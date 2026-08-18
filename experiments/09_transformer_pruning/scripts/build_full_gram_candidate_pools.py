#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.data import load_calibration
from fc_pruning.gram_similarity import build_layer_candidate_pool
from fc_pruning.modeling import load_model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-grams", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-contexts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    calibration = load_calibration(config["calibration_path"])
    input_ids = calibration["input_ids"]
    split = input_ids.shape[0] - args.holdout_contexts
    if split <= 0:
        raise ValueError("holdout-contexts must leave a non-empty fit split")
    width = model.config.intermediate_size
    layers = model.model.layers
    holdout_grams = [
        torch.zeros((width, width), dtype=torch.float32, device=device)
        for _ in layers
    ]
    holdout_sums = [
        torch.zeros(width, dtype=torch.float64, device=device) for _ in layers
    ]

    def make_hook(layer_index: int):
        def hook(_module, hook_args):
            values = hook_args[0][0].float()
            holdout_grams[layer_index].addmm_(values.t(), values)
            holdout_sums[layer_index] += values.sum(dim=0).double()

        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        for offset, sequence in enumerate(input_ids[split:]):
            print(
                f"Gram holdout context={offset + 1:03d}/{args.holdout_contexts:03d}",
                flush=True,
            )
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    total_payload = torch.load(args.total_grams, map_location="cpu", weights_only=True)
    full_stats = torch.load(config["full_stats_path"], map_location="cpu", weights_only=True)
    total_count = int(input_ids.numel())
    holdout_count = int(args.holdout_contexts * input_ids.shape[1])
    fit_count = total_count - holdout_count
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    methods = ("fc", "is_raw", "is_branch")
    pools = {method: [] for method in methods}
    for layer_index, layer in enumerate(layers):
        print(f"Building Gram pools layer={layer_index:02d}", flush=True)
        total_gram = total_payload["layers"][layer_index].to(device)
        holdout_gram = holdout_grams[layer_index]
        fit_gram = total_gram - holdout_gram
        total_sum = full_stats["layers"][layer_index]["sum"]
        holdout_sum = holdout_sums[layer_index].cpu()
        fit_sum = total_sum - holdout_sum
        for method in methods:
            pools[method].append(
                build_layer_candidate_pool(
                    layer,
                    fit_gram,
                    holdout_gram,
                    fit_sum,
                    holdout_sum,
                    total_sum,
                    fit_count,
                    holdout_count,
                    total_count,
                    args.topk,
                    args.block_size,
                    method,
                )
            )
        del total_gram, fit_gram
        torch.cuda.empty_cache()
    protocol = {
        "calibration_path": str(Path(config["calibration_path"]).resolve()),
        "contexts": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "activation_tokens": total_count,
        "fit_contexts": split,
        "holdout_contexts": args.holdout_contexts,
        "fit_tokens": fit_count,
        "holdout_tokens": holdout_count,
        "topk": args.topk,
        "methods": list(methods),
        "test_used": False,
    }
    for method in methods:
        torch.save({"layers": pools[method], "protocol": protocol}, output / f"{method}.pt")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
