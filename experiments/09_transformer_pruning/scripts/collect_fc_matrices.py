#!/usr/bin/env python
"""Stream full-context activations into exact nested-budget FC matrices."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from fc_pruning.modeling import llama_layers, load_model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Llama-3.2-1B")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[8192, 16384, 32768, 65536, 131072, 262144]
    )
    parser.add_argument("--position-seed", type=int, default=32452843)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    calibration_path = (root / args.calibration).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    input_ids = calibration["input_ids"]
    sequence_count, sequence_length = map(int, input_ids.shape)
    budgets = sorted(set(args.budgets))
    if (sequence_count, sequence_length) != (128, 2048):
        raise ValueError(f"Expected calibration shape (128, 2048), got {tuple(input_ids.shape)}")
    if budgets[-1] != sequence_count * sequence_length:
        raise ValueError("Largest budget must use all 128x2048 activation positions")
    if any(budget % sequence_count for budget in budgets):
        raise ValueError("Every budget must be divisible by 128 contexts")
    counts = [budget // sequence_count for budget in budgets]
    if counts != sorted(counts) or counts[-1] != sequence_length:
        raise ValueError("Invalid nested position counts")

    orders = []
    for sequence_index in range(sequence_count):
        order = list(range(sequence_length))
        random.Random(args.position_seed + sequence_index * 1000003).shuffle(order)
        orders.append(order)
    strata = []
    previous = 0
    for count in counts:
        strata.append(
            [
                torch.tensor(order[previous:count], dtype=torch.long)
                for order in orders
            ]
        )
        previous = count

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact 8192x8192 FC matrices")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = load_model(str((root / args.model).resolve()), device)
    model.eval()
    layers = llama_layers(model)
    width = int(model.config.intermediate_size)
    accumulators = [
        [torch.zeros((width, width), dtype=torch.float32, device=device) for _ in strata]
        for _ in layers
    ]
    state = {"sequence": 0}

    def make_hook(layer_index: int):
        def hook(_module, hook_args):
            activation = hook_args[0][0].detach()
            sequence_index = state["sequence"]
            for stratum_index, per_sequence in enumerate(strata):
                indices = per_sequence[sequence_index].to(device)
                selected = activation.index_select(0, indices).float()
                accumulators[layer_index][stratum_index].addmm_(
                    selected.t(), selected
                )
        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_index))
        for layer_index, layer in enumerate(layers)
    ]
    try:
        for sequence_index, sequence in enumerate(input_ids):
            state["sequence"] = sequence_index
            model.model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
            if (sequence_index + 1) % 8 == 0:
                print(f"Processed {sequence_index + 1}/{sequence_count} contexts", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    for layer_index in range(len(layers)):
        cumulative = torch.zeros((width, width), dtype=torch.float32, device=device)
        for stratum_index, budget in enumerate(budgets):
            cumulative.add_(accumulators[layer_index][stratum_index])
            accumulators[layer_index][stratum_index] = None
            norms = cumulative.diagonal().clamp_min(1e-30).sqrt()
            fc = (cumulative / (norms[:, None] * norms[None, :])).abs_()
            torch.save(
                fc.to("cpu", torch.float16),
                output_dir / f"layer_{layer_index:02d}_n{budget}.pt",
            )
            del fc, norms
        del cumulative
        torch.cuda.empty_cache()
        print(f"Saved layer {layer_index:02d}/{len(layers) - 1:02d}", flush=True)

    metadata = {
        "source_calibration": str(calibration_path),
        "source_protocol": calibration.get("protocol", {}),
        "model": str((root / args.model).resolve()),
        "layers": len(layers),
        "intermediate_size": width,
        "contexts": sequence_count,
        "context_length": sequence_length,
        "budgets": budgets,
        "positions_per_context": counts,
        "position_seed": args.position_seed,
        "nested_positions": True,
        "matrix": "absolute uncentered cosine similarity of full SwiGLU z",
        "storage_dtype": "float16",
        "test_used": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
