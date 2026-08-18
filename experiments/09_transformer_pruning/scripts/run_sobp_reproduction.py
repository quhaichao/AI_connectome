#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.modeling import load_model, load_tokenizer
from fc_pruning.sobp_reproduction import apply_sobp_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hessians", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["model_path"])
    hessians = torch.load(args.hessians, map_location="cpu", weights_only=True)
    expected = 128 * 2048
    if hessians["protocol"]["activation_tokens"] != expected:
        raise RuntimeError("SoBP requires exactly 128x2048 activation tokens")
    target = int(round(model.config.intermediate_size * float(config["prune_ratios"][0])))
    plans = []
    for index, (layer, hessian) in enumerate(zip(model.model.layers, hessians["layers"])):
        print(f"SoBP apply layer={index:02d}", flush=True)
        plan = apply_sobp_layer(layer, hessian, target, device)
        plan["layer"] = index
        plans.append(plan)
    evaluation = evaluate_wikitext_ppl(model, tokenizer, config["wikitext_test"], config, device)
    result = {
        "method": "sobp_obc_fixed_ffn_reproduction",
        "fidelity": "SoBP fixed-per-layer FFN adaptation: OBC Hessian score and module-wise Cholesky reconstruction; global knapsack and mask-gradient allocation replaced by fixed 20% quota",
        "calibration_tokens": expected,
        **evaluation,
    }
    result_dir = Path(config["results_dir"]) / "sobp"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "plan.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
    (result_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
