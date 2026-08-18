#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.modeling import load_model, load_tokenizer
from fc_pruning.sobp_reproduction import (
    apply_sobp_layer,
    collect_sobp_mask_gradients,
    sobp_global_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hessians", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    result_dir = Path(config["results_dir"]) / "sobp_adaptive"
    result_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = result_dir / "mask_gradient_scores.pt"
    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["model_path"])
    if gradient_path.exists():
        gradients = torch.load(gradient_path, map_location="cpu", weights_only=True)
    else:
        collect_sobp_mask_gradients(
            model, config["calibration_path"], str(gradient_path), device
        )
        gradients = torch.load(gradient_path, map_location="cpu", weights_only=True)
    hessians = torch.load(args.hessians, map_location="cpu", weights_only=True)
    ratio = float(config["prune_ratios"][0])
    targets = sobp_global_targets(gradients, ratio, maximum_fraction=0.8)
    plans = []
    for index, (layer, hessian, target) in enumerate(
        zip(model.model.layers, hessians["layers"], targets)
    ):
        print(f"SoBP adaptive apply layer={index:02d} target={target}", flush=True)
        plan = apply_sobp_layer(layer, hessian, target, device)
        plan["layer"] = index
        plan["ratio"] = target / model.config.intermediate_size
        plans.append(plan)
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result = {
        "method": "sobp_global_ffn_reproduction",
        "fidelity": "SoBP squared mask-gradient global selection and 80% module cap restricted to FFN; global selection determines layer quotas, followed by OBC Hessian selection and Cholesky reconstruction",
        "calibration_tokens": 128 * 2048,
        "layer_targets": targets,
        "layer_ratios": [target / model.config.intermediate_size for target in targets],
        **evaluation,
    }
    (result_dir / "plan.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
    (result_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
