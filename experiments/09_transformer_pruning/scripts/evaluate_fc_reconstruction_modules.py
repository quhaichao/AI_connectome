#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.fixed_mask_reconstruction import apply_fixed_mask_reconstruction
from fc_pruning.modeling import load_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--hessians", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", choices=("full_ls", "direct_ls"), required=True)
    parser.add_argument(
        "--covariance-modes",
        nargs="+",
        choices=("second_moment", "centered"),
        default=("second_moment", "centered"),
    )
    parser.add_argument(
        "--merge-residual-weights", nargs="+", type=float, default=(0.0,)
    )
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--evaluation-split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    plans = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    hessians = torch.load(args.hessians, map_location="cpu", weights_only=True)
    statistics = torch.load(config["full_stats_path"], map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(config["model_path"])
    evaluation_path = (
        config["wikitext_validation"]
        if args.evaluation_split == "validation"
        else config["wikitext_test"]
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    sample_count = int(hessians["protocol"]["activation_tokens"])
    for mode in args.modes:
        residual_weights = (
            args.merge_residual_weights if mode == "direct_ls" else (0.0,)
        )
        for covariance_mode in args.covariance_modes:
            for merge_residual_weight in residual_weights:
                suffix = (
                    f"_merge_residual_{merge_residual_weight:g}"
                    if merge_residual_weight
                    else ""
                )
                if args.damping != 1e-4:
                    suffix += f"_damping_{args.damping:g}"
                label = f"{mode}_{covariance_mode}{suffix}"
                print(f"Evaluating FC reconstruction mode={label}", flush=True)
                model = load_model(config["model_path"], device)
                audits = []
                for index, (layer, hessian, layer_stats, plan) in enumerate(
                    zip(
                        model.model.layers,
                        hessians["layers"],
                        statistics["layers"],
                        plans,
                    )
                ):
                    print(f"FC reconstruction layer={index:02d} mode={label}", flush=True)
                    mean = layer_stats["sum"].double() / int(layer_stats["count"])
                    audits.append(
                        apply_fixed_mask_reconstruction(
                            layer,
                            hessian,
                            mean,
                            plan,
                            mode,
                            device,
                            damping=args.damping,
                            covariance_mode=covariance_mode,
                            sample_count=sample_count,
                            merge_residual_weight=merge_residual_weight,
                        )
                    )
                evaluation = evaluate_wikitext_ppl(
                    model, tokenizer, evaluation_path, config, device
                )
                result = {
                    "method": f"fc_joint_mask_{label}",
                    "evaluation_split": args.evaluation_split,
                    "source_plan": str(Path(args.plan).resolve()),
                    "calibration_tokens": 128 * 2048,
                    "reconstructed_sources": sum(
                        a["reconstructed_sources"] for a in audits
                    ),
                    "merge_residual_weight": merge_residual_weight,
                    "damping": args.damping,
                    "mean_delta_norm": sum(a["delta_norm"] for a in audits) / len(audits),
                    "mean_bias_norm": sum(a["bias_norm"] for a in audits) / len(audits),
                    **evaluation,
                }
                mode_dir = output / label
                mode_dir.mkdir(parents=True, exist_ok=True)
                (mode_dir / "audits.json").write_text(
                    json.dumps(audits, indent=2), encoding="utf-8"
                )
                (mode_dir / "result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                results.append(result)
                print(json.dumps(result, indent=2), flush=True)
                del model
                torch.cuda.empty_cache()
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
