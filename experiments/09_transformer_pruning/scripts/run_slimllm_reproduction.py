#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.full_stats import load_full_ffn_statistics
from fc_pruning.modeling import load_model, load_tokenizer
from fc_pruning.slimllm_reproduction import (
    SLIMLLM_SELECTION_MODES,
    apply_slimllm_plans,
    build_slimllm_plans,
)


VARIANTS = (
    ("official_piecewise_lr", "official_piecewise", True),
    ("official_piecewise_no_lr", "official_piecewise", False),
    ("direct_lowest_lr", "direct_lowest", True),
    ("direct_lowest_no_lr", "direct_lowest", False),
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_plan(plans: list[dict], ratio: float) -> None:
    if len(plans) != 32:
        raise RuntimeError(f"Expected 32 SlimLLM layer plans; got {len(plans)}")
    for index, plan in enumerate(plans):
        expected = int(round(plan["original_width"] * ratio))
        if plan["layer"] != index or plan["target"] != expected:
            raise RuntimeError(f"Invalid SlimLLM plan at layer {index}")
        if len(plan["pruned"]) != expected or len(set(plan["pruned"])) != expected:
            raise RuntimeError(f"Invalid SlimLLM mask cardinality at layer {index}")


def build_plans(args, config: dict, result_dir: Path, device: torch.device) -> None:
    plan_paths = {
        mode: result_dir / f"plan_{mode}.json" for mode in SLIMLLM_SELECTION_MODES
    }
    if all(path.exists() for path in plan_paths.values()) and not args.overwrite:
        print("SlimLLM plans already exist; skipping plan construction", flush=True)
        return

    model = load_model(config["model_path"], device)
    full = load_full_ffn_statistics(config["full_stats_path"])
    covariance = torch.load(
        args.covariance_stats, map_location="cpu", weights_only=True
    )
    hessians = torch.load(args.hessians, map_location="cpu", weights_only=True)
    plans = build_slimllm_plans(
        model,
        full,
        covariance,
        hessians,
        float(config["prune_ratios"][0]),
        device,
    )
    for mode, mode_plans in plans.items():
        validate_plan(mode_plans, float(config["prune_ratios"][0]))
        write_json(plan_paths[mode], mode_plans)
    protocol = {
        "method": "SlimLLM",
        "paper": "SlimLLM: Accurate Structured Pruning for Large Language Models",
        "official_repository": "https://github.com/guojialong1/Slimllm",
        "scope": "FFN intermediate channels only",
        "fixed_ratio_per_layer": float(config["prune_ratios"][0]),
        "pruned_channels_per_layer": plans["official_piecewise"][0]["target"],
        "calibration_tokens": 128 * 2048,
        "calibration_split": "WikiText-2 train",
        "statistics": "all 128x2048 token positions from the dense model",
        "pca": "official equations 5-8 using centered FFN output covariance",
        "regression": "official independent output-wise affine least squares",
        "controlled_adaptations": [
            "attention head pruning disabled",
            "cross-layer nonuniform allocation disabled",
            "every FFN layer prunes exactly round(11008*0.20)=2202 channels",
            "closed-form sufficient-statistic regression replaces np.polyfit",
        ],
        "selection_modes": {
            "official_piecewise": "official code interval [round(1%*width), round(21%*width))",
            "direct_lowest": "transparent ablation deleting the lowest-scoring 20%",
        },
        "test_used_for_calibration_or_selection": False,
    }
    write_json(result_dir / "protocol.json", protocol)
    del model, full, covariance, hessians, plans
    torch.cuda.empty_cache()


def evaluate_variant(
    config: dict,
    tokenizer,
    plans: list[dict],
    variant: str,
    selection_mode: str,
    use_regression: bool,
    split: str,
    device: torch.device,
) -> dict:
    model = load_model(config["model_path"], device)
    apply_slimllm_plans(model, plans, use_regression)
    data_path = (
        config["wikitext_validation"] if split == "validation" else config["wikitext_test"]
    )
    evaluation = evaluate_wikitext_ppl(model, tokenizer, data_path, config, device)
    row = {
        "variant": variant,
        "selection_mode": selection_mode,
        "linear_regression": use_regression,
        "split": split,
        "ratio": float(config["prune_ratios"][0]),
        "pruned_channels_per_layer": plans[0]["target"],
        "calibration_tokens": 128 * 2048,
        **evaluation,
    }
    del model
    torch.cuda.empty_cache()
    return row


def run_validation(args, config: dict, result_dir: Path, device: torch.device) -> None:
    selection_path = result_dir / "frozen_selection.json"
    if selection_path.exists() and not args.overwrite:
        print("SlimLLM selection is already frozen; skipping validation", flush=True)
        return
    tokenizer = load_tokenizer(config["model_path"])
    results_path = result_dir / "validation_results.json"
    rows = [] if args.overwrite or not results_path.exists() else read_json(results_path)
    completed = {row["variant"] for row in rows}
    for variant, mode, use_regression in VARIANTS:
        if variant in completed:
            continue
        plans = read_json(result_dir / f"plan_{mode}.json")
        validate_plan(plans, float(config["prune_ratios"][0]))
        row = evaluate_variant(
            config, tokenizer, plans, variant, mode, use_regression, "validation", device
        )
        rows.append(row)
        write_json(results_path, rows)
        print(json.dumps(row), flush=True)

    if {row["variant"] for row in rows} != {variant[0] for variant in VARIANTS}:
        raise RuntimeError("Cannot freeze SlimLLM before all validation variants finish")
    priority = {variant[0]: index for index, variant in enumerate(VARIANTS)}
    winner = min(rows, key=lambda row: (row["ppl"], priority[row["variant"]]))
    frozen = {
        "selected_variant": winner["variant"],
        "selection_mode": winner["selection_mode"],
        "linear_regression": winner["linear_regression"],
        "selection_rule": "minimum complete WikiText-2 validation PPL",
        "validation_ppl": winner["ppl"],
        "all_validation_complete": True,
        "test_evaluated_before_freeze": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(selection_path, frozen)
    print(json.dumps(frozen, indent=2), flush=True)


def run_test(args, config: dict, result_dir: Path, device: torch.device) -> None:
    selection_path = result_dir / "frozen_selection.json"
    if not selection_path.exists():
        raise RuntimeError("Refusing test evaluation before validation selection is frozen")
    output = result_dir / "test_result.json"
    if output.exists() and not args.overwrite:
        print("SlimLLM test result already exists; skipping test", flush=True)
        return
    frozen = read_json(selection_path)
    plans = read_json(result_dir / f"plan_{frozen['selection_mode']}.json")
    validate_plan(plans, float(config["prune_ratios"][0]))
    tokenizer = load_tokenizer(config["model_path"])
    row = evaluate_variant(
        config,
        tokenizer,
        plans,
        frozen["selected_variant"],
        frozen["selection_mode"],
        bool(frozen["linear_regression"]),
        "test",
        device,
    )
    result = {
        "method": "SlimLLM FFN-only fixed 20%",
        "frozen_selection": frozen,
        **row,
        "test_used_for_calibration_or_selection": False,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--covariance-stats", required=True)
    parser.add_argument("--hessians", required=True)
    parser.add_argument(
        "--stage", choices=("plan", "validation", "test", "all"), default="all"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config, _root = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaMA-2-7B SlimLLM reproduction")
    device = torch.device(args.device)
    result_dir = Path(config["results_dir"]) / "slimllm"
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("plan", "all"):
        build_plans(args, config, result_dir, device)
    if args.stage in ("validation", "all"):
        run_validation(args, config, result_dir, device)
    if args.stage in ("test", "all"):
        run_test(args, config, result_dir, device)


if __name__ == "__main__":
    main()
