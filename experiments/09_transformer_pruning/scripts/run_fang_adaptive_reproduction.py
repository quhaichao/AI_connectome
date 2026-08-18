#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.adaptive_allocation import fang_asa_targets
from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.fang_reproduction import (
    build_and_apply_fang_flap_plans,
    collect_block_functional_complexity,
)
from fc_pruning.modeling import load_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    result_dir = Path(config["results_dir"]) / "fang_adaptive"
    result_dir.mkdir(parents=True, exist_ok=True)
    complexity_path = result_dir / "functional_complexity.pt"
    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["model_path"])
    if complexity_path.exists():
        complexity = torch.load(complexity_path, map_location="cpu", weights_only=True)
    else:
        complexity = collect_block_functional_complexity(
            model, config["calibration_path"], device
        )
        torch.save(complexity, complexity_path)
    statistics = torch.load(
        Path(config["results_dir"]) / "fang" / "cluster_statistics.pt",
        map_location="cpu",
        weights_only=True,
    )
    ratio = float(config["prune_ratios"][0])
    width = model.config.intermediate_size
    targets, ratios = fang_asa_targets(complexity, width, ratio)
    plans = build_and_apply_fang_flap_plans(
        model, statistics, ratio, clusters=7, temperature=9.0, targets=targets
    )
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result = {
        "method": "fang_flap_asa_paper_reproduction",
        "fidelity": "F-FANG with Eq.8 block functional complexity ASA; ratios bounded to target x [0.5,1.5] and exact global 20% budget; balanced greedy LAP approximation",
        "calibration_tokens": 128 * 2048,
        "functional_complexity": complexity.tolist(),
        "layer_targets": targets,
        "layer_ratios": ratios,
        **evaluation,
    }
    (result_dir / "plan.json").write_text(json.dumps(plans, indent=2, default=lambda x: x.tolist()), encoding="utf-8")
    (result_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
