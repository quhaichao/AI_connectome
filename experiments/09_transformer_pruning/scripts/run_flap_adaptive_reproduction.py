#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.full_stats import load_full_ffn_statistics
from fc_pruning.modeling import load_model, load_tokenizer
from fc_pruning.reproduction import build_and_apply_adaptive_flap_plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["model_path"])
    statistics = load_full_ffn_statistics(config["full_stats_path"])
    ratio = float(config["prune_ratios"][0])
    plans = build_and_apply_adaptive_flap_plans(model, ratio, statistics)
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result = {
        "method": "flap_adaptive_layer_ffn",
        "fidelity": "official FLAP AL global standardized-WIFV allocation restricted to FFN; exact global 20% channel budget",
        "calibration_tokens": 128 * 2048,
        "layer_ratios": [plan["ratio"] for plan in plans],
        **evaluation,
    }
    result_dir = Path(config["results_dir"]) / "flap_adaptive"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "plan.json").write_text(json.dumps(plans, indent=2, default=lambda x: x.tolist()), encoding="utf-8")
    (result_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
