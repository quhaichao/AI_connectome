#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.compute_budget import equal_macs_budget
from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.modeling import load_model, load_tokenizer
from fc_pruning.svd_compression import apply_svd_llm_full_ffn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--covariance-stats", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["model_path"])
    ratio = float(config["prune_ratios"][0])
    budget = equal_macs_budget(
        model.config.hidden_size,
        model.config.intermediate_size,
        int(config["evaluation_length"]),
        ratio,
        number_layers=model.config.num_hidden_layers,
    )
    statistics = torch.load(
        args.covariance_stats, map_location="cpu", weights_only=True
    )
    expected = 128 * 2048
    if statistics["protocol"]["activation_tokens"] != expected:
        raise RuntimeError("SVD-LLM requires exactly 128x2048 activation tokens")
    plans = apply_svd_llm_full_ffn(
        model, budget.svd_rank, statistics, device
    )
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result = {
        "method": "svd_llm_full_whitening_ffn_equal_macs",
        "fidelity": "official full Cholesky whitening for gate/up; algebraically equivalent output-covariance form for down",
        "calibration_tokens": expected,
        **budget.to_dict(),
        **evaluation,
    }
    result_dir = Path(config["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "svd_llm_full_equal_macs_plan.json").write_text(
        json.dumps(plans, indent=2), encoding="utf-8"
    )
    (result_dir / "svd_llm_full_equal_macs_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
