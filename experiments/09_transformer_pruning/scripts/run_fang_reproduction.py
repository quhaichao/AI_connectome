#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.fang_reproduction import (
    build_and_apply_fang_flap_plans,
    collect_fang_cluster_statistics,
    collect_projected_ffn_inputs,
    kmeans_assignments,
    pca_bases_from_covariances,
)
from fc_pruning.modeling import load_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--covariance-stats", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    result_dir = Path(config["results_dir"]) / "fang"
    result_dir.mkdir(parents=True, exist_ok=True)
    group_cache = result_dir / "groups.pt"
    statistic_cache = result_dir / "cluster_statistics.pt"
    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["model_path"])

    if group_cache.exists():
        group_payload = torch.load(group_cache, map_location="cpu", weights_only=True)
    else:
        covariance = torch.load(
            args.covariance_stats, map_location="cpu", weights_only=True
        )
        bases = pca_bases_from_covariances(covariance, 64, device)
        projected = collect_projected_ffn_inputs(
            model, config["calibration_path"], bases, device
        )
        assignments, centers = kmeans_assignments(
            projected, 7, 20, int(config["seed"]), device
        )
        group_payload = {"assignments": assignments, "centers": centers}
        torch.save(group_payload, group_cache)
        del projected, bases, covariance

    if statistic_cache.exists():
        statistics = torch.load(
            statistic_cache, map_location="cpu", weights_only=True
        )
    else:
        statistics = collect_fang_cluster_statistics(
            model,
            config["calibration_path"],
            group_payload["assignments"],
            7,
            device,
        )
        torch.save(statistics, statistic_cache)
    plans = build_and_apply_fang_flap_plans(
        model, statistics, float(config["prune_ratios"][0]), clusters=7, temperature=9.0
    )
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result = {
        "method": "fang_flap_paper_reproduction",
        "fidelity": "PCA64, K=7, Taylor cluster-neuron scores, shared group, tau=9 group-reweighted FLAP; balanced greedy LAP approximation; fixed per-layer 20% disables ASA",
        "calibration_tokens": 128 * 2048,
        **evaluation,
    }
    (result_dir / "plan.json").write_text(
        json.dumps(plans, indent=2, default=lambda value: value.tolist()),
        encoding="utf-8",
    )
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
