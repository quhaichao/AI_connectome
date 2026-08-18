#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.full_stats import load_full_ffn_statistics
from fc_pruning.modeling import load_model, load_tokenizer
from fc_pruning.reproduction import build_and_apply_full_stat_plans


def json_default(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_summary(results_dir: Path, rows: list[dict]) -> None:
    with (results_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Llama-2-7B reproduction")
    device = torch.device("cuda")
    tokenizer = load_tokenizer(config["model_path"])
    statistics = load_full_ffn_statistics(config["full_stats_path"])
    if statistics["protocol"]["activation_tokens"] != 128 * 2048:
        raise RuntimeError("Reproduction requires exactly 128x2048 full-token activations")

    methods = args.methods or config["reproduction_methods"]
    ratio = float(config["prune_ratios"][0])
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    summary_path = results_dir / "summary.json"
    rows = json.load(summary_path.open()) if args.resume and summary_path.exists() else []
    completed = {row["method"] for row in rows}

    for method in methods:
        if method in completed:
            print(f"Skipping completed method={method}", flush=True)
            continue
        model = load_model(config["model_path"], device)
        plans = build_and_apply_full_stat_plans(model, method, ratio, statistics)
        with (results_dir / f"plan_{method}_r{ratio:.3f}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(plans, handle, indent=2, default=json_default)
        evaluation = evaluate_wikitext_ppl(
            model, tokenizer, config["wikitext_test"], config, device
        )
        row = {"method": method, "ratio": ratio, **evaluation}
        rows.append(row)
        print(json.dumps(row), flush=True)
        write_summary(results_dir, rows)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
