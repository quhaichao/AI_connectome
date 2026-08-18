#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.gram_similarity import GramPlanConfig, build_and_apply_gram_plans
from fc_pruning.modeling import load_model, load_tokenizer


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    tokenizer = load_tokenizer(config["model_path"])
    payload = torch.load(args.pool, map_location="cpu", weights_only=True)
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    evaluation_path = (
        config["wikitext_validation"]
        if args.evaluation_split == "validation"
        else config["wikitext_test"]
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    ratio = float(config["prune_ratios"][0])
    for index, values in enumerate(grid):
        name = values.pop("name", f"config_{index:03d}")
        plan_config = GramPlanConfig(**values)
        print(f"Evaluating {name}: {plan_config}", flush=True)
        model = load_model(config["model_path"], device)
        plans, audits = build_and_apply_gram_plans(
            model, payload["layers"], ratio, plan_config
        )
        evaluation = evaluate_wikitext_ppl(
            model, tokenizer, evaluation_path, config, device
        )
        row = {
            "name": name,
            "split": args.evaluation_split,
            "method": payload["layers"][0]["method"],
            **plan_config.__dict__,
            "merges": sum(audit["merges"] for audit in audits),
            "actual_merge_fraction": sum(audit["merges"] for audit in audits)
            / sum(audit["target"] for audit in audits),
            "mean_similarity": sum(
                audit["mean_merge_similarity"] * audit["merges"] for audit in audits
            )
            / max(1, sum(audit["merges"] for audit in audits)),
            "mean_validation_gain": sum(
                audit["mean_validation_gain"] * audit["merges"] for audit in audits
            )
            / max(1, sum(audit["merges"] for audit in audits)),
            "estimated_cost": sum(audit["estimated_cost"] for audit in audits),
            **evaluation,
        }
        rows.append(row)
        (output / f"plan_{name}.json").write_text(
            json.dumps(plans, indent=2, default=_json_default), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
