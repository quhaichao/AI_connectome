#!/usr/bin/env python
"""Validate and summarize the full FFN ratio matrix by calibration domain."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from fc_pruning.modeling import MODEL_KEYS
from fc_pruning.ratio_matrix import METHODS, RATIOS


METHOD_LABELS = {
    "fc_ls": "FC + direct-source LS",
    "flap": "FLAP",
    "sobp": "SoBP",
    "fang": "FANG",
    "slimllm": "SlimLLM",
    "wanda": "Wanda",
}
LEGACY_METHOD_ALIASES = {"fand": "fang"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/ffn_ratio_matrix_pearson")
    parser.add_argument(
        "--output-csv",
        help="Raw CSV output path relative to the project root (default: RESULTS_ROOT/raw_results.csv)",
    )
    parser.add_argument(
        "--aggregate-csv",
        help="Aggregate CSV output path relative to the project root (default: RESULTS_ROOT/aggregate.csv)",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--domains", nargs="+", choices=("c4",), default=["c4"])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / args.results_root
    raw = []
    missing = []
    for model in args.models:
        for domain in args.domains:
            for seed in args.seeds:
                path = root / model / domain / f"seed{seed}" / "summary.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                with path.open(encoding="utf-8") as handle:
                    for source_row in csv.DictReader(handle):
                        row = dict(source_row)
                        row["method"] = LEGACY_METHOD_ALIASES.get(
                            row["method"], row["method"]
                        )
                        if (
                            row["method"] in args.methods
                            and float(row["ratio"]) in args.ratios
                        ):
                            raw.append(row)
    expected = (
        len(args.models)
        * len(args.domains)
        * len(args.seeds)
        * len(args.methods)
        * len(args.ratios)
    )
    if (missing or len(raw) != expected) and not args.allow_incomplete:
        raise RuntimeError(f"Expected {expected} rows, found {len(raw)}; missing={missing}")

    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["domain"], row["model"], row["method"], float(row["ratio"]))].append(
            float(row["ppl"])
        )
    aggregate = []
    for (domain, model, method, ratio), values in sorted(grouped.items()):
        aggregate.append(
            {
                "domain": domain,
                "model": model,
                "method": method,
                "ratio": ratio,
                "mean_ppl": statistics.mean(values),
                "std_ppl": statistics.stdev(values) if len(values) > 1 else 0.0,
                "minimum_ppl": min(values),
                "maximum_ppl": max(values),
                "subsets": len(values),
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    output_csv = (
        project_root / args.output_csv
        if args.output_csv
        else root / "raw_results.csv"
    )
    aggregate_csv = (
        project_root / args.aggregate_csv
        if args.aggregate_csv
        else root / "aggregate.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        if raw:
            writer = csv.DictWriter(handle, fieldnames=raw[0].keys())
            writer.writeheader()
            writer.writerows(raw)
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        if aggregate:
            writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
            writer.writeheader()
            writer.writerows(aggregate)

    for domain in args.domains:
        lines = [
            f"# {domain} calibration: FFN ratio matrix",
            "",
            f"Mean and sample standard deviation over {len(args.seeds)} independent 128x2048 calibration subsets.",
            "WikiText-2 raw test is shared across all rows.",
            "",
        ]
        for model in args.models:
            lines.extend(
                [
                    f"## {model}",
                    "",
                    "| Method | "
                    + " | ".join(f"{ratio:.0%} PPL" for ratio in args.ratios)
                    + " |",
                    "|---|" + "---:|" * len(args.ratios),
                ]
            )
            for method in args.methods:
                cells = []
                for ratio in args.ratios:
                    match = [
                        row for row in aggregate
                        if row["domain"] == domain
                        and row["model"] == model
                        and row["method"] == method
                        and row["ratio"] == ratio
                    ]
                    cells.append(
                        f"{match[0]['mean_ppl']:.4f} +/- {match[0]['std_ppl']:.4f}"
                        if match else "pending"
                    )
                lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
            lines.append("")
        (root / f"summary_{domain}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "completeness.json").write_text(
        json.dumps({
            "expected_rows": expected,
            "actual_rows": len(raw),
            "seeds": args.seeds,
            "fc_similarity": "signed_pearson",
            "raw_results": str(output_csv),
            "aggregate_results": str(aggregate_csv),
            "missing": missing,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "expected": expected,
        "actual": len(raw),
        "groups": len(aggregate),
        "raw_results": str(output_csv),
        "aggregate_results": str(aggregate_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
