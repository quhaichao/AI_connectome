#!/usr/bin/env python
"""Validate and summarize the full FFN ratio matrix by calibration domain."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METHOD_LABELS = {
    "fc_ls": "FC + direct-source LS",
    "flap": "FLAP",
    "sobp": "SoBP",
    "fand": "FAND",
    "slimllm": "SlimLLM",
    "wanda": "Wanda",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/ffn_ratio_matrix")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / args.results_root
    raw = []
    missing = []
    for model in ("llama2_7b", "llama32_1b"):
        for domain in ("c4", "wiki_train"):
            for seed in args.seeds:
                path = root / model / domain / f"seed{seed}" / "summary.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                with path.open(encoding="utf-8") as handle:
                    raw.extend(dict(row) for row in csv.DictReader(handle))
    expected = 2 * 2 * len(args.seeds) * 6 * 4
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
    with (root / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
        if raw:
            writer = csv.DictWriter(handle, fieldnames=raw[0].keys())
            writer.writeheader()
            writer.writerows(raw)
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        if aggregate:
            writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
            writer.writeheader()
            writer.writerows(aggregate)

    for domain in ("c4", "wiki_train"):
        lines = [
            f"# {domain} calibration: FFN ratio matrix",
            "",
            f"Mean and sample standard deviation over {len(args.seeds)} independent 128x2048 calibration subsets.",
            "WikiText-2 raw test is shared across all rows.",
            "",
        ]
        for model in ("llama2_7b", "llama32_1b"):
            lines.extend(
                [
                    f"## {model}",
                    "",
                    "| Method | 20% PPL | 30% PPL | 40% PPL | 50% PPL |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for method in METHOD_LABELS:
                cells = []
                for ratio in (0.2, 0.3, 0.4, 0.5):
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
            "missing": missing,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"expected": expected, "actual": len(raw), "groups": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
