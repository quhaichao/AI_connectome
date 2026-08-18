#!/usr/bin/env python
"""Summarize per-layer direct/merge allocation from saved pruning plans."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/fc_is_layer_merge_shares.csv")
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for samples in (16384, 32768):
        for ratio in args.ratios:
            for method, result_name in (
                ("is", f"long2048_n{samples}_eval2048_ratio_sweep"),
                ("fc_cap15", f"long2048_n{samples}_eval2048_fc_cap15_ratio_sweep"),
            ):
                path = args.root / result_name / f"plan_{'fc' if method == 'fc_cap15' else 'is'}_r{ratio:.3f}.json"
                with path.open(encoding="utf-8") as handle:
                    plans = json.load(handle)
                for layer, plan in enumerate(plans):
                    target = int(plan["target"])
                    merges = len(plan["merges"])
                    direct = len(plan["direct"])
                    rows.append(
                        {
                            "activation_samples": samples,
                            "ratio": ratio,
                            "method": method,
                            "layer": layer,
                            "target": target,
                            "merge_count": merges,
                            "direct_count": direct,
                            "merge_fraction": merges / target,
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
