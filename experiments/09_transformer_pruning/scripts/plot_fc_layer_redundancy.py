#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    pool = torch.load(args.pool, map_location="cpu", weights_only=True)
    plans = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if len(pool["layers"]) != len(plans):
        raise ValueError("FC pool and pruning plan have different layer counts")

    rows = []
    for layer_index, (layer_pool, plan) in enumerate(zip(pool["layers"], plans)):
        maximum_fc = layer_pool["similarity"].float().max(dim=1).values
        high_fc = maximum_fc > args.threshold
        width = int(maximum_fc.numel())
        rows.append(
            {
                "layer": layer_index,
                "ffn_width": width,
                "high_fc_count": int(high_fc.sum()),
                "high_fc_fraction": float(high_fc.float().mean()),
                "accepted_fc_pruning_count": len(plan.get("merges", [])),
                "total_pruning_count": len(plan["pruned"]),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    layers = np.array([row["layer"] for row in rows])
    high_fc_percent = np.array([100.0 * row["high_fc_fraction"] for row in rows])
    pruning_counts = np.array([row["accepted_fc_pruning_count"] for row in rows])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.13},
    )

    axes[0].bar(layers, high_fc_percent, color="#2a9d8f", width=0.78)
    axes[0].set_ylabel("Neurons with max FC > 0.6 (%)")
    axes[0].set_title("Layer-wise high-FC prevalence and accepted FC pruning")
    axes[0].grid(axis="y", color="#d7dce2", linewidth=0.7, alpha=0.8)
    axes[0].set_axisbelow(True)

    axes[1].bar(layers, pruning_counts, color="#457b9d", width=0.78)
    axes[1].axhline(
        330,
        color="#d1495b",
        linestyle="--",
        linewidth=1.2,
        label="15% FC cap (330 neurons)",
    )
    axes[1].set_ylabel("Accepted FC-pruned neurons")
    axes[1].set_xlabel("Transformer layer")
    axes[1].set_xticks(layers)
    axes[1].set_xlim(-0.7, len(layers) - 0.3)
    axes[1].grid(axis="y", color="#d7dce2", linewidth=0.7, alpha=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper center")

    figure.text(
        0.99,
        0.012,
        "High-FC: each neuron has at least one Top-32 candidate above 0.6; "
        "accepted pruning uses the current holdout-gated plan without a 0.6 hard threshold.",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#4f5965",
    )
    figure.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.14)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
