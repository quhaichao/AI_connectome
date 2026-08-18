#!/usr/bin/env python
"""Compare within-domain convergence and cross-domain FC stability."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache")
)

import matplotlib.pyplot as plt
import torch


def matrix_metrics(x: torch.Tensor, y: torch.Tensor, topk: int = 32) -> dict:
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("FC matrices must be matching square tensors")
    width = x.shape[0]
    count = width * width - width
    totals = {key: 0.0 for key in ("x", "y", "xx", "yy", "xy", "abs")}
    block_size = 512
    for begin in range(0, width, block_size):
        end = min(width, begin + block_size)
        xb = x[begin:end].float()
        yb = y[begin:end].float()
        totals["x"] += float(xb.sum())
        totals["y"] += float(yb.sum())
        totals["xx"] += float(xb.square().sum())
        totals["yy"] += float(yb.square().sum())
        totals["xy"] += float((xb * yb).sum())
        totals["abs"] += float((xb - yb).abs().sum())
    diagonal_x = x.diagonal().float()
    diagonal_y = y.diagonal().float()
    totals["x"] -= float(diagonal_x.sum())
    totals["y"] -= float(diagonal_y.sum())
    totals["xx"] -= float(diagonal_x.square().sum())
    totals["yy"] -= float(diagonal_y.square().sum())
    totals["xy"] -= float((diagonal_x * diagonal_y).sum())
    totals["abs"] -= float((diagonal_x - diagonal_y).abs().sum())
    covariance = totals["xy"] - totals["x"] * totals["y"] / count
    variance_x = totals["xx"] - totals["x"] ** 2 / count
    variance_y = totals["yy"] - totals["y"] ** 2 / count
    pearson = covariance / max((variance_x * variance_y) ** 0.5, 1e-30)
    cosine = totals["xy"] / max((totals["xx"] * totals["yy"]) ** 0.5, 1e-30)

    x_for_topk = x.float().clone()
    y_for_topk = y.float().clone()
    x_for_topk.fill_diagonal_(-torch.inf)
    y_for_topk.fill_diagonal_(-torch.inf)
    x_indices = x_for_topk.topk(topk, dim=1).indices
    y_indices = y_for_topk.topk(topk, dim=1).indices
    overlap = (
        (x_indices.unsqueeze(2) == y_indices.unsqueeze(1))
        .any(dim=2)
        .float()
        .mean()
    )
    return {
        "pearson": pearson,
        "cosine": cosine,
        "mae": totals["abs"] / count,
        "top32_overlap": float(overlap),
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["comparison"], row["budget"]), []).append(row)
    output = []
    for (comparison, budget), values in sorted(groups.items()):
        summary = {"comparison": comparison, "budget": budget, "layers": len(values)}
        for metric in ("pearson", "cosine", "mae", "top32_overlap"):
            tensor = torch.tensor([float(row[metric]) for row in values])
            summary[f"{metric}_mean"] = float(tensor.mean())
            summary[f"{metric}_std"] = float(tensor.std(unbiased=False))
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4-dir", default="results/fc_matrix_stability/c4")
    parser.add_argument("--wiki-dir", default="results/fc_matrix_stability/wiki_train")
    parser.add_argument("--output-dir", default="results/fc_matrix_stability")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    c4_dir = (root / args.c4_dir).resolve()
    wiki_dir = (root / args.wiki_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    c4_meta = json.loads((c4_dir / "metadata.json").read_text(encoding="utf-8"))
    wiki_meta = json.loads((wiki_dir / "metadata.json").read_text(encoding="utf-8"))
    if c4_meta["budgets"] != wiki_meta["budgets"]:
        raise ValueError("C4 and Wiki train budgets differ")
    budgets = c4_meta["budgets"]
    full_budget = budgets[-1]
    layers = int(c4_meta["layers"])
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    rows = []
    for layer in range(layers):
        c4 = {
            budget: torch.load(
                c4_dir / f"layer_{layer:02d}_n{budget}.pt", weights_only=True
            ).to(device)
            for budget in budgets
        }
        wiki = {
            budget: torch.load(
                wiki_dir / f"layer_{layer:02d}_n{budget}.pt", weights_only=True
            ).to(device)
            for budget in budgets
        }
        for budget in budgets:
            comparisons = [("cross_domain", c4[budget], wiki[budget])]
            if budget != full_budget:
                comparisons.extend(
                    [
                        ("c4_vs_full", c4[budget], c4[full_budget]),
                        ("wiki_train_vs_full", wiki[budget], wiki[full_budget]),
                    ]
                )
            else:
                comparisons.extend(
                    [
                        ("c4_vs_full", c4[budget], c4[budget]),
                        ("wiki_train_vs_full", wiki[budget], wiki[budget]),
                    ]
                )
            for comparison, x, y in comparisons:
                rows.append(
                    {
                        "comparison": comparison,
                        "budget": budget,
                        "layer": layer,
                        **matrix_metrics(x, y),
                    }
                )
        print(f"Analyzed layer {layer:02d}/{layers - 1:02d}", flush=True)
    summary = summarize(rows)
    with (output_dir / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    labels = {
        "cross_domain": "C4 vs Wiki train",
        "c4_vs_full": "C4 reduced vs full",
        "wiki_train_vs_full": "Wiki train reduced vs full",
    }
    colors = {"cross_domain": "#E45756", "c4_vs_full": "#4C78A8", "wiki_train_vs_full": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for comparison in labels:
        selected = [row for row in summary if row["comparison"] == comparison]
        x = [row["budget"] for row in selected]
        axes[0].plot(x, [row["pearson_mean"] for row in selected], marker="o", label=labels[comparison], color=colors[comparison])
        axes[1].plot(x, [row["top32_overlap_mean"] for row in selected], marker="o", label=labels[comparison], color=colors[comparison])
    for axis, ylabel in zip(axes, ("FC matrix Pearson correlation", "Top-32 neighbor overlap")):
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, [f"{budget // 1024}k" for budget in budgets])
        axis.set_xlabel("Activation samples")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.35)
        axis.set_ylim(0, 1.02)
    axes[0].legend(frameon=False)
    fig.suptitle("FC stability across activation budget and calibration domain")
    fig.savefig(output_dir / "fc_matrix_stability.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
