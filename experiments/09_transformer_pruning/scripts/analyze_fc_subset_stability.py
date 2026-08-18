#!/usr/bin/env python
"""Measure FC stability across independent 128x2048 text subsets."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache")
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from analyze_fc_matrix_stability import matrix_metrics


def load_matrix(directory: Path, layer: int, device: torch.device) -> torch.Tensor:
    return torch.load(
        directory / f"layer_{layer:02d}_n262144.pt", weights_only=True
    ).to(device)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for domain in ("c4", "wiki_train"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        for pair in ("all", "0-1", "0-2", "1-2"):
            selected = domain_rows if pair == "all" else [row for row in domain_rows if row["pair"] == pair]
            record = {"domain": domain, "pair": pair, "comparisons": len(selected)}
            for metric in ("pearson", "cosine", "mae", "top32_overlap"):
                values = torch.tensor([float(row[metric]) for row in selected])
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_std"] = float(values.std(unbiased=False))
            output.append(record)
    return output


def strongest_neurons(matrix: torch.Tensor, count: int) -> torch.Tensor:
    candidate = matrix.float().clone()
    candidate.fill_diagonal_(-torch.inf)
    return candidate.max(dim=1).values.topk(count).indices.sort().values


def clustered_order(matrix: torch.Tensor, clusters: int = 8) -> tuple[torch.Tensor, list[int]]:
    similarity = matrix.float().cpu().numpy().copy()
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    tree = optimal_leaf_ordering(tree, condensed)
    order = leaves_list(tree)
    labels = fcluster(tree, t=clusters, criterion="maxclust")[order]
    boundaries = [
        index
        for index in range(1, len(labels))
        if labels[index] != labels[index - 1]
    ]
    return torch.from_numpy(order.copy()).long(), boundaries


def plot_layer_comparison(
    directories: dict[str, list[Path]],
    rows: list[dict],
    layer: int,
    output: Path,
    device: torch.device,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)
    display_names = {"c4": "C4 train", "wiki_train": "WikiText-2 train"}
    for row_index, domain in enumerate(("c4", "wiki_train")):
        first = load_matrix(directories[domain][0], layer, device)
        second = load_matrix(directories[domain][1], layer, device)
        indices = strongest_neurons(first, 128)
        a = first.index_select(0, indices).index_select(1, indices).float().cpu()
        b = second.index_select(0, indices).index_select(1, indices).float().cpu()
        order, boundaries = clustered_order(a)
        a = a.index_select(0, order).index_select(1, order)
        b = b.index_select(0, order).index_select(1, order)
        a.fill_diagonal_(0)
        b.fill_diagonal_(0)
        difference = (a - b).abs()
        vmax = float(torch.quantile(torch.cat([a.flatten(), b.flatten()]), 0.995))
        diff_vmax = float(torch.quantile(difference.flatten(), 0.995))
        images = [
            axes[row_index, 0].imshow(a, cmap="viridis", vmin=0, vmax=vmax),
            axes[row_index, 1].imshow(b, cmap="viridis", vmin=0, vmax=vmax),
            axes[row_index, 2].imshow(difference, cmap="magma", vmin=0, vmax=diff_vmax),
        ]
        metric = next(
            row
            for row in rows
            if row["domain"] == domain and row["pair"] == "0-1" and row["layer"] == layer
        )
        axes[row_index, 0].set_title(f"{display_names[domain]} subset 0")
        axes[row_index, 1].set_title(f"{display_names[domain]} subset 1")
        axes[row_index, 2].set_title(
            f"Absolute difference\nr={metric['pearson']:.3f}, Top-32={100 * metric['top32_overlap']:.1f}%"
        )
        for axis in axes[row_index]:
            axis.set_xlabel("Selected FFN neuron")
            axis.set_ylabel("Selected FFN neuron")
            for boundary in boundaries:
                axis.axhline(boundary - 0.5, color="white", linewidth=0.45, alpha=0.8)
                axis.axvline(boundary - 0.5, color="white", linewidth=0.45, alpha=0.8)
        fig.colorbar(images[1], ax=axes[row_index, :2], shrink=0.78, label="Absolute FC")
        fig.colorbar(images[2], ax=axes[row_index, 2], shrink=0.78, label="Absolute difference")
    fig.suptitle(
        f"Layer {layer}: hierarchically clustered FC across independent 128x2048 subsets"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/fc_subset_stability")
    parser.add_argument("--visual-layer", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    directories = {
        "c4": [output_dir / f"c4_seed{seed}" for seed in range(3)],
        "wiki_train": [
            root / "results/fc_matrix_stability/wiki_train",
            output_dir / "wiki_seed1",
            output_dir / "wiki_seed2",
        ],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for domain, domain_dirs in directories.items():
        for layer in range(16):
            matrices = [load_matrix(directory, layer, device) for directory in domain_dirs]
            for first_index, second_index in itertools.combinations(range(3), 2):
                rows.append(
                    {
                        "domain": domain,
                        "pair": f"{first_index}-{second_index}",
                        "layer": layer,
                        **matrix_metrics(matrices[first_index], matrices[second_index]),
                    }
                )
            print(f"Analyzed {domain} layer {layer:02d}/15", flush=True)
    summary = summarize(rows)
    with (output_dir / "subset_layer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "subset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "subset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_layer_comparison(
        directories,
        rows,
        args.visual_layer,
        output_dir / f"layer_{args.visual_layer:02d}_subset_fc_comparison.png",
        device,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
