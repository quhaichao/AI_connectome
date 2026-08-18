#!/usr/bin/env python
"""Cluster and downsample complete 8192x8192 FC matrices for visualization."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache")
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def cluster_order(matrix: torch.Tensor, clusters: int) -> tuple[torch.Tensor, list[int]]:
    similarity = matrix.float().numpy()
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    order = leaves_list(tree)
    labels = fcluster(tree, t=clusters, criterion="maxclust")[order]
    boundaries = [
        index
        for index in range(1, len(labels))
        if labels[index] != labels[index - 1]
    ]
    return torch.from_numpy(order.copy()).long(), boundaries


def ordered_block_mean(
    matrix: torch.Tensor, order: torch.Tensor, block: int, device: torch.device
) -> torch.Tensor:
    current = matrix.to(device).index_select(0, order.to(device)).index_select(
        1, order.to(device)
    )
    current.fill_diagonal_(0)
    width = current.shape[0]
    if width % block:
        raise ValueError("Block size must divide FC width")
    reduced = current.float().view(width // block, block, width // block, block).mean(
        dim=(1, 3)
    )
    return reduced.cpu()


def ordered_absolute_difference_mean(
    first: torch.Tensor,
    second: torch.Tensor,
    order: torch.Tensor,
    block: int,
    device: torch.device,
) -> torch.Tensor:
    order_device = order.to(device)
    a = first.to(device).index_select(0, order_device).index_select(1, order_device)
    b = second.to(device).index_select(0, order_device).index_select(1, order_device)
    difference = (a.float() - b.float()).abs()
    difference.fill_diagonal_(0)
    width = difference.shape[0]
    reduced = difference.view(
        width // block, block, width // block, block
    ).mean(dim=(1, 3))
    return reduced.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument(
        "--output", default="results/fc_subset_stability/layer_08_full8192_clustered_fc.png"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = {
        "C4 train": [
            root / "results/fc_subset_stability/c4_seed0",
            root / "results/fc_subset_stability/c4_seed1",
        ],
        "WikiText-2 train": [
            root / "results/fc_matrix_stability/wiki_train",
            root / "results/fc_subset_stability/wiki_seed1",
        ],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), constrained_layout=True)
    for row, (domain, directories) in enumerate(paths.items()):
        first = torch.load(
            directories[0] / f"layer_{args.layer:02d}_n262144.pt", weights_only=True
        )
        second = torch.load(
            directories[1] / f"layer_{args.layer:02d}_n262144.pt", weights_only=True
        )
        order, boundaries = cluster_order(first, args.clusters)
        a = ordered_block_mean(first, order, args.block_size, device)
        b = ordered_block_mean(second, order, args.block_size, device)
        difference = ordered_absolute_difference_mean(
            first, second, order, args.block_size, device
        )
        vmax = float(torch.quantile(torch.cat([a.flatten(), b.flatten()]), 0.995))
        diff_vmax = float(torch.quantile(difference.flatten(), 0.995))
        images = [
            axes[row, 0].imshow(a, cmap="viridis", vmin=0, vmax=vmax),
            axes[row, 1].imshow(b, cmap="viridis", vmin=0, vmax=vmax),
            axes[row, 2].imshow(difference, cmap="magma", vmin=0, vmax=diff_vmax),
        ]
        axes[row, 0].set_title(f"{domain} subset 0")
        axes[row, 1].set_title(f"{domain} subset 1")
        axes[row, 2].set_title("Mean absolute difference")
        scaled_boundaries = [boundary / args.block_size - 0.5 for boundary in boundaries]
        for axis in axes[row]:
            axis.set_xlabel("Clustered FFN neurons (16-neuron blocks)")
            axis.set_ylabel("Clustered FFN neurons (16-neuron blocks)")
            for boundary in scaled_boundaries:
                axis.axhline(boundary, color="white", linewidth=0.35, alpha=0.75)
                axis.axvline(boundary, color="white", linewidth=0.35, alpha=0.75)
        fig.colorbar(images[1], ax=axes[row, :2], shrink=0.76, label="Block-mean absolute FC")
        fig.colorbar(images[2], ax=axes[row, 2], shrink=0.76, label="Block-mean absolute difference")
    fig.suptitle(
        f"Layer {args.layer}: complete 8192x8192 FC matrices after hierarchical clustering\n"
        f"Displayed as {8192 // args.block_size}x{8192 // args.block_size} block means"
    )
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
