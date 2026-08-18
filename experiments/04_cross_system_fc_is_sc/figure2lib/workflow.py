"""Unified per-system workflow for manuscript Figure 2.

Every system follows the same sequence and shares one FC-derived partition
between the diamond matrices, modularity statistics, and network diagrams.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .analysis import matrix_correlation, prepare_system_matrices
from .plotting import (
    calculate_corr_sp_boxplot_trend,
    plot_fc_is_fc_sc_diamonds,
)


SYSTEM_COLORS = {
    "MLP": "#17617A",
    "CNN": "#20979E",
    "Transformer": "#18AFC8",
    "C. elegans": "#8B2E83",
    "Mouse": "#DF3F79",
    "Marmoset": "#EF7B75",
    "Human": "#F2B563",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _positive_density_graph(matrix, edge_density: float) -> nx.Graph:
    """Build an undirected graph from the strongest positive matrix entries."""
    matrix = np.asarray(matrix, dtype=float)
    n_nodes = matrix.shape[0]
    row, column = np.triu_indices(n_nodes, k=1)
    weights = matrix[row, column]
    valid = np.isfinite(weights) & (weights > 0)
    row, column, weights = row[valid], column[valid], weights[valid]

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    if weights.size == 0:
        return graph
    n_possible = n_nodes * (n_nodes - 1) // 2
    n_keep = min(weights.size, max(1, int(np.ceil(edge_density * n_possible))))
    keep = np.argpartition(weights, -n_keep)[-n_keep:]
    graph.add_weighted_edges_from(
        (int(row[index]), int(column[index]), float(weights[index]))
        for index in keep
    )
    return graph


def _best_louvain(
    graph: nx.Graph,
    *,
    resolution: float,
    repeats: int,
    random_state: int,
) -> tuple[list[set[int]], float]:
    if graph.number_of_edges() == 0:
        return [set(graph.nodes())], np.nan
    best_communities = None
    best_q = -np.inf
    for repeat in range(repeats):
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=random_state + repeat,
        )
        q_value = nx.community.modularity(
            graph,
            communities,
            weight="weight",
            resolution=resolution,
        )
        if q_value > best_q:
            best_q = float(q_value)
            best_communities = [set(module) for module in communities]
    return best_communities, best_q


def _labels_from_communities(communities, n_nodes: int) -> np.ndarray:
    labels = np.full(n_nodes, -1, dtype=int)
    for module_id, nodes in enumerate(communities):
        labels[list(nodes)] = module_id
    if np.any(labels < 0):
        raise ValueError("The FC partition does not cover every node.")
    return labels


def _communities_from_labels(labels) -> list[set[int]]:
    labels = np.asarray(labels)
    return [set(np.flatnonzero(labels == label)) for label in np.unique(labels)]


def _modularity(graph: nx.Graph, labels, resolution: float) -> float:
    if graph.number_of_edges() == 0:
        return np.nan
    return float(
        nx.community.modularity(
            graph,
            _communities_from_labels(labels),
            weight="weight",
            resolution=resolution,
        )
    )


def _partition_zscore(
    graph: nx.Graph,
    labels,
    *,
    resolution: float,
    n_null: int,
    random_state: int,
) -> tuple[float, float, float]:
    observed = _modularity(graph, labels, resolution)
    if n_null <= 1 or not np.isfinite(observed):
        return observed, np.nan, np.nan
    rng = np.random.default_rng(random_state)
    null_values = np.asarray(
        [
            _modularity(graph, rng.permutation(labels), resolution)
            for _ in range(n_null)
        ],
        dtype=float,
    )
    null_mean = float(np.nanmean(null_values))
    null_sd = float(np.nanstd(null_values, ddof=1))
    z_score = (observed - null_mean) / null_sd if null_sd > 0 else np.nan
    return observed, float(z_score), null_mean


def calculate_modularity_summary(
    matrices: dict[str, np.ndarray],
    fc_communities,
    *,
    fc_graph: nx.Graph | None = None,
    graph_mode: str = "knn",
    graph_k: int = 50,
    system: str,
    group: str,
    edge_density: float,
    resolution: float,
    louvain_repeats: int,
    n_null: int,
    random_state: int,
) -> pd.DataFrame:
    """Calculate native Q and support for the exact FC diamond partition."""
    labels = _labels_from_communities(fc_communities, matrices["FC"].shape[0])
    rows = []
    for offset, matrix_name in enumerate(("FC", "IS", "SC")):
        if graph_mode == "knn":
            graph = _display_knn_graph(matrices[matrix_name], graph_k)
        elif graph_mode == "density":
            graph = (
                fc_graph.copy()
                if matrix_name == "FC" and fc_graph is not None
                else _positive_density_graph(matrices[matrix_name], edge_density)
            )
        else:
            raise ValueError("graph_mode must be 'knn' or 'density'.")
        if matrix_name == "FC":
            native_q = _modularity(graph, labels, resolution)
            native_modules = len(fc_communities)
        else:
            native_partition, native_q = _best_louvain(
                graph,
                resolution=resolution,
                repeats=louvain_repeats,
                random_state=random_state + 100 * (offset + 1),
            )
            native_modules = len(native_partition)
        fc_partition_q, fc_partition_z, null_mean = _partition_zscore(
            graph,
            labels,
            resolution=resolution,
            n_null=n_null,
            random_state=random_state + 1000 * (offset + 1),
        )
        rows.append(
            {
                "group": group,
                "system": system,
                "matrix": matrix_name,
                "native_Q": native_q,
                "native_n_modules": native_modules,
                "FC_partition_Q": fc_partition_q,
                "FC_partition_Z": fc_partition_z,
                "FC_partition_null_mean": null_mean,
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "n_FC_modules": len(fc_communities),
                "edge_density": edge_density,
                "graph_mode": graph_mode,
                "graph_k": graph_k if graph_mode == "knn" else np.nan,
                "resolution": resolution,
            }
        )
    return pd.DataFrame(rows)


def _select_display_nodes(
    labels,
    max_nodes: int,
    max_modules: int,
    random_state: int,
) -> np.ndarray:
    """Use the legacy module-aware sampling rule for network display."""
    labels = np.asarray(labels)
    if labels.size <= max_nodes:
        return np.arange(labels.size)
    rng = np.random.default_rng(random_state)
    unique, counts = np.unique(labels, return_counts=True)
    ranked = unique[np.argsort(counts)[::-1]]
    keep_modules = ranked[:max_modules]
    per_module = max(8, max_nodes // len(keep_modules))
    selected = []
    for label in keep_modules:
        nodes = np.flatnonzero(labels == label)
        selected.extend(
            rng.choice(nodes, size=min(per_module, len(nodes)), replace=False)
        )
    if len(selected) < max_nodes:
        remaining = np.setdiff1d(np.arange(labels.size), np.asarray(selected))
        n_extra = min(max_nodes - len(selected), len(remaining))
        if n_extra:
            selected.extend(rng.choice(remaining, size=n_extra, replace=False))
    return np.asarray(sorted(selected[:max_nodes]), dtype=int)


def _display_knn_graph(matrix, k: int) -> nx.Graph:
    """Build a readable positive k-nearest-neighbour display graph."""
    matrix = np.asarray(matrix, dtype=float)
    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    k = max(1, min(int(k), matrix.shape[0] - 1))
    for node, row in enumerate(matrix):
        weights = row.copy()
        weights[node] = 0.0
        positive = np.flatnonzero(np.isfinite(weights) & (weights > 0))
        if positive.size > k:
            positive = positive[np.argpartition(weights[positive], -k)[-k:]]
        for neighbour in positive:
            weight = float(weights[neighbour])
            if graph.has_edge(node, int(neighbour)):
                graph[node][int(neighbour)]["weight"] = max(
                    graph[node][int(neighbour)]["weight"],
                    weight,
                )
            else:
                graph.add_edge(node, int(neighbour), weight=weight)
    return graph


def _diamond_fc_display_graph(
    diamond_graph: nx.Graph,
    selected_nodes: np.ndarray,
) -> nx.Graph:
    """Restrict the exact FC graph used by the diamond plot to displayed nodes."""
    node_to_local = {
        int(global_node): local_node
        for local_node, global_node in enumerate(selected_nodes)
    }
    graph = nx.Graph()
    graph.add_nodes_from(range(len(selected_nodes)))
    for source, target, data in diamond_graph.subgraph(selected_nodes).edges(data=True):
        graph.add_edge(
            node_to_local[int(source)],
            node_to_local[int(target)],
            weight=float(data.get("weight", 1.0)),
        )
    return graph


def _connected_layout_copy(graph: nx.Graph) -> nx.Graph:
    """
    Connect components only for coordinate calculation.

    The added weak edges are never drawn. This prevents isolated nodes from
    stretching the spring-layout scale and collapsing the main MLP component
    into an apparently empty point cloud.
    """
    layout_graph = graph.copy()
    components = sorted(
        nx.connected_components(layout_graph),
        key=len,
        reverse=True,
    )
    if len(components) <= 1:
        return layout_graph

    positive_weights = np.asarray(
        [
            float(data.get("weight", 1.0))
            for _, _, data in layout_graph.edges(data=True)
            if float(data.get("weight", 1.0)) > 0
        ],
        dtype=float,
    )
    typical_weight = (
        float(np.median(positive_weights))
        if positive_weights.size
        else 1.0
    )
    anchor = max(
        components[0],
        key=lambda node: layout_graph.degree(node),
    )
    for component in components[1:]:
        representative = max(
            component,
            key=lambda node: layout_graph.degree(node),
        )
        layout_graph.add_edge(
            anchor,
            representative,
            weight=0.25 * typical_weight,
            layout_only=True,
        )
    return layout_graph


def plot_fc_module_networks(
    matrices: dict[str, np.ndarray],
    diamond_result: dict,
    modularity_summary: pd.DataFrame,
    *,
    system: str,
    max_nodes: int = 140,
    max_modules: int = 5,
    k: int = 6,
    layout: str = "spring",
    layout_k: float | None = 0.55,
    random_state: int = 42,
    savepath=None,
):
    """Plot Figure 2g-style networks using the exact diamond FC modules."""
    labels_full = _labels_from_communities(
        diamond_result["communities"],
        matrices["FC"].shape[0],
    )
    nodes = _select_display_nodes(
        labels_full,
        max_nodes,
        max_modules,
        random_state,
    )
    labels = labels_full[nodes]
    module_ids = np.unique(labels)
    palette_values = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(module_ids))))
    palette = {module_id: palette_values[index] for index, module_id in enumerate(module_ids)}

    graphs = {
        "FC": _diamond_fc_display_graph(
            diamond_result["FC_graph"],
            nodes,
        )
    }
    for matrix_name in ("IS", "SC"):
        submatrix = matrices[matrix_name][np.ix_(nodes, nodes)]
        graphs[matrix_name] = _display_knn_graph(submatrix, k)
    layout_graph = _connected_layout_copy(graphs["FC"])
    layout = layout.lower()
    if layout == "spring":
        positions = nx.spring_layout(
            layout_graph,
            seed=random_state,
            weight="weight",
            iterations=800,
            k=layout_k,
        )
    elif layout in {"kamada_kawai", "kamada-kawai"}:
        positive_weights = np.asarray(
            [
                data["weight"]
                for _, _, data in layout_graph.edges(data=True)
                if data["weight"] > 0
            ],
            dtype=float,
        )
        scale = float(np.median(positive_weights)) if positive_weights.size else 1.0
        for _, _, data in layout_graph.edges(data=True):
            strength = max(float(data["weight"]) / max(scale, 1e-12), 1e-6)
            data["layout_distance"] = 1.0 / strength
        positions = nx.kamada_kawai_layout(
            layout_graph,
            weight="layout_distance",
        )
    else:
        raise ValueError("layout must be 'spring' or 'kamada_kawai'.")

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(10.2, 2.6),
        gridspec_kw={"width_ratios": [0.75, 1.0, 1.0, 1.0]},
    )
    q_values = (
        modularity_summary.set_index("matrix")
        .loc[["FC", "IS", "SC"], "FC_partition_Q"]
        .to_numpy(dtype=float)
    )
    axes[0].bar(
        ["FC", "IS", "SC"],
        q_values,
        color=["#2E8B57", "#E68632", "#6B7280"],
        width=0.68,
    )
    axes[0].set(
        title="Network modularity",
        ylabel="Q under FC-defined modules",
    )
    axes[0].spines[["top", "right"]].set_visible(False)

    node_colors = [palette[label] for label in labels]
    for axis, matrix_name in zip(axes[1:], ("FC", "IS", "SC")):
        graph = graphs[matrix_name]
        weights = np.asarray(
            [data["weight"] for _, _, data in graph.edges(data=True)],
            dtype=float,
        )
        if weights.size:
            widths = 0.15 + 1.1 * (weights - weights.min()) / (np.ptp(weights) + 1e-12)
        else:
            widths = []
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            width=widths,
            alpha=0.38,
            edge_color="0.55",
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_size=14,
            node_color=node_colors,
            linewidths=0,
        )
        axis.set_title(f"{matrix_name} network")
        axis.set_axis_off()

    figure.suptitle(
        f"{system}: one FC partition shared by matrices and networks",
        fontsize=10,
    )
    figure.tight_layout()
    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    return figure


def run_figure2_system(
    system: str,
    fc_matrix,
    sc_matrix,
    *,
    is_matrix=None,
    group: str,
    output_dir="./fig/Figure2",
    color: str | None = None,
    num_bins: int = 15,
    edge_density: float = 0.10,
    resolution: float = 1.0,
    louvain_repeats: int = 50,
    min_module_size: float | int = 0.05,
    diamond_sc_range=None,
    diamond_robust_percentiles=(2.0, 98.0),
    diamond_fc_cmap: str = "RdBu_r",
    diamond_is_cmap: str = "PRGn",
    diamond_sc_cmap: str | None = "viridis",
    diamond_figsize=(7.2, 4.6),
    random_state: int = 42,
    show: bool = True,
) -> dict:
    """Create only the Figure 2 trend and FC-module matrix comparison."""
    color = color or SYSTEM_COLORS.get(system, "#438c73")
    matrices = prepare_system_matrices(fc_matrix, sc_matrix, is_matrix=is_matrix)
    output_dir = Path(output_dir)
    system_dir = output_dir / _slug(system)
    system_dir.mkdir(parents=True, exist_ok=True)

    trend_r = calculate_corr_sp_boxplot_trend(
        matrices["IS"],
        "Input similarity (IS)",
        matrices["FC"],
        "Functional correlation (FC)",
        num_bins=num_bins,
        savepath=system_dir / "FC_IS_trend.svg",
        color=color,
        title=system,
    )

    diamond_result = plot_fc_is_fc_sc_diamonds(
        matrices["FC"],
        matrices["IS"],
        matrices["SC"],
        edge_density=edge_density,
        resolution=resolution,
        louvain_repeats=louvain_repeats,
        min_module_size=min_module_size,
        random_state=random_state,
        robust_percentiles=diamond_robust_percentiles,
        fc_cmap=diamond_fc_cmap,
        is_cmap=diamond_is_cmap,
        sc_cmap=diamond_sc_cmap,
        sc_range=diamond_sc_range,
        figsize=diamond_figsize,
        savepath=None,
    )
    pearson_fc_is, pearson_fc_is_p = matrix_correlation(
        matrices["FC"], matrices["IS"], method="pearson"
    )
    pearson_fc_sc, pearson_fc_sc_p = matrix_correlation(
        matrices["FC"], matrices["SC"], method="pearson"
    )
    diamond_result["fig"].suptitle(system, fontsize=11, y=0.995)
    diamond_result["axes"][0].set_title(
        f"FC-IS correspondence (Pearson r = {pearson_fc_is:.2f})",
        fontsize=9,
        pad=2,
    )
    diamond_result["axes"][1].set_title(
        f"FC-SC correspondence (Pearson r = {pearson_fc_sc:.2f})",
        fontsize=9,
        pad=2,
    )
    diamond_result["fig"].savefig(
        system_dir / "FC_IS_SC_diamonds.pdf",
        bbox_inches="tight",
        dpi=1200
    )

    if show:
        plt.show()
    return {
        "system": system,
        "group": group,
        "matrices": matrices,
        "trend_spearman_r": trend_r,
        "diamond": diamond_result,
        "FC_IS_pearson_r": pearson_fc_is,
        "FC_IS_pearson_p": pearson_fc_is_p,
        "FC_SC_pearson_r": pearson_fc_sc,
        "FC_SC_pearson_p": pearson_fc_sc_p,
    }


def summaries_to_frame(results: dict[str, dict]) -> pd.DataFrame:
    """Combine modularity tables from workflows that explicitly created them."""
    if not results:
        return pd.DataFrame()
    tables = [
        result["modularity"]
        for result in results.values()
        if result.get("modularity") is not None
    ]
    if not tables:
        return pd.DataFrame()
    return pd.concat(
        tables,
        ignore_index=True,
    )


def plot_cross_system_summary(summary, *, savepath=None):
    """Plot Figure 2f/h-style correlation and FC-module-alignment radars."""
    summary = pd.DataFrame(summary).copy()
    system_order = [
        "C. elegans",
        "MLP",
        "CNN",
        "Transformer",
        "Human",
        "Marmoset",
        "Mouse",
    ]
    available = [system for system in system_order if system in set(summary["system"])]
    if len(available) < 3:
        raise ValueError("At least three systems are required for a radar summary.")

    per_system = summary.groupby("system", as_index=True).first()
    z_scores = summary.pivot_table(
        index="system",
        columns="matrix",
        values="FC_partition_Z",
        aggfunc="first",
    )
    denominator = z_scores["IS"].abs() + z_scores["SC"].abs()
    contrast = (z_scores["IS"] - z_scores["SC"]) / denominator.replace(0, np.nan)
    contrast = contrast.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    alignment_is = (1 + contrast) / 2
    alignment_sc = (1 - contrast) / 2

    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.8),
        subplot_kw={"projection": "polar"},
    )
    panels = (
        (
            axes[0],
            "FC-SC / FC-IS correlation",
            [
                np.nan_to_num(per_system.loc[system, "FC_IS_pearson_r"], nan=0.0)
                for system in available
            ],
            [
                np.nan_to_num(per_system.loc[system, "FC_SC_pearson_r"], nan=0.0)
                for system in available
            ],
            "FC-IS correlation",
            "FC-SC correlation",
        ),
        (
            axes[1],
            "Alignment with FC-defined modules",
            [alignment_is.loc[system] for system in available],
            [alignment_sc.loc[system] for system in available],
            "FC-IS module alignment",
            "FC-SC module alignment",
        ),
    )
    for axis, title, is_values, sc_values, is_label, sc_label in panels:
        is_closed = np.r_[is_values, is_values[0]]
        sc_closed = np.r_[sc_values, sc_values[0]]
        axis.plot(closed_angles, is_closed, color="#3F88B5", marker="o", label=is_label)
        axis.fill(closed_angles, is_closed, color="#3F88B5", alpha=0.14)
        axis.plot(closed_angles, sc_closed, color="#D45A48", marker="o", label=sc_label)
        axis.fill(closed_angles, sc_closed, color="#D45A48", alpha=0.14)
        axis.set_xticks(angles)
        axis.set_xticklabels(available, fontsize=8)
        axis.set_ylim(0, 1)
        axis.set_title(title, fontsize=10, pad=16)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=7)
    figure.tight_layout()
    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    return figure
