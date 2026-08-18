"""Module-level coarse-graining for manuscript Figure 2.

FC is used to define the common mesoscale coordinate system. FC, IS, and SC
are then averaged within every pair of FC modules. Correspondence is measured
by directly correlating all finite entries of the aligned module-by-module
matrices, including the within-module diagonal entries.

Modules are detected with weighted Louvain clustering over a positive k-nearest
neighbour FC graph. Resolution gamma is scanned instead of forcing an exact
module count. Very small communities are merged into the functionally closest
community before coarse-graining, avoiding unstable singleton blocks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from .analysis import as_square_matrix


SYSTEM_ORDER = [
    "MLP",
    "CNN",
    "Transformer",
    "C. elegans",
    "Mouse",
    "Marmoset",
    "Human",
]


def _relabel_by_first_node(labels: np.ndarray) -> np.ndarray:
    """Relabel modules deterministically by their first node."""
    labels = np.asarray(labels, dtype=int)
    ordered = sorted(
        np.unique(labels),
        key=lambda value: np.flatnonzero(labels == value)[0],
    )
    mapping = {old: new for new, old in enumerate(ordered)}
    return np.asarray([mapping[value] for value in labels], dtype=int)


def _resolve_min_module_size(value: int | float, n_nodes: int) -> int:
    """Convert a node count or node fraction into an integer threshold."""
    if isinstance(value, (float, np.floating)) and 0 < value < 1:
        return min(n_nodes, max(3, int(np.ceil(float(value) * n_nodes))))
    threshold = int(value)
    if threshold < 2:
        raise ValueError("min_module_size must be >=2 or a fraction in (0, 1).")
    return threshold


def build_positive_knn_graph(
    fc_matrix,
    *,
    graph_k: int = 30,
) -> tuple[nx.Graph, dict]:
    """Build a sparse undirected graph from each node's strongest positive FCs.

    The effective neighbourhood size is capped at approximately sqrt(N). This
    keeps small biological networks from becoming almost complete while
    limiting memory use for large ANN matrices. If either endpoint selects an
    edge, the edge is retained; duplicate directed selections use the larger
    positive weight.
    """
    fc = as_square_matrix(fc_matrix, "FC", copy=False)
    n_nodes = fc.shape[0]
    if n_nodes < 3:
        raise ValueError("At least three nodes are required for module analysis.")
    requested_k = int(graph_k)
    if requested_k < 1:
        raise ValueError("graph_k must be a positive integer.")
    effective_k = min(requested_k, n_nodes - 1, max(5, int(round(np.sqrt(n_nodes)))))

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for node in range(n_nodes):
        row = np.asarray(fc[node], dtype=float)
        valid = np.isfinite(row) & (row > 0)
        valid[node] = False
        candidates = np.flatnonzero(valid)
        if candidates.size > effective_k:
            local = np.argpartition(row[candidates], -effective_k)[-effective_k:]
            candidates = candidates[local]
        for neighbour in candidates:
            weight = float(row[neighbour])
            if graph.has_edge(node, int(neighbour)):
                if weight > graph[node][int(neighbour)]["weight"]:
                    graph[node][int(neighbour)]["weight"] = weight
            else:
                graph.add_edge(node, int(neighbour), weight=weight)

    if graph.number_of_edges() == 0:
        raise ValueError("FC contains no finite positive off-diagonal weights.")
    metadata = {
        "graph_k_requested": requested_k,
        "graph_k_effective": effective_k,
        "graph_n_edges": int(graph.number_of_edges()),
        "graph_density": float(nx.density(graph)),
        "graph_n_components": int(nx.number_connected_components(graph)),
    }
    return graph, metadata


def _labels_from_communities(communities, n_nodes: int) -> np.ndarray:
    labels = np.full(n_nodes, -1, dtype=int)
    ordered = sorted(
        (sorted(map(int, community)) for community in communities),
        key=lambda community: community[0],
    )
    for module, nodes in enumerate(ordered):
        labels[nodes] = module
    if np.any(labels < 0):
        raise RuntimeError("Community assignment did not include every node.")
    return labels


def _communities_from_labels(labels: np.ndarray) -> list[set[int]]:
    return [
        set(map(int, np.flatnonzero(labels == module)))
        for module in np.unique(labels)
    ]


def _merge_small_modules(
    labels: np.ndarray,
    fc_matrix,
    *,
    min_module_size: int,
) -> tuple[np.ndarray, dict]:
    """Merge undersized modules into the module with strongest mean FC.

    Positive FC determines the preferred target; mean raw FC, target size and
    module order break ties. Diagnostics quantify how strongly post-processing
    changed the raw Louvain partition.
    """
    fc = as_square_matrix(fc_matrix, "FC", copy=False)
    labels = _relabel_by_first_node(labels)
    moved_nodes = np.zeros(labels.size, dtype=bool)
    merge_steps = 0

    while True:
        modules, sizes = np.unique(labels, return_counts=True)
        small = modules[sizes < min_module_size]
        if small.size == 0 or modules.size <= 1:
            break

        # Merge the smallest community first; the first-node order breaks ties.
        small_sizes = {module: int(np.sum(labels == module)) for module in small}
        source = min(small, key=lambda module: (small_sizes[module], int(module)))
        source_nodes = np.flatnonzero(labels == source)

        best_target = None
        best_positive_score = -np.inf
        best_raw_score = -np.inf
        best_size = -1
        for target in modules:
            if target == source:
                continue
            target_nodes = np.flatnonzero(labels == target)
            values = np.asarray(fc[np.ix_(source_nodes, target_nodes)], dtype=float)
            finite = values[np.isfinite(values)]
            positive_score = (
                float(np.mean(np.clip(finite, 0.0, None)))
                if finite.size else -np.inf
            )
            raw_score = float(np.mean(finite)) if finite.size else -np.inf
            target_size = int(target_nodes.size)
            if (positive_score, raw_score, target_size, -int(target)) > (
                best_positive_score,
                best_raw_score,
                best_size,
                -int(best_target) if best_target is not None else -np.inf,
            ):
                best_target = int(target)
                best_positive_score = positive_score
                best_raw_score = raw_score
                best_size = target_size

        if best_target is None:
            break
        moved_nodes[source_nodes] = True
        labels[labels == source] = best_target
        labels = _relabel_by_first_node(labels)
        merge_steps += 1

    diagnostics = {
        "n_merge_steps": int(merge_steps),
        "n_nodes_merged": int(np.sum(moved_nodes)),
        "fraction_nodes_merged": float(np.mean(moved_nodes)),
    }
    return labels, diagnostics


def detect_fc_modules_louvain(
    fc_matrix,
    *,
    resolution: float = 1.0,
    graph_k: int = 30,
    repeats: int = 10,
    min_module_size: int | float = 0.01,
    random_state: int = 42,
    graph: nx.Graph | None = None,
    graph_metadata: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Detect robust FC modules with repeated weighted Louvain clustering."""
    fc = as_square_matrix(fc_matrix, "FC", copy=False)
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if int(repeats) < 1:
        raise ValueError("repeats must be >=1.")
    if graph is None:
        graph, graph_metadata = build_positive_knn_graph(fc, graph_k=graph_k)
    graph_metadata = dict(graph_metadata or {})

    best_communities = None
    best_modularity = -np.inf
    for repeat in range(int(repeats)):
        communities = nx.algorithms.community.louvain_communities(
            graph,
            weight="weight",
            resolution=float(resolution),
            seed=int(random_state) + repeat,
        )
        modularity = nx.algorithms.community.modularity(
            graph,
            communities,
            weight="weight",
            resolution=float(resolution),
        )
        if modularity > best_modularity:
            best_communities = communities
            best_modularity = float(modularity)

    labels_raw = _labels_from_communities(best_communities, fc.shape[0])
    threshold = _resolve_min_module_size(min_module_size, fc.shape[0])
    labels, merge_metadata = _merge_small_modules(
        labels_raw,
        fc,
        min_module_size=threshold,
    )
    merged_communities = _communities_from_labels(labels)
    modularity_after = nx.algorithms.community.modularity(
        graph,
        merged_communities,
        weight="weight",
        resolution=float(resolution),
    )
    sizes = np.bincount(labels)
    metadata = {
        **graph_metadata,
        "resolution": float(resolution),
        "louvain_repeats": int(repeats),
        "random_state": int(random_state),
        "raw_n_modules": int(len(best_communities)),
        "actual_n_modules": int(len(sizes)),
        "min_module_size_threshold": int(threshold),
        "modularity_before_merge": float(best_modularity),
        "modularity_after_merge": float(modularity_after),
        **merge_metadata,
    }
    return labels, metadata


def coarse_grain_matrix(matrix, labels) -> tuple[np.ndarray, np.ndarray]:
    """Average a node-level matrix within every pair of modules."""
    matrix = as_square_matrix(matrix, copy=False)
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or labels.size != matrix.shape[0]:
        raise ValueError("labels must contain one module label per matrix node.")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        labels = _relabel_by_first_node(labels)
        unique = np.unique(labels)

    n_modules = len(unique)
    block = np.full((n_modules, n_modules), np.nan, dtype=float)
    counts = np.zeros((n_modules, n_modules), dtype=int)
    module_nodes = [np.flatnonzero(labels == module) for module in unique]

    for first in range(n_modules):
        for second in range(first, n_modules):
            nodes_first = module_nodes[first]
            nodes_second = module_nodes[second]
            values = matrix[np.ix_(nodes_first, nodes_second)]
            if first == second:
                if len(nodes_first) < 2:
                    selected = np.asarray([], dtype=float)
                else:
                    selected = values[np.triu_indices(len(nodes_first), k=1)]
            else:
                selected = values.ravel()
            selected = selected[np.isfinite(selected)]
            if selected.size:
                mean_value = float(np.mean(selected))
                block[first, second] = mean_value
                block[second, first] = mean_value
                counts[first, second] = selected.size
                counts[second, first] = selected.size
    return block, counts


def _paired_entries(first, second) -> tuple[np.ndarray, np.ndarray]:
    first = as_square_matrix(first, copy=False)
    second = as_square_matrix(second, copy=False)
    if first.shape != second.shape:
        raise ValueError("Module matrices must have matching shapes.")
    x = first.ravel()
    y = second.ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def correlate_module_matrices(
    first,
    second,
    *,
    method: str = "pearson",
) -> tuple[float, float, int]:
    """Correlate all aligned finite entries of two coarse matrices."""
    x, y = _paired_entries(first, second)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, int(x.size)
    if method == "pearson":
        statistic, p_value = pearsonr(x, y)
    elif method == "spearman":
        statistic, p_value = spearmanr(x, y)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    return float(statistic), float(p_value), int(x.size)


def module_pair_weights(labels) -> np.ndarray:
    """Return the number of unit pairs represented by each module block."""
    labels = _relabel_by_first_node(np.asarray(labels, dtype=int))
    sizes = np.bincount(labels).astype(float)
    weights = np.outer(sizes, sizes)
    np.fill_diagonal(weights, sizes * (sizes - 1.0) / 2.0)
    return weights


def correlate_module_matrices_weighted(
    first,
    second,
    weights,
) -> tuple[float, int, float]:
    """Weighted Pearson correlation across aligned module-pair means.

    Weights are normally the number of underlying unit pairs in each block.
    The effective sample size is returned as a diagnostic; it is not used to
    calculate a parametric p-value because module pairs are not independent.
    """
    first = as_square_matrix(first, copy=False)
    second = as_square_matrix(second, copy=False)
    weights = as_square_matrix(weights, "weights", copy=False)
    if first.shape != second.shape or first.shape != weights.shape:
        raise ValueError("Matrices and weights must have matching shapes.")
    x = first.ravel()
    y = second.ravel()
    w = weights.ravel()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[valid], y[valid], w[valid]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, int(x.size), np.nan
    w = w / np.sum(w)
    x_centered = x - np.sum(w * x)
    y_centered = y - np.sum(w * y)
    covariance = np.sum(w * x_centered * y_centered)
    variance_x = np.sum(w * x_centered**2)
    variance_y = np.sum(w * y_centered**2)
    denominator = np.sqrt(variance_x * variance_y)
    correlation = covariance / denominator if denominator > 0 else np.nan
    effective_n = 1.0 / np.sum(w**2)
    return float(correlation), int(x.size), float(effective_n)


def _fisher_z(value: float, eps: float = 1e-7) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.arctanh(np.clip(value, -1 + eps, 1 - eps)))


def module_segregation(block_matrix) -> float:
    """Return mean within-module minus mean between-module block strength."""
    block = as_square_matrix(block_matrix, copy=False)
    diagonal = np.diag(block)
    between = block[np.triu_indices(block.shape[0], k=1)]
    if not np.any(np.isfinite(diagonal)) or not np.any(np.isfinite(between)):
        return np.nan
    return float(np.nanmean(diagonal) - np.nanmean(between))


def analyze_one_partition(
    *,
    system: str,
    group: str,
    matrices: dict[str, np.ndarray],
    resolution: float,
    clustering_fc=None,
    evaluation_fc=None,
    graph_k: int = 30,
    louvain_repeats: int = 10,
    min_module_size: int | float = 0.01,
    random_state: int = 42,
    graph: nx.Graph | None = None,
    graph_metadata: dict | None = None,
) -> tuple[dict, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Run one Louvain-resolution analysis for one system."""
    clustering_fc = matrices["FC"] if clustering_fc is None else clustering_fc
    evaluation_fc = matrices["FC"] if evaluation_fc is None else evaluation_fc
    labels, clustering_metadata = detect_fc_modules_louvain(
        clustering_fc,
        resolution=resolution,
        graph_k=graph_k,
        repeats=louvain_repeats,
        min_module_size=min_module_size,
        random_state=random_state,
        graph=graph,
        graph_metadata=graph_metadata,
    )
    sizes = np.bincount(labels)

    analysis_matrices = {
        "FC": np.asarray(evaluation_fc, dtype=float),
        "IS": np.asarray(matrices["IS"], dtype=float),
        "SC": np.asarray(matrices["SC"], dtype=float),
    }
    blocks: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for name, matrix in analysis_matrices.items():
        blocks[name], counts[name] = coarse_grain_matrix(matrix, labels)

    r_fi, p_fi, n_entries = correlate_module_matrices(
        blocks["FC"], blocks["IS"], method="pearson"
    )
    r_fs, p_fs, _ = correlate_module_matrices(
        blocks["FC"], blocks["SC"], method="pearson"
    )
    rho_fi, rho_p_fi, _ = correlate_module_matrices(
        blocks["FC"], blocks["IS"], method="spearman"
    )
    rho_fs, rho_p_fs, _ = correlate_module_matrices(
        blocks["FC"], blocks["SC"], method="spearman"
    )
    pair_weights = module_pair_weights(labels)
    weighted_r_fi, weighted_n_pairs, weighted_effective_n = (
        correlate_module_matrices_weighted(
            blocks["FC"],
            blocks["IS"],
            pair_weights,
        )
    )
    weighted_r_fs, _, _ = correlate_module_matrices_weighted(
        blocks["FC"],
        blocks["SC"],
        pair_weights,
    )
    z_fi = _fisher_z(r_fi)
    z_fs = _fisher_z(r_fs)
    weighted_z_fi = _fisher_z(weighted_r_fi)
    weighted_z_fs = _fisher_z(weighted_r_fs)
    finite_primary = np.isfinite(r_fi) and np.isfinite(r_fs)
    threshold = clustering_metadata["min_module_size_threshold"]

    row = {
        "group": group,
        "system": system,
        "resolution": float(resolution),
        "raw_n_modules": clustering_metadata["raw_n_modules"],
        "actual_n_modules": int(len(sizes)),
        "n_nodes": int(len(labels)),
        "min_module_size_threshold": int(threshold),
        "min_module_size": int(sizes.min()),
        "max_module_size": int(sizes.max()),
        "module_sizes": ";".join(map(str, sizes.tolist())),
        "valid_partition": bool(
            len(sizes) >= 3 and sizes.min() >= threshold and finite_primary
        ),
        "n_module_matrix_entries": int(n_entries),
        "FC_IS_pearson": r_fi,
        "FC_IS_pearson_p": p_fi,
        "FC_SC_pearson": r_fs,
        "FC_SC_pearson_p": p_fs,
        "FC_IS_spearman": rho_fi,
        "FC_IS_spearman_p": rho_p_fi,
        "FC_SC_spearman": rho_fs,
        "FC_SC_spearman_p": rho_p_fs,
        "FC_IS_pearson_weighted": weighted_r_fi,
        "FC_SC_pearson_weighted": weighted_r_fs,
        "delta_fisher_z_weighted_IS_minus_SC": (
            weighted_z_fi - weighted_z_fs
        ),
        "weighted_n_module_pairs": int(weighted_n_pairs),
        "weighted_effective_n": weighted_effective_n,
        "FC_IS_fisher_z": z_fi,
        "FC_SC_fisher_z": z_fs,
        "delta_fisher_z_IS_minus_SC": z_fi - z_fs,
        "FC_module_segregation": module_segregation(blocks["FC"]),
        "IS_module_segregation": module_segregation(blocks["IS"]),
        "SC_module_segregation": module_segregation(blocks["SC"]),
        "all_module_matrix_entries_included": True,
        **clustering_metadata,
    }
    return row, blocks, labels, counts


def run_multiresolution_module_analysis(
    datasets: list[dict],
    *,
    resolutions=(0.75, 1.0, 1.25, 1.5, 2.0, 2.5),
    graph_k: int = 30,
    louvain_repeats: int = 10,
    min_module_size: int | float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Analyze every system across Louvain resolution gamma values.

    Each dataset must contain ``group``, ``system``, ``FC``, ``IS``, and
    ``SC``. Optional ``FC_cluster`` and ``FC_eval`` keys support split-half
    analysis. The sparse FC graph is constructed once per system and reused at
    every resolution.
    """
    rows = []
    block_rows = []
    store = {}
    for dataset in datasets:
        system = dataset["system"]
        store[system] = {}
        matrices = {name: dataset[name] for name in ("FC", "IS", "SC")}
        clustering_fc = dataset.get("FC_cluster", dataset["FC"])
        graph, graph_metadata = build_positive_knn_graph(
            clustering_fc,
            graph_k=graph_k,
        )
        for resolution in resolutions:
            row, blocks, labels, counts = analyze_one_partition(
                system=system,
                group=dataset["group"],
                matrices=matrices,
                resolution=float(resolution),
                clustering_fc=clustering_fc,
                evaluation_fc=dataset.get("FC_eval"),
                graph_k=graph_k,
                louvain_repeats=louvain_repeats,
                min_module_size=min_module_size,
                random_state=random_state,
                graph=graph,
                graph_metadata=graph_metadata,
            )
            rows.append(row)
            store[system][float(resolution)] = {
                "blocks": blocks,
                "counts": counts,
                "labels": labels,
                "row": row,
            }
            for matrix_name, block in blocks.items():
                for first in range(block.shape[0]):
                    for second in range(first, block.shape[1]):
                        block_rows.append(
                            {
                                "group": dataset["group"],
                                "system": system,
                                "resolution": float(resolution),
                                "actual_n_modules": int(block.shape[0]),
                                "matrix": matrix_name,
                                "module_a": int(first),
                                "module_b": int(second),
                                "within_module": bool(first == second),
                                "mean_weight": float(block[first, second]),
                                "n_unit_pairs": int(counts[matrix_name][first, second]),
                                "module_a_size": int(np.sum(labels == first)),
                                "module_b_size": int(np.sum(labels == second)),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(block_rows), store


def select_target_module_partitions(
    summary: pd.DataFrame,
    *,
    target_module_counts=(8, 10, 12, 15),
    max_merge_fraction: float = 0.15,
    max_module_count_error: int = 2,
) -> pd.DataFrame:
    """Select comparable, distinct partitions nearest requested module counts.

    Selection uses only valid partitions that move no more than
    ``max_merge_fraction`` of nodes during small-module merging.
    Greedy one-to-one matching prioritizes exact and near-exact K values; a
    target is omitted rather than represented by a partition farther than
    ``max_module_count_error``. This selection never uses FC-IS or FC-SC
    correspondence, so the outcome cannot determine the partition.
    """
    summary = pd.DataFrame(summary).copy()
    required = {
        "group",
        "system",
        "resolution",
        "actual_n_modules",
        "valid_partition",
        "fraction_nodes_merged",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    targets = np.asarray(tuple(map(int, target_module_counts)), dtype=int)
    if targets.size == 0 or len(np.unique(targets)) != targets.size:
        raise ValueError("target_module_counts must contain distinct integers.")

    selected_frames = []
    for (_, _), system_frame in summary.groupby(["group", "system"], sort=False):
        candidates = system_frame.loc[system_frame["valid_partition"]].copy()
        if candidates.empty:
            continue
        candidates = candidates.loc[
            candidates["fraction_nodes_merged"] <= float(max_merge_fraction)
        ]
        if candidates.empty:
            continue

        # Retain the least altered partition for each achieved K.
        candidates = (
            candidates
            .sort_values(
                [
                    "actual_n_modules",
                    "fraction_nodes_merged",
                    "modularity_after_merge",
                ],
                ascending=[True, True, False],
            )
            .drop_duplicates("actual_n_modules", keep="first")
            .reset_index(drop=True)
        )
        if candidates.empty:
            continue

        candidate_counts = candidates["actual_n_modules"].to_numpy(dtype=int)
        candidate_merge = candidates["fraction_nodes_merged"].to_numpy(dtype=float)
        candidate_modularity = candidates["modularity_after_merge"].to_numpy(dtype=float)
        possible_matches = []
        for target_index, target in enumerate(targets):
            for candidate_index, actual in enumerate(candidate_counts):
                error = abs(int(target) - int(actual))
                if error <= int(max_module_count_error):
                    possible_matches.append(
                        (
                            error,
                            candidate_merge[candidate_index],
                            -candidate_modularity[candidate_index],
                            target_index,
                            candidate_index,
                        )
                    )
        assigned_targets = set()
        assigned_candidates = set()
        accepted_matches = []
        for *_, target_index, candidate_index in sorted(possible_matches):
            if target_index in assigned_targets or candidate_index in assigned_candidates:
                continue
            assigned_targets.add(target_index)
            assigned_candidates.add(candidate_index)
            accepted_matches.append((target_index, candidate_index))

        for target_index, candidate_index in accepted_matches:
            row = candidates.iloc[[int(candidate_index)]].copy()
            row["target_n_modules"] = int(targets[int(target_index)])
            row["module_count_error"] = abs(
                int(row["actual_n_modules"].iloc[0])
                - int(targets[int(target_index)])
            )
            row["selection_within_merge_limit"] = bool(
                row["fraction_nodes_merged"].iloc[0] <= max_merge_fraction
            )
            selected_frames.append(row)

    if not selected_frames:
        return pd.DataFrame(columns=[*summary.columns, "target_n_modules"])
    selected = pd.concat(selected_frames, ignore_index=True)
    return selected.sort_values(
        ["group", "system", "target_n_modules"]
    ).reset_index(drop=True)


def select_resolution_by_module_count(
    system_store: dict,
    *,
    target_n_modules: int = 10,
) -> tuple[float, dict]:
    """Select the valid gamma giving the module count closest to a target."""
    candidates = [
        (resolution, result)
        for resolution, result in system_store.items()
        if result["row"]["valid_partition"]
    ]
    if not candidates:
        candidates = list(system_store.items())
    if not candidates:
        raise ValueError("No module partitions are available.")
    resolution, result = min(
        candidates,
        key=lambda item: (
            abs(item[1]["row"]["actual_n_modules"] - int(target_n_modules)),
            -item[1]["row"]["modularity_after_merge"],
            abs(float(item[0]) - 1.0),
        ),
    )
    return float(resolution), result


def _percentile_display(matrix) -> np.ndarray:
    """Convert finite matrix values to within-matrix percentiles for display."""
    matrix = np.asarray(matrix, dtype=float)
    output = np.full_like(matrix, np.nan, dtype=float)
    finite = np.isfinite(matrix)
    if np.any(finite):
        ranks = rankdata(matrix[finite], method="average")
        output[finite] = (ranks - 1) / max(1, len(ranks) - 1)
    return output


def plot_example_module_matrices(
    store: dict,
    *,
    systems=("MLP", "Mouse"),
    target_n_modules: int = 10,
    savepath=None,
):
    """Plot representative matrices at each system's gamma nearest target K."""
    available = [system for system in systems if system in store and store[system]]
    if not available:
        raise ValueError("None of the requested example systems are available.")
    figure, axes = plt.subplots(
        len(available),
        3,
        figsize=(7.8, max(3.0, 2.55 * len(available))),
        squeeze=False,
    )
    image = None
    for row_index, system in enumerate(available):
        resolution, result = select_resolution_by_module_count(
            store[system],
            target_n_modules=target_n_modules,
        )
        blocks = result["blocks"]
        stats = result["row"]
        n_modules = int(stats["actual_n_modules"])
        titles = {
            "FC": f"Module-level FC\nγ={resolution:g}, K={n_modules}",
            "IS": f"Module-level IS\nFC–IS r={stats['FC_IS_pearson']:.2f}",
            "SC": f"Module-level SC\nFC–SC r={stats['FC_SC_pearson']:.2f}",
        }
        for column_index, matrix_name in enumerate(("FC", "IS", "SC")):
            axis = axes[row_index, column_index]
            display = _percentile_display(blocks[matrix_name])
            image = axis.imshow(display, cmap="viridis", vmin=0, vmax=1)
            for diagonal in range(display.shape[0]):
                axis.add_patch(
                    Rectangle(
                        (diagonal - 0.5, diagonal - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.45,
                    )
                )
            axis.set_title(titles[matrix_name], fontsize=9)
            axis.set_xlabel("FC module")
            axis.set_ylabel("FC module" if column_index == 0 else "")
            tick_step = max(1, n_modules // 5)
            axis.set_xticks(range(0, n_modules, tick_step))
            axis.set_yticks(range(0, n_modules, tick_step))
            if column_index == 0:
                axis.text(
                    -0.33,
                    0.5,
                    system,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025)
    colorbar.set_label("Within-matrix percentile (display only)")
    figure.suptitle(
        "FC-defined coarse-grained matrices "
        f"(partition nearest K={target_n_modules}; all matrix entries included)",
        fontsize=10,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.10,
        top=0.76 if len(available) == 1 else 0.88,
        wspace=0.34,
        hspace=0.45,
    )
    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    return figure


def plot_resolution_robustness(
    summary: pd.DataFrame,
    *,
    system_order=SYSTEM_ORDER,
    value_column="delta_fisher_z_IS_minus_SC",
    x_column=None,
    savepath=None,
):
    """Plot FC-IS minus FC-SC correspondence across module scales."""
    summary = pd.DataFrame(summary).copy()
    if value_column not in summary:
        raise ValueError(f"Unknown value column: {value_column}")
    if x_column is None:
        x_column = (
            "target_n_modules"
            if "target_n_modules" in summary.columns
            else "resolution"
        )
    if x_column not in summary:
        raise ValueError(f"Unknown x-axis column: {x_column}")
    available = [system for system in system_order if system in set(summary["system"])]
    x_values = sorted(summary[x_column].unique())
    pivot = summary.pivot_table(
        index="system",
        columns=x_column,
        values=value_column,
        aggfunc="first",
    ).reindex(index=available, columns=x_values)
    module_counts = summary.pivot_table(
        index="system",
        columns=x_column,
        values="actual_n_modules",
        aggfunc="first",
    ).reindex(index=available, columns=x_values)
    pivot["Median"] = pivot.median(axis=1)

    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    maximum = max(0.1, float(np.max(np.abs(finite))) if finite.size else 1.0)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)

    figure, axis = plt.subplots(figsize=(max(5.8, 0.75 * (len(x_values) + 1)), 3.7))
    image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                text_color = "white" if abs(value) > 0.55 * maximum else "black"
                if column < len(x_values):
                    count_value = module_counts.iloc[row, column]
                    label = (
                        f"{value:.2f}\nK={int(count_value)}"
                        if np.isfinite(count_value)
                        else f"{value:.2f}"
                    )
                else:
                    label = f"{value:.2f}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color,
                )
    axis.axvline(len(x_values) - 0.5, color="black", linewidth=1.0)
    axis.set_xticks(range(len(x_values) + 1))
    axis.set_xticklabels([f"{value:g}" for value in x_values] + ["Median"])
    axis.set_yticks(range(len(available)))
    axis.set_yticklabels(available)
    if x_column == "target_n_modules":
        axis.set_xlabel("Target number of FC modules (actual K shown in cells)")
    else:
        axis.set_xlabel("Louvain resolution γ (actual K shown in cells)")
    weighting_note = (
        " (node-pair weighted sensitivity)"
        if "weighted" in value_column
        else " (module-pair primary analysis)"
    )
    axis.set_title(
        "Module-level FC–IS minus FC–SC correspondence"
        + weighting_note
        + "\n"
        r"$\Delta z=\mathrm{atanh}(r_{FC,IS})-\mathrm{atanh}(r_{FC,SC})$"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Δ Fisher z (positive: IS closer to FC)")
    figure.tight_layout()
    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    return figure
