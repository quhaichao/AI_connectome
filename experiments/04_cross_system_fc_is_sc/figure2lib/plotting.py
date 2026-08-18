"""Figure 2 trend and FC-partition diamond plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
from scipy.cluster.hierarchy import (
    leaves_list,
    linkage,
    optimal_leaf_ordering,
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr



def _prepare_matrix(matrix, name, symmetrize="mean"):
    matrix = np.asarray(matrix, dtype=float).copy()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")

    if symmetrize == "mean":
        matrix = np.nanmean(
            np.stack([matrix, matrix.T], axis=0),
            axis=0,
        )
    elif symmetrize == "max":
        matrix = np.fmax(matrix, matrix.T)
    elif symmetrize is not None:
        raise ValueError("symmetrize must be 'mean', 'max', or None.")

    matrix[~np.isfinite(matrix)] = 0.0
    return matrix


def _off_diagonal_values(matrix):
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[mask]
    return values[np.isfinite(values)]


def _robust_norm(
    matrix,
    percentiles=(2.0, 98.0),
    mode="auto",
    symmetric_if_diverging=True,
):
    """
    Estimate color limits from off-diagonal robust percentiles.

    Parameters
    ----------
    mode : {"auto", "diverging", "sequential"}
        auto:
            Use diverging normalization when robust values cross zero;
            otherwise use sequential normalization.
        diverging:
            Force a zero-centred diverging normalization.
        sequential:
            Use percentile-based sequential normalization.
    """
    values = _off_diagonal_values(matrix)

    if len(values) == 0:
        values = np.array([0.0, 1.0])

    low_q, high_q = percentiles
    robust_low, robust_high = np.nanpercentile(
        values,
        [low_q, high_q],
    )

    if mode == "auto":
        mode = (
            "diverging"
            if robust_low < 0 < robust_high
            else "sequential"
        )

    if mode == "diverging":
        if symmetric_if_diverging:
            vmax = max(abs(robust_low), abs(robust_high))
            vmax = max(vmax, 1e-12)
            vmin = -vmax
        else:
            vmin = min(robust_low, 0.0)
            vmax = max(robust_high, 0.0)

        norm = TwoSlopeNorm(
            vmin=vmin,
            vcenter=0.0,
            vmax=vmax,
        )

    elif mode == "sequential":
        # For nonnegative SC, retain the meaningful zero baseline.
        if np.nanmin(values) >= 0:
            vmin = 0.0
        else:
            vmin = robust_low

        vmax = robust_high

        if np.isclose(vmin, vmax):
            delta = max(abs(vmax) * 0.05, 1e-6)
            vmin -= delta
            vmax += delta

        norm = Normalize(
            vmin=vmin,
            vmax=vmax,
            clip=True,
        )

    else:
        raise ValueError(
            "mode must be 'auto', 'diverging', or 'sequential'."
        )

    return norm, (float(vmin), float(vmax)), mode


def _build_fc_graph(fc, edge_density=0.10):
    """
    Construct a graph from the strongest positive FC edges.

    edge_density is relative to all possible undirected edges.
    """
    n = fc.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    weights = fc[iu, ju]

    valid = np.isfinite(weights) & (weights > 0)
    iu = iu[valid]
    ju = ju[valid]
    weights = weights[valid]

    if len(weights) == 0:
        raise ValueError(
            "FC contains no positive off-diagonal values."
        )

    total_possible = n * (n - 1) // 2
    n_keep = max(
        1,
        int(np.ceil(edge_density * total_possible)),
    )
    n_keep = min(n_keep, len(weights))

    keep = np.argpartition(weights, -n_keep)[-n_keep:]

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(
        (
            int(iu[k]),
            int(ju[k]),
            float(weights[k]),
        )
        for k in keep
    )

    return graph


def _resolve_min_module_size(min_module_size, n_nodes):
    """
    Convert a fractional or integer minimum module size to an integer.

    Examples
    --------
    0.05 -> at least 5% of all nodes
    8    -> at least 8 nodes
    """
    if min_module_size is None:
        return 1

    if isinstance(min_module_size, (float, np.floating)):
        if not 0 < min_module_size <= 1:
            raise ValueError(
                "A fractional min_module_size must be in (0, 1]."
            )
        minimum = int(np.ceil(min_module_size * n_nodes))
    else:
        minimum = int(min_module_size)

    return max(1, min(minimum, n_nodes))


def _module_affinity(fc, source_nodes, target_nodes):
    """
    Measure how strongly a small module should be merged into another.

    Positive FC is prioritized. Mean raw FC is used as a tie-breaker.
    """
    block = fc[np.ix_(source_nodes, target_nodes)]

    if block.size == 0:
        return -np.inf, -np.inf

    positive_affinity = np.mean(np.clip(block, 0.0, None))
    raw_affinity = np.mean(block)

    return float(positive_affinity), float(raw_affinity)


def _merge_small_modules(fc, communities, min_module_size):
    """
    Iteratively merge small modules into the module with which they have
    the strongest mean positive FC.

    The procedure stops when every module satisfies min_module_size or
    only one module remains.
    """
    communities = [
        sorted(set(module))
        for module in communities
        if len(module) > 0
    ]

    while len(communities) > 1:
        sizes = np.array([len(module) for module in communities])
        small_indices = np.flatnonzero(sizes < min_module_size)

        if len(small_indices) == 0:
            break

        # Merge the smallest module first.
        source_idx = small_indices[np.argmin(sizes[small_indices])]
        source_nodes = communities[source_idx]

        candidates = [
            idx for idx in range(len(communities))
            if idx != source_idx
        ]

        affinities = [
            _module_affinity(
                fc,
                source_nodes,
                communities[target_idx],
            )
            for target_idx in candidates
        ]

        target_idx = candidates[max(
            range(len(candidates)),
            key=lambda k: affinities[k],
        )]

        communities[target_idx] = sorted(
            set(communities[target_idx]).union(source_nodes)
        )
        del communities[source_idx]

    return communities


def _detect_fc_modules(
    fc,
    edge_density=0.10,
    resolution=1.0,
    louvain_repeats=50,
    min_module_size=0.05,
    random_state=42,
):
    """
    Detect FC modules with repeated Louvain clustering and merge modules
    that are smaller than min_module_size.
    """
    graph = _build_fc_graph(
        fc,
        edge_density=edge_density,
    )

    if not hasattr(nx.community, "louvain_communities"):
        raise ImportError(
            "A recent NetworkX version with "
            "nx.community.louvain_communities is required."
        )

    best_communities = None
    best_q = -np.inf

    for repeat in range(louvain_repeats):
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=random_state + repeat,
        )

        q = nx.community.modularity(
            graph,
            communities,
            weight="weight",
            resolution=resolution,
        )

        if q > best_q:
            best_q = float(q)
            best_communities = [
                sorted(module)
                for module in communities
            ]

    minimum_size = _resolve_min_module_size(
        min_module_size,
        fc.shape[0],
    )

    merged_communities = _merge_small_modules(
        fc,
        best_communities,
        min_module_size=minimum_size,
    )

    # Recalculate modularity after small-module merging.
    merged_q = nx.community.modularity(
        graph,
        [set(module) for module in merged_communities],
        weight="weight",
        resolution=resolution,
    )

    merged_communities.sort(key=lambda x: min(x))

    labels = np.full(fc.shape[0], -1, dtype=int)
    for module_id, nodes in enumerate(merged_communities):
        labels[nodes] = module_id

    return {
        "communities": merged_communities,
        "labels": labels,
        "modularity_before_merge": best_q,
        "modularity_after_merge": float(merged_q),
        "minimum_module_size": minimum_size,
        "graph": graph,
    }


def _hierarchical_order(distance_matrix):
    n = distance_matrix.shape[0]

    if n <= 2:
        return np.arange(n)

    condensed = squareform(
        distance_matrix,
        checks=False,
    )
    z = linkage(condensed, method="average")
    z = optimal_leaf_ordering(z, condensed)

    return leaves_list(z)


def _order_nodes_by_fc_modules(fc, communities):
    """
    Arrange modules contiguously and hierarchically order nodes within
    each module.
    """
    n_modules = len(communities)

    # Order modules using their whole-network FC profiles.
    if n_modules > 2:
        module_profiles = np.vstack([
            np.nanmean(fc[np.asarray(nodes), :], axis=0)
            for nodes in communities
        ])
        module_profiles = np.nan_to_num(module_profiles)

        z_modules = linkage(
            pdist(module_profiles, metric="euclidean"),
            method="average",
        )
        module_order = leaves_list(z_modules)
    else:
        module_order = np.arange(n_modules)

    ordered_nodes = []
    module_boundaries = []
    ordered_module_sizes = []

    start = 0

    for module_idx in module_order:
        nodes = np.asarray(
            communities[module_idx],
            dtype=int,
        )

        if len(nodes) > 2:
            sub_fc = fc[np.ix_(nodes, nodes)]

            distance = 1.0 - np.clip(
                sub_fc,
                -1.0,
                1.0,
            )
            distance = (distance + distance.T) / 2.0
            np.fill_diagonal(distance, 0.0)

            local_order = _hierarchical_order(distance)
            nodes = nodes[local_order]

        end = start + len(nodes)

        ordered_nodes.extend(nodes.tolist())
        module_boundaries.append((start, end))
        ordered_module_sizes.append(len(nodes))

        start = end

    return (
        np.asarray(ordered_nodes, dtype=int),
        module_boundaries,
        ordered_module_sizes,
    )


def _rotate_grid_to_diamond(n):
    x, y = np.meshgrid(
        np.arange(n + 1),
        np.arange(n + 1),
    )

    scale = np.sqrt(2.0)
    u = (x - y) / scale
    v = (x + y) / scale

    return u, v


def _transform_points(points):
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]

    scale = np.sqrt(2.0)

    return np.column_stack([
        (x - y) / scale,
        (x + y) / scale,
    ])


def _draw_module_boxes(
    ax,
    module_boundaries,
    color="#D73027",
    linewidth=1.2,
):
    for start, end in module_boundaries:
        square = np.array([
            [start, start],
            [end, start],
            [end, end],
            [start, end],
        ])

        diamond = _transform_points(square)

        ax.add_patch(
            Polygon(
                diamond,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                joinstyle="miter",
                zorder=10,
            )
        )


def _draw_split_diamond(
    ax,
    left_matrix,
    right_matrix,
    module_boundaries,
    left_label,
    right_label,
    left_cmap,
    right_cmap,
    left_norm,
    right_norm,
    module_color="#D73027",
):
    n = left_matrix.shape[0]
    rows, cols = np.indices((n, n))
    u, v = _rotate_grid_to_diamond(n)

    # Lower triangle becomes the left half.
    left_data = np.ma.masked_where(
        rows <= cols,
        left_matrix,
    )

    # Upper triangle becomes the right half.
    right_data = np.ma.masked_where(
        rows >= cols,
        right_matrix,
    )

    left_artist = ax.pcolormesh(
        u,
        v,
        left_data,
        cmap=left_cmap,
        norm=left_norm,
        shading="flat",
        edgecolors="none",
        rasterized=True,
    )

    right_artist = ax.pcolormesh(
        u,
        v,
        right_data,
        cmap=right_cmap,
        norm=right_norm,
        shading="flat",
        edgecolors="none",
        rasterized=True,
    )

    full_height = n * np.sqrt(2.0)
    half_width = n / np.sqrt(2.0)

    outer_square = np.array([
        [0, 0],
        [n, 0],
        [n, n],
        [0, n],
    ])

    ax.add_patch(
        Polygon(
            _transform_points(outer_square),
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=0.75,
            zorder=11,
        )
    )

    # Main diagonal
    ax.plot(
        [0, 0],
        [0, full_height],
        color="black",
        linewidth=0.5,
        zorder=9,
    )

    _draw_module_boxes(
        ax,
        module_boundaries,
        color=module_color,
    )

    # Direct matrix labels
    ax.text(
        -0.43 * half_width,
        0.12 * full_height,
        left_label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.43 * half_width,
        0.12 * full_height,
        right_label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    padding = 0.05 * full_height

    ax.set_xlim(
        -half_width - padding,
        half_width + padding,
    )
    ax.set_ylim(
        full_height + padding,
        -padding,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    return left_artist, right_artist


def plot_fc_is_fc_sc_diamonds(
    FC,
    IS,
    SC,
    edge_density=0.10,
    resolution=1.0,
    louvain_repeats=50,
    min_module_size=0.05,
    random_state=42,
    symmetrize="mean",
    robust_percentiles=(2.0, 98.0),
    fc_cmap="RdBu_r",
    is_cmap="PuOr_r",
    sc_cmap="viridis",
    sc_range=None,
    figsize=(7.2, 4.6),
    module_color="#D73027",
    show_colorbars=True,
    savepath=None,
):
    """
    Plot two vertically oriented split matrices:

        1. FC on the left and IS on the right
        2. FC on the left and SC on the right

    Improvements
    ------------
    1. FC, IS and SC use different colormaps and separate colorbars.
    2. Color limits are estimated using robust off-diagonal percentiles.
    3. Very small FC modules are merged into the most strongly connected
       neighboring module.

    min_module_size
    ----------------
    float in (0, 1]:
        Minimum fraction of all nodes, e.g. 0.05 means 5%.
    integer:
        Minimum absolute number of nodes, e.g. 8.
    """
    fc = _prepare_matrix(
        FC,
        "FC",
        symmetrize=symmetrize,
    )
    is_matrix = _prepare_matrix(
        IS,
        "IS",
        symmetrize=symmetrize,
    )
    sc = _prepare_matrix(
        SC,
        "SC",
        symmetrize=symmetrize,
    )

    if not (
        fc.shape == is_matrix.shape == sc.shape
    ):
        raise ValueError(
            "FC, IS and SC must have identical shapes."
        )

    # --------------------------------------------------------
    # FC module detection
    # --------------------------------------------------------
    module_result = _detect_fc_modules(
        fc,
        edge_density=edge_density,
        resolution=resolution,
        louvain_repeats=louvain_repeats,
        min_module_size=min_module_size,
        random_state=random_state,
    )

    communities = module_result["communities"]

    order, module_boundaries, module_sizes = (
        _order_nodes_by_fc_modules(
            fc,
            communities,
        )
    )

    fc_ordered = fc[np.ix_(order, order)]
    is_ordered = is_matrix[np.ix_(order, order)]
    sc_ordered = sc[np.ix_(order, order)]

    # --------------------------------------------------------
    # Independent robust color limits
    # --------------------------------------------------------
    fc_norm, fc_limits, fc_mode = _robust_norm(
        fc_ordered,
        percentiles=robust_percentiles,
        mode="diverging",
    )

    is_norm, is_limits, is_mode = _robust_norm(
        is_ordered,
        percentiles=robust_percentiles,
        mode="diverging",
    )

    sc_norm, sc_limits, sc_mode = _robust_norm(
        sc_ordered,
        percentiles=robust_percentiles,
        mode="auto",
    )
    if sc_range is not None:
        sc_min, sc_max = map(float, sc_range)
        if not np.isfinite([sc_min, sc_max]).all() or sc_min >= sc_max:
            raise ValueError("sc_range must contain two finite increasing values.")
        sc_limits = (sc_min, sc_max)
        if sc_min < 0 < sc_max:
            sc_norm = TwoSlopeNorm(vmin=sc_min, vcenter=0.0, vmax=sc_max)
            sc_mode = "diverging"
        else:
            sc_norm = Normalize(vmin=sc_min, vmax=sc_max)
            sc_mode = "sequential"

    # SC uses a sequential map when nonnegative and a diverging map
    # when robust values cross zero.
    if sc_cmap is None:
        sc_cmap = (
            "BrBG"
            if sc_mode == "diverging"
            else "viridis"
        )

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    fc_artist_1, is_artist = _draw_split_diamond(
        axes[0],
        left_matrix=fc_ordered,
        right_matrix=is_ordered,
        module_boundaries=module_boundaries,
        left_label="FC",
        right_label="IS",
        left_cmap=fc_cmap,
        right_cmap=is_cmap,
        left_norm=fc_norm,
        right_norm=is_norm,
        module_color=module_color,
    )

    fc_artist_2, sc_artist = _draw_split_diamond(
        axes[1],
        left_matrix=fc_ordered,
        right_matrix=sc_ordered,
        module_boundaries=module_boundaries,
        left_label="FC",
        right_label="SC",
        left_cmap=fc_cmap,
        right_cmap=sc_cmap,
        left_norm=fc_norm,
        right_norm=sc_norm,
        module_color=module_color,
    )

    axes[0].set_title(
        "FC-IS correspondence",
        fontsize=10,
        pad=2,
    )
    axes[1].set_title(
        "FC-SC correspondence",
        fontsize=10,
        pad=2,
    )

    # --------------------------------------------------------
    # Three independent colorbars
    # --------------------------------------------------------
    if show_colorbars:
        fig.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.92,
            bottom=0.19,
            wspace=0.04,
        )

        colorbar_specs = [
            (
                fc_artist_1,
                [0.12, 0.075, 0.21, 0.020],
                "FC",
                fc_limits,
            ),
            (
                is_artist,
                [0.395, 0.075, 0.21, 0.020],
                "IS",
                is_limits,
            ),
            (
                sc_artist,
                [0.67, 0.075, 0.21, 0.020],
                "SC",
                sc_limits,
            ),
        ]

        colorbars = {}

        for artist, position, label, limits in colorbar_specs:
            cax = fig.add_axes(position)

            cbar = fig.colorbar(
                artist,
                cax=cax,
                orientation="horizontal",
                extend="both",
            )

            cbar.set_label(
                label,
                fontsize=8,
                labelpad=2,
            )
            cbar.ax.xaxis.set_label_coords(0.5, -1.45)
            cbar.ax.tick_params(
                labelsize=7,
                length=2,
                width=0.5,
            )
            cbar.outline.set_linewidth(0.5)

            # Show only the robust endpoints and centre when applicable.
            vmin, vmax = limits

            if vmin < 0 < vmax:
                cbar.set_ticks([vmin, 0.0, vmax])
            else:
                cbar.set_ticks([vmin, vmax])

            colorbars[label] = cbar

    else:
        colorbars = {}
        fig.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.92,
            bottom=0.04,
            wspace=0.04,
        )

    if savepath is not None:
        fig.savefig(
            savepath,
            # dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

    print(
        f"Modules: {len(module_sizes)} | "
        f"sizes: {module_sizes}"
    )
    print(
        "Minimum allowed module size:",
        module_result["minimum_module_size"],
    )
    print(
        "Modularity Q before merging:",
        f"{module_result['modularity_before_merge']:.4f}",
    )
    print(
        "Modularity Q after merging:",
        f"{module_result['modularity_after_merge']:.4f}",
    )
    print(
        "Robust color limits "
        f"({robust_percentiles[0]:g}-{robust_percentiles[1]:g} percentiles):"
    )
    print("  FC:", fc_limits)
    print("  IS:", is_limits)
    print("  SC:", sc_limits)

    return {
        "fig": fig,
        "axes": axes,
        "order": order,
        "FC_ordered": fc_ordered,
        "IS_ordered": is_ordered,
        "SC_ordered": sc_ordered,
        "communities": communities,
        "module_labels": module_result["labels"],
        "FC_graph": module_result["graph"],
        "module_boundaries": module_boundaries,
        "module_sizes": module_sizes,
        "minimum_module_size": (
            module_result["minimum_module_size"]
        ),
        "modularity_before_merge": (
            module_result["modularity_before_merge"]
        ),
        "modularity_after_merge": (
            module_result["modularity_after_merge"]
        ),
        "color_limits": {
            "FC": fc_limits,
            "IS": is_limits,
            "SC": sc_limits,
        },
        "color_modes": {
            "FC": fc_mode,
            "IS": is_mode,
            "SC": sc_mode,
        },
        "colorbars": colorbars,
        "artists": {
            "FC_left_1": fc_artist_1,
            "IS_right": is_artist,
            "FC_left_2": fc_artist_2,
            "SC_right": sc_artist,
        },
    }


def _paired_plot_values(first, second):
    """Accept matching square matrices or matching one-dimensional vectors."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"Input shapes differ: {first.shape} versus {second.shape}.")
    if first.ndim == 2:
        if first.shape[0] != first.shape[1]:
            raise ValueError("Two-dimensional inputs must be square matrices.")
        mask = ~np.eye(first.shape[0], dtype=bool)
        first, second = first[mask], second[mask]
    else:
        first, second = first.ravel(), second.ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    return first[valid], second[valid]


def calculate_corr_sp_boxplot_trend(
    v1,
    v1_label,
    v2,
    v2_label,
    num_bins=15,
    savepath=None,
    color="#438c73",
    title=None,
):
    """Plot binned FC distributions and return the pairwise Spearman r.

    Matrix inputs retain both off-diagonal directions.
    """
    x, y = _paired_plot_values(v1, v2)
    if x.size < 2:
        raise ValueError("At least two finite value pairs are required.")
    correlation, _ = pearsonr(x, y)

    bins = np.linspace(x.min(), x.max(), num_bins + 1)
    if np.unique(bins).size < 2:
        raise ValueError("Input-similarity values are constant; binning is undefined.")
    frame = pd.DataFrame({"X": x, "Y": y})
    frame["bin"] = pd.cut(frame["X"], bins=bins, include_lowest=True)
    grouped = [
        (interval.mid, group["Y"].to_numpy(), group["Y"].median())
        for interval, group in frame.groupby("bin", observed=True)
        if not group.empty
    ]
    if len(grouped) < 2:
        raise ValueError("At least two non-empty bins are required.")
    positions, box_data, medians = map(list, zip(*grouped))

    figure, axis = plt.subplots(figsize=(3.0, 3.0))
    boxplot = axis.boxplot(
        box_data,
        positions=positions,
        widths=(bins[1] - bins[0]) * 0.62,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        zorder=1,
    )
    for patch in boxplot["boxes"]:
        patch.set(facecolor=color, edgecolor=color, alpha=0.9, linewidth=0.5)
    for median in boxplot["medians"]:
        median.set(color="black", linewidth=0.7)
    for artist in boxplot["whiskers"] + boxplot["caps"]:
        artist.set(color=color, linewidth=0.5)

    slope, intercept = np.polyfit(positions, medians, 1)
    x_line = np.asarray(positions)
    axis.plot(
        x_line,
        slope * x_line + intercept,
        color="black",
        linewidth=0.8,
        zorder=3,
    )
    axis.set(
        xlabel=v1_label,
        ylabel=v2_label,
        title=title,
    )
    axis.text(
        0.04,
        0.96,
        f"Pearson r = {correlation:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    return float(correlation)
