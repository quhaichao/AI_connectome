"""Resource-aware analysis and publication figures for the WikiText-2 MoE benchmark.

This module is deliberately analysis-only: it reads artifacts produced by the
isolated sibling ``runner.py`` and never trains, mutates, or saves a model. The v2
analysis separates three questions that should not be collapsed into one rank:

1. construction-only comparison at a shared dense checkpoint and top-1 budget;
2. measured quality-versus-compute comparison across all methods;
3. routing, token dropping, and dynamic-k diagnostics.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Editable text is mandatory for journal-ready SVG/PDF output.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PRIMARY_CONSTRUCTION_METHODS = (
    "fc_guided",
    "random_calibrated",
    "clustering_calibrated",
)

PARTITIONED_METHODS = (
    "fc_guided",
    "random_calibrated",
    "clustering_calibrated",
    "fc_guided_abs",
    "llama_moe_random",
    "llama_moe_clustering",
    "emoe",
    "moefication",
    "switch_partitioned",
    "d2dmoe",
)

FULL_WIDTH_METHODS = (
    "sparse_upcycling",
    "cluster_aware_upcycling",
    "switch_full",
)

METHOD_ORDER = PRIMARY_CONSTRUCTION_METHODS + (
    "fc_guided_abs",
    "llama_moe_random",
    "llama_moe_clustering",
    "emoe",
    "moefication",
    "switch_partitioned",
    "d2dmoe",
) + FULL_WIDTH_METHODS

METHOD_LABELS = {
    "fc_guided": "FC-Hybrid",
    "random_calibrated": "Random + matched ridge",
    "clustering_calibrated": "Clustering + matched ridge",
    "fc_guided_abs": "Abs-FC ablation",
    "llama_moe_random": "LLaMA-MoE random",
    "llama_moe_clustering": "LLaMA-MoE clustering",
    "emoe": "EMoE",
    "moefication": "MoEfication",
    "switch_partitioned": "Regular Switch iso-total",
    "d2dmoe": "D2DMoE",
    "sparse_upcycling": "Sparse Upcycling",
    "cluster_aware_upcycling": "Cluster-aware Upcycling",
    "switch_full": "Regular Switch iso-active",
}

# Related construction methods use one coherent blue family; FC remains the hero.
METHOD_COLORS = {
    "fc_guided": "#B64342",
    "random_calibrated": "#3E668F",
    "clustering_calibrated": "#7E9FC2",
    "fc_guided_abs": "#C88782",
    "llama_moe_random": "#484878",
    "llama_moe_clustering": "#7884B4",
    "emoe": "#9A4D8E",
    "moefication": "#42949E",
    "switch_partitioned": "#767676",
    "d2dmoe": "#E28E2C",
    "sparse_upcycling": "#5B8FD6",
    "cluster_aware_upcycling": "#7BAA5B",
    "switch_full": "#A8A8A8",
}

REGIME_LABELS = {
    "partitioned_iso_total": "Partitioned experts (iso-total)",
    "upcycling_iso_active": "Full-width experts (iso-active)",
}


def apply_publication_style(font_size: float = 7.0) -> None:
    """Apply a compact, vector-editable journal style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "savefig.transparent": False,
        }
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_results(
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Dict[Tuple[str, int], dict]]:
    """Load raw benchmark artifacts without importing the training stack."""
    root = Path(output_dir).expanduser().resolve()
    required = (root / "summary.json", root / "history.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing benchmark result file(s): " + ", ".join(missing) +
            ". Point RESULT_DIR to the completed benchmark output directory."
        )

    summary = pd.DataFrame(_read_json(root / "summary.json"))
    history = pd.DataFrame(_read_json(root / "history.json"))
    config = _read_json(root / "config.json") if (root / "config.json").exists() else {}

    constructions: Dict[Tuple[str, int], dict] = {}
    for path in root.glob("*_seed*_construction.json"):
        stem = path.name[: -len("_construction.json")]
        try:
            method, seed_text = stem.rsplit("_seed", 1)
            constructions[(method, int(seed_text))] = _read_json(path)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.warn("Skipped unreadable construction artifact %s: %s" % (path.name, exc))
    return summary, history, config, constructions


def _ordered_methods(methods: Iterable[str]) -> List[str]:
    methods = list(dict.fromkeys(methods))
    known = [method for method in METHOD_ORDER if method in methods]
    return known + sorted(set(methods) - set(known))


def _label(method: str, summary: Optional[pd.DataFrame] = None) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if summary is not None and {"method", "label"}.issubset(summary.columns):
        match = summary.loc[summary["method"] == method, "label"]
        if not match.empty:
            return str(match.iloc[0]).replace(" (ours)", "")
    return method


def audit_results_v2(
    summary: pd.DataFrame,
    history: pd.DataFrame,
    config: Optional[Mapping[str, object]] = None,
    compute_tolerance: float = 0.10,
) -> pd.DataFrame:
    """Audit completeness and *measured* fairness, returning actionable rows."""
    config = dict(config or {})
    rows: List[dict] = []

    def add(check: str, status: str, detail: str) -> None:
        rows.append({"check": check, "status": status, "detail": detail})

    required_summary = {
        "method", "seed", "regime", "test_loss", "test_perplexity", "train_tokens",
        "measured_expert_flops_per_token", "mean_selected_experts",
        "mean_dropped_fraction",
    }
    missing = sorted(required_summary - set(summary.columns))
    add(
        "required summary fields",
        "PASS" if not missing else "FAIL",
        "all present" if not missing else "missing: " + ", ".join(missing),
    )
    if missing:
        return pd.DataFrame(rows)

    duplicated = summary.duplicated(["method", "seed"], keep=False)
    add(
        "one row per method × seed",
        "PASS" if not duplicated.any() else "FAIL",
        "no duplicates" if not duplicated.any() else "%d duplicated rows" % duplicated.sum(),
    )

    expected_methods = list(config.get("methods", []))
    expected_seeds = list(config.get("seeds", []))
    observed_pairs = set(zip(summary["method"], summary["seed"]))
    if expected_methods and expected_seeds:
        expected_pairs = {(method, seed) for method in expected_methods for seed in expected_seeds}
        missing_pairs = expected_pairs - observed_pairs
        extra_pairs = observed_pairs - expected_pairs
        status = "PASS" if not missing_pairs and not extra_pairs else "FAIL"
        detail = "%d/%d expected pairs" % (len(expected_pairs - missing_pairs), len(expected_pairs))
        if missing_pairs:
            detail += "; missing=" + str(sorted(missing_pairs))
        if extra_pairs:
            detail += "; extra=" + str(sorted(extra_pairs))
        add("method × seed completeness", status, detail)

    finite_columns = [
        "test_loss", "test_perplexity", "train_tokens",
        "measured_expert_flops_per_token", "mean_selected_experts",
        "mean_dropped_fraction",
    ]
    numeric = summary[finite_columns].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(numeric.to_numpy())
    add(
        "finite reported metrics",
        "PASS" if not bad.any() else "FAIL",
        "all finite" if not bad.any() else "%d non-finite cells" % int(bad.sum()),
    )

    token_counts = summary.groupby("seed")["train_tokens"].nunique()
    add(
        "equal charged training tokens within seed",
        "PASS" if (token_counts == 1).all() else "FAIL",
        "; ".join("seed %s: %s unique value(s)" % item for item in token_counts.items()),
    )

    seed_counts = summary.groupby("method")["seed"].nunique()
    min_seeds = int(seed_counts.min()) if not seed_counts.empty else 0
    add(
        "replicate count for inference",
        "PASS" if min_seeds >= 5 else "WARN",
        "minimum n=%d seed(s); paired points and effect sizes are primary when n<5" % min_seeds,
    )

    if "fc_guided" in set(summary["method"]):
        construction = summary[summary["method"].isin(PRIMARY_CONSTRUCTION_METHODS)].copy()
        ratios = []
        for seed, group in construction.groupby("seed"):
            baseline = group.loc[group["method"] == "fc_guided", "measured_expert_flops_per_token"]
            if baseline.empty:
                continue
            base = float(baseline.iloc[0])
            for _, row in group.iterrows():
                ratios.append(float(row["measured_expert_flops_per_token"]) / base)
        max_dev = max((abs(ratio - 1.0) for ratio in ratios), default=float("nan"))
        pass_compute = np.isfinite(max_dev) and max_dev <= compute_tolerance
        add(
            "primary construction measured-compute match",
            "PASS" if pass_compute else "WARN",
            "maximum deviation from FC-guided = %.1f%%" % (100 * max_dev),
        )

        if "method_specific_calibration_steps" in construction.columns:
            calibration_counts = construction.groupby("seed")[
                "method_specific_calibration_steps"
            ].nunique()
            calibration_match = bool((calibration_counts == 1).all())
            add(
                "primary construction method-specific-step match",
                "PASS" if calibration_match else "FAIL",
                "FC-Hybrid, random and clustering must use identical charged method-specific steps",
            )
        router_policy_columns = [
            column for column in (
                "router_kind", "router_target", "router_objective",
                "router_initialization_kind", "joint_router_alignment",
                "joint_router_alignment_steps", "effective_aux_loss_coef",
            )
            if column in construction.columns
        ]
        if router_policy_columns:
            policy_match = all(
                construction.groupby("seed")[column].nunique().eq(1).all()
                for column in router_policy_columns
            )
            add(
                "primary construction router-policy match",
                "PASS" if policy_match else "FAIL",
                "matched fields: " + ", ".join(router_policy_columns),
            )

        d2d = summary[summary["method"] == "d2dmoe"]
        fc = summary[summary["method"] == "fc_guided"]
        joined = fc[["seed", "measured_expert_flops_per_token"]].merge(
            d2d[["seed", "measured_expert_flops_per_token"]], on="seed", suffixes=("_fc", "_d2d")
        )
        if not joined.empty:
            ratio = np.mean(
                joined["measured_expert_flops_per_token_d2d"] /
                joined["measured_expert_flops_per_token_fc"]
            )
            add(
                "D2DMoE measured-compute equivalence",
                "PASS" if abs(ratio - 1.0) <= compute_tolerance else "WARN",
                "D2DMoE uses %.2f× FC-guided expert FLOPs/token; do not treat as iso-compute" % ratio,
            )

    max_drop = float(summary["mean_dropped_fraction"].max())
    add(
        "token dropping below 5%",
        "PASS" if max_drop <= 0.05 else "WARN",
        "maximum dropped-token fraction = %.1f%%" % (100 * max_drop),
    )

    if "valid_perplexity" in history.columns and not history.empty:
        cap = math.exp(20)
        capped = pd.to_numeric(history["valid_perplexity"], errors="coerce") >= cap * 0.999
        add(
            "unclipped learning-curve display",
            "PASS" if not capped.any() else "WARN",
            "use validation NLL: %d point(s) hit the exp(20) perplexity cap" % int(capped.sum()),
        )
    return pd.DataFrame(rows)


def _t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, n - 1))
    except ImportError:
        # Conservative two-sided 95% critical values for the small n used here.
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                 20: 2.086, 30: 2.042}
        df = n - 1
        if df in table:
            return table[df]
        lower = max(key for key in table if key < df)
        upper = min(key for key in table if key > df)
        weight = (df - lower) / float(upper - lower)
        return table[lower] * (1 - weight) + table[upper] * weight


def aggregate_summary_v2(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all numeric metrics and add t-based 95% CIs."""
    group_columns = [column for column in ("method", "label", "regime") if column in summary.columns]
    numeric_columns = [
        column for column in summary.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    rows = []
    for keys, group in summary.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            n = len(values)
            mean = float(np.mean(values)) if n else float("nan")
            sem = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            half = _t_critical_95(n) * sem if n > 1 else float("nan")
            row[column + "_mean"] = mean
            row[column + "_sem"] = sem
            row[column + "_ci95_low"] = mean - half
            row[column + "_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return adjusted
    ordered = valid[np.argsort(values[valid])]
    running = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_comparisons_v2(
    summary: pd.DataFrame,
    baseline: str = "fc_guided",
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute paired seed differences (competitor minus baseline) with Holm correction."""
    methods = list(methods or [method for method in PARTITIONED_METHODS if method != baseline])
    base_columns = ["seed", "test_perplexity", "measured_expert_flops_per_token"]
    base = summary.loc[summary["method"] == baseline, base_columns].rename(
        columns={
            "test_perplexity": "baseline_ppl",
            "measured_expert_flops_per_token": "baseline_flops",
        }
    )
    rows = []
    for method in methods:
        other = summary.loc[summary["method"] == method, base_columns].rename(
            columns={
                "test_perplexity": "competitor_ppl",
                "measured_expert_flops_per_token": "competitor_flops",
            }
        )
        paired = base.merge(other, on="seed", how="inner")
        delta = (paired["competitor_ppl"] - paired["baseline_ppl"]).to_numpy(float)
        n = len(delta)
        mean = float(np.mean(delta)) if n else float("nan")
        sem = float(np.std(delta, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        half = _t_critical_95(n) * sem if n > 1 else float("nan")
        try:
            from scipy.stats import ttest_rel

            p_value = float(ttest_rel(paired["competitor_ppl"], paired["baseline_ppl"]).pvalue) if n > 1 else float("nan")
        except ImportError:
            p_value = float("nan")
        compute_ratio = float(np.mean(paired["competitor_flops"] / paired["baseline_flops"])) if n else float("nan")
        rows.append(
            {
                "baseline": baseline,
                "method": method,
                "comparison_scope": "construction-only" if method in PRIMARY_CONSTRUCTION_METHODS else "adaptation/control",
                "n_pairs": n,
                "delta_ppl_competitor_minus_fc": mean,
                "delta_sem": sem,
                "delta_ci95_low": mean - half,
                "delta_ci95_high": mean + half,
                "fc_wins": int(np.sum(delta > 0)),
                "p_value_paired_t": p_value,
                "measured_expert_flops_ratio_vs_fc": compute_ratio,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm_all_partitioned"] = _holm_adjust(result["p_value_paired_t"])
        primary = result["comparison_scope"] == "construction-only"
        result["p_holm_construction_only"] = np.nan
        result.loc[primary, "p_holm_construction_only"] = _holm_adjust(
            result.loc[primary, "p_value_paired_t"].to_numpy()
        )
    return result


def early_learning_metrics_v2(
    history: pd.DataFrame,
    start_step: int = 100,
    end_step: int = 500,
) -> pd.DataFrame:
    """Summarize early learning with mean validation NLL and normalized AUC."""
    required = {"method", "seed", "global_step", "valid_loss"}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    rows = []
    for (method, seed), group in history.groupby(["method", "seed"]):
        curve = group[["global_step", "valid_loss"]].dropna().sort_values("global_step")
        curve = curve[(curve["global_step"] >= start_step) & (curve["global_step"] <= end_step)]
        if len(curve) < 2:
            continue
        x = curve["global_step"].to_numpy(float)
        y = curve["valid_loss"].to_numpy(float)
        duration = x[-1] - x[0]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "window_first_step": int(x[0]),
                "window_last_step": int(x[-1]),
                "n_evaluations": len(curve),
                "mean_valid_nll": float(np.mean(y)),
                "normalized_nll_auc": float(np.trapz(y, x) / duration) if duration > 0 else float("nan"),
                "last_valid_nll": float(y[-1]),
            }
        )
    return pd.DataFrame(rows)


def pareto_frontier_v2(
    aggregate: pd.DataFrame,
    x: str = "measured_expert_flops_per_token_mean",
    y: str = "test_perplexity_mean",
) -> pd.DataFrame:
    """Return non-dominated rows when both x and y are minimized."""
    frame = aggregate.dropna(subset=[x, y]).sort_values([x, y]).copy()
    keep = []
    best_y = float("inf")
    for index, row in frame.iterrows():
        if float(row[y]) < best_y:
            keep.append(index)
            best_y = float(row[y])
    return frame.loc[keep].sort_values(x)


def _mean_ci(group: pd.DataFrame, metric: str) -> Tuple[float, float, float]:
    values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    half = _t_critical_95(len(values)) * sem
    return mean, mean - half, mean + half


def _panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom")


def _style_axis(ax) -> None:
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.65, zorder=0)
    ax.tick_params(width=0.7, length=2.5)


def save_figure_v2(
    fig,
    output_stem: Optional[str],
    formats: Sequence[str] = ("svg", "pdf", "png"),
    dpi: int = 300,
    close: bool = False,
) -> List[Path]:
    """Export an editable vector pair plus a high-resolution preview."""
    if output_stem is None:
        return []
    stem = Path(output_stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = stem.with_suffix("." + fmt.lower())
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(path)
    if close:
        plt.close(fig)
    return saved


def plot_primary_figure_v2(
    summary: pd.DataFrame,
    history: pd.DataFrame,
    output_stem: Optional[str] = None,
    early_window: Tuple[int, int] = (100, 500),
):
    """Create the primary argument: paired construction, learning, and Pareto cost."""
    apply_publication_style(7.0)
    fig = plt.figure(figsize=(7.2, 3.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.25, 1.18], wspace=0.42)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, index]) for index in range(3)]

    # a | Paired construction effect. Positive values mean FC-Hybrid is better.
    clean = summary[summary["method"].isin(PRIMARY_CONSTRUCTION_METHODS)].copy()
    pivot = clean.pivot_table(index="seed", columns="method", values="test_perplexity", aggfunc="first")
    competitors = PRIMARY_CONSTRUCTION_METHODS[1:]
    x = np.arange(len(competitors))
    for index, method in enumerate(competitors):
        required = ["fc_guided", method]
        paired = pivot.reindex(columns=required).dropna()
        if paired.empty:
            continue
        delta = paired[method] - paired["fc_guided"]
        ax_a.scatter(
            np.full(len(delta), index), delta, color="#F7F7F7",
            edgecolor=METHOD_COLORS[method], s=16, linewidth=0.7, zorder=2,
        )
        mean = float(delta.mean())
        sem = float(delta.sem()) if len(delta) > 1 else 0.0
        half = _t_critical_95(len(delta)) * sem if len(delta) > 1 else 0.0
        ax_a.errorbar(
            index, mean, yerr=half, fmt="o", color=METHOD_COLORS[method],
            markeredgecolor="white", markeredgewidth=0.5, markersize=5.5,
            capsize=2.5, linewidth=1.2, zorder=4,
        )
    ax_a.axhline(0.0, color="#707070", linestyle="--", linewidth=0.8, zorder=0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Random + ridge", "Clustering + ridge"], rotation=18, ha="right")
    ax_a.set_ylabel("Δ test PPL (control − FC-Hybrid) ↑")
    ax_a.set_title("Matched construction effect", loc="left", pad=6)
    ax_a.text(0.02, 0.02, ">0 favors FC-Hybrid; points: seeds; bars: 95% CI", transform=ax_a.transAxes,
              fontsize=5.8, color="#606060")
    _style_axis(ax_a)
    _panel_label(ax_a, "a")

    # b | NLL avoids the exp(20) clipping that made the original PPL curve unreadable.
    start_step, end_step = early_window
    for method in PRIMARY_CONSTRUCTION_METHODS:
        curve = history[(history["method"] == method) &
                        (history["global_step"] >= start_step) &
                        (history["global_step"] <= end_step)]
        if curve.empty:
            continue
        agg = curve.groupby("global_step")["valid_loss"].agg(["mean", "sem"]).reset_index()
        xx = agg["global_step"].to_numpy(float)
        yy = agg["mean"].to_numpy(float)
        ee = agg["sem"].fillna(0).to_numpy(float)
        ax_b.plot(xx, yy, label=_label(method), color=METHOD_COLORS[method])
        ax_b.fill_between(xx, yy - ee, yy + ee, color=METHOD_COLORS[method], alpha=0.14, linewidth=0)
    ax_b.set_xlim(start_step, end_step)
    ax_b.set_xlabel("Charged optimization step")
    ax_b.set_ylabel("Validation NLL ↓")
    ax_b.set_title("Early learning dynamics", loc="left", pad=6)
    ax_b.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
    _style_axis(ax_b)
    _panel_label(ax_b, "b")

    # c | Actual compute, not nominal top-k, determines the comparison geometry.
    aggregate = aggregate_summary_v2(summary)
    frontier = pareto_frontier_v2(aggregate)
    if len(frontier) > 1:
        ax_c.plot(frontier["measured_expert_flops_per_token_mean"] / 1e6,
                  frontier["test_perplexity_mean"], color="#A0A0A0", linestyle="--",
                  linewidth=0.9, zorder=1, label="Pareto frontier")
    label_offsets = {
        "d2dmoe": (4, -8),
        "sparse_upcycling": (4, -8), "cluster_aware_upcycling": (4, 5), "switch_full": (4, 5),
    }
    cost_labels = {
        "fc_guided": "FC-Hybrid", "random_calibrated": "Random + ridge",
        "clustering_calibrated": "Clustering + ridge", "fc_guided_abs": "Abs-FC ablation",
        "llama_moe_random": "Random",
        "llama_moe_clustering": "Clustering", "emoe": "EMoE",
        "moefication": "MoEfication", "switch_partitioned": "Switch iso-total",
        "d2dmoe": "D2DMoE", "sparse_upcycling": "Sparse Upcycling",
        "cluster_aware_upcycling": "Cluster-aware", "switch_full": "Switch iso-active",
    }
    plotted_points = []
    for _, row in aggregate.iterrows():
        method = row["method"]
        marker = "o" if row.get("regime") == "partitioned_iso_total" else "s"
        xx = float(row["measured_expert_flops_per_token_mean"]) / 1e6
        yy = float(row["test_perplexity_mean"])
        ax_c.errorbar(xx, yy,
                      yerr=[[yy - row["test_perplexity_ci95_low"]],
                            [row["test_perplexity_ci95_high"] - yy]],
                      fmt=marker, color=METHOD_COLORS.get(method, "#606060"),
                      markeredgecolor="white", markeredgewidth=0.45, markersize=5.2,
                      capsize=2, linewidth=0.9, zorder=3)
        plotted_points.append((method, xx, yy))

    # The iso-total points occupy a very narrow compute band. Spread their labels
    # vertically and connect them with subtle leaders instead of allowing overlap.
    left_points = sorted((point for point in plotted_points if point[1] < 1.25), key=lambda point: point[2])
    if left_points:
        all_y = aggregate["test_perplexity_mean"].to_numpy(float)
        minimum_gap = max(0.75, 0.035 * (float(np.max(all_y)) - float(np.min(all_y))))
        label_y = [point[2] for point in left_points]
        for index in range(1, len(label_y)):
            label_y[index] = max(label_y[index], label_y[index - 1] + minimum_gap)
        shift = float(np.mean(label_y) - np.mean([point[2] for point in left_points]))
        label_y = [value - shift for value in label_y]
        label_x = max(point[1] for point in left_points) + 0.10
        for (method, xx, yy), text_y in zip(left_points, label_y):
            ax_c.annotate(
                cost_labels.get(method, _label(method, summary)),
                xy=(xx, yy), xytext=(label_x, text_y), textcoords="data",
                fontsize=5.4, color="#303030", ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": "#B8B8B8", "lw": 0.45},
            )
    for method, xx, yy in (point for point in plotted_points if point[1] >= 1.25):
        ax_c.annotate(cost_labels.get(method, _label(method, summary)), (xx, yy),
                      xytext=label_offsets.get(method, (4, 4)), textcoords="offset points",
                      fontsize=5.4, color="#303030")
    ax_c.set_xlabel("Measured expert FLOPs/token (×10⁶) →")
    ax_c.set_ylabel("Test perplexity ↓")
    ax_c.set_title("Quality–compute trade-off", loc="left", pad=6)
    _style_axis(ax_c)
    _panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.86)
    save_figure_v2(fig, output_stem)
    return fig


def extract_d2d_thresholds_v2(
    constructions: Mapping[Tuple[str, int], Mapping[str, object]],
) -> pd.DataFrame:
    rows = []
    for (method, seed), payload in constructions.items():
        if method != "d2dmoe":
            continue
        for point in payload.get("d2d_threshold_curve", []) or []:
            rows.append({"method": method, "seed": seed, **point})
    return pd.DataFrame(rows)


def extract_fc_hybrid_diagnostics_v2(
    constructions: Mapping[Tuple[str, int], Mapping[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten construction-selection and oracle-ridge diagnostics for source data."""
    construction_rows = []
    router_rows = []
    for (method, seed), payload in constructions.items():
        if method != "fc_guided":
            continue
        for row in payload.get("fc_diagnostics", []) or []:
            construction_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "layer": row.get("layer"),
                    "selected_candidate": row.get("selected_candidate"),
                    "selected_strategy": row.get("selected_strategy"),
                    "selected_weights": json.dumps(row.get("selected_weights")),
                    "oracle_normalized_mse": row.get("oracle_normalized_mse"),
                    "oracle_explained_fraction": row.get("oracle_explained_fraction"),
                    "oracle_usage_entropy": row.get("oracle_usage_entropy"),
                    "refinement_accepted": row.get("refinement_accepted"),
                    "candidate_scores": json.dumps(row.get("candidate_scores")),
                }
            )
        for row in payload.get("router_initialization", []) or []:
            router_rows.append({"method": method, "seed": seed, **row})
    return pd.DataFrame(construction_rows), pd.DataFrame(router_rows)


def plot_diagnostics_figure_v2(
    summary: pd.DataFrame,
    constructions: Mapping[Tuple[str, int], Mapping[str, object]],
    output_stem: Optional[str] = None,
):
    """Plot final estimates, routing load, token dropping, and dynamic-k sensitivity."""
    apply_publication_style(7.0)
    aggregate = aggregate_summary_v2(summary)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a | Partitioned estimates, ordered by mean PPL, with an explicit compute cue.
    part = aggregate[aggregate["regime"] == "partitioned_iso_total"].sort_values(
        "test_perplexity_mean", ascending=True
    )
    y = np.arange(len(part))[::-1]
    for yi, (_, row) in zip(y, part.iterrows()):
        method = row["method"]
        mean = row["test_perplexity_mean"]
        low = row["test_perplexity_ci95_low"]
        high = row["test_perplexity_ci95_high"]
        ax_a.plot([low, high], [yi, yi], color=METHOD_COLORS.get(method, "#606060"), linewidth=1.2)
        ax_a.plot(mean, yi, "o", color=METHOD_COLORS.get(method, "#606060"), markersize=4.5)
    labels = []
    for _, row in part.iterrows():
        suffix = "†" if row["measured_expert_flops_per_token_mean"] > 1.5 * part.loc[part["method"] == "fc_guided", "measured_expert_flops_per_token_mean"].iloc[0] else ""
        labels.append(_label(row["method"], summary) + suffix)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlabel("Test perplexity (mean, 95% CI) ↓")
    ax_a.set_title("Partitioned regime", loc="left", pad=6)
    ax_a.text(0.01, -0.22, "† >1.5× FC-guided measured expert FLOPs/token", transform=ax_a.transAxes,
              fontsize=5.8, color="#606060")
    _style_axis(ax_a)
    _panel_label(ax_a, "a")

    # b | Dynamic expert count makes the D2D compute difference visible.
    ordered = aggregate.sort_values("mean_selected_experts_mean", ascending=True)
    yy = np.arange(len(ordered))
    ax_b.barh(yy, ordered["mean_selected_experts_mean"],
              color=[METHOD_COLORS.get(method, "#A8A8A8") for method in ordered["method"]],
              edgecolor="white", linewidth=0.4)
    ax_b.axvline(1.0, color="#606060", linestyle="--", linewidth=0.8)
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels([_label(method, summary) for method in ordered["method"]])
    ax_b.set_xlabel("Selected experts/token →")
    ax_b.set_title("Realized routing width", loc="left", pad=6)
    _style_axis(ax_b)
    _panel_label(ax_b, "b")

    # c | Dropping is a quality and fairness diagnostic, not a hidden implementation detail.
    ordered = aggregate.sort_values("mean_dropped_fraction_mean", ascending=True)
    yy = np.arange(len(ordered))
    ax_c.barh(yy, 100 * ordered["mean_dropped_fraction_mean"],
              color=[METHOD_COLORS.get(method, "#A8A8A8") for method in ordered["method"]],
              edgecolor="white", linewidth=0.4)
    ax_c.axvline(5.0, color="#B64342", linestyle="--", linewidth=0.8)
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels([_label(method, summary) for method in ordered["method"]])
    ax_c.set_xlabel("Dropped-token fraction (%) →")
    ax_c.set_title("Capacity overflow", loc="left", pad=6)
    ax_c.text(5.0, 1.01, "5%", color="#B64342", ha="center", va="bottom",
              transform=ax_c.get_xaxis_transform(), fontsize=5.8)
    _style_axis(ax_c)
    _panel_label(ax_c, "c")

    # d | D2D threshold sweep directly exposes the quality-compute trade-off.
    thresholds = extract_d2d_thresholds_v2(constructions)
    if thresholds.empty:
        ax_d.text(0.5, 0.5, "No D2DMoE threshold artifact", ha="center", va="center",
                  transform=ax_d.transAxes, color="#606060")
        ax_d.set_axis_off()
    else:
        curve = thresholds.groupby("threshold").agg(
            selected_mean=("mean_selected_experts", "mean"),
            selected_sem=("mean_selected_experts", "sem"),
            ppl_mean=("test_perplexity", "mean"),
            ppl_sem=("test_perplexity", "sem"),
        ).reset_index()
        ax_d.errorbar(curve["selected_mean"], curve["ppl_mean"],
                      xerr=curve["selected_sem"].fillna(0), yerr=curve["ppl_sem"].fillna(0),
                      marker="o", color=METHOD_COLORS["d2dmoe"], capsize=2, markersize=4)
        for _, row in curve.iterrows():
            ax_d.annotate("τ=%.2g" % row["threshold"],
                          (row["selected_mean"], row["ppl_mean"]),
                          xytext=(4, 3), textcoords="offset points", fontsize=5.7)
        ax_d.set_xlabel("Selected experts/token →")
        ax_d.set_ylabel("Test perplexity ↓")
        ax_d.set_title("D2DMoE threshold sensitivity", loc="left", pad=6)
        _style_axis(ax_d)
    _panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.12, top=0.94, wspace=0.58, hspace=0.55)
    save_figure_v2(fig, output_stem)
    return fig


def plot_usage_heatmaps_v2(
    constructions: Mapping[Tuple[str, int], Mapping[str, object]],
    summary: Optional[pd.DataFrame] = None,
    output_stem: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
):
    """Plot non-overlapping, seed-averaged layer × expert routing heatmaps."""
    usage: Dict[str, List[np.ndarray]] = {}
    for (method, _seed), payload in constructions.items():
        values = payload.get("test_layer_usage")
        if values is None:
            continue
        usage.setdefault(method, []).append(np.asarray(values, dtype=float))
    if methods is None:
        methods = _ordered_methods(usage)
    methods = [method for method in methods if method in usage]
    if not methods:
        return None

    apply_publication_style(7.0)
    matrices = {method: np.mean(np.stack(usage[method]), axis=0) for method in methods}
    vmax = max(float(np.nanmax(matrix)) for matrix in matrices.values())
    columns = 2
    rows = int(math.ceil(len(methods) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7.2, max(2.2, rows * 1.35)), squeeze=False)
    image = None
    for index, method in enumerate(methods):
        ax = axes.ravel()[index]
        matrix = matrices[method]
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=vmax, cmap="magma", interpolation="nearest")
        ax.set_title(_label(method, summary), loc="left", pad=4)
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.spines[:].set_visible(False)
    for ax in axes.ravel()[len(methods):]:
        ax.set_axis_off()
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.08, top=0.96, hspace=0.58, wspace=0.30)
    if image is not None:
        colorbar_ax = fig.add_axes([0.875, 0.18, 0.018, 0.64])
        cbar = fig.colorbar(image, cax=colorbar_ax)
        cbar.set_label("Fraction of routed tokens")
        cbar.outline.set_linewidth(0.6)
    save_figure_v2(fig, output_stem)
    return fig


def conclusion_table_v2(summary: pd.DataFrame) -> pd.DataFrame:
    """Generate cautious, data-derived statements for the notebook and manuscript draft."""
    aggregate = aggregate_summary_v2(summary).set_index("method")
    paired = paired_comparisons_v2(summary).set_index("method")
    rows = []

    def add(topic: str, statement: str, interpretation: str) -> None:
        rows.append({"topic": topic, "result": statement, "interpretation": interpretation})

    for method in ("random_calibrated", "clustering_calibrated"):
        if method not in paired.index:
            continue
        row = paired.loc[method]
        delta = float(row["delta_ppl_competitor_minus_fc"])
        relative = 100 * delta / float(aggregate.loc["fc_guided", "test_perplexity_mean"])
        add(
            _label(method),
            "competitor − FC = %+.2f PPL (%+.2f%%); FC wins %d/%d paired seeds" %
            (delta, relative, int(row["fc_wins"]), int(row["n_pairs"])),
            "matched construction effect; confirm with more seeds" if abs(relative) >= 0.5 else "practically near-tied at this scale",
        )

    for method in ("fc_guided_abs", "llama_moe_random", "llama_moe_clustering"):
        if method not in paired.index:
            continue
        row = paired.loc[method]
        delta = float(row["delta_ppl_competitor_minus_fc"])
        add(
            _label(method),
            "competitor minus FC-Hybrid = %+.2f PPL; FC-Hybrid wins %d/%d paired seeds" %
            (delta, int(row["fc_wins"]), int(row["n_pairs"])),
            "ablation/original-construction context; not the primary matched attribution test",
        )

    for method in ("emoe", "moefication", "switch_partitioned"):
        if method not in paired.index:
            continue
        row = paired.loc[method]
        delta = float(row["delta_ppl_competitor_minus_fc"])
        add(
            _label(method),
            "competitor − FC = %+.2f PPL; FC wins %d/%d paired seeds" %
            (delta, int(row["fc_wins"]), int(row["n_pairs"])),
            "supporting adaptation/control comparison; n is small",
        )

    if "d2dmoe" in paired.index:
        row = paired.loc["d2dmoe"]
        add(
            "D2DMoE",
            "competitor − FC = %+.2f PPL at %.2f× measured expert FLOPs/token" %
            (float(row["delta_ppl_competitor_minus_fc"]), float(row["measured_expert_flops_ratio_vs_fc"])),
            "better raw PPL is compute-confounded and is not an iso-compute win",
        )

    add(
        "statistical scope",
        "n=%d seed(s) for FC-guided" % int(summary.loc[summary["method"] == "fc_guided", "seed"].nunique()),
        "report paired points, effect sizes and 95% CIs; treat p-values as exploratory when n<5",
    )
    return pd.DataFrame(rows)


def export_source_data_v2(
    output_dir: str,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    audit: pd.DataFrame,
    comparisons: pd.DataFrame,
    early_metrics: pd.DataFrame,
    constructions: Mapping[Tuple[str, int], Mapping[str, object]],
) -> List[Path]:
    """Write traceable source tables used by every v2 quantitative panel."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fc_construction, fc_router = extract_fc_hybrid_diagnostics_v2(constructions)
    tables = {
        "source_summary_by_seed.csv": summary,
        "source_summary_aggregate.csv": aggregate_summary_v2(summary),
        "source_learning_history.csv": history,
        "source_fairness_audit.csv": audit,
        "source_paired_comparisons.csv": comparisons,
        "source_early_learning_metrics.csv": early_metrics,
        "source_d2d_threshold_sweep.csv": extract_d2d_thresholds_v2(constructions),
        "source_fc_hybrid_construction.csv": fc_construction,
        "source_fc_hybrid_router_initialization.csv": fc_router,
    }
    saved = []
    for filename, frame in tables.items():
        path = root / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        saved.append(path)
    return saved
