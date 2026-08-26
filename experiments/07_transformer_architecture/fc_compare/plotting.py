from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


OPTIMIZED_COLORS = {
    "a1": "#C68A3A",
    "a2": "#3B8D87",
    "a3": "#C65D73",
    "a4": "#3E78B2",
}
BASELINE_COLOR = "#6F6F6F"
GROUP_FIGURE_STEMS = {
    "a1": "a1_positional_encoding",
    "a2": "a2_residual_optimized",
    "a3": "a3_sequence_mixer",
    "a4": "a4_normalization_placement",
}

PPL_PLOT_START = 150


def _matrix(summaries: list[dict], key: str, variant: str):
    rows = [summary[key][variant] for summary in summaries]
    steps = np.asarray([item["step"] for item in rows[0]], dtype=float)
    for row in rows[1:]:
        if [item["step"] for item in row] != list(steps.astype(int)):
            raise ValueError("all seeds must share checkpoint steps")
    values = np.asarray([[item["value"] for item in row] for row in rows], dtype=float)
    return steps, values


def _line_with_interval(ax, x, values, label, color):
    center = np.mean(values, axis=0)
    if values.shape[0] > 1:
        low, high = np.quantile(values, [0.025, 0.975], axis=0)
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        for row in values:
            ax.plot(x, row, color=color, alpha=0.16, linewidth=0.55)
    ax.plot(x, center, label=label, color=color, linewidth=1.6)


def _from_initial(values):
    return values - values[:, [0]]


def _write_source_data(path: Path, summaries: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["seed", "metric", "variant", "step", "value"]
        )
        writer.writeheader()
        for summary in summaries:
            metric_keys = [
                ("validation_ppl", "validation_ppl_trajectory"),
                ("top5_fc_mean_position_standardized", "fc_trajectory"),
            ]
            if "raw_fc_trajectory" in summary:
                metric_keys.append(("top5_fc_mean_raw", "raw_fc_trajectory"))
            for metric, key in metric_keys:
                for variant in ("baseline", "optimized"):
                    for row in summary[key][variant]:
                        writer.writerow(
                            {
                                "seed": summary["seed"],
                                "metric": metric,
                                "variant": variant,
                                "step": row["step"],
                                "value": row["value"],
                            }
                        )
            for variant in ("baseline", "optimized"):
                trajectory = summary["fc_trajectory"][variant]
                initial = trajectory[0]["value"]
                for row in trajectory:
                    writer.writerow(
                        {
                            "seed": summary["seed"],
                            "metric": "delta_top5_fc_from_initialization",
                            "variant": variant,
                            "step": row["step"],
                            "value": row["value"] - initial,
                        }
                    )
            baseline_ppl = {
                row["step"]: row["value"]
                for row in summary["validation_ppl_trajectory"]["baseline"]
            }
            optimized_ppl = {
                row["step"]: row["value"]
                for row in summary["validation_ppl_trajectory"]["optimized"]
            }
            if baseline_ppl.keys() != optimized_ppl.keys():
                raise ValueError(
                    "baseline and optimized PPL trajectories must share checkpoint steps"
                )
            for step in baseline_ppl:
                writer.writerow(
                    {
                        "seed": summary["seed"],
                        "metric": "validation_ppl_ratio",
                        "variant": "optimized/baseline",
                        "step": step,
                        "value": optimized_ppl[step] / baseline_ppl[step],
                    }
                )


def _top_fc_title(aggregate: dict):
    percent = 100.0 * aggregate["fc_config"]["top_fraction"]
    selection = aggregate["fc_config"].get("layer_selection")
    if selection == "all":
        return f"Mean top {percent:g}% |FC| across layers"
    if isinstance(selection, int):
        return f"Layer {selection} top {percent:g}% |FC| mean"
    return f"Top {percent:g}% |FC| mean"


def _draw_absolute_fc_panel(
    ax, spec, summaries: list[dict], aggregate: dict, color: str
):
    for variant, label, line_color in (
        ("baseline", spec.baseline_label, BASELINE_COLOR),
        ("optimized", spec.optimized_label, color),
    ):
        steps, values = _matrix(summaries, "fc_trajectory", variant)
        _line_with_interval(ax, steps, values, label, line_color)
    ax.set_title(_top_fc_title(aggregate))
    ax.set_ylabel("|FC|")
    ax.legend(loc="best")


def _draw_delta_fc_panel(
    ax, spec, summaries: list[dict], aggregate: dict, color: str
):
    for variant, label, line_color in (
        ("baseline", spec.baseline_label, BASELINE_COLOR),
        ("optimized", spec.optimized_label, color),
    ):
        steps, values = _matrix(summaries, "fc_trajectory", variant)
        _line_with_interval(ax, steps, _from_initial(values), label, line_color)
    ax.axhline(
        aggregate["fc_config"]["onset_delta"],
        color="#9A9A9A",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=0,
    )
    ax.set_title(f"{_top_fc_title(aggregate)} rise")
    ax.set_ylabel(r"$\Delta$FC from initialization")


# def _draw_ppl_panel(ax, spec, summaries: list[dict], aggregate: dict, color: str):
#     for variant, label, line_color in (
#         ("baseline", spec.baseline_label, BASELINE_COLOR),
#         ("optimized", spec.optimized_label, color),
#     ):
#         steps, values = _matrix(summaries, "validation_ppl_trajectory", variant)
#         _line_with_interval(ax, steps, values, label, line_color)
#     ax.set_title("Validation perplexity")
#     ax.set_ylabel("PPL")
#     ax.set_yscale("log")
#     ax.legend(loc="best")
# def _draw_ppl_panel(ax, spec, summaries: list[dict], aggregate: dict, color: str):
#     for variant, label, line_color in (
#         ("baseline", spec.baseline_label, BASELINE_COLOR),
#         ("optimized", spec.optimized_label, color),
#     ):
#         steps, values = _matrix(
#             summaries,
#             "validation_ppl_trajectory",
#             variant,
#         )
#         keep = steps >= PPL_PLOT_START
#         if not np.any(keep):
#             raise ValueError(
#                 f"No PPL checkpoints at or after step {PPL_PLOT_START}"
#             )
#         # 必须在绘图前过滤，否则隐藏的早期点仍会影响 y 轴自动范围
#         steps = steps[keep]
#         values = values[:, keep]

#         _line_with_interval(
#             ax,
#             steps,
#             values,
#             label,
#             line_color,
#         )

#     ax.set_title(f"Validation perplexity (steps ≥ {PPL_PLOT_START})")
#     ax.set_xlabel("Training step")
#     ax.set_ylabel("PPL")
#     ax.set_yscale("log")
#     ax.set_xlim(left=PPL_PLOT_START)
#     ax.legend(loc="best")

#     # aggregate 中是配对 test PPL 的中位数百分比变化：
#     # 例如 -12.1 表示 optimized 的 PPL 降低了 12.1%
#     percent_change = aggregate["ppl_comparison"]["median_percent_change"]

#     if percent_change <= 0:
#         effect_text = f"Test PPL: {-percent_change:.1f}% lower"
#     else:
#         effect_text = f"Test PPL: {percent_change:.1f}% higher"

#     ax.text(
#         0.98,
#         0.06,
#         effect_text,
#         transform=ax.transAxes,
#         ha="right",
#         va="bottom",
#         fontsize=6.5,
#         fontweight="bold",
#         color=color,
#     )
PPL_FOCUS_STEP = 100


def _draw_ppl_panel(ax, spec, summaries, aggregate, color):
    baseline_steps, baseline = _matrix(
        summaries,
        "validation_ppl_trajectory",
        "baseline",
    )
    optimized_steps, optimized = _matrix(
        summaries,
        "validation_ppl_trajectory",
        "optimized",
    )

    if not np.array_equal(baseline_steps, optimized_steps):
        raise ValueError(
            "baseline and optimized PPL trajectories must share steps"
        )

    # 绘制完整的 0–500 step 曲线
    _line_with_interval(
        ax,
        baseline_steps,
        baseline,
        spec.baseline_label,
        BASELINE_COLOR,
    )
    _line_with_interval(
        ax,
        optimized_steps,
        optimized,
        spec.optimized_label,
        color,
    )

    # 使用 step >= 100 的数据决定合理的 y 轴范围
    focus = baseline_steps >= PPL_FOCUS_STEP
    if not np.any(focus):
        raise ValueError(
            f"No PPL checkpoints at or after step {PPL_FOCUS_STEP}"
        )

    focused_values = np.concatenate(
        [
            baseline[:, focus].reshape(-1),
            optimized[:, focus].reshape(-1),
        ]
    )

    # log y 轴使用乘法 margin
    y_min = np.nanmin(focused_values) / 1.10
    y_max = np.nanmax(focused_values) * 1.10

    # ax.set_xlim(baseline_steps[0], baseline_steps[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xticks([0,100,200,300,400,500])

    ax.set_title("Validation perplexity")
    ax.set_ylabel("PPL")
    ax.legend(loc="best")

    # 明确告诉读者早期高 PPL 超出了显示范围
    early_values = np.concatenate(
        [
            baseline[:, ~focus].reshape(-1),
            optimized[:, ~focus].reshape(-1),
        ]
    )

    if early_values.size and np.nanmax(early_values) > y_max:
        ax.text(
            0.02,
            0.98,
            f"PPL > {y_max:,.0f} clipped",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="#666666",
        )

    # Test PPL 改善百分比
    percent_change = aggregate["ppl_comparison"][
        "median_percent_change"
    ]

    if percent_change <= 0:
        effect_text = f"Test PPL: {-percent_change:.1f}% lower"
    else:
        effect_text = f"Test PPL: {percent_change:.1f}% higher"

    ax.text(
        0.98,
        0.06,
        effect_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color=color,
    )


def _draw_ppl_ratio_panel(
    ax, spec, summaries: list[dict], aggregate: dict, color: str
):
    baseline_steps, baseline = _matrix(
        summaries, "validation_ppl_trajectory", "baseline"
    )
    optimized_steps, optimized = _matrix(
        summaries, "validation_ppl_trajectory", "optimized"
    )
    if not np.array_equal(baseline_steps, optimized_steps):
        raise ValueError(
            "baseline and optimized PPL trajectories must share checkpoint steps"
        )
    if np.any(baseline <= 0) or np.any(optimized <= 0):
        raise ValueError("PPL values must be positive before computing a ratio")
    ratio = optimized / baseline
    ax.axhline(
        1.0,
        color=BASELINE_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=0,
    )
    _line_with_interval(
        ax,
        baseline_steps,
        ratio,
        "Optimized / baseline",
        color,
    )
    ax.set_title("Validation PPL ratio")
    ax.set_ylabel("Optimized / baseline")
    ax.margins(y=0.08)


def plot_group(spec, summaries: list[dict], aggregate: dict, output_stem: Path):
    output_stem = output_stem.parent / GROUP_FIGURE_STEMS.get(
        spec.key, f"{spec.key}_comparison"
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    _write_source_data(
        output_stem.with_name(f"{output_stem.name}_source_data.csv"), summaries
    )
    optimized_color = OPTIMIZED_COLORS[spec.key]
    fig, axes = plt.subplots(1, 4, figsize=(9.20, 2.35), constrained_layout=True)

    absolute_fc_ax, delta_fc_ax, ppl_ax, ppl_ratio_ax = axes
    _draw_absolute_fc_panel(
        absolute_fc_ax, spec, summaries, aggregate, optimized_color
    )
    _draw_delta_fc_panel(delta_fc_ax, spec, summaries, aggregate, optimized_color)
    _draw_ppl_panel(ppl_ax, spec, summaries, aggregate, optimized_color)
    _draw_ppl_ratio_panel(
        ppl_ratio_ax, spec, summaries, aggregate, optimized_color
    )
    for ax in axes:
        ax.set_xlabel("Training step")
    absolute_fc_ax.text(
        -0.12,
        1.08,
        spec.key,
        transform=absolute_fc_ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )
    fig.suptitle(spec.title, fontsize=8)

    paths = []
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_figure6a(rows: list[tuple], output_stem: Path):
    """Assemble all available a1--a4 rows without automatic result judgments."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(9.20, 6.65), constrained_layout=True)
    for row_index, (spec, summaries, aggregate) in enumerate(rows):
        absolute_fc_ax, delta_fc_ax, ppl_ax, ppl_ratio_ax = axes[row_index]
        optimized_color = OPTIMIZED_COLORS[spec.key]
        _draw_absolute_fc_panel(
            absolute_fc_ax, spec, summaries, aggregate, optimized_color
        )
        _draw_delta_fc_panel(
            delta_fc_ax, spec, summaries, aggregate, optimized_color
        )
        _draw_ppl_panel(ppl_ax, spec, summaries, aggregate, optimized_color)
        _draw_ppl_ratio_panel(
            ppl_ratio_ax, spec, summaries, aggregate, optimized_color
        )
        ppl_ax.get_legend().set_title(spec.title)
        absolute_fc_ax.text(
            -0.16,
            1.06,
            spec.key,
            transform=absolute_fc_ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    paths = []
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths
