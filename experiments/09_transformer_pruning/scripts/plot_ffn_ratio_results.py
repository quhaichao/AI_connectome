#!/usr/bin/env python
"""Create separate pruning curves and seed point plots for each model."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ("fc_ls", "flap", "sobp", "fang", "slimllm", "wanda")
METHOD_LABELS = {
    "fc_ls": "FC-pruning",
    "flap": "FLAP",
    "sobp": "SoBP",
    "fang": "FANG",
    "slimllm": "SlimLLM",
    "wanda": "Wanda",
}
MODEL_LABELS = {
    "llama32_1b": "Llama-3.2-1B",
    "qwen25_1_5b": "Qwen2.5-1.5B",
}
COLORS = {
    "fc_ls": "#86383f",
    "flap": "#42949E",
    "sobp": "#4F7CAC",
    "fang": "#9A4D8E",
    "slimllm": "#D28E2D",
    "wanda": "#6F6F6F",
}
PRUNING_RATIOS = np.array([20.0, 30.0, 40.0, 50.0])


def _load_results(path: Path, selected_models: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model", "domain", "seed", "method", "ratio", "ppl"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    frame["method"] = (
        frame["method"].astype(str).str.lower().replace({"fand": "fang"})
    )
    frame["ratio"] = (
        frame["ratio"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )
    if frame["ratio"].max() <= 1:
        frame["ratio"] *= 100
    frame["ppl"] = pd.to_numeric(frame["ppl"], errors="raise")
    frame = frame[
        frame["model"].isin(selected_models)
        & frame["method"].isin(METHOD_ORDER)
        & frame["ratio"].isin(PRUNING_RATIOS)
    ].copy()
    if frame.empty:
        raise ValueError("No requested pruning results were found in the CSV")
    return frame


def _save_figure(fig, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in outputs:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_point_figures(
    frame: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Plot ten seed observations and mean +/- SD, ordered high-to-low."""
    outputs = []
    for (domain, model), model_frame in frame.groupby(
        ["domain", "model"], sort=False
    ):
        for ratio in PRUNING_RATIOS:
            ratio_frame = model_frame[np.isclose(model_frame["ratio"], ratio)]
            if ratio_frame.empty:
                continue
            method_means = ratio_frame.groupby("method")["ppl"].mean()
            ordered_methods = method_means.sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(5.0, 4.2))
            rng = np.random.default_rng(42)
            for method_index, method in enumerate(ordered_methods):
                values = ratio_frame.loc[
                    ratio_frame["method"] == method, "ppl"
                ].to_numpy(dtype=float)
                x = method_index + rng.uniform(-0.12, 0.12, len(values))
                ax.scatter(
                    x,
                    values,
                    s=38,
                    color=COLORS[method],
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.errorbar(
                    method_index,
                    values.mean(),
                    yerr=values.std(ddof=1),
                    fmt="_",
                    color="black",
                    markersize=14,
                    markeredgewidth=2.0,
                    elinewidth=1.2,
                    capsize=3,
                    capthick=1.2,
                    zorder=5,
                )
            ax.set_title(
                f"{domain.upper()} · {MODEL_LABELS.get(model, model)} · "
                f"{ratio:.0f}% pruning",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_ylabel("Perplexity", fontsize=10)
            ax.set_xticks(np.arange(len(ordered_methods)))
            ax.set_xticklabels(
                [METHOD_LABELS[method] for method in ordered_methods],
                rotation=30,
                ha="right",
            )
            ax.tick_params(
                axis="both", labelsize=9, direction="out", length=3.5, width=0.8
            )
            fig.tight_layout()
            prefix = f"{model}_{domain}" if domain != "c4" else model
            outputs.extend(
                _save_figure(
                    fig, output_dir / f"{prefix}_points_{ratio:.0f}pct"
                )
            )
    return outputs


def plot_pruning_curves(
    frame: pd.DataFrame,
    output_dir: Path,
    error_type: str,
) -> list[Path]:
    """Plot mean PPL across pruning ratios with SD or SEM bands."""
    summary = (
        frame.groupby(["model", "domain", "method", "ratio"], as_index=False)[
            "ppl"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    outputs = []
    for (model, domain), model_frame in summary.groupby(
        ["model", "domain"], sort=False
    ):
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        present_methods = set(model_frame["method"])
        for method in METHOD_ORDER:
            if method not in present_methods:
                continue
            method_frame = model_frame[
                model_frame["method"] == method
            ].sort_values("ratio")
            ratios = method_frame["ratio"].to_numpy(dtype=float)
            means = method_frame["mean"].to_numpy(dtype=float)
            errors = method_frame[error_type].to_numpy(dtype=float)
            line_zorder = 10 if method == "fc_ls" else 5
            fill_zorder = 9 if method == "fc_ls" else 4
            ax.plot(
                ratios,
                means,
                color=COLORS[method],
                marker="o",
                linewidth=2.4,
                markersize=4.5,
                markeredgewidth=1.0,
                label=METHOD_LABELS[method],
                zorder=line_zorder,
            )
            ax.fill_between(
                ratios,
                means - errors,
                means + errors,
                color=COLORS[method],
                alpha=0.14,
                linewidth=0,
                zorder=fill_zorder,
            )
        ax.set_title(
            f"{domain.upper()} · {MODEL_LABELS.get(model, model)}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel("Pruning ratio", fontsize=10)
        ax.set_ylabel("Perplexity", fontsize=10)
        ax.set_xticks(PRUNING_RATIOS)
        ax.set_xticklabels([f"{ratio:.0f}%" for ratio in PRUNING_RATIOS])
        ax.tick_params(
            axis="both", labelsize=9, direction="out", length=3.5, width=0.8
        )
        ax.legend(
            title="Method",
            frameon=False,
            fontsize=8.5,
            title_fontsize=9,
            ncol=2,
        )
        fig.tight_layout()
        prefix = f"{model}_{domain}" if domain != "c4" else model
        outputs.extend(
            _save_figure(fig, output_dir / f"{prefix}_pruning_curve")
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="results/ffn_ratio_matrix_pearson/raw_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="pruning_figures/by_model",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_LABELS),
        default=list(MODEL_LABELS),
    )
    parser.add_argument(
        "--error-type",
        choices=("std", "sem"),
        default="std",
    )
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input_csv
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_results(input_path, args.models)

    counts = frame.groupby(["model", "domain", "method", "ratio"])[
        "seed"
    ].nunique()
    incomplete = counts[counts != args.expected_seeds]
    if not incomplete.empty:
        print("Warning: unexpected seed counts:")
        print(incomplete.to_string())

    outputs = plot_point_figures(frame, output_dir)
    outputs.extend(plot_pruning_curves(frame, output_dir, args.error_type))
    print(f"Read {len(frame)} rows from {input_path}")
    print(f"Wrote {len(outputs)} files to {output_dir}")


if __name__ == "__main__":
    main()
