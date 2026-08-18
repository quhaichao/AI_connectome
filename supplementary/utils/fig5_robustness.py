"""Focused multiseed analyses supporting main Figure 5g--i.

Figure 5g--h asks whether Adam and BatchNorm shift FC formation earlier across
operational top-FC fractions and datasets. Figure 5i reuses the exact
FC-guided residual implementation from the main-figure module and tests its
learning advantage across independent seeds and datasets.

The inferential unit is an independently initialized and trained seed. Paired
conditions share initialization and minibatch order; FC edges are never treated
as independent observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_early_high_fc_utils import (
    ExperimentMLP,
    FCGuidedResidualMLP,
    StandardResidualMLP,
    evaluate_residual_block,
    make_train_loader,
    set_seed,
    sync_linear_weights,
)


STANDARD = "#777777"
ADAM = "#D65F5F"
BATCHNORM = "#59A14F"
FC_GUIDED = "#B14E8F"
GATE_ONLY = "#C58DB2"
DARK = "#30343B"
DATASET_COLORS = ("#4C78A8", "#E09F3E", "#59A14F", "#9C755F")


@dataclass(frozen=True)
class Fig5GHConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    hidden_dims: tuple[int, ...] = (300, 300)
    batch_size: int = 256
    total_steps: int = 300
    evaluation_interval: int = 5
    learning_rate: float = 0.05
    adam_learning_rate: float = 1e-3
    top_fc_fractions: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.20)
    smoothing_window: int = 5
    peak_tolerance: float = 0.01
    analysis_samples: int = 2000

    def validate(self) -> None:
        if len(self.seeds) < 20:
            raise ValueError("At least 20 independent seeds are required.")
        if not self.top_fc_fractions or any(
            fraction <= 0 or fraction > 1 for fraction in self.top_fc_fractions
        ):
            raise ValueError("top_fc_fractions must lie in (0, 1].")
        if self.smoothing_window < 3 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be an odd integer >= 3.")
        if not 0 <= self.peak_tolerance < 1:
            raise ValueError("peak_tolerance must lie in [0, 1).")


@dataclass(frozen=True)
class Fig5IConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    hidden_dim: int = 256
    batch_size: int = 512
    total_steps: int = 300
    evaluation_interval: int = 5
    learning_rate: float = 0.05
    analysis_samples: int = 2000
    top_fc_ratio: float = 0.05
    fc_threshold: float = 0.70
    gate_strength: float = 3.0
    gate_top_fraction: float = 0.50
    fc_coupling_strength: float = 0.80
    coupling_steps: int = 40

    def validate(self) -> None:
        if len(self.seeds) < 20:
            raise ValueError("At least 20 independent seeds are required.")
        if self.gate_strength <= 0:
            raise ValueError("gate_strength must be positive.")
        if not 0 < self.gate_top_fraction < 1:
            raise ValueError("gate_top_fraction must lie in (0, 1).")
        if self.fc_coupling_strength < 0:
            raise ValueError("fc_coupling_strength must be non-negative.")
        if not 2 <= self.coupling_steps <= self.total_steps:
            raise ValueError("coupling_steps must lie in [2, total_steps].")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _next_batch(iterator: Any, loader: Any) -> tuple[Any, torch.Tensor, torch.Tensor]:
    try:
        inputs, targets = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        inputs, targets = next(iterator)
    return iterator, inputs, targets


def _evaluation_steps(total_steps: int, interval: int) -> np.ndarray:
    steps = np.arange(0, total_steps, interval, dtype=int)
    if steps[-1] != total_steps - 1:
        steps = np.append(steps, total_steps - 1)
    return steps


def _safe_fc_matrix(activations: np.ndarray) -> np.ndarray:
    valid = np.std(activations, axis=0) > 1e-6
    if valid.sum() < 2:
        return np.eye(2, dtype=float)
    corr = np.corrcoef(activations[:, valid], rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def top_fc_means(corr: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
    """Return mean absolute FC among the strongest requested edge fractions."""

    values = np.sort(np.abs(corr[np.triu_indices(corr.shape[0], k=1)]))
    if not len(values):
        return np.zeros(len(fractions), dtype=float)
    output = []
    for fraction in fractions:
        count = max(1, int(np.ceil(len(values) * float(fraction))))
        output.append(float(values[-count:].mean()))
    return np.asarray(output)


def smooth_fc_curves(
    curves: np.ndarray,
    window: int,
) -> np.ndarray:
    """Centered moving-average smoothing along the training-step axis."""

    curves = np.asarray(curves, dtype=float)
    usable_window = min(
        window,
        curves.shape[-1] if curves.shape[-1] % 2 else curves.shape[-1] - 1,
    )
    if usable_window < 3:
        return curves.copy()
    padding = usable_window // 2
    padded = np.pad(
        curves,
        [(0, 0)] * (curves.ndim - 1) + [(padding, padding)],
        mode="edge",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=usable_window,
        axis=-1,
    )
    return windows.mean(axis=-1)


def steps_to_smoothed_peak(
    curves: np.ndarray,
    steps: np.ndarray,
    window: int,
    tolerance: float,
) -> np.ndarray:
    """Earliest step within ``tolerance`` of each smoothed curve's maximum.

    The tolerance is expressed as a fraction of that curve's smoothed dynamic
    range. A 1% tolerance prevents a tiny late fluctuation from redefining the
    peak of an otherwise flat plateau.
    """

    smoothed = smooth_fc_curves(curves, window)
    maximum = np.nanmax(smoothed, axis=-1)
    minimum = np.nanmin(smoothed, axis=-1)
    threshold = maximum - tolerance * (maximum - minimum)
    reached = smoothed >= threshold[..., None]
    first_index = np.argmax(reached, axis=-1)
    return np.asarray(steps)[first_index]


def gh_peak_steps(result: Mapping[str, np.ndarray], config: Fig5GHConfig) -> np.ndarray:
    return steps_to_smoothed_peak(
        result["fc_curves"],
        result["steps"],
        config.smoothing_window,
        config.peak_tolerance,
    )


def _evaluate_hidden_fc(
    model: ExperimentMLP,
    inputs: torch.Tensor,
    fractions: Sequence[float],
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, hidden = model.forward_with_hidden(inputs)
    return top_fc_means(_safe_fc_matrix(hidden.detach().cpu().numpy()), fractions)


def run_gh_seed(
    experiment: str,
    seed: int,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    config: Fig5GHConfig,
    device: str | torch.device | None = None,
    num_workers: int = 2,
) -> dict[str, np.ndarray]:
    """Run one paired SGD/Adam or no-BN/BatchNorm FC-timing experiment."""

    if experiment not in {"adam", "batchnorm"}:
        raise ValueError("experiment must be 'adam' or 'batchnorm'.")
    config.validate()
    device = resolve_device(device)
    set_seed(seed)
    baseline = ExperimentMLP(hidden_dims=config.hidden_dims).to(device)
    set_seed(seed)
    variant = ExperimentMLP(
        hidden_dims=config.hidden_dims,
        use_batchnorm=(experiment == "batchnorm"),
    ).to(device)
    sync_linear_weights(baseline, variant)
    optimizers = (
        torch.optim.SGD(baseline.parameters(), lr=config.learning_rate),
        torch.optim.Adam(variant.parameters(), lr=config.adam_learning_rate)
        if experiment == "adam"
        else torch.optim.SGD(variant.parameters(), lr=config.learning_rate),
    )
    models = (baseline, variant)
    loader = make_train_loader(train_dataset, config.batch_size, seed, num_workers)
    iterator = iter(loader)
    analysis_inputs = analysis_inputs.to(device)
    steps = _evaluation_steps(config.total_steps, config.evaluation_interval)
    lookup = {int(step): index for index, step in enumerate(steps)}
    fc_curves = np.empty((2, len(config.top_fc_fractions), len(steps)), dtype=float)

    for step in range(config.total_steps):
        iterator, batch_inputs, batch_targets = _next_batch(iterator, loader)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        for model, optimizer in zip(models, optimizers):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(batch_inputs), batch_targets).backward()
            optimizer.step()
        if step in lookup:
            for model_index, model in enumerate(models):
                fc_curves[model_index, :, lookup[step]] = _evaluate_hidden_fc(
                    model, analysis_inputs, config.top_fc_fractions
                )
    return {"seed": np.asarray(seed), "steps": steps, "fc_curves": fc_curves}


def _run_i_models(
    seed: int,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    analysis_targets: torch.Tensor,
    config: Fig5IConfig,
    coupling_strengths: Sequence[float],
    device: str | torch.device | None = None,
    num_workers: int = 2,
) -> dict[str, np.ndarray]:
    """Train a standard residual and matched FC-guided variants."""

    config.validate()
    device = resolve_device(device)
    set_seed(seed)
    standard = StandardResidualMLP(config.hidden_dim).to(device)
    variants = []
    for coupling_strength in coupling_strengths:
        set_seed(seed)
        variants.append(
            FCGuidedResidualMLP(
                hidden_dim=config.hidden_dim,
                gate_strength=config.gate_strength,
                gate_top_fraction=config.gate_top_fraction,
                fc_coupling_strength=float(coupling_strength),
                coupling_steps=config.coupling_steps,
            ).to(device)
        )
    models: tuple[nn.Module, ...] = (standard, *variants)
    optimizers = tuple(
        torch.optim.SGD(model.parameters(), lr=config.learning_rate) for model in models
    )
    loader = make_train_loader(train_dataset, config.batch_size, seed, num_workers)
    iterator = iter(loader)
    analysis_inputs = analysis_inputs.to(device)
    analysis_targets = analysis_targets.to(device)
    steps = _evaluation_steps(config.total_steps, config.evaluation_interval)
    lookup = {int(step): index for index, step in enumerate(steps)}
    curves = np.empty((len(models), 4, len(steps)), dtype=float)

    for step in range(config.total_steps):
        iterator, batch_inputs, batch_targets = _next_batch(iterator, loader)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        for model, optimizer in zip(models, optimizers):
            model.train()
            if isinstance(model, FCGuidedResidualMLP):
                model.set_step(step)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(batch_inputs), batch_targets).backward()
            optimizer.step()
        if step in lookup:
            for model_index, model in enumerate(models):
                loss, accuracy, top_fc, _, corr = evaluate_residual_block(
                    model, analysis_inputs, analysis_targets, config.top_fc_ratio
                )
                upper = corr[np.triu_indices(corr.shape[0], k=1)]
                high_ratio = float(np.mean(upper > config.fc_threshold)) if len(upper) else 0.0
                curves[model_index, :, lookup[step]] = (
                    loss,
                    accuracy,
                    top_fc,
                    high_ratio,
                )
    return {"seed": np.asarray(seed), "steps": steps, "curves": curves}


def run_i_seed(
    seed: int,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    analysis_targets: torch.Tensor,
    config: Fig5IConfig,
    device: str | torch.device | None = None,
    num_workers: int = 2,
) -> dict[str, np.ndarray]:
    """Compare standard and full FC-guided residual MLPs for one seed."""

    return _run_i_models(
        seed,
        train_dataset,
        analysis_inputs,
        analysis_targets,
        config,
        (config.fc_coupling_strength,),
        device,
        num_workers,
    )




def _save_result(result: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)


def _load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _stack(results: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    ordered = sorted(results, key=lambda item: int(item["seed"]))
    data_key = "fc_curves" if "fc_curves" in ordered[0] else "curves"
    return {
        "seeds": np.asarray([int(item["seed"]) for item in ordered]),
        "steps": np.asarray(ordered[0]["steps"]),
        data_key: np.stack([np.asarray(item[data_key]) for item in ordered]),
    }


def _validate_cached_config(config: Any, directory: Path, overwrite: bool) -> None:
    path = directory / "config.json"
    if overwrite or not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        cached = json.load(handle)
    current = json.loads(json.dumps(asdict(config)))
    if cached != current:
        raise ValueError(
            f"Cached results in {directory} use a different configuration. "
            "Set OVERWRITE = True or choose a new RESULT_ROOT."
        )


def _run_or_load(
    directory: Path,
    seeds: Sequence[int],
    config: Any,
    runner: Any,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    directory.mkdir(parents=True, exist_ok=True)
    _validate_cached_config(config, directory, overwrite)
    with open(directory / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    results = []
    for seed in seeds:
        path = directory / f"seed_{seed:03d}.npz"
        result = _load_result(path) if path.exists() and not overwrite else runner(seed)
        if not path.exists() or overwrite:
            _save_result(result, path)
        results.append(result)
    combined = _stack(results)
    np.savez_compressed(directory / "all_seeds.npz", **combined)
    return combined


def run_or_load_gh(
    experiment: str,
    dataset_name: str,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    config: Fig5GHConfig,
    result_root: str | Path,
    device: str | torch.device | None = None,
    num_workers: int = 2,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    directory = Path(result_root) / experiment / dataset_name
    return _run_or_load(
        directory,
        config.seeds,
        config,
        lambda seed: run_gh_seed(
            experiment,
            seed,
            train_dataset,
            analysis_inputs,
            config,
            device,
            num_workers,
        ),
        overwrite,
    )


def run_or_load_i(
    dataset_name: str,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    analysis_targets: torch.Tensor,
    config: Fig5IConfig,
    result_root: str | Path,
    device: str | torch.device | None = None,
    num_workers: int = 2,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    directory = Path(result_root) / dataset_name
    return _run_or_load(
        directory,
        config.seeds,
        config,
        lambda seed: run_i_seed(
            seed,
            train_dataset,
            analysis_inputs,
            analysis_targets,
            config,
            device,
            num_workers,
        ),
        overwrite,
    )






def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _mean_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=axis)
    count = np.sum(np.isfinite(values), axis=axis)
    sem = np.nanstd(values, axis=axis, ddof=1) / np.sqrt(np.maximum(count, 1))
    return mean, 1.96 * sem


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")


def _fraction_labels(fractions: Sequence[float]) -> list[str]:
    return [f"{100 * fraction:g}" for fraction in fractions]


def _plot_peak_timing(
    ax: plt.Axes,
    peak_steps: np.ndarray,
    fractions: Sequence[float],
    labels: Sequence[str],
    colors: Sequence[str],
) -> None:
    x = np.arange(len(fractions))
    for model_index, (label, color) in enumerate(zip(labels, colors)):
        mean, ci = _mean_ci(peak_steps[:, model_index], axis=0)
        ax.errorbar(x, mean, yerr=ci, color=color, marker="o", ms=3.5, lw=1.2, capsize=2, label=label)
    ax.set_xticks(x, _fraction_labels(fractions))
    ax.set_xlabel("Top FC edges (%)")
    ax.set_ylabel("Steps to smoothed FC peak")
    ax.legend(fontsize=6)


def _plot_cross_dataset_acceleration(
    ax: plt.Axes,
    results: Mapping[str, Mapping[str, np.ndarray]],
    config: Fig5GHConfig,
    dataset_order: Sequence[str],
) -> None:
    x = np.arange(len(config.top_fc_fractions))
    for dataset_index, dataset_name in enumerate(dataset_order):
        peak_steps = gh_peak_steps(results[dataset_name], config)
        acceleration = peak_steps[:, 0] - peak_steps[:, 1]
        mean, ci = _mean_ci(acceleration, axis=0)
        ax.errorbar(
            x,
            mean,
            yerr=ci,
            marker="o",
            ms=3.2,
            lw=1.0,
            capsize=1.8,
            color=DATASET_COLORS[dataset_index],
            label=dataset_name,
        )
    ax.axhline(0, color=DARK, lw=0.7, ls="--")
    ax.set_xticks(x, _fraction_labels(config.top_fc_fractions))
    ax.set_xlabel("Top FC edges (%)")
    ax.set_ylabel("Earlier peak (steps)\nbaseline − method")
    ax.legend(fontsize=5.7)


def plot_fig5gh_robustness(
    results: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    config: Fig5GHConfig,
    dataset_order: Sequence[str] = ("MNIST", "FashionMNIST", "KMNIST"),
) -> plt.Figure:
    """Four-panel FC peak-timing robustness figure for Figure 5g--h."""

    set_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    for ax, label in zip(axes.flat, "abcd"):
        _panel_label(ax, label)

    _plot_peak_timing(
        axes[0, 0],
        gh_peak_steps(results["adam"]["MNIST"], config),
        config.top_fc_fractions,
        ("SGD", "Adam"),
        (STANDARD, ADAM),
    )
    axes[0, 0].set_title("Adam advances FC peak timing (MNIST)")
    _plot_peak_timing(
        axes[0, 1],
        gh_peak_steps(results["batchnorm"]["MNIST"], config),
        config.top_fc_fractions,
        ("No BatchNorm", "BatchNorm"),
        (STANDARD, BATCHNORM),
    )
    axes[0, 1].set_title("BatchNorm advances FC peak timing (MNIST)")
    _plot_cross_dataset_acceleration(axes[1, 0], results["adam"], config, dataset_order)
    axes[1, 0].set_title("Adam timing gain across datasets")
    _plot_cross_dataset_acceleration(axes[1, 1], results["batchnorm"], config, dataset_order)
    axes[1, 1].set_title("BatchNorm timing gain across datasets")
    fig.tight_layout()
    return fig


def summarize_gh_results(
    results: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    config: Fig5GHConfig,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Summarize paired peak-timing gains for reporting and cache audit."""

    summary: dict[str, dict[str, dict[str, list[float]]]] = {}
    for experiment, dataset_results in results.items():
        summary[experiment] = {}
        for dataset_name, result in dataset_results.items():
            peak_steps = gh_peak_steps(result, config)
            gain = peak_steps[:, 0] - peak_steps[:, 1]
            mean, ci = _mean_ci(gain, axis=0)
            summary[experiment][dataset_name] = {
                "top_fc_fractions": [float(value) for value in config.top_fc_fractions],
                "mean_earlier_peak_steps": np.asarray(mean).astype(float).tolist(),
                "ci95_earlier_peak_steps": np.asarray(ci).astype(float).tolist(),
                "positive_seed_fraction": np.mean(gain > 0, axis=0).astype(float).tolist(),
            }
    return summary


def i_performance_effects(result: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Paired seed-level effects; positive values favor FC-guided residual."""

    steps = np.asarray(result["steps"])
    curves = np.asarray(result["curves"])
    loss_auc = np.trapz(curves[:, :, 0], x=steps, axis=-1)
    accuracy_auc = np.trapz(curves[:, :, 1], x=steps, axis=-1)
    return {
        "loss_auc_benefit": loss_auc[:, 0] - loss_auc[:, 1],
        "accuracy_auc_benefit": accuracy_auc[:, 1] - accuracy_auc[:, 0],
    }




def _plot_dataset_effects(
    ax: plt.Axes,
    dataset_results: Mapping[str, Mapping[str, np.ndarray]],
    dataset_order: Sequence[str],
    metric: str,
    xlabel: str,
) -> None:
    rng = np.random.default_rng(20260722)
    y = np.arange(len(dataset_order))
    pooled = []
    for index, dataset_name in enumerate(dataset_order):
        values = i_performance_effects(dataset_results[dataset_name])[metric]
        pooled.append(values)
        jitter = rng.normal(0, 0.055, len(values))
        ax.scatter(values, y[index] + jitter, color=FC_GUIDED, s=9, alpha=0.28, linewidths=0)
        mean, ci = _mean_ci(values, axis=0)
        ax.errorbar(mean, y[index], xerr=ci, fmt="o", color=DARK, mfc=FC_GUIDED, ms=4.5, capsize=2)
    pooled_values = np.concatenate(pooled)
    span = max(float(np.ptp(pooled_values)), float(np.nanmax(np.abs(pooled_values))), 1.0)
    text_x = np.nanmax(pooled_values) + 0.06 * span
    for index, values in enumerate(pooled):
        positive = int(np.sum(values > 0))
        ax.text(text_x, index, f"{positive}/{len(values)}", va="center", fontsize=5.8)
    ax.axvline(0, color=DARK, lw=0.7, ls="--")
    ax.set_yticks(y, dataset_order)
    ax.set_xlabel(xlabel)
    ax.set_ylim(len(dataset_order) - 0.55, -0.55)
    ax.set_xlim(
        min(float(np.nanmin(pooled_values)), 0.0) - 0.08 * span,
        max(float(np.nanmax(pooled_values)), 0.0) + 0.24 * span,
    )
    ax.text(0.99, 1.02, "positive seeds", transform=ax.transAxes, ha="right", fontsize=5.5)




def plot_fig5i_stability(
    dataset_results: Mapping[str, Mapping[str, np.ndarray]],
    config: Fig5IConfig,
    dataset_order: Sequence[str] = ("MNIST", "FashionMNIST", "KMNIST"),
) -> plt.Figure:
    """Create the two cross-dataset stability panels retained as Fig. S6m-n."""

    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.25))
    for ax, label in zip(axes, "mn"):
        _panel_label(ax, label)

    _plot_dataset_effects(
        axes[0],
        dataset_results,
        dataset_order,
        "loss_auc_benefit",
        "Loss AUC benefit (standard − FC-guided)",
    )
    axes[0].set_title("Learning benefit across datasets")
    _plot_dataset_effects(
        axes[1],
        dataset_results,
        dataset_order,
        "accuracy_auc_benefit",
        "Accuracy AUC benefit (FC-guided − standard)",
    )
    axes[1].set_title("Accuracy benefit across datasets")
    fig.tight_layout()
    return fig


def summarize_i_results(
    dataset_results: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for dataset_name, result in dataset_results.items():
        effects = i_performance_effects(result)
        row: dict[str, float] = {}
        for metric, values in effects.items():
            mean, ci = _mean_ci(values, axis=0)
            row[f"mean_{metric}"] = float(mean)
            row[f"ci95_{metric}"] = float(ci)
            row[f"positive_seed_fraction_{metric}"] = float(np.mean(values > 0))
        summary[dataset_name] = row
    return summary




def export_figure_bundle(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str,
    source_data: Mapping[str, np.ndarray] | None = None,
    statistics: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "tiff": output_dir / f"{stem}.tiff",
        "png": output_dir / f"{stem}.png",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    if source_data is not None:
        source_path = output_dir / f"{stem}_source_data.npz"
        np.savez_compressed(source_path, **source_data)
        paths["source_data"] = source_path
    if statistics is not None:
        stats_path = output_dir / f"{stem}_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(statistics, handle, indent=2)
        paths["statistics"] = stats_path
    return paths
