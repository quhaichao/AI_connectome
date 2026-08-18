"""Analysis and plotting utilities for Supplementary Figure 2.

This module tests whether the operational functional-connectivity (FC) and
structural-connectivity (SC) definitions are robust to sampling, correlation
estimator, token aggregation, CNN unit granularity, SC aggregation and layer.

It reuses the checkpoint-compatible models and the local-Jacobian routines from
Supplementary Figure 1.  It never trains models or creates simulated figure
results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn as nn

from .fig1_jacobian import (
    cnn_kernel_average_sc_rows,
    collect_inputs,
    pearson_r,
    resolve_device,
    transformer_sample_jacobians,
)


ARCHITECTURES = ("MLP", "CNN", "Transformer")
ARCH_COLORS = {"MLP": "#3E75A6", "CNN": "#2A9D8F", "Transformer": "#B06AB3"}
BLUE = "#3E75A6"
TEAL = "#2A9D8F"
PURPLE = "#B06AB3"
ORANGE = "#D9822B"
RED = "#C84C4C"
GREY = "#6F7782"
LIGHT_GREY = "#E7E9EC"


@dataclass(frozen=True)
class SuppFig2Config:
    seed: int = 42
    mlp_n_samples: int = 512
    cnn_n_samples: int = 256
    transformer_n_samples: int = 64
    cnn_element_units: int = 256
    resampling_repeats: int = 50
    mlp_sample_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    cnn_sample_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    transformer_sample_sizes: tuple[int, ...] = (2, 4, 8, 16, 32)
    transformer_layers: tuple[int, ...] = (0, 1)
    transformer_chunk_size: int = 2
    token_modes: tuple[str, ...] = ("concat", "sample_mean", "position_average")
    transformer_sc_methods: tuple[str, ...] = ("signed_mean", "absolute_mean", "rms")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _first_tensor(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise TypeError("Each data-loader batch must be a tensor or begin with a tensor.")


def correlation_matrix(observations: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Compute unit-by-unit FC from observations-by-units data."""

    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("observations must have shape [observations, units].")
    if observations.shape[0] < 3 or observations.shape[1] < 2:
        raise ValueError("At least three observations and two units are required.")
    method = method.lower()
    if method == "spearman":
        observations = rankdata(observations, axis=0)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'.")
    matrix = np.corrcoef(observations, rowvar=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def row_similarity(profiles: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Correlate incoming SC profiles between units to obtain input similarity."""

    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("profiles must have shape [units, incoming features].")
    return correlation_matrix(profiles.T, method=method)


def matrix_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"Matrix shapes differ: {first.shape} versus {second.shape}.")
    if first.shape[0] == first.shape[1]:
        indices = np.triu_indices(first.shape[0], k=1)
        return pearson_r(first[indices], second[indices])
    return pearson_r(first.ravel(), second.ravel())


def fc_is_correlation(fc: np.ndarray, input_similarity: np.ndarray) -> float:
    return matrix_similarity(fc, input_similarity)


def extract_mlp_activations(
    model: nn.Module,
    data_loader: Iterable[Any],
    n_samples: int,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    device = resolve_device(device)
    model = model.to(device).eval()
    layers: list[list[torch.Tensor]] | None = None
    collected = 0
    with torch.no_grad():
        for batch in data_loader:
            if collected >= n_samples:
                break
            x = _first_tensor(batch).to(device)
            take = min(int(x.shape[0]), n_samples - collected)
            activations = model.get_hidden_activations(x[:take])
            if layers is None:
                layers = [[] for _ in activations]
            for index, activation in enumerate(activations):
                layers[index].append(activation.detach().cpu())
            collected += take
    if not layers or collected == 0:
        raise ValueError("The MLP loader yielded no samples.")
    return {
        f"hidden_{index + 1}": torch.cat(chunks, dim=0).numpy().astype(np.float32)
        for index, chunks in enumerate(layers)
    }


def extract_cnn_activations(
    model: nn.Module,
    data_loader: Iterable[Any],
    n_samples: int,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    device = resolve_device(device)
    model = model.to(device).eval()
    conv1: list[torch.Tensor] = []
    conv2: list[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for batch in data_loader:
            if collected >= n_samples:
                break
            x = _first_tensor(batch).to(device)
            take = min(int(x.shape[0]), n_samples - collected)
            model(x[:take])
            conv1.append(model.activations["conv1"].detach().cpu())
            conv2.append(model.activations["conv2"].detach().cpu())
            collected += take
    if collected == 0:
        raise ValueError("The CNN loader yielded no samples.")
    return {
        "conv1": torch.cat(conv1, dim=0).numpy().astype(np.float32),
        "conv2": torch.cat(conv2, dim=0).numpy().astype(np.float32),
    }


def extract_transformer_activations(
    model: nn.Module,
    data_loader: Iterable[Any],
    n_samples: int,
    layers: Sequence[int],
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    device = resolve_device(device)
    model = model.to(device).eval()
    requested = tuple(sorted(set(int(layer) for layer in layers)))
    if not requested or requested[0] < 0 or requested[-1] >= len(model.layers):
        raise IndexError("A requested Transformer layer does not exist.")
    outputs: dict[int, list[torch.Tensor]] = {layer: [] for layer in requested}
    collected = 0
    with torch.no_grad():
        for batch in data_loader:
            if collected >= n_samples:
                break
            token_ids = _first_tensor(batch).to(device)
            take = min(int(token_ids.shape[0]), n_samples - collected)
            layer_outputs = model.get_layer_outputs(token_ids[:take])
            for layer in requested:
                outputs[layer].append(layer_outputs[layer][1].detach().cpu())
            collected += take
    if collected == 0:
        raise ValueError("The Transformer loader yielded no samples.")
    return {
        f"layer_{layer}": torch.cat(chunks, dim=0).numpy().astype(np.float32)
        for layer, chunks in outputs.items()
    }


def cnn_unit_observations(
    activations: np.ndarray,
    definition: str,
    element_indices: Sequence[int] | None = None,
) -> np.ndarray:
    activations = np.asarray(activations)
    if activations.ndim != 4:
        raise ValueError("CNN activations must have shape [samples, channels, height, width].")
    definition = definition.lower()
    if definition == "element":
        observations = activations.reshape(activations.shape[0], -1)
        if element_indices is not None:
            observations = observations[:, np.asarray(element_indices, dtype=int)]
        return observations
    if definition == "channel":
        return activations.mean(axis=(2, 3))
    if definition == "spatial":
        return activations.mean(axis=1).reshape(activations.shape[0], -1)
    raise ValueError("CNN definition must be 'element', 'channel' or 'spatial'.")


def transformer_fc(
    grouped_activations: np.ndarray,
    token_mode: str = "concat",
    correlation: str = "pearson",
) -> np.ndarray:
    grouped_activations = np.asarray(grouped_activations)
    if grouped_activations.ndim != 3:
        raise ValueError("Transformer activations must have shape [samples, tokens, dimensions].")
    token_mode = token_mode.lower()
    if token_mode == "concat":
        observations = grouped_activations.reshape(-1, grouped_activations.shape[-1])
        return correlation_matrix(observations, correlation)
    if token_mode == "sample_mean":
        return correlation_matrix(grouped_activations.mean(axis=1), correlation)
    if token_mode == "position_average":
        matrices = [
            correlation_matrix(grouped_activations[:, position, :], correlation)
            for position in range(grouped_activations.shape[1])
        ]
        matrix = np.mean(matrices, axis=0)
        np.fill_diagonal(matrix, 1.0)
        return matrix
    raise ValueError("token_mode must be concat, sample_mean or position_average.")


def sample_size_convergence(
    grouped_data: np.ndarray,
    fc_builder: Callable[[np.ndarray], np.ndarray],
    sample_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> dict[str, np.ndarray]:
    grouped_data = np.asarray(grouped_data)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(grouped_data.shape[0])
    half = grouped_data.shape[0] // 2
    if half < 2:
        raise ValueError("At least four grouped samples are required.")
    estimation = grouped_data[permutation[:half]]
    reference = fc_builder(grouped_data[permutation[half : 2 * half]])
    sizes = np.asarray(sorted({int(size) for size in sample_sizes if 2 <= int(size) <= half}))
    if sizes.size == 0:
        raise ValueError("No requested sample size fits within the estimation split.")
    correlations = np.empty((sizes.size, repeats), dtype=float)
    for size_index, size in enumerate(sizes):
        for repeat in range(repeats):
            chosen = rng.choice(half, size=int(size), replace=False)
            correlations[size_index, repeat] = matrix_similarity(
                fc_builder(estimation[chosen]), reference
            )
    return {
        "sizes": sizes,
        "all": correlations,
        "mean": np.nanmean(correlations, axis=1),
        "low": np.nanpercentile(correlations, 2.5, axis=1),
        "high": np.nanpercentile(correlations, 97.5, axis=1),
    }


def repeated_split_reliability(
    grouped_data: np.ndarray,
    fc_builder: Callable[[np.ndarray], np.ndarray],
    repeats: int,
    seed: int,
) -> np.ndarray:
    grouped_data = np.asarray(grouped_data)
    rng = np.random.default_rng(seed)
    half = grouped_data.shape[0] // 2
    values = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        permutation = rng.permutation(grouped_data.shape[0])
        first = fc_builder(grouped_data[permutation[:half]])
        second = fc_builder(grouped_data[permutation[half : 2 * half]])
        values[repeat] = matrix_similarity(first, second)
    return values


def repeated_correlation_method_agreement(
    grouped_data: np.ndarray,
    pearson_builder: Callable[[np.ndarray], np.ndarray],
    spearman_builder: Callable[[np.ndarray], np.ndarray],
    repeats: int,
    seed: int,
) -> np.ndarray:
    grouped_data = np.asarray(grouped_data)
    rng = np.random.default_rng(seed)
    subset_size = max(4, grouped_data.shape[0] // 2)
    values = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        chosen = rng.choice(grouped_data.shape[0], size=subset_size, replace=False)
        subset = grouped_data[chosen]
        values[repeat] = matrix_similarity(
            pearson_builder(subset), spearman_builder(subset)
        )
    return values


def choose_cnn_elements(
    output_shape: Sequence[int], n_units: int, seed: int
) -> np.ndarray:
    total = int(np.prod(output_shape))
    if not 1 <= n_units <= total:
        raise ValueError(f"cnn_element_units must be between 1 and {total}.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=n_units, replace=False)).astype(np.int64)


def cnn_sc_profiles_by_definition(
    model: nn.Module,
    element_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    """Build SC profiles without retaining a full 1600 x 6272 array."""

    channels, height, width = model.get_layer_shapes()["conv2"]
    spatial_positions = height * width
    element_profiles = cnn_kernel_average_sc_rows(model, element_indices).astype(np.float32)
    input_dim = element_profiles.shape[1]
    channel_profiles = np.zeros((channels, input_dim), dtype=np.float32)
    spatial_profiles = np.zeros((spatial_positions, input_dim), dtype=np.float32)
    positions = np.arange(spatial_positions)
    for channel in range(channels):
        indices = channel * spatial_positions + positions
        rows = cnn_kernel_average_sc_rows(model, indices).astype(np.float32)
        channel_profiles[channel] = rows.mean(axis=0)
        spatial_profiles += rows
    spatial_profiles /= channels
    return {
        "element": element_profiles,
        "channel": channel_profiles,
        "spatial": spatial_profiles,
    }


def aggregate_transformer_sc(
    sample_jacobians: np.ndarray,
    method: str,
) -> np.ndarray:
    sample_jacobians = np.asarray(sample_jacobians, dtype=np.float64)
    method = method.lower()
    if method == "signed_mean":
        return sample_jacobians.mean(axis=0)
    if method == "absolute_mean":
        return np.abs(sample_jacobians).mean(axis=0)
    if method == "rms":
        return np.sqrt(np.square(sample_jacobians).mean(axis=0))
    raise ValueError("SC aggregation must be signed_mean, absolute_mean or rms.")


def _conv_channel_profiles(conv: nn.Conv2d) -> np.ndarray:
    return conv.weight.detach().cpu().numpy().reshape(conv.out_channels, -1)


def run_fig1_fc_supp_analysis(
    mlp_model: nn.Module,
    mlp_loader: Iterable[Any],
    cnn_model: nn.Module,
    cnn_loader: Iterable[Any],
    transformer_model: nn.Module,
    transformer_loader: Iterable[Any],
    config: SuppFig2Config,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run only the FC analyses retained as Supplementary Fig. S1a-d."""

    set_seed(config.seed)
    device = resolve_device(device)
    mlp_activations = extract_mlp_activations(
        mlp_model, mlp_loader, config.mlp_n_samples, device
    )
    cnn_activations = extract_cnn_activations(
        cnn_model, cnn_loader, config.cnn_n_samples, device
    )
    transformer_activations = extract_transformer_activations(
        transformer_model,
        transformer_loader,
        config.transformer_n_samples,
        config.transformer_layers,
        device,
    )
    default_transformer_layer = config.transformer_layers[0]

    cnn_element_indices = choose_cnn_elements(
        cnn_model.get_layer_shapes()["conv2"], config.cnn_element_units, config.seed
    )
    default_grouped = {
        "MLP": mlp_activations["hidden_2"],
        "CNN": cnn_unit_observations(
            cnn_activations["conv2"], "element", cnn_element_indices
        ),
        "Transformer": transformer_activations[f"layer_{default_transformer_layer}"],
    }
    default_builders: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "MLP": lambda data: correlation_matrix(data, "pearson"),
        "CNN": lambda data: correlation_matrix(data, "pearson"),
        "Transformer": lambda data: transformer_fc(data, "concat", "pearson"),
    }
    spearman_builders: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "MLP": lambda data: correlation_matrix(data, "spearman"),
        "CNN": lambda data: correlation_matrix(data, "spearman"),
        "Transformer": lambda data: transformer_fc(data, "concat", "spearman"),
    }
    size_map = {
        "MLP": config.mlp_sample_sizes,
        "CNN": config.cnn_sample_sizes,
        "Transformer": config.transformer_sample_sizes,
    }

    results: dict[str, Any] = {
        "architectures": np.asarray(ARCHITECTURES),
        "cnn_element_indices": cnn_element_indices,
    }
    heldout = []
    method_agreement = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        convergence = sample_size_convergence(
            default_grouped[architecture],
            default_builders[architecture],
            size_map[architecture],
            config.resampling_repeats,
            config.seed + architecture_index,
        )
        prefix = architecture.lower()
        for key, value in convergence.items():
            results[f"{prefix}_convergence_{key}"] = value
        heldout.append(
            repeated_split_reliability(
                default_grouped[architecture],
                default_builders[architecture],
                config.resampling_repeats,
                config.seed + 10 + architecture_index,
            )
        )
        method_agreement.append(
            repeated_correlation_method_agreement(
                default_grouped[architecture],
                default_builders[architecture],
                spearman_builders[architecture],
                config.resampling_repeats,
                config.seed + 20 + architecture_index,
            )
        )
        results[f"{prefix}_fc_pearson"] = default_builders[architecture](
            default_grouped[architecture]
        )
        results[f"{prefix}_fc_spearman"] = spearman_builders[architecture](
            default_grouped[architecture]
        )
    results["heldout_reliability_all"] = np.stack(heldout)
    results["pearson_spearman_agreement_all"] = np.stack(method_agreement)

    layer_labels: list[str] = []
    layer_architectures: list[str] = []
    layer_reliability: list[np.ndarray] = []

    for layer_index, key in enumerate(sorted(mlp_activations)):
        observations = mlp_activations[key]
        layer_labels.append(f"MLP H{layer_index + 1}")
        layer_architectures.append("MLP")
        layer_reliability.append(
            repeated_split_reliability(
                observations,
                lambda data: correlation_matrix(data),
                config.resampling_repeats,
                config.seed + 50 + layer_index,
            )
        )

    for layer_index, key in enumerate(("conv1", "conv2")):
        observations = cnn_unit_observations(cnn_activations[key], "channel")
        layer_labels.append(f"CNN C{layer_index + 1}")
        layer_architectures.append("CNN")
        layer_reliability.append(
            repeated_split_reliability(
                observations,
                lambda data: correlation_matrix(data),
                config.resampling_repeats,
                config.seed + 60 + layer_index,
            )
        )

    for layer_offset, layer in enumerate(config.transformer_layers):
        grouped = transformer_activations[f"layer_{layer}"]
        layer_labels.append(f"Transformer L{layer + 1}")
        layer_architectures.append("Transformer")
        layer_reliability.append(
            repeated_split_reliability(
                grouped,
                lambda data: transformer_fc(data, "concat"),
                config.resampling_repeats,
                config.seed + 70 + layer_offset,
            )
        )

    results["layer_labels"] = np.asarray(layer_labels)
    results["layer_architectures"] = np.asarray(layer_architectures)
    results["layer_reliability_all"] = np.stack(layer_reliability)
    results["metadata"] = {
        "config": asdict(config),
        "device": str(device),
        "notes": {
            "cnn_element_resampling": (
                f"A fixed seed-selected subset of {config.cnn_element_units} conv2 feature-map elements "
                "was used for repeated FC calculations."
            ),
            "intervals": "All plotted 95% intervals are computational resampling intervals, not independent training-run confidence intervals.",
        },
    }
    validate_results(results)
    return results


REQUIRED_KEYS = (
    "architectures",
    "heldout_reliability_all",
    "pearson_spearman_agreement_all",
    "layer_labels",
    "layer_architectures",
    "layer_reliability_all",
)


def validate_results(results: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in results]
    for architecture in ARCHITECTURES:
        prefix = architecture.lower()
        for suffix in ("sizes", "all", "mean", "low", "high"):
            key = f"{prefix}_convergence_{suffix}"
            if key not in results:
                missing.append(key)
    if missing:
        raise KeyError(f"Figure 1 FC supplement results are missing: {missing}")
    if np.shape(results["heldout_reliability_all"])[0] != len(ARCHITECTURES):
        raise ValueError("heldout_reliability_all does not match the architecture list.")


def save_results(results: Mapping[str, Any], path: str | Path) -> Path:
    validate_results(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in results.items() if key != "metadata"}
    arrays["metadata_json"] = np.asarray(json.dumps(results.get("metadata", {})))
    np.savez_compressed(path, **arrays)
    return path


def load_results(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        results: dict[str, Any] = {
            key: archive[key] for key in archive.files if key != "metadata_json"
        }
        metadata_text = str(archive["metadata_json"].item()) if "metadata_json" in archive else "{}"
    results["metadata"] = json.loads(metadata_text)
    validate_results(results)
    return results


def publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    return (
        np.nanmean(values, axis=-1),
        np.nanpercentile(values, 2.5, axis=-1),
        np.nanpercentile(values, 97.5, axis=-1),
    )


def _interval_plot(
    axis: plt.Axes,
    values: np.ndarray,
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    mean, low, high = _summary(values)
    x = np.arange(len(labels))
    for index in range(len(labels)):
        axis.errorbar(
            x[index],
            mean[index],
            yerr=[[mean[index] - low[index]], [high[index] - mean[index]]],
            fmt="o",
            ms=4,
            color=colors[index],
            ecolor=colors[index],
            elinewidth=1,
            capsize=2,
        )
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold", pad=2)
    axis.set_ylim(-0.05, 1.05)
    axis.axhline(0, color=LIGHT_GREY, lw=0.6, zorder=0)


def _overview(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Operational robustness tests", loc="left", fontweight="bold", pad=2)
    labels = (
        "Input\nsampling",
        "Correlation\nestimator",
        "Token\naggregation",
        "CNN unit\ngranularity",
        "SC\naggregation",
        "Network\nlayer",
    )
    colors = (BLUE, BLUE, PURPLE, TEAL, ORANGE, GREY)
    xs = np.linspace(0.02, 0.84, len(labels))
    for x, label, color in zip(xs, labels, colors):
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.34),
                0.13,
                0.34,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                edgecolor=color,
                facecolor=mpl.colors.to_rgba(color, 0.10),
                linewidth=0.8,
            )
        )
        axis.text(x + 0.065, 0.51, label, ha="center", va="center")
    for x in xs[:-1]:
        axis.add_patch(
            FancyArrowPatch(
                (x + 0.13, 0.51),
                (x + 0.16, 0.51),
                arrowstyle="-|>",
                mutation_scale=7,
                lw=0.7,
                color="#555555",
            )
        )
    axis.text(0.50, 0.12, "Stable FC/SC conclusion across reasonable operational choices", ha="center", color=GREY)


def _metric_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "YlGnBu",
) -> None:
    image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title, loc="left", fontweight="bold", pad=2)
    threshold = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > threshold else "black",
            )
    return image


def plot_fig1_fc_supp(
    results: Mapping[str, Any],
    output_prefix: str | Path | None = None,
    export_formats: Sequence[str] = ("svg", "pdf", "tiff", "png"),
    dpi: int = 600,
) -> tuple[plt.Figure, dict[str, Any]]:
    validate_results(results)
    publication_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.25, wspace=0.58)
    ax_a, ax_b, ax_c, ax_d = axes

    for architecture in ARCHITECTURES:
        prefix = architecture.lower()
        sizes = np.asarray(results[f"{prefix}_convergence_sizes"])
        mean = np.asarray(results[f"{prefix}_convergence_mean"])
        low = np.asarray(results[f"{prefix}_convergence_low"])
        high = np.asarray(results[f"{prefix}_convergence_high"])
        color = ARCH_COLORS[architecture]
        ax_b.fill_between(sizes, low, high, color=color, alpha=0.14, linewidth=0)
        ax_b.plot(sizes, mean, marker="o", ms=2.8, lw=1.1, color=color, label=architecture)
    ax_b.set_xscale("log", base=2)
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_xlabel("Input samples used for FC")
    ax_b.set_ylabel("Correlation with independent split")
    ax_b.set_title("FC sampling convergence", pad=2)
    ax_b.legend(loc="lower right")

    architecture_colors = [ARCH_COLORS[name] for name in ARCHITECTURES]
    _interval_plot(
        ax_a,
        results["pearson_spearman_agreement_all"],
        ARCHITECTURES,
        architecture_colors,
        "Pearson versus Spearman FC",
        "Matrix agreement",
    )
    _interval_plot(
        ax_c,
        results["heldout_reliability_all"],
        ARCHITECTURES,
        architecture_colors,
        "Held-out FC reproducibility",
        "Split-half matrix correlation",
    )

    layer_labels = [str(value) for value in results["layer_labels"]]
    layer_architectures = [str(value) for value in results["layer_architectures"]]
    layer_colors = [ARCH_COLORS[value] for value in layer_architectures]
    _interval_plot(
        ax_d,
        results["layer_reliability_all"],
        layer_labels,
        layer_colors,
        "Layer-wise FC reproducibility",
        "Split-half matrix correlation",
    )
    for label, axis in zip("abcd", axes):
        _panel_label(axis, label)

    statistics = {
        "heldout_reliability_mean": dict(
            zip(ARCHITECTURES, np.nanmean(results["heldout_reliability_all"], axis=1).tolist())
        ),
        "pearson_spearman_agreement_mean": dict(
            zip(ARCHITECTURES, np.nanmean(results["pearson_spearman_agreement_all"], axis=1).tolist())
        ),
        "layer_reliability_mean": dict(
            zip(layer_labels, np.nanmean(results["layer_reliability_all"], axis=1).tolist())
        ),
    }
    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for extension in export_formats:
            extension = extension.lower().lstrip(".")
            kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
            if extension in {"png", "tif", "tiff"}:
                kwargs["dpi"] = dpi
            fig.savefig(output_prefix.with_suffix(f".{extension}"), **kwargs)
        output_prefix.with_name(output_prefix.name + "_statistics.json").write_text(
            json.dumps(statistics, indent=2), encoding="utf-8"
        )
    return fig, statistics
