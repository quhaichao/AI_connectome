"""ANN-only robustness analyses for Supplementary Figure 3.

The inferential replicate is an independently initialized and trained model.
The module evaluates FC-IS correlations across at least 15 seeds and across
multiple datasets for MLPs, CNNs and Transformers. Biological matrices are
intentionally excluded from this revised figure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, t as student_t
import torch
import torch.nn as nn

from .fig1_jacobian import collect_inputs, resolve_device, transformer_sample_jacobians
from .fig1_fc import (
    correlation_matrix,
    extract_cnn_activations,
    extract_mlp_activations,
    extract_transformer_activations,
    row_similarity,
    transformer_fc,
)


ARCH_COLORS = {
    "MLP": "#4C78A8",
    "CNN": "#2A9D8F",
    "Transformer": "#8A67B2",
}
DATASET_COLORS = {
    "MNIST": "#7DA7D9",
    "FashionMNIST": "#5F8F7B",
    "CIFAR10": "#D08B65",
    "WikiText-2": "#A17DB8",
    "Penn Treebank": "#D5A14A",
}
GREY = "#6F7782"
LIGHT_GREY = "#E5E7EA"
DARK_GREY = "#34383F"


@dataclass(frozen=True)
class SuppFig3Config:
    """Training and analysis defaults aligned with the main code."""

    seeds: tuple[int, ...] = tuple(range(15))
    analysis_seed: int = 42
    minimum_publication_seeds: int = 15
    mlp_eval_samples: int = 1000
    cnn_eval_samples: int = 500
    cnn_units_per_layer: int = 256
    transformer_eval_samples: int = 64
    transformer_jacobian_samples: int = 32
    transformer_layers: tuple[int, ...] = (0, 1)
    transformer_chunk_size: int = 2
    association: str = "pearson"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_slug(value: str) -> str:
    """Return a stable filename component without changing dataset labels."""

    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-").lower()


def initialize_cnn_plateau(model: nn.Module, std: float = 1e-8) -> nn.Module:
    """Match the small-normal CNN initialization used in the main analysis."""

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    return model


def initialize_transformer_xavier(model: nn.Module) -> nn.Module:
    """Match the Transformer Xavier initialization used in the reference code."""

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return model


def train_mlp_model(
    model: nn.Module,
    train_loader: Iterable[Any],
    epochs: int = 5,
    learning_rate: float = 0.05,
    device: str | torch.device | None = None,
) -> nn.Module:
    device = resolve_device(device)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    return model.eval()


def train_cnn_model(
    model: nn.Module,
    train_loader: Iterable[Any],
    epochs: int = 2,
    learning_rate: float = 0.03,
    momentum: float = 0.9,
    device: str | torch.device | None = None,
) -> nn.Module:
    device = resolve_device(device)
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    return model.eval()


def train_transformer_model(
    model: nn.Module,
    train_loader: Iterable[Any],
    epochs: int = 3,
    learning_rate: float = 1e-4,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Train with the same next-sequence targets as the reference code."""

    device = resolve_device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for _ in range(epochs):
        model.train()
        for token_ids, targets in train_loader:
            token_ids, targets = token_ids.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()
    return model.eval()


def evaluate_classifier_accuracy(
    model: nn.Module,
    data_loader: Iterable[Any],
    device: str | torch.device | None = None,
) -> float:
    device = resolve_device(device)
    model = model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return float(correct / total) if total else float("nan")


def evaluate_transformer_perplexity(
    model: nn.Module,
    data_loader: Iterable[Any],
    device: str | torch.device | None = None,
) -> float:
    device = resolve_device(device)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for token_ids, targets in data_loader:
            token_ids, targets = token_ids.to(device), targets.to(device)
            logits = model(token_ids)
            total_loss += float(
                criterion(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                ).item()
            )
            total_tokens += int((targets != 0).sum().item())
    if total_tokens == 0:
        return float("nan")
    return float(np.exp(min(total_loss / total_tokens, 50.0)))


def _association(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method.lower() == "pearson":
        return float(pearsonr(x, y).statistic)
    if method.lower() == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError("Association must be 'pearson' or 'spearman'.")


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _fc_is_record(
    architecture: str,
    dataset: str,
    seed: int,
    layer: str,
    fc: np.ndarray,
    input_similarity: np.ndarray,
    method: str,
    performance_metric: str,
    performance_value: float,
) -> dict[str, Any]:
    fc_values = _upper_values(fc)
    is_values = _upper_values(input_similarity)
    valid = np.isfinite(fc_values) & np.isfinite(is_values)
    return {
        "architecture": architecture,
        "dataset": dataset,
        "seed": int(seed),
        "layer": layer,
        "r_fc_is": _association(is_values[valid], fc_values[valid], method),
        "n_units": int(fc.shape[0]),
        "n_pairs": int(valid.sum()),
        "performance_metric": performance_metric,
        "performance_value": float(performance_value),
    }


def analyze_mlp_run(
    model: nn.Module,
    eval_loader: Iterable[Any],
    seed: int,
    dataset: str,
    config: SuppFig3Config,
    device: str | torch.device | None = None,
) -> list[dict[str, Any]]:
    accuracy = evaluate_classifier_accuracy(model, eval_loader, device)
    activations = extract_mlp_activations(
        model, eval_loader, config.mlp_eval_samples, device
    )
    fc = correlation_matrix(activations["hidden_2"], "pearson")
    profiles = model.linear_layers[1].weight.detach().cpu().numpy()
    input_similarity = row_similarity(profiles, "pearson")
    return [_fc_is_record(
        "MLP", dataset, seed, "H2", fc, input_similarity,
        config.association, "validation_accuracy", accuracy,
    )]


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def conv_pool_kernel_average_rows(
    conv: nn.Conv2d,
    pool: nn.Module,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    output_indices: Sequence[int],
) -> np.ndarray:
    """Expand convolution kernels to pooled, spatially indexed input profiles."""

    input_channels, input_height, input_width = input_shape
    output_channels, output_height, output_width = output_shape
    kernel_h, kernel_w = _pair(conv.kernel_size)
    stride_h, stride_w = _pair(conv.stride)
    pad_h, pad_w = _pair(conv.padding)
    dilation_h, dilation_w = _pair(conv.dilation)
    pool_h, pool_w = _pair(pool.kernel_size)
    pool_stride_h, pool_stride_w = _pair(pool.stride or pool.kernel_size)
    pool_pad_h, pool_pad_w = _pair(pool.padding)
    weights = conv.weight.detach().cpu().numpy()
    rows = np.zeros(
        (len(output_indices), input_channels * input_height * input_width),
        dtype=np.float32,
    )

    for row_index, flat_output in enumerate(output_indices):
        output_channel, residual = divmod(
            int(flat_output), output_height * output_width
        )
        output_h, output_w = divmod(residual, output_width)
        if not 0 <= output_channel < output_channels:
            raise IndexError(f"Output index {flat_output} is outside the layer.")
        pool_positions = [
            (
                output_h * pool_stride_h - pool_pad_h + delta_h,
                output_w * pool_stride_w - pool_pad_w + delta_w,
            )
            for delta_h in range(pool_h)
            for delta_w in range(pool_w)
        ]
        scale = 1.0 / len(pool_positions)
        for prepool_h, prepool_w in pool_positions:
            for input_channel in range(input_channels):
                for kernel_delta_h in range(kernel_h):
                    for kernel_delta_w in range(kernel_w):
                        input_h = (
                            prepool_h * stride_h
                            - pad_h
                            + kernel_delta_h * dilation_h
                        )
                        input_w = (
                            prepool_w * stride_w
                            - pad_w
                            + kernel_delta_w * dilation_w
                        )
                        if 0 <= input_h < input_height and 0 <= input_w < input_width:
                            flat_input = (
                                input_channel * input_height * input_width
                                + input_h * input_width
                                + input_w
                            )
                            rows[row_index, flat_input] += (
                                weights[
                                    output_channel,
                                    input_channel,
                                    kernel_delta_h,
                                    kernel_delta_w,
                                ]
                                * scale
                            )
    return rows


def _fixed_unit_indices(
    output_shape: tuple[int, int, int], n_units: int, seed: int
) -> np.ndarray:
    total = int(np.prod(output_shape))
    n_units = min(int(n_units), total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=n_units, replace=False)).astype(np.int64)


def analyze_cnn_run(
    model: nn.Module,
    eval_loader: Iterable[Any],
    seed: int,
    dataset: str,
    config: SuppFig3Config,
    input_shape: tuple[int, int, int] = (1, 32, 32),
    device: str | torch.device | None = None,
) -> list[dict[str, Any]]:
    accuracy = evaluate_classifier_accuracy(model, eval_loader, device)
    activations = extract_cnn_activations(
        model, eval_loader, config.cnn_eval_samples, device
    )
    layer_shapes = model.get_layer_shapes()
    output_shape = layer_shapes["conv2"]
    indices = _fixed_unit_indices(
        output_shape, config.cnn_units_per_layer, config.analysis_seed + 1,
    )
    observations = activations["conv2"].reshape(activations["conv2"].shape[0], -1)[:, indices]
    profiles = conv_pool_kernel_average_rows(
        model.conv2, model.pool2, layer_shapes["conv1"], output_shape, indices,
    )
    fc = correlation_matrix(observations, "pearson")
    input_similarity = row_similarity(profiles, "pearson")
    return [_fc_is_record(
        "CNN", dataset, seed, "C2", fc, input_similarity,
        config.association, "validation_accuracy", accuracy,
    )]


def analyze_transformer_run(
    model: nn.Module,
    eval_loader: Iterable[Any],
    seed: int,
    dataset: str,
    config: SuppFig3Config,
    device: str | torch.device | None = None,
) -> list[dict[str, Any]]:
    perplexity = evaluate_transformer_perplexity(model, eval_loader, device)
    activations = extract_transformer_activations(
        model,
        eval_loader,
        config.transformer_eval_samples,
        config.transformer_layers,
        device,
    )
    inputs = collect_inputs(eval_loader, config.transformer_jacobian_samples)
    layer = max(config.transformer_layers)
    sample_jacobians = transformer_sample_jacobians(
        model, inputs, layer_index=layer, device=device,
        chunk_size=config.transformer_chunk_size,
    )
    sc_profiles = sample_jacobians.mean(axis=0)
    fc = transformer_fc(activations[f"layer_{layer}"], "concat", "pearson")
    input_similarity = row_similarity(sc_profiles, "pearson")
    return [_fc_is_record(
        "Transformer", dataset, seed, f"L{layer + 1}", fc, input_similarity,
        config.association, "validation_perplexity", perplexity,
    )]


RECORD_KEYS = (
    "architecture",
    "dataset",
    "seed",
    "layer",
    "r_fc_is",
    "n_units",
    "n_pairs",
    "performance_metric",
    "performance_value",
)


def save_run_records(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    if not records:
        raise ValueError("At least one run record is required.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: np.asarray([record[key] for record in records]) for key in RECORD_KEYS
    }
    np.savez_compressed(path, **payload)
    return path


def load_run_records(path: str | Path) -> list[dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = [key for key in RECORD_KEYS if key not in archive]
        if missing:
            raise KeyError(f"Run record cache is missing: {missing}")
        n_records = len(archive["seed"])
        records: list[dict[str, Any]] = []
        for index in range(n_records):
            records.append(
                {
                    "architecture": str(archive["architecture"][index]),
                    "dataset": str(archive["dataset"][index]),
                    "seed": int(archive["seed"][index]),
                    "layer": str(archive["layer"][index]),
                    "r_fc_is": float(archive["r_fc_is"][index]),
                    "n_units": int(archive["n_units"][index]),
                    "n_pairs": int(archive["n_pairs"][index]),
                    "performance_metric": str(archive["performance_metric"][index]),
                    "performance_value": float(archive["performance_value"][index]),
                }
            )
    return records


def assemble_results(
    records: Sequence[Mapping[str, Any]],
    config: SuppFig3Config,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one ANN record is required.")
    results: dict[str, Any] = {
        "architecture": np.asarray([record["architecture"] for record in records]),
        "dataset": np.asarray([record["dataset"] for record in records]),
        "seed": np.asarray([record["seed"] for record in records], dtype=int),
        "layer": np.asarray([record["layer"] for record in records]),
        "r_fc_is": np.asarray([record["r_fc_is"] for record in records], dtype=float),
        "n_units": np.asarray([record["n_units"] for record in records], dtype=int),
        "n_pairs": np.asarray([record["n_pairs"] for record in records], dtype=int),
        "performance_metric": np.asarray(
            [record["performance_metric"] for record in records]
        ),
        "performance_value": np.asarray(
            [record["performance_value"] for record in records], dtype=float
        ),
        "metadata": {
            "config": asdict(config),
            "inferential_unit": "independently initialized and trained model seed",
            **dict(metadata or {}),
        },
    }
    validate_results(results, minimum_seeds=1)
    return results


RESULT_KEYS = (
    "architecture",
    "dataset",
    "seed",
    "layer",
    "r_fc_is",
    "n_units",
    "n_pairs",
    "performance_metric",
    "performance_value",
)


def validate_results(
    results: Mapping[str, Any],
    minimum_seeds: int = 1,
    required_groups: Mapping[str, Sequence[str]] | None = None,
) -> None:
    missing = [key for key in RESULT_KEYS if key not in results]
    if missing:
        raise KeyError(f"Supplementary Figure 3 results are missing: {missing}")
    length = len(results["r_fc_is"])
    for key in RESULT_KEYS:
        if len(results[key]) != length:
            raise ValueError(f"{key} does not align with r_fc_is.")
    architectures = np.asarray(results["architecture"]).astype(str)
    datasets = np.asarray(results["dataset"]).astype(str)
    seeds = np.asarray(results["seed"], dtype=int)
    layers = np.asarray(results["layer"]).astype(str)
    duplicate_rows = set()
    for row in zip(architectures, datasets, seeds, layers):
        if row in duplicate_rows:
            raise ValueError(f"Duplicate architecture/dataset/seed/layer record: {row}")
        duplicate_rows.add(row)
    if required_groups is not None:
        for architecture, dataset_names in required_groups.items():
            for dataset_name in dataset_names:
                mask = (architectures == architecture) & (datasets == dataset_name)
                group_seeds = np.unique(seeds[mask])
                if group_seeds.size < minimum_seeds:
                    raise ValueError(
                        f"{architecture}/{dataset_name} has {group_seeds.size} seeds; "
                        f"at least {minimum_seeds} are required."
                    )


def save_results(results: Mapping[str, Any], path: str | Path) -> Path:
    validate_results(results, minimum_seeds=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(results[key]) for key in RESULT_KEYS}
    arrays["metadata_json"] = np.asarray(json.dumps(results.get("metadata", {})))
    np.savez_compressed(path, **arrays)
    return path


def load_results(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        results: dict[str, Any] = {key: archive[key] for key in RESULT_KEYS}
        metadata_text = (
            str(archive["metadata_json"].item())
            if "metadata_json" in archive
            else "{}"
        )
    results["metadata"] = json.loads(metadata_text)
    validate_results(results, minimum_seeds=1)
    return results


def publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
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
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(student_t.ppf(0.975, df=values.size - 1))
    return mean, mean - critical * sem, mean + critical * sem


def _arrays(results: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(results["architecture"]).astype(str),
        np.asarray(results["dataset"]).astype(str),
        np.asarray(results["seed"], dtype=int),
        np.asarray(results["layer"]).astype(str),
        np.asarray(results["r_fc_is"], dtype=float),
    )


def _plot_seed_layers(
    axis: plt.Axes,
    results: Mapping[str, Any],
    architecture: str,
    dataset_name: str,
) -> None:
    architectures, datasets, seeds, layers, values = _arrays(results)
    mask = (architectures == architecture) & (datasets == dataset_name)
    layer_order = list(dict.fromkeys(layers[mask].tolist()))
    seed_order = np.unique(seeds[mask])
    x = np.arange(len(layer_order))
    color = ARCH_COLORS[architecture]
    for seed in seed_order:
        y = []
        for layer in layer_order:
            selected = values[mask & (seeds == seed) & (layers == layer)]
            y.append(float(selected[0]) if selected.size else np.nan)
        axis.plot(
            x,
            y,
            color=mpl.colors.to_rgba(color, 0.24),
            lw=0.75,
            marker="o",
            ms=2.4,
        )
    for layer_index, layer in enumerate(layer_order):
        layer_values = values[mask & (layers == layer)]
        mean, low, high = _mean_ci(layer_values)
        axis.errorbar(
            layer_index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            ms=5,
            color=color,
            ecolor=color,
            capsize=2.5,
            lw=1.25,
            zorder=4,
        )
    axis.set_xticks(x, layer_order)
    axis.axhline(0, color=LIGHT_GREY, lw=0.75, zorder=0)
    axis.set_ylabel("FC–IS correlation (r)")
    axis.set_title(
        f"{architecture}: {dataset_name}, independent seeds",
        loc="left",
        fontweight="bold",
        pad=2,
    )
    axis.text(
        0.03,
        0.04,
        f"n = {len(seed_order)} seeds\nmean and 95% CI",
        transform=axis.transAxes,
        color=GREY,
        va="bottom",
    )


def _plot_dataset_robustness(
    axis: plt.Axes,
    results: Mapping[str, Any],
    architecture: str,
    dataset_order: Sequence[str],
    primary_layer: str,
    jitter_seed: int,
) -> None:
    architectures, datasets, seeds, layers, values = _arrays(results)
    rng = np.random.default_rng(jitter_seed)
    for dataset_index, dataset_name in enumerate(dataset_order):
        mask = (
            (architectures == architecture)
            & (datasets == dataset_name)
            & (layers == primary_layer)
        )
        dataset_values = values[mask]
        color = DATASET_COLORS.get(dataset_name, ARCH_COLORS[architecture])
        jitter = rng.normal(0, 0.055, size=dataset_values.size)
        axis.scatter(
            dataset_index + jitter,
            dataset_values,
            s=15,
            color=mpl.colors.to_rgba(color, 0.65),
            edgecolors="none",
            zorder=2,
        )
        mean, low, high = _mean_ci(dataset_values)
        axis.errorbar(
            dataset_index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            ms=5,
            color=color,
            ecolor=color,
            capsize=2.5,
            lw=1.25,
            zorder=4,
        )
    axis.set_xticks(
        np.arange(len(dataset_order)),
        [name.replace("FashionMNIST", "Fashion-\nMNIST").replace("Penn Treebank", "Penn\nTreebank") for name in dataset_order],
    )
    axis.axhline(0, color=LIGHT_GREY, lw=0.75, zorder=0)
    axis.set_ylabel(f"FC–IS correlation (r), {primary_layer}")
    axis.set_title(
        f"{architecture}: cross-dataset robustness",
        loc="left",
        fontweight="bold",
        pad=2,
    )
    axis.text(
        0.03,
        0.04,
        "Each point is one independently trained model",
        transform=axis.transAxes,
        color=GREY,
        va="bottom",
        fontsize=5.7,
    )


def plot_fig2_fc_is_supp(
    results: Mapping[str, Any],
    output_prefix: str | Path | None = None,
    dataset_order: Mapping[str, Sequence[str]] | None = None,
    primary_layers: Mapping[str, str] | None = None,
    minimum_seeds: int = 15,
    export_formats: Sequence[str] = ("svg", "pdf", "tiff", "png"),
    dpi: int = 600,
) -> tuple[plt.Figure, dict[str, Any]]:
    dataset_order = dict(
        dataset_order
        or {
            "MLP": ("MNIST", "FashionMNIST", "CIFAR10"),
            "CNN": ("MNIST", "FashionMNIST", "CIFAR10"),
            "Transformer": ("WikiText-2", "Penn Treebank"),
        }
    )
    primary_layers = dict(
        primary_layers or {"MLP": "H2", "CNN": "C2", "Transformer": "L2"}
    )
    validate_results(
        results,
        minimum_seeds=minimum_seeds,
        required_groups=dataset_order,
    )
    architectures_check, datasets_check, seeds_check, layers_check, _ = _arrays(results)
    for architecture, dataset_names in dataset_order.items():
        primary_layer = primary_layers[architecture]
        for dataset_name in dataset_names:
            mask = (
                (architectures_check == architecture)
                & (datasets_check == dataset_name)
                & (layers_check == primary_layer)
            )
            n_primary_seeds = np.unique(seeds_check[mask]).size
            if n_primary_seeds < minimum_seeds:
                raise ValueError(
                    f"{architecture}/{dataset_name}/{primary_layer} has "
                    f"{n_primary_seeds} seeds; at least {minimum_seeds} are required."
                )
    publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    architectures = ("MLP", "CNN", "Transformer")

    for architecture_index, (axis, architecture) in enumerate(
        zip(axes, architectures)
    ):
        _plot_dataset_robustness(
            axis,
            results,
            architecture,
            dataset_order[architecture],
            primary_layers[architecture],
            jitter_seed=42 + architecture_index,
        )

    for axis in axes:
        axis.set_ylim(-0.25, 1.02)
        axis.tick_params(length=2.5, width=0.7)
    _panel_label(axes[0], "a")
    fig.subplots_adjust(
        left=0.085,
        right=0.98,
        top=0.91,
        bottom=0.25,
        wspace=0.44,
    )

    architectures_array, datasets_array, seeds, layers, values = _arrays(results)
    performance_metrics = np.asarray(results["performance_metric"]).astype(str)
    performance_values = np.asarray(results["performance_value"], dtype=float)
    statistics: dict[str, Any] = {
        "inferential_unit": "independently initialized and trained model seed",
        "minimum_seeds_per_architecture_dataset": int(minimum_seeds),
        "association": "Pearson correlation between FC and IS edge values",
        "groups": {},
    }
    for architecture in architectures:
        statistics["groups"][architecture] = {}
        for dataset_name in dataset_order[architecture]:
            group_mask = (
                (architectures_array == architecture)
                & (datasets_array == dataset_name)
            )
            statistics["groups"][architecture][dataset_name] = {}
            for layer in list(dict.fromkeys(layers[group_mask].tolist())):
                mask = group_mask & (layers == layer)
                group_values = values[mask]
                mean, low, high = _mean_ci(group_values)
                unique_seeds = np.unique(seeds[mask])
                performance = performance_values[mask]
                metric_names = np.unique(performance_metrics[mask]).tolist()
                statistics["groups"][architecture][dataset_name][layer] = {
                    "n_independent_seeds": int(unique_seeds.size),
                    "mean_r_fc_is": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "seed_values": group_values.tolist(),
                    "performance_metric": metric_names,
                    "performance_values": performance.tolist(),
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


__all__ = [
    "SuppFig3Config",
    "analyze_cnn_run",
    "analyze_mlp_run",
    "analyze_transformer_run",
    "assemble_results",
    "initialize_cnn_plateau",
    "initialize_transformer_xavier",
    "load_results",
    "load_run_records",
    "plot_fig2_fc_is_supp",
    "safe_slug",
    "save_results",
    "save_run_records",
    "set_seed",
    "train_cnn_model",
    "train_mlp_model",
    "train_transformer_model",
    "validate_results",
]
