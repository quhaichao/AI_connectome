"""Computation and plotting utilities for Supplementary Figure 1.

Scientific question
-------------------
Do the structural-connectivity (SC) definitions used in the manuscript recover
explicit pathway weights when those weights exist, and are Jacobian-derived SC
estimates reproducible when a single explicit pathway matrix does not exist?

The implementation follows the operational definitions used in the main code:

* MLP: compare the hidden-to-hidden weight matrix with the expected local
  Jacobian of the post-activation hidden units.
* CNN: spatially expand and pool-average the second convolutional kernel, then
  compare admissible edges with the expected local Jacobian of the full
  Conv-BN-activation-pooling block.
* Transformer: calculate per-sample local Jacobians for one Transformer block
  and quantify independent split-half reliability and sample-size convergence.

The functions accept trained model objects and data loaders.  They do not train
models, download data or silently substitute simulated results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn as nn


BLUE = "#3E75A6"
ORANGE = "#D9822B"
TEAL = "#2A9D8F"
GREY = "#6F7782"
LIGHT_GREY = "#E7E9EC"
RED = "#C84C4C"
CMAP = "RdBu_r"


@dataclass(frozen=True)
class SuppFig1Config:
    """Analysis settings stored with the exported numerical results."""

    seed: int = 42
    mlp_target_linear_index: int = 1
    mlp_n_samples: int = 512
    cnn_n_samples: int = 64
    cnn_n_output_units: int = 64
    transformer_layer_index: int = 0
    transformer_n_samples: int = 64
    transformer_chunk_size: int = 2
    transformer_repeats: int = 100
    transformer_sample_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_tensor(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise TypeError("Each data-loader batch must be a tensor or begin with a tensor.")


def collect_inputs(data_loader: Iterable[Any], n_samples: int) -> torch.Tensor:
    """Collect a fixed prefix from a loader without changing sample values."""

    chunks: list[torch.Tensor] = []
    count = 0
    for batch in data_loader:
        x = _first_tensor(batch)
        take = min(int(x.shape[0]), n_samples - count)
        if take > 0:
            chunks.append(x[:take].detach().cpu())
            count += take
        if count >= n_samples:
            break
    if count < n_samples:
        raise ValueError(f"Requested {n_samples} samples, but the loader yielded {count}.")
    return torch.cat(chunks, dim=0)


def _off_diagonal_mask(shape: tuple[int, int]) -> np.ndarray:
    if shape[0] != shape[1]:
        return np.ones(shape, dtype=bool)
    return ~np.eye(shape[0], dtype=bool)


def paired_values(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, aligned values from matrices of equal shape."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"Matrix shapes differ: {first.shape} versus {second.shape}.")
    if mask is None:
        mask = _off_diagonal_mask(first.shape)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(first) & np.isfinite(second)
    return first[mask], second[mask]


def pearson_r(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 3 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def spearman_r(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 3:
        return float("nan")
    return pearson_r(rankdata(first), rankdata(second))


def _find_mlp_target_modules(
    model: nn.Module, target_linear_index: int
) -> tuple[list[nn.Module], nn.Linear, nn.Module]:
    if not hasattr(model, "linear_layers") or not hasattr(model, "network"):
        raise AttributeError("MLP must expose 'linear_layers' and sequential 'network'.")
    linear_layers = list(model.linear_layers)
    if target_linear_index <= 0 or target_linear_index >= len(linear_layers) - 1:
        raise ValueError("target_linear_index must identify a non-first hidden layer.")
    target_linear = linear_layers[target_linear_index]
    sequential_modules = list(model.network)
    target_position = next(
        (index for index, module in enumerate(sequential_modules) if module is target_linear),
        None,
    )
    if target_position is None:
        raise RuntimeError("Target linear layer was not found in model.network.")
    activation_types = (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.GELU)
    target_activation = next(
        (
            module
            for module in sequential_modules[target_position + 1 :]
            if isinstance(module, activation_types)
        ),
        None,
    )
    if target_activation is None:
        raise RuntimeError("No element-wise activation follows the target hidden layer.")
    prefix = sequential_modules[:target_position]
    return prefix, target_linear, target_activation


def mlp_weight_sc(model: nn.Module, target_linear_index: int = 1) -> np.ndarray:
    """Return the explicit hidden-to-hidden weight matrix [output, input]."""

    weight = model.linear_layers[target_linear_index].weight
    return to_numpy(weight).astype(np.float64, copy=False)


def mlp_expected_jacobian_sc(
    model: nn.Module,
    data_loader: Iterable[Any],
    target_linear_index: int = 1,
    n_samples: int = 512,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Estimate E[d hidden_l / d hidden_(l-1)] over fixed inputs."""

    device = resolve_device(device)
    model = model.to(device).eval()
    prefix, target_linear, target_activation = _find_mlp_target_modules(
        model, target_linear_index
    )
    output_dim = int(target_linear.out_features)
    input_dim = int(target_linear.in_features)
    jacobian_sum = torch.zeros(output_dim, input_dim, device=device)
    collected = 0

    for batch in data_loader:
        if collected >= n_samples:
            break
        x = _first_tensor(batch).to(device)
        take = min(int(x.shape[0]), n_samples - collected)
        x = x[:take].view(take, -1)
        with torch.no_grad():
            previous = x
            for module in prefix:
                previous = module(previous)
        previous = previous.detach().requires_grad_(True)
        current = target_activation(target_linear(previous))
        for output_index in range(output_dim):
            gradient = torch.autograd.grad(
                current[:, output_index].sum(),
                previous,
                retain_graph=output_index < output_dim - 1,
                create_graph=False,
            )[0]
            jacobian_sum[output_index] += gradient.sum(dim=0).detach()
        collected += take

    if collected == 0:
        raise ValueError("The MLP data loader yielded no samples.")
    return to_numpy(jacobian_sum / collected).astype(np.float64, copy=False)


def run_mlp_validation(
    model: nn.Module,
    data_loader: Iterable[Any],
    config: SuppFig1Config,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    weight = mlp_weight_sc(model, config.mlp_target_linear_index)
    jacobian = mlp_expected_jacobian_sc(
        model,
        data_loader,
        target_linear_index=config.mlp_target_linear_index,
        n_samples=config.mlp_n_samples,
        device=device,
    )
    return {"mlp_weight_sc": weight, "mlp_jacobian_sc": jacobian}


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def choose_cnn_output_units(
    output_shape: tuple[int, int, int], n_units: int, seed: int = 42
) -> np.ndarray:
    total = int(np.prod(output_shape))
    if not 1 <= n_units <= total:
        raise ValueError(f"n_units must be between 1 and {total}.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=n_units, replace=False)).astype(np.int64)


def cnn_kernel_average_sc_rows(
    model: nn.Module,
    output_unit_indices: Sequence[int],
) -> np.ndarray:
    """Expand Conv2 kernels to pooled spatial edges used in the main analysis.

    Each pooled output is represented by the mean of the kernels contributing
    to its pre-pooling positions.  The output contains only requested rows to
    avoid allocating a full 1600 x 6272 matrix unless it is truly needed.
    """

    if not all(hasattr(model, name) for name in ("conv2", "pool2", "get_layer_shapes")):
        raise AttributeError("CNN must expose conv2, pool2 and get_layer_shapes().")
    input_channels, input_height, input_width = model.get_layer_shapes()["conv1"]
    output_channels, output_height, output_width = model.get_layer_shapes()["conv2"]
    conv: nn.Conv2d = model.conv2
    pool: nn.MaxPool2d | nn.AvgPool2d = model.pool2
    kernel_h, kernel_w = _pair(conv.kernel_size)
    stride_h, stride_w = _pair(conv.stride)
    pad_h, pad_w = _pair(conv.padding)
    dilation_h, dilation_w = _pair(conv.dilation)
    pool_h, pool_w = _pair(pool.kernel_size)
    pool_stride_h, pool_stride_w = _pair(pool.stride or pool.kernel_size)
    pool_pad_h, pool_pad_w = _pair(pool.padding)
    weights = to_numpy(conv.weight)

    rows = np.zeros((len(output_unit_indices), input_channels * input_height * input_width))
    for row_index, flat_output_index in enumerate(output_unit_indices):
        channel_out, residual = divmod(int(flat_output_index), output_height * output_width)
        output_h, output_w = divmod(residual, output_width)
        if not 0 <= channel_out < output_channels:
            raise IndexError(f"CNN output unit {flat_output_index} is outside the layer.")
        valid_pool_positions: list[tuple[int, int]] = []
        for pool_delta_h in range(pool_h):
            for pool_delta_w in range(pool_w):
                prepool_h = output_h * pool_stride_h - pool_pad_h + pool_delta_h
                prepool_w = output_w * pool_stride_w - pool_pad_w + pool_delta_w
                valid_pool_positions.append((prepool_h, prepool_w))
        scale = 1.0 / max(len(valid_pool_positions), 1)

        for prepool_h, prepool_w in valid_pool_positions:
            for channel_in in range(input_channels):
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
                                channel_in * input_height * input_width
                                + input_h * input_width
                                + input_w
                            )
                            rows[row_index, flat_input] += (
                                weights[
                                    channel_out,
                                    channel_in,
                                    kernel_delta_h,
                                    kernel_delta_w,
                                ]
                                * scale
                            )
    return rows


def cnn_expected_jacobian_sc_rows(
    model: nn.Module,
    data_loader: Iterable[Any],
    output_unit_indices: Sequence[int],
    n_samples: int = 64,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Estimate E[d pooled_conv2 / d pooled_conv1] for selected outputs."""

    device = resolve_device(device)
    model = model.to(device).eval()
    input_shape = model.get_layer_shapes()["conv1"]
    input_dim = int(np.prod(input_shape))
    output_dim = int(np.prod(model.get_layer_shapes()["conv2"]))
    indices = np.asarray(output_unit_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= output_dim):
        raise IndexError("At least one CNN output-unit index is outside conv2.")
    jacobian_sum = torch.zeros(len(indices), input_dim, device=device)
    collected = 0

    for batch in data_loader:
        if collected >= n_samples:
            break
        x = _first_tensor(batch).to(device)
        take = min(int(x.shape[0]), n_samples - collected)
        x = x[:take]
        with torch.no_grad():
            previous = model.pool1(model.act(model.bn1(model.conv1(x))))
        previous = previous.detach().requires_grad_(True)
        current = model.pool2(model.act(model.bn2(model.conv2(previous))))
        current = current.reshape(take, -1)
        for row_index, output_index in enumerate(indices):
            gradient = torch.autograd.grad(
                current[:, int(output_index)].sum(),
                previous,
                retain_graph=row_index < len(indices) - 1,
                create_graph=False,
            )[0]
            jacobian_sum[row_index] += gradient.reshape(take, -1).sum(dim=0).detach()
        collected += take

    if collected == 0:
        raise ValueError("The CNN data loader yielded no samples.")
    return to_numpy(jacobian_sum / collected).astype(np.float64, copy=False)


def run_cnn_validation(
    model: nn.Module,
    data_loader: Iterable[Any],
    config: SuppFig1Config,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    output_shape = model.get_layer_shapes()["conv2"]
    indices = choose_cnn_output_units(
        output_shape, config.cnn_n_output_units, config.seed
    )
    weight = cnn_kernel_average_sc_rows(model, indices)
    jacobian = cnn_expected_jacobian_sc_rows(
        model,
        data_loader,
        indices,
        n_samples=config.cnn_n_samples,
        device=device,
    )
    return {
        "cnn_weight_sc": weight,
        "cnn_jacobian_sc": jacobian,
        "cnn_output_indices": indices,
    }


def transformer_sample_jacobians(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_index: int = 0,
    device: str | torch.device | None = None,
    chunk_size: int = 2,
) -> np.ndarray:
    """Return one token-averaged local Transformer Jacobian per input sample."""

    try:
        from torch.func import jacrev, vmap
    except ImportError as exc:
        raise ImportError("Transformer validation requires torch.func (PyTorch >= 2.0).") from exc

    device = resolve_device(device)
    model = model.to(device).eval()
    if not 0 <= layer_index < len(model.layers):
        raise IndexError(f"Transformer layer {layer_index} does not exist.")
    layer = model.layers[layer_index]

    def single_sample_forward(hidden_input: torch.Tensor) -> torch.Tensor:
        output = layer(hidden_input.unsqueeze(0))
        return output.mean(dim=1).squeeze(0)

    batched_jacobian = vmap(jacrev(single_sample_forward))
    all_jacobians: list[torch.Tensor] = []
    for start in range(0, int(inputs.shape[0]), chunk_size):
        token_ids = inputs[start : start + chunk_size].to(device)
        length = int(token_ids.shape[1])
        positions = torch.arange(length, device=device).unsqueeze(0)
        with torch.no_grad():
            hidden = model.embedding(token_ids) + model.pos_enc(positions)
            for previous_layer in model.layers[:layer_index]:
                hidden = previous_layer(hidden)
        # [batch, d_out, token, d_in] -> average influence across input tokens.
        jacobian = batched_jacobian(hidden).mean(dim=2)
        all_jacobians.append(jacobian.detach().cpu())
    return torch.cat(all_jacobians, dim=0).numpy().astype(np.float64, copy=False)


def transformer_reliability_from_sample_jacobians(
    sample_jacobians: np.ndarray,
    sample_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32),
    repeats: int = 100,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Quantify independent split-half reliability and convergence."""

    sample_jacobians = np.asarray(sample_jacobians, dtype=float)
    if sample_jacobians.ndim != 3 or sample_jacobians.shape[1] != sample_jacobians.shape[2]:
        raise ValueError("sample_jacobians must have shape [samples, d_model, d_model].")
    if sample_jacobians.shape[0] < 4:
        raise ValueError("At least four Transformer samples are required.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(sample_jacobians.shape[0])
    half = sample_jacobians.shape[0] // 2
    estimation_pool = sample_jacobians[permutation[:half]]
    reference_pool = sample_jacobians[permutation[half : 2 * half]]
    split_a = estimation_pool.mean(axis=0)
    split_b = reference_pool.mean(axis=0)
    mask = _off_diagonal_mask(split_a.shape)
    split_half_r = pearson_r(split_a[mask], split_b[mask])

    usable_sizes = np.asarray(
        sorted({int(size) for size in sample_sizes if 1 <= int(size) <= half}),
        dtype=int,
    )
    if usable_sizes.size == 0:
        raise ValueError("No requested Transformer sample size fits within a split half.")
    reference = split_b
    all_correlations = np.empty((usable_sizes.size, repeats), dtype=float)
    for size_index, size in enumerate(usable_sizes):
        for repeat in range(repeats):
            chosen = rng.choice(half, size=int(size), replace=False)
            estimate = estimation_pool[chosen].mean(axis=0)
            all_correlations[size_index, repeat] = pearson_r(
                estimate[mask], reference[mask]
            )

    return {
        "transformer_split_a_sc": split_a,
        "transformer_split_b_sc": split_b,
        "transformer_split_half_r": np.asarray([split_half_r]),
        "transformer_sample_sizes": usable_sizes,
        "transformer_convergence_all": all_correlations,
        "transformer_convergence_mean": np.nanmean(all_correlations, axis=1),
        "transformer_convergence_low": np.nanpercentile(all_correlations, 2.5, axis=1),
        "transformer_convergence_high": np.nanpercentile(all_correlations, 97.5, axis=1),
    }


def run_transformer_validation(
    model: nn.Module,
    data_loader: Iterable[Any],
    config: SuppFig1Config,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    inputs = collect_inputs(data_loader, config.transformer_n_samples)
    per_sample = transformer_sample_jacobians(
        model,
        inputs,
        layer_index=config.transformer_layer_index,
        device=device,
        chunk_size=config.transformer_chunk_size,
    )
    results = transformer_reliability_from_sample_jacobians(
        per_sample,
        sample_sizes=config.transformer_sample_sizes,
        repeats=config.transformer_repeats,
        seed=config.seed,
    )
    results["transformer_sample_jacobians"] = per_sample
    return results


def assemble_results(
    cnn_results: Mapping[str, np.ndarray],
    transformer_results: Mapping[str, np.ndarray],
    config: SuppFig1Config,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for section in (cnn_results, transformer_results):
        results.update(section)
    results["metadata"] = {"config": asdict(config), **dict(metadata or {})}
    validate_results(results)
    return results


REQUIRED_ARRAY_KEYS = (
    "cnn_weight_sc",
    "cnn_jacobian_sc",
    "cnn_output_indices",
    "transformer_split_a_sc",
    "transformer_split_b_sc",
    "transformer_split_half_r",
    "transformer_sample_sizes",
    "transformer_convergence_mean",
    "transformer_convergence_low",
    "transformer_convergence_high",
)


def validate_results(results: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_ARRAY_KEYS if key not in results]
    if missing:
        raise KeyError(f"Supplementary Figure 1 results are missing: {missing}")
    shape_pairs = (
        ("cnn_weight_sc", "cnn_jacobian_sc"),
        ("transformer_split_a_sc", "transformer_split_b_sc"),
    )
    for first_key, second_key in shape_pairs:
        if np.shape(results[first_key]) != np.shape(results[second_key]):
            raise ValueError(
                f"{first_key} and {second_key} have different shapes: "
                f"{np.shape(results[first_key])} versus {np.shape(results[second_key])}."
            )


def save_results(results: Mapping[str, Any], path: str | Path) -> Path:
    """Save arrays in a non-pickle NPZ plus JSON metadata."""

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
            "savefig.transparent": False,
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _draw_overview(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Validation logic", loc="left", fontweight="bold", pad=2)
    boxes = [
        (0.02, 0.64, 0.25, 0.19, "Explicit pathway\nweights", BLUE),
        (0.38, 0.64, 0.25, 0.19, "Expected local\nJacobian", ORANGE),
        (0.73, 0.64, 0.25, 0.19, "Edge-wise\nagreement", TEAL),
        (0.20, 0.20, 0.25, 0.19, "Independent\ninput splits", GREY),
        (0.56, 0.20, 0.25, 0.19, "Sampling\nconvergence", TEAL),
    ]
    for x, y, width, height, text, color in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=0.8,
                edgecolor=color,
                facecolor=mpl.colors.to_rgba(color, 0.10),
            )
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")
    arrows = [((0.27, 0.735), (0.38, 0.735)), ((0.63, 0.735), (0.73, 0.735)), ((0.45, 0.295), (0.56, 0.295))]
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#444444")
        )
    axis.text(0.50, 0.50, "MLP / CNN", ha="center", va="center", color=GREY)
    axis.text(0.50, 0.08, "Transformer", ha="center", va="center", color=GREY)


def _display_matrix_columns(first: np.ndarray, max_columns: int = 800) -> np.ndarray:
    support = np.any(np.abs(first) > np.finfo(float).eps, axis=0)
    columns = np.flatnonzero(support)
    if columns.size == 0:
        columns = np.arange(first.shape[1])
    if columns.size > max_columns:
        positions = np.linspace(0, columns.size - 1, max_columns).astype(int)
        columns = columns[positions]
    return columns


def _matrix_pair(
    axis: plt.Axes,
    first: np.ndarray,
    second: np.ndarray,
    first_title: str,
    second_title: str,
    title: str,
) -> None:
    axis.set_axis_off()
    axis.set_title(title, loc="left", fontweight="bold", pad=2)
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape[1] > 1000:
        columns = _display_matrix_columns(first)
        first = first[:, columns]
        second = second[:, columns]
    limit = float(np.nanpercentile(np.abs(np.concatenate([first.ravel(), second.ravel()])), 99))
    limit = max(limit, np.finfo(float).eps)
    left = axis.inset_axes([0.00, 0.12, 0.46, 0.75])
    right = axis.inset_axes([0.53, 0.12, 0.46, 0.75])
    left.imshow(first, cmap=CMAP, vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    right.imshow(second, cmap=CMAP, vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    for inner_axis, subtitle in ((left, first_title), (right, second_title)):
        inner_axis.set_title(subtitle, fontsize=6, pad=2)
        inner_axis.set_xticks([])
        inner_axis.set_yticks([])
        for spine in inner_axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_edgecolor("#777777")


def _agreement_plot(
    axis: plt.Axes,
    first: np.ndarray,
    second: np.ndarray,
    title: str,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    first_values, second_values = paired_values(first, second, mask=mask)
    first_z = (first_values - np.mean(first_values)) / max(np.std(first_values), np.finfo(float).eps)
    second_z = (second_values - np.mean(second_values)) / max(np.std(second_values), np.finfo(float).eps)
    correlation = pearson_r(first_values, second_values)
    rank_correlation = spearman_r(first_values, second_values)
    if first_z.size > 4000:
        axis.hexbin(first_z, second_z, gridsize=38, mincnt=1, cmap="Blues", linewidths=0, bins="log")
    else:
        axis.scatter(first_z, second_z, s=4, alpha=0.25, color=BLUE, edgecolors="none")
    slope, intercept = np.polyfit(first_z, second_z, 1)
    x_line = np.linspace(np.nanpercentile(first_z, 1), np.nanpercentile(first_z, 99), 100)
    axis.plot(x_line, slope * x_line + intercept, color=RED, lw=1.1)
    axis.axhline(0, color=LIGHT_GREY, lw=0.5, zorder=0)
    axis.axvline(0, color=LIGHT_GREY, lw=0.5, zorder=0)
    axis.set_title(title, loc="left", fontweight="bold", pad=2)
    axis.set_xlabel("Weight-based SC (z score)")
    axis.set_ylabel("Jacobian SC (z score)")
    axis.text(
        0.04,
        0.96,
        f"Pearson r = {correlation:.2f}\nSpearman ρ = {rank_correlation:.2f}\nn = {first_z.size:,} edges",
        transform=axis.transAxes,
        va="top",
        ha="left",
    )
    return {"pearson_r": correlation, "spearman_r": rank_correlation, "n_edges": int(first_z.size)}


def plot_fig1_jacobian_supp(
    results: Mapping[str, Any],
    output_prefix: str | Path | None = None,
    export_formats: Sequence[str] = ("svg", "pdf", "tiff", "png"),
    dpi: int = 600,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Create the CNN/Transformer Jacobian panels retained as Fig. S1f-j."""

    validate_results(results)
    publication_style()
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=False)
    grid = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.72)
    ax_f = fig.add_subplot(grid[0, :2])
    ax_g = fig.add_subplot(grid[0, 2:])
    ax_h = fig.add_subplot(grid[1, :2])
    ax_i = fig.add_subplot(grid[1, 2])
    ax_j = fig.add_subplot(grid[1, 3])

    _matrix_pair(
        ax_f,
        results["cnn_weight_sc"],
        results["cnn_jacobian_sc"],
        "Kernel-average SC",
        "Expected Jacobian",
        "CNN structural connectivity (sampled outputs)",
    )
    cnn_mask = np.abs(np.asarray(results["cnn_weight_sc"])) > np.finfo(float).eps
    cnn_stats = _agreement_plot(
        ax_g,
        results["cnn_weight_sc"],
        results["cnn_jacobian_sc"],
        "CNN admissible-edge agreement",
        mask=cnn_mask,
    )

    _matrix_pair(
        ax_h,
        results["transformer_split_a_sc"],
        results["transformer_split_b_sc"],
        "Input split A",
        "Input split B",
        "Transformer Jacobian SC reproducibility",
    )
    transformer_stats = _agreement_plot(
        ax_i,
        results["transformer_split_a_sc"],
        results["transformer_split_b_sc"],
        "Split-half agreement",
    )
    ax_i.set_xlabel("Split A SC (z score)")
    ax_i.set_ylabel("Split B SC (z score)")

    sizes = np.asarray(results["transformer_sample_sizes"], dtype=float)
    mean = np.asarray(results["transformer_convergence_mean"], dtype=float)
    low = np.asarray(results["transformer_convergence_low"], dtype=float)
    high = np.asarray(results["transformer_convergence_high"], dtype=float)
    ax_j.fill_between(sizes, low, high, color=TEAL, alpha=0.20, linewidth=0)
    ax_j.plot(sizes, mean, color=TEAL, marker="o", ms=3, lw=1.3)
    ax_j.set_xscale("log", base=2)
    ax_j.set_xticks(sizes)
    ax_j.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_j.set_ylim(max(-0.05, float(np.nanmin(low)) - 0.05), 1.01)
    ax_j.set_xlabel("Samples used for SC")
    ax_j.set_ylabel("Correlation with independent split")
    ax_j.set_title("Transformer sampling convergence", loc="left", fontweight="bold", pad=2)
    ax_j.text(0.04, 0.08, "Mean and 95% interval\nacross repeated subsets", transform=ax_j.transAxes)

    for label, axis in zip("fghij", (ax_f, ax_g, ax_h, ax_i, ax_j)):
        _panel_label(axis, label)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.98, bottom=0.07)
    statistics = {
        "cnn": cnn_stats,
        "transformer": transformer_stats,
        "transformer_split_half_r": float(np.ravel(results["transformer_split_half_r"])[0]),
    }

    if output_prefix is not None:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for extension in export_formats:
            extension = extension.lower().lstrip(".")
            save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
            if extension in {"png", "tif", "tiff"}:
                save_kwargs["dpi"] = dpi
            fig.savefig(output_prefix.with_suffix(f".{extension}"), **save_kwargs)
        output_prefix.with_name(output_prefix.name + "_statistics.json").write_text(
            json.dumps(statistics, indent=2), encoding="utf-8"
        )
    return fig, statistics
