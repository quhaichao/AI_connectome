"""Robustness and reproducibility analyses for Supplementary Figure 6.

The analysis follows the MLP definitions used in the main Figure 3 workflow:
functional connectivity (FC) is the Pearson correlation between hidden-unit
activations on one fixed, class-balanced input set, and input similarity (IS)
is the Pearson correlation between rows of the incoming weight matrix.

Independent training seeds are the inferential replicates. Unit pairs are
summarized within each run and are never treated as independent experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn

from .models import MLP


FC_COLOR = "#2A9D8F"
IS_COLOR = "#E9A03B"
ACCENT_COLOR = "#6F5AA8"
NEUTRAL_COLOR = "#737B85"
LIGHT_GREY = "#E6E8EB"
DARK_GREY = "#30343B"


@dataclass(frozen=True)
class SuppFig6Config:
    """Analysis defaults aligned with the main MLP experiment."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    input_size: int = 784
    hidden_dims: tuple[int, ...] = (100, 100)
    num_classes: int = 10
    activation: str = "relu"
    init_method: str = "normal"
    dropout_p: float = 0.0
    epochs: int = 5
    learning_rate: float = 0.05
    analysis_interval: int = 10
    inference_batch_size: int = 256
    samples_per_class: int = 100
    layer_indices: tuple[int, ...] = (1,)
    primary_edge_threshold: float = 0.70
    primary_onset_fraction: float = 0.25
    primary_smoothing_sigma: float = 1.0
    min_consecutive_points: int = 2
    pair_min_dynamic_range: float = 0.10
    pair_min_is_growth: float = 0.02
    threshold_values: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
    onset_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    smoothing_sigmas: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)
    sampling_strides: tuple[int, ...] = (1, 2, 4)
    max_xcorr_lag_points: int = 12
    bootstrap_repeats: int = 2000
    null_repeats: int = 10000
    analysis_seed: int = 2026


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds and deterministic CuDNN flags."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def collect_balanced_reference_inputs(
    dataset: Any,
    samples_per_class: int = 100,
    num_classes: int = 10,
) -> tuple[torch.Tensor, np.ndarray]:
    """Collect a deterministic class-balanced FC evaluation set.

    Dataset order, rather than a shuffled loader, defines the reference set so
    every seed is evaluated on exactly the same inputs.
    """

    class_samples: dict[int, list[torch.Tensor]] = {
        class_id: [] for class_id in range(num_classes)
    }
    for index in range(len(dataset)):
        sample, target = dataset[index]
        class_id = int(target)
        if class_id in class_samples and len(class_samples[class_id]) < samples_per_class:
            class_samples[class_id].append(sample.detach().cpu().clone())
        if all(len(values) >= samples_per_class for values in class_samples.values()):
            break

    missing = {
        class_id: samples_per_class - len(values)
        for class_id, values in class_samples.items()
        if len(values) < samples_per_class
    }
    if missing:
        raise ValueError(f"Insufficient samples for classes: {missing}")

    inputs = torch.stack(
        [
            sample
            for class_id in range(num_classes)
            for sample in class_samples[class_id][:samples_per_class]
        ]
    )
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    return inputs, labels


def _safe_correlation_matrix(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2:
        raise ValueError("observations must be a samples-by-units matrix")
    matrix = np.corrcoef(observations, rowvar=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _row_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    result = np.corrcoef(matrix)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(result, 1.0)
    return result


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def get_hidden_layer_fc(
    model: MLP,
    fixed_inputs: torch.Tensor,
    layer_index: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute one hidden layer's FC on a fixed input tensor."""

    model.eval()
    device = next(model.parameters()).device
    activation_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fixed_inputs), batch_size):
            batch = fixed_inputs[start : start + batch_size].to(device)
            activations = model.get_hidden_activations(batch)[layer_index]
            activation_batches.append(activations.detach().cpu().numpy())
    activation_matrix = np.concatenate(activation_batches, axis=0)
    return _safe_correlation_matrix(activation_matrix)


def get_hidden_layer_is(model: MLP, layer_index: int) -> np.ndarray:
    """Compute IS from correlations between incoming weight profiles."""

    weights = model.linear_layers[layer_index].weight.detach().cpu().numpy()
    return _row_correlation(weights)


def measure_fc_is_layers(
    model: MLP,
    fixed_inputs: torch.Tensor,
    layer_indices: Sequence[int],
    batch_size: int = 256,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return vectorized upper triangles for all requested hidden layers."""

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer_index in layer_indices:
        if layer_index < 0 or layer_index >= len(model.hidden_dims):
            raise IndexError(f"Invalid hidden layer index: {layer_index}")
        fc = get_hidden_layer_fc(model, fixed_inputs, layer_index, batch_size)
        input_similarity = get_hidden_layer_is(model, layer_index)
        result[int(layer_index)] = (
            _upper_values(fc).astype(np.float32),
            _upper_values(input_similarity).astype(np.float32),
        )
    return result


def evaluate_accuracy(
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


def train_mlp_fc_is_history(
    seed: int,
    train_loader: Iterable[Any],
    fixed_inputs: torch.Tensor,
    config: SuppFig6Config,
    eval_loader: Iterable[Any] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train one independent MLP and measure FC/IS online.

    Only the compact FC/IS histories are retained. Saving every batch-level
    state dictionary is unnecessary for this supplementary robustness test.
    """

    if config.analysis_interval < 1:
        raise ValueError("analysis_interval must be at least 1")
    set_seed(int(seed))
    device = resolve_device(device)
    model = MLP(
        input_size=config.input_size,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
        dropout_p=config.dropout_p,
        activation=config.activation,
        init_weights=True,
        init_method=config.init_method,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    sampled_steps: list[int] = []
    layer_fc: dict[int, list[np.ndarray]] = {
        index: [] for index in config.layer_indices
    }
    layer_is: dict[int, list[np.ndarray]] = {
        index: [] for index in config.layer_indices
    }
    loss_steps: list[int] = []
    losses: list[float] = []

    def record(step: int) -> None:
        measured = measure_fc_is_layers(
            model,
            fixed_inputs,
            config.layer_indices,
            config.inference_batch_size,
        )
        sampled_steps.append(int(step))
        for layer_index, (fc_values, is_values) in measured.items():
            layer_fc[layer_index].append(fc_values)
            layer_is[layer_index].append(is_values)

    record(0)
    global_step = 0
    for _ in range(config.epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_steps.append(global_step)
            losses.append(float(loss.detach().cpu().item()))
            if global_step % config.analysis_interval == 0:
                record(global_step)
                model.train()
    if sampled_steps[-1] != global_step:
        record(global_step)

    accuracy = (
        evaluate_accuracy(model, eval_loader, device)
        if eval_loader is not None
        else float("nan")
    )
    return {
        "seed": int(seed),
        "steps": np.asarray(sampled_steps, dtype=int),
        "loss_steps": np.asarray(loss_steps, dtype=int),
        "losses": np.asarray(losses, dtype=np.float32),
        "final_accuracy": float(accuracy),
        "layer_indices": np.asarray(config.layer_indices, dtype=int),
        "fc": {
            index: np.stack(values).astype(np.float32)
            for index, values in layer_fc.items()
        },
        "is": {
            index: np.stack(values).astype(np.float32)
            for index, values in layer_is.items()
        },
        "metadata": {
            "fc_definition": "Pearson correlation of hidden activations",
            "is_definition": "Pearson correlation of incoming weight rows",
            "fixed_fc_inputs": True,
            "config": asdict(config),
        },
    }


def save_run_history(history: Mapping[str, Any], path: str | Path) -> Path:
    """Save one run without object arrays or pickle."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layer_indices = np.asarray(history["layer_indices"], dtype=int)
    payload: dict[str, Any] = {
        "seed": np.asarray(int(history["seed"])),
        "steps": np.asarray(history["steps"], dtype=int),
        "loss_steps": np.asarray(history["loss_steps"], dtype=int),
        "losses": np.asarray(history["losses"], dtype=np.float32),
        "final_accuracy": np.asarray(float(history["final_accuracy"])),
        "layer_indices": layer_indices,
        "metadata_json": np.asarray(json.dumps(history.get("metadata", {}))),
    }
    for layer_index in layer_indices:
        payload[f"fc_layer_{int(layer_index)}"] = np.asarray(
            history["fc"][int(layer_index)], dtype=np.float32
        )
        payload[f"is_layer_{int(layer_index)}"] = np.asarray(
            history["is"][int(layer_index)], dtype=np.float32
        )
    np.savez_compressed(path, **payload)
    return path


def load_run_history(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        layer_indices = np.asarray(data["layer_indices"], dtype=int)
        metadata = json.loads(str(data["metadata_json"].item()))
        return {
            "seed": int(data["seed"].item()),
            "steps": np.asarray(data["steps"], dtype=int),
            "loss_steps": np.asarray(data["loss_steps"], dtype=int),
            "losses": np.asarray(data["losses"], dtype=np.float32),
            "final_accuracy": float(data["final_accuracy"].item()),
            "layer_indices": layer_indices,
            "fc": {
                int(index): np.asarray(data[f"fc_layer_{int(index)}"], dtype=np.float32)
                for index in layer_indices
            },
            "is": {
                int(index): np.asarray(data[f"is_layer_{int(index)}"], dtype=np.float32)
                for index in layer_indices
            },
            "metadata": metadata,
        }


def validate_histories(
    histories: Sequence[Mapping[str, Any]],
    config: SuppFig6Config,
) -> None:
    if not histories:
        raise ValueError("At least one run history is required")
    reference_steps = np.asarray(histories[0]["steps"])
    expected_layers = tuple(int(value) for value in config.layer_indices)
    seen_seeds: set[int] = set()
    for history in histories:
        seed = int(history["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed: {seed}")
        seen_seeds.add(seed)
        if not np.array_equal(reference_steps, np.asarray(history["steps"])):
            raise ValueError("All histories must use identical sampled training steps")
        actual_layers = tuple(int(value) for value in history["layer_indices"])
        if actual_layers != expected_layers:
            raise ValueError(
                f"Layer mismatch for seed {seed}: {actual_layers} != {expected_layers}"
            )
        for layer_index in expected_layers:
            fc = np.asarray(history["fc"][layer_index])
            input_similarity = np.asarray(history["is"][layer_index])
            if fc.shape != input_similarity.shape or fc.shape[0] != len(reference_steps):
                raise ValueError(f"Invalid history shape for seed {seed}, layer {layer_index}")


def high_edge_proportion(history: np.ndarray, threshold: float) -> np.ndarray:
    history = np.asarray(history, dtype=float)
    return np.nanmean(history > threshold, axis=1)


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if sigma <= 0:
        return values.copy()
    return gaussian_filter1d(values, sigma=float(sigma), axis=-1, mode="nearest")


def _trajectory_endpoints(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return float("nan"), float("nan")
    window = max(2, min(5, values.size // 5))
    baseline = float(np.nanmedian(values[:window]))
    plateau = float(np.nanmedian(values[-window:]))
    return baseline, plateau


def fractional_onset_time(
    steps: np.ndarray,
    values: np.ndarray,
    fraction: float = 0.25,
    smoothing_sigma: float = 1.0,
    min_consecutive: int = 2,
) -> float:
    """First sustained crossing of a fraction of the observed net rise."""

    steps = np.asarray(steps, dtype=float)
    values = _smooth(np.asarray(values, dtype=float), smoothing_sigma)
    if steps.size != values.size or steps.size < 3:
        return float("nan")
    baseline, plateau = _trajectory_endpoints(values)
    rise = plateau - baseline
    if not np.isfinite(rise) or rise <= np.finfo(float).eps:
        return float("nan")
    target = baseline + float(fraction) * rise
    above = np.isfinite(values) & (values >= target)
    width = max(1, int(min_consecutive))
    for index in range(0, len(above) - width + 1):
        if np.all(above[index : index + width]):
            return float(steps[index])
    return float("nan")


def max_slope_time(
    steps: np.ndarray,
    values: np.ndarray,
    smoothing_sigma: float = 1.0,
) -> float:
    steps = np.asarray(steps, dtype=float)
    values = _smooth(np.asarray(values, dtype=float), smoothing_sigma)
    if steps.size != values.size or steps.size < 3:
        return float("nan")
    gradient = np.gradient(values, steps)
    if not np.any(np.isfinite(gradient)):
        return float("nan")
    return float(steps[int(np.nanargmax(gradient))])


def pair_transition_delays(
    steps: np.ndarray,
    fc_history: np.ndarray,
    is_history: np.ndarray,
    smoothing_sigma: float = 1.0,
    min_dynamic_range: float = 0.10,
    min_is_growth: float = 0.02,
) -> np.ndarray:
    """Return pair-level max-slope delays; positive values mean FC leads IS."""

    steps = np.asarray(steps, dtype=float)
    fc = np.asarray(fc_history, dtype=float).T
    input_similarity = np.asarray(is_history, dtype=float).T
    if fc.shape != input_similarity.shape or fc.shape[1] != steps.size:
        raise ValueError("Pair histories must have shape time-by-pairs")

    fc_smoothed = _smooth(fc, smoothing_sigma)
    is_smoothed = _smooth(input_similarity, smoothing_sigma)
    fc_range = np.nanmax(fc_smoothed, axis=1) - np.nanmin(fc_smoothed, axis=1)
    is_range = np.nanmax(is_smoothed, axis=1) - np.nanmin(is_smoothed, axis=1)
    endpoint_window = max(2, min(5, steps.size // 5))
    is_growth = (
        np.nanmedian(is_smoothed[:, -endpoint_window:], axis=1)
        - np.nanmedian(is_smoothed[:, :endpoint_window], axis=1)
    )
    valid = (
        np.isfinite(fc_range)
        & np.isfinite(is_range)
        & (fc_range >= min_dynamic_range)
        & (is_range >= min_dynamic_range)
        & (is_growth >= min_is_growth)
    )
    if not np.any(valid):
        return np.asarray([], dtype=float)
    fc_gradient = np.gradient(fc_smoothed[valid], steps, axis=1)
    is_gradient = np.gradient(is_smoothed[valid], steps, axis=1)
    fc_indices = np.nanargmax(fc_gradient, axis=1)
    is_indices = np.nanargmax(is_gradient, axis=1)
    return steps[is_indices] - steps[fc_indices]


def derivative_xcorr_delay(
    steps: np.ndarray,
    fc_trace: np.ndarray,
    is_trace: np.ndarray,
    smoothing_sigma: float = 1.0,
    max_lag_points: int = 12,
) -> float:
    """Lag maximizing derivative correlation; positive means FC leads IS."""

    steps = np.asarray(steps, dtype=float)
    fc = _smooth(np.asarray(fc_trace, dtype=float), smoothing_sigma)
    input_similarity = _smooth(np.asarray(is_trace, dtype=float), smoothing_sigma)
    if steps.size < 5 or fc.size != steps.size or input_similarity.size != steps.size:
        return float("nan")
    fc_gradient = np.gradient(fc, steps)
    is_gradient = np.gradient(input_similarity, steps)
    max_lag = min(int(max_lag_points), steps.size - 3)
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    for lag in lags:
        if lag > 0:
            x, y = fc_gradient[:-lag], is_gradient[lag:]
        elif lag < 0:
            x, y = fc_gradient[-lag:], is_gradient[:lag]
        else:
            x, y = fc_gradient, is_gradient
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
            correlations.append(np.nan)
        else:
            correlations.append(float(np.corrcoef(x[valid], y[valid])[0, 1]))
    correlations = np.asarray(correlations)
    if not np.any(np.isfinite(correlations)):
        return float("nan")
    best_lag = int(lags[int(np.nanargmax(correlations))])
    step_size = float(np.nanmedian(np.diff(steps)))
    return best_lag * step_size


def _fractional_delay(
    steps: np.ndarray,
    fc_trace: np.ndarray,
    is_trace: np.ndarray,
    fraction: float,
    smoothing_sigma: float,
    min_consecutive: int,
) -> tuple[float, float, float]:
    fc_onset = fractional_onset_time(
        steps, fc_trace, fraction, smoothing_sigma, min_consecutive
    )
    is_onset = fractional_onset_time(
        steps, is_trace, fraction, smoothing_sigma, min_consecutive
    )
    delay = is_onset - fc_onset if np.isfinite(fc_onset + is_onset) else float("nan")
    return fc_onset, is_onset, delay


def exact_sign_flip_null(
    values: np.ndarray,
    repeats: int = 10000,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """One-sided paired-label permutation test on independent run summaries."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=float), float("nan"), float("nan")
    observed = float(np.mean(values))
    if values.size <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=values.size)))
        null = np.mean(signs * values[None, :], axis=1)
        p_value = float(np.mean(null >= observed - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(int(repeats), values.size))
        null = np.mean(signs * values[None, :], axis=1)
        p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return np.asarray(null, dtype=float), observed, p_value


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap independent runs along axis 0."""

    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("values must contain at least one run")
    center = np.nanmean(values, axis=0)
    if values.shape[0] == 1:
        return center, center.copy(), center.copy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(int(repeats), values.shape[0]))
    samples = np.nanmean(values[indices], axis=1)
    lower, upper = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return center, lower, upper


def analyze_sequence_histories(
    histories: Sequence[Mapping[str, Any]],
    config: SuppFig6Config,
) -> dict[str, Any]:
    """Compute only the onset, pair-transition and estimator panels in Fig. S3e-g."""

    validate_histories(histories, config)
    histories = sorted(histories, key=lambda value: int(value["seed"]))
    seeds = np.asarray([int(value["seed"]) for value in histories], dtype=int)
    steps = np.asarray(histories[0]["steps"], dtype=float)
    layers = np.asarray(config.layer_indices, dtype=int)
    n_runs, n_layers = len(histories), len(layers)
    fc_onset = np.full((n_runs, n_layers), np.nan)
    is_onset = np.full_like(fc_onset, np.nan)
    primary_delay = np.full_like(fc_onset, np.nan)
    pair_median_delay = np.full_like(fc_onset, np.nan)
    pair_positive_fraction = np.full_like(fc_onset, np.nan)
    pair_count = np.zeros((n_runs, n_layers), dtype=int)

    estimator_labels = np.asarray(
        ["Fractional onset", "Maximum slope", "Pair transition", "Derivative TLCC"]
    )
    estimator_delay = np.full((n_runs, n_layers, len(estimator_labels)), np.nan)

    for run_index, history in enumerate(histories):
        for layer_position, layer_index in enumerate(layers):
            fc_history = np.asarray(history["fc"][int(layer_index)], dtype=float)
            is_history = np.asarray(history["is"][int(layer_index)], dtype=float)
            primary_fc_trace = high_edge_proportion(
                fc_history, config.primary_edge_threshold
            )
            primary_is_trace = high_edge_proportion(
                is_history, config.primary_edge_threshold
            )
            onset_values = _fractional_delay(
                steps,
                primary_fc_trace,
                primary_is_trace,
                config.primary_onset_fraction,
                config.primary_smoothing_sigma,
                config.min_consecutive_points,
            )
            fc_onset[run_index, layer_position] = onset_values[0]
            is_onset[run_index, layer_position] = onset_values[1]
            primary_delay[run_index, layer_position] = onset_values[2]

            pair_delays = pair_transition_delays(
                steps,
                fc_history,
                is_history,
                config.primary_smoothing_sigma,
                config.pair_min_dynamic_range,
                config.pair_min_is_growth,
            )
            pair_count[run_index, layer_position] = len(pair_delays)
            if pair_delays.size:
                pair_median_delay[run_index, layer_position] = float(
                    np.nanmedian(pair_delays)
                )
                pair_positive_fraction[run_index, layer_position] = float(
                    np.nanmean(pair_delays > 0)
                )

            fc_slope = max_slope_time(
                steps, primary_fc_trace, config.primary_smoothing_sigma
            )
            is_slope = max_slope_time(
                steps, primary_is_trace, config.primary_smoothing_sigma
            )
            slope_delay = (
                is_slope - fc_slope
                if np.isfinite(fc_slope + is_slope)
                else float("nan")
            )
            estimator_delay[run_index, layer_position] = (
                primary_delay[run_index, layer_position],
                slope_delay,
                pair_median_delay[run_index, layer_position],
                derivative_xcorr_delay(
                    steps,
                    primary_fc_trace,
                    primary_is_trace,
                    config.primary_smoothing_sigma,
                    config.max_xcorr_lag_points,
                ),
            )

    seed_primary_delay = np.nanmean(primary_delay, axis=1)
    return {
        "seeds": seeds,
        "steps": steps,
        "layer_indices": layers,
        "layer_labels": np.asarray([f"Hidden layer {index + 1}" for index in layers]),
        "final_accuracy": np.asarray(
            [float(value["final_accuracy"]) for value in histories], dtype=float
        ),
        "fc_onset": fc_onset,
        "is_onset": is_onset,
        "primary_delay": primary_delay,
        "pair_median_delay": pair_median_delay,
        "pair_positive_fraction": pair_positive_fraction,
        "pair_count": pair_count,
        "estimator_labels": estimator_labels,
        "estimator_delay": estimator_delay,
        "primary_edge_threshold": np.asarray(config.primary_edge_threshold),
        "primary_onset_fraction": np.asarray(config.primary_onset_fraction),
        "primary_smoothing_sigma": np.asarray(config.primary_smoothing_sigma),
        "analysis_interval": np.asarray(config.analysis_interval),
        "seed_primary_delay": seed_primary_delay,
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "config": asdict(config),
                    "inferential_unit": "independent training seed",
                    "pair_level_role": "within-run descriptive summary",
                    "positive_delay_definition": "IS transition time minus FC transition time",
                }
            )
        ),
    }


def save_results_npz(results: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: np.asarray(value)
        for key, value in results.items()
        if not isinstance(value, (dict, list, tuple))
    }
    np.savez_compressed(path, **payload)
    return path


def load_results_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _heatmap_limits(matrix: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(matrix, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = max(float(np.nanpercentile(np.abs(finite), 95)), 1.0)
    return -limit, limit


def _draw_delay_heatmap(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
    color_limits: tuple[float, float] | None = None,
) -> mpl.image.AxesImage:
    vmin, vmax = color_limits if color_limits is not None else _heatmap_limits(matrix)
    image = ax.imshow(matrix, cmap="PuOr_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "NA" if not np.isfinite(value) else f"{value:.0f}"
            color = "white" if np.isfinite(value) and abs(value) > 0.58 * vmax else DARK_GREY
            ax.text(column, row, text, ha="center", va="center", fontsize=5.5, color=color)
    return image


def plot_fig3_fc_before_is_supp(
    results: Mapping[str, Any],
    output_dir: str | Path,
    basename: str = "fig3_FC_before_IS_supp",
    export_formats: Sequence[str] = ("svg", "pdf", "tiff", "png"),
    dpi: int = 600,
) -> tuple[mpl.figure.Figure, dict[str, Any]]:
    """Create the onset, pair-transition and estimator panels in Fig. S3e-g."""

    set_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    ax_e, ax_f, ax_g = axes

    seed_fc_onset = np.nanmean(np.asarray(results["fc_onset"], dtype=float), axis=1)
    seed_is_onset = np.nanmean(np.asarray(results["is_onset"], dtype=float), axis=1)
    for fc_value, is_value in zip(seed_fc_onset, seed_is_onset):
        if np.isfinite(fc_value + is_value):
            ax_e.plot((0, 1), (fc_value, is_value), color=LIGHT_GREY, lw=0.9, zorder=1)
    ax_e.scatter(np.zeros_like(seed_fc_onset), seed_fc_onset, color=FC_COLOR, s=20, zorder=2)
    ax_e.scatter(np.ones_like(seed_is_onset), seed_is_onset, color=IS_COLOR, s=20, zorder=2)
    ax_e.set_xticks((0, 1), ("FC", "IS"))
    ax_e.set_xlim(-0.45, 1.45)
    ax_e.set_ylabel("Fractional-onset step")
    ax_e.set_title("Paired onset by seed", loc="left")

    pair_delays = np.nanmean(np.asarray(results["pair_median_delay"], dtype=float), axis=1)
    x_jitter = np.linspace(-0.12, 0.12, len(pair_delays))
    ax_f.axhline(0, color=DARK_GREY, lw=0.8, ls="--")
    ax_f.scatter(x_jitter, pair_delays, color=ACCENT_COLOR, s=24, zorder=3)
    if np.any(np.isfinite(pair_delays)):
        median_value = float(np.nanmedian(pair_delays))
        ax_f.plot((-0.22, 0.22), (median_value, median_value), color=DARK_GREY, lw=1.8)
    ax_f.set_xlim(-0.35, 0.35)
    ax_f.set_xticks((0,), ("Independent seeds",))
    ax_f.set_ylabel("Median pair delay (steps)\nIS transition - FC transition")
    ax_f.set_title("Pair-transition reproducibility", loc="left")

    estimator_values = np.nanmean(np.asarray(results["estimator_delay"], dtype=float), axis=1)
    labels = np.asarray(results["estimator_labels"]).astype(str)
    for estimator_index in range(estimator_values.shape[1]):
        values = estimator_values[:, estimator_index]
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax_g.scatter(estimator_index + jitter, values, color=ACCENT_COLOR,
                     edgecolor="white", linewidth=0.3, s=20, zorder=3)
        if np.any(np.isfinite(values)):
            mean_value = float(np.nanmean(values))
            ax_g.plot((estimator_index - 0.22, estimator_index + 0.22),
                      (mean_value, mean_value), color=DARK_GREY, lw=1.7)
    ax_g.axhline(0, color=DARK_GREY, lw=0.8, ls="--")
    ax_g.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    ax_g.set_ylabel("Run-level delay (steps)")
    ax_g.set_title("Alternative transition estimators", loc="left")

    for label, axis in zip("efg", axes):
        _panel_label(axis, label)
        axis.tick_params(length=2.5, width=0.7)
    fig.tight_layout()

    statistics = {
        "inferential_unit": "independent training seed",
        "n_seeds": int(len(np.asarray(results["seeds"]))),
        "seed_primary_delay": np.asarray(results["seed_primary_delay"], dtype=float).tolist(),
        "pair_median_delay_by_seed": pair_delays.tolist(),
        "pair_counts": np.asarray(results["pair_count"], dtype=int).tolist(),
        "estimator_labels": labels.tolist(),
        "estimator_delay_by_seed": estimator_values.tolist(),
        "final_accuracy": np.asarray(results["final_accuracy"], dtype=float).tolist(),
    }
    for extension in export_formats:
        extension = extension.lower().lstrip(".")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension in {"png", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(output_dir / f"{basename}.{extension}", **kwargs)
    (output_dir / f"{basename}_statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return fig, statistics


__all__ = [
    "SuppFig6Config",
    "analyze_sequence_histories",
    "collect_balanced_reference_inputs",
    "derivative_xcorr_delay",
    "exact_sign_flip_null",
    "fractional_onset_time",
    "load_results_npz",
    "load_run_history",
    "max_slope_time",
    "pair_transition_delays",
    "plot_fig3_fc_before_is_supp",
    "save_results_npz",
    "save_run_history",
    "set_seed",
    "train_mlp_fc_is_history",
    "validate_histories",
]
