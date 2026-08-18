"""Robustness and mechanistic controls for main Figure 5a--f.

This module reproduces the MLP definitions used by the main Figure 5 notebook
and adds independent-seed inference, held-out FC selection, definition
sensitivity, matched controls, acute/persistent masking, representation
quantification and publication-quality plotting.

The independent statistical unit is a complete training seed. Neurons and
neuron pairs are summarized within a seed and are never used as independent
replicates for group-level inference.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon
import torch
import torch.nn as nn

from .models import MLP


HIGH_COLOR = "#D77A61"
MID_COLOR = "#D5A94E"
LOW_COLOR = "#4C78A8"
RANDOM_COLOR = "#8A9099"
MATCHED_COLOR = "#6F5AA8"
UPPER_COLOR = "#2A9D8F"
DARK = "#2D333A"
LIGHT = "#E6E8EB"


@dataclass(frozen=True)
class Fig5SupplementConfig:
    """Defaults aligned with ``MLP_FC_IS_explain.ipynb``."""

    seeds: tuple[int, ...] = tuple(range(15))
    input_size: int = 784
    hidden_dims: tuple[int, ...] = (100, 100)
    num_classes: int = 10
    activation: str = "relu"
    init_method: str = "normal"
    dropout_p: float = 0.0
    learning_rate: float = 0.05
    max_steps: int = 300
    selection_step: int = 180
    snapshot_interval: int = 10
    inference_batch_size: int = 256
    selection_fraction: float = 0.20
    fc_top_k: int = 5
    masking_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40)
    random_mask_repeats: int = 20
    cumulative_grid_size: int = 21
    random_rank_repeats: int = 200
    continuation_steps: int = 120
    continuation_eval_interval: int = 10
    phase_window: int = 60
    analysis_seed: int = 20260718

    def validate(self) -> None:
        if len(self.seeds) < 15:
            raise ValueError("At least 15 independent seeds are required.")
        if self.activation.lower() != "relu":
            raise ValueError("Figure 5 analysis currently assumes ReLU activations.")
        if not 0 < self.selection_fraction < 0.5:
            raise ValueError("selection_fraction must lie between 0 and 0.5.")
        if self.selection_step > self.max_steps:
            raise ValueError("selection_step cannot exceed max_steps.")


def set_seed(seed: int) -> None:
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


def collect_balanced_disjoint_sets(
    dataset: Any,
    samples_per_class: int = 50,
    num_classes: int = 10,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Build deterministic, non-overlapping selection and outcome sets.

    Set A is used only to define FC and neuron groups. Set B is used for
    gradients, error signals, representations and intervention outcomes.
    """

    required = 2 * samples_per_class
    buckets: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_classes)}
    for index in range(len(dataset)):
        sample, target = dataset[index]
        target = int(target)
        if target in buckets and len(buckets[target]) < required:
            buckets[target].append(sample.detach().cpu().clone())
        if all(len(values) >= required for values in buckets.values()):
            break
    missing = {k: required - len(v) for k, v in buckets.items() if len(v) < required}
    if missing:
        raise ValueError(f"Insufficient class-balanced samples: {missing}")

    set_a = torch.stack(
        [x for label in range(num_classes) for x in buckets[label][:samples_per_class]]
    )
    set_b = torch.stack(
        [x for label in range(num_classes) for x in buckets[label][samples_per_class:required]]
    )
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    return set_a, labels.copy(), set_b, labels.copy()


def build_mlp(config: Fig5SupplementConfig, device: torch.device) -> MLP:
    model = MLP(
        input_size=config.input_size,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
        dropout_p=config.dropout_p,
        activation=config.activation,
        init_weights=True,
        init_method=config.init_method,
    )
    return model.to(device)


def _forward_h1(model: MLP, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs.view(inputs.size(0), -1)
    h1 = torch.relu(model.linear_layers[0](x))
    h2 = torch.relu(model.linear_layers[1](h1))
    logits = model.linear_layers[2](h2)
    return logits, h1


def _forward_h1_masked(
    model: MLP,
    inputs: torch.Tensor,
    masked_units: np.ndarray | Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs.view(inputs.size(0), -1)
    h1 = torch.relu(model.linear_layers[0](x))
    if masked_units is not None and len(masked_units) > 0:
        keep = torch.ones(h1.shape[1], device=h1.device, dtype=h1.dtype)
        keep[torch.as_tensor(masked_units, device=h1.device, dtype=torch.long)] = 0.0
        h1 = h1 * keep.unsqueeze(0)
    h2 = torch.relu(model.linear_layers[1](h1))
    return model.linear_layers[2](h2), h1


def _safe_corrcoef(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=float)
    matrix = np.corrcoef(observations, rowvar=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def hidden_activations(
    model: MLP,
    inputs: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            _, h1 = _forward_h1(model, inputs[start : start + batch_size].to(device))
            chunks.append(h1.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def functional_connectivity(activations: np.ndarray) -> np.ndarray:
    return _safe_corrcoef(activations)


def fc_scores(fc: np.ndarray, top_k: int = 5) -> np.ndarray:
    """Return max, top-k mean and 95th-percentile operational FC scores."""

    fc = np.asarray(fc, dtype=float).copy()
    np.fill_diagonal(fc, -np.inf)
    ordered = np.sort(fc, axis=1)
    k = min(max(1, top_k), fc.shape[1] - 1)
    maximum = ordered[:, -1]
    top_mean = np.mean(ordered[:, -k:], axis=1)
    finite = np.where(np.isfinite(fc), fc, np.nan)
    percentile = np.nanpercentile(finite, 95, axis=1)
    return np.stack([maximum, top_mean, percentile], axis=0)


def define_groups(scores: np.ndarray, fraction: float = 0.20) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    n_group = max(1, int(round(len(scores) * fraction)))
    order = np.argsort(scores)
    middle_start = max(0, (len(scores) - n_group) // 2)
    return {
        "low": order[:n_group],
        "mid": order[middle_start : middle_start + n_group],
        "high": order[-n_group:],
    }


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return np.nan_to_num((values - mean) / std)


def matched_control_units(
    target_units: Sequence[int],
    covariates: np.ndarray,
    excluded_units: Sequence[int] = (),
) -> np.ndarray:
    """Greedy unique nearest-neighbour match on standardized covariates."""

    covariates = _zscore_columns(covariates)
    excluded = set(int(x) for x in target_units) | set(int(x) for x in excluded_units)
    available = [i for i in range(len(covariates)) if i not in excluded]
    selected: list[int] = []
    for unit in target_units:
        if not available:
            raise ValueError("Not enough units for covariate matching.")
        distances = np.linalg.norm(covariates[available] - covariates[int(unit)], axis=1)
        choice_position = int(np.argmin(distances))
        selected.append(available.pop(choice_position))
    return np.asarray(selected, dtype=int)


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x[valid], y[valid]).statistic)
    if method == "spearman":
        return float(spearmanr(x[valid], y[valid]).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def error_signal_strength(
    model: MLP,
    inputs: torch.Tensor,
    labels: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-unit RMS of dL/dh1 over held-out samples."""

    model.eval()
    device = next(model.parameters()).device
    sum_sq = np.zeros(model.hidden_dims[0], dtype=float)
    count = 0
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        target = torch.as_tensor(labels[start : start + batch_size], device=device)
        x = batch.view(batch.size(0), -1)
        h1 = torch.relu(model.linear_layers[0](x))
        h1.retain_grad()
        h2 = torch.relu(model.linear_layers[1](h1))
        logits = model.linear_layers[2](h2)
        loss = nn.functional.cross_entropy(logits, target, reduction="sum")
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients = h1.grad.detach().cpu().numpy()
        sum_sq += np.sum(np.square(gradients), axis=0)
        count += len(batch)
    model.zero_grad(set_to_none=True)
    return np.sqrt(sum_sq / max(count, 1))


def evaluate_model(
    model: MLP,
    inputs: torch.Tensor,
    labels: np.ndarray,
    masked_units: Sequence[int] | np.ndarray | None = None,
    batch_size: int = 256,
) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            target = torch.as_tensor(labels[start : start + batch_size], device=device)
            logits, _ = _forward_h1_masked(model, batch, masked_units)
            total_loss += float(nn.functional.cross_entropy(logits, target, reduction="sum"))
            total_correct += int((logits.argmax(1) == target).sum())
    return total_loss / len(inputs), total_correct / len(inputs)


def _snapshot(
    model: MLP,
    selection_inputs: torch.Tensor,
    outcome_inputs: torch.Tensor,
    outcome_labels: np.ndarray,
    config: Fig5SupplementConfig,
) -> dict[str, np.ndarray]:
    activation_a = hidden_activations(model, selection_inputs, config.inference_batch_size)
    activation_b = hidden_activations(model, outcome_inputs, config.inference_batch_size)
    fc = functional_connectivity(activation_a)
    outgoing = model.linear_layers[1].weight.detach().cpu().numpy()
    return {
        "fc_scores": fc_scores(fc, config.fc_top_k),
        "error": error_signal_strength(
            model, outcome_inputs, outcome_labels, config.inference_batch_size
        ),
        "activation_rate": np.mean(activation_a > 0, axis=0),
        "activation_variance": np.var(activation_a, axis=0),
        "outgoing_norm": np.linalg.norm(outgoing, axis=0),
        "outcome_activation": activation_b,
    }


def _next_batch(iterator: Any, loader: Any) -> tuple[Any, torch.Tensor, torch.Tensor]:
    try:
        inputs, labels = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        inputs, labels = next(iterator)
    return iterator, inputs, labels


def train_baseline(
    train_loader: Any,
    selection_inputs: torch.Tensor,
    outcome_inputs: torch.Tensor,
    outcome_labels: np.ndarray,
    config: Fig5SupplementConfig,
    seed: int,
    device: torch.device,
) -> tuple[MLP, dict[str, Any], dict[str, torch.Tensor]]:
    """Train one baseline run and retain compact analysis histories."""

    set_seed(seed)
    model = build_mlp(config, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    snapshot_steps = np.unique(
        np.concatenate(
            ([0, config.selection_step, config.max_steps],
             np.arange(0, config.max_steps + 1, config.snapshot_interval))
        )
    ).astype(int)
    snapshot_steps = snapshot_steps[snapshot_steps <= config.max_steps]
    snapshot_set = set(snapshot_steps.tolist())

    losses = np.empty(config.max_steps, dtype=float)
    grad_norms = np.empty((config.max_steps, config.hidden_dims[0]), dtype=float)
    snapshots: dict[int, dict[str, np.ndarray]] = {
        0: _snapshot(model, selection_inputs, outcome_inputs, outcome_labels, config)
    }
    selection_state: dict[str, torch.Tensor] | None = None
    iterator = iter(train_loader)
    model.train()
    for step in range(1, config.max_steps + 1):
        iterator, inputs, labels = _next_batch(iterator, train_loader)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        gradient = model.linear_layers[0].weight.grad.detach()
        losses[step - 1] = float(loss.detach().cpu())
        grad_norms[step - 1] = torch.linalg.vector_norm(gradient, dim=1).cpu().numpy()
        optimizer.step()
        if step == config.selection_step:
            selection_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step in snapshot_set:
            snapshots[step] = _snapshot(
                model, selection_inputs, outcome_inputs, outcome_labels, config
            )
            model.train()

    if selection_state is None:
        raise RuntimeError("Selection checkpoint was not captured.")
    history = {
        "losses": losses,
        "grad_norms": grad_norms,
        "snapshot_steps": snapshot_steps,
        "error_history": np.stack([snapshots[s]["error"] for s in snapshot_steps]),
        "selection_snapshot": snapshots[config.selection_step],
        "final_snapshot": snapshots[config.max_steps],
    }
    return model, history, selection_state


def _future_energy(grad_norms: np.ndarray, selection_step: int) -> np.ndarray:
    return np.sum(np.square(grad_norms[selection_step - 1 :]), axis=0)


def _cumulative_by_rank(values: np.ndarray, rank_scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(rank_scores)[::-1]
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    denominator = values.sum()
    if denominator <= 0:
        return np.full_like(grid, np.nan, dtype=float)
    cumulative = np.cumsum(values[order]) / denominator
    x = np.arange(1, len(values) + 1) / len(values)
    return np.interp(grid, x, cumulative, left=0.0, right=1.0)


def cumulative_energy_controls(
    energy: np.ndarray,
    early_fc: np.ndarray,
    grid_size: int,
    random_repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, grid_size)
    observed = _cumulative_by_rank(energy, early_fc, grid)
    upper = _cumulative_by_rank(energy, energy, grid)
    random_curves = np.stack(
        [_cumulative_by_rank(energy, rng.permutation(len(energy)), grid)
         for _ in range(random_repeats)]
    )
    return grid, observed, np.nanmean(random_curves, axis=0), upper


def threshold_enrichment(
    energy: np.ndarray,
    early_fc: np.ndarray,
    fractions: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(early_fc)
    total = np.sum(energy)
    high, low = [], []
    for fraction in fractions:
        n = max(1, int(round(len(order) * fraction)))
        expected = fraction
        high.append(np.sum(energy[order[-n:]]) / total / expected)
        low.append(np.sum(energy[order[:n]]) / total / expected)
    return np.asarray(high), np.asarray(low)


def _eta_squared_by_unit(activation: np.ndarray, labels: np.ndarray) -> np.ndarray:
    grand = np.mean(activation, axis=0)
    between = np.zeros(activation.shape[1], dtype=float)
    for label in np.unique(labels):
        group = activation[labels == label]
        between += len(group) * np.square(np.mean(group, axis=0) - grand)
    total = np.sum(np.square(activation - grand), axis=0)
    return np.divide(between, total, out=np.zeros_like(between), where=total > 1e-12)


def _template_similarity(
    incoming_weights: np.ndarray,
    images: torch.Tensor,
    labels: np.ndarray,
) -> np.ndarray:
    flat = images.detach().cpu().numpy().reshape(len(images), -1)
    templates = np.stack([flat[labels == label].mean(0) for label in np.unique(labels)])
    weights = incoming_weights - incoming_weights.mean(1, keepdims=True)
    templates = templates - templates.mean(1, keepdims=True)
    weights /= np.linalg.norm(weights, axis=1, keepdims=True) + 1e-12
    templates /= np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12
    return np.max(np.abs(weights @ templates.T), axis=1)


def _spatial_autocorrelation(incoming_weights: np.ndarray) -> np.ndarray:
    side = int(round(np.sqrt(incoming_weights.shape[1])))
    if side * side != incoming_weights.shape[1]:
        return np.full(incoming_weights.shape[0], np.nan)
    images = incoming_weights.reshape(-1, side, side)
    output = np.empty(len(images), dtype=float)
    for i, image in enumerate(images):
        horizontal = _correlation(image[:, :-1].ravel(), image[:, 1:].ravel(), "pearson")
        vertical = _correlation(image[:-1, :].ravel(), image[1:, :].ravel(), "pearson")
        output[i] = np.nanmean([horizontal, vertical])
    return output


def representation_metrics(
    model: MLP,
    outcome_inputs: torch.Tensor,
    outcome_labels: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    activation = hidden_activations(model, outcome_inputs, batch_size)
    incoming = model.linear_layers[0].weight.detach().cpu().numpy()
    return np.stack(
        [
            _eta_squared_by_unit(activation, outcome_labels),
            _template_similarity(incoming, outcome_inputs, outcome_labels),
            _spatial_autocorrelation(incoming),
        ],
        axis=0,
    )


def acute_masking_analysis(
    model: MLP,
    early_fc: np.ndarray,
    covariates: np.ndarray,
    outcome_inputs: torch.Tensor,
    outcome_labels: np.ndarray,
    config: Fig5SupplementConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return delta loss and delta accuracy for high/low/random/matched masks."""

    base_loss, base_accuracy = evaluate_model(
        model, outcome_inputs, outcome_labels, None, config.inference_batch_size
    )
    order = np.argsort(early_fc)
    loss_delta = np.empty((4, len(config.masking_fractions)), dtype=float)
    accuracy_delta = np.empty_like(loss_delta)
    for d, fraction in enumerate(config.masking_fractions):
        n = max(1, int(round(len(order) * fraction)))
        high = order[-n:]
        low = order[:n]
        # Matching is performed among all non-high-FC neurons. At large doses
        # a fully disjoint high/low/matched partition is mathematically
        # impossible, so overlap with the descriptive low-FC group is allowed.
        matched = matched_control_units(high, covariates)
        random_values: list[tuple[float, float]] = []
        excluded = np.concatenate([high, low])
        available = np.setdiff1d(np.arange(len(order)), excluded)
        if len(available) < n:
            available = np.setdiff1d(np.arange(len(order)), high)
        for _ in range(config.random_mask_repeats):
            mask = rng.choice(available, size=n, replace=False)
            random_values.append(
                evaluate_model(
                    model, outcome_inputs, outcome_labels, mask,
                    config.inference_batch_size,
                )
            )
        conditions = [
            evaluate_model(model, outcome_inputs, outcome_labels, high, config.inference_batch_size),
            evaluate_model(model, outcome_inputs, outcome_labels, low, config.inference_batch_size),
            tuple(np.mean(random_values, axis=0)),
            evaluate_model(model, outcome_inputs, outcome_labels, matched, config.inference_batch_size),
        ]
        for c, (loss, accuracy) in enumerate(conditions):
            loss_delta[c, d] = loss - base_loss
            accuracy_delta[c, d] = base_accuracy - accuracy
    return loss_delta, accuracy_delta


def persistent_masking_analysis(
    selection_state: Mapping[str, torch.Tensor],
    loader_factory: Callable[[int], Any],
    early_fc: np.ndarray,
    covariates: np.ndarray,
    outcome_inputs: torch.Tensor,
    outcome_labels: np.ndarray,
    config: Fig5SupplementConfig,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resume identical checkpoints/batches with fixed masks from step 180."""

    groups = define_groups(early_fc, config.selection_fraction)
    rng = np.random.default_rng(config.analysis_seed + 1009 * seed)
    available = np.setdiff1d(np.arange(len(early_fc)), np.r_[groups["high"], groups["low"]])
    random_units = rng.choice(available, size=len(groups["high"]), replace=False)
    matched = matched_control_units(
        groups["high"], covariates, excluded_units=np.r_[groups["low"], random_units]
    )
    masks: list[np.ndarray | None] = [None, groups["high"], groups["low"], random_units, matched]
    eval_steps = np.arange(
        0, config.continuation_steps + 1, config.continuation_eval_interval, dtype=int
    )
    loss_history = np.empty((len(masks), len(eval_steps)), dtype=float)
    accuracy_history = np.empty_like(loss_history)
    continuation_seed = config.analysis_seed + 100_000 + seed
    for condition, mask in enumerate(masks):
        set_seed(seed)
        model = build_mlp(config, device)
        model.load_state_dict(selection_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        loader = loader_factory(continuation_seed)
        iterator = iter(loader)
        eval_position = 0
        loss_history[condition, 0], accuracy_history[condition, 0] = evaluate_model(
            model, outcome_inputs, outcome_labels, mask, config.inference_batch_size
        )
        for step in range(1, config.continuation_steps + 1):
            iterator, inputs, labels = _next_batch(iterator, loader)
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = _forward_h1_masked(model, inputs, mask)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            if step in set(eval_steps[1:].tolist()):
                eval_position += 1
                loss_history[condition, eval_position], accuracy_history[condition, eval_position] = evaluate_model(
                    model, outcome_inputs, outcome_labels, mask, config.inference_batch_size
                )
    return eval_steps, loss_history, accuracy_history


def detect_loss_drop_onset(losses: np.ndarray) -> int:
    smoothed = gaussian_filter1d(np.asarray(losses, dtype=float), sigma=3.0)
    search_end = max(10, int(round(len(smoothed) * 0.80)))
    return int(np.argmin(np.gradient(smoothed[:search_end])) + 1)


def _group_mean(values: np.ndarray, groups: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.nanmean(values[:, groups[key]], axis=1) for key in ("high", "mid", "low")])


def run_seed_analysis(
    train_loader_factory: Callable[[int], Any],
    selection_inputs: torch.Tensor,
    outcome_inputs: torch.Tensor,
    outcome_labels: np.ndarray,
    config: Fig5SupplementConfig,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run all Figure 5a--f supplementary analyses for one training seed."""

    config.validate()
    device = resolve_device(device)
    loader = train_loader_factory(seed)
    model, history, selection_state = train_baseline(
        loader, selection_inputs, outcome_inputs, outcome_labels, config, seed, device
    )
    selection = history["selection_snapshot"]
    early_fc = selection["fc_scores"][0]
    groups = define_groups(early_fc, config.selection_fraction)
    covariates = np.stack(
        [selection["activation_rate"], selection["activation_variance"], selection["outgoing_norm"]],
        axis=1,
    )
    rng = np.random.default_rng(config.analysis_seed + seed)
    future_energy = _future_energy(history["grad_norms"], config.selection_step)

    group_gradient = _group_mean(history["grad_norms"], groups)
    group_error = _group_mean(history["error_history"], groups)
    gradient_auc = np.trapz(group_gradient[:, config.selection_step - 1 :], axis=1)

    grid, observed, random_curve, upper = cumulative_energy_controls(
        future_energy, early_fc, config.cumulative_grid_size,
        config.random_rank_repeats, rng,
    )
    threshold_high, threshold_low = threshold_enrichment(
        future_energy, early_fc, config.masking_fractions
    )

    representation = representation_metrics(
        model, outcome_inputs, outcome_labels, config.inference_batch_size
    )
    representation_group = np.stack(
        [np.nanmean(representation[:, groups[key]], axis=1) for key in ("high", "low")]
    )

    selection_model = build_mlp(config, device)
    selection_model.load_state_dict(selection_state)
    acute_loss, acute_accuracy = acute_masking_analysis(
        selection_model, early_fc, covariates, outcome_inputs, outcome_labels,
        config, rng,
    )
    persistent_steps, persistent_loss, persistent_accuracy = persistent_masking_analysis(
        selection_state, train_loader_factory, early_fc, covariates,
        outcome_inputs, outcome_labels, config, seed, device,
    )

    return {
        "seed": np.asarray(seed),
        "losses": history["losses"],
        "loss_drop_onset": np.asarray(detect_loss_drop_onset(history["losses"])),
        "snapshot_steps": history["snapshot_steps"],
        "group_gradient": group_gradient,
        "gradient_auc": gradient_auc,
        "cumulative_grid": grid,
        "cumulative_observed": observed,
        "cumulative_random": random_curve,
        "cumulative_upper": upper,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
        "representation_group": representation_group,
        "acute_loss": acute_loss,
        "acute_accuracy": acute_accuracy,
        "persistent_steps": persistent_steps,
        "persistent_loss": persistent_loss,
        "persistent_accuracy": persistent_accuracy,
        "group_error": group_error,
    }


def save_seed_result(result: Mapping[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return path


def load_seed_result(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def stack_seed_results(results: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not results:
        raise ValueError("No seed results were supplied.")
    ordered = sorted(results, key=lambda item: int(item["seed"]))
    keys = ordered[0].keys()
    return {key: np.stack([np.asarray(item[key]) for item in ordered]) for key in keys}


def run_or_load_all_seeds(
    train_loader_factory: Callable[[int], Any],
    selection_inputs: torch.Tensor,
    outcome_inputs: torch.Tensor,
    outcome_labels: np.ndarray,
    config: Fig5SupplementConfig,
    result_dir: str | Path,
    device: str | torch.device | None = None,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in config.seeds:
        path = result_dir / f"fig5a_f_seed_{seed:03d}.npz"
        if path.exists() and not overwrite:
            result = load_seed_result(path)
        else:
            result = run_seed_analysis(
                train_loader_factory, selection_inputs, outcome_inputs,
                outcome_labels, config, seed, device,
            )
            save_seed_result(result, path)
        results.append(result)
    combined = stack_seed_results(results)
    np.savez_compressed(result_dir / "fig5a_f_all_seeds.npz", **combined)
    with open(result_dir / "fig5a_f_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
    return combined


def _mean_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=axis)
    n = np.sum(np.isfinite(values), axis=axis)
    sem = np.nanstd(values, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, 1.96 * sem


def _paired_p(values_a: np.ndarray, values_b: np.ndarray) -> float:
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if valid.sum() < 3 or np.allclose(values_a[valid], values_b[valid]):
        return np.nan
    return float(wilcoxon(values_a[valid], values_b[valid]).pvalue)


def statistical_summary(results: Mapping[str, np.ndarray]) -> dict[str, Any]:
    high_auc = results["gradient_auc"][:, 0]
    low_auc = results["gradient_auc"][:, 2]
    persistent_auc = np.trapz(results["persistent_loss"], x=results["persistent_steps"][0], axis=2)
    return {
        "n_independent_seeds": int(len(results["seed"])),
        "gradient_auc_high_vs_low_wilcoxon_p": _paired_p(high_auc, low_auc),
        "persistent_loss_auc_high_mask_vs_unmasked_wilcoxon_p": _paired_p(
            persistent_auc[:, 1], persistent_auc[:, 0]
        ),
        "acute_primary_fraction": 0.20,
        "inference_unit": "independently initialized and trained model seed",
    }


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.10, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def _plot_seed_points(ax: plt.Axes, values: np.ndarray, labels: Sequence[str], colors: Sequence[str]) -> None:
    x = np.arange(len(labels))
    for row in values:
        ax.plot(x, row, color="#C4C8CD", lw=0.45, alpha=0.6, zorder=1)
        ax.scatter(x, row, color=colors, s=7, alpha=0.65, zorder=2, linewidths=0)
    mean, ci = _mean_ci(values, axis=0)
    ax.errorbar(x, mean, yerr=ci, fmt="none", ecolor=DARK, lw=1.0, capsize=2, zorder=3)
    ax.scatter(x, mean, color=colors, edgecolor=DARK, linewidth=0.4, s=22, zorder=4)
    ax.set_xticks(x, labels)


def plot_fig5a_f_supplement(
    results: Mapping[str, np.ndarray],
    config: Fig5SupplementConfig,
) -> plt.Figure:
    """Create the eight mechanism panels retained as Supplementary Fig. S6a-h."""

    set_nature_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.20, 5.25), constrained_layout=False)
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.965, wspace=0.48, hspace=0.58)
    for ax, label in zip(axes.ravel(), "abcdefgh"):
        _panel_label(ax, label)

    ax = axes[0, 0]
    relative = np.arange(-config.phase_window, config.phase_window + 1)
    aligned = np.full((len(results["seed"]), 3, len(relative)), np.nan)
    for i, onset in enumerate(results["loss_drop_onset"].astype(int)):
        indices = onset + relative - 1
        valid = (indices >= 0) & (indices < config.max_steps)
        aligned[i][:, valid] = results["group_gradient"][i][:, indices[valid]]
    for group, color, name in zip(range(3), [HIGH_COLOR, MID_COLOR, LOW_COLOR], ["High FC", "Middle FC", "Low FC"]):
        mean, ci = _mean_ci(aligned[:, group], axis=0)
        ax.plot(relative, mean, color=color, label=name)
        ax.fill_between(relative, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    ax.axvline(0, color=DARK, lw=0.7, ls="--")
    ax.set(xlabel="Steps from rapid loss drop", ylabel=r"Gradient $L_2$ norm")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    _plot_seed_points(ax, results["gradient_auc"], ["High", "Middle", "Low"], [HIGH_COLOR, MID_COLOR, LOW_COLOR])
    ax.set(ylabel="Post-selection gradient AUC", title="Independent runs")

    ax = axes[0, 2]
    for key, color, label, ls in [
        ("cumulative_observed", HIGH_COLOR, "Ranked by early FC", "-"),
        ("cumulative_random", RANDOM_COLOR, "Random rank", "--"),
        ("cumulative_upper", UPPER_COLOR, "Gradient-rank bound", ":"),
    ]:
        mean, ci = _mean_ci(results[key], axis=0)
        grid = results["cumulative_grid"][0]
        ax.plot(grid * 100, mean * 100, color=color, label=label, ls=ls)
        if key != "cumulative_upper":
            ax.fill_between(grid * 100, (mean - ci) * 100, (mean + ci) * 100, color=color, alpha=0.15, linewidth=0)
    ax.plot([0, 100], [0, 100], color=LIGHT, lw=0.8)
    ax.set(xlabel="Top-ranked neurons (%)", ylabel="Future gradient energy (%)")
    ax.legend(loc="lower right")

    ax = axes[0, 3]
    fractions = np.asarray(config.masking_fractions) * 100
    for key, color, label in [("threshold_high", HIGH_COLOR, "High FC"), ("threshold_low", LOW_COLOR, "Low FC")]:
        mean, ci = _mean_ci(results[key], axis=0)
        ax.errorbar(fractions, mean, yerr=ci, color=color, marker="o", ms=3, capsize=2, label=label)
    ax.axhline(1, color=DARK, lw=0.6, ls="--")
    ax.set(xlabel="Selected fraction (%)", ylabel="Gradient-energy enrichment", title="Threshold robustness")
    ax.legend()

    ax = axes[1, 0]
    differences = results["representation_group"][:, 0] - results["representation_group"][:, 1]
    _plot_seed_points(ax, differences, ["Selectivity", "Template", "Spatial"], [HIGH_COLOR] * 3)
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set(ylabel="High FC - low FC", title="Final representation")

    ax = axes[1, 1]
    for condition, color, label in zip(range(4), [HIGH_COLOR, LOW_COLOR, RANDOM_COLOR, MATCHED_COLOR], ["High FC", "Low FC", "Random", "Matched"]):
        mean, ci = _mean_ci(results["acute_loss"][:, condition], axis=0)
        ax.errorbar(fractions, mean, yerr=ci, color=color, marker="o", ms=2.8, capsize=1.5, label=label)
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set(xlabel="Masked neurons (%)", ylabel=r"Acute $\Delta$ loss")
    ax.legend(ncol=2, columnspacing=0.7, handlelength=1.2)

    ax = axes[1, 2]
    continuation_x = results["persistent_steps"][0]
    for condition, color, label in zip(range(5), [DARK, HIGH_COLOR, LOW_COLOR, RANDOM_COLOR, MATCHED_COLOR], ["Unmasked", "High FC", "Low FC", "Random", "Matched"]):
        mean, ci = _mean_ci(results["persistent_loss"][:, condition], axis=0)
        ax.plot(continuation_x, mean, color=color, label=label)
        ax.fill_between(continuation_x, mean - ci, mean + ci, color=color, alpha=0.10, linewidth=0)
    ax.set(xlabel="Steps after fixed masking", ylabel="Held-out loss")
    ax.legend(ncol=2, columnspacing=0.7, handlelength=1.2)

    ax = axes[1, 3]
    snapshot_x = results["snapshot_steps"][0]
    for group, color, name in zip(range(3), [HIGH_COLOR, MID_COLOR, LOW_COLOR], ["High FC", "Middle FC", "Low FC"]):
        mean, ci = _mean_ci(results["group_error"][:, group], axis=0)
        ax.plot(snapshot_x, mean, color=color, label=name)
        ax.fill_between(snapshot_x, mean - ci, mean + ci, color=color, alpha=0.15, linewidth=0)
    ax.axvline(config.selection_step, color=DARK, lw=0.6, ls="--")
    ax.set(xlabel="Training step", ylabel="Error-signal RMS", title="Groups fixed at step 180")
    return fig

def export_figure_bundle(
    fig: plt.Figure,
    results: Mapping[str, np.ndarray],
    output_dir: str | Path,
    stem: str = "Supplementary_Fig5a_f",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "tiff": output_dir / f"{stem}.tiff",
        "png": output_dir / f"{stem}.png",
        "statistics": output_dir / f"{stem}_statistics.json",
        "source_data": output_dir / f"{stem}_source_data.npz",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    np.savez_compressed(paths["source_data"], **results)
    with open(paths["statistics"], "w", encoding="utf-8") as handle:
        json.dump(statistical_summary(results), handle, ensure_ascii=False, indent=2)
    return paths
