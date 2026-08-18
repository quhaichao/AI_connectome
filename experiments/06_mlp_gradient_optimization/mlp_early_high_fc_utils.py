"""Reusable analyses for main Figure 5g--i.

The module keeps the original analyses (Adam, BatchNorm, standard residual and
FC-guided residual comparisons) while separating model definitions, training,
FC measurements, caching and plotting. Curve statistics are computed from
independently initialized training seeds. FC matrices, network diagrams and
single-value metric bars intentionally use one predeclared representative seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


STANDARD_COLOR = "#777777"
ADAM_COLOR = "#D65F5F"
BATCHNORM_COLOR = "#59A14F"
PLAIN_COLOR = "#777777"
RESIDUAL_COLOR = "#4C78A8"
FC_RESIDUAL_COLOR = "#B14E8F"
LIGHT_GREY = "#D9DDE2"
DARK_GREY = "#30343B"


@dataclass(frozen=True)
class PairExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    representative_seed: int = 0
    hidden_dims: tuple[int, ...] = (300, 300)
    batch_size: int = 256
    total_steps: int = 300
    evaluation_interval: int = 5
    learning_rate: float = 0.05
    adam_learning_rate: float = 1e-3
    top_fc_ratio: float = 0.05
    fc_matrix_steps: tuple[int, ...] = (15, 20, 25)
    metric_step: int = 25
    analysis_samples: int = 2000

    def validate(self) -> None:
        # if len(self.seeds) < 20:
        #     raise ValueError("Curve analyses require at least 20 independent seeds.")
        if self.representative_seed not in self.seeds:
            raise ValueError("representative_seed must be included in seeds.")
        if self.metric_step not in self.fc_matrix_steps:
            raise ValueError("metric_step must be included in fc_matrix_steps.")
        if self.total_steps <= max(self.fc_matrix_steps):
            raise ValueError("total_steps must exceed all fc_matrix_steps.")


@dataclass(frozen=True)
class ResidualExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    representative_seed: int = 0
    hidden_dim: int = 256
    batch_size: int = 512
    total_steps: int = 300
    evaluation_interval: int = 5
    learning_rate: float = 0.05
    gate_strength: float = 2.0
    gate_top_fraction: float = 0.25
    fc_coupling_strength: float = 0.80
    coupling_steps: int = 40
    fc_threshold: float = 0.70
    top_fc_ratio: float = 0.05
    fc_matrix_steps: tuple[int, ...] = (5, 8, 10, 12)
    metric_step: int = 5
    analysis_samples: int = 2000

    def validate(self) -> None:
        # if len(self.seeds) < 20:
        #     raise ValueError("Curve analyses require at least 20 independent seeds.")
        if self.representative_seed not in self.seeds:
            raise ValueError("representative_seed must be included in seeds.")
        if self.metric_step not in self.fc_matrix_steps:
            raise ValueError("metric_step must be included in fc_matrix_steps.")
        if self.total_steps <= max(self.fc_matrix_steps):
            raise ValueError("total_steps must exceed all fc_matrix_steps.")
        if self.gate_strength <= 0:
            raise ValueError("gate_strength must be positive.")
        if not 0 < self.gate_top_fraction < 1:
            raise ValueError("gate_top_fraction must be in (0, 1).")
        if self.fc_coupling_strength < 0:
            raise ValueError("fc_coupling_strength must be non-negative.")
        if not 2 <= self.coupling_steps <= self.total_steps:
            raise ValueError("coupling_steps must be in [2, total_steps].")


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


def make_train_loader(dataset: Any, batch_size: int, seed: int, num_workers: int = 2) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def collect_analysis_batch(dataset: Any, n_samples: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False, num_workers=0)
    inputs, targets = next(iter(loader))
    return inputs, targets


def _next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, torch.Tensor, torch.Tensor]:
    try:
        inputs, targets = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        inputs, targets = next(iterator)
    return iterator, inputs, targets


def _initialize_linear_layers(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0)


class ExperimentMLP(nn.Module):
    """Original two-hidden-layer MLP, optionally with BatchNorm."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: Sequence[int] = (300, 300),
        num_classes: int = 10,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        self.use_batchnorm = use_batchnorm
        self.linear_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        current_dim = input_size
        for hidden_dim in self.hidden_dims:
            self.linear_layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        self.output_layer = nn.Linear(current_dim, num_classes)
        _initialize_linear_layers(self)

    def forward_with_hidden(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = inputs.view(inputs.size(0), -1)
        first_hidden: torch.Tensor | None = None
        for index, linear in enumerate(self.linear_layers):
            hidden = linear(hidden)
            if self.use_batchnorm:
                hidden = self.bn_layers[index](hidden)
            hidden = F.relu(hidden)
            if index == 0:
                first_hidden = hidden
        if first_hidden is None:
            raise RuntimeError("The MLP must contain at least one hidden layer.")
        return self.output_layer(hidden), first_hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(inputs)[0]

    def get_hidden_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(inputs)[1]


def sync_linear_weights(source: ExperimentMLP, target: ExperimentMLP) -> None:
    """Match all comparable linear weights before an optimizer/BN comparison."""

    source_layers = [*source.linear_layers, source.output_layer]
    target_layers = [*target.linear_layers, target.output_layer]
    with torch.no_grad():
        for source_layer, target_layer in zip(source_layers, target_layers):
            target_layer.weight.copy_(source_layer.weight)
            target_layer.bias.copy_(source_layer.bias)


def _safe_fc_matrix(activations: np.ndarray) -> np.ndarray:
    activations = np.asarray(activations, dtype=float)
    valid = np.std(activations, axis=0) > 1e-6
    if valid.sum() < 2:
        return np.eye(2, dtype=float)
    corr = np.corrcoef(activations[:, valid], rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def fc_summary(corr: np.ndarray, top_ratio: float = 0.05, bins: int = 50) -> tuple[float, float]:
    upper = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
    if len(upper) == 0:
        return 0.0, 0.0
    top_k = max(1, int(len(upper) * top_ratio))
    top_fc_mean = float(np.mean(np.sort(upper)[-top_k:]))
    hist, _ = np.histogram(upper, bins=bins, range=(-1, 1), density=True)
    hist = hist + 1e-12
    entropy = float(-np.sum(hist * np.log(hist)))
    return top_fc_mean, entropy


def high_fc_ratio(corr: np.ndarray, threshold: float = 0.70) -> float:
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    return float(np.mean(upper > threshold)) if len(upper) else 0.0


def gradient_l2_summary(weight_gradient: torch.Tensor) -> tuple[float, float]:
    norms = torch.linalg.vector_norm(weight_gradient, dim=1).detach().cpu().numpy()
    top_k = min(10, len(norms))
    concentration = float(np.sort(norms)[-top_k:].sum() / (norms.sum() + 1e-8))
    return float(np.mean(norms)), concentration


def evaluate_first_layer(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    top_fc_ratio: float,
) -> tuple[float, float, float, float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits, hidden = model.forward_with_hidden(inputs)
        loss = float(F.cross_entropy(logits, targets).cpu())
        accuracy = float((logits.argmax(1) == targets).float().mean().cpu() * 100.0)
        corr = _safe_fc_matrix(hidden.detach().cpu().numpy())
    top_fc_mean, entropy = fc_summary(corr, top_fc_ratio)
    return loss, accuracy, top_fc_mean, entropy, corr


def _evaluation_steps(total_steps: int, interval: int) -> np.ndarray:
    steps = np.arange(0, total_steps, interval, dtype=int)
    if steps[-1] != total_steps - 1:
        steps = np.append(steps, total_steps - 1)
    return steps


def run_pair_seed(
    experiment: str,
    seed: int,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    analysis_targets: torch.Tensor,
    config: PairExperimentConfig,
    device: str | torch.device | None = None,
    num_workers: int = 2,
    keep_representative: bool = False,
) -> dict[str, np.ndarray]:
    """Run Standard-vs-Adam or Standard-vs-BatchNorm for one seed."""

    if experiment not in {"adam", "batchnorm"}:
        raise ValueError("experiment must be 'adam' or 'batchnorm'.")
    config.validate()
    device = resolve_device(device)
    set_seed(seed)
    standard = ExperimentMLP(hidden_dims=config.hidden_dims).to(device)
    set_seed(seed)
    variant = ExperimentMLP(
        hidden_dims=config.hidden_dims,
        use_batchnorm=(experiment == "batchnorm"),
    ).to(device)
    sync_linear_weights(standard, variant)
    standard_optimizer = torch.optim.SGD(standard.parameters(), lr=config.learning_rate)
    if experiment == "adam":
        variant_optimizer = torch.optim.Adam(variant.parameters(), lr=config.adam_learning_rate)
    else:
        variant_optimizer = torch.optim.SGD(variant.parameters(), lr=config.learning_rate)

    loader = make_train_loader(train_dataset, config.batch_size, seed, num_workers)
    iterator = iter(loader)
    analysis_inputs = analysis_inputs.to(device)
    analysis_targets = analysis_targets.to(device)
    eval_steps = _evaluation_steps(config.total_steps, config.evaluation_interval)
    eval_lookup = {int(step): index for index, step in enumerate(eval_steps)}
    curves = np.empty((2, 5, len(eval_steps)), dtype=float)
    matrices: dict[str, np.ndarray] = {}

    models = (standard, variant)
    optimizers = (standard_optimizer, variant_optimizer)
    latest_gradient = np.zeros(2, dtype=float)
    for step in range(config.total_steps):
        iterator, batch_inputs, batch_targets = _next_batch(iterator, loader)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        for model_index, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_inputs), batch_targets)
            loss.backward()
            latest_gradient[model_index] = gradient_l2_summary(
                model.linear_layers[0].weight.grad
            )[0]
            optimizer.step()

        needs_curve = step in eval_lookup
        needs_matrix = keep_representative and step in config.fc_matrix_steps
        if not (needs_curve or needs_matrix):
            continue
        for model_index, model in enumerate(models):
            loss, accuracy, top_fc, entropy, corr = evaluate_first_layer(
                model, analysis_inputs, analysis_targets, config.top_fc_ratio
            )
            if needs_curve:
                position = eval_lookup[step]
                curves[model_index, :, position] = (
                    loss, accuracy, top_fc, entropy, latest_gradient[model_index]
                )
            if needs_matrix:
                matrices[f"matrix_{model_index}_{step}"] = corr

    result: dict[str, np.ndarray] = {
        "seed": np.asarray(seed),
        "steps": eval_steps,
        "curves": curves,
    }
    result.update(matrices)
    return result


class PlainMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)
        _initialize_linear_layers(self)

    def forward_with_hidden(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden1 = F.relu(self.fc1(inputs.view(inputs.size(0), -1)))
        hidden2 = F.relu(self.fc2(hidden1))
        return self.fc3(hidden2), hidden1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(inputs)[0]

    def forward_with_block_output(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden1 = F.relu(self.fc1(inputs.view(inputs.size(0), -1)))
        hidden2 = F.relu(self.fc2(hidden1))
        return self.fc3(hidden2), hidden2


class StandardResidualMLP(PlainMLP):
    def forward_with_hidden(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden1 = F.relu(self.fc1(inputs.view(inputs.size(0), -1)))
        hidden2 = F.relu(self.fc2(hidden1))
        return self.fc3(hidden1 + hidden2), hidden1

    def forward_with_block_output(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden1 = F.relu(self.fc1(inputs.view(inputs.size(0), -1)))
        hidden2 = F.relu(self.fc2(hidden1))
        combined = hidden1 + hidden2
        return self.fc3(combined), combined


class FCResidualGate(nn.Module):
    """Stable boost-only residual gate targeted to high-FC units.

    Detached, EMA-smoothed FC values are converted to within-layer ranks. The
    highest-FC fraction receives a smooth positive boost while every other unit
    retains at least the standard residual coefficient of one. Rank gating is
    insensitive to dataset-specific FC scale and never reverses residual signs.
    """

    _RANK_TEMPERATURE = 0.08
    _FC_EMA_MOMENTUM = 0.70

    def __init__(
        self,
        num_features: int,
        strength: float = 2.0,
        top_fraction: float = 0.25,
        coupling_strength: float = 0.80,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if strength <= 0:
            raise ValueError("strength must be positive.")
        if not 0 < top_fraction < 1:
            raise ValueError("top_fraction must be in (0, 1).")
        if coupling_strength < 0:
            raise ValueError("coupling_strength must be non-negative.")
        self.strength = strength
        self.top_fraction = top_fraction
        self.coupling_strength = coupling_strength
        self.eps = eps
        self.register_buffer("running_fc", torch.zeros(num_features))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def _compute_fc(self, activations: torch.Tensor) -> torch.Tensor:
        centered = activations - activations.mean(0, keepdim=True)
        standardized = centered / (centered.std(0, keepdim=True) + self.eps)
        corr = (standardized.T @ standardized) / max(standardized.shape[0] - 1, 1)
        corr.fill_diagonal_(0)
        return corr.abs().mean(dim=1)

    def forward(
        self,
        transformed: torch.Tensor,
        residual: torch.Tensor,
        strength_scale: float = 1.0,
        coupling_scale: float | None = None,
    ) -> torch.Tensor:
        if coupling_scale is None:
            coupling_scale = strength_scale
        if self.training:
            with torch.no_grad():
                batch_fc = self._compute_fc(residual.detach())
                if not bool(self.initialized.item()):
                    self.running_fc.copy_(batch_fc)
                    self.initialized.fill_(True)
                else:
                    self.running_fc.mul_(self._FC_EMA_MOMENTUM).add_(
                        batch_fc, alpha=1.0 - self._FC_EMA_MOMENTUM
                    )
        fc = self.running_fc
        order = torch.argsort(fc)
        ranks = torch.empty_like(fc)
        ranks[order] = torch.linspace(0.0, 1.0, fc.numel(), device=fc.device, dtype=fc.dtype)
        cutoff = 1.0 - self.top_fraction
        boost = torch.sigmoid((ranks - cutoff) / self._RANK_TEMPERATURE)
        gate = 1.0 + float(strength_scale) * self.strength * boost
        # A shared low-rank residual component directly couples the selected
        # high-FC units. The signal is computed within each sample, so inference
        # remains independent of the other samples in a batch.
        shared_activity = (residual * boost.unsqueeze(0)).sum(1, keepdim=True)
        # Sqrt normalization preserves the variance of the shared component.
        # Mean normalization made the common signal vanish as more units were selected.
        shared_activity = shared_activity / torch.sqrt(boost.square().sum() + self.eps)
        coupling = (
            float(coupling_scale)
            * self.coupling_strength
            * shared_activity
            * boost.unsqueeze(0)
        )
        return gate.unsqueeze(0) * residual + transformed + coupling

class FCGuidedResidualMLP(PlainMLP):
    """Residual MLP with four public FC-guidance controls.

    Rank smoothing, EMA smoothing and schedule proportions are implementation
    constants. ``coupling_steps`` scales the complete ramp-hold-decay schedule.
    """

    _GATE_RAMP_STEPS = 5
    _COUPLING_RAMP_FRACTION = 0.25
    _COUPLING_HOLD_FRACTION = 0.625

    def __init__(
        self,
        hidden_dim: int = 256,
        gate_strength: float = 2.0,
        gate_top_fraction: float = 0.25,
        fc_coupling_strength: float = 0.80,
        coupling_steps: int = 40,
    ) -> None:
        super().__init__(hidden_dim)
        self.current_step = 0
        self.coupling_ramp_steps = max(
            1, round(coupling_steps * self._COUPLING_RAMP_FRACTION)
        )
        self.coupling_hold_steps = max(
            self.coupling_ramp_steps,
            round(coupling_steps * self._COUPLING_HOLD_FRACTION),
        )
        self.coupling_decay_steps = max(1, coupling_steps - self.coupling_hold_steps)
        self.fc_gate = FCResidualGate(
            hidden_dim,
            strength=gate_strength,
            top_fraction=gate_top_fraction,
            coupling_strength=fc_coupling_strength,
        )

    def set_step(self, step: int) -> None:
        self.current_step = step

    def _guidance_scales(self) -> tuple[float, float]:
        gate_scale = min(1.0, (self.current_step + 1) / self._GATE_RAMP_STEPS)
        coupling_ramp = min(1.0, (self.current_step + 1) / self.coupling_ramp_steps)
        if self.current_step <= self.coupling_hold_steps:
            coupling_scale = coupling_ramp
        else:
            elapsed = self.current_step - self.coupling_hold_steps
            coupling_scale = max(0.0, 1.0 - elapsed / self.coupling_decay_steps)
        return gate_scale, coupling_scale

    def _combine(self, hidden1: torch.Tensor, hidden2: torch.Tensor) -> torch.Tensor:
        gate_scale, coupling_scale = self._guidance_scales()
        return self.fc_gate(
            hidden2,
            hidden1,
            strength_scale=gate_scale,
            coupling_scale=coupling_scale,
        )

    def forward_with_hidden(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden1 = F.relu(self.fc1(inputs.view(inputs.size(0), -1)))
        hidden2 = F.relu(self.fc2(hidden1))
        combined = self._combine(hidden1, hidden2)
        return self.fc3(combined), hidden1

    def forward_with_block_output(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden1 = F.relu(self.fc1(inputs.view(inputs.size(0), -1)))
        hidden2 = F.relu(self.fc2(hidden1))
        combined = self._combine(hidden1, hidden2)
        return self.fc3(combined), combined


def evaluate_residual_block(
    model: PlainMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    top_fc_ratio: float,
) -> tuple[float, float, float, float, np.ndarray]:
    """Evaluate performance and FC at the representation sent to the classifier."""

    model.eval()
    with torch.no_grad():
        logits, block_output = model.forward_with_block_output(inputs)
        loss = float(F.cross_entropy(logits, targets).detach().cpu())
        accuracy = float((logits.argmax(1) == targets).float().mean().detach().cpu() * 100.0)
        corr = _safe_fc_matrix(block_output.detach().cpu().numpy())
    top_fc_mean, entropy = fc_summary(corr, top_fc_ratio)
    return loss, accuracy, top_fc_mean, entropy, corr


def run_residual_seed(
    seed: int,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    analysis_targets: torch.Tensor,
    config: ResidualExperimentConfig,
    device: str | torch.device | None = None,
    num_workers: int = 2,
    keep_representative: bool = False,
) -> dict[str, np.ndarray]:
    """Compare plain, standard-residual and FC-guided-residual MLPs."""

    config.validate()
    device = resolve_device(device)
    constructors = (
        lambda: PlainMLP(config.hidden_dim),
        lambda: StandardResidualMLP(config.hidden_dim),
        lambda: FCGuidedResidualMLP(
            hidden_dim=config.hidden_dim,
            gate_strength=config.gate_strength,
            gate_top_fraction=config.gate_top_fraction,
            fc_coupling_strength=config.fc_coupling_strength,
            coupling_steps=config.coupling_steps,
        ),
    )
    models = []
    for constructor in constructors:
        set_seed(seed)
        models.append(constructor().to(device))
    optimizers = [torch.optim.SGD(model.parameters(), lr=config.learning_rate) for model in models]
    loader = make_train_loader(train_dataset, config.batch_size, seed, num_workers)
    iterator = iter(loader)
    analysis_inputs = analysis_inputs.to(device)
    analysis_targets = analysis_targets.to(device)
    eval_steps = _evaluation_steps(config.total_steps, config.evaluation_interval)
    eval_lookup = {int(step): index for index, step in enumerate(eval_steps)}
    curves = np.empty((3, 5, len(eval_steps)), dtype=float)
    latest_concentration = np.zeros(3, dtype=float)
    matrices: dict[str, np.ndarray] = {}

    for step in range(config.total_steps):
        iterator, batch_inputs, batch_targets = _next_batch(iterator, loader)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        for model_index, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.train()
            if isinstance(model, FCGuidedResidualMLP):
                model.set_step(step)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_inputs), batch_targets)
            loss.backward()
            latest_concentration[model_index] = gradient_l2_summary(model.fc1.weight.grad)[1]
            optimizer.step()

        needs_curve = step in eval_lookup
        needs_matrix = keep_representative and step in config.fc_matrix_steps
        if not (needs_curve or needs_matrix):
            continue
        for model_index, model in enumerate(models):
            loss, accuracy, top_fc, _, corr = evaluate_residual_block(
                model, analysis_inputs, analysis_targets, config.top_fc_ratio
            )
            if needs_curve:
                position = eval_lookup[step]
                curves[model_index, :, position] = (
                    loss,
                    accuracy,
                    high_fc_ratio(corr, config.fc_threshold),
                    top_fc,
                    latest_concentration[model_index],
                )
            if needs_matrix:
                matrices[f"matrix_{model_index}_{step}"] = corr

    result: dict[str, np.ndarray] = {
        "seed": np.asarray(seed),
        "steps": eval_steps,
        "curves": curves,
    }
    result.update(matrices)
    return result


def save_seed_result(result: Mapping[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return path


def load_seed_result(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _combine_seed_results(results: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    ordered = sorted(results, key=lambda result: int(result["seed"]))
    combined = {
        "seeds": np.asarray([int(result["seed"]) for result in ordered]),
        "steps": np.asarray(ordered[0]["steps"]),
        "curves": np.stack([np.asarray(result["curves"]) for result in ordered]),
    }
    return combined


def _validate_cached_config(config: Any, result_dir: Path, overwrite: bool) -> None:
    """Prevent silently reusing curves computed with a different configuration."""

    config_path = result_dir / "config.json"
    if overwrite or not config_path.exists():
        return
    with open(config_path, "r", encoding="utf-8") as handle:
        cached = json.load(handle)
    current = json.loads(json.dumps(asdict(config)))
    if cached != current:
        raise ValueError(
            f"Cached results in {result_dir} were generated with a different configuration. "
            "Set OVERWRITE = True to recompute them, or use a new result directory."
        )


def run_or_load_pair_repeats(
    experiment: str,
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    analysis_targets: torch.Tensor,
    config: PairExperimentConfig,
    result_dir: str | Path,
    device: str | torch.device | None = None,
    num_workers: int = 2,
    overwrite: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    config.validate()
    result_dir = Path(result_dir) / experiment
    result_dir.mkdir(parents=True, exist_ok=True)
    _validate_cached_config(config, result_dir, overwrite)
    seed_results = []
    representative: dict[str, np.ndarray] | None = None
    for seed in config.seeds:
        path = result_dir / f"seed_{seed:03d}.npz"
        if path.exists() and not overwrite:
            result = load_seed_result(path)
        else:
            result = run_pair_seed(
                experiment,
                seed,
                train_dataset,
                analysis_inputs,
                analysis_targets,
                config,
                device,
                num_workers,
                keep_representative=(seed == config.representative_seed),
            )
            save_seed_result(result, path)
        if seed == config.representative_seed:
            required = [f"matrix_{model}_{step}" for model in range(2) for step in config.fc_matrix_steps]
            if any(key not in result for key in required):
                result = run_pair_seed(
                    experiment, seed, train_dataset, analysis_inputs, analysis_targets,
                    config, device, num_workers, keep_representative=True,
                )
                save_seed_result(result, path)
            representative = result
        seed_results.append(result)
    if representative is None:
        raise RuntimeError("Representative seed result was not found.")
    combined = _combine_seed_results(seed_results)
    np.savez_compressed(result_dir / "all_seed_curves.npz", **combined)
    with open(result_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    return combined, representative


def run_or_load_residual_repeats(
    train_dataset: Any,
    analysis_inputs: torch.Tensor,
    analysis_targets: torch.Tensor,
    config: ResidualExperimentConfig,
    result_dir: str | Path,
    device: str | torch.device | None = None,
    num_workers: int = 2,
    overwrite: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    config.validate()
    result_dir = Path(result_dir) / "residual"
    result_dir.mkdir(parents=True, exist_ok=True)
    _validate_cached_config(config, result_dir, overwrite)
    seed_results = []
    representative: dict[str, np.ndarray] | None = None
    for seed in config.seeds:
        path = result_dir / f"seed_{seed:03d}.npz"
        if path.exists() and not overwrite:
            result = load_seed_result(path)
        else:
            result = run_residual_seed(
                seed,
                train_dataset,
                analysis_inputs,
                analysis_targets,
                config,
                device,
                num_workers,
                keep_representative=(seed == config.representative_seed),
            )
            save_seed_result(result, path)
        if seed == config.representative_seed:
            required = [f"matrix_{model}_{step}" for model in range(3) for step in config.fc_matrix_steps]
            if any(key not in result for key in required):
                result = run_residual_seed(
                    seed, train_dataset, analysis_inputs, analysis_targets,
                    config, device, num_workers, keep_representative=True,
                )
                save_seed_result(result, path)
            representative = result
        seed_results.append(result)
    if representative is None:
        raise RuntimeError("Representative seed result was not found.")
    combined = _combine_seed_results(seed_results)
    np.savez_compressed(result_dir / "all_seed_curves.npz", **combined)
    with open(result_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    return combined, representative


def exponential_smooth(values: np.ndarray, factor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = values.copy()
    for index in range(1, output.shape[-1]):
        output[..., index] = factor * output[..., index - 1] + (1.0 - factor) * values[..., index]
    return output


def _mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, 1.96 * sem


def set_publication_style() -> None:
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
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _plot_curve_with_ci(
    ax: plt.Axes,
    steps: np.ndarray,
    seed_curves: np.ndarray,
    color: str,
    label: str,
    smooth_factor: float,
) -> None:
    smoothed = exponential_smooth(seed_curves, smooth_factor)
    mean, ci = _mean_ci(smoothed)
    ax.plot(steps, mean, color=color, label=label, lw=1.4)
    ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.20, linewidth=0)


def plot_pair_curves(
    results: Mapping[str, np.ndarray],
    experiment: str,
    top_fc_ratio: float = 0.05,
) -> plt.Figure:
    """Five Standard-vs-variant curves; every band summarizes >=20 seeds."""

    if experiment == "adam":
        labels, colors = ("SGD", "Adam"), (STANDARD_COLOR, ADAM_COLOR)
    elif experiment == "batchnorm":
        labels, colors = ("Standard", "BatchNorm"), (STANDARD_COLOR, BATCHNORM_COLOR)
    else:
        raise ValueError("experiment must be 'adam' or 'batchnorm'.")
    set_publication_style()
    steps = results["steps"]
    curves = results["curves"]
    top_fc_label = f"Top {100 * top_fc_ratio:g}% |FC| mean"
    metric_names = ("Validation loss", "Validation accuracy (%)", top_fc_label, "FC entropy", "Mean first-layer gradient norm")
    smooth_factors = (0.5, 0.5, 0.5, 0.8, 0.85)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for metric_index, ax in enumerate(axes.flat[:5]):
        for model_index in range(2):
            _plot_curve_with_ci(
                ax,
                steps,
                curves[:, model_index, metric_index],
                colors[model_index],
                labels[model_index],
                smooth_factors[metric_index],
            )
        ax.set_title(metric_names[metric_index])
        ax.set_xlabel("Training step")
        if metric_index == 0:
            ax.legend()
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.0,
        0.95,
        f"n = {len(results['seeds'])} seeds\nMean ± 95% CI across seeds",
        transform=axes.flat[5].transAxes,
        va="top",
    )
    fig.tight_layout()
    return fig


def plot_residual_curves(
    results: Mapping[str, np.ndarray],
    top_fc_ratio: float = 0.05,
) -> plt.Figure:
    set_publication_style()
    labels = ("No residual", "Standard residual", "FC-guided residual")
    colors = (PLAIN_COLOR, RESIDUAL_COLOR, FC_RESIDUAL_COLOR)
    metric_names = (
        "Validation loss",
        "Validation accuracy (%)",
        "Block-output high-FC ratio (r > 0.7)",
        f"Block-output top {100 * top_fc_ratio:g}% |FC| mean",
        "Gradient concentration (top 10 units)",
    )
    smooth_factors = (0.8, 0.8, 0.8, 0.8, 0.8)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for metric_index, ax in enumerate(axes.flat[:5]):
        for model_index in range(3):
            _plot_curve_with_ci(
                ax,
                results["steps"],
                results["curves"][:, model_index, metric_index],
                colors[model_index],
                labels[model_index],
                smooth_factors[metric_index],
            )
        ax.set_title(metric_names[metric_index])
        ax.set_xlabel("Training step")
        if metric_index == 0:
            ax.legend(fontsize=5.5)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.0,
        0.95,
        f"n = {len(results['seeds'])} seeds\nMean ± 95% CI across seeds",
        transform=axes.flat[5].transAxes,
        va="top",
    )
    fig.tight_layout()
    return fig


def cluster_corr_matrix(corr: np.ndarray) -> np.ndarray:
    distance = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    return corr[order][:, order]


def matrix_metrics(corr: np.ndarray, top_ratio: float = 0.30) -> tuple[float, float]:
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    top_k = max(1, int(len(upper) * top_ratio))
    fc_mean = float(np.mean(np.sort(upper)[-top_k:]))
    eigenvalues = np.maximum(np.linalg.eigvalsh(corr), 1e-12)
    probabilities = eigenvalues / eigenvalues.sum()
    effective_dimension = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return fc_mean, effective_dimension


def network_metrics(corr: np.ndarray, top_ratio: float = 0.05) -> tuple[float, float]:
    graph = _build_fc_graph(corr, top_ratio)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return 0.0, 0.0
    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=0)
        modularity = nx.community.modularity(graph, communities, weight="weight")
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
        modularity = nx.community.modularity(graph, communities, weight="weight")
    clustering = nx.average_clustering(graph, weight="weight")
    return float(modularity), float(clustering)

def full_network_metrics(corr: np.ndarray) -> tuple[float, float]:
    """Weighted modularity and clustering computed from all FC edges."""
    matrix = np.abs(np.asarray(corr, dtype=float)).copy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 0.0)
    # All non-zero FC edges are included; no top-ratio thresholding.
    graph = nx.from_numpy_array(matrix)
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return 0.0, 0.0
    try:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            seed=0,
        )
    except Exception:
        communities = list(
            nx.community.greedy_modularity_communities(
                graph,
                weight="weight",
            )
        )
    modularity = nx.community.modularity(
        graph,
        communities,
        weight="weight",
    )
    clustering = nx.average_clustering(
        graph,
        weight="weight",
    )
    return float(modularity), float(clustering)


def _build_fc_graph(corr: np.ndarray, top_ratio: float) -> nx.Graph:
    matrix = np.asarray(corr, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    threshold = np.quantile(upper, 1.0 - top_ratio)
    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for column in range(row + 1, matrix.shape[0]):
            if matrix[row, column] >= threshold:
                graph.add_edge(row, column, weight=float(matrix[row, column]))
    return graph


def draw_fc_network(
    corr: np.ndarray,
    ax: plt.Axes,
    title: str,
    top_ratio: float = 0.05,
    layout: str = "kamada",
    module_spacing: float = 1.0,
    min_degree: int = 2,
    node_color: str = "#E69F59",
    layout_seed: int = 0,
) -> None:
    graph = _build_fc_graph(corr, top_ratio)
    graph.remove_nodes_from([node for node, degree in graph.degree() if degree < min_degree])
    if graph.number_of_nodes() == 0:
        ax.set_axis_off()
        return
    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=layout_seed)
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    membership = {node: index for index, group in enumerate(communities) for node in group}
    layout = layout.lower()
    if layout in {"kamada", "kamada_kawai"}:
        positions = nx.kamada_kawai_layout(graph, weight="weight")
    elif layout == "spring":
        positions = nx.spring_layout(graph, seed=layout_seed, weight="weight", iterations=300)
    elif layout == "spectral":
        positions = nx.spectral_layout(graph, weight="weight")
    elif layout == "circular":
        positions = nx.circular_layout(graph)
    elif layout == "shell":
        positions = nx.shell_layout(graph)
    else:
        raise ValueError(
            "layout must be one of: 'kamada', 'spring', 'spectral', 'circular', or 'shell'."
        )

    if module_spacing <= 0:
        raise ValueError("module_spacing must be positive.")
    if module_spacing != 1.0 and len(communities) > 1:
        # Preserve the chosen topology-driven layout while moving community
        # centroids farther apart (or closer together). Node colors remain uniform.
        all_xy = np.asarray([positions[node] for node in graph.nodes], dtype=float)
        global_center = all_xy.mean(axis=0)
        adjusted = {}
        for community in communities:
            community = list(community)
            center = np.asarray([positions[node] for node in community]).mean(axis=0)
            shifted_center = global_center + module_spacing * (center - global_center)
            for node in community:
                adjusted[node] = np.asarray(positions[node]) - center + shifted_center
        positions = adjusted

    intra_edges = [edge for edge in graph.edges if membership[edge[0]] == membership[edge[1]]]
    inter_edges = [edge for edge in graph.edges if membership[edge[0]] != membership[edge[1]]]
    weights = np.asarray([graph[u][v]["weight"] for u, v in graph.edges], dtype=float)
    width_lookup = {
        edge: 0.25 + 1.20 * (graph[edge[0]][edge[1]]["weight"] - weights.min())
        / (np.ptp(weights) + 1e-12)
        for edge in graph.edges
    }
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edgelist=inter_edges,
        width=[0.75 * width_lookup[edge] for edge in inter_edges],
        alpha=0.12, edge_color="#70757A",
    )
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edgelist=intra_edges,
        width=[width_lookup[edge] for edge in intra_edges],
        alpha=0.24, edge_color="#70757A",
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=25,
        node_color=node_color,
        edgecolors="white",
        linewidths=0.1,
        alpha=0.95,
    )
    ax.set_title(f"{title}\nTop {100 * top_ratio:g}% edges", fontsize=6.5)
    ax.set_axis_off()


def plot_representative_matrices(
    representative: Mapping[str, np.ndarray],
    model_labels: Sequence[str],
    matrix_steps: Sequence[int],
    representative_seed: int,
) -> plt.Figure:
    set_publication_style()
    fig, axes = plt.subplots(
        len(model_labels),
        len(matrix_steps),
        figsize=(1.65 * len(matrix_steps), 1.55 * len(model_labels)),
        squeeze=False,
    )
    for model_index, label in enumerate(model_labels):
        for step_index, step in enumerate(matrix_steps):
            corr = representative[f"matrix_{model_index}_{step}"]
            mean_fc, dimension = matrix_metrics(corr)
            axes[model_index, step_index].imshow(
                cluster_corr_matrix(corr), cmap="RdBu_r", vmin=-1, vmax=1
            )
            axes[model_index, step_index].set_title(
                f"{label}, step {step}\nFC={mean_fc:.3f}; dim={dimension:.1f}", fontsize=5.8
            )
            axes[model_index, step_index].set_xticks([])
            axes[model_index, step_index].set_yticks([])
    fig.suptitle(f"Representative seed {representative_seed}", fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_representative_networks(
    representative: Mapping[str, np.ndarray],
    model_labels: Sequence[str],
    matrix_steps: Sequence[int],
    representative_seed: int,
    top_ratio: float = 0.05,
    layout: str = "kamada",
    module_spacing: float = 1.0,
    min_degree: int = 2,
    node_color: str = "#E69F59",
) -> plt.Figure:
    set_publication_style()
    fig, axes = plt.subplots(
        len(model_labels),
        len(matrix_steps),
        figsize=(2.0 * len(matrix_steps), 1.9 * len(model_labels)),
        squeeze=False,
    )
    for model_index, label in enumerate(model_labels):
        for step_index, step in enumerate(matrix_steps):
            corr = representative[f"matrix_{model_index}_{step}"]
            modularity, clustering = network_metrics(corr, top_ratio)
            draw_fc_network(
                corr,
                axes[model_index, step_index],
                f"{label}, step {step}\nQ={modularity:.3f}; C={clustering:.3f}",
                top_ratio=top_ratio,
                layout=layout,
                module_spacing=module_spacing,
                min_degree=min_degree,
                node_color=node_color,
                layout_seed=representative_seed,
            )
    fig.suptitle(f"Representative seed {representative_seed}", fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_representative_metric_bars(
    representative: Mapping[str, np.ndarray],
    model_labels: Sequence[str],
    model_colors: Sequence[str],
    metric_step: int,
    representative_seed: int,
    top_ratio: float = 0.05,
) -> plt.Figure:
    set_publication_style()
    values = []
    for model_index in range(len(model_labels)):
        corr = representative[f"matrix_{model_index}_{metric_step}"]
        mean_fc, dimension = matrix_metrics(corr)
        modularity, clustering = network_metrics(corr, top_ratio)
        values.append([mean_fc, dimension, modularity, clustering])
        # mean_fc, dimension = matrix_metrics(corr)
        # modularity, clustering = full_network_metrics(corr)
        # values.append([mean_fc, dimension, modularity, clustering])
    values = np.asarray(values)
    metric_labels = ("FC mean", "Effective dimension", "Modularity", "Clustering")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.8))
    x = np.arange(len(model_labels))
    for metric_index, ax in enumerate(axes):
        bars = ax.bar(x, values[:, metric_index], color=model_colors, width=0.85)
        ax.set_title(metric_labels[metric_index])
        ax.set_xticks(x, model_labels, rotation=30, ha="right")
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}" if metric_index == 1 else f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90
            )
    fig.suptitle(
        f"Seed {representative_seed}, step {metric_step}", fontsize=7.5
    )
    fig.tight_layout()
    return fig


def export_figure(fig: plt.Figure, output_dir: str | Path, stem: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "tiff": output_dir / f"{stem}.tiff",
        "png": output_dir / f"{stem}.png",
    }
    # fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    # fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight")
    # fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    return paths
