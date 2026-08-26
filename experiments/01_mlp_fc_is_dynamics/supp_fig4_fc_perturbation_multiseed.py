"""Standalone, confound-aware multi-seed FC perturbation analysis.

This module implements a paired branching experiment for Supplementary/Extended
Data Fig. 4.  It contains the full experiment, inference and plotting workflow
used by the streamlined multiseed notebook.

Primary design
--------------
1. Train one untreated scout MLP per seed to the pre-intervention checkpoint.
   Select node-disjoint high- and low-FC pairs only when their held-out FC is
   stable across repeated pre-intervention checkpoints.
2. Save the scout state immediately before intervention.  Restore that exact
   checkpoint for every branch, preserving neuron identity, initialization and
   minibatch history while avoiding selection on any intervention outcome.
3. Clone the same warm-up checkpoint for every pair x condition branch and use
   exactly the same minibatch schedule within a seed.
4. Manipulate dependence by leaving one neuron unchanged and rank-remapping the
   other toward, away from or independently of its partner.  The empirical
   marginal distribution of each targeted neuron is preserved exactly within a
   batch, while unnecessary two-neuron sample remapping is avoided.
5. For every pair and intervention step, match control perturbations to the
   realized active-branch activation displacement, sample-mapping disruption
   and RMS effect on downstream logits.  Compare the active manipulation with
   independent-noise, correlation-preserving and no-intervention controls.  The
6. Run the untreated continuation once per seed and reuse its trajectory for
   all selected pairs.  This removes redundant training without changing any
   paired comparison.
7. Treat seed as the inferential replicate.  Pairs are nested observations and
   are averaged within seed before hypothesis tests.

The code does not claim that a manipulation is specific merely because it was
requested to be specific.  It records FC shift, marginal preservation, activation
displacement, logit displacement and instantaneous loss displacement so that
specificity can be checked empirically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST


EPS = 1e-12


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the paired perturbation experiment.

    The reduced profile uses two independent initializations and two
    node-disjoint pairs per FC stratum.  Use ``smoke_config`` only to validate
    plumbing.
    """

    seeds: Tuple[int, ...] = (0, 1)
    hidden_dims: Tuple[int, int] = (100, 100)
    batch_size: int = 256
    learning_rate: float = 0.05
    # Five passes through the 58,000-example SGD pool after reserving 2,000
    # examples for pair selection: 5 * floor(58,000 / 256) = 1,130 steps.
    total_steps: int = 1130
    intervention_start: int = 100
    intervention_end: int = 300
    record_every: int = 5
    n_pairs_per_stratum: int = 2
    selection_samples: int = 2000
    evaluation_samples: int = 2000
    split_seed: int = 20260820
    high_fc_quantile: float = 0.90
    low_abs_fc_quantile: float = 0.15
    # Pair selection uses repeated held-out measurements available strictly
    # before intervention.  This avoids future-trajectory (look-ahead) selection.
    selection_start: int = 60
    selection_every: int = 20
    selection_high_mean_min: float = 0.20
    selection_high_min_fc: float = 0.10
    selection_low_abs_mean_max: float = 0.10
    selection_low_max_abs_fc: float = 0.15
    selection_max_fc_sd: float = 0.10
    stability_penalty: float = 1.0
    min_unit_std: float = 1e-6
    min_unit_active_fraction: float = 0.05
    min_output_weight_quantile: float = 0.25
    target_fc_shift: float = 0.25
    fc_target_tolerance: float = 0.03
    target_logit_rms_ratio: float = 0.05
    target_activation_delta_ratio: float = 0.35
    target_sample_mapping_corr: float = 0.80
    strength_grid: Tuple[float, ...] = (
        0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    )
    rho_grid: Tuple[float, ...] = (-1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0)
    # Four stochastic proposals per strength/rho combination retain a broad
    # candidate search while halving controller work relative to the original
    # eight-draw implementation.  Increase this only if dose/QC checks fail.
    candidate_draws: int = 4
    immediate_post_window: int = 100
    recovery_window: int = 50
    ci_bootstrap_reps: int = 5000
    conditions: Tuple[str, ...] = (
        "none",
        "active_rank",
        "independent_rank",
        "corr_preserving_rank",
    )
    data_root: str = "./data"
    output_dir: str = "./results/supp_fig4_fc_perturbation_multiseed"
    figure_dir: str = "./fig/supp_fig4_fc_perturbation_multiseed"
    device: str = "auto"

    def validate(self) -> None:
        if not (1 <= self.intervention_start <= self.intervention_end < self.total_steps):
            raise ValueError("Require 1 <= intervention_start <= intervention_end < total_steps.")
        if self.n_pairs_per_stratum < 2:
            raise ValueError("Use at least two pairs per stratum.")
        if len(self.seeds) < 2:
            raise ValueError("Use at least two independent seeds.")
        if not (1 <= self.selection_start < self.intervention_start):
            raise ValueError(
                "Require 1 <= selection_start < intervention_start."
            )
        if self.selection_every < 1:
            raise ValueError("selection_every must be positive.")
        if len(_selection_steps(self)) < 3:
            raise ValueError(
                "Stable pair selection requires at least three pre-intervention checkpoints."
            )
        if not (0.0 < self.high_fc_quantile < 1.0):
            raise ValueError("high_fc_quantile must lie strictly between 0 and 1.")
        if not (0.0 < self.low_abs_fc_quantile < 1.0):
            raise ValueError("low_abs_fc_quantile must lie strictly between 0 and 1.")
        if not (
            0.0
            <= self.selection_high_min_fc
            <= self.selection_high_mean_min
            <= 1.0
        ):
            raise ValueError(
                "Require 0 <= selection_high_min_fc <= selection_high_mean_min <= 1."
            )
        if not (
            0.0
            <= self.selection_low_abs_mean_max
            <= self.selection_low_max_abs_fc
            <= 1.0
        ):
            raise ValueError(
                "Require 0 <= selection_low_abs_mean_max <= selection_low_max_abs_fc <= 1."
            )
        if self.selection_max_fc_sd < 0 or self.stability_penalty < 0:
            raise ValueError("Selection FC s.d. and stability penalty must be non-negative.")
        if not (0.0 <= self.min_unit_active_fraction <= 1.0):
            raise ValueError("min_unit_active_fraction must lie in [0, 1].")
        if not (0.0 <= self.min_output_weight_quantile < 1.0):
            raise ValueError("min_output_weight_quantile must lie in [0, 1).")
        if self.fc_target_tolerance <= 0:
            raise ValueError("fc_target_tolerance must be positive.")
        if self.immediate_post_window < 1:
            raise ValueError("immediate_post_window must be positive.")
        if self.intervention_end + self.immediate_post_window > self.total_steps:
            raise ValueError("Immediate post-intervention window exceeds total_steps.")
        if not (-1.0 <= self.target_sample_mapping_corr <= 1.0):
            raise ValueError("target_sample_mapping_corr must lie in [-1, 1].")
        unknown = set(self.conditions) - {
            "none", "active_rank", "independent_rank", "corr_preserving_rank"
        }
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
        matched_controls = {"independent_rank", "corr_preserving_rank"}
        if matched_controls.intersection(self.conditions) and "active_rank" not in self.conditions:
            raise ValueError(
                "active_rank is required when running impact-matched rank controls."
            )
        if "active_rank" in self.conditions and "none" not in self.conditions:
            raise ValueError(
                "none is required because active FC targets use the matched untreated trajectory."
            )


def paper_config(base_dir: Path | str = ".") -> ExperimentConfig:
    base = Path(base_dir)
    return ExperimentConfig(
        data_root=str(base / "data"),
        output_dir=str(base / "results" / "supp_fig4_fc_perturbation_multiseed"),
        figure_dir=str(base / "fig" / "supp_fig4_fc_perturbation_multiseed"),
    )


def smoke_config(base_dir: Path | str = ".") -> ExperimentConfig:
    """Fast structural test.  Results are not suitable for manuscript inference."""

    base = Path(base_dir)
    return replace(
        paper_config(base_dir),
        seeds=(0, 1),
        total_steps=80,
        intervention_start=20,
        intervention_end=50,
        selection_start=5,
        selection_every=7,
        selection_high_mean_min=0.10,
        selection_high_min_fc=0.00,
        selection_low_abs_mean_max=0.25,
        selection_low_max_abs_fc=0.35,
        selection_max_fc_sd=0.25,
        record_every=2,
        n_pairs_per_stratum=2,
        selection_samples=500,
        evaluation_samples=500,
        candidate_draws=2,
        strength_grid=(0.35, 0.75, 1.5, 3.0),
        immediate_post_window=20,
        ci_bootstrap_reps=1000,
        output_dir=str(base / "results" / "supp_fig4_fc_perturbation_multiseed_smoke"),
        figure_dir=str(base / "fig" / "supp_fig4_fc_perturbation_multiseed_smoke"),
    )


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older torch versions do not expose warn_only.
        torch.use_deterministic_algorithms(True)


def derived_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


class MLP(nn.Module):
    """784-100-100-10 ReLU MLP matching the manuscript analysis."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: Sequence[int] = (100, 100),
        num_classes: int = 10,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        dims = [input_size, *hidden_dims, num_classes]
        self.linear_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        for layer in self.linear_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=init_std)
            nn.init.zeros_(layer.bias)

    def hidden2(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        h1 = torch.relu(self.linear_layers[0](x))
        return torch.relu(self.linear_layers[1](h1))

    def forward_with_intervention(
        self,
        x: torch.Tensor,
        controller: Optional["PairIntervention"] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, float]]:
        h2_before = self.hidden2(x)
        if controller is None:
            h2_after = h2_before
            diagnostic = null_diagnostic(h2_before, controller_condition="none")
        else:
            h2_after, diagnostic = controller.apply(
                h2_before,
                self.linear_layers[2].weight,
                self.linear_layers[2].bias,
                int(step or 0),
            )
        logits_before = self.linear_layers[2](h2_before)
        logits_after = self.linear_layers[2](h2_after)
        return logits_after, logits_before, h2_before, h2_after, diagnostic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layers[2](self.hidden2(x))


@dataclass
class MNISTTensors:
    train_images: torch.Tensor
    train_targets: torch.Tensor
    test_images: torch.Tensor
    test_targets: torch.Tensor
    train_pool_indices: np.ndarray
    selection_indices: np.ndarray


def _balanced_reserve_indices(targets: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_class = n_total // 10
    remainder = n_total % 10
    chosen: List[int] = []
    for label in range(10):
        count = per_class + int(label < remainder)
        candidates = np.flatnonzero(targets == label)
        chosen.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def load_mnist_tensors(config: ExperimentConfig, download: bool = True) -> MNISTTensors:
    """Load raw MNIST tensors and reserve a balanced pair-selection split.

    The reserved split is excluded from the SGD schedule, so pair selection does
    not use the same observations that drive the subsequent weight updates.
    """

    train = MNIST(root=config.data_root, train=True, download=download)
    test = MNIST(root=config.data_root, train=False, download=download)
    train_images = train.data.contiguous()
    train_targets = train.targets.contiguous()
    test_images = test.data.contiguous()
    test_targets = test.targets.contiguous()
    selection = _balanced_reserve_indices(
        train_targets.numpy(), config.selection_samples, config.split_seed
    )
    keep = np.ones(len(train_targets), dtype=bool)
    keep[selection] = False
    pool = np.flatnonzero(keep).astype(np.int64)
    return MNISTTensors(
        train_images=train_images,
        train_targets=train_targets,
        test_images=test_images,
        test_targets=test_targets,
        train_pool_indices=pool,
        selection_indices=selection,
    )


def normalize_mnist(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    return ((images.to(device=device, dtype=torch.float32) / 255.0) - 0.1307) / 0.3081


def tensor_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    indices: np.ndarray | Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    x = normalize_mnist(images.index_select(0, idx), device).view(len(idx), -1)
    y = targets.index_select(0, idx).to(device=device, dtype=torch.long)
    return x, y


def make_batch_schedule(
    pool_indices: np.ndarray,
    total_steps: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Deterministic epoch-wise shuffled schedule reused by all branches in a seed."""

    rng = np.random.default_rng(seed)
    batches: List[np.ndarray] = []
    while len(batches) < total_steps:
        order = rng.permutation(pool_indices)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batches.append(order[start : start + batch_size])
            if len(batches) == total_steps:
                break
    return np.stack(batches, axis=0)


def safe_corr_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().double().flatten()
    y = b.detach().double().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if not torch.isfinite(denom) or denom.item() <= EPS:
        return 0.0
    value = torch.dot(x, y) / denom
    return float(torch.clamp(value, -1.0, 1.0).item())


def safe_corr_numpy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.nanstd(a) <= EPS or np.nanstd(b) <= EPS:
        return 0.0
    return float(np.nan_to_num(np.corrcoef(a, b)[0, 1]))


def pair_is(model: MLP, pair: Tuple[int, int]) -> float:
    weight = model.linear_layers[1].weight.detach().cpu().numpy()
    return safe_corr_numpy(weight[pair[0]], weight[pair[1]])


def _rank_zscore(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x.detach())
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(x), device=x.device, dtype=torch.float32)
    ranks = ranks - ranks.mean()
    return ranks / ranks.std(unbiased=False).clamp_min(1e-6)


def rank_remap(original: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Assign the exact sorted values of ``original`` to the rank order of ``score``."""

    target_order = torch.argsort(score.detach())
    sorted_values, _ = torch.sort(original)
    remapped = torch.empty_like(original)
    remapped[target_order] = sorted_values
    return remapped


def _cpu_randn(n: int, generator: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    return torch.randn(n, generator=generator, dtype=torch.float32).to(
        device=like.device, dtype=like.dtype
    )


def rank_noise_proposal(
    a: torch.Tensor,
    b: torch.Tensor,
    strength: float,
    noise_corr: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dependence perturbation with exact empirical marginal preservation.

    ``noise_corr=1`` supplies shared rank noise, ``0`` independent rank noise and
    ``-1`` anti-shared rank noise.  Original activations remain the value pool;
    only their sample assignments change.
    """

    rho = float(np.clip(noise_corr, -1.0, 1.0))
    z1 = _cpu_randn(len(a), generator, a)
    z2 = _cpu_randn(len(a), generator, a)
    eps_a = z1
    eps_b = rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2
    score_a = _rank_zscore(a) + float(strength) * eps_a
    score_b = _rank_zscore(b) + float(strength) * eps_b
    return rank_remap(a, score_a), rank_remap(b, score_b)


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.detach().double() ** 2) + EPS).item())


def _rowwise_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation for corresponding rows without CPU synchronization."""

    x = x.float()
    y = y.float()
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    numerator = torch.sum(x_centered * y_centered, dim=1)
    denominator = torch.linalg.vector_norm(x_centered, dim=1) * torch.linalg.vector_norm(
        y_centered, dim=1
    )
    corr = numerator / denominator.clamp_min(EPS)
    return torch.where(
        denominator > EPS,
        corr.clamp(-1.0, 1.0),
        torch.zeros_like(corr),
    )


def _rank_remap_from_order(original: torch.Tensor, target_order: torch.Tensor) -> torch.Tensor:
    """Differentiably assign sorted original values to a preselected sample order."""

    sorted_values, _ = torch.sort(original)
    remapped = torch.empty_like(original)
    remapped[target_order] = sorted_values
    return remapped


def proposal_diagnostic(
    h2: torch.Tensor,
    proposed_pair: torch.Tensor,
    pair: Tuple[int, int],
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    condition: str,
) -> Dict[str, float]:
    original_pair = h2[:, [pair[0], pair[1]]]
    delta_pair = proposed_pair - original_pair
    base_logits = h2.detach() @ output_weight.detach().T + output_bias.detach()
    delta_logits = delta_pair.detach() @ output_weight.detach()[:, [pair[0], pair[1]]].T

    means_before = original_pair.detach().double().mean(0)
    means_after = proposed_pair.detach().double().mean(0)
    std_before = original_pair.detach().double().std(0, unbiased=False)
    std_after = proposed_pair.detach().double().std(0, unbiased=False)
    sorted_before, _ = torch.sort(original_pair.detach().double(), dim=0)
    sorted_after, _ = torch.sort(proposed_pair.detach().double(), dim=0)

    diag = {
        "condition": condition,
        "fc_before": safe_corr_torch(original_pair[:, 0], original_pair[:, 1]),
        "fc_after": safe_corr_torch(proposed_pair[:, 0], proposed_pair[:, 1]),
        "mean_abs_delta": float(torch.mean(torch.abs(means_after - means_before)).item()),
        "std_relative_error": float(
            torch.mean(torch.abs(std_after - std_before) / std_before.clamp_min(1e-8)).item()
        ),
        "marginal_sorted_rmse": float(
            torch.sqrt(torch.mean((sorted_after - sorted_before) ** 2)).item()
        ),
        "activation_delta_rms_ratio": _rms(delta_pair) / max(_rms(original_pair), EPS),
        "logit_delta_rms_ratio": _rms(delta_logits) / max(_rms(base_logits), EPS),
        "sample_mapping_corr_a": safe_corr_torch(original_pair[:, 0], proposed_pair[:, 0]),
        "sample_mapping_corr_b": safe_corr_torch(original_pair[:, 1], proposed_pair[:, 1]),
    }
    diag["fc_shift"] = diag["fc_after"] - diag["fc_before"]
    return diag


def absolute_state_diagnostic(
    h2_before: torch.Tensor,
    h2_after: torch.Tensor,
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    pair: Tuple[int, int],
) -> Dict[str, float]:
    """Absolute before/after quantities for direct manipulation QC.

    These values complement change scores: they show whether the intervention
    materially alters activation location/scale or the overall logit scale.
    """

    before_pair = h2_before[:, [pair[0], pair[1]]].detach().double()
    after_pair = h2_after[:, [pair[0], pair[1]]].detach().double()
    return {
        "activation_mean_before": float(before_pair.mean().item()),
        "activation_mean_after": float(after_pair.mean().item()),
        "activation_sd_before": float(
            before_pair.std(dim=0, unbiased=False).mean().item()
        ),
        "activation_sd_after": float(
            after_pair.std(dim=0, unbiased=False).mean().item()
        ),
        "activation_rms_before": _rms(before_pair),
        "activation_rms_after": _rms(after_pair),
        "logit_rms_before": _rms(logits_before),
        "logit_rms_after": _rms(logits_after),
    }


def null_diagnostic(h2: torch.Tensor, controller_condition: str = "none") -> Dict[str, float]:
    return {
        "condition": controller_condition,
        "fc_before": np.nan,
        "fc_after": np.nan,
        "fc_shift": 0.0,
        "mean_abs_delta": 0.0,
        "std_relative_error": 0.0,
        "marginal_sorted_rmse": 0.0,
        "activation_delta_rms_ratio": 0.0,
        "logit_delta_rms_ratio": 0.0,
        "sample_mapping_corr_a": 1.0,
        "sample_mapping_corr_b": 1.0,
        "selected_strength": 0.0,
        "selected_noise_corr": 0.0,
        "target_fc": np.nan,
        "untreated_reference_fc": np.nan,
        "fc_vs_untreated": np.nan,
        "target_activation_delta_rms_ratio": np.nan,
        "target_logit_delta_rms_ratio": np.nan,
        "target_sample_mapping_corr": np.nan,
    }


class PairIntervention:
    """Select an impact-matched proposal for one pair on each intervention batch."""

    def __init__(
        self,
        pair: Tuple[int, int],
        stratum: str,
        condition: str,
        config: ExperimentConfig,
        run_seed: int,
        match_targets: Optional[Mapping[int, Mapping[str, float]]] = None,
        untreated_fc_reference: Optional[Mapping[int, float]] = None,
    ) -> None:
        if stratum not in {"high", "low"}:
            raise ValueError("stratum must be 'high' or 'low'.")
        self.pair = tuple(map(int, pair))
        self.stratum = stratum
        self.condition = condition
        self.config = config
        self.match_targets = match_targets or {}
        self.untreated_fc_reference = untreated_fc_reference or {}
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(run_seed))

    @property
    def expected_sign(self) -> int:
        return -1 if self.stratum == "high" else 1

    def active(self, step: int) -> bool:
        return self.config.intervention_start <= step <= self.config.intervention_end

    def _vectorized_candidate_search(
        self,
        h2: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor,
        rho_candidates: Sequence[float],
        target_fc: Optional[float],
        impact_targets: Mapping[str, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
        """Score all rank-remap proposals in one GPU batch.

        Candidate ranking uses float32 tensors and performs a single CPU
        synchronization for the winning index.  The winning proposal is then
        reconstructed from the original activations and receives the full
        double-precision diagnostic in ``apply``.
        """

        cfg = self.config
        candidate_specs = [
            (float(strength), float(rho))
            for strength in cfg.strength_grid
            for rho in rho_candidates
            for _ in range(cfg.candidate_draws)
        ]
        if not candidate_specs:
            raise RuntimeError("No perturbation candidate was configured.")

        target_logit = float(
            impact_targets.get("logit_delta_rms_ratio", cfg.target_logit_rms_ratio)
        )
        target_activation = float(
            impact_targets.get(
                "activation_delta_rms_ratio", cfg.target_activation_delta_ratio
            )
        )
        target_mapping = float(
            impact_targets.get("sample_mapping_corr", cfg.target_sample_mapping_corr)
        )
        impact_scale = max(target_logit, 0.01)
        activation_scale = max(target_activation, 0.05)
        mapping_scale = max(1.0 - target_mapping, 0.10)

        with torch.no_grad():
            device = a.device
            n_candidates = len(candidate_specs)
            n_samples = len(a)
            strengths = torch.tensor(
                [spec[0] for spec in candidate_specs], dtype=torch.float32, device=device
            )
            rhos = torch.tensor(
                [spec[1] for spec in candidate_specs], dtype=torch.float32, device=device
            )
            z1 = torch.randn(
                (n_candidates, n_samples), generator=self.generator, dtype=torch.float32
            ).to(device)
            rank_a = _rank_zscore(a).float()
            rank_b = _rank_zscore(b).float()
            # Leave neuron A unchanged.  Only B is rank-transported toward
            # (rho=+1), away from (rho=-1), or independently of (rho=0) A.
            # This preserves both empirical marginals while avoiding the
            # unnecessary two-neuron sample remapping used by the first version.
            driver_b = rhos[:, None] * rank_a[None, :] + torch.sqrt(
                torch.clamp(1.0 - rhos * rhos, min=0.0)
            )[:, None] * z1
            score_b = rank_b[None, :] + strengths[:, None] * driver_b
            original_order_a = torch.argsort(a.detach().float())
            order_a = original_order_a[None, :].expand(n_candidates, -1)
            order_b = torch.argsort(score_b, dim=1)

            a_detached = a.detach().float()
            b_detached = b.detach().float()
            sorted_b = torch.sort(b_detached).values.expand(n_candidates, -1)
            proposed_a = a_detached[None, :].expand(n_candidates, -1)
            proposed_b = torch.empty_like(score_b).scatter(1, order_b, sorted_b)

            fc_after = _rowwise_corr(proposed_a, proposed_b)
            mapping_corr = 0.5 * (
                _rowwise_corr(proposed_a, a_detached[None, :].expand_as(proposed_a))
                + _rowwise_corr(proposed_b, b_detached[None, :].expand_as(proposed_b))
            )
            original_pair = torch.stack([a_detached, b_detached], dim=1)
            delta_pair = torch.stack(
                [proposed_a - a_detached[None, :], proposed_b - b_detached[None, :]],
                dim=2,
            )
            original_rms = torch.sqrt(torch.mean(original_pair * original_pair) + EPS)
            activation_impact = torch.sqrt(
                torch.mean(delta_pair * delta_pair, dim=(1, 2)) + EPS
            ) / original_rms.clamp_min(EPS)

            base_logits = h2.detach().float() @ output_weight.detach().float().T
            base_logits = base_logits + output_bias.detach().float()
            pair_output_weight = output_weight.detach().float()[:, list(self.pair)]
            delta_logits = torch.matmul(delta_pair, pair_output_weight.T)
            base_logit_rms = torch.sqrt(torch.mean(base_logits * base_logits) + EPS)
            logit_impact = torch.sqrt(
                torch.mean(delta_logits * delta_logits, dim=(1, 2)) + EPS
            ) / base_logit_rms.clamp_min(EPS)

            impact_objective = torch.abs(logit_impact - target_logit) / impact_scale
            impact_objective = impact_objective + 0.20 * torch.abs(
                activation_impact - target_activation
            ) / activation_scale
            impact_objective = impact_objective + 0.20 * torch.abs(
                mapping_corr - target_mapping
            ) / mapping_scale
            if target_fc is not None:
                # FC is the manipulated variable, so it is a feasibility gate
                # rather than one interchangeable term in a soft objective.
                fc_error = torch.abs(fc_after - target_fc)
                feasible = fc_error <= cfg.fc_target_tolerance
                if bool(torch.any(feasible).item()):
                    objective = torch.where(
                        feasible,
                        impact_objective,
                        torch.full_like(impact_objective, torch.inf),
                    )
                else:
                    objective = (
                        fc_error / cfg.fc_target_tolerance
                        + 0.05 * impact_objective
                    )
            else:
                objective = impact_objective

            best_idx = int(torch.argmin(objective).item())
            best_order_a = order_a[best_idx].detach()
            best_order_b = order_b[best_idx].detach()
            best_strength, best_rho = candidate_specs[best_idx]
            best_score = float(objective[best_idx].item())
        return best_order_a, best_order_b, best_strength, best_rho, best_score

    def apply(
        self,
        h2: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Mapping[str, float]]:
        if self.condition == "none" or not self.active(step):
            diag = null_diagnostic(h2, controller_condition=self.condition)
            pair_values = h2[:, [self.pair[0], self.pair[1]]]
            diag["fc_before"] = safe_corr_torch(pair_values[:, 0], pair_values[:, 1])
            diag["fc_after"] = diag["fc_before"]
            return h2, diag
        pair = self.pair
        a = h2[:, pair[0]]
        b = h2[:, pair[1]]
        corr_before = safe_corr_torch(a, b)
        target_fc: Optional[float]
        if self.condition == "active_rank":
            if int(step) not in self.untreated_fc_reference:
                raise RuntimeError(
                    f"Missing untreated FC reference for active step {step}."
                )
            untreated_fc = float(self.untreated_fc_reference[int(step)])
            target_fc = float(
                np.clip(
                    untreated_fc
                    + self.expected_sign * self.config.target_fc_shift,
                    -0.95,
                    0.95,
                )
            )
            rho_candidates = (float(self.expected_sign),)
        elif self.condition == "independent_rank":
            target_fc = None
            rho_candidates = (0.0,)
        elif self.condition == "corr_preserving_rank":
            target_fc = corr_before
            rho_candidates = self.config.rho_grid
        else:
            raise ValueError(f"Unsupported intervention condition: {self.condition}")

        impact_targets: Dict[str, float] = {
            "activation_delta_rms_ratio": self.config.target_activation_delta_ratio,
            "logit_delta_rms_ratio": self.config.target_logit_rms_ratio,
            "sample_mapping_corr": self.config.target_sample_mapping_corr,
        }
        if self.condition in {"independent_rank", "corr_preserving_rank"}:
            impact_targets.update(self.match_targets.get(int(step), {}))

        order_a, order_b, best_strength, best_rho, best_score = (
            self._vectorized_candidate_search(
                h2=h2,
                a=a,
                b=b,
                output_weight=output_weight,
                output_bias=output_bias,
                rho_candidates=rho_candidates,
                target_fc=target_fc,
                impact_targets=impact_targets,
            )
        )
        best_pair = torch.stack(
            [
                _rank_remap_from_order(a, order_a),
                _rank_remap_from_order(b, order_b),
            ],
            dim=1,
        )
        best_diag = proposal_diagnostic(
            h2, best_pair, pair, output_weight, output_bias, self.condition
        )
        best_diag["selected_strength"] = best_strength
        best_diag["selected_noise_corr"] = best_rho
        best_diag["target_fc"] = np.nan if target_fc is None else target_fc
        best_diag["untreated_reference_fc"] = (
            float(self.untreated_fc_reference[int(step)])
            if int(step) in self.untreated_fc_reference
            else np.nan
        )
        best_diag["fc_vs_untreated"] = (
            best_diag["fc_after"] - best_diag["untreated_reference_fc"]
            if np.isfinite(best_diag["untreated_reference_fc"])
            else np.nan
        )
        best_diag["target_activation_delta_rms_ratio"] = impact_targets[
            "activation_delta_rms_ratio"
        ]
        best_diag["target_logit_delta_rms_ratio"] = impact_targets[
            "logit_delta_rms_ratio"
        ]
        best_diag["target_sample_mapping_corr"] = impact_targets[
            "sample_mapping_corr"
        ]
        best_diag["selection_objective"] = best_score
        proposed = h2.clone()
        proposed[:, pair[0]] = best_pair[:, 0]
        proposed[:, pair[1]] = best_pair[:, 1]
        return proposed, best_diag


def _selection_activations(
    model: MLP,
    data: MNISTTensors,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data.selection_indices), config.batch_size):
            idx = data.selection_indices[start : start + config.batch_size]
            x, _ = tensor_batch(data.train_images, data.train_targets, idx, device)
            chunks.append(model.hidden2(x).cpu().numpy())
    activations = np.concatenate(chunks, axis=0)
    if was_training:
        model.train()
    return activations


def _selection_steps(config: ExperimentConfig) -> Tuple[int, ...]:
    """Repeated held-out checkpoints available strictly before intervention."""

    final_pre_step = config.intervention_start - 1
    steps = list(
        range(
            config.selection_start,
            final_pre_step + 1,
            config.selection_every,
        )
    )
    if not steps or steps[-1] != final_pre_step:
        steps.append(final_pre_step)
    return tuple(steps)


def _selection_fc_matrix(
    model: MLP,
    data: MNISTTensors,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    activations = _selection_activations(model, data, config, device)
    unit_std = np.nanstd(activations, axis=0)
    active_fraction = np.mean(activations > 0, axis=0)
    output_norm = torch.linalg.vector_norm(
        model.linear_layers[2].weight.detach(), dim=0
    ).cpu().numpy()
    output_floor = float(np.quantile(output_norm, config.min_output_weight_quantile))
    valid = (
        (unit_std > config.min_unit_std)
        & (active_fraction >= config.min_unit_active_fraction)
        & (output_norm >= output_floor)
    )
    fc = np.full((activations.shape[1], activations.shape[1]), np.nan, dtype=float)
    nonconstant = unit_std > config.min_unit_std
    if np.count_nonzero(nonconstant) >= 2:
        fc_valid = np.corrcoef(activations[:, nonconstant], rowvar=False)
        valid_indices = np.flatnonzero(nonconstant)
        fc[np.ix_(valid_indices, valid_indices)] = fc_valid
    return fc, valid


def _choose_ranked_disjoint(
    candidates: List[Dict[str, object]],
    n: int,
    used: set,
) -> List[Dict[str, object]]:
    """Take the best pre-ranked candidates while enforcing node disjointness."""

    chosen: List[Dict[str, object]] = []
    for candidate in candidates:
        i = int(candidate["unit_a"])
        j = int(candidate["unit_b"])
        if i in used or j in used:
            continue
        chosen.append(candidate)
        used.update((i, j))
        if len(chosen) == n:
            break
    return chosen


def _choose_joint_ranked_disjoint(
    high_candidates: List[Dict[str, object]],
    low_candidates: List[Dict[str, object]],
    n: int,
) -> Optional[Tuple[List[Dict[str, object]], List[Dict[str, object]]]]:
    """Find the first ranked high-pair set that permits a disjoint low set."""

    def search_high(
        start: int,
        selected: List[Dict[str, object]],
        used: set,
    ) -> Optional[Tuple[List[Dict[str, object]], List[Dict[str, object]]]]:
        if len(selected) == n:
            low = _choose_ranked_disjoint(low_candidates, n, set(used))
            if len(low) == n:
                return list(selected), low
            return None
        still_needed = n - len(selected)
        final_start = len(high_candidates) - still_needed
        for candidate_idx in range(start, final_start + 1):
            candidate = high_candidates[candidate_idx]
            i = int(candidate["unit_a"])
            j = int(candidate["unit_b"])
            if i in used or j in used:
                continue
            result = search_high(
                candidate_idx + 1,
                selected + [candidate],
                used | {i, j},
            )
            if result is not None:
                return result
        return None

    return search_high(0, [], set())


def select_fc_pairs(
    fc_by_step: Mapping[int, np.ndarray],
    valid_by_step: Mapping[int, np.ndarray],
    config: ExperimentConfig,
    seed: int,
) -> pd.DataFrame:
    """Select pairs from stable pre-intervention FC trajectories.

    Pair identity uses only repeated held-out measurements available before the
    branch point.  Intervention outcomes, future FC and IS are never used.
    """

    checkpoints = tuple(sorted(fc_by_step))
    expected = _selection_steps(config)
    if checkpoints != expected or tuple(sorted(valid_by_step)) != expected:
        raise ValueError(
            f"Expected pre-intervention FC checkpoints {expected}, received {checkpoints}."
        )
    first_fc = fc_by_step[checkpoints[0]]
    upper_i, upper_j = np.triu_indices(first_fc.shape[0], k=1)
    valid_pairs = np.ones(len(upper_i), dtype=bool)
    trajectories = []
    for step in checkpoints:
        fc = np.asarray(fc_by_step[step], dtype=float)
        valid_units = np.asarray(valid_by_step[step], dtype=bool)
        values = fc[upper_i, upper_j]
        valid_pairs &= (
            valid_units[upper_i]
            & valid_units[upper_j]
            & np.isfinite(values)
        )
        trajectories.append(values)
    trajectory = np.stack(trajectories, axis=1)[valid_pairs]
    ii = upper_i[valid_pairs]
    jj = upper_j[valid_pairs]
    if not len(trajectory):
        raise RuntimeError(f"No valid pre-intervention FC trajectories for seed {seed}.")

    selection_mean = trajectory.mean(axis=1)
    selection_sd = trajectory.std(axis=1, ddof=0)
    selection_min = trajectory.min(axis=1)
    selection_max = trajectory.max(axis=1)
    selection_max_abs = np.abs(trajectory).max(axis=1)
    high_threshold = max(
        float(np.quantile(selection_mean, config.high_fc_quantile)),
        config.selection_high_mean_min,
    )
    low_threshold = min(
        float(np.quantile(np.abs(selection_mean), config.low_abs_fc_quantile)),
        config.selection_low_abs_mean_max,
    )

    candidates: List[Dict[str, object]] = []
    for row_idx, (i, j) in enumerate(zip(ii, jj)):
        row: Dict[str, object] = {
            "unit_a": int(i),
            "unit_b": int(j),
            "selection_mean_fc": float(selection_mean[row_idx]),
            "selection_sd_fc": float(selection_sd[row_idx]),
            "selection_min_fc": float(selection_min[row_idx]),
            "selection_max_fc": float(selection_max[row_idx]),
            "selection_max_abs_fc": float(selection_max_abs[row_idx]),
        }
        row.update(
            {
                f"fc_step_{step}": float(trajectory[row_idx, step_idx])
                for step_idx, step in enumerate(checkpoints)
            }
        )
        candidates.append(row)

    high_candidates = [
        row
        for row in candidates
        if float(row["selection_mean_fc"]) >= high_threshold
        and float(row["selection_sd_fc"]) <= config.selection_max_fc_sd
        and float(row["selection_min_fc"]) >= config.selection_high_min_fc
    ]
    low_candidates = [
        row
        for row in candidates
        if abs(float(row["selection_mean_fc"])) <= low_threshold
        and float(row["selection_sd_fc"]) <= config.selection_max_fc_sd
        and float(row["selection_max_abs_fc"]) <= config.selection_low_max_abs_fc
    ]
    high_candidates.sort(
        key=lambda row: (
            -(
                float(row["selection_mean_fc"])
                - config.stability_penalty * float(row["selection_sd_fc"])
            ),
            int(row["unit_a"]),
            int(row["unit_b"]),
        )
    )
    low_candidates.sort(
        key=lambda row: (
            abs(float(row["selection_mean_fc"]))
            + config.stability_penalty * float(row["selection_sd_fc"]),
            int(row["unit_a"]),
            int(row["unit_b"]),
        )
    )

    joint_selection = _choose_joint_ranked_disjoint(
        high_candidates,
        low_candidates,
        config.n_pairs_per_stratum,
    )
    if joint_selection is None:
        used: set = set()
        high = _choose_ranked_disjoint(
            high_candidates, config.n_pairs_per_stratum, used
        )
        low = _choose_ranked_disjoint(
            low_candidates, config.n_pairs_per_stratum, used
        )
        raise RuntimeError(
            f"Seed {seed}: pre-intervention FC rules yielded {len(high_candidates)} high "
            f"and {len(low_candidates)} low candidates, but only {len(high)} high/"
            f"{len(low)} low node-disjoint pairs could be selected. Thresholds were "
            f"high mean >= {high_threshold:.3f}, high minimum >= "
            f"{config.selection_high_min_fc:.3f}, low |mean| <= {low_threshold:.3f}, "
            f"low max |FC| <= {config.selection_low_max_abs_fc:.3f}, and FC s.d. <= "
            f"{config.selection_max_fc_sd:.3f}; units also required active fraction >= "
            f"{config.min_unit_active_fraction:.3f} and output-weight norm above quantile "
            f"{config.min_output_weight_quantile:.2f}. Inspect the untreated scout trajectories "
            "and explicitly revise the predeclared thresholds if scientifically justified."
        )
    high, low = joint_selection
    rows = []
    for stratum, selected in (("high", high), ("low", low)):
        for rank, candidate in enumerate(selected):
            i = int(candidate["unit_a"])
            j = int(candidate["unit_b"])
            rows.append(
                {
                    "seed": seed,
                    "stratum": stratum,
                    "pair_rank": rank,
                    **candidate,
                    "selection_fc": float(candidate["selection_mean_fc"]),
                    "high_threshold": high_threshold,
                    "low_abs_threshold": low_threshold,
                    "selection_start_step": config.selection_start,
                    "selection_end_step": config.intervention_start - 1,
                    "selection_n_checkpoints": len(checkpoints),
                    "selection_source": "untreated_scout_heldout_preintervention",
                    "pair_id": f"s{seed}_{stratum}_{rank}_{i}_{j}",
                }
            )
    return pd.DataFrame(rows)


def _scout_and_checkpoint(
    seed: int,
    data: MNISTTensors,
    schedule: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
]:
    """Train an untreated reference and retain the exact pre-branch state.

    The same initialization and deterministic minibatch schedule are used by
    the scout and every intervention branch.  Selection is completed using
    held-out data by ``intervention_start - 1``, and all branches restore that
    exact state so neuron indices cannot drift.
    """

    set_seed(seed)
    model = MLP(hidden_dims=config.hidden_dims).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    warm_state: Optional[Dict[str, torch.Tensor]] = None
    if config.intervention_start == 1:
        warm_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    selection_steps = set(_selection_steps(config))
    fc_by_step: Dict[int, np.ndarray] = {}
    valid_by_step: Dict[int, np.ndarray] = {}
    model.train()
    for step in range(1, config.intervention_start):
        x, y = tensor_batch(
            data.train_images, data.train_targets, schedule[step - 1], device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if step == config.intervention_start - 1:
            warm_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        if step in selection_steps:
            fc, valid = _selection_fc_matrix(model, data, config, device)
            fc_by_step[step] = fc
            valid_by_step[step] = valid
            model.train()
    if warm_state is None:
        raise RuntimeError("Pre-intervention scout checkpoint was not created.")
    return warm_state, fc_by_step, valid_by_step


def _phase(step: int, config: ExperimentConfig) -> str:
    if step < config.intervention_start:
        return "pre"
    if step <= config.intervention_end:
        return "intervention"
    return "recovery"


def _gradient_metrics(model: MLP, pair: Tuple[int, int]) -> Tuple[float, float, float]:
    grad = model.linear_layers[1].weight.grad
    if grad is None:
        return 0.0, 0.0, 0.0
    ga = grad[pair[0]].detach()
    gb = grad[pair[1]].detach()
    return (
        safe_corr_torch(ga, gb),
        float(torch.linalg.vector_norm(ga).item()),
        float(torch.linalg.vector_norm(gb).item()),
    )


def _evaluate_model(
    model: MLP,
    data: MNISTTensors,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[float, float]:
    n = min(config.evaluation_samples, len(data.test_targets))
    rng = np.random.default_rng(config.split_seed + 1)
    idx = rng.choice(len(data.test_targets), size=n, replace=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, n, config.batch_size):
            batch_idx = idx[start : start + config.batch_size]
            x, y = tensor_batch(data.test_images, data.test_targets, batch_idx, device)
            logits = model(x)
            total_loss += float(criterion(logits, y).item())
            total_correct += int((logits.argmax(1) == y).sum().item())
    return total_loss / n, total_correct / n


def run_branch(
    warm_state: Mapping[str, torch.Tensor],
    seed: int,
    pair_row: Mapping[str, object],
    condition: str,
    data: MNISTTensors,
    schedule: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    match_targets: Optional[Mapping[int, Mapping[str, float]]] = None,
    untreated_fc_reference: Optional[Mapping[int, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pair = (int(pair_row["unit_a"]), int(pair_row["unit_b"]))
    model = MLP(hidden_dims=config.hidden_dims).to(device)
    model.load_state_dict(warm_state)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    controller = PairIntervention(
        pair=pair,
        stratum=str(pair_row["stratum"]),
        condition=condition,
        config=config,
        run_seed=derived_seed(seed, pair_row["pair_id"], condition),
        match_targets=match_targets,
        untreated_fc_reference=untreated_fc_reference,
    )
    rows: List[Dict[str, object]] = []
    initial_is = pair_is(model, pair)
    rows.append(
        {
            "seed": seed,
            "pair_id": pair_row["pair_id"],
            "stratum": pair_row["stratum"],
            "unit_a": pair[0],
            "unit_b": pair[1],
            "condition": condition,
            "step": config.intervention_start - 1,
            "phase": "pre",
            "is_value": initial_is,
            "loss_before": np.nan,
            "loss_after": np.nan,
            "loss": np.nan,
            "accuracy_before": np.nan,
            "accuracy_after": np.nan,
            "accuracy": np.nan,
            "instant_loss_delta": 0.0,
            "gradient_corr": np.nan,
            "gradient_norm_a": np.nan,
            "gradient_norm_b": np.nan,
            **null_diagnostic(torch.empty(1), controller_condition=condition),
        }
    )

    model.train()
    for step in range(config.intervention_start, config.total_steps + 1):
        x, y = tensor_batch(
            data.train_images, data.train_targets, schedule[step - 1], device
        )
        optimizer.zero_grad(set_to_none=True)
        logits, logits_before, h2_before, h2_after, diag = model.forward_with_intervention(
            x, controller=controller, step=step
        )
        diag = dict(diag)
        diag.update(
            absolute_state_diagnostic(
                h2_before, h2_after, logits_before, logits, pair
            )
        )
        loss = criterion(logits, y)
        with torch.no_grad():
            base_loss = criterion(logits_before, y)
            instant_loss_delta = float((loss.detach() - base_loss).item())
            base_accuracy = float(
                (logits_before.argmax(1) == y).float().mean().item()
            )
            accuracy = float((logits.argmax(1) == y).float().mean().item())
        loss.backward()
        grad_corr, grad_norm_a, grad_norm_b = _gradient_metrics(model, pair)
        optimizer.step()
        should_record = (
            controller.active(step)
            or step % config.record_every == 0
            or step in {
                config.intervention_start,
                config.intervention_end,
                config.intervention_end + 1,
                config.total_steps,
            }
        )
        if should_record:
            rows.append(
                {
                    "seed": seed,
                    "pair_id": pair_row["pair_id"],
                    "stratum": pair_row["stratum"],
                    "unit_a": pair[0],
                    "unit_b": pair[1],
                    "condition": condition,
                    "step": step,
                    "phase": _phase(step, config),
                    "is_value": pair_is(model, pair),
                    "loss_before": float(base_loss.detach().item()),
                    "loss_after": float(loss.detach().item()),
                    "loss": float(loss.detach().item()),
                    "accuracy_before": base_accuracy,
                    "accuracy_after": accuracy,
                    "accuracy": accuracy,
                    "instant_loss_delta": instant_loss_delta,
                    "gradient_corr": grad_corr,
                    "gradient_norm_a": grad_norm_a,
                    "gradient_norm_b": grad_norm_b,
                    **dict(diag),
                }
            )

    final_test_loss, final_test_accuracy = _evaluate_model(model, data, config, device)
    final = {
        "seed": seed,
        "pair_id": pair_row["pair_id"],
        "stratum": pair_row["stratum"],
        "unit_a": pair[0],
        "unit_b": pair[1],
        "condition": condition,
        "selection_fc": float(pair_row["selection_fc"]),
        "initial_is": initial_is,
        "final_is": pair_is(model, pair),
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
    }
    return pd.DataFrame(rows), final


def run_shared_untreated_branch(
    warm_state: Mapping[str, torch.Tensor],
    seed: int,
    pair_table: pd.DataFrame,
    data: MNISTTensors,
    schedule: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the identical untreated continuation once for every pair in a seed.

    The previous implementation retrained the same untreated network once per
    pair.  With no intervention, pair identity cannot affect the model update,
    so one continuation is sufficient; pair-specific FC, IS and gradient
    diagnostics are recorded from that shared model trajectory.
    """

    pair_rows = pair_table.to_dict(orient="records")
    model = MLP(hidden_dims=config.hidden_dims).to(device)
    model.load_state_dict(warm_state)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    rows: List[Dict[str, object]] = []
    initial_is: Dict[str, float] = {}

    for pair_row in pair_rows:
        pair = (int(pair_row["unit_a"]), int(pair_row["unit_b"]))
        pair_id = str(pair_row["pair_id"])
        initial_is[pair_id] = pair_is(model, pair)
        rows.append(
            {
                "seed": seed,
                "pair_id": pair_id,
                "stratum": pair_row["stratum"],
                "unit_a": pair[0],
                "unit_b": pair[1],
                "condition": "none",
                "step": config.intervention_start - 1,
                "phase": "pre",
                "is_value": initial_is[pair_id],
                "loss_before": np.nan,
                "loss_after": np.nan,
                "loss": np.nan,
                "accuracy_before": np.nan,
                "accuracy_after": np.nan,
                "accuracy": np.nan,
                "instant_loss_delta": 0.0,
                "gradient_corr": np.nan,
                "gradient_norm_a": np.nan,
                "gradient_norm_b": np.nan,
                **null_diagnostic(torch.empty(1), controller_condition="none"),
            }
        )

    model.train()
    for step in range(config.intervention_start, config.total_steps + 1):
        x, y = tensor_batch(
            data.train_images, data.train_targets, schedule[step - 1], device
        )
        optimizer.zero_grad(set_to_none=True)
        h2 = model.hidden2(x)
        logits = model.linear_layers[2](h2)
        loss = criterion(logits, y)
        loss.backward()
        gradient_by_pair = {
            str(pair_row["pair_id"]): _gradient_metrics(
                model,
                (int(pair_row["unit_a"]), int(pair_row["unit_b"])),
            )
            for pair_row in pair_rows
        }
        optimizer.step()

        should_record = (
            config.intervention_start <= step <= config.intervention_end
            or step % config.record_every == 0
            or step
            in {
                config.intervention_start,
                config.intervention_end,
                config.intervention_end + 1,
                config.total_steps,
            }
        )
        if not should_record:
            continue

        with torch.no_grad():
            accuracy = float((logits.argmax(1) == y).float().mean().item())
        for pair_row in pair_rows:
            pair = (int(pair_row["unit_a"]), int(pair_row["unit_b"]))
            pair_id = str(pair_row["pair_id"])
            pair_values = h2[:, [pair[0], pair[1]]]
            diag = null_diagnostic(h2, controller_condition="none")
            diag["fc_before"] = safe_corr_torch(
                pair_values[:, 0], pair_values[:, 1]
            )
            diag["fc_after"] = diag["fc_before"]
            diag.update(
                absolute_state_diagnostic(h2, h2, logits, logits, pair)
            )
            grad_corr, grad_norm_a, grad_norm_b = gradient_by_pair[pair_id]
            rows.append(
                {
                    "seed": seed,
                    "pair_id": pair_id,
                    "stratum": pair_row["stratum"],
                    "unit_a": pair[0],
                    "unit_b": pair[1],
                    "condition": "none",
                    "step": step,
                    "phase": _phase(step, config),
                    "is_value": pair_is(model, pair),
                    "loss_before": float(loss.detach().item()),
                    "loss_after": float(loss.detach().item()),
                    "loss": float(loss.detach().item()),
                    "accuracy_before": accuracy,
                    "accuracy_after": accuracy,
                    "accuracy": accuracy,
                    "instant_loss_delta": 0.0,
                    "gradient_corr": grad_corr,
                    "gradient_norm_a": grad_norm_a,
                    "gradient_norm_b": grad_norm_b,
                    **diag,
                }
            )

    final_test_loss, final_test_accuracy = _evaluate_model(
        model, data, config, device
    )
    final_rows = []
    for pair_row in pair_rows:
        pair = (int(pair_row["unit_a"]), int(pair_row["unit_b"]))
        pair_id = str(pair_row["pair_id"])
        final_rows.append(
            {
                "seed": seed,
                "pair_id": pair_id,
                "stratum": pair_row["stratum"],
                "unit_a": pair[0],
                "unit_b": pair[1],
                "condition": "none",
                "selection_fc": float(pair_row["selection_fc"]),
                "initial_is": initial_is[pair_id],
                "final_is": pair_is(model, pair),
                "final_test_loss": final_test_loss,
                "final_test_accuracy": final_test_accuracy,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(final_rows)


def attach_untreated_reference(step_table: pd.DataFrame) -> pd.DataFrame:
    """Attach matched untreated FC/IS trajectories to every branch row.

    Matching is exact on seed, pair identity and training step.  The resulting
    differences are descriptive trajectories; seed remains the inferential unit.
    """

    required = {"seed", "pair_id", "condition", "step", "fc_after", "is_value"}
    missing = required - set(step_table.columns)
    if missing:
        raise ValueError(f"step_table is missing reference columns: {sorted(missing)}")
    keys = ["seed", "pair_id", "step"]
    untreated = step_table.loc[
        step_table["condition"].eq("none"), keys + ["fc_after", "is_value"]
    ].rename(
        columns={
            "fc_after": "untreated_fc",
            "is_value": "untreated_is",
        }
    )
    if untreated.duplicated(keys).any():
        raise ValueError("Untreated trajectory is not unique within seed/pair/step.")
    untreated["untreated_fc"] = (
        untreated.sort_values(keys)
        .groupby(["seed", "pair_id"])["untreated_fc"]
        .transform(lambda values: values.interpolate(limit_direction="both"))
    )
    enriched = step_table.drop(
        columns=["untreated_fc", "untreated_is"], errors="ignore"
    ).merge(untreated, on=keys, how="left", validate="many_to_one")
    missing_fc_reference = enriched["fc_after"].notna() & enriched["untreated_fc"].isna()
    if missing_fc_reference.any() or enriched["untreated_is"].isna().any():
        raise RuntimeError("At least one branch row lacks a matched untreated reference.")
    enriched["fc_vs_untreated"] = enriched["fc_after"] - enriched["untreated_fc"]
    enriched["is_vs_untreated"] = enriched["is_value"] - enriched["untreated_is"]
    return enriched


def build_untreated_fc_reference(
    step_table: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[int, float]:
    """Return a complete finite intervention-window FC reference."""

    untreated = step_table.loc[
        step_table["condition"].eq("none")
        & step_table["phase"].eq("intervention"),
        ["step", "fc_after"],
    ].sort_values("step")
    expected = np.arange(config.intervention_start, config.intervention_end + 1)
    if untreated["step"].duplicated().any() or set(untreated["step"]) != set(expected):
        raise RuntimeError("Untreated branch did not produce a complete FC reference.")
    values = untreated.set_index("step")["fc_after"].reindex(expected)
    values = values.replace([np.inf, -np.inf], np.nan).interpolate(
        limit_direction="both"
    )
    if values.isna().any():
        raise RuntimeError("Untreated FC reference is undefined throughout the window.")
    return {int(step): float(value) for step, value in values.items()}


def run_experiment(
    config: ExperimentConfig,
    data: Optional[MNISTTensors] = None,
    download: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run untreated scouts, select stable pairs, then run all paired branches."""

    config.validate()
    device = resolve_device(config.device)
    data = data or load_mnist_tensors(config, download=download)
    all_steps: List[pd.DataFrame] = []
    all_pairs: List[pd.DataFrame] = []
    finals: List[Dict[str, object]] = []
    total_runs = len(config.seeds) * 2 * config.n_pairs_per_stratum * len(config.conditions)
    completed = 0
    start_time = time.time()
    for seed in config.seeds:
        schedule = make_batch_schedule(
            data.train_pool_indices,
            config.total_steps,
            config.batch_size,
            derived_seed(seed, "minibatches"),
        )
        print(
            f"[scout] seed={seed} training pre-intervention reference; FC at "
            f"steps {_selection_steps(config)}",
            flush=True,
        )
        warm_state, fc_by_step, valid_by_step = _scout_and_checkpoint(
            seed, data, schedule, config, device
        )
        pair_table = select_fc_pairs(
            fc_by_step, valid_by_step, config, seed
        )
        all_pairs.append(pair_table)
        print(
            f"[scout] seed={seed} selected "
            + ", ".join(
                f"{row.pair_id}: mean={row.selection_mean_fc:.3f}, "
                f"sd={row.selection_sd_fc:.3f}"
                for row in pair_table.itertuples(index=False)
            ),
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for pair_row in pair_table.to_dict(orient="records"):
            preferred_order = ("none", "active_rank")
            ordered_conditions = [c for c in preferred_order if c in config.conditions]
            ordered_conditions.extend(c for c in config.conditions if c not in preferred_order)
            active_match_targets: Dict[int, Dict[str, float]] = {}
            untreated_fc_reference: Dict[int, float] = {}
            for condition in ordered_conditions:
                match_targets = (
                    active_match_targets
                    if condition in {"independent_rank", "corr_preserving_rank"}
                    else None
                )
                step_table, final = run_branch(
                    warm_state,
                    seed,
                    pair_row,
                    condition,
                    data,
                    schedule,
                    config,
                    device,
                    match_targets=match_targets,
                    untreated_fc_reference=(
                        untreated_fc_reference if condition == "active_rank" else None
                    ),
                )
                if condition == "none":
                    untreated_fc_reference = build_untreated_fc_reference(
                        step_table, config
                    )
                if condition == "active_rank":
                    active_rows = step_table.loc[
                        step_table["phase"].eq("intervention")
                    ]
                    active_match_targets = {
                        int(row.step): {
                            "activation_delta_rms_ratio": float(
                                row.activation_delta_rms_ratio
                            ),
                            "logit_delta_rms_ratio": float(row.logit_delta_rms_ratio),
                            "sample_mapping_corr": float(
                                0.5
                                * (
                                    row.sample_mapping_corr_a
                                    + row.sample_mapping_corr_b
                                )
                            ),
                        }
                        for row in active_rows.itertuples(index=False)
                    }
                    expected_steps = set(
                        range(config.intervention_start, config.intervention_end + 1)
                    )
                    if set(active_match_targets) != expected_steps:
                        raise RuntimeError(
                            "Active branch did not produce a complete per-step impact profile."
                        )
                all_steps.append(step_table)
                finals.append(final)
                completed += 1
                elapsed = time.time() - start_time
                eta = elapsed / completed * (total_runs - completed)
                print(
                    f"[{completed:>4}/{total_runs}] seed={seed} pair={pair_row['pair_id']} "
                    f"condition={condition} elapsed={elapsed/60:.1f} min ETA={eta/60:.1f} min",
                    flush=True,
                )
    step_table = attach_untreated_reference(pd.concat(all_steps, ignore_index=True))
    return (
        step_table,
        pd.concat(all_pairs, ignore_index=True),
        pd.DataFrame(finals),
    )


def _window_mean(group: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = group.loc[mask, column].dropna()
    return float(values.mean()) if len(values) else np.nan


def summarize_runs(
    step_table: pd.DataFrame,
    final_table: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    keys = ["seed", "pair_id", "stratum", "unit_a", "unit_b", "condition"]
    rows: List[Dict[str, object]] = []
    for key, group in step_table.groupby(keys, sort=False):
        group = group.sort_values("step").copy()
        group["sample_mapping_corr"] = group[
            ["sample_mapping_corr_a", "sample_mapping_corr_b"]
        ].mean(axis=1)
        group["activation_target_abs_error"] = (
            group["activation_delta_rms_ratio"]
            - group["target_activation_delta_rms_ratio"]
        ).abs()
        group["logit_target_abs_error"] = (
            group["logit_delta_rms_ratio"]
            - group["target_logit_delta_rms_ratio"]
        ).abs()
        group["mapping_target_abs_error"] = (
            group["sample_mapping_corr"] - group["target_sample_mapping_corr"]
        ).abs()
        group["fc_target_abs_error"] = (
            group["fc_after"] - group["target_fc"]
        ).abs()
        active = group["phase"].eq("intervention")
        immediate_post = group["step"].between(
            config.intervention_end + 1,
            config.intervention_end + config.immediate_post_window,
        )
        recovery = group["step"].ge(config.total_steps - config.recovery_window + 1)
        end_window = group["step"].between(
            max(config.intervention_start, config.intervention_end - 20),
            config.intervention_end,
        )
        initial_is = float(group.iloc[0]["is_value"])
        immediate_is = _window_mean(group, immediate_post, "is_value")
        post_is = _window_mean(group, recovery, "is_value")
        row = dict(zip(keys, key))
        row.update(
            initial_is=initial_is,
            end_is=_window_mean(group, end_window, "is_value"),
            immediate_post_is=immediate_is,
            delta_is_immediate=immediate_is - initial_is,
            post_is=post_is,
            delta_is_post=post_is - initial_is,
            mean_fc_before=_window_mean(group, active, "fc_before"),
            mean_fc_after=_window_mean(group, active, "fc_after"),
            mean_fc_shift=_window_mean(group, active, "fc_shift"),
            mean_fc_vs_untreated=_window_mean(group, active, "fc_vs_untreated"),
            mean_fc_target_abs_error=_window_mean(
                group, active, "fc_target_abs_error"
            ),
            mean_abs_delta=_window_mean(group, active, "mean_abs_delta"),
            mean_std_relative_error=_window_mean(group, active, "std_relative_error"),
            mean_marginal_sorted_rmse=_window_mean(group, active, "marginal_sorted_rmse"),
            mean_activation_delta_rms_ratio=_window_mean(
                group, active, "activation_delta_rms_ratio"
            ),
            mean_logit_delta_rms_ratio=_window_mean(
                group, active, "logit_delta_rms_ratio"
            ),
            mean_sample_mapping_corr_a=_window_mean(
                group, active, "sample_mapping_corr_a"
            ),
            mean_sample_mapping_corr_b=_window_mean(
                group, active, "sample_mapping_corr_b"
            ),
            mean_selected_strength=_window_mean(group, active, "selected_strength"),
            mean_selected_noise_corr=_window_mean(
                group, active, "selected_noise_corr"
            ),
            mean_activation_target_abs_error=_window_mean(
                group, active, "activation_target_abs_error"
            ),
            mean_logit_target_abs_error=_window_mean(
                group, active, "logit_target_abs_error"
            ),
            mean_mapping_target_abs_error=_window_mean(
                group, active, "mapping_target_abs_error"
            ),
            mean_activation_mean_before=_window_mean(
                group, active, "activation_mean_before"
            ),
            mean_activation_mean_after=_window_mean(
                group, active, "activation_mean_after"
            ),
            mean_activation_sd_before=_window_mean(
                group, active, "activation_sd_before"
            ),
            mean_activation_sd_after=_window_mean(
                group, active, "activation_sd_after"
            ),
            mean_activation_rms_before=_window_mean(
                group, active, "activation_rms_before"
            ),
            mean_activation_rms_after=_window_mean(
                group, active, "activation_rms_after"
            ),
            mean_logit_rms_before=_window_mean(
                group, active, "logit_rms_before"
            ),
            mean_logit_rms_after=_window_mean(
                group, active, "logit_rms_after"
            ),
            mean_loss_before=_window_mean(group, active, "loss_before"),
            mean_loss_after=_window_mean(group, active, "loss_after"),
            mean_accuracy_before=_window_mean(group, active, "accuracy_before"),
            mean_accuracy_after=_window_mean(group, active, "accuracy_after"),
            mean_instant_loss_delta=_window_mean(group, active, "instant_loss_delta"),
            mean_gradient_corr=_window_mean(group, active, "gradient_corr"),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["mean_sample_mapping_corr"] = summary[
        ["mean_sample_mapping_corr_a", "mean_sample_mapping_corr_b"]
    ].mean(axis=1)
    summary = summary.merge(final_table, on=keys, how="left", suffixes=("", "_final"))
    baseline = summary.loc[summary["condition"].eq("none"), keys[:-1] + [
        "delta_is_immediate", "delta_is_post", "final_test_loss", "final_test_accuracy"
    ]].rename(
        columns={
            "delta_is_immediate": "none_delta_is_immediate",
            "delta_is_post": "none_delta_is_post",
            "final_test_loss": "none_final_test_loss",
            "final_test_accuracy": "none_final_test_accuracy",
        }
    )
    summary = summary.merge(baseline, on=keys[:-1], how="left")
    summary["delta_is_immediate_vs_none"] = (
        summary["delta_is_immediate"] - summary["none_delta_is_immediate"]
    )
    summary["delta_is_vs_none"] = summary["delta_is_post"] - summary["none_delta_is_post"]
    summary["test_loss_vs_none"] = summary["final_test_loss"] - summary["none_final_test_loss"]
    summary["test_accuracy_vs_none"] = (
        summary["final_test_accuracy"] - summary["none_final_test_accuracy"]
    )
    return summary


def seed_level_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "delta_is_immediate",
        "delta_is_immediate_vs_none",
        "delta_is_post",
        "delta_is_vs_none",
        "mean_fc_before",
        "mean_fc_after",
        "mean_fc_shift",
        "mean_fc_vs_untreated",
        "mean_fc_target_abs_error",
        "mean_abs_delta",
        "mean_std_relative_error",
        "mean_marginal_sorted_rmse",
        "mean_activation_delta_rms_ratio",
        "mean_logit_delta_rms_ratio",
        "mean_sample_mapping_corr_a",
        "mean_sample_mapping_corr_b",
        "mean_sample_mapping_corr",
        "mean_selected_strength",
        "mean_selected_noise_corr",
        "mean_activation_target_abs_error",
        "mean_logit_target_abs_error",
        "mean_mapping_target_abs_error",
        "mean_activation_mean_before",
        "mean_activation_mean_after",
        "mean_activation_sd_before",
        "mean_activation_sd_after",
        "mean_activation_rms_before",
        "mean_activation_rms_after",
        "mean_logit_rms_before",
        "mean_logit_rms_after",
        "mean_loss_before",
        "mean_loss_after",
        "mean_accuracy_before",
        "mean_accuracy_after",
        "mean_instant_loss_delta",
        "test_loss_vs_none",
        "test_accuracy_vs_none",
    ]
    out = (
        run_summary.groupby(["seed", "stratum", "condition"], as_index=False)[numeric]
        .mean()
    )
    counts = (
        run_summary.groupby(["seed", "stratum", "condition"])["pair_id"]
        .nunique()
        .rename("n_pairs")
        .reset_index()
    )
    return out.merge(counts, on=["seed", "stratum", "condition"], how="left")


def bootstrap_mean_ci(
    values: Sequence[float], reps: int = 5000, seed: int = 0
) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def primary_contrasts(
    seed_summary: pd.DataFrame,
    config: ExperimentConfig,
    controls: Sequence[str] = ("none", "independent_rank", "corr_preserving_rank"),
) -> pd.DataFrame:
    """Two-sided seed-level paired tests; pairs have already been averaged within seed."""

    rows = []
    for stratum in ("high", "low"):
        subset = seed_summary.loc[seed_summary["stratum"].eq(stratum)]
        pivot = subset.pivot(index="seed", columns="condition", values="delta_is_post")
        if "active_rank" not in pivot:
            continue
        for control in controls:
            if control not in pivot:
                continue
            paired = pivot[["active_rank", control]].dropna()
            difference = (
                paired["active_rank"].to_numpy() - paired[control].to_numpy()
            )
            if len(difference) < 2 or np.allclose(difference, 0):
                statistic, p_value = np.nan, 1.0
            else:
                test = stats.wilcoxon(
                    difference,
                    alternative="two-sided",
                    mode="auto",
                )
                statistic, p_value = float(test.statistic), float(test.pvalue)
            ci_low, ci_high = bootstrap_mean_ci(
                difference,
                reps=config.ci_bootstrap_reps,
                seed=derived_seed(stratum, control, "bootstrap"),
            )
            rows.append(
                {
                    "stratum": stratum,
                    "contrast": f"active_rank - {control}",
                    "n_seeds": len(difference),
                    "mean_difference": float(np.mean(difference)),
                    "median_difference": float(np.median(difference)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "p_two_sided": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm"] = holm_adjust(result["p_two_sided"].to_numpy())
    return result


def manipulation_qc(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Condition-level QC with seed as the aggregation level."""

    seed = seed_level_summary(run_summary)
    return (
        seed.groupby(["stratum", "condition"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            fc_shift_mean=("mean_fc_shift", "mean"),
            fc_shift_sd=("mean_fc_shift", "std"),
            fc_vs_untreated_mean=("mean_fc_vs_untreated", "mean"),
            fc_vs_untreated_sd=("mean_fc_vs_untreated", "std"),
            fc_target_error_mean=("mean_fc_target_abs_error", "mean"),
            marginal_rmse_mean=("mean_marginal_sorted_rmse", "mean"),
            std_error_mean=("mean_std_relative_error", "mean"),
            activation_impact_mean=("mean_activation_delta_rms_ratio", "mean"),
            logit_impact_mean=("mean_logit_delta_rms_ratio", "mean"),
            mapping_corr_mean=("mean_sample_mapping_corr", "mean"),
            activation_target_error_mean=("mean_activation_target_abs_error", "mean"),
            logit_target_error_mean=("mean_logit_target_abs_error", "mean"),
            mapping_target_error_mean=("mean_mapping_target_abs_error", "mean"),
            loss_impact_mean=("mean_instant_loss_delta", "mean"),
        )
    )


def save_results(
    step_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    final_table: pd.DataFrame,
    run_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    config: ExperimentConfig,
) -> Path:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    step_table.to_csv(out / "step_metrics.csv.gz", index=False, compression="gzip")
    pair_table.to_csv(out / "selected_pairs.csv", index=False)
    final_table.to_csv(out / "final_metrics.csv", index=False)
    run_summary.to_csv(out / "run_summary.csv", index=False)
    seed_summary.to_csv(out / "seed_summary.csv", index=False)
    contrasts.to_csv(out / "primary_contrasts.csv", index=False)
    manifest = {
        "config": asdict(config),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision_note": "MNIST loaded through torchvision.datasets.MNIST",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "device": str(resolve_device(config.device)),
        "cuda_available": bool(torch.cuda.is_available()),
        "inference_unit": "seed; pairs are nested within seed",
        "primary_test": "two-sided paired Wilcoxon signed-rank on seed means with Holm correction",
        "pair_selection": (
            "untreated scout trajectory on the held-out selection split; repeated FC "
            "measurements end at intervention_start - 1; stable high/low rules use only "
            "pre-intervention FC and common unit-quality filters, never future FC, "
            "intervention outcomes or IS"
        ),
        "branch_identity_control": (
            "every branch restores the exact scout checkpoint at intervention_start - 1 "
            "and reuses the same deterministic minibatch schedule within seed"
        ),
        "candidate_search": (
            "one-neuron rank transport with FC feasibility gating; vectorized float32 "
            "GPU scoring and full double-precision diagnostics only for the selected "
            "proposal; active targets are anchored to the matched untreated trajectory"
        ),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


CONDITION_LABELS = {
    "none": "No intervention",
    "active_rank": "FC-targeted remap",
    "independent_rank": "Independent remap",
    "corr_preserving_rank": "FC-preserving remap",
}

CONDITION_COLORS = {
    "none": "#6B7280",
    "active_rank": "#7A5195",
    "independent_rank": "#3A7CA5",
    "corr_preserving_rank": "#E09F3E",
}


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_publication_figure(
    fig: mpl.figure.Figure,
    base_path: Path | str,
    dpi: int = 600,
    include_tiff: bool = False,
) -> List[Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
    ):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    if include_tiff:
        path = base.with_suffix(".tiff")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    return outputs


def make_selection_stability_figure(
    pair_table: pd.DataFrame,
    config: ExperimentConfig,
) -> mpl.figure.Figure:
    """Plot the untreated pre-intervention trajectories defining pair identity."""

    set_publication_style()
    checkpoint_columns = sorted(
        (
            column
            for column in pair_table.columns
            if column.startswith("fc_step_")
        ),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if not checkpoint_columns:
        raise ValueError("pair_table has no fc_step_* scout trajectory columns.")
    checkpoints = np.asarray(
        [int(column.rsplit("_", 1)[1]) for column in checkpoint_columns]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.7), constrained_layout=True)
    seed_values = sorted(pair_table["seed"].unique())
    palette = plt.get_cmap("tab10")
    seed_colors = {
        seed: palette(index % 10) for index, seed in enumerate(seed_values)
    }
    line_styles = ("-", "--", "-.", ":")

    for ax, stratum in zip(axes, ("high", "low")):
        subset = pair_table.loc[pair_table["stratum"].eq(stratum)].copy()
        if subset.empty:
            raise ValueError(f"No selected {stratum}-FC pairs in pair_table.")
        if stratum == "high":
            ax.axhspan(
                config.selection_high_min_fc,
                1.0,
                color="#DDEFE2",
                alpha=0.65,
                lw=0,
                label="Allowed checkpoint range",
            )
            ax.axhline(
                config.selection_high_mean_min,
                color="#387A4A",
                lw=0.8,
                ls=":",
                label="Configured mean floor",
            )
            ax.set_title("Stable high-FC pairs")
        else:
            ax.axhspan(
                -config.selection_low_max_abs_fc,
                config.selection_low_max_abs_fc,
                color="#E6ECF5",
                alpha=0.75,
                lw=0,
                label="Allowed checkpoint range",
            )
            ax.axhline(
                config.selection_low_abs_mean_max,
                color="#4C6A92",
                lw=0.7,
                ls=":",
                label="Configured |mean| ceiling",
            )
            ax.axhline(
                -config.selection_low_abs_mean_max,
                color="#4C6A92",
                lw=0.7,
                ls=":",
            )
            ax.set_title("Stable low-FC pairs")

        for row in subset.sort_values(["seed", "pair_rank"]).itertuples(index=False):
            values = np.asarray(
                [getattr(row, column) for column in checkpoint_columns],
                dtype=float,
            )
            color = seed_colors[row.seed]
            style = line_styles[int(row.pair_rank) % len(line_styles)]
            label = f"s{int(row.seed)}: {int(row.unit_a)}-{int(row.unit_b)}"
            ax.plot(
                checkpoints,
                values,
                color=color,
                ls=style,
                marker="o",
                ms=2.2,
                lw=1.1,
                label=label,
            )
        # With two pairs per seed, a compact legend is more legible than
        # overlapping direct labels when all low-FC endpoints lie near zero.
        ax.legend(loc="best", ncol=1)
        ax.axvline(config.selection_start, color="#6B7280", lw=0.7, ls="--")
        ax.set_xlabel("Untreated scout step")
        ax.set_ylabel("Held-out functional correlation")
        ax.set_xlim(checkpoints[0] - 0.02 * np.ptp(checkpoints), checkpoints[-1] + 0.02 * np.ptp(checkpoints))
        ax.set_ylim(-1.02, 1.02)

    for label, ax in zip("ab", axes):
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    return fig


def _representative_pair(run_summary: pd.DataFrame, stratum: str) -> str:
    """Choose a deterministic descriptive pair without using its outcome."""

    pair_ids = (
        run_summary.loc[run_summary["stratum"].eq(stratum), "pair_id"]
        .drop_duplicates()
        .sort_values()
    )
    if pair_ids.empty:
        raise ValueError(f"No {stratum}-FC pair is available.")
    return str(pair_ids.iloc[0])


def make_main_figure(
    step_table: pd.DataFrame,
    run_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    config: ExperimentConfig,
) -> mpl.figure.Figure:
    """Six-panel evidence-chain figure for trajectory, specificity and replication."""

    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.08, 5.2), constrained_layout=True)
    trajectory_conditions = ["none", "active_rank", "independent_rank", "corr_preserving_rank"]
    for row, stratum in enumerate(("high", "low")):
        pair_id = _representative_pair(run_summary, stratum)
        sub = step_table.loc[step_table["pair_id"].eq(pair_id)]
        ax_fc = axes[row, 0]
        ax_is = axes[row, 1]
        for condition in trajectory_conditions:
            cond = sub.loc[sub["condition"].eq(condition)].sort_values("step")
            ax_fc.plot(
                cond["step"], cond["fc_after"],
                color=CONDITION_COLORS[condition], lw=1.3,
                label=CONDITION_LABELS[condition],
            )
            ax_is.plot(
                cond["step"], cond["is_value"],
                color=CONDITION_COLORS[condition], lw=1.3,
            )
            if condition == "active_rank" and cond["target_fc"].notna().any():
                target = cond.loc[cond["target_fc"].notna()]
                ax_fc.plot(
                    target["step"], target["target_fc"],
                    color=CONDITION_COLORS[condition], lw=0.8, ls=":",
                    label="Active FC target",
                )
        for ax in (ax_fc, ax_is):
            ax.axvspan(
                config.intervention_start,
                config.intervention_end,
                color="#D1D5DB",
                alpha=0.35,
                lw=0,
            )
            ax.set_xlabel("Training step")
        ax_fc.set_ylabel("Functional correlation")
        ax_is.set_ylabel("Input similarity")
        ax_fc.set_title(f"{'High' if stratum == 'high' else 'Low'}-FC pair: manipulation")
        ax_is.set_title(f"{'High' if stratum == 'high' else 'Low'}-FC pair: downstream IS")
        if row == 0:
            ax_fc.legend(loc="best", ncol=1)

    ax_qc = axes[0, 2]
    qc_conditions = ["active_rank", "independent_rank", "corr_preserving_rank"]
    for x, condition in enumerate(qc_conditions):
        values = seed_summary.loc[
            seed_summary["condition"].eq(condition), "mean_logit_delta_rms_ratio"
        ].dropna().to_numpy()
        jitter = np.linspace(-0.10, 0.10, max(len(values), 1))
        ax_qc.scatter(
            x + jitter[: len(values)], values,
            s=10, alpha=0.65, color=CONDITION_COLORS[condition], edgecolor="none"
        )
        if len(values):
            ax_qc.plot([x - 0.18, x + 0.18], [values.mean(), values.mean()], color="black", lw=1.2)
    ax_qc.axhline(config.target_logit_rms_ratio, color="#111827", ls="--", lw=0.8)
    ax_qc.set_xticks(range(len(qc_conditions)))
    ax_qc.set_xticklabels([CONDITION_LABELS[c] for c in qc_conditions], rotation=30, ha="right")
    ax_qc.set_ylabel("RMS logit change / baseline")
    ax_qc.set_title("Output-impact matching")

    ax_effect = axes[1, 2]
    controls = ["none", "independent_rank", "corr_preserving_rank"]
    x = np.arange(len(controls))
    for offset, stratum in ((-0.12, "high"), (0.12, "low")):
        pivot = seed_summary.loc[seed_summary["stratum"].eq(stratum)].pivot(
            index="seed", columns="condition", values="delta_is_post"
        )
        means, lows, highs = [], [], []
        for control in controls:
            values = pivot["active_rank"] - pivot[control]
            means.append(values.mean())
            lo, hi = bootstrap_mean_ci(
                values.to_numpy(), config.ci_bootstrap_reps,
                derived_seed(stratum, control, "figure-ci"),
            )
            lows.append(values.mean() - lo)
            highs.append(hi - values.mean())
            ax_effect.scatter(
                np.full(len(values), x[len(means) - 1] + offset), values,
                s=9, alpha=0.45,
                color="#4C78A8" if stratum == "high" else "#B279A2",
                edgecolor="none",
            )
        ax_effect.errorbar(
            x + offset, means, yerr=np.vstack([lows, highs]), fmt="o",
            ms=4, capsize=2, lw=1,
            color="#2F5D8A" if stratum == "high" else "#8F4F7A",
            label=f"{stratum}-FC pairs",
        )
    ax_effect.axhline(0, color="#6B7280", lw=0.8)
    ax_effect.set_xticks(x)
    ax_effect.set_xticklabels(["vs none", "vs independent", "vs FC-preserving"], rotation=20, ha="right")
    ax_effect.set_ylabel("Active − control ΔIS")
    ax_effect.set_title("Seed-level replicated effect")
    ax_effect.legend(loc="best")

    for label, ax in zip("abcdef", axes.flat):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    return fig


def make_reference_effect_figure(
    step_table: pd.DataFrame,
    config: ExperimentConfig,
) -> mpl.figure.Figure:
    """Show achieved FC dose and downstream IS specificity at seed level.

    Pairs are averaged within seed before the across-seed mean and 95% normal
    interval are drawn.  The plot is descriptive; formal tests use the
    prespecified seed-level summary endpoints.
    """

    required = {"fc_vs_untreated", "is_value", "condition", "seed", "stratum"}
    missing = required - set(step_table.columns)
    if missing:
        raise ValueError(f"step_table lacks reference-effect fields: {sorted(missing)}")

    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 4.2), constrained_layout=True)
    fc_conditions = ("active_rank", "independent_rank", "corr_preserving_rank")
    is_controls = ("none", "independent_rank", "corr_preserving_rank")

    def plot_seed_band(ax: mpl.axes.Axes, table: pd.DataFrame, value: str,
                       color: str, label: str) -> None:
        summary = table.groupby("step", as_index=False)[value].agg(
            mean="mean", sem="sem", n="count"
        )
        x = summary["step"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, mean, color=color, lw=1.25, label=label)
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.14, lw=0)

    pair_mean = (
        step_table.groupby(
            ["seed", "stratum", "condition", "step"], as_index=False
        )[["fc_vs_untreated", "is_value"]]
        .mean()
    )
    for row, stratum in enumerate(("high", "low")):
        stratum_data = pair_mean.loc[pair_mean["stratum"].eq(stratum)]
        ax_fc, ax_is = axes[row]
        for condition in fc_conditions:
            values = stratum_data.loc[stratum_data["condition"].eq(condition)]
            plot_seed_band(
                ax_fc, values, "fc_vs_untreated",
                CONDITION_COLORS[condition], CONDITION_LABELS[condition],
            )
        expected = (-1.0 if stratum == "high" else 1.0) * config.target_fc_shift
        ax_fc.plot(
            [config.intervention_start, config.intervention_end],
            [expected, expected], color="#111827", lw=0.8, ls=":",
            label="Prespecified target" if row == 0 else None,
        )
        ax_fc.axhline(0, color="#6B7280", lw=0.7)
        ax_fc.set_ylabel("FC minus untreated")
        ax_fc.set_title(f"{'High' if stratum == 'high' else 'Low'}-FC: achieved dose")

        pivot = stratum_data.pivot(
            index=["seed", "step"], columns="condition", values="is_value"
        ).reset_index()
        for control in is_controls:
            if "active_rank" not in pivot or control not in pivot:
                continue
            column = f"active_minus_{control}"
            pivot[column] = pivot["active_rank"] - pivot[control]
            plot_seed_band(
                ax_is, pivot[["seed", "step", column]], column,
                CONDITION_COLORS[control], f"Active vs {CONDITION_LABELS[control]}",
            )
        ax_is.axhline(0, color="#6B7280", lw=0.7)
        ax_is.set_ylabel("Active − control ΔIS")
        ax_is.set_title(f"{'High' if stratum == 'high' else 'Low'}-FC: IS specificity")

        for ax in (ax_fc, ax_is):
            ax.axvspan(
                config.intervention_start, config.intervention_end,
                color="#D1D5DB", alpha=0.30, lw=0, zorder=-10,
            )
            ax.set_xlabel("Training step")
        if row == 0:
            ax_fc.legend(loc="best")
            ax_is.legend(loc="best")

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.17, 1.05, label, transform=ax.transAxes,
                fontweight="bold", fontsize=8)
    return fig


def make_seed_effect_figure(
    seed_summary: pd.DataFrame,
    config: ExperimentConfig,
) -> mpl.figure.Figure:
    """Raw seed-level active-minus-control IS effects without sign reversal."""

    set_publication_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.8), constrained_layout=True)
    controls = ("none", "independent_rank", "corr_preserving_rank")
    x = np.arange(len(controls), dtype=float)
    for offset, stratum, color in (
        (-0.11, "high", "#2F5D8A"),
        (0.11, "low", "#8F4F7A"),
    ):
        pivot = seed_summary.loc[seed_summary["stratum"].eq(stratum)].pivot(
            index="seed", columns="condition", values="delta_is_post"
        )
        means: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        for index, control in enumerate(controls):
            values = (pivot["active_rank"] - pivot[control]).dropna().to_numpy()
            means.append(float(np.mean(values)))
            lo, hi = bootstrap_mean_ci(
                values,
                config.ci_bootstrap_reps,
                derived_seed(stratum, control, "seed-effect-ci"),
            )
            lower.append(means[-1] - lo)
            upper.append(hi - means[-1])
            ax.scatter(
                np.full(len(values), x[index] + offset),
                values,
                s=11,
                alpha=0.45,
                color=color,
                edgecolor="none",
            )
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            ms=4.5,
            capsize=2,
            lw=1,
            color=color,
            label=f"{stratum}-FC pairs",
        )
    ax.axhline(0, color="#6B7280", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["vs none", "vs independent", "vs FC-preserving"],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Active − control ΔIS")
    ax.set_title("Seed-level replicated effect")
    ax.legend(loc="best")
    return fig


def make_qc_figure(
    seed_summary: pd.DataFrame,
) -> mpl.figure.Figure:
    """Direct before/after QC for FC and non-FC quantities.

    Pair observations are already averaged within seed.  For non-FC panels,
    high- and low-FC strata are additionally averaged within seed so every
    displayed paired trajectory remains one independent training replicate.
    """

    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.08, 4.15), constrained_layout=True)
    conditions = (
        "none",
        "active_rank",
        "independent_rank",
        "corr_preserving_rank",
    )

    def paired_panel(
        ax: mpl.axes.Axes,
        table: pd.DataFrame,
        before_col: str,
        after_col: str,
        ylabel: str,
        title: str,
    ) -> None:
        for x, condition in enumerate(conditions):
            subset = table.loc[table["condition"].eq(condition)].sort_values("seed")
            before = subset[before_col].to_numpy(dtype=float)
            after = subset[after_col].to_numpy(dtype=float)
            valid = np.isfinite(before) & np.isfinite(after)
            before, after = before[valid], after[valid]
            color = CONDITION_COLORS[condition]
            for left, right in zip(before, after):
                ax.plot(
                    [x - 0.11, x + 0.11],
                    [left, right],
                    color=color,
                    alpha=0.20,
                    lw=0.65,
                )
            ax.scatter(
                np.full(len(before), x - 0.11),
                before,
                s=9,
                facecolor="white",
                edgecolor=color,
                linewidth=0.6,
                alpha=0.75,
            )
            ax.scatter(
                np.full(len(after), x + 0.11),
                after,
                s=9,
                color=color,
                edgecolor="none",
                alpha=0.65,
            )
            if len(before):
                ax.plot(
                    [x - 0.17, x - 0.05],
                    [before.mean(), before.mean()],
                    color="#111827",
                    lw=1.0,
                )
                ax.plot(
                    [x + 0.05, x + 0.17],
                    [after.mean(), after.mean()],
                    color="#111827",
                    lw=1.0,
                )
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in conditions], rotation=28, ha="right"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    paired_panel(
        axes[0, 0],
        seed_summary.loc[seed_summary["stratum"].eq("high")],
        "mean_fc_before",
        "mean_fc_after",
        "Functional correlation",
        "High-FC: before vs after",
    )
    paired_panel(
        axes[0, 1],
        seed_summary.loc[seed_summary["stratum"].eq("low")],
        "mean_fc_before",
        "mean_fc_after",
        "Functional correlation",
        "Low-FC: before vs after",
    )

    non_fc_columns = [
        "mean_activation_mean_before",
        "mean_activation_mean_after",
        "mean_activation_sd_before",
        "mean_activation_sd_after",
        "mean_logit_rms_before",
        "mean_logit_rms_after",
        "mean_loss_before",
        "mean_loss_after",
    ]
    non_fc = (
        seed_summary.groupby(["seed", "condition"], as_index=False)[non_fc_columns]
        .mean()
    )
    paired_panel(
        axes[0, 2],
        non_fc,
        "mean_activation_mean_before",
        "mean_activation_mean_after",
        "Mean activation",
        "Activation location",
    )
    paired_panel(
        axes[1, 0],
        non_fc,
        "mean_activation_sd_before",
        "mean_activation_sd_after",
        "Activation s.d.",
        "Activation scale",
    )
    paired_panel(
        axes[1, 1],
        non_fc,
        "mean_logit_rms_before",
        "mean_logit_rms_after",
        "RMS logits",
        "Output scale",
    )
    paired_panel(
        axes[1, 2],
        non_fc,
        "mean_loss_before",
        "mean_loss_after",
        "Cross-entropy loss",
        "Instantaneous loss",
    )
    axes[0, 2].plot([], [], "o", ms=3.5, mfc="white", mec="#4B5563", label="Before")
    axes[0, 2].plot([], [], "o", ms=3.5, color="#4B5563", label="After")
    axes[0, 2].legend(loc="best")
    for label, ax in zip("abcdef", axes.flat):
        ax.text(-0.22, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    return fig


def _ema(values: Sequence[float], factor: float = 0.90) -> np.ndarray:
    """Exponential moving average used only as a visual guide."""

    if not 0.0 <= factor < 1.0:
        raise ValueError("smooth factor must lie in [0, 1).")
    return (
        pd.Series(np.asarray(values, dtype=float))
        .ewm(alpha=1.0 - factor, adjust=False, ignore_na=True)
        .mean()
        .to_numpy()
    )


def make_pair_dynamics_figure(
    step_table: pd.DataFrame,
    pair_id: str,
    config: ExperimentConfig,
    conditions: Optional[Sequence[str]] = None,
    smooth_factor: float = 0.90,
) -> mpl.figure.Figure:
    """Legacy-compatible six-panel trajectory figure for one selected pair.

    Raw minibatch values are shown faintly and exponential moving averages are
    overlaid for readability.  Smoothing is visualization-only; all numerical
    summaries and tests continue to use unsmoothed data.
    """

    set_publication_style()
    default_conditions = (
        "none",
        "active_rank",
        "independent_rank",
        "corr_preserving_rank",
    )
    conditions = tuple(conditions or default_conditions)
    subset = step_table.loc[step_table["pair_id"].eq(pair_id)].copy()
    if subset.empty:
        raise ValueError(f"Unknown pair_id: {pair_id}")
    available = set(subset["condition"].unique())
    conditions = tuple(condition for condition in conditions if condition in available)
    if not conditions:
        raise ValueError("None of the requested conditions is present for this pair.")

    first = subset.iloc[0]
    fig, axes = plt.subplots(3, 2, figsize=(7.08, 6.0), constrained_layout=True)
    metric_panels = (
        (axes[0, 0], "fc_after", "Functional correlation", "FC trajectory", 1.0),
        (axes[0, 1], "is_value", "Input similarity", "IS trajectory", 1.0),
        (axes[1, 0], "gradient_corr", "Gradient correlation", "Gradient alignment", 1.0),
        (axes[2, 0], "loss", "Cross-entropy loss", "Training loss", 1.0),
        (axes[2, 1], "accuracy", "Accuracy (%)", "Training accuracy", 100.0),
    )

    for condition in conditions:
        cond = subset.loc[subset["condition"].eq(condition)].sort_values("step")
        steps = cond["step"].to_numpy()
        color = CONDITION_COLORS.get(condition, "#4B5563")
        label = CONDITION_LABELS.get(condition, condition)
        for ax, metric, ylabel, title, scale in metric_panels:
            values = cond[metric].to_numpy(dtype=float) * scale
            ax.plot(steps, values, color=color, lw=0.55, alpha=0.18)
            ax.plot(
                steps,
                _ema(values, smooth_factor),
                color=color,
                lw=1.35,
                label=label if metric == "fc_after" else None,
            )
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        ax_norm = axes[1, 1]
        norm_a = cond["gradient_norm_a"].to_numpy(dtype=float)
        norm_b = cond["gradient_norm_b"].to_numpy(dtype=float)
        ax_norm.plot(steps, norm_a, color=color, lw=0.45, alpha=0.12)
        ax_norm.plot(steps, norm_b, color=color, lw=0.45, alpha=0.12, ls="--")
        ax_norm.plot(steps, _ema(norm_a, smooth_factor), color=color, lw=1.25)
        ax_norm.plot(
            steps, _ema(norm_b, smooth_factor), color=color, lw=1.25, ls="--"
        )

    active_target = subset.loc[
        subset["condition"].eq("active_rank") & subset["target_fc"].notna()
    ].sort_values("step")
    if len(active_target):
        axes[0, 0].plot(
            active_target["step"], active_target["target_fc"],
            color=CONDITION_COLORS["active_rank"], lw=0.9, ls=":",
            label="Active FC target",
        )

    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 1].set_ylabel("Gradient L2 norm")
    axes[1, 1].set_title("Gradient magnitude (solid: A; dashed: B)")
    axes[0, 0].legend(loc="best", ncol=1)

    for ax in axes.flat:
        ax.axvspan(
            config.intervention_start,
            config.intervention_end,
            color="#D1D5DB",
            alpha=0.35,
            lw=0,
            zorder=-10,
        )
        ax.set_xlabel("Training step")
    for label, ax in zip("abcdef", axes.flat):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8)

    fig.suptitle(
        f"Pair {pair_id} | seed {int(first['seed'])} | {first['stratum']}-FC | "
        f"units {int(first['unit_a'])}, {int(first['unit_b'])}",
        fontsize=8.5,
    )
    return fig


def export_pair_dynamics_figures(
    step_table: pd.DataFrame,
    config: ExperimentConfig,
    conditions: Optional[Sequence[str]] = None,
    smooth_factor: float = 0.90,
    include_tiff: bool = False,
) -> List[Path]:
    """Export one comparison trajectory figure for every selected pair."""

    output_dir = Path(config.figure_dir) / "pair_dynamics"
    outputs: List[Path] = []
    for pair_id in step_table["pair_id"].drop_duplicates():
        fig = make_pair_dynamics_figure(
            step_table,
            str(pair_id),
            config,
            conditions=conditions,
            smooth_factor=smooth_factor,
        )
        outputs.extend(
            save_publication_figure(
                fig,
                output_dir / f"pair_dynamics_{pair_id}",
                include_tiff=include_tiff,
            )
        )
        plt.close(fig)
    return outputs


def qc_assertion_table(run_summary: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Advisory checks; failures require inspection rather than automatic exclusion."""

    rank = run_summary.loc[
        run_summary["condition"].isin(
            ["active_rank", "independent_rank", "corr_preserving_rank"]
        )
    ]
    active = rank.loc[rank["condition"].eq("active_rank")]
    seed = seed_level_summary(run_summary)
    checks = [
        (
            "Exact marginal preservation",
            float(rank["mean_marginal_sorted_rmse"].max()),
            1e-7,
            "<=",
        ),
        (
            "Relative s.d. preservation",
            float(rank["mean_std_relative_error"].max()),
            1e-6,
            "<=",
        ),
        (
            "Mean logit impact near target",
            float(abs(rank["mean_logit_delta_rms_ratio"].mean() - config.target_logit_rms_ratio)),
            max(0.02, config.target_logit_rms_ratio * 0.50),
            "<=",
        ),
        (
            "Mean sample-mapping retention near target",
            float(abs(rank["mean_sample_mapping_corr"].mean() - config.target_sample_mapping_corr)),
            0.10,
            "<=",
        ),
        (
            "Active FC shift has intended sign",
            float(
                np.mean(
                    np.sign(active["mean_fc_vs_untreated"].to_numpy())
                    == active["stratum"].map({"high": -1, "low": 1}).to_numpy()
                )
            ),
            0.80,
            ">=",
        ),
        (
            "Active FC reaches untreated-anchored target",
            float(active["mean_fc_target_abs_error"].mean()),
            config.fc_target_tolerance,
            "<=",
        ),
    ]
    impact_tolerances = {
        "mean_activation_delta_rms_ratio": max(
            0.05, config.target_activation_delta_ratio * 0.25
        ),
        "mean_logit_delta_rms_ratio": max(0.01, config.target_logit_rms_ratio * 0.25),
        "mean_sample_mapping_corr": 0.10,
    }
    impact_labels = {
        "mean_activation_delta_rms_ratio": "Activation displacement",
        "mean_logit_delta_rms_ratio": "Logit displacement",
        "mean_sample_mapping_corr": "Sample-mapping disruption",
    }
    for stratum in ("high", "low"):
        subset = seed.loc[seed["stratum"].eq(stratum)]
        for metric, tolerance in impact_tolerances.items():
            pivot = subset.pivot(index="seed", columns="condition", values=metric)
            if "active_rank" not in pivot:
                continue
            for control in ("independent_rank", "corr_preserving_rank"):
                if control not in pivot:
                    continue
                difference = (pivot["active_rank"] - pivot[control]).abs().mean()
                checks.append(
                    (
                        f"{stratum}: {impact_labels[metric]} matched to {control}",
                        float(difference),
                        tolerance,
                        "<=",
                    )
                )
        preserving = subset.loc[
            subset["condition"].eq("corr_preserving_rank"), "mean_fc_shift"
        ].abs().mean()
        checks.append(
            (
                f"{stratum}: FC-preserving control remains near baseline FC",
                float(preserving),
                max(0.05, config.target_fc_shift * 0.20),
                "<=",
            )
        )
    rows = []
    for name, observed, threshold, op in checks:
        passed = observed <= threshold if op == "<=" else observed >= threshold
        rows.append(
            {
                "check": name,
                "observed": observed,
                "criterion": f"{op} {threshold:g}",
                "status": "PASS" if passed else "REVIEW",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Resumable manuscript-scale orchestration
# ---------------------------------------------------------------------------

RESULT_FILENAMES = {
    "step": "step_metrics.csv.gz",
    "pairs": "selected_pairs.csv",
    "final": "final_metrics.csv",
    "runs": "run_summary.csv",
    "seeds": "seed_summary.csv",
    "contrasts": "primary_contrasts.csv",
}


def paper_multiseed_config(
    base_dir: Path | str = ".",
    n_seeds: int = 10,
    n_pairs_per_stratum: int = 4,
) -> ExperimentConfig:
    """Configuration for the streamlined manuscript run."""

    if n_seeds < 2:
        raise ValueError("Use at least two independent seeds.")
    if n_pairs_per_stratum < 2:
        raise ValueError("Use at least two node-disjoint pairs per FC stratum.")
    base = Path(base_dir)
    config = replace(
        paper_config(base),
        seeds=tuple(range(n_seeds)),
        n_pairs_per_stratum=n_pairs_per_stratum,
        conditions=(
            "none",
            "active_rank",
            "independent_rank",
            "corr_preserving_rank",
        ),
        output_dir=str(
            base / "results" / "supp_fig4_fc_perturbation_multiseed_streamlined"
        ),
        figure_dir=str(
            base / "fig" / "supp_fig4_fc_perturbation_multiseed_streamlined"
        ),
    )
    config.validate()
    return config


def smoke_multiseed_config(base_dir: Path | str = ".") -> ExperimentConfig:
    """Short plumbing check; never use its statistics in the manuscript."""

    base = Path(base_dir)
    return replace(
        smoke_config(base),
        conditions=(
            "none",
            "active_rank",
            "independent_rank",
            "corr_preserving_rank",
        ),
        output_dir=str(
            base / "results" / "supp_fig4_fc_perturbation_multiseed_streamlined_smoke"
        ),
        figure_dir=str(
            base / "fig" / "supp_fig4_fc_perturbation_multiseed_streamlined_smoke"
        ),
    )


def analysis_signature(config: ExperimentConfig) -> str:
    """Fingerprint settings that affect training, selection or intervention."""

    payload = asdict(config)
    for key in (
        "seeds",
        "data_root",
        "output_dir",
        "figure_dir",
        "device",
        "ci_bootstrap_reps",
    ):
        payload.pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def estimate_workload(config: ExperimentConfig) -> Dict[str, int]:
    pairs_per_seed = 2 * config.n_pairs_per_stratum
    n_pairs = len(config.seeds) * pairs_per_seed
    output_branches = n_pairs * len(config.conditions)
    trained_branches_per_seed = 1 + pairs_per_seed * (len(config.conditions) - 1)
    trained_branches = len(config.seeds) * trained_branches_per_seed
    updates_per_branch = config.total_steps - config.intervention_start + 1
    return {
        "n_seeds": len(config.seeds),
        "pairs_per_seed": pairs_per_seed,
        "n_pairs_total": n_pairs,
        "n_conditions": len(config.conditions),
        "output_branches": output_branches,
        "trained_branches": trained_branches,
        "scout_updates": len(config.seeds) * (config.intervention_start - 1),
        "branch_updates": trained_branches * updates_per_branch,
    }


def _seed_dir(config: ExperimentConfig, seed: int) -> Path:
    return Path(config.output_dir) / "per_seed" / f"seed_{seed:03d}"


def _seed_paths(config: ExperimentConfig, seed: int) -> Dict[str, Path]:
    directory = _seed_dir(config, seed)
    return {
        "directory": directory,
        "step": directory / RESULT_FILENAMES["step"],
        "pairs": directory / RESULT_FILENAMES["pairs"],
        "final": directory / RESULT_FILENAMES["final"],
        "complete": directory / "COMPLETE.json",
        "error": directory / "ERROR.json",
    }


def _seed_is_complete(config: ExperimentConfig, seed: int) -> bool:
    paths = _seed_paths(config, seed)
    required = (paths["step"], paths["pairs"], paths["final"], paths["complete"])
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
        pairs = pd.read_csv(paths["pairs"])
        final = pd.read_csv(paths["final"])
    except Exception:
        return False
    expected_pairs = 2 * config.n_pairs_per_stratum
    expected_outputs = expected_pairs * len(config.conditions)
    return (
        int(marker.get("seed", -1)) == int(seed)
        and marker.get("analysis_signature") == analysis_signature(config)
        and pairs["pair_id"].nunique() == expected_pairs
        and final[["pair_id", "condition"]].drop_duplicates().shape[0]
        == expected_outputs
        and set(final["condition"].astype(str)) == set(config.conditions)
    )


def _write_seed_result(
    config: ExperimentConfig,
    seed: int,
    step_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    final_table: pd.DataFrame,
) -> None:
    paths = _seed_paths(config, seed)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    step_table.to_csv(paths["step"], index=False, compression="gzip")
    pair_table.to_csv(paths["pairs"], index=False)
    final_table.to_csv(paths["final"], index=False)
    marker = {
        "seed": seed,
        "analysis_signature": analysis_signature(config),
        "n_pairs": int(pair_table["pair_id"].nunique()),
        "n_outputs": int(
            final_table[["pair_id", "condition"]].drop_duplicates().shape[0]
        ),
        "conditions": list(config.conditions),
        "shared_untreated_branch": True,
    }
    paths["complete"].write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )


def _load_seed_result(
    config: ExperimentConfig,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not _seed_is_complete(config, seed):
        raise RuntimeError(f"Seed {seed} does not have a complete compatible checkpoint.")
    paths = _seed_paths(config, seed)
    return (
        pd.read_csv(paths["step"]),
        pd.read_csv(paths["pairs"]),
        pd.read_csv(paths["final"]),
    )


def _active_match_targets(step_table: pd.DataFrame, config: ExperimentConfig) -> Dict[int, Dict[str, float]]:
    active = step_table.loc[step_table["phase"].eq("intervention")]
    targets = {
        int(row.step): {
            "activation_delta_rms_ratio": float(row.activation_delta_rms_ratio),
            "logit_delta_rms_ratio": float(row.logit_delta_rms_ratio),
            "sample_mapping_corr": float(
                0.5 * (row.sample_mapping_corr_a + row.sample_mapping_corr_b)
            ),
        }
        for row in active.itertuples(index=False)
    }
    expected = set(range(config.intervention_start, config.intervention_end + 1))
    if set(targets) != expected:
        raise RuntimeError(
            "The active branch did not produce a complete per-step matching profile."
        )
    return targets


def _run_one_seed(
    config: ExperimentConfig,
    seed: int,
    data: MNISTTensors,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one seed with a shared untreated continuation and paired controls."""

    device = resolve_device(config.device)
    schedule = make_batch_schedule(
        data.train_pool_indices,
        config.total_steps,
        config.batch_size,
        derived_seed(seed, "minibatches"),
    )
    print(f"[scout] seed={seed}: reference training and pair selection", flush=True)
    warm_state, fc_by_step, valid_by_step = _scout_and_checkpoint(
        seed, data, schedule, config, device
    )
    pair_table = select_fc_pairs(fc_by_step, valid_by_step, config, seed)
    print(
        f"[scout] seed={seed}: selected {config.n_pairs_per_stratum} high and "
        f"{config.n_pairs_per_stratum} low pairs",
        flush=True,
    )

    started = time.time()
    untreated_steps, untreated_final = run_shared_untreated_branch(
        warm_state, seed, pair_table, data, schedule, config, device
    )
    step_frames: List[pd.DataFrame] = [untreated_steps]
    final_frames: List[pd.DataFrame] = [untreated_final]
    trained_total = 1 + len(pair_table) * 3
    completed = 1
    print(
        f"[seed {seed}: {completed}/{trained_total}] shared untreated branch complete",
        flush=True,
    )

    for pair_row in pair_table.to_dict(orient="records"):
        pair_id = str(pair_row["pair_id"])
        untreated_pair = untreated_steps.loc[untreated_steps["pair_id"].eq(pair_id)]
        untreated_reference = build_untreated_fc_reference(untreated_pair, config)

        active_steps, active_final = run_branch(
            warm_state,
            seed,
            pair_row,
            "active_rank",
            data,
            schedule,
            config,
            device,
            untreated_fc_reference=untreated_reference,
        )
        step_frames.append(active_steps)
        final_frames.append(pd.DataFrame([active_final]))
        completed += 1
        match_targets = _active_match_targets(active_steps, config)

        for condition in ("independent_rank", "corr_preserving_rank"):
            control_steps, control_final = run_branch(
                warm_state,
                seed,
                pair_row,
                condition,
                data,
                schedule,
                config,
                device,
                match_targets=match_targets,
            )
            step_frames.append(control_steps)
            final_frames.append(pd.DataFrame([control_final]))
            completed += 1
            elapsed = time.time() - started
            eta = elapsed / completed * (trained_total - completed)
            print(
                f"[seed {seed}: {completed}/{trained_total}] pair={pair_id} "
                f"condition={condition}; ETA={eta / 60:.1f} min",
                flush=True,
            )

    combined_steps = attach_untreated_reference(
        pd.concat(step_frames, ignore_index=True)
    )
    return (
        combined_steps,
        pair_table.reset_index(drop=True),
        pd.concat(final_frames, ignore_index=True),
    )


def _check_output_compatibility(config: ExperimentConfig, overwrite: bool) -> None:
    output = Path(config.output_dir)
    manifest_path = output / "multiseed_run_manifest.json"
    if overwrite:
        return
    if not manifest_path.exists():
        per_seed = output / "per_seed"
        if per_seed.exists() and any(per_seed.glob("seed_*")):
            raise RuntimeError(
                "The output directory contains per-seed results but no compatible "
                "manifest. Choose a new output directory or explicitly overwrite."
            )
        return
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("analysis_signature") != analysis_signature(config):
        raise RuntimeError(
            "The output directory contains a different experiment configuration. "
            "Choose a new directory or explicitly overwrite it."
        )


def run_resumable_experiment(
    config: ExperimentConfig,
    data: Optional[MNISTTensors] = None,
    download: bool = True,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run or resume at seed boundaries, then save paper-level tables."""

    config.validate()
    _check_output_compatibility(config, overwrite=overwrite)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis_signature": analysis_signature(config),
        "requested_seeds": list(config.seeds),
        "config": asdict(config),
        "workload": estimate_workload(config),
        "shared_untreated_branch": True,
    }
    (output / "multiseed_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    data = data or load_mnist_tensors(config, download=download)

    step_frames: List[pd.DataFrame] = []
    pair_frames: List[pd.DataFrame] = []
    final_frames: List[pd.DataFrame] = []
    for seed in config.seeds:
        paths = _seed_paths(config, seed)
        if not overwrite and _seed_is_complete(config, seed):
            print(f"[resume] seed={seed}: loading completed checkpoint", flush=True)
            step, pairs, final = _load_seed_result(config, seed)
        else:
            paths["directory"].mkdir(parents=True, exist_ok=True)
            if overwrite and paths["complete"].exists():
                paths["complete"].unlink()
            try:
                step, pairs, final = _run_one_seed(config, seed, data)
                _write_seed_result(config, seed, step, pairs, final)
            except Exception as exc:
                failure = {
                    "seed": seed,
                    "analysis_signature": analysis_signature(config),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                paths["error"].write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                raise
        step_frames.append(step)
        pair_frames.append(pairs)
        final_frames.append(final)

    step_table = pd.concat(step_frames, ignore_index=True)
    pair_table = pd.concat(pair_frames, ignore_index=True)
    final_table = pd.concat(final_frames, ignore_index=True)
    run_summary = summarize_runs(step_table, final_table, config)
    seed_summary = seed_level_summary(run_summary)
    contrasts = primary_contrasts(seed_summary, config)
    save_results(
        step_table,
        pair_table,
        final_table,
        run_summary,
        seed_summary,
        contrasts,
        config,
    )
    validate_inferential_structure(pair_table, seed_summary, config)
    return step_table, pair_table, final_table, run_summary, seed_summary, contrasts


def validate_inferential_structure(
    pair_table: pd.DataFrame,
    seed_summary: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Fail if pairs rather than seeds could accidentally become replicates."""

    expected_pairs = config.n_pairs_per_stratum
    counts = pair_table.groupby(["seed", "stratum"])["pair_id"].nunique()
    if not counts.eq(expected_pairs).all():
        raise AssertionError("Not every seed has the requested number of pairs.")
    expected_rows = len(config.seeds) * 2 * len(config.conditions)
    if len(seed_summary) != expected_rows:
        raise AssertionError("Seed-level table has an unexpected row count.")
    if seed_summary["n_pairs"].min() != expected_pairs:
        raise AssertionError("Pair averaging within seed is incomplete.")


def load_complete_experiment(
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a completed aggregate run and verify its inferential structure."""

    output = Path(config.output_dir)
    paths = {key: output / name for key, name in RESULT_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError("Missing completed output files:\n- " + "\n- ".join(missing))
    tables = (
        pd.read_csv(paths["step"]),
        pd.read_csv(paths["pairs"]),
        pd.read_csv(paths["final"]),
        pd.read_csv(paths["runs"]),
        pd.read_csv(paths["seeds"]),
        pd.read_csv(paths["contrasts"]),
    )
    validate_inferential_structure(tables[1], tables[4], config)
    return tables


__all__ = [
    "ExperimentConfig",
    "paper_multiseed_config",
    "smoke_multiseed_config",
    "estimate_workload",
    "run_resumable_experiment",
    "load_complete_experiment",
    "validate_inferential_structure",
    "manipulation_qc",
    "qc_assertion_table",
    "make_seed_effect_figure",
    "make_reference_effect_figure",
    "make_qc_figure",
    "save_publication_figure",
]
