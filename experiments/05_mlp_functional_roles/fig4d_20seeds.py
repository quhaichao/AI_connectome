"""Twenty-seed replication of the early/late neuron masking analysis.

This is an independent, notebook-style Python script (``# %%`` cells).  It
does not run the t-SNE, sample-difficulty, receptive-field, or frequency
analyses from the original notebook.

Experiment implemented here
---------------------------
1. Train 20 independently initialized two-hidden-layer ReLU MLPs on MNIST,
   Fashion-MNIST or CIFAR-10.
2. Within each seed, reproduce the existing FC-rise pair classification over
   legacy checkpoint indices 0-260 and select 15 disjoint early-rise and 15
   disjoint late-rise neurons.
3. At checkpoint indices 260, 280, 300, 400, 600, 800 and 1100, mask all 15
   neurons in each group by zeroing their classifier-output columns.
4. Record masked accuracy minus the intact checkpoint accuracy.  The
   independent replication unit is the training seed (n=20), not repeated
   subsets drawn from one fixed neuron pool.
5. Export seed-level source data, summaries, paired tests, the current
   multi-step display and a final-checkpoint Fig. 4d-style display.

The dataset and FC-rise threshold are command-line parameters.  The
``--supplementary-suite`` preset runs the four configurations needed for the
current main and Extended Data masking figures:

* MNIST, rise threshold 0.8;
* MNIST, rise threshold 0.5;
* Fashion-MNIST, rise threshold 0.8;
* CIFAR-10, rise threshold 0.8.

The legacy notebook stores its first post-update state at index 0.  To keep the
requested downstream workflow unchanged, ``Step`` below uses the same
zero-based checkpoint index.  ``Optimizer_Update`` is also exported and equals
``Step + 1`` so that this convention is explicit.

Example
-------
    python fig4d_20seeds.py
    python fig4d_20seeds.py --supplementary-suite
    python fig4d_20seeds.py --dataset fashion-mnist --rise-threshold 0.8

Resume is automatic: a completed per-seed CSV is reused.  Pass ``--overwrite``
to rerun every seed.
"""

# %% Imports and experiment contract
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t, ttest_rel
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "mnist"
    input_size: int = 784
    num_classes: int = 10
    seeds: tuple[int, ...] = tuple(range(20))
    checkpoints: tuple[int, ...] = (260, 280, 300, 400, 600, 800, 1100)
    batch_size: int = 256
    test_batch_size: int = 512
    learning_rate: float = 0.05
    hidden_size: int = 100
    n_group_neurons: int = 15
    fc_samples_per_class: int = 100
    fc_eval_batch_size: int = 256
    fc_sample_seed: int = 20260820
    low_threshold: float = 0.5
    high_threshold: float = 0.8
    pre_window: tuple[int, int] = (0, 20)
    rise_window: tuple[int, int] = (20, 260)
    smooth_sigma: float = 2.0
    min_rise_amplitude: float = 0.3
    min_total_pairs: int = 5


GROUP_ORDER = ("Early FC-rise", "Late FC-rise")
GROUP_COLORS = {
    "Early FC-rise": "#c03d3e",
    "Late FC-rise": "#3274a1",
}

DATASET_DISPLAY_NAMES = {
    "mnist": "MNIST",
    "fashion-mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
}

DATASET_INPUT_SIZES = {
    "mnist": 784,
    "fashion-mnist": 784,
    "cifar10": 3072,
}


def condition_signature(config: ExperimentConfig) -> str:
    """Hash every setting that can change training or neuron classification."""
    payload = {
        "dataset": config.dataset,
        "input_size": config.input_size,
        "num_classes": config.num_classes,
        "checkpoints": config.checkpoints,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "hidden_size": config.hidden_size,
        "n_group_neurons": config.n_group_neurons,
        "fc_samples_per_class": config.fc_samples_per_class,
        "fc_sample_seed": config.fc_sample_seed,
        "low_threshold": config.low_threshold,
        "high_threshold": config.high_threshold,
        "pre_window": config.pre_window,
        "rise_window": config.rise_window,
        "smooth_sigma": config.smooth_sigma,
        "min_rise_amplitude": config.min_rise_amplitude,
        "min_total_pairs": config.min_total_pairs,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def configure_plot_style() -> None:
    """Publication-safe defaults; editable text is retained in PDF/SVG."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# %% Model and deterministic data handling
class MLP(nn.Module):
    """The same input-100-100-10 ReLU MLP used by the existing analyses."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 100,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.fc1, self.fc2, self.fc3):
            init.normal_(layer.weight, mean=0.0, std=0.01)
            init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def second_hidden_activation(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is False.")
    return device


def canonical_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "mnist": "mnist",
        "fashion": "fashion-mnist",
        "fashionmnist": "fashion-mnist",
        "fashion-mnist": "fashion-mnist",
        "cifar": "cifar10",
        "cifar-10": "cifar10",
        "cifar10": "cifar10",
    }
    if normalized not in aliases:
        valid = ", ".join(DATASET_DISPLAY_NAMES.values())
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {valid}.")
    return aliases[normalized]


def load_image_dataset(
    dataset_name: str,
    data_root: Path,
    download: bool,
) -> tuple[Dataset, Dataset]:
    """Load one supported dataset with the normalization used in the notebook."""
    dataset_name = canonical_dataset_name(dataset_name)
    if dataset_name == "mnist":
        dataset_class = torchvision.datasets.MNIST
        normalization = ((0.1307,), (0.3081,))
    elif dataset_name == "fashion-mnist":
        dataset_class = torchvision.datasets.FashionMNIST
        normalization = ((0.2860,), (0.3530,))
    elif dataset_name == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
        normalization = (
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        )
    else:  # pragma: no cover - guarded by canonical_dataset_name
        raise AssertionError(f"Unhandled dataset: {dataset_name}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
    )
    train_dataset = dataset_class(
        root=str(data_root), train=True, transform=transform, download=download
    )
    test_dataset = dataset_class(
        root=str(data_root), train=False, transform=transform, download=download
    )
    return train_dataset, test_dataset


def make_train_loader(
    train_dataset: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def make_test_loader(
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def make_fixed_balanced_fc_panel(
    train_dataset: Dataset,
    samples_per_class: int,
    seed: int,
    num_classes: int,
) -> tuple[torch.Tensor, list[int]]:
    """Select one class-balanced panel and reuse it for every seed and step.

    Reusing the same 1,000 images matches the manuscript's stated fixed-panel
    analysis and prevents evaluation-sample changes from masquerading as FC
    dynamics.
    """
    targets = np.asarray(train_dataset.targets)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in range(num_classes):
        candidates = np.flatnonzero(targets == class_id)
        if len(candidates) < samples_per_class:
            raise ValueError(f"Class {class_id} has only {len(candidates)} samples.")
        selected.extend(rng.choice(candidates, size=samples_per_class, replace=False).tolist())
    images = [train_dataset[index][0] for index in selected]
    return torch.stack(images, dim=0), selected


# %% FC trajectory, pair classification and neuron selection
@torch.inference_mode()
def second_layer_fc(
    model: MLP,
    fixed_images_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    activation_batches: list[np.ndarray] = []
    for start in range(0, len(fixed_images_cpu), batch_size):
        batch = fixed_images_cpu[start : start + batch_size].to(device, non_blocking=True)
        activation_batches.append(model.second_hidden_activation(batch).cpu().numpy())
    if was_training:
        model.train()
    activation_matrix = np.concatenate(activation_batches, axis=0)
    fc = np.corrcoef(activation_matrix.T)
    return np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def get_window_indices(sampled_steps: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    indices = np.where((sampled_steps >= window[0]) & (sampled_steps <= window[1]))[0]
    if len(indices) == 0:
        raise ValueError(f"No sampled steps were found in window {window}.")
    return indices


def classify_rising_pairs_by_median_time(
    fc_trajectories: np.ndarray,
    sampled_steps: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Reproduce the existing signed-FC threshold and median-split procedure."""
    n_neurons = fc_trajectories.shape[1]
    pre_indices = get_window_indices(sampled_steps, config.pre_window)
    rise_indices = get_window_indices(sampled_steps, config.rise_window)
    fc_smooth = gaussian_filter1d(fc_trajectories, sigma=config.smooth_sigma, axis=0)

    rising_pairs: list[tuple[int, int]] = []
    rise_times: list[float] = []
    rise_slopes: list[float] = []
    peak_times: list[float] = []
    peak_values: list[float] = []
    pre_values: list[float] = []
    rise_amplitudes: list[float] = []

    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            pre_mean = float(fc_smooth[pre_indices, i, j].mean())
            trajectory = fc_smooth[rise_indices, i, j]
            peak = float(trajectory.max())
            if (
                pre_mean >= config.low_threshold
                or peak < config.high_threshold
                or peak - pre_mean < config.min_rise_amplitude
            ):
                continue

            crossings = np.where(trajectory >= config.high_threshold)[0]
            if len(crossings) == 0:
                continue

            crossing_local = int(crossings[0])
            crossing_global = int(rise_indices[crossing_local])
            crossing_time = float(sampled_steps[crossing_global])
            if crossing_global > 0:
                previous = crossing_global - 1
                delta_t = sampled_steps[crossing_global] - sampled_steps[previous]
                slope = (
                    float(fc_smooth[crossing_global, i, j] - fc_smooth[previous, i, j]) / delta_t
                    if delta_t > 0
                    else 0.0
                )
            else:
                delta_t = sampled_steps[1] - sampled_steps[0]
                slope = (
                    float(fc_smooth[1, i, j] - fc_smooth[0, i, j]) / delta_t
                    if delta_t > 0
                    else 0.0
                )

            peak_local = int(np.argmax(trajectory))
            rising_pairs.append((i, j))
            rise_times.append(crossing_time)
            rise_slopes.append(slope)
            peak_times.append(float(sampled_steps[rise_indices[peak_local]]))
            peak_values.append(peak)
            pre_values.append(pre_mean)
            rise_amplitudes.append(peak - pre_mean)

    if not rising_pairs:
        raise ValueError("No rising pairs met the fixed FC-rise criteria.")

    rise_times_array = np.asarray(rise_times)
    rise_slopes_array = np.asarray(rise_slopes)
    peak_times_array = np.asarray(peak_times)
    peak_values_array = np.asarray(peak_values)
    rise_amplitudes_array = np.asarray(rise_amplitudes)
    median_time = float(np.median(rise_times_array))

    tied_mask = np.isclose(rise_times_array, median_time, rtol=0.0, atol=1e-12)
    early_indices = np.where((rise_times_array < median_time) & ~tied_mask)[0].tolist()
    late_indices = np.where((rise_times_array > median_time) & ~tied_mask)[0].tolist()
    tied_indices = np.where(tied_mask)[0].tolist()
    tied_indices.sort(
        key=lambda k: (
            -rise_slopes_array[k],
            peak_times_array[k],
            -peak_values_array[k],
            -rise_amplitudes_array[k],
            rising_pairs[k][0],
            rising_pairs[k][1],
        )
    )

    target_early_count = (len(rising_pairs) + 1) // 2
    n_tied_to_early = max(0, min(target_early_count - len(early_indices), len(tied_indices)))
    early_indices.extend(tied_indices[:n_tied_to_early])
    late_indices.extend(tied_indices[n_tied_to_early:])

    sort_key = lambda k: (
        rise_times_array[k],
        -rise_slopes_array[k],
        peak_times_array[k],
        -peak_values_array[k],
        -rise_amplitudes_array[k],
        rising_pairs[k][0],
        rising_pairs[k][1],
    )
    early_indices.sort(key=sort_key)
    late_indices.sort(key=sort_key)

    return {
        "rising_pairs": rising_pairs,
        "early_rise_pairs": [rising_pairs[k] for k in early_indices],
        "late_rise_pairs": [rising_pairs[k] for k in late_indices],
        "rise_times": rise_times_array,
        "median_time": median_time,
        "pairs_tied_at_median": len(tied_indices),
    }


def select_top_timing_neurons(
    early_pairs: Sequence[tuple[int, int]],
    late_pairs: Sequence[tuple[int, int]],
    config: ExperimentConfig,
) -> dict[str, np.ndarray]:
    """Select 15 disjoint neurons per group using the existing timing-bias rule."""
    early_count = np.zeros(config.hidden_size)
    late_count = np.zeros(config.hidden_size)
    total_count = np.zeros(config.hidden_size)
    for i, j in early_pairs:
        early_count[[i, j]] += 1
        total_count[[i, j]] += 1
    for i, j in late_pairs:
        late_count[[i, j]] += 1
        total_count[[i, j]] += 1

    early_ratio = early_count / (total_count + 1e-8)
    late_ratio = late_count / (total_count + 1e-8)
    timing_bias = early_ratio - late_ratio
    valid = np.where(total_count >= config.min_total_pairs)[0]
    early_order = sorted(valid.tolist(), key=lambda x: (timing_bias[x], total_count[x]), reverse=True)
    late_order = sorted(valid.tolist(), key=lambda x: (timing_bias[x], -total_count[x]))

    early_nodes = np.asarray(early_order[: config.n_group_neurons], dtype=int)
    early_set = set(early_nodes.tolist())
    late_nodes = np.asarray(
        [node for node in late_order if node not in early_set][: config.n_group_neurons], dtype=int
    )
    if len(early_nodes) != config.n_group_neurons or len(late_nodes) != config.n_group_neurons:
        raise ValueError(
            "The fixed selection rule did not yield 15 disjoint neurons per group: "
            f"early={len(early_nodes)}, late={len(late_nodes)}, valid={len(valid)}."
        )
    return {
        "early_nodes": early_nodes,
        "late_nodes": late_nodes,
        "early_count": early_count,
        "late_count": late_count,
        "total_count": total_count,
        "timing_bias": timing_bias,
    }


# %% Training, checkpoint masking and seed-level records
def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one_seed(
    seed: int,
    train_dataset: Dataset,
    fixed_fc_images: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, torch.Tensor]]]:
    """Train through checkpoint 1100 and retain only data needed here."""
    set_seed(seed)
    train_loader = make_train_loader(
        train_dataset, config.batch_size, seed, num_workers, device
    )
    model = MLP(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
    ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    max_checkpoint = max(config.checkpoints)
    max_fc_step = config.rise_window[1]
    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {}
    fc_trajectories: list[np.ndarray] = []
    sampled_steps: list[int] = []
    state_index = -1

    while state_index < max_checkpoint:
        for images, targets in train_loader:
            model.train()
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            state_index += 1

            if state_index <= max_fc_step:
                fc_trajectories.append(
                    second_layer_fc(
                        model, fixed_fc_images, device, config.fc_eval_batch_size
                    )
                )
                sampled_steps.append(state_index)

            if state_index in config.checkpoints:
                checkpoint_states[state_index] = clone_state_dict_to_cpu(model)

            if state_index >= max_checkpoint:
                break

    missing = sorted(set(config.checkpoints) - set(checkpoint_states))
    if missing:
        raise RuntimeError(f"Missing trained checkpoints: {missing}")
    return np.stack(fc_trajectories), np.asarray(sampled_steps), checkpoint_states


@torch.inference_mode()
def evaluate_accuracy(model: MLP, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    return 100.0 * correct / total


def model_from_state(
    state: dict[str, torch.Tensor], config: ExperimentConfig, device: torch.device
) -> MLP:
    model = MLP(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_output_masked_model(model: MLP, nodes: Sequence[int]) -> MLP:
    """Match the original lesion: zero classifier columns from hidden layer 2."""
    masked_model = copy.deepcopy(model)
    masked_model.eval()
    with torch.no_grad():
        masked_model.fc3.weight[:, list(nodes)] = 0.0
    return masked_model


def evaluate_seed_checkpoints(
    seed: int,
    checkpoint_states: dict[int, dict[str, torch.Tensor]],
    early_nodes: np.ndarray,
    late_nodes: np.ndarray,
    rise_result: dict[str, object],
    test_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    groups = (
        ("Early FC-rise", early_nodes),
        ("Late FC-rise", late_nodes),
    )
    for step in config.checkpoints:
        intact_model = model_from_state(checkpoint_states[step], config, device)
        baseline_accuracy = evaluate_accuracy(intact_model, test_loader, device)
        for group, nodes in groups:
            masked_model = make_output_masked_model(intact_model, nodes).to(device)
            masked_accuracy = evaluate_accuracy(masked_model, test_loader, device)
            records.append(
                {
                    "Condition_Signature": condition_signature(config),
                    "Dataset": config.dataset,
                    "FC_Rise_Threshold": config.high_threshold,
                    "FC_Pre_Threshold": config.low_threshold,
                    "Min_Rise_Amplitude": config.min_rise_amplitude,
                    "Seed": seed,
                    "Step": step,
                    "Optimizer_Update": step + 1,
                    "Group": group,
                    "Baseline_Accuracy": baseline_accuracy,
                    "Masked_Accuracy": masked_accuracy,
                    "Accuracy_Change": masked_accuracy - baseline_accuracy,
                    "N_Masked": len(nodes),
                    "Masked_Neurons": json.dumps([int(node) for node in nodes]),
                    "N_Rising_Pairs": len(rise_result["rising_pairs"]),
                    "N_Early_Pairs": len(rise_result["early_rise_pairs"]),
                    "N_Late_Pairs": len(rise_result["late_rise_pairs"]),
                    "Median_Rise_Time": float(rise_result["median_time"]),
                }
            )
    return pd.DataFrame(records)


def run_one_seed(
    seed: int,
    train_dataset: Dataset,
    fixed_fc_images: torch.Tensor,
    test_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    num_workers: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\n=== Seed {seed}: training and FC classification ===", flush=True)
    fc_trajectories, sampled_steps, checkpoint_states = train_one_seed(
        seed,
        train_dataset,
        fixed_fc_images,
        config,
        device,
        num_workers,
    )
    rise_result = classify_rising_pairs_by_median_time(
        fc_trajectories, sampled_steps, config
    )
    neuron_result = select_top_timing_neurons(
        rise_result["early_rise_pairs"], rise_result["late_rise_pairs"], config
    )
    early_nodes = neuron_result["early_nodes"]
    late_nodes = neuron_result["late_nodes"]
    print(
        f"Seed {seed}: rising pairs={len(rise_result['rising_pairs'])}, "
        f"median rise={rise_result['median_time']}, "
        f"early={early_nodes.tolist()}, late={late_nodes.tolist()}",
        flush=True,
    )

    result = evaluate_seed_checkpoints(
        seed,
        checkpoint_states,
        early_nodes,
        late_nodes,
        rise_result,
        test_loader,
        config,
        device,
    )
    metadata = {
        "condition_signature": condition_signature(config),
        "dataset": config.dataset,
        "input_size": config.input_size,
        "fc_rise_threshold": config.high_threshold,
        "fc_pre_threshold": config.low_threshold,
        "min_rise_amplitude": config.min_rise_amplitude,
        "n_group_neurons": config.n_group_neurons,
        "seed": seed,
        "early_nodes": early_nodes.tolist(),
        "late_nodes": late_nodes.tolist(),
        "n_rising_pairs": len(rise_result["rising_pairs"]),
        "n_early_pairs": len(rise_result["early_rise_pairs"]),
        "n_late_pairs": len(rise_result["late_rise_pairs"]),
        "median_rise_time": float(rise_result["median_time"]),
        "pairs_tied_at_median": int(rise_result["pairs_tied_at_median"]),
    }
    return result, metadata


# %% Summary statistics and the requested current-style figure
def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def make_summary_table(seed_level: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_level.groupby(["Step", "Group"], observed=True)["Accuracy_Change"]
        .agg(Mean="mean", SD="std", N="count")
        .reset_index()
    )
    summary["SEM"] = summary["SD"] / np.sqrt(summary["N"])
    summary["CI95_Low"] = summary["Mean"] - t.ppf(0.975, summary["N"] - 1) * summary["SEM"]
    summary["CI95_High"] = summary["Mean"] + t.ppf(0.975, summary["N"] - 1) * summary["SEM"]
    return summary


def paired_tests(seed_level: pd.DataFrame, checkpoints: Sequence[int]) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for step in checkpoints:
        wide = (
            seed_level.loc[seed_level["Step"] == step]
            .pivot(index="Seed", columns="Group", values="Accuracy_Change")
            .dropna(subset=list(GROUP_ORDER))
        )
        early = wide["Early FC-rise"].to_numpy()
        late = wide["Late FC-rise"].to_numpy()
        differences = early - late
        test = ttest_rel(early, late)
        n = len(differences)
        mean_difference = float(differences.mean())
        sd_difference = float(differences.std(ddof=1))
        sem_difference = sd_difference / np.sqrt(n)
        critical = float(t.ppf(0.975, n - 1))
        records.append(
            {
                "Step": step,
                "N_Paired_Seeds": n,
                "Mean_Difference_Early_minus_Late": mean_difference,
                "SD_Difference": sd_difference,
                "CI95_Low": mean_difference - critical * sem_difference,
                "CI95_High": mean_difference + critical * sem_difference,
                "Paired_t": float(test.statistic),
                "df": n - 1,
                "P_Raw": float(test.pvalue),
            }
        )
    result = pd.DataFrame(records)
    result["P_Holm_Across_Steps"] = holm_adjust(result["P_Raw"])
    return result


def plot_ablation_across_steps(
    seed_level: pd.DataFrame,
    checkpoints: Sequence[int],
    output_dir: Path,
) -> None:
    """Grouped mean +/- SD bars with independent seed points, as currently used."""
    configure_plot_style()
    x = np.arange(len(checkpoints), dtype=float)
    width = 0.34
    offsets = {"Early FC-rise": -width / 2, "Late FC-rise": width / 2}
    rng = np.random.default_rng(20260820)

    fig, ax = plt.subplots(figsize=(180 / 25.4, 95 / 25.4))
    for group in GROUP_ORDER:
        group_df = seed_level.loc[seed_level["Group"] == group]
        means: list[float] = []
        sds: list[float] = []
        for step in checkpoints:
            values = group_df.loc[group_df["Step"] == step, "Accuracy_Change"].to_numpy()
            means.append(float(values.mean()))
            sds.append(float(values.std(ddof=1)))
        positions = x + offsets[group]
        ax.bar(
            positions,
            means,
            width=width,
            yerr=sds,
            capsize=2.0,
            color=GROUP_COLORS[group],
            alpha=0.80,
            edgecolor="black",
            linewidth=0.6,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
            label=group,
            zorder=2,
        )
        for step_index, step in enumerate(checkpoints):
            values = group_df.loc[group_df["Step"] == step].sort_values("Seed")["Accuracy_Change"].to_numpy()
            jitter = rng.uniform(-0.045, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), positions[step_index]) + jitter,
                values,
                s=9,
                color="black",
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.75, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(step) for step in checkpoints])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("FC timing neuron masking across training steps")
    ax.legend(title="", loc="best")
    fig.tight_layout()

    stem = output_dir / "fc_timing_ablation_20_seeds_across_steps"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_ablation_single_step(
    seed_level: pd.DataFrame,
    tests: pd.DataFrame,
    step: int,
    output_dir: Path,
    config: ExperimentConfig,
    show_paired_lines: bool = True,
) -> None:
    """Fig. 4d-style plot for one checkpoint using paired independent seeds."""
    configure_plot_style()
    step_data = seed_level.loc[
        (seed_level["Step"] == step) & seed_level["Group"].isin(GROUP_ORDER)
    ].copy()
    wide = (
        step_data.pivot(index="Seed", columns="Group", values="Accuracy_Change")
        .dropna(subset=list(GROUP_ORDER))
        .loc[:, list(GROUP_ORDER)]
        .sort_index()
    )
    if len(wide) < 2:
        raise ValueError(f"Step {step} has only {len(wide)} complete paired seed(s).")

    early = wide["Early FC-rise"].to_numpy(dtype=float)
    late = wide["Late FC-rise"].to_numpy(dtype=float)
    means = np.asarray([early.mean(), late.mean()])
    sds = np.asarray([early.std(ddof=1), late.std(ddof=1)])

    test_row = tests.loc[tests["Step"] == step]
    if len(test_row) != 1:
        raise ValueError(f"Expected one paired-test row for step {step}, found {len(test_row)}.")
    raw_p = float(test_row.iloc[0]["P_Raw"])
    holm_p = float(test_row.iloc[0]["P_Holm_Across_Steps"])
    stars = significance_stars(holm_p)

    source_data = wide.reset_index().rename(
        columns={
            "Early FC-rise": "Early_Accuracy_Change",
            "Late FC-rise": "Late_Accuracy_Change",
        }
    )
    source_data.insert(0, "FC_Rise_Threshold", config.high_threshold)
    source_data.insert(0, "Dataset", config.dataset)
    source_data.to_csv(output_dir / f"fig4d_step{step}_source_data.csv", index=False)

    x = np.asarray([0.0, 1.0])
    rng = np.random.default_rng(20260820)
    seed_jitter = rng.uniform(-0.075, 0.075, size=len(wide))
    fig, ax = plt.subplots(figsize=(55 / 25.4, 62 / 25.4))
    ax.bar(
        x,
        means,
        yerr=sds,
        width=0.62,
        color=[GROUP_COLORS["Early FC-rise"], GROUP_COLORS["Late FC-rise"]],
        alpha=0.80,
        edgecolor="black",
        linewidth=0.7,
        capsize=2.5,
        error_kw={"linewidth": 0.9, "capthick": 0.9},
        zorder=2,
    )

    for seed_index, (_, row) in enumerate(wide.iterrows()):
        paired_values = row[list(GROUP_ORDER)].to_numpy(dtype=float)
        paired_x = x + seed_jitter[seed_index]
        if show_paired_lines:
            ax.plot(
                paired_x,
                paired_values,
                color="#9a9a9a",
                alpha=0.38,
                linewidth=0.55,
                zorder=3,
            )
        ax.scatter(
            paired_x,
            paired_values,
            s=12,
            color="black",
            alpha=0.70,
            linewidths=0,
            zorder=4,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.65, zorder=1)
    all_values = wide.to_numpy(dtype=float)
    data_min = min(float(all_values.min()), 0.0)
    data_max = max(float(all_values.max()), 0.0)
    y_span = max(data_max - data_min, 1.0)
    bracket_y = data_max + 0.055 * y_span
    bracket_height = 0.025 * y_span
    ax.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color="black",
        linewidth=0.8,
        clip_on=False,
    )
    ax.text(
        0.5,
        bracket_y + bracket_height + 0.015 * y_span,
        stars,
        ha="center",
        va="bottom",
        fontsize=7,
    )

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(
        data_min - 0.08 * y_span,
        bracket_y + bracket_height + 0.16 * y_span,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Early-change", "Late-change"])
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy change (%)")
    ax.set_title("Accuracy change")
    ax.tick_params(axis="both", width=0.7, length=3)
    fig.tight_layout()

    stem = output_dir / f"fig4d_step{step}_20_seeds"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    print(
        f"Step {step} single plot: n={len(wide)} paired seeds, "
        f"raw P={raw_p:.6g}, Holm-adjusted P={holm_p:.6g} ({stars})"
    )


# %% Command-line orchestration with safe resume
def parse_seed_spec(spec: str) -> tuple[int, ...]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start, end = (int(value) for value in spec.split("-", maxsplit=1))
        if end < start:
            raise ValueError("Seed range end must be greater than or equal to its start.")
        return tuple(range(start, end + 1))
    seeds = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not seeds:
        raise ValueError("No seeds were provided.")
    return seeds


def parse_checkpoint_spec(spec: str) -> tuple[int, ...]:
    checkpoints = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not checkpoints:
        raise ValueError("No checkpoints were provided.")
    if any(step < 0 for step in checkpoints):
        raise ValueError("Checkpoints must be non-negative.")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Checkpoints must not contain duplicates.")
    return checkpoints


def threshold_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def condition_slug(dataset: str, rise_threshold: float) -> str:
    return f"{dataset.replace('-', '_')}_rise_{threshold_slug(rise_threshold)}"


def validate_config(config: ExperimentConfig, single_step: int) -> None:
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Seeds must not contain duplicates.")
    if not 0.0 <= config.low_threshold <= 1.0:
        raise ValueError("--pre-threshold must lie between 0 and 1.")
    if not 0.0 <= config.high_threshold <= 1.0:
        raise ValueError("--rise-threshold must lie between 0 and 1.")
    if config.min_rise_amplitude <= 0:
        raise ValueError("--min-rise-amplitude must be positive.")
    if config.n_group_neurons <= 0 or 2 * config.n_group_neurons > config.hidden_size:
        raise ValueError("The two disjoint neuron groups do not fit in the hidden layer.")
    if config.fc_samples_per_class <= 1:
        raise ValueError("--fc-samples-per-class must be greater than 1.")
    if single_step not in config.checkpoints:
        raise ValueError(
            f"Single-step plot checkpoint {single_step} is not present in --checkpoints."
        )


def ensure_output_compatible(
    output_dir: Path,
    config: ExperimentConfig,
    overwrite: bool,
) -> None:
    """Prevent a custom condition from silently replacing another condition."""
    if overwrite:
        return
    config_path = output_dir / "experiment_config.json"
    if config_path.exists():
        try:
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            stored_config = ExperimentConfig(**stored)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot verify the existing output configuration in {config_path}. "
                "Choose a new --output-dir or pass --overwrite explicitly."
            ) from exc
        if condition_signature(stored_config) != condition_signature(config):
            raise RuntimeError(
                f"Output directory {output_dir} contains a different dataset or analysis "
                "configuration. Choose a condition-specific --output-dir; use --overwrite "
                "only if replacing those results is intentional."
            )
        return

    per_seed_dir = output_dir / "per_seed"
    if per_seed_dir.exists() and any(per_seed_dir.glob("seed_*_ablation.csv")):
        raise RuntimeError(
            f"Output directory {output_dir} contains seed results but no readable "
            "experiment_config.json. Choose a new --output-dir or pass --overwrite "
            "explicitly."
        )


def annotate_condition_columns(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Add traceability fields when reusing results from the earlier script."""
    frame = frame.copy()
    frame["Condition_Signature"] = condition_signature(config)
    frame["Dataset"] = config.dataset
    frame["FC_Rise_Threshold"] = config.high_threshold
    frame["FC_Pre_Threshold"] = config.low_threshold
    frame["Min_Rise_Amplitude"] = config.min_rise_amplitude
    return frame


def seed_result_is_complete(path: Path, seed: int, config: ExperimentConfig) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required_columns = {"Seed", "Step", "Group", "N_Masked"}
    if not required_columns.issubset(frame.columns):
        return False
    expected = {(seed, step, group) for step in config.checkpoints for group in GROUP_ORDER}
    observed = {
        (int(row.Seed), int(row.Step), str(row.Group))
        for row in frame[["Seed", "Step", "Group"]].itertuples(index=False)
    }
    if observed != expected or not (frame["N_Masked"] == config.n_group_neurons).all():
        return False

    optional_checks = (
        (
            "Condition_Signature",
            lambda values: set(values.astype(str)) == {condition_signature(config)},
        ),
        ("Dataset", lambda values: set(values.astype(str)) == {config.dataset}),
        (
            "FC_Rise_Threshold",
            lambda values: np.isclose(values, config.high_threshold).all(),
        ),
        (
            "FC_Pre_Threshold",
            lambda values: np.isclose(values, config.low_threshold).all(),
        ),
        (
            "Min_Rise_Amplitude",
            lambda values: np.isclose(values, config.min_rise_amplitude).all(),
        ),
    )
    return all(check(frame[column]) for column, check in optional_checks if column in frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-19", help="Inclusive range (0-19) or comma list.")
    parser.add_argument(
        "--dataset",
        default="mnist",
        help="MNIST, Fashion-MNIST or CIFAR-10 (aliases are accepted).",
    )
    parser.add_argument(
        "--rise-threshold",
        type=float,
        default=0.8,
        help="FC value defining pair rise time (0.8 primary; 0.5 robustness).",
    )
    parser.add_argument(
        "--pre-threshold",
        type=float,
        default=0.5,
        help="Maximum allowed mean FC during updates 0-20.",
    )
    parser.add_argument("--min-rise-amplitude", type=float, default=0.3)
    parser.add_argument("--n-group-neurons", type=int, default=15)
    parser.add_argument("--min-total-pairs", type=int, default=5)
    parser.add_argument("--fc-samples-per-class", type=int, default=100)
    parser.add_argument(
        "--checkpoints",
        default="260,280,300,400,600,800,1100",
        help="Comma-separated zero-based checkpoint indices.",
    )
    parser.add_argument(
        "--single-step",
        type=int,
        default=1100,
        help="Checkpoint for the additional Fig. 4d-style plot.",
    )
    parser.add_argument(
        "--supplementary-suite",
        action="store_true",
        help=(
            "Run MNIST/0.8, MNIST/0.5, Fashion-MNIST/0.8 and CIFAR-10/0.8; "
            "each condition receives an isolated output subdirectory."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "fig4d_20seeds_output",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require the selected torchvision dataset to exist locally.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rerun completed seed files.")
    return parser


def run_condition(
    args: argparse.Namespace,
    dataset_name: str,
    rise_threshold: float,
    output_dir: Path,
) -> dict[str, object]:
    dataset_name = canonical_dataset_name(dataset_name)
    config = ExperimentConfig(
        dataset=dataset_name,
        input_size=DATASET_INPUT_SIZES[dataset_name],
        seeds=parse_seed_spec(args.seeds),
        checkpoints=parse_checkpoint_spec(args.checkpoints),
        n_group_neurons=args.n_group_neurons,
        fc_samples_per_class=args.fc_samples_per_class,
        low_threshold=args.pre_threshold,
        high_threshold=rise_threshold,
        min_rise_amplitude=args.min_rise_amplitude,
        min_total_pairs=args.min_total_pairs,
    )
    validate_config(config, args.single_step)
    device = resolve_device(args.device)
    output_dir = output_dir.resolve()
    ensure_output_compatible(output_dir, config, args.overwrite)
    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(
        f"Condition: {DATASET_DISPLAY_NAMES[config.dataset]}, "
        f"FC-rise threshold={config.high_threshold:g}"
    )
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Independent seeds: {list(config.seeds)}")
    print(f"Checkpoints: {list(config.checkpoints)}")
    print(f"Mask size: {config.n_group_neurons} neurons per group")
    print(f"Output: {output_dir}")

    train_dataset, test_dataset = load_image_dataset(
        config.dataset,
        args.data_root.resolve(),
        download=not args.no_download,
    )
    fixed_fc_images, fc_panel_indices = make_fixed_balanced_fc_panel(
        train_dataset,
        config.fc_samples_per_class,
        config.fc_sample_seed,
        config.num_classes,
    )
    (output_dir / "fixed_fc_panel_training_indices.json").write_text(
        json.dumps(
            {
                "dataset": config.dataset,
                "selection_seed": config.fc_sample_seed,
                "samples_per_class": config.fc_samples_per_class,
                "training_dataset_indices": fc_panel_indices,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    test_loader = make_test_loader(
        test_dataset, config.test_batch_size, args.num_workers, device
    )

    completed_frames: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    metadata_records: list[dict[str, object]] = []
    for seed in config.seeds:
        result_path = per_seed_dir / f"seed_{seed:03d}_ablation.csv"
        metadata_path = per_seed_dir / f"seed_{seed:03d}_metadata.json"
        if (
            not args.overwrite
            and metadata_path.exists()
            and seed_result_is_complete(result_path, seed, config)
        ):
            print(f"Seed {seed}: reusing completed result {result_path.name}")
            completed_frames.append(
                annotate_condition_columns(pd.read_csv(result_path), config)
            )
            if metadata_path.exists():
                metadata_records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue
        try:
            frame, metadata = run_one_seed(
                seed,
                train_dataset,
                fixed_fc_images,
                test_loader,
                config,
                device,
                args.num_workers,
            )
            frame.to_csv(result_path, index=False)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            completed_frames.append(frame)
            metadata_records.append(metadata)
        except Exception as exc:
            failure = {
                "seed": seed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            (per_seed_dir / f"seed_{seed:03d}_ERROR.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            print(f"Seed {seed} failed: {exc}", flush=True)

    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"{len(failures)} seed(s) failed. Successful seed files were kept for resume; "
            f"see {output_dir / 'failures.json'}."
        )
    if len(completed_frames) != len(config.seeds):
        raise RuntimeError("Not all requested independent seeds produced results.")

    seed_level = pd.concat(completed_frames, ignore_index=True)
    seed_level["Group"] = pd.Categorical(
        seed_level["Group"], categories=list(GROUP_ORDER), ordered=True
    )
    seed_level = seed_level.sort_values(["Step", "Group", "Seed"]).reset_index(drop=True)
    summary = make_summary_table(seed_level)
    tests = paired_tests(seed_level, config.checkpoints)

    seed_level.to_csv(output_dir / "seed_level_accuracy_changes.csv", index=False)
    summary.to_csv(output_dir / "accuracy_change_summary_mean_sd_ci.csv", index=False)
    tests.to_csv(output_dir / "paired_early_vs_late_tests.csv", index=False)
    (output_dir / "seed_classification_metadata.json").write_text(
        json.dumps(metadata_records, indent=2), encoding="utf-8"
    )
    plot_ablation_across_steps(seed_level, config.checkpoints, output_dir)
    plot_ablation_single_step(
        seed_level=seed_level,
        tests=tests,
        step=args.single_step,
        output_dir=output_dir,
        config=config,
        show_paired_lines=True,
    )

    print("\nSeed-level accuracy-change summary (mean +/- SD):")
    print(summary[["Step", "Group", "Mean", "SD", "N"]].to_string(index=False))
    print("\nPaired early-versus-late tests:")
    print(tests.to_string(index=False))
    print(f"\nCompleted outputs: {output_dir}")
    return {
        "dataset": config.dataset,
        "rise_threshold": config.high_threshold,
        "output_directory": str(output_dir),
        "n_seeds": len(config.seeds),
        "checkpoints": list(config.checkpoints),
        "single_step": args.single_step,
    }


def main() -> None:
    args = build_parser().parse_args()
    base_output_dir: Path = args.output_dir.resolve()
    if args.supplementary_suite:
        conditions = (
            ("mnist", 0.8),
            ("mnist", 0.5),
            ("fashion-mnist", 0.8),
            ("cifar10", 0.8),
        )
        completed: list[dict[str, object]] = []
        for dataset_name, rise_threshold in conditions:
            has_primary_results = (
                (base_output_dir / "experiment_config.json").exists()
                or (base_output_dir / "per_seed").exists()
            )
            if dataset_name == "mnist" and rise_threshold == 0.8 and has_primary_results:
                output_dir = base_output_dir
            else:
                output_dir = base_output_dir / condition_slug(dataset_name, rise_threshold)
            completed.append(
                run_condition(args, dataset_name, rise_threshold, output_dir)
            )
        base_output_dir.mkdir(parents=True, exist_ok=True)
        (base_output_dir / "supplementary_suite_manifest.json").write_text(
            json.dumps({"conditions": completed}, indent=2),
            encoding="utf-8",
        )
        print(f"\nCompleted all supplementary masking conditions: {base_output_dir}")
    else:
        run_condition(
            args,
            args.dataset,
            args.rise_threshold,
            base_output_dir,
        )


if __name__ == "__main__":
    main()
