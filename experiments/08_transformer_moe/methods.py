"""Paper-aligned expert construction adapters.

The adapters reimplement only the small algorithmic core needed by the local
decoder model.  Original repositories and the exact adaptation boundaries are
documented in BENCHMARK_README.md.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import BenchmarkConfig, METHOD_SPECS, MethodSpec
from .initialization import apply_initialization_policy
from .models import DenseTransformerLM, MoETransformerLM


METHODS_API_VERSION = "2.4.0"


@dataclass
class ActivationCache:
    pre_ffn: List[torch.Tensor]
    hidden: List[torch.Tensor]


@torch.no_grad()
def collect_dense_activations(
    model: DenseTransformerLM,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_batches: int,
    seed: int,
    max_tokens_per_batch: int = 512,
) -> ActivationCache:
    model.eval()
    pre_ffn: List[List[torch.Tensor]] = [[] for _ in model.blocks]
    hidden: List[List[torch.Tensor]] = [[] for _ in model.blocks]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for batch_index, (input_ids, _) in enumerate(batches):
        if batch_index >= num_batches:
            break
        output = model(input_ids, return_intermediates=True)
        for layer, (pre, hid) in enumerate(zip(output.pre_ffn, output.hidden_activations)):
            pre = pre.detach().reshape(-1, pre.shape[-1]).cpu()
            hid = hid.detach().reshape(-1, hid.shape[-1]).cpu()
            count = min(max_tokens_per_batch, pre.shape[0])
            index = torch.randperm(pre.shape[0], generator=generator)[:count]
            pre_ffn[layer].append(pre[index])
            hidden[layer].append(hid[index])
    if not pre_ffn or not pre_ffn[0]:
        raise ValueError("No calibration activations were collected")
    return ActivationCache(
        [torch.cat(parts, dim=0) for parts in pre_ffn],
        [torch.cat(parts, dim=0) for parts in hidden],
    )


def _seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _kmeans_plus_plus(features: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    generator = _seeded_generator(seed)
    n = features.shape[0]
    first = int(torch.randint(n, (1,), generator=generator))
    centers = [features[first]]
    nearest = torch.full((n,), float("inf"), dtype=features.dtype)
    for _ in range(1, k):
        distance = torch.sum((features - centers[-1]) ** 2, dim=1)
        nearest = torch.minimum(nearest, distance)
        if float(nearest.sum()) <= 0:
            index = int(torch.randint(n, (1,), generator=generator))
        else:
            index = int(torch.multinomial(nearest / nearest.sum(), 1, generator=generator))
        centers.append(features[index])
    return torch.stack(centers)


def balanced_kmeans(
    features: torch.Tensor,
    k: int,
    seed: int,
    iterations: int = 30,
    normalize: bool = False,
) -> torch.Tensor:
    """Equal-size k-means used by MoEfication/EMoE/LLaMA-MoE adapters."""
    x = features.detach().float().cpu()
    if x.shape[0] % k:
        raise ValueError("Balanced clustering requires number of neurons divisible by k")
    if normalize:
        x = F.normalize(x, dim=1)
    centers = _kmeans_plus_plus(x, k, seed)
    capacity = x.shape[0] // k
    labels = torch.full((x.shape[0],), -1, dtype=torch.long)
    for _ in range(iterations):
        distances = torch.cdist(x, centers).square()
        confidence = distances.topk(min(2, k), largest=False).values
        margin = confidence[:, 1] - confidence[:, 0] if k > 1 else -confidence[:, 0]
        order = torch.argsort(margin, descending=True)
        remaining = torch.full((k,), capacity, dtype=torch.long)
        new_labels = torch.full_like(labels, -1)
        for point in order.tolist():
            choices = torch.argsort(distances[point])
            for cluster in choices.tolist():
                if remaining[cluster] > 0:
                    new_labels[point] = cluster
                    remaining[cluster] -= 1
                    break
        new_centers = torch.stack([x[new_labels == i].mean(dim=0) for i in range(k)])
        if torch.equal(new_labels, labels):
            break
        labels, centers = new_labels, new_centers
    return labels


def spherical_kmeans(
    features: torch.Tensor, k: int, seed: int, iterations: int = 40
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = F.normalize(features.detach().float().cpu(), dim=1)
    centers = F.normalize(_kmeans_plus_plus(x, k, seed), dim=1)
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    generator = _seeded_generator(seed + 7919)
    for _ in range(iterations):
        new_labels = (x @ centers.t()).argmax(dim=1)
        new_centers = []
        for cluster in range(k):
            members = x[new_labels == cluster]
            if members.numel() == 0:
                members = x[torch.randint(x.shape[0], (1,), generator=generator)]
            new_centers.append(F.normalize(members.mean(dim=0), dim=0))
        new_centers = torch.stack(new_centers)
        if torch.equal(new_labels, labels):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def _balanced_spectral(adjacency: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    matrix = adjacency.detach().float().cpu().clone()
    matrix.fill_diagonal_(0)
    degree = matrix.sum(dim=1).clamp_min(1e-8)
    normalized = matrix / degree.sqrt()[:, None] / degree.sqrt()[None, :]
    _, vectors = torch.linalg.eigh(normalized)
    embedding = F.normalize(vectors[:, -k:], dim=1)
    return balanced_kmeans(embedding, k, seed, normalize=True)


def functional_connectivity_labels(hidden: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Legacy absolute-Pearson construction retained as an explicit ablation."""
    values = hidden.float()
    values = values - values.mean(dim=0, keepdim=True)
    values = values / values.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-6)
    adjacency = (values.t() @ values / max(1, values.shape[0] - 1)).abs()
    return _balanced_spectral(adjacency, k, seed)


def _off_diagonal_mean(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    if n <= 1:
        return matrix.new_zeros(())
    return (matrix.sum() - matrix.diagonal().sum()) / (n * (n - 1))


def _mean_normalize_affinity(matrix: torch.Tensor) -> torch.Tensor:
    value = matrix.detach().float().cpu().clone().clamp_min(0)
    value.fill_diagonal_(0)
    return value / _off_diagonal_mean(value).clamp_min(1e-8)


def _consensus_correlation(
    hidden: torch.Tensor,
    splits: int,
    seed: int,
) -> torch.Tensor:
    """Estimate a signed FC graph by averaging Fisher-z correlations across splits."""
    values = hidden.detach().float().cpu()
    if values.shape[0] < 2:
        raise ValueError("FC construction requires at least two activation samples")
    split_count = min(max(1, splits), max(1, values.shape[0] // 2))
    generator = _seeded_generator(seed)
    shuffled = values[torch.randperm(values.shape[0], generator=generator)]
    fisher_parts = []
    for part in torch.tensor_split(shuffled, split_count, dim=0):
        if part.shape[0] < 2:
            continue
        standardized = part - part.mean(dim=0, keepdim=True)
        standardized = standardized / standardized.std(
            dim=0, unbiased=False, keepdim=True
        ).clamp_min(1e-6)
        correlation = standardized.t() @ standardized / part.shape[0]
        fisher_parts.append(torch.atanh(correlation.clamp(-0.999, 0.999)))
    if not fisher_parts:
        raise ValueError("FC consensus split produced no valid activation subset")
    correlation = torch.tanh(torch.stack(fisher_parts).mean(dim=0))
    correlation.fill_diagonal_(0)
    return correlation


def _balanced_swap_refinement(
    adjacency: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    max_swaps: int,
) -> Tuple[torch.Tensor, int, float]:
    """Increase within-expert affinity using swaps that preserve exact capacities."""
    refined = labels.clone()
    swaps = 0
    total_gain = 0.0
    if max_swaps <= 0:
        return refined, swaps, total_gain
    for _ in range(max_swaps):
        membership = F.one_hot(refined, num_classes=k).to(adjacency.dtype)
        cluster_affinity = adjacency @ membership
        best_gain = 0.0
        best_pair: Optional[Tuple[int, int]] = None
        for left_cluster in range(k):
            left = torch.nonzero(refined == left_cluster, as_tuple=False).squeeze(-1)
            for right_cluster in range(left_cluster + 1, k):
                right = torch.nonzero(refined == right_cluster, as_tuple=False).squeeze(-1)
                gain = (
                    (cluster_affinity[left, right_cluster] - cluster_affinity[left, left_cluster])[:, None]
                    + (cluster_affinity[right, left_cluster] - cluster_affinity[right, right_cluster])[None, :]
                    - 2.0 * adjacency[left][:, right]
                )
                flat_index = int(gain.argmax())
                value = float(gain.reshape(-1)[flat_index])
                if value > best_gain + 1e-9:
                    row = flat_index // right.numel()
                    column = flat_index % right.numel()
                    best_gain = value
                    best_pair = (int(left[row]), int(right[column]))
        if best_pair is None:
            break
        left_index, right_index = best_pair
        left_label = refined[left_index].clone()
        refined[left_index] = refined[right_index]
        refined[right_index] = left_label
        swaps += 1
        total_gain += best_gain
    return refined, swaps, total_gain


def _partition_affinity_diagnostics(
    adjacency: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(labels.numel(), dtype=torch.bool)
    within = adjacency[same & ~diagonal]
    between = adjacency[~same]
    within_mean = float(within.mean()) if within.numel() else 0.0
    between_mean = float(between.mean()) if between.numel() else 0.0
    return {
        "within_affinity": within_mean,
        "between_affinity": between_mean,
        "within_between_ratio": within_mean / max(1e-12, between_mean),
    }


def _fc_contribution_adjacency(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    seed: int,
    cfg: BenchmarkConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    correlation = _consensus_correlation(hidden, cfg.fc_consensus_splits, seed)
    directions = F.normalize(output_weight.detach().float().cpu().t(), dim=1)
    output_cosine = directions @ directions.t()
    positive_fc = correlation.clamp_min(0)
    contribution = (correlation * output_cosine).clamp_min(0)
    positive_fc.fill_diagonal_(0)
    contribution.fill_diagonal_(0)
    positive_scale = _off_diagonal_mean(positive_fc).clamp_min(1e-8)
    contribution_scale = _off_diagonal_mean(contribution).clamp_min(1e-8)
    adjacency = (
        (1.0 - cfg.fc_output_weight) * (positive_fc / positive_scale)
        + cfg.fc_output_weight * (contribution / contribution_scale)
    )
    return _mean_normalize_affinity(adjacency), {
        "positive_fc_mean": float(positive_scale),
        "contribution_affinity_mean": float(contribution_scale),
    }


def _coactivation_adjacency(hidden: torch.Tensor) -> torch.Tensor:
    positive = hidden.detach().float().cpu().clamp_min(0)
    positive = positive / positive.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return _mean_normalize_affinity(positive.t() @ positive)


def _key_adjacency(input_weight: torch.Tensor) -> torch.Tensor:
    keys = F.normalize(input_weight.detach().float().cpu(), dim=1)
    return _mean_normalize_affinity((keys @ keys.t()).clamp_min(0))


def _oracle_reconstruction_context(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = min(max_tokens, hidden.shape[0])
    activation = hidden[:count].detach().float().cpu()
    weight = output_weight.detach().float().cpu()
    dense_output = activation @ weight.t()
    return activation, weight, dense_output


def _oracle_reconstruction_from_context(
    context: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    k: int,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Score a partition by its best possible top-1 dense-FFN reconstruction."""
    activation, weight, dense_output = context
    count = activation.shape[0]
    errors = []
    for expert_index in range(k):
        index = torch.nonzero(labels == expert_index, as_tuple=False).squeeze(-1)
        expert_output = activation[:, index] @ weight[:, index].t()
        errors.append((dense_output - expert_output).square().mean(dim=-1))
    error_matrix = torch.stack(errors, dim=-1)
    best_error, oracle_labels = error_matrix.min(dim=-1)
    dense_energy = dense_output.square().mean().clamp_min(1e-12)
    normalized_error = best_error.mean() / dense_energy
    usage = torch.bincount(oracle_labels, minlength=k).float() / max(1, count)
    entropy = -(usage.clamp_min(1e-12) * usage.clamp_min(1e-12).log()).sum()
    entropy = entropy / math.log(k) if k > 1 else entropy.new_ones(())
    return {
        "oracle_normalized_mse": float(normalized_error),
        "oracle_explained_fraction": float(1.0 - normalized_error),
        "oracle_usage_entropy": float(entropy),
        "oracle_min_usage": float(usage.min()),
        "oracle_max_usage": float(usage.max()),
    }, oracle_labels


def _oracle_reconstruction_metrics(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    max_tokens: int,
) -> Tuple[Dict[str, float], torch.Tensor]:
    context = _oracle_reconstruction_context(hidden, output_weight, max_tokens)
    return _oracle_reconstruction_from_context(context, labels, k)


def _initialize_oracle_ridge_router(
    moe_layer,
    pre_ffn: torch.Tensor,
    hidden: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
    cfg: BenchmarkConfig,
) -> Dict[str, float]:
    """Warm-start a linear Switch router from oracle top-1 reconstruction labels."""
    if moe_layer.router is None or not hasattr(moe_layer.router, "weight"):
        raise ValueError("oracle_ridge initialization requires a linear router")
    metrics, oracle_labels = _oracle_reconstruction_metrics(
        hidden, output_weight, labels, cfg.num_experts, cfg.fc_oracle_tokens
    )
    count = oracle_labels.numel()
    x = pre_ffn[:count].detach().float().cpu()
    target = F.one_hot(oracle_labels, num_classes=cfg.num_experts).float()
    target = target - target.mean(dim=0, keepdim=True)
    gram = x.t() @ x / max(1, count)
    ridge_scale = cfg.fc_router_ridge * gram.diagonal().mean().clamp_min(1e-8)
    covariance = x.t() @ target / max(1, count)
    coefficient = torch.linalg.solve(
        gram + torch.eye(gram.shape[0]) * ridge_scale,
        covariance,
    ).t()
    key_prior = torch.stack(
        [input_weight.detach().float().cpu()[labels == expert].mean(dim=0)
         for expert in range(cfg.num_experts)]
    )
    coefficient_norm = coefficient.norm(dim=1).mean().clamp_min(1e-8)
    key_prior = F.normalize(key_prior, dim=1) * coefficient_norm
    router_weight = (
        (1.0 - cfg.fc_router_key_prior) * coefficient
        + cfg.fc_router_key_prior * key_prior
    )
    with torch.no_grad():
        moe_layer.router.weight.copy_(router_weight.to(moe_layer.router.weight))
    prediction = (x @ router_weight.t()).argmax(dim=-1)
    metrics.update(
        {
            "ridge_accuracy": float((prediction == oracle_labels).float().mean()),
            "ridge_scale": float(ridge_scale),
            "router_weight_norm": float(router_weight.norm(dim=1).mean()),
        }
    )
    return metrics


def _distribute_functional_modules(
    module_labels: torch.Tensor,
    k: int,
    seed: int,
) -> torch.Tensor:
    """Interleave every FC module across experts to maximize top-1 coverage."""
    generator = _seeded_generator(seed)
    distributed = torch.full_like(module_labels, -1)
    for module in range(k):
        members = torch.nonzero(module_labels == module, as_tuple=False).squeeze(-1)
        members = members[torch.randperm(members.numel(), generator=generator)]
        start = module % k
        assignments = (torch.arange(members.numel()) + start) % k
        distributed[members] = assignments
    counts = torch.bincount(distributed, minlength=k)
    if not torch.all(counts == module_labels.numel() // k):
        # The benchmark dimensions are divisible by k^2; retain a safe exact-
        # capacity fallback for custom dimensions that are only divisible by k.
        return random_balanced_labels(module_labels.numel(), k, seed + 1_000_003)
    return distributed


def functional_hybrid_labels(
    hidden: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    k: int,
    seed: int,
    cfg: BenchmarkConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Select an FC-containing multi-view partition on calibration reconstruction."""
    fc_graph, fc_stats = _fc_contribution_adjacency(hidden, output_weight, seed, cfg)
    coactivation_graph = _coactivation_adjacency(hidden)
    key_graph = _key_adjacency(input_weight)
    oracle_context = _oracle_reconstruction_context(
        hidden, output_weight, cfg.fc_oracle_tokens
    )
    candidates: List[Dict[str, object]] = []
    best_labels: Optional[torch.Tensor] = None
    best_adjacency: Optional[torch.Tensor] = None
    best_error = float("inf")
    best_index = -1
    for candidate_index, weights in enumerate(cfg.fc_hybrid_view_weights):
        fc_weight, coactivation_weight, key_weight = weights
        adjacency = (
            fc_weight * fc_graph
            + coactivation_weight * coactivation_graph
            + key_weight * key_graph
        )
        mean_affinity = _off_diagonal_mean(adjacency)
        adjacency = (
            (1.0 - cfg.fc_uniform_shrinkage) * adjacency
            + cfg.fc_uniform_shrinkage * mean_affinity
        )
        adjacency.fill_diagonal_(0)
        clustered = _balanced_spectral(adjacency, k, seed + 104729 * candidate_index)
        distributed = _distribute_functional_modules(
            clustered, k, seed + 130363 * candidate_index
        )
        for strategy, labels in (("clustered", clustered), ("distributed", distributed)):
            metrics, _ = _oracle_reconstruction_from_context(
                oracle_context, labels, k
            )
            candidate = {
                "weights": [float(value) for value in weights],
                "strategy": strategy,
                "oracle_normalized_mse": metrics["oracle_normalized_mse"],
                "oracle_usage_entropy": metrics["oracle_usage_entropy"],
            }
            candidates.append(candidate)
            error = float(metrics["oracle_normalized_mse"])
            if error < best_error:
                best_error = error
                best_labels = labels
                best_adjacency = adjacency
                best_index = len(candidates) - 1
    assert best_labels is not None and best_adjacency is not None
    refined, swaps, refinement_gain = _balanced_swap_refinement(
        best_adjacency, best_labels, k, cfg.fc_refinement_swaps
    )
    refined_metrics, _ = _oracle_reconstruction_from_context(
        oracle_context, refined, k
    )
    if float(refined_metrics["oracle_normalized_mse"]) <= best_error:
        selected = refined
        selected_metrics = refined_metrics
        accepted_refinement = True
    else:
        selected = best_labels
        selected_metrics, _ = _oracle_reconstruction_from_context(
            oracle_context, selected, k
        )
        accepted_refinement = False
    diagnostics: Dict[str, object] = {
        **fc_stats,
        **_partition_affinity_diagnostics(best_adjacency, selected),
        **selected_metrics,
        "selected_candidate": best_index,
        "selected_weights": candidates[best_index]["weights"],
        "selected_strategy": candidates[best_index]["strategy"],
        "candidate_scores": candidates,
        "refinement_swaps": swaps,
        "refinement_gain": refinement_gain,
        "refinement_accepted": accepted_refinement,
        "consensus_splits": min(cfg.fc_consensus_splits, max(1, hidden.shape[0] // 2)),
    }
    return selected, diagnostics


def functional_connectivity_plus_labels(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    k: int,
    seed: int,
    cfg: BenchmarkConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Construct balanced experts from stable FC and neuron contribution geometry.

    The contribution term is positive when two neurons have aligned activation
    covariance and aligned output directions, including the anti/anti case. Each
    term is mean-normalized before mixing so ``fc_output_weight`` has an
    interpretable influence rather than depending on raw matrix scale.
    """
    adjacency, graph_stats = _fc_contribution_adjacency(hidden, output_weight, seed, cfg)
    mean_affinity = _off_diagonal_mean(adjacency)
    adjacency = (
        (1.0 - cfg.fc_uniform_shrinkage) * adjacency
        + cfg.fc_uniform_shrinkage * mean_affinity
    )
    adjacency.fill_diagonal_(0)
    initial = _balanced_spectral(adjacency, k, seed)
    labels, swaps, refinement_gain = _balanced_swap_refinement(
        adjacency, initial, k, cfg.fc_refinement_swaps
    )
    diagnostics = _partition_affinity_diagnostics(adjacency, labels)
    diagnostics.update(
        {
            "consensus_splits": float(min(cfg.fc_consensus_splits, max(1, hidden.shape[0] // 2))),
            **graph_stats,
            "refinement_swaps": float(swaps),
            "refinement_gain": float(refinement_gain),
        }
    )
    return labels, diagnostics


def coactivation_labels(hidden: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    # MoEfication's graph is defined for ReLU; positive GELU activity is the
    # paper-compatible extension also used by D2DMoE's GELU experiments.
    positive = hidden.float().clamp_min(0)
    positive = positive / positive.norm(dim=0, keepdim=True).clamp_min(1e-6)
    adjacency = positive.t() @ positive
    return _balanced_spectral(adjacency, k, seed)


def random_balanced_labels(n: int, k: int, seed: int) -> torch.Tensor:
    generator = _seeded_generator(seed)
    labels = torch.arange(n) % k
    return labels[torch.randperm(n, generator=generator)]


def _copy_shared(dense: DenseTransformerLM, moe: MoETransformerLM) -> None:
    moe.token_embedding.load_state_dict(dense.token_embedding.state_dict())
    moe.position_embedding.load_state_dict(dense.position_embedding.state_dict())
    moe.final_norm.load_state_dict(dense.final_norm.state_dict())
    moe.lm_head.load_state_dict(dense.lm_head.state_dict())
    for dense_block, moe_block in zip(dense.blocks, moe.blocks):
        moe_block.norm1.load_state_dict(dense_block.norm1.state_dict())
        moe_block.attn.load_state_dict(dense_block.attn.state_dict())
        moe_block.norm2.load_state_dict(dense_block.norm2.state_dict())


@torch.no_grad()
def _copy_partitioned_layer(dense_ffn, moe_layer, labels: torch.Tensor) -> None:
    for expert_index, expert in enumerate(moe_layer.experts):
        index = torch.nonzero(labels == expert_index, as_tuple=False).squeeze(-1)
        if index.numel() != expert.fc1.out_features:
            raise ValueError("Construction did not produce equal-size experts")
        index = index.to(dense_ffn.fc1.weight.device)
        expert.fc1.weight.copy_(dense_ffn.fc1.weight[index])
        expert.fc1.bias.copy_(dense_ffn.fc1.bias[index])
        expert.fc2.weight.copy_(dense_ffn.fc2.weight[:, index])
    moe_layer.output_bias.copy_(dense_ffn.fc2.bias)


@torch.no_grad()
def _copy_full_layer(dense_ffn, moe_layer) -> None:
    for expert in moe_layer.experts:
        expert.fc1.weight.copy_(dense_ffn.fc1.weight)
        expert.fc1.bias.copy_(dense_ffn.fc1.bias)
        expert.fc2.weight.copy_(dense_ffn.fc2.weight)
    moe_layer.output_bias.copy_(dense_ffn.fc2.bias)


def _data_aware_low_rank(
    weight: torch.Tensor,
    activations: torch.Tensor,
    energy: float,
    min_rank_fraction: float,
) -> Tuple[torch.Tensor, int]:
    """Cluster-aware data-aware SVD for the first FFN linear map."""
    original_device, original_dtype = weight.device, weight.dtype
    w = weight.detach().float().cpu()
    x = activations.detach().float().cpu()
    covariance = x.t() @ x / max(1, x.shape[0])
    scale = covariance.diag().mean().clamp_min(1e-8)
    covariance = covariance + torch.eye(covariance.shape[0]) * (1e-5 * scale)
    transform = torch.linalg.cholesky(covariance)
    weighted = w @ transform
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    cumulative = singular.square().cumsum(0) / singular.square().sum().clamp_min(1e-12)
    energy_rank = int(torch.searchsorted(cumulative, torch.tensor(energy)).item()) + 1
    minimum = max(1, math.ceil(min(weighted.shape) * min_rank_fraction))
    rank = min(min(weighted.shape), max(minimum, energy_rank))
    approximation = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    restored = torch.linalg.solve_triangular(
        transform.t(), approximation.t(), upper=True
    ).t()
    return restored.to(device=original_device, dtype=original_dtype), rank


def _new_moe(
    dense: Optional[DenseTransformerLM],
    vocab_size: int,
    cfg: BenchmarkConfig,
    spec: MethodSpec,
) -> MoETransformerLM:
    combine = "softmax" if spec.construction in {"scratch", "copy", "cluster_aware_svd"} else "sum"
    model = MoETransformerLM(
        vocab_size=vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        dropout=cfg.dropout,
        expert_width=spec.expert_width,
        router_kind=spec.router,
        router_hidden_size=cfg.router_hidden_size,
        combine_mode=combine,
        capacity_factor=cfg.capacity_factor,
        drop_overflow_tokens=cfg.drop_overflow_tokens,
        dynamic_threshold=cfg.d2d_threshold,
    )
    apply_initialization_policy(
        model,
        tie_embeddings=cfg.tie_embeddings,
        embedding_init_std=cfg.embedding_init_std,
        xavier_initialize=cfg.xavier_initialize,
    )
    if dense is not None:
        _copy_shared(dense, model)
    return model


def construct_method(
    method_name: str,
    dense: Optional[DenseTransformerLM],
    vocab_size: int,
    cfg: BenchmarkConfig,
    cache: Optional[ActivationCache],
    seed: int,
) -> Tuple[MoETransformerLM, Dict[str, object]]:
    spec = METHOD_SPECS[method_name]
    if spec.requires_dense and dense is None:
        raise ValueError(f"{method_name} requires a dense checkpoint")
    model = _new_moe(dense, vocab_size, cfg, spec)
    metadata: Dict[str, object] = {
        "method": method_name,
        "assignments": [],
        "svd_ranks": [],
        "fc_diagnostics": [],
        "router_initialization": [],
    }
    if spec.construction == "scratch":
        return model, metadata

    assert dense is not None
    if spec.construction == "copy":
        for dense_block, moe_block in zip(dense.blocks, model.blocks):
            _copy_full_layer(dense_block.ffn, moe_block.moe)
        return model, metadata

    if spec.construction == "cluster_aware_svd":
        if cache is None:
            raise ValueError("Cluster-aware Upcycling requires cached pre-FFN activations")
        for layer, (dense_block, moe_block) in enumerate(zip(dense.blocks, model.blocks)):
            labels, centroids = spherical_kmeans(cache.pre_ffn[layer], cfg.num_experts, seed + layer)
            _copy_full_layer(dense_block.ffn, moe_block.moe)
            ranks = []
            for expert_index, expert in enumerate(moe_block.moe.experts):
                members = cache.pre_ffn[layer][labels == expert_index]
                approximated, rank = _data_aware_low_rank(
                    dense_block.ffn.fc1.weight,
                    members,
                    cfg.cluster_aware_energy,
                    cfg.cluster_aware_min_rank_fraction,
                )
                with torch.no_grad():
                    expert.fc1.weight.copy_(approximated)
                ranks.append(rank)
            with torch.no_grad():
                moe_block.moe.router.weight.copy_(centroids.to(moe_block.moe.router.weight))
            metadata["assignments"].append(labels.tolist())
            metadata["svd_ranks"].append(ranks)
        return model, metadata

    if cache is None and (
        spec.construction in {"fc", "fc_plus", "fc_hybrid", "coactivation"}
        or spec.router_initialization != "default"
    ):
        raise ValueError(f"{method_name} requires cached hidden activations")
    for layer, (dense_block, moe_block) in enumerate(zip(dense.blocks, model.blocks)):
        if spec.construction == "random":
            labels = random_balanced_labels(cfg.d_ff, cfg.num_experts, seed + layer)
        elif spec.construction == "key_clustering":
            labels = balanced_kmeans(
                dense_block.ffn.fc1.weight, cfg.num_experts, seed + layer, normalize=True
            )
        elif spec.construction == "fc":
            labels = functional_connectivity_labels(cache.hidden[layer], cfg.num_experts, seed + layer)
        elif spec.construction == "fc_plus":
            labels, diagnostics = functional_connectivity_plus_labels(
                cache.hidden[layer],
                dense_block.ffn.fc2.weight,
                cfg.num_experts,
                seed + layer,
                cfg,
            )
            metadata["fc_diagnostics"].append({"layer": layer, **diagnostics})
        elif spec.construction == "fc_hybrid":
            labels, diagnostics = functional_hybrid_labels(
                cache.hidden[layer],
                dense_block.ffn.fc1.weight,
                dense_block.ffn.fc2.weight,
                cfg.num_experts,
                seed + layer,
                cfg,
            )
            metadata["fc_diagnostics"].append({"layer": layer, **diagnostics})
        elif spec.construction == "coactivation":
            labels = coactivation_labels(cache.hidden[layer], cfg.num_experts, seed + layer)
        else:
            raise ValueError(spec.construction)
        _copy_partitioned_layer(dense_block.ffn, moe_block.moe, labels)
        if spec.router_initialization == "oracle_ridge":
            router_diagnostics = _initialize_oracle_ridge_router(
                moe_block.moe,
                cache.pre_ffn[layer],
                cache.hidden[layer],
                dense_block.ffn.fc1.weight,
                dense_block.ffn.fc2.weight,
                labels,
                cfg,
            )
            metadata["router_initialization"].append(
                {"layer": layer, **router_diagnostics}
            )
        elif spec.router_initialization != "default":
            raise ValueError(spec.router_initialization)
        metadata["assignments"].append(labels.tolist())
    return model, metadata


def clone_dense(model: DenseTransformerLM) -> DenseTransformerLM:
    return copy.deepcopy(model)
