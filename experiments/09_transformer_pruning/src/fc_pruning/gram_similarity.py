from __future__ import annotations

from dataclasses import dataclass

import torch

from .pruning import apply_layer_plan


@dataclass(frozen=True)
class GramPlanConfig:
    topk: int = 16
    merge_fraction: float = 0.15
    keeper_capacity: int = 1
    minimum_similarity: float = 0.0
    minimum_output_gain: float = 0.05
    protect_fraction: float = 0.05
    ridge_relative: float = 1e-4
    compensate_merge_mean: bool = True


def _center_gram(
    gram: torch.Tensor,
    sums: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Convert X.T @ X into the centered cross-product used by Pearson r."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    sums = sums.to(device=gram.device, dtype=gram.dtype)
    return gram - torch.outer(sums, sums) / sample_count


def _topk_from_gram(
    gram: torch.Tensor,
    topk: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the highest signed correlations from a centered Gram matrix."""
    width = gram.shape[0]
    if gram.ndim != 2 or gram.shape[1] != width:
        raise ValueError(f"Expected a square Gram matrix, got {tuple(gram.shape)}")
    if not 0 < topk < width:
        raise ValueError(f"topk must be in [1, {width - 1}], got {topk}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    diagonal = gram.diagonal().clamp_min(0.0)
    variance_floor = diagonal.max().clamp_min(1.0) * 1e-12
    valid = diagonal > variance_floor
    inverse_norms = torch.zeros_like(diagonal)
    inverse_norms[valid] = diagonal[valid].rsqrt()
    values = torch.empty((width, topk), dtype=torch.float32)
    indices = torch.empty((width, topk), dtype=torch.int32)
    for begin in range(0, width, block_size):
        end = min(width, begin + block_size)
        similarity = gram[begin:end] * inverse_norms[begin:end, None]
        similarity = similarity * inverse_norms[None, :]
        similarity = similarity.clamp(min=-1.0, max=1.0)
        rows = torch.arange(end - begin, device=gram.device)
        columns = torch.arange(begin, end, device=gram.device)
        similarity[rows, columns] = -torch.inf
        block_values, block_indices = torch.topk(similarity, topk, dim=1)
        values[begin:end].copy_(block_values.float().cpu())
        indices[begin:end].copy_(block_indices.int().cpu())
    return values, indices


def build_layer_candidate_pool(
    layer,
    fit_gram: torch.Tensor,
    holdout_gram: torch.Tensor,
    fit_sum: torch.Tensor,
    holdout_sum: torch.Tensor,
    total_sum: torch.Tensor,
    fit_count: int,
    holdout_count: int,
    total_count: int,
    topk: int,
    block_size: int,
    method: str,
) -> dict:
    device = fit_gram.device
    down = layer.mlp.down_proj.weight.detach().float().to(device)
    output_norm_sq = down.square().sum(dim=0)
    total_second = (fit_gram.diagonal() + holdout_gram.diagonal()) / total_count
    total_mean = total_sum.to(device).float() / total_count
    total_variance = (total_second - total_mean.square()).clamp_min(0.0)
    importance = total_variance * output_norm_sq

    if method != "fc":
        raise ValueError(f"Unsupported similarity method: {method}")
    centered_fit_gram = _center_gram(fit_gram, fit_sum, fit_count)
    similarities, neighbors = _topk_from_gram(
        centered_fit_gram, topk, block_size
    )
    similarity_metric = "signed_pearson"
    del centered_fit_gram

    query = torch.arange(fit_gram.shape[0], device=device).unsqueeze(1)
    query = query.expand(-1, topk)
    neighbor = neighbors.long().to(device)
    query_importance = importance.index_select(0, query.flatten()).view_as(query)
    neighbor_importance = importance.index_select(0, neighbor.flatten()).view_as(neighbor)
    source = torch.where(query_importance <= neighbor_importance, query, neighbor)
    keeper = torch.where(query_importance <= neighbor_importance, neighbor, query)

    def gather(matrix: torch.Tensor) -> torch.Tensor:
        return matrix[source, keeper]

    fit_source_second = fit_gram.diagonal().index_select(0, source.flatten()).view_as(source) / fit_count
    fit_keeper_second = fit_gram.diagonal().index_select(0, keeper.flatten()).view_as(keeper) / fit_count
    hold_source_second = holdout_gram.diagonal().index_select(0, source.flatten()).view_as(source) / holdout_count
    hold_keeper_second = holdout_gram.diagonal().index_select(0, keeper.flatten()).view_as(keeper) / holdout_count
    payload = {
        "source": source.int().cpu(),
        "keeper": keeper.int().cpu(),
        "similarity": similarities,
        "fit_cross": (gather(fit_gram) / fit_count).float().cpu(),
        "fit_source_second": fit_source_second.float().cpu(),
        "fit_keeper_second": fit_keeper_second.float().cpu(),
        "holdout_cross": (gather(holdout_gram) / holdout_count).float().cpu(),
        "holdout_source_second": hold_source_second.float().cpu(),
        "holdout_keeper_second": hold_keeper_second.float().cpu(),
        "fit_mean": (fit_sum.float() / fit_count).cpu(),
        "holdout_mean": (holdout_sum.float() / holdout_count).cpu(),
        "total_mean": total_mean.cpu(),
        "importance": importance.float().cpu(),
        "output_norm_sq": output_norm_sq.float().cpu(),
        "method": method,
        "similarity_metric": similarity_metric,
    }
    return payload


def _candidate_metrics(pool: dict, config: GramPlanConfig) -> dict:
    fit_source_second = pool["fit_source_second"].float()
    fit_keeper_second = pool["fit_keeper_second"].float()
    fit_cross = pool["fit_cross"].float()
    hold_source_second = pool["holdout_source_second"].float()
    hold_keeper_second = pool["holdout_keeper_second"].float()
    hold_cross = pool["holdout_cross"].float()
    source = pool["source"].long()
    keeper = pool["keeper"].long()
    median_energy = float(fit_source_second.median().clamp_min(1e-30))
    ridge = config.ridge_relative * median_energy
    alpha = fit_cross / (fit_keeper_second + ridge)

    fit_residual = (
        fit_source_second - 2.0 * alpha * fit_cross + alpha.square() * fit_keeper_second
    ).clamp_min(0.0)
    hold_residual = (
        hold_source_second
        - 2.0 * alpha * hold_cross
        + alpha.square() * hold_keeper_second
    ).clamp_min(0.0)
    fit_mean = pool["fit_mean"].float()
    holdout_mean = pool["holdout_mean"].float()
    fit_residual_mean = fit_mean[source] - alpha * fit_mean[keeper]
    hold_residual_mean = holdout_mean[source] - alpha * holdout_mean[keeper]
    if config.compensate_merge_mean:
        fit_residual = (fit_residual - fit_residual_mean.square()).clamp_min(0.0)
        hold_residual = (hold_residual - hold_residual_mean.square()).clamp_min(0.0)

    output_norm = pool["output_norm_sq"].float()[source]
    fit_cost = fit_residual * output_norm
    holdout_cost = hold_residual * output_norm
    holdout_direct_variance = (
        hold_source_second - holdout_mean[source].square()
    ).clamp_min(0.0)
    holdout_direct_cost = holdout_direct_variance * output_norm
    output_gain = 1.0 - holdout_cost / holdout_direct_cost.clamp_min(1e-30)
    return {
        "source": source,
        "keeper": keeper,
        "alpha": alpha,
        "fit_cost": fit_cost,
        "holdout_cost": holdout_cost,
        "robust_cost": torch.maximum(fit_cost, holdout_cost),
        "output_gain": output_gain,
    }


def plan_from_candidate_pool(
    layer,
    pool: dict,
    ratio: float,
    config: GramPlanConfig,
) -> tuple[dict, dict]:
    width = pool["importance"].numel()
    target = int(round(width * ratio))
    metrics = _candidate_metrics(pool, config)
    ranks = torch.arange(pool["source"].shape[1]).unsqueeze(0)
    valid = ranks < config.topk
    valid = valid.expand_as(pool["similarity"]).clone()
    valid &= pool["similarity"] >= config.minimum_similarity
    valid &= metrics["output_gain"] >= config.minimum_output_gain

    protected = torch.zeros(width, dtype=torch.bool)
    protect_count = int(round(width * config.protect_fraction))
    if protect_count:
        protected[torch.topk(pool["importance"], protect_count).indices] = True
    valid &= ~protected[metrics["source"]]
    candidate_cost = metrics["robust_cost"].clone()
    candidate_cost[~valid] = torch.inf
    valid_indices = valid.flatten().nonzero(as_tuple=False).flatten()
    valid_costs = candidate_cost.flatten().index_select(0, valid_indices)
    candidate_order = valid_indices[torch.argsort(valid_costs)].tolist()
    direct_order = torch.argsort(pool["importance"]).tolist()
    direct_pointer = 0
    candidate_pointer = 0
    merge_limit = int(round(target * config.merge_fraction))
    pruned: set[int] = set()
    protected_keepers: set[int] = set()
    keeper_load: dict[int, int] = {}
    direct: list[int] = []
    merges: list[dict] = []

    while len(pruned) < target:
        while direct_pointer < width:
            direct_neuron = int(direct_order[direct_pointer])
            if (
                direct_neuron not in pruned
                and direct_neuron not in protected_keepers
                and not protected[direct_neuron]
            ):
                break
            direct_pointer += 1
        while candidate_pointer < len(candidate_order):
            flat_index = candidate_order[candidate_pointer]
            row = flat_index // pool["source"].shape[1]
            rank = flat_index % pool["source"].shape[1]
            merge_source = int(metrics["source"][row, rank])
            merge_keeper = int(metrics["keeper"][row, rank])
            if not torch.isfinite(candidate_cost[row, rank]):
                candidate_pointer = len(candidate_order)
                break
            if (
                len(merges) < merge_limit
                and merge_source not in pruned
                and merge_keeper not in pruned
                and merge_source not in protected_keepers
                and keeper_load.get(merge_keeper, 0) < config.keeper_capacity
            ):
                break
            candidate_pointer += 1

        direct_cost = (
            float(pool["importance"][direct_neuron])
            if direct_pointer < width
            else float("inf")
        )
        merge_cost = (
            float(candidate_cost[row, rank])
            if candidate_pointer < len(candidate_order)
            else float("inf")
        )
        if merge_cost < direct_cost:
            alpha = float(metrics["alpha"][row, rank])
            merges.append(
                {
                    "prune": merge_source,
                    "keep": merge_keeper,
                    "source_similarity": float(pool["similarity"][row, rank]),
                    "alpha": alpha,
                    "cost": merge_cost,
                    "validation_gain": float(metrics["output_gain"][row, rank]),
                }
            )
            pruned.add(merge_source)
            protected_keepers.add(merge_keeper)
            keeper_load[merge_keeper] = keeper_load.get(merge_keeper, 0) + 1
            candidate_pointer += 1
        elif direct_pointer < width:
            direct.append(direct_neuron)
            pruned.add(direct_neuron)
            direct_pointer += 1
        else:
            raise RuntimeError("Unable to satisfy the requested pruning target")

    total_mean = pool["total_mean"].float()
    down = layer.mlp.down_proj.weight.detach().float().cpu()
    bias = torch.zeros(down.shape[0], dtype=torch.float32)
    if direct:
        direct_indices = torch.tensor(direct, dtype=torch.long)
        bias += down[:, direct_indices] @ total_mean[direct_indices]
    if config.compensate_merge_mean:
        for merge in merges:
            source = merge["prune"]
            keeper = merge["keep"]
            residual_mean = total_mean[source] - merge["alpha"] * total_mean[keeper]
            bias += down[:, source] * residual_mean
    plan = {
        "method": pool["method"],
        "similarity_metric": pool.get("similarity_metric", "legacy_unspecified"),
        "target": target,
        "direct": direct,
        "merges": merges,
        "pruned": sorted(pruned),
        "bias_compensation": bias,
        "gram_plan_config": config.__dict__,
    }
    audit = {
        "target": target,
        "merges": len(merges),
        "direct": len(direct),
        "merge_fraction": len(merges) / target,
        "maximum_keeper_load": max(keeper_load.values(), default=0),
        "mean_merge_similarity": sum(m["source_similarity"] for m in merges)
        / max(1, len(merges)),
        "mean_validation_gain": sum(m["validation_gain"] for m in merges)
        / max(1, len(merges)),
        "estimated_cost": sum(m["cost"] for m in merges)
        + float(pool["importance"][torch.tensor(direct)].sum())
        if direct
        else sum(m["cost"] for m in merges),
    }
    return plan, audit


def build_and_apply_gram_plans(
    model,
    pools: list[dict],
    ratio: float,
    config: GramPlanConfig,
) -> tuple[list[dict], list[dict]]:
    plans = []
    audits = []
    for index, (layer, pool) in enumerate(zip(model.model.layers, pools)):
        print(f"Gram similarity plan layer={index:02d}", flush=True)
        plan, audit = plan_from_candidate_pool(layer, pool, ratio, config)
        apply_layer_plan(layer, plan)
        plans.append(plan)
        audits.append(audit)
    return plans, audits
