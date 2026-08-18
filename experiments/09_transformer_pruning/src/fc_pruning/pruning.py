from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .activations import load_layer_activations
from .data import load_calibration
from .modeling import llama_layers


@dataclass
class Merge:
    prune: int
    keep: int
    source_similarity: float
    alpha: float
    cost: float
    validation_gain: float


def output_importance(activations: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    second_moment = activations.float().square().mean(dim=0)
    output_norm_sq = down_weight.detach().float().square().sum(dim=0).cpu()
    return second_moment * output_norm_sq


def flap_importance(activations: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    """FLAP-style activation-variance times output-weight score.

    This is kept separate from ``output_importance`` because the hybrid
    similarity methods can use FLAP for direct deletions while retaining their
    own activation-space similarity for compensated merges.
    """
    variance = activations.float().var(dim=0, unbiased=False)
    output_norm_sq = down_weight.detach().float().square().sum(dim=0).cpu()
    return variance * output_norm_sq


def _row_features(
    method: str, layer, activations: torch.Tensor, config: dict
) -> torch.Tensor:
    base_method = method.removesuffix("_flap")
    if base_method == "fc":
        features = activations.float()
        if config.get("fc_centered", False):
            features = features - features.mean(dim=0, keepdim=True)
        return features.t().contiguous()
    if base_method == "is":
        gate = layer.mlp.gate_proj.weight.detach().float().cpu()
        up = layer.mlp.up_proj.weight.detach().float().cpu()
        return torch.cat([gate, up], dim=1).contiguous()
    raise ValueError(f"Unsupported similarity method: {method}")


def blockwise_topk_receivers(
    features: torch.Tensor,
    block_size: int,
    topk: int,
    energy_relative_floor: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    norms = features.square().sum(dim=1).sqrt()
    floor = max(float(norms.median()) * energy_relative_floor, 1e-12)
    valid = norms > floor
    normalized = features / norms.clamp_min(floor).unsqueeze(1)
    normalized_device = normalized.to(device)
    width = normalized.shape[0]
    out_values = torch.full((width, topk), -1.0, dtype=torch.float32)
    out_indices = torch.full((width, topk), -1, dtype=torch.long)
    valid_device = valid.to(device)

    for begin in range(0, width, block_size):
        end = min(width, begin + block_size)
        query = normalized_device[begin:end]
        similarities = query @ normalized_device.t()
        similarities = similarities.abs()
        row_ids = torch.arange(begin, end, device=device)
        similarities[torch.arange(end - begin, device=device), row_ids] = -1.0
        legal = valid_device.unsqueeze(0).expand(end - begin, -1).clone()
        legal &= valid_device[begin:end].unsqueeze(1)
        similarities.masked_fill_(~legal, -1.0)
        values, indices = similarities.topk(topk, dim=1)
        out_values[begin:end] = values.cpu()
        out_indices[begin:end] = indices.cpu()
        del query, similarities, values, indices
    del normalized_device
    return out_values, out_indices


def _fit_no_intercept(activations: torch.Tensor, prune: int, keep: int, ridge: float):
    source = activations[:, prune].float()
    receiver = activations[:, keep].float()
    source_energy = source.square().mean()
    receiver_energy = receiver.square().mean()
    cross = (source * receiver).mean()
    alpha = cross / (receiver_energy + ridge)
    residual = (source - alpha * receiver).square().mean()
    return float(alpha), float(residual), float(source_energy)


def plan_similarity_pruning(method: str, layer, activations: torch.Tensor, ratio: float, config: dict, device):
    width = activations.shape[1]
    target = int(round(width * ratio))
    fit_count = int(round(activations.shape[0] * float(config["similarity_fit_fraction"])))
    if fit_count <= 0 or fit_count >= activations.shape[0]:
        raise ValueError("similarity_fit_fraction must leave non-empty fit and holdout splits")
    fit_activations = activations[:fit_count]
    validation_activations = activations[fit_count:]
    direct_selection_method = config.get("similarity_direct_method", "output_importance")
    if direct_selection_method == "flap":
        # Match the standalone FLAP baseline exactly. The split below is only
        # for fitting and validating similarity-based merge coefficients.
        importance = flap_importance(activations, layer.mlp.down_proj.weight)
    elif direct_selection_method == "output_importance":
        importance = output_importance(fit_activations, layer.mlp.down_proj.weight)
    else:
        raise ValueError(
            "similarity_direct_method must be 'output_importance' or 'flap'"
        )
    median_importance = float(importance.median().clamp_min(1e-30))
    dead_threshold = median_importance * float(config["dead_importance_median_ratio"])
    dead_order = torch.argsort(importance)
    direct = [int(j) for j in dead_order if float(importance[j]) <= dead_threshold][:target]

    protect_count = int(round(width * float(config["protect_top_importance_fraction"])))
    protected = torch.zeros(width, dtype=torch.bool)
    if protect_count:
        protected[torch.topk(importance, protect_count).indices] = True
    if direct:
        protected[torch.tensor(direct)] = False

    features = _row_features(method, layer, fit_activations, config)
    similarities, receivers = blockwise_topk_receivers(
        features,
        int(config["similarity_block_size"]),
        int(config["topk_receivers"]),
        float(config["activation_energy_relative_floor"]),
        device,
    )
    output_norm_sq = layer.mlp.down_proj.weight.detach().float().square().sum(dim=0).to(device)
    candidates = []
    selection_rule = config.get("selection_rule", "joint_cost")
    context_cost_rules = {
        "context_ucb_cost",
        "context_median_cost",
        "context_q75_cost",
        "context_gain_output_cost",
    }
    need_context_stats = (
        selection_rule in context_cost_rules
        or "minimum_context_gain_lcb" in config
    )
    ridge = float(config["ridge_relative"]) * float(
        fit_activations.float().square().mean(dim=0).median().clamp_min(1e-30)
    )
    minimum_similarity = float(config["minimum_source_similarity"])
    direct_set = set(direct)
    activation_device = fit_activations.float().to(device)
    validation_device = validation_activations.float().to(device)
    activation_energy = activation_device.square().mean(dim=0)
    validation_energy = validation_device.square().mean(dim=0)
    validation_direct_cost = (validation_energy * output_norm_sq).cpu()
    context_direct_ucb = validation_direct_cost
    context_direct_median = validation_direct_cost
    context_direct_q75 = validation_direct_cost
    positions_per_context = int(config["sampled_positions_per_sequence"])
    if need_context_stats:
        if validation_device.shape[0] % positions_per_context:
            raise ValueError("Holdout activations must contain complete contexts")
        validation_contexts = validation_device.shape[0] // positions_per_context
        context_energy = validation_device.view(
            validation_contexts, positions_per_context, width
        ).square().mean(dim=1)
        context_direct_cost = context_energy * output_norm_sq.unsqueeze(0)
        ucb_factor = float(config.get("context_ucb_factor", 1.96))
        context_direct_ucb = (
            context_direct_cost.mean(dim=0)
            + ucb_factor
            * context_direct_cost.std(dim=0, unbiased=True)
            / validation_contexts**0.5
        ).cpu()
        context_direct_median = context_direct_cost.median(dim=0).values.cpu()
        context_direct_q75 = torch.quantile(
            context_direct_cost, 0.75, dim=0
        ).cpu()
        del context_energy, context_direct_cost
    importance_device = importance.to(device)
    candidate_block = 128
    for begin in range(0, width, candidate_block):
        end = min(width, begin + candidate_block)
        receiver_block = receivers[begin:end].to(device)
        similarity_block = similarities[begin:end].to(device)
        safe_receivers = receiver_block.clamp_min(0)
        query_indices = torch.arange(begin, end, device=device).unsqueeze(1).expand_as(
            safe_receivers
        )
        query_importance = importance_device.index_select(0, query_indices.flatten()).view_as(
            query_indices
        )
        receiver_importance = importance_device.index_select(
            0, safe_receivers.flatten()
        ).view_as(safe_receivers)
        prune_indices = torch.where(
            query_importance <= receiver_importance, query_indices, safe_receivers
        )
        keep_indices = torch.where(
            query_importance <= receiver_importance, safe_receivers, query_indices
        )
        source = activation_device.index_select(1, prune_indices.flatten()).view(
            activation_device.shape[0], end - begin, receivers.shape[1]
        )
        receiver = activation_device.index_select(1, keep_indices.flatten()).view(
            activation_device.shape[0], end - begin, receivers.shape[1]
        )
        validation_source = validation_device.index_select(
            1, prune_indices.flatten()
        ).view(validation_device.shape[0], end - begin, receivers.shape[1])
        validation_receiver = validation_device.index_select(
            1, keep_indices.flatten()
        ).view(validation_device.shape[0], end - begin, receivers.shape[1])
        cross = (source * receiver).mean(dim=0)
        source_energy = activation_energy.index_select(0, prune_indices.flatten()).view_as(
            prune_indices
        )
        receiver_energy = activation_energy.index_select(0, keep_indices.flatten()).view_as(
            keep_indices
        )
        alpha = cross / (receiver_energy + ridge)
        residual = (
            source_energy - 2.0 * alpha * cross + alpha.square() * receiver_energy
        ).clamp_min(0.0)
        validation_cross = (validation_source * validation_receiver).mean(dim=0)
        validation_source_energy = validation_energy.index_select(
            0, prune_indices.flatten()
        ).view_as(prune_indices)
        validation_receiver_energy = validation_energy.index_select(
            0, keep_indices.flatten()
        ).view_as(keep_indices)
        validation_residual = (
            validation_source_energy
            - 2.0 * alpha * validation_cross
            + alpha.square() * validation_receiver_energy
        ).clamp_min(0.0)
        validation_gain = 1.0 - validation_residual / validation_source_energy.clamp_min(1e-30)
        prune_importance = importance_device.index_select(
            0, prune_indices.flatten()
        ).view_as(prune_indices)
        actual_cost = validation_residual * output_norm_sq.index_select(
            0, prune_indices.flatten()
        ).view_as(prune_indices)
        fit_actual_cost = residual * output_norm_sq.index_select(
            0, prune_indices.flatten()
        ).view_as(prune_indices)
        context_cost_ucb = actual_cost
        context_cost_median = actual_cost
        context_cost_q75 = actual_cost
        context_gain_lcb = validation_gain
        if need_context_stats:
            block_width = end - begin
            receiver_count = receivers.shape[1]
            context_shape = (
                validation_contexts,
                positions_per_context,
                block_width,
                receiver_count,
            )
            context_source_energy = validation_source.square().view(
                context_shape
            ).mean(dim=1)
            context_residual = (
                validation_source - alpha.unsqueeze(0) * validation_receiver
            ).square().view(context_shape).mean(dim=1)
            source_output_norm = output_norm_sq.index_select(
                0, prune_indices.flatten()
            ).view_as(prune_indices)
            context_output_cost = (
                context_residual * source_output_norm.unsqueeze(0)
            )
            ucb_factor = float(config.get("context_ucb_factor", 1.96))
            context_cost_ucb = (
                context_output_cost.mean(dim=0)
                + ucb_factor
                * context_output_cost.std(dim=0, unbiased=True)
                / validation_contexts**0.5
            )
            context_cost_median = context_output_cost.median(dim=0).values
            context_cost_q75 = torch.quantile(
                context_output_cost, 0.75, dim=0
            )
            context_gain = 1.0 - context_residual / context_source_energy.clamp_min(
                1e-30
            )
            confidence_factor = float(
                config.get("context_confidence_factor", 1.96)
            )
            context_gain_lcb = (
                context_gain.mean(dim=0)
                - confidence_factor
                * context_gain.std(dim=0, unbiased=True)
                / validation_contexts**0.5
            )
            del context_source_energy, context_residual, context_output_cost, context_gain
        valid = receiver_block >= 0
        valid &= similarity_block > minimum_similarity
        valid &= validation_gain >= float(config["minimum_validation_reconstruction_gain"])
        if "minimum_context_gain_lcb" in config:
            valid &= context_gain_lcb >= float(config["minimum_context_gain_lcb"])
        valid &= ~protected.to(device).index_select(0, prune_indices.flatten()).view_as(
            prune_indices
        )
        if direct_set:
            direct_lookup = torch.zeros(width, dtype=torch.bool, device=device)
            direct_lookup[torch.tensor(sorted(direct_set), device=device)] = True
            valid &= ~direct_lookup.index_select(0, prune_indices.flatten()).view_as(
                prune_indices
            )
        positions = valid.nonzero(as_tuple=False).cpu().tolist()
        prune_cpu = prune_indices.cpu()
        keep_cpu = keep_indices.cpu()
        similarity_cpu = similarity_block.cpu()
        importance_cpu = prune_importance.cpu()
        actual_cost_cpu = actual_cost.cpu()
        fit_actual_cost_cpu = fit_actual_cost.cpu()
        context_cost_ucb_cpu = context_cost_ucb.cpu()
        context_cost_median_cpu = context_cost_median.cpu()
        context_cost_q75_cpu = context_cost_q75.cpu()
        context_gain_lcb_cpu = context_gain_lcb.cpu()
        alpha_cpu = alpha.cpu()
        validation_gain_cpu = validation_gain.cpu()
        for row, rank in positions:
            prune = int(prune_cpu[row, rank])
            keep = int(keep_cpu[row, rank])
            candidates.append(
                (
                    -float(similarity_cpu[row, rank]),
                    float(importance_cpu[row, rank]),
                    float(actual_cost_cpu[row, rank]),
                    prune,
                    keep,
                    float(similarity_cpu[row, rank]),
                    float(alpha_cpu[row, rank]),
                    float(validation_gain_cpu[row, rank]),
                    float(fit_actual_cost_cpu[row, rank]),
                    float(context_cost_ucb_cpu[row, rank]),
                    float(context_cost_median_cpu[row, rank]),
                    float(context_cost_q75_cpu[row, rank]),
                    float(context_gain_lcb_cpu[row, rank]),
                )
            )
        del source, receiver, validation_source, validation_receiver, cross, alpha, residual, actual_cost
    del activation_device, validation_device, activation_energy, validation_energy, output_norm_sq
    deduplicated = {}
    for candidate in candidates:
        key = (candidate[3], candidate[4])
        if key not in deduplicated or candidate < deduplicated[key]:
            deduplicated[key] = candidate
    candidates = list(deduplicated.values())

    merges = []
    pruned = set(direct)
    keeper_load: dict[int, int] = {}
    protected_keepers = set()
    if selection_rule == "pair_first":
        operations = [
            (candidate[0], candidate[1], 0, candidate) for candidate in candidates
        ]
    elif selection_rule in {
        "joint_cost",
        "actual_cost",
        "validation_cost",
        "robust_cost",
        "adaptive_output_cost",
        "adaptive_crossfit_cost",
        "context_ucb_cost",
        "context_median_cost",
        "context_q75_cost",
        "context_gain_output_cost",
    }:
        operations = []
        for candidate in candidates:
            (
                _,
                prune_importance,
                actual_cost,
                prune,
                keep,
                similarity,
                alpha,
                validation_gain,
                fit_actual_cost,
                context_cost_ucb,
                context_cost_median,
                context_cost_q75,
                context_gain_lcb,
            ) = candidate
            predicted_cost = prune_importance * max(0.0, 1.0 - similarity * similarity)
            validated_cost = prune_importance * max(0.0, 1.0 - validation_gain)
            if selection_rule == "joint_cost":
                operations.append((predicted_cost, actual_cost, 0, candidate))
            elif selection_rule in {"actual_cost", "adaptive_output_cost"}:
                operations.append((actual_cost, predicted_cost, 0, candidate))
            elif selection_rule == "adaptive_crossfit_cost":
                operations.append(
                    (max(actual_cost, fit_actual_cost), predicted_cost, 0, candidate)
                )
            elif selection_rule == "context_ucb_cost":
                operations.append((context_cost_ucb, predicted_cost, 0, candidate))
            elif selection_rule == "context_median_cost":
                operations.append((context_cost_median, predicted_cost, 0, candidate))
            elif selection_rule == "context_q75_cost":
                operations.append((context_cost_q75, predicted_cost, 0, candidate))
            elif selection_rule == "context_gain_output_cost":
                operations.append((actual_cost, predicted_cost, 0, candidate))
            elif selection_rule == "validation_cost":
                operations.append((validated_cost, predicted_cost, 0, candidate))
            else:
                operations.append(
                    (max(predicted_cost, validated_cost), actual_cost, 0, candidate)
                )
        for neuron in range(width):
            if protected[neuron] or neuron in pruned:
                continue
            if selection_rule == "adaptive_output_cost":
                direct_cost = float(validation_direct_cost[neuron])
            elif selection_rule == "adaptive_crossfit_cost":
                direct_cost = max(
                    float(validation_direct_cost[neuron]),
                    float(importance[neuron]),
                )
            elif selection_rule == "context_ucb_cost":
                direct_cost = float(context_direct_ucb[neuron])
            elif selection_rule == "context_median_cost":
                direct_cost = float(context_direct_median[neuron])
            elif selection_rule == "context_q75_cost":
                direct_cost = float(context_direct_q75[neuron])
            elif selection_rule == "context_gain_output_cost":
                direct_cost = float(validation_direct_cost[neuron])
            else:
                direct_cost = float(importance[neuron])
            operations.append(
                (direct_cost, float(importance[neuron]), 1, neuron)
            )
    else:
        raise ValueError(f"Unknown selection_rule: {selection_rule}")
    operations.sort(key=lambda row: (row[0], row[1], row[2]))
    method_merge_fractions = config.get("max_merge_fraction_by_method", {})
    base_method = method.removesuffix("_flap")
    merge_fraction = method_merge_fractions.get(
        method,
        method_merge_fractions.get(base_method, config.get("max_merge_fraction", 1.0)),
    )
    if not 0.0 <= float(merge_fraction) <= 1.0:
        raise ValueError("max_merge_fraction must be between 0 and 1")
    merge_limit = int(round(target * float(merge_fraction)))

    for _, _, operation_type, payload in operations:
        if len(pruned) >= target:
            break
        if operation_type == 1:
            neuron = int(payload)
            if neuron in pruned or neuron in protected_keepers or protected[neuron]:
                continue
            direct.append(neuron)
            pruned.add(neuron)
            continue
        (
            _,
            _,
            actual_cost,
            prune,
            keep,
            similarity,
            alpha,
            validation_gain,
            _,
            _,
            _,
            _,
            _,
        ) = payload
        if len(merges) >= merge_limit:
            continue
        if prune in pruned or keep in pruned or prune in protected_keepers:
            continue
        if keeper_load.get(keep, 0) >= int(config["max_merges_per_keeper"]):
            continue
        merges.append(Merge(prune, keep, similarity, alpha, actual_cost, validation_gain))
        pruned.add(prune)
        protected_keepers.add(keep)
        keeper_load[keep] = keeper_load.get(keep, 0) + 1

    if len(pruned) < target:
        for neuron in dead_order.tolist():
            if len(pruned) >= target:
                break
            if neuron in pruned or neuron in protected_keepers or protected[neuron]:
                continue
            direct.append(int(neuron))
            pruned.add(int(neuron))
    if len(pruned) != target:
        raise RuntimeError(f"Could only prune {len(pruned)} of {target} requested neurons")
    plan = {
        "method": method,
        "target": target,
        "direct": direct,
        "merges": merges,
        "pruned": sorted(pruned),
        "importance_median": median_importance,
        "dead_threshold": dead_threshold,
        "selection_rule": selection_rule,
        "direct_selection_method": direct_selection_method,
    }
    if config.get("direct_bias_compensation", False) and direct:
        means = activations.float().mean(dim=0)
        down = layer.mlp.down_proj.weight.detach().float().cpu()
        direct_indices = torch.tensor(direct, dtype=torch.long)
        plan["bias_compensation"] = down[:, direct_indices] @ means[direct_indices]
    return plan


def _replace_linear(linear: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None):
    replacement = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    replacement.weight.data.copy_(weight)
    if bias is not None:
        replacement.bias.data.copy_(bias)
    return replacement


@torch.no_grad()
def apply_layer_plan(layer, plan: dict):
    prune_set = set(plan["pruned"])
    width = layer.mlp.gate_proj.weight.shape[0]
    keep_indices = torch.tensor(
        [index for index in range(width) if index not in prune_set],
        device=layer.mlp.gate_proj.weight.device,
    )
    down_original = layer.mlp.down_proj.weight.detach().clone()
    down_updated = down_original.clone()
    for merge in plan.get("merges", []):
        keep = merge["keep"] if isinstance(merge, dict) else merge.keep
        prune = merge["prune"] if isinstance(merge, dict) else merge.prune
        alpha = merge["alpha"] if isinstance(merge, dict) else merge.alpha
        down_updated[:, keep].add_(alpha * down_original[:, prune])
    if "bias_compensation" in plan:
        existing = layer.mlp.down_proj.bias
        bias = torch.zeros(
            down_updated.shape[0], device=down_updated.device, dtype=down_updated.dtype
        ) if existing is None else existing.detach().clone()
        compensation = plan["bias_compensation"]
        if not isinstance(compensation, torch.Tensor):
            compensation = torch.tensor(compensation)
        bias.add_(compensation.to(bias.device, bias.dtype))
    else:
        bias = None if layer.mlp.down_proj.bias is None else layer.mlp.down_proj.bias.detach().clone()

    gate_weight = layer.mlp.gate_proj.weight.index_select(0, keep_indices).contiguous()
    up_weight = layer.mlp.up_proj.weight.index_select(0, keep_indices).contiguous()
    down_weight = down_updated.index_select(1, keep_indices).contiguous()
    layer.mlp.gate_proj = _replace_linear(layer.mlp.gate_proj, gate_weight, None)
    layer.mlp.up_proj = _replace_linear(layer.mlp.up_proj, up_weight, None)
    layer.mlp.down_proj = _replace_linear(layer.mlp.down_proj, down_weight, bias)
    layer.mlp.intermediate_size = int(keep_indices.numel())


def plan_importance_method(method: str, layer, activations: torch.Tensor, ratio: float):
    width = activations.shape[1]
    target = int(round(width * ratio))
    down = layer.mlp.down_proj.weight.detach().float().cpu()
    if method == "magnitude":
        scores = down.norm(dim=0)
    elif method == "wanda":
        activation_rms = activations.float().square().mean(dim=0).sqrt()
        scores = down.abs().mean(dim=0) * activation_rms
    elif method == "flap":
        variance = activations.float().var(dim=0, unbiased=False)
        scores = variance * down.square().sum(dim=0)
    elif method == "importance":
        scores = output_importance(activations, layer.mlp.down_proj.weight)
    else:
        raise ValueError(f"Unsupported importance method: {method}")
    pruned = torch.topk(scores, target, largest=False).indices.sort().values
    plan = {"method": method, "target": target, "direct": pruned.tolist(), "merges": [], "pruned": pruned.tolist()}
    if method == "flap":
        means = activations.float().mean(dim=0)
        plan["bias_compensation"] = down[:, pruned] @ means[pruned]
    return plan


def build_and_apply_plans(model, method: str, ratio: float, config: dict, device):
    plans = []
    for layer_index, layer in enumerate(llama_layers(model)):
        print(f"Planning {method} ratio={ratio:.3f} layer={layer_index:02d}", flush=True)
        activations = load_layer_activations(config["activation_dir"], layer_index)
        if method in {"fc", "is", "fc_flap", "is_flap"}:
            plan = plan_similarity_pruning(method, layer, activations, ratio, config, device)
        else:
            plan = plan_importance_method(method, layer, activations, ratio)
        plans.append(plan)
        del activations
    for layer, plan in zip(llama_layers(model), plans):
        apply_layer_plan(layer, plan)
    model.config.intermediate_size = llama_layers(model)[0].mlp.intermediate_size
    return plans


class _StopAfterFirstLayerInput(Exception):
    pass


@torch.inference_mode()
def _capture_first_layer_inputs(model, calibration: dict, device):
    input_ids = calibration["input_ids"]
    hidden_states = torch.empty(
        (
            input_ids.shape[0],
            input_ids.shape[1],
            model.config.hidden_size,
        ),
        dtype=next(model.parameters()).dtype,
        device="cpu",
    )
    state = {"sequence": 0}
    layer_kwargs = {}
    layers = llama_layers(model)
    original_first_layer = layers[0]

    class InputCatcher(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, current_hidden_states, *args, **kwargs):
            sequence = state["sequence"]
            hidden_states[sequence].copy_(
                current_hidden_states[0].detach().to("cpu")
            )
            if not layer_kwargs:
                for name in (
                    "attention_mask",
                    "position_ids",
                    "cache_position",
                    "position_embeddings",
                ):
                    if name in kwargs:
                        layer_kwargs[name] = kwargs[name]
            raise _StopAfterFirstLayerInput

    layers[0] = InputCatcher(original_first_layer)
    try:
        for sequence_index, sequence in enumerate(input_ids):
            state["sequence"] = sequence_index
            try:
                model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
            except _StopAfterFirstLayerInput:
                pass
    finally:
        layers[0] = original_first_layer
    return hidden_states, layer_kwargs


@torch.inference_mode()
def _collect_current_layer_activations(
    layer,
    hidden_states: torch.Tensor,
    sampled_positions: torch.Tensor,
    layer_kwargs: dict,
    device,
):
    samples_per_sequence = sampled_positions.shape[1]
    activations = torch.empty(
        (hidden_states.shape[0] * samples_per_sequence, layer.mlp.intermediate_size),
        dtype=torch.float16,
        device="cpu",
    )
    state = {"sequence": 0}

    def capture_down_projection_input(_module, args):
        sequence = state["sequence"]
        current = args[0][0]
        selected = current.index_select(
            0, sampled_positions[sequence].to(current.device)
        )
        begin = sequence * samples_per_sequence
        activations[begin : begin + samples_per_sequence].copy_(
            selected.detach().to("cpu", torch.float16)
        )

    handle = layer.mlp.down_proj.register_forward_pre_hook(
        capture_down_projection_input
    )
    try:
        for sequence_index, sequence_hidden_states in enumerate(hidden_states):
            state["sequence"] = sequence_index
            layer(
                sequence_hidden_states.unsqueeze(0).to(device),
                use_cache=False,
                **layer_kwargs,
            )
    finally:
        handle.remove()
    return activations


@torch.inference_mode()
def _propagate_pruned_layer(layer, hidden_states, layer_kwargs, device):
    outputs = torch.empty_like(hidden_states, device="cpu")
    for sequence_index, sequence_hidden_states in enumerate(hidden_states):
        current_output = layer(
            sequence_hidden_states.unsqueeze(0).to(device),
            use_cache=False,
            **layer_kwargs,
        )
        outputs[sequence_index].copy_(current_output[0].detach().to("cpu"))
    return outputs


def build_and_apply_plans_sequential(
    model, method: str, ratio: float, config: dict, device
):
    calibration = load_calibration(config["calibration_path"])
    sampled_positions = calibration["sampled_positions"]
    hidden_states, layer_kwargs = _capture_first_layer_inputs(
        model, calibration, device
    )
    plans = []
    for layer_index, layer in enumerate(llama_layers(model)):
        print(
            f"Sequential planning {method} ratio={ratio:.3f} "
            f"layer={layer_index:02d}",
            flush=True,
        )
        activations = _collect_current_layer_activations(
            layer,
            hidden_states,
            sampled_positions,
            layer_kwargs,
            device,
        )
        if method in {"fc", "is"}:
            plan = plan_similarity_pruning(
                method, layer, activations, ratio, config, device
            )
        else:
            plan = plan_importance_method(method, layer, activations, ratio)
        plans.append(plan)
        apply_layer_plan(layer, plan)
        next_hidden_states = _propagate_pruned_layer(
            layer, hidden_states, layer_kwargs, device
        )
        del activations, hidden_states
        hidden_states = next_hidden_states
    del hidden_states
    model.config.intermediate_size = llama_layers(model)[0].mlp.intermediate_size
    return plans
