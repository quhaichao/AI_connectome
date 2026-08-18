from __future__ import annotations

import torch

from .modeling import llama_layers
from .pruning import apply_layer_plan


def _moments(layer_stats: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(layer_stats["count"])
    mean = layer_stats["sum"].float() / count
    second = layer_stats["sum_sq"].float() / count
    variance = (second - mean.square()).clamp_min(0.0)
    return mean, second, variance


def plan_full_stat_method(method: str, layer, layer_stats: dict, ratio: float) -> dict:
    width = layer.mlp.gate_proj.out_features
    target = int(round(width * ratio))
    mean, second, variance = _moments(layer_stats)
    down = layer.mlp.down_proj.weight.detach().float().cpu()
    down_norm_sq = down.square().sum(dim=0)

    if method == "wanda_sp_official":
        # Official Wanda-sp/FLAP-repository structured adaptation, up to a
        # channel-independent normalization constant.
        scores = down.abs().mean(dim=0) * second.sqrt()
    elif method == "flap_official":
        scores = variance * down_norm_sq
    elif method == "ocp_probe_static":
        probe_count = int(layer_stats["probe_count"])
        probe_second = layer_stats["probe_sum_sq"].float() / probe_count
        scores = down.abs().mean(dim=0) * probe_second.sqrt()
    else:
        raise ValueError(f"Unsupported full-stat reproduction method: {method}")

    pruned = torch.topk(scores, target, largest=False).indices.sort().values
    plan = {
        "method": method,
        "target": target,
        "direct": pruned.tolist(),
        "merges": [],
        "pruned": pruned.tolist(),
    }
    if method == "flap_official":
        plan["bias_compensation"] = down[:, pruned] @ mean[pruned]
    return plan


def build_and_apply_full_stat_plans(
    model, method: str, ratio: float, statistics: dict
) -> list[dict]:
    layers = llama_layers(model)
    if len(statistics["layers"]) != len(layers):
        raise ValueError("Statistics and model have different layer counts")
    plans = []
    for layer_index, (layer, layer_stats) in enumerate(
        zip(layers, statistics["layers"])
    ):
        print(
            f"Planning {method} ratio={ratio:.3f} layer={layer_index:02d}",
            flush=True,
        )
        plans.append(plan_full_stat_method(method, layer, layer_stats, ratio))
    for layer, plan in zip(layers, plans):
        apply_layer_plan(layer, plan)
    model.config.intermediate_size = layers[0].mlp.intermediate_size
    return plans


def build_and_apply_adaptive_flap_plans(
    model, ratio: float, statistics: dict
) -> list[dict]:
    """Apply FLAP's adaptive-layer global WIFV ranking to FFN channels."""
    layers = llama_layers(model)
    if len(statistics["layers"]) != len(layers):
        raise ValueError("Statistics and model have different layer counts")
    scores = []
    moments = []
    for layer, layer_stats in zip(layers, statistics["layers"]):
        mean, _second, variance = _moments(layer_stats)
        down = layer.mlp.down_proj.weight.detach().float().cpu()
        metric = variance * down.square().sum(dim=0)
        standardized = (metric - metric.mean()) / metric.std().clamp_min(1e-12)
        scores.append(standardized)
        moments.append((mean, down))

    width = scores[0].numel()
    total = int(round(len(layers) * width * ratio))
    selected = torch.topk(torch.cat(scores), total, largest=False).indices
    plans = []
    for layer_index, (layer, (mean, down)) in enumerate(zip(layers, moments)):
        begin = layer_index * width
        pruned = (selected[(selected >= begin) & (selected < begin + width)] - begin)
        pruned = pruned.sort().values
        plan = {
            "method": "flap_adaptive_layer_ffn",
            "target": int(pruned.numel()),
            "ratio": float(pruned.numel() / width),
            "direct": pruned.tolist(),
            "merges": [],
            "pruned": pruned.tolist(),
            "bias_compensation": down[:, pruned] @ mean[pruned],
        }
        apply_layer_plan(layer, plan)
        plans.append(plan)
    return plans
