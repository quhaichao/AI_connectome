from __future__ import annotations

import torch
from torch import nn


def _replace_linear(
    linear: nn.Linear,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> nn.Linear:
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
def apply_layer_plan(layer, plan: dict) -> None:
    """Apply one structured gated-FFN plan to a Llama or Qwen2 decoder layer."""
    prune_set = {int(index) for index in plan["pruned"]}
    width = int(layer.mlp.gate_proj.weight.shape[0])
    if any(index < 0 or index >= width for index in prune_set):
        raise IndexError(f"Pruned indices must be within [0, {width - 1}]")
    target = int(plan.get("target", len(prune_set)))
    if len(prune_set) != target:
        raise ValueError(
            f"Plan target is {target}, but it contains {len(prune_set)} unique indices"
        )

    keep_indices = torch.tensor(
        [index for index in range(width) if index not in prune_set],
        device=layer.mlp.gate_proj.weight.device,
        dtype=torch.long,
    )
    down_original = layer.mlp.down_proj.weight.detach().clone()
    down_updated = down_original.clone()
    for merge in plan.get("merges", []):
        keep = int(merge["keep"] if isinstance(merge, dict) else merge.keep)
        prune = int(merge["prune"] if isinstance(merge, dict) else merge.prune)
        alpha = float(merge["alpha"] if isinstance(merge, dict) else merge.alpha)
        if prune not in prune_set:
            raise ValueError(f"Merge source {prune} is not in the pruning mask")
        if keep in prune_set:
            raise ValueError(f"Merge keeper {keep} is also pruned")
        down_updated[:, keep].add_(alpha * down_original[:, prune])

    existing_bias = layer.mlp.down_proj.bias
    bias = None if existing_bias is None else existing_bias.detach().clone()
    if "bias_compensation" in plan:
        if bias is None:
            bias = torch.zeros(
                down_updated.shape[0],
                device=down_updated.device,
                dtype=down_updated.dtype,
            )
        compensation = torch.as_tensor(
            plan["bias_compensation"],
            device=bias.device,
            dtype=bias.dtype,
        )
        if compensation.shape != bias.shape:
            raise ValueError(
                f"Bias compensation shape {tuple(compensation.shape)} "
                f"does not match down-projection output {tuple(bias.shape)}"
            )
        bias.add_(compensation)

    gate_weight = layer.mlp.gate_proj.weight.index_select(
        0, keep_indices
    ).contiguous()
    up_weight = layer.mlp.up_proj.weight.index_select(0, keep_indices).contiguous()
    down_weight = down_updated.index_select(1, keep_indices).contiguous()
    gate_bias = (
        None
        if layer.mlp.gate_proj.bias is None
        else layer.mlp.gate_proj.bias.index_select(0, keep_indices).contiguous()
    )
    up_bias = (
        None
        if layer.mlp.up_proj.bias is None
        else layer.mlp.up_proj.bias.index_select(0, keep_indices).contiguous()
    )
    layer.mlp.gate_proj = _replace_linear(
        layer.mlp.gate_proj, gate_weight, gate_bias
    )
    layer.mlp.up_proj = _replace_linear(
        layer.mlp.up_proj, up_weight, up_bias
    )
    layer.mlp.down_proj = _replace_linear(layer.mlp.down_proj, down_weight, bias)
    layer.mlp.intermediate_size = int(keep_indices.numel())
