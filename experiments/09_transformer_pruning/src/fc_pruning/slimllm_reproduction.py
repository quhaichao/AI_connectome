from __future__ import annotations

import time

import torch
from torch import nn

from .modeling import decoder_layers
from .progress import report_progress


SLIMLLM_SELECTION_MODES = ("official_piecewise", "direct_lowest")


def select_slimllm_channels(
    scores: torch.Tensor,
    target: int,
    mode: str,
    piecewise_fraction: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pruned and kept indices using the two documented SlimLLM rules."""
    if scores.ndim != 1:
        raise ValueError("SlimLLM scores must be one-dimensional")
    width = scores.numel()
    if not 0 < target < width:
        raise ValueError(f"target must be in (0, {width}); got {target}")
    if mode not in SLIMLLM_SELECTION_MODES:
        raise ValueError(f"Unsupported SlimLLM selection mode: {mode}")

    order = torch.argsort(scores.float().cpu(), stable=True)
    if mode == "official_piecewise":
        offset = int(round(piecewise_fraction * width))
        if offset + target > width:
            raise ValueError("Piecewise offset and target exceed channel width")
        pruned = order[offset : offset + target]
    else:
        offset = 0
        pruned = order[:target]
    keep_mask = torch.ones(width, dtype=torch.bool)
    keep_mask[pruned] = False
    keep = keep_mask.nonzero(as_tuple=False).flatten()
    return pruned.sort().values, keep


@torch.inference_mode()
def slimllm_importance_scores(
    layer,
    hidden_second_moment: torch.Tensor,
    intermediate_second_moment: torch.Tensor,
    output_second_moment: torch.Tensor,
    intermediate_sum: torch.Tensor,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Implement SlimLLM equations 5-8 from full-token sufficient statistics."""
    gate = layer.mlp.gate_proj.weight.detach().float().to(device)
    up = layer.mlp.up_proj.weight.detach().float().to(device)
    down_t = layer.mlp.down_proj.weight.detach().float().t().to(device)
    l2_hidden = hidden_second_moment.diagonal().float().clamp_min(0).sqrt().to(device)
    l2_intermediate = intermediate_second_moment.float().clamp_min(0).sqrt().to(device)

    intermediate_sum = intermediate_sum.float().to(device)
    output_sum = down_t.t() @ intermediate_sum
    covariance = output_second_moment.float().to(device)
    covariance = covariance - torch.outer(output_sum, output_sum) / float(count)
    covariance = (covariance + covariance.t()).mul_(0.5)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.abs()
    eigenvalue_weights = torch.sigmoid(
        eigenvalues / eigenvalues.mean().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    )
    mapped_down = (down_t @ eigenvectors).abs()
    mapped_down.mul_(eigenvalue_weights.unsqueeze(0))

    gate_score = torch.linalg.vector_norm(gate * l2_hidden.unsqueeze(0), dim=1)
    up_score = torch.linalg.vector_norm(up * l2_hidden.unsqueeze(0), dim=1)
    down_score = torch.linalg.vector_norm(
        mapped_down * l2_intermediate.unsqueeze(1), dim=1
    )
    scores = gate_score + up_score + down_score
    audit = {
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "gate_score_mean": float(gate_score.mean()),
        "up_score_mean": float(up_score.mean()),
        "down_score_mean": float(down_score.mean()),
        "centered_output_trace": float(covariance.diagonal().sum()),
        "pca_eigenvalue_mean": float(eigenvalues.mean()),
        "pca_eigenvalue_max": float(eigenvalues.max()),
    }
    return scores.cpu(), audit


@torch.inference_mode()
def fit_slimllm_output_affine(
    down_weight: torch.Tensor,
    hessian: torch.Tensor,
    intermediate_sum: torch.Tensor,
    output_second_moment: torch.Tensor,
    pruned: torch.Tensor,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit SlimLLM's independent output-wise affine recovery from moments.

    The dense output is Y=ZW^T and the retained output is X=Y-Z_P W_P^T.
    Computing the diagonal covariance terms through the pruned block avoids
    materializing all 262,144 calibration outputs.
    """
    weight = down_weight.detach().float().to(device)
    pruned = pruned.long().to(device)
    sums = intermediate_sum.float().to(device)
    pruned_weight = weight.index_select(1, pruned)
    pruned_sum = sums.index_select(0, pruned)
    output_sum = weight @ sums
    removed_sum = pruned_weight @ pruned_sum
    retained_sum = output_sum - removed_sum

    hessian_rows = hessian.index_select(0, pruned.cpu()).float().to(device)
    centered_rows = hessian_rows - torch.outer(pruned_sum, sums) / float(count)
    cov_removed_dense = (centered_rows @ weight.t()).t()
    cov_removed_output = (pruned_weight * cov_removed_dense).sum(dim=1)

    centered_pruned = centered_rows.index_select(1, pruned)
    removed_variance = (
        (pruned_weight @ centered_pruned) * pruned_weight
    ).sum(dim=1)
    dense_variance = output_second_moment.diagonal().float().to(device)
    dense_variance = dense_variance - output_sum.square() / float(count)
    retained_variance = dense_variance + removed_variance - 2.0 * cov_removed_output
    retained_dense_covariance = dense_variance - cov_removed_output

    scale_floor = (
        torch.finfo(torch.float32).eps
        * dense_variance.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
    )
    valid = retained_variance > scale_floor
    scale = torch.ones_like(retained_variance)
    scale[valid] = retained_dense_covariance[valid] / retained_variance[valid]
    bias = (output_sum - scale * retained_sum) / float(count)
    if not torch.isfinite(scale).all() or not torch.isfinite(bias).all():
        raise RuntimeError("SlimLLM affine recovery produced non-finite coefficients")

    audit = {
        "scale_min": float(scale.min()),
        "scale_mean": float(scale.mean()),
        "scale_max": float(scale.max()),
        "bias_abs_mean": float(bias.abs().mean()),
        "invalid_variance_dimensions": int((~valid).sum()),
        "retained_variance_min": float(retained_variance.min()),
        "fit": "independent_output_affine_closed_form_equivalent_to_np_polyfit",
    }
    return scale.cpu(), bias.cpu(), audit


def _replace_linear(
    linear: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None
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
def apply_slimllm_plan(layer, plan: dict, use_regression: bool) -> None:
    width = layer.mlp.gate_proj.out_features
    pruned = torch.tensor(plan["pruned"], dtype=torch.long)
    keep_mask = torch.ones(width, dtype=torch.bool)
    keep_mask[pruned] = False
    keep = keep_mask.nonzero(as_tuple=False).flatten().to(
        layer.mlp.gate_proj.weight.device
    )

    gate_weight = layer.mlp.gate_proj.weight.detach().index_select(0, keep)
    up_weight = layer.mlp.up_proj.weight.detach().index_select(0, keep)
    down_weight = layer.mlp.down_proj.weight.detach().index_select(1, keep)
    bias = None
    if use_regression:
        scale = torch.tensor(
            plan["affine_scale"], device=down_weight.device, dtype=down_weight.dtype
        )
        bias = torch.tensor(
            plan["affine_bias"], device=down_weight.device, dtype=down_weight.dtype
        )
        down_weight = down_weight * scale.unsqueeze(1)

    layer.mlp.gate_proj = _replace_linear(
        layer.mlp.gate_proj, gate_weight.contiguous(), None
    )
    layer.mlp.up_proj = _replace_linear(
        layer.mlp.up_proj, up_weight.contiguous(), None
    )
    layer.mlp.down_proj = _replace_linear(
        layer.mlp.down_proj, down_weight.contiguous(), bias
    )
    layer.mlp.intermediate_size = int(keep.numel())


@torch.no_grad()
def apply_slimllm_plans(model, plans: list[dict], use_regression: bool) -> None:
    layers = decoder_layers(model)
    if len(layers) != len(plans):
        raise ValueError("SlimLLM plans and model have different layer counts")
    for layer, plan in zip(layers, plans):
        apply_slimllm_plan(layer, plan, use_regression)
    widths = {layer.mlp.intermediate_size for layer in layers}
    if len(widths) != 1:
        raise RuntimeError(f"SlimLLM fixed-layer plans produced widths {widths}")
    model.config.intermediate_size = widths.pop()


@torch.inference_mode()
def build_slimllm_plans(
    model,
    full_statistics: dict,
    covariance_statistics: dict,
    hessian_statistics: dict,
    ratio: float,
    device: torch.device,
    modes: tuple[str, ...] = SLIMLLM_SELECTION_MODES,
) -> dict[str, list[dict]]:
    layers = decoder_layers(model)
    expected = 128 * 2048
    protocols = (
        full_statistics["protocol"],
        covariance_statistics["protocol"],
        hessian_statistics["protocol"],
    )
    if any(int(protocol["activation_tokens"]) != expected for protocol in protocols):
        raise ValueError("SlimLLM reproduction requires exactly 128x2048 tokens")
    if any(len(payload["layers"]) != len(layers) for payload in (
        full_statistics, covariance_statistics, hessian_statistics
    )):
        raise ValueError("SlimLLM statistics and model have different layer counts")

    width = layers[0].mlp.gate_proj.out_features
    target = int(round(width * ratio))
    invalid_modes = set(modes) - set(SLIMLLM_SELECTION_MODES)
    if invalid_modes:
        raise ValueError(f"Unsupported SlimLLM selection modes: {sorted(invalid_modes)}")
    plans = {mode: [] for mode in modes}
    started_at = time.monotonic()
    for layer_index, layer in enumerate(layers):
        full = full_statistics["layers"][layer_index]
        covariance = covariance_statistics["layers"][layer_index]
        count = int(full["count"])
        if count != int(covariance["count"]):
            raise ValueError(f"Layer {layer_index} statistic counts differ")
        scores, score_audit = slimllm_importance_scores(
            layer,
            covariance["hidden_covariance"],
            full["sum_sq"],
            covariance["down_output_covariance"],
            full["sum"],
            count,
            device,
        )
        for mode in modes:
            pruned, _keep = select_slimllm_channels(scores, target, mode)
            scale, bias, fit_audit = fit_slimllm_output_affine(
                layer.mlp.down_proj.weight,
                hessian_statistics["layers"][layer_index],
                full["sum"],
                covariance["down_output_covariance"],
                pruned,
                count,
                device,
            )
            offset = int(round(0.01 * width)) if mode == "official_piecewise" else 0
            plans[mode].append(
                {
                    "layer": layer_index,
                    "method": "slimllm_ffn_fixed_20pct",
                    "selection_mode": mode,
                    "target": target,
                    "original_width": width,
                    "kept": width - target,
                    "sort_interval": [offset, offset + target],
                    "pruned": pruned.tolist(),
                    "affine_scale": scale.tolist(),
                    "affine_bias": bias.tolist(),
                    "score_audit": score_audit,
                    "fit_audit": fit_audit,
                }
            )
        report_progress(
            "SlimLLM plans",
            layer_index + 1,
            len(layers),
            started_at,
            detail=f"layer={layer_index:02d}",
        )
        del scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return plans
