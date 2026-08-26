from __future__ import annotations

from torch import nn
import torch


def _linear(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
    module = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    module.weight.data.copy_(weight)
    if bias is not None:
        module.bias.data.copy_(bias)
    return module


def _fixed_mask_indices(plan: dict, width: int, device: torch.device):
    pruned = torch.tensor(sorted(plan["pruned"]), dtype=torch.long, device=device)
    keep_mask = torch.ones(width, dtype=torch.bool, device=device)
    keep_mask[pruned] = False
    keep = keep_mask.nonzero(as_tuple=False).flatten()
    return pruned, keep


def _plan_bias(plan: dict, output_size: int, device: torch.device) -> torch.Tensor:
    value = plan.get("bias_compensation")
    if isinstance(value, torch.Tensor):
        return value.float().to(device).clone()
    if value is None:
        return torch.zeros(output_size, dtype=torch.float32, device=device)
    return torch.tensor(value, dtype=torch.float32, device=device)


@torch.no_grad()
def apply_fixed_mask_reconstruction(
    layer,
    hessian: torch.Tensor,
    mean: torch.Tensor,
    plan: dict,
    mode: str,
    device: torch.device,
    damping: float = 1e-4,
    covariance_mode: str = "second_moment",
    sample_count: int | None = None,
    merge_residual_weight: float = 0.0,
) -> dict:
    """Apply a frozen FC mask with full or direct-only joint LS compensation."""
    if mode not in {"full_ls", "direct_ls"}:
        raise ValueError("mode must be full_ls or direct_ls")
    if covariance_mode not in {"second_moment", "centered"}:
        raise ValueError("covariance_mode must be second_moment or centered")
    if covariance_mode == "centered" and (sample_count is None or sample_count <= 0):
        raise ValueError("centered covariance requires a positive sample_count")
    if not 0.0 <= merge_residual_weight <= 1.0:
        raise ValueError("merge_residual_weight must be between zero and one")
    if mode == "full_ls" and merge_residual_weight:
        raise ValueError("merge_residual_weight is only valid for direct_ls")
    gate = layer.mlp.gate_proj
    up = layer.mlp.up_proj
    down = layer.mlp.down_proj
    width = gate.out_features
    pruned, keep = _fixed_mask_indices(plan, width, device)
    weight = down.weight.detach().float().to(device)
    normalizer = sample_count if sample_count is not None else hessian.shape[0]
    matrix = hessian.float().to(device) / normalizer
    mean_device = mean.float().to(device)
    if covariance_mode == "centered":
        matrix.addmm_(mean_device[:, None], mean_device[None, :], beta=1.0, alpha=-1.0)
    diagonal_scale = matrix.diagonal().mean().clamp_min(1e-12)
    matrix.diagonal().add_(damping * diagonal_scale)
    hkk = matrix.index_select(0, keep).index_select(1, keep)
    keep_cholesky = torch.linalg.cholesky(hkk)

    if mode == "full_ls":
        reconstruction_sources = pruned
        base_weight = weight.index_select(1, keep)
    else:
        reconstruction_sources = torch.tensor(
            sorted(plan["direct"]), dtype=torch.long, device=device
        )
        updated = weight.clone()
        for merge in plan.get("merges", []):
            source = int(merge["prune"])
            keeper = int(merge["keep"])
            updated[:, keeper].add_(float(merge["alpha"]) * weight[:, source])
        base_weight = updated.index_select(1, keep)

    if reconstruction_sources.numel():
        hks = matrix.index_select(0, keep).index_select(1, reconstruction_sources)
        rhs = hks @ weight.index_select(1, reconstruction_sources).t()
        delta = torch.cholesky_solve(rhs, keep_cholesky)
    else:
        delta = torch.zeros(
            (keep.numel(), weight.shape[0]), dtype=torch.float32, device=device
        )
    if mode == "direct_ls" and merge_residual_weight and plan.get("merges"):
        merge_sources = torch.tensor(
            [int(item["prune"]) for item in plan["merges"]],
            dtype=torch.long,
            device=device,
        )
        merge_keepers = torch.tensor(
            [int(item["keep"]) for item in plan["merges"]],
            dtype=torch.long,
            device=device,
        )
        merge_alpha = torch.tensor(
            [float(item["alpha"]) for item in plan["merges"]],
            dtype=torch.float32,
            device=device,
        )
        residual_cross = matrix.index_select(0, keep).index_select(1, merge_sources)
        residual_cross -= matrix.index_select(0, keep).index_select(
            1, merge_keepers
        ) * merge_alpha.unsqueeze(0)
        residual_rhs = residual_cross @ weight.index_select(1, merge_sources).t()
        delta += merge_residual_weight * torch.cholesky_solve(
            residual_rhs, keep_cholesky
        )
    reconstructed = base_weight + delta.t()

    if mode == "full_ls":
        bias = (
            weight.index_select(1, pruned) @ mean_device.index_select(0, pruned)
            - delta.t() @ mean_device.index_select(0, keep)
        )
    else:
        bias = _plan_bias(plan, weight.shape[0], device)
        bias -= delta.t() @ mean_device.index_select(0, keep)

    dtype = gate.weight.dtype
    gate_weight = gate.weight.detach().index_select(0, keep).contiguous()
    up_weight = up.weight.detach().index_select(0, keep).contiguous()
    layer.mlp.gate_proj = _linear(gate_weight, None)
    layer.mlp.up_proj = _linear(up_weight, None)
    layer.mlp.down_proj = _linear(reconstructed.to(dtype).contiguous(), bias.to(dtype))
    layer.mlp.intermediate_size = int(keep.numel())
    audit = {
        "mode": mode,
        "covariance_mode": covariance_mode,
        "merge_residual_weight": merge_residual_weight,
        "target": int(pruned.numel()),
        "reconstructed_sources": int(reconstruction_sources.numel()),
        "kept": int(keep.numel()),
        "delta_norm": float(delta.norm()),
        "bias_norm": float(bias.norm()),
    }
    del matrix, hkk, keep_cholesky, delta, weight, base_weight, reconstructed
    torch.cuda.empty_cache()
    return audit
