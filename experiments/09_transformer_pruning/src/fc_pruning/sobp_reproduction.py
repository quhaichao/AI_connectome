from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def apply_sobp_layer(
    layer,
    hessian: torch.Tensor,
    target: int,
    device: torch.device,
    damping: float = 1e-4,
) -> dict:
    """Select and reconstruct one structured FFN mask with the SoBP rule."""
    gate = layer.mlp.gate_proj
    up = layer.mlp.up_proj
    down = layer.mlp.down_proj
    width = gate.out_features
    if not 0 < target < width:
        raise ValueError(f"SoBP target must be within (0, {width}), got {target}")

    weight = down.weight.detach().float().to(device)
    matrix = hessian.float().to(device)
    diagonal_scale = matrix.diagonal().mean().clamp_min(1e-12)
    matrix.diagonal().add_(damping * diagonal_scale)
    chol = torch.linalg.cholesky(matrix)
    inverse = torch.cholesky_inverse(chol)
    scores = weight.square().sum(dim=0) / inverse.diagonal().clamp_min(1e-30)
    pruned = torch.topk(scores, target, largest=False).indices.sort().values
    keep_mask = torch.ones(width, dtype=torch.bool, device=device)
    keep_mask[pruned] = False
    keep = keep_mask.nonzero(as_tuple=False).flatten()

    # Module-wise SoBP reconstruction: solve H_KK delta = H_KS W_S^T.
    hkk = matrix.index_select(0, keep).index_select(1, keep)
    hks = matrix.index_select(0, keep).index_select(1, pruned)
    rhs = hks @ weight.index_select(1, pruned).t()
    keep_chol = torch.linalg.cholesky(hkk)
    delta = torch.cholesky_solve(rhs, keep_chol)
    reconstructed_down = weight.index_select(1, keep) + delta.t()

    dtype = gate.weight.dtype
    module_device = gate.weight.device
    gate_new = nn.Linear(
        gate.in_features,
        keep.numel(),
        bias=False,
        device=module_device,
        dtype=dtype,
    )
    up_new = nn.Linear(
        up.in_features,
        keep.numel(),
        bias=False,
        device=module_device,
        dtype=dtype,
    )
    down_new = nn.Linear(
        keep.numel(),
        down.out_features,
        bias=False,
        device=module_device,
        dtype=dtype,
    )
    gate_new.weight.data.copy_(gate.weight.detach().index_select(0, keep))
    up_new.weight.data.copy_(up.weight.detach().index_select(0, keep))
    down_new.weight.data.copy_(reconstructed_down.to(dtype))
    layer.mlp.gate_proj = gate_new
    layer.mlp.up_proj = up_new
    layer.mlp.down_proj = down_new
    layer.mlp.intermediate_size = int(keep.numel())

    audit = {
        "target": target,
        "pruned": pruned.cpu().tolist(),
        "kept": int(keep.numel()),
        "mean_score": float(scores.mean()),
    }
    del matrix, chol, inverse, hkk, hks, rhs, keep_chol, delta, weight
    torch.cuda.empty_cache()
    return audit
