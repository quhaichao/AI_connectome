from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .adaptive_allocation import global_layer_targets
from .data import load_calibration
from .modeling import llama_layers


@torch.inference_mode()
def collect_sobp_hessians(
    model,
    calibration_path: str,
    output_path: str,
    device: torch.device,
) -> dict:
    """Collect FFN down-input Hessians H=X^T X over all calibration tokens."""
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    width = model.config.intermediate_size
    hessians = [
        torch.zeros((width, width), dtype=torch.float32, device=device)
        for _ in layers
    ]
    counts = [0 for _ in layers]

    def make_hook(layer_index: int):
        def hook(_module, args):
            values = args[0][0].float()
            hessians[layer_index].addmm_(values.t(), values)
            counts[layer_index] += values.shape[0]

        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        for index, sequence in enumerate(input_ids):
            print(
                f"SoBP Hessian context={index + 1:03d}/{input_ids.shape[0]:03d}",
                flush=True,
            )
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = int(input_ids.numel())
    if any(count != expected for count in counts):
        raise RuntimeError(f"SoBP Hessian counts {counts} do not equal {expected}")
    payload = {
        "layers": [hessian.cpu() for hessian in hessians],
        "protocol": {
            "calibration_path": str(Path(calibration_path).resolve()),
            "contexts": int(input_ids.shape[0]),
            "sequence_length": int(input_ids.shape[1]),
            "activation_tokens": expected,
            "all_token_positions": True,
            "statistic": "down_input_hessian_XT_X",
            "accumulation_dtype": "float32",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload["protocol"]


def collect_sobp_mask_gradients(
    model,
    calibration_path: str,
    output_path: str,
    device: torch.device,
) -> dict:
    """Collect SoBP Eq. 7 mask gradients for all FFN neurons."""
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    width = model.config.intermediate_size
    gradients = [
        torch.zeros(width, dtype=torch.float64, device=device) for _ in layers
    ]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def make_hook(layer_index: int):
        def hook(_module, args):
            activation = args[0]
            if not activation.requires_grad:
                activation.requires_grad_(True)
            values = activation[0].detach().float()

            def gradient_hook(gradient):
                contribution = (values * gradient[0].float()).sum(dim=0)
                gradients[layer_index].add_(contribution.double())

            activation.register_hook(gradient_hook)

        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        for index, sequence in enumerate(input_ids):
            print(
                f"SoBP mask gradient context={index + 1:03d}/{input_ids.shape[0]:03d}",
                flush=True,
            )
            tokens = sequence.unsqueeze(0).to(device)
            logits = model(input_ids=tokens, use_cache=False).logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1),
            )
            loss.backward()
            del tokens, logits, loss
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    scores = [gradient.square().float().cpu() for gradient in gradients]
    payload = {
        "scores": scores,
        "protocol": {
            "calibration_path": str(Path(calibration_path).resolve()),
            "contexts": int(input_ids.shape[0]),
            "sequence_length": int(input_ids.shape[1]),
            "activation_tokens": int(input_ids.numel()),
            "statistic": "squared_ffn_mask_gradient",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload["protocol"]


def sobp_global_targets(
    gradient_payload: dict,
    ratio: float,
    maximum_fraction: float = 0.8,
) -> list[int]:
    scores = gradient_payload["scores"]
    total = int(round(sum(score.numel() for score in scores) * ratio))
    return global_layer_targets(scores, total, maximum_fraction)


@torch.no_grad()
def apply_sobp_layer(
    layer,
    hessian: torch.Tensor,
    target: int,
    device: torch.device,
    damping: float = 1e-4,
) -> dict:
    gate = layer.mlp.gate_proj
    up = layer.mlp.up_proj
    down = layer.mlp.down_proj
    width = gate.out_features
    weight = down.weight.detach().float().to(device)
    matrix = hessian.float().to(device) / hessian.shape[0]
    diagonal_scale = matrix.diagonal().mean().clamp_min(1e-12)
    matrix.diagonal().add_(damping * diagonal_scale)
    chol = torch.linalg.cholesky(matrix)
    inverse = torch.cholesky_inverse(chol)
    scores = weight.square().sum(dim=0) / inverse.diagonal().clamp_min(1e-30)
    pruned = torch.topk(scores, target, largest=False).indices.sort().values
    keep_mask = torch.ones(width, dtype=torch.bool, device=device)
    keep_mask[pruned] = False
    keep = keep_mask.nonzero(as_tuple=False).flatten()

    # Module-wise SoBP reconstruction: compensate the surviving down columns
    # by solving H_KK delta = H_KS W_S^T.
    hkk = matrix.index_select(0, keep).index_select(1, keep)
    hks = matrix.index_select(0, keep).index_select(1, pruned)
    rhs = hks @ weight.index_select(1, pruned).t()
    keep_chol = torch.linalg.cholesky(hkk)
    delta = torch.cholesky_solve(rhs, keep_chol)
    reconstructed_down = weight.index_select(1, keep) + delta.t()

    dtype = gate.weight.dtype
    gate_weight = gate.weight.detach().index_select(0, keep)
    up_weight = up.weight.detach().index_select(0, keep)
    down_weight = reconstructed_down.to(dtype)
    device_for_new = gate.weight.device
    gate_new = nn.Linear(gate.in_features, keep.numel(), bias=False, device=device_for_new, dtype=dtype)
    up_new = nn.Linear(up.in_features, keep.numel(), bias=False, device=device_for_new, dtype=dtype)
    down_new = nn.Linear(keep.numel(), down.out_features, bias=False, device=device_for_new, dtype=dtype)
    gate_new.weight.data.copy_(gate_weight)
    up_new.weight.data.copy_(up_weight)
    down_new.weight.data.copy_(down_weight)
    layer.mlp.gate_proj = gate_new
    layer.mlp.up_proj = up_new
    layer.mlp.down_proj = down_new
    del matrix, chol, inverse, hkk, hks, rhs, keep_chol, delta, weight
    torch.cuda.empty_cache()
    return {
        "target": target,
        "pruned": pruned.cpu().tolist(),
        "kept": int(keep.numel()),
        "mean_score": float(scores.mean()),
    }
