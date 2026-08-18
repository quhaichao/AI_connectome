from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import load_calibration
from .modeling import llama_layers


class LowRankLinear(nn.Module):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        rank, in_features = first.shape
        out_features, second_rank = second.shape
        if rank != second_rank:
            raise ValueError("Low-rank factors have incompatible shapes")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.v_proj = nn.Linear(in_features, rank, bias=False)
        self.u_proj = nn.Linear(rank, out_features, bias=False)
        self.v_proj.weight.data.copy_(first)
        self.u_proj.weight.data.copy_(second)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(inputs))


@torch.inference_mode()
def collect_svd_diagonal_statistics(
    model,
    calibration_path: str,
    output_path: str,
    device: torch.device,
) -> dict:
    """Collect all-token input second moments for FFN gate/up projections."""
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    hidden_size = model.config.hidden_size
    sums = [torch.zeros(hidden_size, dtype=torch.float64) for _ in layers]
    counts = [0 for _ in layers]

    def make_hook(layer_index: int):
        def hook(_module, args):
            hidden = args[0][0].float()
            sums[layer_index].add_(hidden.square().sum(dim=0).double().cpu())
            counts[layer_index] += hidden.shape[0]

        return hook

    handles = [
        layer.mlp.gate_proj.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        for index, sequence in enumerate(input_ids):
            print(
                f"SVD diagonal statistics context={index + 1:03d}/{input_ids.shape[0]:03d}",
                flush=True,
            )
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    expected = int(input_ids.numel())
    if any(count != expected for count in counts):
        raise RuntimeError(f"SVD statistics counts {counts} do not equal {expected}")
    payload = {
        "layers": [
            {"count": count, "input_sum_sq": value}
            for count, value in zip(counts, sums)
        ],
        "protocol": {
            "calibration_path": str(Path(calibration_path).resolve()),
            "contexts": int(input_ids.shape[0]),
            "sequence_length": int(input_ids.shape[1]),
            "activation_tokens": expected,
            "all_token_positions": True,
            "statistic": "diagonal_input_second_moment",
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["protocol"]


@torch.inference_mode()
def collect_svd_full_covariances(
    model,
    calibration_path: str,
    output_path: str,
    device: torch.device,
) -> dict:
    """Collect SVD-LLM-equivalent full covariances over every token.

    Gate and up share the post-RMSNorm FFN input covariance. For down_proj we
    accumulate its 4096-dimensional output covariance. If W is down_proj and
    H is its 11008-dimensional input covariance, this output covariance is
    W H W^T. Its leading eigenvectors are exactly the left singular vectors
    of W chol(H), avoiding an otherwise costly 11008-square Cholesky.
    """
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    hidden_size = model.config.hidden_size
    hidden_covariances = [
        torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=device)
        for _ in layers
    ]
    down_output_covariances = [
        torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=device)
        for _ in layers
    ]
    counts = [0 for _ in layers]

    def make_input_hook(layer_index: int):
        def hook(_module, args):
            hidden = args[0][0].float()
            hidden_covariances[layer_index].addmm_(hidden.t(), hidden)
            counts[layer_index] += hidden.shape[0]

        return hook

    def make_output_hook(layer_index: int):
        def hook(_module, _args, output):
            projected = output[0].float()
            down_output_covariances[layer_index].addmm_(
                projected.t(), projected
            )

        return hook

    handles = []
    for index, layer in enumerate(layers):
        handles.append(
            layer.mlp.gate_proj.register_forward_pre_hook(make_input_hook(index))
        )
        handles.append(
            layer.mlp.down_proj.register_forward_hook(make_output_hook(index))
        )
    try:
        for index, sequence in enumerate(input_ids):
            print(
                f"SVD full covariance context={index + 1:03d}/{input_ids.shape[0]:03d}",
                flush=True,
            )
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    expected = int(input_ids.numel())
    if any(count != expected for count in counts):
        raise RuntimeError(f"SVD covariance counts {counts} do not equal {expected}")
    payload = {
        "layers": [
            {
                "count": count,
                "hidden_covariance": hidden.cpu(),
                "down_output_covariance": output.cpu(),
            }
            for count, hidden, output in zip(
                counts, hidden_covariances, down_output_covariances
            )
        ],
        "protocol": {
            "calibration_path": str(Path(calibration_path).resolve()),
            "contexts": int(input_ids.shape[0]),
            "sequence_length": int(input_ids.shape[1]),
            "activation_tokens": expected,
            "all_token_positions": True,
            "hidden_covariance_dimension": hidden_size,
            "down_statistic": "output_covariance_equals_W_H_WT",
            "accumulation_dtype": "float32",
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["protocol"]


def _activation_scale(sum_sq: torch.Tensor, count: int, damping: float) -> torch.Tensor:
    second = sum_sq.float() / count
    floor = damping * second.mean().clamp_min(1e-30)
    return (second + floor).sqrt()


@torch.inference_mode()
def factorize_linear_svd_llm_diagonal(
    linear: nn.Linear,
    rank: int,
    input_scale: torch.Tensor,
    device: torch.device,
) -> LowRankLinear:
    """SVD-LLM's whitening/truncation formula with diagonal whitening.

    The official algorithm uses a full Cholesky factor of X^T X. This
    controlled FFN-only variant retains the activation-aware coordinate
    scaling but uses its diagonal so the 11,008-dimensional down projection
    remains tractable with all 262,144 calibration tokens.
    """
    dtype = linear.weight.dtype
    weight = linear.weight.detach().float().to(device)
    scale = input_scale.float().to(device).clamp_min(1e-12)
    weighted = weight * scale.unsqueeze(0)
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    singular = singular[:rank].sqrt()
    second = u[:, :rank] * singular.unsqueeze(0)
    first = singular.unsqueeze(1) * vh[:rank]
    first = first / scale.unsqueeze(0)
    result = LowRankLinear(first.to(dtype).cpu(), second.to(dtype).cpu()).to(dtype=dtype)
    del weight, scale, weighted, u, singular, vh, first, second
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def factorize_input_whitened_linear(
    linear: nn.Linear,
    rank: int,
    input_covariance: torch.Tensor,
    device: torch.device,
) -> LowRankLinear:
    """Apply the official SVD-LLM Cholesky whitening and truncate."""
    dtype = linear.weight.dtype
    covariance = input_covariance.double().to(device)
    try:
        cholesky = torch.linalg.cholesky(covariance)
    except torch.linalg.LinAlgError:
        minimum = torch.linalg.eigvalsh(covariance)[0]
        jitter = (-minimum + 1e-6).clamp_min(1e-6)
        covariance.diagonal().add_(jitter)
        cholesky = torch.linalg.cholesky(covariance)
    weight = linear.weight.detach().float().to(device)
    weighted = weight @ cholesky.float()
    left, _singular, _right = torch.linalg.svd(weighted, full_matrices=False)
    left = left[:, :rank]
    # U_r U_r^T W is algebraically identical to
    # U_r S_r V_r^T chol(H)^-1, but avoids an unstable explicit inverse.
    first = left.t() @ weight
    result = LowRankLinear(first.to(dtype).cpu(), left.to(dtype).cpu()).to(
        dtype=dtype
    )
    del covariance, cholesky, weight, weighted, left, first, _singular, _right
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def factorize_from_output_covariance(
    linear: nn.Linear,
    rank: int,
    output_covariance: torch.Tensor,
    device: torch.device,
) -> LowRankLinear:
    """Exact left singular subspace of W chol(H) from W H W^T."""
    dtype = linear.weight.dtype
    covariance = output_covariance.float().to(device)
    _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    left = eigenvectors[:, -rank:].flip(1)
    weight = linear.weight.detach().float().to(device)
    first = left.t() @ weight
    result = LowRankLinear(first.to(dtype).cpu(), left.to(dtype).cpu()).to(
        dtype=dtype
    )
    del covariance, _eigenvalues, eigenvectors, left, weight, first
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def apply_svd_llm_full_ffn(
    model,
    rank: int,
    covariance_statistics: dict,
    device: torch.device,
) -> list[dict]:
    layers = llama_layers(model)
    if len(covariance_statistics["layers"]) != len(layers):
        raise ValueError("Covariance statistics and model layer counts differ")
    plans = []
    for index, layer in enumerate(layers):
        statistics = covariance_statistics["layers"][index]
        print(f"SVD-LLM full FFN layer={index:02d} gate", flush=True)
        layer.mlp.gate_proj = factorize_input_whitened_linear(
            layer.mlp.gate_proj,
            rank,
            statistics["hidden_covariance"],
            device,
        )
        print(f"SVD-LLM full FFN layer={index:02d} up", flush=True)
        layer.mlp.up_proj = factorize_input_whitened_linear(
            layer.mlp.up_proj,
            rank,
            statistics["hidden_covariance"],
            device,
        )
        print(f"SVD-LLM full FFN layer={index:02d} down", flush=True)
        layer.mlp.down_proj = factorize_from_output_covariance(
            layer.mlp.down_proj,
            rank,
            statistics["down_output_covariance"],
            device,
        )
        layer.mlp.to(device)
        plans.append(
            {
                "layer": index,
                "rank": rank,
                "gate_up_whitening": "full_input_covariance_cholesky",
                "down_whitening": "equivalent_output_covariance_eigenspace",
            }
        )
    return plans


@torch.inference_mode()
def apply_svd_llm_diagonal_ffn(
    model,
    rank: int,
    input_statistics: dict,
    activation_statistics: dict,
    device: torch.device,
    damping: float = 1e-6,
) -> list[dict]:
    layers = llama_layers(model)
    if len(input_statistics["layers"]) != len(layers):
        raise ValueError("Input statistics and model layer counts differ")
    if len(activation_statistics["layers"]) != len(layers):
        raise ValueError("Activation statistics and model layer counts differ")

    plans = []
    for index, layer in enumerate(layers):
        print(f"SVD-LLM diagonal FFN layer={index:02d} gate", flush=True)
        input_stats = input_statistics["layers"][index]
        activation_stats = activation_statistics["layers"][index]
        hidden_scale = _activation_scale(
            input_stats["input_sum_sq"], int(input_stats["count"]), damping
        )
        z_scale = _activation_scale(
            activation_stats["sum_sq"], int(activation_stats["count"]), damping
        )
        layer.mlp.gate_proj = factorize_linear_svd_llm_diagonal(
            layer.mlp.gate_proj, rank, hidden_scale, device
        )
        print(f"SVD-LLM diagonal FFN layer={index:02d} up", flush=True)
        layer.mlp.up_proj = factorize_linear_svd_llm_diagonal(
            layer.mlp.up_proj, rank, hidden_scale, device
        )
        print(f"SVD-LLM diagonal FFN layer={index:02d} down", flush=True)
        layer.mlp.down_proj = factorize_linear_svd_llm_diagonal(
            layer.mlp.down_proj, rank, z_scale, device
        )
        layer.mlp.to(device)
        plans.append(
            {
                "layer": index,
                "rank": rank,
                "gate_shape": [model.config.intermediate_size, model.config.hidden_size],
                "up_shape": [model.config.intermediate_size, model.config.hidden_size],
                "down_shape": [model.config.hidden_size, model.config.intermediate_size],
                "whitening": "diagonal_all_token_second_moment",
            }
        )
    return plans
