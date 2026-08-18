from __future__ import annotations

from pathlib import Path

import torch

from .data import load_calibration
from .modeling import llama_layers


@torch.inference_mode()
def collect_full_ffn_statistics(
    model,
    calibration_path: str,
    output_path: str,
    device: torch.device,
    ocp_token_fraction: float = 0.5,
) -> dict:
    """Collect FFN statistics over every token in every calibration context."""
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    width = model.config.intermediate_size
    if input_ids.ndim != 2:
        raise ValueError("Calibration input_ids must have shape [contexts, length]")
    if not 0.0 < ocp_token_fraction <= 1.0:
        raise ValueError("ocp_token_fraction must be in (0, 1]")

    stats = [
        {
            "count": 0,
            "sum": torch.zeros(width, dtype=torch.float64),
            "sum_sq": torch.zeros(width, dtype=torch.float64),
            "abs_sum": torch.zeros(width, dtype=torch.float64),
            "probe_count": 0,
            "probe_sum": torch.zeros(width, dtype=torch.float64),
            "probe_sum_sq": torch.zeros(width, dtype=torch.float64),
            "probe_abs_sum": torch.zeros(width, dtype=torch.float64),
        }
        for _ in layers
    ]
    probe_positions: dict[int, torch.Tensor] = {}
    sensitivities = [
        (
            layer.mlp.gate_proj.weight.detach().float().abs().sum(dim=0)
            + layer.mlp.up_proj.weight.detach().float().abs().sum(dim=0)
        )
        for layer in layers
    ]

    def make_gate_hook(layer_index: int):
        def hook(_module, args):
            hidden = args[0][0].float()
            sensitivity = sensitivities[layer_index].to(hidden.device)
            token_scores = (hidden * sensitivity).square().sum(dim=1)
            keep = max(1, int(round(hidden.shape[0] * ocp_token_fraction)))
            probe_positions[layer_index] = torch.topk(
                token_scores, keep, largest=True, sorted=False
            ).indices

        return hook

    def make_down_hook(layer_index: int):
        def hook(_module, args):
            activation = args[0][0].float()
            layer_stats = stats[layer_index]
            layer_stats["count"] += activation.shape[0]
            layer_stats["sum"] += activation.sum(dim=0).double().cpu()
            layer_stats["sum_sq"] += activation.square().sum(dim=0).double().cpu()
            layer_stats["abs_sum"] += activation.abs().sum(dim=0).double().cpu()

            selected = activation.index_select(0, probe_positions[layer_index])
            layer_stats["probe_count"] += selected.shape[0]
            layer_stats["probe_sum"] += selected.sum(dim=0).double().cpu()
            layer_stats["probe_sum_sq"] += selected.square().sum(dim=0).double().cpu()
            layer_stats["probe_abs_sum"] += selected.abs().sum(dim=0).double().cpu()

        return hook

    handles = []
    for layer_index, layer in enumerate(layers):
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(make_gate_hook(layer_index)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_down_hook(layer_index)))
    try:
        for sequence_index, sequence in enumerate(input_ids):
            print(
                f"Full-token statistics context={sequence_index + 1:03d}/{input_ids.shape[0]:03d}",
                flush=True,
            )
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    expected = int(input_ids.shape[0] * input_ids.shape[1])
    for layer_index, layer_stats in enumerate(stats):
        if layer_stats["count"] != expected:
            raise RuntimeError(
                f"Layer {layer_index} collected {layer_stats['count']} tokens; expected {expected}"
            )

    payload = {
        "layers": stats,
        "protocol": {
            "calibration_path": str(Path(calibration_path).resolve()),
            "contexts": int(input_ids.shape[0]),
            "sequence_length": int(input_ids.shape[1]),
            "activation_tokens": expected,
            "all_token_positions": True,
            "ocp_token_fraction": float(ocp_token_fraction),
            "intermediate_size": int(width),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload["protocol"]


def load_full_ffn_statistics(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)
