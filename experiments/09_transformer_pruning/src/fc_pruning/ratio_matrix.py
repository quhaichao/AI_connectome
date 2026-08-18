from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import load_calibration
from .fang_reproduction import (
    kmeans_assignments,
    pca_bases_from_covariances,
)
from .modeling import llama_layers


METHODS = ("fc_ls", "flap", "sobp", "fand", "slimllm", "wanda")
RATIOS = (0.2, 0.3, 0.4, 0.5)


def _protocol(calibration_path: str, input_ids: torch.Tensor) -> dict:
    return {
        "calibration_path": str(Path(calibration_path).resolve()),
        "contexts": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "activation_tokens": int(input_ids.numel()),
        "fit_contexts": int(round(input_ids.shape[0] * 0.75)),
        "holdout_contexts": int(input_ids.shape[0] - round(input_ids.shape[0] * 0.75)),
        "all_token_positions": True,
    }


@torch.inference_mode()
def collect_ratio_statistics(model, calibration_path: str, device: torch.device) -> dict:
    """Collect all dense sufficient statistics needed by the six methods.

    Hessians are split into fit/holdout contexts for FC selection. All other
    matrices use all contexts. Only O(width^2) matrices are retained; token
    activations are never written to disk.
    """
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    if tuple(input_ids.shape) != (128, 2048):
        raise ValueError(f"Expected (128, 2048) calibration, got {tuple(input_ids.shape)}")
    layers = llama_layers(model)
    width = int(model.config.intermediate_size)
    hidden = int(model.config.hidden_size)
    fit_contexts = 96
    fit_grams = [torch.zeros((width, width), dtype=torch.float32, device=device) for _ in layers]
    hold_grams = [torch.zeros_like(item) for item in fit_grams]
    hidden_cov = [torch.zeros((hidden, hidden), dtype=torch.float32, device=device) for _ in layers]
    output_cov = [torch.zeros_like(item) for item in hidden_cov]
    sums = [torch.zeros(width, dtype=torch.float64) for _ in layers]
    sums_sq = [torch.zeros(width, dtype=torch.float64) for _ in layers]
    fit_sums = [torch.zeros(width, dtype=torch.float64) for _ in layers]
    holdout_sums = [torch.zeros(width, dtype=torch.float64) for _ in layers]
    counts = [0 for _ in layers]
    state = {"context": 0}

    def make_input_hook(index: int):
        def hook(_module, args):
            values = args[0][0].float()
            hidden_cov[index].addmm_(values.t(), values)
        return hook

    def make_down_hook(index: int):
        def hook(_module, args):
            values = args[0][0].float()
            if state["context"] < fit_contexts:
                fit_grams[index].addmm_(values.t(), values)
                fit_sums[index] += values.sum(dim=0).double().cpu()
            else:
                hold_grams[index].addmm_(values.t(), values)
                holdout_sums[index] += values.sum(dim=0).double().cpu()
            sums[index] += values.sum(dim=0).double().cpu()
            sums_sq[index] += values.square().sum(dim=0).double().cpu()
            counts[index] += values.shape[0]
        return hook

    def make_output_hook(index: int):
        def hook(_module, _args, output):
            values = output[0].float()
            output_cov[index].addmm_(values.t(), values)
        return hook

    handles = []
    for index, layer in enumerate(layers):
        handles.extend(
            [
                layer.mlp.gate_proj.register_forward_pre_hook(make_input_hook(index)),
                layer.mlp.down_proj.register_forward_pre_hook(make_down_hook(index)),
                layer.mlp.down_proj.register_forward_hook(make_output_hook(index)),
            ]
        )
    try:
        for context, sequence in enumerate(input_ids):
            state["context"] = context
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
            if (context + 1) % 8 == 0:
                print(f"Ratio statistics context={context + 1}/128", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    expected = int(input_ids.numel())
    if any(count != expected for count in counts):
        raise RuntimeError(f"Statistics token counts are {counts}; expected {expected}")
    layers_payload = []
    for index in range(len(layers)):
        layers_payload.append(
            {
                "count": counts[index],
                "sum": sums[index],
                "sum_sq": sums_sq[index],
                "fit_sum": fit_sums[index],
                "holdout_sum": holdout_sums[index],
                "fit_gram": fit_grams[index].cpu(),
                "holdout_gram": hold_grams[index].cpu(),
                "hidden_covariance": hidden_cov[index].cpu(),
                "down_output_covariance": output_cov[index].cpu(),
            }
        )
        fit_grams[index] = None
        hold_grams[index] = None
        hidden_cov[index] = None
        output_cov[index] = None
        torch.cuda.empty_cache()
    return {"layers": layers_payload, "protocol": _protocol(calibration_path, input_ids)}


def save_ratio_statistics(payload: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_ratio_statistics(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def collect_fand_statistics_fast(
    model,
    calibration_path: str,
    ratio_statistics: dict,
    device: torch.device,
    clusters: int = 7,
    pca_components: int = 64,
    kmeans_iterations: int = 20,
    seed: int = 0,
) -> dict:
    """Collect FAND/FANG cluster Taylor statistics in one backward per context."""
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    bases = pca_bases_from_covariances(
        {"layers": [{"hidden_covariance": x["hidden_covariance"]} for x in ratio_statistics["layers"]]},
        pca_components,
        device,
    )
    projected = torch.empty(
        (len(layers), int(input_ids.numel()), pca_components), dtype=torch.float16
    )
    context_state = {"index": 0}
    device_bases = [basis.to(device) for basis in bases]

    def make_project_hook(index: int):
        def hook(_module, args):
            values = args[0][0].float()
            begin = context_state["index"] * input_ids.shape[1]
            projected[index, begin : begin + values.shape[0]].copy_(
                (values @ device_bases[index]).half().cpu()
            )
        return hook

    handles = [
        layer.mlp.gate_proj.register_forward_pre_hook(make_project_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        with torch.inference_mode():
            for context, sequence in enumerate(input_ids):
                context_state["index"] = context
                model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assignments, centers = kmeans_assignments(
        projected, clusters, kmeans_iterations, seed, device
    )
    del projected, bases, device_bases, centers
    torch.cuda.empty_cache()

    width = int(model.config.intermediate_size)
    sums = torch.zeros((len(layers), clusters, width), device=device)
    sums_sq = torch.zeros_like(sums)
    taylor = torch.zeros_like(sums)
    counts = torch.zeros((len(layers), clusters), dtype=torch.long, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def make_hook(index: int):
        def hook(_module, args):
            activation = args[0]
            activation.requires_grad_(True)
            values = activation[0].detach().float()
            begin = context_state["index"] * input_ids.shape[1]
            labels = assignments[index, begin : begin + values.shape[0]].long().to(device)
            sums[index].index_add_(0, labels, values)
            sums_sq[index].index_add_(0, labels, values.square())
            counts[index].index_add_(0, labels, torch.ones_like(labels))

            def gradient_hook(gradient):
                taylor[index].index_add_(0, labels, values * gradient[0].float())

            activation.register_hook(gradient_hook)
        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        for context, sequence in enumerate(input_ids):
            context_state["index"] = context
            tokens = sequence.unsqueeze(0).to(device)
            logits = model(input_ids=tokens, use_cache=False).logits.float()
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1),
            )
            loss.backward()
            model.zero_grad(set_to_none=True)
            del tokens, logits, loss
            torch.cuda.empty_cache()
            if (context + 1) % 8 == 0:
                print(f"FAND Taylor context={context + 1}/128", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    divisor = counts.clamp_min(1).unsqueeze(-1)
    result = {
        "counts": counts.cpu(),
        "mean": (sums / divisor).cpu(),
        "variance": (sums_sq / divisor - (sums / divisor).square()).clamp_min(0.0).cpu(),
        "taylor": (taylor / divisor).abs().cpu(),
        "protocol": {
            **ratio_statistics["protocol"],
            "clusters": clusters,
            "pca_components": pca_components,
            "kmeans_iterations": kmeans_iterations,
            "statistic": "all-layer one-backward-per-context cluster Taylor",
        },
    }
    del sums, sums_sq, taylor, counts, assignments
    torch.cuda.empty_cache()
    return result


def as_full_statistics(payload: dict) -> tuple[dict, dict, dict]:
    full = {
        "layers": [
            {
                "count": layer["count"],
                "sum": layer["sum"],
                "sum_sq": layer["sum_sq"],
            }
            for layer in payload["layers"]
        ],
        "protocol": payload["protocol"],
    }
    covariance = {
        "layers": [
            {
                "count": layer["count"],
                "hidden_covariance": layer["hidden_covariance"],
                "down_output_covariance": layer["down_output_covariance"],
            }
            for layer in payload["layers"]
        ],
        "protocol": payload["protocol"],
    }
    hessians = {
        "layers": [layer["fit_gram"] + layer["holdout_gram"] for layer in payload["layers"]],
        "protocol": payload["protocol"],
    }
    return full, covariance, hessians
