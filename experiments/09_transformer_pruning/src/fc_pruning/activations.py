from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import load_calibration
from .modeling import llama_layers


@torch.inference_mode()
def collect_down_projection_inputs(model, calibration_path: str, output_dir: str, device):
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    positions = calibration["sampled_positions"]
    layers = llama_layers(model)
    samples_per_sequence = positions.shape[1]
    total_samples = input_ids.shape[0] * samples_per_sequence
    d_ff = model.config.intermediate_size
    buffers = [torch.empty((total_samples, d_ff), dtype=torch.float16) for _ in layers]
    state = {"sequence": 0}

    def make_hook(layer_index: int):
        def hook(_module, args):
            activation = args[0][0]
            selected = activation.index_select(0, positions[state["sequence"]].to(activation.device))
            begin = state["sequence"] * samples_per_sequence
            end = begin + samples_per_sequence
            buffers[layer_index][begin:end].copy_(selected.detach().to("cpu", torch.float16))

        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_index))
        for layer_index, layer in enumerate(layers)
    ]
    try:
        for sequence_index, sequence in enumerate(input_ids):
            state["sequence"] = sequence_index
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for layer_index, buffer in enumerate(buffers):
        torch.save(buffer, output / f"layer_{layer_index:02d}.pt")
    metadata = {
        "layers": len(layers),
        "samples": total_samples,
        "intermediate_size": d_ff,
        "dtype": "float16",
        "source_calibration": str(Path(calibration_path).resolve()),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def load_layer_activations(output_dir: str, layer_index: int) -> torch.Tensor:
    path = Path(output_dir) / f"layer_{layer_index:02d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)
