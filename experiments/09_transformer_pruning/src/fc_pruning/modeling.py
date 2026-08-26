from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    path: str
    display_name: str
    model_type: str


MODEL_SPECS = {
    "llama32_1b": ModelSpec(
        path="models/Llama-3.2-1B",
        display_name="Llama-3.2-1B",
        model_type="llama",
    ),
    "qwen25_1_5b": ModelSpec(
        path="models/Qwen2.5-1.5B",
        display_name="Qwen2.5-1.5B",
        model_type="qwen2",
    ),
}
MODEL_KEYS = tuple(MODEL_SPECS)


def resolve_model_path(project_root: str | Path, model_key: str) -> Path:
    try:
        spec = MODEL_SPECS[model_key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model {model_key!r}; choose one of {MODEL_KEYS}"
        ) from error
    model_path = (Path(project_root) / spec.path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != spec.model_type:
        raise ValueError(
            f"{model_key} expects model_type={spec.model_type!r}, "
            f"but {model_path} contains {config.model_type!r}"
        )
    return model_path


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    model = model.to(device)
    validate_prunable_model(model)
    return model


def decoder_layers(model):
    """Return decoder blocks shared by supported Llama and Qwen2 models."""
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise TypeError(
            f"{type(model).__name__} does not expose decoder layers as model.layers"
        )
    return layers


def validate_prunable_model(model) -> None:
    """Fail early when a checkpoint does not expose the expected gated FFN."""
    model_type = getattr(model.config, "model_type", None)
    if model_type not in {spec.model_type for spec in MODEL_SPECS.values()}:
        raise TypeError(
            f"Unsupported model_type={model_type!r}; expected Llama or Qwen2"
        )
    layers = decoder_layers(model)
    if not layers:
        raise ValueError("The model contains no decoder layers")
    required = ("gate_proj", "up_proj", "down_proj")
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        missing = [name for name in required if not hasattr(mlp, name)]
        if missing:
            raise TypeError(f"Layer {index} MLP is missing projections: {missing}")
        gate_width = int(mlp.gate_proj.out_features)
        up_width = int(mlp.up_proj.out_features)
        down_width = int(mlp.down_proj.in_features)
        if len({gate_width, up_width, down_width}) != 1:
            raise ValueError(
                f"Layer {index} has inconsistent FFN widths: "
                f"gate={gate_width}, up={up_width}, down={down_width}"
            )
