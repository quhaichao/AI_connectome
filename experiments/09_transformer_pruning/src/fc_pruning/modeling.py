from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_path: str, device: torch.device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    return model.to(device)


def llama_layers(model):
    return model.model.layers
