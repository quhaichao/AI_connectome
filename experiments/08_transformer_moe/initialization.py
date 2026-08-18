"""Shared v2 initialization policy, independent of model constructor signatures."""

from __future__ import annotations

import torch.nn as nn


INITIALIZATION_API_VERSION = "2.4.0"


def apply_initialization_policy(
    model: nn.Module,
    tie_embeddings: bool,
    embedding_init_std: float,
    xavier_initialize: bool,
) -> nn.Module:
    """Apply one initialization/head policy to dense and every MoE baseline."""
    if not hasattr(model, "token_embedding") or not hasattr(model, "lm_head"):
        raise TypeError("Expected a language model with token_embedding and lm_head")

    if tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight
    else:
        # Old model constructors tie these tensors. Detach the head before init.
        model.lm_head.weight = nn.Parameter(
            model.token_embedding.weight.detach().clone(), requires_grad=True
        )

    if xavier_initialize:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=embedding_init_std)
            elif isinstance(module, nn.Linear):
                if tie_embeddings and module is model.lm_head:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    if tie_embeddings:
        # Preserve identity even if future initialization code replaces a Parameter.
        model.lm_head.weight = model.token_embedding.weight
    return model
