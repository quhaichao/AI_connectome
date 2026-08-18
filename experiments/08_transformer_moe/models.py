"""Small decoder Transformer and unified sparse-MoE layers for the benchmark.

All methods share this implementation so differences come from construction and
routing, not from unrelated framework kernels.  The layer supports both
partitioned experts (total-parameter matched) and full-width experts (active-
compute matched upcycling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor
    routing: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    pre_ffn: List[torch.Tensor] = field(default_factory=list)
    hidden_activations: List[torch.Tensor] = field(default_factory=list)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return self.dropout(y)


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hidden = F.gelu(self.fc1(x))
        y = self.dropout(self.fc2(hidden))
        return (y, hidden) if return_hidden else y


class DenseBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = DenseFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        x = x + self.attn(self.norm1(x))
        ffn_input = self.norm2(x)
        if return_intermediates:
            y, hidden = self.ffn(ffn_input, return_hidden=True)
            return x + y, ffn_input, hidden
        return x + self.ffn(ffn_input)


class DenseTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DenseBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, return_intermediates: bool = False) -> ModelOutput:
        batch, length = input_ids.shape
        if length > self.seq_len:
            raise ValueError(f"Sequence length {length} exceeds configured {self.seq_len}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions)[None])
        pre_ffn: List[torch.Tensor] = []
        hidden_activations: List[torch.Tensor] = []
        for block in self.blocks:
            if return_intermediates:
                x, ffn_input, hidden = block(x, return_intermediates=True)
                pre_ffn.append(ffn_input)
                hidden_activations.append(hidden)
            else:
                x = block(x)
        logits = self.lm_head(self.final_norm(x))
        return ModelOutput(
            logits=logits,
            aux_loss=logits.new_zeros(()),
            pre_ffn=pre_ffn,
            hidden_activations=hidden_activations,
        )


class ExpertFFN(nn.Module):
    """An expert with a shared output bias held by the parent MoE layer."""

    def __init__(self, d_model: int, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hidden = F.gelu(self.fc1(x))
        y = self.dropout(self.fc2(hidden))
        return (y, hidden) if return_hidden else y


class TwoLayerRouter(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, num_experts: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)

    def forward(self, x: torch.Tensor, nonnegative: bool = False) -> torch.Tensor:
        value = self.fc2(F.gelu(self.fc1(x)))
        return value.abs() if nonnegative else value


class SparseMoEFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        router_kind: str = "switch",
        router_hidden_size: int = 128,
        combine_mode: str = "sum",
        capacity_factor: float = 1.25,
        drop_overflow_tokens: bool = True,
        dynamic_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_kind = router_kind
        self.combine_mode = combine_mode
        self.capacity_factor = capacity_factor
        self.drop_overflow_tokens = drop_overflow_tokens
        self.dynamic_threshold = dynamic_threshold
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, hidden_size, dropout) for _ in range(num_experts)]
        )
        self.output_bias = nn.Parameter(torch.zeros(d_model))
        if router_kind in {"moefication_mlp", "d2d_regression"}:
            self.router = TwoLayerRouter(d_model, router_hidden_size, num_experts)
        elif router_kind in {"switch", "centroid"}:
            self.router = nn.Linear(d_model, num_experts, bias=False)
        elif router_kind == "avg_key":
            self.router = None
        else:
            raise ValueError(f"Unsupported router kind: {router_kind}")

    def router_scores(self, flat_x: torch.Tensor) -> torch.Tensor:
        if self.router_kind == "avg_key":
            # EMoE Eq. 4: the gate stays tied to the mean key of each expert.
            keys = torch.stack([expert.fc1.weight.mean(dim=0) for expert in self.experts])
            return flat_x @ keys.t()
        if self.router_kind == "d2d_regression":
            return self.router(flat_x, nonnegative=True)
        if self.router_kind == "moefication_mlp":
            return self.router(flat_x, nonnegative=False)
        return self.router(flat_x)

    def _fixed_k_mask(self, probabilities: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(probabilities, self.top_k, dim=-1)
        mask = torch.zeros_like(probabilities, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        if not self.drop_overflow_tokens:
            return mask
        tokens = probabilities.shape[0]
        capacity = max(1, math.ceil(self.capacity_factor * tokens * self.top_k / self.num_experts))
        kept = torch.zeros_like(mask)
        for expert_index in range(self.num_experts):
            candidates = torch.nonzero(mask[:, expert_index], as_tuple=False).squeeze(-1)
            if candidates.numel() <= capacity:
                kept[candidates, expert_index] = True
            else:
                scores = probabilities[candidates, expert_index]
                chosen = candidates[torch.topk(scores, capacity, sorted=False).indices]
                kept[chosen, expert_index] = True
        return kept

    def _dynamic_mask(self, scores: torch.Tensor) -> torch.Tensor:
        maxima = scores.max(dim=-1, keepdim=True).values
        mask = scores >= self.dynamic_threshold * maxima
        # abs-valued routers can return all zeros; always execute one expert.
        empty = ~mask.any(dim=-1)
        if empty.any():
            rows = torch.nonzero(empty, as_tuple=False).squeeze(-1)
            columns = scores[rows].argmax(dim=-1)
            mask[rows, columns] = True
        return mask

    def _dense_ensemble(self, flat_x: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        values = torch.stack([expert(flat_x) for expert in self.experts], dim=1)
        return (values * probabilities.unsqueeze(-1)).sum(dim=1) + self.output_bias

    def expert_targets(self, flat_x: torch.Tensor, target_kind: str) -> torch.Tensor:
        """Calibration targets used only for independent router fitting."""
        outputs = []
        hidden_values = []
        for expert in self.experts:
            # Calibration/alignment targets represent the deterministic expert
            # function; training-time dropout would otherwise make the oracle
            # label change randomly from one step to the next.
            hidden = F.gelu(expert.fc1(flat_x))
            value = expert.fc2(hidden)
            outputs.append(value)
            hidden_values.append(hidden)
        stacked_outputs = torch.stack(outputs, dim=1)
        if target_kind == "output_norm":
            return stacked_outputs.norm(p=2, dim=-1)
        if target_kind == "positive_activation":
            return torch.stack(
                [hidden.clamp_min(0).sum(dim=-1) for hidden in hidden_values], dim=-1
            )
        if target_kind == "reconstruction_score":
            dense_partition_output = stacked_outputs.sum(dim=1, keepdim=True)
            # Higher is better. The shared FFN output bias cancels in this
            # expert-vs-dense comparison and therefore need not be added.
            return -(stacked_outputs - dense_partition_output).square().mean(dim=-1)
        raise ValueError(target_kind)

    def forward(self, x: torch.Tensor, dense_routing: bool = False):
        original_shape = x.shape
        flat_x = x.reshape(-1, self.d_model)
        scores = self.router_scores(flat_x)
        probabilities = torch.softmax(scores, dim=-1)
        if dense_routing:
            dense = self._dense_ensemble(flat_x, probabilities)
            return dense.reshape(original_shape), dense.new_zeros(()), {
                "probabilities": probabilities,
                "mask": torch.ones_like(probabilities, dtype=torch.bool),
                "selected_per_token": torch.full(
                    (flat_x.shape[0],), self.num_experts, device=x.device, dtype=torch.long
                ),
                "dropped_fraction": dense.new_zeros(()),
            }

        if self.router_kind == "d2d_regression":
            mask = self._dynamic_mask(scores)
        else:
            mask = self._fixed_k_mask(probabilities)

        if self.combine_mode == "softmax":
            weights = probabilities * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            weights = mask.to(flat_x.dtype)

        output = torch.zeros_like(flat_x)
        for expert_index, expert in enumerate(self.experts):
            token_index = torch.nonzero(mask[:, expert_index], as_tuple=False).squeeze(-1)
            if token_index.numel():
                value = expert(flat_x[token_index])
                output[token_index] += value * weights[token_index, expert_index, None]
        output = output + self.output_bias

        importance = probabilities.mean(dim=0)
        dispatch = mask.float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(importance * dispatch)
        selected = mask.sum(dim=-1)
        requested = self.top_k if self.router_kind != "d2d_regression" else selected.float().mean()
        if self.router_kind != "d2d_regression":
            dropped = (self.top_k - selected).clamp_min(0).float().sum()
            dropped = dropped / max(1, flat_x.shape[0] * self.top_k)
        else:
            dropped = output.new_zeros(())
        telemetry = {
            "probabilities": probabilities,
            "mask": mask,
            "usage": mask.float().mean(dim=0),
            "selected_per_token": selected,
            "mean_selected": selected.float().mean(),
            "requested": torch.as_tensor(requested, device=x.device),
            "dropped_fraction": dropped,
        }
        return output.reshape(original_shape), aux_loss, telemetry


class MoEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        router_kind: str,
        router_hidden_size: int,
        combine_mode: str,
        capacity_factor: float,
        drop_overflow_tokens: bool,
        dynamic_threshold: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SparseMoEFFN(
            d_model=d_model,
            hidden_size=expert_hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            router_kind=router_kind,
            router_hidden_size=router_hidden_size,
            combine_mode=combine_mode,
            capacity_factor=capacity_factor,
            drop_overflow_tokens=drop_overflow_tokens,
            dynamic_threshold=dynamic_threshold,
        )

    def forward(self, x: torch.Tensor, dense_routing: bool = False):
        x = x + self.attn(self.norm1(x))
        ffn_input = self.norm2(x)
        y, aux, telemetry = self.moe(ffn_input, dense_routing=dense_routing)
        return x + y, ffn_input, aux, telemetry


class MoETransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        expert_width: str,
        router_kind: str,
        router_hidden_size: int,
        combine_mode: str,
        capacity_factor: float,
        drop_overflow_tokens: bool,
        dynamic_threshold: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_width = expert_width
        expert_hidden = d_ff if expert_width == "full" else d_ff // num_experts
        self.expert_hidden_size = expert_hidden
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                MoEBlock(
                    d_model, n_heads, expert_hidden, num_experts, top_k, dropout,
                    router_kind, router_hidden_size, combine_mode, capacity_factor,
                    drop_overflow_tokens, dynamic_threshold,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, dense_routing: bool = False) -> ModelOutput:
        _, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions)[None])
        routing: List[Dict[str, torch.Tensor]] = []
        pre_ffn: List[torch.Tensor] = []
        aux_losses = []
        for block in self.blocks:
            x, ffn_input, aux, telemetry = block(x, dense_routing=dense_routing)
            pre_ffn.append(ffn_input)
            routing.append(telemetry)
            aux_losses.append(aux)
        logits = self.lm_head(self.final_norm(x))
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        return ModelOutput(logits=logits, aux_loss=aux_loss, routing=routing, pre_ffn=pre_ffn)


def parameter_report(model: MoETransformerLM) -> Dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    expert_total = sum(
        parameter.numel()
        for block in model.blocks
        for expert in block.moe.experts
        for parameter in expert.parameters()
    )
    router_total = sum(
        parameter.numel()
        for block in model.blocks
        if block.moe.router is not None
        for parameter in block.moe.router.parameters()
    )
    shared = total - expert_total - router_total
    per_expert = expert_total / (len(model.blocks) * model.num_experts)
    nominal_active = shared + router_total + per_expert * len(model.blocks) * model.top_k
    expert_flops_per_token = 2.0 * model.d_model * model.expert_hidden_size * model.top_k * len(model.blocks)
    return {
        "total_parameters": float(total),
        "shared_parameters": float(shared),
        "expert_parameters": float(expert_total),
        "router_parameters": float(router_total),
        "nominal_active_parameters": float(nominal_active),
        "nominal_expert_flops_per_token": float(expert_flops_per_token),
    }
