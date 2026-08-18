from __future__ import annotations

import math
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int, dtype: torch.dtype):
        return self.pe[:length].to(dtype=dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int, base: float):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        frequencies = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, length: int, dtype: torch.dtype):
        return self.cos[:length].to(dtype=dtype), self.sin[:length].to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class CausalMHA(nn.Module):
    def __init__(self, config: ModelConfig, max_len: int):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = (
            RotaryEmbedding(self.head_dim, max_len, config.rope_base)
            if config.position == "rope"
            else None
        )

    def forward(self, x: torch.Tensor):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.rope is not None:
            cos, sin = self.rope(length, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(out)


class OfficialMamba2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as error:
            raise ImportError(
                "a3 requires the official `mamba-ssm` package; the experiment will "
                "not substitute a simplified SSM. On the Linux/CUDA server install "
                "the pinned requirements-mamba2.txt environment first."
            ) from error
        self.mixer = Mamba2(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
        )

    def forward(self, x: torch.Tensor):
        return self.mixer(x)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.out_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        return self.out_proj(self.dropout(F.gelu(self.in_proj(x), approximate="tanh")))


class DynamicHyperConnection(nn.Module):
    """Paper-faithful Dynamic Hyper-Connection for one residual branch.

    ``state`` has shape [batch, tokens, streams, features]. Static routing is
    initialized to the Pre-Norm residual topology. Input-dependent routing uses
    zero-initialized projections, tanh and a small learnable scale, following
    Appendix J of Zhu et al. (2024).
    """

    def __init__(
        self,
        dim: int,
        streams: int,
        branch_index: int,
        dynamic_scale: float,
        use_tanh: bool,
    ):
        super().__init__()
        if streams < 2:
            raise ValueError("Hyper-Connections require at least two streams")
        self.streams = streams
        self.norm = nn.LayerNorm(dim)
        self.static_beta = nn.Parameter(torch.ones(streams))
        alpha_input = torch.zeros(streams, 1)
        alpha_input[branch_index % streams, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat((alpha_input, torch.eye(streams)), dim=1)
        )
        self.register_buffer(
            "_initial_static_alpha", self.static_alpha.detach().clone(), persistent=False
        )
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, streams + 1))
        self.dynamic_alpha_scale = nn.Parameter(torch.tensor(float(dynamic_scale)))
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
        self.dynamic_beta_scale = nn.Parameter(torch.tensor(float(dynamic_scale)))
        self.activation = torch.tanh if use_tanh else lambda value: value

    def width_connection(self, state: torch.Tensor):
        if state.ndim != 4 or state.shape[2] != self.streams:
            raise ValueError(
                "DHC state must have shape [batch, tokens, streams, features]"
            )
        normalized = self.norm(state)
        dynamic_alpha = self.activation(normalized @ self.dynamic_alpha_fn)
        alpha = (
            dynamic_alpha * self.dynamic_alpha_scale
            + self.static_alpha[None, None, :, :]
        )
        dynamic_beta = self.activation(normalized @ self.dynamic_beta_fn)
        beta = (
            dynamic_beta * self.dynamic_beta_scale
            + self.static_beta[None, None, :]
        )
        # Batched matmul is substantially faster than the equivalent einsum on
        # common CUDA kernels and avoids the old implementation's routing bottleneck.
        mixed = torch.matmul(alpha.transpose(-2, -1), state)
        return mixed[:, :, 0, :], mixed[:, :, 1:, :], beta

    @staticmethod
    def depth_connection(
        residual_streams: torch.Tensor,
        branch_output: torch.Tensor,
        beta: torch.Tensor,
    ):
        return residual_streams + branch_output.unsqueeze(2) * beta.unsqueeze(-1)


class TransformerBlock(nn.Module):
    def __init__(
        self, config: ModelConfig, mixer_type: str, max_len: int, layer_index: int
    ):
        super().__init__()
        self.config = config
        if mixer_type == "mha":
            self.mixer = CausalMHA(config, max_len)
        elif mixer_type == "mamba2":
            self.mixer = OfficialMamba2(config)
        else:
            raise ValueError(f"Unknown mixer {mixer_type!r}")
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.residual_dropout = nn.Dropout(config.dropout)
        if config.residual == "dhc" and config.norm_position != "pre":
            raise ValueError("The controlled DHC comparison uses Pre-Norm blocks")
        self.attention_hc = (
            DynamicHyperConnection(
                config.d_model,
                config.hc_streams,
                branch_index=2 * layer_index,
                dynamic_scale=config.hc_dynamic_scale,
                use_tanh=config.hc_tanh,
            )
            if config.residual == "dhc"
            else None
        )
        self.ffn_hc = (
            DynamicHyperConnection(
                config.d_model,
                config.hc_streams,
                branch_index=2 * layer_index + 1,
                dynamic_scale=config.hc_dynamic_scale,
                use_tanh=config.hc_tanh,
            )
            if config.residual == "dhc"
            else None
        )

    def forward(self, state: torch.Tensor, return_effective_input: bool = False):
        if self.attention_hc is not None:
            branch_input, residual_streams, beta = self.attention_hc.width_connection(
                state
            )
            effective_input = branch_input
            branch_output = self.residual_dropout(
                self.mixer(self.norm1(branch_input))
            )
            state = self.attention_hc.depth_connection(
                residual_streams, branch_output, beta
            )
            branch_input, residual_streams, beta = self.ffn_hc.width_connection(state)
            branch_output = self.residual_dropout(self.ffn(self.norm2(branch_input)))
            state = self.ffn_hc.depth_connection(
                residual_streams, branch_output, beta
            )
            return (state, effective_input) if return_effective_input else state
        effective_input = state
        if self.config.norm_position == "pre":
            state = state + self.residual_dropout(self.mixer(self.norm1(state)))
            state = state + self.residual_dropout(self.ffn(self.norm2(state)))
        else:
            state = self.norm1(state + self.residual_dropout(self.mixer(state)))
            state = self.norm2(state + self.residual_dropout(self.ffn(state)))
        return (state, effective_input) if return_effective_input else state


class AdaptiveStreamReadout(nn.Module):
    """Token-adaptive final pooling over HC streams.

    All parameters start at zero, so the initial readout is exactly the uniform
    stream mean. This module is downstream of every intermediate FC tap and
    therefore never enters the FC forward calculation.
    """

    def __init__(self, dim: int, streams: int):
        super().__init__()
        self.global_logits = nn.Parameter(torch.zeros(streams))
        self.query = nn.Parameter(torch.zeros(dim))

    def forward(self, state: torch.Tensor):
        normalized = F.layer_norm(state, (state.shape[-1],))
        content_logits = normalized @ self.query / math.sqrt(state.shape[-1])
        weights = torch.softmax(
            content_logits + self.global_logits[None, None, :], dim=2
        )
        return (state * weights.unsqueeze(-1)).sum(dim=2)


class CausalLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig, max_len: int):
        super().__init__()
        if config.vocab_size <= 2:
            raise ValueError("vocab_size must be filled from the training split")
        if len(config.resolved_mixers()) != config.n_layers:
            raise ValueError("mixer_pattern length must equal n_layers")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = (
            SinusoidalEncoding(config.d_model, max_len)
            if config.position == "sinusoidal"
            else None
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(config, mixer, max_len, layer_index)
            for layer_index, mixer in enumerate(config.resolved_mixers())
        )
        # Kept in both Pre- and Post-Norm variants so placement inside blocks is
        # the only changed field in a4.
        self.final_norm = nn.LayerNorm(config.d_model)
        self.stream_readout = (
            AdaptiveStreamReadout(config.d_model, config.hc_streams)
            if config.residual == "dhc" and config.hc_adaptive_readout
            else None
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        # Preserve Mamba-2's targeted SSM/dt initialization. The official project
        # explicitly warns against blanket post-initialization of its internals.
        protected = set()
        for module in self.modules():
            if isinstance(module, OfficialMamba2):
                protected.update(id(child) for child in module.modules())
        for module in self.modules():
            if id(module) not in protected:
                self._init_weights(module)
        residual_scale = 1.0 / math.sqrt(2.0 * config.n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("mixer.out_proj.weight") or name.endswith("ffn.out_proj.weight"):
                parameter.data.mul_(residual_scale)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initial_state(self, tokens: torch.Tensor):
        # Original Transformer convention: scale token embeddings before adding
        # fixed sinusoidal PE. The same scaling is retained in the RoPE branch so
        # a1 changes the positional mechanism, not residual-stream magnitude.
        state = self.embedding(tokens) * math.sqrt(self.config.d_model)
        if self.position is not None:
            state = state + self.position(tokens.shape[1], state.dtype)
        state = self.embedding_dropout(state)
        return state

    def canonical_state(self, state: torch.Tensor):
        # The LM head always consumes D operational units. Adaptive stream
        # pooling is used only after the final block; intermediate FC taps return
        # the actual routed D-dimensional branch input before this readout.
        if state.ndim == 4:
            return (
                self.stream_readout(state)
                if self.stream_readout is not None
                else state.mean(dim=2)
            )
        return state

    def forward(self, tokens: torch.Tensor, return_layer_states: bool = False):
        state = self._initial_state(tokens)
        if self.config.residual == "dhc":
            state = state.unsqueeze(2).expand(
                -1, -1, self.config.hc_streams, -1
            )
        taps: list[torch.Tensor] = []
        for layer in self.layers:
            if return_layer_states:
                state, effective_input = layer(
                    state, return_effective_input=True
                )
                taps.append(effective_input)
            else:
                state = layer(state)
        hidden = self.final_norm(self.canonical_state(state))
        logits = self.lm_head(hidden)
        return (logits, taps) if return_layer_states else logits


def build_model(config: ModelConfig, max_len: int):
    return CausalLanguageModel(config, max_len)


def copy_shared_initialization(source: nn.Module, target: nn.Module):
    """Copy every name-and-shape-compatible tensor and return an audit record."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied: list[str] = []
    for name, tensor in target_state.items():
        if name in source_state and source_state[name].shape == tensor.shape:
            target_state[name] = source_state[name].detach().clone()
            copied.append(name)
    target.load_state_dict(target_state)
    return copied


@torch.inference_mode()
def hyper_connection_diagnostics(model: nn.Module):
    """Summarize learned DHC routing without storing token-level activations."""
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, DynamicHyperConnection):
            continue
        rows.append(
            {
                "module": name,
                "static_alpha_delta_l2": float(
                    (module.static_alpha - module._initial_static_alpha).norm().cpu()
                ),
                "static_beta_delta_l2": float(
                    (module.static_beta - 1.0).norm().cpu()
                ),
                "dynamic_alpha_projection_l2": float(
                    module.dynamic_alpha_fn.norm().cpu()
                ),
                "dynamic_beta_projection_l2": float(
                    module.dynamic_beta_fn.norm().cpu()
                ),
                "dynamic_alpha_scale": float(module.dynamic_alpha_scale.cpu()),
                "dynamic_beta_scale": float(module.dynamic_beta_scale.cpu()),
            }
        )
    readout = getattr(model, "stream_readout", None)
    if readout is not None:
        rows.append(
            {
                "module": "stream_readout",
                "global_weights": torch.softmax(
                    readout.global_logits, dim=0
                ).detach().cpu().tolist(),
                "global_logits_l2": float(readout.global_logits.norm().cpu()),
                "query_l2": float(readout.query.norm().cpu()),
            }
        )
    return rows


def with_vocab(config: ModelConfig, vocab_size: int):
    return replace(config, vocab_size=vocab_size)
