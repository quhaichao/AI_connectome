"""Checkpoint-compatible model definitions for Supplementary Figure 1.

The classes mirror the architectures used in the main MLP, CNN and
Transformer analyses.  Training stays in the original analysis notebooks;
this module only reconstructs models so saved ``state_dict`` objects can be
loaded without executing those notebooks from top to bottom.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    """Two-or-more-hidden-layer MLP used by the main MLP notebook."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: tuple[int, ...] | list[int] = (100, 100),
        num_classes: int = 10,
        dropout_p: float = 0.0,
        activation: str = "relu",
        init_weights: bool = True,
        init_method: str = "kaiming",
    ) -> None:
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.linear_layers = nn.ModuleList()
        layers: list[nn.Module] = []

        current_dim = input_size
        for hidden_dim in self.hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            self.linear_layers.append(linear)
            layers.extend([linear, self._get_activation(activation)])
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim

        output_layer = nn.Linear(current_dim, num_classes)
        self.linear_layers.append(output_layer)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)

        if init_weights:
            self._initialize_weights(init_method)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(f"Unsupported activation: {name}")

    def _initialize_weights(self, method: str) -> None:
        method = method.lower()
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if method == "kaiming":
                init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif method == "xavier":
                init.xavier_normal_(module.weight)
            elif method == "orthogonal":
                init.orthogonal_(module.weight)
            elif method == "normal":
                init.normal_(module.weight, mean=0.0, std=0.01)
            else:
                raise ValueError(f"Unsupported initialization: {method}")
            if module.bias is not None:
                init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(x.size(0), -1))

    def get_hidden_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.view(x.size(0), -1)
        activations: list[torch.Tensor] = []
        activation_types = (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)
        for layer in self.network:
            x = layer(x)
            if isinstance(layer, activation_types):
                activations.append(x.clone())
        return activations


class SimpleCNN(nn.Module):
    """CNN used in the main CNN notebook (32 x 32 inputs)."""

    def __init__(self, in_channels: int = 1, activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 5 * 5, 10)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activations: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act(self.bn1(self.conv1(x))))
        self.activations["conv1"] = x.clone()
        x = self.pool2(self.act(self.bn2(self.conv2(x))))
        self.activations["conv2"] = x.clone()
        return self.fc(x.view(x.size(0), -1))

    @staticmethod
    def get_layer_shapes() -> dict[str, tuple[int, int, int]]:
        return {"conv1": (32, 14, 14), "conv2": (64, 5, 5)}


class TransformerEncoderLayerCustom(nn.Module):
    """Pre-Norm Transformer block used in the main Transformer analysis."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout(attended)
        transformed = self.ff2(self.dropout(self.act(self.ff1(self.norm2(x)))))
        return x + self.dropout(transformed)


class TransformerLM(nn.Module):
    """Word-level language model used in ``transformer_sc_fc_analysis.py``."""

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        seq_len: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerCustom(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length = x.shape
        positions = torch.arange(length, device=x.device).unsqueeze(0)
        h = self.dropout(self.embedding(x) + self.pos_enc(positions))
        for layer in self.layers:
            h = layer(h)
        return self.fc_out(self.norm(h))

    def get_layer_outputs(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        _, length = x.shape
        positions = torch.arange(length, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_enc(positions)
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            h_in = h
            h = layer(h)
            outputs.append((h_in, h))
        return outputs


def strip_state_dict_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str = "module."
) -> dict[str, torch.Tensor]:
    """Remove a DataParallel prefix without altering other state keys."""

    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def load_state_dict_file(
    model: nn.Module,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load either a raw state dict or a checkpoint containing ``state_dict``."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a 'state_dict' key.")
    model.load_state_dict(strip_state_dict_prefix(checkpoint), strict=strict)
    return model

