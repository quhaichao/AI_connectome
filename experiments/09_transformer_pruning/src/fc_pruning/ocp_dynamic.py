from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class OCPController:
    def __init__(
        self,
        number_layers: int,
        first_pruned_layer: int,
        overall_target: float,
        beta: float = 0.95,
        gamma: float = 0.1,
        clip_delta: float = 0.1,
    ):
        self.number_layers = number_layers
        self.first_pruned_layer = first_pruned_layer
        self.active_layers = number_layers - first_pruned_layer
        self.active_target = overall_target * number_layers / self.active_layers
        self.beta = beta
        self.gamma = gamma
        self.minimum = max(0.0, self.active_target - clip_delta)
        self.maximum = min(1.0, self.active_target + clip_delta)
        self.history = torch.zeros(number_layers, dtype=torch.float64)
        self.seen = torch.zeros(number_layers, dtype=torch.long)
        self.drift = 0.0
        self.batch_index = 0
        self.records: list[dict] = []

    def begin_batch(self) -> None:
        self.drift = 0.0
        self.batch_index += 1

    def allocate(self, layer_index: int, density: float) -> float:
        self.history[layer_index] = (
            self.beta * self.history[layer_index] + (1.0 - self.beta) * density
        )
        self.seen[layer_index] += 1
        active_history = self.history[self.first_pruned_layer :]
        global_mean = float(active_history.mean())
        base = self.active_target - self.gamma * (
            float(self.history[layer_index]) - global_mean
        )
        remaining = self.number_layers - layer_index
        ratio = min(
            self.maximum,
            max(self.minimum, base + self.drift / remaining),
        )
        self.drift += self.active_target - ratio
        self.records.append(
            {
                "batch": self.batch_index,
                "layer": layer_index,
                "outlier_density": density,
                "history_density": float(self.history[layer_index]),
                "base_ratio": base,
                "prune_ratio": ratio,
                "drift_after": self.drift,
            }
        )
        return ratio


class OCPDynamicMLP(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        layer_index: int,
        controller: OCPController,
        token_fraction: float = 0.5,
    ):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        self.layer_index = layer_index
        self.controller = controller
        self.token_fraction = token_fraction
        sensitivity = (
            self.gate_proj.weight.detach().float().abs().sum(dim=0)
            + self.up_proj.weight.detach().float().abs().sum(dim=0)
        )
        self.register_buffer("sensitivity", sensitivity, persistent=False)
        down_norm_sq = self.down_proj.weight.detach().float().square().sum(dim=0)
        self.register_buffer("down_norm_sq", down_norm_sq, persistent=False)

    def _probe(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _hidden = hidden.shape
        sensitivity = self.sensitivity.to(hidden.device, hidden.dtype)
        scores = (hidden * sensitivity).float().square().sum(dim=-1)
        token_count = max(1, int(round(length * self.token_fraction)))
        positions = torch.topk(scores, token_count, dim=1, sorted=False).indices
        gather = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return hidden.gather(1, gather)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.layer_index == self.controller.first_pruned_layer:
            self.controller.begin_batch()
        probe_hidden = self._probe(hidden)
        probe_activation = self.act_fn(self.gate_proj(probe_hidden)) * self.up_proj(
            probe_hidden
        )
        absolute = probe_activation.float().abs()
        threshold = absolute.mean() + 2.0 * absolute.std(unbiased=False)
        density = float((absolute > threshold).float().mean())
        ratio = self.controller.allocate(self.layer_index, density)
        target = int(round(probe_activation.shape[-1] * ratio))

        importance = (
            probe_activation.float().square().mean(dim=(0, 1))
            * self.down_norm_sq.to(probe_activation.device)
        )
        pruned = torch.topk(importance, target, largest=False).indices
        keep_mask = torch.ones(
            importance.numel(), dtype=torch.bool, device=importance.device
        )
        keep_mask[pruned] = False
        keep = keep_mask.nonzero(as_tuple=False).flatten()

        gate = F.linear(hidden, self.gate_proj.weight.index_select(0, keep))
        up = F.linear(hidden, self.up_proj.weight.index_select(0, keep))
        activation = self.act_fn(gate) * up
        output = F.linear(
            activation, self.down_proj.weight.index_select(1, keep)
        )
        return output


def install_ocp_dynamic(
    model,
    overall_target: float,
    first_pruned_layer: int = 3,
    token_fraction: float = 0.5,
    beta: float = 0.95,
    gamma: float = 0.1,
    clip_delta: float = 0.1,
) -> OCPController:
    layers = model.model.layers
    controller = OCPController(
        len(layers),
        first_pruned_layer,
        overall_target,
        beta=beta,
        gamma=gamma,
        clip_delta=clip_delta,
    )
    for index in range(first_pruned_layer, len(layers)):
        layers[index].mlp = OCPDynamicMLP(
            layers[index].mlp,
            index,
            controller,
            token_fraction=token_fraction,
        )
    return controller
