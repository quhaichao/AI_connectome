from __future__ import annotations

import math

import torch

from .config import FCConfig


def resolve_fc_layers(selection, n_layers: int) -> tuple[int, ...]:
    """Resolve ``"all"`` or one zero-based block index to concrete layers."""
    if selection == "all":
        return tuple(range(n_layers))
    if isinstance(selection, int) and not isinstance(selection, bool):
        if selection < 0 or selection >= n_layers:
            raise IndexError(
                f"FC layer {selection} is outside a {n_layers}-layer model"
            )
        return (selection,)
    raise ValueError('FC layer_selection must be "all" or a zero-based integer')


def _top_edge_summary(observations: torch.Tensor, config: FCConfig):
    """Return the signed top-edge mean for a 2-D observation-by-unit matrix."""
    observations = observations - observations.mean(dim=0, keepdim=True)
    std = observations.std(dim=0, unbiased=True)
    valid = torch.isfinite(std) & (std > config.min_feature_std)
    observations = observations[:, valid]
    if observations.shape[1] < 2:
        raise ValueError("fewer than two non-constant operational units in FC probe")
    observations = observations / observations.std(
        dim=0, unbiased=True
    ).clamp_min(config.min_feature_std)
    correlation = observations.T @ observations / max(observations.shape[0] - 1, 1)
    indices = torch.triu_indices(
        correlation.shape[0], correlation.shape[1], offset=1, device=correlation.device
    )
    edges = correlation[indices[0], indices[1]]
    edges = edges[torch.isfinite(edges)]
    if edges.numel() == 0:
        raise ValueError("no finite off-diagonal FC values")
    k = max(1, math.ceil(config.top_fraction * edges.numel()))
    top = torch.topk(edges, k=k, largest=True, sorted=False).values
    return {
        "top_fc_mean": float(top.mean().cpu()),
        "n_features": int(observations.shape[1]),
        "n_observations": int(observations.shape[0]),
        "n_edges": int(edges.numel()),
        "n_top_edges": int(k),
    }


def top_fc_mean(activations: torch.Tensor, config: FCConfig):
    """Compute the signed mean of the largest 5% pairwise Pearson FC values.

    ``activations`` is [sequences, tokens, features]. The primary estimator first
    z-scores each feature across sequences *within each token position*, then
    concatenates positions and sequences. This retains the manuscript definition
    of a feature dimension as an operational unit while conditioning out both the
    deterministic positional mean and position-dependent variance. The latter is
    essential for comparing additive sinusoidal encoding with RoPE fairly.

    ``raw_top_fc_mean`` is retained as a sensitivity diagnostic but is never used
    for the primary onset test.
    """
    if activations.ndim != 3:
        raise ValueError("activations must have shape [sequences, tokens, features]")
    values = activations.detach().float()
    raw = _top_edge_summary(values.reshape(-1, values.shape[-1]), config)
    if config.position_standardize:
        position_mean = values.mean(dim=0, keepdim=True)
        position_std = values.std(dim=0, unbiased=True, keepdim=True)
        valid = torch.isfinite(position_std).all(dim=1).squeeze(0)
        valid &= (position_std > config.min_feature_std).all(dim=1).squeeze(0)
        values = values[:, :, valid]
        position_mean = position_mean[:, :, valid]
        position_std = position_std[:, :, valid]
        if values.shape[-1] < 2:
            raise ValueError(
                "fewer than two units vary at every token position in the FC probe"
            )
        values = (values - position_mean) / position_std.clamp_min(
            config.min_feature_std
        )
    primary = _top_edge_summary(values.reshape(-1, values.shape[-1]), config)
    primary.update(
        {
            "estimator": (
                "position-standardized concatenated Pearson"
                if config.position_standardize
                else "raw concatenated Pearson"
            ),
            "raw_top_fc_mean": raw["top_fc_mean"],
            "raw_n_features": raw["n_features"],
        }
    )
    return primary


@torch.inference_mode()
def measure_model_fc(
    model,
    probe_tokens: torch.Tensor,
    config: FCConfig,
    device: torch.device,
    batch_size: int,
    use_bf16: bool = False,
):
    model.eval()
    layer_indices = resolve_fc_layers(config.layer_selection, len(model.layers))
    by_layer = {layer: [] for layer in layer_indices}
    for start in range(0, probe_tokens.shape[0], batch_size):
        tokens = probe_tokens[start : start + batch_size].to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and use_bf16,
        ):
            _, taps = model(tokens, return_layer_states=True)
        if len(taps) != len(model.layers):
            raise RuntimeError(
                f"model returned {len(taps)} FC taps for {len(model.layers)} layers"
            )
        for layer in layer_indices:
            by_layer[layer].append(taps[layer].detach().cpu())
    layer_results = {
        str(layer): top_fc_mean(torch.cat(chunks, dim=0), config)
        for layer, chunks in by_layer.items()
    }
    primary = sum(item["top_fc_mean"] for item in layer_results.values()) / len(layer_results)
    raw = sum(item["raw_top_fc_mean"] for item in layer_results.values()) / len(
        layer_results
    )
    return {
        "top_fc_mean": primary,
        "raw_top_fc_mean": raw,
        "layers": layer_results,
        "layer_selection": config.layer_selection,
        "resolved_layer_indices": list(layer_indices),
        "layer_aggregation": (
            "equal-weight arithmetic mean of the independently computed "
            "per-layer top-edge FC means"
        ),
        "tap_semantics": (
            "D-dimensional effective input to each selected block; ordinary "
            "residual state for baseline and routed branch input for DHC"
        ),
    }
