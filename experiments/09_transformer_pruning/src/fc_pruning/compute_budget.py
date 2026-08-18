from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CompressionBudget:
    hidden_size: int
    intermediate_size: int
    sequence_length: int
    number_layers: int
    ffn_prune_ratio: float
    baseline_layer_macs: int
    target_layer_macs: float
    target_model_macs: float
    target_ffn_macs: float
    svd_rank: int
    svd_ffn_macs: int
    slice_hidden_size: int
    slice_layer_macs: int
    slice_model_macs: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "svd_ffn_target_ratio": self.svd_ffn_macs
                / self.target_ffn_macs,
                "slice_layer_target_ratio": self.slice_layer_macs
                / self.target_layer_macs,
                "slice_model_target_ratio": self.slice_model_macs
                / self.target_model_macs,
                "target_layer_baseline_ratio": self.target_layer_macs
                / self.baseline_layer_macs,
                "slice_layer_baseline_ratio": self.slice_layer_macs
                / self.baseline_layer_macs,
            }
        )
        return payload


def llama_layer_macs(
    hidden_size: int,
    intermediate_size: int,
    sequence_length: int,
) -> int:
    """MACs for attention projections/matmuls and a gated Llama FFN.

    Embedding lookup, normalization, rotary operations and the LM head are
    excluded because the comparison concerns the repeated decoder layer.
    Llama-2-7B has ordinary multi-head attention, so Q/K/V/O each use a
    hidden_size by hidden_size projection.
    """
    d = hidden_size
    m = intermediate_size
    length = sequence_length
    attention_projections = 4 * length * d * d
    attention_scores_and_values = 2 * length * length * d
    gated_ffn = 3 * length * d * m
    return attention_projections + attention_scores_and_values + gated_ffn


def equal_macs_budget(
    hidden_size: int,
    intermediate_size: int,
    sequence_length: int,
    ffn_prune_ratio: float,
    slice_round_interval: int = 8,
    number_layers: int = 32,
) -> CompressionBudget:
    if not 0.0 <= ffn_prune_ratio < 1.0:
        raise ValueError("ffn_prune_ratio must be in [0, 1)")
    if slice_round_interval <= 0:
        raise ValueError("slice_round_interval must be positive")
    if number_layers <= 0:
        raise ValueError("number_layers must be positive")

    d = hidden_size
    m = intermediate_size
    length = sequence_length
    keep = 1.0 - ffn_prune_ratio
    baseline_ffn = 3 * length * d * m
    baseline_layer = llama_layer_macs(d, m, length)
    target_ffn = keep * baseline_ffn
    target_layer = baseline_layer - baseline_ffn + target_ffn
    target_model = number_layers * target_layer

    # Each of gate/up/down becomes two linear maps with rank r.
    exact_rank = keep * d * m / (d + m)
    rank_candidates = {
        max(1, min(d, int(exact_rank))),
        max(1, min(d, int(exact_rank) + 1)),
    }
    svd_rank = min(
        rank_candidates,
        key=lambda rank: abs(3 * length * rank * (d + m) - target_ffn),
    )
    svd_ffn = 3 * length * svd_rank * (d + m)

    # SliceGPT reduces the residual stream but retains Llama's original
    # attention head dimension. Its learned/estimated per-block rotations are
    # explicit residual-path matmuls. The final MLP returns to the unsliced
    # dimension because the LM head is not sliced in the official default.
    def slicegpt_model_macs(candidate: int) -> int:
        attention = (
            4 * length * d * candidate + 2 * length * length * d
        )
        regular_ffn = 3 * length * intermediate_size * candidate
        regular_shortcuts = 2 * length * candidate * candidate
        regular_layer = attention + regular_ffn + regular_shortcuts
        final_ffn = (
            2 * length * intermediate_size * candidate
            + length * intermediate_size * d
        )
        final_shortcuts = (
            length * candidate * candidate + length * candidate * d
        )
        final_layer = attention + final_ffn + final_shortcuts
        return (number_layers - 1) * regular_layer + final_layer

    dimensions = range(slice_round_interval, d + 1, slice_round_interval)
    slice_hidden = min(
        dimensions,
        key=lambda candidate: abs(slicegpt_model_macs(candidate) - target_model),
    )
    slice_model = slicegpt_model_macs(slice_hidden)
    slice_layer = slice_model // number_layers

    return CompressionBudget(
        hidden_size=d,
        intermediate_size=m,
        sequence_length=length,
        number_layers=number_layers,
        ffn_prune_ratio=ffn_prune_ratio,
        baseline_layer_macs=baseline_layer,
        target_layer_macs=target_layer,
        target_model_macs=target_model,
        target_ffn_macs=target_ffn,
        svd_rank=svd_rank,
        svd_ffn_macs=svd_ffn,
        slice_hidden_size=slice_hidden,
        slice_layer_macs=slice_layer,
        slice_model_macs=slice_model,
    )
