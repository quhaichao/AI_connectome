"""Read-only expert-specialization analysis for saved Core-6 checkpoints.

This module never trains or changes a checkpoint.  It uses the routing
probabilities returned by the common MoE forward pass, which also supports EMoE's
parameter-free average-key router.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import BenchmarkConfig, METHOD_SPECS
from .initialization import apply_initialization_policy
from .models import MoETransformerLM


METRICS = (
    "cosine_similarity",
    "routing_purity",
    "conditional_entropy",
    "nmi",
)


def config_from_saved(payload: Mapping[str, object]) -> BenchmarkConfig:
    values = dict(payload)
    for name in ("methods", "seeds", "d2d_threshold_sweep", "fc_hybrid_view_weights"):
        if name in values:
            values[name] = tuple(
                tuple(row) if name == "fc_hybrid_view_weights" else row
                for row in values[name]
            )
    cfg = BenchmarkConfig(**values)
    cfg.validate()
    return cfg


def build_checkpoint_model(method: str, vocab_size: int, cfg: BenchmarkConfig):
    spec = METHOD_SPECS[method]
    combine = (
        "softmax"
        if spec.construction in {"scratch", "copy", "cluster_aware_svd"}
        else "sum"
    )
    model = MoETransformerLM(
        vocab_size=vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        dropout=cfg.dropout,
        expert_width=spec.expert_width,
        router_kind=spec.router,
        router_hidden_size=cfg.router_hidden_size,
        combine_mode=combine,
        capacity_factor=cfg.capacity_factor,
        drop_overflow_tokens=cfg.drop_overflow_tokens,
        dynamic_threshold=cfg.d2d_threshold,
    )
    return apply_initialization_policy(
        model,
        tie_embeddings=cfg.tie_embeddings,
        embedding_init_std=cfg.embedding_init_std,
        xavier_initialize=cfg.xavier_initialize,
    )


def load_checkpoint_strict(model, path: str | Path) -> None:
    path = Path(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        state = torch.load(path, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch for {path.name}: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def prepare_pos_probe(
    dataset,
    vocabulary,
    max_batches: int,
    batch_size: int,
):
    """Prepare the identical validation tokens and POS labels used by every model."""
    count = min(len(dataset), max_batches * batch_size)
    input_batches = [
        dataset.inputs[start : min(start + batch_size, count)].contiguous()
        for start in range(0, count, batch_size)
    ]
    token_ids = torch.cat(input_batches, dim=0).reshape(-1).numpy()
    special_ids = {
        vocabulary.stoi[token]
        for token in ("<pad>", "<unk>", "<eos>")
        if token in vocabulary.stoi
    }
    valid_mask = np.asarray([int(token) not in special_ids for token in token_ids])
    valid_tokens = [
        vocabulary.itos[int(token)]
        for token, keep in zip(token_ids, valid_mask)
        if keep
    ]
    try:
        import nltk

        pos_tags = np.asarray([tag for _, tag in nltk.pos_tag(valid_tokens)], dtype=object)
    except LookupError as exc:
        raise RuntimeError(
            "NLTK POS tagger data are missing. Install/copy the English averaged "
            "perceptron tagger into this environment before running the notebook."
        ) from exc
    return input_batches, valid_mask, pos_tags, valid_tokens


@torch.inference_mode()
def collect_layer_routes(
    model,
    input_batches: Sequence[torch.Tensor],
    valid_mask: np.ndarray,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    model.eval()
    routes: Dict[int, List[np.ndarray]] = {layer: [] for layer in range(len(model.blocks))}
    for input_ids in input_batches:
        output = model(input_ids.to(device, non_blocking=True))
        if len(output.routing) != len(model.blocks):
            raise RuntimeError("Model did not return routing telemetry for every layer")
        for layer, telemetry in enumerate(output.routing):
            probabilities = telemetry["probabilities"]
            routes[layer].append(probabilities.argmax(dim=-1).detach().cpu().numpy())
    result = {}
    for layer, parts in routes.items():
        flattened = np.concatenate(parts).reshape(-1)
        if flattened.shape[0] != valid_mask.shape[0]:
            raise RuntimeError(
                f"Layer {layer}: routing/token mismatch {flattened.shape[0]} != {valid_mask.shape[0]}"
            )
        result[layer] = flattened[valid_mask]
    return result


def _mutual_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(labels_a, labels_b))


def compute_specialization_metrics(
    pos_tags: np.ndarray,
    routing: np.ndarray,
    num_experts: int,
) -> Dict[str, float]:
    if pos_tags.shape[0] != routing.shape[0] or pos_tags.shape[0] == 0:
        raise ValueError("POS and routing arrays must be aligned and non-empty")
    table = pd.crosstab(
        pd.Series(pos_tags, name="pos"),
        pd.Series(routing, name="expert"),
    ).reindex(columns=range(num_experts), fill_value=0)
    counts = table.to_numpy(dtype=np.float64)
    total = float(counts.sum())

    # Match the original notebook: cosine similarity is computed between each
    # active expert's POS-count vector; inactive experts are reported separately
    # and excluded so dead routing cannot masquerade as specialization.
    active = counts.sum(axis=0) > 0
    expert_vectors = counts[:, active].T
    if expert_vectors.shape[0] >= 2:
        normalized = F.normalize(torch.from_numpy(expert_vectors), dim=1).numpy()
        similarity = normalized @ normalized.T
        upper = similarity[np.triu_indices(similarity.shape[0], k=1)]
        cosine = float(upper.mean())
    else:
        # Preserve the original notebook convention. The accompanying
        # active_experts audit makes a collapsed single-expert solution visible.
        cosine = 1.0

    purity = float(counts.max(axis=1).sum() / total)
    pos_totals = counts.sum(axis=1)
    conditional_entropy = 0.0
    for row, row_total in zip(counts, pos_totals):
        if row_total <= 0:
            continue
        probability = row[row > 0] / row_total
        conditional_entropy += (row_total / total) * float(
            -(probability * np.log2(probability)).sum()
        )

    usage = counts.sum(axis=0) / total
    active_usage = usage[usage > 0]
    routing_entropy = float(
        -(active_usage * np.log2(active_usage)).sum() / math.log2(num_experts)
    )
    return {
        "cosine_similarity": cosine,
        "routing_purity": purity,
        "conditional_entropy": conditional_entropy,
        "nmi": _mutual_information(pos_tags, routing),
        "active_experts": int(active.sum()),
        "routing_entropy": routing_entropy,
        "tokens_analyzed": int(total),
    }


def analyze_model(
    model,
    input_batches: Sequence[torch.Tensor],
    valid_mask: np.ndarray,
    pos_tags: np.ndarray,
    device: torch.device,
    num_experts: int,
) -> List[Dict[str, float]]:
    routes = collect_layer_routes(model, input_batches, valid_mask, device)
    return [
        {"layer": layer, **compute_specialization_metrics(pos_tags, route, num_experts)}
        for layer, route in routes.items()
    ]


def bootstrap_mean_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260723,
) -> Tuple[float, float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(samples, alpha)), float(
        np.quantile(samples, 1.0 - alpha)
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_fc_comparisons(
    seed_level: pd.DataFrame,
    fc_method: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    lower_is_better = {"cosine_similarity", "conditional_entropy"}
    rows = []
    for metric in METRICS:
        metric_rows = []
        pivot = seed_level.pivot(index="seed", columns="method", values=metric)
        for method in methods:
            paired = pivot[[fc_method, method]].dropna()
            difference = paired[fc_method] - paired[method]
            advantage = -difference if metric in lower_is_better else difference
            if len(paired) < 2 or np.allclose(difference, 0):
                statistic, p_value = float("nan"), 1.0
            else:
                result = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            metric_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{fc_method} vs {method}",
                    "baseline": method,
                    "n_seeds": len(paired),
                    "mean_raw_difference_fc_minus_baseline": float(difference.mean()),
                    "mean_fc_advantage": float(advantage.mean()),
                    "median_fc_advantage": float(advantage.median()),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, value in zip(metric_rows, adjusted):
            row["p_holm_within_metric"] = float(value)
        rows.extend(metric_rows)
    return pd.DataFrame(rows)
