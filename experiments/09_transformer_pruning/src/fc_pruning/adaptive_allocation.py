from __future__ import annotations

import torch


def exact_integer_targets(
    desired: torch.Tensor,
    total: int,
    lower: int = 0,
    upper: int | None = None,
) -> list[int]:
    """Round real-valued layer budgets while preserving the exact total."""
    values = desired.double()
    if upper is None:
        upper = int(torch.ceil(values.max()).item())
    targets = values.floor().long().clamp(lower, upper)
    difference = int(total - targets.sum().item())
    fractions = values - values.floor()
    if difference > 0:
        order = torch.argsort(fractions, descending=True).tolist()
        while difference:
            changed = False
            for index in order:
                if targets[index] >= upper:
                    continue
                targets[index] += 1
                difference -= 1
                changed = True
                if difference == 0:
                    break
            if not changed:
                raise ValueError("Upper bounds cannot satisfy the requested total")
    elif difference < 0:
        order = torch.argsort(fractions, descending=False).tolist()
        while difference:
            changed = False
            for index in order:
                if targets[index] <= lower:
                    continue
                targets[index] -= 1
                difference += 1
                changed = True
                if difference == 0:
                    break
            if not changed:
                raise ValueError("Lower bounds cannot satisfy the requested total")
    return targets.tolist()


def fang_asa_targets(
    functional_complexity: torch.Tensor,
    width: int,
    ratio: float,
) -> tuple[list[int], list[float]]:
    """Allocate FANG ASA sparsity from block functional complexity.

    FANG defines complexity as one minus block input/output cosine and assigns
    sparsity proportional to one minus complexity. Centering the proportional
    signal and using the largest admissible linear scale preserves the exact
    global budget while keeping every layer inside [0.5 sp, 1.5 sp].
    """
    complexity = functional_complexity.double()
    pruning_signal = 1.0 - complexity
    centered = pruning_signal - pruning_signal.mean()
    lower = 0.5 * ratio
    upper = 1.5 * ratio
    scales = []
    positive = float(centered.max())
    negative = float((-centered.min()).clamp_min(0.0))
    if positive > 0:
        scales.append((upper - ratio) / positive)
    if negative > 0:
        scales.append((ratio - lower) / negative)
    scale = min(scales) if scales else 0.0
    ratios = (ratio + scale * centered).clamp(lower, upper)
    total = int(round(len(complexity) * width * ratio))
    targets = exact_integer_targets(
        ratios * width,
        total,
        lower=int(torch.ceil(torch.tensor(lower * width)).item()),
        upper=int(torch.floor(torch.tensor(upper * width)).item()),
    )
    return targets, [target / width for target in targets]


def global_layer_targets(
    scores: list[torch.Tensor],
    total: int,
    maximum_fraction: float = 0.8,
) -> list[int]:
    """Select globally least-important units with a per-layer safety bound."""
    if not scores:
        return []
    widths = [score.numel() for score in scores]
    offsets = torch.tensor([0, *torch.tensor(widths).cumsum(0).tolist()])
    flat = torch.cat([score.detach().double().flatten().cpu() for score in scores])
    order = torch.argsort(flat).tolist()
    maxima = [int(width * maximum_fraction) for width in widths]
    targets = [0 for _ in widths]
    selected = 0
    for flat_index in order:
        layer_index = int(torch.searchsorted(offsets[1:], flat_index, right=True))
        if targets[layer_index] >= maxima[layer_index]:
            continue
        targets[layer_index] += 1
        selected += 1
        if selected == total:
            break
    if selected != total:
        raise ValueError("Per-layer bounds cannot satisfy the global budget")
    return targets
