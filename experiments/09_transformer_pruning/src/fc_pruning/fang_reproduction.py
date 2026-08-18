from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import load_calibration
from .modeling import llama_layers
from .pruning import apply_layer_plan


@torch.inference_mode()
def pca_bases_from_covariances(
    covariance_statistics: dict,
    components: int,
    device: torch.device,
) -> list[torch.Tensor]:
    bases = []
    for index, statistics in enumerate(covariance_statistics["layers"]):
        print(f"FANG PCA layer={index:02d}", flush=True)
        covariance = statistics["hidden_covariance"].float().to(device)
        _values, vectors = torch.linalg.eigh(covariance)
        bases.append(vectors[:, -components:].flip(1).cpu())
        del covariance, _values, vectors
        torch.cuda.empty_cache()
    return bases


@torch.inference_mode()
def collect_projected_ffn_inputs(
    model,
    calibration_path: str,
    bases: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    total = int(input_ids.numel())
    components = bases[0].shape[1]
    projected = torch.empty(
        (len(layers), total, components), dtype=torch.float16
    )
    context = {"index": 0}
    device_bases = [basis.to(device) for basis in bases]

    def make_hook(layer_index: int):
        def hook(_module, args):
            hidden = args[0][0].float()
            begin = context["index"] * input_ids.shape[1]
            end = begin + hidden.shape[0]
            projected[layer_index, begin:end].copy_(
                (hidden @ device_bases[layer_index]).half().cpu()
            )

        return hook

    handles = [
        layer.mlp.gate_proj.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        for index, sequence in enumerate(input_ids):
            context["index"] = index
            print(
                f"FANG projected inputs context={index + 1:03d}/{input_ids.shape[0]:03d}",
                flush=True,
            )
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return projected


@torch.inference_mode()
def kmeans_assignments(
    projected: torch.Tensor,
    clusters: int,
    iterations: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    all_assignments = []
    all_centers = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for layer_index in range(projected.shape[0]):
        print(f"FANG K-means layer={layer_index:02d}", flush=True)
        data = projected[layer_index].float().to(device)
        initial = torch.randperm(data.shape[0], generator=generator)[:clusters]
        centers = data.index_select(0, initial.to(device)).clone()
        for _iteration in range(iterations):
            distances = torch.cdist(data, centers)
            assignment = distances.argmin(dim=1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, assignment, data)
            counts = torch.bincount(assignment, minlength=clusters).clamp_min(1)
            centers = sums / counts.unsqueeze(1)
        all_assignments.append(assignment.to(torch.uint8).cpu())
        all_centers.append(centers.cpu())
        del data, centers, distances, assignment, sums, counts
        torch.cuda.empty_cache()
    return torch.stack(all_assignments), all_centers


def collect_fang_cluster_statistics(
    model,
    calibration_path: str,
    assignments: torch.Tensor,
    clusters: int,
    device: torch.device,
) -> dict:
    calibration = load_calibration(calibration_path)
    input_ids = calibration["input_ids"]
    layers = llama_layers(model)
    width = model.config.intermediate_size
    all_sums = []
    all_sums_sq = []
    all_taylor = []
    all_counts = []
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Retain only one layer's activation gradient graph at a time. Keeping
    # hooks on all 32 layers simultaneously exceeds 7B-model GPU memory.
    for layer_index, layer in enumerate(layers):
        sums = torch.zeros((clusters, width), device=device)
        sums_sq = torch.zeros_like(sums)
        taylor = torch.zeros_like(sums)
        counts = torch.zeros(clusters, dtype=torch.long, device=device)
        context = {"index": 0}

        def hook(_module, args):
            activation = args[0]
            if not activation.requires_grad:
                activation.requires_grad_(True)
            begin = context["index"] * input_ids.shape[1]
            end = begin + input_ids.shape[1]
            labels = assignments[layer_index, begin:end].long().to(device)
            # Keep only detached values in the tensor hook. Capturing the
            # graph-bearing activation here creates a reference cycle and
            # retains every previous window's backward graph.
            values = activation[0].detach().float()
            for cluster in range(clusters):
                mask = labels == cluster
                selected = values[mask]
                counts[cluster] += selected.shape[0]
                sums[cluster] += selected.sum(dim=0)
                sums_sq[cluster] += selected.square().sum(dim=0)

            def gradient_hook(gradient):
                grad = gradient[0].float()
                for cluster in range(clusters):
                    mask = labels == cluster
                    taylor[cluster] += (values[mask] * grad[mask]).sum(dim=0)

            activation.register_hook(gradient_hook)

        handle = layer.mlp.down_proj.register_forward_pre_hook(hook)
        try:
            for index, sequence in enumerate(input_ids):
                context["index"] = index
                print(
                    f"FANG Taylor layer={layer_index:02d} "
                    f"context={index + 1:03d}/{input_ids.shape[0]:03d}",
                    flush=True,
                )
                tokens = sequence.unsqueeze(0).to(device)
                logits = model(input_ids=tokens, use_cache=False).logits.float()
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    tokens[:, 1:].reshape(-1),
                )
                loss.backward()
                del tokens, logits, loss
                torch.cuda.empty_cache()
        finally:
            handle.remove()
        all_counts.append(counts.cpu())
        all_sums.append(sums.cpu())
        all_sums_sq.append(sums_sq.cpu())
        all_taylor.append(taylor.cpu())
        del sums, sums_sq, taylor, counts
        torch.cuda.empty_cache()

    counts = torch.stack(all_counts)
    sums = torch.stack(all_sums)
    sums_sq = torch.stack(all_sums_sq)
    taylor = torch.stack(all_taylor)
    divisor = counts.clamp_min(1).unsqueeze(-1)
    return {
        "counts": counts.cpu(),
        "mean": (sums / divisor).cpu(),
        "variance": (sums_sq / divisor - (sums / divisor).square())
        .clamp_min(0.0)
        .cpu(),
        "taylor": (taylor / divisor).abs().cpu(),
    }


def _balanced_function_groups(
    scores: torch.Tensor,
    shared: torch.Tensor,
    group_size: int,
) -> list[torch.Tensor]:
    clusters, width = scores.shape
    shared_mask = torch.zeros(width, dtype=torch.bool)
    shared_mask[shared] = True
    candidates = (~shared_mask).nonzero(as_tuple=False).flatten()
    pairs = []
    for cluster in range(clusters):
        order = candidates[torch.argsort(scores[cluster, candidates], descending=True)]
        pairs.extend((float(scores[cluster, neuron]), cluster, int(neuron)) for neuron in order)
    pairs.sort(reverse=True)
    assigned = torch.zeros(width, dtype=torch.bool)
    assigned[shared] = True
    groups = [[] for _ in range(clusters)]
    for _score, cluster, neuron in pairs:
        if assigned[neuron] or len(groups[cluster]) >= group_size:
            continue
        groups[cluster].append(neuron)
        assigned[neuron] = True
        if all(len(group) == group_size for group in groups):
            break
    if not all(len(group) == group_size for group in groups):
        raise RuntimeError("Balanced FANG assignment did not fill every group")
    return [torch.tensor(group, dtype=torch.long) for group in groups]


def build_and_apply_fang_flap_plans(
    model,
    statistics: dict,
    ratio: float,
    clusters: int = 7,
    temperature: float = 9.0,
    targets: list[int] | None = None,
    apply: bool = True,
) -> list[dict]:
    layers = llama_layers(model)
    width = model.config.intermediate_size
    group_size = width // (clusters + 1)
    if targets is None:
        targets = [int(round(width * ratio)) for _ in layers]
    if len(targets) != len(layers):
        raise ValueError("FANG target count must match the number of layers")
    plans = []
    for layer_index, layer in enumerate(layers):
        target = int(targets[layer_index])
        print(f"F-FANG plan layer={layer_index:02d}", flush=True)
        scores = statistics["taylor"][layer_index]
        per_cluster_top = torch.topk(scores, group_size, dim=1).indices
        frequency = torch.zeros(width, dtype=torch.long)
        score_sum = torch.zeros(width)
        for cluster in range(clusters):
            frequency[per_cluster_top[cluster]] += 1
            score_sum[per_cluster_top[cluster]] += scores[
                cluster, per_cluster_top[cluster]
            ]
        shared_order = sorted(
            range(width),
            key=lambda neuron: (int(frequency[neuron]), float(score_sum[neuron])),
            reverse=True,
        )
        shared = torch.tensor(shared_order[:group_size], dtype=torch.long)
        groups = _balanced_function_groups(scores, shared, group_size)

        cluster_means = statistics["mean"][layer_index]
        center_distances = torch.cdist(cluster_means, cluster_means)
        relevance = torch.softmax(-center_distances / temperature, dim=1)
        variance = statistics["variance"][layer_index]
        down_norm_sq = (
            layer.mlp.down_proj.weight.detach().float().square().sum(dim=0).cpu()
        )
        quotas = [target // clusters for _ in range(clusters)]
        for index in range(target % clusters):
            quotas[index] += 1
        pruned_parts = []
        weighted_means = torch.zeros(width)
        for group_index, group in enumerate(groups):
            weighted_variance = relevance[group_index] @ variance[:, group]
            importance = weighted_variance * down_norm_sq[group]
            local = torch.topk(
                importance, quotas[group_index], largest=False
            ).indices
            selected = group[local]
            pruned_parts.append(selected)
            weighted_means[selected] = relevance[group_index] @ cluster_means[:, selected]
        pruned = torch.cat(pruned_parts).sort().values
        compensation = (
            layer.mlp.down_proj.weight.detach().float().cpu()[:, pruned]
            @ weighted_means[pruned]
        )
        plan = {
            "method": "fang_flap_paper_reproduction",
            "target": target,
            "direct": pruned.tolist(),
            "merges": [],
            "pruned": pruned.tolist(),
            "bias_compensation": compensation,
            "shared": shared.tolist(),
            "group_sizes": [len(group) for group in groups],
            "temperature": temperature,
        }
        if apply:
            apply_layer_plan(layer, plan)
        plans.append(plan)
    model.config.intermediate_size = layers[0].mlp.intermediate_size
    return plans


@torch.inference_mode()
def collect_block_functional_complexity(
    model,
    calibration_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Collect FANG Eq. 8 block input/output cosine complexity."""
    input_ids = load_calibration(calibration_path)["input_ids"]
    layers = llama_layers(model)
    cosine_sums = torch.zeros(len(layers), dtype=torch.float64)
    counts = torch.zeros(len(layers), dtype=torch.long)

    def make_hook(layer_index: int):
        def hook(_module, args, output):
            block_input = args[0][0].float()
            block_output = output[0][0].float()
            cosine = F.cosine_similarity(block_input, block_output, dim=-1)
            cosine_sums[layer_index] += cosine.sum().double().cpu()
            counts[layer_index] += cosine.numel()

        return hook

    handles = [layer.register_forward_hook(make_hook(index)) for index, layer in enumerate(layers)]
    try:
        for index, sequence in enumerate(input_ids):
            print(
                f"FANG ASA complexity context={index + 1:03d}/{input_ids.shape[0]:03d}",
                flush=True,
            )
            model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = int(input_ids.numel())
    if not torch.all(counts == expected):
        raise RuntimeError(f"FANG ASA counts {counts.tolist()} do not equal {expected}")
    return 1.0 - cosine_sums / counts.double()
