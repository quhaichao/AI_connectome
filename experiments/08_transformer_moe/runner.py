"""Fair training/evaluation harness for all registered MoE methods."""

from __future__ import annotations

import csv
import inspect
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import BenchmarkConfig, METHOD_SPECS
from .data import WikiTextBundle, evaluation_batches, infinite_batches
from .initialization import INITIALIZATION_API_VERSION, apply_initialization_policy
from .methods import (
    ActivationCache,
    METHODS_API_VERSION,
    clone_dense,
    collect_dense_activations,
    construct_method,
)
from .models import DenseTransformerLM, MoETransformerLM, parameter_report


def runtime_compatibility_report() -> Dict[str, object]:
    """Report whether the loaded v2 modules belong to one compatible code snapshot."""
    expected_config = {
        "tie_embeddings",
        "embedding_init_std",
        "xavier_initialize",
        "fc_consensus_splits",
        "fc_output_weight",
        "fc_uniform_shrinkage",
        "fc_refinement_swaps",
        "fc_hybrid_view_weights",
        "fc_oracle_tokens",
        "fc_router_ridge",
        "fc_router_key_prior",
        "router_calibration_temperature",
        "router_alignment_coef",
        "router_alignment_steps",
        "restore_best_validation",
    }
    signature = inspect.signature(MoETransformerLM.__init__)
    missing_config = sorted(expected_config - set(BenchmarkConfig.__dataclass_fields__))
    package_dir = Path(__file__).resolve().parent
    model_file = Path(inspect.getfile(MoETransformerLM)).resolve()
    method_file = Path(inspect.getfile(construct_method)).resolve()
    initialization_file = Path(inspect.getfile(apply_initialization_policy)).resolve()
    same_package = all(
        path.parent == package_dir for path in (model_file, method_file, initialization_file)
    )
    component_versions_match = (
        METHODS_API_VERSION == "2.4.0" and INITIALIZATION_API_VERSION == "2.4.0"
    )
    return {
        "compatible": not missing_config and same_package and component_versions_match,
        "runner_file": str(Path(__file__).resolve()),
        "models_file": str(model_file),
        "methods_file": str(method_file),
        "initialization_file": str(initialization_file),
        "moe_constructor": str(signature),
        "initialization_policy": "external_factory_v2",
        "methods_api_version": METHODS_API_VERSION,
        "initialization_api_version": INITIALIZATION_API_VERSION,
        "missing_config_fields": missing_config,
        "all_modules_from_same_package": same_package,
        "component_versions_match": component_versions_match,
    }


def assert_runtime_compatibility() -> Dict[str, object]:
    """Fail before training when Jupyter holds stale or partially synced v2 modules."""
    report = runtime_compatibility_report()
    if not report["compatible"]:
        raise RuntimeError(
            "Incompatible/stale MoE_benchmark_v2 modules are loaded. "
            "Synchronize the entire v2 folder and restart the Jupyter kernel, then run all cells. "
            f"Details: {json.dumps(report, ensure_ascii=False, indent=2)}"
        )
    return report


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def build_dense_model(vocab_size: int, cfg: BenchmarkConfig) -> DenseTransformerLM:
    model = DenseTransformerLM(
        vocab_size=vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    return apply_initialization_policy(
        model,
        tie_embeddings=cfg.tie_embeddings,
        embedding_init_std=cfg.embedding_init_std,
        xavier_initialize=cfg.xavier_initialize,
    )


def _ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _hoyer_loss(hidden_activations: List[torch.Tensor]) -> torch.Tensor:
    values = []
    for hidden in hidden_activations:
        flat = hidden.reshape(-1, hidden.shape[-1])
        numerator = flat.abs().sum(dim=-1).square()
        denominator = flat.square().sum(dim=-1).clamp_min(1e-12)
        values.append((numerator / denominator).mean())
    return torch.stack(values).mean()


@torch.no_grad()
def evaluate(
    model,
    dataset,
    cfg: BenchmarkConfig,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    layer_usage: Optional[List[torch.Tensor]] = None
    selected_sum = 0.0
    selected_tokens = 0
    dropped_sum = 0.0
    batches_seen = 0
    for input_ids, targets in evaluation_batches(
        dataset, cfg.batch_size, max_batches or cfg.eval_batches, device
    ):
        output = model(input_ids)
        nll = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        tokens = targets.numel()
        total_nll += float(nll)
        total_tokens += tokens
        batches_seen += 1
        if output.routing:
            if layer_usage is None:
                layer_usage = [torch.zeros_like(item["usage"], device="cpu") for item in output.routing]
            for layer, item in enumerate(output.routing):
                layer_usage[layer] += item["usage"].detach().cpu()
                selected_sum += float(item["selected_per_token"].sum())
                selected_tokens += item["selected_per_token"].numel()
                dropped_sum += float(item["dropped_fraction"])
    mean_nll = total_nll / max(1, total_tokens)
    result: Dict[str, object] = {
        "loss": mean_nll,
        "perplexity": math.exp(min(20.0, mean_nll)),
        "tokens": total_tokens,
    }
    if layer_usage is not None:
        result.update(
            {
                "layer_usage": [(value / batches_seen).tolist() for value in layer_usage],
                "mean_selected_experts": selected_sum / max(1, selected_tokens),
                "mean_dropped_fraction": dropped_sum / max(1, batches_seen * len(layer_usage)),
            }
        )
    return result


def _optimizer(model, cfg: BenchmarkConfig):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _set_progress(progress, description: str, **postfix: object) -> None:
    """Update an optional tqdm-like object without making tqdm a hard dependency."""
    if progress is None:
        return
    progress.set_description_str(description, refresh=False)
    if postfix:
        progress.set_postfix(postfix, refresh=False)


def _advance_progress(progress, **postfix: object) -> None:
    if progress is None:
        return
    if postfix:
        progress.set_postfix(postfix, refresh=False)
    progress.update(1)


def train_dense_warmup(
    model: DenseTransformerLM,
    stream: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    cfg: BenchmarkConfig,
    progress=None,
    progress_label: str = "dense warmup",
) -> List[Dict[str, float]]:
    model.train()
    optimizer = _optimizer(model, cfg)
    logs = []
    _set_progress(progress, progress_label)
    for step in range(1, cfg.dense_warmup_steps + 1):
        input_ids, targets = next(stream)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids)
        loss = _ce_loss(output.logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        _advance_progress(
            progress,
            phase_step=f"{step}/{cfg.dense_warmup_steps}",
            train_loss=f"{float(loss.detach()):.4f}",
        )
        if step == 1 or step == cfg.dense_warmup_steps:
            logs.append({"phase_step": step, "train_loss": float(loss.detach())})
    return logs


def d2d_sparsity_calibration(
    model: DenseTransformerLM,
    stream: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    cfg: BenchmarkConfig,
    progress=None,
    progress_label: str = "D2DMoE sparsity calibration",
) -> List[Dict[str, float]]:
    model.train()
    optimizer = _optimizer(model, cfg)
    logs = []
    _set_progress(progress, progress_label)
    for step in range(1, cfg.d2d_sparsity_steps + 1):
        input_ids, targets = next(stream)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, return_intermediates=True)
        task = _ce_loss(output.logits, targets)
        sparsity = _hoyer_loss(output.hidden_activations)
        alpha = cfg.d2d_hoyer_alpha * step / max(1, cfg.d2d_sparsity_steps)
        loss = task + alpha * sparsity
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        _advance_progress(
            progress,
            phase_step=f"{step}/{cfg.d2d_sparsity_steps}",
            train_loss=f"{float(task.detach()):.4f}",
        )
        if step == 1 or step == cfg.d2d_sparsity_steps:
            logs.append(
                {
                    "phase_step": step,
                    "train_loss": float(task.detach()),
                    "hoyer": float(sparsity.detach()),
                    "hoyer_alpha": alpha,
                }
            )
    return logs


def calibrate_routers(
    model: MoETransformerLM,
    dense_reference: DenseTransformerLM,
    stream: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    method_name: str,
    cfg: BenchmarkConfig,
    progress=None,
    progress_label: str = "router calibration",
) -> List[Dict[str, float]]:
    """Fit routers on a shared dense representation under a charged token budget."""
    spec = METHOD_SPECS[method_name]
    routers = [block.moe.router for block in model.blocks if block.moe.router is not None]
    parameters = [parameter for router in routers for parameter in router.parameters()]
    if not parameters or cfg.router_calibration_steps <= 0:
        return []
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=0.0)
    if spec.router_target == "none":
        raise ValueError(f"{method_name} enables calibration without a router target")
    logs = []
    model.eval()
    dense_reference.eval()
    _set_progress(progress, progress_label)
    for step in range(1, cfg.router_calibration_steps + 1):
        input_ids, _ = next(stream)
        with torch.no_grad():
            dense_output = dense_reference(input_ids, return_intermediates=True)
        losses = []
        for layer, block in enumerate(model.blocks):
            x = dense_output.pre_ffn[layer].detach().reshape(-1, model.d_model)
            with torch.no_grad():
                target = block.moe.expert_targets(x, spec.router_target)
                if spec.router_target == "positive_activation":
                    target = torch.log1p(target)
            prediction = block.moe.router_scores(x)
            if spec.router_objective == "soft_ce":
                temperature = cfg.router_calibration_temperature
                if spec.router_target == "reconstruction_score":
                    centered_target = target - target.mean(dim=-1, keepdim=True)
                    target_scale = centered_target.std(
                        dim=-1, unbiased=False, keepdim=True
                    ).clamp_min(1e-6)
                    target_logits = centered_target / target_scale
                else:
                    target_logits = torch.log1p(target)
                target_probability = F.softmax(target_logits / temperature, dim=-1)
                prediction_log_probability = F.log_softmax(prediction / temperature, dim=-1)
                losses.append(
                    F.kl_div(
                        prediction_log_probability,
                        target_probability,
                        reduction="batchmean",
                    )
                    * temperature**2
                )
            elif spec.router_objective == "mse":
                if spec.router_target == "positive_activation":
                    prediction = F.softplus(prediction)
                scale = target.mean(dim=0, keepdim=True).clamp_min(1e-6)
                losses.append(F.mse_loss(prediction / scale, target / scale))
            else:
                raise ValueError(
                    f"Unsupported router objective for {method_name}: {spec.router_objective}"
                )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
        optimizer.step()
        _advance_progress(
            progress,
            phase_step=f"{step}/{cfg.router_calibration_steps}",
            router_loss=f"{float(loss.detach()):.4f}",
        )
        if step == 1 or step == cfg.router_calibration_steps:
            logs.append({"phase_step": step, "router_loss": float(loss.detach())})
    return logs


def _joint_reconstruction_alignment_loss(
    model: MoETransformerLM,
    pre_ffn: List[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    losses = []
    for layer, block in enumerate(model.blocks):
        x = pre_ffn[layer].detach().reshape(-1, model.d_model)
        with torch.no_grad():
            target = block.moe.expert_targets(x, "reconstruction_score")
            centered = target - target.mean(dim=-1, keepdim=True)
            scale = centered.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
            target_probability = F.softmax(centered / scale / temperature, dim=-1)
        prediction_log_probability = F.log_softmax(
            block.moe.router_scores(x) / temperature, dim=-1
        )
        losses.append(
            F.kl_div(
                prediction_log_probability,
                target_probability,
                reduction="batchmean",
            )
            * temperature**2
        )
    return torch.stack(losses).mean()


@torch.no_grad()
def _ema_update(teacher: MoETransformerLM, student: MoETransformerLM, decay: float) -> None:
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


def train_moe(
    model: MoETransformerLM,
    stream: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    global_start_step: int,
    cfg: BenchmarkConfig,
    valid_dataset,
    device: torch.device,
    distillation: bool = False,
    joint_router_alignment: bool = False,
    aux_loss_coef: Optional[float] = None,
    progress=None,
    progress_label: str = "MoE training",
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    optimizer = _optimizer(model, cfg)
    teacher = None
    if distillation and cfg.cluster_aware_distill_coef > 0:
        teacher = clone_model(model).to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    logs: List[Dict[str, object]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_validation_loss = float("inf")
    best_global_step = -1
    effective_aux_coef = cfg.aux_loss_coef if aux_loss_coef is None else aux_loss_coef
    _set_progress(progress, progress_label)
    for local_step in range(1, steps + 1):
        model.train()
        input_ids, targets = next(stream)
        output = model(input_ids)
        task_loss = _ce_loss(output.logits, targets)
        loss = task_loss + effective_aux_coef * output.aux_loss
        router_alignment_loss = output.logits.new_zeros(())
        if (
            joint_router_alignment
            and cfg.router_alignment_coef > 0
            and local_step <= cfg.router_alignment_steps
        ):
            router_alignment_loss = _joint_reconstruction_alignment_loss(
                model, output.pre_ffn, cfg.router_calibration_temperature
            )
            loss = loss + cfg.router_alignment_coef * router_alignment_loss
        distill_loss = output.logits.new_zeros(())
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(input_ids, dense_routing=True).logits
            # Language-model adaptation of EESD: distil the full-expert EMA
            # ensemble distribution instead of CLIP features.
            student_log_prob = F.log_softmax(
                output.logits.float().reshape(-1, output.logits.shape[-1]), dim=-1
            )
            teacher_prob = F.softmax(
                teacher_logits.float().reshape(-1, teacher_logits.shape[-1]), dim=-1
            )
            distill_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
            loss = loss + cfg.cluster_aware_distill_coef * distill_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if teacher is not None:
            _ema_update(teacher, model, cfg.ema_decay)

        global_step = global_start_step + local_step
        progress_postfix: Dict[str, object] = {
            "method_step": f"{local_step}/{steps}",
            "train_loss": f"{float(task_loss.detach()):.4f}",
        }
        if local_step == 1 or local_step == steps or global_step % cfg.eval_interval == 0:
            _set_progress(
                progress,
                f"{progress_label} | validation at step {global_step}",
                **progress_postfix,
            )
            validation = evaluate(model, valid_dataset, cfg, device)
            if float(validation["loss"]) < best_validation_loss:
                best_validation_loss = float(validation["loss"])
                best_global_step = global_step
                if cfg.restore_best_validation:
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            logs.append(
                {
                    "global_step": global_step,
                    "train_loss": float(task_loss.detach()),
                    "aux_loss": float(output.aux_loss.detach()),
                    "router_alignment_loss": float(router_alignment_loss.detach()),
                    "distill_loss": float(distill_loss.detach()),
                    "valid_loss": validation["loss"],
                    "valid_perplexity": validation["perplexity"],
                    "mean_selected_experts": validation.get("mean_selected_experts"),
                    "mean_dropped_fraction": validation.get("mean_dropped_fraction"),
                }
            )
            progress_postfix["valid_ppl"] = f"{float(validation['perplexity']):.2f}"
            _set_progress(progress, progress_label, **progress_postfix)
        _advance_progress(progress, **progress_postfix)
    if cfg.restore_best_validation and best_state is not None:
        model.load_state_dict(best_state)
    return logs, {
        "best_validation_loss": best_validation_loss,
        "best_validation_step": float(best_global_step),
    }


def clone_model(model: MoETransformerLM) -> MoETransformerLM:
    import copy

    return copy.deepcopy(model)


def _stream_after(
    dataset,
    cfg: BenchmarkConfig,
    seed: int,
    device: torch.device,
    skip: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    stream = infinite_batches(dataset, cfg.batch_size, seed, device)
    for _ in range(skip):
        next(stream)
    return stream


def run_single_method(
    method_name: str,
    seed: int,
    dense_checkpoint: DenseTransformerLM,
    activation_cache: ActivationCache,
    bundle: WikiTextBundle,
    cfg: BenchmarkConfig,
    device: torch.device,
    progress=None,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object], MoETransformerLM]:
    spec = METHOD_SPECS[method_name]
    seed_everything(seed + 100_003, cfg.deterministic)
    start = time.perf_counter()
    stream = _stream_after(bundle.train, cfg, seed, device, 0)
    dense_for_method: Optional[DenseTransformerLM]
    phase_logs: Dict[str, object] = {}
    if spec.requires_dense:
        dense_for_method = clone_dense(dense_checkpoint).to(device)
        for _ in range(cfg.dense_warmup_steps):
            next(stream)
    else:
        dense_for_method = None

    consumed = cfg.dense_warmup_steps if spec.requires_dense else 0
    if spec.sparsity_calibration:
        phase_logs["sparsity_calibration"] = d2d_sparsity_calibration(
            dense_for_method,
            stream,
            cfg,
            progress=progress,
            progress_label=f"seed {seed} | {spec.label} | sparsity calibration",
        )
        consumed += cfg.d2d_sparsity_steps

    _set_progress(progress, f"seed {seed} | {spec.label} | constructing experts")
    model, construction = construct_method(
        method_name,
        dense_for_method,
        len(bundle.vocab.itos),
        cfg,
        activation_cache,
        seed,
    )
    model = model.to(device)
    if spec.router_calibration:
        if dense_for_method is None:
            raise ValueError(f"{method_name} needs a dense reference for router calibration")
        phase_logs["router_calibration"] = calibrate_routers(
            model,
            dense_for_method,
            stream,
            method_name,
            cfg,
            progress=progress,
            progress_label=f"seed {seed} | {spec.label} | router calibration",
        )
        consumed += cfg.router_calibration_steps

    remaining = cfg.total_steps - consumed
    logs, selection = train_moe(
        model,
        stream,
        remaining,
        consumed,
        cfg,
        bundle.valid,
        device,
        distillation=spec.distillation,
        joint_router_alignment=spec.joint_router_alignment,
        aux_loss_coef=(
            spec.aux_loss_coef_override
            if spec.aux_loss_coef_override >= 0
            else cfg.aux_loss_coef
        ),
        progress=progress,
        progress_label=f"seed {seed} | {spec.label} | LM training",
    )
    _set_progress(progress, f"seed {seed} | {spec.label} | final evaluation")
    valid = evaluate(model, bundle.valid, cfg, device, max_batches=10**9)
    test = evaluate(model, bundle.test, cfg, device, max_batches=10**9)
    report = parameter_report(model)
    mean_selected = float(test.get("mean_selected_experts", cfg.top_k))
    report["measured_active_parameters"] = (
        report["shared_parameters"]
        + report["router_parameters"]
        + (report["expert_parameters"] / cfg.num_experts) * mean_selected
    )
    report["measured_expert_flops_per_token"] = (
        2.0 * cfg.d_model * model.expert_hidden_size * mean_selected * cfg.n_layers
    )
    summary: Dict[str, object] = {
        "method": method_name,
        "label": spec.label,
        "regime": spec.regime,
        "construction_kind": spec.construction,
        "router_kind": spec.router,
        "router_target": spec.router_target,
        "router_objective": spec.router_objective,
        "router_initialization_kind": spec.router_initialization,
        "joint_router_alignment": spec.joint_router_alignment,
        "joint_router_alignment_steps": (
            min(cfg.router_alignment_steps, remaining)
            if spec.joint_router_alignment else 0
        ),
        "effective_aux_loss_coef": (
            spec.aux_loss_coef_override
            if spec.aux_loss_coef_override >= 0
            else cfg.aux_loss_coef
        ),
        "seed": seed,
        "train_tokens": cfg.total_steps * cfg.batch_size * cfg.seq_len,
        "gradient_steps_total": cfg.total_steps,
        "dense_warmup_steps": cfg.dense_warmup_steps if spec.requires_dense else 0,
        "method_specific_calibration_steps": consumed - (cfg.dense_warmup_steps if spec.requires_dense else 0),
        "moe_end_to_end_steps": remaining,
        "best_validation_step": int(selection["best_validation_step"]),
        "best_validation_loss_during_training": selection["best_validation_loss"],
        "valid_loss": valid["loss"],
        "valid_perplexity": valid["perplexity"],
        "test_loss": test["loss"],
        "test_perplexity": test["perplexity"],
        "mean_selected_experts": mean_selected,
        "mean_dropped_fraction": test.get("mean_dropped_fraction", 0.0),
        "elapsed_seconds": time.perf_counter() - start,
        **report,
    }
    construction["phase_logs"] = phase_logs
    construction["test_layer_usage"] = test.get("layer_usage")
    if method_name == "d2dmoe":
        threshold_curve = []
        original_thresholds = [block.moe.dynamic_threshold for block in model.blocks]
        for threshold in cfg.d2d_threshold_sweep:
            for block in model.blocks:
                block.moe.dynamic_threshold = threshold
            point = evaluate(model, bundle.test, cfg, device, max_batches=10**9)
            threshold_curve.append(
                {
                    "threshold": threshold,
                    "test_perplexity": point["perplexity"],
                    "mean_selected_experts": point.get("mean_selected_experts"),
                }
            )
        for block, threshold in zip(model.blocks, original_thresholds):
            block.moe.dynamic_threshold = threshold
        construction["d2d_threshold_curve"] = threshold_curve
    return summary, logs, construction, model


def run_benchmark(
    bundle: WikiTextBundle,
    cfg: BenchmarkConfig,
    save_models: bool = False,
    show_progress: bool = False,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    assert_runtime_compatibility()
    cfg.validate()
    device = resolve_device(cfg.device)
    output_dir = cfg.output_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summaries: List[Dict[str, object]] = []
    histories: List[Dict[str, object]] = []
    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise RuntimeError(
                "show_progress=True requires tqdm. Install it with `pip install tqdm`."
            ) from exc
        per_seed_steps = cfg.dense_warmup_steps
        for method_name in cfg.methods:
            spec = METHOD_SPECS[method_name]
            consumed = cfg.dense_warmup_steps if spec.requires_dense else 0
            calibration_steps = 0
            if spec.sparsity_calibration:
                consumed += cfg.d2d_sparsity_steps
                calibration_steps += cfg.d2d_sparsity_steps
            if spec.router_calibration:
                consumed += cfg.router_calibration_steps
                calibration_steps += cfg.router_calibration_steps
            per_seed_steps += calibration_steps + max(0, cfg.total_steps - consumed)
        progress = tqdm(
            total=len(cfg.seeds) * per_seed_steps,
            desc="MoE benchmark training",
            unit="step",
            dynamic_ncols=True,
        )
    try:
        for seed in cfg.seeds:
            seed_everything(seed, cfg.deterministic)
            dense = build_dense_model(len(bundle.vocab.itos), cfg).to(device)
            warmup_stream = infinite_batches(bundle.train, cfg.batch_size, seed, device)
            dense_logs = train_dense_warmup(
                dense,
                warmup_stream,
                cfg,
                progress=progress,
                progress_label=f"seed {seed} | shared dense warmup",
            )
            calibration_stream = infinite_batches(
                bundle.train, cfg.batch_size, seed + 50_000, device
            )
            _set_progress(progress, f"seed {seed} | collecting dense activations")
            cache = collect_dense_activations(
                dense,
                calibration_stream,
                cfg.construction_batches,
                seed=seed + 60_000,
            )
            for method_name in cfg.methods:
                summary, logs, construction, model = run_single_method(
                    method_name,
                    seed,
                    dense,
                    cache,
                    bundle,
                    cfg,
                    device,
                    progress=progress,
                )
                summaries.append(summary)
                histories.extend(
                    {"method": method_name, "seed": seed, **entry} for entry in logs
                )
                stem = f"{method_name}_seed{seed}"
                (output_dir / f"{stem}_construction.json").write_text(
                    json.dumps(construction, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                if save_models:
                    torch.save(model.state_dict(), output_dir / f"{stem}.pt")
                _write_results(output_dir, summaries, histories)
                _set_progress(
                    progress,
                    f"seed {seed} | completed {METHOD_SPECS[method_name].label}",
                    test_ppl=f"{float(summary['test_perplexity']):.2f}",
                )
            (output_dir / f"dense_seed{seed}_warmup.json").write_text(
                json.dumps(dense_logs, indent=2), encoding="utf-8"
            )
    finally:
        if progress is not None:
            progress.close()
    return summaries, histories


def _write_results(
    output_dir: Path,
    summaries: List[Dict[str, object]],
    histories: List[Dict[str, object]],
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(histories, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if summaries:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)


def load_results(output_dir: str):
    root = Path(output_dir)
    summaries = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    histories = json.loads((root / "history.json").read_text(encoding="utf-8"))
    return summaries, histories
