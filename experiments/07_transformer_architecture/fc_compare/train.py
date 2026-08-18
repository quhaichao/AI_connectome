from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Must be set before CUDA initializes to make cuBLAS GEMMs reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .analysis import summarize_group
from .audit import audit_built_pair, audit_spec
from .config import (
    ComparisonSpec,
    DataConfig,
    FCConfig,
    TrainConfig,
    as_serializable_dict,
)
from .data import eval_loader, fixed_probe, load_wikitext2, train_loader
from .fc import measure_model_fc
from .models import (
    build_model,
    copy_shared_initialization,
    hyper_connection_diagnostics,
    with_vocab,
)


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _optimizer(model, config: TrainConfig, device: torch.device):
    shared_decay, shared_no_decay = [], []
    hc_decay, hc_no_decay = [], []
    hc_readout = []
    for name, parameter in model.named_parameters():
        # HC paper recipe: global/static routing coefficients are not decayed,
        # while the input-dependent projection matrices are. Biases, gains and
        # normalization parameters follow the usual no-decay rule.
        is_hc_static = name.endswith("static_alpha") or name.endswith("static_beta")
        is_hc_parameter = ".attention_hc." in name or ".ffn_hc." in name
        is_hc_readout = name.startswith("stream_readout.")
        use_decay = parameter.ndim >= 2 and not is_hc_static
        destination = (
            hc_readout
            if is_hc_readout
            else hc_decay
            if is_hc_parameter and use_decay
            else hc_no_decay
            if is_hc_parameter
            else shared_decay
            if use_decay
            else shared_no_decay
        )
        destination.append(parameter)
    groups = []
    for parameters, weight_decay, lr, role in (
        (shared_decay, config.weight_decay, config.learning_rate, "shared_decay"),
        (shared_no_decay, 0.0, config.learning_rate, "shared_no_decay"),
        (
            hc_decay,
            config.weight_decay,
            config.learning_rate * config.hc_lr_multiplier,
            "hc_routing_decay",
        ),
        (
            hc_no_decay,
            0.0,
            config.learning_rate * config.hc_lr_multiplier,
            "hc_routing_no_decay",
        ),
        (
            hc_readout,
            0.0,
            config.learning_rate * config.hc_readout_lr_multiplier,
            "hc_adaptive_readout",
        ),
    ):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "weight_decay": weight_decay,
                    "lr": lr,
                    "role": role,
                }
            )
    kwargs = dict(
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )
    try:
        return torch.optim.AdamW(groups, fused=device.type == "cuda", **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, **kwargs)


def _scheduler(optimizer, config: TrainConfig):
    def multiplier(step: int):
        if step < config.warmup_steps:
            return max(step, 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            config.max_steps - config.warmup_steps, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.inference_mode()
def evaluate_ppl(model, loader, device: torch.device, use_bf16: bool):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for tokens, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, use_bf16):
            logits = model(tokens)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
        total_nll += float(nll)
        total_tokens += targets.numel()
    mean_nll = total_nll / total_tokens
    return {"nll": mean_nll, "ppl": math.exp(mean_nll), "tokens": total_tokens}


def _train_one_step(
    model,
    optimizer,
    scheduler,
    tokens,
    targets,
    device,
    config: TrainConfig,
    dropout_seed: int,
):
    torch.manual_seed(dropout_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(dropout_seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, config.use_bf16):
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), float(grad_norm)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_seed(
    spec: ComparisonSpec,
    datasets,
    data: DataConfig,
    train: TrainConfig,
    fc: FCConfig,
    seed: int,
    output_dir: Path,
    device: torch.device,
    seed_index: int = 1,
    total_seeds: int = 1,
):
    started_at = time.perf_counter()
    set_seed(seed, train.deterministic)
    baseline = build_model(spec.baseline, data.seq_len)
    set_seed(seed, train.deterministic)
    optimized = build_model(spec.optimized, data.seq_len)
    copied = copy_shared_initialization(baseline, optimized)
    built_audit = audit_built_pair(spec, baseline, optimized, copied)
    baseline.to(device)
    optimized.to(device)

    train_batches = train_loader(datasets["train"], data, seed)
    validation_batches = eval_loader(datasets["validation"], data)
    test_batches = eval_loader(datasets["test"], data)
    probe = fixed_probe(datasets["validation"], fc.probe_sequences)

    optimizers = {
        "baseline": _optimizer(baseline, train, device),
        "optimized": _optimizer(optimized, train, device),
    }
    schedulers = {
        name: _scheduler(optimizer, train) for name, optimizer in optimizers.items()
    }
    models = {"baseline": baseline, "optimized": optimized}
    metrics: list[dict] = []
    trajectories = {
        "validation_ppl": {"baseline": [], "optimized": []},
        "fc": {"baseline": [], "optimized": []},
        "raw_fc": {"baseline": [], "optimized": []},
    }
    best = {
        "baseline": {"ppl": math.inf, "step": None},
        "optimized": {"ppl": math.inf, "step": None},
    }
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(step: int, do_ppl: bool, do_fc: bool):
        for name, model in models.items():
            if do_ppl:
                result = evaluate_ppl(model, validation_batches, device, train.use_bf16)
                trajectories["validation_ppl"][name].append(
                    {"step": step, "value": result["ppl"]}
                )
                metrics.append(
                    {
                        "seed": seed,
                        "variant": name,
                        "step": step,
                        "split": "validation",
                        "metric": "ppl",
                        "value": result["ppl"],
                    }
                )
                if result["ppl"] < best[name]["ppl"]:
                    best[name] = {"ppl": result["ppl"], "step": step}
                    torch.save(model.state_dict(), seed_dir / f"best_{name}.pt")
            if do_fc:
                result = measure_model_fc(
                    model,
                    probe,
                    fc,
                    device=device,
                    batch_size=data.batch_size,
                    use_bf16=train.use_bf16,
                )
                trajectories["fc"][name].append(
                    {"step": step, "value": result["top_fc_mean"]}
                )
                trajectories["raw_fc"][name].append(
                    {"step": step, "value": result["raw_top_fc_mean"]}
                )
                metrics.append(
                    {
                        "seed": seed,
                        "variant": name,
                        "step": step,
                        "split": "fixed_validation_probe",
                        "metric": "top5_fc_mean_position_standardized"
                        if fc.position_standardize
                        else "top5_fc_mean_raw",
                        "value": result["top_fc_mean"],
                        "raw_value": result["raw_top_fc_mean"],
                        "tap_semantics": result["tap_semantics"],
                        "details": result["layers"],
                    }
                )

    checkpoint(0, do_ppl=True, do_fc=True)
    iterator = iter(train_batches)
    progress = tqdm(
        range(1, train.max_steps + 1),
        total=train.max_steps,
        desc=f"{spec.key} seed {seed} ({seed_index}/{total_seeds})",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not train.show_progress,
    )
    for step in progress:
        try:
            tokens, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_batches)
            tokens, targets = next(iterator)
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        common_dropout_seed = seed * 1_000_000 + step
        latest_losses = {}
        for name in ("baseline", "optimized"):
            loss, grad_norm = _train_one_step(
                models[name],
                optimizers[name],
                schedulers[name],
                tokens,
                targets,
                device,
                train,
                common_dropout_seed,
            )
            latest_losses[name] = loss
            metrics.append(
                {
                    "seed": seed,
                    "variant": name,
                    "step": step,
                    "split": "train",
                    "metric": "loss",
                    "value": loss,
                    "grad_norm": grad_norm,
                    "learning_rate": optimizers[name].param_groups[0]["lr"],
                }
            )
        if step % train.eval_interval == 0 or step == train.max_steps:
            checkpoint(step, do_ppl=True, do_fc=False)
        if step % train.fc_interval == 0 or step == train.max_steps:
            checkpoint(step, do_ppl=False, do_fc=True)
        if step == 1 or step % 5 == 0 or step == train.max_steps:
            progress.set_postfix(
                base=f"{latest_losses['baseline']:.3f}",
                opt=f"{latest_losses['optimized']:.3f}",
                refresh=False,
            )

    test_ppl = {}
    for name, model in models.items():
        state = torch.load(
            seed_dir / f"best_{name}.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(state)
        test_ppl[name] = evaluate_ppl(model, test_batches, device, train.use_bf16)["ppl"]

    with (seed_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "seed": seed,
        "best_validation": best,
        "test_ppl": test_ppl,
        "validation_ppl_trajectory": trajectories["validation_ppl"],
        "fc_trajectory": trajectories["fc"],
        "raw_fc_trajectory": trajectories["raw_fc"],
        "built_pair_audit": built_audit,
        "optimized_routing_diagnostics": hyper_connection_diagnostics(
            models["optimized"]
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    _write_json(seed_dir / "summary.json", summary)
    return summary


def run_comparison(
    spec: ComparisonSpec,
    data: DataConfig,
    train: TrainConfig,
    fc: FCConfig,
    output_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_audit = audit_spec(spec, data, train, fc)
    datasets, _, itos, data_metadata = load_wikitext2(data, output_dir / "cache")
    spec = replace(
        spec,
        baseline=with_vocab(spec.baseline, len(itos)),
        optimized=with_vocab(spec.optimized, len(itos)),
    )
    # vocab_size is a common, data-derived field and is not an experimental difference.
    design_audit = audit_spec(spec, data, train, fc)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    snapshot = {
        "spec": as_serializable_dict(spec),
        "data": as_serializable_dict(data),
        "train": as_serializable_dict(train),
        "fc": as_serializable_dict(fc),
        "design_audit": design_audit,
        "data_metadata": data_metadata,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    _write_json(output_dir / "experiment_config.json", snapshot)
    summaries = []
    for seed_index, seed in enumerate(train.seeds, start=1):
        summaries.append(
            run_seed(
                spec,
                datasets,
                data,
                train,
                fc,
                seed,
                output_dir,
                device,
                seed_index=seed_index,
                total_seeds=len(train.seeds),
            )
        )
    aggregate = summarize_group(summaries, train, fc)
    aggregate.update(
        {
            "group": spec.key,
            "baseline_label": spec.baseline_label,
            "optimized_label": spec.optimized_label,
            "hypothesis": spec.hypothesis,
        }
    )
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    from .plotting import plot_group

    figure_paths = plot_group(spec, summaries, aggregate, output_dir / "figure")
    return {
        "output_dir": str(output_dir),
        "aggregate": aggregate,
        "figure_paths": [str(path) for path in figure_paths],
    }
