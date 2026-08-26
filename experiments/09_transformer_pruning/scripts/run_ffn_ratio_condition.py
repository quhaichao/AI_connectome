#!/usr/bin/env python
"""Run one model/seed condition of the C4 20/30/40/50% FFN matrix."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch

from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.fang_reproduction import build_and_apply_fang_flap_plans
from fc_pruning.fixed_mask_reconstruction import apply_fixed_mask_reconstruction
from fc_pruning.gram_similarity import GramPlanConfig, build_layer_candidate_pool, plan_from_candidate_pool
from fc_pruning.modeling import (
    MODEL_KEYS,
    decoder_layers,
    load_model,
    load_tokenizer,
    resolve_model_path,
)
from fc_pruning.pruning import apply_layer_plan
from fc_pruning.progress import report_progress
from fc_pruning.ratio_matrix import (
    METHODS,
    RATIOS,
    as_full_statistics,
    collect_fang_statistics_fast,
    collect_ratio_statistics,
    load_ratio_statistics,
    save_ratio_statistics,
)
from fc_pruning.slimllm_reproduction import apply_slimllm_plan, build_slimllm_plans
from fc_pruning.sobp_reproduction import apply_sobp_layer

def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    _write_json(path.with_suffix(".json"), rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _key(method: str, ratio: float) -> tuple[str, int]:
    return method, int(round(ratio * 1_000_000))


def _simple_plan(method: str, layer, statistics: dict, ratio: float) -> dict:
    count = int(statistics["count"])
    second = statistics["sum_sq"].float() / count
    mean = statistics["sum"].float() / count
    down = layer.mlp.down_proj.weight.detach().float().cpu()
    if method == "wanda":
        scores = down.abs().mean(dim=0) * second.clamp_min(0).sqrt()
    elif method == "flap":
        variance = (second - mean.square()).clamp_min(0.0)
        scores = variance * down.square().sum(dim=0)
    else:
        raise ValueError(method)
    target = int(round(layer.mlp.gate_proj.out_features * ratio))
    pruned = torch.topk(scores, target, largest=False).indices.sort().values
    plan = {
        "method": method,
        "target": target,
        "direct": pruned.tolist(),
        "merges": [],
        "pruned": pruned.tolist(),
    }
    if method == "flap":
        plan["bias_compensation"] = down[:, pruned] @ mean[pruned]
    return plan


def _build_fc_pools(model, statistics: dict, device: torch.device) -> list[dict]:
    config = GramPlanConfig(
        topk=32,
        merge_fraction=0.15,
        keeper_capacity=1,
        minimum_similarity=0.0,
        minimum_output_gain=0.05,
        protect_fraction=0.05,
        ridge_relative=1e-4,
        compensate_merge_mean=True,
    )
    pools = []
    protocol = statistics["protocol"]
    sequence_length = int(protocol["sequence_length"])
    fit_count = int(protocol["fit_contexts"]) * sequence_length
    holdout_count = int(protocol["holdout_contexts"]) * sequence_length
    layers = decoder_layers(model)
    started_at = time.monotonic()
    for index, layer in enumerate(layers):
        item = statistics["layers"][index]
        pool = build_layer_candidate_pool(
            layer,
            item["fit_gram"].to(device),
            item["holdout_gram"].to(device),
            item["fit_sum"].to(device),
            item["holdout_sum"].to(device),
            item["sum"].to(device),
            fit_count,
            holdout_count,
            item["count"],
            32,
            512,
            "fc",
        )
        pool["_config"] = config
        pools.append(pool)
        torch.cuda.empty_cache()
        report_progress(
            "Pearson candidate pools",
            index + 1,
            len(layers),
            started_at,
            detail=f"layer={index:02d}",
        )
    return pools


def _build_plans(
    model,
    statistics: dict,
    fang_statistics: dict | None,
    methods: list[str],
    ratios: list[float],
    device: torch.device,
    result_dir: Path,
) -> None:
    full, covariance, hessians = as_full_statistics(statistics)
    fc_pools = _build_fc_pools(model, statistics, device) if "fc_ls" in methods else None
    for method in methods:
        if method == "fc_ls":
            started_at = time.monotonic()
            completed = 0
            total = len(ratios) * len(decoder_layers(model))
            for ratio in ratios:
                plans = []
                for layer_index, (layer, pool) in enumerate(
                    zip(decoder_layers(model), fc_pools)
                ):
                    plan, _audit = plan_from_candidate_pool(
                        layer, pool, ratio, pool["_config"]
                    )
                    plans.append(plan)
                    completed += 1
                    report_progress(
                        "FC-LS plans",
                        completed,
                        total,
                        started_at,
                        detail=f"ratio={ratio:.0%} layer={layer_index:02d}",
                    )
                _write_json(result_dir / f"plan_{method}_r{ratio:.3f}.json", plans)
        elif method in {"flap", "wanda"}:
            for ratio in ratios:
                plans = [
                    _simple_plan(method, layer, item, ratio)
                    for layer, item in zip(decoder_layers(model), full["layers"])
                ]
                _write_json(result_dir / f"plan_{method}_r{ratio:.3f}.json", plans)
        elif method == "fang":
            if fang_statistics is None:
                raise RuntimeError("FANG statistics are required")
            for ratio in ratios:
                plans = build_and_apply_fang_flap_plans(
                    model,
                    fang_statistics,
                    ratio,
                    clusters=7,
                    temperature=9.0,
                    apply=False,
                )
                _write_json(result_dir / f"plan_{method}_r{ratio:.3f}.json", plans)
        elif method == "slimllm":
            for ratio in ratios:
                slim_plans = build_slimllm_plans(
                    model,
                    full,
                    covariance,
                    hessians,
                    ratio,
                    device,
                    modes=("official_piecewise",),
                )
                _write_json(
                    result_dir / f"plan_{method}_r{ratio:.3f}.json",
                    slim_plans["official_piecewise"],
                )
        elif method == "sobp":
            # SoBP reconstruction depends on the surviving block, so masks and
            # reconstructed weights are applied during evaluation below.
            continue
        else:
            raise ValueError(method)
    del fc_pools
    torch.cuda.empty_cache()


def _evaluate_one(
    model,
    method: str,
    ratio: float,
    plans: list[dict] | None,
    statistics: dict,
    device: torch.device,
    tokenizer,
    test_path: str,
    config: dict,
) -> dict:
    if method == "fc_ls":
        for layer, plan, item in zip(decoder_layers(model), plans, statistics["layers"]):
            hessian = item["fit_gram"] + item["holdout_gram"]
            apply_fixed_mask_reconstruction(
                layer,
                hessian,
                item["sum"].double() / int(item["count"]),
                plan,
                "direct_ls",
                device,
                damping=1e-4,
                covariance_mode="second_moment",
                sample_count=int(item["count"]),
            )
            del hessian
    elif method == "sobp":
        target = int(round(model.config.intermediate_size * ratio))
        for layer, item in zip(decoder_layers(model), statistics["layers"]):
            hessian = item["fit_gram"] + item["holdout_gram"]
            apply_sobp_layer(layer, hessian, target, device)
            del hessian
    else:
        for layer, plan in zip(decoder_layers(model), plans):
            if method == "slimllm":
                apply_slimllm_plan(layer, plan, use_regression=True)
            else:
                apply_layer_plan(layer, plan)
        model.config.intermediate_size = decoder_layers(model)[0].mlp.intermediate_size
    if method in {"fc_ls", "sobp"}:
        model.config.intermediate_size = decoder_layers(model)[0].mlp.intermediate_size
    evaluation = evaluate_wikitext_ppl(model, tokenizer, test_path, config, device)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--domain", choices=("c4",), default="c4")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--results-root", default="results/ffn_ratio_matrix_pearson")
    parser.add_argument("--cache-dir", default="/tmp/ffn_ratio_matrix_pearson")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    root = Path(__file__).resolve().parents[1]
    model_path = resolve_model_path(root, args.model)
    result_dir = root / args.results_root / args.model / args.domain / f"seed{args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = (root / args.calibration).resolve()
    test_path = str((root / "data/wikitext2_raw/test-00000-of-00001.parquet").resolve())
    protocol_path = result_dir / "protocol.json"
    existing_protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else {}
    )
    existing_fc_plans = list(result_dir.glob("plan_fc_ls_r*.json"))
    existing_fc_plans_are_signed = all(
        all(
            plan.get("similarity_metric") == "signed_pearson"
            for plan in json.loads(path.read_text(encoding="utf-8"))
        )
        for path in existing_fc_plans
    )
    if (
        "fc_ls" in args.methods
        and existing_fc_plans
        and existing_protocol.get("fc_similarity") != "signed_pearson"
        and not existing_fc_plans_are_signed
    ):
        raise RuntimeError(
            "Existing FC-LS plans were not marked as signed Pearson. "
            "Use the default results/ffn_ratio_matrix_pearson directory or "
            "choose a new --results-root; old plans cannot be resumed."
        )

    def record_protocol() -> None:
        previous = existing_protocol
        seen_methods = set(previous.get("methods", [])) | set(args.methods)
        seen_ratios = {float(value) for value in previous.get("ratios", [])} | set(args.ratios)
        _write_json(protocol_path, {
            "model": args.model,
            "domain": args.domain,
            "seed": args.seed,
            "calibration": str(calibration_path),
            "calibration_tokens": 262144,
            "positions_per_context": 2048,
            "fc_similarity": "signed_pearson",
            "methods": [method for method in METHODS if method in seen_methods],
            "ratios": sorted(seen_ratios),
            "test_path": test_path,
            "test_used_for_calibration": False,
        })

    summary_path = result_dir / "summary.csv"
    rows = []
    if args.resume and summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {_key(row["method"], float(row["ratio"])) for row in rows}
    wanted = {_key(method, ratio) for method in args.methods for ratio in args.ratios}
    if wanted <= completed:
        record_protocol()
        print(f"Condition complete: {result_dir}", flush=True)
        return

    cache_dir = Path(args.cache_dir) / args.model / args.domain / f"seed{args.seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / "ratio_statistics.pt"
    fang_path = cache_dir / "fang_statistics.pt"
    model = load_model(str(model_path), device)
    if stats_path.exists():
        print(f"Loading ratio statistics {stats_path}", flush=True)
        statistics = load_ratio_statistics(stats_path)
    else:
        print(f"Collecting ratio statistics {args.model}/{args.domain}/seed{args.seed}", flush=True)
        statistics = collect_ratio_statistics(model, str(calibration_path), device)
        save_ratio_statistics(statistics, stats_path)
    fang_statistics = None
    if "fang" in args.methods:
        if fang_path.exists():
            fang_statistics = load_ratio_statistics(fang_path)
        else:
            print("Collecting FANG statistics", flush=True)
            fang_statistics = collect_fang_statistics_fast(
                model, str(calibration_path), statistics, device, seed=args.seed
            )
            save_ratio_statistics(fang_statistics, fang_path)

    plan_complete = all(
        (result_dir / f"plan_{method}_r{ratio:.3f}.json").exists()
        for method in args.methods
        for ratio in args.ratios
        if method != "sobp"
    )
    if not plan_complete:
        print("Building plans", flush=True)
        _build_plans(model, statistics, fang_statistics, args.methods, args.ratios, device, result_dir)
    del fang_statistics
    torch.cuda.empty_cache()
    base_model = model
    tokenizer = load_tokenizer(str(model_path))
    config = {
        "evaluation_length": 2048,
        "evaluation_max_windows": None,
        "evaluation_batch_size": 1,
    }
    for ratio in args.ratios:
        for method in args.methods:
            if _key(method, ratio) in completed:
                continue
            print(f"Evaluate {args.model}/{args.domain}/seed{args.seed} {method} r={ratio:.2f}", flush=True)
            model = copy.deepcopy(base_model)
            plans = None
            if method != "sobp":
                plans = json.loads(
                    (result_dir / f"plan_{method}_r{ratio:.3f}.json").read_text(encoding="utf-8")
                )
            evaluation = _evaluate_one(
                model, method, ratio, plans, statistics, device, tokenizer, test_path, config
            )
            row = {
                "model": args.model,
                "domain": args.domain,
                "seed": args.seed,
                "method": method,
                "ratio": ratio,
                "calibration_tokens": 262144,
                "calibration_positions": "all_128x2048",
                **evaluation,
            }
            rows.append(row)
            completed.add(_key(method, ratio))
            _write_rows(summary_path, rows)
            print(json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
    del base_model
    record_protocol()
    if not args.keep_cache:
        for path in (stats_path, fang_path):
            if path.exists():
                path.unlink()
        try:
            cache_dir.rmdir()
        except OSError:
            pass
    print(f"Finished {result_dir}", flush=True)


if __name__ == "__main__":
    main()
