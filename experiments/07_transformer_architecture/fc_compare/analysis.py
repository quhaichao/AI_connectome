from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from .config import FCConfig, TrainConfig


def bootstrap_ci(values, samples: int, confidence: float, seed: int = 20260725):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [math.nan, math.nan]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    statistics = np.median(draws, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(np.quantile(statistics, alpha)), float(np.quantile(statistics, 1 - alpha))]


def fc_onset(trajectory: list[dict], config: FCConfig):
    steps = np.asarray([row["step"] for row in trajectory], dtype=np.float64)
    values = np.asarray([row["value"] for row in trajectory], dtype=np.float64)
    order = np.argsort(steps)
    steps, values = steps[order], values[order]
    initial = values[0]
    delta = values - initial
    interval = np.median(np.diff(steps)) if len(steps) > 1 else 1.0
    censor_step = float(steps[-1] + interval)
    threshold = initial + config.onset_delta
    if not np.isfinite(delta).all() or np.nanmax(delta) < config.onset_delta:
        return {
            "onset_step": censor_step,
            "censored": True,
            "initial": float(initial),
            "delta_threshold": float(config.onset_delta),
            "threshold": float(threshold),
            "max_delta": float(np.nanmax(delta)),
        }
    sustain = config.onset_sustain_checkpoints
    above = values >= threshold
    for index in range(0, len(values) - sustain + 1):
        if bool(np.all(above[index : index + sustain])):
            return {
                "onset_step": float(steps[index]),
                "censored": False,
                "initial": float(initial),
                "delta_threshold": float(config.onset_delta),
                "threshold": float(threshold),
                "max_delta": float(np.nanmax(delta)),
            }
    return {
        "onset_step": censor_step,
        "censored": True,
        "initial": float(initial),
        "delta_threshold": float(config.onset_delta),
        "threshold": float(threshold),
        "max_delta": float(np.nanmax(delta)),
    }


def early_auc(trajectory: list[dict], config: FCConfig):
    steps = np.asarray([row["step"] for row in trajectory], dtype=np.float64)
    values = np.asarray([row["value"] for row in trajectory], dtype=np.float64)
    cutoff = steps[0] + config.early_fraction * (steps[-1] - steps[0])
    if cutoff <= steps[0] or steps[-1] <= steps[0]:
        return math.nan
    keep = steps < cutoff
    x = np.concatenate([steps[keep], [cutoff]])
    y_values = np.concatenate([values[keep], [np.interp(cutoff, steps, values)]])
    y = np.maximum(y_values - values[0], 0.0)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x) / max(x[-1] - x[0], 1.0))


def summarize_group(seed_summaries: list[dict], train: TrainConfig, fc: FCConfig):
    log_ppl_differences = []
    onset_differences = []
    early_auc_differences = []
    onset_rows = []
    for summary in seed_summaries:
        base_ppl = summary["test_ppl"]["baseline"]
        opt_ppl = summary["test_ppl"]["optimized"]
        log_ppl_differences.append(math.log(opt_ppl) - math.log(base_ppl))
        base_onset = fc_onset(summary["fc_trajectory"]["baseline"], fc)
        opt_onset = fc_onset(summary["fc_trajectory"]["optimized"], fc)
        onset_differences.append(opt_onset["onset_step"] - base_onset["onset_step"])
        early_auc_differences.append(
            early_auc(summary["fc_trajectory"]["optimized"], fc)
            - early_auc(summary["fc_trajectory"]["baseline"], fc)
        )
        onset_rows.append(
            {"seed": summary["seed"], "baseline": base_onset, "optimized": opt_onset}
        )

    ppl_ci = bootstrap_ci(
        log_ppl_differences, train.bootstrap_samples, train.confidence, seed=1001
    )
    ppl_percent_changes = 100.0 * (np.exp(log_ppl_differences) - 1.0)
    ppl_percent_ci = bootstrap_ci(
        ppl_percent_changes, train.bootstrap_samples, train.confidence, seed=1004
    )
    onset_ci = bootstrap_ci(
        onset_differences, train.bootstrap_samples, train.confidence, seed=1002
    )
    auc_ci = bootstrap_ci(
        early_auc_differences, train.bootstrap_samples, train.confidence, seed=1003
    )
    return {
        "n_seeds": len(seed_summaries),
        "ppl_comparison": {
            "estimand": "paired log(test PPL optimized / baseline), best validation checkpoint",
            "paired_values": log_ppl_differences,
            "median": float(np.median(log_ppl_differences)),
            "bootstrap_ci": ppl_ci,
            "paired_percent_changes": ppl_percent_changes.tolist(),
            "median_percent_change": float(np.median(ppl_percent_changes)),
            "percent_change_bootstrap_ci": ppl_percent_ci,
        },
        "fc_timing": {
            "estimand": "optimized minus baseline onset step; negative means earlier",
            "paired_values": onset_differences,
            "median": float(np.median(onset_differences)),
            "bootstrap_ci": onset_ci,
            "onsets": onset_rows,
        },
        "early_fc_auc_comparison": {
            "estimand": "optimized minus baseline early positive baseline-adjusted AUC",
            "paired_values": early_auc_differences,
            "median": float(np.nanmedian(early_auc_differences)),
            "bootstrap_ci": auc_ci,
        },
        "fc_config": asdict(fc),
    }
