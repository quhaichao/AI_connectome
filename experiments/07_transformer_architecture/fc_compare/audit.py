from __future__ import annotations

from dataclasses import asdict

from .config import ComparisonSpec, DataConfig, FCConfig, TrainConfig


def audit_spec(
    spec: ComparisonSpec,
    data: DataConfig,
    train: TrainConfig,
    fc: FCConfig,
):
    """Record the configured comparison without accepting or rejecting it."""
    baseline = asdict(spec.baseline)
    optimized = asdict(spec.optimized)
    data_record = asdict(data)
    data_record["data_dir"] = str(data_record["data_dir"])
    differences = {
        key: {"baseline": baseline[key], "optimized": optimized[key]}
        for key in baseline
        if baseline[key] != optimized[key]
    }
    return {
        "group": spec.key,
        "declared_model_differences": list(spec.allowed_model_differences),
        "observed_model_differences": differences,
        "training_config": asdict(train),
        "data_config": data_record,
        "fc_config": asdict(fc),
    }


def audit_built_pair(
    spec: ComparisonSpec,
    baseline,
    optimized,
    copied_names: list[str],
):
    """Record construction metadata; never gate training or plotting."""
    return {
        "shared_tensors_copied": len(copied_names),
    }
