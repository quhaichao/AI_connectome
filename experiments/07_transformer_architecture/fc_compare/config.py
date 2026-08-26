from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, Optional, Union


DEFAULT_DATA_DIR = Path(
    "/mnt/Data16T/Data/haichao/code/AI_connectom/story/"
    "story_part2_struc_func/Transformer/data/wikitext-2"
)


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    seq_len: int = 128
    batch_size: int = 32
    max_vocab_size: Optional[int] = None
    num_workers: int = 2


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 0  # Filled after the training-only vocabulary is built.
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 8
    dropout: float = 0.1
    position: Literal["sinusoidal", "rope"] = "rope"
    norm_position: Literal["pre", "post"] = "pre"
    residual: Literal["standard", "dhc"] = "standard"
    mixer_pattern: tuple[str, ...] = field(default_factory=tuple)
    rope_base: float = 10_000.0
    hc_streams: int = 4
    hc_dynamic_scale: float = 0.02
    hc_tanh: bool = True
    hc_adaptive_readout: bool = True
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 4
    mamba_headdim: int = 64

    def resolved_mixers(self) -> tuple[str, ...]:
        return self.mixer_pattern or ("mha",) * self.n_layers


@dataclass(frozen=True)
class FCConfig:
    # "all" averages the independently computed FC statistic across every
    # Transformer block. An integer selects one zero-based block index.
    layer_selection: Union[Literal["all"], int] = "all"
    top_fraction: float = 0.05
    probe_sequences: int = 256
    position_standardize: bool = True
    min_feature_std: float = 1e-6
    onset_delta: float = 0.03
    onset_sustain_checkpoints: int = 3
    early_fraction: float = 0.25


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 500
    eval_interval: int = 25
    fc_interval: int = 10
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    hc_lr_multiplier: float = 1.0
    hc_readout_lr_multiplier: float = 8.0
    use_bf16: bool = True
    seeds: tuple[int, ...] = (11,)
    bootstrap_samples: int = 10_000
    confidence: float = 0.95
    deterministic: bool = True
    show_progress: bool = True


# @dataclass(frozen=True)
# class TrainConfig:
#     max_steps: int = 2000
#     eval_interval: int = 100
#     fc_interval: int = 50
#     learning_rate: float = 3e-4
#     min_lr_ratio: float = 0.1
#     warmup_steps: int = 200
#     weight_decay: float = 0.1
#     beta1: float = 0.9
#     beta2: float = 0.95
#     adam_eps: float = 1e-8
#     grad_clip: float = 1.0
#     hc_lr_multiplier: float = 1.0
#     hc_readout_lr_multiplier: float = 8.0
#     use_bf16: bool = True
#     seeds: tuple[int, ...] = (11,)
#     bootstrap_samples: int = 10_000
#     confidence: float = 0.95
#     deterministic: bool = True
#     show_progress: bool = True


@dataclass(frozen=True)
class ComparisonSpec:
    key: Literal["a1", "a2", "a3", "a4"]
    title: str
    baseline_label: str
    optimized_label: str
    baseline: ModelConfig
    optimized: ModelConfig
    allowed_model_differences: tuple[str, ...]
    hypothesis: str


def paper_config(group: str, data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
    """Return the preregistered configuration for one Figure 6a comparison.

    Groups are intentionally independent and receive separately frozen training
    recipes. Within a group, shared Transformer tensors and data exposure remain
    controlled. In a2, adaptive stream readout is an intrinsic part of the
    explicitly labelled DHC recipe rather than a claim of isolated module causality.
    """
    group = group.lower()
    data = DataConfig(data_dir=Path(data_dir))
    train = TrainConfig()
    fc = FCConfig()

    common = ModelConfig()
    if group == "a1":
        train = replace(
            train,
            learning_rate=1e-4,
            warmup_steps=40,
            weight_decay=0.08,
            min_lr_ratio=0.10,
            seeds=(11, 22, 33, 44, 55),
        )
        spec = ComparisonSpec(
            key="a1",
            title="Positional encoding",
            baseline_label="Sinusoidal PE",
            optimized_label="RoPE",
            baseline=replace(common, position="sinusoidal"),
            optimized=replace(common, position="rope"),
            allowed_model_differences=("position",),
            hypothesis="RoPE has lower held-out PPL and an earlier rise in top-5% FC.",
        )
    elif group == "a2":
        # Increase depth while narrowing so residual routing acts more often
        # without materially increasing width-squared compute:
        # 24*176^2 is approximately 20*192^2. The smaller D also reduces FC cost.
        deeper = replace(
            common,
            # d_model=176,
            # n_heads=8,
            # d_ff=704,
            n_layers=6,
        )
        spec = ComparisonSpec(
            key="a2",
            title="Residual-optimized recipe",
            baseline_label="Residual",
            optimized_label="DHC×4 + adaptive readout",
            baseline=replace(deeper, residual="standard"),
            optimized=replace(deeper, residual="dhc", dropout=0.05),
            allowed_model_differences=("residual", "dropout"),
            hypothesis=(
                "Dynamic Hyper-Connections with four residual streams have lower "
                "held-out PPL and an earlier rise in top-5% FC."
            ),
        )
        # The preceding seed sets informed the short-horizon development. The
        # revised PPL-only recipe is frozen and confirmed on untouched seeds.
        train = replace(
            train,
            learning_rate=7e-5,
            warmup_steps=30,
            weight_decay=0.05,
            min_lr_ratio=0.15,
            hc_lr_multiplier=6.0,
            hc_readout_lr_multiplier=8.0,
            grad_clip=0.5,
            seeds=(11, 22, 33, 44, 55),
        )
    elif group == "a3":
        train = replace(
            train,
            learning_rate=5e-5,
            warmup_steps=30,
            weight_decay=0.05,
            min_lr_ratio=0.10,
            seeds=(11, 22, 33, 44, 55),
        )
        # Keep depth and FFN blocks fixed. Only the sequence mixer changes in the
        # even-numbered blocks; the official mamba_ssm.Mamba2 implementation is used.
        all_mha = ("mha",) * common.n_layers
        # hybrid = tuple("mha" if i % 2 == 0 else "mamba2" for i in range(common.n_layers))
        hybrid = (
                    "mamba2", "mamba2", "mamba2", "mha",
                    "mamba2", "mamba2", "mamba2", "mha",
                )
        spec = ComparisonSpec(
            key="a3",
            title="Sequence mixer",
            baseline_label="MHA",
            optimized_label="Hybrid MHA + Mamba-2",
            baseline=replace(common, mixer_pattern=all_mha),
            optimized=replace(common, mixer_pattern=hybrid),
            allowed_model_differences=("mixer_pattern",),
            hypothesis="The hybrid mixer has lower held-out PPL and an earlier rise in top-5% FC.",
        )
    elif group == "a4":
        train = replace(
            train,
            learning_rate=8e-5,
            warmup_steps=100,
            weight_decay=0.05,
            min_lr_ratio=0.20,
            seeds=(11, 22, 33, 44, 55),

        )
        deeper = replace(common, n_layers=6)
        spec = ComparisonSpec(
            key="a4",
            title="Normalization placement",
            baseline_label="Post-Norm",
            optimized_label="Pre-Norm",
            baseline=replace(deeper, norm_position="post"),
            optimized=replace(deeper, norm_position="pre"),
            allowed_model_differences=("norm_position",),
            hypothesis="Pre-Norm has lower held-out PPL and an earlier rise in top-5% FC.",
        )
    else:
        raise ValueError(f"Unknown group {group!r}; expected a1, a2, a3, or a4")
    return spec, data, train, fc


def smoke_config(group: str, data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
    """Small configuration for pipeline checks; never use it for manuscript claims."""
    spec, data, train, fc = paper_config(group, data_dir)
    data = replace(data, seq_len=32, batch_size=4, num_workers=0, max_vocab_size=1000)
    train = replace(
        train,
        max_steps=4,
        eval_interval=2,
        fc_interval=2,
        warmup_steps=1,
        seeds=(7,),
        bootstrap_samples=200,
        use_bf16=False,
    )
    fc = replace(fc, probe_sequences=8)
    spec = replace(
        spec,
        baseline=replace(spec.baseline, d_model=32, n_heads=4, d_ff=64, n_layers=2),
        optimized=replace(spec.optimized, d_model=32, n_heads=4, d_ff=64, n_layers=2),
    )
    if group == "a3":
        spec = replace(
            spec,
            baseline=replace(spec.baseline, mixer_pattern=("mha", "mha"), mamba_headdim=16),
            optimized=replace(spec.optimized, mixer_pattern=("mha", "mamba2"), mamba_headdim=16),
        )
    return spec, data, train, fc


def as_serializable_dict(value):
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return convert(asdict(value))
