"""Self-contained v2 configuration and method registry for the WikiText-2 benchmark.

The registry separates parameter-partitioned conversions from full-width
upcycling methods.  They answer different efficiency questions and should not
be ranked in a single parameter-matched table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    regime: str
    construction: str
    router: str
    expert_width: str
    requires_dense: bool = True
    router_calibration: bool = False
    router_target: str = "none"
    router_objective: str = "mse"
    router_initialization: str = "default"
    joint_router_alignment: bool = False
    aux_loss_coef_override: float = -1.0
    sparsity_calibration: bool = False
    distillation: bool = False
    notes: str = ""


METHOD_SPECS: Dict[str, MethodSpec] = {
    "fc_guided": MethodSpec(
        "fc_guided", "FC-Hybrid (ours)", "partitioned_iso_total", "fc_hybrid",
        "switch", "partitioned", router_calibration=False,
        router_target="reconstruction_score", router_objective="soft_ce",
        router_initialization="oracle_ridge", joint_router_alignment=True,
        aux_loss_coef_override=0.01,
        notes=("Calibration-selected blend of consensus FC, co-activation and key geometry, "
               "followed by oracle-reconstruction ridge router initialization."),
    ),
    "fc_guided_abs": MethodSpec(
        "fc_guided_abs", "FC-guided abs-FC (ablation)", "partitioned_iso_total", "fc",
        "switch", "partitioned", router_calibration=False,
        router_target="reconstruction_score", router_objective="soft_ce",
        router_initialization="oracle_ridge", joint_router_alignment=True,
        aux_loss_coef_override=0.01,
        notes="Original absolute-Pearson FC partition with matched oracle-ridge router initialization.",
    ),
    "random_calibrated": MethodSpec(
        "random_calibrated", "Random + matched ridge", "partitioned_iso_total", "random",
        "switch", "partitioned", router_calibration=False,
        router_target="reconstruction_score", router_objective="soft_ce",
        router_initialization="oracle_ridge", joint_router_alignment=True,
        aux_loss_coef_override=0.01,
        notes="Random balanced partition with exactly the same oracle-ridge router initialization as FC-Hybrid.",
    ),
    "clustering_calibrated": MethodSpec(
        "clustering_calibrated", "Clustering + matched ridge",
        "partitioned_iso_total", "key_clustering", "switch", "partitioned",
        router_calibration=False, router_target="reconstruction_score", router_objective="soft_ce",
        router_initialization="oracle_ridge", joint_router_alignment=True,
        aux_loss_coef_override=0.01,
        notes="LLaMA-style key clustering with the same oracle-ridge router initialization as FC-Hybrid.",
    ),
    "moefication": MethodSpec(
        "moefication", "MoEfication", "partitioned_iso_total", "coactivation",
        "moefication_mlp", "partitioned", router_calibration=True,
        router_target="positive_activation", router_objective="mse",
        notes="Co-activation graph partition plus independently calibrated MLP selector.",
    ),
    "emoe": MethodSpec(
        "emoe", "EMoE", "partitioned_iso_total", "key_clustering",
        "avg_key", "partitioned",
        notes="Constrained key clustering; gate is tied to expert mean keys.",
    ),
    "d2dmoe": MethodSpec(
        "d2dmoe", "D2DMoE", "partitioned_iso_total", "key_clustering",
        "d2d_regression", "partitioned", router_calibration=True,
        router_target="output_norm", router_objective="mse", sparsity_calibration=True,
        notes="Hoyer sparsification, output-norm regression router, dynamic-k threshold.",
    ),
    "llama_moe_random": MethodSpec(
        "llama_moe_random", "LLaMA-MoE (random)", "partitioned_iso_total", "random",
        "switch", "partitioned",
    ),
    "llama_moe_clustering": MethodSpec(
        "llama_moe_clustering", "LLaMA-MoE (clustering)", "partitioned_iso_total",
        "key_clustering", "switch", "partitioned",
    ),
    "switch_partitioned": MethodSpec(
        "switch_partitioned", "Regular Switch (iso-total)", "partitioned_iso_total",
        "scratch", "switch", "partitioned", requires_dense=False,
        notes="From-scratch Switch control with d_ff / E hidden units per expert.",
    ),
    "sparse_upcycling": MethodSpec(
        "sparse_upcycling", "Sparse Upcycling", "upcycling_iso_active", "copy",
        "switch", "full",
        notes="Every expert is a full-width copy of the dense FFN.",
    ),
    "cluster_aware_upcycling": MethodSpec(
        "cluster_aware_upcycling", "Cluster-aware Upcycling", "upcycling_iso_active",
        "cluster_aware_svd", "centroid", "full", distillation=True,
        notes="Spherical activation clusters, data-aware SVD, centroid router and EMA distillation.",
    ),
    "switch_full": MethodSpec(
        "switch_full", "Regular Switch (iso-active)", "upcycling_iso_active",
        "scratch", "switch", "full", requires_dense=False,
        notes="From-scratch full-width Switch anchor for the upcycling regime.",
    ),
}


DEFAULT_METHODS: Tuple[str, ...] = (
    "fc_guided",
    "random_calibrated",
    "clustering_calibrated",
    "fc_guided_abs",
    "moefication",
    "emoe",
    "d2dmoe",
    "llama_moe_random",
    "llama_moe_clustering",
    "switch_partitioned",
    "sparse_upcycling",
    "cluster_aware_upcycling",
    "switch_full",
)


@dataclass
class BenchmarkConfig:
    data_dir: str = "/mnt/Data16T/Data/haichao/code/AI_connectom/story/story_part2_struc_func/Transformer/data/wikitext-2"
    output_dir: str = "benchmark_outputs_v2"
    seq_len: int = 128
    batch_size: int = 32
    max_vocab_size: int = 5000

    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_layers: int = 5
    num_experts: int = 4
    top_k: int = 1
    dropout: float = 0.1
    # v2 defaults reproduce the original FC-guided notebook more closely.
    tie_embeddings: bool = False
    embedding_init_std: float = 0.02
    xavier_initialize: bool = True
    capacity_factor: float = 2.0
    drop_overflow_tokens: bool = True

    lr: float = 5e-4
    weight_decay: float = 0.01
    total_steps: int = 2000
    dense_warmup_steps: int = 100
    eval_interval: int = 20
    eval_batches: int = 30
    restore_best_validation: bool = True
    construction_batches: int = 20
    aux_loss_coef: float = 0.1
    grad_clip: float = 5.0

    # Method-specific work is charged to the same total training-token budget.
    router_calibration_steps: int = 20
    d2d_sparsity_steps: int = 20
    d2d_hoyer_alpha: float = 1e-4
    d2d_threshold: float = 0.5
    d2d_threshold_sweep: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9)
    router_hidden_size: int = 128
    router_calibration_temperature: float = 1.0
    router_alignment_coef: float = 0.02
    router_alignment_steps: int = 300

    # FC-Hybrid construction. Candidate selection uses training-only activations.
    fc_consensus_splits: int = 4
    fc_output_weight: float = 0.35
    fc_uniform_shrinkage: float = 0.05
    fc_refinement_swaps: int = 64
    # Candidate weights are (FC contribution, MoEfication co-activation, key geometry).
    fc_hybrid_view_weights: Tuple[Tuple[float, float, float], ...] = (
        (1.00, 0.00, 0.00),
        (0.70, 0.15, 0.15),
        (0.55, 0.25, 0.20),
        (0.40, 0.30, 0.30),
    )
    fc_oracle_tokens: int = 4096
    fc_router_ridge: float = 0.05
    fc_router_key_prior: float = 0.20

    cluster_aware_energy: float = 0.90
    cluster_aware_min_rank_fraction: float = 0.50
    cluster_aware_distill_coef: float = 0.1
    ema_decay: float = 0.999

    seeds: Tuple[int, ...] = (0, 1, 2)
    methods: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_METHODS)
    device: str = "auto"
    deterministic: bool = True

    def validate(self) -> None:
        unknown = sorted(set(self.methods) - set(METHOD_SPECS))
        if unknown:
            raise ValueError(f"Unknown benchmark methods: {unknown}")
        if self.d_ff % self.num_experts:
            raise ValueError("d_ff must be divisible by num_experts for partitioned methods")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")
        if self.embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        if self.capacity_factor < 1.0:
            raise ValueError("capacity_factor must be at least 1.0")
        if self.construction_batches < 1:
            raise ValueError("construction_batches must be positive")
        if self.eval_interval < 1 or self.eval_batches < 1:
            raise ValueError("eval_interval and eval_batches must be positive")
        if self.dense_warmup_steps >= self.total_steps:
            raise ValueError("dense_warmup_steps must be smaller than total_steps")
        reserved = self.router_calibration_steps + self.d2d_sparsity_steps
        if reserved >= self.total_steps - self.dense_warmup_steps:
            raise ValueError("D2DMoE calibration consumes the entire continuation budget")
        if any(value < 0 or value > 1 for value in self.d2d_threshold_sweep):
            raise ValueError("Every D2DMoE threshold must lie in [0, 1]")
        if self.router_calibration_temperature <= 0:
            raise ValueError("router_calibration_temperature must be positive")
        if self.router_alignment_coef < 0:
            raise ValueError("router_alignment_coef must be non-negative")
        if self.router_alignment_steps < 0:
            raise ValueError("router_alignment_steps must be non-negative")
        if self.fc_consensus_splits < 1:
            raise ValueError("fc_consensus_splits must be at least one")
        if not 0 <= self.fc_output_weight <= 1:
            raise ValueError("fc_output_weight must lie in [0, 1]")
        if not 0 <= self.fc_uniform_shrinkage <= 1:
            raise ValueError("fc_uniform_shrinkage must lie in [0, 1]")
        if self.fc_refinement_swaps < 0:
            raise ValueError("fc_refinement_swaps must be non-negative")
        if self.fc_oracle_tokens < 32:
            raise ValueError("fc_oracle_tokens must be at least 32")
        if self.fc_router_ridge <= 0:
            raise ValueError("fc_router_ridge must be positive")
        if not 0 <= self.fc_router_key_prior <= 1:
            raise ValueError("fc_router_key_prior must lie in [0, 1]")
        if not self.fc_hybrid_view_weights:
            raise ValueError("fc_hybrid_view_weights must contain at least one candidate")
        for weights in self.fc_hybrid_view_weights:
            if len(weights) != 3 or any(value < 0 for value in weights):
                raise ValueError("Every FC hybrid candidate needs three non-negative weights")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Every FC hybrid candidate must sum to one")
            if weights[0] < 0.35:
                raise ValueError("Every FC hybrid candidate must retain at least 35% FC weight")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["methods"] = list(self.methods)
        value["seeds"] = list(self.seeds)
        value["d2d_threshold_sweep"] = list(self.d2d_threshold_sweep)
        return value


def method_table(config: BenchmarkConfig):
    """Return registry rows without importing pandas."""
    rows = []
    for name in config.methods:
        spec = METHOD_SPECS[name]
        rows.append(asdict(spec))
    return rows
