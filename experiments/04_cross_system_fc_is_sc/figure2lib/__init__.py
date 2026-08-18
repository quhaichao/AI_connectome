"""Unified, manuscript-aligned analysis code for Figure 2."""

from .analysis import (
    calculate_pearson_similarity_matrix,
    matrix_correlation,
    paired_matrix_values,
    prepare_fc_sc,
    prepare_system_matrices,
)
from .celegans import get_matlab_string
from .plotting import (
    calculate_corr_sp_boxplot_trend,
    plot_fc_is_fc_sc_diamonds,
)
from .module_level import (
    analyze_one_partition,
    build_positive_knn_graph,
    coarse_grain_matrix,
    correlate_module_matrices,
    correlate_module_matrices_weighted,
    detect_fc_modules_louvain,
    module_pair_weights,
    module_segregation,
    plot_example_module_matrices,
    plot_resolution_robustness,
    run_multiresolution_module_analysis,
    select_resolution_by_module_count,
    select_target_module_partitions,
)
from .workflow import (
    SYSTEM_COLORS,
    calculate_modularity_summary,
    plot_cross_system_summary,
    plot_fc_module_networks,
    run_figure2_system,
    summaries_to_frame,
)

__all__ = [
    "SYSTEM_COLORS",
    "calculate_corr_sp_boxplot_trend",
    "calculate_modularity_summary",
    "calculate_pearson_similarity_matrix",
    "analyze_one_partition",
    "build_positive_knn_graph",
    "coarse_grain_matrix",
    "correlate_module_matrices",
    "correlate_module_matrices_weighted",
    "detect_fc_modules_louvain",
    "module_pair_weights",
    "get_matlab_string",
    "matrix_correlation",
    "paired_matrix_values",
    "plot_cross_system_summary",
    "plot_fc_is_fc_sc_diamonds",
    "plot_fc_module_networks",
    "plot_example_module_matrices",
    "plot_resolution_robustness",
    "prepare_fc_sc",
    "prepare_system_matrices",
    "run_figure2_system",
    "run_multiresolution_module_analysis",
    "select_resolution_by_module_count",
    "select_target_module_partitions",
    "module_segregation",
    "summaries_to_frame",
]
