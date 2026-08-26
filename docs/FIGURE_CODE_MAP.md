# Figure-to-code map

This index follows the current manuscript numbering (main Figs. 1–6 and Extended Data Figs. 1–8).

## Main figures

| Figure panels | Analysis | Primary entry point(s) |
| --- | --- | --- |
| Fig. 1a,e | Biological FC and anatomical SC definitions | `experiments/04_cross_system_fc_is_sc/biological_networks_fc_is_sc.ipynb` |
| Fig. 1b,f | MLP FC and direct/Jacobian SC | `experiments/01_mlp_fc_is_dynamics/MLP_FC_IS_analysis.ipynb`; `supplementary/supp_fig1_mlp_jacobian.ipynb` |
| Fig. 1c,g | CNN FC and kernel/Jacobian SC | `experiments/02_cnn_fc_is/cnn_fc_is_analysis.ipynb`; `supplementary/supp_fig1_cnn_transformer_jacobian.ipynb` |
| Fig. 1d,h | Transformer FC and Jacobian SC | `experiments/03_transformer_fc_is/transformer_fc_is_sc_analysis.py`; `supplementary/supp_fig1_cnn_transformer_jacobian.ipynb` |
| Fig. 2a | Input-similarity definition across systems | Core MLP/CNN/Transformer analyses plus `experiments/04_cross_system_fc_is_sc/figure2lib/analysis.py` |
| Fig. 2b,e | Artificial-network FC–IS–SC comparisons | `experiments/04_cross_system_fc_is_sc/artificial_networks_fc_is_sc.ipynb` |
| Fig. 2c,f | Biological-network FC–IS–SC comparisons | `experiments/04_cross_system_fc_is_sc/biological_networks_fc_is_sc.ipynb` |
| Fig. 2d | Seven-system FC–IS versus FC–SC summary | Both cross-system notebooks plus `figure2lib/workflow.py` |
| Fig. 3a–e | Formation and timing of the MLP FC–IS relationship | `experiments/01_mlp_fc_is_dynamics/MLP_FC_IS_analysis.ipynb` |
| Fig. 3f–g | FC/IS modularity and within/between-module allocation | `experiments/01_mlp_fc_is_dynamics/fig3_modularity_mechanism.ipynb` plus the core MLP notebook |
| Fig. 4a–c,e–j | Early- and late-changing FC pairs in classification | `experiments/05_mlp_functional_roles/fig4_early_late_fc_roles.ipynb` |
| Fig. 4d | Independent 20-seed masking of early- versus late-FC-rise neurons | `experiments/05_mlp_functional_roles/fig4d_20seeds.py` |
| Fig. 4k–m | Stable/fluctuating FC pairs in frequency learning | `experiments/05_mlp_functional_roles/fig4_frequency_learning.ipynb` |
| Fig. 5a–f | FC, gradient flow, receptive fields and masking | `experiments/06_mlp_gradient_optimization/fig5_fc_gradient_mechanism.ipynb` |
| Fig. 5g–i | Optimizer, BatchNorm and FC-guided residual comparisons | `experiments/06_mlp_gradient_optimization/fig5_fc_guided_optimization.ipynb`; `mlp_early_high_fc_utils.py` |
| Fig. 6a1–a4 | RoPE, residual, attention and normalization comparisons | `experiments/07_transformer_architecture/fig6a1_rope_vs_sinusoidal.ipynb`; `fig6a2_hyperconnection_vs_residual.ipynb`; `fig6a3_hybrid_vs_standard_attention.ipynb`; `fig6a4_pre_norm_vs_post_norm.ipynb` |
| Fig. 6b–c | FC-guided MoE construction and FC trajectories | `experiments/08_transformer_moe/fig6b_c_fc_guided_moe.ipynb` |
| Fig. 6d | Six-method MoE perplexity benchmark | `experiments/08_transformer_moe/fig6d_moe_benchmark.ipynb` |
| Fig. 6e | Expert semantic specialization | `experiments/08_transformer_moe/fig6e_s8bc_expert_specialization.ipynb` |
| Fig. 6f | Stable high-FC redundancy | `experiments/09_transformer_pruning/scripts/collect_fc_matrices.py`; `analyze_fc_matrix_stability.py`; `plot_fc_layer_redundancy.py` |
| Fig. 6g | FC-guided structured-pruning workflow | `experiments/09_transformer_pruning/src/fc_pruning/` |
| Fig. 6h | Structured-pruning perplexity comparison | `experiments/09_transformer_pruning/fig6h_s8d_pruning_results.ipynb`; `FC_pruning_raw_results.csv` |

## Extended Data figures

| Figure panels | Analysis | Primary entry point(s) |
| --- | --- | --- |
| Extended Data Fig. 1a–d | FC definition robustness and sampling reproducibility | `supplementary/supp_fig1_fc_robustness.ipynb` |
| Extended Data Fig. 1e | MLP weight/Jacobian comparison | `supplementary/supp_fig1_mlp_jacobian.ipynb` |
| Extended Data Fig. 1f–j | CNN and Transformer Jacobian validation | `supplementary/supp_fig1_cnn_transformer_jacobian.ipynb` |
| Extended Data Fig. 2a | Multi-seed, multi-dataset FC–IS reproducibility | `supplementary/supp_fig2_fc_is_reproducibility.ipynb` |
| Extended Data Fig. 2b–g | Cross-system FC/IS/SC matrices and within-module comparisons | Artificial and biological notebooks in `experiments/04_cross_system_fc_is_sc/` |
| Extended Data Fig. 2h–i | Representative networks, modularity and module support | `experiments/04_cross_system_fc_is_sc/supp_fig2_modularity.ipynb` |
| Extended Data Fig. 3a–b | MLP optimizer and density robustness | `experiments/01_mlp_fc_is_dynamics/MLP_FC_IS_analysis.ipynb` |
| Extended Data Fig. 3c | CNN pair-level timing | `experiments/02_cnn_fc_is/cnn_fc_is_analysis.ipynb` |
| Extended Data Fig. 3d | Transformer pair-level timing | `experiments/03_transformer_fc_is/transformer_fc_is_sc_analysis.py` |
| Extended Data Fig. 3e–g | Cross-seed FC-before-IS robustness | `supplementary/supp_fig3_fc_before_is.ipynb` |
| Extended Data Fig. 4 | Direct FC perturbations and independent-seed extension | `experiments/01_mlp_fc_is_dynamics/supp_fig4_fc_perturbation.ipynb`; `experiments/01_mlp_fc_is_dynamics/supp_fig4_fc_perturbation_multiseed.ipynb`; `experiments/01_mlp_fc_is_dynamics/supp_fig4_fc_perturbation_multiseed.py` |
| Extended Data Fig. 5a–h | Multi-step and threshold sensitivity for Fig. 4 | `experiments/05_mlp_functional_roles/fig4_early_late_fc_roles.ipynb`; `fig4d_20seeds.py --supplementary-suite` for seed-level masking robustness |
| Extended Data Fig. 6a–d | Dataset generality of early/late FC roles | `experiments/05_mlp_functional_roles/fig4_early_late_fc_roles.ipynb`; `fig4d_20seeds.py --supplementary-suite` for Fashion-MNIST and CIFAR-10 masking |
| Extended Data Fig. 6e–g | Stable/fluctuating FC frequency analyses | `experiments/05_mlp_functional_roles/fig4_frequency_learning.ipynb` |
| Extended Data Fig. 7a–n | FC–gradient and optimization robustness | `supplementary/supp_fig7_fc_mechanism_robustness.ipynb`; `supplementary/utils/fig5_mechanism.py`; `fig5_robustness.py` |
| Extended Data Fig. 8a | Expert-cluster-reordered FC matrices | `experiments/08_transformer_moe/fig6b_c_fc_guided_moe.ipynb` |
| Extended Data Fig. 8b–c | Layer-wise and seed-level expert specialization | `experiments/08_transformer_moe/fig6e_s8bc_expert_specialization.ipynb` |
| Extended Data Fig. 8d | Multi-ratio, multi-seed pruning results | `experiments/09_transformer_pruning/fig6h_s8d_pruning_results.ipynb`; `FC_pruning_raw_results.csv` |

## Suggested execution order

1. Run the core MLP, CNN and Transformer FC–IS–SC analyses.
2. Run the artificial/biological cross-system aggregation and modularity analysis.
3. Run the dedicated Fig. 4, Fig. 5 and Fig. 6 experiment families.
4. Run the supplementary robustness notebooks last, reusing model-level caches where applicable.

## Fig. 4 independent-seed masking

`experiments/05_mlp_functional_roles/fig4d_20seeds.py` is the primary entry point for the Fig. 4d accuracy-masking result. It trains 20 independently initialized MLPs and uses the training seed as the replication unit. Within each seed, it identifies 15 disjoint early-FC-rise and 15 disjoint late-FC-rise neurons, masks each complete group, and records masked accuracy minus the corresponding intact-checkpoint accuracy. The repeated random-subset analysis from one fixed neuron pool is not used for inference.

Run the primary MNIST analysis from the repository root:

```bash
python experiments/05_mlp_functional_roles/fig4d_20seeds.py
```

Run the full masking robustness suite (MNIST at FC-rise thresholds 0.8 and 0.5, Fashion-MNIST at 0.8, and CIFAR-10 at 0.8):

```bash
python experiments/05_mlp_functional_roles/fig4d_20seeds.py --supplementary-suite
```

The default checkpoints are zero-based model-state indices `260,280,300,400,600,800,1100`; exported `Optimizer_Update` values equal `Step + 1`. Completed per-seed CSV files are reused automatically, while `--overwrite` reruns them. Outputs are written under `experiments/05_mlp_functional_roles/fig4d_20seeds_output/` and include seed-level source data, summary statistics with confidence intervals, paired early-versus-late tests, classification metadata, and PDF/SVG/PNG/TIFF figures.
