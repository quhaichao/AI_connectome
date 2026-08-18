# Early functional correlation facilitates learning in artificial neural networks

Code accompanying the manuscript **“Early functional correlation facilitates learning in artificial neural networks.”** The repository is organized by analysis domain while retaining the paper's main-figure and supplementary-figure correspondence.

## Repository layout

| Path | Contents |
| --- | --- |
| `experiments/01_mlp_fc_is_dynamics/` | MLP FC–IS–SC analysis, training dynamics, modularity and causal perturbation |
| `experiments/02_cnn_fc_is/` | CNN FC–IS–SC and pair-level dynamics |
| `experiments/03_transformer_fc_is/` | Transformer FC–IS–SC and Jacobian analysis |
| `experiments/04_cross_system_fc_is_sc/` | Artificial/biological cross-system comparisons and modularity |
| `experiments/05_mlp_functional_roles/` | Early/late FC functional roles and frequency-learning analyses |
| `experiments/06_mlp_gradient_optimization/` | FC–gradient mechanism, optimizer, BatchNorm and residual analyses |
| `experiments/07_transformer_architecture/` | Transformer architecture comparisons in Fig. 6a |
| `experiments/08_transformer_moe/` | FC-guided mixture-of-experts analyses in Fig. 6b–e and Fig. S8a–c |
| `experiments/09_transformer_pruning/` | FC-guided structured pruning in Fig. 6f–h and Fig. S8d |
| `supplementary/` | Dedicated robustness analyses for Figs. S1, S2, S3 and S7 |
| `requirements/` | Environment-specific dependency lists |
| `docs/FIGURE_CODE_MAP.md` | Panel-level paper-to-code index |
| `docs/PROVENANCE.md` | Source-to-release renaming and inclusion record |
| `docs/KNOWN_LIMITATIONS.md` | Portability and reproducibility notes that still require author action |

## Start here

1. Find the target panel in [`docs/FIGURE_CODE_MAP.md`](docs/FIGURE_CODE_MAP.md).
2. Create a separate environment for the relevant analysis family. The MLP/CNN/cross-system notebooks can start from `requirements/core.txt`; each Transformer analysis has its own dependency file.
3. Launch Jupyter from the directory that contains the selected notebook. Several notebooks resolve sibling helper modules relative to the current working directory.
4. Provide the datasets, checkpoints and output directories described in the notebook. Large datasets, trained weights and machine-specific caches are intentionally not included.

Example for the common analyses:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements/core.txt
jupyter lab
```

Transformer architecture, MoE and pruning experiments require distinct environments because their PyTorch/CUDA and package constraints differ. See [`requirements/README.md`](requirements/README.md).


Before a public release, review [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md), replace machine-specific data paths, add the final paper citation/DOI, and choose a license for the authors' code. No repository-wide license has been assigned in this draft.

