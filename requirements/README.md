# Environments

Use separate virtual environments for the major experiment families.

| Analysis family | Dependency file |
| --- | --- |
| MLP, CNN, core Transformer FC–IS, cross-system and supplementary analyses | `core.txt` |
| Transformer architecture comparisons (Fig. 6a) | `transformer_architecture.txt` |
| Optional Mamba-2 architecture comparison | `transformer_architecture_mamba2.txt` after installing a matching CUDA PyTorch build |
| Transformer MoE (Fig. 6b–e, Fig. S8a–c) | `transformer_moe.txt` |
| Transformer pruning (Fig. 6f–h, Fig. S8d) | `transformer_pruning.txt` |

The module-specific files are byte-identical copies of the active source requirements. `core.txt` is an unpinned compatibility starting point inferred from the common notebook imports; freeze exact versions only after a clean end-to-end rerun.

