"""Deterministically rebuild the four thin experiment notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


GROUPS = {
    "a1": {
        "filename": "transformer_optim_rope.ipynb",
        "title": "a1 | Sinusoidal positional encoding versus RoPE",
        "changed": "Only `ModelConfig.position` changes: fixed sinusoidal addition versus RoPE on q/k.",
    },
    "a2": {
        "filename": "transformer_optim_residual.ipynb",
        "title": "a2 | Ordinary residual connections versus optimized DHC recipe",
        "changed": "The optimized recipe uses DHC×4, a token-adaptive final stream readout and training dropout 0.05 instead of 0.10. The readout is downstream of every FC tap, and dropout is disabled for every FC probe; neither directly alters the measured activation. Both variants use the same 24-layer, 176-wide backbone, tokens and FC estimator. This row tests the manuscript's performance–FC association, not isolated causal superiority of DHC.",
        "output_name": "a2_dhc4_adaptive",
        "rerun_note": (
            "Rerunning uses the frozen a2-specific optimizer and fresh confirmation "
            "seeds 121/232/343/454/565. It writes to "
            "`results_v3/a2_dhc4_adaptive/`, preserving earlier diagnostics."
        ),
    },
    "a3": {
        "filename": "transformer_optim_attention.ipynb",
        "title": "a3 | MHA versus hybrid MHA + Mamba-2",
        "changed": "Only `ModelConfig.mixer_pattern` changes. Mamba-2 comes from the official `mamba_ssm.Mamba2`; no surrogate SSM is allowed.",
    },
    "a4": {
        "filename": "transformer_optim_norm.ipynb",
        "title": "a4 | Post-Norm versus Pre-Norm",
        "changed": "Only `ModelConfig.norm_position` changes. Both variants use LayerNorm and the same final normalization/head.",
    },
}


def markdown(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(group: str, metadata: dict):
    rerun_note = metadata.get(
        "rerun_note",
        "Saved output cells may contain earlier diagnostics. Rerunning uses the "
        "current frozen analysis and writes to the configured `results_v3/` directory.",
    )
    cells = [
        markdown(
            f"# Figure 6{metadata['title']}\n\n"
            "**Core conclusion examined:** a variant with lower held-out PPL should show an earlier rise in "
            "position-standardized top-5% functional correlation (FC).\n\n"
            f"**Within-group comparison:** {metadata['changed']}\n\n"
            "The notebook is intentionally thin. Data handling, architectures, paired initialization, training, "
            "FC estimation, descriptive paired statistics and publication exports live in the shared `fc_compare` package so "
            "the four comparisons cannot silently drift apart.\n\n"
            f"> **Rerun note:** {rerun_note}\n\n"
            "Training displays a per-seed progress bar with the latest paired losses."
        ),
        markdown(
            "## Preregistered measurement\n\n"
            "- Corpus: the three local WikiText-2 splits; vocabulary is built from training only.\n"
            "- PPL: complete validation/test splits, token-weighted cross-entropy, best validation checkpoint.\n"
            "- FC units: the D-dimensional effective input to each selected block. `FC_LAYER = 'all'` computes FC independently in every layer and then takes their equal-weight mean; use a zero-based integer for one layer. For DHC, each tap is the actually routed branch input, not the mean of four streams.\n"
            "- FC profile: fixed validation sequences; each feature is z-scored across sequences within each token position before concatenation.\n"
            "- Primary FC statistic: mean of the largest signed 5% off-diagonal Pearson correlations.\n"
            "- Earlier rise: first three-checkpoint sustained increase of at least 0.03 above step 0.\n"
            "- Reporting: paired estimates and bootstrap confidence intervals are saved without automatic pass/fail decisions.\n"
            "- Figure panels: absolute top-n% FC mean, its change from initialization, and validation PPL. All panels are drawn whenever training completes."
        ),
        code(
            "from dataclasses import replace\n"
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n\n"
            "HERE = Path.cwd().resolve()\n"
            "if not (HERE / 'fc_compare').is_dir():\n"
            "    raise RuntimeError('Start Jupyter with FC_compare as the working directory.')\n"
            "sys.path.insert(0, str(HERE))\n\n"
            "from fc_compare import paper_config, run_comparison, smoke_config\n"
        ),
        code(
            f"GROUP = '{group}'\n"
            "DATA_DIR = Path('/mnt/Data16T/Data/haichao/code/AI_connectom/story/story_part2_struc_func/Transformer/data/wikitext-2')\n"
            "MODE = 'paper'  # use 'smoke' only to test the pipeline; smoke results are not scientific evidence\n"
            "FC_LAYER = 'all'  # 'all' averages per-layer FC; use a zero-based integer for one layer\n"
            f"OUTPUT_DIR = HERE / 'results_v3' / '{metadata.get('output_name', group)}'\n\n"
            "factory = paper_config if MODE == 'paper' else smoke_config\n"
            "spec, data, train, fc = factory(GROUP, DATA_DIR)\n"
            "fc = replace(fc, layer_selection=FC_LAYER)\n"
            "print(spec)\n"
            "print(data)\n"
            "print(train)\n"
            "print(fc)\n"
        ),
        code(
            "result = run_comparison(\n"
            "    spec=spec,\n"
            "    data=data,\n"
            "    train=train,\n"
            "    fc=fc,\n"
            "    output_dir=OUTPUT_DIR,\n"
            ")\n"
            "print(json.dumps(result['aggregate'], indent=2, ensure_ascii=False))\n"
        ),
        code(
            "from IPython.display import SVG, display\n"
            "display(SVG(filename=result['figure_paths'][0]))\n"
        ),
        markdown(
            "## Result inspection\n\n"
            "The notebook always displays the three final panels after a successful run. "
            "Use `aggregate_summary.json` for the paired PPL change, FC-onset difference, early-FC AUC difference "
            "and their bootstrap confidence intervals; interpret these values together with the plotted trajectories."
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def preserve_outputs(new_notebook: dict, old_notebook: dict):
    new_code = [
        cell for cell in new_notebook["cells"] if cell["cell_type"] == "code"
    ]
    old_code = [
        cell for cell in old_notebook["cells"] if cell["cell_type"] == "code"
    ]
    if len(new_code) != len(old_code):
        raise RuntimeError(
            f"Cannot preserve outputs across {len(old_code)} -> {len(new_code)} code cells"
        )
    for new_cell, old_cell in zip(new_code, old_code):
        new_cell["execution_count"] = old_cell.get("execution_count")
        new_cell["outputs"] = old_cell.get("outputs", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(GROUPS), default=None)
    parser.add_argument(
        "--preserve-outputs",
        action="store_true",
        help="Keep code-cell outputs while replacing notebook source cells.",
    )
    args = parser.parse_args()
    selected = (args.group,) if args.group else tuple(GROUPS)
    for group in selected:
        metadata = GROUPS[group]
        path = ROOT / metadata["filename"]
        rebuilt = notebook(group, metadata)
        if args.preserve_outputs and path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            preserve_outputs(rebuilt, existing)
        path.write_text(
            json.dumps(rebuilt, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
