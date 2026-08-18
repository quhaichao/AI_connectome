# Structured FFN pruning for Llama

This repository contains two retained experiment tracks for structured FFN
pruning: the final controlled six-method ratio matrix, and the earlier broad
Llama-2-7B 20% benchmark against OCP, SVD-LLM, SliceGPT, and PruneNet.
Attention-head pruning and rejected search branches are outside the final
scope.

## Final protocol

- Models: Llama-2-7B and Llama-3.2-1B.
- Methods: FC-LS, FLAP, SoBP, FAND, SlimLLM, and Wanda.
- FFN pruning ratios: 20%, 30%, 40%, and 50% in every layer.
- Calibration domains: C4 and WikiText-2 train.
- Calibration subsets: seeds 0 through 9, independently sampled per model and
  domain.
- Calibration budget: 128 sequences x 2048 tokens = 262,144 tokens.
- Evaluation: WikiText-2 raw test, 165 complete 2048-token windows and 337,755
  predicted positions.

The complete matrix contains 960 unique evaluations:

```text
2 models x 2 domains x 10 seeds x 6 methods x 4 ratios = 960
```

## Repository layout

```text
src/fc_pruning/              Core pruning, reconstruction, and evaluation code
scripts/                     Calibration, execution, summary, and plotting tools
tests/                       Focused CPU unit tests
results/ffn_ratio_matrix/    Raw and aggregate final results
results/fc_layer_analysis/   Layer-wise FC statistics and figures
results/prior_20pct_benchmarks/  Earlier broad baseline and cost tables
external/                    Official-source snapshots used by reproductions
```

Models and datasets are intentionally not committed. Expected local paths are:

```text
models/Llama-2-7b-hf/
models/Llama-3.2-1B/
data/c4/raw/c4-train.00000-of-01024.json.gz
data/wikitext2_raw/{train,validation,test}-00000-of-00001.parquet
```

## Reproduction

Use an environment with a working NVIDIA driver and install the Python
dependencies in `requirements.txt`. The original experiments used the local
`transformer` environment.

```bash
export PYTHONPATH=src:.
PYTHON=/path/to/transformer/bin/python

$PYTHON scripts/build_ffn_ratio_matrix_calibrations.py
$PYTHON -u scripts/run_ffn_ratio_matrix.py --gpus 0 1
$PYTHON scripts/summarize_ffn_ratio_matrix.py
```

The runner supports `--resume`: completed method/ratio rows in each condition
are skipped. Per-condition plans and sufficient-statistic caches are generated
locally and are excluded from Git because they are reproducible and large.

The earlier 20% baselines use
`configs/llama2_7b_20pct_full_wiki_train_reproduction.json`. Their individual
entry points are retained under `scripts/run_*_reproduction.py`,
`scripts/run_slicegpt_equal_macs.py`, and `scripts/run_svd_llm_*_equal_macs.py`.
The official FLAP, Wanda, SVD-LLM, SliceGPT, and PruneNet source snapshots are
kept under `external/` for provenance.

`residual_channel_analysis/` is a separate local project and is intentionally
excluded from this repository.

See `RESULTS.md` for conclusions and `results/README.md` for artifact details.
