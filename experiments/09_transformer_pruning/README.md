# FC-pruning reproduction

This directory contains the code used to reproduce the C4 structured FFN
pruning experiments and figures for:

- Llama-3.2-1B
- Qwen2.5-1.5B

The same pipeline evaluates FC-pruning, FLAP, SoBP, FANG, SlimLLM, and Wanda at
20%, 30%, 40%, and 50% pruning. Each result is averaged over ten independently
sampled C4 calibration subsets. WikiText-2 raw test perplexity is the evaluation
metric.

The internal result key for our method is `fc_ls`; figures display it as
**FC-pruning**. FC-pruning uses signed Pearson correlation for functional
connectivity.

Models, datasets, calibration tensors, pruning plans, caches, result tables,
and generated figures are intentionally excluded from this repository.

## 1. Environment

Python 3.12 was used for the reported experiments. Install a CUDA-compatible
PyTorch build for your machine first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src:.
PYTHON=python
```

All commands below are run from this directory.

## 2. Local models and datasets

Download the model checkpoints from Hugging Face and place them at exactly:

```text
models/Llama-3.2-1B/
models/Qwen2.5-1.5B/
```

The directories must include `config.json`, tokenizer files, and all model
weight files. Model loading is offline (`local_files_only=True`).

Place the required dataset files at:

```text
data/c4/raw/c4-train.00000-of-01024.json.gz
data/wikitext2_raw/test-00000-of-00001.parquet
```

The experiment uses the English C4 training shard
`c4-train.00000-of-01024.json.gz` and the
`Salesforce/wikitext` `wikitext-2-raw-v1` test parquet.

## 3. Build calibration subsets

Calibration data is tokenized independently for each model so Qwen never reuses
Llama token IDs:

```bash
$PYTHON scripts/build_ffn_ratio_matrix_calibrations.py \
  --models llama32_1b qwen25_1_5b
```

This creates ten `128 x 2048` C4 calibration tensors per model and writes
`data/ffn_ratio_matrix/manifest.json`.

## 4. Run the experiments

For one GPU:

```bash
$PYTHON -u scripts/run_ffn_ratio_matrix.py \
  --models llama32_1b qwen25_1_5b \
  --gpus 0
```

To run only one model, pass either `--models llama32_1b` or
`--models qwen25_1_5b`. The scheduler resumes completed conditions
automatically. Progress, elapsed time, and ETA are printed to the terminal and
also saved under `results/ffn_ratio_matrix_pearson/logs/`.

Statistics caches are stored under `/tmp/ffn_ratio_matrix_pearson` while a
condition is running. Allow roughly 20 GB of temporary disk space per active
condition.

## 5. Summarize and plot

Combine all seed-level result files:

```bash
$PYTHON scripts/summarize_ffn_ratio_matrix.py \
  --models llama32_1b qwen25_1_5b
```

This writes:

```text
results/ffn_ratio_matrix_pearson/raw_results.csv
results/ffn_ratio_matrix_pearson/aggregate.csv
results/ffn_ratio_matrix_pearson/summary_c4.md
```

Generate one pruning curve per model and separate 20%, 30%, 40%, and 50% seed
point plots:

```bash
$PYTHON scripts/plot_ffn_ratio_results.py \
  --models llama32_1b qwen25_1_5b
```

Figures are written as both PNG and PDF under
`pruning_figures/by_model/`. Methods in every point plot are ordered by mean PPL
from high to low. Shaded curve bands and black point-plot error bars represent
the sample standard deviation over seeds. Use `--error-type sem` to show
standard error bands in the curves instead.

The lightweight notebook
`notebooks/fig6h_s8d_pruning_results.ipynb` runs the same plotting command and
previews the generated PNG files.

Because the two models use different tokenizers, compare pruning methods within
each model; absolute PPL values should not be compared directly between models.

## 6. Tests

```bash
$PYTHON -m unittest discover -s tests -v
```

The tests cover the shared Llama/Qwen model adapter, signed Pearson candidate
selection, pruning plans, reconstruction, progress reporting, and the SlimLLM
reproduction.
