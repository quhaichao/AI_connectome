#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.activations import collect_down_projection_inputs
from fc_pruning.config import load_config
from fc_pruning.data import (
    prepare_c4_calibration,
    prepare_wikitext_validation_calibration,
)
from fc_pruning.modeling import load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokens-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    tokenizer = load_tokenizer(config["model_path"])
    calibration_path = Path(config["calibration_path"])
    if args.force or not calibration_path.exists():
        source = config.get("calibration_source", "c4")
        if source == "c4":
            payload = prepare_c4_calibration(config, tokenizer)
        elif source == "wikitext_validation":
            payload = prepare_wikitext_validation_calibration(config, tokenizer)
        else:
            raise ValueError(f"Unsupported calibration_source: {source}")
        print(json.dumps(payload["protocol"], indent=2))
    else:
        print(f"Reusing {calibration_path}")
    if args.tokens_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to collect full Llama-3.2-1B activations.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    metadata = collect_down_projection_inputs(
        model, config["calibration_path"], config["activation_dir"], device
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
