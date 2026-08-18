#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from fc_pruning.config import load_config
from fc_pruning.modeling import load_model
from fc_pruning.svd_compression import collect_svd_diagonal_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    protocol = collect_svd_diagonal_statistics(
        model, config["calibration_path"], args.output, device
    )
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
