#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from fc_pruning.config import load_config
from fc_pruning.modeling import load_model
from fc_pruning.sobp_reproduction import collect_sobp_hessians


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
    print(json.dumps(collect_sobp_hessians(model, config["calibration_path"], args.output, device), indent=2))


if __name__ == "__main__":
    main()
