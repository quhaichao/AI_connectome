#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from fc_pruning.config import load_config
from fc_pruning.full_stats import collect_full_ffn_statistics
from fc_pruning.modeling import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-token Llama statistics")
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    protocol = collect_full_ffn_statistics(
        model,
        config["calibration_path"],
        config["full_stats_path"],
        device,
        float(config.get("ocp_token_fraction", 0.5)),
    )
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
