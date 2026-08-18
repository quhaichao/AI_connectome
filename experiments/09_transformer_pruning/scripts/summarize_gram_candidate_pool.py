#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from fc_pruning.gram_similarity import GramPlanConfig, _candidate_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    args = parser.parse_args()
    payload = torch.load(args.pool, map_location="cpu", weights_only=True)
    similarities = []
    gains = []
    for layer in payload["layers"]:
        similarities.append(layer["similarity"].float())
        gains.append(_candidate_metrics(layer, GramPlanConfig())["output_gain"])
    similarity = torch.cat([value.flatten() for value in similarities])
    gain = torch.cat([value.flatten() for value in gains])
    quantiles = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    result = {
        "method": payload["layers"][0]["method"],
        "candidates": int(similarity.numel()),
        "similarity_quantiles": torch.quantile(similarity, quantiles).tolist(),
        "output_gain_quantiles": torch.quantile(gain, quantiles).tolist(),
        "counts": {
            f"sim_ge_{similarity_threshold}_gain_ge_{gain_threshold}": int(
                ((similarity >= similarity_threshold) & (gain >= gain_threshold)).sum()
            )
            for similarity_threshold in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8)
            for gain_threshold in (0.0, 0.05, 0.1, 0.2)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
