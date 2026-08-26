#!/usr/bin/env python
"""Build independent full-position C4 calibration subsets.

Each payload is tokenized with the model that will consume it. This avoids
silently reusing Llama token IDs for Qwen and records a manifest for every
model/seed condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fc_pruning.data import prepare_c4_calibration
from fc_pruning.modeling import (
    MODEL_KEYS,
    load_tokenizer,
    resolve_model_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--output-root", default="data/ffn_ratio_matrix")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = root / args.output_root
    manifest = {
        "sequence_count": 128,
        "sequence_length": 2048,
        "positions_per_sequence": 2048,
        "activation_tokens": 262144,
        "seeds": args.seeds,
        "conditions": [],
    }

    for model_name in args.models:
        model_path = resolve_model_path(root, model_name)
        tokenizer = load_tokenizer(str(model_path))
        for seed in args.seeds:
            condition = output_root / model_name / "c4" / f"seed{seed}"
            calibration_path = condition / "calibration.pt"
            if calibration_path.exists():
                print(f"Reuse {calibration_path}", flush=True)
            else:
                prepare_c4_calibration(
                    {
                        "seed": seed,
                        "calibration_sequences": 128,
                        "calibration_length": 2048,
                        "sampled_positions_per_sequence": 2048,
                        "minimum_sample_position": 0,
                        "c4_shard": str(
                            (root / "data/c4/raw/c4-train.00000-of-01024.json.gz").resolve()
                        ),
                        "c4_reservoir_size": 8192,
                        "c4_minimum_characters": 7000,
                        "calibration_path": str(calibration_path),
                    },
                    tokenizer,
                )
            protocol = json.loads(
                calibration_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )["protocol"]
            if int(protocol["activation_tokens"]) != 128 * 2048:
                raise RuntimeError(f"Invalid calibration token count: {calibration_path}")
            manifest["conditions"].append(
                {
                    "model": model_name,
                    "model_path": str(model_path),
                    "domain": "c4",
                    "seed": seed,
                    "calibration_path": str(calibration_path.resolve()),
                    "protocol": protocol,
                }
            )

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "conditions": len(manifest["conditions"])}, indent=2))


if __name__ == "__main__":
    main()
