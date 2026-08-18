#!/usr/bin/env python
"""Build independent full-position C4/WikiText calibration subsets.

Each payload is tokenized with the model that will consume it. This avoids
silently reusing Llama-3 token IDs for Llama-2 and records a manifest for every
model/domain/seed condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fc_pruning.data import prepare_c4_calibration
from fc_pruning.modeling import load_tokenizer
from build_fc_matrix_stability_data import build_wiki_train_payload, write_payload


MODELS = {
    "llama2_7b": "models/Llama-2-7b-hf",
    "llama32_1b": "models/Llama-3.2-1B",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=tuple(MODELS))
    parser.add_argument("--output-root", default="data/ffn_ratio_matrix")
    parser.add_argument("--max-windows-per-article", type=int, default=2)
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
        model_path = (root / MODELS[model_name]).resolve()
        tokenizer = load_tokenizer(str(model_path))
        for domain in ("c4", "wiki_train"):
            for seed in args.seeds:
                condition = output_root / model_name / domain / f"seed{seed}"
                calibration_path = condition / "calibration.pt"
                if calibration_path.exists():
                    print(f"Reuse {calibration_path}", flush=True)
                elif domain == "c4":
                    payload = prepare_c4_calibration(
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
                else:
                    payload = build_wiki_train_payload(
                        root / "data/wikitext2_raw/train-00000-of-00001.parquet",
                        root / "data/wikitext2_raw/validation-00000-of-00001.parquet",
                        root / "data/wikitext2_raw/test-00000-of-00001.parquet",
                        tokenizer,
                        128,
                        2048,
                        seed=seed,
                        max_windows_per_article=args.max_windows_per_article,
                    )
                    write_payload(payload, calibration_path)
                protocol = json.loads(
                    calibration_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
                )["protocol"]
                if int(protocol["activation_tokens"]) != 128 * 2048:
                    raise RuntimeError(f"Invalid calibration token count: {calibration_path}")
                manifest["conditions"].append(
                    {
                        "model": model_name,
                        "model_path": str(model_path),
                        "domain": domain,
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
