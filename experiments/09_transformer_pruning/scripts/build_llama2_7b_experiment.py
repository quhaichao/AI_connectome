#!/usr/bin/env python
"""Build matched C4/Wiki-train calibration and configs for Llama-2-7B."""
from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import torch

from build_fc_matrix_stability_data import build_wiki_train_payload, write_payload
from fc_pruning.data import prepare_c4_calibration
from fc_pruning.modeling import load_tokenizer


def shared_positions(sequence_count: int, length: int, count: int, seed: int) -> torch.Tensor:
    positions = []
    for sequence_index in range(sequence_count):
        rng = random.Random(seed + sequence_index * 1000003)
        positions.append(sorted(rng.sample(range(128, length), count)))
    return torch.tensor(positions, dtype=torch.long)


def rewrite_payload(payload: dict, path: Path, positions: torch.Tensor, source: str) -> None:
    payload["sampled_positions"] = positions.clone()
    payload["protocol"] = {
        **payload["protocol"],
        "positions_per_sequence": int(positions.shape[1]),
        "activation_tokens": int(positions.numel()),
        "shared_position_schedule": True,
        "experiment_source": source,
    }
    write_payload(payload, path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models/Llama-2-7b-hf"
    tokenizer = load_tokenizer(str(model_path))
    data_dir = root / "data/llama2_7b"
    c4_path = data_dir / "c4_train_128x2048_n16384.pt"
    wiki_path = data_dir / "wikitext2_train_128x2048_n16384.pt"
    sequence_count = 128
    sequence_length = 2048
    position_count = 128
    position_seed = 104729
    positions = shared_positions(
        sequence_count, sequence_length, position_count, position_seed
    )

    c4_config = {
        "seed": 0,
        "calibration_sequences": sequence_count,
        "calibration_length": sequence_length,
        "sampled_positions_per_sequence": position_count,
        "minimum_sample_position": 128,
        "c4_shard": str(root / "data/c4/raw/c4-train.00000-of-01024.json.gz"),
        "c4_reservoir_size": 4096,
        "c4_minimum_characters": 7000,
        "calibration_path": str(c4_path),
    }
    c4_payload = prepare_c4_calibration(c4_config, tokenizer)
    rewrite_payload(c4_payload, c4_path, positions, "c4_train")

    wiki_payload = build_wiki_train_payload(
        root / "data/wikitext2_raw/train-00000-of-00001.parquet",
        root / "data/wikitext2_raw/validation-00000-of-00001.parquet",
        root / "data/wikitext2_raw/test-00000-of-00001.parquet",
        tokenizer,
        sequence_count,
        sequence_length,
        seed=0,
        max_windows_per_article=2,
    )
    rewrite_payload(wiki_payload, wiki_path, positions, "wikitext2_raw_train")

    base = {
        "model_path": str(model_path),
        "wikitext_validation": str(
            root / "data/wikitext2_raw/validation-00000-of-00001.parquet"
        ),
        "wikitext_test": str(
            root / "data/wikitext2_raw/test-00000-of-00001.parquet"
        ),
        "seed": 0,
        "calibration_sequences": sequence_count,
        "calibration_length": sequence_length,
        "sampled_positions_per_sequence": position_count,
        "minimum_sample_position": 128,
        "evaluation_length": 2048,
        "evaluation_max_windows": None,
        "calibration_batch_size": 1,
        "evaluation_batch_size": 1,
        "prune_ratios": [0.2],
        "methods": ["dense", "wanda", "flap", "importance", "is", "fc"],
        "similarity_block_size": 512,
        "topk_receivers": 32,
        "max_merges_per_keeper": 8,
        "protect_top_importance_fraction": 0.1,
        "dead_importance_median_ratio": 0.001,
        "ridge_relative": 0.0001,
        "minimum_source_similarity": 0.0,
        "similarity_fit_fraction": 0.75,
        "minimum_validation_reconstruction_gain": 0.02,
        "selection_rule": "joint_cost",
        "activation_energy_relative_floor": 0.0001,
        "activation_storage_dtype": "float16",
        "max_merge_fraction": 1.0,
        "max_merge_fraction_by_method": {"fc": 0.15, "is": 0.15},
        "pruning_execution": "static_dense",
        "test_used_for_calibration_or_selection": False,
    }
    variants = {}
    for source, calibration_path in (("c4", c4_path), ("wiki_train", wiki_path)):
        config = deepcopy(base)
        config.update(
            {
                "calibration_source": source,
                "calibration_path": str(calibration_path),
                "activation_dir": str(data_dir / f"activations_{source}_n16384"),
                "results_dir": str(root / "results/llama2_7b_20pct" / source),
                "protocol_name": f"Llama2-7B-{source}-C128-L2048-P128-N16384-R20-cap15",
            }
        )
        config_path = root / "configs" / f"llama2_7b_20pct_{source}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        variants[source] = str(config_path)
    manifest = {
        "model": str(model_path),
        "calibration": {"c4": str(c4_path), "wiki_train": str(wiki_path)},
        "configs": variants,
        "contexts": sequence_count,
        "context_length": sequence_length,
        "positions_per_context": position_count,
        "activation_samples": int(positions.numel()),
        "pruning_ratio": 0.2,
        "test_used_for_calibration_or_selection": False,
    }
    (root / "configs/llama2_7b_20pct_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
