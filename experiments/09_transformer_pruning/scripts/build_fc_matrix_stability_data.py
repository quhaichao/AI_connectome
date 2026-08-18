#!/usr/bin/env python
"""Build matched 128x2048 C4 and WikiText-2 train calibration payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from array import array
from pathlib import Path

import torch

from fc_pruning.data import (
    _contains_aligned_token_sequence,
    _load_wikitext_rows,
    _sha256,
    _wikitext_articles_without_headings,
)
from fc_pruning.modeling import load_tokenizer


def token_bytes(tokens: torch.Tensor) -> bytes:
    return array("I", map(int, tokens.tolist())).tobytes()


def build_wiki_train_payload(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    tokenizer,
    sequence_count: int,
    sequence_length: int,
    seed: int,
    max_windows_per_article: int,
) -> dict:
    articles = _wikitext_articles_without_headings(_load_wikitext_rows(train_path))
    candidates = []
    for article_index, (title, body_rows) in enumerate(articles):
        tokens = tokenizer(
            "\n\n".join(body_rows),
            add_special_tokens=False,
            return_tensors="pt",
            verbose=False,
        )["input_ids"].squeeze(0)
        window_count = min(tokens.numel() // sequence_length, max_windows_per_article)
        for window_index in range(window_count):
            start = window_index * sequence_length
            candidates.append(
                (article_index, title, window_index, start, tokens[start : start + sequence_length])
            )
    if len(candidates) < sequence_count:
        raise RuntimeError(f"Only {len(candidates)} eligible WikiText train windows")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:sequence_count]
    sequences = torch.stack([row[4] for row in selected])
    manifest = []
    for article_index, title, window_index, start, sequence in selected:
        manifest.append(
            {
                "split": "train",
                "article_index": article_index,
                "article_title": title,
                "article_window": window_index,
                "token_start": start,
                "token_end": start + sequence_length,
                "token_sha256": hashlib.sha256(token_bytes(sequence)).hexdigest(),
            }
        )

    heldout_tokens = {}
    exact_matches = {}
    for split, path in (("validation", validation_path), ("test", test_path)):
        text = "\n\n".join(_load_wikitext_rows(path))
        tokens = tokenizer(
            text, add_special_tokens=False, return_tensors="pt", verbose=False
        )["input_ids"].squeeze(0)
        heldout_tokens[split] = int(tokens.numel())
        stream = token_bytes(tokens)
        exact_matches[split] = sum(
            _contains_aligned_token_sequence(stream, token_bytes(sequence))
            for sequence in sequences
        )
    if any(exact_matches.values()):
        raise RuntimeError(f"WikiText split audit failed: {exact_matches}")

    positions = torch.arange(sequence_length).repeat(sequence_count, 1)
    return {
        "input_ids": sequences,
        "sampled_positions": positions,
        "manifest": manifest,
        "protocol": {
            "source": str(train_path.resolve()),
            "source_dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
            "source_split": "train",
            "seed": seed,
            "sequence_count": sequence_count,
            "sequence_length": sequence_length,
            "positions_per_sequence": sequence_length,
            "activation_tokens": sequence_count * sequence_length,
            "articles_available": len(articles),
            "eligible_article_windows": len(candidates),
            "max_windows_per_article": max_windows_per_article,
            "headings_removed": True,
            "sampling": "seeded article-contained non-overlapping train windows",
            "heldout_audit": {
                "validation_sha256": _sha256(validation_path),
                "test_sha256": _sha256(test_path),
                "heldout_tokens": heldout_tokens,
                "selected_windows_found_in_heldout": exact_matches,
            },
        },
    }


def write_payload(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {"protocol": payload["protocol"], "samples": payload["manifest"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Llama-3.2-1B")
    parser.add_argument("--c4-calibration", default="data/c4/calibration_llama32_1b_seed0.pt")
    parser.add_argument("--wiki-train", default="data/wikitext2_raw/train-00000-of-00001.parquet")
    parser.add_argument("--wiki-validation", default="data/wikitext2_raw/validation-00000-of-00001.parquet")
    parser.add_argument("--wiki-test", default="data/wikitext2_raw/test-00000-of-00001.parquet")
    parser.add_argument("--output-dir", default="data/fc_matrix_stability")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-windows-per-article", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sequence_count = 128
    sequence_length = 2048
    c4_source = (root / args.c4_calibration).resolve()
    c4 = torch.load(c4_source, map_location="cpu", weights_only=False)
    if tuple(c4["input_ids"].shape) != (sequence_count, sequence_length):
        raise ValueError(f"Expected C4 shape (128, 2048), got {tuple(c4['input_ids'].shape)}")
    c4_payload = {
        "input_ids": c4["input_ids"].clone(),
        "sampled_positions": torch.arange(sequence_length).repeat(sequence_count, 1),
        "manifest": c4["manifest"],
        "protocol": {
            **c4["protocol"],
            "positions_per_sequence": sequence_length,
            "activation_tokens": sequence_count * sequence_length,
            "derived_from": str(c4_source),
            "full_position_diagnostic": True,
        },
    }
    tokenizer = load_tokenizer(str((root / args.model).resolve()))
    wiki_payload = build_wiki_train_payload(
        (root / args.wiki_train).resolve(),
        (root / args.wiki_validation).resolve(),
        (root / args.wiki_test).resolve(),
        tokenizer,
        sequence_count,
        sequence_length,
        args.seed,
        args.max_windows_per_article,
    )
    output_dir = (root / args.output_dir).resolve()
    c4_output = output_dir / "c4_train_128x2048.pt"
    wiki_output = output_dir / "wikitext2_train_128x2048.pt"
    write_payload(c4_payload, c4_output)
    write_payload(wiki_payload, wiki_output)
    manifest = {
        "model": str((root / args.model).resolve()),
        "sequence_count": sequence_count,
        "sequence_length": sequence_length,
        "maximum_activation_tokens": sequence_count * sequence_length,
        "budgets": [8192, 16384, 32768, 65536, 131072, 262144],
        "c4_calibration": str(c4_output),
        "wiki_train_calibration": str(wiki_output),
        "test_used_for_fc": False,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
