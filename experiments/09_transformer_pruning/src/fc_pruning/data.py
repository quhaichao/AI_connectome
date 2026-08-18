from __future__ import annotations

import gzip
import hashlib
import json
import random
from array import array
from pathlib import Path

import pyarrow.parquet as pq
import torch


def _load_wikitext_rows(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".parquet":
        rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
        return [row for row in rows if row]
    return [path.read_text(encoding="utf-8")]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_aligned_token_sequence(haystack: bytes, needle: bytes) -> bool:
    offset = haystack.find(needle)
    while offset >= 0:
        if offset % 4 == 0:
            return True
        offset = haystack.find(needle, offset + 1)
    return False


def _wikitext_heading_level(text: str) -> int:
    compact = "".join(text.strip().split())
    if not compact or not compact.startswith("=") or not compact.endswith("="):
        return 0
    leading = len(compact) - len(compact.lstrip("="))
    trailing = len(compact) - len(compact.rstrip("="))
    return leading if leading == trailing else 0


def _wikitext_articles_without_headings(rows: list[str]) -> list[tuple[str, list[str]]]:
    articles: list[tuple[str, list[str]]] = []
    title: str | None = None
    body: list[str] = []
    for row in rows:
        level = _wikitext_heading_level(row)
        if level == 1:
            if title is not None:
                articles.append((title, body))
            title = row.strip()
            body = []
        elif level == 0 and title is not None and row.strip():
            body.append(row)
    if title is not None:
        articles.append((title, body))
    return articles


def _reservoir_c4_documents(
    shard_path: str | Path,
    reservoir_size: int,
    minimum_characters: int,
    seed: int,
) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    eligible = 0
    with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < minimum_characters:
                continue
            eligible += 1
            item = (line_index, text)
            if len(reservoir) < reservoir_size:
                reservoir.append(item)
                continue
            replacement = rng.randrange(eligible)
            if replacement < reservoir_size:
                reservoir[replacement] = item
    if len(reservoir) < reservoir_size:
        raise RuntimeError(
            f"Only {len(reservoir)} eligible C4 documents found; expected {reservoir_size}."
        )
    rng.shuffle(reservoir)
    return reservoir


def prepare_c4_calibration(config: dict, tokenizer) -> dict:
    seed = int(config["seed"])
    sequence_count = int(config["calibration_sequences"])
    sequence_length = int(config["calibration_length"])
    positions_per_sequence = int(config["sampled_positions_per_sequence"])
    minimum_position = int(config["minimum_sample_position"])
    rng = random.Random(seed)

    reservoir = _reservoir_c4_documents(
        config["c4_shard"],
        int(config["c4_reservoir_size"]),
        int(config["c4_minimum_characters"]),
        seed,
    )
    sequences = []
    sampled_positions = []
    manifest = []
    for line_index, text in reservoir:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < sequence_length:
            continue
        start = rng.randint(0, len(token_ids) - sequence_length)
        segment = token_ids[start : start + sequence_length]
        positions = sorted(
            rng.sample(range(minimum_position, sequence_length), positions_per_sequence)
        )
        sequences.append(segment)
        sampled_positions.append(positions)
        manifest.append(
            {
                "c4_line_index": line_index,
                "token_start": start,
                "document_tokens": len(token_ids),
            }
        )
        if len(sequences) == sequence_count:
            break
    if len(sequences) != sequence_count:
        raise RuntimeError(
            f"Found {len(sequences)} usable C4 sequences, expected {sequence_count}. "
            "Increase c4_reservoir_size or lower c4_minimum_characters."
        )

    payload = {
        "input_ids": torch.tensor(sequences, dtype=torch.long),
        "sampled_positions": torch.tensor(sampled_positions, dtype=torch.long),
        "manifest": manifest,
        "protocol": {
            "source": str(Path(config["c4_shard"]).resolve()),
            "seed": seed,
            "sequence_count": sequence_count,
            "sequence_length": sequence_length,
            "positions_per_sequence": positions_per_sequence,
            "activation_tokens": sequence_count * positions_per_sequence,
        },
    }
    output_path = Path(config["calibration_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    with output_path.with_suffix(".manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": payload["protocol"], "samples": manifest}, handle, indent=2)
    return payload


def prepare_wikitext_validation_calibration(config: dict, tokenizer) -> dict:
    """Build calibration contexts exclusively from WikiText-2 validation.

    Windows are non-overlapping within validation. The manifest and explicit
    audit make it possible to verify that no selected validation sequence
    occurs anywhere in the held-out test token stream.
    """
    validation_path = Path(config["wikitext_validation"])
    test_path = Path(config["wikitext_test"])
    if validation_path.resolve() == test_path.resolve():
        raise ValueError("WikiText validation and test must be different files")

    seed = int(config["seed"])
    sequence_count = int(config["calibration_sequences"])
    sequence_length = int(config["calibration_length"])
    positions_per_sequence = int(config["sampled_positions_per_sequence"])
    minimum_position = int(config["minimum_sample_position"])
    if positions_per_sequence > sequence_length - minimum_position:
        raise ValueError("Not enough legal positions for the requested activation samples")

    validation_rows = _load_wikitext_rows(validation_path)
    test_rows = _load_wikitext_rows(test_path)
    test_text = "\n\n".join(test_rows)
    test_tokens = tokenizer(
        test_text, add_special_tokens=False, return_tensors="pt", verbose=False
    )["input_ids"].squeeze(0)
    max_windows_per_article = int(config.get("max_windows_per_article", 2))
    articles = _wikitext_articles_without_headings(validation_rows)
    candidates = []
    retained_validation_rows = []
    for article_index, (title, body_rows) in enumerate(articles):
        retained_validation_rows.extend(body_rows)
        article_tokens = tokenizer(
            "\n\n".join(body_rows),
            add_special_tokens=False,
            return_tensors="pt",
            verbose=False,
        )["input_ids"].squeeze(0)
        article_windows = min(
            article_tokens.numel() // sequence_length, max_windows_per_article
        )
        for article_window in range(article_windows):
            token_start = article_window * sequence_length
            candidates.append(
                (
                    article_index,
                    title,
                    article_window,
                    token_start,
                    article_tokens[token_start : token_start + sequence_length],
                )
            )
    if sequence_count > len(candidates):
        raise RuntimeError(
            f"WikiText validation has {len(candidates)} eligible article-contained "
            f"windows with max_windows_per_article={max_windows_per_article}; "
            f"requested {sequence_count}."
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected_windows = candidates[:sequence_count]
    sequences = []
    sampled_positions = []
    manifest = []
    for article_index, title, article_window, token_start, sequence in selected_windows:
        positions = sorted(
            rng.sample(range(minimum_position, sequence_length), positions_per_sequence)
        )
        sequences.append(sequence)
        sampled_positions.append(positions)
        manifest.append(
            {
                "split": "validation",
                "article_index": article_index,
                "article_title": title,
                "article_window": article_window,
                "token_start": token_start,
                "token_end": token_start + sequence_length,
                "token_sha256": hashlib.sha256(
                    array("I", map(int, sequence.tolist())).tobytes()
                ).hexdigest(),
            }
        )

    test_bytes = array("I", map(int, test_tokens.tolist())).tobytes()
    exact_test_matches = 0
    for sequence in sequences:
        sequence_bytes = array("I", map(int, sequence.tolist())).tobytes()
        exact_test_matches += int(
            _contains_aligned_token_sequence(test_bytes, sequence_bytes)
        )
    retained_validation_row_hashes = {
        hashlib.sha256(row.strip().encode("utf-8")).hexdigest()
        for row in retained_validation_rows
        if row.strip()
    }
    test_row_hashes = {
        hashlib.sha256(row.strip().encode("utf-8")).hexdigest()
        for row in test_rows
        if row.strip()
    }
    shared_nonempty_rows = len(retained_validation_row_hashes & test_row_hashes)
    if exact_test_matches:
        raise RuntimeError(
            f"Leakage audit found {exact_test_matches} selected calibration windows in test"
        )
    if shared_nonempty_rows:
        raise RuntimeError("Leakage audit found shared calibration prose rows in test")

    fit_sequences = int(round(sequence_count * float(config["similarity_fit_fraction"])))
    audit = {
        "validation_sha256": _sha256(validation_path),
        "test_sha256": _sha256(test_path),
        "paths_are_distinct": True,
        "shared_exact_calibration_prose_rows": shared_nonempty_rows,
        "selected_windows_found_anywhere_in_test": exact_test_matches,
        "test_tokens_scanned": int(test_tokens.numel()),
    }
    payload = {
        "input_ids": torch.stack(sequences),
        "sampled_positions": torch.tensor(sampled_positions, dtype=torch.long),
        "manifest": manifest,
        "protocol": {
            "source": str(validation_path.resolve()),
            "source_dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
            "source_split": "validation",
            "evaluation_split": "test",
            "seed": seed,
            "sequence_count": sequence_count,
            "sequence_length": sequence_length,
            "positions_per_sequence": positions_per_sequence,
            "activation_tokens": sequence_count * positions_per_sequence,
            "fit_sequences": fit_sequences,
            "fit_activation_tokens": fit_sequences * positions_per_sequence,
            "holdout_sequences": sequence_count - fit_sequences,
            "holdout_activation_tokens": (sequence_count - fit_sequences)
            * positions_per_sequence,
            "articles_available": len(articles),
            "eligible_article_windows": len(candidates),
            "max_windows_per_article": max_windows_per_article,
            "headings_removed_from_calibration": True,
            "sampling": "seeded article-contained non-overlapping validation windows",
            "leakage_audit": audit,
        },
    }
    output_path = Path(config["calibration_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    with output_path.with_suffix(".manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"protocol": payload["protocol"], "samples": manifest}, handle, indent=2
        )
    return payload


def load_calibration(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_wikitext_test_tokens(path: str | Path, tokenizer) -> torch.Tensor:
    text = "\n\n".join(_load_wikitext_rows(path))
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt", verbose=False)[
        "input_ids"
    ]
    return ids.squeeze(0)
