from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


SPLIT_FILES = {
    "train": "wiki.train.tokens",
    "validation": "wiki.valid.tokens",
    "test": "wiki.test.tokens",
}


def _resolve_split(data_dir: Path, split: str) -> Path:
    path = data_dir / SPLIT_FILES[split]
    if not path.is_file():
        expected = ", ".join(SPLIT_FILES.values())
        raise FileNotFoundError(
            f"WikiText-2 split not found: {path}. Expected {expected} in {data_dir}. "
            "The experiment deliberately has no network fallback so every pair uses "
            "the same local corpus version."
        )
    return path


def read_words(path: Path) -> list[str]:
    """Whitespace tokenization with an explicit sentence/document boundary token."""
    words: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            words.extend(line.strip().split())
            words.append("<eos>")
    return words


def build_vocab(train_words: list[str], max_vocab_size: Optional[int] = None):
    counts = Counter(train_words)
    counts.pop("<unk>", None)
    counts.pop("<eos>", None)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if max_vocab_size is not None:
        ordered = ordered[: max(0, max_vocab_size - 2)]
    itos = ["<unk>", "<eos>"] + [token for token, _ in ordered]
    stoi = {token: idx for idx, token in enumerate(itos)}
    return stoi, itos


def encode(words: list[str], stoi: dict[str, int]) -> torch.Tensor:
    unk = stoi["<unk>"]
    return torch.tensor([stoi.get(word, unk) for word in words], dtype=torch.long)


class TokenBlockDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.n_blocks = max(0, (len(token_ids) - 1) // seq_len)
        if self.n_blocks == 0:
            raise ValueError("split is too short for the configured sequence length")

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, index):
        start = int(index) * self.seq_len
        block = self.token_ids[start : start + self.seq_len + 1]
        return block[:-1], block[1:]


def load_wikitext2(config: DataConfig, cache_dir: Path):
    data_dir = Path(config.data_dir)
    split_words = {
        split: read_words(_resolve_split(data_dir, split)) for split in SPLIT_FILES
    }
    stoi, itos = build_vocab(split_words["train"], config.max_vocab_size)
    encoded = {split: encode(words, stoi) for split, words in split_words.items()}

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "vocab.json").write_text(
        json.dumps(itos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "data_dir": str(data_dir),
        "tokenizer": "whitespace + <eos> per source line",
        "vocab_source": "training split only",
        "vocab_size": len(itos),
        "tokens": {key: int(value.numel()) for key, value in encoded.items()},
        "unk_rate": {
            key: float((value == stoi["<unk>"]).float().mean())
            for key, value in encoded.items()
        },
    }
    (cache_dir / "data_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    datasets = {
        split: TokenBlockDataset(ids, config.seq_len) for split, ids in encoded.items()
    }
    return datasets, stoi, itos, metadata


def train_loader(dataset: Dataset, config: DataConfig, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def eval_loader(dataset: Dataset, config: DataConfig):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def fixed_probe(dataset: TokenBlockDataset, n_sequences: int):
    """Evenly sample a fixed held-out probe instead of taking an ordered prefix."""
    n = min(n_sequences, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, steps=n).round().long().unique()
    xs = [dataset[int(index)][0] for index in indices]
    return torch.stack(xs, dim=0)
