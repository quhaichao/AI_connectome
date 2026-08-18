"""Deterministic WikiText-2 loading used by every benchmark method."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("<pad>", "<unk>", "<eos>")


def _split_words(text: str) -> List[str]:
    # Matches the whitespace-tokenized protocol in the original notebook.
    return re.findall(r"\S+", text)


def _resolve_split(data_dir: Path, split: str) -> Path:
    candidates = {
        "train": ("wiki.train.tokens", "wiki.train.raw"),
        "valid": ("wiki.valid.tokens", "wiki.valid.raw", "wiki.validation.tokens"),
        "test": ("wiki.test.tokens", "wiki.test.raw"),
    }[split]
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find WikiText-2 {split} split under {data_dir}")


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode(self, words: Sequence[str]) -> List[int]:
        unk = self.unk_id
        return [self.stoi.get(word, unk) for word in words]


class CausalSequenceDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], seq_len: int):
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        usable = ((ids.numel() - 1) // seq_len) * seq_len
        if usable < seq_len:
            raise ValueError("Split is too short for the configured seq_len")
        self.inputs = ids[:usable].view(-1, seq_len)
        self.targets = ids[1 : usable + 1].view(-1, seq_len)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


@dataclass
class WikiTextBundle:
    vocab: Vocabulary
    train: CausalSequenceDataset
    valid: CausalSequenceDataset
    test: CausalSequenceDataset


def load_wikitext2(data_dir: str, seq_len: int, max_vocab_size: int) -> WikiTextBundle:
    root = Path(data_dir).expanduser().resolve()
    texts = {
        split: _resolve_split(root, split).read_text(encoding="utf-8")
        for split in ("train", "valid", "test")
    }
    train_words = _split_words(texts["train"])
    counts = Counter(train_words)
    keep = max(0, max_vocab_size - len(SPECIAL_TOKENS))
    itos = list(SPECIAL_TOKENS) + [word for word, _ in counts.most_common(keep)]
    vocab = Vocabulary({word: i for i, word in enumerate(itos)}, itos)

    datasets = {}
    for split, text in texts.items():
        # Preserve document boundaries with EOS, instead of silently joining lines.
        words: List[str] = []
        for line in text.splitlines():
            words.extend(_split_words(line))
            words.append("<eos>")
        datasets[split] = CausalSequenceDataset(vocab.encode(words), seq_len)
    return WikiTextBundle(vocab, datasets["train"], datasets["valid"], datasets["test"])


def infinite_batches(
    dataset: CausalSequenceDataset,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield an identical shuffled batch stream for every method at a given seed."""
    if len(dataset) < batch_size:
        raise ValueError("Training dataset is smaller than batch_size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    while True:
        order = torch.randperm(len(dataset), generator=generator)
        usable = (len(order) // batch_size) * batch_size
        for offset in range(0, usable, batch_size):
            index = order[offset : offset + batch_size]
            yield dataset.inputs[index].to(device), dataset.targets[index].to(device)


def evaluation_batches(
    dataset: CausalSequenceDataset,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for offset in range(0, len(dataset), batch_size):
        if offset // batch_size >= max_batches:
            break
        end = min(offset + batch_size, len(dataset))
        yield dataset.inputs[offset:end].to(device), dataset.targets[offset:end].to(device)
