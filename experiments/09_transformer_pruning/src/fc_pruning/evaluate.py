from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import load_wikitext_test_tokens


@torch.inference_mode()
def evaluate_wikitext_ppl(model, tokenizer, test_path: str, config: dict, device):
    tokens = load_wikitext_test_tokens(test_path, tokenizer)
    sequence_length = int(config["evaluation_length"])
    windows = tokens.numel() // sequence_length
    maximum = config.get("evaluation_max_windows")
    if maximum is not None:
        windows = min(windows, int(maximum))
    if windows == 0:
        raise RuntimeError("WikiText-2 test contains no complete evaluation window.")

    total_nll = 0.0
    total_predictions = 0
    batch_size = int(config["evaluation_batch_size"])
    for begin in range(0, windows, batch_size):
        current = min(batch_size, windows - begin)
        batch = torch.stack(
            [
                tokens[(begin + offset) * sequence_length : (begin + offset + 1) * sequence_length]
                for offset in range(current)
            ]
        ).to(device)
        logits = model(input_ids=batch, use_cache=False).logits.float()
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        total_nll += F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="sum",
        ).item()
        total_predictions += shift_labels.numel()
    return {
        "ppl": math.exp(total_nll / total_predictions),
        "nll": total_nll,
        "predicted_tokens": total_predictions,
        "windows": windows,
        "sequence_length": sequence_length,
    }
