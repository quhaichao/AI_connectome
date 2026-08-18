#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from fc_pruning.compute_budget import equal_macs_budget
from fc_pruning.config import load_config
from fc_pruning.data import load_calibration
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.modeling import load_tokenizer
from fc_pruning.slicegpt_compat import build_current_llama_adapter


class CalibrationWindows(Dataset):
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.input_ids[index]
        return {
            "input_ids": tokens,
            "attention_mask": torch.ones_like(tokens),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--final-orientation", choices=["random", "pca"], default="random")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Imports below are the unchanged official SliceGPT rotation/slicing code.
    from slicegpt import layernorm_fusion, rotate
    from slicegpt.config import config as slicegpt_config
    from slicegpt.slicing_scheduler import ConstSlicingScheduler

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    slicegpt_config.device = device

    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    tokenizer = load_tokenizer(config["model_path"])
    Adapter = build_current_llama_adapter()
    adapter = Adapter(model)

    ratio = float(config["prune_ratios"][0])
    budget = equal_macs_budget(
        model.config.hidden_size,
        model.config.intermediate_size,
        int(config["evaluation_length"]),
        ratio,
        number_layers=model.config.num_hidden_layers,
    )
    calibration = load_calibration(config["calibration_path"])
    input_ids = calibration["input_ids"]
    if tuple(input_ids.shape) != (128, 2048):
        raise RuntimeError(f"SliceGPT requires 128x2048 windows, got {tuple(input_ids.shape)}")
    loader = DataLoader(
        CalibrationWindows(input_ids),
        batch_size=args.batch_size,
        shuffle=False,
    )

    layernorm_fusion.replace_layers(adapter)
    layernorm_fusion.fuse_modules(adapter)
    scheduler = ConstSlicingScheduler(budget.slice_hidden_size)
    rotate.rotate_and_slice(
        adapter,
        loader,
        scheduler,
        final_orientation=args.final_orientation,
    )

    model.to(device)
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result = {
        "method": "slicegpt_equal_macs",
        "fidelity": "official rotation/slicing with a transformers-4.57 forward compatibility adapter",
        "calibration_tokens": int(input_ids.numel()),
        "final_orientation": args.final_orientation,
        **budget.to_dict(),
        **evaluation,
    }
    result_dir = Path(config["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "slicegpt_equal_macs_config.json").write_text(
        adapter.slicing_conf.to_json_string(), encoding="utf-8"
    )
    (result_dir / "slicegpt_equal_macs_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
