#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "external" / "PruneNet-main" / "prunenet"))

from SparsityPredictor import SparsityPredictor
from prunenet_utils import _slicing

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.modeling import load_tokenizer
from fc_pruning.prunenet_reproduction import (
    select_prunenet_channels,
    set_prunenet_seed,
    train_prunenet_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    ratio = float(config["prune_ratios"][0])
    seed = int(config["seed"])
    set_prunenet_seed(seed)

    result_dir = Path(config["results_dir"]) / "prunenet"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "training_checkpoint.pt"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    layers = list(model.model.layers)
    action_model = SparsityPredictor(
        model.config.hidden_size, model.config.intermediate_size
    ).to(device)
    resume_payload = None
    if args.resume and checkpoint_path.exists():
        resume_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state = train_prunenet_policy(
        layers,
        action_model,
        ratio,
        args.num_episodes,
        args.learning_rate,
        device,
        lambda payload: torch.save(payload, checkpoint_path),
        resume_payload,
    )
    action_model.load_state_dict(state.best_model_state)
    selections = select_prunenet_channels(layers, action_model, ratio, device)
    plans = []
    for index, (layer, selection) in enumerate(zip(layers, selections)):
        _slicing("llama", layer, selection)
        plans.append(
            {
                "layer": index,
                "kept": int(selection.numel()),
                "pruned": int(model.config.intermediate_size - selection.numel()),
                "kept_indices": selection.tolist(),
            }
        )

    model.to(dtype=torch.bfloat16, device=device)
    tokenizer = load_tokenizer(config["model_path"])
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result = {
        "method": "prunenet_official_core_ffn_uniform",
        "fidelity": "official policy, sampling, KS spectral reward and FFN slicing; layer-streamed equivalent training",
        "calibration_tokens": 0,
        "compression_ratio": ratio,
        "num_episodes": args.num_episodes,
        "learning_rate": args.learning_rate,
        "best_total_reward": state.best_score,
        "episode_history": state.episode_rows,
        **evaluation,
    }
    torch.save(state.best_model_state, result_dir / "best_action_model.pt")
    (result_dir / "plan.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
    (result_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
