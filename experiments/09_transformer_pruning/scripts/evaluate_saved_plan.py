#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.modeling import llama_layers, load_model, load_tokenizer
from fc_pruning.pruning import apply_layer_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--drop-merges", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for plan evaluation.")
    device = torch.device("cuda")
    tokenizer = load_tokenizer(config["model_path"])
    model = load_model(config["model_path"], device)
    with Path(args.plan).open("r", encoding="utf-8") as handle:
        plans = json.load(handle)
    for plan in plans:
        if "bias_compensation" in plan and not isinstance(
            plan["bias_compensation"], torch.Tensor
        ):
            plan["bias_compensation"] = torch.tensor(plan["bias_compensation"])
    if args.drop_merges:
        for plan in plans:
            plan["merges"] = []
    for layer, plan in zip(llama_layers(model), plans):
        apply_layer_plan(layer, plan)
    model.config.intermediate_size = llama_layers(model)[0].mlp.intermediate_size
    result = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    result.update(
        {
            "plan": str(Path(args.plan).resolve()),
            "drop_merges": args.drop_merges,
        }
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
