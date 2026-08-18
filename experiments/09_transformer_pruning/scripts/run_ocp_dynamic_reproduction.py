#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fc_pruning.config import load_config
from fc_pruning.evaluate import evaluate_wikitext_ppl
from fc_pruning.modeling import load_model, load_tokenizer
from fc_pruning.ocp_dynamic import install_ocp_dynamic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["model_path"])
    target = float(config["prune_ratios"][0])
    controller = install_ocp_dynamic(
        model,
        target,
        first_pruned_layer=3,
        token_fraction=0.5,
        beta=0.95,
        gamma=0.1,
        clip_delta=0.1,
    )
    evaluation = evaluate_wikitext_ppl(
        model, tokenizer, config["wikitext_test"], config, device
    )
    ratios = [record["prune_ratio"] for record in controller.records]
    result = {
        "method": "ocp_dynamic_paper_logic_pp_output_score",
        "fidelity": "paper OCP probing and adaptive allocation; PPsp replaced by probe output-impact score because PPsp is external to the paper",
        "calibration_tokens": 0,
        "first_pruned_layer": controller.first_pruned_layer,
        "active_target": controller.active_target,
        "mean_dynamic_ratio": sum(ratios) / len(ratios),
        "minimum_dynamic_ratio": min(ratios),
        "maximum_dynamic_ratio": max(ratios),
        **evaluation,
    }
    result_dir = Path(config["results_dir"]) / "ocp_dynamic"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "allocation_records.json").write_text(
        json.dumps(controller.records, indent=2), encoding="utf-8"
    )
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
