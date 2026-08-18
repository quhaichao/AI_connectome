#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def predicted_saving(plan: dict) -> float:
    total = 0.0
    for merge in plan.get("merges", []):
        gain = min(float(merge.get("validation_gain", 0.0)), 0.999)
        cost = float(merge.get("cost", 0.0))
        total += cost * gain / max(1e-6, 1.0 - gain)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fc-plan", required=True)
    parser.add_argument("--flap-plan", required=True)
    parser.add_argument("--top-layers", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fc_plans = json.loads(Path(args.fc_plan).read_text(encoding="utf-8"))
    flap_plans = json.loads(Path(args.flap_plan).read_text(encoding="utf-8"))
    if len(fc_plans) != len(flap_plans):
        raise ValueError("FC and FLAP plans must have the same number of layers")
    scores = []
    for index, (fc_plan, flap_plan) in enumerate(zip(fc_plans, flap_plans)):
        fc_mask = set(fc_plan["pruned"])
        flap_mask = set(flap_plan["pruned"])
        scores.append(
            {
                "layer": index,
                "predicted_saving": predicted_saving(fc_plan),
                "fc_merges": len(fc_plan.get("merges", [])),
                "mask_replacements": len(fc_mask - flap_mask),
            }
        )
    order = [
        item["layer"]
        for item in sorted(
            scores,
            key=lambda item: (item["predicted_saving"], item["fc_merges"]),
            reverse=True,
        )
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for top_layers in args.top_layers:
        if not 0 <= top_layers <= len(fc_plans):
            raise ValueError("top-layers must be between zero and the layer count")
        selected = set(order[:top_layers])
        hybrid = [
            fc_plan if index in selected else flap_plan
            for index, (fc_plan, flap_plan) in enumerate(zip(fc_plans, flap_plans))
        ]
        label = f"top_{top_layers:02d}"
        path = output / f"plan_{label}.json"
        path.write_text(json.dumps(hybrid, indent=2), encoding="utf-8")
        manifests.append(
            {
                "label": label,
                "plan": str(path.resolve()),
                "fc_layers": sorted(selected),
                "predicted_saving": sum(
                    item["predicted_saving"]
                    for item in scores
                    if item["layer"] in selected
                ),
            }
        )
    (output / "manifest.json").write_text(
        json.dumps({"layer_scores": scores, "plans": manifests}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
