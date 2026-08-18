#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    sources = {
        "FANG + ASA": (root / "fang_adaptive" / "result.json", "#277da1"),
        "FLAP adaptive": (root / "flap_adaptive" / "result.json", "#d1495b"),
        "SoBP global": (root / "sobp_adaptive" / "result.json", "#2a9d8f"),
    }
    figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for label, (path, color) in sources.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        ratios = [100.0 * value for value in payload["layer_ratios"]]
        axis.plot(range(len(ratios)), ratios, marker="o", markersize=3.5, linewidth=1.8, label=label, color=color)
    axis.axhline(20.0, color="#444444", linestyle="--", linewidth=1.3, label="Uniform 20%")
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("FFN channels pruned (%)")
    axis.set_xticks(range(0, 32, 2))
    axis.set_xlim(-0.5, 31.5)
    axis.set_ylim(0, 46)
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    output = root / "adaptive_layer_ratios.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
