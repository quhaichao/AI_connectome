from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from fc_compare import paper_config, run_comparison, smoke_config


def parse_fc_layer(value: str):
    if value.lower() == "all":
        return "all"
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--fc-layer must be 'all' or a zero-based integer"
        ) from error


def parse_args():
    parser = argparse.ArgumentParser(description="Run one controlled Figure 6a comparison")
    parser.add_argument("--group", required=True, choices=("a1", "a2", "a3", "a4"))
    parser.add_argument("--mode", choices=("paper", "smoke"), default="paper")
    parser.add_argument(
        "--data-dir",
        default=(
            "/mnt/Data16T/Data/haichao/code/AI_connectom/story/"
            "story_part2_struc_func/Transformer/data/wikitext-2"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument(
        "--fc-layer",
        type=parse_fc_layer,
        default="all",
        help="'all' averages independently computed per-layer FC; an integer selects one layer",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    factory = paper_config if args.mode == "paper" else smoke_config
    spec, data, train, fc = factory(args.group, args.data_dir)
    fc = replace(fc, layer_selection=args.fc_layer)
    default_name = "a2_dhc4_adaptive" if args.group == "a2" else args.group
    output = Path(args.output_dir or Path("results_v3") / default_name)
    result = run_comparison(spec, data, train, fc, output, device=args.device)
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
