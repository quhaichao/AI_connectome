from __future__ import annotations

import argparse
import json
from pathlib import Path

from fc_compare.config import paper_config
from fc_compare.plotting import plot_figure6a


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble controlled Figure 6a rows")
    parser.add_argument("--results-root", default="results_v3")
    parser.add_argument("--output", default="results_v3/Figure6a_controlled")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    rows = []
    group_dirs = {
        "a1": "a1",
        "a2": "a2_dhc4_adaptive",
        "a3": "a3",
        "a4": "a4",
    }
    for group in ("a1", "a2", "a3", "a4"):
        spec = paper_config(group)[0]
        group_dir = root / group_dirs[group]
        aggregate_path = group_dir / "aggregate_summary.json"
        if not aggregate_path.is_file():
            raise FileNotFoundError(f"missing {aggregate_path}")
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        summaries = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(group_dir.glob("seed_*/summary.json"))
        ]
        if not summaries:
            raise FileNotFoundError(f"no per-seed summaries in {group_dir}")
        rows.append((spec, summaries, aggregate))
    paths = plot_figure6a(rows, Path(args.output))
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
