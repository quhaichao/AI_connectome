#!/usr/bin/env python
"""Schedule all model/domain/seed conditions across independent GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/ffn_ratio_matrix/manifest.json")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--results-root", default="results/ffn_ratio_matrix")
    parser.add_argument("--cache-dir", default="/tmp/ffn_ratio_matrix")
    parser.add_argument("--models", nargs="+", default=["llama2_7b", "llama32_1b"])
    parser.add_argument("--domains", nargs="+", default=["c4", "wiki_train"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    selected = [
        item
        for item in manifest["conditions"]
        if item["model"] in args.models
        and item["domain"] in args.domains
        and int(item["seed"]) in args.seeds
    ]
    # Alternate model sizes so both GPUs do not always enter the largest
    # collection phase simultaneously.
    selected.sort(key=lambda x: (x["seed"], x["domain"], x["model"]), reverse=True)
    queue = deque(selected)
    logs = root / args.results_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = {}
    failures = []

    while queue or running:
        for gpu in args.gpus:
            if gpu in running or not queue:
                continue
            item = queue.popleft()
            label = f"{item['model']}_{item['domain']}_seed{item['seed']}"
            log_path = logs / f"{label}.log"
            command = [
                sys.executable,
                "-u",
                "scripts/run_ffn_ratio_condition.py",
                "--model",
                item["model"],
                "--domain",
                item["domain"],
                "--seed",
                str(item["seed"]),
                "--calibration",
                item["calibration_path"],
                "--results-root",
                args.results_root,
                "--cache-dir",
                args.cache_dir,
                "--device",
                "cuda:0",
                "--resume",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONPATH"] = "src:."
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, handle, item, log_path)
            print(f"GPU {gpu}: started {label} pid={process.pid}", flush=True)
        time.sleep(args.poll_seconds)
        for gpu, state in list(running.items()):
            process, handle, item, log_path = state
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            label = f"{item['model']}_{item['domain']}_seed{item['seed']}"
            if return_code:
                failures.append({"condition": label, "return_code": return_code, "log": str(log_path)})
                print(f"GPU {gpu}: FAILED {label} rc={return_code}", flush=True)
            else:
                print(f"GPU {gpu}: completed {label}", flush=True)
            del running[gpu]

    status = {"selected_conditions": len(selected), "failures": failures}
    status_path = root / args.results_root / "scheduler_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
