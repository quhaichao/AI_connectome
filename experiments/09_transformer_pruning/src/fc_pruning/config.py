from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    root = path.parent.parent
    for key in (
        "model_path",
        "calibration_path",
        "activation_dir",
        "full_stats_path",
        "wikitext_validation",
        "wikitext_test",
        "results_dir",
    ):
        if key not in config:
            continue
        value = Path(config[key])
        config[key] = str((root / value).resolve()) if not value.is_absolute() else str(value)
    if "c4_shard" in config:
        value = Path(config["c4_shard"])
        config["c4_shard"] = str(
            (root / value).resolve() if not value.is_absolute() else value
        )
    return config, root
