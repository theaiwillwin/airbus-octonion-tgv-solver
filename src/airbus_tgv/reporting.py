"""CSV / JSON / YAML reporting utilities."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import csv
import json

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def write_metrics_csv(rows: Iterable[dict], path):
    rows = list(rows)
    if not rows:
        Path(path).write_text("")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_summary_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_config_yaml(cfg: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if _HAS_YAML:
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
    else:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
