"""Generic utilities: timing, memory, IO."""
from __future__ import annotations
import os
import time
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


@contextmanager
def timer():
    """Context manager yielding a callable returning elapsed seconds."""
    t0 = time.perf_counter()
    elapsed = {"t": 0.0}
    try:
        yield elapsed
    finally:
        elapsed["t"] = time.perf_counter() - t0


def process_memory_mb() -> float:
    """Resident memory in MB. 0.0 if psutil is unavailable."""
    if not _HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def field_memory_mb(*arrays: np.ndarray) -> float:
    """Dense memory footprint of a tuple of ndarrays."""
    return sum(a.nbytes for a in arrays) / (1024.0 * 1024.0)


def ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def dump_json(obj, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"unhandled type {type(o)}")
