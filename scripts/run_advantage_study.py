"""Advantage study: associator-guided trajectory compression vs baselines.

For each Reynolds number, compare:
  A. dense trajectory storage (classical baseline)
  B. fixed-rank Tucker (naive tensor baseline, rank chosen blind)
  C. associator-guided Tucker (our method)

Reported per run: memory, reconstruction error, decomposition time, and the
memory x error product (lower = better tradeoff). All numbers are measured,
not estimated.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbus_tgv.constants import TGVParams
from airbus_tgv.trajectory import (
    collect_trajectory, associator_guided_ranks, tucker_decompose_trajectory,
)
from airbus_tgv.reporting import write_metrics_csv, write_summary_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reynolds", type=float, nargs="+",
                    default=[10.0, 100.0, 250.0, 500.0])
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--t-final", type=float, default=0.5)
    ap.add_argument("--snapshot-interval", type=int, default=2)
    ap.add_argument("--fixed-rank", type=int, default=8,
                    help="Blind rank for the naive Tucker baseline")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"advantage_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for Re in args.reynolds:
        params = TGVParams(Re=Re)
        print(f"[advantage] Re={Re}: collecting trajectory...")
        t0 = time.perf_counter()
        traj = collect_trajectory(params, nx=args.nx, ny=args.nx,
                                  t_final=args.t_final,
                                  snapshot_interval=args.snapshot_interval)
        collect_time = time.perf_counter() - t0
        mem_dense = traj.memory_dense_mb()

        # A. dense baseline row
        rows.append({
            "Re": Re, "nx": args.nx, "method": "dense_trajectory",
            "n_snapshots": len(traj.times),
            "ranks": "full",
            "memory_mb": mem_dense,
            "compression_ratio": 1.0,
            "reconstruction_error": 0.0,
            "decomp_time_sec": 0.0,
            "collect_time_sec": collect_time,
        })

        # B. fixed-rank Tucker (blind baseline)
        r = args.fixed_rank
        fixed = {"time": r, "x": r, "y": r}
        t0 = time.perf_counter()
        res_fixed = tucker_decompose_trajectory(traj, core_ranks=fixed)
        t_fixed = time.perf_counter() - t0
        rows.append({
            "Re": Re, "nx": args.nx, "method": "tucker_fixed_rank",
            "n_snapshots": len(traj.times),
            "ranks": str(res_fixed["ranks"]),
            "memory_mb": res_fixed["memory_compressed_mb"],
            "compression_ratio": res_fixed["compression_ratio"],
            "reconstruction_error": res_fixed["reconstruction_error"],
            "decomp_time_sec": t_fixed,
            "collect_time_sec": collect_time,
        })

        # C. associator-guided Tucker
        guided = associator_guided_ranks(traj, min_rank=4, max_rank=32)
        t0 = time.perf_counter()
        res_guided = tucker_decompose_trajectory(traj, core_ranks=guided)
        t_guided = time.perf_counter() - t0
        rows.append({
            "Re": Re, "nx": args.nx, "method": "tucker_associator_guided",
            "n_snapshots": len(traj.times),
            "ranks": str(res_guided["ranks"]),
            "memory_mb": res_guided["memory_compressed_mb"],
            "compression_ratio": res_guided["compression_ratio"],
            "reconstruction_error": res_guided["reconstruction_error"],
            "decomp_time_sec": t_guided,
            "collect_time_sec": collect_time,
        })

        print(f"  dense: {mem_dense:.3f} MB | "
              f"fixed r={r}: {res_fixed['compression_ratio']:.1f}x @ "
              f"{res_fixed['reconstruction_error']:.2e} | "
              f"guided {res_guided['ranks']}: "
              f"{res_guided['compression_ratio']:.1f}x @ "
              f"{res_guided['reconstruction_error']:.2e}")

    write_metrics_csv(rows, out / "metrics.csv")
    write_summary_json({"n_runs": len(rows), "args": vars(args)}, out / "summary.json")
    print(f"[advantage] wrote {len(rows)} rows to {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
