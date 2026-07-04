"""Scalability projection: Tucker compression memory vs grid size.

Runs measured benchmarks at nx = 32, 64, 128 (Re=100) and projects analytically
to nx = 256, 512 using the Tucker memory formula:

  mem_tucker = mem(core) + sum(mem(factor_i))
             = 8 * r_t * r_x * r_y * 8 bytes    (core, float64)
               + (nt * r_t + nx * r_x + ny * r_y) * 8 bytes  (factors)

versus classical dense storage:
  mem_dense = 2 * nt * nx * ny * 8 bytes  (u and v)

Projections fix ranks at the values measured at nx=128 (largest measured grid)
since rank is a physics property, not a grid property: it tracks flow complexity,
which converges as the grid refines under a fixed physical problem.

Output: results/scalability_<timestamp>/metrics.csv and a printed table.
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


def tucker_memory_bytes(nt: int, nx: int, ny: int,
                        r_t: int, r_x: int, r_y: int,
                        n_fields: int = 2) -> float:
    """Analytical Tucker memory in bytes (float64, n_fields velocity components)."""
    bytes_per_float = 8
    core = r_t * r_x * r_y * bytes_per_float
    factors = (nt * r_t + nx * r_x + ny * r_y) * bytes_per_float
    return n_fields * (core + factors)


def dense_memory_bytes(nt: int, nx: int, ny: int, n_fields: int = 2) -> float:
    """Dense trajectory memory in bytes."""
    return n_fields * nt * nx * ny * 8


def project_row(nx: int, nt_per_nx: float,
                r_t: int, r_x: int, r_y: int,
                label: str = "projected") -> dict:
    """Compute a projection row given measured ranks."""
    nt = max(1, int(nt_per_nx * nx))
    ny = nx
    mem_d = dense_memory_bytes(nt, nx, ny)
    mem_t = tucker_memory_bytes(nt, nx, ny, r_t, r_x, r_y)
    return {
        "nx": nx, "nt": nt,
        "r_t": r_t, "r_x": r_x, "r_y": r_y,
        "memory_dense_mb": mem_d / (1024 ** 2),
        "memory_tucker_mb": mem_t / (1024 ** 2),
        "compression_ratio": mem_d / max(mem_t, 1e-30),
        "measurement": label,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--t-final", type=float, default=0.5)
    ap.add_argument("--snapshot-interval", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"scalability_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    params = TGVParams(Re=args.re)
    measured_rows = []
    last_ranks = None
    last_nt_per_nx = None

    print(f"\nScalability projection  Re={args.re}")
    print(f"{'nx':>6}  {'nt':>5}  {'ranks (t,x,y)':>14}  "
          f"{'dense MB':>10}  {'Tucker MB':>10}  {'ratio':>7}  {'note':>12}")
    print("-" * 75)

    for nx in sorted(args.grids):
        print(f"  [run] nx={nx} ...")
        t0 = time.perf_counter()
        traj = collect_trajectory(params, nx=nx, ny=nx,
                                  t_final=args.t_final,
                                  snapshot_interval=args.snapshot_interval)
        collect_time = time.perf_counter() - t0

        # max_rank is intentionally grid-independent: rank tracks flow
        # complexity, not resolution. Letting the cap grow with nx inflates
        # ranks far beyond what the flow needs (verified: rank (15,30,30)
        # reaches 1.4e-15 error even at nx=256).
        guided = associator_guided_ranks(traj, min_rank=4, max_rank=32)
        t0 = time.perf_counter()
        res = tucker_decompose_trajectory(traj, core_ranks=guided)
        decomp_time = time.perf_counter() - t0

        nt = len(traj.times)
        r_t, r_x, r_y = res["ranks"]
        mem_dense = traj.memory_dense_mb()
        mem_tucker = res["memory_compressed_mb"]
        ratio = res["compression_ratio"]

        last_ranks = (r_t, r_x, r_y)
        last_nt_per_nx = nt / nx

        row = {
            "nx": nx, "nt": nt,
            "r_t": r_t, "r_x": r_x, "r_y": r_y,
            "memory_dense_mb": mem_dense,
            "memory_tucker_mb": mem_tucker,
            "compression_ratio": ratio,
            "reconstruction_error": res["reconstruction_error"],
            "decomp_time_sec": decomp_time,
            "collect_time_sec": collect_time,
            "measurement": "measured",
        }
        measured_rows.append(row)
        print(f"  {nx:>6}  {nt:>5}  {str((r_t, r_x, r_y)):>14}  "
              f"  {mem_dense:>8.3f}    {mem_tucker:>8.3f}  {ratio:>7.1f}×  measured")

    # Analytical projections using ranks from the largest measured grid
    all_rows = list(measured_rows)
    if last_ranks is not None:
        r_t, r_x, r_y = last_ranks
        projected_grids = [g for g in [256, 512, 1024] if g > max(args.grids)]
        for nx in projected_grids:
            row = project_row(nx, last_nt_per_nx, r_t, r_x, r_y, label="projected")
            row["reconstruction_error"] = float("nan")
            row["decomp_time_sec"] = float("nan")
            row["collect_time_sec"] = float("nan")
            all_rows.append(row)
            print(f"  {nx:>6}  {row['nt']:>5}  {str((r_t, r_x, r_y)):>14}  "
                  f"  {row['memory_dense_mb']:>8.3f}    {row['memory_tucker_mb']:>8.3f}"
                  f"  {row['compression_ratio']:>7.1f}×  projected*")

    print()
    print("* Projections use ranks from the largest measured grid.")
    print("  Rank is a physics property (tracks flow complexity), not a grid property;")
    print("  it is expected to be stable under grid refinement for fixed physics.")
    print()

    write_metrics_csv(all_rows, out / "metrics.csv")
    write_summary_json({
        "re": args.re,
        "measured_grids": sorted(args.grids),
        "projected_grids": [g for g in [256, 512, 1024] if g > max(args.grids)],
        "last_ranks": list(last_ranks) if last_ranks else None,
    }, out / "summary.json")
    print(f"[scalability] wrote {len(all_rows)} rows to {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
