"""Perturbed-flow advantage study: does associator guidance grow with flow
complexity?

Compares the smooth TGV against multi-mode perturbed TGV flows. For each flow:
  1. Collect the (time, x, y) trajectory.
  2. Tucker-decompose with associator-guided ranks; record memory and error.
  3. Sweep uniform ranks r=2..max and find the smallest uniform rank whose
     reconstruction error <= guided error (error-matched head-to-head).
  4. Report the memory ratio (uniform / guided) at matched error.

If the associator hypothesis holds, the guided-vs-uniform memory advantage
should grow as the flow gains multi-mode structure, because asymmetric rank
allocation matters more when mode content differs across tensor directions.

Reconstruction error is measured against the trajectory itself, so no exact
solution is required for the perturbed flows.
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
from airbus_tgv.perturbation import perturbed_tgv_initial_condition
from airbus_tgv.trajectory import (
    collect_trajectory, associator_guided_ranks, tucker_decompose_trajectory,
)
from airbus_tgv.reporting import write_metrics_csv, write_summary_json


def error_matched_uniform(traj, target_error: float, max_rank: int) -> dict:
    """Smallest uniform rank whose reconstruction error <= target_error."""
    best = None
    for r in range(2, max_rank + 1):
        res = tucker_decompose_trajectory(
            traj, core_ranks={"time": r, "x": r, "y": r})
        if res["reconstruction_error"] <= target_error:
            best = {"rank": r,
                    "memory_mb": res["memory_compressed_mb"],
                    "reconstruction_error": res["reconstruction_error"]}
            break
    if best is None:
        res = tucker_decompose_trajectory(
            traj, core_ranks={"time": max_rank, "x": max_rank, "y": max_rank})
        best = {"rank": max_rank,
                "memory_mb": res["memory_compressed_mb"],
                "reconstruction_error": res["reconstruction_error"],
                "note": "max rank reached without matching error"}
    return best


def study_flow(label: str, params: TGVParams, nx: int, t_final: float,
               snapshot_interval: int, initial_condition=None) -> dict:
    print(f"[perturbed-study] {label}: collecting trajectory...")
    traj = collect_trajectory(params, nx=nx, ny=nx, t_final=t_final,
                              snapshot_interval=snapshot_interval,
                              initial_condition=initial_condition)
    mem_dense = traj.memory_dense_mb()

    guided_ranks = associator_guided_ranks(traj, min_rank=4,
                                           max_rank=max(32, nx // 2))
    t0 = time.perf_counter()
    guided = tucker_decompose_trajectory(traj, core_ranks=guided_ranks)
    guided_time = time.perf_counter() - t0

    matched = error_matched_uniform(traj, guided["reconstruction_error"],
                                    max_rank=nx // 2)

    ratio = matched["memory_mb"] / max(guided["memory_compressed_mb"], 1e-30)
    row = {
        "flow": label,
        "Re": params.Re,
        "nx": nx,
        "n_snapshots": len(traj.times),
        "memory_dense_mb": mem_dense,
        "guided_ranks": str(guided["ranks"]),
        "guided_memory_mb": guided["memory_compressed_mb"],
        "guided_error": guided["reconstruction_error"],
        "guided_compression_ratio": guided["compression_ratio"],
        "matched_uniform_rank": matched["rank"],
        "matched_uniform_memory_mb": matched["memory_mb"],
        "matched_uniform_error": matched["reconstruction_error"],
        "guided_vs_uniform_memory_advantage": ratio,
        "decomp_time_sec": guided_time,
        "notes": matched.get("note", ""),
    }
    print(f"  guided {guided['ranks']}: {guided['memory_compressed_mb']:.3f} MB "
          f"@ {guided['reconstruction_error']:.2e} | matched uniform r="
          f"{matched['rank']}: {matched['memory_mb']:.3f} MB @ "
          f"{matched['reconstruction_error']:.2e} | advantage {ratio:.2f}x")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--t-final", type=float, default=0.5)
    ap.add_argument("--snapshot-interval", type=int, default=2)
    ap.add_argument("--amplitudes", type=float, nargs="+",
                    default=[0.1, 0.3])
    ap.add_argument("--n-modes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"perturbed_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    params = TGVParams(Re=args.re)
    rows = [study_flow("smooth_tgv", params, args.nx, args.t_final,
                       args.snapshot_interval)]
    for amp in args.amplitudes:
        ic = perturbed_tgv_initial_condition(n_modes=args.n_modes,
                                             amplitude=amp, seed=args.seed)
        rows.append(study_flow(f"perturbed_amp{amp}", params, args.nx,
                               args.t_final, args.snapshot_interval,
                               initial_condition=ic))

    write_metrics_csv(rows, out / "metrics.csv")
    write_summary_json({"args": vars(args), "n_flows": len(rows)},
                       out / "summary.json")
    print(f"[perturbed-study] wrote {len(rows)} rows to {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
