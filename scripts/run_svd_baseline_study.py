"""The reviewer's test: associator-guided ranks vs standard adaptive SVD.

Standard baseline: HOSVD energy truncation. For each mode, unfold the
trajectory tensor, compute singular values, and keep the smallest rank whose
retained energy fraction exceeds (1 - tol). This is the textbook classical
adaptive method the octonion layer must beat (or match at lower cost) to
justify its existence.

For each flow (smooth TGV + perturbed variants) we report, at matched
reconstruction error:
  - ranks, memory, error, and rank-selection wall time for both methods.

Honest reporting: whatever the outcome, it ships.
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


def hosvd_energy_ranks(traj, tol: float) -> dict:
    """Standard classical baseline: per-mode SVD energy truncation.

    Rank per mode = smallest r with sum(s[:r]^2) >= (1 - tol) * sum(s^2),
    computed on the u-velocity trajectory tensor unfoldings (v gives the
    same ranks for this flow to within 1).
    """
    U, V = traj.to_tensor()
    ranks = {}
    for axis, name in enumerate(("time", "x", "y")):
        unfolding = np.moveaxis(U, axis, 0).reshape(U.shape[axis], -1)
        s = np.linalg.svd(unfolding, compute_uv=False)
        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        r = int(np.searchsorted(energy, 1.0 - tol) + 1)
        ranks[name] = max(1, min(r, U.shape[axis]))
    return ranks


def match_tolerance(traj, target_error: float, tols) -> tuple:
    """Find the loosest HOSVD tolerance whose Tucker error <= target_error."""
    best = None
    for tol in sorted(tols, reverse=True):  # loosest first
        t0 = time.perf_counter()
        ranks = hosvd_energy_ranks(traj, tol)
        select_time = time.perf_counter() - t0
        res = tucker_decompose_trajectory(traj, core_ranks=ranks)
        if res["reconstruction_error"] <= target_error:
            return tol, ranks, res, select_time
        best = (tol, ranks, res, select_time)
    return best  # tightest attempted, even if it missed


def study(label, traj):
    # associator-guided
    t0 = time.perf_counter()
    guided_ranks = associator_guided_ranks(traj, min_rank=4, max_rank=32)
    guided_select_time = time.perf_counter() - t0
    guided = tucker_decompose_trajectory(traj, core_ranks=guided_ranks)

    # standard adaptive SVD, error-matched to the guided result
    tols = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20,
            1e-22, 1e-24, 1e-26, 1e-28, 1e-30]
    tol, svd_ranks, svd_res, svd_select_time = match_tolerance(
        traj, guided["reconstruction_error"], tols)

    row = {
        "flow": label,
        "guided_ranks": str(guided["ranks"]),
        "guided_memory_mb": guided["memory_compressed_mb"],
        "guided_error": guided["reconstruction_error"],
        "guided_select_time_sec": guided_select_time,
        "svd_tol": tol,
        "svd_ranks": str(svd_res["ranks"]),
        "svd_memory_mb": svd_res["memory_compressed_mb"],
        "svd_error": svd_res["reconstruction_error"],
        "svd_select_time_sec": svd_select_time,
        "memory_ratio_svd_over_guided":
            svd_res["memory_compressed_mb"] / max(guided["memory_compressed_mb"], 1e-30),
        "select_time_ratio_guided_over_svd":
            guided_select_time / max(svd_select_time, 1e-30),
    }
    print(f"[{label}]")
    print(f"  guided : ranks {guided['ranks']}  {guided['memory_compressed_mb']:.4f} MB "
          f"@ {guided['reconstruction_error']:.2e}  (select {guided_select_time*1000:.0f} ms)")
    print(f"  svd    : ranks {svd_res['ranks']}  {svd_res['memory_compressed_mb']:.4f} MB "
          f"@ {svd_res['reconstruction_error']:.2e}  (tol {tol:.0e}, select {svd_select_time*1000:.0f} ms)")
    print(f"  memory svd/guided: {row['memory_ratio_svd_over_guided']:.2f}x | "
          f"select-time guided/svd: {row['select_time_ratio_guided_over_svd']:.2f}x")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--t-final", type=float, default=0.5)
    ap.add_argument("--snapshot-interval", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"svd_baseline_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    params = TGVParams(Re=args.re)
    rows = []

    print("collecting smooth TGV trajectory...")
    traj = collect_trajectory(params, nx=args.nx, ny=args.nx,
                              t_final=args.t_final,
                              snapshot_interval=args.snapshot_interval)
    rows.append(study("smooth_tgv", traj))

    for amp in (0.1, 0.3):
        print(f"collecting perturbed trajectory (amp={amp})...")
        ic = perturbed_tgv_initial_condition(n_modes=4, amplitude=amp, seed=0)
        traj_p = collect_trajectory(params, nx=args.nx, ny=args.nx,
                                    t_final=args.t_final,
                                    snapshot_interval=args.snapshot_interval,
                                    initial_condition=ic)
        rows.append(study(f"perturbed_amp{amp}", traj_p))

    write_metrics_csv(rows, out / "metrics.csv")
    write_summary_json({"args": vars(args)}, out / "summary.json")
    print(f"\n[svd-baseline] wrote {len(rows)} rows to {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
