"""Benchmark trajectory compression: full-fidelity solve vs Tucker+ROM.

Collect a trajectory, decompose it via Tucker with associator-guided ranks,
and benchmark memory/accuracy gains.
"""
from __future__ import annotations
import argparse
import time
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbus_tgv.constants import TGVParams
from airbus_tgv.trajectory import (
    collect_trajectory, associator_guided_ranks, tucker_decompose_trajectory,
    trajectory_kinetic_energy_comparison,
)
from airbus_tgv.rom import rom_speedup_estimate
from airbus_tgv.reporting import write_summary_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--t-final", type=float, default=0.5)
    ap.add_argument("--snapshot-interval", type=int, default=5,
                    help="Collect every Nth timestep")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    params = TGVParams(Re=args.re)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"trajectory_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[trajectory] collecting Re={args.re} nx={args.nx} t_final={args.t_final}")
    t0 = time.perf_counter()
    traj = collect_trajectory(params, nx=args.nx, ny=args.nx, t_final=args.t_final,
                              snapshot_interval=args.snapshot_interval)
    collect_time = time.perf_counter() - t0
    print(f"[trajectory] collected {len(traj.times)} snapshots in {collect_time:.2f} s")

    # Associator-guided rank selection
    print("[trajectory] computing associator-guided ranks...")
    ranks = associator_guided_ranks(traj, min_rank=4, max_rank=32)
    print(f"[trajectory] selected ranks: {ranks}")

    # Tucker decomposition
    print("[trajectory] Tucker decomposition...")
    try:
        t0 = time.perf_counter()
        result = tucker_decompose_trajectory(traj, core_ranks=ranks)
        decomp_time = time.perf_counter() - t0
        print(f"[trajectory] decomposition completed in {decomp_time:.2f} s")
    except RuntimeError as e:
        print(f"[trajectory] ERROR: {e}")
        print("[trajectory] (install tensorly with: pip install tensorly)")
        result = None

    # Kinetic energy comparison
    print("[trajectory] computing kinetic energy comparison...")
    ke_comp = trajectory_kinetic_energy_comparison(traj, params)

    # Assemble output
    output = {
        "Re": args.re,
        "nx": args.nx,
        "t_final": args.t_final,
        "n_snapshots": len(traj.times),
        "collect_time_sec": collect_time,
        "decomp_time_sec": decomp_time if result else 0.0,
        "ranks": ranks,
        "compression_ratio": result["compression_ratio"] if result else 0.0,
        "reconstruction_error": result["reconstruction_error"] if result else 0.0,
        "memory_dense_mb": result["memory_dense_mb"] if result else 0.0,
        "memory_compressed_mb": result["memory_compressed_mb"] if result else 0.0,
        "ke_error_mean": ke_comp["ke_error_mean"],
        "ke_error_max": ke_comp["ke_error_max"],
    }

    if result:
        output["rom_speedup_estimate"] = rom_speedup_estimate(
            result["factors_u"], (len(traj.times), args.nx, args.nx)
        )

    write_summary_json(output, out / "summary.json")
    print(f"[trajectory] wrote results to {out}")
    for k, v in output.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
