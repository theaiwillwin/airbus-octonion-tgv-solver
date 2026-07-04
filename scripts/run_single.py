"""Run a single benchmark configuration."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys

# Make src/ importable without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbus_tgv.constants import TGVParams
from airbus_tgv.benchmark import run_method
from airbus_tgv.reporting import write_metrics_csv, write_summary_json, write_config_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--t-final", type=float, default=0.5)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--method", type=str, default="classical_fv")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ny = args.ny if args.ny is not None else args.nx
    params = TGVParams(Re=args.re)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    res = run_method(args.method, params, args.nx, ny, args.t_final, cfl=args.cfl)
    row = res.to_row()
    write_metrics_csv([row], out / "metrics.csv")
    write_summary_json(row, out / "summary.json")
    write_config_yaml({"Re": args.re, "nx": args.nx, "ny": ny,
                       "t_final": args.t_final, "cfl": args.cfl,
                       "method": args.method}, out / "config.yaml")
    print(f"[run_single] wrote results to {out}")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
