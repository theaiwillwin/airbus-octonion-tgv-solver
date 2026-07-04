"""Regenerate figures from the most recent sweep results."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbus_tgv.constants import TGVParams
from airbus_tgv.finite_volume import run_fv_solver
from airbus_tgv.exact_solution import exact_velocity
from airbus_tgv.associator_metrics import associator_field
from airbus_tgv import plotting as pl


def _latest_run_dir() -> Path:
    runs = sorted((ROOT / "results").glob("run_*"))
    if not runs:
        raise SystemExit("No results/run_* directories found; run a sweep first.")
    return runs[-1]


def _load_rows(metrics_csv: Path):
    with open(metrics_csv) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        rows = _load_rows(metrics_csv)
        for r in rows:
            for k, v in r.items():
                try:
                    if k in ("method", "notes"):
                        continue
                    r[k] = float(v)
                except (TypeError, ValueError):
                    continue
        pl.plot_metric_vs_re(rows, "l2_velocity_error",
                             fig_dir / "l2_vs_re.png", "L2 velocity error", log=True)
        pl.plot_metric_vs_re(rows, "runtime_sec",
                             fig_dir / "runtime_vs_re.png", "runtime (s)")
        pl.plot_metric_vs_re(rows, "memory_compressed_mb",
                             fig_dir / "memory_vs_re.png", "compressed memory (MB)")
        pl.plot_compression_vs_error(rows, fig_dir / "compression_vs_error.png")

    # Always also generate a fresh illustrative single-run set of field figures.
    params = TGVParams(Re=100.0)
    res = run_fv_solver(params, nx=64, ny=64, t_final=0.5)
    X = res.diagnostics["X"]; Y = res.diagnostics["Y"]
    dx, dy = res.diagnostics["dx"], res.diagnostics["dy"]
    u_ex, v_ex = exact_velocity(X, Y, res.t, params)
    A = associator_field(res.u, res.v, None, dx, dy, mode="strain")["A"]

    pl.plot_velocity_quiver(X, Y, res.u, res.v, fig_dir / "velocity_final.png",
                            title=f"u at t={res.t:.2f}, Re={params.Re}")
    pl.plot_error_heatmap(res.u, res.v, u_ex, v_ex, fig_dir / "error_heatmap.png")
    pl.plot_associator_field(A, fig_dir / "associator_field.png")
    pl.plot_associator_vs_error(A, res.u, res.v, u_ex, v_ex,
                                fig_dir / "associator_vs_error.png")
    print(f"[figures] wrote figures into {fig_dir}")


if __name__ == "__main__":
    main()
