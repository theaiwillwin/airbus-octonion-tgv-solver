"""Run a Reynolds-number sweep across methods and grids."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbus_tgv.constants import TGVParams
from airbus_tgv.benchmark import run_method
from airbus_tgv.reporting import write_metrics_csv, write_summary_json, write_config_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "reynolds_sweep.yaml"))
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["sweep"]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for Re in cfg["reynolds"]:
        for nx in cfg["grids"]:
            for method in cfg["methods"]:
                params = TGVParams(Re=float(Re))
                print(f"[sweep] Re={Re} nx={nx} method={method}")
                try:
                    r = run_method(method, params, nx, nx,
                                   t_final=cfg["t_final"], cfl=cfg["cfl"])
                    rows.append(r.to_row())
                except Exception as exc:
                    rows.append({
                        "method": method,
                        "output_label": "Experimental research output – not a validated engineering deliverable.",
                        "Re": float(Re), "nx": nx, "ny": nx,
                        "stable": False, "notes": f"ERROR: {exc}",
                        "dt": 0, "dt_mean": 0, "n_steps": 0, "final_time": 0,
                        "runtime_sec": 0, "memory_dense_mb": 0,
                        "memory_compressed_mb": 0, "compression_ratio": 1.0,
                        "compression_rel_err": 0, "l2_velocity_error": float("nan"),
                        "kinetic_energy_num": 0, "kinetic_energy_exact": 0,
                        "kinetic_energy_rel_error": float("nan"),
                        "divergence_l2": float("nan"),
                        "associator_mean": 0, "associator_p95": 0, "associator_max": 0,
                        "associator_dt_factor_mean": 1.0,
                        "associator_dt_factor_min": 1.0,
                    })

    write_metrics_csv(rows, out / "metrics.csv")
    write_summary_json({"n_runs": len(rows), "config": cfg}, out / "summary.json")
    write_config_yaml(cfg, out / "config.yaml")
    print(f"[sweep] wrote {len(rows)} rows to {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
