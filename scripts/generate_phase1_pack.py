"""Assemble a 'Phase 1 proposal pack': aggregated metrics + key docs into one folder."""
from __future__ import annotations
import argparse
import shutil
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / f"phase1_pack_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    docs = ROOT / "docs"
    for name in ("EXECUTIVE_SUMMARY.md", "PHASE1_PROPOSAL_DRAFT.md",
                 "METHOD.md", "VALIDATION_PLAN.md",
                 "AIRBUS_REQUIREMENTS_TRACE.md", "TECHNICAL_RISK_REGISTER.md",
                 "UAV_ASSOCIATOR_TRANSFER.md", "QUANTUM_HARDWARE_PATH.md"):
        p = docs / name
        if p.exists():
            shutil.copy2(p, out / name)

    runs = sorted((ROOT / "results").glob("run_*"))
    if runs:
        latest = runs[-1]
        for fname in ("metrics.csv", "summary.json", "config.yaml"):
            src = latest / fname
            if src.exists():
                shutil.copy2(src, out / fname)
        fig_dir = latest / "figures"
        if fig_dir.exists():
            shutil.copytree(fig_dir, out / "figures", dirs_exist_ok=True)
    print(f"[phase1-pack] assembled at {out}")


if __name__ == "__main__":
    main()
