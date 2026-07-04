# Phase 1 Submission Package — Manifest

**Archive:** `airbus-octonion-tgv-solver_phase1_submission_20260704.zip`

**Challenge:** 2026 Global Quantum + AI Challenge — Airbus "Quantum Solvers: Enhancing Predictive Aerodynamic Modeling Capabilities"

**Submission Date:** 2026-07-04

---

## Contents

The package contains a standalone Phase 1 proposal pack with all evidence, figures, and reproduction details.

### Key Documents (Read in This Order)

1. **EXECUTIVE_SUMMARY.md** — One-page judge-facing summary with all measured results and caveats
2. **PHASE1_PROPOSAL_DRAFT.md** — Full 9-section proposal with detailed methodology, results, risks, and quantum pathway
3. **QUANTUM_HARDWARE_PATH.md** — Detailed qubit counts, circuit depth, and fault-tolerance requirements for the Tucker-to-TTN-to-quantum mapping

### Supporting Documentation

- **METHOD.md** — Algorithm descriptions and design choices
- **VALIDATION_PLAN.md** — Testing strategy and scope
- **AIRBUS_REQUIREMENTS_TRACE.md** — Alignment with challenge requirements
- **TECHNICAL_RISK_REGISTER.md** — Risks, mitigations, and status
- **UAV_ASSOCIATOR_TRANSFER.md** — Prior work and transfer learning notes

### Data and Figures

- **metrics.csv** — Complete benchmark results (Re sweep, grid convergence, method comparison)
- **summary.json** — Run metadata and parameters
- **config.yaml** — Solver configuration used for all measurements
- **figures/** — 8 publication-quality plots:
  - `associator_field.png` — Octonion associator torsion visualization
  - `associator_vs_error.png` — Steering factor vs L2 error correlation
  - `compression_vs_error.png` — Tucker memory vs reconstruction error Pareto front
  - `error_heatmap.png` — L2 error across Re and methods
  - `l2_vs_re.png` — Convergence vs Reynolds number
  - `memory_vs_re.png` — Storage requirements vs Reynolds number
  - `runtime_vs_re.png` — Wall-clock time vs Reynolds number
  - `velocity_final.png` — Final velocity field snapshot (Re=100, t=0.5)

---

## Key Results (at a Glance)

| Metric | Value | Evidence |
|--------|-------|----------|
| Solver convergence order | 1.99 (design 2.0) | 32²→128² grid refinement |
| Stable Reynolds range | Re = 10–2000 | 20× mandatory range |
| Compression @ nx=64 | 7.8× @ 1.1e-15 error | Measured |
| Compression @ nx=256 | **1010× @ 1.4e-15 error** | Measured (not projected) |
| Associator advantage (smooth TGV) | 1.22× vs uniform rank | Measured at matched error |
| Associator advantage (perturbed TGV) | **1.73× at 10% complexity** | Directional validation |
| Quantum hardware requirement | ~50,000 physical qubits | Fault-tolerant surface code |
| CFL steering net advantage | **None on this benchmark** | Honest ablation result |

---

## Reproducibility

All numbers in the proposal regenerate from the companion source code repository (not included in this package; available at https://github.com/[user]/airbus-octonion-tgv-solver or equivalent).

Scripts that produced each result:

1. **Solver verification, Re sweep, grid convergence**
   ```
   python scripts\run_reynolds_sweep.py --config configs\reynolds_sweep.yaml
   ```

2. **Compression advantage study (dense vs Tucker baselines)**
   ```
   python scripts\run_advantage_study.py --reynolds 100 --nx 64
   ```

3. **Scalability measured through nx=256**
   ```
   python scripts\run_scalability_projection.py --re 100 --grids 32 64 128 256
   ```

4. **Complexity scaling with perturbed initial conditions**
   ```
   python scripts\run_perturbed_advantage_study.py --re 100 --nx 64 --amplitudes 0.1 0.3
   ```

5. **Automated test suite**
   ```
   python -m pytest
   ```
   Result: 48/48 pass

---

## Technical Approach (Summary)

**Baseline:** 2nd-order cell-centered finite-volume solver with spectral pressure projection, validated against exact analytical TGV solution.

**Novel layer:** Octonion-associator-guided tensor compression. Flow states embedded as 8-component octonions; associator torsion (a measure of non-associativity) used to steer:
- Per-step CFL safety factor (adaptive time stepping)
- Per-mode Tucker rank selection (compression allocation)
- Space-time trajectory storage via Tucker decomposition

**Measurement approach:** Three-way comparison (classical dense storage, uniform fixed-rank Tucker, associator-guided asymmetric-rank Tucker) at matched reconstruction error.

**Quantum pathway:** Tucker decomposition maps to tree tensor network state (TTN); standard log₂ qubit encoding yields ~50–60 logical qubits at measured ranks, growing logarithmically with grid size while classical storage grows linearly.

---

## Quality Assurance

- ✅ 48 automated tests (unit + integration)
- ✅ Measured through nx=256 (not extrapolated from smaller grids)
- ✅ Rank saturation verified by direct decomposition
- ✅ Honest ablation: CFL steering advantage did not survive control experiment
- ✅ Caveat: TGV is spectrally favorable to low-rank methods; perturbed-flow study measures degradation direction
- ✅ All numbers regenerate from documented scripts on standard CPU (no GPU, no cluster, no external data required)

---

## Questions?

Refer to the EXECUTIVE_SUMMARY.md for the quick version, PHASE1_PROPOSAL_DRAFT.md for full details, and the source repository (not in this package) for code.

