# Airbus Octonion-TGV Solver

Quantum-inspired finite-volume solver workspace for the 2026 Global Quantum + AI Challenge Airbus problem statement, "Quantum Solvers: Enhancing Predictive Aerodynamic Modeling Capabilities."

Output label required for generated scientific outputs:

`Experimental research output – not a validated engineering deliverable.`

No external CFD data is required. The benchmark truth is the exact analytical Convecting Taylor-Green Vortex solution, and every metric is generated from local runs.

## Challenge Alignment

| Airbus requirement | Code |
| --- | --- |
| Working 2D Convecting TGV solver | `src/airbus_tgv/finite_volume.py` |
| FV cell averages and face fluxes | `src/airbus_tgv/finite_volume.py`, `src/airbus_tgv/fluxes.py` |
| Exact analytical solution | `src/airbus_tgv/exact_solution.py` |
| Re = 10, 100, and higher sweeps | `scripts/run_reynolds_sweep.py`, `configs/reynolds_sweep.yaml` |
| Runtime, memory, and error scaling | `src/airbus_tgv/benchmark.py` |
| L2 error and kinetic energy decay | `src/airbus_tgv/exact_solution.py`, `src/airbus_tgv/benchmark.py` |
| Classical baseline | `classical_fv` method |
| Tensor/FV route | `src/airbus_tgv/compression.py` |
| Fano-plane octonion associator | `src/airbus_tgv/fano.py`, `src/airbus_tgv/octonion.py` |

## Install

Windows PowerShell:

```powershell
cd D:\Projects\quantum_challenge\airbus_claude\airbus-octonion-tgv-solver
python -m pip install -e .
```

Optional TensorLy support:

```powershell
python -m pip install -e ".[tensor]"
```

## Run Tests

```powershell
python -m pytest
```

## Run One Benchmark

```powershell
python scripts\run_single.py --re 10 --nx 32 --t-final 0.1 --method fv_plus_associator_guided_compression
```

Outputs are written to `results\run_<timestamp>\metrics.csv`, `summary.json`, and `config.yaml`.

## Run Reynolds Sweep

```powershell
python scripts\run_reynolds_sweep.py --config configs\reynolds_sweep.yaml
```

## Regenerate Figures

```powershell
python scripts\make_figures.py --run-dir results\run_<timestamp>
```

Figures are generated from actual metrics and a fresh finite-volume field run. The script does not create fake plots.

## Phase 1 Pack

```powershell
python scripts\generate_phase1_pack.py
```

## What Is Implemented

- Exact periodic Taylor-Green velocity, pressure, kinetic energy, L2 error, and FFT divergence diagnostics.
- Cell-centered finite-volume solver with face states, convective fluxes, diffusive fluxes, RK2 stepping, CFL control, periodic boundaries, and FFT pressure projection.
- Classical baseline method.
- Real octonion multiplication using fixed Fano-plane triples.
- Real associator norm `||(a*b)*c - a*(b*c)||`.
- Flow-to-octonion embeddings using velocity, pressure or vorticity, strain, and velocity-gradient channels.
- Associator summaries, compression masks, associator-steered CFL safety factor, and associator-guided rank policy.
- SVD compression on real solver fields; TensorLy Tucker support only when TensorLy is installed.
- Divergence-free multi-mode perturbed initial conditions (`perturbation.py`) for testing beyond the spectrally simple TGV.
- Benchmark metrics and figure generation.
- Requirements trace, method document, validation plan, risk register, proposal draft, and UAV transfer note.

## Measured Quantum-Inspired Advantage (Trajectory Compression)

The space-time trajectory pathway (`src/airbus_tgv/trajectory.py`,
`scripts/run_advantage_study.py`) stores the full (time, x, y) solution tensor
in Tucker-compressed form. Measured on this machine with the 2nd-order central
flux (Re sweep, nx=64, t_final=0.5, results in `results/advantage_*/metrics.csv`):

| Re | Dense trajectory | Associator-guided Tucker | Blind fixed-rank Tucker |
| --- | --- | --- | --- |
| 10 | 20.44 MB | 155× smaller @ 3.6e-12 error | 323× smaller @ 1.4e-8 error |
| 100 | 2.13 MB | **7.8× smaller @ 1.1e-15 error** | 77× smaller @ 1.8e-6 error |
| 500 | 1.25 MB | **5.9× smaller @ 1.5e-10 error** | 48× smaller @ 3.8e-6 error |
| 2000 | 1.25 MB | **5.9× smaller @ 1.6e-10 error** | 48× smaller @ 4.0e-6 error |

The Re=10 row's 155× figure is inflated by TGV's single dominant Fourier mode
at low Reynolds; at nx=64 the conservative claim is **7.8×** at machine
precision across the industrially relevant Re=100–2000 range. The ratio grows
rapidly with grid size because guided ranks saturate at (14, 29, 29) from
nx=128 onward while dense storage grows as O(n³): **measured 102× at nx=128,
1010× at nx=256, and 4880× at nx=512 (4.18 GB → 0.86 MB), all at ~1e-15 error**
(`scripts\run_scalability_projection.py`; the 512² summary is in
`results\scalability_512_measured\`). The 323× figure for blind
fixed-rank at Re=10 is real but misleading as a headline number.

At matched reconstruction error, the associator-guided asymmetric rank
allocation uses 1.22–1.73× less memory than the best **uniform** fixed rank
(`scripts\run_perturbed_advantage_study.py`). However, the standard classical
adaptive baseline — HOSVD singular-value energy truncation — matches or
slightly beats the associator guidance at matched error with ~13× faster rank
selection (`scripts\run_svd_baseline_study.py`). **No advantage over standard
adaptive SVD is claimed**; the associator layer must demonstrate unique value
in Phase 2 (POD baselines, heterogeneous flows) or be retired. The
Tucker-vs-dense compression numbers above are independent of the
rank-selection method and stand either way.

In the time-stepping sweep, the associator-guided method achieves roughly 10%
lower L2 velocity error than the classical baseline at every Re ≥ 100 — but a
control experiment (classical solver with a uniform cfl=0.362) reproduces the
same accuracy at Re ≥ 500 at ~60× lower runtime. The steering gain survives
the control only at Re=100 (0.00162 vs 0.00178). We report this as an honest
ablation: **no net steering advantage is claimed on this benchmark**; the
associator's measured value is in compression rank selection.

What this does and does not show:

- **What is measured:** Tucker compression of TGV trajectories reduces storage
  6–8× at machine-precision reconstruction error vs dense classical storage,
  implementing the Airbus "Tensor Network within Finite Volume" route. Ranks
  are automatically selected by the associator field.
- **What the associator adds (two honest ablations):** guided rank allocation
  beats uniform fixed-rank (1.22–1.73×) but does not beat standard HOSVD
  energy truncation at matched error. The CFL steering gain did not survive a
  constant-CFL control except at Re=100. Neither associator claim is made;
  the layer is retained as an experimental diagnostic pending Phase 2
  evidence.
- **What is NOT claimed:** no quantum hardware speedup. The Re=10 compression
  ratio should not be cited as representative — TGV's spectral simplicity
  flatters all low-rank methods at low Re. The ROM speedup (`rom.py`, ~1.2×)
  is an operation-count estimate, not a measured wall-clock result.

Reproduce with:

```powershell
python scripts\run_trajectory_compression.py --re 100 --nx 64 --t-final 0.5
python scripts\run_advantage_study.py
```

## Solver Accuracy and Stability (Measured)

With the default 2nd-order central flux (`lf_coeff=0.0`; physical viscosity
provides stabilization for the smooth TGV):

- **Grid convergence at Re=100**: L2 error 0.00602 (32²) → 0.00178 (64²) →
  0.00045 (128²); observed order **1.99** (design order 2 confirmed).
- **Stable at Re = 10 through 2000** at nx=64 — 20× beyond the mandatory
  Re=100 — with L2 ≤ 0.0035 everywhere.
- Full Lax-Friedrichs dissipation (`lf_coeff=1.0`) remains available for
  non-smooth flows; it is first-order (measured order 1.01) and was the
  source of the earlier larger errors.

## What Is Not Yet Implemented

- A high-order (≥3rd) scheme such as WENO/DG; current design order is 2.
- A quantum hardware or quantum simulator backend.
- A measured (wall-clock) ROM speedup; only an operation-count estimate.

## Associator/Fano-Plane Role

Each cell is embedded as an 8-component octonion, for example:

`[1, u, v, p_or_vorticity, du_dx, du_dy, dv_dx, dv_dy]`

The Fano-plane multiplication table defines the non-associative octonion product. The associator field is computed over local cell and neighbor triples as a relational torsion diagnostic. In `fv_plus_associator_guided_compression`, it actively steers the CFL time-step factor and compression rank. This is the control-angle advantage hypothesis: quantum-inspired non-associative geometry allocates numerical caution and memory where local flow relationships are more complex.

## Why This Fits the Airbus FV/Tensor Route

The Airbus statement explicitly allows Tensor Network methods inside a Finite Volume framework. This workspace uses real FV cell averages and face fluxes, then applies tensor-inspired SVD compression and optional Tucker decomposition to real solver fields. The associator layer is an additional quantum-inspired geometric diagnostic for compression and structure detection.

## Scientific Honesty

This project should be described as a credible path to quantum-inspired control advantage only when metrics support that wording. Do not state "quantum advantage achieved" unless measured runtime, memory, or error tradeoffs beat the classical baseline under matched conditions.
