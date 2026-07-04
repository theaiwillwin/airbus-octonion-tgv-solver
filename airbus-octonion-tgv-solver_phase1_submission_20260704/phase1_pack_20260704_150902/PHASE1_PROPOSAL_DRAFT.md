# Phase 1 Proposal Draft

## 1. Problem Framing

Airbus seeks more efficient PDE solvers for aerodynamic flow prediction, where conventional high-performance solvers face scalability pressure and physical testing remains expensive near demanding operating regimes. We address the required 2D Convecting Taylor-Green Vortex benchmark because it has an exact analytical solution and therefore supports rigorous error, runtime, memory, and kinetic-energy validation.

## 2. Technical Approach

We propose a quantum-inspired finite-volume solver workspace. The baseline is a real periodic cell-centered finite-volume solver with face fluxes and FFT pressure projection. The benchmark truth is the exact Taylor-Green solution, so no external CFD dataset is required.

The novel layer is an octonionic Fano-plane associator controller. Local cell, face-neighbor, and directional flow states are embedded into 8-component octonion vectors containing velocity, vorticity or pressure, and velocity-gradient channels. The exact non-associative associator norm `||(a*b)*c - a*(b*c)||` is computed as a relational torsion diagnostic.

The advantage hypothesis is a control and steering angle: the non-associative associator field steers numerical resources by reducing the CFL time step in high-torsion states and selecting compression rank where the vortex-interface geometry is more complex. This keeps the FV physics intact while using quantum-inspired algebra as a stability and compression controller.

Compression is implemented at two levels. Per-snapshot: a truncated SVD baseline with associator-guided rank policy. Full space-time: the (time, x, y) solution tensor is stored in Tucker-decomposed form (TensorLy), with per-mode ranks selected automatically from associator torsion statistics — the Tensor-Network-within-Finite-Volume pathway named in the challenge statement. The submitted evidence compares classical finite volume, associator diagnostics, associator-guided compression, and blind fixed-rank tensor baselines.

## 3. Feasibility and Resource Requirements

The current implementation runs locally on a normal Windows CPU using NumPy, SciPy-compatible FFT operations, pandas, matplotlib, PyYAML, pytest, psutil, and tqdm. The explicit solver is intentionally lightweight for Phase 1 reproducibility. It supports Re = 10 and Re = 100 as mandatory cases and attempts higher Re values where the time step and grid resolution remain stable.

## 4. Expected Impact and Measured Results

We present a verified 2nd-order finite-volume baseline, then report preliminary evidence that octonion-associator-guided tensor compression and CFL adaptation provide measurable — though necessarily modest on the spectrally clean TGV benchmark — advantages over naive alternatives, with a clear path to rigorous validation against proper baselines in Phase 2.

All results are locally reproducible against the analytical benchmark (2nd-order central-flux solver, nx=64, t_final=0.5).

1. **Solver verification (strongest result).** The underlying FV solver achieves theoretical convergence order (1.99 measured, 32² → 128², Re=100), divergence L2 bounded near 1e-7 by FFT pressure projection, and stable operation across Re = 10–2000 (20× the mandatory range). All 48 automated tests pass. This verification baseline ensures any compression or steering effects are measured against a credible classical solver.

2. **Compression on the prescribed benchmark, measured through nx=256.** Tucker decomposition achieves 7.8× storage reduction at machine-precision reconstruction error at the baseline nx=64 grid, growing to a **measured 1010× at nx=256** (523 MB → 0.52 MB at 1.4e-15 error) because the guided ranks saturate at (14, 29, 29) while dense storage grows as O(n³) — see Section 9. TGV's spectral narrowness makes it favorable to low-rank storage, and we state this plainly; the perturbed-flow study (result 3) measures how the method degrades — gracefully — as mode content is added. The Tensor Network within Finite Volume pathway named in the challenge statement is implemented and verified.

3. **Associator rank allocation adapts to physics — and its advantage grows with flow complexity.** At matched reconstruction error, the associator-guided asymmetric rank allocation outperforms uniform-rank Tucker by 1.22× on the smooth TGV, rising to **1.73× on a multi-mode perturbed TGV**. All runs at Re=100, nx=64, t_final=0.5, with divergence-free streamfunction perturbations: at 10% amplitude, guided ranks (13, 26, 26) give 11.1× compression at 8.0e-8 error (advantage 1.73×); at 30% amplitude, ranks (11, 22, 22) give 16.4× at 5.3e-6 (advantage 1.67×). Compression remains substantial on the harder flows — the guided policy trades a modest error increase for lower rank as mode content grows, exactly the adaptive behaviour claimed. Reproduce with `scripts/run_perturbed_advantage_study.py`. This directly addresses the concern that TGV's spectral simplicity masks the method's value: the guided allocation matters *more* as the flow gains structure. Spatial ranks also adapt automatically to Reynolds number (19 → 31, Re=10 → 2000) without manual tuning. Comparisons against POD and adaptive SVD are planned for Phase 2.

4. **Associator CFL steering: honest ablation result.** Associator-adaptive time stepping yields approximately 10% lower L2 velocity error than the classical baseline at every Re ≥ 100. However, our own control experiment shows that a uniform 10% CFL reduction (cfl=0.362, no associator) reproduces the same accuracy at Re ≥ 500 at roughly 60× lower runtime; the adaptive factor genuinely outperforms the constant control only at Re=100 (0.00162 vs 0.00178). We therefore do not claim a net steering advantage on this benchmark. The associator's measured contribution is rank selection (result 3); whether its steering value grows on flows with genuine spatial heterogeneity — where a constant factor cannot adapt — is a Phase 2 question.

## 5. Validation Plan

Validation records time-to-solution, memory estimates, L2 velocity error, kinetic-energy decay, divergence norm, associator statistics, compression ratio, and compression error. The mandatory Re = 10 and Re = 100 runs are complete and extended through Re = 2000, with grid refinement (32², 64², 128²; observed order 1.99) and ablations across baseline FV, FV plus associator diagnostic, FV plus associator-guided compression, and blind fixed-rank tensor baselines. All 48 unit and integration tests pass; every reported number regenerates from `scripts/run_reynolds_sweep.py`, `scripts/run_advantage_study.py`, `scripts/run_trajectory_compression.py`, `scripts/run_scalability_projection.py`, and `scripts/run_perturbed_advantage_study.py`.

## 6. Hybrid / Cross-Domain Integration

The backend separates PDE solving, octonion embedding, associator diagnostics, compression, benchmarking, and plotting into independent modules. The octonion algebra layer is domain-agnostic: a future adapter would supply a different flow-to-octonion embedding while reusing the same Fano-plane multiplication, associator computation, and Tucker compression pipeline. Within aerodynamics, the natural extension is from smooth single-mode TGV to multi-mode or perturbed initial conditions, where the associator's sensitivity to flow complexity should become more pronounced and provide a stronger baseline for Phase 2 claims.

## 7. Team Capability

The workspace is designed for reproducible local execution: tests, benchmark scripts, configs, metrics, figures, and trace documentation are versioned. The approach combines numerical PDE implementation, geometric algebra diagnostics, compression baselines, and conservative scientific reporting.

## 8. Risks and Mitigations

Two originally identified risks have been retired by measurement: solver accuracy (now verified 2nd-order with the central flux; the first-order Lax-Friedrichs path is retained for non-smooth flows) and unproven compression advantage (now measured head-to-head at matched error). Remaining risks: (a) the TGV's spectral simplicity flatters low-rank storage — mitigation is evaluation on perturbed/multi-mode initial conditions before scaling claims; (b) the per-step associator diagnostic adds runtime — mitigation is amortized evaluation every N steps; (c) the ROM speedup is currently an operation-count estimate — mitigation is hyper-reduction (DEIM) for a measured wall-clock result in Phase 2.

## 9. Scalability and Quantum Hardware Pathway

**Scalability to larger grids.** Tucker compression memory scales as O(r·n) per mode versus O(n³) for classical dense storage. At fixed physics (Re=100, same TGV problem), Tucker rank is determined by flow complexity, not grid resolution: the guided ranks saturate at (14, 29, 29) from nx=128 onward, and we verified by direct measurement that these ranks reach machine-precision reconstruction error even at nx=256. The compression ratio therefore grows rapidly with grid size — **measured through nx=256**, projected beyond (see `scripts/run_scalability_projection.py`):

| nx   | Dense (Re=100) | Tucker (guided) | Ratio   | Error   | Ranks (t,x,y) | Note     |
|------|---------------|-----------------|---------|---------|---------------|----------|
| 32   | 0.16 MB       | 0.04 MB         | 4.4×    | 1.5e-6  | (5, 16, 16)   | measured |
| 64   | 2.1 MB        | 0.27 MB         | 7.8×    | 1.1e-15 | (15, 30, 30)  | measured |
| 128  | 32.8 MB       | 0.32 MB         | 102×    | 1.2e-15 | (14, 29, 29)  | measured |
| 256  | 523 MB        | 0.52 MB         | **1010×** | 1.4e-15 | (14, 29, 29)  | measured |
| 512  | 4.1 GB        | 0.86 MB         | 4887×   | —       | (14, 29, 29)  | projected* |

\* Projections fix ranks at their measured saturation values. The saturation is verified through nx=256 by direct measurement (identical ranks, machine-precision error), so the 512 projection rests on measured rank stability, not extrapolated rank stability. These extreme ratios are specific to the spectrally narrow single-mode TGV; industrially relevant multi-mode flows would compress less (Section 4, result 3 measures the degradation direction). Reproduce with `python scripts\run_scalability_projection.py --re 100 --grids 32 64 128 256`.

**Rank-selection policy (stated for reproducibility).** Guided ranks come from `associator_guided_ranks` with a grid-independent cap (min_rank=4, max_rank=32): rank must track flow complexity, not resolution, so the cap does not grow with nx. An earlier internal draft used a grid-dependent cap (max_rank=nx/2), which inflated ranks to (29, 59, 59) at nx=128 with no error benefit; we verified by direct decomposition of the nx=256 trajectory that the smaller ranks reach identical machine-precision error, corrected the policy, and re-measured everything above. The small variation between nx=64 (15, 30, 30) and nx≥128 (14, 29, 29) reflects grid-dependent associator statistics feeding the same policy, not a policy change.

**Quantum hardware pathway.** The Tucker decomposition used in this workspace maps directly to a tree tensor network (TTN) state. Under the standard log₂ encoding, each spatial mode of dimension n requires ⌈log₂(n)⌉ qubits. Using **measured ranks** from `scripts/run_scalability_projection.py`:

| Grid | Ranks (t,x,y) | Logical qubits | Classical Tucker factors | Note |
|------|--------------|----------------|--------------------------|------|
| 64²  | (15, 30, 30) | ~46 qubits     | ~4,350 floats            | measured |
| 128² | (14, 29, 29) | ~50 qubits     | ~9,258 floats            | measured |
| 256² | (14, 29, 29) | ~54 qubits     | ~22,170 floats           | measured |
| 512² | (14, 29, 29) | ~57 qubits     | ~44,340 floats           | projected |

As nx quadruples from 128 → 512, the qubit count grows by ~7 (logarithmically) while classical Tucker factor storage grows ~5× (linearly with n). The compression ratio of the quantum encoding versus classical Tucker factors is the exponential advantage the tensor-network literature describes; the compression ratio of Tucker versus dense classical storage is already measured above.

No quantum hardware is claimed. Current devices (≤133 qubits, shallow coherent depth) cannot execute the O(10⁴)-gate preparation circuit at these scales. Fault-tolerant hardware at ~50,000 physical qubits is required for the nx=128 case. Detailed qubit counts, circuit depth estimates, and the honest caveats are in `docs/QUANTUM_HARDWARE_PATH.md`.
