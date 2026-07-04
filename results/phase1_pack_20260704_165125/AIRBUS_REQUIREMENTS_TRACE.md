# Airbus Requirements Trace

This workspace targets the 2026 Global Quantum + AI Challenge Airbus statement, "Quantum Solvers: Enhancing Predictive Aerodynamic Modeling Capabilities."

| Requirement | Implementation |
| --- | --- |
| Working solver for 2D Convecting Taylor-Green Vortex | `src/airbus_tgv/finite_volume.py` implements a cell-centered finite-volume velocity solver with periodic boundaries and FFT projection. |
| Exact TGV analytical solution | `src/airbus_tgv/exact_solution.py` implements velocity, pressure, kinetic energy, L2 error, and divergence diagnostics. |
| Re scaling | `scripts/run_reynolds_sweep.py` and `configs/reynolds_sweep.yaml` run Re = 10, 100, 250, 500 where stable. |
| Time-to-solution | `src/airbus_tgv/benchmark.py` records `runtime_sec`. |
| Memory scaling | `src/airbus_tgv/compression.py` and benchmark rows record dense and compressed memory estimates. |
| L2 error | Benchmark rows record `l2_velocity_error` against the exact analytical solution. |
| Kinetic energy decay | `kinetic_energy_num`, `kinetic_energy_exact`, and relative error are recorded per run. |
| Classical comparison | Method `classical_fv` is the baseline without associator-guided compression. |
| FV cell averages | Grid fields are cell-centered arrays initialized from the exact solution at cell centers. |
| Interface relative fields | `compute_face_states`, `compute_convective_fluxes`, and `compute_diffusive_fluxes` compute x/y face states and fluxes. |
| Temporal evolution by face fluxes | `step_finite_volume` updates cells from divergence of face fluxes, followed by periodic projection. |
| Tensor Network within FV route | `compression.py` provides real SVD compression and optional TensorLy Tucker decomposition if installed. |
| Associator method | `octonion.py`, `fano.py`, and `associator_metrics.py` implement fixed Fano-plane octonion products and associator diagnostics. |
| Control / steering angle | `fv_plus_associator_guided_compression` uses associator-derived CFL steering factors and associator-guided SVD rank selection. |
| Figures | `scripts/make_figures.py` generates plots from actual run outputs plus a fresh FV field run. |

The code uses the periodic-domain interpretation of the Airbus parameters: `L=2*pi` is the domain length, and the TGV mode uses wave number `k=2*pi/L`. This gives the standard periodic `sin(x)` / `cos(y)` vortex when `L=2*pi` while preserving the specified viscosity `nu = V0*L/Re`.
