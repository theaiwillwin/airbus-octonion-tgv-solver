# Method

## Finite-Volume Formulation

The solver stores cell-centered velocity fields `u` and `v` on a periodic square domain. Face values are reconstructed by arithmetic averaging, conservative convective momentum fluxes are computed on x/y faces, viscous fluxes use face-normal gradients, and the cell update is the divergence of the net face flux. A second-order midpoint time step is used with CFL and viscous stability limits.

After each explicit update, `project_divergence_free` applies a periodic FFT projection. The projection removes the irrotational component of the velocity field in Fourier space and returns a pressure-like projection potential.

## Exact TGV Benchmark

The Airbus statement sets `L=2*pi`, `V0=1`, `Uc=1`, `Vc=0`, `rho=1`, `p0=0`, and `nu=V0*L/Re`. For a periodic domain of length `L`, this implementation uses `k=2*pi/L`:

`u = Uc + V0 sin(k(x-Uc t)) cos(k(y-Vc t)) exp(-2 nu k^2 t)`

`v = Vc - V0 cos(k(x-Uc t)) sin(k(y-Vc t)) exp(-2 nu k^2 t)`

The pressure field is translated and decayed consistently as `exp(-4 nu k^2 t)`.

## Octonion Algebra

Octonions are represented as 8-component real vectors. Multiplication is generated from the fixed Fano-plane triples:

`(1,2,3), (1,4,5), (1,6,7), (2,4,6), (2,5,7), (3,4,7), (3,5,6)`.

The real basis element is the identity, imaginary basis elements square to `-1`, and reversing a Fano product changes sign.

## Associator Operator

The associator is implemented exactly:

`[a,b,c] = (a*b)*c - a*(b*c)`

`associator_norm(a,b,c) = ||[a,b,c]||`

This is not a proxy correlation or cosine metric. It is the real non-associative octonion associator induced by the fixed Fano-plane multiplication table.

## Flow Embedding

Each cell is mapped into an octonion vector. The pressure mode uses:

`[1, u, v, p_or_vorticity, du_dx, du_dy, dv_dx, dv_dy]`

The pressure-free strain mode uses:

`[1, u, v, vorticity, strain_xx, strain_xy, strain_yx, strain_yy]`

Directional probes compare center, x-neighbor, y-neighbor, and one-dimensional neighbor triples. The output fields are `A`, `A_x`, `A_y`, and `A_diag`.

## Associator-Guided Control and Compression

The baseline compression is truncated SVD applied to real solver fields. The associator layer can select a higher retained rank when the p95 associator norm is high. If TensorLy is installed, Tucker decomposition is exposed explicitly; if it is not installed, the code raises a clear runtime error rather than pretending Tucker support exists.

The associator-guided method also uses the non-associative field as a steering signal for the time step. At each step, the current octonionic associator field is summarized and converted into a CFL safety factor in `(0, 1]`. Larger relational torsion reduces the step size. This is the concrete control angle: the Fano-plane associator does not replace the FV equations, but it steers numerical resource allocation and stability control.

This is the proposed quantum-inspired advantage mechanism: non-associative geometric structure supplies a low-cost controller for where to spend rank and time-step budget. It remains an advantage hypothesis until measured against the classical baseline.

## Limitations

This is a CPU-friendly research workspace, not a production CFD code. The current FV scheme is low order and the associator-guided compression is an ablation pathway, not proven quantum advantage. Any claim of advantage must come from measured runtime, memory, or error tradeoffs in `results/run_*/metrics.csv`.
