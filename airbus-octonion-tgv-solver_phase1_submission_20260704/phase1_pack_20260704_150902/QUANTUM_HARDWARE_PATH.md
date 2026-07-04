# Quantum Hardware Pathway

This document maps the workspace's classical tensor-network pipeline to a
quantum circuit model. No quantum hardware is claimed. The mapping is provided
to answer the question: "if fault-tolerant quantum hardware were available at
the required scale, what circuit resources would this algorithm need?"

## Background: Tucker Decomposition as Tensor Network State

A Tucker-decomposed tensor of shape (nt, nx, ny) with ranks (r_t, r_x, r_y)
is equivalent to a rank-bounded tensor network state. The core tensor and three
factor matrices together define a tree-structured tensor network (TN):

```
  factor_t (nt × r_t)
       |
  core (r_t × r_x × r_y) — factor_x (nx × r_x)
       |
  factor_y (ny × r_y)
```

This is a 3-mode Hierarchical Tucker (HT) tree with leaf dimensions (nt, nx, ny)
and bond dimensions (r_t, r_x, r_y). It is a special case of the tree tensor
network (TTN) architecture.

## Qubit Count Estimate

The standard encoding maps each mode dimension d to ⌈log₂(d)⌉ qubits.

Ranks below are **measured** from `scripts/run_scalability_projection.py` at
Re=100, t_final=0.5. The guided ranks saturate at (14, 29, 29) from nx=128
onward; the saturation is verified by direct measurement at nx=256, where
these ranks still reach machine-precision reconstruction error (1.4e-15).

**nx=64, measured ranks (r_t=15, r_x=30, r_y=30), nt=34:**

| Mode | Dimension | Rank | Qubits (dim) | Qubits (rank) |
|------|-----------|------|--------------|---------------|
| Time | 34        | 15   | 6            | 4             |
| x    | 64        | 30   | 6            | 5             |
| y    | 64        | 30   | 6            | 5             |

- Factor qubits: (6+4) + (6+5) + (6+5) = 32 logical qubits
- Core register: ⌈log₂(15·30·30)⌉ = ⌈log₂(13500)⌉ = 14 qubits
- **Total: ~46 logical qubits** at nx=64

**nx=128, measured ranks (r_t=14, r_x=29, r_y=29), nt=131:**

| Mode | Dimension | Rank | Qubits (dim) | Qubits (rank) |
|------|-----------|------|--------------|---------------|
| Time | 131       | 14   | 8            | 4             |
| x    | 128       | 29   | 7            | 5             |
| y    | 128       | 29   | 7            | 5             |

- Factor qubits: (8+4) + (7+5) + (7+5) = 36 logical qubits
- Core register: ⌈log₂(14·29·29)⌉ = ⌈log₂(11774)⌉ = 14 qubits
- **Total: ~50 logical qubits** at nx=128

**nx=256, measured ranks (14, 29, 29), nt=523:**

| Mode | Dimension | Rank | Qubits (dim) | Qubits (rank) |
|------|-----------|------|--------------|---------------|
| Time | 523       | 14   | 10           | 4             |
| x    | 256       | 29   | 8            | 5             |
| y    | 256       | 29   | 8            | 5             |

- Factor qubits: (10+4) + (8+5) + (8+5) = 40 logical qubits
- Core register: 14 qubits (unchanged — ranks saturated)
- **Total: ~54 logical qubits** at nx=256 (measured ranks, 1.4e-15 error)

**nx=512, projected:** ~57 logical qubits (ranks still saturated; only the
mode-dimension qubits grow by 1 per doubling).

The logarithmic scaling with grid dimension is the core claim: as nx doubles,
the spatial mode qubit count increases by 1 (⌈log₂(2n)⌉ = ⌈log₂(n)⌉ + 1),
while classical storage grows as O(n). At nx=512 vs nx=64, classical Tucker
factor storage grows ~10×; quantum encoding grows by ~11 qubits total.

## Circuit Depth Estimate

For a tree tensor network preparation circuit (standard isometry compilation):

- Each factor matrix (m × r) is prepared using O(m · r) parameterized Givens
  rotation gates.
- At nx=64, measured ranks: O(34·15 + 64·30 + 64·30) ≈ O(4350) factor gates.
- Core tensor loading: O(r_t · r_x · r_y) = O(15·30·30) = O(13500) gates.
- Estimated **total circuit depth: O(2×10⁴) gates** at nx=64.

At nx=128, measured ranks (14, 29, 29):
- Factor gates: O(131·14 + 128·29 + 128·29) ≈ O(9300)
- Core: O(14·29·29) ≈ O(11800) gates
- Estimated **total: O(2×10⁴) gates** at nx=128.

At nx=256 with saturated ranks (measured), the core term is unchanged at
~O(1.2×10⁴) gates; factor gates grow only as O(n · r) with r constant:
O(523·14 + 256·29 + 256·29) ≈ O(22200) —
**total ~O(3.4×10⁴) gates** at nx=256.

## What This Means for Fault-Tolerant Quantum Hardware

With surface code overhead (~1000 physical qubits per logical qubit at
code distance d=15 for a target logical error rate of 10⁻¹⁵):

- nx=64 (~46 logical qubits): ~46,000 physical qubits
- nx=128 (~50 logical qubits): ~50,000 physical qubits
- nx=256 (~54 logical qubits): ~54,000 physical qubits

The qubit count grows only logarithmically with grid size; the circuit depth
grows linearly with the Tucker bond dimensions. For the smooth TGV, bond
dimension is bounded by the rank of the solution (approximately the number of
active Fourier modes), which grows slowly with Re.

## Honest Caveats

1. **No quantum hardware exists at this scale today.** The IBM Heron processor
   (133 qubits, 2024) and similar systems cannot coherently execute circuits
   of depth 10⁴–10⁵; surface-code fault tolerance requires a processor ~1000×
   larger than current devices.

2. **The tensor network preparation circuit does not by itself solve the PDE.**
   The advantage is in compressed storage and reconstruction queries, not in
   the time integration, which remains classical.

3. **The log(n) qubit scaling is real.** A classical representation of a
   rank-30 Tucker factor in a 64-point spatial mode requires 64·30 = 1920
   floats; a quantum state requires ⌈log₂(64)⌉ + ⌈log₂(30)⌉ = 11 qubits.
   This is the sense in which the Tucker → TTN → quantum circuit chain
   constitutes a credible path to exponential quantum memory advantage, not
   a current result.

4. **The associator layer maps to quantum non-associativity.** The octonion
   associator (a,b,c) = (ab)c − a(bc) computes a quantity with no classical
   analogue in standard Lie algebra; it is native to quantum geometric algebra.
   Whether this translates to a circuit-level advantage is an open research
   question not claimed here.

## Reproducible Parameters

All ranks in this document are measured outputs of the scalability run at
Re=100, t_final=0.5: (15, 30, 30) at nx=64 and (14, 29, 29) at nx=128 and
nx=256 (saturated, machine-precision error verified at both grids). To
reproduce the rank measurements:

```powershell
python scripts\run_scalability_projection.py --re 100 --grids 32 64 128 256
```

The per-Re guided ranks in the advantage study are reproduced with:

```powershell
python scripts\run_advantage_study.py --reynolds 100 --nx 64
```
