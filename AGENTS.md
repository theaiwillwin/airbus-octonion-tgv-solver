# Project Instructions

Build complete, local, reproducible Python research code for the Airbus 2026 Quantum + AI Challenge Taylor-Green Vortex solver.

No placeholders, fake data, silent failures, or unsupported advantage claims. The analytical Taylor-Green solution is the reference truth, so no external CFD dataset is required.

The octonion layer must compute the real associator norm:

`associator_norm(a,b,c) = ||(a*b)*c - a*(b*c)||`

using the fixed Fano-plane structure constants in the project statement. Do not replace it with correlation, cosine similarity, or a heuristic proxy.

All scientific outputs and external-facing text must preserve the statement:

`Experimental research output - not a validated engineering deliverable.`

Keep the backend modular: separate PDE solving, octonion embedding, associator diagnostics, compression, benchmarking, plotting, and domain constants.
