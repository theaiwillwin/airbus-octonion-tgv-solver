# Validation Plan

## Mandatory Cases

Run Re = 10 and Re = 100 with at least 32x32 and 64x64 grids. Re = 250 and Re = 500 are attempted where the explicit solver remains stable within the configured runtime.

## Grid Refinement

For each Reynolds number, compare L2 velocity error and kinetic-energy relative error across grid sizes. A credible result should show reduced error with refinement for stable cases.

## Analytical Comparison

Every run compares numerical velocity at final time against the exact TGV solution. The benchmark records L2 velocity error, numerical kinetic energy, exact kinetic energy, kinetic-energy relative error, and divergence norm.

## Ablation Study

The required methods are:

1. `classical_fv`: finite-volume baseline.
2. `fv_plus_associator_diagnostic`: same solver plus associator field diagnostics.
3. `fv_plus_associator_guided_compression`: same solver plus associator-steered CFL control and SVD compression rank selected from associator statistics.

## Failure Criteria

A run is failed if fields contain NaN or Inf, the run does not reach final time, the divergence norm grows without bound, or the final velocity error is not finite.

## Acceptance Criteria

Phase 1 acceptance for this workspace means tests pass, Re = 10 and Re = 100 runs complete at 32x32, metrics are saved, and figures are generated from real outputs. Competitiveness requires stronger evidence: grid-refined scaling, memory/error Pareto improvement from compression, and a fair comparison against a stronger classical reference.
