# Technical Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Low-order FV scheme may be too dissipative | Error may dominate associator/compression comparisons | Add higher-order reconstruction and compare grid-refined results. |
| Explicit time stepping limits high-Re scaling | Higher Re may require very small dt | Add semi-implicit diffusion or stronger CFL analysis before claiming scaling. |
| Associator diagnostic may not improve compression Pareto front | No demonstrated advantage | Report as diagnostic only unless metrics show a benefit. |
| SVD is not a full tensor-network solver | Challenge fit may be weaker than true TN implementation | Use TensorLy Tucker when installed and document SVD as baseline fallback. |
| Projection and FV operators are mixed-order | Divergence is controlled but not a production CFD discretization | Track divergence every run and document numerical method limits. |
| Formula ambiguity in source statement | Reproducibility risk | Use periodic-domain interpretation `k=2*pi/L`, document it, and keep parameters configurable. |
| Runtime on CPU at 128x128 sweeps | Long local runs | Keep default sweep modest and allow targeted configs. |
