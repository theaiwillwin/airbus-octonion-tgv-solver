"""Benchmark harness: run methods, collect metrics, save artifacts."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import time
import numpy as np

from .constants import TGVParams, OUTPUT_DISCLAIMER
from .finite_volume import run_fv_solver, build_periodic_grid
from .exact_solution import exact_velocity, kinetic_energy, l2_error, divergence
from .associator_metrics import (
    associator_field, associator_summary_stats, associator_guided_dt_factor,
)
from .compression import (
    svd_compress_field, reconstruct_svd, compression_error,
    memory_estimate_dense, memory_estimate_compressed,
    associator_guided_rank_policy,
)
from .utils import process_memory_mb, timer


@dataclass
class BenchmarkResult:
    method: str
    output_label: str
    Re: float
    nx: int
    ny: int
    dt: float
    dt_mean: float
    n_steps: int
    final_time: float
    runtime_sec: float
    memory_dense_mb: float
    memory_compressed_mb: float
    compression_ratio: float
    compression_rel_err: float
    l2_velocity_error: float
    kinetic_energy_num: float
    kinetic_energy_exact: float
    kinetic_energy_rel_error: float
    divergence_l2: float
    associator_mean: float
    associator_p95: float
    associator_max: float
    associator_dt_factor_mean: float
    associator_dt_factor_min: float
    stable: bool
    notes: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def _final_metrics(res, params: TGVParams, method: str,
                   compression_rank: Optional[int]) -> BenchmarkResult:
    u, v = res.u, res.v
    X = res.diagnostics["X"]; Y = res.diagnostics["Y"]
    dx = res.diagnostics["dx"]; dy = res.diagnostics["dy"]
    t = res.t
    u_ex, v_ex = exact_velocity(X, Y, t, params)
    ke_num = kinetic_energy(u, v, params.rho)
    ke_ex = kinetic_energy(u_ex, v_ex, params.rho)
    div = divergence(u, v, dx, dy)
    A = associator_field(u, v, None, dx, dy, mode="strain")["A"]
    stats = associator_summary_stats(A)
    steering = res.diagnostics.get("dt_steering_factors") or []

    mem_dense = memory_estimate_dense(u) + memory_estimate_dense(v)
    mem_comp = mem_dense
    comp_err = 0.0
    if compression_rank is not None:
        Uu, su, Vtu = svd_compress_field(u, compression_rank)
        Uv, sv, Vtv = svd_compress_field(v, compression_rank)
        u_rec = reconstruct_svd(Uu, su, Vtu)
        v_rec = reconstruct_svd(Uv, sv, Vtv)
        comp_err = 0.5 * (compression_error(u, u_rec) + compression_error(v, v_rec))
        mem_comp = (memory_estimate_compressed(Uu, su, Vtu)
                    + memory_estimate_compressed(Uv, sv, Vtv))

    return BenchmarkResult(
        method=method,
        output_label=OUTPUT_DISCLAIMER,
        Re=params.Re,
        nx=u.shape[0], ny=u.shape[1],
        dt=float(res.dt_history[-1]) if res.dt_history else 0.0,
        dt_mean=float(np.mean(res.dt_history)) if res.dt_history else 0.0,
        n_steps=res.n_steps,
        final_time=t,
        runtime_sec=res.diagnostics.get("runtime_sec", 0.0),
        memory_dense_mb=mem_dense,
        memory_compressed_mb=mem_comp,
        compression_ratio=(mem_dense / mem_comp) if mem_comp > 0 else 1.0,
        compression_rel_err=comp_err,
        l2_velocity_error=l2_error(u, v, u_ex, v_ex),
        kinetic_energy_num=ke_num,
        kinetic_energy_exact=ke_ex,
        kinetic_energy_rel_error=abs(ke_num - ke_ex) / (abs(ke_ex) + 1e-30),
        divergence_l2=float(np.sqrt(np.mean(div ** 2))),
        associator_mean=stats["associator_mean"],
        associator_p95=stats["associator_p95"],
        associator_max=stats["associator_max"],
        associator_dt_factor_mean=float(np.mean(steering)) if steering else 1.0,
        associator_dt_factor_min=float(np.min(steering)) if steering else 1.0,
        stable=not res.diagnostics.get("unstable", False),
        notes=res.diagnostics.get("reason") or "",
    )


def run_method(method: str, params: TGVParams, nx: int, ny: int, t_final: float,
               cfl: float = 0.4, lf_coeff: float = 0.0) -> BenchmarkResult:
    """Run a method. Supported:
        - 'classical_fv'
        - 'fv_plus_associator_diagnostic'
        - 'fv_plus_associator_guided_compression'

    lf_coeff=0.0 (default) uses the pure 2nd-order central flux, verified
    stable for the smooth TGV up to Re=2000; lf_coeff=1.0 restores full
    Lax-Friedrichs dissipation (first-order, robust for non-smooth flows).
    """
    dt_controller = None
    if method == "fv_plus_associator_guided_compression":
        def dt_controller(u, v, dx, dy, step, t):
            A = associator_field(u, v, None, dx, dy, mode="strain")["A"]
            return associator_guided_dt_factor(A, floor=0.35)

    t0 = time.perf_counter()
    res = run_fv_solver(params, nx=nx, ny=ny, t_final=t_final, cfl=cfl,
                        lf_coeff=lf_coeff, dt_controller=dt_controller)
    runtime = time.perf_counter() - t0
    res.diagnostics["runtime_sec"] = runtime

    rank = None
    if method == "classical_fv":
        rank = None
    elif method == "fv_plus_associator_diagnostic":
        rank = None
    elif method == "fv_plus_associator_guided_compression":
        dx = res.diagnostics["dx"]; dy = res.diagnostics["dy"]
        A = associator_field(res.u, res.v, None, dx, dy, mode="strain")["A"]
        rank = associator_guided_rank_policy(A, min_rank=4, max_rank=min(32, nx))
    else:
        raise ValueError(f"unknown method '{method}'")

    return _final_metrics(res, params, method, rank)
