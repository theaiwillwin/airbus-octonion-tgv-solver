# =============================================================================
# SUBMISSION EVIDENCE - Airbus 2026 Quantum + AI Challenge, Phase 1
# Verbatim copy of src/airbus_tgv/finite_volume.py from the project repository.
# The 2D Convecting Taylor-Green Vortex finite-volume solver: cell-centered
# periodic FV, face fluxes, RK2 stepping, FFT (spectral) pressure projection.
# Relative imports (.constants, .exact_solution, .fluxes) refer to sibling
# modules in the airbus_tgv package; full reproduction commands are in the
# accompanying PDF report, Section 9.
# =============================================================================

"""Cell-centered, periodic finite-volume solver for the 2D incompressible
Navier-Stokes equations using a fractional-step projection method.

State variables: cell-averaged velocities u, v on an nx x ny grid covering
[0, 2*pi]^2 with periodic BCs. Pressure is computed each step via an FFT
Poisson solve to enforce incompressibility.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from .constants import TGVParams
from .exact_solution import exact_velocity, divergence as div_centered
from .fluxes import (
    convective_fluxes, diffusive_fluxes, divergence_from_faces,
    face_average_x, face_average_y,
)


def build_periodic_grid(nx: int, ny: int, L_domain: float = 2.0 * np.pi):
    dx = L_domain / nx
    dy = L_domain / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


def initialize_cell_averages_from_exact(nx: int, ny: int, params: TGVParams):
    L_domain = 2.0 * np.pi
    X, Y, dx, dy = build_periodic_grid(nx, ny, L_domain)
    u, v = exact_velocity(X, Y, 0.0, params)
    return X, Y, dx, dy, u, v


def poisson_solve_periodic(rhs: np.ndarray, dx: float, dy: float) -> np.ndarray:
    nx, ny = rhs.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX * KX + KY * KY
    K2[0, 0] = 1.0
    rhs_hat = np.fft.fftn(rhs - rhs.mean())
    phi_hat = -rhs_hat / K2
    phi_hat[0, 0] = 0.0
    return np.real(np.fft.ifftn(phi_hat))


def project_divergence_free(u, v, dx, dy):
    nx, ny = u.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX * KX + KY * KY
    K2[0, 0] = 1.0
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    div_hat = 1j * (KX * u_hat + KY * v_hat)
    phi_hat = -div_hat / K2
    phi_hat[0, 0] = 0.0
    u_proj_hat = u_hat - 1j * KX * phi_hat
    v_proj_hat = v_hat - 1j * KY * phi_hat
    phi = np.real(np.fft.ifftn(phi_hat))
    return np.real(np.fft.ifftn(u_proj_hat)), np.real(np.fft.ifftn(v_proj_hat)), phi


def compute_face_states(u, v):
    """Return arithmetic face states for x/y faces from cell averages."""
    return {
        "u_x": face_average_x(u),
        "v_x": face_average_x(v),
        "u_y": face_average_y(u),
        "v_y": face_average_y(v),
    }


def compute_convective_fluxes(u, v):
    """Public wrapper for conservative momentum fluxes across volume faces."""
    return convective_fluxes(u, v, 0.0, 0.0)


def compute_diffusive_fluxes(u, v, nu, dx, dy):
    """Public wrapper for viscous face fluxes."""
    return diffusive_fluxes(u, v, nu, dx, dy)


def rhs_fv(u, v, nu, dx, dy, lf_coeff: float = 1.0):
    Fx_u, Fx_v, Fy_u, Fy_v = convective_fluxes(u, v, 0.0, 0.0, lf_coeff=lf_coeff)
    Gx_u, Gx_v, Gy_u, Gy_v = diffusive_fluxes(u, v, nu, dx, dy)
    dudt = -divergence_from_faces(Fx_u - Gx_u, Fy_u - Gy_u, dx, dy)
    dvdt = -divergence_from_faces(Fx_v - Gx_v, Fy_v - Gy_v, dx, dy)
    return dudt, dvdt


def cfl_dt(u, v, dx, dy, nu, cfl: float = 0.4, vis_cfl: float = 0.2) -> float:
    umax = float(np.max(np.abs(u))) + 1e-12
    vmax = float(np.max(np.abs(v))) + 1e-12
    dt_adv = cfl / (umax / dx + vmax / dy)
    dt_vis = vis_cfl * 0.5 / (nu * (1.0 / dx ** 2 + 1.0 / dy ** 2) + 1e-30)
    return float(min(dt_adv, dt_vis))


def step_finite_volume(u, v, dt, nu, dx, dy, project: bool = True,
                       lf_coeff: float = 1.0):
    du1, dv1 = rhs_fv(u, v, nu, dx, dy, lf_coeff=lf_coeff)
    u_mid = u + 0.5 * dt * du1
    v_mid = v + 0.5 * dt * dv1
    du2, dv2 = rhs_fv(u_mid, v_mid, nu, dx, dy, lf_coeff=lf_coeff)
    u_new = u + dt * du2
    v_new = v + dt * dv2
    if project:
        u_new, v_new, _ = project_divergence_free(u_new, v_new, dx, dy)
    return u_new, v_new


@dataclass
class FVRunResult:
    u: np.ndarray
    v: np.ndarray
    t: float
    n_steps: int
    dt_history: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def run_fv_solver(params: TGVParams,
                  nx: int = 64, ny: int = 64,
                  t_final: float = 1.0,
                  cfl: float = 0.4,
                  project: bool = True,
                  lf_coeff: float = 1.0,
                  dt_controller: Optional[Callable] = None,
                  callback: Optional[Callable] = None,
                  initial_condition: Optional[Callable] = None,
                  max_steps: int = 200_000) -> FVRunResult:
    """initial_condition, if given, is a callable (X, Y, params) -> (u, v)
    replacing the default exact-TGV initialization."""
    X, Y, dx, dy, u, v = initialize_cell_averages_from_exact(nx, ny, params)
    if initial_condition is not None:
        u, v = initial_condition(X, Y, params)
    if project:
        u, v, _ = project_divergence_free(u, v, dx, dy)
    nu = params.nu
    t = 0.0
    dt_hist = []
    steering_factors = []
    diag = {"unstable": False, "reason": None}
    for step in range(max_steps):
        if t >= t_final - 1e-15:
            break
        dt = cfl_dt(u, v, dx, dy, nu, cfl=cfl)
        if dt_controller is not None:
            factor = float(dt_controller(u, v, dx, dy, step, t))
            if not np.isfinite(factor) or factor <= 0.0 or factor > 1.0:
                raise ValueError(f"dt_controller returned invalid factor {factor}")
            dt *= factor
            steering_factors.append(factor)
        if t + dt > t_final:
            dt = t_final - t
        u, v = step_finite_volume(u, v, dt, nu, dx, dy, project=project,
                                  lf_coeff=lf_coeff)
        t += dt
        dt_hist.append(dt)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            diag["unstable"] = True
            diag["reason"] = f"NaN/Inf at step {step}"
            break
        if callback is not None:
            callback(step, t, u, v)
    diag.update({"X": X, "Y": Y, "dx": dx, "dy": dy,
                 "dt_steering_factors": steering_factors})
    return FVRunResult(u=u, v=v, t=t, n_steps=len(dt_hist),
                       dt_history=dt_hist, diagnostics=diag)
