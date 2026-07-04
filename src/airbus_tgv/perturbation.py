"""Divergence-free multi-mode perturbations of the TGV initial condition.

The perturbation is built from a streamfunction so it is divergence-free by
construction:

    psi(x, y) = sum_k a_k * sin(kx_k * x + phi_k) * sin(ky_k * y + chi_k)
    u' =  d(psi)/dy,   v' = -d(psi)/dx

Wavenumbers are integers, so on the periodic grid the analytic derivatives
are exact Fourier modes and the spectral divergence of the sampled field is
zero to machine precision. This gives a physically meaningful "harder" flow:
the single-mode TGV plus higher-mode content that low-rank methods cannot
absorb into one Fourier mode.
"""
from __future__ import annotations
import numpy as np

from .constants import TGVParams
from .exact_solution import exact_velocity


def streamfunction_perturbation(X: np.ndarray, Y: np.ndarray,
                                n_modes: int = 4,
                                amplitude: float = 0.1,
                                seed: int = 0) -> tuple:
    """Velocity perturbation (u', v') from a random multi-mode streamfunction.

    Modes use integer wavenumbers 2..(n_modes+1) in each direction; amplitudes
    decay as 1/k^2 so the perturbation has a physical-looking spectrum. The
    overall field is rescaled so max(|u'|, |v'|) == amplitude.
    """
    rng = np.random.default_rng(seed)
    u_p = np.zeros_like(X)
    v_p = np.zeros_like(X)
    for k in range(2, 2 + n_modes):
        a_k = rng.uniform(0.5, 1.0) / (k * k)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        chi = rng.uniform(0.0, 2.0 * np.pi)
        # psi = a_k sin(k x + phi) sin(k y + chi)
        # u' = d psi / dy = a_k k sin(k x + phi) cos(k y + chi)
        # v' = -d psi / dx = -a_k k cos(k x + phi) sin(k y + chi)
        u_p += a_k * k * np.sin(k * X + phi) * np.cos(k * Y + chi)
        v_p += -a_k * k * np.cos(k * X + phi) * np.sin(k * Y + chi)
    scale = max(float(np.max(np.abs(u_p))), float(np.max(np.abs(v_p))), 1e-30)
    u_p *= amplitude / scale
    v_p *= amplitude / scale
    return u_p, v_p


def perturbed_tgv_initial_condition(n_modes: int = 4,
                                    amplitude: float = 0.1,
                                    seed: int = 0):
    """Return an initial_condition callable for run_fv_solver.

    The callable takes (X, Y, params) and returns (u, v): the exact TGV
    velocity at t=0 plus a divergence-free multi-mode perturbation.
    """
    def ic(X, Y, params: TGVParams):
        u, v = exact_velocity(X, Y, 0.0, params)
        u_p, v_p = streamfunction_perturbation(X, Y, n_modes=n_modes,
                                               amplitude=amplitude, seed=seed)
        return u + u_p, v + v_p
    return ic
