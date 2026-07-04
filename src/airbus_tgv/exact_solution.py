"""Exact analytical solution for the 2D Convecting Taylor-Green Vortex.

The Airbus statement gives L = 2*pi as the domain length. For a periodic
finite-volume benchmark on [0, L]^2, the closed-form mode is evaluated with
k = 2*pi/L. With the Airbus value L = 2*pi this reduces to the standard
sin(x), cos(y) Taylor-Green vortex while retaining nu = V0*L/Re.
"""
from __future__ import annotations
import numpy as np
from .constants import TGVParams


def exact_velocity(x: np.ndarray, y: np.ndarray, t: float, params: TGVParams):
    """Exact analytical velocity (u, v) at coordinates (x, y) and time t."""
    k = params.wave_number
    decay = np.exp(-2.0 * params.nu * (k ** 2) * t)
    xs = k * (x - params.Uc * t)
    ys = k * (y - params.Vc * t)
    u = params.Uc + params.V0 * np.sin(xs) * np.cos(ys) * decay
    v = params.Vc - params.V0 * np.cos(xs) * np.sin(ys) * decay
    return u, v


def exact_pressure(x: np.ndarray, y: np.ndarray, t: float, params: TGVParams):
    """Exact pressure at time t.

    For the convecting TGV, the unsteady pressure field is the initial profile
    translated by the convection velocity (Uc, Vc) and decayed by exp(-4*nu*t/L^2).
    """
    k = params.wave_number
    decay = np.exp(-4.0 * params.nu * (k ** 2) * t)
    xs = k * (x - params.Uc * t)
    ys = k * (y - params.Vc * t)
    return params.p0 + 0.25 * params.rho * params.V0 ** 2 * (
        np.cos(2.0 * xs) + np.cos(2.0 * ys)
    ) * decay


def kinetic_energy(u: np.ndarray, v: np.ndarray, rho: float = 1.0) -> float:
    """Volume-averaged kinetic energy: 0.5 * rho * <u^2 + v^2>."""
    return 0.5 * rho * float(np.mean(u * u + v * v))


def l2_error(u_num, v_num, u_ex, v_ex) -> float:
    """Discrete L2 norm of velocity error."""
    diff = (u_num - u_ex) ** 2 + (v_num - v_ex) ** 2
    return float(np.sqrt(np.mean(diff)))


def divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Periodic spectral divergence used by the FFT pressure projection."""
    nx, ny = u.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    du_dx = np.real(np.fft.ifftn(1j * KX * np.fft.fftn(u)))
    dv_dy = np.real(np.fft.ifftn(1j * KY * np.fft.fftn(v)))
    return du_dx + dv_dy
