"""Associator-based relational diagnostics for finite-volume flow states.

Each cell is embedded as an 8-component octonion built from local flow state.
The associator over (left, center, right) along an axis measures the
non-associative torsion between neighbouring states. Large associator norm
flags regions where local relational geometry is sharply changing -- vortex
cores, shear interfaces, or under-resolved features.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from .octonion import associator_norm


def vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Periodic central-difference vorticity dv/dx - du/dy."""
    dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
    return dv_dx - du_dy


def velocity_gradients(u, v, dx, dy):
    du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
    dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
    dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dy)
    return du_dx, du_dy, dv_dx, dv_dy


def embed_octonion_field(u, v, dx, dy, p: Optional[np.ndarray] = None,
                         mode: str = "pressure") -> np.ndarray:
    """Map (u, v[, p]) on an nx*ny grid to an (nx, ny, 8) octonion field.

    mode = 'pressure': o = (1, u, v, p_or_vort, du/dx, du/dy, dv/dx, dv/dy)
    mode = 'strain':   o = (1, u, v, vort, exx, exy, eyx, eyy)
    """
    du_dx, du_dy, dv_dx, dv_dy = velocity_gradients(u, v, dx, dy)
    o = np.empty(u.shape + (8,), dtype=float)
    o[..., 0] = 1.0
    o[..., 1] = u
    o[..., 2] = v
    if mode == "pressure":
        o[..., 3] = p if p is not None else (dv_dx - du_dy)
        o[..., 4] = du_dx
        o[..., 5] = du_dy
        o[..., 6] = dv_dx
        o[..., 7] = dv_dy
    elif mode == "strain":
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)
        eyx = exy
        o[..., 3] = dv_dx - du_dy
        o[..., 4] = exx
        o[..., 5] = exy
        o[..., 6] = eyx
        o[..., 7] = eyy
    else:
        raise ValueError(f"unknown embedding mode: {mode}")
    return o


def associator_field(u, v, p, dx, dy, mode: str = "pressure") -> dict:
    """Compute directional associator-norm fields A_x, A_y, A_diag and combined A."""
    o = embed_octonion_field(u, v, dx, dy, p=p, mode=mode)
    o_xp = np.roll(o, -1, axis=0)
    o_xm = np.roll(o, +1, axis=0)
    o_yp = np.roll(o, -1, axis=1)
    o_ym = np.roll(o, +1, axis=1)

    A_x = associator_norm(o_xm, o, o_xp)
    A_y = associator_norm(o_ym, o, o_yp)
    A_diag = associator_norm(o, o_xp, o_yp)
    A = np.sqrt(A_x ** 2 + A_y ** 2 + A_diag ** 2)
    return {"A": A, "A_x": A_x, "A_y": A_y, "A_diag": A_diag}


def associator_summary_stats(A: np.ndarray) -> dict:
    flat = A.ravel()
    return {
        "associator_mean": float(np.mean(flat)),
        "associator_std": float(np.std(flat)),
        "associator_p50": float(np.percentile(flat, 50)),
        "associator_p95": float(np.percentile(flat, 95)),
        "associator_max": float(np.max(flat)),
    }


def associator_guided_compression_mask(A: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """Boolean mask: True where associator > quantile threshold (keep full rank)."""
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    thr = np.quantile(A, quantile)
    return A > thr


def associator_guided_dt_factor(A: np.ndarray, floor: float = 0.25) -> float:
    """Return a multiplicative CFL safety factor in [floor, 1.0] based on A_max.

    Larger relational torsion => smaller dt. Purely heuristic; documented as such.
    """
    a_max = float(np.max(A))
    a_mean = float(np.mean(A)) + 1e-12
    ratio = a_max / a_mean
    factor = 1.0 / (1.0 + 0.1 * max(0.0, ratio - 1.0))
    return float(max(floor, min(1.0, factor)))
