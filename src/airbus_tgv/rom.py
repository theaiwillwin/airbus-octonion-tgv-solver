"""Reduced-order model (ROM) using compressed trajectory basis.

Project the full FV solver onto a low-rank subspace learned from a
previous high-fidelity trajectory via Tucker decomposition.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

try:
    import tensorly as tl  # type: ignore
    _HAS_TENSORLY = True
except Exception:
    _HAS_TENSORLY = False


class TrajectoryROM:
    """Reduced-order model based on Tucker factors from trajectory."""

    def __init__(self, factors_u, factors_v):
        """Initialize with Tucker factors (list of mode matrices).

        Args:
            factors_u, factors_v: List of factor matrices from Tucker decomposition.
                factors[0] is the time mode, factors[1] is x, factors[2] is y.
        """
        if not _HAS_TENSORLY:
            raise RuntimeError("tensorly required for ROM")
        self.factors_u = factors_u
        self.factors_v = factors_v
        self.nt = factors_u[0].shape[1] if factors_u else None
        self.nx = factors_u[1].shape[1] if len(factors_u) > 1 else None
        self.ny = factors_u[2].shape[1] if len(factors_u) > 2 else None

    def project_snapshot(self, u, v):
        """Project a full (nx, ny) snapshot onto the reduced basis.

        Returns reduced u, v of shape (nt, nx, ny).
        """
        # Unfold u and v and project using the spatial factors
        U_x = self.factors_u[1].T  # (nx_reduced, nx)
        U_y = self.factors_u[2].T  # (ny_reduced, ny)

        u_spatial = U_x @ u @ U_y.T  # (nx_reduced, ny_reduced)
        v_spatial = U_x @ v @ U_y.T

        return u_spatial, v_spatial

    def lift_snapshot(self, u_red, v_red):
        """Reconstruct full snapshot from reduced coefficients.

        Returns full u, v of shape (nx, ny).
        """
        U_x = self.factors_u[1]  # (nx, nx_reduced)
        U_y = self.factors_u[2]  # (ny, ny_reduced)

        u = U_x @ u_red @ U_y.T  # (nx, ny)
        v = U_x @ v_red @ U_y.T

        return u, v

    def summary(self):
        """Return summary of ROM dimensions."""
        return {
            "nx_reduced": self.nx,
            "ny_reduced": self.ny,
            "nt_basis": self.nt,
        }


def rom_speedup_estimate(factors_u, full_shape, rom_solve_cost_frac=0.1):
    """Estimate ROM speedup for solving on compressed basis.

    Args:
        factors_u: Tucker factors from decomposition.
        full_shape: Original tensor shape (nt, nx, ny).
        rom_solve_cost_frac: Cost of solving in ROM as fraction of full solve.

    Returns:
        Estimated speedup factor (>1 means ROM is faster).
    """
    nt, nx, ny = full_shape
    nx_red = factors_u[1].shape[1]
    ny_red = factors_u[2].shape[1]

    # Cost of full solve: O(nt * nx * ny) ops per timestep
    # Cost of ROM solve: O(nt * nx_red * ny_red) ops per timestep
    # Plus projection/lift overhead (assumed small)

    full_cost = nt * nx * ny
    rom_cost = nt * nx_red * ny_red
    projection_cost = 2 * (nx * ny * nx_red + nx * ny_red * ny)

    speedup = (full_cost + projection_cost) / (rom_cost + projection_cost + 1e-30)
    return float(speedup)
