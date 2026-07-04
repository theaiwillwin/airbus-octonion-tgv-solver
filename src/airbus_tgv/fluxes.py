"""Face-flux computations for the finite-volume scheme.

Convention: u, v are cell-centered. Faces are indexed by the left/lower cell.
Periodic boundary conditions everywhere.

For face state reconstruction we use a simple arithmetic average (2nd order
central) and a Lax-Friedrichs stabilization for the nonlinear convective part.
This gives a conservative scheme with controllable dissipation.
"""
from __future__ import annotations
import numpy as np


def face_average_x(phi: np.ndarray) -> np.ndarray:
    """Average to x-faces: face i lies between cell i and i+1."""
    return 0.5 * (phi + np.roll(phi, -1, axis=0))


def face_average_y(phi: np.ndarray) -> np.ndarray:
    return 0.5 * (phi + np.roll(phi, -1, axis=1))


def convective_fluxes(u, v, Uc, Vc, lf_coeff: float = 1.0):
    """Conservative momentum fluxes through x- and y-faces.

    F_x[u]: flux of u-momentum through x-face = (u_face)^2  + Uc * u_face
            (using u as transported quantity; advection by total velocity Uc+u)
    Actually the conservative form is rho * U * U where U = full velocity.
    We use rho = 1 and treat the *full* velocities (Uc + u', Vc + v') so the
    decomposition stays simple: define total velocity U = u (which already
    includes Uc in the convecting TGV convention used elsewhere... but here
    `u` is the conservative variable already containing the Uc offset).
    """
    u_xf = face_average_x(u)
    v_xf = face_average_x(v)
    u_yf = face_average_y(u)
    v_yf = face_average_y(v)

    # Fluxes through x-faces (normal velocity = u_xf)
    Fx_u = u_xf * u_xf
    Fx_v = u_xf * v_xf
    # Fluxes through y-faces (normal velocity = v_yf)
    Fy_u = v_yf * u_yf
    Fy_v = v_yf * v_yf

    # Local Lax-Friedrichs stabilization. lf_coeff=1.0 gives the robust
    # first-order-dissipative scheme; lf_coeff=0.0 gives the pure 2nd-order
    # central flux (appropriate for smooth flows like TGV, where the physical
    # viscosity provides the stabilization).
    if lf_coeff > 0.0:
        a_x = lf_coeff * np.maximum(np.abs(u), np.abs(np.roll(u, -1, axis=0)))
        a_y = lf_coeff * np.maximum(np.abs(v), np.abs(np.roll(v, -1, axis=1)))
        Fx_u -= 0.5 * a_x * (np.roll(u, -1, axis=0) - u)
        Fx_v -= 0.5 * a_x * (np.roll(v, -1, axis=0) - v)
        Fy_u -= 0.5 * a_y * (np.roll(u, -1, axis=1) - u)
        Fy_v -= 0.5 * a_y * (np.roll(v, -1, axis=1) - v)
    return Fx_u, Fx_v, Fy_u, Fy_v


def diffusive_fluxes(u, v, nu, dx, dy):
    """Viscous fluxes nu * grad(phi) at faces."""
    Gx_u = nu * (np.roll(u, -1, axis=0) - u) / dx
    Gx_v = nu * (np.roll(v, -1, axis=0) - v) / dx
    Gy_u = nu * (np.roll(u, -1, axis=1) - u) / dy
    Gy_v = nu * (np.roll(v, -1, axis=1) - v) / dy
    return Gx_u, Gx_v, Gy_u, Gy_v


def divergence_from_faces(Fx, Fy, dx, dy):
    """Cell divergence of a face flux: (F_x[i] - F_x[i-1])/dx + (F_y[j] - F_y[j-1])/dy."""
    return (Fx - np.roll(Fx, 1, axis=0)) / dx + (Fy - np.roll(Fy, 1, axis=1)) / dy
