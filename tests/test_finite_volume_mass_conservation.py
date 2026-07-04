import numpy as np

from airbus_tgv.constants import TGVParams
from airbus_tgv.finite_volume import (
    run_fv_solver, project_divergence_free, build_periodic_grid,
)
from airbus_tgv.exact_solution import exact_velocity, divergence


def test_projection_removes_divergence():
    X, Y, dx, dy = build_periodic_grid(32, 32)
    # Build a velocity field with deliberate divergence
    u = np.sin(X) * np.cos(Y) + 0.3 * np.sin(2 * X)
    v = -np.cos(X) * np.sin(Y) + 0.2 * np.cos(3 * Y)
    u2, v2, _ = project_divergence_free(u, v, dx, dy)
    d = divergence(u2, v2, dx, dy)
    assert float(np.max(np.abs(d))) < 1e-8


def test_short_run_remains_stable_and_close_to_exact():
    params = TGVParams(Re=100.0)
    res = run_fv_solver(params, nx=32, ny=32, t_final=0.2, cfl=0.4)
    assert not res.diagnostics["unstable"]
    X = res.diagnostics["X"]; Y = res.diagnostics["Y"]
    u_ex, v_ex = exact_velocity(X, Y, res.t, params)
    err = np.sqrt(np.mean((res.u - u_ex) ** 2 + (res.v - v_ex) ** 2))
    # Loose bound for a 32^2 grid with LF dissipation; mostly a non-blow-up check
    assert err < 0.5


def test_mass_conservation_total_momentum_drift_small():
    """Sum over cells of (u, v) should drift only slowly for the projected scheme."""
    params = TGVParams(Re=100.0)
    res = run_fv_solver(params, nx=32, ny=32, t_final=0.1, cfl=0.4)
    X = res.diagnostics["X"]; Y = res.diagnostics["Y"]
    u0, v0 = exact_velocity(X, Y, 0.0, params)
    m0 = (u0.mean(), v0.mean())
    m1 = (res.u.mean(), res.v.mean())
    assert abs(m1[0] - m0[0]) < 1e-6
    assert abs(m1[1] - m0[1]) < 1e-6
