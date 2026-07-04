"""Tests for divergence-free multi-mode perturbations."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbus_tgv.constants import TGVParams
from airbus_tgv.exact_solution import divergence
from airbus_tgv.finite_volume import build_periodic_grid, run_fv_solver
from airbus_tgv.perturbation import (
    streamfunction_perturbation, perturbed_tgv_initial_condition,
)


def test_perturbation_is_divergence_free():
    X, Y, dx, dy = build_periodic_grid(64, 64)
    u_p, v_p = streamfunction_perturbation(X, Y, n_modes=4, amplitude=0.1)
    div = divergence(u_p, v_p, dx, dy)
    assert float(np.max(np.abs(div))) < 1e-10


def test_perturbation_amplitude_scaling():
    X, Y, dx, dy = build_periodic_grid(64, 64)
    u_p, v_p = streamfunction_perturbation(X, Y, amplitude=0.25)
    peak = max(float(np.max(np.abs(u_p))), float(np.max(np.abs(v_p))))
    assert peak == pytest.approx(0.25, rel=1e-12)


def test_perturbation_deterministic_by_seed():
    X, Y, _, _ = build_periodic_grid(32, 32)
    u1, v1 = streamfunction_perturbation(X, Y, seed=7)
    u2, v2 = streamfunction_perturbation(X, Y, seed=7)
    u3, _ = streamfunction_perturbation(X, Y, seed=8)
    assert np.array_equal(u1, u2) and np.array_equal(v1, v2)
    assert not np.array_equal(u1, u3)


def test_perturbed_ic_contains_tgv_plus_perturbation():
    params = TGVParams(Re=100.0)
    X, Y, dx, dy = build_periodic_grid(32, 32)
    ic = perturbed_tgv_initial_condition(amplitude=0.1, seed=0)
    u, v = ic(X, Y, params)
    from airbus_tgv.exact_solution import exact_velocity
    u0, v0 = exact_velocity(X, Y, 0.0, params)
    diff = max(float(np.max(np.abs(u - u0))), float(np.max(np.abs(v - v0))))
    assert 0.0 < diff <= 0.1 + 1e-12


def test_solver_stable_with_perturbed_ic():
    params = TGVParams(Re=100.0)
    ic = perturbed_tgv_initial_condition(n_modes=4, amplitude=0.1, seed=0)
    res = run_fv_solver(params, nx=32, ny=32, t_final=0.1, cfl=0.4,
                        lf_coeff=0.0, initial_condition=ic)
    assert not res.diagnostics["unstable"]
    assert np.isfinite(res.u).all() and np.isfinite(res.v).all()
    # incompressibility is maintained by projection
    dx = res.diagnostics["dx"]; dy = res.diagnostics["dy"]
    div = divergence(res.u, res.v, dx, dy)
    assert float(np.sqrt(np.mean(div ** 2))) < 1e-6
