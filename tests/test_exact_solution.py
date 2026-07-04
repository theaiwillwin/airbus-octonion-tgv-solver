import numpy as np
import pytest

from airbus_tgv.constants import TGVParams
from airbus_tgv.exact_solution import (
    exact_velocity, exact_pressure, kinetic_energy, l2_error, divergence,
)
from airbus_tgv.finite_volume import build_periodic_grid


@pytest.fixture
def grid():
    X, Y, dx, dy = build_periodic_grid(32, 32)
    return X, Y, dx, dy


def test_initial_condition_matches_problem_statement(grid):
    X, Y, dx, dy = grid
    params = TGVParams(Re=100.0)
    u, v = exact_velocity(X, Y, 0.0, params)
    k = params.wave_number
    u_ic = params.Uc + params.V0 * np.sin(k * X) * np.cos(k * Y)
    v_ic = params.Vc - params.V0 * np.cos(k * X) * np.sin(k * Y)
    assert np.allclose(u, u_ic)
    assert np.allclose(v, v_ic)


def test_decay_factor_decreases(grid):
    X, Y, _, _ = grid
    params = TGVParams(Re=10.0)  # higher viscosity -> noticeable decay
    u0, v0 = exact_velocity(X, Y, 0.0, params)
    u1, v1 = exact_velocity(X, Y, 1.0, params)
    # Remove mean convection to compare fluctuation magnitudes
    amp0 = np.std(u0 - params.Uc) + np.std(v0 - params.Vc)
    amp1 = np.std(u1 - params.Uc) + np.std(v1 - params.Vc)
    assert amp1 < amp0


def test_shapes(grid):
    X, Y, _, _ = grid
    params = TGVParams()
    u, v = exact_velocity(X, Y, 0.0, params)
    p = exact_pressure(X, Y, 0.0, params)
    assert u.shape == v.shape == p.shape == X.shape


def test_kinetic_energy_positive_and_decays(grid):
    X, Y, _, _ = grid
    params = TGVParams(Re=10.0)
    u0, v0 = exact_velocity(X, Y, 0.0, params)
    u1, v1 = exact_velocity(X, Y, 1.0, params)
    # Subtract the mean (convective) part: it does not decay.
    ke0 = kinetic_energy(u0 - params.Uc, v0 - params.Vc)
    ke1 = kinetic_energy(u1 - params.Uc, v1 - params.Vc)
    assert ke0 > 0 and ke1 > 0 and ke1 < ke0


def test_l2_error_zero_for_identical(grid):
    X, Y, _, _ = grid
    params = TGVParams()
    u, v = exact_velocity(X, Y, 0.5, params)
    assert l2_error(u, v, u, v) == 0.0


def test_divergence_of_exact_is_small(grid):
    X, Y, dx, dy = grid
    params = TGVParams()
    u, v = exact_velocity(X, Y, 0.0, params)
    d = divergence(u, v, dx, dy)
    assert float(np.max(np.abs(d))) < 0.05  # 2nd-order on 32^2 grid
