import numpy as np

from airbus_tgv.constants import TGVParams
from airbus_tgv.finite_volume import build_periodic_grid
from airbus_tgv.exact_solution import exact_velocity, exact_pressure
from airbus_tgv.associator_metrics import (
    associator_field, associator_summary_stats,
    associator_guided_compression_mask, associator_guided_dt_factor,
    embed_octonion_field,
)


def test_associator_field_shapes_and_finiteness():
    X, Y, dx, dy = build_periodic_grid(32, 32)
    params = TGVParams()
    u, v = exact_velocity(X, Y, 0.0, params)
    p = exact_pressure(X, Y, 0.0, params)
    out = associator_field(u, v, p, dx, dy, mode="pressure")
    for k in ("A", "A_x", "A_y", "A_diag"):
        assert out[k].shape == u.shape
        assert np.all(np.isfinite(out[k]))


def test_embedding_modes():
    X, Y, dx, dy = build_periodic_grid(16, 16)
    params = TGVParams()
    u, v = exact_velocity(X, Y, 0.0, params)
    p = exact_pressure(X, Y, 0.0, params)
    op = embed_octonion_field(u, v, dx, dy, p=p, mode="pressure")
    os = embed_octonion_field(u, v, dx, dy, mode="strain")
    assert op.shape == u.shape + (8,)
    assert os.shape == u.shape + (8,)


def test_summary_stats_keys():
    X, Y, dx, dy = build_periodic_grid(16, 16)
    params = TGVParams()
    u, v = exact_velocity(X, Y, 0.0, params)
    A = associator_field(u, v, None, dx, dy, mode="strain")["A"]
    stats = associator_summary_stats(A)
    for k in ("associator_mean", "associator_std",
              "associator_p50", "associator_p95", "associator_max"):
        assert k in stats


def test_compression_mask_fraction():
    X, Y, dx, dy = build_periodic_grid(32, 32)
    params = TGVParams()
    u, v = exact_velocity(X, Y, 0.0, params)
    A = associator_field(u, v, None, dx, dy, mode="strain")["A"]
    mask = associator_guided_compression_mask(A, quantile=0.9)
    frac = mask.mean()
    assert 0.0 < frac < 0.2  # roughly the top decile


def test_dt_factor_bounds():
    A = np.random.rand(8, 8)
    f = associator_guided_dt_factor(A)
    assert 0.25 <= f <= 1.0
