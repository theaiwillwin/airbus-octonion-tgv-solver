import math
from airbus_tgv.constants import TGVParams
from airbus_tgv.benchmark import run_method


def test_classical_fv_short_run():
    params = TGVParams(Re=100.0)
    r = run_method("classical_fv", params, nx=24, ny=24, t_final=0.05)
    assert r.stable
    assert r.l2_velocity_error < 1.0
    assert r.divergence_l2 < 1e-4
    assert r.compression_ratio == 1.0
    assert math.isfinite(r.associator_mean)


def test_associator_diagnostic_run():
    params = TGVParams(Re=100.0)
    r = run_method("fv_plus_associator_diagnostic", params, nx=24, ny=24, t_final=0.05)
    assert r.stable
    assert r.associator_p95 >= r.associator_mean


def test_associator_guided_compression():
    params = TGVParams(Re=100.0)
    r = run_method("fv_plus_associator_guided_compression",
                   params, nx=24, ny=24, t_final=0.05)
    assert r.stable
    assert r.memory_compressed_mb > 0
    assert r.compression_ratio >= 1.0
    assert 0.0 <= r.compression_rel_err < 1.0
