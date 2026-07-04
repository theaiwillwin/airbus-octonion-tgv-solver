"""Tests for trajectory collection and compression."""
import numpy as np
import pytest

from airbus_tgv.constants import TGVParams
from airbus_tgv.trajectory import (
    Trajectory, collect_trajectory, associator_guided_ranks,
    trajectory_kinetic_energy_comparison,
)


def test_trajectory_add_snapshot():
    traj = Trajectory()
    u = np.random.rand(32, 32)
    v = np.random.rand(32, 32)
    traj.add_snapshot(u, v, 0.1)
    assert len(traj.times) == 1
    assert traj.times[0] == 0.1
    assert traj.u_snapshots[0].shape == (32, 32)


def test_trajectory_to_tensor():
    traj = Trajectory()
    for i in range(5):
        u = np.random.rand(16, 16)
        v = np.random.rand(16, 16)
        traj.add_snapshot(u, v, float(i) * 0.1)
    U, V = traj.to_tensor()
    assert U.shape == (5, 16, 16)
    assert V.shape == (5, 16, 16)


def test_trajectory_memory_dense():
    traj = Trajectory()
    for i in range(3):
        traj.add_snapshot(np.zeros((32, 32)), np.zeros((32, 32)), 0.0)
    mem = traj.memory_dense_mb()
    # 3 snapshots * 2 fields * 32*32 * 8 bytes / 1MB ≈ 0.049 MB
    assert mem > 0


def test_collect_trajectory_short():
    """Quick trajectory collection."""
    params = TGVParams(Re=100.0)
    traj = collect_trajectory(params, nx=24, ny=24, t_final=0.05, snapshot_interval=10)
    assert len(traj.times) > 0
    assert traj.X is not None and traj.Y is not None
    assert traj.dx > 0 and traj.dy > 0


def test_associator_guided_ranks():
    """Rank selection from associator."""
    params = TGVParams(Re=100.0)
    traj = collect_trajectory(params, nx=24, ny=24, t_final=0.05, snapshot_interval=20)
    ranks = associator_guided_ranks(traj, min_rank=2, max_rank=16)
    assert "time" in ranks and "x" in ranks and "y" in ranks
    # Ranks are clamped to [min_rank, max_rank] but the mode-dimension cap
    # (dim // 2, never below 1) wins for tiny modes such as few snapshots.
    assert all(1 <= r <= 16 for r in ranks.values())
    nt = len(traj.times)
    assert ranks["time"] <= max(1, nt // 2)
    assert ranks["x"] <= 12 and ranks["y"] <= 12  # 24 // 2


def test_trajectory_kinetic_energy_comparison():
    """KE comparison numerical vs exact."""
    params = TGVParams(Re=100.0)
    traj = collect_trajectory(params, nx=24, ny=24, t_final=0.05, snapshot_interval=30)
    ke_comp = trajectory_kinetic_energy_comparison(traj, params)
    assert "ke_error_mean" in ke_comp
    assert ke_comp["ke_error_mean"] < 0.1  # Loose bound for 2nd-order FV


def test_tucker_decompose_requires_tensorly():
    """Tucker decomposition requires tensorly."""
    try:
        import tensorly  # noqa: F401
        tensorly_available = True
    except ImportError:
        tensorly_available = False

    if not tensorly_available:
        pytest.skip("tensorly not installed")

    params = TGVParams(Re=100.0)
    traj = collect_trajectory(params, nx=24, ny=24, t_final=0.05, snapshot_interval=30)
    from airbus_tgv.trajectory import tucker_decompose_trajectory
    result = tucker_decompose_trajectory(traj)
    assert "compression_ratio" in result
    assert result["compression_ratio"] > 1.0
    assert result["reconstruction_error"] < 0.01
