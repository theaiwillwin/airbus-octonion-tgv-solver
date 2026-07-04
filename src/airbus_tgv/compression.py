"""Field compression utilities.

Primary baseline: truncated SVD ("matrix-factorized tensor-inspired
compression baseline" in the documentation). If `tensorly` is available we
expose Tucker decomposition as well, but never silently substitute it.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    import tensorly as tl  # type: ignore
    from tensorly.decomposition import tucker  # type: ignore
    _HAS_TENSORLY = True
except Exception:
    _HAS_TENSORLY = False


def svd_compress_field(field: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD of a 2D field. Returns (U, s, Vt) truncated to `rank`."""
    if field.ndim != 2:
        raise ValueError("svd_compress_field expects 2D input")
    U, s, Vt = np.linalg.svd(field, full_matrices=False)
    rank = max(1, min(rank, s.size))
    return U[:, :rank], s[:rank], Vt[:rank, :]


def reconstruct_svd(U, s, Vt) -> np.ndarray:
    return (U * s) @ Vt


def compression_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.linalg.norm(original - reconstructed) / (np.linalg.norm(original) + 1e-30))


def memory_estimate_dense(field: np.ndarray) -> float:
    """MB of a dense ndarray."""
    return field.nbytes / (1024.0 * 1024.0)


def memory_estimate_compressed(U, s, Vt) -> float:
    return (U.nbytes + s.nbytes + Vt.nbytes) / (1024.0 * 1024.0)


def associator_guided_rank_policy(A: np.ndarray, min_rank: int = 4,
                                  max_rank: int = 32) -> int:
    """Choose a rank in [min_rank, max_rank] proportional to associator p95.

    Larger p95 (more relational torsion) => keep more rank.
    """
    if min_rank < 1 or max_rank < min_rank:
        raise ValueError("invalid rank range")
    p95 = float(np.percentile(A, 95))
    mean = float(np.mean(A)) + 1e-12
    ratio = min(p95 / mean, 10.0) / 10.0  # in [0, 1]
    rank = int(round(min_rank + ratio * (max_rank - min_rank)))
    return max(min_rank, min(max_rank, rank))


def tucker_compress_field(field: np.ndarray, ranks):
    """Optional Tucker decomposition. Raises if tensorly is not installed."""
    if not _HAS_TENSORLY:
        raise RuntimeError(
            "tensorly is not installed; install with `pip install tensorly` "
            "to enable Tucker compression. The SVD baseline remains available."
        )
    return tucker(tl.tensor(field), rank=ranks)
