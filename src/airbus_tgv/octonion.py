"""Octonion algebra built from the Fano plane multiplication table.

An octonion is represented as an 8-component real vector
    o = (o0, o1, o2, o3, o4, o5, o6, o7)
where o0 is the real part and o1..o7 are the imaginary basis coefficients.
"""
from __future__ import annotations
import numpy as np

from .fano import MULT_TABLE


def _ensure(a):
    a = np.asarray(a, dtype=float)
    if a.shape[-1] != 8:
        raise ValueError(f"octonion arrays must have last dim 8, got {a.shape}")
    return a


# Pre-build the structure constants once. C[i,j,k] is the coefficient of e_k
# in e_i * e_j, with 0 = real unit.
def _structure_constants() -> np.ndarray:
    C = np.zeros((8, 8, 8), dtype=float)
    # real * anything = anything, anything * real = anything
    for k in range(8):
        C[0, k, k] = 1.0
        C[k, 0, k] = 1.0
    # i, j in 1..7
    for i in range(1, 8):
        for j in range(1, 8):
            sign, k = MULT_TABLE[(i, j)]
            C[i, j, k] += float(sign)
    return C


_C = _structure_constants()


def octonion_mul(a, b):
    """Multiply two octonions (broadcasts over leading dims)."""
    a = _ensure(a)
    b = _ensure(b)
    # c_k = sum_{i,j} C[i,j,k] * a_i * b_j
    return np.einsum("ijk,...i,...j->...k", _C, a, b)


def octonion_conj(a):
    a = _ensure(a).copy()
    a[..., 1:] *= -1.0
    return a


def octonion_norm(a) -> np.ndarray:
    a = _ensure(a)
    return np.sqrt(np.sum(a * a, axis=-1))


def associator(a, b, c):
    """Associator [a, b, c] = (a*b)*c - a*(b*c)."""
    return octonion_mul(octonion_mul(a, b), c) - octonion_mul(a, octonion_mul(b, c))


def associator_norm(a, b, c) -> np.ndarray:
    return octonion_norm(associator(a, b, c))


# Convenience constructors -----------------------------------------------------

def e(i: int) -> np.ndarray:
    """Return the i-th standard basis octonion (i in 0..7)."""
    v = np.zeros(8)
    v[i] = 1.0
    return v
