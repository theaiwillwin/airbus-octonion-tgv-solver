import numpy as np
import pytest

from airbus_tgv.octonion import octonion_mul, octonion_conj, octonion_norm, associator, e
from airbus_tgv.fano import FANO_TRIPLES


def test_identity():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    e0 = e(0)
    assert np.allclose(octonion_mul(e0, a), a)
    assert np.allclose(octonion_mul(a, e0), a)


@pytest.mark.parametrize("i", list(range(1, 8)))
def test_imaginary_units_square_to_minus_one(i):
    ei = e(i)
    prod = octonion_mul(ei, ei)
    expected = -e(0)
    assert np.allclose(prod, expected)


@pytest.mark.parametrize("triple", FANO_TRIPLES)
def test_fano_triples_cyclic(triple):
    a, b, c = triple
    ea, eb, ec = e(a), e(b), e(c)
    # e_a * e_b = +e_c
    assert np.allclose(octonion_mul(ea, eb), ec)
    # e_b * e_a = -e_c
    assert np.allclose(octonion_mul(eb, ea), -ec)


def test_norm_multiplicative_on_basis():
    for i in range(8):
        assert octonion_norm(e(i)) == pytest.approx(1.0)


def test_conjugate():
    a = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
    aa = octonion_mul(a, octonion_conj(a))
    # a * conj(a) = |a|^2 * e0
    assert aa[0] == pytest.approx(np.sum(a * a))
    assert np.allclose(aa[1:], 0.0, atol=1e-10)


def test_known_nonzero_associator():
    # e1, e2, e5 do not lie in one quaternionic Fano line, so the associator is non-zero.
    A = associator(e(1), e(2), e(5))
    assert np.linalg.norm(A) > 0.5


def test_quaternionic_subalgebra_associates():
    # Any Fano triple generates an associative subalgebra (quaternionic).
    for a, b, c in FANO_TRIPLES:
        ea, eb, ec = e(a), e(b), e(c)
        for x in (ea, eb, ec):
            for y in (ea, eb, ec):
                for z in (ea, eb, ec):
                    assert np.allclose(associator(x, y, z), 0.0, atol=1e-10)
