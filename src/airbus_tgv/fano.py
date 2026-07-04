"""Fano plane oriented triples used to define octonion multiplication.

The seven oriented lines (mod sign) of the Fano plane define the multiplication
table of the imaginary octonion units e1..e7. Convention used here matches the
problem statement:

    (1,2,3), (1,4,5), (1,6,7), (2,4,6), (2,5,7), (3,4,7), (3,5,6)

Within each cyclic triple (a, b, c):
    e_a * e_b = +e_c,   e_b * e_c = +e_a,   e_c * e_a = +e_b
and reversing the order flips the sign.
"""
from __future__ import annotations
from typing import Dict, Tuple

FANO_TRIPLES: Tuple[Tuple[int, int, int], ...] = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 6, 7),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 5, 6),
)


def build_multiplication_table() -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Return mapping (i, j) -> (sign, k) for i, j in 1..7 (imaginary basis)."""
    table: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for a, b, c in FANO_TRIPLES:
        # cyclic positive
        table[(a, b)] = (+1, c)
        table[(b, c)] = (+1, a)
        table[(c, a)] = (+1, b)
        # anticyclic negative
        table[(b, a)] = (-1, c)
        table[(c, b)] = (-1, a)
        table[(a, c)] = (-1, b)
    # squares
    for i in range(1, 8):
        table[(i, i)] = (-1, 0)  # e_i^2 = -1 (encoded as -e_0)
    return table


MULT_TABLE: Dict[Tuple[int, int], Tuple[int, int]] = build_multiplication_table()
