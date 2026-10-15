#!/usr/bin/env python3
"""Regenerate constants.json for the haner-piecewise circuit by
running the interval search.

The datapath evaluates u^{-1/2} on the mantissa u = n2 / 2^e in [1, 2).
The search tiles the 25-bit mantissa words [2^24, 2^25) left to right.
For each interval, bisection finds the largest right threshold that
passes the acceptance test:

  1. mpmath fits the degree-2 minimax (Remez) polynomial of u^{-1/2} on
     the interval at a fixed working precision.
  2. The coefficients round to the signed (26, 2) fixed-point grid, one
     bank per exponent parity (the odd bank scaled by 2^-1/2).
  3. The exact integer datapath (pre-truncated Horner rows, rescale
     truncation) runs on every achievable n2 whose mantissa word falls in
     the interval.  The interval passes when every output is within the
     task band ALLOWED_ULP of the ideal table.

The accepted tables therefore meet the band by construction.  checks.py
re-proves the worst error over the whole domain, with ``horner_step`` as
its multiplier model.

constants.json holds the two tables circuit.py builds from:

    qtab        qtab[parity][interval] = [q0, q1, q2], the quantized
                power-basis coefficients (integers on the 2^-24 grid)
    thresholds  the interior interval boundaries as mantissa words:
                interval i + 1 starts at thresholds[i]

Determinism: there is no RNG and no time dependence.  The fit runs in
mpmath software arithmetic at a fixed precision, and the datapath runs
in exact integer arithmetic, so every machine writes the same tables.

RUN: uv run python gen_constants.py   (writes constants.json;
     review the change with git diff)
"""

from __future__ import annotations

import json
import pathlib

import mpmath
import numpy as np

from uncompute.testing import load_contract, load_module, task_dir

HERE = pathlib.Path(__file__).resolve().parent
CONSTANTS = HERE / "constants.json"
R = load_module(task_dir(__file__) / "reference.py", "coulomb_potential_reference")
ALLOWED_ULP = load_contract(__file__)["EPS"]

mpmath.mp.prec = 128  # fixed working precision of the fit

DEGREE = 2  # polynomial degree of this configuration
WORD_BITS = 26  # n: signed fixed-point word
INT_BITS = 2  # p: integer bits (fixed by 1 <= u < 2)
FRAC_BITS = WORD_BITS - INT_BITS  # 24
REMEZ_ITERATIONS = 6  # exchange steps; the reference is converged by then
U_LO = 1 << FRAC_BITS  # mantissa word of u = 1
U_HI = 1 << (FRAC_BITS + 1)  # mantissa word of u = 2 (exclusive)


# ---------------------------------------------------------------------------
# The fit: degree-2 minimax of u^{-1/2} in mpmath
# ---------------------------------------------------------------------------


def error_extrema(coefficients, a, b):
    """Return the two interior extrema of the error p(u) - u^{-1/2} on
    [a, b].  The error's derivative g = c1 + 2 c2 u + u^{-3/2} / 2 is
    convex, so it has at most two roots, one on each side of its minimum.
    """
    _c0, c1, c2 = coefficients

    def slope(u):  # g(u), the error's derivative
        return c1 + 2 * c2 * u + u**-1.5 / 2

    bottom = (3 / (8 * c2)) ** (mpmath.mpf(2) / 5)  # where g' = 2 c2 - (3/4) u^{-5/2} = 0
    assert a < bottom < b
    return [
        mpmath.findroot(slope, (lo, hi), solver="anderson") for lo, hi in ((a, bottom), (bottom, b))
    ]


def minimax(a, b):
    """Return the degree-2 minimax coefficients [c0, c1, c2] of u^{-1/2}
    on [a, b] by the Remez exchange.
    """
    quarter = (b - a) / 4  # Chebyshev extrema of degree 3: cos(k pi / 3)
    nodes = [a, a + quarter, b - quarter, b]
    for _ in range(REMEZ_ITERATIONS):
        # Solve p(x_k) + (-1)^k E = x_k^{-1/2} for (c0, c1, c2, E).
        system = mpmath.matrix([[1, x, x * x, (-1) ** k] for k, x in enumerate(nodes)])
        solution = mpmath.lu_solve(system, mpmath.matrix([x**-0.5 for x in nodes]))
        coefficients = [solution[j] for j in range(DEGREE + 1)]
        nodes = [a, *error_extrema(coefficients, a, b), b]
    return coefficients


def quantized_banks(lo: int, hi: int) -> list[list[int]]:
    """Fit the interval [lo, hi) of mantissa words; return the two parity
    banks [[q0, q1, q2], [q0, q1, q2]] rounded onto the 2^-24 grid.
    """
    coefficients = minimax(mpmath.mpf(lo) / U_LO, mpmath.mpf(hi) / U_LO)
    return [
        [int(mpmath.nint(c * scale * U_LO)) for c in coefficients]
        for scale in (1, 1 / mpmath.sqrt(2))
    ]


# ---------------------------------------------------------------------------
# The datapath: exact integer model on the achievable domain
# ---------------------------------------------------------------------------

DOMAIN = np.asarray(R.DOMAIN, dtype=np.uint64)
EXPONENT = np.asarray([int(v).bit_length() - 1 for v in R.DOMAIN], dtype=np.uint64)
U_WORD = DOMAIN << (np.uint64(FRAC_BITS) - EXPONENT)
PARITY = EXPONENT & np.uint64(1)
HALF_EXPONENT = EXPONENT >> np.uint64(1)
IDEAL = np.asarray([R.ideal_f(int(v)) for v in R.DOMAIN], dtype=np.int64)
MASK = np.uint64((1 << WORD_BITS) - 1)


def horner_step(value, u_word, coefficient: int):
    """One Horner step on the word: (value (x) u) + coefficient, with (x)
    the HRS signed pre-truncated multiplier.  Row ``bit`` adds u shifted
    by bit - 24 under control bit ``bit``; the top row subtracts (the
    two's-complement sign).
    """
    product = np.zeros(len(u_word), dtype=np.uint64)
    for bit in range(WORD_BITS):
        shift = bit - FRAC_BITS
        row = (u_word << np.uint64(shift)) & MASK if shift >= 0 else u_word >> np.uint64(-shift)
        row = row * ((value >> np.uint64(bit)) & np.uint64(1))
        product = product - row if bit == WORD_BITS - 1 else product + row
    return (product + np.uint64(coefficient & int(MASK))) & MASK


def worst_error(lo: int, hi: int, banks: list[list[int]]) -> int:
    """Return the worst |output - ideal| over the achievable n2 whose
    mantissa word lies in [lo, hi).  A negative final word (sign bit set)
    shifts into a huge output, so it fails the band too.
    """
    worst = 0
    for parity, bank in enumerate(banks):
        selected = (U_WORD >= lo) & (U_WORD < hi) & (PARITY == parity)
        if not selected.any():
            continue
        u_word = U_WORD[selected]
        value = np.full(len(u_word), np.uint64(bank[-1] & int(MASK)))
        for coefficient in reversed(bank[:-1]):
            value = horner_step(value, u_word, coefficient)
        output = (value >> HALF_EXPONENT[selected]).astype(np.int64)
        worst = max(worst, int(np.abs(output - IDEAL[selected]).max()))
    return worst


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def passes(lo: int, hi: int) -> bool:
    return worst_error(lo, hi, quantized_banks(lo, hi)) <= ALLOWED_ULP


def largest_passing_threshold(lo: int) -> int:
    """Return the largest hi in (lo, U_HI] with [lo, hi) passing."""
    if passes(lo, U_HI):
        return U_HI
    good, bad = lo + 1, U_HI
    assert passes(lo, good)
    while bad - good > 1:
        mid = (good + bad) // 2
        if passes(lo, mid):
            good = mid
        else:
            bad = mid
    return good


def main():
    lefts = [U_LO]
    while (right := largest_passing_threshold(lefts[-1])) < U_HI:
        lefts.append(right)
    rights = [*lefts[1:], U_HI]
    banks = [quantized_banks(lo, hi) for lo, hi in zip(lefts, rights)]
    constants = {
        "qtab": [[bank[parity] for bank in banks] for parity in (0, 1)],
        "thresholds": lefts[1:],
    }
    CONSTANTS.write_text(json.dumps(constants, indent=1) + "\n")
    print(f"wrote {CONSTANTS.name}: {len(banks)} intervals per parity")


if __name__ == "__main__":
    main()
