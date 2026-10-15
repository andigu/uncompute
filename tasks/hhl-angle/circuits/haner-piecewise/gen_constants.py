#!/usr/bin/env python3
"""Regenerate constants.json for hhl-angle haner-piecewise by
running its partition search.

constants.json holds the two tables that circuit.py loads:

    qtab     one [q0, q1, q2, q3] row per interval: the degree-3
             coefficients in units of 2^-38, the signed (44, 6) fixed
             point of the datapath
    bounds   the interior interval boundaries as raw X values, ascending

The search
----------
The search grows the partition left to right over the raw input grid.
For each interval it finds the widest stop that passes ``accepts``.
``accepts`` fits a degree-3 minimax polynomial by Remez exchange,
rounds the coefficients to 2^-38, and runs the circuit's fixed-point
datapath on every integer input of the interval.  It accepts the
interval only if the worst output error is within the contract band
(eps_ulp = 429 units of 2^-32).  The search budgets coefficient
rounding and product truncation directly, so it needs no safety margin.
checks.py re-proves the band over all accepted inputs, against a float
ideal and with ``datapath`` as its model.

Determinism
-----------
All real arithmetic runs in mpmath at a fixed precision.  mpmath is
software arithmetic, so every machine derives the same tables.  The
datapath runs on exact numpy integer arrays.  There is no RNG and no
time dependence.  The run takes about a minute; most of it computes
the exact ideal outputs.

RUN: uv run python gen_constants.py   (writes constants.json)
"""

import json
import pathlib

import numpy as np
from mpmath import mp

from uncompute.testing import load_sibling

S = load_sibling(__file__)
CONSTANTS = pathlib.Path(__file__).resolve().parent / "constants.json"

mp.dps = 30  # about 100 bits: far below every error the search measures
REMEZ_ITERS = 12
NEWTON_ITERS = 8
DEGREE = S.DEGREE

MASK = np.uint64(S.WORD_MASK)
SIGN = np.uint64(1 << (S.WORD_BITS - 1))
RAW = np.arange(S.DOMAIN_RAW_LO, S.DOMAIN_RAW_HI, dtype=np.uint64)


# ---------------------------------------------------------------------------
# The exact ideal outputs
# ---------------------------------------------------------------------------
def ideal_outputs():
    """Return round(arcsin(2^15/X) * 2^32) for every accepted X, rounded
    half away from zero as the grader rounds.  The values are positive,
    so the rule is floor(v + 1/2).  The assert keeps every value far from
    a rounding tie, so mpmath's precision decides each rounding exactly.
    """
    out = np.empty(len(RAW), dtype=np.int64)
    scale = mp.mpf(2) ** S.OUTPUT_FRAC_BITS
    tie_gap = mp.mpf(2) ** -40
    for i, raw in enumerate(range(S.DOMAIN_RAW_LO, S.DOMAIN_RAW_HI)):
        v = mp.asin(mp.mpf(1 << 15) / raw) * scale
        out[i] = int(mp.floor(v + mp.mpf(0.5)))
        assert abs(v - mp.floor(v) - mp.mpf(0.5)) > tie_gap, raw
    return out


# ---------------------------------------------------------------------------
# The degree-3 minimax fit (Remez exchange) of f(x) = arcsin(1/(2x))
# ---------------------------------------------------------------------------
def f(x):
    return mp.asin(1 / (2 * x))


def f_slopes(x):
    """Return f'(x) and f''(x): -1/(x s) and (8x^2 - 1)/(x^2 s^3), with
    s = sqrt(4x^2 - 1).
    """
    s = mp.sqrt(4 * x * x - 1)
    return -1 / (x * s), (8 * x * x - 1) / (x * x * s**3)


def poly(c, x):
    return c[0] + x * (c[1] + x * (c[2] + x * c[3]))


def error_extremum(c, x, a, b):
    """Move x toward a root of the error slope p' - f' by Newton steps,
    clamped to the open interval (a, b).
    """
    for _ in range(NEWTON_ITERS):
        first, second = f_slopes(x)
        slope = c[1] + 2 * c[2] * x + 3 * c[3] * x * x - first
        curve = 2 * c[2] + 6 * c[3] * x - second
        x = min(max(x - slope / curve, a), b)
    return x


def remez(a, b):
    """Return the power-basis coefficients [c0, c1, c2, c3] of the
    minimax fit of f on [a, b], or None if the exchange breaks the
    reference order.  That failure occurs only on very wide probes, and
    the search treats it as a failing interval.
    """
    nodes = [a + (b - a) * (1 - mp.cos(mp.pi * i / (DEGREE + 1))) / 2 for i in range(DEGREE + 2)]
    for _ in range(REMEZ_ITERS):
        # Equioscillation: p(x_i) + (-1)^i E = f(x_i) on the five nodes.
        system = mp.matrix(
            [[x**j for j in range(DEGREE + 1)] + [(-1) ** i] for i, x in enumerate(nodes)]
        )
        c = list(mp.lu_solve(system, mp.matrix([f(x) for x in nodes])))[: DEGREE + 1]
        nodes = [a, *(error_extremum(c, x, a, b) for x in nodes[1:-1]), b]
        if not all(u < v for u, v in zip(nodes, nodes[1:])):
            return None
        errors = [abs(poly(c, x) - f(x)) for x in nodes]
        if max(errors) - min(errors) <= max(errors) * mp.mpf(2) ** -20:
            break
    return c


def quantize(c):
    """Round each coefficient to the nearest multiple of 2^-FRAC_BITS."""
    return [int(mp.floor(cj * 2**S.FRAC_BITS + mp.mpf(0.5))) for cj in c]


# ---------------------------------------------------------------------------
# The circuit's fixed-point datapath on exact integer arrays
# ---------------------------------------------------------------------------
def horner_product(v, x):
    """Return the circuit's pre-truncated signed product of the words v
    and the raw inputs x, mod 2^44.  Row ``bit`` adds x shifted by
    bit - 16 (truncated below the word's 2^-38 place) under bit ``bit``
    of v.  The top row subtracts, because that bit carries weight -2^5.
    """
    total = np.zeros(len(x), dtype=np.uint64)
    for bit in range(S.WORD_BITS):
        shift = bit - S.INPUT_FRAC_BITS
        term = (x << np.uint64(shift)) if shift >= 0 else (x >> np.uint64(-shift))
        term = term * ((v >> np.uint64(bit)) & np.uint64(1))
        total = total - term if bit == S.WORD_BITS - 1 else total + term
    return total & MASK


def datapath(q, x):
    """Return the circuit's outputs for the quantized row q on raw inputs
    x, or None if the final word is negative (the round would wrap).
    """
    words = [np.uint64(qj & S.WORD_MASK) for qj in q]  # two's complement
    value = np.full(len(x), words[DEGREE])
    for word in reversed(words[:DEGREE]):
        value = (horner_product(value, x) + word) & MASK
    if np.any(value >= SIGN):
        return None
    return ((value + np.uint64(S.ROUND_HALF)) >> np.uint64(S.ROUND_SHIFT)).astype(np.int64)


# ---------------------------------------------------------------------------
# The partition search
# ---------------------------------------------------------------------------
def fit_interval(start, stop, ideal):
    """Fit raw inputs [start, stop); return the quantized row if the
    datapath meets the contract band on every input, else None.
    """
    one = mp.mpf(1 << S.INPUT_FRAC_BITS)
    c = remez(start / one, (stop - 1) / one)
    if c is None:
        return None
    q = quantize(c)
    lo, hi = start - S.DOMAIN_RAW_LO, stop - S.DOMAIN_RAW_LO
    actual = datapath(q, RAW[lo:hi])
    if actual is None or np.max(np.abs(actual - ideal[lo:hi])) > S.ALLOWED_ULP:
        return None
    return q


def widest_interval(start, guess, ideal):
    """Return (stop, row) for the widest passing interval [start, stop).

    The search starts at width ``guess``.  It doubles the width while the
    interval passes, halves it while the interval fails, and then bisects
    between the widest pass and the narrowest fail.  Every returned stop
    passed its own fit.
    """
    end = S.DOMAIN_RAW_HI

    def probe(width):
        stop = min(start + width, end)
        return stop, fit_interval(start, stop, ideal)

    stop, row = probe(guess)
    if row is not None:
        good, bad = (stop, row), end + 1
        while good[0] < end:
            stop, row = probe(2 * (good[0] - start))
            if row is None:
                bad = stop
                break
            good = (stop, row)
    else:
        bad, width = stop, guess // 2
        while (row := fit_interval(start, start + width, ideal)) is None:
            bad, width = start + width, width // 2
        good = (start + width, row)
    while bad - good[0] > 1:
        mid = (good[0] + bad) // 2
        row = fit_interval(start, mid, ideal)
        good, bad = ((mid, row), bad) if row is not None else (good, mid)
    return good


def tables():
    """Run the partition search; return (qtab, bounds)."""
    ideal = ideal_outputs()
    qtab, bounds, start, width = [], [], S.DOMAIN_RAW_LO, 1 << 12
    while True:
        stop, row = widest_interval(start, width, ideal)
        qtab.append(row)
        if stop == S.DOMAIN_RAW_HI:
            return qtab, bounds
        bounds.append(stop)
        width, start = stop - start, stop


def main():
    qtab, bounds = tables()
    rows = ",\n    ".join(json.dumps(row) for row in qtab)
    new = f'{{\n  "qtab": [\n    {rows}\n  ],\n  "bounds": {json.dumps(bounds)}\n}}\n'
    old = CONSTANTS.read_text() if CONSTANTS.exists() else None
    CONSTANTS.write_text(new)
    print(f"{len(qtab)} intervals; constants.json {'unchanged' if new == old else 'WRITTEN'}")


if __name__ == "__main__":
    main()
