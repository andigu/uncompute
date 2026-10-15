#!/usr/bin/env python3
"""Regenerate constants.json for the haner-piecewise circuit: sweep the
configuration envelope and fit the chosen configuration's tables.

The sweep picks the configuration of the direct HRS design
(arXiv:1805.12445): the argmin of the paper's own closed-form
accounting over a bounded envelope.  The envelope is the four printed
degrees 3..6, word sizes n = 39..42 at the domain-forced integer width
p = 7, and every applicable Horner pebbling schedule in the paper's
Table I.  Each candidate builds its largest-passing-interval partition
with ``build_partition`` below, so every row already meets the 1e-7
error tier.  The sweep does not search circuit identities, nonuniform
fixed-point formats, lookup layouts, exceptional values, or arithmetic
blocks outside the paper.

Partition.  Each interval grows to the largest right endpoint whose
error bound stays inside the tier.  A Remez fit of exp(-t) on the
interval gives the polynomial; the fit rounds its local-coordinate
coefficients to the n-bit word.  The error bound then evaluates that
ROUNDED polynomial against exp exactly on a dense grid.  It adds the
interpolation remainder between grid points, derived from a bound on
the error's second derivative.  It also adds the datapath's truncation
and rounding bounds.  So the bound holds by construction, with no
safety factor.

Determinism.  The fit uses mpmath at a fixed precision for exp, cos,
and pi, and exact integers and Fractions for everything else.  No
hardware float reaches the tables, so every machine writes the same
constants.json.

Overflow policy.  quantized_fit raises when a local coefficient misses
the signed n-bit word.  The sweep prices such a probe as infeasible
(bound = +inf).  A configuration whose partition then accepts no
interval is infeasible.  With the exact fit, no probe in the envelope
overflows.

Pricing is the HRS closed form: the
paper's T_fma / T_extra / LABEL formulas and its factor 2 for
uncompute.  The sweep extends it with the local-coordinate offset, the
sign fold, and a 2-per-carry-position accumulate.

constants.json holds the argmin configuration and its tables:

    LEFTS         the left endpoint of each interval, as raw w = x^2 * 2^frac
    COEFFICIENTS  one [c0, ..., c_degree] row of n-bit two's-complement
                  local-coordinate coefficient patterns per interval

RUN: uv run python gen_constants.py   (writes constants.json;
     review the change with git diff)
"""

from __future__ import annotations

import collections
import functools
import itertools
import json
import math
import operator
import pathlib
from fractions import Fraction

import mpmath

from uncompute.testing import load_contract

HERE = pathlib.Path(__file__).resolve().parent
CONSTANTS = HERE / "constants.json"

_CONTRACT = load_contract(__file__)
INPUT_BITS = _CONTRACT["n_x"]  # X: 39-bit two's complement input
OUTPUT_BITS = _CONTRACT["m"]  # k: 39-bit modular accumulator
OUTPUT_FRAC = _CONTRACT["frac_bits"]  # 32

TIER_ULP = Fraction(1 << OUTPUT_FRAC, 10**7)  # the HRS 1e-7 tier, continuous
assert math.floor(TIER_ULP) == _CONTRACT["eps_ulp"]
GRID_SIZE = 4000  # dense grid u = i / (GRID_SIZE - 1) on the unit interval
PREC = 96  # fraction bits of the fixed-point fit arithmetic
ONE = 1 << PREC
# The fixed-point exp targets on the grid err by under 2^13 units of
# 2^-PREC: 1.5 units per recurrence step over 4000 steps, plus the
# amplitude product.
TARGET_SLACK = Fraction(1 << 13, ONE)

SWEEP_DEGREES = range(3, 7)  # HRS's four printed degrees
SWEEP_WORD_BITS = range(39, 43)  # word sizes around the selected n
SWEEP_INT_BITS = 7  # domain-forced: w = x^2 < 64 + sign

# HRS Table I: pebble moves to leave the last iterate while clearing the
# earlier ones, indexed by {sequential stages: {live registers: moves}}.
PEBBLE_MOVES = {
    4: {3: 9, 4: 7},
    5: {4: 11, 5: 9},
    6: {4: 15, 5: 13, 6: 11},
    7: {4: 19, 5: 17, 6: 15, 7: 13},
}


# One partition interval [left_raw, right_raw) of raw w = x^2 * 2^frac:
# its quantized coefficient word patterns and its error bound.
Interval = collections.namedtuple(
    "Interval", ["left_raw", "right_raw", "coefficients", "error_bound_ulp"]
)


def exp_neg(x: Fraction) -> int:
    """exp(-x) for rational x >= 0, rounded to the 2^-PREC grid."""
    with mpmath.workprec(PREC + 32):
        return int(mpmath.nint(mpmath.exp(-mpmath.mpf(x.numerator) / x.denominator) * ONE))


def exp_grid(width: Fraction) -> list[int]:
    """exp(-width u) at every grid point u, on the 2^-PREC grid, by
    repeated multiplication with one step factor.
    """
    step = exp_neg(width / (GRID_SIZE - 1))
    multiply = lambda value, _: value * step >> PREC
    return list(itertools.accumulate(range(GRID_SIZE - 1), multiply, initial=ONE))


@functools.cache
def chebyshev_nodes(degree: int) -> tuple[Fraction, ...]:
    """Return the degree + 2 Chebyshev extrema (1 - cos(pi k / (degree + 1))) / 2
    on [0, 1], rounded to the 2^-PREC grid: the Remez starting nodes.
    """
    with mpmath.workprec(PREC + 32):
        return tuple(
            Fraction(
                int(mpmath.nint((1 - mpmath.cos(mpmath.pi * k / (degree + 1))) / 2 * ONE)), ONE
            )
            for k in range(degree + 2)
        )


def solve(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Solve a square linear system exactly (Gauss-Jordan over Fractions)."""
    rows = [row + [value] for row, value in zip(matrix, rhs)]
    size = len(rows)
    for column in range(size):
        pivot = next(r for r in range(column, size) if rows[r][column] != 0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        pivot_row = rows[column]
        for r in range(size):
            factor = rows[r][column] / pivot_row[column]
            if r != column and factor != 0:
                rows[r] = [a - factor * b for a, b in zip(rows[r], pivot_row)]
    return [rows[r][size] / rows[r][r] for r in range(size)]


def grid_values(coefficients: list[int]) -> list[int]:
    """sum_k coefficients[k] * i^k * N^(degree - k) at every grid index i,
    with N = GRID_SIZE - 1: the polynomial at u = i / N, times N^degree.

    The values are exact.  Repeated running sums build them from the
    forward differences at i = 0.
    """
    degree = len(coefficients) - 1
    last = GRID_SIZE - 1
    scaled = [c * last ** (degree - power) for power, c in enumerate(coefficients)]
    row = [sum(c * i**power for power, c in enumerate(scaled)) for i in range(degree + 1)]
    differences = []
    while row:
        differences.append(row[0])
        row = [b - a for a, b in zip(row, row[1:])]
    values = [differences[degree]] * (GRID_SIZE - degree)
    for level in range(degree - 1, -1, -1):
        values = list(itertools.accumulate(values, initial=differences[level]))
    return values


def remez_exp(width: Fraction, degree: int) -> list[int]:
    """Degree-``degree`` minimax fit of exp(-width u) on u in [0, 1], over
    the dense grid.  Returns the power-basis coefficients as integers on
    the 2^-PREC grid.
    """
    last = GRID_SIZE - 1
    grid_targets = exp_grid(width)
    target_scale = last**degree
    scaled_targets = [value * target_scale for value in grid_targets]

    nodes = chebyshev_nodes(degree)
    node_targets = [exp_neg(width * u) for u in nodes]
    coefficients = []
    for _ in range(60):
        matrix = [
            [u**power for power in range(degree + 1)] + [(-1) ** r] for r, u in enumerate(nodes)
        ]
        solution = solve(matrix, [Fraction(value, ONE) for value in node_targets])
        coefficients = [round(c * ONE) for c in solution[:-1]]
        # Errors in units of 2^-PREC / last^degree.
        errors = list(map(operator.sub, grid_values(coefficients), scaled_targets))
        # Index i is an extremum iff (e[i]-e[i-1])*(e[i+1]-e[i]) <= 0.
        steps = list(map(operator.sub, errors[1:], errors))
        turns = [a * b <= 0 for a, b in zip(steps, steps[1:])]
        extrema = [0, *itertools.compress(range(1, last), turns), last]
        alternating = []
        for index in extrema:
            if alternating and (errors[index] > 0) == (errors[alternating[-1]] > 0):
                if abs(errors[index]) > abs(errors[alternating[-1]]):
                    alternating[-1] = index
            else:
                alternating.append(index)
        needed = degree + 2
        if len(alternating) < needed:
            break
        if len(alternating) > needed:
            offset = max(
                range(len(alternating) - needed + 1),
                key=lambda start: min(
                    abs(errors[index]) for index in alternating[start : start + needed]
                ),
            )
            alternating = alternating[offset : offset + needed]
        new_nodes = tuple(Fraction(index, last) for index in alternating)
        node_targets = [grid_targets[index] for index in alternating]
        if new_nodes == nodes:
            break
        nodes = new_nodes
    return coefficients


def quantized_fit(left_raw: int, right_raw: int, degree: int, n: int, p: int):
    """Local-coordinate coefficients on [left, right): fit exp on the
    interval, rescale to t = w - left, and round to the 2^-(n-p) grid.
    Returns (n-bit word patterns, signed integer coefficients).
    """
    scale = 1 << (n - p)
    width = Fraction(right_raw - left_raw, scale)
    amplitude = Fraction(exp_neg(Fraction(left_raw, scale)), ONE)
    patterns = []
    quantized = []
    mask = (1 << n) - 1
    sign = 1 << (n - 1)
    for power, coefficient in enumerate(remez_exp(width, degree)):
        local_coefficient = amplitude * Fraction(coefficient, ONE) / width**power
        q = round(local_coefficient * scale)
        if not -sign <= q < sign:
            raise AssertionError("quantized coefficient overflow")
        patterns.append(q & mask)
        quantized.append(q)
    return tuple(patterns), quantized


def rounded_polynomial_error(left_raw: int, right_raw: int, quantized: list[int], scale: int):
    """Upper bound on |Q(t) - exp(-(a + t))| over t in [0, W], in value
    units.  Q(t) = sum_k quantized[k] t^k / scale is the rounded local
    polynomial, a = left_raw / scale, and W = (right_raw - left_raw) / scale.

    The bound is the exact maximum over the dense grid, plus the linear
    interpolation remainder h^2 / 8 * max|e''| between grid points, plus
    TARGET_SLACK for the fixed-point exp targets.
    """
    degree = len(quantized) - 1
    last = GRID_SIZE - 1
    span = right_raw - left_raw
    width = Fraction(span, scale)
    amplitude_fixed = exp_neg(Fraction(left_raw, scale))
    amplitude_upper = Fraction(amplitude_fixed + 1, ONE)

    # Q(W i / last) * scale^(degree + 1) * last^degree, exact on the grid.
    unit = [q * span**power * scale ** (degree - power) for power, q in enumerate(quantized)]
    values = grid_values(unit)
    targets = [amplitude_fixed * g >> PREC for g in exp_grid(width)]
    value_scale = scale ** (degree + 1) * last**degree
    worst = max(abs(v * ONE - t * value_scale) for v, t in zip(values, targets))
    grid_error = Fraction(worst, ONE * value_scale)

    # |e''| <= |Q''| + exp(-a - t) <= sum k(k-1)|q_k| W^(k-2) / scale + exp(-a).
    curvature = (
        sum(
            power * (power - 1) * abs(q) * width ** (power - 2)
            for power, q in enumerate(quantized)
            if power >= 2
        )
        / scale
        + amplitude_upper
    )
    spacing = width / last
    return grid_error + spacing * spacing / 8 * curvature + TARGET_SLACK


def interval_error_bound(left_raw: int, right_raw: int, degree: int, n: int, p: int):
    """Error bound for one interval, in output units: the rounded
    polynomial's approximation error + multiplier truncation + squaring
    truncation + ideal rounding + final rounding.  Returns (bound,
    coefficient patterns).
    """
    frac = n - p
    scale = 1 << frac
    output_per_work_ulp = Fraction(1 << OUTPUT_FRAC, scale)
    width = Fraction(right_raw - left_raw, scale)
    patterns, quantized = quantized_fit(left_raw, right_raw, degree, n, p)
    approximation = rounded_polynomial_error(left_raw, right_raw, quantized, scale) * (
        1 << OUTPUT_FRAC
    )
    truncation = (patterns[degree] & (scale - 1)).bit_count() * output_per_work_ulp
    for _ in range(degree - 1):
        truncation = truncation * width + frac * output_per_work_ulp
    amplitude_upper = Fraction(exp_neg(Fraction(left_raw, scale)) + 1, ONE)
    squaring = frac * output_per_work_ulp * amplitude_upper
    final_rounding = Fraction(1, 2) if frac > OUTPUT_FRAC else 0
    total = (
        approximation
        + truncation
        + squaring
        + Fraction(1, 2)  # ideal-output rounding
        + final_rounding
    )
    return total, patterns


def build_partition(degree: int, n: int, p: int, *, price=None) -> list[Interval]:
    """Largest-passing-interval partition of w = x^2 in [0, 64): grow each
    interval to the largest right endpoint whose error bound stays inside
    the 1e-7 tier; close with an all-zero tail once the ideal itself is
    inside the tier.  ``price`` replaces ``interval_error_bound``; the
    sweep passes a probe that prices an overflowing fit as infeasible.
    """
    price = price or interval_error_bound
    frac = n - p
    scale = 1 << frac
    domain_right = 64 << frac
    left = 0
    intervals = []
    while True:
        # Once the ideal itself is inside the tier, use an all-zero tail.
        ideal_upper = Fraction((exp_neg(Fraction(left, scale)) + 1) << OUTPUT_FRAC, ONE)
        if ideal_upper + Fraction(1, 2) <= TIER_ULP:
            intervals.append(Interval(left, domain_right, (0,) * (degree + 1), TIER_ULP))
            break

        lo = min(left + (1 << max(0, frac - 8)), domain_right)
        while price(left, lo, degree, n, p)[0] > TIER_ULP:
            lo = left + (lo - left) // 2
            if lo <= left:
                raise RuntimeError(f"partition stalled at raw w={left}")
        hi = domain_right
        if price(left, hi, degree, n, p)[0] <= TIER_ULP:
            lo = hi
        while hi - lo > 1:
            midpoint = (lo + hi) // 2
            if price(left, midpoint, degree, n, p)[0] <= TIER_ULP:
                lo = midpoint
            else:
                hi = midpoint
        bound, coefficients = price(left, lo, degree, n, p)
        intervals.append(Interval(left, lo, coefficients, bound))
        if lo >= domain_right:
            break
        left = lo
    return intervals


def _probe_bound(left_raw, right_raw, degree, n, p):
    """interval_error_bound, with an overflowing fit priced as an
    infeasible probe instead of an error.
    """
    try:
        return interval_error_bound(left_raw, right_raw, degree, n, p)
    except AssertionError:  # quantized coefficient overflow
        return math.inf, None


def t_fma(n: int, p: int) -> int:
    """Return the paper's fused-multiply-add Toffoli formula."""
    return (3 * n * n + 7 * n) // 2 + 3 * n * p - 3 * p * p + 3 * p - 1  # 1.5n^2 + 3.5n + ...


def resources(
    degree: int,
    intervals: int,
    n: int,
    p: int,
    *,
    pebble_registers: int | None = None,
    pebble_moves: int | None = None,
) -> tuple[int, int]:
    """(Toffoli, qubits) of one configuration under the closed form: the
    paper's T_fma/T_extra/LABEL and factor 2 for uncompute, extended
    with the local-coordinate offset, the sign fold, and the
    accumulate.  ``pebble_*=None`` is the direct (unpebbled) ladder.
    """
    label_bits = (intervals - 1).bit_length()  # ceil(log2(intervals))
    fma = t_fma(n, p)
    extra = 2 * intervals * (4 * label_bits - 8)
    label = 4 * intervals * n

    # One square followed by ``degree`` Horner products, or Table I moves.
    stages = degree + 1 if pebble_moves is None else pebble_moves
    core_forward = stages * (fma + extra) + label

    # Load and clear one selected left endpoint around a modular
    # subtraction; a label-pattern flag costs label_bits - 1 ANDs.
    offset = 2 * (intervals - 1) * (label_bits - 1) + (2 * n - 2)
    fold = INPUT_BITS - 1
    accumulate = 2 * OUTPUT_BITS - 2
    toffoli = 2 * (core_forward + offset + fold) + accumulate

    live = degree + 1 if pebble_registers is None else pebble_registers
    qubits = (
        live * n
        + label_bits
        + INPUT_BITS
        + OUTPUT_BITS
        + 1  # comparator output or accumulator carry
        + 1
    )  # copied sign during the even-function fold
    return toffoli, qubits


def run_sweep():
    """Every (degree, n, schedule) row of the envelope: returns (feasible
    rows sorted by closed-form product, infeasible configs, the partition
    of every feasible (n, degree)).
    """
    rows = []
    infeasible = []
    partitions = {}
    for n in SWEEP_WORD_BITS:
        for degree in SWEEP_DEGREES:
            try:
                partition = build_partition(degree, n, SWEEP_INT_BITS, price=_probe_bound)
            except RuntimeError:  # partition stalled: no interval fits
                infeasible.append((n, degree))
                continue
            partitions[n, degree] = partition
            intervals = len(partition)
            t, q = resources(degree, intervals, n, SWEEP_INT_BITS)
            rows.append((t * q, n, degree, intervals, "direct", t, q))
            for live, moves in PEBBLE_MOVES.get(degree + 1, {}).items():
                t, q = resources(
                    degree, intervals, n, SWEEP_INT_BITS, pebble_registers=live, pebble_moves=moves
                )
                rows.append((t * q, n, degree, intervals, f"pebble-{live}", t, q))
    return sorted(rows), infeasible, partitions


def render(row, partition):
    """Return the constants.json text for the argmin row: the
    configuration and the tables that circuit.py reads.
    """
    _product, n, degree, _intervals, schedule, _t, _q = row
    assert max(interval.error_bound_ulp for interval in partition) <= TIER_ULP
    doc = {
        "DEGREE": degree,
        "WORD_BITS": n,
        "INT_BITS": SWEEP_INT_BITS,
        "SCHEDULE": schedule,
        "LEFTS": [interval.left_raw for interval in partition],
        "COEFFICIENTS": [list(interval.coefficients) for interval in partition],
    }
    return json.dumps(doc, indent=1) + "\n"


def main():
    rows, _infeasible, partitions = run_sweep()
    _product, n, degree, intervals, schedule, t, q = rows[0]
    print(
        f"argmin: degree {degree:d}, (n, p) = ({n:d}, {SWEEP_INT_BITS:d}), "
        f"{intervals:d} intervals, {schedule}; "
        f"closed form {t:d} Toffoli x {q:d} qubits"
    )
    new = render(rows[0], partitions[n, degree])
    old = CONSTANTS.read_text() if CONSTANTS.exists() else None
    CONSTANTS.write_text(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
