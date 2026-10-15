"""Grader for `gaussian`.

Spec: |X>|k> -> |X>|k + round(exp(-(X/2^32)^2)*2^32)
mod 2^39>. X is a 39-qubit two's-complement fixed-point input (7 integer bits
incl. sign, 32 fractional bits; x = X/2^32), restricted to x in [-8, 8)
(half-open — the upper endpoint is EXCLUDED); k is a 39-qubit additive
accumulator, randomized per shot.

Banded check, no failure budget: register 0 must round-trip exactly;
register 1 passes iff the centered residue of read_k - ideal_k_out mod 2^39
has |delta| <= EPS = 429 ULP (1 ULP = 2^-32), that is, floor(1e-7 * 2^32).

Grader ideal: round(exp(-(xs/2^32)^2)*2^32) with f64 exp (careful f64 is the
pinned definition). Rounding is round-half-away-from-zero (C ``round``
semantics), reproduced bit-exactly.

Run `python3 grader.py` for the self-test (the pinned anchors).
"""

import json
import math
from pathlib import Path

from uncompute.evalcore import QUBIT, RegSpec, Task, TestCase

# Input width: 39-qubit two's-complement pattern; x = decode(X)/2^32.
N_X = 39
# Output/accumulator width; the accumulator is mod 2^39.
M = 39
# Domain: decode(X) in [-8*2^32, 8*2^32), i.e. x in [-8, 8) (half-open).
X_MAG_HALF = 8 << 32
# Correctness band (centered residue, inclusive), in ULP = 2^-32:
# 429 * 2^-32 = 9.9884e-8, the largest integer band inside the 1e-7 bound.
EPS = 429
# round(exp(0)*2^32) = 2^32 exactly (the maximum of the function, at x=0).
OUT_MAX = 1 << 32

MASK39 = (1 << 39) - 1
MOD = 1 << M

# contract.json is the shared statement of these derived values for circuits;
# the inline derivation above stays authoritative — assert-only consistency.
_contract = json.loads((Path(__file__).resolve().parent / "contract.json").read_text())
assert _contract["n_x"] == N_X and _contract["m"] == M
assert _contract["frac_bits"] == 32 and _contract["out_max"] == OUT_MAX
assert _contract["x_mag_half"] == X_MAG_HALF
assert _contract["eps_ulp"] == EPS


def _round_half_away(v):
    """Round half away from zero (C ``round`` semantics); exact for v >= 0 (floor and
    v - floor(v) are exact in f64). Python's round() is half-to-even.
    """
    f = math.floor(v)
    return f + 1 if v - f >= 0.5 else f


def decode_x(pattern):
    """Decode a 39-bit two's-complement pattern to a signed value."""
    assert 0 <= pattern <= MASK39
    return pattern - MOD if (pattern >> 38) & 1 else pattern


def in_domain(xs):
    """x = xs/2^32 in [-8, 8) (half-open at +8)."""
    return -X_MAG_HALF <= xs < X_MAG_HALF


def ideal_gauss(xs):
    """round(exp(-(xs/2^32)^2)*2^32), unsigned (always in [0, 2^32])."""
    assert in_domain(xs)
    x = xs / 4294967296.0
    return _round_half_away(math.exp(-(x * x)) * 4294967296.0)


def ideal_k_out(xs, k_in):
    """(k + round(exp(-(xs/2^32)^2)*2^32)) mod 2^39."""
    return (k_in + ideal_gauss(xs)) & MASK39


class Gaussian(Task):
    name = "gaussian"
    seed_domain = b"gaussian"
    # Domain x in [-8,8) is EXACTLY 1/8 of the 39-bit pattern space:
    # 72192 draws yields exactly 9024 expected live shots.
    num_tests = 72192

    def schema(self):
        return [
            RegSpec(width=N_X, kind=QUBIT),  # 0: X (preserved)
            RegSpec(width=M, kind=QUBIT),  # 1: k accumulator
        ]

    def gen_test(self, xof):
        # Constant 10 bytes/shot: 5 for X, 5 for k — drawn BEFORE any skip
        # decision, both masked to 39 bits (2^40 mod 2^39 == 0: exactly
        # uniform). Out-of-domain X (skip probability 7/8) is
        # consumed-then-skipped.
        xb = xof.read(5)
        kb = xof.read(5)
        xpat = int.from_bytes(xb, "little") & MASK39
        k = int.from_bytes(kb, "little") & MASK39
        xs = decode_x(xpat)
        if not in_domain(xs):
            return None
        return TestCase(
            inputs=[xpat, k],
            expected=[xpat, ideal_k_out(xs, k)],
            expected_phase=False,
        )

    def check_shot(self, tc, read):
        # Banded acceptance: X exact; k within the centered mod-2^39 band.
        if read[0] != tc.inputs[0]:
            return False
        d = (read[1] - tc.expected[1]) % MOD
        if d > MOD // 2:
            d -= MOD
        return abs(d) <= EPS


TASK = Gaussian()


if __name__ == "__main__":
    # Self-test: the pinned anchors.

    # 1. anchors (mpmath-50dps pins).
    ANCHORS = [
        (0, 4294967296),
        (1, 4294967296),
        (65536, 4294967295),
        (134217728, 4290775039),
        (268435456, 4278222805),
        (1073741824, 4034748382),
        (4294967296, 1580030169),
        (8589934592, 78665070),
        (17179869184, 483),
        (25769803776, 0),
        (-1, 4294967296),
        (-65536, 4294967295),
        (-4294967296, 1580030169),
        (-17179869184, 483),
        (-(8 << 32), 0),
    ]
    for xs, want in ANCHORS:
        got = ideal_gauss(xs)
        assert got == want, f"anchor mismatch at X={xs}: got {got}, want {want}"

    # 2. decode_and_domain.
    assert decode_x(0) == 0
    assert decode_x(MASK39 - X_MAG_HALF + 1) == -X_MAG_HALF  # -8 pattern
    assert in_domain(-X_MAG_HALF) and not in_domain(X_MAG_HALF)
    assert in_domain(X_MAG_HALF - 1) and not in_domain(-X_MAG_HALF - 1)

    # 3. ideal_domain_sampled (pinned stride).
    prev = None
    xs = 0
    while xs < X_MAG_HALF:
        v = ideal_gauss(xs)
        assert ideal_gauss(-xs) == v, f"not even at X={xs}"
        assert prev is None or v <= prev, f"not monotone at X={xs}"
        assert 0 <= v <= OUT_MAX
        prev = v
        xs += 1 << 13
    assert ideal_gauss(0) == OUT_MAX
    assert ideal_gauss(X_MAG_HALF - 1) == 0

    # 4. check_shot_band_semantics.
    task = TASK
    for xs, k_in in [(0, 5), (-X_MAG_HALF, MOD - 5), (134217728, 12345)]:
        xpat = xs % MOD
        ideal = ideal_k_out(xs, k_in)
        tc = TestCase(inputs=[xpat, k_in], expected=[xpat, ideal])
        for d in (-EPS, -1, 0, 1, EPS):
            assert task.check_shot(tc, [xpat, (ideal + d) % MOD]), (xs, d)
        for d in (-(EPS + 1), EPS + 1):
            assert not task.check_shot(tc, [xpat, (ideal + d) % MOD]), (xs, d)
        assert not task.check_shot(tc, [xpat ^ 1, ideal])  # X corruption

    # 5. gen_test_contract (SHAKE stream, skip rate ~7/8).
    import hashlib

    class _Xof:
        def __init__(self, tag):
            self._buf = hashlib.shake_256(tag).digest(10 * 80_000)
            self._pos = 0

        def read(self, n):
            out = self._buf[self._pos : self._pos + n]
            self._pos += n
            return out

    xof = _Xof(task.seed_domain + b"/gen_test_contract_test")
    live = 0
    for _ in range(80_000):
        tc = task.gen_test(xof)
        if tc is not None:
            live += 1
            xpat, k = tc.inputs
            assert in_domain(decode_x(xpat)) and k <= MASK39
            assert tc.expected == [xpat, ideal_k_out(decode_x(xpat), k)]
    expected = 80_000 / 8
    sigma = (80_000 * (1 / 8.0) * (7 / 8.0)) ** 0.5
    assert expected - 3 * sigma <= live <= expected + 3 * sigma, live

    print("gaussian grader.py: all self-checks OK")
