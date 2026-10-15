"""Grader for `hhl-angle`.

Spec: |X>|k> -> |X>|k + round(arcsin(2^15/X)*2^32) mod 2^33>. X is a 21-bit
unsigned fixed-point input (x = X/2^16), in-domain X in [2^16, 2^21 - 1]
(both endpoints in-domain); k is a 33-bit additive accumulator, randomized
per shot. Banded check, no failure budget: register 0 must round-trip
exactly; register 1 passes iff the centered residue of read_k - ideal_k_out
mod 2^33 has |delta| <= EPS = 429 ULP (1 ULP = 2^-32 rad), that is,
floor(1e-7 * 2^32): an error below 1e-7 rad.

Grader ideal: round(asin(2^15/X)*2^32) with f64 asin (careful f64 is the
pinned definition: faithful to < 2 f64 ULP here, six orders below the band).
Rounding is round-half-away-from-zero (C ``round`` semantics), reproduced bit-exactly.

Run `python3 grader.py` for the self-test (the pinned unit-test anchors).
"""

import math

from uncompute.evalcore import QUBIT, RegSpec, Task, TestCase

# Input register width (bits): 21-bit unsigned fixed-point X.
N_X = 21
# Output/accumulator register width (bits): 33-bit additive accumulator k.
M = 33
# Smallest in-domain X (inclusive): x = X/2^16 = 1.
X_MIN = 1 << 16
# Largest in-domain X (inclusive, = 2^21 - 1, the full 21-bit mask).
X_MAX = (1 << 21) - 1
# Correctness band (centered residue, inclusive), in ULP = 2^-32 rad.
EPS = 429
# round(arcsin(1/2)*2^32) = round(pi/6*2^32): the map's maximum output.
OUT_MAX = 2248839617
# The map's minimum output, at X = X_MAX.
OUT_MIN = 67111627

MASK21 = (1 << 21) - 1
MASK33 = (1 << 33) - 1
MOD = 1 << M

# Contract cross-check (assert-only): contract.json is the shared copy of the
# task-defining derived constants that circuits load; the inline
# derivations above stay authoritative and must match it exactly.
import json as _json
from pathlib import Path as _Path

_CONTRACT = _json.loads((_Path(__file__).resolve().parent / "contract.json").read_text())
assert (_CONTRACT["x_bits"], _CONTRACT["k_bits"]) == (N_X, M), _CONTRACT
assert (_CONTRACT["x_min"], _CONTRACT["x_max"]) == (X_MIN, X_MAX), _CONTRACT
assert _CONTRACT["eps_ulp"] == EPS, _CONTRACT
assert (_CONTRACT["out_max"], _CONTRACT["out_min"]) == (OUT_MAX, OUT_MIN), _CONTRACT


def _round_half_away(v):
    """Round half away from zero (C ``round`` semantics).

    Exact for v >= 0:
    floor(v) is exact, and v - floor(v) is exact in f64 (Sterbenz for v >= 1,
    trivial for v < 1), so the >= 0.5 comparison never misrounds. (Python's
    built-in round() is half-to-even — NOT this rule.)
    """
    f = math.floor(v)
    return f + 1 if v - f >= 0.5 else f


def ideal_h(x):
    """round(arcsin(2^15/X)*2^32), always positive for in-domain X."""
    assert X_MIN <= x <= X_MAX
    return _round_half_away(math.asin((1 << 15) / x) * 4294967296.0)


def ideal_k_out(x, k_in):
    """(k + round(arcsin(2^15/X)*2^32)) mod 2^33."""
    return (k_in + ideal_h(x)) & MASK33


class HhlAngle(Task):
    name = "hhl-angle"
    seed_domain = b"hhl-angle"
    # 3-byte draw masked to 21 bits: acceptance 31/32, E[live] = 8742.
    num_tests = 9024

    def schema(self):
        return [
            RegSpec(width=N_X, kind=QUBIT),  # 0: X (preserved)
            RegSpec(width=M, kind=QUBIT),  # 1: k accumulator
        ]

    def gen_test(self, xof):
        # Constant 8 bytes/shot: 3 for X (21 bits masked — exactly uniform),
        # 5 for k (33 bits masked — exactly uniform). Both drawn BEFORE the
        # skip decision. Reject X < 2^16 (below X_MIN); acceptance 31/32.
        xb = xof.read(3)
        kb = xof.read(5)
        xpat = int.from_bytes(xb, "little") & MASK21
        k = int.from_bytes(kb, "little") & MASK33
        if xpat < X_MIN:
            return None
        # xpat <= MASK21 = X_MAX always holds (masked to 21 bits).
        return TestCase(
            inputs=[xpat, k],
            expected=[xpat, ideal_k_out(xpat, k)],
            expected_phase=False,
        )

    def check_shot(self, tc, read):
        # Banded acceptance: X exact; k within the centered mod-2^33 band.
        if read[0] != tc.inputs[0]:
            return False
        d = (read[1] - tc.expected[1]) % MOD
        if d > MOD // 2:
            d -= MOD
        return abs(d) <= EPS


TASK = HhlAngle()


if __name__ == "__main__":
    # Self-test: the pinned unit-test anchors.

    # 1. anchor_vectors (mpmath-50dps pins, cross-checked against f64).
    pins = [
        (65536, 2248839617),
        (65537, 2248801781),
        (81920, 1767451396),
        (98304, 1459588412),
        (114688, 1244474082),
        (131072, 1085253432),
        (262144, 538278936),
        (290126, 486128218),
        (524288, 268610527),
        (768349, 183224271),
        (909458, 154782251),
        (1048576, 134239583),
        (1199617, 117333279),
        (1312143, 107268888),
        (1372325, 102563799),
        (2097150, 67111659),
        (2097151, 67111627),
    ]
    for x, want in pins:
        got = ideal_h(x)
        assert got == want, f"anchor mismatch at X={x}: got {got}, want {want}"
    # Accumulator map anchors incl. the mod-2^33 seam.
    assert ideal_k_out(X_MIN, 0) == OUT_MAX
    assert ideal_k_out(X_MIN, MASK33) == (OUT_MAX - 1) & MASK33
    assert ideal_k_out(X_MAX, 0) == OUT_MIN

    # 2. domain_bounds.
    assert X_MIN == 1 << 16 and X_MAX == (1 << 21) - 1 == MASK21

    # 3. ideal_monotone_decreasing (strided, pinned stride).
    prev = None
    x = X_MIN
    while x <= X_MAX:
        v = ideal_h(x)
        assert v > 0
        assert prev is None or v <= prev, f"not monotone at X={x}"
        assert OUT_MIN <= v <= OUT_MAX
        prev = v
        x += 1 << 10
    assert ideal_h(X_MIN) == OUT_MAX
    assert ideal_h(X_MAX) == OUT_MIN

    # 4. check_shot_band_semantics.
    task = TASK
    for x, k_in in [(X_MIN, 5), (X_MAX, MASK33 - 5), (1610612, 0x123456789 & MASK33)]:
        tc = TestCase(inputs=[x, k_in], expected=[x, ideal_k_out(x, k_in)])
        ideal = ideal_k_out(x, k_in)
        for d in (-EPS, -1, 0, 1, EPS):
            assert task.check_shot(tc, [x, (ideal + d) % MOD]), (x, d)
        for d in (-(EPS + 1), EPS + 1):
            assert not task.check_shot(tc, [x, (ideal + d) % MOD]), (x, d)
        assert not task.check_shot(tc, [x ^ 1, ideal % MOD])

    # 5. gen_test_contract against a real SHAKE256 stream (domain and skip-rate check).
    import hashlib

    class _Xof:
        def __init__(self, tag):
            self._buf = hashlib.shake_256(tag).digest(8 * 80_000)
            self._pos = 0

        def read(self, n):
            out = self._buf[self._pos : self._pos + n]
            self._pos += n
            return out

    xof = _Xof(b"hhl-angle gen_test contract test")
    kept = skipped = 0
    for _ in range(80_000):
        tc = task.gen_test(xof)
        if tc is None:
            skipped += 1
            continue
        kept += 1
        xpat, k = tc.inputs
        assert X_MIN <= xpat <= X_MAX and k <= MASK33
        assert tc.expected == [xpat, ideal_k_out(xpat, k)]
    assert kept + skipped == 80_000
    expected_skips = 80_000 / 32.0
    sigma = (80_000 * (1 / 32.0) * (31 / 32.0)) ** 0.5
    assert expected_skips - 4 * sigma <= skipped <= expected_skips + 4 * sigma, skipped

    print("hhl-angle grader.py: all self-checks OK")
