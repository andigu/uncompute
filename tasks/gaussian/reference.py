#!/usr/bin/env python3
"""gaussian — self-checking classical reference (stdlib only).

Ground truth for |X>|k> -> |X>|k + round(exp(-(X/2^32)^2)*2^32) mod 2^39>, X a
39-bit two's-complement pattern with decode(X) in [-8*2^32, 8*2^32) (x in
[-8, 8)), banded acceptance |centered residue| <= EPS = 429 ULP
(1 ULP = 2^-32), that is, an error below 1e-7.

Run:  python3 reference.py            (anchor + shape + band self-checks, ~2 s)
      python3 reference.py --mpmath   (also regenerate anchors at 50 dps)
"""

import math
import sys

N_X = 39
M = 39
MOD = 1 << 39
FRAC = 32
X_MAG_HALF = 8 << FRAC  # 8 * 2^32 = 2^35; domain is the half-open [-8, 8)
EPS = 429
OUT_MAX = 1 << FRAC  # round(exp(0)*2^32) = 2^32 exactly (x=0 is the max)

# round(exp(-(X/2^32)^2)*2^32), mpmath 50 dps (regenerate with --mpmath). Every anchor's
# true value sits well clear of a round-half boundary (min observed distance
# 0.125 at X=268435456, x=1/16), so careful f64 reproduces all of them.
ANCHORS = {
    0: 4294967296,
    1: 4294967296,
    7: 4294967296,
    65536: 4294967295,
    1048576: 4294967040,
    134217728: 4290775039,  # x = 1/32
    268435456: 4278222805,  # x = 1/16
    1073741824: 4034748382,  # x = 1/4
    4294967296: 1580030169,  # x = 1
    8589934592: 78665070,  # x = 2
    17179869184: 483,  # x = 4
    25769803776: 0,  # x = 6
    30064771072: 0,  # x = 7
    (8 << FRAC) - 1: 0,  # x = 8 - 2^-32 (domain max, endpoint excluded from [8,.))
    -1: 4294967296,
    -65536: 4294967295,
    -4294967296: 1580030169,  # x = -1
    -17179869184: 483,  # x = -4
    -(8 << FRAC): 0,  # x = -8 (domain min, endpoint included)
}


def decode_x(pattern):
    """39-bit two's-complement pattern -> signed value."""
    assert 0 <= pattern < MOD
    return pattern - MOD if (pattern >> 38) & 1 else pattern


def in_domain(xs):
    """x = xs/2^32 in [-8, 8) -- half-open."""
    return -X_MAG_HALF <= xs < X_MAG_HALF


def round_half_away(v):
    """Round v >= 0 to the nearest integer, with ties away from zero.

    This is the grader's rule. Python round() sends ties to the even
    integer instead. floor(v) and v - floor(v) are exact in f64.
    """
    f = math.floor(v)
    return f + 1 if v - f >= 0.5 else f


def ideal_gauss(xs):
    """Unsigned round(exp(-(xs/2^32)^2)*2^32) in f64 (the pinned definition).

    The platform C library exp is faithful to well under 2 f64 ULP. At this
    magnitude, that error is many orders of magnitude inside the +-429 band
    (README §3.3 derives the band). Ties round away from zero.
    """
    assert in_domain(xs)
    x = xs / 4294967296.0
    return round_half_away(math.exp(-(x * x)) * 4294967296.0)


def ideal_k_out(xs, k_in):
    return (k_in + ideal_gauss(xs)) % MOD


def centered(v):
    r = v % MOD
    return r - MOD if r > MOD // 2 else r


def check_shot(xpat, k_in, read_x, read_k):
    """Apply the grader's acceptance: X exact, k within the centered band."""
    if read_x != xpat:
        return False
    want = ideal_k_out(decode_x(xpat), k_in)
    return abs(centered(read_k - want)) <= EPS


def main():
    # 1. Anchors: careful f64 reproduces every mpmath-pinned value.
    for xs, want in ANCHORS.items():
        got = ideal_gauss(xs)
        assert got == want, f"anchor mismatch at X={xs}: got {got}, want {want}"

    # 2. Optional: regenerate anchors at 50 decimal digits.
    if "--mpmath" in sys.argv:
        import mpmath as mp

        mp.mp.dps = 50
        for xs, want in ANCHORS.items():
            v = int(mp.nint(mp.e ** (-((mp.mpf(xs) / 2**32) ** 2)) * 2**32))
            assert v == want, f"mpmath regeneration mismatch at X={xs}: {v} != {want}"
        print("mpmath 50-dps anchor regeneration: OK")

    # 3. Domain shape, strided (2^36 in-domain values too large to enumerate):
    #    even function, monotonically decreasing in |x|, bounded by OUT_MAX,
    #    nonnegative (exp never reaches 0 in exact math, but the 2^-32 output
    #    grid rounds it to 0 for |x| > ~4.78).
    prev = None
    for xs in range(0, X_MAG_HALF, 1 << 16):
        v = ideal_gauss(xs)
        assert ideal_gauss(-xs) == v, f"not even at X={xs}"
        assert prev is None or v <= prev, f"not monotone decreasing at X={xs}"
        assert 0 <= v <= OUT_MAX
        prev = v
    assert ideal_gauss(0) == OUT_MAX
    assert ideal_gauss(X_MAG_HALF - 1) == 0
    assert ideal_gauss(-X_MAG_HALF) == 0

    # 4. Band semantics incl. both mod-2^39 seams.
    for xs, k_in in ((0, 5), (-X_MAG_HALF, MOD - 5), (134217728, 12345)):
        xpat = xs % MOD
        ideal = ideal_k_out(xs, k_in)
        for d in (-EPS, -1, 0, 1, EPS):
            assert check_shot(xpat, k_in, xpat, (ideal + d) % MOD), (xs, d)
        for d in (-(EPS + 1), EPS + 1):
            assert not check_shot(xpat, k_in, xpat, (ideal + d) % MOD), (xs, d)
        assert not check_shot(xpat, k_in, xpat ^ 1, ideal)  # X corruption

    # 5. Domain edges / decode.
    assert decode_x(0) == 0
    assert decode_x(X_MAG_HALF) == X_MAG_HALF  # +8, just OUTSIDE domain
    assert decode_x(MOD - X_MAG_HALF) == -X_MAG_HALF  # -8, in-domain endpoint
    assert in_domain(-X_MAG_HALF) and not in_domain(X_MAG_HALF)
    assert in_domain(X_MAG_HALF - 1) and not in_domain(-X_MAG_HALF - 1)

    print(
        f"reference.py: all self-checks OK (EPS={EPS}, OUT_MAX={OUT_MAX}, {len(ANCHORS)} anchors)"
    )


if __name__ == "__main__":
    main()
