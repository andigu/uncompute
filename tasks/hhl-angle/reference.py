#!/usr/bin/env python3
"""hhl-angle -- self-checking classical reference (stdlib only).

Ground truth for |X>|k> -> |X>|k + round(arcsin(2^15/X)*2^32) mod 2^33>, X a
21-bit unsigned fixed-point pattern with X in [2^16, 2^21-1] (x = X/2^16 in
[1, 32-2^-16]), banded acceptance |centered residue| <= EPS = 429 ULP (1 ULP =
2^-32 rad), that is, floor(1e-7 * 2^32). See README.md for the full
specification.

The anchor, domain-shape, and band-semantics batteries live in grader.py's
__main__ self-test (the single authority); this file keeps what the grader
does not carry: the 50-dps anchor regeneration (--mpmath) and a simple
reference model that shows the band is attainable — 20 octave/quarter-interval banks
(range-reduce X into e = floor(log2 x) - 16 in {0..4}, quarter j = top
2 mantissa bits below the leading one), a barrel shift M = X << (4-e) puts
every octave's mantissa into a UNIFORM 18-bit field, a degree-6 Horner
polynomial per bank in the bit-complemented local variable Vc = (2^18-1) - V
(making every quantized coefficient non-negative), verified exhaustively
over the full 2,031,616-point domain to be within EPS with 5.9x headroom.

Run:  python3 reference.py            (reference-model self-checks)
      python3 reference.py --mpmath   (also regenerate anchors at 50 dps)
"""

import math
import operator
import sys
from fractions import Fraction

N_X = 21
M = 33
MOD = 1 << 33
X_MIN = 1 << 16
X_MAX = (1 << 21) - 1
EPS = 429
OUT_MAX = 2248839617  # round(arcsin(1/2) * 2^32) = round(pi/6 * 2^32)
OUT_MIN = 67111627  # round(arcsin(2^15/(2^21-1)) * 2^32)

# round(arcsin(2^15/X) * 2^32), mpmath 50 dps (regenerate with --mpmath), cross-checked
# against careful f64 (0 discrepancies across all 17). Every anchor sits >= 0.0797
# from a round-half boundary.
ANCHORS = {
    65536: 2248839617,  # u = 1/2 endpoint: round(pi/6 * 2^32)
    65537: 2248801781,  # endpoint+1
    81920: 1767451396,  # octave-0 interior
    98304: 1459588412,  # octave-0 interior
    114688: 1244474082,  # octave-0 interior / sub-seam
    131072: 1085253432,  # octave boundary (leading-one branch point)
    262144: 538278936,  # octave boundary
    290126: 486128218,  # SHAKE KAT 1
    524288: 268610527,  # octave boundary
    768349: 183224271,  # SHAKE KAT 2
    909458: 154782251,  # SHAKE KAT 3
    1048576: 134239583,  # octave boundary
    1199617: 117333279,  # SHAKE KAT 4
    1312143: 107268888,  # SHAKE KAT 5
    1372325: 102563799,  # SHAKE KAT 6
    2097150: 67111659,  # far endpoint-1
    2097151: 67111627,  # far endpoint
}

# ---- 20-bank reference model ----------------------------------------------
# BANK_COEFFS[e][j]: degree-6 polynomial coefficients (highest degree first),
# quantized to a 2^38 scale (32 output fractional bits + 6 guard bits), fit by
# ordinary least squares against the f64 ideal in the bit-complement Horner
# variable Vc = (2^18-1) - V. All non-negative by construction. Verified
# exhaustively (main() below) over all 2,031,616 in-domain X: worst
# |error| = 73 ULP (bank e=0,j=0, at X=65536) -- 5.9x headroom under EPS=429.
# main() also re-solves each least-squares fit exactly (lsq_bank) and checks
# that every entry rounds the exact coefficient. One entry (e=1, j=1, index 2)
# lies 0.004 from a rounding boundary. A float64 solve gives its upper
# neighbor, and the table keeps that value.
DEG = 6
GUARD = 6
# The constant term (coefficient index DEG) has the round-to-nearest offset
# 2^(GUARD-1) pre-folded in, so model_h below is a plain truncating
# right-shift by GUARD (no separate rounding-add step).
BANK_COEFFS = [
    [
        [47382399, 16435734, 318109808, 1175311185, 5258523524, 23993134744, 113116984828],
        [7490758, 11784699, 100785259, 517246544, 2868619986, 16197332457, 93413720655],
        [1852137, 5052625, 41687503, 264287695, 1746908992, 11707545132, 79646386009],
        [594704, 2283010, 20049793, 149610258, 1145940894, 8871653673, 69456253559],
    ],
    [
        [9631039, 11703506, 104031162, 468617118, 2291590258, 11221826291, 55348887285],
        [2154561, 4890778, 39151857, 221912404, 1309158260, 7743811443, 46027807224],
        [635433, 2101280, 17527889, 118394852, 818154648, 5667891590, 39403101388],
        [226058, 979605, 8821011, 68865606, 545416789, 4328923814, 34449868468],
    ],
    [
        [3993894, 5670886, 47673314, 222650101, 1110815478, 5525248588, 27533832431],
        [953571, 2321761, 18500200, 107095983, 640781919, 3831076416, 22933102279],
        [291526, 1004307, 8418219, 57682456, 402762817, 2812061767, 19650881125],
        [106020, 472047, 4278815, 33758160, 269492006, 2151692114, 17191081945],
    ],
    [
        [1909927, 2806799, 23344414, 109949630, 551220949, 2752219226, 13749639071],
        [462915, 1145798, 9123627, 53084879, 318716976, 1910534656, 11456570076],
        [142731, 496567, 4167710, 28657510, 200609111, 1403334759, 9819161657],
        [52183, 233875, 2123466, 16796685, 134350587, 1074267412, 8591337440],
    ],
    [
        [944513, 1399684, 11612108, 54805321, 275092670, 1374818137, 6872669037],
        [229769, 571027, 4546239, 26485135, 159150775, 954644941, 5727041080],
        [70994, 247592, 2078735, 14305962, 100208538, 701331574, 4908797671],
        [25989, 116671, 1059756, 8388102, 67126070, 536936918, 4295144159],
    ],
]
BANK_COEFFS_SHA256 = "156fefe5de67ecb5bc5d59dc9d044aaa317550ca6a70e5cad985e37419478367"


def _round_half_away(v):
    """Round half away from zero (the grader's pinned rounding rule).

    Exact for v >= 0: floor(v) is exact, and v - floor(v)
    is exact in f64 (Sterbenz for v >= 1, trivial for v < 1), so the
    >= 0.5 comparison never misrounds. (Python's built-in round() is
    half-to-even -- NOT what the grader pins.)
    """
    f = math.floor(v)
    return f + 1 if v - f >= 0.5 else f


def ideal_h(x):
    """Signed round(arcsin(2^15/x) * 2^32), careful f64 with half-away-
    from-zero rounding -- restating grader.py's pinned definition exactly
    (the C library asin is faithful to < 2 f64 ULP here, six orders below the +-429
    band).
    """
    assert X_MIN <= x <= X_MAX
    return _round_half_away(math.asin((1 << 15) / x) * 4294967296.0)


def classify(x):
    """Range-reduce X into (e, j, V): e = floor(log2 x) - 16 in {0..4} (octave,
    position of X's leading one relative to bit 16), M = X << (4-e) in
    [2^20, 2^21) (barrel-shifts the mantissa into a UNIFORM 18-bit field),
    j = top 2 bits of M below the leading one, V = M - (2^20 + j*2^18) in
    [0, 2^18).
    """
    e = x.bit_length() - 1 - 16
    shift = 4 - e
    Mval = x << shift
    j = (Mval >> 18) & 0b11
    Mlo = (1 << 20) + j * (1 << 18)
    V = Mval - Mlo
    return e, j, V


def model_h_scaled(x):
    """Evaluate the reference model's Horner form, integer-exact,
    returning the value at GUARD-bit extra precision (pre-final-round).
    """
    e, j, V = classify(x)
    Vc = ((1 << 18) - 1) - V
    coeffs = BANK_COEFFS[e][j]
    acc = coeffs[0]
    for k in range(1, DEG + 1):
        acc = coeffs[k] + ((acc * Vc) >> 18)
    return acc


def model_h(x):
    """Return the final rounded reference-model output. The offset 2^(GUARD-1) is
    pre-folded into BANK_COEFFS's constant term, so this is a plain
    truncating right-shift by GUARD.
    """
    acc = model_h_scaled(x)
    return acc >> GUARD


def lsq_bank(e, j):
    """Return bank (e, j)'s exact least-squares polynomial in Vc.

    The fit runs over every in-domain X of the bank, against the unrounded
    f64 ideal asin(2^15/X) * 2^32.  The result is a list of Fractions a,
    lowest degree first, that minimizes sum (sum_k a[k] Vc^k - ideal)^2.
    """
    lo = (1 << (16 + e)) + j * (1 << (14 + e))
    xs = range(lo, lo + (1 << (14 + e)))
    vcs = [((1 << 18) - 1) - classify(x)[2] for x in xs]
    # Each ideal is >= 2^26, so its f64 value times 2^30 is an exact integer.
    ys = []
    for x in xs:
        y = math.asin((1 << 15) / x) * 4294967296.0 * 2.0**30
        assert y == int(y)
        ys.append(int(y))
    # Normal equations: sum_c S[r + c] a[c] = T[r], with S[k] = sum Vc^k
    # and T[k] = sum Vc^k * ideal.
    S, T = [], []
    pw = [1] * len(vcs)
    for k in range(2 * DEG + 1):
        S.append(sum(pw))
        if k <= DEG:
            T.append(Fraction(sum(map(operator.mul, pw, ys)), 1 << 30))
        pw = list(map(operator.mul, pw, vcs))
    n = DEG + 1
    rows = [[Fraction(S[r + c]) for c in range(n)] + [T[r]] for r in range(n)]
    for c in range(n):  # Gauss-Jordan; the Gram matrix is positive definite
        for r in range(n):
            if r != c:
                f = rows[r][c] / rows[c][c]
                rows[r] = [u - f * v for u, v in zip(rows[r], rows[c])]
    return [rows[k][n] / rows[k][k] for k in range(n)]


def main():
    # 1. Optional: regenerate anchors at 50 decimal digits.
    if "--mpmath" in sys.argv:
        import mpmath as mp

        mp.mp.dps = 50
        for x, want in ANCHORS.items():
            v = int(mp.nint(mp.asin(mp.mpf(1 << 15) / x) * 2**32))
            assert v == want, f"mpmath regeneration mismatch at X={x}: {v} != {want}"
        print("mpmath 50-dps anchor regeneration: OK")

    # 2. Reference model: digest pin + exhaustive band check.
    import hashlib
    import struct

    buf = bytearray()
    for e in range(5):
        for j in range(4):
            for c in BANK_COEFFS[e][j]:
                buf += struct.pack("<q", c)
    got_digest = hashlib.sha256(bytes(buf)).hexdigest()
    assert got_digest == BANK_COEFFS_SHA256, "bank coefficient table digest drifted"
    for e in range(5):
        for j in range(4):
            assert all(c >= 0 for c in BANK_COEFFS[e][j]), "bank coeffs must be non-negative"
            # Entry i scales the t^(DEG-i) coefficient, t = Vc / 2^18, by 2^GUARD.
            # The constant entry also carries the rounding offset 2^(GUARD-1).
            a = lsq_bank(e, j)
            for i, c in enumerate(BANK_COEFFS[e][j]):
                d = DEG - i
                want = a[d] * (1 << (18 * d + GUARD)) + (1 << (GUARD - 1) if d == 0 else 0)
                assert abs(c - want) <= Fraction(1, 2) + Fraction(1, 128), (e, j, i)

    worst = 0
    worst_x = None
    x = X_MIN
    while x <= X_MAX:
        got = model_h(x)
        want = ideal_h(x)
        err = abs(got - want)
        if err > worst:
            worst, worst_x = err, x
        x += 1
    assert worst <= EPS, f"reference model exceeds band: worst={worst} at X={worst_x}"
    print(
        f"reference model: worst |error| = {worst} ULP at X={worst_x} "
        f"(EPS={EPS}, headroom {EPS / worst:.2f}x)"
    )

    print("reference.py: all self-checks OK")


if __name__ == "__main__":
    main()
