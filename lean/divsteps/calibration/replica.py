#!/usr/bin/env python3
"""replica.py — plain-Python replica of the hddivsteps convex-hull certificate
(hddivsteps_base.v / hddivsteps590.v of Wuille (2021), sipa/safegcd-bounds, MIT).

Faithfulness notes (vs the Coq source):

* Dyadics are kept in Dred-canonical form (odd mantissa, or (0,0) for zero).
  This is a faithful quotient: every decision the Coq computation makes —
  DDOrder comparisons (AVL set membership/dedup/min/max/ascending fold),
  `orientation` (exact Dsub/Dmult then Dcompare), `narrow` (Dmult then Dltb) —
  is a function of dyadic VALUES only (Dcompare aligns exponents before
  comparing mantissas).  MSetAVL keeps the first-inserted representative, but
  no representative-dependent quantity is ever observable.  With canonical
  forms, value equality == tuple equality, so Python sets dedup correctly;
  ORDER comparisons still align exponents (never compare mantissas raw).
* convexHull folds over set elements in ASCENDING pair-value order (DDSet.fold
  visits ascending), addUpperPoint chain ++ addLowerPoint chain, then set-dedup.
* odd branch: delta' = INC - delta with INC = 2 (half-delta variant), applied
  when delta > 0; else delta' = INC + delta.  Points are (g, f) pairs.
* narrow takes min/max under the PAIR order and reads only the first
  coordinate; strict comparisons -1 < M*l and M*h < 1; drop happens BEFORE the
  hull is computed (hull only for surviving keys).
"""

import sys
from functools import cmp_to_key

# ---------- dyadics: canonical (mantissa, exponent), mantissa odd or (0,0) ----


def dred(m, e):
    if m == 0:
        return (0, 0)
    while m & 1 == 0:
        m >>= 1
        e += 1
    return (m, e)


def dcmp(a, b):
    """Value comparison (Dcompare): align exponents, compare mantissas."""
    (ma, ea), (mb, eb) = a, b
    if ea >= eb:
        ma <<= ea - eb
    else:
        mb <<= eb - ea
    return (ma > mb) - (ma < mb)


def dadd(a, b):
    (ma, ea), (mb, eb) = a, b
    if ea >= eb:
        return dred((ma << (ea - eb)) + mb, eb)
    return dred(ma + (mb << (eb - ea)), ea)


def dsub(a, b):
    (ma, ea), (mb, eb) = a, b
    if ea >= eb:
        return dred((ma << (ea - eb)) - mb, eb)
    return dred(ma - (mb << (eb - ea)), ea)


def dmult(a, b):
    return dred(a[0] * b[0], a[1] + b[1])


def dhalf(a):
    return (a[0], a[1] - 1) if a[0] else (0, 0)


# ---------- points: (g, f) pairs, lexicographic by value ---------------------


def ddcmp(p, q):
    c = dcmp(p[0], q[0])
    return c if c else dcmp(p[1], q[1])


DDKEY = cmp_to_key(ddcmp)


def orientation(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return dcmp(dmult(dsub(x1, x3), dsub(y2, y3)), dmult(dsub(y1, y3), dsub(x2, x3)))


def add_chain_point(p, l, keep_sign):
    """addLowerPoint (keep_sign=-1: keep when orientation=Lt) /
    addUpperPoint (keep_sign=+1: keep when orientation=Gt).
    """
    while True:
        if not l:
            return [p]
        if len(l) == 1:
            q = l[0]
            return [p] if ddcmp(p, q) == 0 else [p, q]
        q, r = l[0], l[1]
        if orientation(p, q, r) == keep_sign:
            return [p] + l
        l = l[1:]


def convex_hull(s):
    pts = sorted(s, key=DDKEY)
    upper = []
    for p in pts:
        upper = add_chain_point(p, upper, +1)
    lower = []
    for p in pts:
        lower = add_chain_point(p, lower, -1)
    return frozenset(upper + lower)


# ---------- transitions and the state map ------------------------------------


def odd_pos_trans(p):
    g, f = p
    return (dhalf(dsub(g, f)), g)


def odd_nonpos_trans(p):
    g, f = p
    return (dhalf(dadd(g, f)), f)


def even_trans(p):
    g, f = p
    return (dhalf(g), f)


INC = 2


def narrow(Md, s):
    if not s:
        return True
    lo = min(s, key=DDKEY)
    hi = max(s, key=DDKEY)
    l, h = lo[0], hi[0]
    return dcmp((-1, 0), dmult(Md, l)) < 0 and dcmp(dmult(Md, h), (1, 0)) < 0


def process_divstep(M, state):
    Md = dred(M, 0)
    s1 = {}
    for k, v in state.items():
        ev = frozenset(even_trans(p) for p in v)
        s1[INC + k] = s1.get(INC + k, frozenset()) | ev
        if k > 0:
            od = frozenset(odd_pos_trans(p) for p in v)
            s1[INC - k] = s1.get(INC - k, frozenset()) | od
        else:
            od = frozenset(odd_nonpos_trans(p) for p in v)
            s1[INC + k] = s1[INC + k] | od
    out = {}
    for k, v in s1.items():
        if not narrow(Md, v):
            out[k] = convex_hull(v)
    return out


def state0():
    one = (1, 0)
    zero = (0, 0)
    return {1: frozenset([(zero, one), (one, one)])}


def run(M, n, trace=False):
    st = state0()
    sizes = []
    for i in range(n):
        st = process_divstep(M, st)
        sizes.append((i + 1, len(st), sum(len(v) for v in st.values())))
        if trace:
            print(
                f"  iter {i + 1:3d}: keys={len(st):3d} points={sum(len(v) for v in st.values()):4d}"
            )
        if not st:
            break
    return st, sizes


# Small calibration fixtures. M = 0x1DFF certifies at 30 iterations but not at
# 29, and M = 0x1E00 does not certify at 30, so the three runs probe both sides
# of the threshold in M and in the iteration count.
FIXTURES = [  # (M, iters, expect_empty)
    (0x1DFF, 30, True),
    (0x1DFF, 29, False),
    (0x1E00, 30, False),
]

# (iterations, M) thresholds of the exampleN lemmas in hddivsteps_examples.v,
# Wuille (2021), sipa/safegcd-bounds (MIT).
LADDER = [
    (50, 0x30EF80),
    (60, 0x3CFAB01),
    (70, 0x4ED603AD),
    (80, 0x62079D724),
    (90, 0x7AE312BF9F),
    (100, 0x9EB8BB022C3),
    (120, 0xF91D7384FEC8F),
    (140, 0x190C14A6B68D1755),
    (160, 0x28ADCA2F6746850761),
    (180, 0x41164837DBA63CC865F5),
    (200, 0x6800E4D2AC85589F17D4B0),
    (250, 0x14FE91F2A268CC215F1586A78F32),
    (300, 0x454CE3C97F2E8F66511DB7C277DE001AB),
    (350, 0xE18E996A73C8649B67A59B136A03AE874088DA),
    (400, 0x2E2AA07E7688B076C9B312EB5C8AFB58BCCA81E09729),
    (450, 0x96AC54F418D8174B771B32D8EC83D1A044F3543DF0418CF70),
    (500, 0x1E7C8CE70BEDE0E24BE604945C3E4CC5D99270CD57644386F873692),
]

# The 590-iteration threshold of example590 in hddivsteps590.v (same source).
MSTAR = 0x101C840FAED222B9EF3BFE933011B277F84EE3D3F6DC8B496D8CD75B8315AFEC2


def main():
    import time

    ok = True
    print("calibration fixtures:")
    for M, n, expect_empty in FIXTURES:
        t0 = time.monotonic()
        st, _ = run(M, n)
        dt = time.monotonic() - t0
        got_empty = not st
        status = "OK" if got_empty == expect_empty else "FAIL"
        ok &= got_empty == expect_empty
        print(f"  M={M:#x} @ {n}: empty={got_empty} (expect {expect_empty}) [{dt:.3f}s] {status}")
    if not ok:
        sys.exit("CALIBRATION FAILED")

    if "--sizes" in sys.argv:
        print("per-iteration sizes for M=0x1dff (30 iters):")
        _, sizes = run(0x1DFF, 30)
        print(" ", sizes)
        return

    if "--ladder" in sys.argv:
        upto = int(sys.argv[sys.argv.index("--ladder") + 1])
        print("ladder fixtures:")
        for n, M in LADDER:
            if n > upto:
                break
            t0 = time.monotonic()
            st, _ = run(M, n)
            dt = time.monotonic() - t0
            status = "OK" if not st else "FAIL"
            ok &= not st
            print(f"  example{n}: empty={not st} [{dt:.1f}s] {status}")
        if not ok:
            sys.exit("LADDER FAILED")

    if "--full" in sys.argv:
        t0 = time.monotonic()
        st, sizes = run(MSTAR, 590, trace=True)
        dt = time.monotonic() - t0
        print(f"example590: empty={not st} [{dt:.1f}s]")
        if st:
            sys.exit("EXAMPLE590 FAILED")

    print("ALL REQUESTED CHECKS PASSED")


if __name__ == "__main__":
    main()
