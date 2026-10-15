#!/usr/bin/env python3
"""Regenerate constants.json for the dqi-reed-solomon discovered circuit.

main() reruns every search and writes constants.json.  The searches, in
file order:

  Formulas     the multiplication formulas and tower maps (KT12, KT6,
               KT3, CUBE3Q, PAIR3, LIN).  Deterministic; seconds.
  BM windows   the per-round coefficient windows (IMAXV, KMSV, KMMV).
               search.c samples the full Berlekamp-Massey recursion over
               403.2M weight-law draws (56 fixed seeds, 7.2M draws each)
               and counts, per (round, slot), how often the slot changes
               state.  Each window is the smallest one whose clipped
               slots all have mass under 4e-5 per draw.  About 2.7
               CPU-hours; the seeds run on a thread pool (3.5 min on 56
               cores).
  Psi windows  the cofactor windows of the Psi route: per CR op, the
               minimal cyclic interval covering the union support where
               the op acts, over 400k law draws (16 fixed seeds) plus
               the w = 0 anchor and every w = 1 position.  BM runs under
               the freshly derived BM windows.

The models section holds the bit-exact classical models of the circuit
that the Psi window search runs: the fused-Q BM recursion and the Psi
route (stages 5-7).  The script derives the instance from contract.json
and does not import circuit.py, so it runs without constants.json.

Formula derivation.  The field is GF(2^12), poly 4331, x = 2
primitive.  The tower is GF(2^12) = GF(64)(t) = GF(8)(s)(t), read top
down: A = x + x^64, t = x / A, nu = x^65 / A^2, u = x^65,
A' = u + u^8, s = u / A', nu' = u^9 / A'^2, v = u^9.  Tower
coordinates are E12[i + 3j + 6k] = v^i s^j t^k.  The norm descends
both quadratic levels (N = x xbar in GF(64), then M = N Nbar in GF(8)),
so the only nonlinear inverse left is M -> M^3 in GF(8), a 3-product
quadratic form; M^-1 = (M^3)^2 is a free squaring.  A full inversion
costs N (15) + M (6) + M^3 (3) + Nbar M^-1 (12) + xbar N^-1 (30) = 66
Toffoli, and every stage maps 0 to 0.  Rejected towers: GF(64)^2 alone
needs N^31 (five 1-bits, so a generic GF(64) product on top of two
quadratic forms); GF(16)^3 pays a cubic norm; plain Itoh-Tsujii needs
at least four 42-Toffoli products.

  KT12    GF(2^12) = GF(4)[x]: a 6-term product over GF(4) by CRT at
          the four GF(4) points, infinity, and the irreducible
          quadratics x^2+x+w, x^2+x+w^2, x^2+wx+1.  14 GF(4) products
          x 3 = 42; the greedy prune removes nothing.  GF(8)-CRT and
          Chudnovsky-style curves over GF(4) tie at 42; Karatsuba
          variants give 45-51.
  KT6     GF(64) = GF(4)[u], 3 terms, CRT at four points + infinity:
          15 products, the best known bilinear complexity of GF(2^6).
  KT3     the 3-term formula M3TERMS reduced modulo v's minimal
          polynomial: 6 products.
  CUBE3Q  exhaustive search over product subsets: 3 is the minimum.
  PAIR3   elimination rank decomposition of Tr3(r s^2): rank 3.
  LIN     the tower's linear maps (pure linear algebra on the
          elements above).

RUN: uv run python gen_constants.py   (writes constants.json)
"""

import collections
import ctypes
import json
import os
import pathlib
import random
import sys
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np

from uncompute.classical import GF2n, f2_matinv, f2_transpose, parity
from uncompute.testing import build_clib, load_contract

HERE = pathlib.Path(__file__).resolve().parent
PATH = HERE / "constants.json"

# The instance (contract.json), derived as circuit.py derives it.
CONTRACT = load_contract(__file__)
M = CONTRACT["m"]  # code length = nonzero field points
N = CONTRACT["n_synd"]  # syndrome symbols = BM rounds
B = CONTRACT["b"]  # bits per symbol
W = CONTRACT["w_max"]  # unique-decoding radius n/2 = max locator degree
LAW_CUM = CONTRACT["weight_cum_2p16"]  # the grader's 2^16 cumulative weight law
NH = W // 2 + 1  # words per Euclid half E', O' (and per cofactor)
FIELD = GF2n(B, CONTRACT["gf_poly"])
EXP, LOG = FIELD.exp, FIELD.log
gmul, ginv, gsqrt, trace = FIELD.mul, FIELD.inv, FIELD.sqrt, FIELD.trace
DWB = (N - 2).bit_length() + 1  # the signed d counter dW: two's complement
DMAX = 1 << (DWB - 1)  # two's-complement range of dW


def bitlist(m, n):
    return [i for i in range(n) if (m >> i) & 1]


def bitmask(n, pred):
    return sum(1 << i for i in range(n) if pred(i))


DWMASK = (1 << DWB) - 1  # the dW register (two's complement, DWB bits)
MMASK = 31  # m = (dW + 2) >> 2 lives in 5 bits (dW[2:7])
# The cores this process may use (its CPU affinity set, not the whole node).
NCPU = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

# #########################################################################
# Formulas
# #########################################################################
# Every table is deterministic; the module docstring gives the derivation.
# search_kt keeps the CRT formula only if certify_kt accepts it.

# ---------------------------------------------------------------------------
# small F2 helpers
# ---------------------------------------------------------------------------


def rows_apply(rows, x):
    """bit r of the result = parity(rows[r] & x)."""
    v = 0
    for r in range(len(rows)):
        if parity(rows[r] & x):
            v |= 1 << r
    return v


def compose_rows(A, B, nb):
    """Rows of A o B (apply B, then A); B has nb rows."""
    out = []
    for r in range(len(A)):
        m = 0
        for i in bitlist(A[r], nb):
            m ^= B[i]
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# GF(2^b) arithmetic
# ---------------------------------------------------------------------------


def reduce_raw(raw):
    """Reduce a raw F2[x] product (degree <= 2B - 2) mod the field
    polynomial: x^k = EXP[k], since gamma = x.
    """
    v = 0
    for k in bitlist(raw, 2 * B - 1):
        v ^= EXP[k]
    return v


# ---------------------------------------------------------------------------
# polynomial-multiplication formulas (Karatsuba fallback, GF(8) terms)
# ---------------------------------------------------------------------------

M3TERMS = [
    (1, 1, 0b00111),
    (2, 2, 0b01110),
    (4, 4, 0b11100),
    (3, 3, 0b00010),
    (6, 6, 0b01000),
    (5, 5, 0b00100),
]


def kara(n):
    """Recursive symmetric Karatsuba formula for n-term F2 polynomial
    multiplication, [(Amask, Bmask, raw-product mask)].
    """
    if n == 1:
        return [(1, 1, 1)]
    if n == 3:
        return list(M3TERMS)
    n0 = n // 2
    n1 = n - n0
    m = max(n0, n1)
    out = []
    for A, B, C in kara(n0):
        out.append((A, B, C ^ (C << n0)))
    for A, B, C in kara(n1):
        out.append((A << n0, B << n0, (C << n0) ^ (C << (2 * n0))))
    for A, B, C in kara(m):
        Am = 0
        Bm = 0
        for i in range(m):
            if (A >> i) & 1:
                if i < n0:
                    Am ^= 1 << i
                if i < n1:
                    Am ^= 1 << (n0 + i)
            if (B >> i) & 1:
                if i < n0:
                    Bm ^= 1 << i
                if i < n1:
                    Bm ^= 1 << (n0 + i)
        out.append((Am, Bm, C << n0))
    return out


# ---------------------------------------------------------------------------
# GF(4)-tower CRT bilinear formulas (deterministic; KT / KT12 / KT6)
# ---------------------------------------------------------------------------


class F4:
    """GF(4) = {0, 1, w, w+1} inside fld, index-encoded (bit 0 = 1, bit 1 =
    w), so index addition is XOR.
    """

    def __init__(self, fld):
        gmul = fld.mul
        w3 = fld.exp[fld.order // 3]
        assert gmul(w3, w3) == (w3 ^ 1) and gmul(w3, w3 ^ 1) == 1
        self.w3 = w3
        self.E = [0, 1, w3, w3 ^ 1]
        self.I = {self.E[i]: i for i in range(4)}
        self.gmul = gmul

    def mul(self, a, bq):
        return self.I[self.gmul(self.E[a], self.E[bq])]


def f4_irreducible_quadratics(f4):
    """Monic irreducible x^2 + a x + b over GF(4), (a, b) in index order."""
    out = []
    for a in range(4):
        for bq in range(4):
            if all((f4.mul(r, r) ^ f4.mul(a, r) ^ bq) != 0 for r in range(4)):
                out.append((a, bq))
    assert len(out) == 6
    return out


def f4_lams(f4, k, nquad):
    """Return the CRT evaluation functionals for a k-term product over GF(4):
    evaluation at 0, 1, w, w^2, at infinity, then for each of the first
    nquad irreducible quadratics q the two reduction coordinates
    (x^i mod q = c0_i + c1_i x) and their sum (Karatsuba over GF(4)).
    Modulus degree = 4 + 1 + 2 nquad; the product has 2k - 1 coefficients.
    """
    assert 4 + 1 + 2 * nquad == 2 * k - 1, (k, nquad)
    lams = []
    for pt in range(4):
        lam, p = [], 1
        for _ in range(k):
            lam.append(p)
            p = f4.mul(p, pt)
        lams.append(lam)
    lams.append([0] * (k - 1) + [1])
    for a, bq in f4_irreducible_quadratics(f4)[:nquad]:
        L0, L1 = [], []
        c0, c1 = 1, 0
        for _ in range(k):
            L0.append(c0)
            L1.append(c1)
            c0, c1 = f4.mul(c1, bq), c0 ^ f4.mul(c1, a)
        lams.extend([L0, L1, [p ^ q for p, q in zip(L0, L1)]])
    return lams


def f4_digits(f4, fld, coords, gens, k):
    """GF(4)-digit extraction for an n = 2k dimensional coordinate system:
    gens[i] (i < k) are field elements whose GF(4)-span {gens[i], gens[i]
    w} is the space; coords(elt) -> n-bit coordinate vector.  Returns
    DIG[bq] = the k GF(4)-digit indices of coordinate basis vector bq.
    """
    n = 2 * k
    colsE = []
    for i in range(k):
        for j in range(2):
            e = gens[i]
            if j:
                e = fld.mul(e, f4.w3)
            colsE.append(coords(e))
    EXTR = f2_matinv(f2_transpose(colsE, n), n)
    DIG = []
    for bq in range(n):
        c = rows_apply(EXTR, 1 << bq)
        DIG.append([(c >> (2 * i)) & 3 for i in range(k)])
    return DIG


def solve_sym(n, prodlist, mulcoord):
    """Reconstruction solve for symmetric operand masks: target masks T[t]
    per product such that a*b = XOR_t parity(a&m_t) parity(b&m_t) T[t] on all
    n x n basis pairs (verified), or None.
    """
    piv = {}
    for i in range(n):
        for j in range(n):
            a = 0
            for t, mk in enumerate(prodlist):
                if ((mk >> i) & 1) and ((mk >> j) & 1):
                    a |= 1 << t
            r = mulcoord(i, j)
            while a:
                cc = a.bit_length() - 1
                if cc in piv:
                    pa, pr = piv[cc]
                    a ^= pa
                    r ^= pr
                else:
                    piv[cc] = (a, r)
                    a = 0
                    r = 0
            if r:
                return None
    T = [0] * len(prodlist)
    for cc in sorted(piv):
        a, r = piv[cc]
        acc = r
        aa = a & ~(1 << cc)
        while aa:
            c2 = aa.bit_length() - 1
            aa ^= 1 << c2
            acc ^= T[c2]
        T[cc] = acc
    for i in range(n):
        for j in range(n):
            acc = 0
            for t, mk in enumerate(prodlist):
                if ((mk >> i) & 1) and ((mk >> j) & 1):
                    acc ^= T[t]
            if acc != mulcoord(i, j):
                return None
    return T


def crt_f4_formula(n, DIG, k, lams, f4mul, mulcoord):
    """GF(4)-CRT bilinear formula on n = 2k coordinate bits: products =
    3 F2 parities per lambda, reconstruction solved by
    Gaussian elimination over all n x n basis pairs, deterministic greedy
    prune in index order.  Returns [(prodmask, targetmask)] (symmetric
    operand masks) or None.
    """
    prods = []
    for lam in lams:
        m0 = 0
        m1 = 0
        for bq in range(n):
            U = 0
            for i in range(k):
                U ^= f4mul(lam[i], DIG[bq][i])
            if U & 1:
                m0 |= 1 << bq
            if U & 2:
                m1 |= 1 << bq
        prods.extend([m0, m1, m0 ^ m1])
    assert len(prods) == 3 * len(lams)

    T = solve_sym(n, prods, mulcoord)
    if T is None:
        return None
    keep = list(range(len(prods)))
    for t in range(len(prods)):
        if t not in keep:
            continue
        trial = [prods[u] for u in keep if u != t]
        if solve_sym(n, trial, mulcoord) is not None:
            keep = [u for u in keep if u != t]
    plist = [prods[u] for u in keep]
    T = solve_sym(n, plist, mulcoord)
    if T is None:
        return None
    out = []
    for t, mk in enumerate(plist):
        if T[t] and mk:
            out.append((mk, T[t]))
    return out


def build_f4_crt_formula(fld):
    """GF(2^b) multiplication over the poly basis via GF(2^b) = GF(4)[x]:
    b/2 terms over GF(4), moduli = 4 points + infinity + (b/2 - 3)
    irreducible quadratics (b = 12: 42 products).
    """
    b = fld.n
    k = b // 2
    try:
        f4 = F4(fld)
        theta = 2
        gens = [fld.pow(theta, i) for i in range(k)]
        DIG = f4_digits(f4, fld, lambda e: e, gens, k)
        for v in range(1 << b):  # exhaustive digit round-trip
            dg = [0] * k
            for bq in bitlist(v, b):
                dg = [x ^ y for x, y in zip(dg, DIG[bq])]
            w = 0
            for i in range(k):
                w ^= fld.mul(gens[i], f4.E[dg[i]])
            assert w == v
        lams = f4_lams(f4, k, k - 3)
        return crt_f4_formula(b, DIG, k, lams, f4.mul, lambda i, j: fld.mul(1 << i, 1 << j))
    except Exception as ex:  # noqa: BLE001  (best-effort alternative: any failure falls back to kara(b))
        print("f4-crt formula construction failed:", repr(ex))
        return None


def search_kt(fld):
    """Return the GF(2^b) multiplication formula, constants.json shape
    [[Ab, Bb, Tb], ...].  Deterministic: the Karatsuba formula kara(b)
    (b = 12: 54 products), replaced by the GF(4)-tower CRT formula when
    that is smaller.
    """
    b = fld.n
    RAW = kara(b)
    # basis-exhaustive check of RAW (complete for a bilinear form)
    for i in range(b):
        for j in range(b):
            pr = 0
            for Am, Bm, Cm in RAW:
                if parity((1 << i) & Am) and parity((1 << j) & Bm):
                    pr ^= Cm
            assert reduce_raw(pr) == fld.mul(1 << i, 1 << j)
    KT = []
    for Am, Bm, Cm in RAW:
        tm = reduce_raw(Cm)
        if tm:
            KT.append([bitlist(Am, b), bitlist(Bm, b), bitlist(tm, b)])
    F4KT = build_f4_crt_formula(fld)
    if F4KT is not None:
        KT_NEW = [[bitlist(mk, b), bitlist(mk, b), bitlist(tm, b)] for (mk, tm) in F4KT]
        if certify_kt(KT_NEW, fld) and len(KT_NEW) < len(KT):
            KT = KT_NEW
    return KT


# ---------------------------------------------------------------------------
# The towers
# ---------------------------------------------------------------------------


class Tower:
    """The (2, 2, 3) tower of the module docstring.  GF(2^12) = GF(64)(t),
    GF(64) = GF(8)(s), GF(8) = F2[v]; E_list = [v^i s^j t^k] indexed
    i + 3j + 6k; the GF(64) coordinates are E6 = [v^i s^j] (i + 3j), the
    GF(8) coordinates E3 = [v^i].
    """

    def __init__(self, fld):
        self.fld = fld
        b = fld.n
        assert b == 12
        h = b // 2
        EXP, gmul, ginv, gpow = fld.exp, fld.mul, fld.inv, fld.pow
        Ael = EXP[1 << h] ^ 2  # x^(2^h) + x = Tr(x)
        t_el = gmul(2, ginv(Ael))  # t = x / A
        u_el = EXP[(1 << h) + 1]  # u = x^(2^h + 1) = N(x)
        nu_el = gmul(u_el, ginv(gmul(Ael, Ael)))  # nu = u / A^2
        assert gpow(u_el, 1 << h) == u_el and u_el != 0
        assert gpow(Ael, 1 << h) == Ael and Ael != 0
        assert gpow(nu_el, 1 << h) == nu_el
        assert gmul(t_el, t_el) ^ t_el == nu_el
        self.Ael, self.t_el, self.u_el, self.nu_el = Ael, t_el, u_el, nu_el
        self.h = h
        # sub-level GF(64) = GF(8)(s)
        Ap = gpow(u_el, 8) ^ u_el  # Tr_{6/3}(u)
        s_el = gmul(u_el, ginv(Ap))
        v_el = gpow(u_el, 9)  # N_{6/3}(u)
        nup_el = gmul(v_el, ginv(gmul(Ap, Ap)))
        assert gpow(Ap, 8) == Ap and Ap != 0
        assert gpow(v_el, 8) == v_el and gpow(v_el, 7) == 1 and v_el != 1
        assert gpow(nup_el, 8) == nup_el
        assert gmul(s_el, s_el) ^ s_el == nup_el
        self.Ap, self.s_el, self.v_el, self.nup_el = (Ap, s_el, v_el, nup_el)
        E3 = [gpow(v_el, i) for i in range(3)]
        E6 = E3 + [gmul(e, s_el) for e in E3]
        E_list = E6 + [gmul(e, t_el) for e in E6]
        self.E_sub = E6
        self.E3 = E3
        self.E_list = E_list
        self.MINVR = f2_matinv(f2_transpose(E_list, b), b)
        for v in range(1 << b):
            c = self.coords_full(v)
            w = 0
            for u in bitlist(c, b):
                w ^= E_list[u]
            assert w == v
        # projections poly -> (a0, a1) and the conjugate's constant part
        self.A0ROWS = self.MINVR[0:h]
        self.A1ROWS = self.MINVR[h:b]
        self.A01ROWS = [self.A0ROWS[r] ^ self.A1ROWS[r] for r in range(h)]
        # squaring and multiply-by-nu on subfield coords
        self.SQR = f2_transpose([self.coords_sub(gmul(e, e)) for e in self.E_sub], h)
        self.NUR = f2_transpose([self.coords_sub(gmul(nu_el, e)) for e in self.E_sub], h)
        NSR = compose_rows(self.NUR, self.SQR, h)
        # linear part of the norm N = a0 a1 + a0^2 + nu a1^2 on poly words
        self.LNROWS = []
        for r in range(h):
            m = 0
            for j in bitlist(self.SQR[r], h):
                m ^= self.A0ROWS[j]
            for j in bitlist(NSR[r], h):
                m ^= self.A1ROWS[j]
            self.LNROWS.append(m)
        # GF(8) level: coords3 = low 3 bits of the GF(64) coords of a
        # GF(8) element; RED3 = v^k for the KT3 reduction
        self.RED3 = [self.coords3(gpow(v_el, k)) for k in range(5)]
        self.SQ3R = f2_transpose([self.coords3(gmul(e, e)) for e in self.E3], 3)
        self.NU3R = f2_transpose([self.coords3(gmul(nup_el, e)) for e in self.E3], 3)
        NSR3 = compose_rows(self.NU3R, self.SQ3R, 3)
        # linear part of M = n0 n1 + n0^2 + nu' n1^2 on GF(64) coords
        self.LMROWS = [self.SQ3R[r] ^ (NSR3[r] << 3) for r in range(3)]
        self.FCUBE3 = [self.coords3(gpow(self.elt3(c), 3)) for c in range(8)]

    # -- coordinates --
    def coords_full(self, v):
        return rows_apply(self.MINVR, v)

    def coords_sub(self, v):
        c = self.coords_full(v)
        assert c < (1 << self.h), "not a subfield element"
        return c

    def coords3(self, v):
        c = self.coords_full(v)
        assert c < 8, "not a GF(8) element"
        return c

    def elt3(self, c):
        w = 0
        for i in bitlist(c, 3):
            w ^= self.E3[i]
        return w

    def lin_tables(self):
        """Return the LIN block of constants.json: the linear maps circuit.py
        reads.
        """
        return {
            "E12": list(self.E_list),
            "A0ROWS": list(self.A0ROWS),
            "A1ROWS": list(self.A1ROWS),
            "LNROWS": list(self.LNROWS),
            "SQ3R": list(self.SQ3R),
            "LMROWS": list(self.LMROWS),
        }


# ---------------------------------------------------------------------------
# subfield multiplication formulas
# ---------------------------------------------------------------------------


def search_kt6(fld, tower):
    """Return the GF(64) multiplication formula in tower coords (E6), 15
    products: GF(64) = GF(4)[u], 3 terms over GF(4), CRT at the four points
    + infinity, built directly on the 6-bit coordinate vectors.
    """
    f4 = F4(fld)
    u = tower.u_el
    gens = [fld.pow(u, i) for i in range(3)]
    DIG = f4_digits(f4, fld, tower.coords_sub, gens, 3)
    lams = f4_lams(f4, 3, 0)
    assert len(lams) == 5
    E6 = tower.E_sub
    F = crt_f4_formula(
        6, DIG, 3, lams, f4.mul, lambda i, j: tower.coords_sub(fld.mul(E6[i], E6[j]))
    )
    assert F is not None, "GF(64) CRT formula failed"
    return [[bitlist(mk, 6), bitlist(mk, 6), bitlist(tm, 6)] for (mk, tm) in F]


def search_kt3(tower):
    """Return the GF(8) multiplication formula in the v-basis, 6 products:
    M3TERMS reduced modulo the minimal polynomial of v.
    """
    KT3 = []
    for Am, Bm, Cm in M3TERMS:
        tm = 0
        for kk in range(5):
            if (Cm >> kk) & 1:
                tm ^= tower.RED3[kk]
        if tm:
            KT3.append([bitlist(Am, 3), bitlist(Bm, 3), bitlist(tm, 3)])
    return KT3


# ---------------------------------------------------------------------------
# quadratic-form formulas: exact solve and exhaustive search (CUBE3Q)
# ---------------------------------------------------------------------------


def qp_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _offvec(n, u, w):
    v = 0
    for idx, (i, j) in enumerate(qp_pairs(n)):
        if (((u >> i) & (w >> j)) ^ ((u >> j) & (w >> i))) & 1:
            v |= 1 << idx
    return v


def quad_solve(n, prods, fvals):
    """Given product pairs (u, w),
    solve for output masks + linear correction so the quadratic formula
    reproduces fvals on all 2^n elements; returns (tris, corr) or None.
    """
    pairs = qp_pairs(n)
    bas = {}
    for t_, (u, w) in enumerate(prods):
        x = _offvec(n, u, w)
        sel = 1 << t_
        while x:
            mb = x.bit_length() - 1
            if mb in bas:
                x ^= bas[mb][0]
                sel ^= bas[mb][1]
            else:
                bas[mb] = (x, sel)
                break
    tg = [0] * n
    for idx, (i, j) in enumerate(pairs):
        cdiff = fvals[(1 << i) | (1 << j)] ^ fvals[1 << i] ^ fvals[1 << j]
        for m in range(n):
            if (cdiff >> m) & 1:
                tg[m] |= 1 << idx
    outs = []
    for m in range(n):
        x = tg[m]
        sel = 0
        while x:
            mb = x.bit_length() - 1
            if mb not in bas:
                return None
            x ^= bas[mb][0]
            sel ^= bas[mb][1]
        outs.append(sel)
    tris = []
    for t_, (u, w) in enumerate(prods):
        om = 0
        for m in range(n):
            if (outs[m] >> t_) & 1:
                om |= 1 << m
        if om:
            tris.append((u, w, om))
    corr = [0] * n
    for i in range(n):
        cv = fvals[1 << i]
        for u, w, om in tris:
            if (u >> i) & 1 and (w >> i) & 1:
                cv ^= om
        for m in range(n):
            if (cv >> m) & 1:
                corr[m] |= 1 << i
    for x in range(1 << n):
        if quad_apply((tris, corr), x, n) != fvals[x]:
            return None
    return (tris, corr)


def quad_exhaustive(n, fvals):
    """EXHAUSTIVE minimum-product quadratic formula for fvals on n bits:
    candidate products are the unordered pairs u < w of nonzero masks (a
    pair (u, u) is linear and never needed); subsets are tried in
    increasing size, lexicographic order, and the first solvable one wins.
    Feasible for n <= 4; used for GF(8).
    """
    from itertools import combinations

    cands = [(u, w) for u in range(1, 1 << n) for w in range(u + 1, 1 << n)]
    for r in range(len(qp_pairs(n)) + 1):
        for sub in combinations(cands, r):
            res = quad_solve(n, list(sub), fvals)
            if res is not None:
                tris, corr = res
                return [[list(t) for t in tris], list(corr)]
    raise AssertionError("no quadratic formula (impossible)")


def search_cube3q(tower):
    """Exhaustive, deterministic: x -> x^3 in GF(8), v-basis."""
    return quad_exhaustive(3, tower.FCUBE3)


# ---------------------------------------------------------------------------
# pairing-form rank decomposition (elimination)
# ---------------------------------------------------------------------------


def pair_decomp(Brows, n):
    """B[u] = mask over v of the form on basis pair (u, v).  Returns
    [(p, qm), ...] with B(a, c) = XOR_k parity(a & p_k) & parity(c & qm_k).
    """
    Rw = list(Brows)
    out = []
    while True:
        u0 = None
        for u in range(n):
            if Rw[u]:
                u0 = u
                break
        if u0 is None:
            break
        qm = Rw[u0]
        v0 = (qm & -qm).bit_length() - 1
        p = bitmask(n, lambda u: (Rw[u] >> v0) & 1)
        for u in bitlist(p, n):
            Rw[u] ^= qm
        out.append((p, qm))
    chk = [0] * n
    for p, qm in out:
        for u in bitlist(p, n):
            chk[u] ^= qm
    assert chk == list(Brows)
    return out


def search_pair3(fld, tower):
    """Rank decomposition of B(r, s) = Tr3(r s^2) on GF(8) v-coords."""
    gmul = fld.mul
    E3 = tower.E3
    B33 = [bitmask(3, lambda j: trace(gmul(E3[i], gmul(E3[j], E3[j])), 3)) for i in range(3)]
    return [[p, qm] for (p, qm) in pair_decomp(B33, 3)]


# ---------------------------------------------------------------------------
# formula evaluation (search_kt's certify_kt, quad_solve's quad_apply)
# ---------------------------------------------------------------------------


def kt_mul_fn(KT):
    ktm = [
        (sum(1 << u for u in Ab), sum(1 << u for u in Bb), sum(1 << u for u in Tb))
        for (Ab, Bb, Tb) in KT
    ]

    def mul(a, bb):
        acc = 0
        for Am, Bm, Tm in ktm:
            if parity(a & Am) and parity(bb & Bm):
                acc ^= Tm
        return acc

    return mul


def certify_kt(KT, fld):
    """Return True iff KT multiplies exactly in GF(2^b) over the poly basis:
    checked on the b x b basis pairs, complete for a bilinear form.
    """
    kt_mul = kt_mul_fn(KT)
    return all(
        kt_mul(1 << i, 1 << j) == fld.mul(1 << i, 1 << j)
        for i in range(fld.n)
        for j in range(fld.n)
    )


def quad_apply(Q, x, n):
    tris, corr = Q
    acc = 0
    for u, w, om in tris:
        if parity(u & x) and parity(w & x):
            acc ^= om
    for m in range(n):
        if parity(corr[m] & x):
            acc ^= 1 << m
    return acc


def run_b12():
    """All formula searches (every one deterministic); returns (tables,
    timings) with tables in constants.json shape.
    """
    fld = FIELD
    tower = Tower(fld)
    tables, tm = {}, {}
    for name, search in (
        ("KT12", lambda: search_kt(fld)),
        ("KT6", lambda: search_kt6(fld, tower)),
        ("KT3", lambda: search_kt3(tower)),
        ("CUBE3Q", lambda: search_cube3q(tower)),
        ("PAIR3", lambda: search_pair3(fld, tower)),
    ):
        t0 = time.time()
        tables[name] = search()
        tm[name + "_s"] = time.time() - t0
    tables["LIN"] = tower.lin_tables()
    return tables, tm


# #########################################################################
# Models
# #########################################################################
# Bit-exact classical models of the circuit (Builder-free): the one Python
# model of its fused-Q BM recursion, and the Psi route (stages 5-7) op for
# op.  The Psi window search runs them.

# =========================================================================
# BM recursion (q-form conditional-swap, fused-Q B bank)
# =========================================================================
# The B bank rotates at bank-1.  A per-round hook instruments the loop.


def bm_rounds(S_, win, bank=W + 2, freelast=False, hook=None):
    """Run the recursion on the syndrome list S_ (S_1 first).

    win = (IMAXV, KMSV, KMMV): the circuit's truncation windows, with
    its (1,1) swap elision; win=None runs untruncated.  freelast=True
    takes the would-be ctl branch at r=N-1 instead of the circuit's
    elision (the outputs are monicize-equal).  hook, when given, fires
    once per round AFTER the round's update with the instrumented round
    state: r, d_old (d entering the round), ctl, parked (the post-park
    discrepancy), delc, m, b_rot0 (bbm[0] after rotation), sig_old0
    (sig[0] before the ctl swap), sig_in / bbm_in (the sig / rotated bbm
    lists before the swap), swapped (the slots the ctl swap exchanged),
    kmm (the update window), and the live sig/bbm lists.
    Returns (sig, d) at the end of the last round.  Field products run in
    the log domain (LS: the syndrome logs, -1 for zero).
    """
    sig = [0] * (W + 1)
    sig[0] = 1
    bbm = [0] * bank
    bbm[0] = 1
    delc = 1
    d = 0
    LS = [LOG[s] if s else -1 for s in S_]
    for r in range(N):
        bbm = [bbm[bank - 1]] + bbm[: bank - 1]
        if win is not None:
            imax, kms_, kmm_ = win[0][r], win[1][r], win[2][r]
        else:
            imax = min(r, W)
            kms_ = kmm_ = min(r + 1, W)
        D = 0
        for i in range(imax + 1):
            a, ls = sig[i], LS[r - i]
            if a and ls >= 0:
                D ^= EXP[LOG[a] + ls]
        ctl = 1 if (D != 0 and d >= 0) else 0
        if r == N - 1 and not freelast:
            ctl = 0
        if ctl:
            D, delc = delc, D
        assert delc != 0  # DEL starts at 1; replaced only by nonzero D
        m = EXP[(LOG[D] - LOG[delc]) % M] if D else 0
        if hook is not None:
            sig_in, bbm_in = sig[:], bbm[:]
        swapped = []
        if ctl:
            for k in range(kms_ + 1):
                if win is not None and r == 1 and k == 1:
                    continue  # the circuit's proven (1,1) swap identity
                sig[k], bbm[k] = bbm[k], sig[k]
                swapped.append(k)
        if m:
            lm = LOG[m]
            for k in range(kmm_ + 1):
                if bbm[k]:
                    sig[k] ^= EXP[lm + LOG[bbm[k]]]
        d_old = d
        if r < N - 1:
            d = (-d) if ctl else (d + 1)
            assert -DMAX <= d <= DMAX - 1
        if hook is not None:
            hook(
                {
                    "r": r,
                    "d_old": d_old,
                    "ctl": ctl,
                    "parked": D,
                    "delc": delc,
                    "m": m,
                    "b_rot0": bbm_in[0],
                    "sig_old0": sig_in[0],
                    "sig_in": sig_in,
                    "bbm_in": bbm_in,
                    "swapped": swapped,
                    "kmm": kmm_,
                    "sig": sig,
                    "bbm": bbm,
                }
            )
    return sig, d


# =========================================================================
# Psi route (bit-exact model of circuit stages 5-7)
# =========================================================================
# Every op maps 1:1 to a circuit gadget of circuit.py (psi_forward,
# psi_pre, psi_seed, build_R): relabels and swaps (free), XOR-product
# accumulates (paid), zero tests (flags), controlled swaps and rotations,
# inversions (totalized: inv(0) = 0, as the tower chain), and free
# measurement erasures whose CLAIMS are checked here (a failed claim = the
# circuit shot fails; hmr resets the wire, so the failure stays on that
# shot).  The cofactor ops take the circuit's windows (win, a
# {(kind, j): (lo, n)} map) when given; win=None runs them unwindowed.
# Consumer: the Psi window search below.


# ---------------------------------------------------------------- sampling
def sample(rng):
    """One weight-law draw: (w, positions j, values e_j)."""
    u = rng.getrandbits(16)
    w = 0
    while LAW_CUM[w] <= u:
        w += 1
    pos = rng.sample(range(M), w)
    val = [rng.randrange(1, M + 1) for _ in range(w)]
    return w, pos, val


def syndromes(pos, val):
    """S_r = sum_j val_j gamma^(pos_j r), r = 1..N, in the log domain."""
    S = [0] * N
    for p, v in zip(pos, val):
        lv = LOG[v]
        e = 0
        for r in range(N):
            e += p
            if e >= M:
                e -= M
            S[r] ^= EXP[lv + e]
    return S


def locator(pos):
    """sigma(z) = prod (1 - gamma^p z), ascending, 36 coefficients."""
    s = [1]
    for p in pos:
        # s_i ^= s_{i-1} * gamma^p, multiplied in the log domain
        s = [
            (s[i] if i < len(s) else 0) ^ (EXP[LOG[s[i - 1]] + p] if i and s[i - 1] else 0)
            for i in range(len(s) + 1)
        ]
    return s + [0] * (W + 1 - len(s))


# ---------------------------------------------------------------- BM
def bm(S_, win):
    """Return the circuit's windowed BM recursion (bm_rounds) plus the
    circuit's final dW (= model d - 1: the r=N-2 increment is dropped).
    """
    sig, d = bm_rounds(S_, win=win)
    return sig, (d - 1) & DWMASK


# ---------------------------------------------------------------- pipeline
class ClaimError(Exception):
    """A free measurement-erase claim failed: the circuit shot fails."""


def claim(ok, what):
    if not ok:
        raise ClaimError(what)


def split_pad(sigm, dW):
    """Stage 4-6: E_i = sqrt(sigma_2i), O_i = sqrt(sigma_2i+1) (E_0 = ONE = 1),
    m = (dW + 2) >> 2 (5 bits of the in-place dW + 2), non-cyclic shift up by m
    as the barrel of cswap chains (a permutation on every input).
    """
    E = [1] + [gsqrt(sigm[2 * i]) for i in range(1, 18)]
    O = [gsqrt(sigm[2 * i + 1]) for i in range(18)]
    m = ((dW + 2) & DWMASK) >> 2 & MMASK
    for s in (16, 8, 4, 2, 1):
        if m & s:
            for X in (E, O):
                for k in range(17 - s, -1, -1):
                    X[k], X[k + s] = X[k + s], X[k]
    return E, O, m


def rot_up(g, lo=0):
    return g[:lo] + [g[17]] + g[lo:17]


def rot_dn(g, lo=0):
    return g[:lo] + g[lo + 1 :] + [g[lo]]


def forward(E, O, pre_exact=True):
    """F: round 1, the block-0 zero-pivot repair, and 17 blocks, each a
    flag-controlled swap, two top eliminations (q_hi, q_lo stored in the
    freed top slots) and a one-time zero-pivot repair recorded as (eps,
    idx).  Returns (f, g, eps, idx, pre, log) or raises ClaimError.
    """
    f, g = E[:], O[:]
    pre = 0
    if f[17] == 0 and pre_exact:
        pre = 1  # flag [f17 == 0] + 18-word cswap
        f, g = g, f
    log = []
    # round 1: one elimination of g's top by f's top
    fi = ginv(f[17])
    q1 = gmul(g[17], fi)
    claim(g[17] == gmul(q1, f[17]), "r1 g17")
    for k in range(17):
        g[k] ^= gmul(q1, f[k])
    g[17] = q1
    g = rot_up(g)
    eps, idx = 0, 0
    zp = int(g[17] == 0)
    if zp:
        g = rot_up(g, lo=1)
        eps, idx = 1, 0
    log.append((0, 0, zp, zp))
    for j in range(1, 18):
        f[j - 1], g[j - 1] = g[j - 1], f[j - 1]  # boundary swap (relabel)
        flag = int(g[17] != 0)
        if flag:
            for k in range(j, 18):
                f[k], g[k] = g[k], f[k]
        claim(flag == int(g[17] != 0), f"flag erase j{j}")
        fi = ginv(f[17])
        qh = gmul(g[17], fi)
        t = gmul(qh, f[16]) if j <= 16 else 0  # elim1's window is [j, 17]
        ql = gmul(g[16] ^ t, fi)
        # block elimination: two sequential top eliminations with a full
        # rot_up between (the circuit fuses them into one (18-j)-product
        # evaluation/interpolation block)
        claim(g[17] == gmul(qh, f[17]), f"g17 j{j}")
        for k in range(j, 18):
            g[k] ^= gmul(qh, f[k])
        g[17] = qh
        g = rot_up(g)
        claim(g[17] == gmul(ql, f[17]), f"g16 j{j}")
        for k in range(j, 18):
            g[k] ^= gmul(ql, f[k])
        g[17] = ql
        g = rot_up(g)
        zp = int(g[17] == 0)
        c = zp & (1 - eps)
        if c:
            g = rot_up(g, lo=j + 1)
            eps, idx = 1, j
        claim(c == (eps & int(idx == j)), f"c erase j{j}")
        claim(zp == (c | int(g[17] == 0)), f"zp erase j{j}")
        log.append((j, flag, zp, c))
    return f, g, eps, idx, pre, log


def _supp(v):
    return frozenset(k for k in range(NH) if v[k])


def _wset(win, kind, j):
    if win is None:
        return set(range(NH))
    lo, n = win[(kind, j)]
    return {(lo + i) % NH for i in range(n)}


def _crot(v, win, j):
    """Apply the controlled rotation b <- z b restricted to its 'rz' window (the
    circuit rotates words lo..lo+n cyclically by one).
    """
    if win is None:
        return rot_up(v)
    lo, n = win[("rz", j)]
    if n == 0:
        return v[:]
    ks = [(lo + i) % NH for i in range(min(n + 1, NH))]
    out = v[:]
    for i in range(len(ks)):
        out[ks[i]] = v[ks[i - 1]]
    return out


def _cmul(a, b0, qs, win, j):
    """Apply a ^= (sum_i q_i z^(i+1)) * b0 over b0's window (cyclic)."""
    ws = _wset(win, "mul", j)
    a = a[:]
    for i, q in enumerate(qs):
        for k in ws:
            a[(k + i + 1) % NH] ^= gmul(q, b0[k])
    return a


def _cswap(a, b, win, j):
    ws = _wset(win, "swap", j)
    a2, b2 = a[:], b[:]
    for k in ws:
        a2[k], b2[k] = b[k], a[k]
    return a2, b2


def backward(f, g, pre, m, log, hook=None, win=None):
    """R: journal-reverse of F interleaved with the cofactor recurrence CR
    (cofactor windows win, None = unwindowed).  Seed a[(m+1) % 18] =
    inv(h), h = f17 at F's end.  Returns restored (E', O') and the cofactor
    pair (a, b).
    """
    f, g = f[:], g[:]
    a, b = [0] * NH, [0] * NH
    a[(m + 1) % NH] = ginv(f[17])
    for j in range(17, 0, -1):
        _, flag, _zp, c = log[j]
        if c:  # repair undo + CR repair shift
            if hook:
                hook(("rz", j), _supp(b))
            g = rot_dn(g, lo=j + 1)
            b = _crot(b, win, j)
        # elimination undo (q_lo then q_hi), CR a ^= zQ b in between shifts
        g = rot_dn(g)
        ql = g[17]
        g[17] = 0
        for k in range(j, 18):
            g[k] ^= gmul(ql, f[k])
        g = rot_dn(g)
        qh = g[17]
        g[17] = 0
        for k in range(j, 18):
            g[k] ^= gmul(qh, f[k])
        if hook and (ql or qh):
            hook(("mul", j), _supp(b))
        a = _cmul(a, b, [ql, qh], win, j)
        b = rot_up(rot_up(b))
        if flag:
            if hook:
                hook(("swap", j), _supp(a) | _supp(b))
            for k in range(j, 18):
                f[k], g[k] = g[k], f[k]
            a, b = _cswap(a, b, win, j)
        f[j - 1], g[j - 1] = g[j - 1], f[j - 1]
    if log[0][3]:
        if hook:
            hook(("rz", 0), _supp(b))
        g = rot_dn(g, lo=1)
        b = _crot(b, win, 0)
    g = rot_dn(g)
    q1 = g[17]
    g[17] = 0
    for k in range(18):
        g[k] ^= gmul(q1, f[k])
    if hook and q1:
        hook(("mul", 0), _supp(b))
    a = _cmul(a, b, [q1], win, 0)
    b = rot_up(b)
    if pre:
        if hook:
            hook(("swap", 0), _supp(a) | _supp(b))
        f, g = g, f
        a, b = _cswap(a, b, win, 0)
    return f, g, a, b


def monicize(sig):
    """sigma / sigma_0 (the circuit's monicize; ginv(0) = 0)."""
    iv = ginv(sig[0])
    return [1] + [gmul(iv, s) for s in sig[1:]]


def prep(pos, val, win):
    """Return the monic windowed-BM locator and the circuit's final dW, under
    the BM windows win.
    """
    sig, dW = bm(syndromes(pos, val), win)
    return monicize(sig), dW


# #########################################################################
# BM windows
# #########################################################################
# search.c implements the full (untruncated) recursion with the slot
# instrumentation and the weight-law draw scheme.  The library compiles
# on demand into __pycache__/ (gitignored), staleness-checked by a
# source-sha256 sidecar and written atomically.  CC and CFLAGS are
# honored from the environment.

NSIG = W + 1
NBANK = NSIG + 1
NOMM = W
TRACE_LEN = 2 + N * 7 + NSIG + NOMM

SLOT_THRESH = 4e-5  # marginal-mass bar for a clipped slot
SLOT_DRAWS = 7_200_000  # draws per seed
SLOT_SEEDS = [20260501 + i for i in range(28)] + [20261501 + i for i in range(28)]

_SRC = HERE / "search.c"
_LIB = HERE / "__pycache__" / ("libsearch.dylib" if sys.platform == "darwin" else "libsearch.so")
_lib = None


def load():
    """Return the compiled search.c, with its instance constants asserted."""
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(str(build_clib(_SRC, _LIB, os.environ.get("CFLAGS", "-O2").split())))
    i16p = np.ctypeslib.ndpointer(np.int16, flags="C")
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C")
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C")
    u32p = np.ctypeslib.ndpointer(np.uint32, flags="C")
    lib.rs_params.argtypes = [i32p]
    lib.rs_slots.argtypes = [
        i16p,
        i16p,
        u32p,
        ctypes.c_uint64,
        ctypes.c_int64,
        i64p,
        i64p,
        i64p,
        i64p,
    ]
    p = np.zeros(10, np.int32)
    lib.rs_params(p)
    assert p.tolist() == [B, M, M, N, W, NSIG, NBANK, NOMM, 128, TRACE_LEN], (
        "search.c instance constants drifted",
        p.tolist(),
    )
    _lib = lib
    return lib


def _slot_shard(seed):
    """One seed's full-run (mac_hit, swap_hit, ni_hist) counters."""
    exp = np.array(EXP[:M], np.int16)
    log = np.array(LOG, np.int16)
    cum = np.array(LAW_CUM, np.uint32)
    out = np.zeros(2, np.int64)
    mac, swp, nih = (np.zeros(N * NSIG, np.int64) for _ in range(3))
    load().rs_slots(exp, log, cum, seed, SLOT_DRAWS, out, mac, swp, nih)
    assert out[1] == 0, ("model invariant violated", seed, int(out[1]))
    return mac, swp, nih


def slot_census():
    """Return the summed per-(round, slot) counters over every seed.  ctypes
    releases the GIL, so the seeds run in parallel threads.
    """
    load()
    with ThreadPool(NCPU) as pool:
        shards = pool.map(_slot_shard, SLOT_SEEDS)
    return tuple(np.sum([s[i] for s in shards], axis=0).reshape(N, NSIG) for i in range(3))


def derive_schedule(mac, swp, nih, draws, thresh=SLOT_THRESH):
    """Minimal windows under the per-slot bar: for each round the
    smallest imax / kms / kmm such that every clipped slot above it has
    mass < thresh * draws, with the r=0 / r=N-1 endpoints pinned by
    the circuit.
    """
    bar = thresh * draws
    imx, kms, kmm = [0] * N, [0] * N, [0] * N
    for r in range(N):
        imax_f = min(r, W)
        kmax_f = min(r + 1, W)
        k = imax_f
        while k > 0 and nih[r, k:].sum() < bar:
            k -= 1
        imx[r] = k
        k = kmax_f
        while k > 0 and swp[r, k] < bar:
            k -= 1
        kms[r] = k
        k = kmax_f
        while k > 0 and mac[r, k] < bar:
            k -= 1
        kmm[r] = k
    imx[0], kms[0], kmm[0] = 0, 1, 0
    imx[N - 1] = kms[N - 1] = kmm[N - 1] = W
    return imx, kms, kmm


def bm_windows():
    """Return the derived per-round windows (IMAXV, KMSV, KMMV)."""
    mac, swp, nih = slot_census()
    return derive_schedule(mac, swp, nih, SLOT_DRAWS * len(SLOT_SEEDS))


# #########################################################################
# Psi windows
# #########################################################################
# Draw scheme: shard i uses random.Random(PSI_SEED + i) and sample().

PSI_DRAWS = 400_000
PSI_SEED = 300
PSI_SHARDS = 16


def cyc_interval(ix):
    """Minimal cyclic interval (lo, n) of Z/18 covering the index set."""
    ix = sorted(set(ix))
    if not ix:
        return 0, 0
    if len(ix) == 18:
        return 0, 18
    gaps = [((ix[(i + 1) % len(ix)] - ix[i]) % 18 or 18, i) for i in range(len(ix))]
    g, i = max(gaps)
    return ix[(i + 1) % len(ix)], 18 - g + 1


def _supports(cases, win):
    U = collections.defaultdict(set)

    def hook(key, sup):
        U[key] |= sup

    for pos, val in cases:
        sigm, dW = prep(pos, val, win)
        if sigm != locator(pos):
            continue
        E, O, m = split_pad(sigm, dW)
        try:
            f, g, _eps, _idx, pre, log = forward(E, O)
        except ClaimError:
            continue
        backward(f, g, pre, m, log, hook=hook)
    return U


def _shard_windows(args):
    seed, n, win = args
    rng = random.Random(seed)
    cases = [sample(rng)[1:] for _ in range(n)]
    return {k: sorted(v) for k, v in _supports(cases, win).items()}


def anchor_cases():
    rng = random.Random(3)
    return [([], [])] + [([p], [rng.randrange(1, M + 1)]) for p in range(M)]


def table(U):
    return {
        kind: {j: cyc_interval(U.get((kind, j), ())) for j in range(17, -1, -1)}
        for kind in ("mul", "rz", "swap")
    }


def psi_windows(win):
    """Return the Psi cofactor windows, with BM run under the windows win."""
    per = PSI_DRAWS // PSI_SHARDS
    with Pool(NCPU) as pool:
        res = pool.map(_shard_windows, [(PSI_SEED + i, per, win) for i in range(PSI_SHARDS)])
    U = collections.defaultdict(set)
    for r in res:
        for k, v in r.items():
            U[k] |= set(v)
    for k, v in _supports(anchor_cases(), win).items():
        U[k] |= v
    return {k: {str(j): list(v) for j, v in d.items()} for k, d in table(U).items()}


# #########################################################################
# constants.json
# #########################################################################


def generate():
    """Run every search; return the constants.json bytes."""
    t0 = time.time()
    tables, _ = run_b12()
    print("formulas: %.1fs" % (time.time() - t0), flush=True)
    imx, kms, kmm = bm_windows()
    print("bm_windows: %.0fs" % (time.time() - t0), flush=True)
    psi = psi_windows((imx, kms, kmm))
    print("psi_windows: %.0fs" % (time.time() - t0), flush=True)
    consts = {
        **{k: tables[k] for k in ("KT12", "KT6", "KT3", "CUBE3Q", "PAIR3", "LIN")},
        "bm_windows": {"IMAXV": imx, "KMSV": kms, "KMMV": kmm},
        "psi_windows": psi,
    }
    return (json.dumps(consts, indent=1) + "\n").encode()


def main():
    new = generate()
    old = PATH.read_bytes() if PATH.exists() else None
    PATH.write_bytes(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
