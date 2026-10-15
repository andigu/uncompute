#!/usr/bin/env python3
"""dqi-reed-solomon discovered circuit: fused-Q truncated Berlekamp-Massey
decode at (m, n, b) = (4095, 70, 12), then a half-degree Bezout pass that
turns the per-point Forney division into a precomputed Psi = 1/sigma'
mod sigma, so the Chien/Forney sweep costs 23 Toffoli per point (root
test 11 + a rank-12 gated pairing 12) instead of 80.  Built natively on
uncompute.Builder.

The BM body runs the (2,2,3)-tower field arithmetic under the measured
window schedule and stores no accepted discrepancy: the scale identity
sigma_0 * DEL == 1 replaces it (README).
The Psi route (stages 5-7) holds the Bezout pair in the syndrome TAIL,
zeroed by extending the in-place omega transform to all 70 rows.

Layout (flat):
  circuit.py         the tables (field, formulas, schedule; loaded from
                    constants.json), the circuit layer (term lists,
                    emitter, Psi-route gadgets, sweep body), and build(),
                    the stage sequence with its phase marks and cost pins.
  constants.json    every searched value: the formula tables, the BM
                    windows (bm_windows), the Psi windows (psi_windows).
  gen_constants.py  reruns every search and writes constants.json; it
                    holds the bit-exact classical models the searches run.
  search.c          the C sampler behind the BM window census.

RUN: uv run python circuit.py -- builds, asserts its cost, and
streams to $UNCOMPUTE_OPS_PATH (else ./ops.bin).  Importing is cheap and
builds nothing; build() feeds the occupancy/liveness profilers.
Evaluate: python tools/eval.py dqi-reed-solomon.

README.md in this folder describes the construction.
"""

import functools
import json
import pathlib
from contextlib import contextmanager

import numpy as np

from uncompute import Builder
from uncompute.bennett import Journal
from uncompute.classical import GF2n, f2_matinv, f2_transpose, parity
from uncompute.gadgets import declare_registers, linear_in_place
from uncompute.testing import assert_metrics, emit_and_report, load_contract

HERE = pathlib.Path(__file__).resolve().parent


# =========================================================================
# Field and instance
# =========================================================================

CONTRACT = load_contract(__file__)
POLY = CONTRACT["gf_poly"]
M = CONTRACT["m"]  # code length = nonzero field points
N = CONTRACT["n_synd"]  # syndrome symbols = BM rounds
B = CONTRACT["b"]  # bits per symbol
W = CONTRACT["w_max"]  # unique-decoding radius n/2 = max locator degree
LAW_CUM = CONTRACT["weight_cum_2p16"]  # the grader's 2^16 cumulative weight law
assert M == (1 << B) - 1 and N == 2 * W
NH = W // 2 + 1  # words per Euclid half E', O' (and per cofactor)
SQRT_EXP = (M + 1) // 2  # sqrt(x) = x^SQRT_EXP (2048)

# EXP/LOG tables (gamma = 2): EXP[k] = gamma^k for k < 2M, LOG its inverse.
# ginv is totalized (zero maps to zero), as the tower inversion chain.
FIELD = GF2n(B, POLY)
assert FIELD.gen == 2, "gamma = 2 not primitive?!"
EXP, LOG = FIELD.exp, FIELD.log
gmul, ginv, gpow, gsqrt = FIELD.mul, FIELD.inv, FIELD.pow, FIELD.sqrt
# trace(x) is the absolute trace; trace(x, 3) the GF(8) trace of a GF(8) element.
trace = FIELD.trace


def bitlist(m, n):
    return [i for i in range(n) if (m >> i) & 1]


def bitmask(n, pred):
    return sum(1 << i for i in range(n) if pred(i))


def gf_matinv(Mx):
    """Inverse of a square matrix over GF(2^B) (Gauss-Jordan)."""
    n = len(Mx)
    A = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(Mx)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col])
        A[col], A[piv] = A[piv], A[col]
        iv = ginv(A[col][col])
        A[col] = [gmul(iv, v) for v in A[col]]
        for r in range(n):
            if r != col and A[r][col]:
                f = A[r][col]
                A[r] = [A[r][c] ^ gmul(f, A[col][c]) for c in range(2 * n)]
    return [row[n:] for row in A]


# =========================================================================
# A. Fixup vector c (contract.json; task README section 3.2:
# counter-mode SHA-256 XOF over c_tag, grader-asserted)
# =========================================================================
_buf = bytes.fromhex(CONTRACT["c_stream_hex"])
CBITS = [(_buf[t >> 3] >> (t & 7)) & 1 for t in range(M * B)]
CMASKJ = [bitmask(B, lambda bp: CBITS[B * j + bp]) for j in range(M)]

# =========================================================================
# B. Frozen multiplication formulas + tower row masks (constants.json
# beside circuit.py, written by gen_constants.py).
# =========================================================================
CONSTS = json.loads((HERE / "constants.json").read_bytes())
KT = [tuple(map(tuple, t)) for t in CONSTS["KT12"]]  # GF(2^12) mult: 42 products
KT6 = [tuple(map(tuple, t)) for t in CONSTS["KT6"]]  # GF(64) mult: 15 products
KT3 = [tuple(map(tuple, t)) for t in CONSTS["KT3"]]  # GF(8) mult: 6 products
# x^3 in GF(8): 3 products
CUBE3Q = tuple(map(tuple, CONSTS["CUBE3Q"][0])), tuple(CONSTS["CUBE3Q"][1])
PAIR3 = [tuple(pq) for pq in CONSTS["PAIR3"]]  # Tr3(r s^2) pairing: rank 3
LIN = CONSTS["LIN"]  # the tower's linear maps

# =========================================================================
# C. Omega schedule: in-place triangular product from middle-product
# blocks, each at its field-rank a+b-1 by evaluation-interpolation.
# Constant GF(2^B) maps are free CX networks, so evaluating both
# operands at a+b-1 fixed points and interpolating the results back
# costs only the pointwise quantum-quantum products; the recursion
# below (balanced binary) is the DP optimum of the alias-free schedule
# family: 147 products against the dense triangle's W(W+1)/2 = 630.
# Blocks read syn columns strictly below the rows they write, so the
# hi -> block -> lo emission order stays in-place.
# =========================================================================
_OMC = {1: 0}
_OMS = {1: None}
for _n in range(2, W + 1):
    _best = None
    _bh = None
    for _h in range(1, _n // 2 + 1):
        _v = _OMC[_h] + _OMC[_n - _h]
        if _best is None or _v < _best:
            _best, _bh = _v, _h
    assert _best is not None  # range(1, _n // 2 + 1) is non-empty
    _OMC[_n] = (_n - 1) + _best
    _OMS[_n] = _bh

MACS = []  # (r0, a, c0, b): reads syn cols [c0, c0+b), writes [r0, r0+a)


def _build_tri(lo, n):
    if n == 1:
        return
    h = _OMS[n]
    _build_tri(lo + h, n - h)
    MACS.append((lo + h, n - h, lo, h))
    _build_tri(lo, h)


# omega70: the same in-place transform extended to all N = 2W rows.  Rows
# W..N-1 of sigma_m*S mod z^N vanish on every correct decode (the key
# equation: deg omega < deg sigma <= W), which frees the 35-word syndrome
# TAIL as workspace for the Bezout pair (stage 5).  Emission
# order hi -> lo keeps it in place: the tail triangle (rows W..N-1 from
# rows W..N-2, diagonals 1..W-1), then one off-diagonal middle-product
# block (rows W..N-1 from rows 0..W-1; its Toeplitz diagonals W+1..N-1
# are zero -- sigma has W+1 coefficients -- and are clipped from the
# sigma-side evaluation), then the omega triangle.  The map is unit
# triangular on all N rows, so its literal reverse restores S on EVERY
# path, including approximate-BM paths where the tail is not zero.
_build_tri(W, W)
MACS.append((W, W, 0, W))
_build_tri(0, W)
assert len(MACS) == 2 * (W - 1) + 1
for _r0, _a, _c0, _b in MACS:
    # alias-free in place: every block reads rows strictly below the rows
    # it writes; the triangles span diagonals 1..W-1, the off-diagonal
    # block 1..N-1 (clipped to 1..W at emission)
    assert _c0 + _b == _r0, (_r0, _a, _c0, _b)
    assert _r0 + _a - 1 - _c0 <= (N - 1 if _b == W else W - 1), (_r0, _a, _c0, _b)

_VINV = {}


def vinv(npts):
    """(points, inverse Vandermonde) for evaluation at gamma^0..gamma^(npts-1),
    cached per size.
    """
    if npts not in _VINV:
        pts = [EXP[t] for t in range(npts)]
        _VINV[npts] = (
            pts,
            gf_matinv([[gpow(pts[t], k) for k in range(npts)] for t in range(npts)]),
        )
    return _VINV[npts]


@functools.cache
def cmrows(c):
    """Row masks of the constant multiplication x -> c*x (a free CX network)."""
    return f2_transpose([gmul(c, 1 << j) for j in range(B)], B)


# =========================================================================
# D. (2,2,3) tower: GF(2^12) = GF(64)(t) = GF(8)(s)(t), read top-down
# (gen_constants.py docstring).  A 12-bit tower word is a0 (low 6) + a1 (high 6) t with
# t^2 + t = nu, conjugate xbar = (a0 + a1) + a1 t; a GF(64) word is
# n0 (low 3) + n1 (high 3) s, conjugate Nbar = (n0 + n1) + n1 s.  The
# norm descends both quadratic levels (N = x xbar in GF(64) = a0 a1 +
# linear, M = N Nbar in GF(8) = n0 n1 + linear); the only nonlinear
# inverse left is M -> M^3 (a 3-product quadratic form), M^-1 = (M^3)^2
# a free squaring.  Coordinates: E12[i + 3j + 6k] = v^i s^j t^k; the
# GF(8) projection of a GF(64) word is a bit slice (no row parities).
# Every stage maps 0 to 0, so the zflag armor (monicize) carries over.
# local: gadgets.gf_tower.GF asserts q0 in {4, 8}; this tower is
# GF(8)(s)(t) and its in-circuit chain runs on A0/A1 row projections of
# 12-bit poly-basis words, never on tower-packed words.
# =========================================================================
E12 = LIN["E12"]  # the 12 tower basis elements (poly ints)
A0ROWS = LIN["A0ROWS"]  # poly(12) -> a0 (GF(64) coords)
A1ROWS = LIN["A1ROWS"]  # poly(12) -> a1
A01ROWS = [A0ROWS[r] ^ A1ROWS[r] for r in range(6)]  # a0 + a1 (xbar's constant part)
LNROWS = LIN["LNROWS"]  # poly(12) -> the linear part of N: a0^2 + nu a1^2
LMROWS = LIN["LMROWS"]  # GF(64) coords -> the linear part of M: n0^2 + nu' n1^2
SQ3R = LIN["SQ3R"]  # squaring on GF(8) coords
assert (len(E12), len(A0ROWS), len(A1ROWS), len(LNROWS), len(LMROWS), len(SQ3R)) == (
    12,
    6,
    6,
    6,
    3,
    3,
)
# the tower elements re-derived from the field (the recipe in the
# gen_constants.py docstring); E12 must be their products
_A_el = 2 ^ gpow(2, 64)
_t_el = gmul(2, ginv(_A_el))
_u_el = gpow(2, 65)
_Ap_el = _u_el ^ gpow(_u_el, 8)
_s_el = gmul(_u_el, ginv(_Ap_el))
_v_el = gpow(_u_el, 9)
assert E12 == [
    gmul(gmul(gpow(_v_el, i), gpow(_s_el, j)), gpow(_t_el, k))
    for k in range(2)
    for j in range(2)
    for i in range(3)
], "E12 != v^i s^j t^k"
MINVR = f2_matinv(f2_transpose(E12, B), B)
assert MINVR[0:6] == A0ROWS and MINVR[6:12] == A1ROWS, "A0/A1 rows != inverse basis matrix"

# =========================================================================
# E. Trace dual, free-Z interpolation split, GF(8) fixed pairing form
# =========================================================================
TRT = [trace(x) for x in range(1 << B)]  # absolute trace table
TINV = f2_matinv([bitmask(B, lambda c: TRT[gmul(1 << a, 1 << c)]) for a in range(B)], B)
MU = f2_transpose(TINV, B)
for _a in range(B):
    for _bb in range(B):
        assert TRT[gmul(1 << _a, MU[_bb])] == (1 if _a == _bb else 0)

LAMJ = [0] * M
for _j in range(M):
    _v = 0
    for _bpos in bitlist(CMASKJ[_j], B):
        _v ^= MU[_bpos]
    LAMJ[_j] = _v

# ---- move N sweep points into free Z gates (Vandermonde interpolation) --
KILLJ = [j for j in range(M) if LAMJ[j] == 0]
KILLJ += [j for j in range(M) if LAMJ[j] != 0][: N - len(KILLJ)]
assert len(KILLJ) == N and len(set(KILLJ)) == N
_tgt = [gmul(LAMJ[j], ginv(EXP[j])) for j in KILLJ]
_Ai = gf_matinv([[gpow(EXP[j], i) for i in range(N)] for j in KILLJ])
UCO = [0] * (N + 1)
for _r in range(N):
    _v = 0
    for _c in range(N):
        _v ^= gmul(_Ai[_r][_c], _tgt[_c])
    UCO[_r + 1] = _v


def U_eval(j):
    acc = 0
    for i in range(1, N + 1):
        if UCO[i]:
            acc ^= EXP[(LOG[UCO[i]] + i * j) % M]
    return acc


LAMJ2 = [LAMJ[j] ^ U_eval(j) for j in range(M)]
for _j in KILLJ:
    assert LAMJ2[_j] == 0, ("interpolation failed", _j)
NLIVE = sum(1 for j in range(M) if LAMJ2[j])
ZMASKI = [bitmask(B, lambda bb: TRT[gmul(UCO[i + 1], 1 << bb)]) for i in range(N)]


def _elt3(c):
    w = 0
    for i in bitlist(c, 3):
        w ^= gpow(_v_el, i)
    return w


def _tr3(x):
    t = trace(x, 3)
    assert t in (0, 1)
    return t


# the GF(8) pairing form B(r, s) = Tr3(r * s^2) as PAIR3 spells it: rank 3
for _r in range(8):
    for _s3 in range(8):
        _acc = 0
        for _p, _qm in PAIR3:
            _acc ^= parity(_r & _p) & parity(_s3 & _qm)
        assert _acc == _tr3(gmul(_elt3(_r), gmul(_elt3(_s3), _elt3(_s3)))), (
            "pairing form drifted",
            _r,
            _s3,
        )


def pair_rows(lam):
    """r_a(lam), a < B: parity(r_a & U) = Tr(e_a sqrt(lam U)), lam != 0."""
    assert lam
    sq = [gsqrt(gmul(lam, 1 << c)) for c in range(B)]
    return [bitmask(B, lambda c: TRT[gmul(1 << a_, sq[c])]) for a_ in range(B)]


# =========================================================================
# Constant linear maps of monicize and the Euclid pass
# =========================================================================
# MONB[k] = the (a, b) basis pairs whose product has bit k set
MONB = [[] for _ in range(B)]
for _a in range(B):
    for _b2 in range(B):
        _vv = gmul(1 << _a, 1 << _b2)
        for _k2 in range(B):
            if (_vv >> _k2) & 1:
                MONB[_k2].append((_a, _b2))
SQRT_ROWS = f2_transpose([gsqrt(1 << j) for j in range(B)], B)
SQUARE_ROWS = f2_transpose([gmul(1 << j, 1 << j) for j in range(B)], B)


# =========================================================================
# Truncation schedule
# =========================================================================
# Reversal-stack capacity (tagged slots; overflow shots form error
# class E2, the stack overflow that README.md describes).  Overflow rate
# by cap under the DQI weight law, measured under the derived windows
# (403.2M draws): cap 1 -> 1.02e-4 [1.01e-4, 1.03e-4], cap 2 -> 4.1e-7,
# cap 3 -> 0 seen; max pushes 3.
# Cap 1 saves STKW = 19 wires at the global peak (both slots materialize
# by round 4 and stay live through the sweep) and 19 Toffoli per push
# cascade forward and again in the reverse pass: 2,470 Toffoli and 19
# qubits against cap 2, a 1.5% lower TQ-cost.  The aggregate sign-error
# rate is ~1.4e-4 (E1 3.8e-5 + E2 1.02e-4), 7x under the 1e-3 allowance
# (cap 2: 3.8e-5, 26x).
STK_CAP = 1

# Per-round coefficient windows (MEASURED; constants.json bm_windows,
# derived by gen_constants.py from 403.2M draws under the DQI weight
# law).  IMAXV bounds the discrepancy convolution, KMSV the
# flag-controlled slot swaps, KMMV the update macs.  The rule: the
# smallest per-round windows such that every clipped slot has marginal
# mass < 4e-5; r=0 and r=N-1 are pinned by the circuit.  Events outside
# a window form error class E1, the cutoff crossings of README.md.
_WIN = CONSTS["bm_windows"]
IMAXV = list(_WIN["IMAXV"])
KMSV = list(_WIN["KMSV"])
KMMV = list(_WIN["KMMV"])
assert len(IMAXV) == len(KMSV) == len(KMMV) == N
for _r in range(N):
    # never wider than the reachable envelope, never narrower than
    # round 0's seed path needs
    assert IMAXV[_r] <= min(_r, W, (_r + 1) // 2 + 1)
    assert KMMV[_r] <= KMSV[_r] <= min(_r + 1, W, _r // 2 + 2)
assert IMAXV[N - 1] == W and KMSV[N - 1] == KMMV[N - 1] == W
assert KMSV[0] == 1 and KMMV[0] == 0

# Signed d counter: two's complement, wide enough for the all-zero
# syndrome's monotone climb (d = N-2 after the last increment) -- DWB-1
# magnitude bits + the sign bit dW[DWB-1].
DWB = (N - 2).bit_length() + 1
assert (1 << (DWB - 1)) - 1 >= N - 2
SGN = DWB - 1  # the sign bit index (pqc / ctl anchor)
DMAX = 1 << (DWB - 1)  # two's-complement range of dW
# Stack round tag: TAGB bits hold every pushing round 0..N-1.
TAGB = (N - 1).bit_length()
TAGMASK = (1 << TAGB) - 1
STKW = B + TAGB  # stack slot: B value lanes + the round tag
assert DWB == 8  # MBITS = dW[2:7] and search.c hard-code the 8-bit dW


def caps_schedule():
    """Exact reachable-(d, depth) DP over the d-automaton: d>=0 branches to
    d+1 (ctl=0, needs D==0) or -d (ctl=1); d<0 forces ctl=0 and a PUSH
    (pqc = dW[SGN]), d -> d+1.  r=N-1 pushes unconditionally.  CAPS[r] =
    max reachable stack depth AFTER a push at round r = the emitted
    cascade extent; CAPS[r]==0 => a push is unreachable => the cascade is
    elided.
    """
    states = {(0, 0)}
    caps = [0] * N
    for r in range(N):
        ns = set()
        mx = 0
        for d_, k_ in states:
            if r == N - 1:
                ns.add((d_, k_ + 1))
                mx = max(mx, k_ + 1)
            elif d_ < 0:
                ns.add((d_ + 1, k_ + 1))
                mx = max(mx, k_ + 1)
            else:
                ns.add((d_ + 1, k_))
                ns.add((-d_, k_))
        caps[r] = mx
        states = ns
    return caps


CAPS_FULL = caps_schedule()
assert max(CAPS_FULL) <= W, ("stack cap bound broken", CAPS_FULL)
assert CAPS_FULL[0] == 0 and CAPS_FULL[1] == 0
# reachable depth is W-1 before the unconditional r=N-1 push, W after
assert max(CAPS_FULL[: N - 1]) == W - 1 and CAPS_FULL[N - 1] == W
# CAP truncation: clamp every emitted cascade extent to STK_CAP.
# min(c, STK_CAP) == 0 iff c == 0, so the set of rounds carrying a cascade
# is unchanged; only extents shrink.  Overflow shots (depth would exceed
# STK_CAP) are the failure class E2 (README.md).  With
# STK_CAP = 1 every cascade is the single Dw <-> STK[0] Fredkin level.
CAPS = [min(c, STK_CAP) for c in CAPS_FULL]
assert max(CAPS) == STK_CAP

# Psi-route cofactor windows (cyclic intervals (lo, n) of the 18-word
# registers, array-index frame) per CR op of R: the measured union
# support, recorded only where the op acts (swap: flag = 1; rotation:
# c = 1; product: (q_lo, q_hi) != 0), over 400k weight-law draws plus the
# w = 0 / w = 1 anchors (constants.json psi_windows, searched by
# gen_constants.py).  A support outside a window only mis-sets the
# Psi value (a sign error on that shot): R^-1 mirrors R op for op, so the
# registers are restored exactly either way.
PSI_WIN_IV = {k: {int(j): tuple(v) for j, v in d.items()} for k, d in CONSTS["psi_windows"].items()}
assert sorted(PSI_WIN_IV) == ["mul", "rz", "swap"] and all(
    sorted(d) == list(range(NH)) for d in PSI_WIN_IV.values()
)
PSI_WIN = {(k, jj): v for k, d in PSI_WIN_IV.items() for jj, v in d.items()}


# =========================================================================
# Terms (Builder-free term-list construction)
# =========================================================================
# A term list is the build's unit of arithmetic: a list of normalized
# terms over hard wire ids,
#
#   ('L', ws, c, tws)             tws ^= parity(ws) ^ c      (free CX/X)
#   ('P', wa, ca, wb, cb, tws)    tws ^= (parity(wa) ^ ca) AND
#                                        (parity(wb) ^ cb)   (one CCX)
#
# built by folding the frozen formula tables (section B) through operand
# specs (one wire-set per operand bit).  Em emits a term list and
# erases it.  Everything here is pure construction: no gate is emitted.


def spec_plain(ws):
    return [frozenset((w,)) for w in ws]


def spec_rows(rows, ws):
    return [
        frozenset(ws[j] for j in range(len(ws)) if (rows[r] >> j) & 1) for r in range(len(rows))
    ]


def fold(spec, bits_):
    ws = frozenset()
    for i in bits_:
        ws = ws ^ spec[i]
    return tuple(sorted(ws))


def tmask(Tb):
    """Return a formula product's target index set as a B-bit mask."""
    u = 0
    for jj in Tb:
        u ^= 1 << jj
    return u


def xfold(pairs):
    """Operand spec bit from (word, row mask) pairs: the XOR (not union)
    of the selected wires, so repeated wires cancel.
    """
    s = set()
    for word, row in pairs:
        for j in bitlist(row, B):
            s ^= {word[j]}
    return frozenset(s)


def norm_term(wa, ca, wb, cb, tws):
    if (not wa) and ca == 0:
        return None
    if (not wb) and cb == 0:
        return None
    if not wa:
        return ("L", wb, cb, tws)
    if not wb:
        return ("L", wa, ca, tws)
    if wa == wb:
        if ca == cb:
            return ("L", wa, ca, tws)
        return None
    return ("P", wa, ca, wb, cb, tws)


def bil_terms(rt, table, specA, specB, tgt=None, tws_of=None):
    """Bilinear terms of one frozen formula table over operand specs:
    fold each product's operand bit-sets through the specs, normalize,
    append.  Targets come from tgt (indexed by the table's target set)
    or a custom tws_of(Tb).
    """
    for Ab, Bb, Tb in table:
        tws = tws_of(Tb) if tws_of else [tgt[jj] for jj in Tb]  # ty: ignore[not-subscriptable]  # tgt is given when tws_of is not
        tm = norm_term(fold(specA, Ab), 0, fold(specB, Bb), 0, tws)
        if tm:
            rt.append(tm)


def kt6_terms_poly(rt, specA, specB, tgt12, off):
    """KT6 terms whose GF(64)-coordinate targets land in the 12-bit poly
    word tgt12 through the tower basis: coord j -> E12[j + off].
    """

    def tws_of(Tb):
        pm = 0
        for u in Tb:
            pm ^= E12[u + off]
        return [tgt12[jj] for jj in bitlist(pm, B)]

    bil_terms(rt, KT6, specA, specB, tws_of=tws_of)


def quad_terms(rt, Qf, Xs, tgt, n):
    tris, corr = Qf
    sp = spec_plain(Xs)
    for u, w, om in tris:
        tm = norm_term(
            fold(sp, bitlist(u, n)), 0, fold(sp, bitlist(w, n)), 0, [tgt[m] for m in bitlist(om, n)]
        )
        if tm:
            rt.append(tm)
    for m in range(n):
        if corr[m]:
            rt.append(("L", tuple(sorted(Xs[i] for i in bitlist(corr[m], n))), 0, (tgt[m],)))


def lin_rows_terms(rt, rows, src, tgt):
    """Free linear rows: tgt[r] ^= parity(rows[r] & src)."""
    for rr in range(len(rows)):
        if rows[rr]:
            rt.append(
                ("L", tuple(sorted(src[jj] for jj in bitlist(rows[rr], len(src)))), 0, (tgt[rr],))
            )


def prod_rt(tgt, wa, wb):
    """tgt ^= wa * wb (KT12, 42 products) as a term list."""
    rt = []
    bil_terms(rt, KT, spec_plain(wa), spec_plain(wb), tgt)
    return rt


def one_rt(wa, ca, wb, cb, t):
    """t ^= (wa ^ ca) AND (wb ^ cb) over wire tuples as a one-term list."""
    return [norm_term(tuple(sorted(wa)), ca, tuple(sorted(wb)), cb, [t])]


# ---- AND ladders ----------------------------------------------------------
def ladder_rt(rungs, tree):
    """AND-ladder term lists over rungs [(wire tuple, polarity), ...]:
    tree[i-1] <- AND of the first i+1 rungs, each parity tested against
    its polarity bit (len(rungs) - 1 products).
    """
    (w0, c0), (w1, c1) = rungs[0], rungs[1]
    rts = [one_rt(w0, c0, w1, c1, tree[0])]
    for i in range(2, len(rungs)):
        ws, cc = rungs[i]
        rts.append(one_rt((tree[i - 2],), 0, ws, cc, tree[i - 1]))
    return rts


def eq_rungs(bits, val):
    """Rungs whose ladder top is [bits == val] (val = 0: the zero test)."""
    return [((w,), 1 - ((val >> i) & 1)) for i, w in enumerate(bits)]


# ---- the tower inversion chain -------------------------------------------
CHAIN_WINS = [("nqd", 6), ("m3d", 3), ("s3d", 3), ("yd", 6)]


def norm_chain_terms(src, a0, a1, nq, m3, s3):
    """Term lists for the norm chain on src: N (KT6 bilinear + linear
    rows into nq), M (KT3 on the halves of nq + linear rows into m3),
    M^3 (CUBE3Q into s3).
    """
    rtN = []
    bil_terms(rtN, KT6, a0, a1, nq)
    lin_rows_terms(rtN, LNROWS, src, nq)
    rtM = []
    bil_terms(rtM, KT3, spec_plain(nq[0:3]), spec_plain(nq[3:6]), m3)
    lin_rows_terms(rtM, LMROWS, nq, m3)
    rtS = []
    quad_terms(rtS, CUBE3Q, m3, s3, 3)
    return rtN, rtM, rtS


def chain_rts(src, wins, out):
    """Forward tower inversion-chain term lists for out = inv(src): the
    norm chain, N^-1 = Nbar M^-1 (two KT3 into yd, M^-1 = (M^3)^2 folded
    into the operand spec as SQ3R rows), then x^-1 = xbar N^-1 (two KT6)
    recombined into the 12-bit poly-basis out.  wins are the CHAIN_WINS
    scratch words; rts[:-1] write wins in order, rts[-1] writes out.
    """
    nq, m3, s3, yd = wins
    a0 = spec_rows(A0ROWS, src)
    a1 = spec_rows(A1ROWS, src)
    a01 = spec_rows(A01ROWS, src)
    rtN, rtM, rtS = norm_chain_terms(src, a0, a1, nq, m3, s3)
    m6 = spec_rows(SQ3R, s3)
    n0 = spec_plain(nq[0:3])
    n1 = spec_plain(nq[3:6])
    n01 = [n0[i] ^ n1[i] for i in range(3)]
    rtY = []
    bil_terms(rtY, KT3, n01, m6, yd[0:3])
    bil_terms(rtY, KT3, n1, m6, yd[3:6])
    y = spec_plain(yd)
    rtI = []
    kt6_terms_poly(rtI, a01, y, out, 0)
    kt6_terms_poly(rtI, a1, y, out, 6)
    return [rtN, rtM, rtS, rtY, rtI]


# ---- evaluation/interpolation products -----------------------------------
def polyprod_rt(Qw, Fw, tgt):
    """Evaluation/interpolation product: for the polynomials Q(y) =
    sum_i Qw[i] y^i and F(y) = sum_k Fw[k] y^k (lists of words), XOR
    coefficient n of Q*F into every word in tgt[n] (a list; empty = drop).
    Costs len(Qw) + len(Fw) - 1 products (one scalar product when both
    have one word); operand evaluations and the interpolation rows are
    free CX networks folded into the operand specs and target sets.
    """
    npts = len(Qw) + len(Fw) - 1
    assert len(tgt) == npts
    if npts == 1:
        rt = []
        for T in tgt[0]:
            bil_terms(rt, KT, spec_plain(Qw[0]), spec_plain(Fw[0]), T)
        return rt
    pts, Iv = vinv(npts)
    rt = []
    for t in range(npts):
        cq = [cmrows(gpow(pts[t], i)) for i in range(len(Qw))]
        cf = [cmrows(gpow(pts[t], k)) for k in range(len(Fw))]
        sA = [xfold([(Qw[i], cq[i][bb]) for i in range(len(Qw))]) for bb in range(B)]
        sB = [xfold([(Fw[k], cf[k][bb]) for k in range(len(Fw))]) for bb in range(B)]

        def tws_of(Tb, t=t):
            u = tmask(Tb)
            acc = {}
            for n in range(npts):
                if not tgt[n]:
                    continue
                v = gmul(Iv[n][t], u)
                for T in tgt[n]:
                    for mm in bitlist(v, B):
                        acc[T[mm]] = acc.get(T[mm], 0) ^ 1
            return [w for w in sorted(acc) if acc[w]]

        bil_terms(rt, KT, sA, sB, tws_of=tws_of)
    return [tm for tm in rt if tm[-1]]


def omega_rts(sig, syn):
    """Return the in-place omega70 block term lists (MACS order) for the
    monic sigma bank sig (words sig[1..W]) and the syndrome words syn.
    Each block folds its evaluation maps into the operand parity specs
    (sigma side over the bank, S side over the rows it reads) and its
    interpolation rows into the KT target sets, so the a+b-1 pointwise
    products are the only Toffoli gates.
    """
    out = []
    for r0, a_, c0, b_ in MACS:
        npts = a_ + b_ - 1
        pts, Iv = vinv(npts)
        rt = []
        for t in range(npts):
            cmA = [cmrows(Iv[k][t]) for k in range(npts)]
            cmB = [cmrows(gpow(pts[t], j)) for j in range(b_)]
            # diagonals beyond W are zero (clipped): sigma has W+1 coefficients
            sA = [
                frozenset(
                    sig[1 + k][j] for k in range(min(npts, W)) for j in bitlist(cmA[k][bb], B)
                )
                for bb in range(B)
            ]
            sB = [
                frozenset(
                    syn[c0 + b_ - 1 - j][jb] for j in range(b_) for jb in bitlist(cmB[j][bb], B)
                )
                for bb in range(B)
            ]
            wcoef = [gpow(pts[t], i2) for i2 in range(a_)]

            def om_tws(Tb, wcoef=wcoef, r0=r0, a_=a_):
                u = tmask(Tb)
                tws = []
                for i2 in range(a_):
                    tws.extend(syn[r0 + i2][m] for m in bitlist(gmul(wcoef[i2], u), B))
                return tws

            bil_terms(rt, KT, sA, sB, tws_of=om_tws)
        out.append(rt)
    return out


# =========================================================================
# Emission (term-list emitter, journaled vocabulary, allocators)
# =========================================================================
# Em emits the term lists of the Terms section through a live Builder: emit_mac
# writes a term list forward (one CCX per product; a shared multi-target
# product fans out through the sT scratch and erases it for free), and
# emit_free erases a written term list by demolition measurement + phase
# fixups, 0 Toffoli.  The journaled vocabulary (mac / xor / free / x /
# cx / cswap / swap / swaps, and the ladder, chain and increment helpers
# built on them) runs each step now and records its conjugate in a
# uncompute.bennett.Journal, so a later Journal.reverse() replays the
# conjugates over the same hard wire ids.
#
# Scratch comes from Em's current allocator, an Alloc: it round-trips
# windows through the Builder allocator and logs the window ids a journal
# touches, so a replay can hold exactly those ids (Alloc.replay).


def word_pairs(xs, ys):
    """Bitwise wire pairs of two word lists: word by word, bit 0 first."""
    return [(x[t], y[t]) for x, y in zip(xs, ys) for t in range(B)]


# NOJ never records: unjournaled emission through the journaled vocabulary.
NOJ = Journal()
NOJ.recording = False


class Em:
    """Term-list emitter over the held pivots sA, sB and the fan-out
    scratch sT, with the journaled vocabulary.  The vocabulary records
    into the current journal J (NOJ outside any using() block), and the
    scratch helpers draw from the current allocator A.
    """

    def __init__(self, b, sA, sB, sT, A):
        self.b = b
        self.sA = sA
        self.sB = sB
        self.sT = sT
        self.J = NOJ
        self.A = A

    @contextmanager
    def using(self, J=None, A=None):
        """Make J and/or A the current journal / allocator inside the block."""
        prev = self.J, self.A
        if J is not None:
            self.J = J
        if A is not None:
            self.A = A
        yield
        self.J, self.A = prev

    # ---- the unjournaled core --------------------------------------------
    def _build(self, ws, cc, s):
        if len(ws) == 1 and cc == 0:
            return ws[0], False
        for w in ws:
            self.b.cx(w, s)
        if cc:
            self.b.x(s)
        return s, True

    def _unbuild(self, ws, cc, s, built):
        if not built:
            return
        if cc:
            self.b.x(s)
        for w in reversed(ws):
            self.b.cx(w, s)

    def pivot_pair(self, wa, ca, wb, cb, mid):
        """Emit the shared pivot-operand sandwich: build both pivot operands
        into sA/sB, run the caller's middle 2-qubit op on the built
        pivots (ua, ub), then unbuild in reverse.  One core for the
        single-target CCX in emit_mac, the conditioned-CZ repayment in
        emit_free, and fan_pair's full sT window.
        """
        ua, ba = self._build(wa, ca, self.sA)
        ub, bb2 = self._build(wb, cb, self.sB)
        mid(ua, ub)
        self._unbuild(wb, cb, self.sB, bb2)
        self._unbuild(wa, ca, self.sA, ba)

    def fan_pair(self, wa, ca, wb, cb, mid):
        """Emit the product-pair skeleton over pivot_pair: CCX the pivots'
        AND into the sT fan-out scratch, run the caller's mid(sT) on it,
        then measure sT away with its phase repayment.
        """
        b = self.b

        def sandwich(ua, ub):
            b.ccx(ua, ub, self.sT)
            mid(self.sT)
            m = b.hmr(self.sT)
            b.cz(ua, ub, cond=m)

        self.pivot_pair(wa, ca, wb, cb, sandwich)

    def emit_mac(self, rt):
        b = self.b
        for t in rt:
            k = t[0]
            if k == "L":
                ws, cc, tws = t[1], t[2], t[3]
                for tw in tws:
                    for w in ws:
                        b.cx(w, tw)
                    if cc:
                        b.x(tw)
            else:
                wa, ca, wb, cb, tws = t[1], t[2], t[3], t[4], t[5]
                if len(tws) == 1:
                    self.pivot_pair(wa, ca, wb, cb, lambda ua, ub, tw=tws[0]: b.ccx(ua, ub, tw))
                else:

                    def mid(fan, tws=tws):
                        for tw in tws:
                            b.cx(fan, tw)

                    self.fan_pair(wa, ca, wb, cb, mid)

    def emit_free(self, rt):
        b = self.b
        per = {}
        order = []
        for t in rt:
            for tw in t[-1]:
                if tw not in per:
                    per[tw] = []
                    order.append(tw)
                per[tw].append(t)
        for tw in reversed(order):
            m = b.hmr(tw)
            for t in per[tw]:
                k = t[0]
                if k == "L":
                    for w in t[1]:
                        b.z(w, cond=m)
                    if t[2]:
                        b.neg(cond=m)
                else:
                    self.pivot_pair(t[1], t[2], t[3], t[4], lambda ua, ub: b.cz(ua, ub, cond=m))

    def swap_fn(self, c, pairs):
        """One closure applying cswap(c, x, y) over pairs in order (c None:
        plain swaps).
        """
        b = self.b
        if c is None:

            def fn():
                for x, y in pairs:
                    b.swap(x, y)
        else:

            def fn():
                for x, y in pairs:
                    b.cswap(c, x, y)

        return fn

    # ---- the journaled vocabulary ----------------------------------------
    def mac(self, rt):
        """Write a term list into ZERO targets; the conjugate erases it
        for free.
        """
        self.J.step(lambda: self.emit_mac(rt), lambda: self.emit_free(rt))
        return rt

    def xor(self, rt):
        """XOR-accumulate a term list into live targets: self-inverse."""
        self.J.step(lambda: self.emit_mac(rt))
        return rt

    def free(self, rt):
        """Measurement-erase targets that equal the term list (a claim);
        the conjugate recomputes them.
        """
        self.J.step(lambda: self.emit_free(rt), lambda: self.emit_mac(rt))

    def macs(self, rts):
        for rt in rts:
            self.mac(rt)

    def frees(self, rts):
        """Erase term lists in reverse order."""
        for rt in reversed(rts):
            self.free(rt)

    def x(self, q):
        self.J.step(lambda: self.b.x(q))

    def cx(self, c, t):
        self.J.step(lambda: self.b.cx(c, t))

    def cswap(self, c, x, y):
        self.J.step(lambda: self.b.cswap(c, x, y))

    def swap(self, x, y):
        self.J.step(lambda: self.b.swap(x, y))

    def swaps(self, c, pairs):
        """cswap(c, .) (c None: swap) over pairs as ONE journal step."""
        self.J.step(self.swap_fn(c, pairs))

    def ladder(self, rungs, tree):
        """Emit an AND ladder (ladder_rt) and return its term
        lists; the caller consumes the top and erases with frees().
        """
        rts = ladder_rt(rungs, tree)
        self.macs(rts)
        return rts

    def inc1(self, x, tree):
        """x += 1 (mod 2^len(x)) as a constant ripple: len(x) - 2 CCX
        ladder carries into tree (len(x) - 2 zero wires), each erased by
        measurement right after its use.
        """
        crts = self.ladder([((x[i],), 0) for i in range(len(x) - 1)], tree)
        for j in range(len(x) - 1, 1, -1):
            self.cx(tree[j - 2], x[j])
            self.free(crts[j - 2])
        self.cx(x[0], x[1])
        self.x(x[0])


def chain_open(E, src, out):
    """Open the inversion-chain scratch on E's allocator and build the
    chain term lists for out = inv(src) (chain_rts).
    """
    wins = [E.A.open(name, n) for (name, n) in CHAIN_WINS]
    return wins, chain_rts(src, wins, out)


def chain_close(E, wins):
    for (name, _n), win in zip(reversed(CHAIN_WINS), reversed(wins)):
        E.A.close(name, win)


def chain_fwd(E, src, out):
    wins, rts = chain_open(E, src, out)
    E.macs(rts)
    return wins, rts


def chain_undo(E, wins, rts):
    E.frees(rts)
    chain_close(E, wins)


@contextmanager
def chain_live(E, src, out):
    """out <- inv(src) with the chain intermediates live inside the block;
    exit erases out and the intermediates.
    """
    wins, rts = chain_fwd(E, src, out)
    yield
    chain_undo(E, wins, rts)


@contextmanager
def opened(E, name, n):
    """Open an n-wire scratch window on E's allocator; exit closes it."""
    qs = E.A.open(name, n)
    yield qs
    E.A.close(name, qs)


@contextmanager
def eq_test(E, bits, val):
    """Yield the top wire of a [bits == val] ladder; exit
    erases the ladder.
    """
    with opened(E, "tree", len(bits) - 1) as tr:
        rts = E.ladder(eq_rungs(bits, val), tr)
        yield tr[len(bits) - 2]
        E.frees(rts)


class Alloc:
    """Scratch windows through the Builder allocator: open allocs and
    claims a window, close releases it and frees it in reverse alloc
    order.  Journals replay closures over hard wire ids, so open records
    each window id into the current log set (logging()), and replay()
    re-materializes exactly a logged set via reclaim before a replay.
    """

    def __init__(self, b):
        self.b = b
        self.log = None  # the set open records window ids into, if any
        self.round = None  # BM round tag for bank bookkeeping
        self.bank_log = {}  # round -> bank registers first touched there

    def open(self, name, n):
        qs = self.b.qubits(n)
        if self.log is not None:
            self.log.update(qs)
        self.b.claim(name, qs)
        return qs

    def reopen(self, name, qs):
        """Re-acquire these exact (free) ids as a window."""
        self.b.reclaim(qs)
        self.b.claim(name, qs)

    def close(self, name, qs):
        self.b.release_recycle(name, qs)

    @contextmanager
    def logging(self, ids):
        """Record every window id opened inside the block into the set ids."""
        prev, self.log = self.log, ids
        yield
        self.log = prev

    @contextmanager
    def replay(self, ids, name="jscratch"):
        """Hold exactly the (free) ids inside the block, for a journal replay."""
        pool = sorted(ids)
        if pool:
            self.reopen(name, pool)
        yield
        if pool:
            self.close(name, pool)


class BankReg:
    """One lazy bank register: B wires materialized at the first operand
    touch (allocated, claimed into the bank's set, and logged with the
    touch round).  The fused-Q rotation relabels handles for free, so a
    physical slot's first touch is discovered mechanically instead of
    derived from the clipped-window schedule.  The mirror replay of the
    touch round returns the slot to |0> (post-replay state equals the
    forward state before that round), which is where the reverse pass
    frees it.
    """

    __slots__ = ("A", "name", "round", "width", "wires")

    def __init__(self, A, name, width=B):
        self.A = A
        self.name = name
        self.width = width
        self.wires = None
        self.round = None

    def mat(self):
        if self.wires is None:
            A = self.A
            self.wires = A.b.qubits(self.width)
            A.b.claim(self.name, self.wires)
            self.round = A.round
            A.bank_log.setdefault(A.round, []).append(self)
        return self.wires

    def __getitem__(self, k):
        return self.mat()[k]

    def __iter__(self):
        return iter(self.mat())

    def __len__(self):
        return self.width


# =========================================================================
# Psi route (half-degree Euclid F, R, cofactor recurrence)
# =========================================================================
# README.md, "Division-free Forney recovery".  The half-degree Euclid
# pass F compresses the padded locator halves (E', O') into quotients;
# R replays F backwards (restoring E', O') while the cofactor recurrence
# CR builds the Bezout pair (a, b), A E' + B O' = z^m, so
# Psi = B^2 + z A^2 = 1/sigma_m' at every root.  Every step runs
# through a uncompute.bennett.Journal over hard wire ids: F records into its
# journal JF; R is a deferred journal that replays JF's block segments
# backwards (interleaved with the CR steps), R^-1 replays them forwards,
# and F^-1 replays all of JF backwards.  The allocator logs every scratch
# id these journals touch (PSI_IDS), and R and R^-1..F^-1 hold exactly that
# set, so every replay reuses exactly the forward wire ids.  gen_constants.py holds the bit-exact
# classical model of these gadgets, op for op.


def flag_nz(E, word, q):
    """q ^= [word != 0]: a zero-test ladder, then NOT top into q (free:
    CX + X, one journal step).
    """
    b = E.b
    with eq_test(E, word, 0) as top:

        def fn():
            b.cx(top, q)
            b.x(q)

        E.J.step(fn)


@contextmanager
def inv_word(E, src):
    """Yield out = inv(src) (66 Toffoli), chain intermediates erased for
    free; exit erases out (36 Toffoli: the intermediates are recomputed).
    """
    with opened(E, "inv", B) as out:
        wins, rts = chain_fwd(E, src, out)
        chain_undo(E, wins, rts[:-1])
        yield out
        # the recompute runs rts built on the chain's exact ids
        for (name, _n), w in zip(CHAIN_WINS, wins):
            E.A.reopen(name, w)
        E.macs(rts[:-1])
        E.free(rts[-1])
        E.frees(rts[:-1])
        chain_close(E, wins)


def pad_barrel(E, mbits, ew, ow):
    """Shift both halves up by m = the 5-bit register mbits (non-cyclic:
    the barrel of cswap chains, one per set bit of m).
    """
    for sft in (16, 8, 4, 2, 1):
        ctl = mbits[sft.bit_length() - 1]
        for X in (ew, ow):
            for k in range(NH - 1 - sft, -1, -1):
                E.swaps(ctl, word_pairs([X[k]], [X[k + sft]]))


def psi_pre(E, ew, ow, pre):
    """pre <- [E'_17 == 0]; cswap the halves on pre (odd L with
    sigma_{L-1} = 0).  pre stays live until the literal reverse.
    """
    with eq_test(E, ew[17], 0) as top:
        E.cx(top, pre)
    E.swaps(pre, word_pairs(ew, ow))


def psi_forward(E, f, g, eps, idx):
    """F on the Euclid bank (f, g: lists of 18 words, top-aligned, f[17]
    the pivot): round 1, the block-0 zero-pivot repair, then 17 blocks
    s0..s9.  Mirrors gen_constants.py forward() op for op.  Every free
    erase claim holds whenever f17 != 0 at the elimination, i.e. on every
    correct decode (the pre-swap and padding guarantee deg E' = 17) and on
    every garbage locator the model has seen.  Returns the final (f, g)
    word arrays and the per-block records that build_R consumes: the
    flag / c / q wires and the journal marks where R interleaves the
    cofactor ops (start; swap, after the flag cswap; q, quotients ready;
    c, c set; end).
    """
    f, g = list(f), list(g)
    recs = {}
    # ---- round 1: q1 = g17/f17; g -= q1 f; store q1 in g17's word ----
    mk = {"start": E.J.mark(), "swap": E.J.mark()}  # block 0 swaps on pre, before F
    with inv_word(E, f[17]) as fi:
        with opened(E, "q1", B) as q1:
            E.mac(prod_rt(q1, g[17], fi))
            E.free(prod_rt(g[17], q1, f[17]))  # claim g17 == q1 f17
            E.swaps(None, word_pairs([q1], [g[17]]))  # g17's word <- q1
        mk["q"] = E.J.mark()  # q1 lives in g[17]'s word
        q1w = g[17]
        E.xor([op for k in range(17) for op in prod_rt(g[k], q1w, f[k])])
    g = g[17:] + g[:17]  # rot_up (relabel)
    # ---- block 0: zero-pivot test + repair (degree defect after round 1)
    with eq_test(E, g[17], 0) as top:
        E.cx(top, eps)  # eps <- [g17 == 0]
    mk["c"] = E.J.mark()
    for k in range(17, 1, -1):  # rot_up(g, lo=1) on eps
        E.swaps(eps, word_pairs([g[k]], [g[k - 1]]))
    mk["end"] = E.J.mark()
    recs[0] = {"marks": mk, "c": eps, "q": [q1w]}
    for j in range(1, 18):
        mk = {"start": E.J.mark()}
        f[j - 1], g[j - 1] = g[j - 1], f[j - 1]  # boundary swap (relabel)
        with opened(E, "flag", 1) as (flag,):
            # s0: set flag to [g17 != 0]
            flag_nz(E, g[17], flag)
            # s1: cswap window [j, 17] on flag
            E.swaps(flag, word_pairs([f[k] for k in range(j, 18)], [g[k] for k in range(j, 18)]))
            mk["swap"] = E.J.mark()
            # s2: erase flag against [g17 != 0] (post-swap)
            flag_nz(E, g[17], flag)
        # s3: invert f17 into fi; s9 (the with exit) erases it
        with inv_word(E, f[17]) as fi:
            # s4: q_hi, t, q_lo
            with opened(E, "qh", B) as qh, opened(E, "ql", B) as ql:
                E.mac(prod_rt(qh, g[17], fi))
                if j <= 16:
                    with opened(E, "t", B) as t:
                        trt = E.mac(prod_rt(t, qh, f[16]))
                        rtl = []
                        bil_terms(
                            rtl,
                            KT,
                            [frozenset((g[16][i], t[i])) for i in range(B)],
                            spec_plain(fi),
                            ql,
                        )
                        E.mac(rtl)
                        E.free(trt)
                else:
                    E.mac(prod_rt(ql, g[16], fi))
                mk["q"] = E.J.mark()
                # s5: block elimination: target word g[j-1+n] ^= coef_n of
                # (ql + qh y) * (f[j] + f[j+1] y + ... + f[16] y^(16-j)), n = 0..17-j
                if j <= 16:
                    E.xor(
                        polyprod_rt(
                            [ql, qh],
                            [f[k] for k in range(j, 17)],
                            [[g[j - 1 + n]] for n in range(18 - j)],
                        ),
                    )
                E.free(prod_rt(g[17], qh, f[17]))  # claim g17 == qh f17
                E.free(prod_rt(g[16], ql, f[17]))  # claim g16' == ql f17
                E.swaps(None, word_pairs([qh, ql], [g[17], g[16]]))
            g = g[16:] + g[:16]  # rot_up twice (relabel)
            with opened(E, "c", 1) as (c,):
                # s6: c = [g17 == 0] AND NOT eps
                with eq_test(E, g[17], 0) as top:
                    E.mac(one_rt((top,), 0, (eps,), 1, c))
                mk["c"] = E.J.mark()
                # s7: controlled rot_up(g, lo=j+1) on c; eps ^= c; idx ^= c*j
                for k in range(17, j + 1, -1):
                    E.swaps(c, word_pairs([g[k]], [g[k - 1]]))
                E.cx(c, eps)
                for bp in bitlist(j, 5):
                    E.cx(c, idx[bp])
                # s8: erase c == eps AND [idx == j]
                with eq_test(E, idx, j) as top:
                    E.free(one_rt((top,), 0, (eps,), 0, c))
        mk["end"] = E.J.mark()
        recs[j] = {"marks": mk, "flag": flag, "c": c, "q": [ql, qh]}
    return f, g, recs


def build_R(E, JF, recs, A, Bw, pre, win, eb):
    """R as a deferred Journal: JF's block segments replayed backwards,
    interleaved with the cofactor recurrence CR on the 18-word cyclic
    registers A (a) and Bw (b).  win maps ('mul'|'rz'|'swap', j) to the
    measured union support (lo, n) of the register(s) the op touches
    (array index frame).  R.forward() runs R; R.reverse() runs R^-1.
    Returns (R, a, b), the cofactor word lists in their final labels.
    """
    a, bb = list(A), list(Bw)
    R = Journal()

    def seg_rev(x, y):
        R.defer(lambda: JF.reverse(x, y), lambda: JF.forward(x, y))

    def cr_rz_ctl(c, j):
        # controlled cyclic rotation up by one (new ws[k] = old ws[k-1])
        # of the window words: len-1 word cswaps
        lo, n = win[("rz", j)]
        if n == 0:
            return
        ws = [bb[(lo + i) % NH] for i in range(min(n + 1, NH))]
        hi = [ws[k] for k in range(len(ws) - 1, 0, -1)]
        low = [ws[k - 1] for k in range(len(ws) - 1, 0, -1)]
        R.defer(E.swap_fn(c, word_pairs(hi, low)), E.swap_fn(c, word_pairs(hi[::-1], low[::-1])))

    def cr_mul(qws, j):
        # b <- z^len(qws) b (relabel); a ^= (sum_i q_i z^(i+1)) b_old,
        # with qws listed low -> high (ql, qh for a block; [q1] for round 1)
        b0 = list(bb)
        lo, n = win[("mul", j)]
        if n:
            Fw = [b0[(lo + i) % NH] for i in range(n)]
            tgt = [[a[(lo + k + 1) % NH]] for k in range(len(qws) + n - 1)]
            rt = polyprod_rt(list(qws), Fw, tgt)
            R.defer(lambda: E.emit_mac(rt))
        return b0[-len(qws) :] + b0[: -len(qws)]

    def cr_swap(c, j):
        lo, n = win[("swap", j)]
        ks = [(lo + i) % NH for i in range(n)]
        R.defer(E.swap_fn(c, word_pairs([a[k] for k in ks], [bb[k] for k in ks])))

    for j in range(17, -1, -1):
        r = recs[j]
        mk = r["marks"]
        seg_rev(mk["c"], mk["end"])
        cr_rz_ctl(r["c"], j)
        seg_rev(mk["q"], mk["c"])
        bb = cr_mul(r["q"], j)
        seg_rev(mk["swap"], mk["q"])
        cr_swap(pre if j == 0 else r["flag"], j)
        seg_rev(mk["start"], mk["swap"])
    # undo the pre-swap of the halves too: the sweep's root test needs E'
    # in the E words (psi_pre's swap itself stays live until the literal
    # reverse, which redoes this swap first)
    ew, ow = eb
    R.defer(E.swap_fn(pre, word_pairs(ew, ow)))
    return R, a, bb


def psi_seed(E, h, mbits, A):
    """A[(m+1) % 18] ^= inv(h), m = the 5-bit register mbits (0..17):
    one [m == v] ladder (4 Toffoli) and one controlled 12-bit copy per v
    (12 Toffoli, paid again by the literal reverse).
    """
    with inv_word(E, h) as ih:
        for v in range(NH):
            with eq_test(E, mbits, v) as top:
                # XOR-accumulate, recomputed on the reverse: the TAIL word is zero
                # only on a correct decode, so a measurement erase here would
                # break syndrome restoration on approximate-BM paths
                E.xor([norm_term((top,), 0, (ih[t],), 0, [A[(v + 1) % NH][t]]) for t in range(B)])


# =========================================================================
# Sweep (Chien/Forney root test + Psi pairing, one gated repeat body)
# =========================================================================
# Per point x = gamma^t (error position j = M - t): with E'_k, b_k read as
# if rotated by gamma^(kt) and O'_k, a_k by gamma^((k+SQRT_EXP)t) (parity sets
# over the unrotated words; no rotation is emitted),
#   XOR of the 36 EB words = E'(x) + sqrt(x) O'(x) = x^m sqrt(sigma_m(x)),
#   XOR of the 36 (a, b) words = V = B(x) + sqrt(x) A(x),  Psi(x) = V^2,
# so flag = [XOR of EB == 0] (11 Toffoli) and the Forney bit
# Tr(lam omega(x) / sigma_m'(x)) = Tr(lam omega(x) Psi(x)) = Tr(sqrt(lam U) V)
# = sum_a parity(r_a(lam) . U) V_a (U = omega(x), loaded into som): 12
# flag-gated rank-1 terms, one Toffoli each.  23 Toffoli per point.
#
# Loop form: the words never move, so every CX of every point acts on
# fixed wires; only WHICH stored bits a fan-in reads changes with t.  At
# point t, word k would hold g_k^t times its stored value, so bit bb of the
# rotated word is the parity of the stored bits selected by row bb of
# cmrows(g_k^t).  The body emits the union of each fan-in's possible bits
# in the per-point emission order, each CX data-gated on its membership
# bit at t, so a point's executed ops are exactly its per-point spelling
# (the Em core's pivot_pair / fan_pair / emit_free order).  Every other op
# is gated on the point being live (lam != 0).  A pairing row r_a(lam)
# with a single bit uses that som wire directly (no sA build), so the
# pairing CCX/CZ appear once per case (built sA, or som[c]), each gated on
# its case.

LOG_EB = list(range(NH)) + [(k + SQRT_EXP) % M for k in range(NH)]
LOG_AB = [(k + SQRT_EXP) % M for k in range(NH)] + list(range(NH))
CM_ARR = np.array(
    [[0] * B] + [cmrows(EXP[e]) for e in range(M)], dtype=np.int64
)  # CM_ARR[1 + e, bb]: cmrows(gamma^e)


def rot_member(logs):
    """mem[bb, k, j, t]: bit j of word k is in operand group bb at t."""
    t = np.arange(M, dtype=np.int64)
    mem = np.zeros((B, len(logs), B, M), dtype=bool)
    for k, lg in enumerate(logs):
        rows = CM_ARR[1 + (lg * t) % M]  # [t, bb]
        for j in range(B):
            mem[:, k, j, :] = ((rows >> j) & 1).T.astype(bool)
    return mem


LIVE = np.array([bool(LAMJ2[(M - t) % M]) for t in range(M)])
assert int(LIVE.sum()) == NLIVE


def sweep_plan(b, A, omw, ebw, abw):
    """Return the body as [(gate vector over t, emitter(s_))] in emission order,
    s_ the body's wire handles; a None vector emits ungated (the tree
    window open on allocator A).  Mirrors the per-point form: som fan-ins, the root-test
    ladder (pivot_pair builds, sorted operands), the rank-12 pairing
    (fan_pair), the ladder erase (emit_free, reversed), the som unwind.
    """
    m_om = rot_member(list(range(W))) & LIVE  # som fan-ins
    m_eb = rot_member(LOG_EB) & LIVE  # root-test groups
    m_v = rot_member(LOG_AB) & LIVE  # pairing V groups
    prow = np.zeros((B, M), dtype=np.int64)  # r_a(lam) masks
    for t in range(M):
        if LIVE[t]:
            prow[:, t] = pair_rows(LAMJ2[(M - t) % M])
    single = np.stack([prow == (1 << c) for c in range(B)], axis=1)  # [a, c, t]
    built = ~single.any(axis=1) & LIVE  # [a, t]
    m_sa = np.stack([((prow >> c) & 1).astype(bool) & built for c in range(B)], axis=1)  # [a, c, t]
    # the per-point spelling the body mirrors: every root-test group is
    # non-empty (both pivots build), every V group has >= 2 wires (it builds)
    assert (m_eb.any(axis=(1, 2)) | ~LIVE).all(), "empty root-test group"
    assert (m_v.sum(axis=(1, 2)) >= 2)[:, LIVE].all(), "degenerate V group"
    eb_sorted = sorted(
        (w, k, j) for k, word in enumerate(ebw) for j, w in enumerate(word)
    )  # terms sort operands
    assert len({w for w, _, _ in eb_sorted}) == len(eb_sorted)

    plan = []

    def op(vec, fn):
        plan.append((vec, fn))

    def eb_build(g, s, inverse):
        cands = [(w, m_eb[g, k, j]) for w, k, j in eb_sorted]
        if inverse:
            op(LIVE, lambda s_: b.x(s_[s]))
            cands.reverse()
        for w, v in cands:
            op(v, lambda s_, w=w: b.cx(w, s_[s]))
        if not inverse:
            op(LIVE, lambda s_: b.x(s_[s]))

    def ua(i, s_):
        return s_["sA"] if i == 1 else s_["tr"][i - 2]

    def rung(i, mid):
        """ladder rung i >= 1: wa = group 0 (i == 1, built in sA) or
        tree[i-2] (direct), wb = group i (built in sB).
        """
        if i == 1:
            eb_build(0, "sA", False)
        eb_build(i, "sB", False)
        op(LIVE, mid)
        eb_build(i, "sB", True)
        if i == 1:
            eb_build(0, "sA", True)

    om = [(bb, k, j) for bb in range(B) for k in range(W) for j in range(B)]
    for bb, k, j in om:
        op(m_om[bb, k, j], lambda s_, bb=bb, k=k, j=j: b.cx(omw[k][j], s_["som"][bb]))
    op(None, lambda s_: s_.__setitem__("tr", A.open("tree", B - 1)))
    for i in range(1, B):
        rung(i, lambda s_, i=i: b.ccx(ua(i, s_), s_["sB"], s_["tr"][i - 1]))
    vw = [(k, j) for k in range(len(abw)) for j in range(B)]
    for a_ in range(B):
        cases = [(built[a_], lambda s_: s_["sA"])] + [
            (single[a_, c] & LIVE, lambda s_, c=c: s_["som"][c]) for c in range(B)
        ]
        for c in range(B):
            op(m_sa[a_, c], lambda s_, c=c: b.cx(s_["som"][c], s_["sA"]))
        for k, j in vw:
            op(m_v[a_, k, j], lambda s_, k=k, j=j: b.cx(abw[k][j], s_["sB"]))
        for v, u in cases:
            op(v, lambda s_, u=u: b.ccx(u(s_), s_["sB"], s_["sT"]))
        op(LIVE, lambda s_: b.cz(s_["tr"][B - 2], s_["sT"]))
        op(LIVE, lambda s_: s_.__setitem__("m", b.hmr(s_["sT"])))
        for v, u in cases:
            op(v, lambda s_, u=u: b.cz(u(s_), s_["sB"], cond=s_["m"]))
        for k, j in reversed(vw):
            op(m_v[a_, k, j], lambda s_, k=k, j=j: b.cx(abw[k][j], s_["sB"]))
        for c in reversed(range(B)):
            op(m_sa[a_, c], lambda s_, c=c: b.cx(s_["som"][c], s_["sA"]))
    for i in range(B - 1, 0, -1):
        op(LIVE, lambda s_, i=i: s_.__setitem__("m", b.hmr(s_["tr"][i - 1])))
        rung(i, lambda s_, i=i: b.cz(ua(i, s_), s_["sB"], cond=s_["m"]))
    op(LIVE, lambda s_: A.close("tree", s_["tr"]))
    for bb, k, j in reversed(om):
        op(m_om[bb, k, j], lambda s_, bb=bb, k=k, j=j: b.cx(omw[k][j], s_["som"][bb]))
    return plan


def emit_sweep(b, E, A, omw, ebw, abw):
    """Emit the sweep: one gate-table row of M bits per distinct gate
    vector (the forward and unwind passes share rows; row r gates
    iteration t on bit r * M + t), then the repeat(M) body over the held
    pivots and the som / tree windows of allocator A.
    """
    plan = sweep_plan(b, A, omw, ebw, abw)
    rows = {}
    for vec, _ in plan:
        if vec is not None:
            rows.setdefault(vec.tobytes(), vec)
    rows.setdefault(LIVE.tobytes(), LIVE)
    row_of = {key: r for r, key in enumerate(rows)}
    table = b.declare_table(np.concatenate(list(rows.values())).astype(np.uint8).tolist())
    with b.repeat(M) as it:

        def gate(vec):
            return b.gated(table, it + row_of[vec.tobytes()] * M)

        s_ = {"sA": E.sA, "sB": E.sB, "sT": E.sT, "som": A.open("som", B)}
        for vec, fn in plan:
            if vec is None:
                fn(s_)
            else:
                with gate(vec):
                    fn(s_)
        with gate(LIVE):
            A.close("som", s_["som"])


# =========================================================================
# BM (stage 2 forward, stage 10 journaled reverse)
# =========================================================================
def flag_test(E, src, gate, ctl):
    """ctl <- [src != 0] AND parity(gate) (gate=() folds to the bare
    nonzero test): the tree window, the (B-1)-rung zero-test ladder, the
    flag AND, and the rung teardown.
    local: every zero-test tree here has gadgets.mcx's n-1 Toffoli cost,
    but must hold its flag live across later gates and journal each step
    for the reverse pass; mcx computes and releases inside one call and
    records nothing.
    """
    with eq_test(E, src, 0) as top:
        E.mac(one_rt((top,), 1, gate, 1, ctl))


class BM:
    """The q-form conditional-swap BM forward (stage 2) and its journaled
    reverse (stage 10).  Round 0 and round N-1 are peeled; the general
    round is a sequence of step methods over explicit round state.  run()
    makes the BM journal J current, so every step records
    into it; JROUND holds the
    journal index at the seed start and at every round start.
    """

    def __init__(self, E, A, syn, Dw, dW):
        self.b, self.E, self.A, self.J = E.b, E, A, Journal()
        self.J.recording = False
        self.pools = {}  # round -> window ids its journal touches
        self.syn, self.Dw, self.dW = syn, Dw, dW
        self.SIG = [BankReg(A, "sig") for _ in range(W + 1)]  # sigma_0..sigma_W
        self.BB = [BankReg(A, "bb") for _ in range(W + 2)]  # B bank (fused-Q: wrap at W+1)
        # STK_CAP-slot LIFO push stack, STKW wires per slot: B value lanes + a
        # TAGB-bit round tag.  The tag records the pushing round and gives
        # every round's control flag an erase anchor at the round tail --
        # after the d complement, ctl == dW[SGN] AND NOT [top tag == r] -- so
        # no flag outlives its round.
        self.STK = [BankReg(A, "stk", STKW) for _ in range(STK_CAP)]
        assert max(CAPS) <= len(self.STK), ("push cascade would overrun STK", max(CAPS))
        self.JROUND = [0]

    def round_head(self, r, bb_l):
        """Open round r (inside its bm/rNN phase block): journal mark, the
        round's cascade stack slots, and the fused-Q bank rotation (a
        relabel).  Returns the rotated bank.
        """
        self.A.round = r
        self.A.log = self.pools.setdefault(r, set())
        self.JROUND.append(self.J.mark())
        # Lazy-materialization invariant: a register touched mid-round may
        # only take free-list ids that strictly earlier rounds' windows freed
        # (those return before the descending replay reaches them).  At the
        # bank/ctl touch sites this holds because the round's own freed ids
        # are always reconsumed by the windows still open there; the cascade
        # runs after every window closed, so its stack depths materialize at
        # the round head instead.  reclaim fails closed on any breach.
        for s in self.STK[: CAPS[r]]:
            s.mat()
        return [bb_l[W + 1]] + bb_l[: W + 1]

    def discrepancy(self, r):
        """Dw <- sum_{i <= IMAXV[r]} sigma_i S_{r-i} (the truncated window)."""
        rtD = []
        for i in range(IMAXV[r] + 1):
            bil_terms(rtD, KT, spec_plain(self.SIG[i]), spec_plain(self.syn[r - i]), self.Dw)
        self.E.mac(rtD)

    @contextmanager
    def quotient(self, ctl, bb_l):
        """Yield the round quotient m without a stored accepted discrepancy.
        Invariant (README): sigma_0 * DEL == 1 on every
        correct-path round, DEL the accepted discrepancy of the q-form
        recursion, so the recursion's m = D'/DEL' is
          ctl=0: m = Delta * sigma_0 = p,
          ctl=1: m = DEL_old / Delta = 1/(Delta * sigma_0) = inv(p).
        Exit erases m (a function of p, inv(p), ctl), the chain and inv(p),
        then p.  Dw = Delta on every branch there; p = Delta * sigma_0_old,
        and sigma_0_old now sits in bb_l[0] (ctl=1) or SIG[0] (ctl=0, where
        Delta != 0 forces bb_l[0] == 0); on ctl=1 the update left SIG[0] =
        1/Delta, so Delta*(bb_l[0] + SIG[0]) = p + ctl:
          p == Dw * (bb_l[0] ^ SIG[0]) ^ ctl*[1]   (one KT claim).
        """
        E = self.E
        with opened(E, "pq", B) as pv:
            E.mac(prod_rt(pv, self.Dw, self.SIG[0]))
            with opened(E, "dinv", B) as dv, chain_live(E, pv, dv), opened(E, "m", B) as mq:
                # m = p ^ ctl*(p ^ inv(p)): B free copies + B CCX
                rtM = []
                for t_ in range(B):
                    rtM.append(("L", (pv[t_],), 0, (mq[t_],)))
                    rtM.append(norm_term((ctl,), 0, tuple(sorted((pv[t_], dv[t_]))), 0, (mq[t_],)))
                E.mac(rtM)
                yield mq
                E.free(rtM)
            rtPe = []
            bil_terms(
                rtPe,
                KT,
                spec_plain(self.Dw),
                [frozenset((bb_l[0][j], self.SIG[0][j])) for j in range(B)],
                pv,
            )
            rtPe.append(("L", (ctl,), 0, (pv[0],)))
            E.free(rtPe)

    def bank_swaps(self, r, ctl, bb_l):
        """cswap(ctl) of the sigma and B banks over slots 0..KMSV[r]."""
        for k2 in range(KMSV[r] + 1):
            if r == 1 and k2 == 1:
                # provable identity: both branches leave sigma_1 ==
                # B_1 entering this swap (2-branch enumeration)
                continue
            for t_ in range(B):
                self.E.cswap(ctl, self.SIG[k2][t_], bb_l[k2][t_])

    def update(self, r, k0, mq, bb_l):
        """sigma_k ^= m * B_k over slots k0..KMMV[r] (one KT12 each)."""
        for k2 in range(k0, KMMV[r] + 1):
            self.E.xor(prod_rt(self.SIG[k2], mq, bb_l[k2]))

    def push(self, r, ctl=True):
        """Emit the fused-Q push cascade on pqc = dW[SGN] (d<0 forces ctl=0);
        ctl=False spells it with plain swaps (the unconditional r=N-1
        push, 0 Toffoli, no tag).  Extent CAPS[r] = min(DP extent,
        STK_CAP): the CLAMPED Fredkin ladder still acts on pairwise-distinct
        qubits; on a full stack it rotates STK[STK_CAP-1] into Dw (overflow
        drop, the E2 class); on a not-full stack it is the identical
        shift as the uncapped ladder.  Inter-slot swaps carry all STKW
        lanes; the Dw level is value-only.
        """
        E, STK = self.E, self.STK
        capr = CAPS[r]
        if not capr:
            return
        pqc = self.dW[SGN]

        def sw(x, y):
            if ctl:
                E.cswap(pqc, x, y)
            else:
                E.swap(x, y)

        for i2 in range(capr - 2, -1, -1):
            for t_ in range(STKW):
                sw(STK[i2][t_], STK[i2 + 1][t_])
        for t_ in range(B):
            sw(self.Dw[t_], STK[0][t_])
        if not ctl:
            return
        # tag the incoming entry with its round: the vacated slot's tag lanes
        # are |0> below overflow, so free CX from the push flag writes r
        # exactly (overflow shots are the E2 class)
        for bp in bitlist(r, TAGB):
            E.cx(pqc, STK[0][B + bp])

    def erase_dw(self, ctl):
        """Erase Dw.  Post-cascade Dw == ctl * Delta on every branch (d<0:
        pushed, Dw = 0; ctl=0, d>=0: Delta = 0), and ctl=1 left sigma_0 =
        1/Delta: Dw == ctl * inv(sigma_0), one chain on SIG[0] and a B-term
        claim (stack overflow is the E2 class).
        """
        E = self.E
        with opened(E, "wreg", B) as wr, chain_live(E, self.SIG[0], wr):
            E.free([norm_term((ctl,), 0, (wr[t_],), 0, (self.Dw[t_],)) for t_ in range(B)])

    def close_ctl(self, r, ctl):
        """d complement if ctl, then erase ctl (the caller closes its
        window).  Post-complement dW[SGN] = [ctl OR pushed], and a push at
        this round iff the top tag reads r,
        so ctl == dW[SGN] AND NOT [top tag == r].  Rounds the DP proves
        push-free (and r=0) anchor on dW[SGN] alone, for free.  On a stack
        overflow the rotated-out tag corrupts the compare -- overflow shots
        are already the E2 class.
        """
        E, dW = self.E, self.dW
        for i in range(DWB):
            E.cx(ctl, dW[i])
        if CAPS_FULL[r] == 0:
            E.free([norm_term((dW[SGN],), 0, (), 1, (ctl,))])
        else:
            tag = self.STK[0][B:]  # materialized at round_head
            with opened(E, "eqt", TAGB - 1) as eqw:
                flips = bitlist(~r & TAGMASK, TAGB)
                for bp in flips:
                    E.x(tag[bp])
                eqr = E.ladder([((tag[i],), 0) for i in range(TAGB)], eqw)
                E.free([norm_term((dW[SGN],), 0, (eqw[TAGB - 2],), 1, (ctl,))])
                E.frees(eqr)
                for bp in flips:
                    E.x(tag[bp])

    def inc_d(self):
        """d += 1: the constant +1 ripple on dW (DWB-2 CCX, MBU-erased
        journaled carries).
        local: gadgets.arith.add has no constant-operand form (an all-None
        operand is a no-op) and no journal hook.
        """
        with opened(self.E, "inc", DWB - 2) as ic:
            self.E.inc1(self.dW, ic)

    def round0(self, bb_l):
        """r=0 needs no working discrepancy register: D(0)=S_1 lives in
        syn[0].  ctl = [S_1 != 0] off syn[0], and the round quotient is
        m = inv(S_1) on both branches (the chain maps 0 to 0).
        """
        E, SIG, syn0 = self.E, self.SIG, self.syn[0]
        with self.b.phase("bm/r00"):
            bb_l = self.round_head(0, bb_l)
            with opened(E, "ctl", 1) as (ctl,):
                flag_test(E, syn0, (), ctl)
                with opened(E, "m", B) as mq:
                    wins, rts = chain_fwd(E, syn0, mq)
                    # SIG[0]=|1>, SIG[1]=|0>, bb_l[0]=|0>, bb_l[1]=|1> are KNOWN basis
                    # states: the two register cswaps collapse to 4 free CX.
                    assert KMSV[0] == 1
                    E.cx(ctl, SIG[0][0])
                    E.cx(ctl, bb_l[0][0])
                    E.cx(ctl, SIG[1][0])
                    E.cx(ctl, bb_l[1][0])
                    # post-swap bb_l[0] value = ctl, bb_l[1] value = 1^ctl; m nonzero
                    # only when ctl=1, hence m*bb0 = m (free CX) and m*bb1 = 0 (skip).
                    for t_ in range(B):
                        E.cx(mq[t_], SIG[0][t_])
                    # erase m: post-update m == ctl * sigma_0 on both branches; the
                    # chain intermediates are functions of syn[0]
                    E.free([norm_term((ctl,), 0, (SIG[0][t_],), 0, (mq[t_],)) for t_ in range(B)])
                    chain_undo(E, wins, rts[:-1])
                self.close_ctl(0, ctl)
            self.inc_d()
        return bb_l

    def round(self, r, bb_l):
        """Emit a general round 0 < r < N-1."""
        with self.b.phase(f"bm/r{r:02d}"):
            bb_l = self.round_head(r, bb_l)
            with opened(self.E, "ctl", 1) as (ctl,):
                self.discrepancy(r)
                # ctl is set iff Dw != 0 and d >= 0
                flag_test(self.E, self.Dw, (self.dW[SGN],), ctl)
                with self.quotient(ctl, bb_l) as mq:
                    self.bank_swaps(r, ctl, bb_l)
                    self.update(r, 0, mq, bb_l)
                self.push(r)
                self.erase_dw(ctl)
                self.close_ctl(r, ctl)
            if r < N - 2:
                # +1 always -> d = ctl ? -d : d+1 (at r=N-2 the increment is
                # dropped: the counter has no later consumer and the complement
                # alone arms the erase anchor)
                self.inc_d()
        return bb_l

    def round_last(self, bb_l):
        """r=N-1: no ctl.  m = p, computed straight into the m window; the
        update skips slot 0 (with a length change due (w=W mass) zB occupies
        slots [N-1-r*, W] and never wraps to 0; otherwise M=0); the push is
        unconditional (push(ctl=False)), and r=N-1 writes no tag (no later
        round reads it).
        """
        r = N - 1
        with self.b.phase(f"bm/r{r:02d}"):
            bb_l = self.round_head(r, bb_l)
            self.discrepancy(r)
            with opened(self.E, "m", B) as mq:
                rtM = self.E.mac(prod_rt(mq, self.Dw, self.SIG[0]))
                self.update(r, 1, mq, bb_l)
                # m is a function of Dw and the untouched sigma_0
                self.E.free(rtM)
            assert CAPS[r] == STK_CAP
            self.push(r, ctl=False)

    def run(self):
        """Stage 2: the seed, round 0, rounds 1..N-2, round N-1; the journal
        is frozen afterwards (stages 3..9 are mirrored explicitly).
        """
        E, J, A = self.E, self.J, self.A
        J.recording = True
        A.round = -1  # seed segment: banks touched here free last
        A.log = self.pools.setdefault(-1, set())
        with E.using(J=J, A=A):
            with self.b.phase("bm/seed"):
                E.x(self.SIG[0][0])
                E.x(self.BB[0][0])
            bb_l = self.round0(list(self.BB))
            for r in range(1, N - 1):
                bb_l = self.round(r, bb_l)
            self.round_last(bb_l)
        J.recording = False
        self.JROUND.append(J.mark())
        A.round = None
        A.log = None
        # every lazy register must have materialized inside BM: a first touch
        # in a later stage would have no mirror round to free it at
        assert all(reg.wires is not None for reg in self.SIG + self.BB + self.STK), (
            "lazy register untouched during BM"
        )

    def reverse(self):
        """Stage 10, the journaled measurement-based reverse pass.  Every BM
        step went through J with its conjugate (macs record
        release-or-recompute per the emit-time pebbling choice), so
        replaying the conjugates in reverse re-materializes each round's
        intermediates at the mirrored points and the replayed Fredkin
        cascades pop the stack at exactly the right moments.  The journal
        replays closures over hard wire ids: each round's window pool is
        re-materialized with reclaim right before its replay and freed
        right after, and every bank slot is freed after the replay of its
        first-touch round (its state there equals the forward state before
        that round: |0>).  The availability lines up by time symmetry -- an
        id a bank slot took at forward round f is released before the
        replays of rounds < f, exactly when their pools want it back -- and
        reclaim fails closed on any gap.
        """
        b, A, JR = self.b, self.A, self.JROUND
        freed = 0
        for k in range(len(JR) - 2, -1, -1):
            rr = k - 1  # round -1 = the seed
            with b.phase("reverse/seed" if rr < 0 else f"reverse/r{rr:02d}"):
                if JR[k + 1] > JR[k]:
                    with A.replay(self.pools.get(rr, ())):
                        self.J.reverse(since=JR[k], until=JR[k + 1])
                for reg in reversed(A.bank_log.get(rr, [])):
                    A.close(reg.name, reg.wires)
                    freed += 1
        assert freed == len(self.SIG) + len(self.BB) + len(self.STK), (
            "lazy registers leaked past the reverse",
            freed,
        )


# =========================================================================
# (Un)monicize (stages 3 and 9) and the in-place omega70 (stages 4 and 8)
# =========================================================================
# ARMORED: zflag = [sigma_0 == 0] XORed into W0[0] and SIG[0][0] around the
# slot loop makes the scaling invertible for ALL data (scale defaults to 1
# when sigma_0 = 0), so truncation deviations can never make phase garbage.


def w0_pass(E, SIG, mid, W0):
    """One w0 window: open the chain scratch, run the SIG[0] inversion
    chain intermediates forward, apply mid to the W0 term list (emit_mac
    to create W0, emit_free to erase it), then erase the intermediates
    and close the chain scratch.  The create and destroy windows differ
    only in mid.
    """
    wins, rts = chain_open(E, SIG[0], W0)
    E.macs(rts[:-1])
    mid(rts[-1])
    E.frees(rts[:-1])
    chain_close(E, wins)


def mon_slot(E, sig_i, scale, eraser, ymon):
    """sig_i <- scale * sig_i: ymon = scale * sig_i, a demolition erase of
    sig_i against eraser x ymon through Em's measurement-erase core
    (per-target hmr + conditioned-CZ repayment), then swap ymon into
    sig_i.  monicize: (scale, eraser) = (W0, SIG[0]); unmonicize: (SIG[0],
    W0).
    """
    E.emit_mac(prod_rt(ymon, scale, sig_i))
    E.emit_free(
        [
            ("P", (eraser[a_],), 0, (ymon[b_],), 0, (sig_i[k2],))
            for k2 in range(B)
            for (a_, b_) in MONB[k2]
        ]
    )
    for t_ in range(B):
        E.b.swap(ymon[t_], sig_i[t_])


def mon_stage(E, SIG, prefix, slot_range, forward):
    """One whole (un)monicize stage: create W0, arm the zflag armor,
    run the W scaling slots, disarm, erase W0.  monicize and
    unmonicize are the same block up to phase prefix, slot order, and
    the (scale, eraser) pair.
    """
    b = E.b
    # W0 and the zflag ladder (eq_test's two halves) outlive the phase that
    # opens them, so they open and close by hand in their first/last phase.
    with b.phase(prefix + "/w0+"):
        W0 = E.A.open("w0", B)
        w0_pass(E, SIG, E.emit_mac, W0)
    with b.phase(prefix + "/zflag+"):
        tr = E.A.open("tree", B - 1)
        rts = E.ladder(eq_rungs(SIG[0], 0), tr)
        top = tr[B - 2]
        b.cx(top, W0[0])
        b.cx(top, SIG[0][0])
    scale, eraser = (W0, SIG[0]) if forward else (SIG[0], W0)
    for i in slot_range:
        with b.phase(f"{prefix}/s{i:02d}"), opened(E, "ymon", B) as ym:
            mon_slot(E, SIG[i], scale, eraser, ym)
    with b.phase(prefix + "/zflag-"):
        b.cx(top, SIG[0][0])
        b.cx(top, W0[0])
        E.frees(rts)
        E.A.close("tree", tr)
    with b.phase(prefix + "/w0-"):
        w0_pass(E, SIG, E.emit_free, W0)
        E.A.close("w0", W0)


def omega_pass(E, macrt, prefix, order):
    """Emit the omega70 block term lists in order, one phase block each."""
    for bi in order:
        with E.b.phase(f"{prefix}/m{bi:02d}"):
            E.emit_mac(macrt[bi])


def build():
    """Build the circuit and assert its pins.  Returns (builder, Toffoli
    count); the occupancy/liveness profilers consume its wire_log.
    """
    # ---- registers, held scratch, allocators --------------------------------
    b = Builder()
    with b.phase("init"):  # profile-stage blocks (metadata; emit nothing)
        syn = list(declare_registers(b, [(f"s{r:02d}", B) for r in range(N)]).values())
        assert all(list(syn[r]) == list(range(B * r, B * r + B)) for r in range(N)), (
            "syn ids not contiguous"
        )  # qubits Br..Br+B-1

        ANC = []

        def An(n):
            qs = b.qubits(n)
            ANC.extend(qs)
            return qs

        Dw = An(B)  # fused-Q working discrepancy register
        dW = An(DWB)  # two's-complement d counter
        OMW = [syn[i] for i in range(W)]  # omega lives IN PLACE in syn[0..W-1]

        # Ownership wire-sets (claim/release bookkeeping; emits no op).  Every
        # long-lived wire joins exactly one named set with its full id list, so
        # the anonymous pool live - sum(owned) is empty from init to cleanup;
        # tools/occupancy_profile.py validates the discipline fail-closed.
        b.claim("syn", [w for reg in syn for w in reg])
        b.claim("dw", Dw)
        b.claim("dcount", dW)

        # Transient scratch round-trips through the allocator (A.open /
        # A.close).  Only state that persists between windows (discrepancies,
        # stack, controls, counters, the Em pivots) stays held; everything that
        # returns to |0> mid-circuit is freed there, so peak_qubits is the
        # measured max co-live width and the liveness profile shows the true
        # windows.  Journal replays (the BM reverse, the Psi route) run
        # closures over hard wire ids, so A logs the window ids a journal
        # touches and re-materializes exactly that set before its replay
        # (A.replay; reclaim fails closed when any id is not free).
        A = Alloc(b)

        sA = An(1)[0]
        b.claim("pivA", [sA])
        sB = An(1)[0]
        b.claim("pivB", [sB])
        sT = An(1)[0]
        b.claim("fanT", [sT])
        E = Em(b, sA, sB, sT, A)
        bm = BM(E, A, syn, Dw, dW)
        SIG = bm.SIG

    # Width ledger: N * B syn + (B + DWB + 3) held (Dw, dW, pivots) + the
    # lazily-materialized wires ((2W+3) B bank + STK_CAP tagged stack slots)
    # + the late BM/reverse rounds' window tiling.  The peak assert below
    # enforces the total.

    # ---- 1. zlin: free Z part sum_i Tr(u_i S_i) -----------------------------
    with b.phase("zlin"):
        for i in range(N):
            for bb in bitlist(ZMASKI[i], B):
                b.z(syn[i][bb])

    # ---- 2. bm: BM forward (q-form conditional-swap, single-chain m) --------
    bm.run()

    # ---- 3. monicize: SIG[i] <- inv(sigma_0)*SIG[i] for i=1..W --------------
    mon_stage(E, SIG, "monicize", range(1, W + 1), True)

    # ---- 4. omega70 = sigma_m*S mod z^N, IN PLACE in syn[0..N-1] ------------
    # sigma_m is monic: the triangles' diagonals span sigma_1..sigma_{W-1},
    # the off-diagonal tail block's sigma_1..sigma_W (omega_rts).
    # Rows W..N-1 (the TAIL) end at 0 on every correct decode and host the
    # Bezout pair (stage 5).
    MACRT = omega_rts(SIG, syn)
    omega_pass(E, MACRT, "omega", range(len(MACRT)))

    # ---- 5. euclid: the half-degree Bezout pass (Psi route) -----------------
    # sigma_m = E(z)^2 + z O(z)^2 with E_i = sqrt(sigma_2i), O_i =
    # sqrt(sigma_2i+1): SIG[1..35] are square-rooted IN PLACE (a free linear
    # map) and the Euclid bank EB is E = [ONE, SIG2, .., SIG34], O = [SIG1, ..,
    # SIG35] (ONE is a fresh word holding 1 = E_0).  m = floor((35 - L)/2) =
    # (dW + 2) >> 2 (dW = the BM counter, +2 in place), and the barrel pad
    # shifts both halves up by m words so deg E' = 17 on every correct decode.
    # JSET journals the setup (ONE, the in-place square roots, dW += 2) so
    # stage 7 undoes it with one reverse.
    # PSI_IDS logs every scratch id the Psi journals touch; stages R and 7
    # hold exactly that set around their replays.
    with b.phase("euclid/setup"):
        ONE = A.open("one", B)
        EPS = A.open("zprec", 6)  # eps + 5-bit idx
        PRE = A.open("pre", 1)[0]
        PSI_IDS = set()
        JSET, JPAD, JPRE, JF, JSEED = (Journal(), Journal(), Journal(), Journal(), Journal())
        with E.using(J=JSET), A.logging(PSI_IDS):
            E.x(ONE[0])
            for i in range(1, W + 1):
                JSET.step(
                    lambda r=list(SIG[i]): linear_in_place(b, r, SQRT_ROWS),
                    lambda r=list(SIG[i]): linear_in_place(b, r, SQUARE_ROWS),
                )
            # dW += 2: a constant increment on bits 1..7 (ladder carries, MBU-erased)
            with opened(E, "inc", DWB - 3) as tr:
                E.inc1(dW[1:DWB], tr)
        EW = [ONE] + [list(SIG[2 * i]) for i in range(1, NH)]
        OWD = [list(SIG[2 * i + 1]) for i in range(NH)]
        MBITS = dW[2:7]  # m = (dW + 2) >> 2
    with b.phase("euclid/pad"), E.using(J=JPAD):
        pad_barrel(E, MBITS, EW, OWD)
    with b.phase("euclid/pre"), E.using(J=JPRE), A.logging(PSI_IDS):
        psi_pre(E, EW, OWD, PRE)
    with b.phase("euclid/F"), E.using(J=JF), A.logging(PSI_IDS):
        F_FIN, _G_FIN, RECS = psi_forward(E, EW, OWD, EPS[0], EPS[1:6])

    # ---- seed + R: the Bezout pair (a, b) grows in the TAIL + one EXT word --
    # a[(m+1) % 18] <- inv(h) (h = the final F pivot), then R = the journal
    # reverse of F interleaved with the cofactor recurrence: on exit EB holds
    # (E', O') again and (a, b) = (A, B) with A E' + B O' = z^m, deg <= 17.
    with b.phase("euclid/seed"):
        TAILW = [list(syn[i]) for i in range(W, N)]
        AW = TAILW[:NH]
        with E.using(J=JSEED), A.logging(PSI_IDS):
            psi_seed(E, F_FIN[NH - 1], MBITS, AW)
    with b.phase("euclid/R"):
        with A.replay(PSI_IDS, "psipool"):
            EXT = A.open("ext", B)  # opened under the replay hold: never a PSI_IDS id
            BWD = TAILW[NH:] + [EXT]
            JR, AFIN, BFIN = build_R(E, JF, RECS, AW, BWD, PRE, PSI_WIN, (EW, OWD))
            JR.forward()
        A.close("zprec", EPS)

    # ---- 6. sweep: root flag + rank-12 Psi pairing over the M points --------
    with b.phase("sweep"):
        emit_sweep(b, E, A, OMW, EW + OWD, AFIN + BFIN)

    # ---- 7. euclid_inv: R^-1, unseed, F^-1, un-pre, un-pad, un-setup --------
    # The zprec and psipool holds span R..setup, so they open and close by
    # hand (A.replay's two halves) in the first and last phase.
    PSI_POOL = sorted(PSI_IDS)
    with b.phase("euclid_inv/R"):
        A.reopen("zprec", EPS)
        A.reopen("psipool", PSI_POOL)
        JR.reverse()
        A.close("ext", EXT)
    with b.phase("euclid_inv/seed"):
        JSEED.reverse()
    with b.phase("euclid_inv/F"):
        JF.reverse()
    with b.phase("euclid_inv/pre"):
        JPRE.reverse()
    with b.phase("euclid_inv/pad"):
        JPAD.reverse()
    with b.phase("euclid_inv/setup"):
        JSET.reverse()
        A.close("psipool", PSI_POOL)
        A.close("pre", [PRE])
        A.close("zprec", EPS)
        A.close("one", ONE)

    # ---- 8. omega_restore: exact reversed block list ------------------------
    # each block's targets are disjoint from its operands, so re-emitting the
    # same term list un-XORs it; the reversed block order restores S exactly.
    omega_pass(E, MACRT, "omega_restore", reversed(range(len(MACRT))))

    # ---- 9. unmonicize: SIG[i] <- sigma_0*SIG[i] for i=W..1 -----------------
    mon_stage(E, SIG, "unmonicize", range(W, 0, -1), False)

    # ---- 10. reverse: the journaled BM reverse pass -------------------------
    bm.reverse()

    # ---- 11. cleanup ---------------------------------------------------------
    # Release-before-free through the only teardown path: every workspace set
    # returns its wires (full id lists, reverse claim order) ahead of the
    # frees; the N * B syndrome wires stay owned to the end as the declared
    # registers.
    with b.phase("cleanup"):
        b.release("fanT", [sT])
        b.release("pivB", [sB])
        b.release("pivA", [sA])
        b.release("dcount", dW)
        b.release("dw", Dw)
        b.recycle_all(reversed(ANC))

    # =========================================================================
    # Metrics gate, pre-emission: the pins under the measured window
    # schedule and STK_CAP = 1.
    # =========================================================================
    TOF = assert_metrics(b, __file__)

    return b, TOF


if __name__ == "__main__":
    emit_and_report(build()[0])
