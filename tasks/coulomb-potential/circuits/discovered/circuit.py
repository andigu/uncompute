#!/usr/bin/env python3
"""coulomb-potential circuits/discovered: a window-fitted linear
interpolator — the acceptance windows of the +-32-unit output
allowance, pulled back through the circuit's own arithmetic, fit a
128-leaf full-depth table read by a binary QROM; segmented Bennett
mirror whose unlookup is measurement-based: the table words are
demolished in the X basis and the phase damage repaid by one floored
repair walk over the address tree.

Certified cost: 1,140 Toffoli x 142 qubits = 161,880 (assert_metrics
in the epilogue).

RUN: uv run python circuit.py   (stream to $UNCOMPUTE_OPS_PATH, else ./ops.bin)

README.md in this folder describes the circuit.
"""

import math
from contextlib import nullcontext

from uncompute import Builder
from uncompute.gadgets import (
    and_ladder,
    and_ladder_release,
    cdkm_add_co,
    declare_registers,
    xor_word,
)
from uncompute.testing import assert_metrics, emit_and_report, load_contract, load_pinned

# Task contract pins (contract.json beside grader.py, asserted there):
# coord bits, k width, output scale 2^24, the inclusive +-EPS band, max n2.
_CONTRACT = load_contract(__file__)
B_C = _CONTRACT["B"]
M_K = _CONTRACT["M"]
N2_MAX = _CONTRACT["N2_MAX"]
C_SCALE = _CONTRACT["C_SCALE"]
EPS = _CONTRACT["EPS"]

# Fixed-point architecture of the interpolator.  n2 (18 bits) is normalized
# to the mantissa m = n2 << (17-e), m in [2^17, 2^18), e = floor(log2 n2).
# The QROM is addressed by (e mod 2, top A fractional mantissa bits); the R
# bits of m from bit 10 down to bit RL — disjoint from the address — form
# the remainder that drives a linear correction; the bits below RL are
# truncated outright (part of the error spend).  With no address bit shared
# into the remainder, no leaf can serve two address slots, so the fit
# bottoms out at the full-depth 128-leaf tree — the trade that makes the
# measurement-based unlookup below pay: lookup grows by 35 path ANDs while
# unlookup drops from a second full walk to a 17-Toffoli repair.
# Each leaf stores an 18-bit intercept g whose top bit is always
# set and therefore implicit (only GLOW = 17 bits ride on wires) and a
# slope s in units of 2^-SH.  The datapath computes
#     gs = g - sum_i rem_i * ((s << i) >> SH),
# with the tables scaled so gs ~= REF(n2) * 2^(h-Q), h = floor(e/2); the
# variable left shift + rounded accumulate then recover
#     DELTA = round(gs * 2^(6-h))  ~= REF(n2)
# (delta_of below is the exact arithmetic model of that tail).
A = 6  # QROM mantissa address bits (NF = 2^(A+1) entries incl. parity)
Q = 6  # intercept scale: gs ~= REF * 2^(h-Q), keeping gs in [2^17, 2^18)
R = 6  # remainder bits driving the linear correction (NR = 2^R)
RL = 5  # lowest mantissa bit kept: bits RL-1..0 are truncated
SH = 2  # slope fixed-point shift (slope rows are (s << i) >> SH)
GW = 18
GMAX = (1 << GW) - 1  # intercept word width / max
SREGW = 25
FB = 24  # widened intercept register: GW + 7 shift headroom
GLOW = GW - 1  # stored intercept bits (msb implicit)
NF = 1 << (A + 1)
NR = 1 << R
HBIT = 1 << (GW - 1)  # the implicit leading bit of every intercept


def REF(n2):
    """Return the ideal value: round(2^24 / sqrt(n2))."""
    return round(C_SCALE / math.sqrt(n2))


# REF must equal the grader oracle byte-for-byte: grader.ideal_f rounds
# half AWAY from zero, Python round() rounds half to even, and the two
# agree only because no f64 value C_SCALE/sqrt(n2) lands exactly on .5
# anywhere in the domain (tie-freeness).  Proved on every build against
# the contract digests: the achievable domain (sums of three squares of
# |d| <= 255) and [REF(n2)] over it must hash to DOM_DIGEST / TBL_DIGEST,
# the grader's own SHA-256 pins (~0.2 s).
_ach = bytearray(N2_MAX + 1)
for _dx in range(256):
    _x2 = _dx * _dx
    for _dy in range(_dx, 256):
        _s2 = _x2 + _dy * _dy
        for _dz in range(_dy, 256):
            _ach[_s2 + _dz * _dz] = 1
_ach[0] = 0
_dom = [_v for _v in range(1, N2_MAX + 1) if _ach[_v]]
load_pinned(_dom, _CONTRACT["DOM_DIGEST"])
load_pinned([REF(_v) for _v in _dom], _CONTRACT["TBL_DIGEST"])
del _ach, _dom


# PS(s, r) is the exact value the multiply stage computes for slope s and
# remainder r: the sum of the surviving truncated rows.  make_psub tabulates
# it for all NR remainders by lowest-set-bit DP.
def PS(s, r):
    v = 0
    for i in range(R):
        if (r >> i) & 1:
            v += (s << i) >> SH
    return v


def make_psub(s):
    tb = [0] * NR
    row = [(s << i) >> SH for i in range(R)]
    for r in range(1, NR):
        lb = r & (-r)
        tb[r] = tb[r ^ lb] + row[lb.bit_length() - 1]
    return tb


# MSH[n] = the 18-bit normalized mantissa of n; FIDX[n] = its QROM address
# = (e&1) << A | (top A fractional mantissa bits).  Bit 17 of MSH is the
# implicit leading 1 and is not part of the address.
FIDX = [0] * (N2_MAX + 1)
MSH = [0] * (N2_MAX + 1)
for _n in range(1, N2_MAX + 1):
    _e = _n.bit_length() - 1
    _m = _n << (17 - _e)
    MSH[_n] = _m
    FIDX[_n] = ((_e & 1) << A) | ((_m >> 11) & 63)


# Exact arithmetic model of the datapath tail: the circuit left-shifts the
# widened intercept by floor(z/2) = 8-h (z = normalization shift), then the
# accumulate feeds bits [2:] into k with bit 1 as carry-in, i.e.
# DELTA = round(gs * 2^(8-h) / 4).  For h < 7 the shift makes the /4 exact;
# for h >= 7 the round-to-nearest can add one ulp (see hi_eps below).
def delta_of(gs, h):
    a = 8 - h
    v = gs << a
    return (v >> 2) + ((v >> 1) & 1)


# ---- the interpolator table, fitted on acceptance windows -----------------
# For every n2 the final DELTA must land in [REF-32, REF+32].  Pulling that
# band backwards through delta_of turns it into an interval [lo, hi] of
# admissible gs values; LO/HI intersect those intervals over every n2 that
# shares a (QROM address, remainder) cell.  _fit_line() then searches for one
# quantized slope s such that, after adding back the exact partial-product
# table, a single intercept g covers every remainder of a whole address
# block — i.e. one (s, g) line serves the block.  _leaf_tree() makes the tree
# adaptive: try a line over the full block, split in half on failure, so
# leaves sit exactly as shallow as the +-32 band allows.  The result is
# certified exhaustively over all n2 in [4, N2_MAX] before use (n2 <= 3
# under-normalizes and is handled by the tiny-n2 patch instead).
def _acceptance_windows():
    """Per (QROM address, remainder) cell, the admissible gs interval [LO, HI].

    Returns None when some n2 has an empty window.
    """
    LO = [[0] * NR for _ in range(NF)]
    HI = [[GMAX] * NR for _ in range(NF)]
    for n2 in range(4, N2_MAX + 1):
        e = n2.bit_length() - 1
        h = e >> 1
        t = REF(n2)
        d = h - Q
        # h >= 7: delta_of's round-to-nearest can add one ulp, so the top
        # of the band tightens from +EPS to +(EPS-1); below that /4 is exact.
        hi_eps = EPS - 1 if h >= 7 else EPS
        # invert the tail: gs range whose DELTA stays in [t-EPS, t+hi_eps]
        if d >= 0:
            lo = (t - EPS) << d
            hi = ((t + hi_eps + 1) << d) - 1
        else:
            lo = (t - EPS + (1 << -d) - 1) >> (-d)
            hi = (t + hi_eps) >> (-d)
        lo = max(lo, 0)
        hi = min(hi, GMAX)
        if lo > hi:
            return None
        f = FIDX[n2]
        rem = (MSH[n2] >> RL) & (NR - 1)
        LO[f][rem] = max(LO[f][rem], lo)
        HI[f][rem] = min(HI[f][rem], hi)
    return LO, HI


def _fit_line(LO, HI, f0, f1):
    """One (slope, intercept) line that serves addresses f0..f1-1, or None."""
    LOn = [0] * NR
    HIn = [GMAX] * NR
    for f in range(f0, f1):
        Lf = LO[f]
        Hf = HI[f]
        for r in range(NR):
            LOn[r] = max(LOn[r], Lf[r])
            HIn[r] = min(HIn[r], Hf[r])
    # analytic seed: over this block gs ~= kap/sqrt(x) with x = m/2^17,
    # so s0 = the expected per-remainder-step drop, in slope units of
    # 2^-SH; then search slopes outward from the seed.
    p = f0 >> A
    a = f0 & 63
    kap = (1 << 24) * (2**-0.5 if p else 1.0) / (1 << Q)
    x0 = 1.0 + a / 64.0
    span = NR * (1 << RL) / 131072.0
    s0 = round((kap / math.sqrt(x0) - kap / math.sqrt(x0 + span)) / NR * (1 << SH))
    for ds in range(700):
        for s in (s0,) if ds == 0 else (s0 + ds, s0 - ds):
            if s < 0:
                continue
            tb = make_psub(s)
            # slope s is feasible iff one g satisfies
            # LOn[r]+PS(s,r) <= g <= HIn[r]+PS(s,r) for ALL remainders;
            # intersect and take the midpoint for symmetric margin.
            glo = 0
            ghi = GMAX
            ok = True
            for r in range(NR):
                sub = tb[r]
                l = LOn[r] + sub
                glo = max(glo, l)
                h2 = HIn[r] + sub
                ghi = min(ghi, h2)
                if glo > ghi:
                    ok = False
                    break
            if ok:
                return s, (glo + ghi) // 2
    return None


def _leaf_tree(LO, HI):
    """Build the adaptive leaf tree {(depth, prefix): (slope, intercept)}, or None.

    Each address block gets one leaf when feasible, else splits in half,
    so the (depth, prefix) keys are exactly the QROM's early-terminating tree.
    """
    LEAF = {}

    def grow(depth, prefix):
        rr = _fit_line(LO, HI, prefix << (7 - depth), (prefix + 1) << (7 - depth))
        if rr is not None:
            LEAF[(depth, prefix)] = rr
            return True
        if depth == 7:
            return False
        okl = grow(depth + 1, 2 * prefix)
        okr = grow(depth + 1, 2 * prefix + 1)
        return okl and okr

    # a depth-0 leaf never occurs in practice and qrom() has no
    # unconditional write path for it — reject rather than special-case
    if not grow(0, 0) or (0, 0) in LEAF:
        return None
    return LEAF


def _flatten(LEAF):
    """Per-address slope and intercept tables (SLf, G0f) from the leaf tree."""
    SLf = [0] * NF
    G0f = [0] * NF
    for f in range(NF):
        for depth in range(8):
            key = (depth, f >> (7 - depth))
            if key in LEAF:
                SLf[f], G0f[f] = LEAF[key]
                break
    return SLf, G0f


def _certified(LEAF, SLf, G0f):
    """Exhaustive certification over every n2 in [4, N2_MAX].

    Every reachable gs keeps its implicit msb (stays in [2^17, 2^18))
    and its DELTA lands inside the +-32 band.
    """
    for _sl, g in LEAF.values():
        if not (HBIT <= g <= GMAX):
            return False
    for n2 in range(4, N2_MAX + 1):
        f = FIDX[n2]
        gs = G0f[f] - PS(SLf[f], (MSH[n2] >> RL) & (NR - 1))
        if not (HBIT <= gs <= GMAX):
            return False
        h = (n2.bit_length() - 1) >> 1
        err = abs(delta_of(gs, h) - REF(n2))
        if err > EPS:
            return False
    return True


def build_tables():
    windows = _acceptance_windows()
    if windows is None:
        return None
    LEAF = _leaf_tree(*windows)
    if LEAF is None:
        return None
    SLf, G0f = _flatten(LEAF)
    if not _certified(LEAF, SLf, G0f):
        return None
    # datapath width guards: the product word fits under the stored
    # intercept bits and the widened register has shift headroom
    SMAX = max(max(SLf), 1)
    SW = SMAX.bit_length()
    WP = max(PS(SMAX, NR - 1).bit_length(), A + 2, 10)
    if not (WP <= GLOW and GLOW + SW <= SREGW and 2 * GLOW - WP < FB):
        return None
    return SLf, G0f, LEAF, SW, WP


_tabs = build_tables()  # ~1.5 s, certified exhaustively inside
assert _tabs is not None, "table infeasible"
SLf, G0f, LEAF, SW, WP = _tabs
SMAX = max(max(SLf), 1)
# leaf_f indexes the leaves by the top 7 mantissa bits: the tree is full depth
assert all(depth == 7 for (depth, _p) in LEAF)
print(f"tables ok: leaves={len(LEAF)} SW={SW} WP={WP}")


# ---- tiny-n2 constants: with the iterative (capped-at-15) flags, n2=v<=3
# normalizes to v<<15 with z=15, so a = z>>1 = 7 and the datapath holds the
# deterministic leaf intercept G_v pre-shift (rem bits all zero => prod=0).
# The pre-shift patch rewrites sreg to Q_v; post-shift the accumulate adds
# (Q_v<<5) + 2^23, which is within the inclusive +-32 band of F(v). ----
def leaf_f(v):
    return ((v << 15) >> (17 - A)) & ((1 << A) - 1)


GV = {v: G0f[leaf_f(v)] for v in (1, 2, 3)}
QV = {}
for _v in (1, 2, 3):
    _qv = min((REF(_v) - (1 << 23) + 16) >> 5, (1 << GW) - 1)
    assert 0 <= _qv < (1 << GW)
    assert abs(((_qv << 5) + (1 << 23)) - REF(_v)) <= EPS
    QV[_v] = _qv
# n2 = 1 clips QV at 2^18 - 1 and the circuit lands k + 2^24 - 32 —
# delta = -EPS exactly, the input that spends the whole band.
AV = {v: GV[v] ^ QV[v] for v in (1, 2, 3)}
XEc = AV[1] ^ AV[2] ^ AV[3]
XBc = AV[2] ^ AV[3]
XCc = AV[1] ^ AV[3]
for _c in (XEc, XBc, XCc):
    assert 0 <= _c < (1 << GW)


# ------------------------------------------------ squarer specs & DP plan
# d^2 = sum_i d_i 4^i + sum_{i<j} d_i d_j 2^{i+j+1} for 8-bit d, packed into
# four column groups G1..G4, each with at most ONE wire per column so a
# single sparse radd accumulates it.  Column 8 natively carries two terms
# (d_4 and d_0 d_7); they are merged into a half-adder pair
#   h8 = d_0 d_7 XOR d_4  (col 8),   h9 = d_0 d_7 AND d_4  (col 9),
# restoring the one-wire-per-column invariant.  The decomposition is
# checked exhaustively over all 256 d below.
def sq_specs():
    def Ap(i, j):
        return ("a", i, j)

    G1 = (
        0,
        {
            0: ("q", 0),
            2: Ap(0, 1),
            3: Ap(0, 2),
            4: Ap(0, 3),
            5: Ap(0, 4),
            6: Ap(0, 5),
            7: Ap(0, 6),
            8: ("h8",),
            9: ("h9",),
            10: Ap(4, 5),
            11: Ap(4, 6),
            12: Ap(4, 7),
            14: Ap(6, 7),
        },
    )
    G2 = (
        2,
        {
            2: ("q", 1),
            4: Ap(1, 2),
            5: Ap(1, 3),
            6: Ap(1, 4),
            7: Ap(1, 5),
            8: Ap(1, 6),
            9: Ap(1, 7),
            10: ("q", 5),
            12: Ap(5, 6),
            13: Ap(5, 7),
            14: ("q", 7),
        },
    )
    G3 = (
        4,
        {
            4: ("q", 2),
            6: Ap(2, 3),
            7: Ap(2, 4),
            8: Ap(2, 5),
            9: Ap(2, 6),
            10: Ap(2, 7),
            12: ("q", 6),
        },
    )
    G4 = (6, {6: ("q", 3), 8: Ap(3, 4), 9: Ap(3, 5), 10: Ap(3, 6), 11: Ap(3, 7)})
    return [G1, G2, G3, G4]


SPECS = sq_specs()


def spec_value(spec, d):
    db = [(d >> i) & 1 for i in range(8)]
    v = 0
    for pos, e in spec.items():
        if e[0] == "q":
            bit = db[e[1]]
        elif e[0] == "a":
            bit = db[e[1]] & db[e[2]]
        elif e[0] == "h8":
            bit = (db[0] & db[7]) ^ db[4]
        else:
            bit = db[0] & db[7] & db[4]
        v += bit << pos
    return v


for _d in range(256):
    assert sum(spec_value(SPECS[_g][1], _d) for _g in range(4)) == _d * _d


# Exact DP over the order in which the 11 remaining column groups (axis 0
# opens with G1 unconditionally — it seeds the n2 word for free) are
# accumulated.  After each accumulate, the maximum achievable value of the
# partial n2 bounds the carry cap: emitting group g (base column `off`)
# when the running total fits in `cap` bits costs cap-off-1 adder ANDs.
# States are per-axis subsets of emitted groups; axm[ax][mask] is the
# per-axis max-value contribution, so the cap bound is a plain sum.  The
# DP minimizes total adder ANDs; plan_squarer also recomputes the
# axis-major reference order (all of one axis's groups, then the next
# axis's) under the same cap rule, and the assert below fails the build
# if a spec change regresses the DP past it.
def plan_squarer():
    import itertools

    VALt = [[spec_value(SPECS[g][1], d) for d in range(256)] for g in range(4)]
    gl = [(1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)]
    axm = []
    for ax in range(3):
        n = len(gl[ax])
        arr = []
        for m in range(1 << n):
            best = 0
            for d in range(256):
                v = VALt[0][d] if ax == 0 else 0
                for pi in range(n):
                    if (m >> pi) & 1:
                        v += VALt[gl[ax][pi]][d]
                best = max(best, v)
            arr.append(best)
        axm.append(arr)
    INF = 10**9
    states = sorted(
        itertools.product(range(8), range(16), range(16)),
        key=lambda s: bin(s[0] | (s[1] << 3) | (s[2] << 7)).count("1"),
    )
    F = {(0, 0, 0): 0}
    PRE = {}
    for st in states:
        if st not in F:
            continue
        cur = F[st]
        for ax in range(3):
            for pi, g in enumerate(gl[ax]):
                if (st[ax] >> pi) & 1:
                    continue
                nm0 = [st[0], st[1], st[2]]
                nm0[ax] |= 1 << pi
                nm = tuple(nm0)
                bound = axm[0][nm[0]] + axm[1][nm[1]] + axm[2][nm[2]]
                cap = bound.bit_length()
                off = SPECS[g][0]
                top = max(SPECS[g][1])
                assert top + 1 <= cap <= 18
                cost = cap - off - 1
                if cur + cost < F.get(nm, INF):
                    F[nm] = cur + cost
                    PRE[nm] = (st, ax, g, cap)
    goal = (7, 15, 15)
    sched = []
    st = goal
    while st != (0, 0, 0):
        p, ax, g, cap = PRE[st]
        sched.append((ax, g, cap))
        st = p
    sched.reverse()
    # the axis-major reference cost under the same cap rule (the DP cost
    # must be at most this)
    axc = 0
    masks = [0, 0, 0]
    for ax in range(3):
        for pi, g in enumerate(gl[ax]):
            masks[ax] |= 1 << pi
            bound = axm[0][masks[0]] + axm[1][masks[1]] + axm[2][masks[2]]
            axc += bound.bit_length() - SPECS[g][0] - 1
    return sched, F[goal], axc


SQ_SCHED, SQCOST, AXCOST = plan_squarer()
print(f"squarer plan: adder-ANDs={SQCOST} (axis-major {AXCOST})")
assert SQCOST <= AXCOST


# ---------------------------------------- Fredkin shifter rows (stages 3 & 8)
# Both shifters are TRUNCATED: each row moves only the window a full shift
# could populate.  normalize rows (s_amt, ereg index, window [lo, hi)) stop
# at lo = 0/0/RL-3/RL-1, so mantissa positions below RL may keep stale
# unshifted bits — no read position sources from a stale bit: the QROM
# address, the remainder rows, and the patch discriminators all sit at
# positions >= RL; the stage-8 rows (s_amt, ereg index, lim)
# cap at lim = 18/19/21, the highest reachable top bit per row.  build()
# emits exactly these tables.
NORM_ROWS = [(8, 3, 0, 10), (4, 2, 0, 14), (2, 1, RL - 3, 16), (1, 0, RL - 1, 17)]
SHIFT_ROWS = [(1, 1, GW), (2, 2, GW + 1), (4, 3, GW + 3)]


def build():  # noqa: C901, PLR0915 - straight-line circuit emitter
    """Construct the discovered circuit on a fresh Builder.

    Returns (b, RI, RJ, KQ, n2b): the Builder, the ri/rj axis
    registers, the k register, and the n2 word's wires.  Also the entry
    point of tools/occupancy_profile.py, which replays the ownership
    wire-set log fail-closed and charts the region occupancy
    (``circ, *_ = mod.build()``).
    """
    # ------------------------------------------- Builder + the mirror journal
    # The Builder is live from here on: every gadget call below emits ops
    # directly.  Register ids follow the grader schema in order — ri axes
    # 0..2 on qubits 0..23, rj on 24..47, k on 48..72 (six 8-qubit
    # registers, then the 25-bit accumulator).  The evaluator writes and
    # reads registers by that schema order, so these seven declarations
    # must come first and in this order.
    b = Builder()
    _r = declare_registers(
        b, [(n, B_C) for n in ("rix", "riy", "riz", "rjx", "rjy", "rjz")] + [("k", M_K)]
    )
    RI = [_r["rix"], _r["riy"], _r["riz"]]
    RJ = [_r["rjx"], _r["rjy"], _r["rjz"]]
    KQ = _r["k"]
    b.claim("ri", RI[0] + RI[1] + RI[2])
    b.claim("rj", RJ[0] + RJ[1] + RJ[2])
    b.claim("k", KQ)

    # The Builder journals the semantic structure itself: checkpoint() takes an
    # opaque Mark, and mirror(since, until=) emits the exact Bennett inverse of
    # the region between two marks — and_compute <-> and_release, alloc <->
    # free (the LIFO free-list returns every mirrored ancilla to the same
    # physical slot, asserted by the Builder), X/CX/CCX re-emitted verbatim.
    # ---------------------------------------------------------------- adders
    # radd: Gidney temporary-AND ripple adder, acc += addend (+ optional cin;
    #   carry XORed onto cout).  addend may contain None holes for sparse
    #   operands.  Its recs list journals each carry step so the unwind can
    #   emit the exact inverse — the same journal-and-replay discipline
    #   b.mirror() applies to the whole forward region, at gadget scale.
    #   local: shared arith.add emits a different op sequence — it ANDs (operand,
    #   target) where radd ANDs (target, addend), diverging at the first op
    #   (36 vs 38 ops on a 4-bit cout add, measured) — and its cin only
    #   seeds the [CDKM] corner, while subtract/hi9 rides a live Gidney
    #   carry-in (cin=bq1).
    # cadd: controlled acc += addend under q via one shared AND ladder (the
    #   in-place Gidney-chain controlled adder of pipeline stage 5).
    #   local: shared arith.add(ctrl=) is a different circuit.  It fuses
    #   the control into every position, so it holds about 2n transient
    #   wires where cadd holds n.  It takes no cout, so a carry-out row
    #   needs one more target bit.  That costs 2n Toffoli against cadd's
    #   2n+1; rows without cout tie at 2n-1 (measured, n = 8..16).
    # The accumulate (stage 9) calls the shared uncompute.gadgets.cdkm_add_co
    #   directly: the tail runs at the 142-qubit peak
    #   with zero width headroom, and a Gidney chain would hold ~23 carry
    #   ANDs live at that moment.
    def radd(acc, addend, cin=None, cout=None):
        n = len(acc)
        assert len(addend) <= n
        full = cout is not None
        recs: list[tuple] = []
        c = cin
        for i in range(n):
            bi = addend[i] if i < len(addend) else None
            ai = acc[i]
            if i == n - 1 and not full:
                if bi is not None:
                    b.cx(bi, ai)
                if c is not None:
                    b.cx(c, ai)
                break
            if bi is None and c is None:
                continue
            if bi is None:
                t = b.and_compute(c, ai)
                b.cx(c, ai)
                recs.append(("u", t, c, ai))
            elif c is None:
                t = b.and_compute(ai, bi)
                b.cx(bi, ai)
                recs.append(("f", t, ai, bi))
            else:
                b.cx(c, ai)
                b.cx(c, bi)
                t = b.and_compute(ai, bi)
                b.cx(c, t)
                recs.append(("m", t, ai, bi, c))
            c = t
        if cout is not None and c is not None and recs:
            b.cx(c, cout)
        for rec_ in reversed(recs):
            kind = rec_[0]
            if kind == "u":
                _, t, cp, ai = rec_
                b.cx(cp, ai)
                b.and_release(t, cp, ai)
                b.cx(cp, ai)
            elif kind == "f":
                _, t, ai, bi = rec_
                b.cx(bi, ai)
                b.and_release(t, ai, bi)
                b.cx(bi, ai)
            else:
                _, t, ai, bi, cp = rec_
                b.cx(cp, t)
                b.and_release(t, ai, bi)
                b.cx(cp, bi)
                b.cx(bi, ai)

    def cadd(q, av, tv, cout):
        n = len(av)
        top = n if cout is not None else n - 1
        m_ = [None] * max(top, 0)
        for j in range(n):
            if j:
                b.cx(m_[j - 1], av[j])
                b.cx(m_[j - 1], tv[j])
            if j < top:
                m_[j] = b.and_compute(av[j], tv[j])
                if j:
                    b.cx(m_[j - 1], m_[j])
        if cout is not None:
            b.ccx(q, m_[n - 1], cout)
        for j in reversed(range(n)):
            if j < top:
                if j:
                    b.cx(m_[j - 1], m_[j])
                b.and_release(m_[j], av[j], tv[j])
            b.ccx(q, av[j], tv[j])
            if j:
                b.cx(m_[j - 1], tv[j])
                b.cx(m_[j - 1], av[j])

    # ---------------------------------------------------------------- circuit
    # ---- 1. in-place diffs ri <- |ri-rj|, sign kept on SGN ----
    _M0 = b.checkpoint()  # forward-region start (everything until _MFWD mirrors)
    with b.phase("diffs/setup"):
        SGN = [b.qubit() for _ in range(3)]
        b.claim("sign", SGN)
    for ax in range(3):
        with b.phase(f"diffs/ax{ax}"):
            riq = RI[ax]
            rjq = RJ[ax]
            sg = SGN[ax]
            for q in riq:
                b.x(q)
            radd(riq, rjq, cout=sg)
            for q in riq:
                b.x(q)
            for q in riq:
                b.cx(sg, q)
            radd(riq, [sg])

    # ---- 2. n2 = sum of squares (DP-scheduled column-packed squarer) ----
    _MSQ = b.checkpoint()
    n2b = [None] * 18

    def g1_axis0():
        # Axis 0's group 1 goes straight into freshly allocated n2b wires — the
        # first group needs no adder at all (CX/AND writes onto zeros).  The
        # wiring derives from the same G1 spec emit_group consumes: the 'q'
        # bit lands by CX, the spec's holes become fresh |0> allocations
        # (ascending), the pair products follow in spec order, and the h8/h9
        # half-adder pair comes LAST — emit_group computes that pair FIRST;
        # the certified stream pins each order.
        spec = SPECS[0][1]
        D = RI[0]
        (qpos,) = (p for p, e in spec.items() if e[0] == "q")
        n2b[qpos] = b.qubit()
        b.cx(D[spec[qpos][1]], n2b[qpos])
        for pos in sorted(set(range(len(n2b))) - set(spec)):
            n2b[pos] = b.qubit()
        for pos, e in spec.items():
            if e[0] == "a":
                n2b[pos] = b.and_compute(D[e[1]], D[e[2]])
        (h8p,) = (p for p, e in spec.items() if e[0] == "h8")
        (h9p,) = (p for p, e in spec.items() if e[0] == "h9")
        n2b[h8p] = b.and_compute(D[0], D[7])
        n2b[h9p] = b.and_compute(n2b[h8p], D[4])
        b.cx(D[4], n2b[h8p])

    def emit_group(D, off, spec, cap):
        # accumulate one column group into n2b[off:cap] with a single sparse
        # radd; pair-product ANDs are released again right after (the h8/h9
        # half-adder pair needs its own compute/uncompute choreography)
        p07 = t89 = None
        if 8 in spec and spec[8][0] == "h8":
            p07 = b.and_compute(D[0], D[7])
            t89 = b.and_compute(p07, D[4])
            b.cx(D[4], p07)
        prods = []
        xb = []
        for pos in range(off, cap):
            e = spec.get(pos)
            if e is None:
                xb.append(None)
            elif e[0] == "q":
                xb.append(D[e[1]])
            elif e[0] == "h8":
                xb.append(p07)
            elif e[0] == "h9":
                xb.append(t89)
            else:
                t = b.and_compute(D[e[1]], D[e[2]])
                prods.append((D[e[1]], D[e[2]], t))
                xb.append(t)
        radd(n2b[off:cap], xb)
        for u, v, t in reversed(prods):
            b.and_release(t, u, v)
        if p07 is not None:
            b.cx(D[4], p07)
            b.and_release(t89, p07, D[4])
            b.and_release(p07, D[0], D[7])

    with b.phase("squarer/a0g1"):
        g1_axis0()
        b.claim("n2", n2b)  # all 18 wires exist once a0g1 has seeded the word
    for _ax, _g, _cap in SQ_SCHED:
        with b.phase(f"squarer/a{_ax}g{_g + 1}"):
            emit_group(RI[_ax], SPECS[_g][0], SPECS[_g][1], _cap)

    # ---- 3. normalize: iterative binary-search flags + staged cond. shifts ----
    # z = 8*ereg[3] + 4*ereg[2] + 2*ereg[1] + ereg[0] is the left shift that
    # brings the top set bit of n2 to bit 17; e = 17 - z.  The shift is capped
    # at 15 (= 8+4+2+1), enough for every n2 >= 4; n2 <= 3 under-shifts and is
    # corrected by the tiny-n2 patch in stage 7.  Each row's AND ladder
    # flags that the current top s_amt bits of n2b are all zero, and one CX
    # flips the row's ereg bit under the flag.
    _MNORM = b.checkpoint()
    with b.phase("normalize/setup"):
        ereg = [b.qubit() for _ in range(4)]
        b.claim("exp-flag", ereg)

    # Shared arith.mcx would close the ladder with b.ccx(ladder, last,
    # fq) — same s_amt-1 Toffoli, one fewer transient AND wire; this
    # full-ladder + CX form is what the certified stream pins, and the
    # extra wire is metric-free (normalize peaks at 105 of the 142 peak).
    for s_amt, jz, lo, hi in NORM_ROWS:
        with b.phase(f"normalize/z{s_amt}"):
            flag, rec = and_ladder(b, [(n2b[17 - t], 0) for t in range(s_amt)])
            b.cx(flag, ereg[jz])
            and_ladder_release(b, rec)
            for i in reversed(range(lo, hi)):
                # operand order fixes the stream encoding (swap is
                # semantically symmetric; the certified stream pins it)
                b.cswap(ereg[jz], n2b[i], n2b[i + s_amt])

    # ---- 4. adaptive QROM ----
    _MQROM = b.checkpoint()
    with b.phase("qrom/setup"):
        slope = [b.qubit() for _ in range(SW)]
        sreg = [b.qubit() for _ in range(GLOW)]
        b.claim("slope", slope)
        b.claim("intercept", sreg)
        addr_bits = [ereg[0]] + [n2b[16 - i] for i in range(A)]

        # Early-terminating binary QROM over addr_bits = (e&1, top mantissa bits):
        # each internal node costs one AND (Gidney) or one CX-pair (root); at a
        # leaf, the intercept — msb stripped, it is implicit — and the slope are
        # CX-copied onto sreg/slope under the path flag.  Leaves sit exactly as
        # shallow as build_tables' interval fitting allowed — under the R=6
        # economics that is the full-depth 128-leaf tree.
        # local: shared qrom.pruned_walk costs the same one AND per live
        # internal node (126 on this 128-leaf tree) but roots on the bare address
        # bit, clear child first; this root builds a CX-parity wire and takes
        # the set-parity child first — the certified stream pins that order.
        # The root parity wire is one extra transient, metric-free (qrom peaks
        # below the 142 peak).  The un-qrom repair walk reuses the same walker.
        def walk(visit, flag, depth, prefix):
            # visit(flag, depth, prefix) serves a leaf and returns True, or
            # returns False to descend
            if visit(flag, depth, prefix):
                return
            bq = addr_bits[depth]
            ch = b.and_compute(flag, bq)
            walk(visit, ch, depth + 1, 2 * prefix + 1)
            b.cx(flag, ch)
            walk(visit, ch, depth + 1, 2 * prefix)
            b.cx(flag, ch)
            b.and_release(ch, flag, bq)

        def walk_root(visit, phase=None):
            # phase names the sub-phases of the root's two children and close
            b.x(ereg[0])  # address parity bit: e&1 = (17-z)&1 = NOT ereg[0]
            rt = b.qubit()
            b.claim("qrom-root", [rt])
            b.cx(addr_bits[0], rt)

            def sub(name):
                return b.phase(f"{phase}/{name}") if phase else nullcontext()

            with sub("parity1"):
                walk(visit, rt, 1, 1)
                b.x(rt)
            with sub("parity0"):
                walk(visit, rt, 1, 0)
                b.x(rt)
            with sub("close"):
                b.cx(addr_bits[0], rt)
                b.release_recycle("qrom-root", [rt])
                b.x(ereg[0])

        def load_leaf(flag, depth, prefix):
            lf = LEAF.get((depth, prefix))
            if lf is None:
                return False
            sl, g = lf
            xor_word(b, flag, g - HBIT, sreg)
            xor_word(b, flag, sl, slope)
            return True

        walk_root(load_leaf, "qrom")

    # ---- 5. multiply: prod = sum_i rem_i * floor(slope*2^i/4) ----
    # Row 0 lands as fresh ANDs onto the new product wires; rows 1..5 are
    # in-place controlled adds (cadd).  _acc tracks the maximum achievable
    # running sum, proving each row's carry-out target is still zero when used.
    _MMUL = b.checkpoint()
    with b.phase("multiply/row0"):
        rb0 = n2b[RL]
        prod = [b.and_compute(rb0, q2) for q2 in slope[SH:]]
        prod += [b.qubit() for _ in range(WP - len(prod))]
        b.claim("product", prod)
        _acc = (SMAX << 0) >> SH
    for i in range(1, R):
        with b.phase(f"multiply/row{i}"):
            rb = n2b[RL + i]
            sh = i - SH
            if sh < 0:
                avw = slope[-sh:]
                pos = 0
            else:
                avw = slope
                pos = sh
            n = len(avw)
            assert _acc < (1 << (pos + n)), "cout target not zero"
            cout = prod[pos + n] if pos + n < WP else None
            cadd(rb, avw, prod[pos : pos + n], cout)
            _acc += (SMAX << i) >> SH
    assert _acc < (1 << WP)

    # ---- 6. gs_low -= prod (17-bit chunked complement subtract, no overall
    # borrow); the chunk-borrow wire bq1 stays LIVE through the accumulate and
    # the exact mirror erases it for free — an early erase would pay the
    # borrow recompute twice, and lo8 pins the 142 peak either way ----
    _MSUB = b.checkpoint()
    with b.phase("subtract/lo8"):
        for j in range(GLOW):
            b.x(sreg[j])
        bq1 = b.qubit()
        b.claim("borrow", [bq1])
        radd(sreg[0:8], prod[0:8], cout=bq1)
    with b.phase("subtract/hi9"):
        radd(sreg[8:GLOW], prod[8:WP], cin=bq1)
    with b.phase("subtract/restore"):
        for j in range(GLOW):
            b.x(sreg[j])

    with b.phase("subtract/widen"):
        sreg += [b.qubit() for _ in range(SREGW - GLOW - 1)]
        b.claim("widen", sreg[GLOW:])
        b.x(sreg[GW - 1])

    # ---- 7. tiny-n2 patch (pre-shift): rewrite sreg from the deterministic
    # leaf intercept G_v to Q_v.  Flags are AND-allocated wires released by
    # free measurement-based uncomputes.  Discriminators (post-normalization,
    # n2<=3 only): n2=1 -> n2b[15]=1,n2b[16]=0; n2=2 -> 0,1; n2=3 -> 1,1. ----
    _MPATCH = b.checkpoint()
    with b.phase("patch"):
        e4 = n2b[17]
        b.x(e4)
        t3f = b.and_compute(e4, n2b[15])
        xor_word(b, t3f, XBc, sreg[:GW])
        b.and_release(t3f, e4, n2b[15])
        t3c = b.and_compute(e4, n2b[16])
        xor_word(b, t3c, XCc, sreg[:GW])
        b.and_release(t3c, e4, n2b[16])
        xor_word(b, e4, XEc, sreg[:GW])
        b.x(e4)
        sreg.append(b.qubit())
        b.claim("widen", [sreg[-1]])
        assert len(sreg) == SREGW

    # ---- 8. variable left shift ----
    _MSHIFT = b.checkpoint()
    for s_amt, jz, lim in SHIFT_ROWS:
        with b.phase(f"shift/s{s_amt}"):
            fq = ereg[jz]
            for p in reversed(range(lim)):
                # operands (lo, hi) — normalize's rows pass (hi, lo); the two
                # stream encodings differ deliberately, the semantics do not
                b.cswap(fq, sreg[p + s_amt], sreg[p])

    _MFWD = b.checkpoint()  # forward-region end; the accumulate is not mirrored

    # ---- 9. accumulate into k (not mirrored) ----
    # DELTA = round(sreg / 4): sreg[2:] is the integer part, sreg[1] enters as
    # the carry-in (round-to-nearest) — this is the /4-with-round of delta_of.
    # The b.x(n2b[17]) sandwich adds 2^23 exactly when the top mantissa bit is
    # CLEAR, i.e. on the under-normalized tiny-n2 (n2 <= 3) branch.
    with b.phase("accumulate"):
        b.x(n2b[17])
        # cin = sreg[1] is a LIVE data wire: add()'s MAJ seed carries cin's
        # value (added, then UMA-restored) — that is the advertised
        # round-to-nearest.  The gadget docstring's |0> carry-in wording is its
        # zero-ancilla corner, not a precondition here.
        cdkm_add_co(b, sreg[2:25] + [n2b[17]], KQ[0:24], KQ[24], sreg[1])
        b.x(n2b[17])

    # ---- mirror: the conjugation F . accumulate . F^-1 ----
    # Segment mirrors, one per forward stage in reverse order, emit the exact
    # Bennett inverse of every forward stage EXCEPT the QROM: the un-qrom
    # segment below erases the table words by X-basis demolition and repays
    # the phase with one floored repair walk, so the second tree walk the
    # mirror would re-bill never runs.  The segmentation carries the
    # ownership discipline: each set releases just before the segment whose
    # frees consume its wires, so release precedes free through the whole
    # teardown.  Each b.phase block names one teardown step and opens before
    # its releases (emits no op), so the releases and the segment's tail
    # frees attribute to the un-* phase rather than to accumulate; the
    # reflected sub-phase labels come from each segment's own marks.
    with b.phase("un-shift"):
        b.mirror(_MSHIFT, until=_MFWD)  # frees nothing
    with b.phase("un-patch"):
        b.release("widen", [sreg[SREGW - 1]])
        b.mirror(_MPATCH, until=_MSHIFT)  # frees the patch widen wire
    with b.phase("un-subtract"):
        b.release("widen", sreg[GLOW : SREGW - 1])
        b.release("borrow", [bq1])
        b.mirror(_MSUB, until=_MPATCH)  # frees bq1 + widen wires
    with b.phase("un-multiply"):
        b.release("product", prod)
        b.mirror(_MMUL, until=_MSUB)  # releases/frees prod
    # ---- un-qrom: measurement-based unlookup + floored phase-repair walk ----
    # After un-multiply the slope/intercept wires hold exactly the looked-up
    # table words again — a pure function of the address bits, all still
    # live.  Each X-basis demolition (hmr) is free, forces its wire to |0>,
    # and leaves the shot's sign multiplied by (-1)^<outcomes, word(addr)>;
    # the walk below repays that mask, so clearing the words costs 17
    # Toffoli instead of the mirror's second 126-AND tree walk.  Frees run
    # in reverse alloc order — the LIFO free-list state the later mirrors
    # assert is exactly the one the mirrored unlookup would have left.
    with b.phase("un-qrom/measure"):
        b.release("intercept", sreg[:GLOW])
        b.release("slope", slope)
        m_sreg = [None] * GLOW
        m_slope = [None] * SW
        for j in reversed(range(GLOW)):
            m_sreg[j] = b.hmr(sreg[j])
            b.recycle(sreg[j])
        for j in reversed(range(SW)):
            m_slope[j] = b.hmr(slope[j])
            b.recycle(slope[j])

        # Per-flag repair coefficients in the monomial (Reed-Muller) basis.
        # word(f) packs the 24 measured table bits of leaf f; W(p, S) XORs the
        # words over the low-nibble subcube v <= S, so the phase to repay at
        # flag p factors as  sum_S <outcomes, W(p, S)> * mono_S(low bits).
        # The Mobius transform runs on the classical table at build time, and
        # the runtime coefficients stay plain outcome XOR chains (free OXORs;
        # None = zero mask, nothing to repay).  Nibble bit FL_M-1-t of v and
        # of S corresponds to low[t], matching the walk's msb-first prefixes.
        FL_M = 4  # low address bits served by monomials
        FL_D = 7 - FL_M  # repair-walk depth (high address bits)
        low = addr_bits[FL_D:]

        def word(f):
            return (G0f[f] - HBIT) | (SLf[f] << GLOW)

        def gamma(p, S):
            w = 0
            for v in range(1 << FL_M):
                if v & ~S == 0:
                    w ^= word((p << FL_M) | v)
            ms = [m_sreg[j] for j in range(GLOW) if (w >> j) & 1]
            ms += [m_slope[j] for j in range(SW) if (w >> (GLOW + j)) & 1]
            acc = None
            for m2 in ms:
                acc = m2 if acc is None else acc ^ m2
            return acc

        GAM = [[gamma(p, S) for S in range(1 << FL_M)] for p in range(1 << FL_D)]

    # Monomial bank over the FL_M low address bits (m[14..11]): one AND
    # wire per product of two or more bits — six pairs, four triples (a
    # pair wire AND one more bit), one quad — 11 Toffoli.  Singleton
    # monomials are the address wires themselves and the bank floors the
    # repair walk: leaves below depth FL_D are served entirely by free
    # conditioned phase gates, so repair cost stops growing with leaf
    # count — the economics the R=6 refit banks on.
    with b.phase("un-qrom/monomials"):

        def smask(ts):  # subset of low-bit indices -> nibble-style mask
            v = 0
            for t in ts:
                v |= 1 << (FL_M - 1 - t)
            return v

        MONO = {}
        MORDER = []
        for ts in (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 2, 3),
        ):
            base = MONO[smask(ts[:-1])] if len(ts) > 2 else low[ts[0]]
            w = b.and_compute(base, low[ts[-1]])
            MONO[smask(ts)] = w
            MORDER.append((w, base, low[ts[-1]]))

    # Floored repair walk: path ANDs over the three high address bits only
    # (2 + 4 = 6); at each depth-3 flag the sixteen Reed-Muller
    # coefficients condition free phase gates — Z on the flag for S = 0,
    # CZ against a bare low address wire or a bank monomial otherwise.
    # 6 + 11 = 17 Toffoli repays the whole unlookup.  The forward walker
    # runs it, so the root CX-parity trick and child order match.
    with b.phase("un-qrom/fix"):

        def repair_leaf(flag, depth, prefix):
            if depth != FL_D:
                return False
            g = GAM[prefix]
            if g[0] is not None:
                b.z(flag, cond=g[0])
            for S in range(1, 1 << FL_M):
                if g[S] is None:
                    continue
                if S & (S - 1) == 0:
                    b.cz(flag, low[FL_M - S.bit_length()], cond=g[S])
                else:
                    b.cz(flag, MONO[S], cond=g[S])
            return True

        walk_root(repair_leaf)

    # Unwind the monomial bank: free measurement-based releases in exact
    # reverse alloc order (bases outlive their products), restoring the
    # free-list for the mirrors below.
    with b.phase("un-qrom/close"):
        for w, c1, c2 in reversed(MORDER):
            b.and_release(w, c1, c2)

    with b.phase("un-normalize"):
        b.release("exp-flag", ereg)
        b.mirror(_MNORM, until=_MQROM)  # frees ereg
    with b.phase("un-squarer"):
        b.release("n2", n2b)
        b.mirror(_MSQ, until=_MNORM)  # releases/frees n2
    with b.phase("un-diffs"):
        b.release("sign", SGN)
        b.mirror(_M0, until=_MSQ)  # frees the sign wires
    return b, RI, RJ, KQ, n2b


b, RI, RJ, KQ, n2b = build()

# ---------------------------------------------------------------- accounting
# io_qubits 73 = ri (3x8) + rj (3x8) + k (25)
assert_metrics(b, __file__)

# ---------------------------------------------------------------- emission
if __name__ == "__main__":
    emit_and_report(b)
