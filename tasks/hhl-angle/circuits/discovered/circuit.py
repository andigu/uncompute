#!/usr/bin/env python3
"""hhl-angle circuits/discovered: a 345-leaf adaptive
piecewise-quadratic QROM with both multiplies run as signed Booth
radix-4 digit rows — zero-alloc in-place recode, pair-select addend
columns demolished inline — under a bias-capped v-subtraction and a
deferred conditioned-CZ fixup walk.

Certified cost: 1,348 Toffoli x 115 qubits = 155,020 (assert_metrics
in the epilogue).

RUN: uv run python circuit.py   (stream to $UNCOMPUTE_OPS_PATH, else ./ops.bin)
     uv run python checks.py   (exhaustive worst-case error proof)

README.md in this folder describes the circuit.
"""

import contextlib
import json
import math
import pathlib

import numpy as np

from uncompute import Builder
from uncompute.gadgets import (
    add,
    booth_columns,
    booth_columns_release,
    declare_registers,
    gidney_add,
)
from uncompute.testing import assert_metrics, emit_and_report, load_contract

LO, HI = 1 << 16, 1 << 21
MOD = 1 << 33

# ---------------------------------------------------------------------------
# Shared task contract: grader.py asserts contract.json's derived
# constants against its own inline derivation.
# ---------------------------------------------------------------------------
CONTRACT = load_contract(__file__)
EPS = CONTRACT["eps_ulp"]  # the task band, 429 output units

# ---------------------------------------------------------------------------
# Frozen chosen config + leaf table (constants.json; gen_constants.py
# beside this file documents the fields, runs the searches, and writes
# the file)
# ---------------------------------------------------------------------------
_CONST = json.loads((pathlib.Path(__file__).resolve().parent / "constants.json").read_text())
CFG = _CONST["cfg"]
TAB = {k: np.asarray(v, dtype=np.int64) for k, v in _CONST["leaves"].items()}

NDN = CFG["NDN"]
GUARD = CFG["GUARD"]
RPROD = CFG["RPROD"]
G1W = CFG["G1W"]
TAU = CFG["TAU"]
RQ = CFG["RQ"]
G2W = CFG["G2W"]
J0 = CFG["J0"]
C0D = CFG["C0DROP"]
QCAP = CFG["QCAP"]
B4 = CFG["B4"]
VORD = CFG["VORD"]
C0GL = CFG["C0GL"]
PCAP = CFG["PCAP"]
VW = G1W + GUARD  # v register width (guards + slope word)
NDQ = NDN - TAU  # quadratic coordinate width
C0W = 32 - C0D  # stored intercept width
BIAS = 1 << QCAP  # the slope-word bias 2^16
GLIM = (1 << G1W) - (BIAS >> GUARD)  # g1 range so W fits in G1W+GUARD bits
# Per-digit product truncations: row t adds d_t (v >> sig_t) << B4[t],
# the |d|=2 candidate one v bit deeper (magnitude folded into the shift
# before truncation).
SIG = [B4[t] + RPROD - J0 - 2 * t for t in range(7)]
assert QCAP + 1 <= VW and QCAP >= RPROD
assert len(B4) == 7
assert sorted(VORD) == [0, 1, 2] and len(VORD) == 3

# ---------------------------------------------------------------------------
# Leaf tree.  The frozen rows are an exact dyadic partition of [LO, HI);
# the rebuilt walk order — leftmost leaf first within each octave,
# octaves 2^16..2^20 in order — is load-bearing: it is the visit order
# of the certified stream.
# ---------------------------------------------------------------------------


class Leaf:
    __slots__ = ("bb", "coef", "lo")

    def __init__(self, lo, bb, coef):
        self.lo, self.bb, self.coef = lo, bb, coef


class Node:
    __slots__ = ("bit", "left", "right")

    def __init__(self, bit, left, right):
        self.bit, self.left, self.right = bit, left, right


LSET = {
    (int(lo), int(bb)): (int(g0), int(g1), int(g2), int(sh))
    for lo, bb, g0, g1, g2, sh in zip(
        TAB["lo"], TAB["bb"], TAB["g0"], TAB["g1"], TAB["g2"], TAB["sh"]
    )
}


def tree_from_leafset(lset):
    def go(lo, bb):
        if (lo, bb) in lset:
            return Leaf(lo, bb, lset[(lo, bb)])
        assert bb > 7, "frozen table is not a dyadic partition"
        h = 1 << (bb - 1)
        return Node(bb - 1, go(lo, bb - 1), go(lo + h, bb - 1))

    Ts = [go(1 << (16 + e), 16 + e) for e in range(5)]
    return Node(20, Node(19, Node(18, Node(17, Ts[0], Ts[1]), Ts[2]), Ts[3]), Ts[4])


def collect(nd, out=None):
    if out is None:
        out = []
    if isinstance(nd, Leaf):
        out.append(nd)
    else:
        collect(nd.left, out)
        collect(nd.right, out)
    return out


ROOT = tree_from_leafset(LSET)
LEAVES = collect(ROOT)

# ---------------------------------------------------------------------------
# Builder-native measurement-based erasure: b.hmr(w) demolishes a wire into
# an Outcome handle and b.cz(a, c, cond=m) hosts the deferred phase fixup.
# The Booth rows' [GID]-prefix dial reads the *current* live count against
# the frozen live-qubit cap through b.active_qubits.  That accounting is
# load-bearing: the prefix schedule is part of the certified stream, so
# any change to what counts as live moves the emitted bytes.
# ---------------------------------------------------------------------------


def build():  # noqa: C901, PLR0915 - straight-line circuit emitter
    """Build the discovered circuit on a fresh Builder; return (b, regs).

    Circuit helpers are closures over ``b``.  Registers come first, so
    Builder qubit ids 0..53 are the persistent I/O wires.  build() is
    also the entry point of tools/occupancy_profile.py, which replays
    the ownership wire-set log fail-closed and charts the occupancy.
    """
    b = Builder()
    _r = declare_registers(b, [("X", 21), ("k", 33)])  # |X> preserved; |k> acc
    XQ, KQ = _r["X"], _r["k"]
    b.claim("X", XQ)
    b.claim("k", KQ)

    # ---------------------------------------------------------------------------
    # Local arithmetic form: the signed Booth digit row.  The shared
    # general adder is uncompute.gadgets.arith.add (its neg input runs the
    # conditional negate + sign extension, its gl dial the [GID]/[CDKM]
    # carry split), and uncompute.gadgets.arith.booth_columns builds the
    # pair-select addend columns and demolishes them; the row supplies
    # the recode and the |d|=2 selector.
    # ---------------------------------------------------------------------------

    def brow(recode_cxs, s1w, m1w, signw, pairs, T, gl):
        """T += d * (operand row) — one signed Booth radix-4 digit row.

        The digit's three source wires are recoded in place by
        ``recode_cxs`` (allocating nothing): ``s1w`` ends up holding the
        |d|=1 selector (the XOR of the low source pair), ``m1w`` the XOR
        of the high pair, and ``signw`` — a still-intact source wire —
        the digit sign, fed to the adder's negate input.  The |d|=2
        selector m2 = (NOT s1w) AND m1w is the row's only non-column AND.

        Each addend column is the pair-select (s1w AND v_a) XOR
        (m2 AND v_b) — v_a, v_b the |d|=1 / |d|=2 candidate operand bits,
        one shift apart — built X-sandwiched into one's-complement form
        under the sign wire.  Right after the adder returns, every column
        AND is demolished by measurement and its phase kick repaired
        inline by two conditioned CZs on the still-live operand pair.

        ``gl`` is the adder's [GID]-prefix length (the (qubits, Toffoli)
        dial): prefix carries ride measured ANDs at 1 Toffoli + 1 live
        wire, the rest of the addend span rides [CDKM] MAJ/UMA carries in
        the column temps at 2 Toffoli + 0 wires, and the sign-extension
        tail is always [GID].
        """
        for u, v in recode_cxs:
            b.cx(u, v)
        m2 = b.and_compute(s1w, m1w, pol=(False, True))
        cols, reps = booth_columns(b, s1w, m2, pairs, neg=signw)
        add(b, T, cols, neg=signw, gl=gl)
        booth_columns_release(b, cols, reps, neg=signw)
        b.and_release(m2, s1w, m1w, pol=(False, True))
        for u, v in reversed(recode_cxs):
            b.cx(u, v)

    def row_gl(m, n):
        """[GID]-prefix length for a capped product row.  At the chain
        top, m column temps + the digit AND + the gl prefix carries + the
        (n - 1 - m) sign-extension tail carries are all live above the
        stage base (n live wires at gl = 0), so the prefix takes what the
        frozen cap leaves, two wires of headroom kept under it.
        """
        return max(0, min(m, PCAP - b.active_qubits - n - 2))

    # ---------------------------------------------------------------------------
    # The circuit — emitted gate by gate through a live Builder.  Registers
    # first: Builder qubit ids 0..53 are the persistent I/O wires.
    # ---------------------------------------------------------------------------

    def crotl(ctrl, amt):
        """Rotate X[0..14] left by ``amt`` under ``ctrl``, as disjoint swap
        cycles of Fredkins (Builder.cswap, 1 Toffoli per swapped pair; its
        first swap operand is the CX pivot, and the operand order is
        load-bearing — swapping it is semantically equal but moves the
        certified stream): 14 Toffoli per rotation (gcd(15, amt) = 1,
        asserted — one 15-cycle, 14 swapped pairs).
        """
        n = NDN
        amt = amt % n
        if amt == 0:
            return
        assert math.gcd(n, amt) == 1
        seen = [False] * n
        for s in range(n):
            if seen[s]:
                continue
            cyc = []
            i = s
            while not seen[i]:
                seen[i] = True
                cyc.append(i)
                i = (i + amt) % n
            for j in range(len(cyc) - 1, 0, -1):
                b.cswap(ctrl, XQ[cyc[j - 1]], XQ[cyc[j]])

    def leaf_bits(lf):
        """Return the 1-bits of a leaf's four stored words, as (word, bit) pairs
        in emission order: g2, sh, W >> GUARD, g0 >> C0D.  The single
        source for the lookup writes (visit1) and the fixup kick list
        (leaf_mbs) — the two sides must stay in exact lockstep or the
        fixup mis-repairs the demolition kickback.
        """
        g0, g1, g2, sh = lf.coef
        # stored slope word: W >> GUARD = g1 + (BIAS >> GUARD)
        for w, (word, width) in enumerate(
            ((g2, G2W), (sh, 3), (g1 + (BIAS >> GUARD), G1W), (g0 >> C0D, C0W))
        ):
            for i in range(width):
                if (word >> i) & 1:
                    yield w, i

    # ---- 1. lookup: adaptive QROM walk writes the coefficient words ----
    # Unary-iteration tree walk (one temporary AND per internal node, the
    # CX(ctrl, a) sibling-flip trick reuses it for both children — the QROM
    # with data-dependent depth): leaves sit exactly as shallow as the
    # certified error band allowed the fit to place them.
    # Coefficient wires materialize at their first writing leaf (the
    # walk below): a wire exists exactly from its first data bit on, and
    # a bit no leaf ever sets would fail the post-walk coverage assert
    # instead of riding as a dead wire.
    g2q = [None] * G2W
    shq = [None] * 3
    wsl = [None] * G1W
    c0q = [None] * C0W

    def _mat(bank, name, i):
        if bank[i] is None:
            bank[i] = b.qubit()
            b.claim(name, [bank[i]])
        return bank[i]

    def walk(visit, floor=-1):
        """Unary-iteration tree walk: ``visit(ctrl, leaf)`` fires once per
        leaf, with ``ctrl`` the wire that is 1 exactly on the leaf's X range
        (None above the uncontrolled top split).  One temporary AND per
        internal node; the CX(ctrl, a) sibling flip reuses each AND for both
        children.  Internal nodes below tree level ``floor`` are not
        descended: every leaf beneath fires with the node's own control (the
        fixup hands those leaves to the one-hot bank); floor=-1 descends
        everywhere.  Runs both the lookup (stage 1) and the fixup re-walk
        (stage 9).
        local: gadgets.qrom.pruned_walk expresses the same adaptive depth
        (uniform=) at the same 1 Toffoli + 1 AND per node, but visits the
        right child first and re-points each AND with 2 CX per node; this
        walk is left-first with the 1-CX sibling flip, and its visit order
        is the certified stream's.
        """

        def go(nd, ctrl):
            if isinstance(nd, Leaf):
                visit(ctrl, nd)
                return
            if nd.bit < floor:
                for lf in collect(nd):
                    visit(ctrl, lf)
                return
            bq = XQ[nd.bit]
            if ctrl is None:
                b.x(bq)
                go(nd.left, bq)
                b.x(bq)
                go(nd.right, bq)
            else:
                a = b.and_compute(ctrl, bq, pol=(True, False))
                go(nd.left, a)
                b.cx(ctrl, a)
                go(nd.right, a)
                b.and_release(a, ctrl, bq)

        go(ROOT, None)

    _oct = None
    # The open lookup/oct{o} phase block: an octave's phase runs from its
    # first leaf to the next octave's first leaf, walk ANDs included.
    octave = contextlib.ExitStack()

    def visit1(ctrl, lf):
        nonlocal _oct
        o = lf.lo.bit_length() - 17
        if o != _oct:
            _oct = o
            octave.close()
            octave.enter_context(b.phase(f"lookup/oct{o}"))
        for w, i in leaf_bits(lf):
            bank, nm = ((g2q, "g2"), (shq, "shift"), (wsl, "wslope"), (c0q, "intercept"))[w]
            b.cx(ctrl, _mat(bank, nm, i))

    with b.phase("lookup/setup"), octave:
        walk(visit1)
    assert all(q is not None for bank in (g2q, shq, wsl, c0q) for q in bank), (
        "coefficient bit never written by any leaf"
    )

    # ---- 2. rotate: align the leaf's varying bits into the coordinate dn ----
    with b.phase("rotate/s1"):
        crotl(shq[0], 1)
    with b.phase("rotate/s2"):
        crotl(shq[1], 2)
    with b.phase("rotate/s4"):
        crotl(shq[2], 4)

    dqb = [XQ[TAU + t] for t in range(NDQ)]  # dq = dn >> 3, on the X wires

    def demolish(rset, qs):
        """Demolish a coefficient word, top bit down: hmr + release + free
        per wire.  Returns the X-measurement bits, indexed like ``qs`` —
        the fixup's kick list (leaf_mbs).
        """
        mb = [None] * len(qs)
        for i in range(len(qs) - 1, -1, -1):
            mb[i] = b.hmr(qs[i])
            b.release_recycle(rset, [qs[i]])
        return mb

    # ---- 3. c0add: k += g0; demolish the intercept word ----
    # Gidney form with the carry chain truncated to gl=C0GL live AND
    # wires; the prefix length is a searched dial (gen_constants.py).  No
    # carry-in wire exists in this form.
    with b.phase("c0add/add"):
        add(b, KQ[C0D : C0D + len(c0q)], c0q, gl=C0GL, cout=KQ[C0D + len(c0q)])
    with b.phase("c0add/c0erase"):
        mb_c0 = demolish("intercept", c0q)

    # ---- Booth digit row tables --------------------------------------------
    # Product digits t = 0..6 recode Y = dn >> 2 (Y bits y_j live on
    # XQ[j + J0]): d_t = y_{2t-1} + y_{2t} - 2 y_{2t+1}, boundary bits
    # zero.  After the in-place CXs, s1w holds the |d|=1 selector
    # y_{2t-1} ^ y_{2t}, m1w holds y_{2t} ^ y_{2t+1}, and the untouched
    # high source wire y_{2t+1} is the sign.  The boundary rows carry no
    # recode CX (t=0) and no sign (t=6: d_6 = y_11 + y_12 >= 0).

    def prod_row_spec(t):
        lo, mid, hi = XQ[2 * t + 1], XQ[2 * t + 2], XQ[2 * t + 3] if 2 * t + 3 <= NDN - 1 else None
        if t == 0:
            return [], mid, hi, hi
        if hi is None:
            return [(mid, lo)], lo, mid, None
        return [(mid, lo), (hi, mid)], lo, mid, hi

    def prod_pairs(t):
        sig = SIG[t]
        out = []
        for c in range(VW - sig + 1):
            va = vq[sig + c] if sig + c <= VW - 1 else None
            vb = vq[sig - 1 + c]
            out.append((va, vb))
        return out

    # Curvature digits s = 0..2 recode g2 (5 bits, VORD emission order):
    # e_s = g2_{2s-1} + g2_{2s} - 2 g2_{2s+1}, boundary bits zero; row s
    # adds e_s (dq << 2s) >> RQ into the 17-bit window, the |e|=2
    # candidate one dq bit deeper.

    def q_row_spec(s):
        if s == 0:
            return [], g2q[0], g2q[1], g2q[1]
        if s == 2:
            return [(g2q[4], g2q[3])], g2q[3], g2q[4], None
        return [(g2q[2], g2q[1]), (g2q[3], g2q[2])], g2q[1], g2q[2], g2q[3]

    def q_pairs(s):
        out = []
        for c in range(NDQ - RQ + 2 * s + 1):
            ia, ib = c + RQ - 2 * s, c + RQ - 2 * s - 1
            va = dqb[ia] if 0 <= ia <= NDQ - 1 else None
            vb = dqb[ib] if 0 <= ib <= NDQ - 1 else None
            out.append((va, vb))
        return out

    # ---- 4. vcompute: v = W - Q(dn) inside the 17-bit bias window ----
    # The X sandwich turns the signed digit rows into a two's-complement
    # subtract; the per-prefix window invariant (certified, frozen VORD
    # order) guarantees no carry ever crosses bit 16, so only
    # tgtv = vq[:17] is conjugated and every row chain is 17 bits wide.
    with b.phase("vcompute/setup"):
        guards = [b.qubit() for _ in range(GUARD)]
        b.claim("guard", guards)
        vq = list(guards) + list(wsl)
        tgtv = vq[: QCAP + 1]
        for q in tgtv:
            b.x(q)
    for s in VORD:
        with b.phase(f"vcompute/s{s}"):
            cxs, s1w, m1w, signw = q_row_spec(s)
            brow(cxs, s1w, m1w, signw, q_pairs(s), tgtv, None)
    with b.phase("vcompute/close"):
        for q in tgtv:
            b.x(q)

    # ---- 5. product: k -= P(dn, v), 7 signed Booth digit rows ----
    with b.phase("product/setup"):
        for q in KQ:
            b.x(q)
    for t in range(7):
        with b.phase(f"product/t{t}"):
            cxs, s1w, m1w, signw = prod_row_spec(t)
            m, n = VW - SIG[t] + 1, 33 - B4[t]
            brow(cxs, s1w, m1w, signw, prod_pairs(t), KQ[B4[t] :], row_gl(m, n))
    with b.phase("product/close"):
        for q in KQ:
            b.x(q)

    # ---- 6. comp: cancel the bias, k += (dn >> 2) << 7 ----
    # The Booth recode preserves sum d_t 4^t = dn >> 2 exactly, so every
    # product row carries the bias term of its digit and one carry-only
    # Gidney add removes them all exactly.
    with b.phase("comp"):
        # gadgets.arith gidney_add: 13 addend bits against a 26-bit target,
        # the zero tail rippling carry only — 25 Toffoli, every carry AND
        # uncomputed for free.
        gidney_add(b, [XQ[j] for j in range(J0, NDN)], KQ[QCAP - RPROD + J0 :])

    # ---- 7. vuncompute: mirrored rows restore W ----
    # No X sandwich: the same signed rows now add Q back.  The row order
    # is VORD reversed, so the uncompute prefixes revisit exactly the
    # certified compute prefixes (the window invariant covers both
    # directions with one frozen order).
    for s in reversed(VORD):
        with b.phase(f"vuncompute/s{s}"):
            cxs, s1w, m1w, signw = q_row_spec(s)
            brow(cxs, s1w, m1w, signw, q_pairs(s), tgtv, None)
            if s == VORD[0]:
                # Guards end |0> without measurement: stage 7's mirror restored
                # v = W exactly, and W mod 2^GUARD = 0 (asserted at werase below) —
                # freed here, after the last mirrored row, at the end of their
                # liveness window.
                b.release_recycle("guard", guards)

    # ---- 8. unrotate: restore X; demolish slope, shift, g2 words ----
    with b.phase("unrotate/s4"):
        crotl(shq[2], NDN - 4)
    with b.phase("unrotate/s2"):
        crotl(shq[1], NDN - 2)
    with b.phase("unrotate/s1"):
        crotl(shq[0], NDN - 1)
    with b.phase("unrotate/werase"):
        mb_g1 = demolish("wslope", vq[GUARD:])
        assert BIAS % (1 << GUARD) == 0  # W = (g1 << GUARD) + BIAS: W mod 2^GUARD = 0
    with b.phase("unrotate/sherase"):
        mb_sh = demolish("shift", shq)
    with b.phase("unrotate/g2erase"):
        mb_g2 = demolish("g2", g2q)

    # ---- 9. fixup: one conditioned-CZ walk repairs all phase kickback ----
    # Demolishing a coefficient wire that held bit b of the leaf's word kicks
    # (-1)^b exactly when the measurement bit came up 1; the repair is a CZ
    # between the leaf's path flag and a constant-|1> kick wire, conditioned
    # on that classical bit.  Below tree level BESTM the walk is replaced by a
    # one-hot expansion of X[7..BESTM-1]: the unary wires serve every leaf
    # of width < 2^BESTM at once, cheaper than descending their subtree
    # nodes.  BESTM minimizes walk-ANDs + one-hot cost subject to the
    # one-hot bank fitting under the peak cap.
    # local: gadgets.qrom.erase_register + phase_fixup serve one demolished
    # word per fixed-depth window walk; this fixup serves all four words
    # (g2, sh, W, g0) in a single adaptive floored walk.

    def fixup_cost(m):
        state = {"ands": 0, "stopped": False}

        def go(nd, has):
            if isinstance(nd, Leaf):
                return
            if nd.bit < m:
                state["stopped"] = True
                return
            if has:
                state["ands"] += 1
            go(nd.left, True)
            go(nd.right, True)

        go(ROOT, False)
        oh = max(0, (1 << (m - 7)) - 2) if state["stopped"] else 0
        return state["ands"] + oh, state["stopped"]

    BESTM, BESTC = 7, None
    for m_try in range(7, 14):
        cst, stp = fixup_cost(m_try)
        if stp and 56 + (1 << (m_try - 7)) + (21 - m_try) > PCAP:
            continue
        if BESTC is None or cst < BESTC:
            BESTM, BESTC = m_try, cst

    def onehot_build(bits):
        """One-hot bank over ``bits``: U[y] = [bits == y], grown one bit at a
        time with 2^t ANDs at level t (2^j - 2 Toffoli total).
        local: gadgets.qrom.phase_fixup routes its one-hot window by cswap
        (2^j - 1 Toffoli) and consumes it inside one fixup call; this
        AND-grown bank costs 2^j - 2 and stays claimed across the whole
        floored walk.
        """
        j = len(bits)
        f1 = b.qubit()
        b.cx(bits[0], f1)
        f0 = b.qubit()
        b.cx(bits[0], f0)
        b.x(f0)
        U = [None] * (1 << j)
        U[0], U[1] = f0, f1
        b.claim("onehot", [f0, f1])
        for t in range(1, j):
            for y in range(1 << t):
                n = b.and_compute(U[y], bits[t])
                b.cx(n, U[y])
                U[y + (1 << t)] = n
        b.claim("onehot", U[2:])  # the AND-grown bank wires, now long-lived
        return U

    def onehot_free(U, bits):
        b.release("onehot", U)
        j = len(bits)
        for t in range(j - 1, 0, -1):
            for y in range((1 << t) - 1, -1, -1):
                n = U[y + (1 << t)]
                b.cx(n, U[y])
                b.and_release(n, U[y], bits[t])
        b.x(U[0])
        b.cx(bits[0], U[0])
        b.recycle(U[0])
        b.cx(bits[0], U[1])
        b.recycle(U[1])

    def leaf_mbs(lf):
        """Return the measurement bits recorded as 1-kicks for this leaf's words."""
        return [(mb_g2, mb_sh, mb_g1, mb_c0)[w][i] for w, i in leaf_bits(lf)]

    with b.phase("fixup/kick"):
        kick = b.qubit()
        b.claim("kick", [kick])
        b.x(kick)
    with b.phase("fixup/onehot"):
        ohbits = [XQ[t] for t in range(7, BESTM)]
        U = onehot_build(ohbits)

    def emit_leaf(ctrl, lf):
        if lf.bb >= BESTM:
            for cbit in leaf_mbs(lf):
                b.cz(ctrl, kick, cond=cbit)
        else:
            mask = (1 << (BESTM - 7)) - 1
            xb = (lf.lo >> 7) & mask
            mbs = leaf_mbs(lf)
            for xr in range(1 << (lf.bb - 7)):
                u = U[xb + xr]
                for cbit in mbs:
                    b.cz(ctrl, u, cond=cbit)

    with b.phase("fixup/walk"):
        walk(emit_leaf, floor=BESTM)
    with b.phase("fixup/unhot"):
        onehot_free(U, ohbits)
    with b.phase("fixup/close"):
        b.x(kick)
        b.release_recycle("kick", [kick])
    return b, _r


# ------------------------------------------------------------------ entry
if __name__ == "__main__":
    b, _ = build()
    # -------------------------------------------------------- accounting
    assert_metrics(b, __file__)
    # --------------------------------------------------------- emission
    emit_and_report(b)
