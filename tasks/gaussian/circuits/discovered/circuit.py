#!/usr/bin/env python3
"""gaussian circuits/discovered: a content-pruned QROM serving
a 514-segment piecewise-quadratic table, evaluated by two Booth
radix-4 multiplies with measurement-based deferred uncomputation and
pooled certified-zero spares.

Certified cost: 1,583 Toffoli x 139 qubits = 220,037 (assert_metrics
in the epilogue).

RUN: uv run python circuit.py   (stream to $UNCOMPUTE_OPS_PATH, else ./ops.bin)

README.md in this folder describes the circuit.
"""

import json
from pathlib import Path

from uncompute import Builder
from uncompute.gadgets import (
    WirePool,
    add,
    booth_columns,
    booth_columns_release,
    declare_registers,
    pruned_walk,
)
from uncompute.testing import assert_metrics, emit_and_report, load_contract

# Task contract (grader.py asserts the same values from its own
# inline derivation): the register widths.
_CONTRACT = load_contract(__file__)
N_X = _CONTRACT["n_x"]  # 39: two's-complement input width
KW = _CONTRACT["m"]  # 39: accumulator width (mod 2^39)

NADDR = 1024
FBITS = 21


# ---------------------------------------------------------------------------
# Frozen chosen config (gen_constants.py runs the config search and
# writes constants.json)
# ---------------------------------------------------------------------------
def load_cfg():
    """Return the frozen config from constants.json.

    build() calls load_cfg(), so gen_constants.py can import this module
    before constants.json exists.
    """
    cfg = json.loads((Path(__file__).resolve().parent / "constants.json").read_text())
    cfg["words"] = [int(w, 16) for w in cfg["words"]]
    ncut = cfg["Ncut"]

    # Table shape at the region seams, asserted: the zero region stores
    # all-zero words; the constant tail (s0 <= idx < Ncut) stores B1p = 0 and
    # E = 2^18 exactly, so its dq field is zero and only a0 survives.
    assert all(w == 0 for w in cfg["words"][ncut:])
    assert all(cfg["B1p"][a] == 0 and cfg["E"][a] == 1 << 18 for a in range(cfg["s0"], ncut))
    # The x = 0 clamp, asserted: segment 0 is the only segment at the CLAMP
    # saturation value 2^A0BITS - 1.
    assert cfg["A0p"][0] == (1 << cfg["A0BITS"]) - 1
    assert all(a < (1 << cfg["A0BITS"]) - 1 for a in cfg["A0p"][1:ncut])
    return cfg


# The pooled spare-wire bank (uncompute.gadgets.WirePool): after the sign
# fold, X[35..37] are sign-extension copies XORed to |0> for every
# in-domain input (x in [-8, 8)).  Lending them as pooled ancillas between
# fold and unfold is pure Python bookkeeping: a pooled "alloc" emits no op
# (the wire is persistent and already zero), a pooled AND is a bare CCX, a
# pooled release is hmr + fixup with no reset and no free.  While lent,
# the spares leave the x ownership set.
SPARES = (35, 36, 37)

# ---------------------------------------------------------------------------
# The circuit
# ---------------------------------------------------------------------------


def build(cfg=None):  # noqa: C901, PLR0915 - straight-line circuit emitter
    """Build the discovered circuit on a fresh Builder; return (b, regs).

    Circuit helpers are closures over ``b`` and the pool ``wp``.
    Registers come first, so Builder qubit ids 0..77 are the persistent
    I/O wires and every temp id matches the certified stream.  build()
    is also the entry point of tools/occupancy_profile.py, which
    replays the ownership wire-set log fail-closed and charts the
    occupancy.  ``cfg`` defaults to the frozen config (load_cfg());
    gen_constants.py passes each candidate config to rank it by circuit
    cost.
    """
    if cfg is None:
        cfg = load_cfg()
    WORDS = cfg["words"]
    M, CAPW, A0SH, A0BITS = cfg["M"], cfg["CAPW"], cfg["A0SH"], cfg["A0BITS"]
    EQ, RB, I0, FLOW = cfg["EQ"], cfg["RB"], cfg["I0"], cfg["FLOW"]
    NT, FW, EWID = cfg["NT"], cfg["FW"], cfg["EWID"]
    # Wiring couplings: booth_row indexes qFw[2i + 1 - FLOW] against the
    # FBITS-bit model, and the walks assume one word per address.
    assert FBITS == FW + FLOW and len(WORDS) == NADDR

    # ---- liveness plan: raw peak budget CAPRAW = lookup peak (78 I/O + 55
    # word bits + 9 walker ANDs = 142; raw counts the three spare-hosted
    # slope bits twice, so the Builder peak is 142 - 3 = 139).  Every k-add
    # gets a [GID] prefix of gl measured-carry positions (1 Toffoli and +1
    # live wire each, measurement-erased) sized so the runway moment never
    # exceeds the budget; the rest stays [CDKM] (in-place carries).
    WORDW = FW + EWID + A0BITS
    CAPRAW = 78 + WORDW + 9
    ROWBASE = 78 + FW + EWID
    A0RUN = 38 - (A0SH + A0BITS)
    GL0 = max(0, min(A0BITS - 1, CAPRAW - (78 + WORDW) - max(0, A0RUN)))

    b = Builder()
    _r = declare_registers(b, [("x", N_X), ("k", KW)])
    XQ, KQ = _r["x"], _r["k"]
    b.claim("x", XQ)
    b.claim("k", KQ)

    # ------------------------------------------------------------ wiring
    Dq = XQ[0:25]
    Uq = XQ[25 - M : 25]
    addr_msb = [XQ[34 - i] for i in range(10)]
    low_msb = [XQ[28], XQ[27], XQ[26], XQ[25]]
    hi_msb = [XQ[34], XQ[33], XQ[32], XQ[31], XQ[30], XQ[29]]

    # ---- fold: sign-fold x, pool the certified-zero spares, allocate words
    with b.phase("fold/sign"):
        for i in range(38):
            b.cx(XQ[38], XQ[i])
        wp = WirePool(b, SPARES)
        b.release("x", SPARES)  # the spares leave x while lent to the pool

    with b.phase("fold/words"):
        # The pool bank empties into the slope immediately: qFw[0..2] bind
        # the spares here (pooled allocs emit no op; the wires are the
        # persistent certified-zero x bits).  Every other packed-word wire
        # materializes at the first leaf that sets its bit, and a bit no
        # address ever sets fails the post-walk coverage assert instead of
        # riding as a dead wire.
        TW = FW + EWID + A0BITS
        T = [None] * TW
        for _j in range(3):
            T[_j] = wp.qubit()
        assert set(T[:3]) == set(SPARES)  # the pool bank empties into the slope
        b.claim("slope", T[:3])

    def _matT(j):
        if T[j] is None:
            T[j] = wp.qubit()
            b.claim("slope" if j < FW else "correction" if j < FW + EWID else "constant", [T[j]])
        return T[j]

    # ---- lookup: content-pruned QROM walk writes the packed word --------
    CUM = [0] * (NADDR + 1)
    for a in range(NADDR):
        CUM[a + 1] = CUM[a] + (1 if WORDS[a] else 0)

    def nz_main(lo, span):
        return CUM[lo + span] - CUM[lo] > 0

    def unif_main(lo, span):
        w = WORDS[lo]
        if w == 0:
            return False
        for a in range(lo + 1, lo + span):
            if WORDS[a] != w:
                return False
        return True

    def leaf_main(a, wire):
        w = WORDS[a]
        j = 0
        while w:
            if w & 1:
                if wire is None:
                    b.x(_matT(j))
                else:
                    b.cx(wire, _matT(j))
            w >>= 1
            j += 1

    def lmark(lo, span):
        if span == 128:
            b.set_phase(f"lookup/b{lo // 128}")

    # descend = the walk levels above the first 128-address subtree; each
    # b<n> mark then owns that subtree's leaves plus the pops back to the
    # next one
    with b.phase("lookup/descend"):
        pruned_walk(
            b, addr_msb, 0, NADDR, leaf_main, nz_main, uniform=unif_main, mark=lmark, pool=wp
        )
    assert all(q is not None for q in T), "packed-word bit never set"
    qFw, qEw, qA0w = T[:FW], T[FW : FW + EWID], T[FW + EWID :]
    # qEw now directly holds dq in two's complement (= the stored word
    # pattern)

    # ---- a0add: k += a0 << A0SH, then demolish the constant word --------
    # The demolition bits sbA0 stay pending until the unlook walk; carrying
    # them there is what makes the constant word's uncompute free.
    with b.phase("a0add/add"):
        add(b, KQ[A0SH:], qA0w, gl=GL0, pool=wp)
    with b.phase("a0add/meas"):
        sbA0 = []
        for q in qA0w:
            sbA0.append(b.hmr(q))
        b.release("constant", qA0w)  # demolished: ownership ends before the frees
        for q in reversed(qA0w):
            wp.recycle(q)

    # The Booth-row skeleton shared by mid_row and booth_row: the
    # X-sandwiched pair AND (m1, partner) -> m2 inside the ``flip`` CX
    # fold, the fused columns (booth_columns), the caller's pre-add
    # segment, the in-place add, its mirror, the column demolition
    # (booth_columns_release), and the mirrored pair-AND undo.  ``cols``
    # holds one (a1, a2) pair per column — a1 is m1's AND partner (None
    # for a column covered by m2 alone); ``colneg`` CXes that wire onto
    # every column as it is built and repairs it before the demolition;
    # ``wrap(m2, ts)`` emits the pre-add ops and returns (the addend A,
    # the adder kwargs, the post-add segment).
    def acc_row(m1, partner, flip, cols, wrap, T, colneg=None):
        for u, v in flip:
            b.cx(u, v)
        b.x(m1)
        m2 = wp.and_compute(m1, partner)
        b.x(m1)
        for u, v in flip:
            b.cx(u, v)
        ts, reps = booth_columns(b, m1, m2, cols, neg=colneg, pool=wp)
        A, akw, unwrap = wrap(m2, ts)
        add(b, T, A, pool=wp, **akw)
        unwrap()
        booth_columns_release(b, ts, reps, neg=colneg, pool=wp)
        for u, v in flip:
            b.cx(u, v)
        b.x(m1)
        wp.and_release(m2, m1, partner)
        b.x(m1)
        for u, v in flip:
            b.cx(u, v)

    # ---- Booth radix-4 mid multiply: F = B1 + sum of 6 signed d-rows ----
    # qEw holds signed dq; each row's high columns are covered by the sign
    # wire sw = AND(m1 xor m2, qEw[top]) reused (shared) across all
    # columns.  The CX-built recoding pair (m1 where allocated, z) is the
    # row's only non-AND scratch, claimed into "recode" for its lifetime.
    def mid_row(t, invert):
        j0, j1, j2 = 2 * t - 1, 2 * t, 2 * t + 1
        SH1 = M + 4 - 2 * t
        qlo = max(FLOW, EQ - SH1 + 1)
        alloc_m1 = j0 >= 0
        if alloc_m1:
            m1 = wp.qubit()
            b.cx(Uq[j1], m1)
            b.cx(Uq[j0], m1)
        else:
            m1 = Uq[j1]
        z = wp.qubit()
        b.claim("recode", ([m1] if alloc_m1 else []) + [z])
        if j2 < M:
            b.cx(Uq[j2], z)
        b.cx(Uq[j1], z)
        neg = Uq[j2] if j2 < M else None
        q2top = EQ + EWID - SH1
        cols = [(qEw[q + SH1 - EQ], qEw[q + SH1 - EQ - 1]) for q in range(qlo, q2top)]

        def wrap(m2, ts):
            # sign wire (covers columns q2top .. FW-1)
            b.cx(m1, m2)
            sw = wp.and_compute(m2, qEw[EWID - 1])
            b.cx(m1, m2)
            ext = FW - (q2top - FLOW)
            ext = max(ext, 0)
            for qq in qFw:
                if invert:
                    b.x(qq)
                if neg is not None:
                    b.cx(neg, qq)

            def unwrap():
                for qq in qFw:
                    if neg is not None:
                        b.cx(neg, qq)
                    if invert:
                        b.x(qq)
                cbit = b.hmr(sw)
                b.cx(m1, m2)
                b.cz(m2, qEw[EWID - 1], cond=cbit)
                b.cx(m1, m2)
                wp.recycle(sw)

            return ts + [sw] * ext, {"shared": sw}, unwrap

        acc_row(m1, z, [], cols, wrap, qFw[qlo - FLOW :])
        if j2 < M:
            b.cx(Uq[j2], z)
        b.cx(Uq[j1], z)
        b.release("recode", [z])
        wp.recycle(z)
        if alloc_m1:
            b.cx(Uq[j1], m1)
            b.cx(Uq[j0], m1)
            b.release("recode", [m1])
            wp.recycle(m1)

    for t in range(NT):
        with b.phase(f"mid_mul/t{t}"):
            mid_row(t, False)

    # ---- final rows: Booth rows of F x D accumulated into k -------------
    # k is complemented around the whole block (the X sandwich), turning
    # the subtraction k - P of the model into additions of the Booth rows.
    def booth_row(i, drop):
        sh = FBITS - 2 * i
        f1w = qFw[2 * i - FLOW]
        have2 = 2 * i + 1 < FBITS
        f2w = qFw[2 * i + 1 - FLOW] if have2 else None
        Aw = qFw[2 * i - 1 - FLOW]
        b.cx(f1w, Aw)
        neg = f2w if have2 else None
        cols = [
            (Dq[sh + q] if sh + q <= 24 else None, Dq[sh + q - 1]) for q in range(drop, 2 * i + 5)
        ]
        m = len(cols) + 1
        # [GID]-prefix length: the largest that keeps this row's runway
        # moment (row wires + carries + remaining headroom term) under
        # CAPRAW
        gl = max(0, min(m - 1, CAPRAW - ROWBASE - m - (33 - 2 * i)))

        def wrap(g, ts):
            b.cx(Aw, g)
            if neg is not None:
                b.cx(neg, g)

            def unwrap():
                if neg is not None:
                    b.cx(neg, g)
                b.cx(Aw, g)

            return [g] + ts, {"neg": neg, "gl": gl}, unwrap

        acc_row(
            Aw,
            f2w if have2 else f1w,
            [(f1w, f2w)] if have2 else [],
            cols,
            wrap,
            KQ[drop - 1 :],
            colneg=neg,
        )
        b.cx(f1w, Aw)

    with b.phase("final_rows/neg"):
        for q in KQ:
            b.x(q)
    for i in range(I0, 11):
        with b.phase(f"final_rows/i{i}"):
            drop = max(RB, (2 * i + 5) - CAPW)
            booth_row(i, drop)
    with b.phase("final_rows/unneg"):
        for q in KQ:
            b.x(q)

    # ---- mid_unmul: exact reverse of mid_mul returns F to B1 ------------
    for t in reversed(range(NT)):
        with b.phase(f"mid_unmul/t{t}"):
            mid_row(t, True)

    # One-hot rail over the 4 low address bits (16 addressable leaves;
    # 30 claimed wires with every level kept live), used once by the
    # fixup walk so each of its leaves can address all 16 word slots with
    # plain conditioned CZs instead of 16 more tree levels.  The rail —
    # the CX-built lo wires and the AND-grown hi wires alike — lives
    # across the whole walk, so every rail wire is claimed into "onehot".
    def build_onehot(bits_msb):
        hb = bits_msb[0]
        w1 = wp.qubit()
        b.cx(hb, w1)
        w0 = wp.qubit()
        b.x(w0)
        b.cx(hb, w0)
        b.claim("onehot", [w0, w1])
        cur = [w0, w1]
        levs = []
        for bit in bits_msb[1:]:
            lev = []
            new = []
            for w in cur:
                hi = wp.and_compute(w, bit)
                lo = wp.qubit()
                b.cx(w, lo)
                b.cx(hi, lo)
                new.extend([lo, hi])
                lev.append((w, bit, lo, hi))
            b.claim("onehot", new)
            levs.append(lev)
            cur = new
        return cur, ((hb, w0, w1), levs)

    def un_onehot(meta):
        (hb, w0, w1), levs = meta
        for lev in reversed(levs):
            b.release("onehot", [q for (_w, _bit, lo, hi) in lev for q in (lo, hi)])
            for w, bit, lo, hi in reversed(lev):
                b.cx(hi, lo)
                b.cx(w, lo)
                wp.recycle(lo)
                wp.and_release(hi, w, bit)
        b.release("onehot", [w0, w1])
        b.cx(hb, w0)
        b.x(w0)
        wp.recycle(w0)
        b.cx(hb, w1)
        wp.recycle(w1)

    # The deferred-fixup walk: demolish the surviving word wires to bits,
    # then one QROM walk over the 6 high address bits (x one-hot rail over
    # the 4 low bits) applies, per address, the conditioned CZs that cancel
    # every recorded kickback, the a0 bits from stage a0add included.
    # local: the library unlookup (gadgets.erase_register / phase_fixup)
    # walks a dense window on a cswap-routed one-hot and, with a merged
    # fixup function, shares all three words like this walk; at its own
    # tuned point (berry_ku -> ku = 32) it costs 30 walk ANDs + 31 route
    # cswaps = 61 Toffoli on 32 fresh wires (the wire-matched ku = 16
    # point costs 62 + 15 = 77).  This content-pruned walk costs 50 =
    # 14 rail ANDs + 36 walk ANDs (the phase split onehot 16 + w* 34
    # books 2 descent ANDs under onehot).  The rail claims 30 wires —
    # 3 on pool spares, 27 fresh — inside unlook's width slack (stage
    # peak 110 < the 139 ceiling).
    def fixup(qubits, pre_sb, wbits, low_msb, hi_msb, lam):
        # The parameters are coupled to the closure: the demolition order
        # must match the slope + correction releases, and lam must be the
        # rail size the leaf indexes.
        assert qubits == qFw + qEw and lam == 1 << len(low_msb)
        with b.phase("unlook/meas"):
            sb = []
            for q in qubits:
                sb.append(b.hmr(q))
            b.release("slope", qFw)  # demolished: ownership ends
            b.release("correction", qEw)  # before the frees
            for q in reversed(qubits):
                wp.recycle(q)
        sb = sb + list(pre_sb)
        with b.phase("unlook/onehot"):
            oh, meta = build_onehot(low_msb)
            assert len(oh) == lam

            def nzf(lo, span):
                a0 = lo * lam
                a1 = min((lo + span) * lam, NADDR)
                for a in range(a0, a1):
                    if wbits[a]:
                        return True
                return False

            def leaf(h, wire):
                for l in range(lam):
                    a = h * lam + l
                    if a >= NADDR:
                        break
                    w = wbits[a]
                    j = 0
                    while w:
                        if w & 1:
                            b.cz(wire, oh[l], cond=sb[j])
                        w >>= 1
                        j += 1

            def wmark(lo, span):
                if span == 8:
                    b.set_phase(f"unlook/w{lo}")

            pruned_walk(b, hi_msb, 0, 1 << len(hi_msb), leaf, nzf, mark=wmark, pool=wp)
        with b.phase("unlook/unhot"):
            un_onehot(meta)

    # ---- unlook: demolish slope+correction, one walk fixes all kickback -
    fixup(qFw + qEw, sbA0, WORDS, low_msb, hi_msb, 16)

    assert len(wp._free) == 3 and set(wp._free) == set(SPARES), "spares in flight"
    b.claim("x", SPARES)  # the certified-zero spares re-join x

    # ---- unfold ---------------------------------------------------------
    with b.phase("unfold"):
        for i in range(38):
            b.cx(XQ[38], XQ[i])
    return b, _r


# ------------------------------------------------------------------ entry
if __name__ == "__main__":
    b, _ = build()
    # -------------------------------------------------------- accounting
    assert_metrics(b, __file__)
    # --------------------------------------------------------- emission
    emit_and_report(b)
