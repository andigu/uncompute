#!/usr/bin/env python3
"""modinv-point-add: the divstep circuit, in-place point addition
|P> -> |P+Q> at 4,006,673 executed Toffoli x 935 peak qubits.

This file spells the circuit's phase flow flat, top to bottom, so each
phase mark sits on the code that opens the phase:

    init          registers (x1, y1), the addend bit registers, the
                  A word and the multiplier machine (singles + lazy pool)
    s1-const-subs x1 -= Q.x, y1 -= Q.y mod p (classical-bit networks)
    s2-*          y1 <- y1/x1 by a divstep walk pair: walk A forward,
                  the seam row multiply and the measurement erase of y1,
                  walk B in reverse with the halving rider, then the
                  recompute, phase fixup, erase and swap
    s4-sub-squ    x1 <- x1 + 3*Q.x - lambda^2 mod p; the fused square
                  (squ-a-lo, squ-b-hi, squ-c-sum) subtracts lambda^2
    s6-*          y1 <- x1*y1 by the second walk pair: the product,
                  walk C forward with the halving rider, the seam
                  apply/erase, walk D in reverse with the doubling
                  rider, then the swap
    s7-final      x1 <- Q.x - x1; y1 -= Q.y
    close         machine teardown to |0>

engine.py beside this file holds everything that opens no phase: the
walk legs and their gadgets, the walk sandwich (HmrDivstepInv: row
multiply, sign fix, rider), the multiplier machine (MulMachine), the
k-block multiplier, the fused square (FusedSquare), the parameter
tables and the emit tail.  This file runs the Hmr engine
(hmr_build_circuit with WIDE off) at this circuit's dials.  The other
circuit files (fill.py, j5.py .. j9.py, wide.py) set other dials through
the engine; README.md lists the cost of each.
"""

import pathlib

from uncompute import Builder
from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent

# The engine (engine.py, loaded by path): the walk and arithmetic stack
# this file builds on, and the ../_shared/luo_core.py layer (E.LC).
E = load_module(HERE / "engine.py", "divstep_engine")
LC = E.LC

N = E.N
P_MOD = E.P_MOD

# ---------------------------------------------------------------------------
# dials: this circuit's configuration (the other circuit files set other
# dials through the engine)
# ---------------------------------------------------------------------------
# The arena ceiling (hmr_composed_params Qcap): the lowest ceiling this
# build attains, found by gen_constants.py (935).
QCAP = E.qcap_floor("circuit")
E.hmr_configure({"qcap": QCAP, "wide": False})


# ---------------------------------------------------------------------------
# the fused square: x -= lambda^2 mod p, one half-width square per phase
# ---------------------------------------------------------------------------


def _subtract_square(b, lam, out):
    """out -= lam^2 mod p with the engine's FusedSquare: the low, high and
    cross squares each fold into out and clear before the next starts.
    """
    sq = E.FusedSquare(b, lam, out, QCAP)
    b.set_phase("squ-a-lo")
    sq.low_term()
    b.set_phase("squ-b-hi")
    sq.high_term()
    b.set_phase("squ-c-sum")
    sq.cross_term()
    sq.close()


# ---------------------------------------------------------------------------
# the point-add flow
# ---------------------------------------------------------------------------

# Pool sizing (E.HMR_POOL_PLATEAU, E._hmr_pool_off): each mul pool
# rides its phase's width ceiling at the arena dial, and the pool
# allocates lazily (MulMachine.pool_width), so no phase pays for a
# pool it does not use.


def build_circuit(b, p=P_MOD, n=N):
    """Emit the point addition |P> -> |P+Q>; Q per shot on the
    Bit-kind inputs.  Returns (x1, y1, ox, oy).
    """
    x1, y1, ox, oy = LC.declare_point_io(b, n)
    # per-shot cbspecs (free classical networks of the addend bits)
    S_x2, S_y2, S_3x2 = LC.point_cbspecs(b, ox, oy, p, n)

    A = b.qubits(n)
    mm = E.MulMachine(b)
    singles, ma = mm.singles, mm.ma
    b.claim("io/x1", x1)
    b.claim("io/y1", y1)
    b.claim("A", A)
    mm.claim()

    inv = E.HmrDivstepInv(b, x1)

    with b.phase("s1-const-subs"):
        LC.const_modp(b, x1, S_x2, singles["flag"], y1, singles["clean3"], p, n, inv=True)
        LC.const_modp(b, y1, S_y2, singles["flag"], x1, singles["clean3"], p, n, inv=True)

    # ================= s2-idiv: y1 <- y1 / x1 ==========================
    # walk-A fwd (g rides x1; f/gv host = the |0> A word) -> closer
    # (fv parked signed; A word back |0>) -> plateau row-mul A =
    # |fv|*y1*2^{-Tmin} + sign fix (A = y1*x1^{-1}*2^{tstop-Tmin}) ->
    # HMR(y1) -> closer_inv (f/gv rehomed to the freed y1 word) ->
    # walk-B rev with rider halves (A *= 2^{Tmin-tstop}) -> recompute/
    # phase-fix/erase y1 -> swap(y1, A).
    # The mul pool and its plan exist only at mul sites: the walk legs
    # allocate their own scratch (decode chains, zero-test keepers,
    # rider fold anc), so the pool parks at width 0 around them and
    # the walk census does not carry idle pool wires.
    with b.phase("s2-walkA"):
        mm.off()
        inv.start(A, "s2-walkA", "s2-walkB")
        inv.walk.emit_fwd()
    with b.phase("s2-seam"):
        mm.on(E.HMR_POOL_PLATEAU, seam=True)
        inv.walk.closer()
        inv.row_mul(y1, A, ma)
        inv.sign_fix(A, ma)
        mbits = [b.hmr(q) for q in y1]
        inv.row_unpark(y1, ma)
        inv.walk.pool.rebind_band("fw", list(y1))
        inv.walk.closer_inv()
        inv.arm_rider(A, {"rev": "halve"})
        mm.off()
    with b.phase("s2-walkB"):
        inv.walk.emit_rev()
        inv.walk.rider = None
        inv.finish()
    with b.phase("s2-recompute"):
        mm.on(E._hmr_pool_off())
        E.mul_modp(b, x1, A, y1, p, n, ma)  # y1 = x1*A = y exact
        for q, m in zip(y1, mbits):
            b.z(q, cond=m)  # Z^b phase fixup
        E.mul_modp(b, x1, A, y1, p, n, ma, inv=True)
        for yq, aq in zip(y1, A):
            b.swap(yq, aq)

    # ============ s4: x1 <- x1 + 3*Q.x - lambda^2 = P.x + 2*Q.x - lambda^2
    # The fused square subtracts lambda^2 from x1 term by term, so no
    # square register survives the phase.  The A word is |0> and idle
    # here; it is freed so the square's carries can use its width, and
    # allocated again afterwards.
    with b.phase("s4-sub-squ"):
        mm.pool_width(0)
        b.release_recycle("A", A)
        _subtract_square(b, y1, x1)
        b.set_phase("s4-sub-squ")
        LC.const_modp(b, x1, S_3x2, singles["flag"], y1, singles["clean3"], p, n)
        A[:] = b.qubits(n)
        b.claim("A", A)
        mm.pool_width(E._hmr_pool_off())

    # ================= s6-imul: y1 <- x1 * y1 ==========================
    # A = x1*y1 (exact, tape empty) -> HMR(y1) -> walk-C fwd on x1
    # (f/gv host = the freed y1 word; rider halves: A *= 2^{Tmin-tstop})
    # -> closer -> plateau row-mul y1 = |fv|*A*2^{-Tmin} + sign fix (= y exact)
    # -> Z^b -> erase pair -> closer_inv -> walk-D rev with rider
    # doubles (A restored to x1*y1) -> swap(y1, A).
    with b.phase("s6-mul1"):
        E.mul_modp(b, x1, y1, A, p, n, ma)
    with b.phase("s6-walkC"):
        mbits2 = [b.hmr(q) for q in y1]
        mm.off()
        inv.start(y1, "s6-walkC", "s6-walkD")
        inv.arm_rider(A, {"fwd": "halve", "rev": "double"})
        inv.walk.emit_fwd()
    with b.phase("s6-seam"):
        mm.on(E.HMR_POOL_PLATEAU, seam=True)
        inv.walk.closer()
        inv.row_mul(A, y1, ma)  # y1 = |fv|*A*2^-Tmin
        inv.sign_fix(y1, ma)  # y1 = y exact
        for q, m in zip(y1, mbits2):
            b.z(q, cond=m)
        inv.sign_fix(y1, ma, inv=True)
        # erase pair: the facing lifts cancel (the row parks in
        # magnitude form through the sign fix + fixup, which never
        # touch the row ids; the inv call carries the un-reduce +
        # un-lift)
        inv.row_mul(A, y1, ma, inv=True)
        inv.walk.closer_inv()
        mm.off()
    with b.phase("s6-walkD"):
        inv.walk.emit_rev()
        inv.walk.rider = None
        inv.finish()
        mm.on(E._hmr_pool_off())
        for yq, aq in zip(y1, A):
            b.swap(yq, aq)

    # ================= s7-final ========================================
    with b.phase("s7-final"):
        LC.neg_modp(b, x1, y1, singles["clean3"], p, n)
        LC.const_modp(b, x1, S_x2, singles["flag"], y1, singles["clean3"], p, n)
        LC.const_modp(b, y1, S_y2, singles["flag"], x1, singles["clean3"], p, n, inv=True)

    # ================= teardown ========================================
    b.set_phase("close")
    b.release("A", A)
    mm.off()
    for w in reversed(list(A)):
        b.recycle(w)
    return x1, y1, ox, oy


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------


def main():
    """Build, then the engine's emit tail: per-phase census table, the
    cost pin and ancilla-leak guard (assert_metrics), emit_and_report.
    """
    b = Builder()
    b.set_phase("init")
    build_circuit(b)
    E.emit_tail(b, "hmr", __file__)


if __name__ == "__main__":
    main()
