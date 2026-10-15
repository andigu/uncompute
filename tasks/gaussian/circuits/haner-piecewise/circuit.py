#!/usr/bin/env python3
"""gaussian haner-piecewise: the direct HRS
piecewise-polynomial Gaussian circuit (arXiv:1805.12445) — sign
fold, pre-truncated square, degree 3 on 32 Remez intervals in signed
fixed point (40, 7) evaluated in local coordinates, full Bennett
mirror.

Certified cost: 23,784 Toffoli x 362 qubits = 8,609,808
(assert_metrics in the epilogue).

constants.json holds the swept configuration and its fitted tables;
gen_constants.py writes it.

RUN: uv run python circuit.py   (stream to $UNCOMPUTE_OPS_PATH, else ./ops.bin)

README.md in this folder describes the circuit.
"""

from __future__ import annotations

import json
import pathlib

from uncompute import Builder
from uncompute.bennett import Stages
from uncompute.gadgets import (
    add,
    and_ladder,
    and_ladder_release,
    declare_registers,
    gidney_add,
    gidney_sub,
    hrs_mul_rows,
    label_increment,
    xor_word,
)
from uncompute.testing import assert_metrics, emit_and_report

# ---------------------------------------------------------------------------
# Instance constants (semantic scalars; the widths repeat the values in
# the task's contract.json)
# ---------------------------------------------------------------------------
INPUT_BITS = 39  # X: two's complement, 7 integer bits incl. sign
OUTPUT_BITS = 39  # k: modular accumulator
OUTPUT_FRAC = 32
MAG_BITS = 36  # |X| <= 2^35 on the domain |x| <= 8

# The configuration and tables that gen_constants.py sweeps and fits.
_HERE = pathlib.Path(__file__).resolve().parent
_C = json.loads((_HERE / "constants.json").read_bytes())

# The swept configuration of the published design.  The circuit below
# is written for the direct (unpebbled) degree-3 ladder.
DEGREE = _C["DEGREE"]  # 3
WORD_BITS = _C["WORD_BITS"]  # n: signed fixed-point word
INT_BITS = _C["INT_BITS"]  # p: w = x^2 < 64 needs 6 + sign
assert (DEGREE, _C["SCHEDULE"]) == (3, "direct")
LEFTS = _C["LEFTS"]  # left endpoint of each interval, raw w = x^2 * 2^33
COEFFICIENTS = [tuple(row) for row in _C["COEFFICIENTS"]]  # [c0..c3] word patterns
INTERVALS = len(LEFTS)  # 32
FRAC_BITS = WORD_BITS - INT_BITS  # 33

LABEL_BITS = 5  # ceil(log2(INTERVALS))
assert (1 << LABEL_BITS) >= INTERVALS > (1 << (LABEL_BITS - 1))
ROUND_SHIFT = FRAC_BITS - OUTPUT_FRAC  # 1: bits dropped at round
ROUND_HALF = 1 << (ROUND_SHIFT - 1)  # +1, folded into c0
assert (ROUND_SHIFT, ROUND_HALF) == (1, 1)  # the add-1-drop-bit-0 rule
WORD_MASK = (1 << WORD_BITS) - 1
DOMAIN_W_RAW = 64 << FRAC_BITS  # w = x^2 domain end, 2^39


# ---------------------------------------------------------------------------
# The circuit.  Every helper closes over the live Builder; AND ancillas are
# the only anonymous transients (Gidney pairs).
# ---------------------------------------------------------------------------


def build_circuit(coefficients, lefts, bounds):
    """Emit the full circuit for the given quantized partition; return
    (Builder, register dict).
    """
    b = Builder()
    st = Stages(b)
    regs = declare_registers(b, [("x", INPUT_BITS), ("k", OUTPUT_BITS)])
    XQ, KQ = regs["x"], regs["k"]
    b.claim("x", XQ)
    b.claim("k", KQ)

    # Coefficient words as loaded (two's complement on the 40-bit word);
    # the +1 rounding offset rides the degree-0 word — the round stage is
    # then a plain add of the final word's bits [1:40) into k, exact on
    # the pattern regardless of the word's sign.
    w3 = [c[3] for c in coefficients]
    w2 = [c[2] for c in coefficients]
    w1 = [c[1] for c in coefficients]
    w0 = [(c[0] + ROUND_HALF) & WORD_MASK for c in coefficients]
    assert [(w - ROUND_HALF) & WORD_MASK for w in w0] == [c[0] for c in coefficients]

    # ---- coefficient selection: label-match flag + CX loads -------------
    # gadgets.mcx pays LABEL_BITS - 1 = 4 Toffoli per target bit;
    # and_ladder computes one 4-AND match flag and rides it across all
    # (up to 40) CX loads of the word, 4 Toffoli per interval total.
    # gadgets.qrom.walk selects all 32 at 30 Toffoli but is unary
    # iteration, which HRS does not use.
    def coef_load(pattern, word, W):
        """CX-load ``word`` into W under [label == pattern]: one AND
        ladder over the 5 label bits (open bits X-conjugated for the whole
        ladder lifetime, releases included).
        """
        flag, rec = and_ladder(b, [(lbl[j], (pattern >> j) & 1) for j in range(LABEL_BITS)])
        xor_word(b, flag, word, W)
        and_ladder_release(b, rec)

    def word_alloc(setname):
        W = b.qubits(WORD_BITS)
        b.claim(setname, W)
        return W

    def selected_load(words, W):
        for pattern, word in enumerate(words):
            if word:  # a zero word loads nothing
                coef_load(pattern, word, W)

    def square_rows():
        """Emit the truncated-square rows: m = 2|X| lives in XQ shifted by one
        (m's bit ``bit`` is XQ[bit-1]; in-domain m has no bits above 36),
        so row ``bit`` adds trunc(m * 2^(bit-33)) — the XQ window shifted
        by bit-32 — under control XQ[bit-1].  Where the window crosses its
        own control (the diagonal, m[bit]*m[bit] = m[bit]), a CX copy
        stands in for the control so ``add(ctrl=)``'s operands stay
        distinct.
        """
        for bit in range(1, MAG_BITS + 1):
            b.set_phase(f"square/b{bit:02d}")
            ctrl = XQ[bit - 1]
            T = Wq[max(0, bit - (FRAC_BITS - 1)) :]
            A = list(XQ[max(0, (FRAC_BITS - 1) - bit) : MAG_BITS])[: len(T)]
            if ctrl in A:
                dq = b.qubit()
                b.claim("diag", [dq])
                b.cx(ctrl, dq)
                A[A.index(ctrl)] = dq
                add(b, T, A, ctrl=ctrl)
                b.cx(ctrl, dq)
                b.release_recycle("diag", [dq])
            else:
                add(b, T, A, ctrl=ctrl)

    def offset_stage(un=False):
        """t = w - a_l (un=False) and its exact inverse (un=True):
        label-match CX-load the selected left endpoint a_l =
        ``lefts[label]`` into a transient word, one in-place subtract
        (add on the way back), label-match clear, free.  The word lives
        only inside this stage, so its 40 wires never widen the Horner
        peak.
        """
        pre = "un-offset" if un else "offset"
        b.set_phase(f"{pre}/{'clear' if un else 'load'}")
        Aof = word_alloc("aoff")
        selected_load(lefts, Aof)  # interval 0 starts at w = 0
        b.set_phase(f"{pre}/sub")
        (gidney_add if un else gidney_sub)(b, Aof, Wq)
        b.set_phase(f"{pre}/{'load' if un else 'clear'}")
        selected_load(lefts, Aof)
        b.release_recycle("aoff", Aof)

    # ---- 1. fold: |X| by one's-complement fold + conditional +1 ---------
    # In-domain |X| <= 2^35, so the increment's carries die inside
    # XQ[0:36] and XQ[36..37] are zero by the domain promise until
    # un-fold.  The stage head is empty (fold allocates
    # nothing), so the whole stage mirrors as body.
    with st.stage("fold/mirror"):
        pass
    b.set_phase("fold/sign")
    for i in range(INPUT_BITS - 1):
        b.cx(XQ[INPUT_BITS - 1], XQ[i])
    b.set_phase("fold/abs")
    gidney_add(b, [XQ[INPUT_BITS - 1]], XQ[:MAG_BITS])

    # ---- 2. square: w = truncated_square(2|X|) --------------------------
    with st.stage("square/alloc", set_name="w"):
        b.set_phase("square/alloc")
        Wq = word_alloc("w")
    square_rows()

    # ---- 3. label: 31 comparators drive the label counter ---------------
    with st.stage("label/setup", set_name="label"):
        b.set_phase("label/setup")
        lbl = b.qubits(LABEL_BITS)
        b.claim("label", lbl)
    for i, bound in enumerate(bounds):
        b.set_phase(f"label/b{i:02d}")
        # The bounds ascend strictly (asserted), so the label counts
        # the bounds <= w at 0 Toffoli beyond each comparator.
        label_increment(b, Wq, bound, i, lbl)

    # ---- 4. offset: t = w - a_l -----------------------------------------
    # The word lives only inside the stage, so its inverse is the
    # explicit offset_stage(un=True) unwind callback, never a mirror.
    with st.stage("offset", unwind=lambda: offset_stage(un=True)):
        offset_stage()

    # ---- 5..8. lead/h2/h1/h0: the unpebbled Horner ladder ---------------
    # One core, four stages: the stage head allocates the stage word,
    # then the body CX-loads the selected coefficient and (h-stages only)
    # runs the multiplier rows against the previous word.  The h0 word
    # table (w0) carries the folded +1.
    stages = [
        ("lead", "poly3", w3),  # V3 = c3[label]
        ("h2", "poly2", w2),  # V2 = c2[label] + V3 (x) t
        ("h1", "poly1", w1),  # V1 = c1[label] + V2 (x) t
        ("h0", "poly0", w0),
    ]  # V0 = c0[label] + 1 + V1 (x) t
    V = None
    for stage, setname, words in stages:
        with st.stage(f"{stage}/load", set_name=setname):
            b.set_phase(f"{stage}/load")
            W = word_alloc(setname)
        selected_load(words, W)
        if V is not None:
            hrs_mul_rows(b, stage, V, W, Wq, FRAC_BITS)
        V = W

    mark_round = b.checkpoint()
    V0 = V  # the poly0 word: Horner value + 1

    # ---- 9. round: k += V0[1:40) mod 2^39 -------------------------------
    b.set_phase("round/add")
    gidney_add(b, V0[ROUND_SHIFT:], KQ)  # ty: ignore[not-subscriptable]  # stages is non-empty, so V is set

    # ---- 10. un-*: Bennett mirror of stages 8..1 ------------------------
    # st.unwind replays every recorded stage in reverse — per stage:
    # set_phase(un-<name>); mirror the body (the stage's wires return to
    # |0>); release the ownership set; mirror the head (replaying the
    # frees) — so every set is released exactly when its wires are
    # provably |0>.  The offset stage carries its unwind= callback
    # instead: offset_stage(un=True) — the same loads around the inverse
    # add, claims included — emitted between the word mirrors and the
    # label mirror.
    st.unwind(until=mark_round)

    return b, regs


def build():
    """Run the raw build (also the profiler tools' entry): check the
    tables from constants.json and construct the circuit.  Returns
    (Builder, register dict).
    """
    assert len(COEFFICIENTS) == INTERVALS
    assert all(len(row) == DEGREE + 1 for row in COEFFICIENTS)
    assert all(0 <= c <= WORD_MASK for row in COEFFICIENTS for c in row)
    assert LEFTS[0] == 0
    assert all(a < c for a, c in zip(LEFTS, LEFTS[1:]))
    assert LEFTS[-1] < DOMAIN_W_RAW
    assert COEFFICIENTS[-1] == (0,) * (DEGREE + 1)
    bounds = LEFTS[1:]  # interval i + 1 starts where interval i ends

    # x = 0 corner: w = t = 0, so the executed addend is the
    # interval-0 folded degree-0 word dropped by one bit — 391 ULP under
    # the ideal 2^32.

    return build_circuit(COEFFICIENTS, LEFTS, bounds)


if __name__ == "__main__":
    b, _regs = build()
    # The peak sits at the widest multiplier row (h0/b33) and decomposes
    # as the register file (x + k + label + the square/local word + 4
    # polynomial words = 283) + that row's transients (39 chain ANDs +
    # 39 addend ANDs + the top addend AND = 79).
    assert_metrics(b, __file__)
    emit_and_report(b)
