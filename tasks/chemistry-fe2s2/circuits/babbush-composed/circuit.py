#!/usr/bin/env python3
"""chemistry-fe2s2 circuits/babbush-composed — Babbush coherent alias
sampling composed
with an independent unary-iteration SELECT: the published circuits end
to end, nothing shared between the two walks ([BAB] Babbush et al.
(2018), arXiv:1805.03662; [BER] Berry et al. (2019), arXiv:1902.02134;
[GID] Gidney (2018), arXiv:1709.06648, temporary AND).  The alias lookup
carries the standard variable-spaced refinement ([SAN] Sanders et al.
(2020), arXiv:2007.07391, Fig. 3):
identical-word subtrees of the frame-block-ordered table load once at the
subtree root, and out-of-domain subtrees (i >= L) are never entered.
The unlookup window is k' = 32 — the peak-times-Toffoli-optimal member
of [BER] App. C's k' family (the T-optimal k' = 256 would park a
256-wire one-hot window and more than double the peak).

The selection is interior — sel lives on a 16-wire
scratch, SELECT walks it while the alias word is still live, and the
scratch erases by X-basis demolition whose kickback <m, sel> =
<m, alt> ^ f.<m, i^alt> resolves through outcome-conditioned Z/CZ
against the live alt carriers, the index register, and a held f copy
(zero Toffolis).  The f copy clears against the keep word by one extra
comparator pass (16 Toffolis).

RUN: python circuit.py                    builds, asserts metrics, emits
     python tools/eval.py chemistry-fe2s2 --circuit <this file>
                                          evaluates (repo root)

README.md in this folder describes the circuit.
"""

import pathlib

from uncompute import Builder
from uncompute.gadgets import declare_registers, pruned_walk, xor_word
from uncompute.testing import assert_metrics, emit_and_report, load_module

# The comparator, multiplexor, unlookup, SELECT leaf and erasure tail are
# loaded from ../_shared/baseline.py.
_sh = load_module(
    pathlib.Path(__file__).resolve().parent.parent / "_shared" / "baseline.py", "alias_baseline"
)
L, A, S, X, B = _sh.L, _sh.A, _sh.S, _sh.X, _sh.B
DATA = _sh.DATA
KU, LOG_KU = 32, 5
assert KU == 1 << LOG_KU
# [BER]'s T-optimal window is berry_ku(L) = 256; k' = 32 is the
# TQ-cost-optimal member of the same published family (see module docstring).


def build():
    b = Builder()
    regs = declare_registers(b, [("i", A), ("sigma", S), ("x", X)])
    idx, sigma, x = regs["i"], regs["sigma"], regs["x"]
    for name in ("i", "sigma", "x"):
        b.claim(name, regs[name])
    word = b.qubits(B)
    b.claim("word", word)

    # [SAN] Fig. 3 content pruning of the [BAB] walk: an aligned span
    # whose in-domain addresses all store the same word writes it once
    # at the span root; out-of-domain spans (i >= L) are never entered.
    def in_domain(lo, _span):
        return lo < L

    def same_word(lo, span):
        return len(set(DATA[lo : min(lo + span, L)])) <= 1

    def leaf(lo, ctrl):
        xor_word(b, ctrl, DATA[lo], word)

    b.set_phase("lookup")
    pruned_walk(b, list(reversed(idx)), 0, 1 << A, leaf, in_domain, uniform=same_word)
    b.set_phase("cmp/fwd")
    carries = _sh.compare(b, sigma, word[:16])
    b.set_phase("cmp/select")
    sel = b.qubits(A)
    b.claim("sel", sel)
    _sh.select(b, idx, word, sel, carries[-1])
    # f rides one held wire through SELECT so the sel erasure can
    # condition on it; allocated after select() frees its theta scratch,
    # keeping the phase peak at the comparator stage.
    flag = b.qubit()
    b.claim("flag", [flag])
    b.cx(carries[-1], flag)
    b.set_phase("cmp/rev")
    _sh.uncompare(b, sigma, word[:16], carries)
    b.set_phase("select-pauli")
    _sh.select_pauli(b, sel, x)
    # ---- round-trip erasure: sel, the f copy, then the word.
    _sh.erase(b, idx, sigma, word, sel, flag, KU)
    return b, idx, sigma, x


def main():
    b, _, _, _ = build()
    # The variable-spaced alias lookup (13,061 — the frame-blocked
    # order gathers runs of identical words and the melt schedule
    # extends them), the dense SELECT walk over the [0, 41052) window
    # (41,050), two comparator passes (32 — the second erases the held
    # f copy), the mux incl. the theta scratch (18) and the k'=32
    # unlookup window (31 prep + 1,281 fixup) total 55,473.  The sel
    # erasure is measurement plus outcome-conditioned phase — zero
    # Toffolis.  Peak 139 = 72 I/O (16+16+40) + 34 word + 16 comparator
    # carries + 16 sel scratch + the mux theta scratch, reached inside
    # cmp/select.  assert_metrics also proves every ancilla was freed.
    assert_metrics(b, __file__)
    emit_and_report(b)


if __name__ == "__main__":
    main()
