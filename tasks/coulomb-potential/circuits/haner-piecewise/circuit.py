#!/usr/bin/env python3
"""coulomb-potential haner-piecewise: the normalized
HRS piecewise-polynomial circuit (arXiv:1805.12445) — exact
squared-distance front end, first-one normalization, degree 2 on 16
parity banks in signed fixed point (26, 2), barrel rescale,
segmented Bennett mirror.

Certified cost: 6,355 Toffoli x 252 qubits = 1,601,460
(assert_metrics in the epilogue).

RUN: uv run python circuit.py          (stream to $UNCOMPUTE_OPS_PATH, else ./ops.bin)
     uv run python gen_constants.py   (refit the tables into constants.json)
     uv run python checks.py          (the exhaustive datapath check)

README.md in this folder describes the circuit.
"""

from __future__ import annotations

import json
import math
import pathlib

from uncompute import Builder
from uncompute.bennett import Stages
from uncompute.gadgets import (
    and_ladder,
    and_ladder_release,
    declare_registers,
    gidney_add,
    gidney_sub,
    hrs_mul_rows,
    label_increment,
    xor_word,
)
from uncompute.testing import assert_metrics, emit_and_report, load_contract

# ---------------------------------------------------------------------------
# Instance constants (semantic scalars; the task-defining derived
# constants come from the shared contract.json below)
# ---------------------------------------------------------------------------
COORD_BITS = 8  # per-coordinate register width
ACC_BITS = 25  # accumulator width (mod 2^25)
OUT_SCALE_BITS = 24  # F(n2) = round(2^24 / sqrt(n2))
N2_BITS = 18  # n2 <= 3 * 255^2 = 195,075 < 2^18

WORD_BITS = 26  # n: signed fixed-point word
INT_BITS = 2  # p: integer bits (fixed by 1 <= u < 2)
FRAC_BITS = WORD_BITS - INT_BITS  # 24
INTERVALS = 8  # per parity; 16 coefficient banks

LABEL_BITS = 3  # ceil(log2(INTERVALS))
assert (1 << LABEL_BITS) >= INTERVALS > (1 << (LABEL_BITS - 1))
EXP_MAX = N2_BITS - 1  # e in [0, 17]
HMAX = EXP_MAX >> 1  # h = floor(e/2) in [0, 8]
H_BITS = 4
assert HMAX < (1 << H_BITS)  # the deepest shift fits the shift word
PAD_BITS = HMAX  # rescale extension: 8 zero wires below V0
WORD_MASK = (1 << WORD_BITS) - 1

_CONTRACT = load_contract(__file__)
ALLOWED_ULP = _CONTRACT["EPS"]  # the band the exhaustive check prices against
N2_MAX = _CONTRACT["N2_MAX"]
assert N2_MAX.bit_length() == N2_BITS
N_ACHIEVABLE = _CONTRACT["N_ACHIEVABLE"]
TBL_DIGEST = _CONTRACT["TBL_DIGEST"]
DOM_DIGEST = _CONTRACT["DOM_DIGEST"]


# ---------------------------------------------------------------------------
# The tables: gen_constants.py fits them (interval search + quantization)
# and writes constants.json; checks.py proves their worst error.
# ---------------------------------------------------------------------------


def load_tables():
    """Load and check the quantized banks and interval thresholds.

    Returns (qtab, thresholds): qtab maps (parity, interval) to [q0, q1,
    q2], every entry in the signed 26-bit word; thresholds are the
    ascending mantissa words where intervals 1.. start (u is an integer
    on the 2^-24 grid, so [u >= left_i] == [u_word >= T_i] exactly).
    """
    constants = json.loads((pathlib.Path(__file__).with_name("constants.json")).read_text())
    sign = 1 << (WORD_BITS - 1)
    qtab = {}
    for parity in (0, 1):
        rows = constants["qtab"][parity]
        assert len(rows) == INTERVALS
        for i, row in enumerate(rows):
            assert len(row) == 3 and all(-sign <= q < sign for q in row)
            qtab[(parity, i)] = row
    ths = constants["thresholds"]
    assert len(ths) == INTERVALS - 1
    assert all((1 << FRAC_BITS) < t < (1 << (FRAC_BITS + 1)) for t in ths)
    assert all(a < c for a, c in zip(ths, ths[1:]))
    return qtab, ths


# ---------------------------------------------------------------------------
# The circuit.  Every helper closes over the live Builder; AND ancillas are
# the only anonymous transients (Gidney pairs).
# ---------------------------------------------------------------------------


def build_circuit(qtab, thresholds):  # noqa: C901, PLR0915 - straight-line circuit emitter
    """Emit the full circuit for the given quantized banks and interval
    thresholds; return (Builder, register dict).
    """
    b = Builder()
    st = Stages(b)
    regs = declare_registers(
        b, [(n, COORD_BITS) for n in ("rix", "riy", "riz", "rjx", "rjy", "rjz")] + [("k", ACC_BITS)]
    )
    RI = [regs["rix"], regs["riy"], regs["riz"]]
    RJ = [regs["rjx"], regs["rjy"], regs["rjz"]]
    KQ = regs["k"]
    b.claim("ri", RI[0] + RI[1] + RI[2])
    b.claim("rj", RJ[0] + RJ[1] + RJ[2])
    b.claim("k", KQ)

    # Coefficient words as loaded (two's complement on the 26-bit word),
    # bank pattern = interval | parity << LABEL_BITS.
    banks = {}
    for parity in (0, 1):
        for i in range(INTERVALS):
            pattern = i | (parity << LABEL_BITS)
            banks[pattern] = [q & WORD_MASK for q in qtab[(parity, i)]]

    # ---- 1. dist: n2 = sum of squared signed differences ----------------
    # Per axis (short recomputable-difference lifetime): 9-bit difference,
    # sign-conditioned negation, schoolbook square rows, immediate clear.
    with st.stage("dist", set_name="n2"):
        b.set_phase("dist/setup")
        n2b = b.qubits(N2_BITS)
        b.claim("n2", n2b)

    def square_rows(mag):
        """n2b += mag^2: schoolbook row i (bit j = mag_i AND mag_j, the
        diagonal bit mag_i itself) Gidney-added at offset i; partial
        products released right after each row.
        """
        for i in range(COORD_BITS):
            row, tmp = [], []
            for j in range(COORD_BITS):
                if j == i:
                    row.append(mag[i])
                else:
                    t = b.and_compute(mag[i], mag[j])
                    row.append(t)
                    tmp.append((t, j))
            gidney_add(b, row, n2b[i:])
            for t, j in reversed(tmp):
                b.and_release(t, mag[i], mag[j])

    for ax in range(3):
        b.set_phase(f"dist/ax{ax}")
        D = b.qubits(COORD_BITS + 1)
        b.claim("diff", D)
        sgn, mag = D[COORD_BITS], D[:COORD_BITS]
        for j in range(COORD_BITS):  # D = r_i,c
            b.cx(RI[ax][j], D[j])
        gidney_sub(b, RJ[ax], D)  # D = r_i,c - r_j,c (9-bit 2c)
        for q in mag:  # sign-conditioned negation
            b.cx(sgn, q)
        gidney_add(b, [sgn], mag)  # mag = |d|, sgn parked
        square_rows(mag)
        gidney_sub(b, [sgn], mag)  # immediate clear: un-negate,
        for q in mag:
            b.cx(sgn, q)
        gidney_add(b, RJ[ax], D)  # un-subtract,
        for j in range(COORD_BITS):
            b.cx(RI[ax][j], D[j])
        b.release_recycle("diff", D)  # the mirror recreates them

    # ---- 2. scan: one-hot exponent flags + steering bits -----------------
    # HRS first-one scan: z = "no set bit above e yet"; f_e = z AND n2_e,
    # then z ^= f_e keeps z running.  z ends [n2 == 0] = 0 on every graded
    # shot (equal points are domain-rejected) and is freed; the flags stay
    # live to steer normalize, the bank match, and the rescale.  Two
    # ownership sets survive the stage, so its unwind is the un_scan
    # callback: both releases, then one whole-stage mirror.
    def un_scan():
        b.set_phase("un-scan")
        b.release("exp", flags)  # the mirrored chain and_releases free them
        b.release("steer", [pv] + hq)
        b.mirror(mark_dist, until=mark_scan)

    mark_dist = b.checkpoint()
    with st.stage("scan", unwind=un_scan):
        b.set_phase("scan/flags")
        flags = [None] * N2_BITS
        f_top = b.qubit()
        b.cx(n2b[EXP_MAX], f_top)
        flags[EXP_MAX] = f_top
        z = b.qubit()
        b.claim("scan-z", [z])
        b.x(z)
        b.cx(n2b[EXP_MAX], z)
        for e in range(EXP_MAX - 1, -1, -1):
            fe = b.and_compute(z, n2b[e])
            b.cx(fe, z)
            flags[e] = fe
        b.release_recycle("scan-z", [z])
        b.claim("exp", flags)
        # Any function of e is a free CX-load from the one-hot flags.
        b.set_phase("scan/steer")
        pv = b.qubit()  # parity: e mod 2
        hq = b.qubits(H_BITS)  # shift word: h = floor(e/2)
        b.claim("steer", [pv] + hq)
        for e in range(N2_BITS):
            if e & 1:
                b.cx(flags[e], pv)
            for j in range(H_BITS):
                if ((e >> 1) >> j) & 1:
                    b.cx(flags[e], hq[j])
    mark_scan = b.checkpoint()

    # ---- 3. normalize: M = n2 << (24 - e) --------------------------------
    # Exactly one flag fires, so M's leading bit is one unconditional X;
    # per e, e Fredkins move n2's low bits (the swapped-in M bits are |0>,
    # so the moves also clear n2) and one CX clears the leading n2 bit.
    # The stage frees n2 mid-body and its inverse must re-claim the
    # recreated word, so its unwind is the un_normalize callback: the
    # clear segment first (recreates n2), the claim, the mant release,
    # then the setup segment (frees M).
    def un_normalize():
        b.set_phase("un-normalize")
        b.mirror(mark_nclear, until=mark_normalize)  # recreates the n2 word
        b.claim("n2", n2b)  # owned from its recreation here until the
        # dist unwind releases and frees it
        b.release("mant", M)
        b.mirror(mark_scan, until=mark_nclear)  # frees M

    with st.stage("normalize", unwind=un_normalize):
        b.set_phase("normalize/setup")
        M = b.qubits(FRAC_BITS + 1)  # bits 0..24; bit 25 of the
        b.claim("mant", M)  # (26,2) word is always 0 and
        b.x(M[FRAC_BITS])  # never allocated
        for e in range(N2_BITS):
            b.set_phase(f"normalize/e{e:02d}")
            for j in range(e):
                b.cswap(flags[e], n2b[j], M[FRAC_BITS - e + j])
            b.cx(flags[e], n2b[e])
        mark_nclear = b.checkpoint()
        b.set_phase("normalize/clear")
        b.release_recycle("n2", n2b)  # all 18 wires provably |0>
    mark_normalize = b.checkpoint()

    # ---- 4. label: 7 comparators drive the label counter -----------------
    with st.stage("label", set_name="label"):
        b.set_phase("label/setup")
        lbl = b.qubits(LABEL_BITS)
        b.claim("label", lbl)
    # The increment's soundness is the telescope identity: the XOR of
    # i ^ (i + 1) over a fired prefix i < m equals m, for every count m
    # the counter can reach (free classical assert; emits no op).
    prefix_xor = 0
    for m_fired in range(INTERVALS):
        assert prefix_xor == m_fired
        prefix_xor ^= m_fired ^ (m_fired + 1)
    for i, threshold in enumerate(thresholds):
        b.set_phase(f"label/b{i + 1:02d}")
        # The comparator runs on M's 24 fractional bits: M's always-set
        # integer bit drops out of [M >= threshold].  The thresholds
        # ascend strictly (asserted), so the label counts the thresholds
        # <= M at 0 Toffoli beyond each comparator.
        label_increment(b, M[:FRAC_BITS], threshold - (1 << FRAC_BITS), i, lbl)

    # ---- coefficient selection: bank-match flag + CX loads ---------------
    # Per-bank match ladders are the published HRS selection this
    # baseline reproduces — one 3-AND match flag rides across all (up to
    # 26) CX loads of the word, 3 Toffoli per bank, 48 per load stage.
    # gadgets.qrom.walk over the 4-wire (label, parity) address would
    # load all 16 banks at 2^4 - 2 = 14 Toffoli per load stage but
    # restructures the HRS per-interval select; gadgets.mcx would
    # pay len(mw) - 1 = 3 Toffoli per set target bit.
    match_wires = list(lbl) + [pv]  # bit 3 = exponent parity

    def coef_load(pattern, word, W):
        """CX-load ``word`` into W under [(label, parity) == pattern]: one
        AND ladder over the 4 match wires (open bits X-conjugated for the
        whole ladder lifetime, releases included).
        """
        flag, rec = and_ladder(b, [(q, (pattern >> j) & 1) for j, q in enumerate(match_wires)])
        xor_word(b, flag, word, W)
        and_ladder_release(b, rec)

    # ---- 5..7. lead/h1/h0: the unpebbled Horner ladder --------------------
    stages = [
        ("lead", 2, "poly2"),  # V2 = c2[bank]
        ("h1", 1, "poly1"),  # V1 = c1[bank] + V2 (x) u
        ("h0", 0, "poly0"),
    ]  # V0 = c0[bank] + V1 (x) u
    V = None
    for stage, deg, setname in stages:
        with st.stage(stage, set_name=setname):
            b.set_phase(f"{stage}/load")
            W = b.qubits(WORD_BITS)
            b.claim(setname, W)
        for pattern in sorted(banks):
            coef_load(pattern, banks[pattern][deg], W)
        if V is not None:
            # u's always-zero bit 25 is not allocated: M holds bits 0..24
            hrs_mul_rows(b, stage, V, W, M, FRAC_BITS)
        V = W
    V0 = V  # the poly0 word: V0 < 2^25 (checked)

    # ---- 8. rescale: barrel-rotate [pad | V0] right by h ------------------
    # Four controlled cyclic rotations (s = 1, 2, 4, 8 under h's bits);
    # each cycle of length L costs L - 1 Fredkins, 34 - gcd(34, s) total.
    # Afterwards bits [8:33) hold V0 >> h and the truncated low bits sit
    # in the pad wires (reversible; the mirror unwinds them).
    with st.stage("rescale", set_name="pad"):
        b.set_phase("rescale/pad")
        pad = b.qubits(PAD_BITS)
        b.claim("pad", pad)
    w34 = list(pad) + list(V0)  # ty: ignore[invalid-argument-type]  # stages is non-empty, so V is set

    def rot_right(ctrl, wires, s):
        n = len(wires)
        g = math.gcd(n, s)
        for c0 in range(g):
            cycle = [(c0 + s * t) % n for t in range(n // g)]
            for a, c in zip(cycle, cycle[1:]):
                b.cswap(ctrl, wires[a], wires[c])

    for s, hbit in ((1, hq[0]), (2, hq[1]), (4, hq[2]), (8, hq[3])):
        b.set_phase(f"rescale/s{s}")
        rot_right(hbit, w34, s)

    mark_round = b.checkpoint()

    # ---- 9. accumulate: k += (V0 >> h) mod 2^25 (not mirrored) -----------
    # Bit 33 of the rotated word is provably 0 (V0 < 2^25 and the wrapped
    # positions are pad zeros), so the 25-bit window [8:33) is the whole
    # value; truncation is the checked rescale rule.
    b.set_phase("accumulate/add")
    gidney_add(b, w34[PAD_BITS : PAD_BITS + ACC_BITS], KQ)

    # ---- 10. un-*: Bennett mirror of stages 8..1 --------------------------
    # st.unwind replays the recorded stages in reverse.  Per standard
    # stage: set_phase(un-<name>); mirror the body (the stage's wires
    # return to |0>); release the ownership set; mirror the head (the
    # frees) — so release precedes free through the whole teardown, and
    # the reflected sub-phase labels come from each segment's own forward
    # marks.  The scan and normalize stages contribute their unwind=
    # callbacks instead (two surviving sets; the n2 re-claim).  The
    # segment boundaries partition the forward item range exactly, and
    # set_phase/claim/release/checkpoint emit no ops, so together the
    # calls emit the same reversed item range as one whole-region mirror
    # and the op stream is byte-identical to it.
    st.unwind(until=mark_round)

    return b, regs


def build():
    """Run the raw build (also the profiler tools' entry): load the tables
    and construct the circuit.  Returns (Builder, register dict).
    """
    return build_circuit(*load_tables())


if __name__ == "__main__":
    b, _regs = build()
    # The peak sits at the widest multiplier row (h0/b24) and decomposes
    # as the persistent file (I/O + flags + steer + mantissa + label + 3
    # words) + that row's transients (25 chain ANDs + 25 addend ANDs = 50).
    assert_metrics(b, __file__)
    emit_and_report(b)
