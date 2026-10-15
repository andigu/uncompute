#!/usr/bin/env python3
"""lattice-gauge-plaquette circuits/discovered: the global
inversion-walk chain on the net-flip exact-DP schedule, accumulating
all 576 Wilson-loop traces in one walk.  The circuit adds the weighted sum of the 192
plaquette and 384 rectangle traces of a 4^3 periodic lattice of
Sigma(72x3) links (9 qubits each) into a 51-bit accumulator modulo
2^51, restoring every link.

The build is loop-form: the schedule's raster periodicity (three row
programs; constants.json) becomes op-stream LOOP structure directly.
Encode and decode are one repeat(192) each with the link wires affine
in the loop index; the walk rides a slab conveyor -- Swap-rotations of
the link array between rows and planes keep every row program on one
fixed set of frame wires, so each of the three row bodies is recorded
once and repeated.  The conveyor is Clifford-only (no Toffoli, no
allocations) and nets to the identity before decode.

Certified cost: 172,128 Toffoli x 1,859 peak qubits = 319,985,952;
io_qubits 1,779.

RUN: uv run python circuit.py -- re-derives every table from the
group generators, builds the loop-form circuit, asserts the certified
cost, and emits to $UNCOMPUTE_OPS_PATH (else ./ops.bin).
IMPORTING this module runs the same build + emission by design;
build() is the raw-build accessor for the profilers.  Evaluate:
python tools/eval.py lattice-gauge-plaquette.

README.md in this folder describes the construction.
"""

import functools
import json
import pathlib

from uncompute import Builder
from uncompute.builder import WireList
from uncompute.gadgets import declare_registers
from uncompute.ops import OpKind
from uncompute.testing import assert_metrics, emit_and_report, load_contract

print = functools.partial(print, flush=True)

CH_PEAK = 1859  # the certified peak (README, Cost)

# ---------------------------------------------------------------------------
# Contract constants (README section 3.2; PARAMS-DERIVATION.md sections 4 and 7),
# loaded from the task-root contract.json — the sidecar shared by the
# grader and every circuit (grader.py asserts it against its own inline
# derivation).
# ---------------------------------------------------------------------------
_CONTRACT = load_contract(__file__)
assert _CONTRACT["m"] == 51
A_P = int(_CONTRACT["a_P"], 16)
A_R = int(_CONTRACT["a_R"], 16)
# Direct addends: a_P is odd (adds touch bits [0,51)); a_R = ...0b1010
# (bits [1,51)).  No accumulator rotation: the adds are full-width exact.
assert A_P % 2 == 1 and A_R % 2 == 0 and (A_R >> 1) % 2 == 1


# ---------------------------------------------------------------------------
# A. Exact group model.  Sigma(72x3) is built over Z[w] (w a primitive cube
# root of unity, elements a+b*w as integer pairs) as 3x3 matrices with a
# power-of-3 denominator, exactly as in reference.py and the grader.
# ---------------------------------------------------------------------------
def zm(x, y):
    a, b = x
    c, d = y
    return (a * c - b * d, a * d + b * c - b * d)


def za(x, y):
    return (x[0] + y[0], x[1] + y[1])


def zn(x):
    return (-x[0], -x[1])


def zc(x):
    a, b = x
    return (a - b, -b)


Z0 = (0, 0)
ONE = (1, 0)
W = (0, 1)
W2 = (-1, -1)


def wp(e):
    return [ONE, W, W2][e % 3]


def norm(m, k):
    while k > 0 and all(a % 3 == 0 and b % 3 == 0 for (a, b) in m):
        m = tuple((a // 3, b // 3) for (a, b) in m)
        k -= 1
    return (m, k)


def mmul(A, B):
    ma, ka = A
    mb, kb = B
    out = []
    for i in (0, 3, 6):
        for j in (0, 1, 2):
            s = Z0
            for l in (0, 1, 2):
                s = za(s, zm(ma[i + l], mb[3 * l + j]))
            out.append(s)
    return norm(tuple(out), ka + kb)


def mdag(A):
    m, k = A
    return (tuple(zc(m[3 * j + i]) for i in range(3) for j in range(3)), k)


ID = (tuple(ONE if i == j else Z0 for i in range(3) for j in range(3)), 0)


def mdet1(A):
    m, k = A
    d = Z0
    d = za(d, zm(m[0], zm(m[4], m[8])))
    d = za(d, zm(m[1], zm(m[5], m[6])))
    d = za(d, zm(m[2], zm(m[3], m[7])))
    d = za(d, zn(zm(m[2], zm(m[4], m[6]))))
    d = za(d, zn(zm(m[0], zm(m[5], m[7]))))
    d = za(d, zn(zm(m[1], zm(m[3], m[8]))))
    return d[1] == 0 and d[0] == 3 ** (3 * k)


def mpow(A, n):
    R = ID
    for _ in range(n):
        R = mmul(R, A)
    return R


def mk(rows):
    return (tuple(rows[i][j] for i in range(3) for j in range(3)), 0)


wI = mk([[W, Z0, Z0], [Z0, W, Z0], [Z0, Z0, W]])
Cg = mk([[ONE, Z0, Z0], [Z0, W, Z0], [Z0, Z0, W2]])
Eg = mk([[Z0, ONE, Z0], [Z0, Z0, ONE], [ONE, Z0, Z0]])
eta = (W2[0] - W[0], W2[1] - W[1])


def frac(rows):
    return norm(tuple(zm(eta, rows[i][j]) for i in range(3) for j in range(3)), 1)


Vg = frac([[ONE, ONE, ONE], [ONE, W, W2], [ONE, W2, W]])
assert mmul(Vg, mdag(Vg)) == ID and mdet1(Vg) and mpow(Vg, 4) == ID


def ce(q, r):
    return mmul(mpow(Cg, q), mpow(Eg, r))


delta = {}
for p in range(3):
    for q in range(3):
        for r in range(3):
            delta[mmul(mpow(wI, p), ce(q, r))] = (p, q, r)
assert len(delta) == 27
LAM = None
for l in range(3):
    if mmul(Eg, Cg) == mmul(mpow(wI, l), mmul(Cg, Eg)):
        LAM = l
assert LAM in (1, 2)


def closure(gens, cap=700):
    seen = {ID}
    frontier = [ID]
    while frontier:
        nf = []
        for A in frontier:
            for g in gens:
                B = mmul(A, g)
                if B not in seen:
                    seen.add(B)
                    nf.append(B)
                    if len(seen) > cap:
                        return None
        frontier = nf
    return seen


sig36 = closure([Cg, Eg, wI, Vg], cap=200)
assert sig36 is not None and len(sig36) == 108
V2m = mpow(Vg, 2)
Vinvm = mpow(Vg, 3)
Vp = [mpow(Vg, m) for m in range(4)]

# The contract X generator completing Sigma(36x3) -> Sigma(72x3): a unitary
# determinant-1 normalizer of the diagonal subgroup, outside sig36, whose
# closure with the other generators has exactly 216 elements.
Xc = frac([[ONE, ONE, W2], [ONE, W, W], [W, ONE, W]])


def member_ok(A):
    Ad = mdag(A)
    if mmul(A, Ad) != ID or not mdet1(A):
        return False
    if mmul(mmul(A, Cg), Ad) not in delta:
        return False
    if mmul(mmul(A, Eg), Ad) not in delta:
        return False
    if A in sig36:
        return False
    cl = closure([Cg, Eg, wI, Vg, A], cap=300)
    return cl is not None and len(cl) == 216


assert member_ok(Xc), "contract X generator invalid"
G = closure([Cg, Eg, wI, Vg, Xc], cap=300)
assert G is not None and len(G) == 216
Xcp = [ID, Xc]


# ---------------------------------------------------------------------------
# B. Codes and coordinates.  The contract 9-bit code packs
# g = w^p C^q E^r V^(2s+t) X^u; the datapath works in "mirror coordinates"
# (P, Q, R, M, U) with g = w^P V^M X^U C^Q E^R (the V/X factors pulled to
# the FRONT), because then right-multiplication only permutes (Q, R) through
# fixed conjugation tables and adds trits -- the whole chain state fits on
# three one-hot trit rails + 3 flags.
# ---------------------------------------------------------------------------
def fields(c):
    return (c & 3, (c >> 2) & 3, (c >> 4) & 3, (c >> 6) & 1, (c >> 7) & 1, (c >> 8) & 1)


valid_codes = [
    c for c in range(512) if (c & 3) != 3 and ((c >> 2) & 3) != 3 and ((c >> 4) & 3) != 3
]
assert len(valid_codes) == 216
code_mat = {}
mat_code = {}
for c in valid_codes:
    p, q, r, s, t, u = fields(c)
    Mx = mmul(mmul(mmul(mmul(mpow(wI, p), mpow(Cg, q)), mpow(Eg, r)), Vp[2 * s + t]), Xcp[u])
    assert Mx not in mat_code
    code_mat[c] = Mx
    mat_code[Mx] = c


def conj_tab(A):
    """(q, r) -> (phase, q', r') with A (C^q E^r) A^-1 = w^phase C^q' E^r'."""
    Ai = mdag(A)
    tab = {}
    for q in range(3):
        for r in range(3):
            ph, q2, r2 = delta[mmul(mmul(A, ce(q, r)), Ai)]
            tab[(q, r)] = (ph, q2, r2)
    return tab


conjV = conj_tab(Vg)
conjVi = conj_tab(Vinvm)
conjV2 = conj_tab(V2m)
for q in range(3):
    for r in range(3):
        assert conjVi[(q, r)][1:] == (r, (3 - q) % 3)
        assert conjV[(q, r)][1:] == ((3 - r) % 3, q)
        assert conjV2[(q, r)] == (0, (3 - q) % 3, (3 - r) % 3)
        if q == 0 or r == 0:
            assert conjVi[(q, r)][0] == 0 and conjV[(q, r)][0] == 0

# The V-conjugation phase is proportional to q*r; GAM is the coboundary
# section exponent that makes the primed phase P' = P + GAM*Q*R carry-free
# through V-stages, and CO_Q/CO_R are the residual q*r cross terms the
# multiply must add (the fused rotation clusters in prot_cluster).
CVI = None
for c_ in range(3):
    if all(conjVi[(q, r)][0] == (c_ * q * r) % 3 for q in range(3) for r in range(3)):
        CVI = c_
        break
assert CVI is not None, "V phase not proportional to qr"
GAM = (3 - CVI) % 3
assert all(conjV[(q, r)][0] == (CVI * q * r) % 3 for q in range(3) for r in range(3))
CO_R = (LAM + GAM) % 3
CO_Q = GAM % 3
assert (LAM + 2 * GAM) % 3 == 0, "QRC must vanish for this datapath"
# Pins: the vanishing-coefficient guards below exist but must not
# fire for this group model.
assert (CO_Q, CO_R) == (1, 2)
assert (-(LAM + GAM)) % 3 == 1, "qr_corr_ops coefficient must be 1"
print(f"section: CVI={CVI} GAM={GAM} LAM={LAM} CO_R={CO_R} CO_Q={CO_Q}")


def xi_groups(conjT, permT):
    """Cells (q, r) where conjugating by T needs a residual w-phase after
    the GAM section is applied, grouped by phase amount.
    """
    g = {1: [], 2: []}
    for q in range(3):
        for r in range(3):
            q2, r2 = permT[(q, r)]
            amt = (conjT[(q, r)][0] + GAM * (q2 * r2 - q * r)) % 3
            if amt:
                g[amt].append((q, r))
    return g


def perm_swaps(pt):
    """Decompose a 9-cell permutation into transpositions (J-cell swaps)."""
    sigma = {3 * q + r: 3 * pt[(q, r)][0] + pt[(q, r)][1] for q in range(3) for r in range(3)}
    seen = set()
    swaps = []
    for st in range(9):
        if st in seen or sigma[st] == st:
            seen.add(st)
            continue
        cyc = [st]
        cur = sigma[st]
        while cur != st:
            cyc.append(cur)
            cur = sigma[cur]
        seen.update(cyc)
        swaps += [(cyc[i], cyc[i + 1]) for i in range(len(cyc) - 1)][::-1]
    return swaps


def lin_st(mset):
    """Affine form a ^ (b & m1) ^ (c & m0) matching a subset of {0..3}."""
    for a in range(2):
        for bb in range(2):
            for cc in range(2):
                if all(
                    (a ^ (bb & (m >> 1)) ^ (cc & (m & 1))) == (1 if m in mset else 0)
                    for m in range(4)
                ):
                    return (a, bb, cc)
    return None


# ---------------------------------------------------------------------------
# C. The internal X representative and the mirror algebra.  The chain works
# with an internal generator Xo (in place of the contract Xc) chosen so
# that X^2 = V^2, X V X^-1 = V^-1, and
# Xc*Xo^-1 lies in Delta(27) = <w,C,E> -- then the
# U-flag composes linearly with the M-counter and the encode transform
# (contract code -> internal code) is cheap.  No tie-break is needed: the
# algebraic filter admits EXACTLY ONE candidate (asserted; a group-model
# change that widens the set fails the build).
# ---------------------------------------------------------------------------
XO_CANDS = []
for g in sorted(G):
    if g in sig36:
        continue
    if mpow(g, 2) != V2m:
        continue
    if mmul(mmul(g, Vg), mdag(g)) != Vinvm:
        continue
    if mmul(Xc, mdag(g)) in delta:
        XO_CANDS.append(g)
assert len(XO_CANDS) == 1, "internal X representative is unique"
Xo = XO_CANDS[0]

# Encode displacement D = Xc * Xo^-1 = w^aD C^bD E^cD, conjugated through
# each V-power m: PHI/BM/CM are the (phase, C, E) trit increments the
# U-conditioned encode transform must add per m.
Dabc = delta[mmul(Xc, mdag(Xo))]
a_D, b_D, c_D = Dabc
PHI = {}
BM = {}
CM = {}
for m in range(4):
    ph, bm, cm = delta[mmul(mmul(Vp[m], ce(b_D, c_D)), Vp[(4 - m) % 4])]
    PHI[m] = (a_D + ph) % 3
    BM[m] = bm
    CM[m] = cm
assert set(PHI.values()) <= {0, 1}  # pin: no PHI value reaches 2
print(f"encode D = w^{a_D} C^{b_D} E^{c_D} ; PHI={PHI} BM={BM} CM={CM}")

Xp = [ID, Xo]

# Mirror coordinates: g = w^P V^M X^U C^Q E^R.
rev_mat = {}
mat_rev = {}
for P in range(3):
    for Q in range(3):
        for R in range(3):
            for M in range(4):
                for U in range(2):
                    Mx = mmul(mmul(mpow(wI, P), mmul(Vp[M], Xp[U])), ce(Q, R))
                    rev_mat[(P, Q, R, M, U)] = Mx
                    mat_rev[Mx] = (P, Q, R, M, U)
assert len(mat_rev) == 216

conjX = conj_tab(Xo)
conjXi = conj_tab(mdag(Xo))
assert conjXi[(0, 0)] == (0, 0, 0) and conjX[(0, 0)] == (0, 0, 0)
permXi = {k: v[1:] for k, v in conjXi.items()}
permX = {k: v[1:] for k, v in conjX.items()}

# Both xi-phase groups of the selected Xo are EMPTY: the U-stage of the
# chain multiply needs J-cell swaps only, no accP correction.  Asserted:
# a nonempty group would need an accP rotation between fold and swaps,
# and the build fails closed rather than emit without it.
grpXi2 = xi_groups(conjXi, permXi)
grpX2 = xi_groups(conjX, permX)
assert grpXi2 == {1: [], 2: []} and grpX2 == {1: [], 2: []}, (
    "xi phase groups must vanish for the correction-free u-stage"
)
SWAPS_Xi = perm_swaps(permXi)
SWAPS_X = perm_swaps(permX)
assert len(SWAPS_Xi) == 6 and len(SWAPS_X) == 6

# ---------------------------------------------------------------------------
# D. Trace classes in mirror coordinates.  For fixed (M, U, Q, R) the trace
# of w^P V^M X^U C^Q E^R is 0 ('Z'), a unit-class value ('U': f = 2*sg or
# -sg depending on P), or the top class ('T': f = 6 or -3, only at the
# origin).  The extraction gadget synthesizes a 6-wire indicator model
# of f.
# ---------------------------------------------------------------------------
CLS = {}
for M in range(4):
    for U in range(2):
        for Q in range(3):
            for R in range(3):
                A = mmul(mmul(Vp[M], Xp[U]), ce(Q, R))
                m9, k = A
                t = za(za(m9[0], m9[4]), m9[8])
                cls = None
                if t == (0, 0):
                    cls = ("Z", 0, 0)
                else:
                    for sg in (1, -1):
                        for e in range(3):
                            bb = wp(e)
                            if t == (sg * bb[0] * 3**k, sg * bb[1] * 3**k):
                                cls = ("U", sg, e)
                            if t == (3 * sg * bb[0] * 3**k, 3 * sg * bb[1] * 3**k):
                                cls = ("T", sg, e)
                assert cls is not None, "unexpected trace value"
                if cls[0] == "T":
                    assert (M, U, Q, R) == (0, 0, 0, 0) and cls[1] == 1 and cls[2] == 0
                CLS[(M, U, Q, R)] = cls

# ---------------------------------------------------------------------------
# E. F2 rank decompositions for the extraction scans.  Each per-(M,U)-row
# 9-cell indicator table is decomposed as offset ^ complement ^ sum of
# rank-1 (rail-parity AND cell-parity) products; the exhaustive scan over
# the 512 offsets (and the delta/gamma window and 12/1n/2n pairings)
# minimizes the number of paid Toffoli products.
# ---------------------------------------------------------------------------
F9m = (1 << 9) - 1


def decomp8x9(S):
    best = None
    for v in range(512):
        eps = 0
        piv = []
        coeffs = []
        for i in range(8):
            rw = S[i] ^ v
            if rw & 1:
                rw ^= F9m
                eps |= 1 << i
            cm = 0
            for j in range(len(piv)):
                pv = piv[j]
                if rw & (pv & -pv):
                    rw ^= pv
                    cm |= 1 << j
            if rw:
                piv.append(rw)
                cm |= 1 << (len(piv) - 1)
            coeffs.append(cm)
        if best is None or len(piv) < best[0]:
            best = (len(piv), v, eps, list(piv), list(coeffs))
    _cost, v, eps, piv, coeffs = best
    for i in range(8):
        acc = v ^ (F9m if (eps >> i) & 1 else 0)
        for j in range(len(piv)):
            if (coeffs[i] >> j) & 1:
                acc ^= piv[j]
        assert acc == S[i], "decomposition reconstruction failed"
    return best


def amt_tables(dlt, gg):
    S1 = []
    S2 = []
    SNE = []
    for M in range(4):
        for U in range(2):
            r1 = r2 = rne = 0
            for Q in range(3):
                for R in range(3):
                    cell = 1 << (3 * Q + R)
                    if (M, U) == (0, 0):
                        amt = dlt if (Q, R) == (0, 0) else gg
                    else:
                        cls = CLS[(M, U, Q, R)]
                        amt = (cls[2] - GAM * Q * R + dlt) % 3 if cls[0] == "U" else 0
                    if amt == 1:
                        r1 |= cell
                    elif amt == 2:
                        r2 |= cell
                    if amt:
                        rne |= cell
            S1.append(r1)
            S2.append(r2)
            SNE.append(rne)
    return S1, S2, SNE


EBEST = None
for _d in range(3):
    for _g in range(3):
        _S1, _S2, _SNE = amt_tables(_d, _g)
        d1 = decomp8x9(_S1)
        d2 = decomp8x9(_S2)
        dn = decomp8x9(_SNE)
        for kind, da, db in (("12", d1, d2), ("1n", d1, dn), ("2n", d2, dn)):
            cst = da[0] + db[0]
            if EBEST is None or cst < EBEST[0]:
                EBEST = (cst, kind, da, db, _d)
EIDX = EBEST[4]
print(f"EXT scan: paid={EBEST[0]} kind={EBEST[1]} delta={EIDX}")

# nu (off-diagonal unit-class) and sg (negative-sign) indicators: both
# decompose at rank 0 -- pure rail/cell parities, no paid Toffoli.
SNU = []
SSGb = []
for M in range(4):
    for U in range(2):
        rn = rs = 0
        if (M, U) != (0, 0):
            for Q in range(3):
                for R in range(3):
                    cls = CLS[(M, U, Q, R)]
                    if cls[0] == "U":
                        rn |= 1 << (3 * Q + R)
                        if cls[1] == -1:
                            rs |= 1 << (3 * Q + R)
        SNU.append(rn)
        SSGb.append(rs)
DNU = decomp8x9(SNU)
DSG = None
for r0v in (0, F9m):
    dd = decomp8x9([r0v] + SSGb[1:])
    if DSG is None or dd[0] < DSG[0]:
        DSG = dd

# The fast encode-transform predicate: all four PHI/BM/CM columns are
# affine in (s, t), so the U-conditioned recode runs on linear forms
# instead of a generic per-(s,t) case tree.  Asserted: no slow path
# exists, so a group model that breaks the predicate fails the build.
_B1 = {m for m in range(4) if BM[m] == 1}
_C1 = {m for m in range(4) if CM[m] == 1}
_P1 = {m for m in range(4) if (PHI[m] % 3) == 1}
_P2 = {m for m in range(4) if (PHI[m] % 3) == 2}
XF_FAST = (
    all(BM[m] in (1, 2) for m in range(4))
    and all(CM[m] in (1, 2) for m in range(4))
    and lin_st(_B1) is not None
    and lin_st(_C1) is not None
    and ((not _P2 and lin_st(_P1) is not None) or (not _P1 and lin_st(_P2) is not None))
)
assert XF_FAST, "fast encode predicate must hold"

# ---------------------------------------------------------------------------
# F. The loop schedule.  Conventions pinned by PARAMS-DERIVATION.md
# sections 3 and 7 / reference.py: site s = x + 4y + 16z, link l = 3s + mu,
# inverses on return links.
# ---------------------------------------------------------------------------

# The global chain-walk schedule.  Ops: 'f'/'i' push the link / its
# inverse (35-Toffoli multiplies); 'li'/'ul' open/close at the identity
# with a 1-Toffoli inverse-load / its reverse; 'v' inverts the register in
# place (21 Toffoli) so subsequent right-multiplies edit the word's other
# end; 'XP'/'XR' fire a plaquette/rectangle trace-add.  Exact DP optimum
# of the net-flip-relaxed morph model (odd inversion counts per morph
# allowed): 3,136 multiplies + 480 inversions + 64 anchors.
#
# The DP optimum is row-periodic: every y-junction row of the x-fastest
# raster is one lattice translate of a single row program, every
# plane-crossing row another, so constants.json stores THREE row
# programs (open_row, then y_row x12 and z_row x3 by translation) plus
# the closing tail, and this build repeats them through a slab conveyor
# (section L) instead of unrolling 4,256 tokens.  y_row/z_row link ids
# are FRAME-LOCAL: link 3*(x + 4*dy + 16*dz) + mu addresses the link dy
# rows / dz planes above the row being processed (mod 4); open_row and
# tail are absolute.  gen_constants.py runs the search and writes the
# file.
_CONSTS = json.loads((pathlib.Path(__file__).resolve().parent / "constants.json").read_bytes())


def _decode_prog(blobs):
    KIND = {"f": "f", "i": "i", "l": "li", "u": "ul", "v": "inv", "P": "XP", "R": "XR"}
    out = []
    for tok in blobs.split():
        kd = KIND[tok[0]]
        out.append((kd, int(tok[1:])) if tok[0] in "filu" else (kd,))
    return out


OPEN_ROW = _decode_prog(" ".join(_CONSTS["open_row"]))
Y_ROW = _decode_prog(" ".join(_CONSTS["y_row"]))
Z_ROW = _decode_prog(" ".join(_CONSTS["z_row"]))
TAIL = _decode_prog(" ".join(_CONSTS["tail"]))

# gen_constants.reconstruct is the one authority for row-form -> flat
# translation.

# ---------------------------------------------------------------------------
# G. Full-width exact accumulation.  Each trace-add folds a_P*f or a_R*f
# directly into k over bits [LO, 51): nothing is windowed, nothing is
# dropped -- the emitted map is k += a_P*S_P + a_R*S_R mod 2^51 exactly
# (the graded ideal map, which the evaluator checks end to end).
# ---------------------------------------------------------------------------
WBITS = 51

# ---------------------------------------------------------------------------
# H. Wire map.  Only the 1,779 I/O wires (192 links x 9, then k) are
# persistent.  The 80-wire working set lives and dies with its data: the
# nine hot utility wires span the whole program and the 12-wire chain
# state spans the loops stage (explicit stage-boundary lifecycles); the
# other 59 wires -- the J grid, the extraction scaffolding and class
# basis, and the 30 dedicated adder carries -- are allocated when their gadget
# builds them and freed when it (or its measurement mirror) tears them
# down.  At every plaquette trace-add ripple all 80 are simultaneously
# live, so the skyline peaks at exactly 1,859 (rectangle adds top out at
# 1,858: LO_R = 1 drops one carry) while breathing everywhere else.
# ---------------------------------------------------------------------------
NIO = 1779
links = [[9 * l + i for i in range(9)] for l in range(192)]  # wires 0..1727
kreg = list(range(1728, 1779))  # wires 1728..1778
accP = [1779, 1780, 1781]  # one-hot rails of the primed phase trit P'
Qm = [1782, 1783, 1784]  # one-hot rails of the C-exponent trit Q
Rm = [1785, 1786, 1787]  # one-hot rails of the E-exponent trit R
accS, accT, accU = 1788, 1789, 1790  # V-counter bits M = 2S+T; X flag U
BANK = list(range(1791, 1829))  # the 38-wire pooled ancilla bank
candB = BANK[0:4]
xorB0 = BANK[4]
ctlB = BANK[5:7]
J = BANK[7:16]  # 3x3 one-hot product grid J[3q+r] = [Q=q][R=r]
a_st = BANK[16]
r10 = BANK[17]
r01 = BANK[18]
r00 = BANK[19]  # M-decoder
rMU1 = BANK[20:24]
rMU0 = BANK[24:28]  # (M,U) one-hot row rails
nuq = BANK[28]
sgq = BANK[29]  # unit-class / negative-sign flags
scr0 = BANK[30]
t1q = BANK[31]
f2q = BANK[32]
t2q = BANK[33]
fm1q = BANK[34]
cen2q = BANK[35]
f6q = BANK[36]  # trace-class basis wires
addd = BANK[37]  # the full-width adder's addend wire
ecq = ctlB[0]  # e-copy flag lives on ctlB[0]: free during extraction
CARRY = list(range(1829, 1859))  # 30 dedicated full-width adder carries
# lifecycle classes (see section I): breathing = everything except the
# hot utilities and the chain state.  Two UTIL members -- xor0 and the
# addend wire -- carry data only in the loops stage yet ride this
# whole-program block; both are live at every trace-add ripple, so the
# whole-program allocation adds nothing to the peak.  The other seven
# utilities are hot in all three stages.
UTIL = list(candB) + [xorB0] + list(ctlB) + [scr0, addd]
CHAINW = list(accP) + list(Qm) + list(Rm) + [accS, accT, accU]
BREATHING = {w for w in range(1779, 1859) if w not in UTIL and w not in CHAINW}
# The working set: 9 hot utilities + 12 chain wires + 59 breathing
# wires = all 80 above the 1,779 I/O; the bank holds 38.
assert len(UTIL) + len(CHAINW) + len(BREATHING) == CH_PEAK - NIO

# The profiling contract consumed by tools/liveness_profile.py and
# tools/occupancy_profile.py in builder mode: WIRE_TAGS maps each wire
# to its ownership-set name, claimed at every allocation of that wire
# and released before every free by the emission layer (section L).
WIRE_TAGS = {}


# Ownership sets (the meta.json roles; hierarchical names coarsen to
# the set at the first '/').  The persistent I/O sets g and k are
# block-claimed in section L; every working wire is tagged here and
# claimed/released by the replay loop at each alloc/free.  A wire keeps
# its set through every service it performs (the J grid and the decoder
# rows also serve as adder carries inside kadd_ops).
for _i, _q in enumerate(accP):
    WIRE_TAGS[_q] = f"chain/accP/{_i}"
for _i, _q in enumerate(Qm):
    WIRE_TAGS[_q] = f"chain/Q/{_i}"
for _i, _q in enumerate(Rm):
    WIRE_TAGS[_q] = f"chain/R/{_i}"
WIRE_TAGS[accS] = "chain/M/1"
WIRE_TAGS[accT] = "chain/M/0"
WIRE_TAGS[accU] = "chain/U"
for _i, _q in enumerate(candB):
    WIRE_TAGS[_q] = f"util/cand/{_i}"
WIRE_TAGS[xorB0] = "util/xor0"
for _i, _q in enumerate(ctlB):
    WIRE_TAGS[_q] = f"util/ctl/{_i}"
for _i, _q in enumerate(J):
    WIRE_TAGS[_q] = f"scaffold/J/{_i}"
WIRE_TAGS[a_st] = "scaffold/row11"
WIRE_TAGS[r10] = "scaffold/row10"
WIRE_TAGS[r01] = "scaffold/row01"
WIRE_TAGS[r00] = "scaffold/row00"
for _i, _q in enumerate(rMU1):
    WIRE_TAGS[_q] = f"scaffold/railU1/{_i}"
for _i, _q in enumerate(rMU0):
    WIRE_TAGS[_q] = f"scaffold/railU0/{_i}"
WIRE_TAGS[nuq] = "scaffold/nu"
WIRE_TAGS[sgq] = "scaffold/sg"
WIRE_TAGS[scr0] = "util/scr"
WIRE_TAGS[t1q] = "basis/t1"
WIRE_TAGS[f2q] = "basis/f2"
WIRE_TAGS[t2q] = "basis/t2"
WIRE_TAGS[fm1q] = "basis/fm1"
WIRE_TAGS[cen2q] = "basis/cen2"
WIRE_TAGS[f6q] = "basis/f6"
WIRE_TAGS[addd] = "util/addend"
for _i, _q in enumerate(CARRY):
    WIRE_TAGS[_q] = f"carry/{_i}"

ROWQ = []
for M in range(4):
    for U in range(2):
        ROWQ.append(rMU1[M] if U else rMU0[M])


def rails_of(mask):
    return tuple(ROWQ[i] for i in range(8) if (mask >> i) & 1)


def cells_of(mask):
    return tuple(J[c] for c in range(9) if (mask >> c) & 1)


def bx_args(d):
    """decomp8x9 output -> (paid rank-1 products, free parity wires, const)."""
    _cost, v, eps, piv, coeffs = d
    prods = []
    for j in range(len(piv)):
        um = 0
        for i in range(8):
            if (coeffs[i] >> j) & 1:
                um |= 1 << i
        prods.append((rails_of(um), cells_of(piv[j]), 0))
    oc = 0
    if bin(eps).count("1") > 4:
        eps ^= 255
        oc ^= 1
    if bin(v).count("1") > 4:
        v ^= F9m
        oc ^= 1
    frees = rails_of(eps) + cells_of(v)
    return tuple(prods), frees, oc


# ---------------------------------------------------------------------------
# I. Macro vocabulary and its expansion to flat ops.  Gadgets are authored
# as macro tuples, inverted at the macro level, and expanded 1:1 into
# the flat op-tuple program.  Gadget pops and whole-gadget mirrors at
# the emission sites (emit_seg rev=True) are derived by inv_op -- the
# guard against copy-paste drift between a build and its mirror.  The
# in-gadget scaffold unbuilds (J_unbuild_ops, shared by ustage_ops and
# invert_macros, and build_ext_ops' Clifford-only teardown) are
# hand-written:
#
#     ('cx'/'x'/'ccx', ...)      as-is
#     ('cswap', c, a, b)        CX-conjugated Toffoli
#     ('inc3', b0, b1, amt, c)  (controlled) +amt mod 3 on a binary trit
#     ('andc', w, a, b)         w (a pooled |0> wire) ^= a AND b  [1 Toffoli]
#     ('andr', w, a, b)         release w = a AND b by X-basis measurement
#                               + one classically-conditioned CZ  [0 Toffoli]
#     ('bx', t, prods, frees, oc)   t ^= oc ^ frees-parity ^ sum of rank-1
#                               (parity AND parity) products  [1 T/product]
#     ('mx', t, prods, frees, oc)   measurement-based release of the same
#
# The macro expansion lowers to the flat vocabulary x/cx/ccx/alloc/free
# plus the two measurement kinds:
#     ("hmr", q, m)   X-basis demolition measurement of q into classical
#                     bit m; q ends |0> but stays allocated (pooled)
#     ("czc", a, c, m)  CZ(a, c) conditioned on bit m (the phase fixup)
# Both are Toffoli-free and liveness-neutral, and every kind maps 1:1
# onto one Builder call in the replay loop (section L): hmr returns a
# Builder Outcome handle and czc conditions a cz on it.
# The cswap/andc/andr lowerings below are byte-identical to the Builder
# canonical forms (Builder.cswap: CX(b,a) CCX(c,a,b) CX(b,a);
# and_compute: alloc + CCX; and_release: hmr + conditioned CZ + free),
# re-derived here only because the replay maps primitive kinds 1:1 and
# the alloc/claim interleave needs the split ops -- a Builder change to
# those forms cannot move this stream, and vice versa.
# ---------------------------------------------------------------------------
_mbit = [0]

# BREATHING (defined with the wire map above): wires with real
# lifecycles, allocated when their gadget builds them and freed when it
# tears them down.  The nine hot utility wires and the 12-wire chain
# state span whole stages instead; the 1,779 I/O wires are the only
# persistent ones.


# local: Builder.checkpoint/mirror and gadgets.Journal are the library
# reverse pass, but macro inversion here runs pre-Builder on flat op
# lists, and the replay's bare hmr/czc ops are outside mirror's
# invertible vocabulary.
def inv_op(o):
    t = o[0]
    if t == "inc3":
        return ("inc3", o[1], o[2], (3 - o[3]) % 3, o[4])
    if t == "andc":
        return ("andr",) + o[1:]
    if t == "andr":
        return ("andc",) + o[1:]
    if t == "bx":
        return ("mx",) + o[1:]
    if t == "mx":
        return ("bx",) + o[1:]
    if t == "al":
        return ("fr", o[1])
    if t == "fr":
        return ("al", o[1])
    return o  # x, cx, ccx, cswap are self-inverse


def _prep(out, A, B, cf):
    a0, b0 = A[0], B[0]
    for q in A[1:]:
        out.append(("cx", q, a0))
    for q in B[1:]:
        out.append(("cx", q, b0))
    if cf:
        out.append(("x", b0))
    return a0, b0


def _unprep(out, A, B, cf):
    a0, b0 = A[0], B[0]
    if cf:
        out.append(("x", b0))
    for q in B[1:]:
        out.append(("cx", q, b0))
    for q in A[1:]:
        out.append(("cx", q, a0))


def expand(L, rev=False):  # noqa: C901 - flat op-kind dispatch
    """Macro list -> flat low-level op list (x/cx/ccx/hmr/czc)."""
    out = []
    if rev:
        L = [inv_op(o) for o in reversed(L)]
    for o in L:
        t = o[0]
        if t == "cx" or t == "x" or t == "ccx":
            out.append(o)
        elif t == "al":
            out.append(("alloc", o[1]))
        elif t == "fr":
            out.append(("free", o[1]))
        elif t == "cswap":
            c, a1, a2 = o[1], o[2], o[3]
            out += [("cx", a2, a1), ("ccx", c, a1, a2), ("cx", a2, a1)]
        elif t == "inc3":
            b0q, b1q, amt, c = o[1], o[2], o[3], o[4]
            if amt % 3 == 1:
                if c is None:
                    out += [("cx", b1q, b0q), ("cx", b0q, b1q), ("x", b0q)]
                else:
                    out += [("ccx", c, b1q, b0q), ("ccx", c, b0q, b1q), ("cx", c, b0q)]
            elif amt % 3 == 2:
                if c is None:
                    out += [("x", b0q), ("cx", b0q, b1q), ("cx", b1q, b0q)]
                else:
                    out += [("cx", c, b0q), ("ccx", c, b0q, b1q), ("ccx", c, b1q, b0q)]
        elif t == "andc":
            if o[1] in BREATHING:
                out.append(("alloc", o[1]))
            out.append(("ccx", o[2], o[3], o[1]))
        elif t == "andr":
            m = _mbit[0]
            _mbit[0] += 1
            out += [("hmr", o[1], m), ("czc", o[2], o[3], m)]
            if o[1] in BREATHING:
                out.append(("free", o[1]))
        elif t == "bx":
            tgt, prods, frees, oc = o[1], o[2], o[3], o[4]
            if tgt in BREATHING:
                out.append(("alloc", tgt))
            if oc:
                out.append(("x", tgt))
            for q in frees:
                out.append(("cx", q, tgt))
            for A, B, cf in prods:
                a0, b0 = _prep(out, A, B, cf)
                out.append(("ccx", a0, b0, tgt))
                _unprep(out, A, B, cf)
        elif t == "mx":
            tgt, prods, frees, oc = o[1], o[2], o[3], o[4]
            if oc:
                out.append(("x", tgt))
            for q in frees:
                out.append(("cx", q, tgt))
            m = _mbit[0]
            _mbit[0] += 1
            out.append(("hmr", tgt, m))
            for A, B, cf in prods:
                a0, b0 = _prep(out, A, B, cf)
                out.append(("czc", a0, b0, m))
                _unprep(out, A, B, cf)
            if tgt in BREATHING:
                out.append(("free", tgt))
        else:
            raise ValueError(t)
    return out


# ---------------------------------------------------------------------------
# J. The gadget library (macro level), local to this file: the shared
# uncompute.gadgets arithmetic operates on binary registers, and nothing
# there speaks one-hot trit rails or XOR-parity addends, so each gadget
# here has no shared equivalent at these costs.
# ---------------------------------------------------------------------------


def core3_ops(rails, c1, c2, xw):
    """One-hot trit rotate by (c1 + 2*c2) mod 3 (binary control wires),
    3 Toffoli, using xw as the XOR pivot.
    """
    return [
        ("cswap", c1, rails[1], rails[2]),
        ("cx", c1, xw),
        ("cx", c2, xw),
        ("cswap", xw, rails[0], rails[1]),
        ("cx", c1, xw),
        ("cx", c2, xw),
        ("cswap", c2, rails[1], rails[2]),
    ]


def bin3_add(b0q, b1q, y1w, y2w):
    """Binary-coded trit (b0,b1) += (y1 + 2*y2) mod 3, 3 Toffoli."""
    return [
        ("x", b1q),
        ("ccx", y1w, b1q, b0q),
        ("x", b1q),
        ("cx", y2w, y1w),
        ("x", b0q),
        ("ccx", y1w, b0q, b1q),
        ("x", b0q),
        ("cx", y2w, y1w),
        ("x", b1q),
        ("ccx", y2w, b1q, b0q),
        ("x", b1q),
    ]


def prot_cluster(fams):
    """Fused one-hot accP rotation: total amount = sum of family amounts.
    fam ('d', c1w, c2w): amount = c1w + 2*c2w (binary wires).
    fam ('k', acc, bits, k): amount = k*A*j (A the one-hot trit on acc
    rails, j the binary-coded link trit on bits) -- the CO_Q/CO_R cross
    terms.  Each 'k' family costs 3 Karatsuba product cells; adjacent
    same-pair controlled swaps are XOR-merged across families (the fusion
    that makes the cluster cheaper than separate rotations).
    """
    pool = [candB[0], candB[1], candB[2], candB[3], ctlB[0], ctlB[1]]
    L = []
    U = []
    toks = []
    pi = [0]

    def push(kind, ws):
        if toks and toks[-1][0] == kind:
            toks[-1] = (kind, toks[-1][1] ^ set(ws))
        else:
            toks.append((kind, set(ws)))

    for fam in fams:
        if fam[0] == "d":
            push("a", {fam[1]})
            push("b", {fam[1], fam[2]})
            push("a", {fam[2]})
        else:
            acc, bits, k = fam[1], fam[2], fam[3] % 3
            assert k in (1, 2)
            w1 = pool[pi[0]]
            w2 = pool[pi[0] + 1]
            w3 = pool[pi[0] + 2]
            pi[0] += 3
            pr3 = (((acc[1], acc[2]), (bits[0], bits[1]), 0),)
            L += [
                ("andc", w1, acc[1], bits[0]),
                ("andc", w2, acc[2], bits[1]),
                ("bx", w3, pr3, (), 0),
            ]
            U = [
                ("mx", w3, pr3, (), 0),
                ("andr", w2, acc[2], bits[1]),
                ("andr", w1, acc[1], bits[0]),
            ] + U
            d = {w1, w2}
            t = {w3}
            if k == 1:
                push("a", d)
                push("b", t)
                push("a", d ^ t)
            else:
                push("a", d ^ t)
                push("b", t)
                push("a", d)
    for kind, ws in toks:
        wl = sorted(ws)
        if not wl:
            continue
        pr = (accP[1], accP[2]) if kind == "a" else (accP[0], accP[1])
        c0 = wl[0]
        for w in wl[1:]:
            L.append(("cx", w, c0))
        L.append(("cswap", c0, pr[0], pr[1]))
        for w in wl[1:]:
            L.append(("cx", w, c0))
    return L + U


def J_build_ops():
    """Build the 3x3 one-hot grid J[3q+r] = Qm[q] AND Rm[r] from the rails:
    4 Toffoli (the (a,b) in {1,2}^2 cells), the rest by the one-hot
    invariants rail0 = 1 ^ rail1 ^ rail2 (free CX).
    """
    L: list = []
    for a in (1, 2):
        for bb in (1, 2):
            L.append(("andc", J[3 * a + bb], Qm[a], Rm[bb]))
    for a in (1, 2):
        L += [
            ("al", J[3 * a]),
            ("cx", Qm[a], J[3 * a]),
            ("cx", J[3 * a + 1], J[3 * a]),
            ("cx", J[3 * a + 2], J[3 * a]),
        ]
    for bb in (1, 2):
        L += [
            ("al", J[bb]),
            ("cx", Rm[bb], J[bb]),
            ("cx", J[3 + bb], J[bb]),
            ("cx", J[6 + bb], J[bb]),
        ]
    L.append(("al", J[0]))
    L.append(("x", J[0]))
    for i in range(1, 9):
        L.append(("cx", J[i], J[0]))
    return L


def J_fold_ops():
    """Fold the J grid onto the rails -- each Qm rail XORs its row of
    cells, each Rm rail its column (free CX).  Self-inverse: the same
    sequence unfolds.
    """
    L = []
    for a in range(3):
        L += [("cx", J[3 * a], Qm[a]), ("cx", J[3 * a + 1], Qm[a]), ("cx", J[3 * a + 2], Qm[a])]
    for bb in range(3):
        L += [("cx", J[bb], Rm[bb]), ("cx", J[3 + bb], Rm[bb]), ("cx", J[6 + bb], Rm[bb])]
    return L


def J_unbuild_ops():
    """Tear the J grid down: uncompute the derived cells via the one-hot
    invariants (free CX, rows before columns) and release the four AND
    cells by measurement (andr, free).
    """
    L: list = []
    for i in range(1, 9):
        L.append(("cx", J[i], J[0]))
    L.append(("x", J[0]))
    L.append(("fr", J[0]))
    for a in (1, 2):
        L += [
            ("cx", Qm[a], J[3 * a]),
            ("cx", J[3 * a + 1], J[3 * a]),
            ("cx", J[3 * a + 2], J[3 * a]),
            ("fr", J[3 * a]),
        ]
    for bb in (1, 2):
        L += [
            ("cx", Rm[bb], J[bb]),
            ("cx", J[3 + bb], J[bb]),
            ("cx", J[6 + bb], J[bb]),
            ("fr", J[bb]),
        ]
    for a in (1, 2):
        for bb in (1, 2):
            L.append(("andr", J[3 * a + bb], Qm[a], Rm[bb]))
    return L


def qr_swap_ops(tctl):
    """Swap the Q<->R rails under control in 2 Toffoli via the one-hot invariant
    rail0 = 1 ^ rail1 ^ rail2 (rail0 derived, fixed up with free CX).
    """
    pre = [("cx", Qm[1], Qm[0]), ("cx", Qm[2], Qm[0]), ("cx", Rm[1], Rm[0]), ("cx", Rm[2], Rm[0])]
    return pre + [("cswap", tctl, Qm[1], Rm[1]), ("cswap", tctl, Qm[2], Rm[2])] + list(pre)


def ustage_ops(uctl, swaps):
    """U-conditioned (Q,R) permutation: build J, fold J onto the rails,
    apply the 6 controlled J-cell swaps of the X-conjugation permutation,
    unfold and unbuild.  10 Toffoli: the 4-Toffoli J build + the 6 swaps
    (the measurement-based J teardown is free).  The xi-phase correction
    that would sit between fold and swaps VANISHES for the selected Xo
    (both phase groups empty -- asserted at derivation), which is what
    keeps this stage free of an accP rotation.
    """
    L = J_build_ops()
    L += J_fold_ops()
    for i, jj in swaps:
        L.append(("cswap", uctl, J[i], J[jj]))
    L += J_fold_ops()
    L += J_unbuild_ops()
    return L


def stages_fwd_ops(tctl, sctl, uctl):
    """Emit the V/X part of a forward multiply: t-stage (derived-rail Q<->R
    swap), s-stage (double negation), u-stage (X permutation).
    """
    L = qr_swap_ops(tctl)
    L.append(("cswap", sctl, Qm[1], Qm[2]))
    L += [
        ("cx", tctl, xorB0),
        ("cx", sctl, xorB0),
        ("cswap", xorB0, Rm[1], Rm[2]),
        ("cx", tctl, xorB0),
        ("cx", sctl, xorB0),
    ]
    L += ustage_ops(uctl, SWAPS_Xi)
    return L


def stages_inv_ops(tctl, sctl, uctl):
    L = ustage_ops(uctl, SWAPS_X)
    L += qr_swap_ops(tctl)
    L.append(("cswap", sctl, Rm[1], Rm[2]))
    L += [
        ("cx", tctl, xorB0),
        ("cx", sctl, xorB0),
        ("cswap", xorB0, Qm[1], Qm[2]),
        ("cx", tctl, xorB0),
        ("cx", sctl, xorB0),
    ]
    return L


def mul_fwd_ops(g):
    """Chain push: state *= link (stored st2 code on wires g[0..8]).
    35 Toffoli: fused phase cluster (13) + two rail rotations (3 each) +
    V/X stages (14) + the M/U counter update (2).
    """
    p0, p1, q0, q1, r0, r1, sB, tB, uB = g
    fams = [("d", p0, p1)]
    if CO_Q:
        fams.append(("k", Qm, (r0, r1), CO_Q))
    if CO_R:
        fams.append(("k", Rm, (q0, q1), CO_R))
    L = prot_cluster(fams)
    L += core3_ops(Qm, q0, q1, xorB0)
    L += core3_ops(Rm, r0, r1, xorB0)
    L += stages_fwd_ops(tB, sB, uB)
    L += [
        ("cx", accU, accT),
        ("ccx", accT, tB, accS),
        ("cx", accU, accT),
        ("cx", tB, accT),
        ("cx", sB, accS),
        ("ccx", accU, uB, accS),
        ("cx", uB, accU),
    ]
    return L


def mul_inv_ops(g):
    """Chain push of the link INVERSE (not the pop of mul_fwd_ops -- pops
    are exact macro mirrors of whatever was pushed).
    """
    p0, p1, q0, q1, r0, r1, sB, tB, uB = g
    L = stages_inv_ops(tB, sB, uB)
    L += [
        ("cx", accU, accT),
        ("cx", uB, accT),
        ("ccx", accT, tB, accS),
        ("cx", uB, accT),
        ("cx", accU, accT),
        ("ccx", accU, uB, accS),
        ("cx", tB, accT),
        ("cx", sB, accS),
        ("cx", uB, accS),
        ("cx", tB, accS),
        ("cx", uB, accU),
    ]
    fams = [("d", p1, p0)]
    if CO_Q:
        fams.append(("k", Qm, (r0, r1), 3 - CO_Q))
    if CO_R:
        fams.append(("k", Rm, (q0, q1), 3 - CO_R))
    L += prot_cluster(fams)
    L += core3_ops(Qm, q1, q0, xorB0)
    L += core3_ops(Rm, r1, r0, xorB0)
    return L


def invload_ops(g):
    """Chain anchor: load the primed state of g^-1 from the identity rails
    in 1 Toffoli (a 35-Toffoli multiply would do the same from the
    identity; the 64 scheduled anchors save 34 Toffoli each, 2,176
    total).  From the stored code
    (p2,q,r,s,t,u) the inverse state is exactly P' = -p2, Q = -q, R = -r,
    U = u, T = t, S = s^u^t^(t&u); trit negations are rail-swapped CX.
    """
    p0, p1, q0, q1, r0, r1, sB, tB, uB = g
    return [
        ("cx", p1, accP[1]),
        ("cx", p0, accP[2]),
        ("cx", p0, accP[0]),
        ("cx", p1, accP[0]),
        ("cx", q1, Qm[1]),
        ("cx", q0, Qm[2]),
        ("cx", q0, Qm[0]),
        ("cx", q1, Qm[0]),
        ("cx", r1, Rm[1]),
        ("cx", r0, Rm[2]),
        ("cx", r0, Rm[0]),
        ("cx", r1, Rm[0]),
        ("cx", uB, accU),
        ("cx", tB, accT),
        ("cx", sB, accS),
        ("cx", uB, accS),
        ("cx", tB, accS),
        ("ccx", tB, uB, accS),
    ]


def xform_ops(g):
    """In-place contract->internal recode of one link register (17
    Toffoli; with qr_corr_ops the 23-Toffoli encode unit).  Runs on
    linear forms of (s, t) under the fast
    predicate (XF_FAST, asserted): the U-conditioned trit increments
    PHI/BM/CM and the LAM*r*BM phase cross-term via two bin3_adds each.
    """
    p0, p1, q0, q1, r0, r1, sB, tB, uB = g
    y1 = ctlB[0]
    y2 = ctlB[1]
    yp = candB[0]
    ys = candB[1]
    z1 = candB[2]
    zt = candB[3]
    ys2 = scr0

    def uand(lin, tgt):
        a, bb, cc = lin
        if not bb and not cc:
            if a:
                return [("cx", uB, tgt)], [("cx", uB, tgt)]
            return [], []
        t_ = ((uB,), tuple(([sB] if bb else []) + ([tB] if cc else [])), a)
        return [("bx", tgt, (t_,), (), 0)], [("mx", tgt, (t_,), (), 0)]

    L = []
    b_y1, c_y1 = uand(lin_st(_B1), y1)
    L += b_y1
    L += [("cx", uB, y2), ("cx", y1, y2)]
    if _P1 or _P2:
        amtP, Pset = (1, _P1) if _P2 == set() else (2, _P2)
        b_yp, c_yp = uand(lin_st(Pset), yp)
        L += b_yp
        L.append(("inc3", p0, p1, amtP, yp))
        L += c_yp
    k1w, k2w = (y1, y2) if LAM == 1 else (y2, y1)
    prz = (((k1w,), (r0,), 0), ((k2w,), (r1,), 0))
    prt = (((uB,), (r0, r1), 0),)
    L += [("bx", z1, prz, (), 0), ("bx", zt, prt, (), 0), ("cx", z1, zt)]
    L += bin3_add(p0, p1, z1, zt)
    L += [("cx", z1, zt), ("mx", zt, prt, (), 0), ("mx", z1, prz, (), 0)]
    L += bin3_add(q0, q1, y1, y2)
    b_ys, c_ys = uand(lin_st(_C1), ys)
    L += b_ys
    L += [("cx", uB, ys2), ("cx", ys, ys2)]
    L += bin3_add(r0, r1, ys, ys2)
    L += [("cx", uB, ys2), ("cx", ys, ys2)]
    L += c_ys
    L += [("cx", uB, y2), ("cx", y1, y2)]
    L += c_y1
    return L


def qr_corr_ops(g):
    """Apply the coboundary-section correction p2 -= (LAM+GAM)*q*r of the stored
    code (6 Toffoli: 3 Karatsuba product cells for the q*r trit product +
    a 3-Toffoli bin3_add).
    """
    coef = (-(LAM + GAM)) % 3  # ty: ignore[unsupported-operator]  # LAM asserted in (1, 2)
    if coef == 0:
        return []
    p0, p1, q0, q1, r0, r1 = g[0], g[1], g[2], g[3], g[4], g[5]
    x1 = candB[0]
    x2 = candB[1]
    if coef == 1:
        pr1 = (((q0,), (r0,), 0), ((q1,), (r1,), 0))
    else:
        pr1 = (((q0,), (r1,), 0), ((q1,), (r0,), 0))
    prt = (((q0, q1), (r0, r1), 0),)
    L = [("bx", x1, pr1, (), 0), ("bx", x2, prt, (), 0), ("cx", x1, x2)]
    L += bin3_add(p0, p1, x1, x2)
    L += [("cx", x1, x2), ("mx", x2, prt, (), 0), ("mx", x1, pr1, (), 0)]
    return L


# ---- the trace extraction gadget (one per trace-add, mirrored after) ------
def build_ext_ops():
    """Map the chain state to the 6-wire trace-class basis
    (cen2, f6, t1, f2, t2, fm1): decode M into row rails, build J, then
    synthesize the e-trit / nu / sg indicators from the F2 scans and
    combine.  Ends with a Clifford-only teardown of the linear scaffolding
    so those wires can serve as the full-width adder's carry pool (rebuilt
    for free by the mirror on the way back).
    """
    kind = EBEST[1]  # ty: ignore[not-subscriptable]  # EBEST set by the module scan
    pa, fa, oa = bx_args(EBEST[2])  # ty: ignore[not-subscriptable]  # EBEST set by the module scan
    pb, fb, ob = bx_args(EBEST[3])  # ty: ignore[not-subscriptable]  # EBEST set by the module scan
    pn, fn, on = bx_args(DNU)
    ps, fs, osg = bx_args(DSG)
    USED = set()
    for pp, ff in ((pa, fa), (pb, fb), (pn, fn), (ps, fs)):
        for Ax, _Bx, _cf in pp:
            USED.update(Ax)
        USED.update(ff)
    USED &= set(ROWQ)
    E: list = [
        ("andc", a_st, accS, accT),
        ("al", r10),
        ("cx", accS, r10),
        ("cx", a_st, r10),
        ("al", r01),
        ("cx", accT, r01),
        ("cx", a_st, r01),
        ("al", r00),
        ("x", r00),
        ("cx", accS, r00),
        ("cx", accT, r00),
        ("cx", a_st, r00),
    ]
    built0 = []
    for m, rq in ((0, r00), (1, r01), (2, r10), (3, a_st)):
        n0 = (rMU0[m] in USED) or (m == 0)
        n1 = (rMU1[m] in USED) or n0
        if n1:
            E.append(("andc", rMU1[m], rq, accU))
        if n0:
            E += [("al", rMU0[m]), ("cx", rq, rMU0[m]), ("cx", rMU1[m], rMU0[m])]
            built0.append((m, rq))
    E += J_build_ops()
    w1, w2 = candB[0], candB[1]
    if kind == "12":
        pre = [("bx", w1, pa, fa, oa), ("bx", w2, pb, fb, ob)]
        post = [("mx", w2, pb, fb, ob), ("mx", w1, pa, fa, oa)]
    elif kind == "1n":
        pre = [("bx", w1, pa, fa, oa), ("bx", w2, pb, fb, ob), ("cx", w1, w2)]
        post = [("cx", w1, w2), ("mx", w2, pb, fb, ob), ("mx", w1, pa, fa, oa)]
    else:
        pre = [("bx", w2, pa, fa, oa), ("bx", w1, pb, fb, ob), ("cx", w2, w1)]
        post = [("cx", w2, w1), ("mx", w1, pb, fb, ob), ("mx", w2, pa, fa, oa)]
    i1 = (EIDX - 1) % 3
    i2 = (EIDX - 2) % 3
    ecp = (((w1,), (accP[i1], accP[EIDX]), 0), ((w2,), (accP[i2], accP[EIDX]), 0))
    ecf = (accP[EIDX],)
    E += pre
    E += [("bx", ecq, ecp, ecf, 0)]
    E += [("bx", nuq, pn, fn, on), ("bx", sgq, ps, fs, osg)]
    E += [
        ("andc", t1q, nuq, ecq),
        ("al", t2q),
        ("cx", nuq, t2q),
        ("cx", t1q, t2q),
        ("x", sgq),
        ("andc", f2q, t1q, sgq),
        ("andc", fm1q, t2q, sgq),
        ("x", sgq),
        ("andc", cen2q, J[0], rMU0[0]),
        ("andc", f6q, cen2q, ecq),
    ]
    E += [("mx", ecq, ecp, ecf, 0)]
    E += post
    # Clifford-only teardown: reclaim the linear scaffolding for the
    # full-width adder carry pool (rebuilt for free on the way back).
    # Deliberately partial -- the four AND cells stay live -- and
    # deliberately ordered cols (J[1], J[2]) before rows (J[3], J[6]),
    # the reverse of the J_unbuild_ops order.  The two
    # blocks commute (each writes only cells the other does not read),
    # but the free order feeds the Builder LIFO free-list, so
    # harmonizing it would permute later qubit ids and move the stream.
    for m, rq in built0:
        E += [("cx", rq, rMU0[m]), ("cx", rMU1[m], rMU0[m]), ("fr", rMU0[m])]
    E += [
        ("cx", accS, r10),
        ("cx", a_st, r10),
        ("fr", r10),
        ("cx", accT, r01),
        ("cx", a_st, r01),
        ("fr", r01),
        ("x", r00),
        ("cx", accS, r00),
        ("cx", accT, r00),
        ("cx", a_st, r00),
        ("fr", r00),
    ]
    E += [("cx", J[i], J[0]) for i in range(1, 9)] + [("x", J[0]), ("fr", J[0])]
    for bb in (1, 2):
        E += [
            ("cx", Rm[bb], J[bb]),
            ("cx", J[3 + bb], J[bb]),
            ("cx", J[6 + bb], J[bb]),
            ("fr", J[bb]),
        ]
    for a in (1, 2):
        E += [
            ("cx", Qm[a], J[3 * a]),
            ("cx", J[3 * a + 1], J[3 * a]),
            ("cx", J[3 * a + 2], J[3 * a]),
            ("fr", J[3 * a]),
        ]
    return E


EXT_OPS = build_ext_ops()

# ---- full-width XOR-basis trace-adds over the 6 scaffold wires -------------
BASISQS = [cen2q, f6q, t1q, f2q, t2q, fm1q]
OUTV = (6, -3, 2, -2, 1, -1)


def kadd_consts(mult):
    """Per-bit XOR-parity matrices for adding mult*f through the basis
    wires: pats[j] is the constant added when basis wire j is set.
    """
    Wm = 1 << WBITS
    cm = {v: (v * mult) % Wm for v in OUTV}
    pats = [cm[-3], cm[6] ^ cm[-3], cm[-2], cm[2] ^ cm[-2], cm[1], cm[-1] ^ cm[1]]
    mats = [[BASISQS[jj] for jj in range(6) if (pats[jj] >> i) & 1] for i in range(51)]
    lo = min((cv & -cv).bit_length() - 1 for cv in cm.values() if cv)
    return mats, lo


MATS_P, LO_P = kadd_consts(A_P)
MATS_R, LO_R = kadd_consts(A_R)

# The adder carry pool: 8 static bank seeds, the 12 reclaimed
# extraction wires, and the 30 dedicated CARRY wires -- 50 in all, one
# per interior bit of a full-width add.  The pool is pinned literally
# here in derivation order (the 12 reclaimed wires hold |0> at the k-add
# point, scanned in BANK order).  Any order change permutes carry
# placement and moves the stream.
# local: gadgets.wires.WirePool fronts live Builder calls; this flat-
# program layer places its pooled carries by static wire id against the
# same proven-|0> census, for the same 0-fresh-alloc effect.
KPOOL = [candB[0], candB[1], candB[2], candB[3], ctlB[0], ctlB[1], xorB0, scr0]
KPOOL += [J[0], J[1], J[2], J[3], J[6], r10, r01, r00, rMU0[0], rMU0[1], rMU0[2], rMU0[3]]
KPOOL += CARRY


def kadd_ops(mats, xconst=0, lo=0, hi=51, *, pool):
    """In-place k += addend over bits [lo, hi): per bit the addend parity
    is folded onto addd (free CX), one AND makes the carry on a pooled
    wire, and the return sweep releases every carry by measurement
    (1 Toffoli per interior bit; Gidney ripple).
    local: gadgets.add/gidney_add is the same Gidney (2018) 1-T/interior-bit
    ripple but requires a register operand wire per addend position
    (shared= restores one repeated wire; it cannot refold a per-bit XOR
    parity of the 6 basis wires onto the single addd wire), which is
    the inexpressible part; its pool= parameter already covers pooled
    carries.
    Carries come only from the fixed pool, and pool sufficiency is
    asserted fail-closed: no dynamic allocation exists, so a wider add
    cannot silently push past the 1,859-wire peak.
    """
    L = []

    def put(i):
        for fq in mats[i]:
            L.append(("cx", fq, addd))
        if (xconst >> i) & 1:
            L.append(("x", addd))

    carries = {}
    c = None
    for i in range(lo, hi):
        put(i)
        if i < hi - 1:
            idx = i - lo
            assert idx < len(pool), "carry pool exhausted"
            nc = pool[idx]
            if nc in BREATHING:
                L.append(("al", nc))
            if c is None:
                L.append(("ccx", kreg[i], addd, nc))
                L.append(("cx", addd, kreg[i]))
            else:
                L += [
                    ("cx", c, kreg[i]),
                    ("cx", c, addd),
                    ("ccx", kreg[i], addd, nc),
                    ("cx", c, nc),
                    ("cx", c, addd),
                    ("cx", addd, kreg[i]),
                ]
            carries[i] = nc
            c = nc
        else:
            if c is not None:
                L.append(("cx", c, kreg[i]))
            L.append(("cx", addd, kreg[i]))
        put(i)
    for i in range(hi - 2, lo - 1, -1):
        cc = carries[i]
        cprev = carries.get(i - 1)
        put(i)
        L.append(("cx", addd, kreg[i]))
        if cprev is not None:
            L += [("cx", cprev, addd), ("cx", cprev, cc)]
        L.append(("andr", cc, kreg[i], addd))
        if cprev is not None:
            L.append(("cx", cprev, addd))
        L.append(("cx", addd, kreg[i]))
        put(i)
    return L


# ---- in-place state inversion st -> st^-1 (21 Toffoli) ---------------------
# Free rail negations of P', Q, R; the (Q,R) map factorizes as
# W^M B^U (-I) with B the X-conjugation (SWAPS_X on the joint one-hot J
# grid, 6 controlled cell swaps) and W = [[0,2],[1,0]] (a t-conditioned
# neg_R + derived-rail Q<->R swap); M2 = -M (U=0) / M+2 (U=1) is one
# Toffoli.  P-corrections ride J as one U-conditioned table stage whose
# cells fold together the X-conjugation pullback T02(B(.)), the
# (U)-conditioned squares, and the prim-shear conjugation (the stored
# P' = P + GAM*Q*R at GAM = 1 cancels what would otherwise be two more
# stages).
_T02 = {(q, r): (q * q + 2 * r * r + 2 * q * r) % 3 for q in range(3) for r in range(3)}
_BM = lambda q, r: ((2 * q + r) % 3, (q + r) % 3)
_IMER = {
    (q, r): (_T02[_BM(q, r)] + q * q + 2 * r * r + q * r - _BM(q, r)[0] * _BM(q, r)[1]) % 3
    for q in range(3)
    for r in range(3)
}
_IG1 = [c for c, v in _IMER.items() if v == 1]
_IG2 = [c for c, v in _IMER.items() if v == 2]


def invert_macros():
    def _sw(a, bq):
        return [("cx", a, bq), ("cx", bq, a), ("cx", a, bq)]

    L = []
    L += _sw(accP[1], accP[2]) + _sw(Qm[1], Qm[2]) + _sw(Rm[1], Rm[2])
    L += J_build_ops()
    L += J_fold_ops()
    # U-conditioned P-correction table stage on the J cells (one-hot: the
    # cell group's OR is its XOR, collected with free CX)
    for c in _IG1:
        L.append(("cx", J[3 * c[0] + c[1]], scr0))
    for c in _IG2:
        L.append(("cx", J[3 * c[0] + c[1]], xorB0))
    L += [("andc", ctlB[0], accU, scr0), ("andc", ctlB[1], accU, xorB0)]
    L += core3_ops(accP, ctlB[0], ctlB[1], addd)
    L += [("andr", ctlB[1], accU, xorB0), ("andr", ctlB[0], accU, scr0)]
    for c in _IG2:
        L.append(("cx", J[3 * c[0] + c[1]], xorB0))
    for c in _IG1:
        L.append(("cx", J[3 * c[0] + c[1]], scr0))
    # B swaps on J conditioned on U
    for i, jj in SWAPS_X:
        L.append(("cswap", accU, J[i], J[jj]))
    # re-derive Qm/Rm from the permuted grid; tear J down
    L += J_fold_ops()
    L += J_unbuild_ops()
    # W1 = neg_R then derived-rail Q<->R swap, both conditioned on t
    L += [("cswap", accT, Rm[1], Rm[2])]
    L += qr_swap_ops(accT)
    # W2 = -I conditioned on s
    L += [("cswap", accS, Qm[1], Qm[2]), ("cswap", accS, Rm[1], Rm[2])]
    # M-fix: accS ^= (accT & ~U) ^ U
    L += [("x", accU), ("ccx", accU, accT, accS), ("x", accU), ("cx", accU, accS)]
    return L


INVERT_M = invert_macros()


# ---------------------------------------------------------------------------
# K. Lower the schedule to flat op chunks.
#
# Every schedule token lowers through the same macro segments; the row
# programs are frame-local, so their link operands are frame wires and
# one lowering serves every translate of a row.  A row lowers to its 36
# (inter-fire morph + trace-add) chunks: each is balanced (frees what
# it allocates) and under the op stream's u16 LOOP-item cap, so a chunk is
# exactly one repeat() body.
# ---------------------------------------------------------------------------
# The per-link encode unit (contract -> stored code) on the wires of
# link 0; the encode/decode loops make its link operands affine.
XF_UNIT = xform_ops(links[0]) + qr_corr_ops(links[0])
XF_FWD = expand(XF_UNIT)
XF_REV = expand(XF_UNIT, rev=True)


def morph_ops(toks):
    """Expand a run of walk tokens (no fires) to flat low-level ops."""
    L = []
    for op in toks:
        k = op[0]
        if k == "f":
            L += expand(mul_fwd_ops(links[op[1]]))
        elif k == "i":
            L += expand(mul_inv_ops(links[op[1]]))
        elif k == "li":
            L += expand(invload_ops(links[op[1]]))
        elif k == "ul":
            L += expand(invload_ops(links[op[1]]), rev=True)
        elif k == "inv":
            L += expand(INVERT_M)
        else:
            raise AssertionError(op)
    return L


def row_chunks(form):
    """Lower one row program to its 36 (morph + trace-add) chunks."""
    chunks, cur = [], []
    for op in form:
        if op[0] not in ("XP", "XR"):
            cur.append(op)
            continue
        mats, lo = (MATS_P, LO_P) if op[0] == "XP" else (MATS_R, LO_R)
        ops_ = morph_ops(cur)
        ops_ += expand(EXT_OPS)
        ops_ += expand(kadd_ops(mats, lo=lo, hi=WBITS, pool=KPOOL))
        ops_ += expand(EXT_OPS, rev=True)
        chunks.append(ops_)
        cur = []
    assert not cur
    return chunks


OPEN_CHUNKS = row_chunks(OPEN_ROW)
Y_CHUNKS = row_chunks(Y_ROW)
Z_CHUNKS = row_chunks(Z_ROW)
TAIL_OPS = morph_ops(TAIL)
assert max(len(c) for c in OPEN_CHUNKS + Y_CHUNKS + Z_CHUNKS) < 0xFFFF

# ---------------------------------------------------------------------------
# L. Emit the loop-form circuit through a native Builder.  play()
# replays flat chunk ops 1:1 onto Builder calls -- the working set is
# dynamic (alloc/free ride the Builder's LIFO free-list, so the
# emitted stream's peak equals the skyline peak), hmr returns Builder
# Outcome handles and czc conditions a cz on them, and the ownership
# sets ride along: g and k are block-claimed once and owned end to end,
# every allocation claims the wire's tagged set, and the single free
# path releases before freeing.  Marks and claim/release are pure
# bookkeeping (the emitted bytes are identical with or without them);
# tools/occupancy_profile.py and tools/liveness_profile.py consume the
# wire_log they leave, the former validating the ownership discipline
# fail-closed.  play()'s wire hook lets the encode/decode loops make
# the link operands affine in the loop index.
#
# The conveyor: link data lives on the 1,728 link wires as four y-slabs
# of four z-slabs.  y_shift()/z_shift() rotate the array one slab down
# (48 four-cycles x 9 wires, three Swaps each), so after a shift the
# next row's links sit on the frame-row wires and one recorded row body
# serves every translate.  Four y-shifts per plane and four z-shifts in
# all: the conveyor nets to the identity before decode.  Chunks sit in
# repeat(1) bodies inside the row loops purely to stay under the
# op stream's u16 LOOP-item cap; the balanced-allocation rule then hands
# every iteration the same working-wire ids.
# ---------------------------------------------------------------------------
b = Builder()
_bio = declare_registers(b, [(f"g{_l:03d}", 9) for _l in range(192)] + [("k", 51)])
_bregs = [_bio[f"g{_l:03d}"] for _l in range(192)]
_bk = _bio["k"]

# The persistent I/O sets, block-claimed with their full id lists and
# never released: they end the build as the two declared registers.
b.claim("g", [_w for _lk in links for _w in _lk])
b.claim("k", list(kreg))

LNK = WireList([_w for _lk in links for _w in _lk])  # link wires 0..1727

_wq = {w: w for w in range(NIO)}  # program wire -> live Builder qubit


def _res(w, wire):
    if wire is not None:
        _r = wire(w)
        if _r is not None:
            return _r
    return _wq[w]


def play(ops_, wire=None):
    """Replay flat low-level ops onto the Builder.  ``wire`` maps a
    program wire id to an operand override (or None); everything else
    routes through the live program-wire map.
    """
    outs = {}
    for _op in ops_:
        _k = _op[0]
        if _k == "cx":
            b.cx(_res(_op[1], wire), _res(_op[2], wire))
        elif _k == "ccx":
            b.ccx(_res(_op[1], wire), _res(_op[2], wire), _res(_op[3], wire))
        elif _k == "x":
            b.x(_res(_op[1], wire))
        elif _k == "hmr":
            outs[_op[2]] = b.hmr(_res(_op[1], wire))
        elif _k == "czc":
            b.cz(_res(_op[1], wire), _res(_op[2], wire), cond=outs[_op[3]])
        elif _k == "alloc":
            _q = b.qubit()
            _wq[_op[1]] = _q
            b.claim(WIRE_TAGS[_op[1]], [_q])  # every working wire is tagged
        elif _k == "free":
            _q = _wq.pop(_op[1])
            b.release(WIRE_TAGS[_op[1]], [_q])  # release-before-free, one path
            b.recycle(_q)
        else:
            raise AssertionError(_op)


def y_shift():
    """Rotate the link array one y-slab down (next row -> frame row):
    1,296 Swaps.
    """
    with b.repeat(4) as _zb, b.repeat(108) as _w:
        _e = 432 * _zb + _w
        b.swap(LNK[_e], LNK[_e + 108])
        b.swap(LNK[_e + 108], LNK[_e + 216])
        b.swap(LNK[_e + 216], LNK[_e + 324])


def z_shift():
    """Rotate the link array one z-slab down (next plane -> frame
    plane): 1,296 Swaps.
    """
    with b.repeat(432) as _w:
        b.swap(LNK[_w], LNK[_w + 432])
        b.swap(LNK[_w + 432], LNK[_w + 864])
        b.swap(LNK[_w + 864], LNK[_w + 1296])


def emit_row(chunks):
    for _ops in chunks:
        with b.repeat(1):
            play(_ops)


# ---- 0/1. utility wires; encode: one affine repeat over the 192 links ------
with b.phase("encode"):
    play([("alloc", _w) for _w in UTIL])
    with b.repeat(192) as _i:
        play(XF_FWD, wire=lambda w: LNK[9 * _i + w] if w < 9 else None)

# ---- 2. loops: the conveyor walk + 576 trace-adds ---------------------------
with b.phase("loops/id-set"):
    play([("alloc", _w) for _w in CHAINW] + [("x", accP[0]), ("x", Qm[0]), ("x", Rm[0])])

with b.phase("loops/row-open"):
    for _ops in OPEN_CHUNKS:
        play(_ops)
with b.repeat(3):  # plane 0, rows y = 1..3
    with b.phase("loops/shift"):
        y_shift()
    with b.phase("loops/row-y"):
        emit_row(Y_CHUNKS)
with b.phase("loops/shift"):
    y_shift()
with b.repeat(3):  # planes z = 1..3
    with b.phase("loops/shift"):
        z_shift()
    with b.phase("loops/row-z"):
        emit_row(Z_CHUNKS)
    with b.repeat(3):
        with b.phase("loops/shift"):
            y_shift()
        with b.phase("loops/row-y"):
            emit_row(Y_CHUNKS)
    with b.phase("loops/shift"):
        y_shift()
with b.phase("loops/shift"):
    z_shift()

with b.phase("loops/close"):
    play(TAIL_OPS)
with b.phase("loops/id-clear"):
    play([("x", accP[0]), ("x", Qm[0]), ("x", Rm[0])] + [("free", _w) for _w in CHAINW])

# ---- 3/4. decode: exact mirror of encode; release the utilities ------------
with b.phase("decode"):
    with b.repeat(192) as _i:
        play(XF_REV, wire=lambda w: LNK[1719 - 9 * _i + w] if w < 9 else None)
    play([("free", _w) for _w in UTIL])

assert_metrics(b, __file__)

# The measurement count: 384 link recodes x 7 + 3,136 multiplies x 10
# + 480 inversions x 6 + 192 P trace-adds x 72 + 384 R trace-adds x 71.
N_HMR = sum(1 for _t in b.iter_expanded() if _t[0] == OpKind.HMR)


def build():
    """Raw-build accessor for the occupancy/liveness profilers: the
    module-level build above is the only build (no post pass touches
    the allocator), so the tools consume its wire_log directly.
    """
    return b, _bregs + [_bk]


# Emission.  The evaluator checks the circuit: its sampled run covers
# every failure class, so the epilogue carries no Simulator replay
# (uncompute.testing).
emit_and_report(b, extra=f"bits={N_HMR}")
