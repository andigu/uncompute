#!/usr/bin/env python3
"""chemistry-fe2s2 ground truth: the pinned [2Fe-2S] (30e,20o) alias-select table.

The task fuses the alias-sampling decision with the SELECT application
of the chosen Jordan-Wigner Pauli string to a 40-qubit system register x, then
erases the decision:

    |i>_16 |sigma>_16 |x>_40
        -> (-1)^phase |i> |sigma> |x XOR flip(sel)>

    f     = [sigma < keep_i]                (strict comparator)
    sel   = f ? i : alt_i
    phase = theta_sel XOR ny2(sel) XOR parity(x AND zy(sel))

per row: flip = X-or-Y support of the Pauli string, zy = Z-or-Y support,
ny2 = (Y-count mod 4)/2 (Y-count is even on every row: real coefficients),
theta_sel = coefficient sign of the selected row.

The values f and sel are internal to the circuit and must be returned to zero;
there is no output register for them.

This script re-derives the table from first principles (the pinned FCIDUMP
-> openfermion JW -> magnitude truncation -> deterministic integer alias
construction) and asserts byte-equality with the pinned in-repo copies
(`data/table.json.xz`, `data/masks.json.xz`). Run it after any toolchain
change to confirm the pins still reproduce; run with --emit to (re)write
the data files.  The FCIDUMP step needs pyscf and openfermion, which the
project does not install:

    uv run --with pyscf --with openfermion python reference.py

--skip-derive runs every other check without them.

Pinned generation contract (every step deterministic):

  1. Hamiltonian: the (30e,20o) active-space FCIDUMP of the [2Fe-2S]
     complex (Li-Chan iron-sulfur model space;
     github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters,
     path Fe2S2_and_Fe4S4/Fe2S2/fe2s2), pinned as data/fcidump.xz with
     FCIDUMP_SHA256 over the decompressed bytes. pyscf reads the dump;
     the spatial integrals lift to spin-orbitals (openfermion
     spinorb_from_spatial, interleaved up/down), fermion -> qubit via
     Jordan-Wigner. 40 qubits; the identity coefficient is dropped.
  2. Truncation: keep exactly the terms with |w_i| >= 1e-4 Ha, the sparse
     qubitization threshold practice; L = 41,052 kept terms
     (asserted, and the margin around the threshold is asserted wide
     enough that a re-derivation cannot flip membership).
  3. S_z pairing: the spin-restricted integrals make the term set exactly
     symmetric under the spin flip alpha<->beta.  On JW strings the flip
     is the qubit permutation 2p<->2p+1 followed by a Z at the partner
     site of every X/Y endpoint, multiplied out in the Pauli algebra
     (sz_image below).  Every kept term either pairs with a distinct
     partner of exactly equal |w| (20,400 pairs) or is self-symmetric
     (252 rows); asserted complete.
  3b. Letter pairing: the mask-level involution C (c_image_zy below;
     flips unchanged, zy gains the same-spin-sector flip parity minus
     the self term — the X<->Y letter exchange of the SU(2) degeneracy
     structure, a CZ-network conjugation at the circuit level) matches
     most S_z pairs into quartet tokens: C maps pair A's representative
     mask onto a member of pair B.  C commutes with the S_z map in this
     gauge, so the matching is a perfect pairing on the matched set;
     unmatched pairs (C-image self, absent, or truncated away) form the
     residual block.  Matching is exact on (flip, zy) — no coefficient
     tolerance anywhere.
  4. Row order (frame blocks): a chain of exact
     mask-level regroupings — quartet tokens at four-address groups
     (address bit 0 the spin-flip bit, bit 1 the letter bit;
     quartet_order below), reflection-merged octuples under D
     (group_order), thiolate-swap refinement under sigma
     (final_order), a frame-set partition over the
     eight-involution commuting family (frameset_order) — arrives at
     the final partition over the SEVEN-frame commuting family
     {F, C, D, D2, s_thio, s_d45, x_muS} (frame_order below), D2 the
     composed hybrid reflection: every row takes the deepest of 13
     fixed frame sets whose full mask orbit survives in the
     table, a rank-k block spans 2^k aligned addresses with branch
     bit p keyed to the p-th frame of its set, blocks are laid out
     size-descending, then by frame set, then by descending |w|,
     with identical-coefficient-vector duplicate blocks contiguous
     at aligned slots inside each region; the 14 maximal
     same-frame-set runs are the contract regions.
  5. lambda = sum(|w_i|) over the KEPT terms (built-in sum, IEEE-754
     doubles, table order).
  6. Integer weights: targets t_i = |w_i|/lambda * L * 2^MU (doubles), m_i
     by largest-remainder rounding to sum EXACTLY L * 2^MU (ties: lower
     index first, table order).
  7. Alias table: the credit-aligned two-phase melt scheduler
     (run_atomic_alias below -> alias_sched.melt_align) over the
     run-atomic small stream — phase A melts heavy larges onto shared
     keep words via exact whole-entry subset fills (zero run
     boundaries crossed), with window-closed larges' donations banked
     and routed so row-adjacent melted larges share one keep-word
     class; phase B closes the near-threshold larges through credit
     chains; one conservation-closure large takes keep=0 / alt=i.
     The alias distribution reproduces m_i EXACTLY (asserted).
  8. Word layout (MSB..LSB): theta(1) | alt_theta(1) | alt(16) | keep(16)
     = 34 bits, theta_i = sign bit of w_i, alt_theta_i = theta[alt_i].
     Mask layout: flip(40) | zy(40)<<40 | ny2(1)<<80, 81 bits.
  9. Digest pins: DATA_SHA256 over the words as little-endian u64;
     MASKS_SHA256 over the masks as 16-byte little-endian blocks.

REPRODUCIBILITY CONTRACT: the PINNED
COEFFICIENTS (hex doubles in data/table.json.xz) are the ground truth;
steps 3-8 run on them bit-exactly and this script asserts they reproduce
the pinned words + digests. The FCIDUMP -> JW step is a physics check, not
a bit check: re-derived coefficients must match the pinned ones
term-by-term to |delta| <= 1e-9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FCIDUMP_XZ = DATA_DIR / "fcidump.xz"
TABLE_XZ = DATA_DIR / "table.json.xz"
MASKS_XZ = DATA_DIR / "masks.json.xz"

MU = 16
T = 1 << MU
IDX_BITS = 16
B_WORD = 1 + 1 + IDX_BITS + MU  # 34
N_QUBITS = 40
THRESHOLD = 1e-4  # Ha; the sparse-method truncation cutoff
L_EXPECTED = 41052
N_PAIRS_EXPECTED = 20400  # S_z pairs; 252 self-symmetric rows
N_TOKENS_EXPECTED = 9942  # quartet tokens; 516 residual pairs
N_GROUPS_EXPECTED = 4152  # D-merged token pairs (octuples)
# refinement censuses (sigma = the thiolate swap (0 1)(18 19)):
N_SG_EXPECTED = 612  # sigma-matched D-group pairs (16-blocks)
N_ST_EXPECTED = 66  # sigma-matched D-unmatched token pairs
N_DR_EXPECTED = 245  # D-matched residual-pair quads
N_DT_EXPECTED = 123  # D-matched self-symmetric doubles
# frame-partition censuses (D2 = the composed hybrid reflection):
N_FRAME_SPANS_EXPECTED = 3221
N_FRAME_REGIONS_EXPECTED = 14
FRAME_SIZES_EXPECTED = {1: 60, 2: 12, 4: 870, 8: 1026, 16: 676, 32: 577}
N_ELEC = 30
HF_STATE = (1 << N_ELEC) - 1  # 30 electrons in the first 30 spin-orbitals

SEED_DOMAIN = b"chemistry-fe2s2"  # the grader's seed domain (x probes)

FCIDUMP_SHA256 = "95d8786af06eeea2107e19ffd98c66a6ca97fc8c9864175a4f6d64512b6f2df9"
# Digest pins over the constructed tables (recorded from the emit run;
# check_pin_internal_consistency asserts them).
DATA_SHA256 = "1fd83441ba295e3dace47ee25224bfb84a539d1c63c14e0a1ce02c2fb8c03e45"
MASKS_SHA256 = "b970bec6c3689c53de7738c7f6a7454cdf07e641e27e4b048fc1b5eb52a02751"


def term_str(t) -> str:
    return " ".join(f"{p}{q}" for q, p in t) if t else "I"


def read_fcidump_bytes() -> bytes:
    raw = lzma.decompress(FCIDUMP_XZ.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == FCIDUMP_SHA256, "pinned FCIDUMP drifted"
    return raw


def derive_terms():
    """FCIDUMP -> JW re-derivation (heavy deps; import inside).

    Returns ALL non-identity terms as [(term_tuple, coeff)] — truncation
    is applied by the caller so the boundary margin can be asserted.
    """
    import tempfile

    import numpy as np
    from openfermion import InteractionOperator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    with tempfile.NamedTemporaryFile(suffix=".fcidump") as fh:
        fh.write(read_fcidump_bytes())
        fh.flush()
        d = fcidump.read(fh.name)
    n = d["NORB"]
    assert 2 * n == N_QUBITS and d["NELEC"] == N_ELEC
    h1 = d["H1"]
    eri = ao2mo.restore(1, d["H2"], n)  # full (pq|rs), chemist
    two = np.asarray(eri.transpose(0, 2, 3, 1), order="C")  # physicist
    so1, so2 = spinorb_from_spatial(h1, two)  # interleaved spins
    op = InteractionOperator(float(d["ECORE"]), so1, 0.5 * so2)
    qop = jordan_wigner(op)
    qop.compress(1e-12)
    terms = []
    for term, coeff in qop.terms.items():
        if term == ():
            continue  # identity: not loaded
        assert abs(coeff.imag) < 1e-9, (term, coeff)
        terms.append((term, float(coeff.real)))
    return terms


def truncate(named):
    """Step 2 on [(pauli_str, coeff)]: keep |c| >= THRESHOLD; assert the
    count and a re-derivation-proof margin around the cutoff.
    """
    kept = [(p, c) for p, c in named if abs(c) >= THRESHOLD]
    dropped_max = max((abs(c) for _, c in named if abs(c) < THRESHOLD), default=0.0)
    kept_min = min(abs(c) for _, c in kept)
    assert len(kept) == L_EXPECTED, f"L drifted: {len(kept)}"
    # the physics check tolerates 1e-9 coefficient drift; the threshold
    # margin must dominate it so membership cannot flip
    assert kept_min - THRESHOLD > 1e-9 or kept_min == THRESHOLD
    assert THRESHOLD - dropped_max > 1e-9
    return kept


# ---------------------------------------------------------------------------
# The S_z involution on JW strings (contract step 3).
# ---------------------------------------------------------------------------
_LETTER = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_NAME = "IXYZ"
_PROD = {}
for _a in range(4):
    _PROD[(0, _a)] = (0, _a)
    _PROD[(_a, 0)] = (0, _a)
    _PROD[(_a, _a)] = (0, 0)
_PROD[(1, 2)] = (1, 3)
_PROD[(2, 1)] = (3, 3)
_PROD[(2, 3)] = (1, 1)
_PROD[(3, 2)] = (3, 1)
_PROD[(3, 1)] = (1, 2)
_PROD[(1, 3)] = (3, 2)


def sz_image(pauli: str) -> str:
    """Return the S_z-flip image of a JW term string: permute 2p<->2p+1, then
    left-multiply a Z at the partner of every X/Y site (Pauli algebra;
    the accumulated phase is +-1 and does not affect the string).
    """
    ops = {}
    for tok in pauli.split():
        ops[int(tok[1:]) ^ 1] = _LETTER[tok[0]]
    for q in [qq for qq, a in list(ops.items()) if a in (1, 2)]:
        part = q ^ 1
        _k, c = _PROD[(3, ops.get(part, 0))]
        if c == 0:
            ops.pop(part, None)
        else:
            ops[part] = c
    return " ".join(f"{_NAME[a]}{q}" for q, a in sorted(ops.items()))


# ---------------------------------------------------------------------------
# The letter involution C on (flip, zy) masks (contract step 3b).
# ---------------------------------------------------------------------------
EVN_MASK = 0x5555555555  # even (spin-up) mode positions
ODD_MASK = EVN_MASK << 1  # odd (spin-down) mode positions


def c_image_zy(flip: int, zy: int) -> int:
    """Return the letter frame's zy image: every site gains the parity of the
    same-spin-sector flips EXCLUDING itself (the K_evens + K_odds CZ
    graph — the pi-symmetric gauge, which makes C commute with the S_z
    map).  Flips are unchanged; ny2 follows from (flip, zy).
    """
    we = bin(flip & EVN_MASK).count("1") & 1
    wo = bin(flip & ODD_MASK).count("1") & 1
    br = (EVN_MASK if we else 0) | (ODD_MASK if wo else 0)
    return zy ^ br ^ flip


# ---------------------------------------------------------------------------
# The reflection involution D on (flip, zy) masks (contract step 3c):
# the C2 Fe1<->Fe2 orbital reflection p -> 19-p as a fermionic mode
# permutation, composed from adjacent FSWAP conjugations.  NOT a
# Hamiltonian symmetry (coefficients differ by a few percent) — the row
# order only needs the MASK SET to be closed on the matched part, and
# the SELECT frames are weight-blind.
# ---------------------------------------------------------------------------
def _perm_ts(orb):
    """Adjacent-transposition network of an orbital permutation lifted
    to interleaved spin-orbital modes (bubble sort of the mode image).
    """
    modes = [0] * N_QUBITS
    for p in range(20):
        modes[2 * p] = 2 * orb[p]
        modes[2 * p + 1] = 2 * orb[p] + 1
    ts = []
    for _ in range(N_QUBITS):
        for j in range(N_QUBITS - 1):
            if modes[j] > modes[j + 1]:
                modes[j], modes[j + 1] = modes[j + 1], modes[j]
                ts.append(j)
    return ts


def _fswap_image(ts, flip, zy):
    """Mask action of an adjacent-FSWAP composition."""
    for m in ts:
        fm = (flip >> m) & 1
        fm1 = (flip >> (m + 1)) & 1
        zm = (zy >> m) & 1
        zm1 = (zy >> (m + 1)) & 1
        flip ^= ((fm ^ fm1) << m) | ((fm ^ fm1) << (m + 1))
        nzm = zm1 ^ fm
        nzm1 = zm ^ fm1
        zy ^= ((zm ^ nzm) << m) | ((zm1 ^ nzm1) << (m + 1))
    return flip, zy


_REFL_TS = _perm_ts([19 - p for p in range(20)])


def d_image_fz(flip, zy):
    """Apply the reflection frame's mask action: the composition of adjacent
    fermionic-SWAP conjugations realizing the mode permutation.
    """
    return _fswap_image(_REFL_TS, flip, zy)


def quartet_order(named):
    """Contract steps 3-4 (pairing + first stage): the quartet
    machinery.  ``named``
    is [(pauli_str, coeff)] (any order); returns the reordered list with
    matched tokens at four-address groups, then residual pairs at
    even/odd couples, then the self-symmetric block, plus the counts
    (n_tokens, n_residual_pairs, n_pairs_total).
    """
    by_str = {p: i for i, (p, _c) in enumerate(named)}
    seen = set()
    pairs = []
    selfsym = []
    for i, (p, _c) in enumerate(named):
        if i in seen:
            continue
        img = sz_image(p)
        j = by_str.get(img)
        assert j is not None, f"unpaired term {p!r}: S_z closure broken"
        if j == i:
            selfsym.append(i)
            seen.add(i)
            continue
        assert j not in seen
        pi, pj = named[i][0], named[j][0]
        rep, mate = (i, j) if pi < pj else (j, i)
        assert abs(abs(named[i][1]) - abs(named[j][1])) < 1e-12
        pairs.append((rep, mate))
        seen.add(i)
        seen.add(j)

    # step 3b: mask-level C matching of pairs into quartet tokens
    fz = {}
    row_of_fz = {}
    for i, (p, _c) in enumerate(named):
        flip, zy, _ny = parse_pauli(p)
        fz[i] = (flip, zy)
        assert (flip, zy) not in row_of_fz, "duplicate (flip, zy) mask"
        row_of_fz[(flip, zy)] = i
    pair_of_row = {}
    for k, (rep, mate) in enumerate(pairs):
        pair_of_row[rep] = k
        pair_of_row[mate] = k

    def pkey(k):
        rep = pairs[k][0]
        return (-abs(named[rep][1]), named[rep][0])

    match = {}
    for k in range(len(pairs)):
        if k in match:
            continue
        flip, zy = fz[pairs[k][0]]
        img_row = row_of_fz.get((flip, c_image_zy(flip, zy)))
        if img_row is None:
            continue  # C-image absent: residual
        kb = pair_of_row.get(img_row)
        if kb is None or kb == k:
            continue  # C-image absent or self: residual
        # C is an involution commuting with S_z in this gauge, so the
        # reverse match must agree.
        fb, zb = fz[img_row]
        back = row_of_fz.get((fb, c_image_zy(fb, zb)))
        assert back == pairs[k][0], "C matching is not an involution?"
        assert kb not in match, "C matching is not a perfect pairing?"
        match[k] = kb
        match[kb] = k

    tokens = []
    for k in sorted(match, key=pkey):
        kb = match[k]
        if pkey(kb) < pkey(k):
            continue  # emit each token once, from A
        tokens.append((k, kb))
    residual = sorted((k for k in range(len(pairs)) if k not in match), key=pkey)
    selfsym.sort(key=lambda i: (-abs(named[i][1]), named[i][0]))

    order = []
    for a, _b in tokens:
        rep_a, mate_a = pairs[a]
        fa, za = fz[rep_a]
        img_row = row_of_fz[(fa, c_image_zy(fa, za))]
        b = pair_of_row[img_row]
        rep_b, mate_b = pairs[b]
        other = mate_b if img_row == rep_b else rep_b
        order.append(named[rep_a])  # 4t     : rep_A
        order.append(named[mate_a])  # 4t + 1 : S_z image of rep_A
        order.append(named[img_row])  # 4t + 2 : C image of rep_A
        order.append(named[other])  # 4t + 3 : S_z image of the above
    for k in residual:
        rep, mate = pairs[k]
        order.append(named[rep])
        order.append(named[mate])
    for i in selfsym:
        order.append(named[i])
    assert len(order) == len(named)
    return order, len(tokens), len(residual), len(pairs)


def _pp(pairs):
    """Orbital involution from a transposition product."""
    p = list(range(20))
    for a, b in pairs:
        p[a], p[b] = b, a
    return p


_SIGMA_TS = _perm_ts(_pp([(0, 1), (18, 19)]))


def s_image_fz(flip, zy):
    """Apply the thiolate-swap frame sigma = orbital (0 1)(18 19) as a
    fermionic mode permutation (adjacent-FSWAP composition).  Like the
    reflection D it is NOT a Hamiltonian symmetry — the row order only
    needs mask-set closure on the matched part.
    """
    return _fswap_image(_SIGMA_TS, flip, zy)


def _pi40(v):
    """Swap adjacent bit pairs (2p <-> 2p+1) of a 40-bit mask."""
    return (((v & EVN_MASK) << 1) | ((v >> 1) & EVN_MASK)) & ((1 << N_QUBITS) - 1)


def _f_image_fz(flip, zy):
    """Apply the S_z frame's mask action in the circuit gauge."""
    return _pi40(flip), _pi40(zy) ^ flip


def _c_image_fz(flip, zy):
    return flip, c_image_zy(flip, zy)


# ---------------------------------------------------------------------------
# The eight-involution family and its frame-set partition
# (contract step 4, intermediate stage).  Eight commuting mask
# involutions — the refinement frames F, C, D plus five
# orbital-involution frames, all legal circuit conjugations, all
# weight-blind (mask-set closure only):
#
#     s_thio = (0 1)(18 19)     s_d45 = (4 5)(14 15)
#     s_muS  = (7 8)(11 12)     s_d23 = (2 3)(16 17)
#     x_muS  = (7 11)(8 12)
#
# SIG_SETS is the fixed list of frame sets (a design constant of the pinned
# layout): tuples of frame indices in APPLICATION order (position p =
# branch bit p of the block; innermost frame first).
# ---------------------------------------------------------------------------
SIG_GEN_NAMES = ["F", "C", "D", "s_thio", "s_d45", "s_muS", "s_d23", "x_muS"]
_SIG_PERM_TS = {
    "s_d45": _perm_ts(_pp([(4, 5), (14, 15)])),
    "s_muS": _perm_ts(_pp([(7, 8), (11, 12)])),
    "s_d23": _perm_ts(_pp([(2, 3), (16, 17)])),
    "x_muS": _perm_ts(_pp([(7, 11), (8, 12)])),
}
SIG_IMAGES = [
    _f_image_fz,
    _c_image_fz,
    d_image_fz,
    s_image_fz,
    lambda f, z: _fswap_image(_SIG_PERM_TS["s_d45"], f, z),
    lambda f, z: _fswap_image(_SIG_PERM_TS["s_muS"], f, z),
    lambda f, z: _fswap_image(_SIG_PERM_TS["s_d23"], f, z),
    lambda f, z: _fswap_image(_SIG_PERM_TS["x_muS"], f, z),
]
SIG_SETS = [
    (0, 1),  # F C
    (0, 1, 2),  # F C D
    (0, 2),  # F D
    (0, 1, 2, 5),  # F C D s_muS
    (0, 1, 2, 6),  # F C D s_d23
    (0, 1, 2, 3),  # F C D s_thio
    (1, 2),  # C D
    (0, 1, 2, 4),  # F C D s_d45
    (0, 1, 7),  # F C x_muS
    (0, 1, 2, 6, 5),  # F C D s_d23 s_muS
    (0, 1, 2, 6, 4),  # F C D s_d23 s_d45
    (0, 1, 2, 7),  # F C D x_muS
    (0, 1, 3),  # F C s_thio
    (0, 1, 2, 6, 3),  # F C D s_d23 s_thio
    (0, 1, 2, 5, 3),  # F C D s_muS s_thio
    (0, 1, 6),  # F C s_d23
    (1,),  # C
]


def frameset_order(named):
    """Contract step 4 (intermediate): re-partition the refined order
    into frame blocks.  Scanning rows in refined
    order, each unassigned row takes the DEEPEST frame set from
    SIG_SETS (ties by frame-set tuple) whose full orbit lies in the
    table and unassigned; a rank-k block spans 2^k addresses with
    branch bit p keyed to the p-th frame of its set, rows in branch
    order.  Blocks are laid out size-descending (every block start
    stays aligned to its span), then by frame-set tuple, then by
    descending |w| of the base row; maximal same-frame-set runs are
    the stage's regions.  All matching is exact on (flip, zy) —
    weight-blind by design.
    """
    order_r, refc, ngrp, ntok, nres, npairs = final_order(named)
    L = len(order_r)
    fzs = [parse_pauli(p)[:2] for p, _c in order_r]
    row_of = {v: i for i, v in enumerate(fzs)}
    ptr = [[row_of.get(img(*fzs[r]), -1) for r in range(L)] for img in SIG_IMAGES]
    asig = sorted(SIG_SETS, key=lambda t: (-len(t), t))
    assigned = [False] * L
    blocks = []
    for r in range(L):
        if assigned[r]:
            continue
        hit = None
        for t in asig:
            rows = [r]
            ok = True
            for g in t:
                new = []
                for q in rows:
                    x = ptr[g][q]
                    if x < 0 or assigned[x]:
                        ok = False
                        break
                    new.append(x)
                if not ok or set(new) & set(rows):
                    ok = False
                    break
                rows = rows + new
            if ok:
                hit = (rows, t)
                break
        rows, t = hit if hit else ([r], ())
        for q in rows:
            assigned[q] = True
        blocks.append((rows, t))
    blocks.sort(key=lambda bt: (-len(bt[0]), bt[1], -abs(order_r[bt[0][0]][1])))
    out = []
    regions: list[list] = []
    for rows, t in blocks:
        if regions and regions[-1][0] == t:
            regions[-1][2] += len(rows)
        else:
            regions.append([t, len(out), len(out) + len(rows)])
        out.extend(rows)
    assert sorted(out) == list(range(L))

    # Fail-closed layout identities: every block start aligned to its
    # span, every branch row the composed frame image of its base.
    fz_out = [fzs[r] for r in out]
    for t, lo, hi in regions:
        k = len(t)
        n = 1 << k
        assert (hi - lo) % n == 0, (t, lo, hi)
        for base in range(lo, hi, n):
            assert base % n == 0, (t, base)
            for bmask in range(1, n):
                f, z = fz_out[base]
                for i in range(k):
                    if (bmask >> i) & 1:
                        f, z = SIG_IMAGES[t[i]](f, z)
                assert fz_out[base + bmask] == (f, z), (t, base, bmask)
    return (
        [order_r[r] for r in out],
        [tuple(rg) for rg in regions],
        len(blocks),
        refc,
        ngrp,
        ntok,
        nres,
        npairs,
    )


# ---------------------------------------------------------------------------
# The final frame family and partition (contract step 4).  Seven
# frames: the generators F, C, D, s_thio, s_d45, x_muS plus the
# composed hybrid reflection
#
#     D2 = D * s_thio * s_muS * s_910 * s_d23  # noqa: ERA001 - displayed equation
#        = (0 18)(1 19)(2 16)(3 17)(4 15)(5 14)(6 13)(7 11)(8 12),
#
# an element of the same commuting group that closes 13,448 rows
# (32.8% of the table) on which at least one of its factor generators
# fails: a generator-only census misses blocks closed under a product
# whose factors individually fail.  (Its 89.9% closure does NOT exceed
# every generator's -- s_thio closes 91.0%.)  FRAME_SETS
# is the fixed list of frame sets (a design constant of the pinned layout).
# ---------------------------------------------------------------------------
FRAME_GEN_NAMES = ["F", "C", "D", "D2", "s_thio", "s_d45", "x_muS"]
_D2_ORB = [18, 19, 16, 17, 15, 14, 13, 11, 12, 9, 10, 7, 8, 6, 5, 4, 2, 3, 0, 1]
_FRAME_PERM_TS = {
    "D2": _perm_ts(_D2_ORB),
    "s_d45": _SIG_PERM_TS["s_d45"],
}
FRAME_IMAGES = [
    _f_image_fz,
    _c_image_fz,
    d_image_fz,
    lambda f, z: _fswap_image(_FRAME_PERM_TS["D2"], f, z),
    s_image_fz,
    lambda f, z: _fswap_image(_FRAME_PERM_TS["s_d45"], f, z),
    lambda f, z: _fswap_image(_SIG_PERM_TS["x_muS"], f, z),
]
FRAME_SETS = [
    (0, 1, 2, 3, 4),  # F C D D2 s_thio
    (0, 1, 2, 3, 5),  # F C D D2 s_d45
    (0, 1, 2, 3, 6),  # F C D D2 x_muS
    (0, 1, 2, 3),  # F C D D2
    (0, 1, 2, 5),  # F C D s_d45
    (0, 1, 2, 6),  # F C D x_muS
    (0, 1, 2),  # F C D
    (0, 1, 3),  # F C D2
    (0, 2, 3),  # F D D2
    (0, 1),  # F C
    (0, 2),  # F D
    (1, 2),  # C D
    (1,),  # C
]


def _orbit(r, t, ptr, assigned):
    """Rows of row r's orbit under the frame generators t.

    Returns None when the orbit leaves the table, reaches an assigned
    row, or overlaps itself.
    """
    rows = [r]
    for g in t:
        new = [ptr[g][q] for q in rows]
        if any(x < 0 or assigned[x] for x in new):
            return None
        if set(new) & set(rows):
            return None
        rows = rows + new
    return rows


def _partition_blocks(L, ptr):
    """Cover rows 0..L-1 by blocks: each unassigned row takes the deepest
    surviving orbit of FRAME_SETS (scanned deepest-first), else stays single.
    """
    asig = sorted(FRAME_SETS, key=lambda t: (-len(t), t))
    assigned = [False] * L
    blocks = []
    for r in range(L):
        if assigned[r]:
            continue
        rows, t = [r], ()
        for sig in asig:
            orbit = _orbit(r, sig, ptr, assigned)
            if orbit is not None:
                rows, t = orbit, sig
                break
        for q in rows:
            assigned[q] = True
        blocks.append((rows, t))
    return blocks


def _region_runs(blocks):
    """Merge consecutive same-frame-set blocks into [t, lo, hi] regions."""
    runs: list[list] = []
    pos = 0
    for rows, t in blocks:
        if runs and runs[-1][0] == t:
            runs[-1][2] += len(rows)
        else:
            runs.append([t, pos, pos + len(rows)])
        pos += len(rows)
    return runs


def _find_slot(free, cnt, step):
    """First slot at a multiple of step that starts cnt free slots, or None."""
    nb = len(free)
    for s in range(0, nb, step):
        if s + cnt <= nb and all(free[s : s + cnt]):
            return s
    return None


def _layout_region(t, lo, hi, chex, cval):
    """Order one region's blocks: duplicate groups contiguous at aligned
    slots, then single blocks by descending |w|.

    Returns the region's row permutation and its duplicate groups.
    """
    n = 1 << len(t)
    bases = list(range(lo, hi, n))
    groups = defaultdict(list)
    for b in bases:
        groups[tuple(chex[b : b + n])].append(b)
    dups = sorted(
        (bs for bs in groups.values() if len(bs) > 1), key=lambda bs: (-len(bs), -cval[bs[0]])
    )
    singles = sorted((bs[0] for bs in groups.values() if len(bs) == 1), key=lambda b: -cval[b])
    nb = len(bases)
    free = [True] * nb
    slot_of = {}
    for bs in dups:
        cnt = len(bs)
        al = 1
        while al * 2 <= cnt:
            al *= 2
        posn = _find_slot(free, cnt, al)
        if posn is None:
            posn = _find_slot(free, cnt, 1)
        assert posn is not None
        for j, b in enumerate(bs):
            free[posn + j] = False
            slot_of[posn + j] = b
    it = iter(singles)
    for s in range(nb):
        if free[s]:
            slot_of[s] = next(it)
            free[s] = False
    assert next(it, None) is None
    perm = []
    base_new = {}
    for s in range(nb):
        b = slot_of[s]
        base_new[b] = lo + s * n
        perm.extend(range(b, b + n))
    dup_groups = [(n, [base_new[b] for b in bs]) for bs in dups]
    return perm, dup_groups


def _check_frame_layout(fz_out, regions):
    """Fail-closed layout identities: each aligned block is its base row's frame orbit."""
    for t, lo, hi in regions:
        k = len(t)
        n = 1 << k
        assert (hi - lo) % n == 0, (t, lo, hi)
        for base in range(lo, hi, n):
            assert base % n == 0, (t, base)
            for bmask in range(1, n):
                f, z = fz_out[base]
                for i in range(k):
                    if (bmask >> i) & 1:
                        f, z = FRAME_IMAGES[t[i]](f, z)
                assert fz_out[base + bmask] == (f, z), (t, base, bmask)


def frame_order(named):
    """Contract step 4 (final): re-partition the intermediate order into
    blocks over the seven-frame family (deepest of FRAME_SETS whose
    orbit survives, scanned in that order, deepest-first), lay
    blocks out size-descending then by frame set then by descending
    |w| of the base row — and within each region place duplicate
    blocks (identical coefficient vectors) contiguously at
    power-of-two-aligned slots, remaining blocks by descending |w|.
    Returns the order, the regions, the duplicate groups (for the
    run-atomic alias construction), and the span count.
    """
    (order_s, _regions_s, _nspans_s, refc, ngrp, ntok, nres, npairs) = frameset_order(named)
    L = len(order_s)
    fzs = [parse_pauli(p)[:2] for p, _c in order_s]
    row_of = {v: i for i, v in enumerate(fzs)}
    ptr = [[row_of.get(img(*fzs[r]), -1) for r in range(L)] for img in FRAME_IMAGES]
    blocks = _partition_blocks(L, ptr)
    blocks.sort(key=lambda bt: (-len(bt[0]), bt[1], -abs(order_s[bt[0][0]][1])))
    # region runs of the size-desc layout (pre-tie-break)
    runs = _region_runs(blocks)

    mid = [r for rows, _t in blocks for r in rows]
    chex = [order_s[r][1].hex() for r in mid]
    cval = [abs(order_s[r][1]) for r in mid]
    perm = []
    dup_groups = []
    for t, lo, hi in runs:
        region_perm, region_dups = _layout_region(t, lo, hi, chex, cval)
        perm.extend(region_perm)
        dup_groups.extend(region_dups)
    assert sorted(perm) == list(range(L))
    out = [mid[j] for j in perm]
    regions = [(tuple(t), lo, hi) for t, lo, hi in runs]
    _check_frame_layout([fzs[r] for r in out], regions)
    return (
        [order_s[r] for r in out],
        regions,
        dup_groups,
        len(blocks),
        refc,
        ngrp,
        ntok,
        nres,
        npairs,
    )


# Melt-align scheduler knobs (contract constants; deterministic).
MELT_PULL_BUDGET = 380  # stage-2 credited-fill pull budget
MELT_TRIES = 12  # bounded repair tries per credited fill


def run_atomic_alias(m, dup_groups):
    """Contract step 7: the credit-aligned two-phase melt alias
    scheduler.  Phase A melts heavy larges onto shared keep words via
    exact whole-entry subset fills over the run-atomic small stream
    (zero run boundaries crossed); window-closed larges' donations are
    banked and routed so row-adjacent melted larges share one
    keep-word class, circumventing the coarse pool's mod-4 donation
    lattice.  Phase B closes the near-threshold larges through credit
    chains.  The full algorithm is alias_sched.melt_align (a sibling
    module of this file); exact by construction — the closing
    assertion reproduces the integer weights bit-perfectly.
    """
    import sys

    sys.path.insert(0, str(HERE))
    import alias_sched

    keep, alt = alias_sched.melt_align(
        m,
        dup_groups,
        pull_budget=MELT_PULL_BUDGET,
        tries=MELT_TRIES,
    )
    L = len(m)
    T16 = 1 << MU
    D = [0] * L
    for j in range(L):
        assert keep[j] is not None and alt[j] is not None, j
        D[j] += keep[j]
        D[alt[j]] += T16 - keep[j]
    assert D == m, "alias table does not reproduce integer weights"
    return keep, alt


def group_order(named):
    """Contract step 4 (regrouping): regroup the quartet order under
    the reflection involution D.  Tokens whose four masks map bijectively
    onto another token's masks pair into eight-address groups
    (8s + o = token-A row o; 8s + 4 + o = its D image), sorted by
    (-|w_A|, rep_A string); D-unmatched tokens follow at four-address
    groups, then residual pairs, then the self-symmetric block.
    Matching is exact on (flip, zy) — weight-blind by design.
    """
    order3, ntok, nres, npairs = quartet_order(named)
    fzs = [parse_pauli(p)[:2] for p, _c in order3]
    row_of = {v: i for i, v in enumerate(fzs)}
    tok_of = {}
    for t in range(ntok):
        for o in range(4):
            tok_of[4 * t + o] = t
    partner = {}
    for t in range(ntok):
        imgs = [d_image_fz(*fzs[4 * t + o]) for o in range(4)]
        if not all(v in row_of for v in imgs):
            continue
        tg = {tok_of.get(row_of[v]) for v in imgs}
        if len(tg) == 1:
            t2 = tg.pop()
            if t2 is not None and t2 != t:
                partner[t] = t2
    merged = []
    seen = set()
    for t in sorted(partner):
        t2 = partner[t]
        if t in seen or t2 in seen or partner.get(t2) != t:
            continue
        merged.append((t, t2))
        seen.add(t)
        seen.add(t2)
    unmerged = [t for t in range(ntok) if t not in seen]

    def tkey(t):
        return (-abs(order3[4 * t][1]), order3[4 * t][0])

    order = []
    for a, _b in sorted(
        ((min(p, key=tkey), max(p, key=tkey)) for p in merged), key=lambda ab: tkey(ab[0])
    ):
        lower = [4 * a + o for o in range(4)]
        order.extend(order3[r] for r in lower)
        order.extend(order3[row_of[d_image_fz(*fzs[r])]] for r in lower)
    for t in sorted(unmerged, key=tkey):
        order.extend(order3[4 * t + o] for o in range(4))
    order.extend(order3[4 * ntok :])
    assert len(order) == len(named)
    return order, len(merged), ntok, nres, npairs


def final_order(named):
    """Contract step 4 (refinement): refine the group order under the
    thiolate frame sigma and the block-level reflection pairings.
    Regions, magnitude-sorted within each: sigma-matched D-group pairs
    as 16-row blocks (bit 3 = sigma), sigma-unmatched D-groups,
    sigma-matched D-unmatched token pairs as 8-row blocks (bit 2 =
    sigma there), remaining tokens, D-matched residual-pair quads
    (bit 1 = D there), remaining residual pairs, D-matched
    self-symmetric doubles (bit 0 = D there), remaining tail rows.
    All matching is exact on (flip, zy).
    """
    order4, ngrp, ntok, nres, npairs = group_order(named)
    fzs = [parse_pauli(p)[:2] for p, _c in order4]
    row_of = {v: i for i, v in enumerate(fzs)}
    d_end = 8 * ngrp
    c_end = 4 * ntok
    tail0 = 2 * npairs
    L = len(order4)

    def unit_of(r):
        if r < d_end:
            return ("G", r // 8)
        if r < c_end:
            return ("t", (r - d_end) // 4)
        if r < tail0:
            return ("p", (r - c_end) // 2)
        return ("s", r - tail0)

    def match(kind, count, base_of, n, img):
        partner = {}
        for k in range(count):
            imgs = [img(*fzs[base_of(k) + o]) for o in range(n)]
            if not all(v in row_of for v in imgs):
                continue
            us = {unit_of(row_of[v]) for v in imgs}
            if len(us) == 1:
                u = us.pop()
                if u[0] == kind and u[1] != k:
                    partner[k] = u[1]
        out = []
        seen = set()
        for k in sorted(partner):
            k2 = partner[k]
            if k in seen or k2 in seen or partner.get(k2) != k:
                continue
            out.append((k, k2))
            seen.add(k)
            seen.add(k2)
        return out, seen

    sg, seen_g = match("G", ngrp, lambda k: 8 * k, 8, s_image_fz)
    n_tok_u = (c_end - d_end) // 4
    st, seen_t = match("t", n_tok_u, lambda k: d_end + 4 * k, 4, s_image_fz)
    n_res = (tail0 - c_end) // 2
    dr, seen_r = match("p", n_res, lambda k: c_end + 2 * k, 2, d_image_fz)
    dtl, seen_s = match("s", L - tail0, lambda k: tail0 + k, 1, d_image_fz)

    def gkey(base):
        return (-abs(order4[base][1]), order4[base][0])

    out = []
    for a, b in sorted(sg, key=lambda ab: gkey(8 * min(ab))):
        a, b = (a, b) if gkey(8 * a) <= gkey(8 * b) else (b, a)
        lower = list(range(8 * a, 8 * a + 8))
        out.extend(lower)
        out.extend(row_of[s_image_fz(*fzs[r])] for r in lower)
    for g in sorted((g for g in range(ngrp) if g not in seen_g), key=lambda g: gkey(8 * g)):
        out.extend(range(8 * g, 8 * g + 8))
    for a, b in sorted(st, key=lambda ab: gkey(d_end + 4 * min(ab))):
        a, b = (a, b) if gkey(d_end + 4 * a) <= gkey(d_end + 4 * b) else (b, a)
        lower = list(range(d_end + 4 * a, d_end + 4 * a + 4))
        out.extend(lower)
        out.extend(row_of[s_image_fz(*fzs[r])] for r in lower)
    for t in sorted(
        (t for t in range(n_tok_u) if t not in seen_t), key=lambda t: gkey(d_end + 4 * t)
    ):
        out.extend(range(d_end + 4 * t, d_end + 4 * t + 4))
    for a, b in sorted(dr, key=lambda ab: gkey(c_end + 2 * min(ab))):
        a, b = (a, b) if gkey(c_end + 2 * a) <= gkey(c_end + 2 * b) else (b, a)
        out.extend((c_end + 2 * a, c_end + 2 * a + 1))
        out.extend(row_of[d_image_fz(*fzs[c_end + 2 * a + o])] for o in range(2))
    for k in sorted(
        (k for k in range(n_res) if k not in seen_r), key=lambda k: gkey(c_end + 2 * k)
    ):
        out.extend((c_end + 2 * k, c_end + 2 * k + 1))
    for a, b in sorted(dtl, key=lambda ab: gkey(tail0 + min(ab))):
        a, b = (a, b) if gkey(tail0 + a) <= gkey(tail0 + b) else (b, a)
        out.extend((tail0 + a, row_of[d_image_fz(*fzs[tail0 + a])]))
    for k in sorted(
        (k for k in range(L - tail0) if k not in seen_s), key=lambda k: gkey(tail0 + k)
    ):
        out.append(tail0 + k)
    assert sorted(out) == list(range(L))
    counts = (len(sg), len(st), len(dr), len(dtl))
    return [order4[r] for r in out], counts, ngrp, ntok, nres, npairs


def parse_pauli(pauli: str):
    flip = zy = 0
    ny = 0
    for tok in pauli.split():
        p, q = tok[0], int(tok[1:])
        assert p in "XYZ" and 0 <= q < N_QUBITS, tok
        if p in "XY":
            flip |= 1 << q
        if p in "ZY":
            zy |= 1 << q
        if p == "Y":
            ny += 1
    assert ny % 2 == 0, "odd Y-count would give a +-i phase: out of model"
    return flip, zy, ny


def build_table(terms):
    """Run steps 3-9 of the contract: frame-block order -> integer alias
    table -> words + masks.  ``terms`` is a list of ``(pauli_string,
    coeff)`` pairs (already truncated); the row order is recomputed
    here.
    """
    (ordered, regions, dup_groups, nspans, refc, ngrp, ntok, nres, npairs) = frame_order(
        list(terms)
    )
    n_sg, n_st, n_dr, n_dt = refc
    assert npairs == N_PAIRS_EXPECTED, f"pair census drifted: {npairs}"
    assert 2 * ntok + nres == npairs
    if N_TOKENS_EXPECTED is not None:
        assert ntok == N_TOKENS_EXPECTED, f"token census drifted: {ntok}"
    if N_GROUPS_EXPECTED is not None:
        assert ngrp == N_GROUPS_EXPECTED, f"group census drifted: {ngrp}"
    for got, want, nm in (
        (n_sg, N_SG_EXPECTED, "sigma-group"),
        (n_st, N_ST_EXPECTED, "sigma-token"),
        (n_dr, N_DR_EXPECTED, "residual-quad"),
        (n_dt, N_DT_EXPECTED, "tail-double"),
    ):
        if want is not None:
            assert got == want, f"{nm} census drifted: {got}"
    assert nspans == N_FRAME_SPANS_EXPECTED, f"span census drifted: {nspans}"
    assert len(regions) == N_FRAME_REGIONS_EXPECTED, len(regions)
    sizes = {}
    for t, lo, hi in regions:
        n = 1 << len(t)
        sizes[n] = sizes.get(n, 0) + (hi - lo) // n
    assert sizes == FRAME_SIZES_EXPECTED, f"size census drifted: {sizes}"

    ws = [c for _, c in ordered]
    L = len(ws)
    lam = sum(abs(c) for c in ws)

    targets = [abs(c) / lam * L * T for c in ws]
    floors = [int(t) for t in targets]
    rem = L * T - sum(floors)
    order = sorted(range(L), key=lambda i: (-(targets[i] - floors[i]), i))
    m = floors[:]
    for i in order[:rem]:
        m[i] += 1
    assert sum(m) == L * T

    # the credit-aligned melt scheduler (exactness asserted inside)
    keep, alt = run_atomic_alias(m, dup_groups)

    theta = [1 if c < 0 else 0 for c in ws]
    words = [
        (theta[i] << (B_WORD - 1)) | (theta[alt[i]] << (B_WORD - 2)) | (alt[i] << MU) | keep[i]
        for i in range(L)
    ]
    assert all(0 <= w < (1 << B_WORD) for w in words)
    masks = []
    for p, _ in ordered:
        flip, zy, ny = parse_pauli(p)
        masks.append(flip | (zy << N_QUBITS) | (((ny % 4) // 2) << 80))
    return (
        ordered,
        lam,
        m,
        keep,
        alt,
        theta,
        words,
        masks,
        regions,
        nspans,
        refc,
        ngrp,
        ntok,
        nres,
    )


def words_sha256(words) -> str:
    return hashlib.sha256(b"".join(w.to_bytes(8, "little") for w in words)).hexdigest()


def masks_sha256_of(masks) -> str:
    return hashlib.sha256(b"".join(mk.to_bytes(16, "little") for mk in masks)).hexdigest()


def oracle(terms, i, sigma, x):
    """Return internal f||sel, followed by the task's x_out and phase.

    The first value is useful for derivation checks but is not a boundary
    output.  terms is the alt/theta/keep/flip/zy/ny2 dict (per-row lists).
    """
    f = int(sigma < terms["keep"][i])
    sel = i if f else terms["alt"][i]
    th = terms["theta"][i] if f else terms["theta"][sel]
    word = (f << IDX_BITS) | sel
    x_out = x ^ terms["flip"][sel]
    phase = th ^ terms["ny2"][sel] ^ (bin(x & terms["zy"][sel]).count("1") & 1)
    return word, x_out, phase


def probe_sigma(terms, i, j):
    """sigma probe slot j for row i: both boundaries and both extremes."""
    k = terms["keep"][i]
    return [0, 0xFFFF, max(0, k - 1), k, min(0xFFFF, k + 1)][j]


def probe_x(terms, i, k):
    """x probe slot k for row i: the named states plus three SHA-derived
    values; slot 5 is forced to odd parity so every row deterministically
    exercises both parity sectors.
    """
    if k == 0:
        return 0
    if k == 1:
        return (1 << N_QUBITS) - 1  # all-ones (weight 40)
    if k == 2:
        return HF_STATE  # 30 electrons (weight 30)
    if k == 3:
        return terms["flip"][i]  # even weight (superselection structure)
    if k == 4:
        return terms["flip"][terms["alt"][i]]
    h = hashlib.sha256()
    h.update(SEED_DOMAIN)
    h.update(b"x-probe")
    h.update(int(i).to_bytes(4, "little"))
    h.update(bytes([k - 5]))
    d = h.digest()
    x = int.from_bytes(d[:5], "little") & ((1 << N_QUBITS) - 1)
    if k == 5 and not bin(x).count("1") & 1:
        x ^= 1  # slot 5: force odd parity (an exact 2-to-1 map)
    return x


def brute_apply(pauli: str, x: int):
    """Literal Pauli application to |x>: (x_out, phase in {+1,-1})."""
    ph = 1 + 0j
    xo = x
    for tok in pauli.split():
        p, q = tok[0], int(tok[1:])
        bit = (xo >> q) & 1
        if p == "Z":
            ph *= (-1) ** bit
        elif p == "X":
            xo ^= 1 << q
        elif p == "Y":
            ph *= 1j if bit == 0 else -1j
            xo ^= 1 << q
    assert abs(ph.imag) < 1e-12
    return xo, int(ph.real)


def emit(ordered, lam, m, keep, alt, theta, words, masks, regions, nspans, refc, ngrp, ntok, nres):
    DATA_DIR.mkdir(exist_ok=True)
    table = {
        "ordering": (
            "bounded-signature frame blocks over the seven-"
            "frame commuting family {F, C, D, D2, s_thio, "
            "s_d45, x_muS} (D2 = the composed hybrid "
            "reflection (0 18)(1 19)(2 16)(3 17)(4 15)(5 14)"
            "(6 13)(7 11)(8 12)): every row takes the deepest "
            "of 13 fixed signatures whose orbit survives in "
            "the table, a rank-k block "
            "spans 2^k aligned addresses with branch bit p "
            "keyed to the signature's p-th frame; blocks laid "
            "out size-descending, then by signature, then by "
            "descending |w|, with identical-coefficient-"
            "vector duplicate blocks contiguous at aligned "
            "slots inside each region.  The 14 maximal same-"
            "signature runs are the regions field; the alias "
            "table is the two-phase melt schedule over the "
            "run-atomic small stream (heavy larges melt onto "
            "four shared keep words via exact whole-entry "
            "subset fills, near-threshold larges close "
            "through credit chains, one conservation-closure "
            "zero-keep row)."
        ),
        "n_pairs": N_PAIRS_EXPECTED,
        "n_tokens": ntok,
        "n_groups": ngrp,
        "n_residual_pairs": nres,
        "n_sigma_groups": refc[0],
        "n_sigma_tokens": refc[1],
        "n_residual_quads": refc[2],
        "n_tail_doubles": refc[3],
        "n_spans": nspans,
        "regions": [
            {"gens": [FRAME_GEN_NAMES[g] for g in t], "start": lo, "end": hi}
            for t, lo, hi in regions
        ],
        "system": "[2Fe-2S] (30e,20o) active space",
        "fcidump": {
            "file": "data/fcidump.xz",
            "sha256": FCIDUMP_SHA256,
            "source": (
                "github.com/zhendongli2008/"
                "Active-space-model-for-Iron-Sulfur-Clusters"
                " (Fe2S2_and_Fe4S4/Fe2S2/fe2s2)"
            ),
        },
        "encoding": "jordan-wigner (interleaved spin-orbitals)",
        "n_qubits": N_QUBITS,
        "n_electrons": N_ELEC,
        "hf_state": f"{HF_STATE:#x}",
        "truncation_threshold": repr(THRESHOLD),
        "L": len(words),
        "mu": MU,
        "idx_bits": IDX_BITS,
        "b_word": B_WORD,
        "word_layout": "theta(1)|alt_theta(1)|alt(16)|keep(16), MSB..LSB",
        "lambda": repr(float(lam)),
        "toolchain": "pyscf 2.14.0, openfermion 1.8.1, numpy (venv)",
        "sha256_words_le_u64": words_sha256(words),
        "terms": [
            {
                "i": i,
                "pauli": t,
                "coeff": c.hex(),
                "m": m[i],
                "keep": keep[i],
                "alt": alt[i],
                "theta": theta[i],
                "word": f"{words[i]:#011x}",
            }
            for i, (t, c) in enumerate(ordered)
        ],
    }
    payload = (json.dumps(table, indent=1) + "\n").encode()
    TABLE_XZ.write_bytes(lzma.compress(payload, preset=9))
    mdigest = masks_sha256_of(masks)
    mdoc = {
        "_generated": "by reference.py --emit - DO NOT EDIT BY HAND",
        "description": (
            "chemistry-fe2s2 Pauli-action masks for the "
            "[2Fe-2S] (30e,20o) JW table (data/table.json.xz)"
        ),
        "packing": "flip(40) | zy(40) << 40 | ny2(1) << 80",
        "L": len(masks),
        "sha256_le_16B": mdigest,
        "masks": [f"{mk:#x}" for mk in masks],
    }
    mpayload = (json.dumps(mdoc, indent=1) + "\n").encode()
    MASKS_XZ.write_bytes(lzma.compress(mpayload, preset=9))
    print(
        f"emitted data/table.json.xz ({len(words)} words, "
        f"{TABLE_XZ.stat().st_size} B) sha256_words={words_sha256(words)}"
    )
    print(f"emitted data/masks.json.xz ({MASKS_XZ.stat().st_size} B) sha256_masks={mdigest}")
    print(f"lambda_kept={lam!r}")


def load_pinned():
    """Pinned terms as [(pauli_str, coeff_float)], plus words and the
    raw table dict.
    """
    table = json.loads(lzma.decompress(TABLE_XZ.read_bytes()))
    pinned_terms = [(t["pauli"], float.fromhex(t["coeff"])) for t in table["terms"]]
    words = [int(t["word"], 16) for t in table["terms"]]
    return pinned_terms, words, table


def check_pin_internal_consistency():
    """Bit-exact: rebuild steps 3-8 from the PINNED coefficients; must
    equal the pinned words, masks and digests.
    """
    pinned_terms, pinned_words, table = load_pinned()
    assert len(pinned_words) == L_EXPECTED
    assert DATA_SHA256 is not None and MASKS_SHA256 is not None, (
        "digest pins not populated (run --emit and record them)"
    )
    assert words_sha256(pinned_words) == DATA_SHA256 == table["sha256_words_le_u64"]
    rebuilt = build_table([(p, c) for p, c in pinned_terms])
    (_, lam, _, _, _, _, words, masks, _, _, _, _, _, _) = rebuilt
    assert words == pinned_words, "pinned coeffs do not rebuild the pinned words"
    assert masks_sha256_of(masks) == MASKS_SHA256
    mdoc = json.loads(lzma.decompress(MASKS_XZ.read_bytes()))
    assert [int(mk, 16) for mk in mdoc["masks"]] == masks
    assert repr(float(lam)) == table["lambda"]
    print(
        f"pin self-check OK: L={len(pinned_words)}, lambda={lam:.4f}, "
        f"words sha256={DATA_SHA256[:16]}..., "
        f"masks sha256={MASKS_SHA256[:16]}..."
    )
    return pinned_terms, words, masks


def selfcheck_physics(pinned_terms, words, masks):
    """Structure pins + closed-form sign vs literal Pauli application."""
    L = len(words)
    keep = [w & 0xFFFF for w in words]
    alt = [(w >> MU) & 0xFFFF for w in words]
    theta = [(w >> (B_WORD - 1)) & 1 for w in words]
    alt_theta = [(w >> (B_WORD - 2)) & 1 for w in words]
    assert all(a < L for a in alt), "alt escapes the graded domain"
    assert all(alt_theta[i] == theta[alt[i]] for i in range(L)), (
        "theta_alt prefetch identity broken"
    )
    for i in range(L):
        if keep[i] == 0:
            assert alt[i] == i, f"row {i}: zero-keep row must self-alias"
    fw = {}
    for mk in masks:
        w = bin(mk & ((1 << N_QUBITS) - 1)).count("1")
        assert w % 2 == 0, "odd flip weight breaks superselection"
        fw[w] = fw.get(w, 0) + 1
    # closed form vs literal application on a deterministic sample
    step = max(1, L // 257)
    for i in list(range(0, L, step)) + [L - 1]:
        p = pinned_terms[i][0]
        mk = masks[i]
        flip = mk & ((1 << N_QUBITS) - 1)
        zy = (mk >> N_QUBITS) & ((1 << N_QUBITS) - 1)
        ny2 = (mk >> 80) & 1
        for x in (0, (1 << N_QUBITS) - 1, HF_STATE, flip, (i * 2654435761) & ((1 << N_QUBITS) - 1)):
            xo, ph = brute_apply(p, x)
            assert xo == x ^ flip
            assert ph == (-1) ** (ny2 ^ (bin(x & zy).count("1") & 1)), (i, x)
    print(
        f"physics self-check OK: alt bound, prefetch identity, zero-keep "
        f"self-alias, even flip weights {sorted(fw)} , closed-form sign "
        f"vs literal Pauli on {len(range(0, L, step)) + 1} rows x 5 states"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--emit",
        action="store_true",
        help="(re)write the data files from a fresh FCIDUMP -> JW derivation (needs pyscf)",
    )
    ap.add_argument(
        "--emit-from-pin",
        action="store_true",
        help="(re)write the data files from the PINNED coefficients (steps 3-9 only; no pyscf)",
    )
    ap.add_argument(
        "--skip-derive", action="store_true", help="only self-check the pinned files (no pyscf run)"
    )
    args = ap.parse_args()

    if args.emit or args.emit_from_pin:
        if args.emit_from_pin:
            kept = [(p, c) for p, c in load_pinned()[0]]
        else:
            named = [(term_str(t), c) for t, c in derive_terms()]
            kept = truncate(named)
        out = build_table(kept)
        emit(*out)
        return 0

    pinned_terms, words, masks = check_pin_internal_consistency()
    selfcheck_physics(pinned_terms, words, masks)
    if args.skip_derive:
        return 0

    # Physics check: fresh FCIDUMP -> JW run must agree with the pin to
    # TOL per coefficient and reproduce the same kept set.
    TOL = 1e-9
    fresh_named = [(term_str(t), c) for t, c in derive_terms()]
    fresh = {p: c for p, c in truncate(fresh_named)}
    assert set(fresh) == {p for p, _ in pinned_terms}, "term set drifted"
    worst = max(abs(fresh[p] - c) for p, c in pinned_terms)
    assert worst <= TOL, f"coefficient drift {worst:.3e} > {TOL:.0e}"
    print(f"FCIDUMP re-derivation agrees with pin: max |delta coeff| = {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
