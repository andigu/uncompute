#!/usr/bin/env python3
"""chemistry-fe2s2 circuits/discovered: a melt-scheduled seven-frame block walk —
the SELECT descent stops at 3,221 frame-block roots, the alias lookup
skips the spans the melt schedule makes constant under exact
hold-budget DPs, and one deferred plane-collapsed pass repairs both
measurement kickbacks jointly.  The selection swap (sel <-> i under
f) carries sel through SELECT and the identical swap restores it; the
keep bank stays live so the identical comparator pass erases f, and
both banks then demolish into the joint deferred fixup — no selection
data survives to the boundary.

Cost: 14,978 Toffoli x 126 qubits = 1,887,228 (assert_metrics in the
epilogue).

RUN: uv run python circuit.py          builds, asserts metrics, emits
     python tools/eval.py chemistry-fe2s2    evaluates (repo root)
build() returns (b, regs); profilers import the module and call build()
without emitting.

README.md in this folder describes the circuit.
"""

import functools
import json
import pathlib

from uncompute import Builder
from uncompute.classical import f2_matinv, parity
from uncompute.gadgets import and_chain, declare_registers, release_chain, xor_outcomes, xor_word
from uncompute.testing import assert_metrics, emit_and_report, load_contract, load_module

L, A, S, X = 41052, 16, 16, 40
B = 15  # word bank: keep bits 0-14 (bit 15 -> kc)
# The round-trip peak sits at the SELECT stage (112 + up to 14 transient
# = 126, the keep bank held through it for the f erasure); the hold
# budgets below spend the matching headroom in every other phase.  Each
# walk DP saturates below its budget (lookup at 15 holds, fixup at 11),
# so the budgets cost no qubit.
PEAK = 112 + 14  # the SELECT-stage peak
H_LOOK = PEAK - 104  # lookup-walk hold budget: the lookup holds 104 live wires
KW, LOG_KW = 16, 4  # fixup window slots (bits 2-5)
FIX_WIN = 64  # fixup window rows (bits 0-5)
N_FIX = -(-L // FIX_WIN)  # fixup windows (642)
H_FIX = PEAK - 88  # fixup-walk hold budget: the fixup holds 88 live wires
CMP_CHAIN = 16  # comparator borrow stages: all on temporary ANDs
assert KW == 1 << LOG_KW
# Berry et al.'s T-optimal window is berry_ku(L) = 256; the peak budget holds
# the one-hot at 16 slots (windows of 64 rows, planes for the low 2).
CONTRACT = load_contract(__file__)
XMASK = (1 << X) - 1
EVN = 0x5555555555  # even (spin-up) mode positions
ODM = EVN << 1  # odd (spin-down) mode positions

# The frame family, indexed as in contract regions; the application
# (nesting) order is the index order, F innermost.  GAUGE_A[g] /
# GAUGE_C[g] dress frame g with inner X^a / Z^c layers (free CX / CZ
# on the flag): the sign shift gains parity(a AND z) / parity(c AND f)
# on the pre-image mask.  The Z masks — CZ on orbital pairs each frame
# exchanges — minimize the per-block monomial channel over the X/Z
# gauge family (channel cost 117).  The walk-node channel below then
# carries the emitted total to 108.  gen_constants.py searches both
# sets and writes constants.json; sign_tables() reads it.
GEN_NAMES = ["F", "C", "D", "D2", "s_thio", "s_d45", "x_muS"]
PRIORITY = list(range(len(GEN_NAMES)))
GAUGE_A: dict = {}  # X^a layers: sign parity(a & z)
_GIDX = {n: i for i, n in enumerate(GEN_NAMES)}
REGIONS = [
    (tuple(_GIDX[g] for g in rg["gens"]), rg["start"], rg["end"]) for rg in CONTRACT["regions"]
]
_prev = 0
for _t, _lo, _hi in REGIONS:
    assert _lo == _prev and list(_t) == sorted(_t)
    assert (_hi - _lo) % (1 << len(_t)) == 0
    _prev = _hi
assert _prev == L


# The pinned table, parsed by ../_shared/baseline.py (the baseline's
# loader; it also asserts the table shape).
_sh = load_module(
    pathlib.Path(__file__).resolve().parent.parent / "_shared" / "baseline.py", "alias_baseline"
)
WORDS, MASKS = _sh.DATA, _sh.MASKS
# Sign-stripped lookup fields; keep and alt never coexist in scratch.
DATA = [w & 0xFFFFFFFF for w in WORDS]
KEEPW = [w & 0xFFFF for w in DATA]
ALTW = [(w >> 16) & 0xFFFF for w in DATA]
SIGN = [((w >> 33) ^ (m >> 80)) & 1 for w, m in zip(WORDS, MASKS)]
assert all(DATA)


def _pi(v):
    """Swap adjacent bit pairs (2p <-> 2p+1) of a 40-bit mask."""
    return ((v & EVN) << 1) | ((v >> 1) & EVN)


def _g(f):
    """g(f) = sum_p f_{2p} f_{2p+1} mod 2 — the F frame's sign shift."""
    return parity(f & (f >> 1) & EVN)


def _czy(f, z):
    """Return the C frame's zy image (K_evens + K_odds CZ graph)."""
    we = parity(f & EVN)
    wo = parity(f & ODM)
    return z ^ (EVN if we else 0) ^ (ODM if wo else 0) ^ f


def _s(f):
    """s(f) = e2(f_even) ^ e2(f_odd) mod 2 — the C frame's sign shift."""
    we = bin(f & EVN).count("1")
    wo = bin(f & ODM).count("1")
    return ((we * (we - 1) // 2) ^ (wo * (wo - 1) // 2)) & 1


# ---- permutation-frame machinery (D, D2 and the orbital swaps) ------------
def _perm_machinery(orb):
    """From an orbital involution: the mode permutation PERM, the mask
    image map (adjacent-FSWAP composition), and the legal decomposition
    of the controlled frame — the CZ graph S = Q P and its
    prefix-parity collapse into bilinear pairs + linear terms.
    """
    modes = [0] * X
    for p in range(20):
        modes[2 * p] = 2 * orb[p]
        modes[2 * p + 1] = 2 * orb[p] + 1
    seq = list(modes)
    ts = []
    for _ in range(X):
        for j in range(X - 1):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                ts.append(j)

    def dimg(f, z):
        for m in ts:
            fm = (f >> m) & 1
            fm1 = (f >> (m + 1)) & 1
            zm = (z >> m) & 1
            zm1 = (z >> (m + 1)) & 1
            f ^= ((fm ^ fm1) << m) | ((fm ^ fm1) << (m + 1))
            nzm = zm1 ^ fm
            nzm1 = zm ^ fm1
            z ^= ((zm ^ nzm) << m) | ((zm1 ^ nzm1) << (m + 1))
        return f, z

    pcols, qcols = [], []
    for k in range(X):
        f1, z1 = dimg(1 << k, 0)
        pcols.append(f1)
        qcols.append(z1)
    assert all(bin(c).count("1") == 1 for c in pcols)
    perm = [c.bit_length() - 1 for c in pcols]
    for k in range(X):
        f1, z1 = dimg(0, 1 << k)
        assert f1 == 0 and z1 == pcols[k]
    scols = [qcols[perm[v]] for v in range(X)]
    for u in range(X):
        assert not (scols[u] >> u) & 1
        for v in range(u + 1, X):
            assert ((scols[v] >> u) & 1) == ((scols[u] >> v) & 1)
    # prefix-parity collapse of the S form: x_k = u_k ^ u_{k-1}
    form = {}
    for u in range(X):
        for v in range(u + 1, X):
            if not (scols[v] >> u) & 1:
                continue
            for a in (u,) if u == 0 else (u, u - 1):
                for bb in (v,) if v == 0 else (v, v - 1):
                    key = (min(a, bb), max(a, bb))
                    form[key] = form.get(key, 0) ^ 1
    bilin = sorted(k for k, c in form.items() if c and k[0] != k[1])
    lin = sorted(k[0] for k, c in form.items() if c and k[0] == k[1])
    return perm, dimg, scols, bilin, lin


def _ppp(pairs):
    p = list(range(20))
    for a, b in pairs:
        p[a], p[b] = b, a
    return p


PERM_MACH = {
    2: _perm_machinery([19 - p for p in range(20)]),  # D
    3: _perm_machinery(
        [18, 19, 16, 17, 15, 14, 13, 11, 12, 9, 10, 7, 8, 6, 5, 4, 2, 3, 0, 1]
    ),  # D2
    4: _perm_machinery(_ppp([(0, 1), (18, 19)])),  # s_thio
    5: _perm_machinery(_ppp([(4, 5), (14, 15)])),  # s_d45
    6: _perm_machinery(_ppp([(7, 11), (8, 12)])),  # x_muS
}


def _apply_perm(perm, f):
    out = 0
    for k in range(X):
        if (f >> k) & 1:
            out |= 1 << perm[k]
    return out


def _qs_graph(scols, h):
    """Edges of a CZ graph inside h — a permutation frame's sign."""
    acc = 0
    hh = h
    while hh:
        lb = hh & -hh
        u = lb.bit_length() - 1
        hh ^= lb
        acc ^= parity(scols[u] & hh)
    return acc


def _fz(r):
    return MASKS[r] & XMASK, (MASKS[r] >> X) & XMASK


def _img(g, f, z):
    if g == 0:
        return _pi(f), _pi(z) ^ f
    if g == 1:
        return f, _czy(f, z)
    return PERM_MACH[g][1](f, z)


def _sgn(g, f, z, gauge_c):
    if g == 0:
        base = _g(f)
    elif g == 1:
        base = _s(f)
    else:
        perm, _dimg, scols, _b, _l = PERM_MACH[g]
        base = _qs_graph(scols, _apply_perm(perm, f))
    a = GAUGE_A.get(g)
    if a:
        base ^= parity(a & z)
    c = gauge_c.get(g)
    if c:
        base ^= parity(c & f)
    return base


# Block partition: a block of 2^k rows starts at every aligned base of a
# rank-k region; SPAN_AT maps each block base to its row count.
SPAN_AT = {}
for _t, _lo, _hi in REGIONS:
    _n = 1 << len(_t)
    for _base in range(_lo, _hi, _n):
        assert _base % _n == 0, (_t, _base)
        SPAN_AT[_base] = _n
del _t, _lo, _hi, _n, _base


@functools.cache
def sign_tables():
    """Return (GAUGE_C, LIN_SIGNS, MONO_SIGNS, NODE_MONOS).

    The first call reads constants.json, so gen_constants.py can import
    this module before constants.json exists.

    Table shape — asserted, not just measured: every block of every
    region reproduces its branch-composed frame images row by row, and
    the sign residual of the fixed nesting (gauge dressing included) is
    extracted as a GF(2) polynomial per block.

    Walk-node monomial channel: each toggle (lo, hi, monomial) hosts one
    sign monomial at the select-walk node covering [lo, hi) — the node
    gate fires exactly where every covered block's copy would, so the
    per-block polynomials absorb the flip and emit only the leftover.
    Legality is asserted: [lo, hi) is a walk-aligned span tiled EXACTLY
    by blocks whose frame-set rank exceeds the monomial's top branch
    bit (branch bit p of a block is sel bit p).
    """
    cj = json.loads((pathlib.Path(__file__).resolve().parent / "constants.json").read_bytes())
    gauge_c = {
        int(g): int(c, 16)  # Z^c layers: sign parity(c & f)
        for g, c in cj["GAUGE_C"].items()
    }
    lin_signs = {}
    mono_signs = {}
    for t, lo, hi in REGIONS:
        k = len(t)
        if k == 0:
            continue
        n = 1 << k
        for base in range(lo, hi, n):
            st = {0: (*_fz(base), 0)}
            resid = [0] * n
            for bm in range(1, n):
                i = bm.bit_length() - 1
                f0, z0, s0 = st[bm ^ (1 << i)]
                f1, z1 = _img(t[i], f0, z0)
                sa = s0 ^ _sgn(t[i], f0, z0, gauge_c)
                st[bm] = (f1, z1, sa)
                assert _fz(base + bm) == (f1, z1), (t, base, bm)
                resid[bm] = SIGN[base] ^ sa ^ SIGN[base + bm]
            mob = resid[:]
            for i in range(k):
                stp = 1 << i
                for lo2 in range(0, n, stp << 1):
                    for j in range(stp):
                        mob[lo2 + stp + j] ^= mob[lo2 + j]
            assert mob[0] == 0, base
            lin = tuple(i for i in range(k) if mob[1 << i])
            mono = sorted(
                (
                    tuple(i for i in range(k) if (m >> i) & 1)
                    for m in range(1, n)
                    if mob[m] and bin(m).count("1") >= 2
                ),
                key=lambda mn: (len(mn), mn),
            )
            if lin:
                lin_signs[base] = lin
            if mono:
                mono_signs[base] = tuple(mono)

    node_monos: dict = {}
    for lo, hi, mn in ((lo, hi, tuple(mn)) for lo, hi, mn in cj["NODE_MONOS"]):
        sp = hi - lo
        assert sp & (sp - 1) == 0 and lo % sp == 0 and lo in SPAN_AT
        cov = [bb for bb in range(lo, hi) if bb in SPAN_AT]
        assert sum(SPAN_AT[bb] for bb in cov) == sp  # exact tiling
        for bb in cov:
            assert bb + SPAN_AT[bb] <= hi
            assert SPAN_AT[bb].bit_length() - 1 > max(mn)  # branch support
            ms = set(mono_signs.pop(bb, ()))
            ms ^= {mn}
            if ms:
                mono_signs[bb] = tuple(sorted(ms, key=lambda mn: (len(mn), mn)))
        node_monos.setdefault((lo, sp), []).append(mn)
    for monos in node_monos.values():
        monos.sort(key=lambda mn: (len(mn), mn))
    return gauge_c, lin_signs, mono_signs, node_monos


# Gate intervals: per (frame, branch-bit position), maximal runs of
# adjacent regions sharing the assignment.
GATE_IVS: list = []
_opened: dict = {}
for _t, _lo, _hi in REGIONS:
    for _p, _g2 in enumerate(_t):
        _iv = _opened.get((_g2, _p))
        if _iv is not None and GATE_IVS[_iv][3] == _lo:
            GATE_IVS[_iv] = (_g2, _p, GATE_IVS[_iv][2], _hi)
        else:
            _opened[(_g2, _p)] = len(GATE_IVS)
            GATE_IVS.append((_g2, _p, _lo, _hi))
GEN_IVS = {g: [(p, lo, hi) for g2, p, lo, hi in GATE_IVS if g2 == g] for g in range(len(GEN_NAMES))}
del _t, _lo, _hi, _p, _g2, _iv, _opened


# ---- strict comparator, hybrid (Cuccaro ripple + top borrow chain) --------
# Cuccaro et al. (2004), arXiv:quant-ph/0410184: the MAJ ripple blocks.
def compare_into(b, sigma, keep_bits, target):
    """target ^= [sigma < keep]: the borrow of sigma - keep ripples IN
    PLACE through the low S - CMP_CHAIN sigma wires (Cuccaro MAJ
    blocks, 2 Toffoli each; one |0> seed ancilla); the top CMP_CHAIN
    stages carry the borrow on temporary ANDs (1 Toffoli each,
    released measurement-based) — t = MAJ(~sigma_k, keep_k, borrow)
    via the CX-prepped AND.  2(S - CMP_CHAIN) + CMP_CHAIN Toffoli,
    peak +1 seed + CMP_CHAIN chain wires; both registers restored
    exactly.
    """
    c0 = b.qubit()
    b.claim("carry", [c0])
    prev = c0
    r = S - CMP_CHAIN
    for k in range(r):
        a, kb = sigma[k], keep_bits[k]
        b.x(a)
        b.cx(a, kb)
        b.cx(a, prev)
        b.ccx(prev, kb, a)
        prev = a
    chain = []
    for k in range(r, S):
        a, kb = sigma[k], keep_bits[k]
        b.x(a)
        b.cx(prev, a)
        b.cx(prev, kb)
        t = b.and_compute(a, kb)
        b.claim("carry", [t])
        b.cx(prev, t)  # t = MAJ(~sigma_k, keep_k, prev)
        chain.append((t, a, kb, prev))
        prev = t
    b.cx(prev, target)
    for t, a, kb, pr in reversed(chain):
        b.cx(pr, t)
        b.release("carry", [t])
        b.and_release(t, a, kb)
        b.cx(pr, kb)
        b.cx(pr, a)
        b.x(a)
    for k in reversed(range(r)):
        a, kb = sigma[k], keep_bits[k]
        pr = sigma[k - 1] if k else c0
        b.ccx(pr, kb, a)
        b.cx(a, pr)
        b.cx(a, kb)
        b.x(a)
    b.release_recycle("carry", [c0])


# ---- the periodicity-pruned, adaptively capped lookup walk ----------------
def _chain_hold(b, lits):
    """Held temporary-AND chain (Gidney 2018, arXiv:1709.06648) over
    >= 2 (wire, polarity) literals, every AND wire in the "hold" set;
    returns (final wire, trail).  Unwind
    with _chain_unwind.
    """
    top, trail = and_chain(b, lits)
    b.claim("hold", [t for t, *_ in trail])
    return top, trail


def _chain_unwind(b, trail):
    b.release("hold", [t for t, *_ in reversed(trail)])
    release_chain(b, trail)


def _acc_build(b, lits):
    """Build the sawtooth accumulator: build the chain (k-1 ANDs), release
    every intermediate immediately (reverse order — each release's
    inputs are still live), keep ONLY the final wire.
    """
    acc, trail = _chain_hold(b, lits)
    _chain_unwind(b, trail[:-1])
    return acc


def _acc_teardown(b, lits, acc):
    """Release the accumulator: rebuild the held prefix (k-2 ANDs),
    release acc against it, unwind the prefix — 2k-3 total with the
    build, exactly the borrow-chain recompute idiom.
    """
    b.release("hold", [acc])
    if len(lits) == 2:
        b.and_release(acc, lits[0][0], lits[1][0], pol=(lits[0][1], lits[1][1]))
        return
    pre, chain = _chain_hold(b, lits[:-1])
    b.and_release(acc, pre, lits[-1][0], pol=(True, lits[-1][1]))
    _chain_unwind(b, chain)


# The adaptive cap: a walk's content tree (leaf > skip > split, the
# circuit predicate order) is priced by an exact DP over walker states
# (r = remaining hold budget, t = pending control terms — a held ctrl
# wire and bare path literals are cost-equivalent), and the realizing
# policy drives the emission.  Mechanisms priced: leaf chain (t-1
# ANDs, all held over the payload), leaf accumulator (2t-3, ONE held),
# accumulator conversion at a node (2t-3, descend from one wire),
# single-wire split (1 AND serving both children), literal stacking
# (free), skip and one-sided descent (free: rows i >= L are untested).
_INF = float("inf")


class _WNode:
    __slots__ = ("kids", "kind", "lo", "load", "span")

    def __init__(self, kind, kids=(), lo=0, span=0, load=0):
        self.kind = kind
        self.kids = kids
        self.lo = lo
        self.span = span
        self.load = load


def _content_tree(lo, span, limit, uniform, skip, leaf_load):
    if lo >= limit:
        return None
    if span == 1 or (uniform is not None and uniform(lo, span)):
        return _WNode("leaf", lo=lo, span=span, load=leaf_load(lo) if leaf_load else 0)
    half = span >> 1
    if skip is not None and skip(lo, span):
        return _WNode("skip", (_content_tree(lo, half, limit, uniform, skip, leaf_load),), lo, span)
    left = _content_tree(lo, half, limit, uniform, skip, leaf_load)
    right = _content_tree(lo + half, half, limit, uniform, skip, leaf_load)
    if right is None:
        return _WNode("one", (left,), lo, span)
    return _WNode("two", (left, right), lo, span)


def _walk_policy(root, H):
    """-> (minimum walk ANDs under hold budget H, realizing choice per
    (node, r, t) state).  Leaf loads are the payload's own transient
    wires (plane ANDs), priced into feasibility.
    """
    memo = {}
    choice = {}

    def f(n, r, t):
        if t > r + 1:
            return _INF
        key = (id(n), r, t)
        v = memo.get(key)
        if v is not None:
            return v
        if n.kind == "leaf":
            if t >= 2:
                v, ch = _INF, None
                if t - 1 + n.load <= r:
                    v, ch = t - 1, ("chain",)
                if t - 1 <= r and 1 + n.load <= r and 2 * t - 3 < v:
                    v, ch = 2 * t - 3, ("acc",)
                choice[key] = ch
            else:
                v = 0 if n.load <= r else _INF
        elif n.kind in ("skip", "one"):
            v = f(n.kids[0], r, t)
            ch = ("desc",)
            if t >= 2 and t - 1 <= r:
                alt = 2 * t - 3 + f(n.kids[0], r - 1, 1)
                if alt < v:
                    v, ch = alt, ("conv-desc",)
            choice[key] = ch
        else:
            left, right = n.kids
            best, ch = _INF, None

            def opts(r0, t0):
                res = []
                if t0 == 0:
                    res.append((f(left, r0, 1) + f(right, r0, 1), "bare"))
                elif t0 == 1 and r0 >= 1:
                    res.append((1 + f(left, r0 - 1, 1) + f(right, r0 - 1, 1), "split"))
                if t0 >= 1:
                    res.append((f(left, r0, t0 + 1) + f(right, r0, t0 + 1), "lit"))
                return res

            for c, name in opts(r, t):
                if c < best:
                    best, ch = c, (name,)
            if t >= 2 and t - 1 <= r:
                for c, name in opts(r - 1, 1):
                    if 2 * t - 3 + c < best:
                        best, ch = 2 * t - 3 + c, ("conv", name)
            choice[key] = ch
            v = best
        memo[key] = v
        return v

    return f(root, H, 0), choice


def _policy_walk(b, root, choice, H, bit_of, leaf_body):
    """Emit the walk the DP policy prescribes, byte-idiom-identical to
    the chain/accumulator helpers above.
    """

    def go(n, r, lits):
        t = len(lits)
        if n.kind == "leaf":
            assert t >= 1
            if t == 1:
                w, pol = lits[0]
                if not pol:
                    b.x(w)
                leaf_body(n.lo, w)
                if not pol:
                    b.x(w)
            elif choice[(id(n), r, t)][0] == "chain":
                acc, chain = _chain_hold(b, lits)
                leaf_body(n.lo, acc)
                _chain_unwind(b, chain)
            else:
                acc = _acc_build(b, lits)
                leaf_body(n.lo, acc)
                _acc_teardown(b, lits, acc)
            return
        if n.kind in ("skip", "one"):
            if choice.get((id(n), r, t), ("desc",))[0] == "conv-desc":
                acc = _acc_build(b, lits)
                go(n.kids[0], r - 1, [(acc, True)])
                _acc_teardown(b, lits, acc)
            else:
                go(n.kids[0], r, lits)
            return
        ch = choice[(id(n), r, t)]
        if ch[0] == "conv":
            acc = _acc_build(b, lits)
            split(n, r - 1, [(acc, True)], ch[1])
            _acc_teardown(b, lits, acc)
        else:
            split(n, r, lits, ch[0])

    def split(n, r, lits, action):
        left, right = n.kids
        hb = bit_of(n)
        if action == "bare":
            b.x(hb)
            go(left, r, [(hb, True)])
            b.x(hb)
            go(right, r, [(hb, True)])
        elif action == "split":
            w, pol = lits[0]

            def flip():
                if not pol:
                    b.x(w)
                b.cx(w, anc)
                if not pol:
                    b.x(w)

            anc = b.and_compute(w, hb, pol=(pol, True))
            b.claim("hold", [anc])
            go(right, r - 1, [(anc, True)])
            flip()
            go(left, r - 1, [(anc, True)])
            flip()
            b.release("hold", [anc])
            b.and_release(anc, w, hb, pol=(pol, True))
        else:  # "lit": stack the address bit
            go(left, r, lits + [(hb, False)])
            go(right, r, lits + [(hb, True)])

    go(root, H, [])


def _lk_uniform(lo, span):
    hi = min(lo + span, L)
    first = DATA[lo]
    return all(v == first for v in DATA[lo + 1 : hi])


def _lk_skip(lo, span):
    half = span >> 1
    rh = min(lo + span, L) - (lo + half)
    return rh > 0 and DATA[lo : lo + rh] == DATA[lo + half : lo + half + rh]


_LK_ROOT = _content_tree(0, 1 << A, L, _lk_uniform, _lk_skip, None)
_LK_ANDS, _LK_POLICY = _walk_policy(_LK_ROOT, H_LOOK)
assert _LK_ANDS < _INF


def lookup_walk(b, idx_msb, leaf_at):
    """One content-pruned descent over [0, 2^A): periodic skip (two
    halves with identical in-domain content recurse once, no AND) and
    one-sided descent past L are free; everywhere else the load-time
    DP policy places the held chain, split, and accumulator wires.
    """
    _policy_walk(
        b, _LK_ROOT, _LK_POLICY, H_LOOK, lambda n: idx_msb[A - (n.span.bit_length() - 1)], leaf_at
    )


# ---- Berry et al. App. C: measurement-based erase -------------------------
def erase_wires(b, wires):
    """X-measure a list of wires (release from their sets first via the
    caller); the kickback is repaired by the B(2) fixup.
    """
    return [b.hmr(q) for q in wires]


def _xor_terms(*lists):
    """Symmetric difference of outcome-term lists (GF(2) sum)."""
    cnt: dict = {}
    order = []
    for terms in lists:
        for o in terms:
            if o in cnt:
                cnt[o] ^= 1
            else:
                cnt[o] = 1
                order.append(o)
    return [o for o in order if cnt[o]]


# Per-window plane census (the fixup leaves' own transient wires, fed
# to the DP as leaf loads): 2 held wires when the pair plane fires (it
# chains through a held single), 1 for a lone single plane, else 0.
# XOR over the 32-bit (keep | alt << 16) row patterns is exactly the
# outcome-term symmetric difference fixup_leaf computes at build time
# (asserted there).
def _plane_set(wv):
    base = wv * FIX_WIN
    planes = set()
    for sm in (1, 2, 3):
        for lv in range(KW):
            gbase = base + (lv << 2)
            if gbase >= L:
                continue
            acc = 0
            for rr in range(4):
                if not rr & ~sm:
                    acc ^= DATA[gbase + rr if gbase + rr < L else gbase]
            if acc:
                planes.add(sm)
                break
    return frozenset(planes)


_PLANE_SETS = [_plane_set(_wv) for _wv in range(N_FIX)]
_FX_ROOT = _content_tree(
    0,
    1 << (A - 6),
    N_FIX,
    None,
    None,
    lambda wv: 2 if 3 in _PLANE_SETS[wv] else (1 if _PLANE_SETS[wv] else 0),
)
_FX_ANDS, _FX_POLICY = _walk_policy(_FX_ROOT, H_FIX)
assert _FX_ANDS < _INF


def _capped_window_walk(b, bits, body):
    """Walk the dense windows over [0, N_FIX) in [0, 2^10) under the
    lookup's DP policy at hold budget H_FIX — one-sided nodes imply
    the bit (the walk gadget's control-removal rule, out-of-window
    addresses being i >= L), and the per-window plane transients are
    priced into the leaf states.  ``bits`` little-endian, as the
    window suffix of the address register.
    """
    _policy_walk(
        b,
        _FX_ROOT,
        _FX_POLICY,
        H_FIX,
        lambda n: bits[n.span.bit_length() - 2],
        lambda lo, w: body(w, lo),
    )


def b2_fixup(b, idx, m_keep, m_alt):  # noqa: C901 - straight-line circuit emitter
    """Emit the joint deferred unlookup repair at B(2) granularity: a dense
    window walk over bits 6-15 (642 windows of 64 rows, DP-capped at
    hold budget H_FIX), the 16-slot one-hot routing of bits 2-5,
    per-slot outcome-conditioned CZs for the quad-base fix, and
    Reed-Muller plane ANDs for the low two bits — one temporary AND
    per (window, plane) with free conditioned CZs.  Out-of-domain rows
    inside the last window inherit their quad base (rows i >= L are
    untested).
    """

    def row_terms(iv):
        t = [m_keep[t2] for t2 in range(16) if (KEEPW[iv] >> t2) & 1]
        t += [m_alt[t2] for t2 in range(16) if (ALTW[iv] >> t2) & 1]
        return t

    mid, hu = idx[2:6], idx[6:]
    with b.phase("unlookup/window:prep"):
        u = b.qubits(KW)
        b.claim("window", u)
        b.x(u[0])
        for i in reversed(range(LOG_KW)):
            for s in range(0, KW, 1 << (i + 1)):
                b.cswap(mid[i], u[s], u[s + (1 << i)])

    def fixup_leaf(ctrl, wv):
        base = wv * FIX_WIN
        planes = {}
        for Smask in (1, 2, 3):
            need = False
            for lv in range(KW):
                gbase = base + (lv << 2)
                if gbase >= L:
                    continue
                acc = []
                for rr in range(4):
                    if rr & ~Smask:
                        continue
                    row = gbase + rr
                    acc = _xor_terms(acc, row_terms(row if row < L else gbase))
                if xor_outcomes(acc) is not None:
                    need = True
                    break
            if need:
                planes[Smask] = None
        assert frozenset(planes) == _PLANE_SETS[wv]
        win_terms = {}
        for lv in range(KW):
            gbase = base + (lv << 2)
            if gbase >= L:
                continue
            win_terms[lv] = {
                rr: row_terms(gbase + rr if gbase + rr < L else gbase) for rr in range(4)
            }
            gb = xor_outcomes(win_terms[lv][0])
            if gb is not None:
                b.cz(ctrl, u[lv], cond=gb)

        def emit_plane(pw, Smask):
            for lv, terms in win_terms.items():
                acc = []
                for rr in range(4):
                    if rr & ~Smask:
                        continue
                    acc = _xor_terms(acc, terms[rr])
                gb = xor_outcomes(acc)
                if gb is not None:
                    b.cz(pw, u[lv], cond=gb)

        # the r0r1 plane chains through a HELD single-bit plane: two
        # plane wires live only across the pair's emission (the same
        # transient the build itself has), so its release recomputes
        # nothing, and a needed single rides the held wire for free
        done = set()
        if 3 in planes:
            lo_S = 1 if 1 in planes else 2
            lo_bit, hi_bit = (idx[0], idx[1]) if lo_S == 1 else (idx[1], idx[0])
            plo = b.and_compute(ctrl, lo_bit)
            p3 = b.and_compute(plo, hi_bit)
            emit_plane(p3, 3)
            b.and_release(p3, plo, hi_bit)
            if lo_S in planes:
                emit_plane(plo, lo_S)
                done.add(lo_S)
            b.and_release(plo, ctrl, lo_bit)
            done.add(3)
        for Sm, ibit in ((1, idx[0]), (2, idx[1])):
            if Sm in planes and Sm not in done:
                pw = b.and_compute(ctrl, ibit)
                emit_plane(pw, Sm)
                b.and_release(pw, ctrl, ibit)

    with b.phase("unlookup/window:fixup"):
        _capped_window_walk(b, hu, fixup_leaf)
    with b.phase("unlookup/window:unprep"):
        for i in range(LOG_KW):
            for s in range(0, KW, 1 << (i + 1)):
                v = u[s + (1 << i)]
                b.cx(v, u[s])
                b.release("window", [v])
                b.and_release(v, mid[i], u[s])
        b.x(u[0])
        b.release_recycle("window", [u[0]])


# ---- frames ---------------------------------------------------------------
def conjugate_f(b, ctrl, x):
    for p in range(X // 2):
        lo, hi = x[2 * p], x[2 * p + 1]
        b.cx(hi, lo)
        b.ccx(ctrl, lo, hi)
        b.cx(hi, lo)
        b.ccz(ctrl, lo, hi)


# ---- minimal-rank conjugation forms ---------------------------------------
# A controlled quadratic phase form needs exactly rank(B)/2 CCZs behind
# a free GF(2) congruence (CX net): the symplectic reduction below is
# proved function-identical at synthesis time — the decomposition is
# asserted against the target form on a determining set (all e_k and
# e_j+e_k fix a quadratic), and the net is asserted to realize the dual
# basis matrix exactly.


def prefix_mat(n):
    """u_k = x_0 ^ ... ^ x_k as row bitmasks over x."""
    return [(1 << (k + 1)) - 1 for k in range(n)]


def q_eval(edges, lin, T, x):
    """q(x) = sum_{(a,b) in edges} u_a u_b + sum_lin u_c with u = T x."""
    u = [parity(T[r] & x) for r in range(len(T))]
    v = 0
    for a, b in edges:
        v ^= u[a] & u[b]
    for c in lin:
        v ^= u[c]
    return v


def symplectic_pairs(edges, lin, T, n):
    """-> (dual pair functionals, ell bitmask, dual matrix rows) with
    q(x) = sum (alpha_i.x)(beta_i.x) ^ (ell.x), verified on the
    determining set.
    """
    B2 = [0] * n
    qe = [q_eval(edges, lin, T, 1 << k) for k in range(n)]
    for j in range(n):
        for k in range(j + 1, n):
            if q_eval(edges, lin, T, (1 << j) | (1 << k)) ^ qe[j] ^ qe[k]:
                B2[j] |= 1 << k
                B2[k] |= 1 << j

    def bform(u, v):
        acc = 0
        uu = u
        while uu:
            lb = uu & -uu
            j = lb.bit_length() - 1
            uu ^= lb
            acc ^= parity(B2[j] & v)
        return acc

    basis = [1 << k for k in range(n)]
    pairs = []
    while True:
        hit = None
        for i, v in enumerate(basis):
            for j, w in enumerate(basis):
                if i != j and bform(v, w):
                    hit = (i, j)
                    break
            if hit:
                break
        if hit is None:
            break
        i, j = hit
        a, b2 = basis[i], basis[j]
        pairs.append((a, b2))
        rest = []
        for k, u in enumerate(basis):
            if k in (i, j):
                continue
            u ^= b2 if bform(u, a) else 0
            u ^= a if bform(u, b2) else 0
            rest.append(u)
        basis = rest
    radical = basis
    # dual functionals: x = sum_r xi_r vecs[r]; xi_r = row r of V_col^-1
    vecs = [w for p in pairs for w in p] + radical
    assert len(vecs) == n
    vcol = [0] * n
    for r, v in enumerate(vecs):
        for j in range(n):
            if (v >> j) & 1:
                vcol[j] |= 1 << r
    inv = f2_matinv(vcol, n)
    duals = [(inv[2 * i], inv[2 * i + 1]) for i in range(len(pairs))]
    ell = 0
    for k in range(n):
        v = qe[k]
        for a, b2 in duals:
            v ^= parity(a & (1 << k)) & parity(b2 & (1 << k))
        if v:
            ell |= 1 << k

    def decomp(x):
        v = parity(ell & x)
        for a, b2 in duals:
            v ^= parity(a & x) & parity(b2 & x)
        return v

    for j in range(n):
        assert decomp(1 << j) == qe[j]
        for k in range(j + 1, n):
            x = (1 << j) | (1 << k)
            assert decomp(x) == q_eval(edges, lin, T, x)
    return duals, ell, inv


def synth_net(M, n):
    """CX ops [(src,dst)] turning wire functionals I into rows of M."""
    A2 = list(M)
    ops = []
    for col in range(n):
        piv = next(r for r in range(col, n) if (A2[r] >> col) & 1)
        if piv != col:
            A2[col] ^= A2[piv]
            ops.append((piv, col))
            A2[piv] ^= A2[col]
            ops.append((col, piv))
            A2[col] ^= A2[piv]
            ops.append((piv, col))
        for r in range(n):
            if r != col and (A2[r] >> col) & 1:
                A2[r] ^= A2[col]
                ops.append((col, r))
    assert A2 == [1 << k for k in range(n)]
    net = [(s, d) for s, d in reversed(ops)]
    S2 = [1 << k for k in range(n)]
    for s, d in net:
        S2[d] ^= S2[s]
    assert S2 == list(M), "net does not realize M"
    return net


def solve_transpose(M, ell, n):
    """c with sum_j c_j M[j] = ell."""
    A2 = list(M)
    c_track = [1 << j for j in range(n)]
    piv_cols = []
    for col in range(n):
        piv = None
        for r in range(len(piv_cols), n):
            if (A2[r] >> col) & 1:
                piv = r
                break
        if piv is None:
            continue
        k = len(piv_cols)
        A2[k], A2[piv] = A2[piv], A2[k]
        c_track[k], c_track[piv] = c_track[piv], c_track[k]
        for r in range(n):
            if r != k and (A2[r] >> col) & 1:
                A2[r] ^= A2[k]
                c_track[r] ^= c_track[k]
        piv_cols.append(col)
    res = ell
    combo = 0
    for k, col in enumerate(piv_cols):
        if (res >> col) & 1:
            res ^= A2[k]
            combo ^= c_track[k]
    assert res == 0
    acc = 0
    for j in range(n):
        if (combo >> j) & 1:
            acc ^= M[j]
    assert acc == ell
    return combo


def min_form_net(edges, lin, T, n):
    """-> (net, ccz_pairs, cz_wires): after net, wires 2i/2i+1 hold the
    dual pair functionals; q = sum of their products plus ell on CZ
    wires.  End-to-end asserted on the determining set.
    """
    duals, ell, inv = symplectic_pairs(edges, lin, T, n)
    M = inv
    c = solve_transpose(M, ell, n)
    net = synth_net(M, n)
    ccz = [(2 * i, 2 * i + 1) for i in range(len(duals))]
    cz = [j for j in range(n) if (c >> j) & 1]
    S2 = [1 << k for k in range(n)]
    for s, d in net:
        S2[d] ^= S2[s]
    for j in range(n):
        for k in range(j, n):
            x = (1 << j) | (1 << k)
            v = 0
            for a, b2 in ccz:
                v ^= parity(S2[a] & x) & parity(S2[b2] & x)
            for w in cz:
                v ^= parity(S2[w] & x)
            assert v == q_eval(edges, lin, T, x), (j, k)
    return net, ccz, cz


# Minimal-rank realizations of the phase forms: rank(B)/2 CCZs after a
# free congruence net (proved function-identical inside min_form_net).
_MIN_FORM = {g: min_form_net(PERM_MACH[g][3], PERM_MACH[g][4], prefix_mat(X), X) for g in PERM_MACH}
_C_FORM = min_form_net([(j - 1, j) for j in range(1, 20)], list(range(19)), prefix_mat(20), 20)


def _run_form(b, ctrl, wires, form):
    net, ccz, cz = form
    for s, d in net:
        b.cx(wires[s], wires[d])
    for a, bb in ccz:
        b.ccz(ctrl, wires[a], wires[bb])
    for c in cz:
        b.cz(ctrl, wires[c])
    for s, d in reversed(net):
        b.cx(wires[s], wires[d])


def conjugate_c(b, ctrl, x):
    """C^ctrl: per spin sector, e2(y) at its bilinear rank — 10 CCZs
    behind a free congruence net (19 in prefix coordinates).
    """
    for par in (0, 1):
        sec = [x[2 * p + par] for p in range(X // 2)]
        _run_form(b, ctrl, sec, _C_FORM)


def _pf_perm(b, ctrl, x, perm):
    done = set()
    for k in range(X):
        j = perm[k]
        if k in done or j == k:
            continue
        done.add(j)
        b.cx(x[j], x[k])
        b.ccx(ctrl, x[k], x[j])
        b.cx(x[j], x[k])


def _gauge_layer(b, ctrl, x, g, rev=False):
    gauge_c = sign_tables()[0]

    def xs(a):
        j = 0
        while a:
            if a & 1:
                b.cx(ctrl, x[j])
            a >>= 1
            j += 1

    def zs(c):
        j = 0
        while c:
            if c & 1:
                b.cz(ctrl, x[j])
            c >>= 1
            j += 1

    if rev:
        zs(gauge_c.get(g, 0))
        xs(GAUGE_A.get(g, 0))
    else:
        xs(GAUGE_A.get(g, 0))
        zs(gauge_c.get(g, 0))


def conjugate_open(b, ctrl, x, g):
    if g == 0:
        conjugate_f(b, ctrl, x)
        return
    if g == 1:
        conjugate_c(b, ctrl, x)
        return
    perm = PERM_MACH[g][0]
    _pf_perm(b, ctrl, x, perm)
    _run_form(b, ctrl, x, _MIN_FORM[g])
    _gauge_layer(b, ctrl, x, g)  # inner gauge layer (free CX/CZ)


def conjugate_close(b, ctrl, x, g):
    if g == 0:
        conjugate_f(b, ctrl, x)
        return
    if g == 1:
        conjugate_c(b, ctrl, x)
        return
    perm = PERM_MACH[g][0]
    _gauge_layer(b, ctrl, x, g, rev=True)
    _run_form(b, ctrl, x, _MIN_FORM[g])
    _pf_perm(b, ctrl, x, perm)


def xor_gate_flags(b, sel_bits, flags):
    """XOR every gate interval's [lo <= chosen < hi] AND chosen_p onto
    its frame flag, trie-shared: the half-plane split turns the 14
    intervals into 28 [chosen < K] uses drawing from the DISTINCT
    region-boundary constants K (lo = 0 endpoints vanish).  A borrow
    stage at bit j depends only on (K's lowest set bit, K mod 2^(j+1)),
    so constants sharing their low set bit form a trie over the higher
    bits: the DFS over the sorted constants builds each stage ONCE and
    unwinds only to the fork point — one AND + one CX per
    (K, branch-bit) use, only one chain ever alive.  Contributions are
    CXes onto flag wires, so they commute freely; the identical call
    at teardown XORs every one back out.
    """
    uses = {}
    for g in PRIORITY:
        for p, lo_k, hi_k in GEN_IVS[g]:
            uses.setdefault(hi_k, {}).setdefault(p, []).append(g)
            if lo_k:
                uses.setdefault(lo_k, {}).setdefault(p, []).append(g)
    groups = {}
    for K in uses:
        groups.setdefault((K & -K).bit_length() - 1, []).append(K)
    for lo_bit, ks in sorted(groups.items()):
        # trie DFS order: low-bit-first lexicographic over bits above lo_bit
        ks.sort(key=lambda K: tuple((K >> j) & 1 for j in range(lo_bit + 1, A)))
        stack = []  # (bit, wire, in1, in2, pol, xed, K mod 2^(bit+1))

        def unwind_to(depth):
            while len(stack) > depth:
                _j, t, c1, c2, pol, xed, _val = stack.pop()
                b.release("blk", [t])
                if xed:
                    b.x(t)
                b.and_release(t, c1, c2, pol=pol)

        for K in ks:
            keep = 0
            for d, st in enumerate(stack):
                if K & ((1 << (st[0] + 1)) - 1) != st[6]:
                    break
                keep = d + 1
            unwind_to(keep)
            if stack:
                cur, cpol, start = stack[-1][1], True, stack[-1][0] + 1
            else:
                cur, cpol, start = sel_bits[lo_bit], False, lo_bit + 1
            for j in range(start, A):
                if (K >> j) & 1:
                    pol = (True, not cpol)
                    t = b.and_compute(sel_bits[j], cur, pol=pol)
                    b.x(t)
                    xed = True
                else:
                    pol = (False, cpol)
                    t = b.and_compute(sel_bits[j], cur, pol=pol)
                    xed = False
                b.claim("blk", [t])
                stack.append((j, t, sel_bits[j], cur, pol, xed, K & ((1 << (j + 1)) - 1)))
                cur, cpol = t, True
            assert cpol
            for p, gs in sorted(uses[K].items()):
                tc = b.and_compute(sel_bits[p], cur)
                for g in gs:
                    b.cx(tc, flags[g])
                b.and_release(tc, sel_bits[p], cur)
        unwind_to(0)


# ---- Babbush/Sanders: the frame-block SELECT descent ---------------------
def select_walk(b, bits_msb, enter, leaf):
    """Content-pruned descent over the block partition.  At a
    one-sided node (right half entirely >= L) the parent control
    already implies the branch on every graded shot, so the walk
    descends with it unchanged — no narrowing AND (the spare copies
    land on sel values no shot reaches).  ``enter(lo, span, ctrl)``
    hooks every controlled node entry — the walk-node monomial channel
    emits there; ``leaf(lo, ctrl)`` fires per block.
    """

    def rec(bits, lo, span, ctrl):
        if lo >= L:
            return
        if ctrl is not None:
            enter(lo, span, ctrl)
            if span == 1 or SPAN_AT.get(lo) == span:
                leaf(lo, ctrl)
                return
        hb = bits[0]
        half = span >> 1
        if ctrl is None:
            b.x(hb)
            rec(bits[1:], lo, half, hb)
            b.x(hb)
            rec(bits[1:], lo + half, half, hb)
        elif lo + half < L:
            anc = b.and_compute(ctrl, hb)
            rec(bits[1:], lo + half, half, anc)
            b.cx(ctrl, anc)
            rec(bits[1:], lo, half, anc)
            b.cx(ctrl, anc)
            b.and_release(anc, ctrl, hb)
        else:
            rec(bits[1:], lo, half, ctrl)  # one-sided node: no narrowing

    rec(bits_msb, 0, 1 << A, None)


def select_pauli(b, sel_bits, x):
    """One content-pruned descent over the 3,221 frame blocks; seven
    flag wires XOR-accumulate the 14 interval gates; frames nest in
    the fixed application order (F innermost); per-block sign
    polynomials ride free channels plus one CCZ per surviving
    monomial (shared prefix AND per degree-3 group), the seven
    node-hosted monomials emitting once at their walk nodes.
    """
    _gauge_c, lin_signs, mono_signs, node_monos = sign_tables()

    def apply_row(ctrl, iv):
        m = MASKS[iv]
        for j in range(X):  # Z-part first: CZ sees the INPUT x values
            if (m >> (X + j)) & 1:
                b.cz(ctrl, x[j])
        xor_word(b, ctrl, m & XMASK, x)  # then the X-part flips
        if SIGN[iv]:  # constant sign (-1)^(theta ^ ny2)
            b.z(ctrl)
        for i in lin_signs.get(iv, ()):
            b.cz(ctrl, sel_bits[i])
        pref = pref_bit = None
        for mn in mono_signs.get(iv, ()):
            if len(mn) == 2:
                b.ccz(ctrl, sel_bits[mn[0]], sel_bits[mn[1]])
            else:  # degree >= 3: prefix AND per group
                if pref_bit != mn[0]:
                    if pref is not None:
                        b.and_release(pref, ctrl, sel_bits[pref_bit])
                    pref = b.and_compute(ctrl, sel_bits[mn[0]])
                    pref_bit = mn[0]
                if len(mn) == 3:
                    b.ccz(pref, sel_bits[mn[1]], sel_bits[mn[2]])
                else:
                    t2 = b.and_compute(pref, sel_bits[mn[1]])
                    b.ccz(t2, sel_bits[mn[2]], sel_bits[mn[3]])
                    b.and_release(t2, pref, sel_bits[mn[1]])
        if pref is not None:
            b.and_release(pref, ctrl, sel_bits[pref_bit])

    hosted = set()

    def node_enter(lo, span, ctrl):
        monos = node_monos.get((lo, span))
        if not monos:
            return
        for mn in monos:
            if len(mn) == 2:
                b.ccz(ctrl, sel_bits[mn[0]], sel_bits[mn[1]])
            else:
                t2 = b.and_compute(ctrl, sel_bits[mn[0]])
                b.ccz(t2, sel_bits[mn[1]], sel_bits[mn[2]])
                b.and_release(t2, ctrl, sel_bits[mn[0]])
        hosted.add((lo, span))

    with b.phase("select-blk"):
        flags = {}
        for g in PRIORITY:
            if not GEN_IVS[g]:
                continue
            fq = b.qubit()
            b.claim("flag", [fq])
            flags[g] = fq
        xor_gate_flags(b, sel_bits, flags)

    with b.phase("select-conj"):
        for g in reversed(PRIORITY):  # outermost first, F innermost last
            if g in flags:
                conjugate_open(b, flags[g], x, g)
    with b.phase("select-pauli"):
        select_walk(b, list(reversed(sel_bits)), node_enter, lambda lo, ctrl: apply_row(ctrl, lo))
        assert hosted == set(node_monos), "node-hosted monomial span missed"
    with b.phase("select-unconj"):
        for g in PRIORITY:  # innermost first
            if g in flags:
                conjugate_close(b, flags[g], x, g)

    with b.phase("select-unblk"):
        xor_gate_flags(b, sel_bits, flags)
        for g in reversed(PRIORITY):
            if g not in flags:
                continue
            b.release_recycle("flag", [flags[g]])


def build():
    b = Builder()
    regs = declare_registers(b, [("i", A), ("sigma", S), ("x", X)])
    idx, sigma, x = regs["i"], regs["sigma"], regs["x"]
    for name in ("i", "sigma", "x"):
        b.claim(name, regs[name])

    # ONE walk carries BOTH payloads: keep bits 0-14 into the 15-wire
    # word bank plus one carrier for bit 15, alt into the 16-wire
    # selection scratch (alt before the swap, sel during SELECT).
    word = b.qubits(B)
    b.claim("word", word)
    kc = b.qubit()
    b.claim("word", [kc])
    keep_carriers = list(word) + [kc]
    sel = b.qubits(A)
    b.claim("sel", sel)

    def leaf(lo, ctrl):
        xor_word(b, ctrl, KEEPW[lo], keep_carriers)
        xor_word(b, ctrl, ALTW[lo], sel)

    with b.phase("lookup"):
        lookup_walk(b, list(idx[::-1]), leaf)

    # f = [sigma < keep] on one scratch wire; the keep bank stays live
    # through SELECT so the identical comparator pass can erase f after
    # the selection is unwound
    with b.phase("cmp"):
        f = b.qubit()
        b.claim("carry", [f])
        compare_into(b, sigma, keep_carriers, f)

    # the selection swap: sel <-> i under f — the index wires carry the
    # unselected branch through SELECT and the identical swap restores
    # them after it
    with b.phase("cmp/select"):
        for j in range(A):
            b.cswap(f, sel[j], idx[j])

    select_pauli(b, sel, x)

    with b.phase("cmp/unselect"):
        for j in range(A):
            b.cswap(f, sel[j], idx[j])
    with b.phase("cmp/rev"):
        compare_into(b, sigma, keep_carriers, f)
        b.release_recycle("carry", [f])

    b.release("word", word)
    b.release("word", [kc])
    with b.phase("unlookup/measure-keep"):
        m_keep = erase_wires(b, keep_carriers)
        for q in keep_carriers:
            b.recycle(q)
    b.release("sel", sel)
    with b.phase("unlookup/measure-alt"):
        m_alt = erase_wires(b, sel)
        for q in sel:
            b.recycle(q)

    b2_fixup(b, idx, m_keep, m_alt)
    return b, regs


if __name__ == "__main__":
    b, _ = build()
    assert_metrics(b, __file__)
    emit_and_report(b)
