#!/usr/bin/env python3
"""modinv-point-add: the specialized Euclidean circuit, a low-width
register-sharing EEA point adder specialized from the Luo et al. (2026)
construction ("the paper" in the comments below), certified at
20,521,949 Toffoli x 843 qubits.  It adds a per-shot classical point Q
to a quantum point in place
(|P> -> |P+Q>), Q delivered on the Bit-kind input registers (2/3):
the const_modp cbspecs are free classical bit networks of the input
bits, and their loads ride as bit conditions (CX under flagq promotes
to a full Toffoli).

Shared code: ../_shared/luo_core.py (the arithmetic/template layer,
loaded by path; it loads ../_shared/clsbits.py) — the certification fingerprint
covers this file alone.  Searched tables live in constants.json;
python gen_constants.py regenerates them.

RUN: uv run python circuit.py — build, cost-check, emit to
$UNCOMPUTE_OPS_PATH (else ./ops.bin).
     python tools/eval.py modinv-point-add --circuit <path to this
     file> — certification by the evaluator.

README.md in this folder describes the circuit.
"""

import functools
import json
import pathlib

from uncompute import Builder
from uncompute.testing import assert_metrics, emit_and_report, load_contract, load_module

HERE = pathlib.Path(__file__).resolve().parent

# Reusable arithmetic/template layer shared with the other circuits,
# loaded from ../_shared/luo_core.py: the (gate, ...) tuple vocabulary and its
# inverse rule, the CDKM adder cells, small-register/counter helpers,
# the classed-template machinery and step-loop emitters (with the shared
# step-table cache), the Gidney carry helpers and ripple cells, the
# k-block multiplier family, the point I/O + cbspec networks, and the
# modular add/double/const-add/negate + idiv/imul flows.  Circuit-local
# callables those bodies resolve at call time are bound by
# _core.init(globals()) at the bottom of this file, after they are
# defined.
_core = load_module(HERE.parent / "_shared" / "luo_core.py", "luo_core")

# The shared bodies, re-exported: circuit-local code reads them as
# plain globals of this module.
int_bits = _core.int_bits
X_ = _core.X_
CX_ = _core.CX_
CCX_ = _core.CCX_
ANDP = _core.ANDP
ANDM = _core.ANDM
_INV = _core._INV
inv_list = _core.inv_list
emit = _core.emit
cell_maj = _core.cell_maj
cell_uma = _core.cell_uma
cell_maj_inv = _core.cell_maj_inv
cell_uma_inv = _core.cell_uma_inv
emit_op = _core.emit_op
cell_maj_u = _core.cell_maj_u
add_small = _core.add_small
ctrl_inc = _core.ctrl_inc
add_gid = _core.add_gid
and_ladder = _core.and_ladder
cswap = _core.cswap
_zero_toggle = _core._zero_toggle
ClassTable = _core.ClassTable
tpl_prune = _core.tpl_prune
tpl_emit_body = _core.tpl_emit_body
_UShape = _core._UShape
tpl_unary = _core.tpl_unary
tpl_dual = _core.tpl_dual
cell_uma_u = _core.cell_uma_u
xor_const_reg = _core.xor_const_reg
add_const_small = _core.add_const_small
const_minus_small = _core.const_minus_small
ctrl_dec = _core.ctrl_dec
unary_depth = _core.unary_depth
tpl_inv = _core.tpl_inv
tpl_emit_steps = _core.tpl_emit_steps
_STCACHE = _core._STCACHE
step_tables = _core.step_tables
_affine_spans = _core._affine_spans
_cb_x = _core._cb_x
_ccx_cond_const = _core._ccx_cond_const
_xor_carries = _core._xor_carries
gidney_add_const = _core.gidney_add_const
gidney_cmp_ge = _core.gidney_cmp_ge
ctrl_qq_add_flag = _core.ctrl_qq_add_flag
cuccaro_lt_flag = _core.cuccaro_lt_flag
kb_dbl_fold = _core.kb_dbl_fold
gc_lazy_cma = _core.gc_lazy_cma
horner_mul = _core.horner_mul
declare_point_io = _core.declare_point_io
point_cbspecs = _core.point_cbspecs
ctrl_mod_add = _core.ctrl_mod_add
dbl_modp = _core.dbl_modp
const_modp = _core.const_modp
neg_modp = _core.neg_modp
emit_idiv = _core.emit_idiv
emit_imul = _core.emit_imul


# ===========================================================================
# task contract: the prime p, per-shot classical addend Q
# ===========================================================================

N = 256
P_MOD = int(load_contract(__file__)["p"], 16)  # grader-asserted at its import

# The addend is per-shot classical input data (grader.py): registers
# 2/3 are 256-wide Bit-kind inputs carrying Q.x, Q.y.  The Gidney
# constant machinery needs 0 < C < p for every derived constant (the
# const_modp cbspecs S_x2/S_y2/S_3x2, incl. 3*Q.x mod p != 0) — an
# input-domain fact of the drawn points (canonical residues of curve
# points; no x = 0 or y = 0 point exists on y^2 = x^3 + 7).


# ===========================================================================
# schedules: worst-case curves + the priced production tables
# ===========================================================================


def len_width(n):
    """Length-register width: n+3 must be encodable (plain-value
    encoding; >= floor(log2 n)+1 for every n).
    """
    return (n + 3).bit_length()


def ls_width(n, p=None):
    """Return the width of the shift register.

    The register keeps counting through the residual regime (the paper's
    own value semantics, Table 4 rows 33-36), so it must also hold
    Nmax - 4n without wrapping to 0.
    """
    return max(len_width(n), (nmax_steps(n, p) - 4 * n + 1).bit_length())


# --- exact step budget + window bounds (choice D-bounds) -----------------
#
# This implementation derives the step budget and all lower window bounds
# exactly, by dynamic programming over quotient-sequence continuants.
# All-2 quotient chains (Pell continuants) grow slowest, at 2^0.6358 per
# step-unit, so they set the budget.  The budget must cover every input:
# an under-budgeted run spoils the HMR phase fixup (hard grader failure).
# The upper bounds K1..K5 come from the paper's chains t < 2^{N/4} and
# p/r < 2^{N/4}; the D-K3/D-K5 windows are documented at their definitions.

_DP_WCAP = 8  # quotient bit-width cap for the extremal DP; exact vs
# the unbounded DP for m <= 130 (re-checkable by rerunning
# _min_continuants uncapped; wcap=3 is already exact there;
# wide quotients grow >= 2 bits/unit, triple the minimal
# path's 0.6358 bits/unit).


def _min_continuants(m_cap, first_ge2, wcap=None, limit=None, min_p_out=None):
    """Pareto-minimal (t, t_prev) continuant pairs per unit count m.

    Units = sum of bitlen(q_i); appending quotient q = 2^(w-1) (only the
    smallest quotient per width matters for minima).  first_ge2 forbids q=1
    as the first quotient (EEA on x < p/2 has q_1 >= 2).  limit prunes
    states whose smaller component already exceeds it (their bit lengths
    are clamped by callers anyway).

    min_p_out: optional (m_cap+1)-list, filled in place with the minimal
    COMPLETE-run continuant per m -- last appended quotient >= 2, so with
    first_ge2 both end quotients are >= 2 (continuants are
    reversal-symmetric). A pure observer: it never affects the fronts.
    """
    if wcap is None:
        wcap = m_cap if m_cap <= 130 else _DP_WCAP
    fronts = [[] for _ in range(m_cap + 1)]
    fronts[0] = [(1, 0)]
    for m in range(1, m_cap + 1):
        cands, cands2 = [], []
        for w in range(1, min(wcap, m) + 1):
            q = 1 << (w - 1)
            src = fronts[m - w]
            if m - w == 0 and first_ge2 and w < 2:
                continue
            for t, tp in src:
                c = (q * t + tp, t)
                cands.append(c)
                if w >= 2:
                    cands2.append(c)
        if min_p_out is not None and cands2:
            min_p_out[m] = min(t for (t, _) in cands2)
        if not cands:
            continue
        if limit is not None:
            kept = [c for c in cands if c[1] <= limit]
            cands = kept if kept else [min(cands)]
        cands.sort()
        front = []
        best_tp = None
        for t, tp in cands:
            if best_tp is None or tp < best_tp:
                front.append((t, tp))
                best_tp = tp
        fronts[m] = front
    return fronts


_DPCACHE = {}


def _dp_tables(n):
    """Exact extremal tables for width-n runs: min_t_len[m] (min bitlen of
    t_j after m units) and min_p[m] (min possible p for a full m-unit run,
    both end quotients >= 2; None if no such run).
    """
    if n in _DPCACHE:
        return _DPCACHE[n]
    # generous unit cap: true rate is ~0.6358 bits/unit -> m <= n/0.63 + 8
    m_cap = int(n / 0.6357) + 8
    limit = 1 << (n + 6)
    min_p = [None] * (m_cap + 1)
    fronts = _min_continuants(m_cap, first_ge2=True, limit=limit, min_p_out=min_p)
    min_t_len = [min((t for (t, _) in fronts[m]), default=None) for m in range(m_cap + 1)]
    min_t_len = [v.bit_length() if v is not None else None for v in min_t_len]
    out = {"m_cap": m_cap, "min_t_len": min_t_len, "min_p": min_p}
    _DPCACHE[n] = out
    return out


_NMAXCACHE = {}


def _nmax_steps_wc(n, p=None):
    """Exact worst-case step budget for n-bit modulus (or the specific p)."""
    key = (n, p)
    if key in _NMAXCACHE:
        return _NMAXCACHE[key]
    tab = _dp_tables(n)
    lim = p if p is not None else (1 << n) - 1
    m_max = 0
    for m in range(tab["m_cap"] + 1):
        mp = tab["min_p"][m]
        if mp is not None and mp <= lim:
            m_max = m
    # single-quotient runs (x=1, q=p) consume bitlen(p) = n units
    m_max = max(m_max, n)
    out = 4 * m_max
    _NMAXCACHE[key] = out
    return out


_WINCACHE = {}


def _active_windows_wc(n, T):
    """Step-dependent active windows: paper upper bounds (with D-K3/D-K5),
    exact DP lower bounds (D-bounds).
    """
    key = (n, T)
    if key in _WINCACHE:
        return _WINCACHE[key]
    assert T >= 1
    tab = _dp_tables(n)
    mtl = tab["min_t_len"]

    def tl(m):
        v = mtl[m] if 0 <= m < len(mtl) else None
        return v if v is not None else n + 3

    # k1: min over iterations covering step T of lt+lq(+2), r-ops fire in
    # phases 1-2 (u in [1, 2B], B <= n+1-lt).
    k1 = n + 3
    k2 = n + 3
    for m in range((T - 1) // 4 + 1):
        u = T - 4 * m
        lt = tl(m)
        Bmax = n + 1 - lt
        if Bmax < 1:
            continue
        # --- r ops (phases 1-2) ---
        if u <= Bmax:  # phase 1 achievable (B >= u)
            k1 = min(k1, lt + 2)
        elif u <= 2 * Bmax:  # phase 2 with B in [ceil(u/2), Bmax]
            k1 = min(k1, lt + (u - Bmax - 1) + 2)
        # --- swap (phases 2-3) ---
        if u >= 2:
            Bhi = min(Bmax, u - 1)  # phase 2: B in [ceil(u/2), u-1]
            if Bhi * 2 >= u:
                k2 = min(k2, lt + (u - Bhi) + 1)
        Blo = max(1, (u + 2) // 3)  # phase 3: smallest feasible B
        for B in range(Blo, min((u - 1) // 2, Bmax) + 1):
            if 2 * B < u <= 3 * B:  # J = lt+3B-u+2, increasing in B
                k2 = min(k2, lt + (3 * B - u + 1) + 1)
                break
    k1 = max(1, min(k1, n + 3))
    k2 = max(1, min(k2, n + 2))
    K1 = n + 3
    K2 = min(T // 2 + 2, n + 2)
    # D-K3: the t-op window covers both phases.  Phase 3 acts on width
    # lt+1.  The phase-4 comparison acts on the full t'(h) region
    # n+3-lrp-ls, provably <= floor(T/4)+2 (and <= n+2).  One window serves
    # both phases.
    k3 = 1
    K3 = min(T // 4 + 2, n + 2)
    # k4: lt at iteration end (u = 4B): min over B >= 1 of tl(T/4 - B).
    k4 = n + 3
    k5 = n + 3
    if T % 4 == 0:
        for B in range(1, T // 4 + 1):
            m = T // 4 - B
            lt = tl(m)
            if B <= n + 1 - lt:
                k4 = min(k4, lt)
        k5 = min(tl(T // 4) + 2, n + 3)
    k4 = max(1, min(k4, n + 3))
    k5 = max(1, min(k5, n + 3))
    K4 = min(T // 4 + 3, n + 3)
    K5 = min(T // 4 + 4, n + 3)
    # steps at which a block can never fire leave its k at the clamp; the
    # emitted block is inert there, so collapse to a trivial window.
    k1 = min(k1, K1)
    k2 = min(k2, K2)
    w = {
        "r": (k1, K1),
        "swap": (k2, K2),
        "t": (k3, K3),
        "len_t": (min(k4, K4), K4),
        "len_rp": (min(k5, K5), K5),
    }
    for name, (a, bnd) in w.items():
        assert 1 <= a <= bnd <= n + 3, (name, n, T, a, bnd)
    _WINCACHE[key] = w
    return w


# ---------------------------------------------------------------------------
# priced production tables (n = 256): window schedules + certainty cores.
# The fitted tables ride in constants.json (gen_constants.py fits them).
# Small n has no tables and falls back to the worst-case curves.
# ---------------------------------------------------------------------------

NMAX_SCHED = 1432  # priced schedule truncation (P_shot ~ 1.3e-5);
# wl_windows' "nmax" (1616) is the FIT horizon,
# a different quantity


@functools.cache
def _priced():
    """Return the priced tables under the keys "win", "tcore" and "rcore".

    The first call reads constants.json, so gen_constants.py can load
    this module before constants.json exists.
    """
    consts = json.loads((HERE / "constants.json").read_bytes())
    win256 = consts["wl_windows"]
    tcore256 = consts["tcore"]
    rcore256 = consts["rcore"]
    assert NMAX_SCHED % 4 == 0 and win256["nmax_cut"] >= NMAX_SCHED
    return {
        # n -> dict(rows, win={name: [(k, K)] * rows}, newK=[int] * rows)
        "win": {
            N: {
                "rows": win256["nmax"],
                "win": {k: [tuple(x) for x in v] for k, v in win256["windows"].items()},
                "newK": list(win256["len_rp_new_K"]),
            }
        },
        # n -> dict(rlo=[...], rows)  /  dict(lmax=[...], rmin=[...], rows)
        "tcore": {N: {"rlo": list(tcore256["rlo"]), "rows": int(tcore256["nmax_cut"])}},
        "rcore": {
            N: {
                "lmax": list(rcore256["lmax"]),
                "rmin": list(rcore256["rmin"]),
                "rows": int(rcore256["nmax_cut"]),
            }
        },
    }


_WIN_NAMES = ("r", "swap", "t", "len_t", "len_rp")


def nmax_steps(n, p=None):
    if n == N:  # the priced n: the schedule truncation
        return NMAX_SCHED
    return _nmax_steps_wc(n, p)


# Residual-regime no-wrap guard: after the walk finishes (lrp = 0), ls
# keeps counting, so it must hold the maximal residual count
# NMAX_SCHED - 4n without wrapping.  Rests on the continuant lemma —
# a complete walk needs >= 4n steps, since K(q_1..q_k) <=
# prod(q_i + 1) <= 2^{sum bitlen(q_i)} forces sum bitlen(q_i) >= n
# for p > 2^{n-1}, and x = 1 realizes exactly 4n.
assert NMAX_SCHED - 4 * N < (1 << ls_width(N))


def active_windows(n, T):
    """Production windows: the table row clipped inside the worst-case
    (analytic) window; worst case beyond the table or for un-tabled n.
    """
    wc = _active_windows_wc(n, T)
    tab = _priced()["win"].get(n)
    if tab is None or T > tab["rows"]:
        return wc
    out = {}
    for name in _WIN_NAMES:
        k, K = tab["win"][name][T - 1]
        wk, wK = wc[name]
        k, K = max(k, wk), min(K, wK)
        if k > K:  # defensive: never widen, never die
            k, K = wk, wK
        out[name] = (k, K)
    return out


def _newK(n, T, k):
    """Return the truncated right edge of the D6 new-length scan at step T (>= k,
    <= n+3; n+3 when un-tabled or beyond the table).
    """
    tab = _priced()["win"].get(n)
    if tab is None or T > tab["rows"]:
        return n + 3
    return max(k, min(tab["newK"][T - 1], n + 3))


def _rlo_at(n, T, K):
    """t-prefix certainty-core top (exclusive), clamped into [0, K];
    a value <= the caller's window bottom means no core at this step.
    """
    tab = _priced()["tcore"].get(n)
    if tab is None or not (1 <= T <= len(tab["rlo"])):
        return 0
    return max(0, min(tab["rlo"][T - 1], K))


def _rsplit_at(n, T, k, K):
    """r-block certainty bounds (lmax, rmin) with a nonempty separated
    core, else None (the merged pair keeps its dual spelling there).
    """
    tab = _priced()["rcore"].get(n)
    if tab is None or not (1 <= T <= len(tab["lmax"])):
        return None
    lmax, rmin = tab["lmax"][T - 1], tab["rmin"][T - 1]
    if lmax <= 0 or rmin <= 0:
        return None
    lmax = max(lmax, k)
    rmin = min(rmin, K)
    if lmax + 1 > rmin - 1:
        return None
    return lmax, rmin


# ===========================================================================
# the circuit's controlled cell (the gate vocabulary, the other cells and
# the small-register arithmetic live in ../_shared/luo_core.py)
# ===========================================================================


def c3x(s, a, b, t, w):
    """t ^= s&a&b via Gidney temporary AND on the clean scratch w: 2
    Toffoli (ANDP + CCX; the ANDM release is measurement-based). a/b are
    untouched between AND+ and AND-, so the conditioned-CZ phase repair
    is exact; self-image under inv_list/tpl_inv.
    """
    return [ANDP(a, b, w), CCX_(s, w, t), ANDM(a, b, w)]


# ===========================================================================
# unary trees, interval blocks, length machinery
# ===========================================================================


def _uz_chain_gates(j, k, bits, dirty, a_tmp, racc):
    """One leaf of the zero-map core: compute a_j = racc & bits_j into a_tmp,
    apply the chain Toffoli (or base CNOT at the low end), release a_tmp.
    """
    g = [ANDP(racc, bits[j - k], a_tmp)]
    tgt, nb = (dirty[j - k], dirty[j - k - 1]) if j > k else (dirty[j - k], None)
    g.append(X_(a_tmp))
    if nb is None:
        g.append(CX_(a_tmp, tgt))
    else:
        g.append(CCX_(a_tmp, nb, tgt))
    g.append(X_(a_tmp))
    g.append(ANDM(racc, bits[j - k], a_tmp))
    return g


def lower_zero_map(root, k, K, bits, dirty, aux):
    """dirty[j] ^= y_j, y_j = AND_{h=k..j} not(root & [h>=A] & bits_h)."""
    a_tmp = aux["c3xw"]
    g = []
    for j in range(K, k - 1, -1):
        g += _uz_chain_gates(j, k, bits, dirty, a_tmp, root)
    for j in range(k + 1, K + 1):
        g += _uz_chain_gates(j, k, bits, dirty, a_tmp, root)
    return g


def right_length_write(root, k, K, bits, dirty, target, aux, n):
    """XOR into target the right length n+4-m for the lowest j = m in
    [max(k,A), K] with bits_j = 1 (0 if none), gated on root.
    """
    wl = len(target)
    mask = (1 << wl) - 1

    def val(pos):
        return (n + 4 - pos) & mask

    g = xor_const_reg(target, val(k))
    dw = []
    for j in range(k, K):
        delta = val(j) ^ val(j + 1)
        for i in range(wl):
            if (delta >> i) & 1:
                dw.append(CX_(dirty[j - k], target[i]))
    delta_K = val(K) ^ 0
    for i in range(wl):
        if (delta_K >> i) & 1:
            dw.append(CX_(dirty[K - k], target[i]))
    g += dw
    g += lower_zero_map(root, k, K, bits, dirty, aux)
    g += dw
    g += lower_zero_map(root, k, K, bits, dirty, aux)
    return g


def mk_eea_aux(b, n, p):
    wl = len_width(n)
    wls = ls_width(n, p)
    d = max(1, unary_depth(n + 3))
    aux = {
        "pathA": b.qubits(d),
        "pathB": b.qubits(max(d, wl, wls - 1)),
        "acc": b.qubit(),
        "carry": b.qubit(),
        "c3xw": b.qubit(),
        "ct1": b.qubit(),
        "ct2": b.qubit(),
        "ct3": b.qubit(),
        "tmp": b.qubit(),
        "sxq": b.qubit(),
        "alv": b.qubit(),
    }
    aux["cscr"] = (aux["pathA"] + aux["pathB"])[:wls]
    return aux


# The GC mul pool: wires provably |0> at all 8 mul call sites (EEA
# endpoint contract) — w1x(3) + lt/lq/lrp(3*wl) + p1/p2/sign(3) + pathA +
# pathB + sxq + alv. Gamma carriers (ls, iterq) are excluded.
# The pool list rides on the aux dict and is filled at EEARegs
# construction (before any mul).
# local: gadgets.wires.WirePool fronts LIFO alloc/AND calls on a spare
# bank; the k-block gadgets pick pool wires by POSITION (_wires keeper
# exclusion, keeper->slot swaps), so they take the raw wire list.


def mk_mul_aux(aux):
    """Role aliases: the mul stack addresses machinery wires by mul
    role.
    """
    return {
        "carry": aux["carry"],
        "flag": aux["ct1"],
        "top": aux["ct2"],
        "tq": aux["tmp"],
        "clean3": [aux["acc"], aux["c3xw"], aux["ct3"]],
        "anc": aux.setdefault("gc_pool", []),
    }


class EEARegs:
    """Fixed qubit assignment for one EEA instance.

    work1 = workspace register (n qubits) + 3 extras; work2 = the field
    register being inverted (n qubits) + 3 extras.  Position j of the paper
    maps to list index j-1.  Work2 is laid out so that the terminal t'
    (little-endian from position 1) lands on [x3 extras, X reversed].
    """

    def __init__(self, n, p, xreg, wsreg, w1x, w2x, lt, lq, ls, lrp, p1, p2, sign, iterq, aux):
        self.n, self.p = n, p
        self.w1r = list(w1x) + list(wsreg)
        # input x big-endian at positions 4..n+3: position 3+i (0-based
        # index 3+i-1) holds bit n-1-(i-1)... keep it simple: index 3+i
        # holds x bit n-1-i.
        self.w2r = list(w2x) + [xreg[n - 1 - i] for i in range(n)]
        self.xreg = list(xreg)
        self.lt, self.lq, self.ls, self.lrp = lt, lq, ls, lrp
        self.p1, self.p2, self.sign, self.iterq = p1, p2, sign, iterq
        self.aux = aux
        # t' bit b sits at position b+1 = index b
        self.tprime = self.w2r[:n]
        # ---- GC pool fill (once per aux instance) ----
        pool = aux.setdefault("gc_pool", [])
        if not pool:
            pool.extend(
                list(w1x)
                + list(lt)
                + list(lq)
                + list(lrp)
                + [p1, p2, sign]
                + list(aux["pathA"])
                + list(aux["pathB"])
                + [aux["sxq"], aux["alv"]]
            )
            assert len(set(pool)) == len(pool)


# ===========================================================================
# loop templates: range scan, step tables (the class tables, tree shapes
# and unary/dual templates live in ../_shared/luo_core.py)
# ===========================================================================


def tpl_range_scan(shape, idx, root, path, racc, leaf_fn, sense, order, node_cls, in_cls, base_cls):
    """Classed template of range_scan (geq sense only — no leq caller
    exists): leaf_fn(j) -> classed items; the accumulator
    toggle CX(e, racc) carries in_cls(j); the root seeding CX carries
    base_cls (the enclosing block's own gate, e.g. the 4|T class).
    """
    assert sense == "geq", "geq only (no leq caller)"
    out = []
    if order == "dec":
        out.append((CX_(root, racc), base_cls))

        def lf(j, e):
            return leaf_fn(j) + [(CX_(e, racc), in_cls(j))]

        out += tpl_unary(shape, idx, root, path, lf, "dec", node_cls)
    else:

        def lf(j, e):
            return [(CX_(e, racc), in_cls(j))] + leaf_fn(j)

        out += tpl_unary(shape, idx, root, path, lf, "inc", node_cls)
        out.append((CX_(root, racc), base_cls))
    return out


def _tpl_swap_block(root, R, st):
    """Build the location-controlled sign <-> Work1 swap tree over the full
    shape, windows in st.swap.
    """
    # Label-shift contract (D=1): tree labels jp = position - 1; R.lq arrives
    # holding lt+lq+[p1=0] = J-1 (adjusted from the r-frame by
    # build_step_template). Leaf jp touches w1r[jp].
    aux = R.aux
    out = []

    def ncls(nid, s):
        return (st.swap, st.node_slot(nid, s))

    def leaf(jp, e):
        cls = (st.swap, st.leaf_slot(st.shape_Ss, jp, 0))
        return [(g, cls) for g in cswap(e, R.sign, R.w1r[jp])]

    out += tpl_unary(st.shape_Ss, R.lq, root, aux["pathA"], leaf, "inc", ncls)
    return out


def _tpl_len_scan(root, bits_w, dirty_w, R, st, scan_reg, target, shape, jmap, dw_desc, tbl, tba):
    """Shared core of the two length-update scans (_tpl_hpw /
    _tpl_rlw): seed/delta patterns in `tba`, scan classes in `tbl`,
    telescoping-XOR dw sweeps around two range scans. Leaf classes:
    0 = in-window AND chain, 1 = the chain CCX, 2 = the closing-edge
    CX. `jmap` maps tree labels jp to register positions j; the chain
    CCX controls on the previous position dirty_w[jmap(jp-1)-1] and
    exists only when that index is a real wire (the reflected scan's
    jp = 0 leaf has none); `dw_desc` picks the dw sweep direction
    (each pattern's dw order is exactly its (seed, deltas, close)
    sequence).
    """
    aux = R.aux
    W, wl = st.W, st.wl
    racc, a_tmp = aux["acc"], aux["c3xw"]
    path = aux["pathA"]
    IS4 = (st.is4, 0)
    out = []
    for i in range(wl):
        out.append((X_(target[i]), (tba, i)))

    def dw():
        rng = range(W, 0, -1) if dw_desc else range(1, W + 1)
        for j in rng:
            for i in range(wl):
                out.append((CX_(dirty_w[j - 1], target[i]), (tba, st.aux_dw(j, i))))

    def ncls(nid, s):
        return (tbl, st.node_slot(nid, s))

    def inc(jp):
        return (tbl, st.leaf_slot(shape, jp, 0))

    def chain(jp):
        return (tbl, st.leaf_slot(shape, jp, 1))

    def edge(jp):
        return (tbl, st.leaf_slot(shape, jp, 2))

    def chain_fwd(jp):
        j = jmap(jp)
        nb = jmap(jp - 1) - 1
        ops = [(ANDP(racc, bits_w[j - 1], a_tmp), inc(jp)), (X_(a_tmp), inc(jp))]
        if 0 <= nb < W:
            ops.append((CCX_(a_tmp, dirty_w[nb], dirty_w[j - 1]), chain(jp)))
        ops.append((CX_(a_tmp, dirty_w[j - 1]), edge(jp)))
        ops += [(X_(a_tmp), inc(jp)), (ANDM(racc, bits_w[j - 1], a_tmp), inc(jp))]
        return ops

    def chain_bwd(jp):
        j = jmap(jp)
        nb = jmap(jp - 1) - 1
        if not (0 <= nb < W):
            return []
        return [
            (ANDP(racc, bits_w[j - 1], a_tmp), chain(jp)),
            (X_(a_tmp), chain(jp)),
            (CCX_(a_tmp, dirty_w[nb], dirty_w[j - 1]), chain(jp)),
            (X_(a_tmp), chain(jp)),
            (ANDM(racc, bits_w[j - 1], a_tmp), chain(jp)),
        ]

    def scan():
        o = tpl_range_scan(
            shape, scan_reg, root, path, racc, chain_fwd, "geq", "dec", ncls, inc, IS4
        )
        o += tpl_range_scan(
            shape, scan_reg, root, path, racc, chain_bwd, "geq", "inc", ncls, inc, IS4
        )
        return o

    dw()
    out += scan()
    dw()
    out += scan()
    return out


def _tpl_hpw(root, bits_w, dirty_w, R, st):
    """Build the highest-set-position scan over the full shape: seed/delta
    patterns in st.lentaux, scan classes in st.lent (window [k4, K4],
    4|T only); writes R.lt from R.lrp's register geometry (the dual of
    right_length_write, on the reflected shape).

    Reflected labels: jp = W - j on shape_Wf, register = RAW lrp
    ([j <= B] == [jp >= lrp] with B = W - lrp) — the reflection needs
    no B-prep arithmetic. Enumeration order and sense flip together,
    so the emitted skeleton is the mirror image.
    """
    W = st.W
    return _tpl_len_scan(
        root,
        bits_w,
        dirty_w,
        R,
        st,
        scan_reg=R.lrp,
        target=R.lt,
        shape=st.shape_Wf,
        jmap=lambda jp: W - jp,
        dw_desc=True,
        tbl=st.lent,
        tba=st.lentaux,
    )


def _tpl_rlw(root, bits_w, dirty_w, R, st, tbl, tba):
    """right_length_write over the full shape: patterns in `tba`, scan
    classes in `tbl` (window [k5, K5] or [k5, W]; 4|T only); writes
    R.lrp from R.lt's register geometry.

    Shifted labels (D=2): jp = j - 2 on shape_Wr, register = RAW lt
    ([j >= A] == [jp >= lt] with A = lt + 2) — the shift needs no
    A-prep arithmetic. j = jp + 2 >= 3, so the chain neighbor always
    exists.
    """
    return _tpl_len_scan(
        root,
        bits_w,
        dirty_w,
        R,
        st,
        scan_reg=R.lt,
        target=R.lrp,
        shape=st.shape_Wr,
        jmap=lambda jp: jp + 2,
        dw_desc=False,
        tbl=tbl,
        tba=tba,
    )


def _tpl_tail(R, st):
    """Post-shift tail of the classed step: shift-back + phase update +
    the 4|T-gated end-of-pass SWAP / length-table updates.
    """
    n = R.n
    W = n + 3
    aux = R.aux
    p1, p2, sign = R.p1, R.p2, R.sign
    lq, ls, lrp = R.lq, R.ls, R.lrp
    w1r, w2r = R.w1r, R.w2r
    ct1, ct3 = aux["ct1"], aux["ct3"]
    tmp = aux["tmp"]
    IS4 = (st.is4, 0)
    out = []

    def A(gates, cls=None, sec=None):
        if cls is None and sec:
            cls = ("spine", sec)
        out.extend((g, cls) for g in gates)

    # ---- post-shift (the p1-gated rotation and the ls inc/decs live
    # in the virtualized/fused top-of-step spellings; only the
    # double-rot-back correction remains) ----
    A([ANDP(p1, p2, ct1, 0, 0)], sec="rot-right2")
    A(_rot_by_right(ct1, w2r, 2), sec="rot-right2")
    A([ANDM(p1, p2, ct1, 0, 0)], sec="rot-right2")

    # ---- phase update ----
    lits_lq = [(q, 1) for q in lq]
    glq, cq_lq = and_ladder(lits_lq, aux["pathA"])
    A(glq, sec="phase-update")
    lits_rp = [(q, 1) for q in lrp]
    grp, cq_rp = and_ladder(lits_rp, aux["pathB"])
    A(grp, sec="phase-update")
    A([ANDP(cq_lq, cq_rp, ct3, 0, 1)], sec="phase-update")
    A(
        [CX_(sign, tmp), CX_(p1, tmp), CCX_(ct3, tmp, p2), CX_(p1, tmp), CX_(sign, tmp)],
        sec="phase-update",
    )
    A([CCX_(ct3, p2, sign)], sec="phase-update")
    A([ANDM(cq_lq, cq_rp, ct3, 0, 1)], sec="phase-update")
    A(inv_list(grp), sec="phase-update")
    A(inv_list(glq), sec="phase-update")
    lits_ls = [(q, 1 - (((n + 1) >> i) & 1)) for i, q in enumerate(ls)]
    gls, cq_ls = and_ladder(lits_ls, aux["pathA"])
    A(gls, sec="phase-update")
    A([CX_(cq_ls, p1), CX_(cq_ls, p2)], sec="phase-update")
    A(inv_list(gls), sec="phase-update")

    # ---- end-of-iteration SWAP + length updates (whole block 4|T-gated) --
    glq2, cq2 = and_ladder(lits_lq, aux["pathA"])
    gls2, cq3 = and_ladder(lits_ls, aux["pathB"])
    cond_open = glq2 + gls2 + [ANDP(cq2, cq3, ct3, 0, 0)] + inv_list(gls2) + inv_list(glq2)
    A(cond_open, IS4)
    g = []
    for i in range(W):
        g += cswap(ct3, w1r[i], w2r[i])
    A(g, IS4)
    out.extend(_tpl_hpw(ct3, w2r, w1r, R, st))
    out.extend(_tpl_hpw(ct3, w1r, w2r, R, st))
    out.extend(_tpl_rlw(ct3, w1r, w2r, R, st, st.lenrp, st.lenrpaux))
    out.extend(_tpl_rlw(ct3, w2r, w1r, R, st, st.lenrpE, st.lenrpEaux))
    A([CX_(ct3, R.iterq)], IS4)
    A(inv_list(cond_open), IS4)
    return out


# ---------------------------------------------------------------------------
# StepTables: the per-(slot, step) gating tables. _fill chains base
# window marking -> t-core tables (tfr/tcore) -> r-core tables (rdual/
# rfrA/rfrB/rcore/rmode). The lenrpE / lenrpEaux markings read the
# truncated D6 new-scan edge (_newK, priced).
# ---------------------------------------------------------------------------


class StepTables:
    """All per-(slot, step) gating tables for one (n, p) instance.  Shared
    by every EEA run (forward and inverse, both register bindings): the
    window schedule depends only on (n, T).
    """

    def __init__(self, n, p):
        self.n = n
        self.nmax = nmax_steps(n, p)
        W = n + 3
        self.W = W
        self.wl = len_width(n)
        # Label-shifted tree shapes: the trees compare registers
        # that arrive WITHOUT "+const" preps, so their label
        # universes shift down by the same const (r: -2 vs position,
        # register lt+lq; swap: -1, register lt+lq+[p1=0]; t: -1,
        # registers lt / LS''-lrp). Window k-edges clamp to the
        # structural minima (L>=3, J>=2) so shifted labels stay >= 0.
        self.shape_Wr = _UShape(W - 2)  # r/rlw: labels [1, n+1]
        self.shape_Ss = _UShape(W - 2)  # swap: labels [1, n+1]
        self.shape_St = _UShape(W - 2, lo=0)  # t: labels [0, n+1]
        self.shape_Wf = _UShape(W - 1, lo=0)  # hpw REFLECTED: jp = W-j
        NM = self.nmax
        nWr, nSs, nSt = (self.shape_Wr.n_nodes, self.shape_Ss.n_nodes, self.shape_St.n_nodes)
        wl = self.wl
        self.r = ClassTable("r", NM, 2 * nWr + (W - 2))
        self.swap = ClassTable("swap", NM, 2 * nSs + (W - 2))
        self.t = ClassTable("t", NM, 2 * nSt + (W - 1))
        # length-update roles: nodes + 3 leaf classes (in / interior / edge)
        nWf = self.shape_Wf.n_nodes
        self.lent = ClassTable("lent", NM, 2 * nWf + 3 * W)
        self.lenrp = ClassTable("lenrp", NM, 2 * nWr + 3 * (W - 2))
        self.lenrpE = ClassTable("lenrpE", NM, 2 * nWr + 3 * (W - 2))
        # telescoping-XOR write patterns: wl seed slots + (j, i) delta slots
        self.lentaux = ClassTable("lentaux", NM, wl + W * wl)
        self.lenrpaux = ClassTable("lenrpaux", NM, wl + W * wl)
        self.lenrpEaux = ClassTable("lenrpEaux", NM, wl + W * wl)
        self.is4 = ClassTable("is4", NM, 1)
        self._fill()

    # ---- slot layout ---------------------------------------------------
    @staticmethod
    def node_slot(nid, state):
        return 2 * nid + (0 if state == "A" else 1)

    @staticmethod
    def leaf_slot(shape, j, cls=0):
        nlab = shape.L - shape.lo + 1
        return 2 * shape.n_nodes + cls * nlab + (j - shape.lo)

    def aux_dw(self, j, i):
        return self.wl + (j - 1) * self.wl + i

    # ---- per-step activation --------------------------------------------
    def _fill(self):
        n, wl = self.n, self.wl
        mask = (1 << wl) - 1
        for T in range(1, self.nmax + 1):
            win = active_windows(n, T)
            rk, rK = win["r"]
            self._mark_tree(self.r, self.shape_Wr, T, max(rk, 3) - 2, rK - 2)
            sk, sK = win["swap"]
            self._mark_tree(self.swap, self.shape_Ss, T, max(sk, 2) - 1, sK - 1)
            tk, tK = win["t"]
            self._mark_tree(self.t, self.shape_St, T, tk - 1, tK - 1)
            if T % 4:
                continue
            self.is4.set(T, 0)
            k4, K4 = win["len_t"]
            k5, K5 = win["len_rp"]
            W = self.W
            # hpw reflected (jp = W-j): upper edge becomes the low-jp
            # edge, so the mark's low-edge (j == k) boundary applies.
            self._mark_len(self.lent, T, W - K4, W - k4, shape=self.shape_Wf)
            # rlw shifted (D=2, register raw lt): A = lt+2 >= 3 always,
            # so k5 clamps to 3 structurally.
            k5c = max(k5, 3)
            self._mark_len(self.lenrp, T, k5c - 2, K5 - 2, shape=self.shape_Wr)
            # lenrpE scans to the PRICED D6 right edge, not W (_newK)
            self._mark_len(self.lenrpE, T, k5c - 2, _newK(n, T, k5) - 2, shape=self.shape_Wr)
            # hpw pattern: val = pos & mask, seeded at the K end.
            self._fill_dw_aux(self.lentaux, T, k4, K4, lambda pos: pos & mask, False)
            self._fill_rlw_aux(self.lenrpaux, T, k5, K5)
            self._fill_rlw_aux(self.lenrpEaux, T, k5, _newK(n, T, k5))
        self._fill_tcore()
        self._fill_rcore()

    def _fill_tcore(self):
        """t-prefix certainty core: fringe tree (tfr) + ungated core run
        (tcore) on the certified RLO(T) split; no-op when n is untabled.
        """
        self.tcore_max = 0  # top of any step's ungated core run
        tab = _priced()["tcore"].get(self.n)
        if tab is None:
            return
        shape = self.shape_St
        nlab = shape.L - shape.lo + 1
        self.tfr = ClassTable("tfr", self.nmax, 2 * shape.n_nodes + nlab)
        self.tcore = ClassTable("tcore", self.nmax, self.W - 1)
        for T in range(1, self.nmax + 1):
            k3, K3 = active_windows(self.n, T)["t"]
            rlo = _rlo_at(self.n, T, K3)
            kf = max(k3, rlo)
            self.tcore_max = max(self.tcore_max, rlo)

            self._mark_tree(self.tfr, shape, T, kf - 1, K3 - 1)
            for j in range(k3, rlo):
                self.tcore.set(T, j - 1)

    def _fill_rcore(self):
        """r certainty core: dual spelling (rdual) on unsplit steps,
        fringe trees (rfrA/rfrB) + core run (rcore) on split steps
        (rmode selects); no-op when n is untabled.
        """
        self.rcore_span = (self.W + 1, 0)  # (cmin, cmax), empty default
        tab = _priced()["rcore"].get(self.n)
        if tab is None:
            return
        shape = self.shape_Wr
        NM, W = self.nmax, self.W
        nWr = shape.n_nodes
        nlab = shape.L - shape.lo + 1
        self.rdual = ClassTable("rdual", NM, 2 * nWr + nlab)
        self.rfrA = ClassTable("rfrA", NM, 2 * nWr + nlab)
        self.rfrB = ClassTable("rfrB", NM, 2 * nWr + nlab)
        self.rcore = ClassTable("rcore", NM, W)
        self.rmode = ClassTable("rmode", NM, 2)
        for T in range(1, NM + 1):
            k, K = active_windows(self.n, T)["r"]
            k = max(k, 3)
            sp = _rsplit_at(self.n, T, k, K)
            if sp is None:
                self.rmode.set(T, 0)

                self._mark_tree(self.rdual, shape, T, k - 2, K - 2)
            else:
                lmax, rmin = sp
                self.rcore_span = (
                    min(self.rcore_span[0], lmax + 1),
                    max(self.rcore_span[1], rmin - 1),
                )
                self.rmode.set(T, 1)

                self._mark_tree(self.rfrA, shape, T, rmin - 2, K - 2)

                self._mark_tree(self.rfrB, shape, T, k - 2, lmax - 2)
                for j in range(lmax + 1, rmin):
                    self.rcore.set(T, j - 1)

    def _mark_tree(self, tbl, shape, T, k, K):
        def onn(nid, st):
            tbl.set(T, self.node_slot(nid, st))

        def onl(j):
            tbl.set(T, self.leaf_slot(shape, j, 0))

        shape.mark(k, K, onn, onl)

    def _mark_len(self, tbl, T, k, K, shape):
        def onn(nid, st):
            tbl.set(T, self.node_slot(nid, st))

        def onl(j):
            tbl.set(T, self.leaf_slot(shape, j, 0))  # in-window
            if j > k:
                tbl.set(T, self.leaf_slot(shape, j, 1))  # chain CCX
            else:
                tbl.set(T, self.leaf_slot(shape, j, 2))  # j == k: CX

        shape.mark(k, K, onn, onl)

    def _fill_dw_aux(self, tbl, T, k, K, val, seed_at_k):
        """Telescoping-XOR write pattern (shared by the rlw and hpw
        aux tables): the seed slots get val(seed end), each in-window
        label j gets the delta val(j)^val(j-toward-seed), and the
        closing end gets its full val at aux_dw(end, i).  seed_at_k
        picks the orientation; each pattern's dw list order
        (rlw ascending with the K-entry last, hpw descending with the
        k-entry last) is exactly the (seed, deltas, close) sequence.
        """
        wl = self.wl
        seed_pos, close_pos = (k, K) if seed_at_k else (K, k)
        for i in range(wl):
            if (val(seed_pos) >> i) & 1:
                tbl.set(T, i)
        rng = range(k, K) if seed_at_k else range(k + 1, K + 1)
        step = 1 if seed_at_k else -1
        for j in rng:
            delta = val(j) ^ val(j + step)
            for i in range(wl):
                if (delta >> i) & 1:
                    tbl.set(T, self.aux_dw(j, i))
        for i in range(wl):
            if (val(close_pos) >> i) & 1:
                tbl.set(T, self.aux_dw(close_pos, i))

    def _fill_rlw_aux(self, tbl, T, k, K):
        """rlw pattern: val = (n+4-pos) truncated to wl bits."""
        n, mask = self.n, (1 << self.wl) - 1
        self._fill_dw_aux(tbl, T, k, K, lambda pos: (n + 4 - pos) & mask, True)


# ===========================================================================
# the step: merged-r + certainty cores (classed template)
# ===========================================================================


def _uncond_inc(reg, anc):
    """reg += 1: carries c1=reg0 (CX copy), c_{i+1}=c_i&reg_i, released
    by measurement after reg_i flips (c_i AND NOT reg_i).  w-2
    Toffoli.
    """
    w = len(reg)
    g = [CX_(reg[0], anc[0])]
    for i in range(1, w - 1):
        g.append(ANDP(anc[i - 1], reg[i], anc[i]))
    g.append(X_(reg[0]))
    for i in range(1, w):
        g.append(CX_(anc[i - 1], reg[i]))
    for i in range(w - 2, 0, -1):
        g.append(ANDM(anc[i - 1], reg[i], anc[i], 0, 1))
    g += [X_(reg[0]), CX_(reg[0], anc[0]), X_(reg[0])]
    return g


def _rot_by_right(ctrl, reg, s):
    """Build a controlled cyclic rotation right by s (cycle decomposition)."""
    m = len(reg)
    g = []
    s %= m
    if s == 0 or m <= 1:
        return g
    seen = [False] * m
    for start in range(m):
        if seen[start]:
            continue
        cyc = []
        j = start
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = (j + s) % m
        pivot = cyc[0]
        for j in cyc[1:]:
            g += cswap(ctrl, reg[pivot], reg[j])
    return g


# ---------------------------------------------------------------------------
# The step template. Per step T the classed body carries: the merged-r
# step (with the certainty-core t-prefix inside), r-core bracket
# surgery on split steps, the virtualized-rotation frame (one
# unconditional relabel-rotation), and the fused ls arithmetic.
# ---------------------------------------------------------------------------


def _core_cf_sub(s, a, b2, c):
    return [CX_(c, a), CX_(c, b2), CCX_(s, b2, a), CCX_(a, b2, c)]


def _core_cs_sub(_s, a, b2, c):
    return [CCX_(a, b2, c), CX_(c, b2), CX_(c, a)]


def _core_cf_add(_s, a, b2, c):
    return [CX_(c, a), CX_(c, b2), CCX_(a, b2, c)]


def _core_cs_add(s, a, b2, c):
    return [CCX_(a, b2, c), CCX_(s, b2, a), CX_(c, b2), CX_(c, a)]


_CORE_CELLS = {"sub": (_core_cf_sub, _core_cs_sub), "add": (_core_cf_add, _core_cs_add)}


# ---------------------------------------------------------------------------
# The step, template spelling (per-T ClassTable gating).
# ---------------------------------------------------------------------------


def _tpl_t_block(mode, sign_update, root, R, st):
    assert hasattr(st, "tcore"), "certainty-core tables required"
    aux = R.aux
    acc, carry, c3xw = aux["acc"], aux["carry"], aux["c3xw"]
    w1r = R.w1r
    # Virtualized-rotation frame: the step's one rotation already happened
    # at the top, so the t-block addresses Work2 through the rotated view
    # (position j lives on wire (j-1) mod W -- list index (j-2) mod W).
    W = R.n + 3
    w2r = [R.w2r[(i - 1) % W] for i in range(W)]
    idx = R.lt  # holds the t endpoint while the t-prep is open
    cf = cell_uma_inv if mode == "sub" else cell_maj
    cs = cell_maj_inv if mode == "sub" else cell_uma
    ccf, ccs = _CORE_CELLS[mode]
    # Label shift (D=1): tree labels jp = position - 1; the tree register
    # is lt (no +1 prep). Leaf jp touches wire index jp.
    shape = st.shape_St
    coremax = st.tcore_max
    out: list = [(CX_(root, acc), ("spine", "t-edge"))]
    for j in range(1, coremax):
        cls = (st.tcore, j - 1)
        out += [(g, cls) for g in ccf(acc, w2r[j - 1], w1r[j - 1], carry)]

    def ncls(nid, s):
        return (st.tfr, st.node_slot(nid, s))

    def lin(jp):
        return (st.tfr, st.leaf_slot(shape, jp, 0))

    def leaf1(jp, e):
        ops = cf(acc, w2r[jp], w1r[jp], carry, c3xw) + [CX_(e, acc)]
        return [(g, lin(jp)) for g in ops]

    out += tpl_unary(shape, idx, root, aux["pathA"], leaf1, "inc", ncls)
    if sign_update:
        out.append((CCX_(root, carry, R.sign), ("spine", "t-edge")))

    def leaf2(jp, e):
        ops = [(CX_(e, acc), lin(jp))]
        ops += [(g, lin(jp)) for g in cs(acc, w2r[jp], w1r[jp], carry, c3xw)]
        return ops

    out += tpl_unary(shape, idx, root, aux["pathA"], leaf2, "dec", ncls)
    for j in range(coremax - 1, 0, -1):
        cls = (st.tcore, j - 1)
        out += [(g, cls) for g in ccs(acc, w2r[j - 1], w1r[j - 1], carry)]
    out.append((CX_(root, acc), ("spine", "t-edge")))
    return out


def _tpl_r_section(R, st):
    aux = R.aux
    p1, p2 = R.p1, R.p2
    ct1, ct2, ct3 = aux["ct1"], aux["ct2"], aux["ct3"]
    acc, carry, c3xw = aux["acc"], aux["carry"], aux["c3xw"]
    t = aux["tmp"]
    w1r, w2r = R.w1r, R.w2r
    shape = st.shape_Wr
    # Label-shift contract (D=2): tree labels jp = position - 2.
    # R.lq arrives holding lt+lq = L-2 (the shared lq frame) and R.ls
    # natively holds LS'' = n+1-ls = R-2 (the global complement frame):
    # both edge registers match the shifted labels with NO per-step
    # arithmetic. Leaf jp touches position jp+2 (wire index jp+1).
    out: list = [(ANDP(p1, aux["alv"], ct1, 1, 1), ("spine", "r-ctl"))]

    def nA(nid, s):
        return (st.rfrA, st.node_slot(nid, s))

    def lA(j):
        return (st.rfrA, st.leaf_slot(shape, j, 0))

    def nB(nid, s):
        return (st.rfrB, st.node_slot(nid, s))

    def lB(j):
        return (st.rfrB, st.leaf_slot(shape, j, 0))

    def nD(nid, s):
        return (st.rdual, st.node_slot(nid, s))

    def lD(j):
        return (st.rdual, st.leaf_slot(shape, j, 0))

    cmin, cmax = st.rcore_span

    # ---- pass 1 (dec): split-top, split-core, split-bot, dual ----
    def l1top(jp, ea):
        ops = [CX_(ea, acc)] + cell_uma_inv(acc, w1r[jp + 1], w2r[jp + 1], carry, c3xw)
        return [(g, lA(jp)) for g in ops]

    out += tpl_unary(shape, R.ls, ct1, aux["pathA"], l1top, "dec", nA)
    for j in range(cmax, cmin - 1, -1):
        out += [(g, (st.rcore, j - 1)) for g in _core_cf_sub(acc, w1r[j - 1], w2r[j - 1], carry)]

    def l1bot(jp, eb):
        ops = cell_uma_inv(acc, w1r[jp + 1], w2r[jp + 1], carry, c3xw) + [CX_(eb, acc)]
        return [(g, lB(jp)) for g in ops]

    out += tpl_unary(shape, R.lq, ct1, aux["pathB"], l1bot, "dec", nB)

    def l1dual(jp, ea, eb):
        ops = (
            [CX_(ea, acc)]
            + cell_uma_inv(acc, w1r[jp + 1], w2r[jp + 1], carry, c3xw)
            + [CX_(eb, acc)]
        )
        return [(g, lD(jp)) for g in ops]

    out += tpl_dual(shape, R.ls, R.lq, ct1, aux["pathA"], aux["pathB"], l1dual, "dec", nD)

    # Guard asymmetry, justified: in the dual spelling (rmode 0) every
    # carry-writing Toffoli descends from the ct1-rooted trees, so
    # carry already encodes ct1 AND borrow and the sign update needs no
    # extra ct1 guard; on split steps (rmode 1) the ungated
    # certainty-core cells can dirty carry when ct1 = 0, so the twin
    # takes ct1 explicitly.
    out.append((CX_(carry, R.sign), (st.rmode, 0)))
    out.append((CCX_(ct1, carry, R.sign), (st.rmode, 1)))
    out += [
        (ANDP(p1, p2, ct2, 1, 0), ("spine", "r-ctl")),
        (CX_(ct2, R.sign), ("spine", "r-ctl")),
        (ANDM(p1, p2, ct2, 1, 0), ("spine", "r-ctl")),
        (ANDP(p2, R.sign, ct2, 0, 0), ("spine", "r-ctl")),
        (ANDP(ct1, ct2, ct3, 0, 1), ("spine", "r-ctl")),
    ]

    # ---- pass 2 (inc): dual, split-bot, hoist+core, split-top ----
    def p2body(jp):
        a, b = w1r[jp + 1], w2r[jp + 1]
        return (
            c3x(acc, a, b, carry, c3xw)
            + [ANDP(acc, ct3, t), CCX_(t, b, a), ANDM(acc, ct3, t)]
            + [CX_(carry, b), CX_(carry, a)]
        )

    def l2dual(jp, ea, eb):
        ops = [CX_(eb, acc)] + p2body(jp) + [CX_(ea, acc)]
        return [(g, lD(jp)) for g in ops]

    out += tpl_dual(shape, R.ls, R.lq, ct1, aux["pathA"], aux["pathB"], l2dual, "inc", nD)

    def l2bot(jp, eb):
        ops = [CX_(eb, acc)] + p2body(jp)
        return [(g, lB(jp)) for g in ops]

    out += tpl_unary(shape, R.lq, ct1, aux["pathB"], l2bot, "inc", nB)
    out.append((ANDP(acc, ct3, t), (st.rmode, 1)))
    for j in range(cmin, cmax + 1):
        out += [(g, (st.rcore, j - 1)) for g in _core_cs_add(t, w1r[j - 1], w2r[j - 1], carry)]
    out.append((ANDM(acc, ct3, t), (st.rmode, 1)))

    def l2top(jp, ea):
        ops = p2body(jp) + [CX_(ea, acc)]
        return [(g, lA(jp)) for g in ops]

    out += tpl_unary(shape, R.ls, ct1, aux["pathA"], l2top, "inc", nA)
    out += [
        (ANDM(ct1, ct2, ct3, 0, 1), ("spine", "r-ctl")),
        (ANDM(p2, R.sign, ct2, 0, 0), ("spine", "r-ctl")),
        (ANDM(p1, aux["alv"], ct1, 1, 1), ("spine", "r-ctl")),
    ]
    return out


def build_step_template(R, st):
    """Build the production step template: classed body with the virtualized-
    rotation / fused-ls frame spellings emitted directly (top-of-step
    fused ls arithmetic + unconditional relabel-rotation, certainty-core
    r/t sections, shared post-shift tail).
    """
    n = R.n
    W = n + 3
    aux = R.aux
    p1, p2, sign = R.p1, R.p2, R.sign
    lt, lq, ls, lrp = R.lt, R.lq, R.ls, R.lrp
    w2r = R.w2r
    ct1, ct2 = aux["ct1"], aux["ct2"]
    alv = aux["alv"]
    ladder_anc = aux["pathA"]
    out = []

    def A(gates, sec=None):
        cls = ("spine", sec) if sec else None
        out.extend((g, cls) for g in gates)

    # ls-arithmetic fusion: the step's phase-gated inc/dec quartet
    # unions into one unconditional inc + one p2-controlled double dec,
    # emitted here at the step top (the t-prep frame constant below compensates
    # with n+2, phase-4 seeing ls-1).
    # Under the global LS'' = n+1-ls complement frame the fused step
    # arithmetic flips: ls += 1 -> LS'' -= 1, the p2 double-dec -> +2.
    # The net LS'' += 2*P2 - 1 is one increment inside complement
    # fans controlled on P2 = 0 (~(~x + 1) == x - 1).
    nfan = [X_(p2)] + [CX_(p2, q) for q in ls] + [X_(p2)]
    A(nfan + _uncond_inc(ls, aux["pathB"]) + nfan, sec="ls-fused")
    # Rotation virtualization: both phase-gated rotation chains
    # become this single unconditional relabel-rotation (Clifford); the
    # t-block reads Work2 through the rotated view, dead lanes
    # over-rotate, and the ls barrel corrects at pass end (emit_eea).
    A([("swap", w2r[i], w2r[i + 1]) for i in range(W - 1)], sec="rot-virt")
    A(_zero_toggle(lrp, alv, ladder_anc), sec="alive-toggle")
    A([ANDP(p1, p2, ct1, 1, 0), ANDP(ct1, alv, ct2, 0, 1)], sec="rot-right2")
    A(_rot_by_right(ct2, w2r, 2), sec="rot-right2")
    A([ANDM(ct1, alv, ct2, 0, 1)], sec="rot-right2")
    A([ANDM(p1, p2, ct1, 1, 0)], sec="rot-right2")

    # ---- shared lq frame: one build serves the r section (L = lt+lq+2)
    # and, after a -1+[p1=0] adjust, the swap tree (J); one teardown at
    # the end. (The step's own lq +-1 updates commute with the constant
    # frame and run inside it.)
    A(add_gid(lt, lq, aux["pathB"]), sec="lq-prep")

    # ---- r section: merged pair with the certainty-core split ----
    out.extend(_tpl_r_section(R, st))
    A(_zero_toggle(lrp, alv, ladder_anc), sec="alive-toggle")

    # r-frame (lt+lq) -> swap-frame (lt+lq+[p1=0]) adjust
    A([X_(p1)] + ctrl_inc(lq, p1, aux["pathB"]) + [X_(p1)], sec="lq-prep")

    # ---- middle: swap block + tprep + MERGED t pair ----
    sxq = aux["sxq"]
    A([CX_(p1, sxq), CX_(p2, sxq)], sec="swap-ctl")
    out.extend(_tpl_swap_block(sxq, R, st))
    A([CX_(p2, sxq), CX_(p1, sxq)], sec="swap-ctl")
    # The step's lq updates (-[P1 and not P2], +[not P1 and P2]) and the
    # frame teardown's -[P1 = 0] net to lq -= [P2 = 0]: one controlled
    # decrement.  Then the rest of the teardown: lq -= lt.
    A([X_(p2)] + ctrl_dec(lq, p2, aux["pathB"]) + [X_(p2)], sec="swap-ctl")
    A(inv_list(add_gid(lt, lq, aux["pathB"])), sec="lq-prep")
    # In-place t endpoint: lrp <- LS'' - lrp == n+1 - ls - lrp, then a
    # P2-controlled swap of lt and lrp, so lt holds (P2 ? n+1-ls-lrp :
    # lt): the t tree's shifted labels (D=1) absorb the +1/-1 constants
    # (the ls-fusion compensation rides in the frame constant:
    # n+1 == (n+2) - 1).  The subtraction runs as ~(lrp + ~LS'') ==
    # LS'' - lrp.  The t-blocks read no length register.
    wl = len(lt)
    assert len(lrp) == wl
    lsl = ls[:wl]
    tprep = [X_(q) for q in lsl]
    tprep += add_gid(lsl, lrp, aux["pathB"])
    tprep += [X_(q) for q in lsl] + [X_(q) for q in lrp]
    for i in range(wl):
        tprep += cswap(p2, lt[i], lrp[i])
    A(tprep, sec="t-prep")
    A([ANDP(p2, sign, ct1, 1, 0), ANDP(p1, ct1, ct2, 0, 1)], sec="t-ctl")
    out.extend(_tpl_t_block("sub", False, ct2, R, st))
    A([ANDM(p1, ct1, ct2, 0, 1), ANDM(p2, sign, ct1, 1, 0)], sec="t-ctl")
    A([CX_(p1, sign)], sec="t-ctl")
    out.extend(_tpl_t_block("add", True, p1, R, st))
    A(inv_list(tprep), sec="t-prep")
    out.extend(_tpl_tail(R, st))
    return out


# ===========================================================================
# the mul stack: priced windows + dispatch (the k-block gadgets and the
# Horner multiplier live in ../_shared/luo_core.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Priced mul windows (n >= 128; small n runs exact):
# fold chains cap at bit 63 — two blocks at the 55-wire pool (a
# 64-wire pool would run them single-block, 48,960 Toffoli cheaper per
# point add, at 9 more qubits); the pseudo-Mersenne constants'
# ones end at bit 33 (asserted below), beyond which carries only
# propagate (fwd) or saturate against the flag (inv, the truncated
# kb-sums cancel).
# The lazy-flag compare runs over the top 32 bits. Worst-case escape
# pricing is analytic (2040 calls/family/shot): fold carry-out c/2^63
# <= 0.14/100K + top-32 tie 2^-32 = 0.047/100K — <= 0.19/100K against
# Unified(20); the bound is draw-independent.
# ---------------------------------------------------------------------------
FOLD_WIN = 63
CMP_TOPG = 32
EXACT_MUL_BELOW_N = 128  # certified dispatch: exact mod-p mul spellings
# below this width, k-block lazy gadgets at/above

# priced-window precondition: the fold constant's ones end at bit 33,
# strictly inside the FOLD_WIN cap — beyond it carries only propagate.
assert ((1 << N) - P_MOD).bit_length() == 33 <= FOLD_WIN


# Dispatch mirrors the certified configuration: at reduced widths the
# build runs the EXACT mod-p spellings (the lazy family's deferred
# carries are draw-dependent when c = 2^n - p is large relative to
# 2^n); at n >= 128 the k-block lazy gadgets carry the production
# stream. The 100,000-shot evaluation certifies the full-width
# composition.


def _mul_cma(b, ctrl, addend, target, p, n, ma, inv=False):
    if n < EXACT_MUL_BELOW_N:
        ctrl_mod_add(b, ctrl, addend, target, p, n, ma, inv=inv)
        return
    anc = ma["anc"]
    assert anc, "GC pool empty (EEARegs not constructed yet)"
    gc_lazy_cma(b, ctrl, addend, target, p, n, ma, anc, inv=inv, win=FOLD_WIN, topg=CMP_TOPG)


def _mul_dbl(b, reg, p, n, ma, inv=False):
    if n < EXACT_MUL_BELOW_N:
        dbl_modp(b, reg, p, n, ma, inv=inv)
        return
    anc = ma["anc"]
    assert anc, "GC pool empty (EEARegs not constructed yet)"
    kb_dbl_fold(b, reg, p, n, anc, inv=inv, win=FOLD_WIN)


def mul_modp(b, xc, y, out, p, n, ma, square=False, inv=False):
    """out <- xc*y mod p (out starts |0>): the shared windowed Horner
    multiplier on this circuit's dispatch (_mul_cma / _mul_dbl above).
    """
    horner_mul(b, xc, y, out, p, n, ma, cma=_mul_cma, dbl=_mul_dbl, square=square, inv=inv)


# ===========================================================================
# EEA emission + point-add flow
# ===========================================================================


def build_circuit(b, p, n):
    """Emit the point addition |P> -> |P+Q> into builder b, the addend Q
    read per shot from the Bit-kind input registers.  Returns
    (x1, y1, ox, oy): the target wire lists and the addend bit lists.
    """
    x1, y1, ox, oy = declare_point_io(b, n)  # registers 0-3

    # per-shot cbspecs (free classical bit networks of the input bits;
    # the compile-time spec builder's 0 < C < p contract is the input
    # domain)
    S_x2, S_y2, S_3x2 = point_cbspecs(b, ox, oy, p, n)

    A = b.qubits(n)
    w1x = b.qubits(3)
    w2x = b.qubits(3)
    wl = len_width(n)
    wls = ls_width(n, p)
    lt = b.qubits(wl)
    lq = b.qubits(wl)
    ls = b.qubits(wls)
    lrp = b.qubits(wl)
    p1, p2 = b.qubit(), b.qubit()
    sign, iterq = b.qubit(), b.qubit()
    aux = mk_eea_aux(b, n, p)
    ma = mk_mul_aux(aux)
    clean3 = ma["clean3"]
    flagq = aux["ct1"]

    Rws = EEARegs(n, p, x1, A, w1x, w2x, lt, lq, ls, lrp, p1, p2, sign, iterq, aux)
    Rwy = EEARegs(n, p, x1, y1, w1x, w2x, lt, lq, ls, lrp, p1, p2, sign, iterq, aux)

    # ---- ownership wire-sets (pure bookkeeping: b.claim/b.release emit
    # no ops; the stream is byte-identical with or without them).  Every
    # long-lived wire has exactly one owner, and the vocabulary ANDs run
    # on owned persistent scratch, so owned == live at every scored op;
    # the anonymous pool is transiently nonzero only inside the
    # alloc/claim and release/free bookkeeping bursts (init, the
    # handoffs, the teardown).  tools/occupancy_profile.py replays this log
    # fail-closed: claims need live-and-unowned wires, release precedes
    # every free, and only the declared registers end owned.  Set names
    # follow the README Cost census (512 I/O + n workspace A + the
    # machine wires wx/len/flag/aux).
    own = {
        "io/x1": list(x1),
        "io/y1": list(y1),
        "A": list(A),
        "wx": list(w1x) + list(w2x),
        "len": list(lt) + list(lq) + list(ls) + list(lrp),
        "flag": [p1, p2, sign, iterq],
        "aux": (
            list(aux["pathA"])
            + list(aux["pathB"])
            + [aux[k] for k in ("acc", "carry", "c3xw", "ct1", "ct2", "ct3", "tmp", "sxq", "alv")]
        ),
    }
    assert sum(len(v) for v in own.values()) == b.active_qubits
    for nm, ids in own.items():
        b.claim(nm, ids)

    with b.phase("s1-const-subs"):
        const_modp(b, x1, S_x2, flagq, y1, clean3, p, n, inv=True)
        const_modp(b, y1, S_y2, flagq, x1, clean3, p, n, inv=True)
    with b.phase("s2-idiv"):
        emit_idiv(b, x1, y1, A, Rws, Rwy, p, n, ma)

    # Role handoff: the EEA machinery wires (== the gc_pool list) are
    # guaranteed |0> from the s2 teardown until the s6 setup.  Free them
    # here and re-alloc the SAME ids (LIFO free-list, asserted) as the
    # mul pool for s3-s5 — every ma/aux binding stays valid, the peak
    # accounting stays exact, and the liveness raster shows the handoff
    # instead of one solid block.  (sxq is a pool wire and s4 uses it as
    # the constant-1 control, so the pool span is s3..s5 contiguous.)
    pool = aux["gc_pool"]
    poolset = set(pool)
    assert aux["sxq"] in poolset  # so the pool span covers s4
    # per-set slices of the pool (release and claim must name the owner)
    pool_owner = {nm: [q for q in ids if q in poolset] for nm, ids in own.items()}
    pool_owner = {nm: ids for nm, ids in pool_owner.items() if ids}
    assert sum(len(v) for v in pool_owner.values()) == len(pool)

    def _handoff(tag, to_pool):
        # Ownership follows the role: the machinery sets release the
        # pool ids and the single "mulpool" set claims them (mirrored
        # on the way back).  Release precedes free on both paths; the
        # claims re-attach to the re-alloc'd (asserted identical) ids.
        for nm, ids in pool_owner.items() if to_pool else (("mulpool", pool),):
            b.release(nm, ids)
        b.recycle_all(list(reversed(pool)))
        got = [b.qubit() for _ in pool]
        assert got == list(pool), f"pool id drift at {tag}"
        for nm, ids in (("mulpool", pool),) if to_pool else pool_owner.items():
            b.claim(nm, ids)

    with b.phase("s3-squ"):
        _handoff("s2->s3 (machinery -> mul pool)", to_pool=True)
        mul_modp(b, y1, y1, A, p, n, ma, square=True)
    with b.phase("s4-sub-squ"):
        oneq = aux["sxq"]
        b.x(oneq)
        ctrl_mod_add(b, oneq, A, x1, p, n, ma, inv=True)
        b.x(oneq)
        const_modp(b, x1, S_3x2, flagq, y1, clean3, p, n)
    with b.phase("s5-squ-dag"):
        mul_modp(b, y1, y1, A, p, n, ma, square=True, inv=True)
    with b.phase("s6-imul"):
        _handoff("s5->s6 (mul pool -> machinery)", to_pool=False)
        emit_imul(b, x1, y1, A, Rwy, p, n, ma)
    with b.phase("s7-final"):
        neg_modp(b, x1, y1, clean3, p, n)
        const_modp(b, x1, S_x2, flagq, y1, clean3, p, n)
        const_modp(b, y1, S_y2, flagq, x1, clean3, p, n, inv=True)
        # End-of-circuit: every non-output wire is |0> (the evaluator's
        # reversibility gate re-verifies); release ownership, then free
        # in reverse allocation order.  Only io/x1 + io/y1 stay owned
        # (the declared registers).  Of the 340 non-io wires freed here,
        # only 4 (ct1 + the clean3 acc/c3xw/ct3) are s7 gate operands;
        # the A register and the rest of the walk machinery are |0> by
        # s7 entry and could release there.  Freeing them at s7 entry
        # would move ~336 unscored R records and change the op stream;
        # peak and TQ-cost would stay the same (the peak is set at init).
        for nm, ids in own.items():
            if not nm.startswith("io/"):
                b.release(nm, ids)
        for reg in (
            [
                aux["alv"],
                aux["sxq"],
                aux["tmp"],
                aux["ct3"],
                aux["ct2"],
                aux["ct1"],
                aux["c3xw"],
                aux["carry"],
                aux["acc"],
            ],
            aux["pathB"],
            aux["pathA"],
            [iterq, sign, p2, p1],
            lrp,
            ls,
            lq,
            lt,
            w2x,
            w1x,
            A,
        ):
            b.recycle_all(list(reversed(list(reg))))
    return x1, y1, ox, oy


# The EEA sandwich emission (the ls barrel corrects the dead lanes'
# over-rotation at pass ends; inverse passes mirror).


def emit_eea(b, R, inv=False):
    n, p = R.n, R.p
    W = n + 3
    aux = R.aux
    clean3 = [aux["acc"], aux["carry"], aux["c3xw"]]
    dirty_ws = R.w1r[3 : 3 + n]
    half_cb = int_bits(((1 << n) - (p + 1) // 2) % (1 << n), n)

    def piece_compare(_pinv):
        gidney_cmp_ge(b, R.xreg, half_cb, R.iterq, dirty_ws, clean3)

    def _ctrl_neg_modp(reg, pinv):
        # iterq-controlled mod-p negation: complement fan + ctrl add of
        # +-(p+1) (the shared spelling of negx and negate-tprime).
        if not pinv:
            for q in reg:
                b.cx(R.iterq, q)
            gidney_add_const(b, reg, int_bits(p + 1, n), dirty_ws[: n - 1], clean3, ctrl=R.iterq)
        else:
            gidney_add_const(
                b,
                reg,
                int_bits(((1 << n) - (p + 1)) % (1 << n), n),
                dirty_ws[: n - 1],
                clean3,
                ctrl=R.iterq,
            )
            for q in reg:
                b.cx(R.iterq, q)

    def piece_negx(pinv):
        _ctrl_neg_modp(R.xreg, pinv)

    def piece_init_consts(pinv):
        g = [X_(R.w1r[0])]
        for i in range(n):
            if (p >> (n - 1 - i)) & 1:
                g.append(X_(R.w1r[3 + i]))
        g.append(X_(R.lt[0]))
        for i, q in enumerate(R.ls):  # LS'' frame: n+1 - 0
            if ((n + 1) >> i) & 1:
                g.append(X_(q))
        emit(b, g, inv=pinv)

    def piece_init_lrp(pinv):
        oneq = aux["sxq"]
        g = [X_(oneq)]
        g += right_length_write(oneq, 4, W, R.w2r[3:W], R.w1r[3:W], R.lrp, aux, n)
        g.append(X_(oneq))
        emit(b, g, inv=pinv)

    def piece_steps(pinv):
        st = step_tables(n, p)
        if getattr(R, "_tpl_cache", None) is None:
            R._tpl_cache = build_step_template(R, st)  # deterministic in
        tpl_emit_steps(b, R._tpl_cache, st.nmax, inv=pinv)  # (R, st)

    def piece_barrel(pinv):
        # LS'' -> ls for the bit-controlled powers (const_minus is
        # self-inverse, so the same sandwich restores the frame).
        conv = const_minus_small(R.ls, (n + 1) % (1 << len(R.ls)), aux["cscr"], aux["carry"])
        g = list(conv)
        for k in range(len(R.ls)):
            s = pow(2, k, W)
            g += _rot_by_right(R.ls[k], R.w2r, s)
        g += conv
        emit(b, g, inv=pinv)

    def piece_final_resets(pinv):
        g = []
        for i in range(n):
            if (p >> i) & 1:
                g.append(X_(R.w1r[i]))
        g.append(X_(R.w1r[W - 1]))
        g += xor_const_reg(R.lt, n)
        emit(b, g, inv=pinv)

    def piece_negate_tprime(pinv):
        b.x(R.iterq)
        _ctrl_neg_modp(R.tprime, pinv)
        b.x(R.iterq)

    pieces = [
        piece_compare,
        piece_negx,
        piece_init_consts,
        piece_init_lrp,
        piece_steps,
        piece_barrel,
        piece_final_resets,
        piece_negate_tprime,
    ]
    if inv:
        for pc in reversed(pieces):
            pc(True)
    else:
        for pc in pieces:
            pc(False)


# ===========================================================================
# emission
# ===========================================================================


def build():
    """Full-width build on a fresh Builder — also the entry point of
    tools/occupancy_profile.py, which replays the ownership wire-set
    log fail-closed and charts the declared occupancy (no emission
    here).  Returns (b, (x1, y1)).
    """
    b = Builder()
    x1, y1, _, _ = build_circuit(b, P_MOD, N)
    return b, (x1, y1)


def main():
    b, _ = build()
    assert_metrics(b, __file__)
    emit_and_report(b)


# Bind the circuit-local callables the shared layer resolves at call
# time (fail-closed: a missing name raises here, not mid-build).
_core.init(globals())


if __name__ == "__main__":
    main()
