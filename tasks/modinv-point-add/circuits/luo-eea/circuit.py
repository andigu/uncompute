#!/usr/bin/env python3
"""modinv-point-add: baseline reconstruction of the space-efficient EEA
affine point addition of Luo et al. (2026), arXiv:2607.13816 ("the paper"
in the comments below).  The circuit is built from the paper's text; README.md
in this folder states and prices its implementation choices.  It is
certified at 69,529,757 Toffoli x 838 qubits.  It adds a per-shot
classical point Q to a quantum point in place (|P> -> |P+Q>), Q delivered
on the Bit-kind input registers (2/3): the const_modp cbspecs are free
classical bit networks of the input bits, and their loads ride as bit
conditions (CX under flagq promotes to a full Toffoli).  README.md also
holds the accounting of published against certified cost.

Shared code: ../_shared/luo_core.py (the arithmetic/template layer,
loaded by path; it loads ../_shared/clsbits.py) — the certification fingerprint
covers this file alone.

RUN: uv run python circuit.py — build ~9 s + emit ~1 s to
$UNCOMPUTE_OPS_PATH (else ./ops.bin).
     python tools/eval.py modinv-point-add --circuit <path to this
     file> — certification by the evaluator.
"""

import pathlib

from uncompute import Builder
from uncompute.testing import assert_metrics, emit_and_report, load_contract, load_module

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
_core = load_module(
    pathlib.Path(__file__).resolve().parent.parent / "_shared" / "luo_core.py", "luo_core"
)

# The shared bodies, re-exported: circuit-local code reads them as plain
# globals of this module.
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
_span_tables = _core._span_tables
declare_point_io = _core.declare_point_io
point_cbspecs = _core.point_cbspecs
ctrl_mod_add = _core.ctrl_mod_add
dbl_modp = _core.dbl_modp
const_modp = _core.const_modp
neg_modp = _core.neg_modp
emit_idiv = _core.emit_idiv
emit_imul = _core.emit_imul

# ---------------------------------------------------------------- instance
N = 256

# The addend is per-shot classical input data (grader.py): registers
# 2/3 are 256-wide Bit-kind inputs carrying Q.x, Q.y.  The curve
# constants below come from the task contract (grader-asserted at
# its own import, fail-closed).
_contract = load_contract(__file__)
P_MOD = int(_contract["p"], 16)
GX = int(_contract["gx"], 16)
GY = int(_contract["gy"], 16)
ORDER = int(_contract["n_order"], 16)


def len_width(n):
    """Length-register width: floor(log2 n)+1, bumped so n+3 is encodable
    (plain-value encoding; the bump only matters for the tiny selftest n).
    """
    return max(n.bit_length() - 1 + 1, (n + 3).bit_length())


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
# The search is finite without a heuristic width cap.  If a chain has k
# quotients and m total quotient bits, its leading continuant K obeys
#
#     K >= product(q_i) >= 2^(m-k),       K >= F_(k+1).
#
# For K < 2^n this gives m-k <= n-1; the Fibonacci inequality bounds k.
# Moreover every q_i <= K < 2^n, so every quotient width is at most n.


def _finite_dp_limits(n):
    """Return the proved (unit-count, quotient-width) limits for n-bit K."""
    assert n >= 2
    pmax = (1 << n) - 1
    # Largest j with F_j <= pmax; then a k-quotient continuant satisfies
    # F_(k+1) <= K <= pmax, hence k <= j-1.
    fm2, fm1, j = 0, 1, 1
    while fm2 + fm1 <= pmax:
        fm2, fm1 = fm1, fm2 + fm1
        j += 1
    kmax = j - 1
    return (n - 1) + kmax, n


def _pareto_front(cands, limit):
    """Pareto-minimal front of (t, t_prev) candidate pairs: ascending t,
    strictly descending t_prev.  If limit is given, states whose leading
    continuant already exceeds it are infeasible and are discarded.
    """
    if limit is not None:
        cands = [c for c in cands if c[0] <= limit]
    cands.sort()
    front = []
    best_tp = None
    for t, tp in cands:
        if best_tp is None or tp < best_tp:
            front.append((t, tp))
            best_tp = tp
    return front


def _min_continuants(m_cap, first_ge2, wcap=None, limit=None):
    """Pareto-minimal (t, t_prev) continuant pairs per unit count m.

    Units = sum of bitlen(q_i); appending quotient q = 2^(w-1) (only the
    smallest quotient per width matters for minima).  first_ge2 forbids q=1
    as the first quotient (EEA on x < p/2 has q_1 >= 2).  limit prunes
    states whose smaller component already exceeds it (their bit lengths
    are clamped by callers anyway).
    """
    if wcap is None:
        wcap = m_cap
    fronts = [[] for _ in range(m_cap + 1)]
    fronts[0] = [(1, 0)]
    for m in range(1, m_cap + 1):
        cands = []
        for w in range(1, min(wcap, m) + 1):
            q = 1 << (w - 1)
            src = fronts[m - w]
            if m - w == 0 and first_ge2 and w < 2:
                continue
            for t, tp in src:
                cands.append((q * t + tp, t))
        if not cands:
            continue
        fronts[m] = _pareto_front(cands, limit)
    return fronts


_DPCACHE = {}


def _dp_tables(n):
    """Exact extremal tables for width-n runs: min_t_len[m] (min bitlen of
    t_j after m units) and min_p[m] (min possible p for a full m-unit run,
    both end quotients >= 2; None if no such run).
    """
    if n in _DPCACHE:
        return _DPCACHE[n]
    m_cap, wcap = _finite_dp_limits(n)
    limit = (1 << n) - 1
    # Build quotient chains in reverse order.  Excluding width one at the
    # first transition enforces q_last >= 2.  A completed state additionally
    # requires width >= 2 on the final transition, enforcing q_1 >= 2.
    fronts = [[] for _ in range(m_cap + 1)]
    min_p = [None] * (m_cap + 1)
    fronts[0] = [(1, 0)]
    for m in range(1, m_cap + 1):
        cands, cands2 = [], []
        for w in range(1, min(wcap, m) + 1):
            q = 1 << (w - 1)
            if m - w == 0 and w < 2:
                continue
            for t, tp in fronts[m - w]:
                c = (q * t + tp, t)
                cands.append(c)
                if w >= 2:
                    cands2.append(c)
        if cands2:
            min_p[m] = min(t for (t, _) in cands2)
        if not cands:
            continue
        fronts[m] = _pareto_front(cands, limit)
    min_t_len = [min((t for (t, _) in fronts[m]), default=None) for m in range(m_cap + 1)]
    min_t_len = [v.bit_length() if v is not None else None for v in min_t_len]
    out = {"m_cap": m_cap, "wcap": wcap, "min_t_len": min_t_len, "min_p": min_p}
    _DPCACHE[n] = out
    return out


_NMAXCACHE = {}


def nmax_steps(n, p=None):
    """Certified step budget for an n-bit modulus (or the specific p)."""
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
    # Cap-slack guard: the extremal run must sit strictly inside the DP
    # horizon, with the whole tail above it infeasible.  A clipped
    # horizon would under-budget the run -- the corner that spoils the
    # HMR phase fixup (hard sign failure), so it is asserted, not prose.
    assert m_max < tab["m_cap"], (n, m_max, tab["m_cap"])
    assert all(
        tab["min_p"][m] is None or tab["min_p"][m] > lim for m in range(m_max + 1, tab["m_cap"] + 1)
    )
    # single-quotient runs (x=1, q=p) consume bitlen(p) = n units
    m_max = max(m_max, n)
    out = 4 * m_max
    _NMAXCACHE[key] = out
    return out


_WINCACHE = {}


def active_windows(n, T):
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
    # n+3-lrp-ls, provably <= floor(T/4)+2.  One window serves both phases.
    # Cap n+1 (v3's): the t-ops fire only with P1 = 1, never in the
    # residual regime (P1 = P2 = 0 there).  Phase 3: lt <= n (t < p).
    # Phase 4: lrp >= 1 (alive) and ls >= 1 (the post-shift decrement
    # follows), so n+3-lrp-ls <= n+1.
    k3 = 1
    K3 = min(T // 4 + 2, n + 1)
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
    # K4 bounds B = n+3-lrp; the lt update runs only on an alive
    # iteration end (lrp >= 1), so the cap is n+2 (v3's).
    K4 = min(T // 4 + 3, n + 2)
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


# ===========================================================================
# SECTION B -- loop-compressed emission infrastructure (op stream).
#
# Emission targets the stock uncompute.Builder and ends with b.emit()
# (UNCOMPUTE_OPS_PATH FIFO when the evaluator drives us, else ./ops.bin).
# Expanded, the circuit is ~260M ops; the compact spelling keeps the
# recorded program (and the stream) program-sized:
#
#   * The NMAX = nmax_steps(n, p) EEA steps are ONE recorded step body
#     inside b.repeat(NMAX).  Every step-dependent structure -- the
#     active-window unary-iteration trees, the 4|T end-of-iteration block,
#     the telescoping-XOR write patterns -- is data-gated by per-(slot,
#     step) class tables (op-stream DATA segments), so the expansion at step
#     T is EXACTLY the gate sequence a pruned per-step generator
#     produces, plus PASS-through CX
#     pairs where the pruned tree contracts a single-child chain
#     (Clifford-only; executed Toffoli is identical -- see _UShape).  The
#     inverse-EEA step loop is the mirrored template with the step index
#     reversed ((NMAX-1-s), an affine gate index into the same tables).
#   * The App.-A Horner multiplier records one iteration inside
#     b.repeat(n); the first/last-iteration doubling skip is table-gated
#     and the per-iteration control bit is an affine operand (mul_modp).
#   * Everything else (wrapper arithmetic, Gidney ripples outside the
#     Horner loop) is program-sized already and stays flat.
#
# The class tables (ClassTable), the tree shape (_UShape), the unary/dual
# templates and the prune/emit pair live in ../_shared/luo_core.py.
# ===========================================================================


def tpl_expand(items, T, inv=False):
    """Python-side expansion of a classed template at step T: the vocab ops
    whose class bit is set (None = always on).  This is the test oracle for
    the Builder-loop emission of the same template.
    """
    seq = tpl_inv(items) if inv else items
    return [g for (g, cls) in seq if cls is None or cls[0].get(T, cls[1])]


def tpl_range_scan(shape, idx, root, path, racc, leaf_fn, sense, order, node_cls, in_cls, base_cls):
    """Classed template of range_scan: leaf_fn(j) -> classed items; the
    accumulator toggle CX(e, racc) carries in_cls(j); the root seeding CX
    carries base_cls (the enclosing block's own gate, e.g. the 4|T class).
    """
    out = []
    if sense == "leq":
        if order == "inc":
            out.append((CX_(root, racc), base_cls))

            def lf(j, e):
                return leaf_fn(j) + [(CX_(e, racc), in_cls(j))]

            out += tpl_unary(shape, idx, root, path, lf, "inc", node_cls)
        else:

            def lf(j, e):
                return [(CX_(e, racc), in_cls(j))] + leaf_fn(j)

            out += tpl_unary(shape, idx, root, path, lf, "dec", node_cls)
            out.append((CX_(root, racc), base_cls))
    else:
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


# ===========================================================================
# SECTION C -- compile-time constants (curve-derived; the per-shot addend
# enters only through the Bit-kind input registers and their cbspec
# networks in SECTION H).
# ===========================================================================

# ===========================================================================
# SECTION D -- gate-list vocabulary and primitives.
#
# Measurement outcomes: HMR returns an Outcome handle, read exactly once
# (by its conditioned Z/CZ fixup) and dead afterwards. Outcome ids are
# minted per RECORD site (one id per loop-body HMR regardless of iteration
# count), so the recorded site count IS the id count -- what keeps this
# 21.7M-measurement circuit under the interpreter's 2^24 bit-address cap
# (asserted in fullwidth_emit: 68,452 ids at full width, measured; a
# fresh id per expanded measurement would blow the cap).
#
# Blocks are plain lists of tuples.  inv_list() produces the exact inverse:
# reversal, with the measurement-based AND pair swapped ("and+" <-> "and-").
# All EEA-internal logic uses this vocabulary so EEA-dagger is a literal
# mirror; the Gidney classical-quantum arithmetic (which measures mid-block)
# is emitted functionally with an explicit inverse instead.  The vocabulary,
# emit_op, the uncontrolled cells and the shared small-register helpers
# live in ../_shared/luo_core.py.
# ===========================================================================


# --- controlled Cuccaro cells (paper Fig. 11, wires: s=enable, a=target,
#     b=addend, c=shared carry; w = one clean C3X scratch) ------------------
# local: gadgets.arith maj/uma are the uncontrolled 1-Toffoli cells; these
# carry the location control s through a clean-ancilla C3X (3 Toffoli) --
# the realization behind the paper's 11(K-k)/9(K-k) block prices.


def c3x(s, a, b, t, w):
    """t ^= s&a&b with one clean scratch (3 Toffoli, the paper's cell
    realization; the scratch returns to |0>).
    """
    return [CCX_(a, b, w), CCX_(s, w, t), CCX_(a, b, w)]


# --- small-register arithmetic (length registers) ---------------------------


def inc_small(reg, anc):
    """reg += 1 (mod 2^w), uncontrolled.  anc: w-2 clean scratch.
    w-2 Toffoli: carry ANDs over the low bits, each released by
    measurement right after the bit above it flips (its own inputs are
    still unflipped then).
    """
    w = len(reg)
    if w == 0:
        return []
    if w == 1:
        return [X_(reg[0])]

    def c(i):  # wire holding the carry into bit i
        return reg[0] if i == 1 else anc[i - 2]

    g = [ANDP(c(i - 1), reg[i - 1], anc[i - 2]) for i in range(2, w)]
    for i in range(w - 1, 1, -1):
        g.append(CX_(c(i), reg[i]))
        g.append(ANDM(c(i - 1), reg[i - 1], anc[i - 2]))
    g += [CX_(reg[0], reg[1]), X_(reg[0])]
    return g


def add_const_inc(reg, val, anc):
    """reg += val (mod 2^w) as one incrementer per set bit of val
    (reg[b:] += 1 for each bit b): sum over set bits of (w-b-2)
    Toffoli.  anc: w-2 clean scratch.
    """
    w = len(reg)
    val %= 1 << w
    g = []
    for bit in range(w):
        if (val >> bit) & 1:
            g += inc_small(reg[bit:], anc)
    return g


def const_minus_inc(reg, C, anc):
    """reg <- (C - reg) mod 2^w (an involution): ~reg + C + 1."""
    return [X_(q) for q in reg] + add_const_inc(reg, C + 1, anc)


# --- pruned unary iteration over a promised interval (Sec. 4.4) -------------
# local: gadgets.qrom walk/pruned_walk descend on transient and_compute
# nodes; this is the paper's promised-interval iteration on persistent path
# registers -- the same M-1 node Toffoli for M labels -- kept gate-list so
# _UShape can class-gate the identical skeleton per step (SECTION E2).


def _split_bit(labels):
    lo, hi = labels[0], labels[-1]
    x = lo ^ hi
    return x.bit_length() - 1


def unary_tree(labels, idx, root, path, leaf_fn, order="inc"):
    """Pruned unary iteration: leaf_fn(j, e_j) -> gates, e_j fires iff the
    idx register equals j AND root=1 (for register values inside the
    promised label set; out-of-range values only fire leaves when root=1,
    so callers must keep root=0 whenever the promise may fail).
    """
    g = []

    def rec(labs, ctrl, depth):
        if len(labs) == 1:
            g.extend(leaf_fn(labs[0], ctrl))
            return
        bit = _split_bit(labs)
        z = [x for x in labs if not (x >> bit) & 1]
        o = [x for x in labs if (x >> bit) & 1]
        h = path[depth]
        first, second = (z, o) if order == "inc" else (o, z)
        g.append(ANDP(ctrl, idx[bit], h, 0, 1 if first is z else 0))
        rec(first, h, depth + 1)
        g.append(CX_(ctrl, h))
        rec(second, h, depth + 1)
        g.append(ANDM(ctrl, idx[bit], h, 0, 1 if second is z else 0))

    rec(list(labels), root, 0)
    return g


def range_scan(labels, idx, root, path, racc, leaf_fn, sense, order):
    """Visit labels in `order`, providing leaf_fn(j, racc) with racc =
    root AND [j <= B] (sense='leq', B = idx value) or root AND [j >= B]
    (sense='geq').  Promise: B inside [labels[0], labels[-1]].
    racc starts and ends |0>.
    """
    g = []
    if sense == "leq":
        if order == "inc":
            g.append(CX_(root, racc))  # B >= k promised

            def lf(j, e):
                return leaf_fn(j, racc) + [CX_(e, racc)]

            g += unary_tree(labels, idx, root, path, lf, "inc")
        else:

            def lf(j, e):
                return [CX_(e, racc)] + leaf_fn(j, racc)

            g += unary_tree(labels, idx, root, path, lf, "dec")
            g.append(CX_(root, racc))
    else:
        if order == "dec":
            g.append(CX_(root, racc))  # B <= K promised

            def lf(j, e):
                return leaf_fn(j, racc) + [CX_(e, racc)]

            g += unary_tree(labels, idx, root, path, lf, "dec")
        else:

            def lf(j, e):
                return [CX_(e, racc)] + leaf_fn(j, racc)

            g += unary_tree(labels, idx, root, path, lf, "inc")
            g.append(CX_(root, racc))
    return g


# --- length-update circuits (Sec. 4.4, telescoping XOR) ----------------------


def _uz_chain_gates(j, k, K, bits, dirty, a_tmp, racc, upper):
    """One leaf of the zero-map core: compute a_j = racc & bits_j into a_tmp,
    apply the chain Toffoli (or base CNOT), release a_tmp.
    """
    g = [ANDP(racc, bits[j - k], a_tmp)]
    if upper:
        tgt, nb = (dirty[j - k], dirty[j - k + 1]) if j < K else (dirty[j - k], None)
    else:
        tgt, nb = (dirty[j - k], dirty[j - k - 1]) if j > k else (dirty[j - k], None)
    g.append(X_(a_tmp))
    if nb is None:
        g.append(CX_(a_tmp, tgt))
    else:
        g.append(CCX_(a_tmp, nb, tgt))
    g.append(X_(a_tmp))
    g.append(ANDM(racc, bits[j - k], a_tmp))
    return g


def lower_zero_map(root, k, K, bits, dirty, areg, aux, boundary):
    """dirty[j] ^= y_j, y_j = AND_{h=k..j} not(root & [h>=A] & bits_h)."""
    labels = list(range(k, K + 1))
    path, racc, a_tmp = aux["pathA"], aux["acc"], aux["c3xw"]
    g = []
    if boundary:

        def fwd(j, rq):
            return _uz_chain_gates(j, k, K, bits, dirty, a_tmp, rq, False)

        g += range_scan(labels, areg, root, path, racc, fwd, "geq", "dec")

        def bwd(j, rq):
            if j <= k:
                return []
            return _uz_chain_gates(j, k, K, bits, dirty, a_tmp, rq, False)

        g += range_scan(labels, areg, root, path, racc, bwd, "geq", "inc")
    else:
        for j in range(K, k - 1, -1):
            g += _uz_chain_gates(j, k, K, bits, dirty, a_tmp, root, False)
        for j in range(k + 1, K + 1):
            g += _uz_chain_gates(j, k, K, bits, dirty, a_tmp, root, False)
    return g


def right_length_write(root, k, K, bits, dirty, areg, target, aux, n, boundary=True):
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
    g += lower_zero_map(root, k, K, bits, dirty, areg, aux, boundary)
    g += dw
    g += lower_zero_map(root, k, K, bits, dirty, areg, aux, boundary)
    return g


# ===========================================================================
# SECTION E -- the EEA step and Algorithm-1 wrapper.
# ===========================================================================


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


def tree_depth(L):
    """Return the path-register length for the pruned unary iteration over [1, L].

    The register holds one path qubit per node level of the _UShape tree.
    (The core's unary_depth(L) = ceil(log2(L-1)) + 1 over-counts by one
    when L-1 is just above a power of two: 10 against 9 at L = 259.)
    """
    shape = _UShape(L)

    def levels(ref):
        if ref[0] == "leaf":
            return 0
        _, _, _, z, o = shape.nodes[ref[1]]
        return 1 + max(levels(z), levels(o))

    return max(1, levels(shape.root))


def mk_eea_aux(b, n, p):
    wl = len_width(n)
    wls = ls_width(n, p)
    d = tree_depth(n + 3)
    aux = {
        "pathA": b.qubits(d),
        # pathB: the dual trees never open to the single-tree
        # depth d, so max(wl, wls - 1) slots suffice (the
        # selftest ladder covers n = 6/8; fullwidth_emit
        # asserts the peak).
        "pathB": b.qubits(max(wl, wls - 1)),
        "acc": b.qubit(),
        "carry": b.qubit(),
        "c3xw": b.qubit(),
        "ct3": b.qubit(),
        "alv": b.qubit(),
    }
    aux["cscr"] = (aux["pathA"] + aux["pathB"])[:wls]
    return aux


def _rot_right2(ctrl, reg):
    """Build a controlled cyclic rotation right by two (cycle decomposition)."""
    m = len(reg)
    g = []
    if m <= 2:
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
            j = (j + 2) % m
        pivot = cyc[0]
        for j in cyc[1:]:
            g += cswap(ctrl, reg[pivot], reg[j])
    return g


# --- twin-spelling shared preludes -------------------------------------------
# A pruned per-step generator and the loop template (build_step_template)
# must spell these preludes identically; emitting each from one place
# keeps the twins from drifting.


def r_endpoint_prep(ltreg, lqreg, lsreg, aux, n):
    """Endpoint transforms for the r-ops window: lq <- lt+lq+2 (holds L),
    ls <- n+3-ls (holds R).  Callers close with inv_list.
    """
    return r_endpoint_prep_lq(ltreg, lqreg, aux) + r_endpoint_prep_ls(lsreg, aux, n)


def r_endpoint_prep_lq(ltreg, lqreg, aux):
    """Build the L half of r_endpoint_prep: lq <- lt+lq+2."""
    return add_gid(ltreg, lqreg, aux["pathB"]) + add_const_inc(lqreg, 2, aux["pathB"])


def r_endpoint_prep_ls(lsreg, aux, n):
    """Build the R half of r_endpoint_prep: ls <- n+3-ls (an involution)."""
    return const_minus_inc(lsreg, n + 3, aux["pathB"])


def swap_endpoint_prep(ltreg, lqreg, p1q, aux):
    """Endpoint transform for the merged swap: lq <- lt+lq+1+[Phase1=0]
    (holds J, choice D-swap).  Callers close with inv_list.
    """
    carry = aux["carry"]
    prep = add_small(ltreg, lqreg, carry)
    prep += add_const_small(lqreg, 1, aux["cscr"], carry)
    # +1 more iff Phase1 = 0
    prep += [X_(p1q)]
    prep += ctrl_inc(lqreg, p1q, aux["pathB"])
    prep += [X_(p1q)]
    return prep


def ls_update(ls, p1, p2, aux, pre):
    """Build the step's shift-count update, fused: Algorithm 3 adds 1 to ls and,
    in phase 2 / phase 4, subtracts 2, on the P1 = 0 side before the
    r-ops (pre=True) and on the P1 = 1 side after the t-ops (pre=False).
    The net is ls += c*(1 - 2*P2), c = [P1 = 0] or [P1 = 1]: one
    controlled increment inside P2-controlled complement fans
    (~(~ls + c) = ls - c).
    """
    fan = [CX_(p2, q) for q in ls]
    flip = [X_(p1)] if pre else []
    return fan + flip + ctrl_inc(ls, p1, aux["pathB"]) + flip + fan


def swap_from_r_prep(lq, p1, aux):
    """Merged-swap endpoint from the open r-side frame: lq holds
    L = lt+lq+2, so lq <- L - P1 = lt+lq+1+[Phase1=0] = J (choice
    D-swap).  Callers close with inv_list.
    """
    return ctrl_dec(lq, p1, aux["pathB"])


def lq_update_from_j(lq, p2, aux):
    """Build the step's lq update (Algorithm 3: lq -= [P1 and not P2],
    lq += [not P1 and P2]) applied from J = L - P1: the net from J is
    lq += P2, one controlled increment.  It closes swap_from_r_prep;
    the r-side frame closes after it.
    """
    return ctrl_inc(lq, p2, aux["pathB"])


def t_endpoint_prep(lt, lrp, rreg, p2, aux):
    """lt <- (P2 ? n+3-lrp-ls : lt+1), the phase-selected right endpoint
    of the t-side ops (choice D-K3), computed in place in the length
    registers while the r-side frame still holds R = n+3-ls in rreg:
    lt <- lt+1 and lrp <- R-lrp (both bijections), then a P2-controlled
    swap of lt and lrp.  R-lrp is spelled ~(lrp + ~R).  The t-blocks
    index their unary tree on lt and read no other length register.
    Callers close with inv_list.
    """
    wl = len(lt)
    assert len(lrp) == wl
    rl = rreg[:wl]
    tprep = inc_small(lt, aux["pathB"])
    tprep += [X_(q) for q in rl]
    tprep += add_gid(rl, lrp, aux["pathB"])
    tprep += [X_(q) for q in rl] + [X_(q) for q in lrp]
    for i in range(wl):
        tprep += cswap(p2, lt[i], lrp[i])
    return tprep


def phase_update_gates(lq, lrp, ls, p1, p2, sign, ct3, aux):
    """Build the end-of-step phase update (verbatim Algorithm 3; residual
    ls != 0 keeps it inert, choice D-freeze).
    """
    glq, cq_lq = and_ladder([(q, 1) for q in lq], aux["pathA"])
    g = list(glq)  # cq_lq = [lq == 0]
    grp, cq_rp = and_ladder([(q, 1) for q in lrp], aux["pathB"])
    g += grp  # cq_rp = [lrp == 0]
    g.append(ANDP(cq_lq, cq_rp, ct3, 0, 1))  # lq=0 and lrp>0
    # P2 ^= ct3 & (Sign ^ P1), with Sign itself holding Sign ^ P1
    g += [CX_(p1, sign), CCX_(ct3, sign, p2), CX_(p1, sign)]
    g.append(CCX_(ct3, p2, sign))
    g.append(ANDM(cq_lq, cq_rp, ct3, 0, 1))
    g += inv_list(grp)
    g += inv_list(glq)
    gls, cq_ls = and_ladder([(q, 1) for q in ls], aux["pathA"])
    g += gls
    g += [CX_(cq_ls, p1), CX_(cq_ls, p2)]
    g += inv_list(gls)
    return g


def iter_end_cond_open(lq, ls, ct3, aux):
    """Open the 4|T condition: ct3 <- [lq == 0 and ls == 0], both
    ladders released at once by recompute (lq and ls are untouched by
    the gated block, so the closing recompute is exact).  Callers close
    with inv_list.
    """
    glq, cq2 = and_ladder([(q, 1) for q in lq], aux["pathA"])
    gls, cq3 = and_ladder([(q, 1) for q in ls], aux["pathB"])
    return glq + gls + [ANDP(cq2, cq3, ct3, 0, 0)] + inv_list(gls) + inv_list(glq)


# ===========================================================================
# SECTION E2 -- the loop-compressed step: class tables + step template.
#
# One template (per EEA register binding) spells ALL NMAX steps; the
# per-step differences -- exactly the active windows of Sec. 4.5 and the
# 4|T end-of-iteration block -- live in ClassTable bits.  The expansion at
# step T is the pruned per-step gate sequence with PASS-through CX pairs
# added where the
# pruned tree contracts chains (Toffoli-identical; see _UShape).
# ===========================================================================


class StepTables:
    """All per-(slot, step) gating tables for one (n, p) instance.  Shared
    by every EEA run (forward and inverse, both register bindings): the
    window schedule depends only on (n, T).
    """

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.nmax = nmax_steps(n, p)
        W = n + 3
        self.W = W
        self.wl = len_width(n)
        self.shape_W = _UShape(W)  # r-ops / length scans: labels [1, n+3]
        self.shape_S = _UShape(W - 1)  # swap / t scans: labels [1, n+2]
        NM = self.nmax
        nW, nS = self.shape_W.n_nodes, self.shape_S.n_nodes
        wl = self.wl
        self.r = ClassTable("r", NM, 2 * nW + W)
        self.swap = ClassTable("swap", NM, 2 * nS + (W - 1))
        self.t = ClassTable("t", NM, 2 * nS + (W - 1))
        # length-update roles: nodes + 3 leaf classes (in / interior / edge)
        self.lent = ClassTable("lent", NM, 2 * nW + 3 * W)
        self.lenrp = ClassTable("lenrp", NM, 2 * nW + 3 * W)
        self.lenrpE = ClassTable("lenrpE", NM, 2 * nW + 3 * W)
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
        return 2 * shape.n_nodes + cls * shape.L + (j - 1)

    def aux_dw(self, j, i):
        return self.wl + (j - 1) * self.wl + i

    # ---- per-step activation --------------------------------------------
    def _fill(self):
        n, W, wl = self.n, self.W, self.wl
        mask = (1 << wl) - 1
        for T in range(1, self.nmax + 1):
            win = active_windows(n, T)
            self._mark_tree(self.r, self.shape_W, T, *win["r"])
            self._mark_tree(self.swap, self.shape_S, T, *win["swap"])
            self._mark_tree(self.t, self.shape_S, T, *win["t"])
            if T % 4:
                continue
            self.is4.set(T, 0)
            k4, K4 = win["len_t"]
            k5, K5 = win["len_rp"]
            self._mark_len(self.lent, T, k4, K4, upper=True)
            self._mark_len(self.lenrp, T, k5, K5, upper=False)
            self._mark_len(self.lenrpE, T, k5, W, upper=False)
            # hpw pattern: seed = K & mask; deltas j^(j-1) on (k, K],
            # boundary pattern k & mask at j == k (highest_position_write's
            # dw list order is descending j with the k-entry last).
            for i in range(wl):
                if (K4 >> i) & 1:
                    self.lentaux.set(T, i)
            for j in range(k4 + 1, K4 + 1):
                delta = (j ^ (j - 1)) & mask
                for i in range(wl):
                    if (delta >> i) & 1:
                        self.lentaux.set(T, self.aux_dw(j, i))
            for i in range(wl):
                if ((k4 & mask) >> i) & 1:
                    self.lentaux.set(T, self.aux_dw(k4, i))
            self._fill_rlw_aux(self.lenrpaux, T, k5, K5)
            self._fill_rlw_aux(self.lenrpEaux, T, k5, W)

    def _mark_tree(self, tbl, shape, T, k, K):
        def onn(nid, st):
            tbl.set(T, self.node_slot(nid, st))

        def onl(j):
            tbl.set(T, self.leaf_slot(shape, j, 0))

        shape.mark(k, K, onn, onl)

    def _mark_len(self, tbl, T, k, K, upper):
        shape = self.shape_W

        def onn(nid, st):
            tbl.set(T, self.node_slot(nid, st))

        def onl(j):
            tbl.set(T, self.leaf_slot(shape, j, 0))  # in-window
            if upper:
                if j < K:
                    tbl.set(T, self.leaf_slot(shape, j, 1))  # chain CCX
                else:
                    tbl.set(T, self.leaf_slot(shape, j, 2))  # j == K: CX
            else:
                if j > k:
                    tbl.set(T, self.leaf_slot(shape, j, 1))  # chain CCX
                else:
                    tbl.set(T, self.leaf_slot(shape, j, 2))  # j == k: CX

        shape.mark(k, K, onn, onl)

    def _fill_rlw_aux(self, tbl, T, k, K):
        """rlw pattern: seed = val(k); deltas val(j)^val(j+1) on [k, K);
        val(K) at j == K (right_length_write's dw order is ascending j,
        K-entry last).
        """
        n, wl = self.n, self.wl
        mask = (1 << wl) - 1

        def val(pos):
            return (n + 4 - pos) & mask

        for i in range(wl):
            if (val(k) >> i) & 1:
                tbl.set(T, i)
        for j in range(k, K):
            delta = val(j) ^ val(j + 1)
            for i in range(wl):
                if (delta >> i) & 1:
                    tbl.set(T, self.aux_dw(j, i))
        for i in range(wl):
            if (val(K) >> i) & 1:
                tbl.set(T, self.aux_dw(K, i))


# ---- block templates (classed mirrors of the SECTION D/E blocks) -----------


def _tpl_r_block(mode, sign_update, root, R, st, prep=True):
    """r_interval_block over the full shape, windows in st.r.  With
    prep=False the caller holds L in lq and R in ls (r_endpoint_prep
    open around both r blocks).
    """
    aux = R.aux
    n = R.n
    out = []
    prep = r_endpoint_prep(R.lt, R.lq, R.ls, aux, n) if prep else []
    out.extend((g, None) for g in prep)
    acc, carry, c3xw = aux["acc"], aux["carry"], aux["c3xw"]
    w1r, w2r = R.w1r, R.w2r
    cf = cell_uma_inv if mode == "sub" else cell_maj
    cs = cell_maj_inv if mode == "sub" else cell_uma

    def ncls(nid, s):
        return (st.r, st.node_slot(nid, s))

    def lin(j):
        return (st.r, st.leaf_slot(st.shape_W, j, 0))

    def leaf1(j, ea, eb):
        ops = [CX_(ea, acc)] + cf(acc, w1r[j - 1], w2r[j - 1], carry, c3xw) + [CX_(eb, acc)]
        return [(g, lin(j)) for g in ops]

    out += tpl_dual(st.shape_W, R.ls, R.lq, root, aux["pathA"], aux["pathB"], leaf1, "dec", ncls)
    if sign_update:
        out.append((CX_(carry, R.sign), None))

    def leaf2(j, ea, eb):
        ops = [CX_(eb, acc)] + cs(acc, w1r[j - 1], w2r[j - 1], carry, c3xw) + [CX_(ea, acc)]
        return [(g, lin(j)) for g in ops]

    out += tpl_dual(st.shape_W, R.ls, R.lq, root, aux["pathA"], aux["pathB"], leaf2, "inc", ncls)
    out.extend((g, None) for g in inv_list(prep))
    return out


def _tpl_swap_block(root, R, st, prep=True):
    """lc_swap_block over the full shape, windows in st.swap.  With
    prep=False the caller holds J in lq.
    """
    aux = R.aux
    out = []
    prep = swap_endpoint_prep(R.lt, R.lq, R.p1, aux) if prep else []
    out.extend((g, None) for g in prep)

    def ncls(nid, s):
        return (st.swap, st.node_slot(nid, s))

    def leaf(j, e):
        cls = (st.swap, st.leaf_slot(st.shape_S, j, 0))
        return [(g, cls) for g in cswap(e, R.sign, R.w1r[j - 1])]

    out += tpl_unary(st.shape_S, R.lq, root, aux["pathA"], leaf, "inc", ncls)
    out.extend((g, None) for g in inv_list(prep))
    return out


def _tpl_t_block(mode, sign_update, root, R, st):
    """t_prefix_block over the full shape, windows in st.t."""
    aux = R.aux
    acc, carry, c3xw = aux["acc"], aux["carry"], aux["c3xw"]
    w1r, w2r = R.w1r, R.w2r
    idx = R.lt  # holds the t endpoint while t_endpoint_prep is open
    cf = cell_uma_inv if mode == "sub" else cell_maj
    cs = cell_maj_inv if mode == "sub" else cell_uma
    out = [(CX_(root, acc), None)]

    def ncls(nid, s):
        return (st.t, st.node_slot(nid, s))

    def lin(j):
        return (st.t, st.leaf_slot(st.shape_S, j, 0))

    def leaf1(j, e):
        ops = cf(acc, w2r[j - 1], w1r[j - 1], carry, c3xw) + [CX_(e, acc)]
        return [(g, lin(j)) for g in ops]

    out += tpl_unary(st.shape_S, idx, root, aux["pathA"], leaf1, "inc", ncls)
    if sign_update:
        out.append((CX_(carry, R.sign), None))

    def leaf2(j, e):
        ops = [(CX_(e, acc), lin(j))]
        ops += [(g, lin(j)) for g in cs(acc, w2r[j - 1], w1r[j - 1], carry, c3xw)]
        return ops

    out += tpl_unary(st.shape_S, idx, root, aux["pathA"], leaf2, "dec", ncls)
    out.append((CX_(root, acc), None))
    return out


def _tpl_len_write(root, bits_w, dirty_w, idxreg, target, R, st, tbl, tba, up):
    """highest_position_write (up=True, scan boundary [j <= B], window
    [k4, K4]) or right_length_write (up=False, boundary [j >= A], window
    [k5, K5] / [k5, W]) over the full shape: seed/delta write patterns
    in `tba`, scan classes in `tbl` (4|T only).  idxreg holds the
    boundary register (B or A); the zero-map chain runs toward the
    window edge on the `up` side, and the edge leaf writes by CX where
    the interior writes by chain CCX.
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
        for j in range(W, 0, -1) if up else range(1, W + 1):
            for i in range(wl):
                out.append((CX_(dirty_w[j - 1], target[i]), (tba, st.aux_dw(j, i))))

    def ncls(nid, s):
        return (tbl, st.node_slot(nid, s))

    def inc(j):
        return (tbl, st.leaf_slot(st.shape_W, j, 0))

    def mid(j):
        return (tbl, st.leaf_slot(st.shape_W, j, 1))

    def edge(j):
        return (tbl, st.leaf_slot(st.shape_W, j, 2))

    def interior(j):
        return j < W if up else j > 1

    def nb(j):
        return dirty_w[j] if up else dirty_w[j - 2]

    def chain_fwd(j):
        ops = [(ANDP(racc, bits_w[j - 1], a_tmp), inc(j)), (X_(a_tmp), inc(j))]
        if interior(j):
            ops.append((CCX_(a_tmp, nb(j), dirty_w[j - 1]), mid(j)))
        ops.append((CX_(a_tmp, dirty_w[j - 1]), edge(j)))
        ops += [(X_(a_tmp), inc(j)), (ANDM(racc, bits_w[j - 1], a_tmp), inc(j))]
        return ops

    def chain_bwd(j):
        if not interior(j):
            return []  # bwd fires only strictly inside the window
        return [
            (ANDP(racc, bits_w[j - 1], a_tmp), mid(j)),
            (X_(a_tmp), mid(j)),
            (CCX_(a_tmp, nb(j), dirty_w[j - 1]), mid(j)),
            (X_(a_tmp), mid(j)),
            (ANDM(racc, bits_w[j - 1], a_tmp), mid(j)),
        ]

    def zmap():
        sense = "leq" if up else "geq"
        o = tpl_range_scan(
            st.shape_W,
            idxreg,
            root,
            path,
            racc,
            chain_fwd,
            sense,
            "inc" if up else "dec",
            ncls,
            inc,
            IS4,
        )
        o += tpl_range_scan(
            st.shape_W,
            idxreg,
            root,
            path,
            racc,
            chain_bwd,
            sense,
            "dec" if up else "inc",
            ncls,
            inc,
            IS4,
        )
        return o

    dw()
    out += zmap()
    dw()
    out += zmap()
    return out


def build_step_template(R, st):
    """Build the complete classed step body: tpl_expand(template, T) equals
    the pruned per-step gate sequence at step T plus PASS Cliffords, for
    every T in [1, NMAX].
    """
    n = R.n
    W = n + 3
    aux = R.aux
    p1, p2, sign = R.p1, R.p2, R.sign
    lt, lq, ls, lrp = R.lt, R.lq, R.ls, R.lrp
    w1r, w2r = R.w1r, R.w2r
    ct3 = aux["ct3"]
    # Condition temporaries ride idle path qubits: pathA[0..1] wherever
    # no tree is open, pathB[0..1] around the t-sub (its tree runs on
    # pathA alone).  The r-sub root is ct3; the r-add root is ct3 with
    # its two temporaries released before the dual trees open and
    # recomputed after.
    ct1, ct2 = aux["pathA"][0], aux["pathA"][1]
    tc1, tc2 = aux["pathB"][0], aux["pathB"][1]
    alv = aux["alv"]
    ladder_anc = aux["pathA"]
    IS4 = (st.is4, 0)
    out = []

    def A(gates, cls=None):
        out.extend((g, cls) for g in gates)

    # alive flag
    A(_zero_toggle(lrp, alv, ladder_anc))

    # ---- pre-shift ----
    A([ANDP(p1, alv, ct1, 1, 1)])
    g = []
    for i in range(W - 1):
        g += cswap(ct1, w2r[i], w2r[i + 1])
    A(g)
    A([ANDM(p1, alv, ct1, 1, 1)])
    A(ls_update(ls, p1, p2, aux, pre=True))
    A([ANDP(p1, p2, ct1, 1, 0), ANDP(ct1, alv, ct2, 0, 1)])
    A(_rot_right2(ct2, w2r))
    A([ANDM(ct1, alv, ct2, 0, 1)])
    A([ANDM(p1, p2, ct1, 1, 0)])

    # ---- r-side endpoints, open around both r blocks: lq holds L,
    # ls holds R (neither is read or written between the blocks) ----
    rprep_lq = r_endpoint_prep_lq(lt, lq, aux)
    rprep_ls = r_endpoint_prep_ls(ls, aux, n)
    A(rprep_lq + rprep_ls)

    # ---- location-controlled subtraction on r's ----
    A([ANDP(p1, alv, ct3, 1, 1)])
    out.extend(_tpl_r_block("sub", True, ct3, R, st, prep=False))
    A([ANDM(p1, alv, ct3, 1, 1)])

    # Sign toggles on [not P1 and P2]
    A([ANDP(p1, p2, ct1, 1, 0), CX_(ct1, sign), ANDM(p1, p2, ct1, 1, 0)])

    # ---- conditional addition back on r's ----
    racq = [ANDP(p2, sign, ct1, 0, 0), ANDP(p1, ct1, ct2, 1, 1)]
    A(racq + [ANDP(ct2, alv, ct3, 0, 1)] + inv_list(racq))
    out.extend(_tpl_r_block("add", False, ct3, R, st, prep=False))
    A(racq + [ANDM(ct2, alv, ct3, 0, 1)] + inv_list(racq))

    # release alive flag
    A(_zero_toggle(lrp, alv, ladder_anc))

    # ---- location-controlled swap (root P1 xor P2, held in P2) at
    # J = L - P1, from the still-open r-side frame ----
    jprep = swap_from_r_prep(lq, p1, aux)
    A(jprep)
    A([CX_(p1, p2)])
    out.extend(_tpl_swap_block(p2, R, st, prep=False))
    A([CX_(p1, p2)])
    A(lq_update_from_j(lq, p2, aux))
    A(inv_list(rprep_lq))

    # ---- t-side endpoint prep ----
    tprep = t_endpoint_prep(lt, lrp, ls, p2, aux)
    A(tprep)

    # ---- location-controlled subtraction on t's ----
    A([ANDP(p2, sign, tc1, 1, 0), ANDP(p1, tc1, tc2, 0, 1)])
    out.extend(_tpl_t_block("sub", False, tc2, R, st))
    A([ANDM(p1, tc1, tc2, 0, 1), ANDM(p2, sign, tc1, 1, 0)])

    A([CX_(p1, sign)])

    # ---- addition on t's with Sign update ----
    out.extend(_tpl_t_block("add", True, p1, R, st))
    A(inv_list(tprep))
    A(inv_list(rprep_ls))

    # ---- post-shift ----
    g = []
    for i in range(W - 1):
        g += cswap(p1, w2r[i], w2r[i + 1])
    A(g)
    A(ls_update(ls, p1, p2, aux, pre=False))
    A([ANDP(p1, p2, ct1, 0, 0)])
    A(_rot_right2(ct1, w2r))
    A([ANDM(p1, p2, ct1, 0, 0)])

    # ---- phase update ----
    A(phase_update_gates(lq, lrp, ls, p1, p2, sign, ct3, aux))

    # ---- end-of-iteration SWAP + length updates (whole block 4|T-gated) --
    cond_open = iter_end_cond_open(lq, ls, ct3, aux)
    A(cond_open, IS4)
    g = []
    for i in range(W):
        g += cswap(ct3, w1r[i], w2r[i])
    A(g, IS4)
    bprep = const_minus_inc(lrp, n + 3, aux["pathB"])
    A(bprep, IS4)
    out.extend(_tpl_len_write(ct3, w2r, w1r, lrp, lt, R, st, st.lent, st.lentaux, up=True))
    out.extend(_tpl_len_write(ct3, w1r, w2r, lrp, lt, R, st, st.lent, st.lentaux, up=True))
    A(inv_list(bprep), IS4)
    aprep = add_const_inc(lt, 2, aux["pathB"])
    A(aprep, IS4)
    out.extend(_tpl_len_write(ct3, w1r, w2r, lt, lrp, R, st, st.lenrp, st.lenrpaux, up=False))
    out.extend(_tpl_len_write(ct3, w2r, w1r, lt, lrp, R, st, st.lenrpE, st.lenrpEaux, up=False))
    A(inv_list(aprep), IS4)
    A([CX_(ct3, R.iterq)], IS4)
    A(inv_list(cond_open), IS4)
    return out


# ===========================================================================
# SECTION F -- the modular multiplier (paper App. A).  Its Gidney
# classical-quantum arithmetic (arXiv:2507.23079) and quantum-quantum
# ripple pieces live in ../_shared/luo_core.py.  They measure mid-block
# (HMR + conditioned Z), so they are emitted functionally with explicit
# inverses instead of via gate-list reversal.
# ===========================================================================


def _mul_step_tables(b, n):
    """Per-Builder DATA tables gating the Horner doubling: bit t of the
    first table is [t > 0] (forward: no doubling before the MSB add), of
    the second [t < n-1] (inverse: no doubling after the last).
    """
    cache = getattr(b, "_luo_mul_tables", None)
    if cache is None:
        cache = b._luo_mul_tables = {}
    if n not in cache:
        cache[n] = (
            b.declare_table([1 if t > 0 else 0 for t in range(n)]),
            b.declare_table([1 if t < n - 1 else 0 for t in range(n)]),
        )
    return cache[n]


def mul_modp(b, xc, y, out, p, n, ma, square=False, inv=False):
    """out ^= x*y mod p, Horner MSB->LSB (n ctrl-mod-adds + (n-1)
    doublings = 17n^2 + O(n log n) Toffoli, paper App. A), recorded as ONE
    iteration inside b.repeat(n): the doubling skip on the first (forward)
    / last (inverse) iteration is table-gated, and the per-iteration
    control bit is an affine operand of the loop index.  For square=True y
    is xc and each control bit is copied to a temp first (the control
    qubit is also a multiplicand operand, and gate operands must be
    distinct).  A control register that is not id-affine in the Horner
    order (the Work2-embedded t' -- one layout serving both endianness
    views as index maps: 3 extras + reversed X) is routed through the
    same temp with per-span gated copies -- 2 extra executed CX per
    iteration, 0 extra Toffoli (the pruned per-step spelling controls
    those gates on the t' qubit directly).
    """
    tq = ma["tq"]
    ma = dict(ma)
    ma["dirty"] = list(y) + [tq]  # doubling borrows n+1 dirty qubits
    tnot0, tnotl = _mul_step_tables(b, n)
    order = range(n - 1, -1, -1) if not inv else range(n)
    ids = [xc[i] for i in order]
    spans = _affine_spans(ids)
    direct = (not square) and len(spans) == 1
    span_tbls = _span_tables(b, n, spans) if len(spans) > 1 else None

    with b.repeat(n) as t:

        def ctrl_expr():
            (t0, _ln, base, stride) = spans[0]
            return base - t0 * stride + t * stride

        def copy_ctrl():
            if span_tbls is None:
                b.cx(ctrl_expr(), tq)
            else:
                for (t0, _ln, base, stride), tb in zip(spans, span_tbls):
                    with b.gated(tb, t):
                        b.cx(base - t0 * stride + t * stride, tq)

        if not inv:
            with b.gated(tnot0, t):
                dbl_modp(b, out, p, n, ma)
            if direct:
                ctrl_mod_add(b, ctrl_expr(), y, out, p, n, ma)
            else:
                copy_ctrl()
                ctrl_mod_add(b, tq, y, out, p, n, ma)
                copy_ctrl()
        else:
            if direct:
                ctrl_mod_add(b, ctrl_expr(), y, out, p, n, ma, inv=True)
            else:
                copy_ctrl()
                ctrl_mod_add(b, tq, y, out, p, n, ma, inv=True)
                copy_ctrl()
            with b.gated(tnotl, t):
                dbl_modp(b, out, p, n, ma, inv=True)


# ===========================================================================
# SECTION G -- the full EEA wrapper (Algorithm 1) and IDiv / IMul.
# ===========================================================================


def emit_eea(b, R, inv=False, flat_steps=False):
    """Forward (or exact inverse) space-efficient EEA on R: X-register value
    x -> x^{-1} mod p in R.tprime (little-endian), workspace and controls
    reset to constants; Gamma(x) = (Iter, ls) stays until the inverse run.

    flat_steps=True emits the unrolled template spelling (the per-step
    Python expansion of the same template) instead of the b.repeat(NMAX)
    loop -- the selftest oracle proving the loop spelling expands to the
    identical op sequence.
    """
    n, p = R.n, R.p
    W = n + 3
    aux = R.aux
    clean3 = [aux["acc"], aux["carry"], aux["c3xw"]]
    dirty_ws = R.w1r[3 : 3 + n]  # workspace field qubits as dirty
    half_cb = int_bits(((1 << n) - (p + 1) // 2) % (1 << n), n)

    def piece_compare(_pinv):
        # Iter toggles on [x >= (p+1)/2]  (self-inverse)
        gidney_cmp_ge(b, R.xreg, half_cb, R.iterq, dirty_ws, clean3)

    def _ctrl_negate(reg, pinv):
        """reg <- p - reg iff Iter: complement sweep + controlled +(p+1)
        (inverse = the mirrored pair with the complement constant).
        """
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
        _ctrl_negate(R.xreg, pinv)

    def piece_init_consts(pinv):
        g = [X_(R.w1r[0])]  # t = 1
        for i in range(n):  # r = p big-endian
            if (p >> (n - 1 - i)) & 1:
                g.append(X_(R.w1r[3 + i]))
        g.append(X_(R.lt[0]))  # lt = 1
        emit(b, g, inv=pinv)

    def piece_init_lrp(pinv):
        oneq = aux["alv"]  # idle outside the step loop
        g = [X_(oneq)]
        g += right_length_write(
            oneq, 4, W, R.w2r[3:W], R.w1r[3:W], R.lt, R.lrp, aux, n, boundary=False
        )
        g.append(X_(oneq))
        emit(b, g, inv=pinv)

    def piece_steps(pinv):
        # The loop-compressed step section: ONE recorded step body under
        # b.repeat(NMAX), per-step windows via class tables (SECTION E2).
        # The inverse run is the mirrored template on the reversed step
        # index -- exactly inv_list of each step, steps in reverse order.
        st = step_tables(n, p)
        items = build_step_template(R, st)
        if flat_steps:
            rng_T = range(1, st.nmax + 1)
            for T in reversed(rng_T) if pinv else rng_T:
                emit(b, tpl_expand(items, T, inv=pinv))
        else:
            tpl_emit_steps(b, items, st.nmax, inv=pinv)

    def piece_final_resets(pinv):
        g = []
        for i in range(n):  # terminal t = p (LE)
            if (p >> i) & 1:
                g.append(X_(R.w1r[i]))
        g.append(X_(R.w1r[W - 1]))  # terminal r = 1
        g += xor_const_reg(R.lt, n)  # lt = n -> 0
        emit(b, g, inv=pinv)

    def piece_negate_tprime(pinv):
        # if Iter = 0: t' <- p - t'
        b.x(R.iterq)
        _ctrl_negate(R.tprime, pinv)
        b.x(R.iterq)

    pieces = [
        piece_compare,
        piece_negx,
        piece_init_consts,
        piece_init_lrp,
        piece_steps,
        piece_final_resets,
        piece_negate_tprime,
    ]
    if inv:
        for pc in reversed(pieces):
            pc(True)
    else:
        for pc in pieces:
            pc(False)


def mk_mul_aux(aux):
    # flag/top ride idle path qubits between the EEAs
    return {
        "carry": aux["carry"],
        "flag": aux["pathA"][0],
        "top": aux["pathA"][1],
        "tq": aux["alv"],
        "clean3": [aux["acc"], aux["c3xw"], aux["ct3"]],
    }


# ===========================================================================
# SECTION H -- the full point-addition circuit (Fig. 13, ctrl dropped).
# ===========================================================================


def build_circuit(b, p, n):
    """Emit the point addition |P> -> |P+Q> into builder b, the addend Q
    read per shot from the Bit-kind input registers.  Returns
    (x1, y1, ox, oy): the target wire lists and the addend bit lists.
    """
    x1, y1, ox, oy = declare_point_io(b, n)  # registers 0-3

    # per-shot cbspecs (free classical bit networks of the input bits;
    # the compile-time builder's 0 < C < p contract is the input domain)
    S_qx, S_qy, S_3qx = point_cbspecs(b, ox, oy, p, n)

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
    flagq = aux["pathA"][0]  # idle outside the EEA step loop

    # OWNERSHIP (wire-axis annotation; claims and releases write wire_log
    # only, so they add no ops).  Every wire is claimed at birth, so the
    # anonymous pool is empty outside the init and teardown bookkeeping
    # bursts: the vocabulary ANDs write into persistent pre-allocated
    # targets, never and_compute transients (SECTION D, emit_op NOTE).
    # Set heads mirror the peak itemization (README Cost): x1 + y1 +
    # A + wext = 3n+6 field/work qubits, then len, ctrl, scr.  The s7
    # teardown releases every non-io set and frees its wires (one
    # unscored R each); only x1/y1 end owned and live.
    # tools/occupancy_profile.py replays the log fail-closed (0
    # violations).
    assert len(x1) + len(y1) + len(A) + len(w1x) + len(w2x) == 3 * n + 6
    b.claim("x1", x1)  # target_x; EEA x-lane inside Work2
    b.claim("y1", y1)  # target_y; EEA workspace in Rwy
    b.claim("A", A)  # product accumulator; workspace in Rws
    b.claim("wext/w1", w1x)  # Work1 positions 1..3
    b.claim("wext/w2", w2x)  # Work2 positions 1..3
    b.claim("len/lt", lt)  # length of t
    b.claim("len/lq", lq)  # length of q
    b.claim("len/ls", ls)  # shift count (keeps counting residually)
    b.claim("len/lrp", lrp)  # length of r'
    b.claim("ctrl", [p1, p2, sign, iterq])  # Phase1/Phase2/Sign/Iter
    b.claim("scr/path", aux["pathA"] + aux["pathB"])  # unary-tree paths,
    #                          ladder ancillas, borrowed small-reg scratch
    b.claim(
        "scr/flags", [aux["acc"], aux["carry"], aux["c3xw"], aux["ct3"], aux["alv"]]
    )  # accumulator / carry /
    #                          C3X ancilla / condition + alive flags

    Rws = EEARegs(n, p, x1, A, w1x, w2x, lt, lq, ls, lrp, p1, p2, sign, iterq, aux)
    Rwy = EEARegs(n, p, x1, y1, w1x, w2x, lt, lq, ls, lrp, p1, p2, sign, iterq, aux)

    with b.phase("s1-const-subs"):
        const_modp(b, x1, S_qx, flagq, y1, clean3, p, n, inv=True)
        const_modp(b, y1, S_qy, flagq, x1, clean3, p, n, inv=True)
    with b.phase("s2-idiv"):
        emit_idiv(b, x1, y1, A, Rws, Rwy, p, n, ma)
    with b.phase("s3-squ"):
        mul_modp(b, y1, y1, A, p, n, ma, square=True)
    with b.phase("s4-sub-squ"):
        oneq = aux["pathB"][0]  # idle between the EEAs (alv is tq)
        b.x(oneq)
        ctrl_mod_add(b, oneq, A, x1, p, n, ma, inv=True)
        b.x(oneq)
        const_modp(b, x1, S_3qx, flagq, y1, clean3, p, n)
    with b.phase("s5-squ-dag"):
        mul_modp(b, y1, y1, A, p, n, ma, square=True, inv=True)
    with b.phase("s6-imul"):
        emit_imul(b, x1, y1, A, Rwy, p, n, ma)
    with b.phase("s7-final"):
        neg_modp(b, x1, y1, clean3, p, n)
        const_modp(b, x1, S_qx, flagq, y1, clean3, p, n)
        const_modp(b, y1, S_qy, flagq, x1, clean3, p, n, inv=True)
        # End-of-circuit teardown: every non-io wire is |0> here (the
        # evaluator's reversibility gate re-verifies), so release every
        # workspace ownership set and free its wires in reverse
        # allocation order (LIFO free-list).  Only x1/y1 -- the declared
        # registers -- stay owned and live.
        for nm, ids in (
            ("A", A),
            ("wext/w1", w1x),
            ("wext/w2", w2x),
            ("len/lt", lt),
            ("len/lq", lq),
            ("len/ls", ls),
            ("len/lrp", lrp),
            ("ctrl", [p1, p2, sign, iterq]),
            ("scr/path", aux["pathA"] + aux["pathB"]),
            ("scr/flags", [aux["acc"], aux["carry"], aux["c3xw"], aux["ct3"], aux["alv"]]),
        ):
            b.release(nm, ids)
        for reg in (
            [aux["alv"], aux["ct3"], aux["c3xw"], aux["carry"], aux["acc"]],
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


def build():
    """Raw full-width build for the profiling tools: no selftests, no pin
    asserts, no emission.  Returns ``(b,)``, the finished Builder.  This
    is the entry point of tools/occupancy_profile.py, which replays the
    ownership wire-set log fail-closed (claims at birth, the s7
    release/free teardown in build_circuit) and charts the band
    occupancy over the loop-compressed Toffoli axis.
    """
    b = Builder()
    build_circuit(b, P_MOD, N)
    return (b,)


def fullwidth_emit():
    """n=256 build on the stock Builder, then the standard epilogue
    (uncompute.testing): assert_metrics pins the certified invariants --
    69,529,757 emitted == executed Toffoli x 838 peak qubits -- and the
    ancilla-leak guard (the s7 teardown leaves only the 512 schema
    qubits live); emit_and_report streams the op stream to
    $UNCOMPUTE_OPS_PATH (the evaluator FIFO) or ./ops.bin.
    """
    # The model tier runs the generic-modulus budget and width
    # (nmax_steps(n), ls_width(n)); the emitted circuit runs the
    # p-specific ones (step_tables, the ls register).  They coincide
    # at full width (the P1 exact-DP budget of 1616 steps, residual
    # cap 1616 - 4n = 592):
    assert nmax_steps(N) == nmax_steps(N, P_MOD)
    assert ls_width(N) == ls_width(N, P_MOD)
    b = Builder()
    build_circuit(b, P_MOD, N)
    s = b.phase_summary()
    print(
        f"build: peak_qubits={b.peak_qubits}, outcomes={b.next_out:,}, "
        f"emitted Toffoli={s['total_emitted_toffoli']:,}, "
        f"Clifford={s['total_emitted_clifford']:,}"
    )
    assert_metrics(b, __file__)
    # Outcome ids are per-record-site (SECTION D header); the program is
    # addressable only while they fit the interpreter's 2^24 bit-address
    # cap (68,452 minted at full width, measured).
    assert b.next_out < (1 << 24), ("outcome-id cap", b.next_out)
    emit_and_report(b)
    return b


def main():
    # Transport is the golden-path convention only: $UNCOMPUTE_OPS_PATH
    # (the evaluator's FIFO) or ./ops.bin.  There is deliberately no
    # positional out-path: runpy-based tools (the profilers) run this
    # file with their own argv in place, and runpy sets argv[0] to this
    # file, so a positional would read a foreign token as the ops
    # destination and misroute the emission (the evaluator
    # sanitizes argv to [circuit_path] against exactly that).
    fullwidth_emit()


# Bind the circuit-local callables the shared layer resolves at call
# time (fail-closed: a missing name raises here, not mid-build).
_core.init(globals())


if __name__ == "__main__":
    main()
