"""Shared arithmetic/template layer for the modinv-point-add
circuits (circuits/divsteps, circuits/euclidean, circuits/luo-eea).

The bodies here are the byte-for-byte common core of the three circuits'
code: the (gate, ...) tuple vocabulary, its inverse rule and
emitter, the CDKM adder cells, small-register/counter helpers and AND
ladders, the classed-template machinery (class tables, the unary tree
shape, the unary/dual templates, prune, inverse and step-loop emitters
with the step-table cache), the Gidney carry helpers (cbspec bits may
be compile-time ints or per-shot Bit handles) with the borrowed-dirty
constant add/compare ladder and the controlled ripple add/compare
cells, the k-block lazy modular multiplication family, and the modular
add/double/const-add/negate + idiv/imul flows.  It re-exports the
classical-bit input layer of ./clsbits.py (the point I/O registers,
ClsBits + const_modp_bit_specs deriving per-shot cbspecs from the
Bit-kind input registers).  "The paper" in the comments below means
Luo et al. (2026), arXiv:2607.13816.

Loading contract: each circuit.py
loads this file by path (uncompute.testing.load_module, no sys.modules
registration -- every load is a private instance).  Circuit-specific
callables (the _DEPS tuple) stay in each circuit.py and are bound into this
namespace by init(globals()) after they are defined; the bodies below
resolve them at call time.  Calling one before init() raises (the
_unbound placeholder), and init() raises KeyError on a missing name.
"""

import math
import pathlib

from uncompute.gadgets import maj, maj_inv
from uncompute.testing import load_module

# Circuit-local callables, bound by init().  The placeholders keep the
# names statically defined; calling one before init() fails loudly.
_DEPS = (
    "StepTables",
    "c3x",
    "emit_eea",
    "mul_modp",
)


def _unbound(*_a, **_k):
    raise RuntimeError("luo_core.init() not called")


StepTables = _unbound
c3x = _unbound
emit_eea = _unbound
mul_modp = _unbound


def init(circuit_globals):
    """Bind the circuit's _DEPS implementations into this module."""
    for _name in _DEPS:
        globals()[_name] = circuit_globals[_name]


# ===========================================================================
# gate vocabulary
# ===========================================================================


def int_bits(v, w):
    return [(v >> i) & 1 for i in range(w)]


def X_(q):
    return ("x", q)


def CX_(a, t):
    return ("cx", a, t)


def CCX_(a, b, t):
    return ("ccx", a, b, t)


def ANDP(c1, c2, t, n1=0, n2=0):
    """t (clean) <- (c1^n1) & (c2^n2): one Toffoli (temporary AND)."""
    return ("and+", c1, c2, t, n1, n2)


def ANDM(c1, c2, t, n1=0, n2=0):
    """Release t = (c1^n1)&(c2^n2): HMR + conditioned CZ, zero Toffoli
    (paper Fig. 8 measurement uncompute).
    """
    return ("and-", c1, c2, t, n1, n2)


_INV = {"and+": "and-", "and-": "and+"}


def inv_list(gates):
    return [((_INV[g[0]],) + g[1:] if g[0] in _INV else g) for g in reversed(gates)]


def emit_op(b, g):
    """Emit one vocabulary tuple into Builder b.  Works identically in flat
    context and inside repeat()/gated() recording (the measurement outcome
    of "and-" is written by its own hmr and read once by the following
    conditioned CZ -- one outcome id per record site, every iteration).

    NOTE: "and+"/"and-" deliberately do NOT use Builder.and_compute/
    and_release. The vocabulary ANDs write into persistent pre-allocated
    scratch (no alloc/free per AND) and "and-" omits the redundant R on the
    already-|0> wire that and_release's recycle() would emit — both are part
    of the certified op count. Swapping in the built-ins is STREAM-CHANGING.
    A gadgets.WirePool front over the scratch has the same mismatch: pooled
    wires pop LIFO, while the vocabulary targets fixed per-role wires (the
    lockstep pathA/pathB trees), and pooled ANDs carry no negations.
    """
    k = g[0]
    if k == "x":
        b.x(g[1])
    elif k == "cx":
        b.cx(g[1], g[2])
    elif k == "ccx":
        b.ccx(g[1], g[2], g[3])
    elif k == "swap":
        # produced by rotation virtualization (the unconditional
        # relabel-rotation)
        b.swap(g[1], g[2])
    elif k == "and+":
        _, c1, c2, t, n1, n2 = g
        if n1:
            b.x(c1)
        if n2:
            b.x(c2)
        b.ccx(c1, c2, t)
        if n2:
            b.x(c2)
        if n1:
            b.x(c1)
    elif k == "and-":
        _, c1, c2, t, n1, n2 = g
        m = b.hmr(t)
        if n1:
            b.x(c1)
        if n2:
            b.x(c2)
        b.cz(c1, c2, cond=m)
        if n2:
            b.x(c2)
        if n1:
            b.x(c1)
    else:
        raise ValueError(k)


def emit(b, gates, inv=False):
    for g in inv_list(gates) if inv else gates:
        emit_op(b, g)


# ===========================================================================
# adder cells
# ===========================================================================


def cell_maj(s, a, b, c, w):
    return [CX_(c, a), CX_(c, b)] + c3x(s, a, b, c, w)


def cell_uma(s, a, b, c, w):
    return c3x(s, a, b, c, w) + [CCX_(s, b, a), CX_(c, b), CX_(c, a)]


def cell_maj_inv(s, a, b, c, w):
    return c3x(s, a, b, c, w) + [CX_(c, b), CX_(c, a)]


def cell_uma_inv(s, a, b, c, w):
    return [CX_(c, a), CX_(c, b), CCX_(s, b, a)] + c3x(s, a, b, c, w)


def cell_maj_u(a, b, c):
    """Uncontrolled MAJ (s == 1).
    local: same 1 Toffoli as gadgets.arith.maj, but vocabulary-spelled
    (inv_list mirror) with the carry held on c and the CCX controls in
    (a, b) order -- swapping in the shared cell is stream-changing.
    """
    return [CX_(c, a), CX_(c, b), CCX_(a, b, c)]


def cell_uma_u(a, b, c):
    return [CCX_(a, b, c), CX_(b, a), CX_(c, b), CX_(c, a)]


# ===========================================================================
# small-register arithmetic, AND ladders, controlled swap
# ===========================================================================


def add_small(A, B, carry):
    """B += A mod 2^w (equal widths), carry clean and returned clean.
    local: gadgets.arith.cdkm_add_co is the same 2w-Toffoli [CDKM]
    chain; the step template needs classable (gate, class) tuples, and
    its CCX operand order is part of the certified stream.
    """
    w = len(B)
    assert len(A) == w
    g = []
    for i in range(w):
        g += cell_maj_u(B[i], A[i], carry)
    for i in range(w - 1, -1, -1):
        g += cell_uma_u(B[i], A[i], carry)
    return g


def ctrl_inc(reg, ctrl, anc):
    """reg += ctrl (mod 2^w).  anc: w-1 clean scratch.  w-1 Toffoli.
    The carry ANDs release by measurement: after reg[i] flips under
    src, anc[i] = src AND NOT reg[i], an "and-" with a negated literal.
    local: gadgets.gf_tower.GF.ctr_inc/ctr_dec are the same w-1
    Toffoli; the classed step template needs (gate, class) tuples.
    """
    w = len(reg)
    g = []
    prev = ctrl
    for i in range(w - 1):
        g.append(ANDP(prev, reg[i], anc[i]))
        prev = anc[i]
    g.append(CX_(ctrl, reg[0]))
    for i in range(1, w):
        g.append(CX_(anc[i - 1], reg[i]))
    for i in range(w - 2, -1, -1):
        src = ctrl if i == 0 else anc[i - 1]
        g.append(ANDM(src, reg[i], anc[i], 0, 1))
    return g


def add_gid(A, B, anc):
    """B += A (mod 2^w), equal widths.  anc: w-1 clean scratch holding
    the carries.  w-1 Toffoli: each carry is one AND on the
    carry-conjugated operands (MAJ), released by measurement on the way
    back (Gidney, arXiv:1709.06648).
    """
    w = len(B)
    assert len(A) == w
    if w == 0:
        return []

    def c(i):  # carry into bit i (i >= 1)
        return anc[i - 1]

    g = []
    if w >= 2:
        g.append(ANDP(A[0], B[0], c(1)))
    for i in range(1, w - 1):
        g += [CX_(c(i), A[i]), CX_(c(i), B[i]), ANDP(A[i], B[i], c(i + 1)), CX_(c(i), c(i + 1))]
    top = w - 1
    g.append(CX_(A[top], B[top]))
    if top >= 1:
        g.append(CX_(c(top), B[top]))
    for i in range(w - 2, 0, -1):
        g += [CX_(c(i), c(i + 1)), ANDM(A[i], B[i], c(i + 1)), CX_(c(i), A[i]), CX_(A[i], B[i])]
    if w >= 2:
        g += [ANDM(A[0], B[0], c(1)), CX_(A[0], B[0])]
    return g


def and_ladder(literals, anc):
    """Compute AND of literals [(qubit, negate), ...] into anc[m-2]
    (returns gates, ctrl_qubit).  len >= 2; release with inv_list.
    local: gadgets.arith.mcx matches the m-1 Toffoli count but
    allocates its ladder and targets one X; this writes owned
    persistent scratch, keeps the top AND wire for reuse, and returns
    classable tuples.
    """
    m = len(literals)
    assert m >= 2 and len(anc) >= m - 1
    g = [ANDP(literals[0][0], literals[1][0], anc[0], literals[0][1], literals[1][1])]
    cur = anc[0]
    for i in range(2, m):
        g.append(ANDP(cur, literals[i][0], anc[i - 1], 0, literals[i][1]))
        cur = anc[i - 1]
    return g, cur


def cswap(q, a, b):
    # local: same CX/CCX/CX bytes as Builder.cswap; templates need
    # classable (gate, class) tuples.
    return [CX_(b, a), CCX_(q, a, b), CX_(b, a)]


def _zero_toggle(reg, target, anc):
    """target ^= [reg == 0] (compute-use-release, anc transient).
    local: gadgets.mcx open-controlled is the same m-1 Toffoli on m-2
    transient and_compute wires; this rides the persistent ladder anc
    and stays gate-list (it composes and_ladder -- see its carve-out).
    """
    lits = [(q, 1) for q in reg]
    if len(lits) == 1:
        return [X_(reg[0]), CX_(reg[0], target), X_(reg[0])]
    g, cq = and_ladder(lits, anc)
    return g + [CX_(cq, target)] + inv_list(g)


def xor_const_reg(reg, val):
    return [X_(reg[i]) for i in range(len(reg)) if (val >> i) & 1]


def add_const_small(B, val, scratch, carry):
    """B += val mod 2^w via a scratch register set to the constant."""
    w = len(B)
    val %= 1 << w
    if val == 0:
        return []
    s = scratch[:w]
    g = xor_const_reg(s, val)
    g += add_small(s, B, carry)
    g += xor_const_reg(s, val)
    return g


def const_minus_small(B, C, scratch, carry):
    """B <- (C - B) mod 2^w (involution)."""
    w = len(B)
    g = [X_(q) for q in B]
    g += add_const_small(B, (C + 1) % (1 << w), scratch, carry)
    return g


def ctrl_dec(reg, ctrl, anc):
    """reg -= ctrl (mod 2^w): complement, increment, complement."""
    fan = [X_(q) for q in reg]
    return fan + ctrl_inc(reg, ctrl, anc) + fan


def unary_depth(m_labels):
    if m_labels <= 1:
        return 0
    return math.ceil(math.log2(max(m_labels - 1, 1))) + 1


# ===========================================================================
# loop templates
# ===========================================================================


def _bcache(b, name):
    """Per-Builder memo dict (Builders are the cache scope: DATA tables
    must be declared once per emission).
    """
    cache = getattr(b, name, None)
    if cache is None:
        cache = {}
        setattr(b, name, cache)
    return cache


class ClassTable:
    """One per-role gating table: nsteps rows x width slots, row-major (bit
    index = (T-1)*width + slot, T the 1-based step).  Filled once at build
    time; declared as an op-stream DATA segment per Builder on first use.
    """

    __slots__ = ("_any_memo", "buf", "name", "nsteps", "width")

    def __init__(self, name, nsteps, width):
        self.name = name
        self.nsteps = nsteps
        self.width = width
        self.buf = bytearray(nsteps * width)
        self._any_memo = {}

    def set(self, T, slot):
        self.buf[(T - 1) * self.width + slot] = 1

    def get(self, T, slot):
        return self.buf[(T - 1) * self.width + slot]

    def any(self, slot):
        """Return True if the slot fires on any step (all-dead slots are dropped
        from the recorded template -- they would expand to nothing).
        """
        hit = self._any_memo.get(slot)
        if hit is None:
            hit = self._any_memo[slot] = any(self.buf[slot :: self.width])
        return hit

    def handle(self, b):
        cache = _bcache(b, "_luo_tables")
        h = cache.get(id(self))
        if h is None:
            h = cache[id(self)] = b.declare_table(self.buf)
        return h


def tpl_prune(items):
    """Drop ops whose class never fires on any step (their gated records
    would expand to nothing on every iteration). Spine-tagged items
    (("spine", name) classes -- census attribution only) are always-on.
    """
    alive = {}
    out = []
    for g, cls in items:
        if cls is not None and cls[0] != "spine":
            key = (id(cls[0]), cls[1])
            ok = alive.get(key)
            if ok is None:
                ok = alive[key] = cls[0].any(cls[1])
            if not ok:
                continue
        out.append((g, cls))
    return out


def tpl_emit_body(b, seq, sexpr):
    """Record one pass of a (pre-inverted, pre-pruned) template inside the
    enclosing repeat: each op data-gated on its class bit at step `sexpr`
    (an affine expression of the step loop index).  Consecutive ops sharing
    a class share one gated() block.
    """
    i, m = 0, len(seq)
    while i < m:
        g, cls = seq[i]
        if cls is None or cls[0] == "spine":  # spine tag = always-on
            emit_op(b, g)
            i += 1
            continue
        j = i + 1
        while j < m and seq[j][1] is not None and seq[j][1][0] is cls[0] and seq[j][1][1] == cls[1]:
            j += 1
        tbl, slot = cls
        with b.gated(tbl.handle(b), sexpr * tbl.width + slot):
            for k in range(i, j):
                emit_op(b, seq[k][0])
        i = j


def _split_bit(lo, hi):
    return (lo ^ hi).bit_length() - 1


class _UShape:
    """Fixed full-range pruned-unary-iteration tree shape over labels [1, L].

    A per-step window [k, K] is realized by 3-state node gating:

      AND  (window straddles both children) -> the plain ANDP/CX/ANDM node
      PASS (window inside one child)        -> CX(ctrl, h) pass-through pair
      OFF  (window misses the subtree)      -> no ops in the expansion

    Contracting the PASS chains yields exactly the plain unary tree built
    directly on [k, K]: the same split bits are tested in the same order
    (splits are at absolute bit boundaries, so the first straddling bit
    going down equals the window's own top differing bit, recursively), the
    same leaves fire in the same order, and the executed Toffoli count is
    identical (M-1 node ANDs for an M-label window; PASS pairs are
    Cliffords; ANDM releases are measurement-based).  The PASS child reads
    its control from the parent's path qubit instead of skipping levels --
    path qubits are interchangeable clean scratch, so only the spelling
    differs from the level-skipping form.

    local: gadgets.qrom.walk emits a per-window tree at build time (the
    same M-1 ANDs per M-address window); inside b.repeat the body must
    be ONE fixed shape for all steps, re-windowed per step by this
    3-state ClassTable gating.
    """

    def __init__(self, L, lo=1):
        self.L = L  # highest label
        self.lo = lo  # lowest label (>= 0: bit-split tree)
        assert lo >= 0
        self.nodes = []  # nid -> (lo, hi, bit, zref, oref)

        def rec(lo, hi):
            if lo == hi:
                return ("leaf", lo)
            bit = _split_bit(lo, hi)
            x = (hi >> bit) << bit  # smallest label with `bit` set
            nid = len(self.nodes)
            self.nodes.append(None)
            z = rec(lo, x - 1)
            o = rec(x, hi)
            self.nodes[nid] = (lo, hi, bit, z, o)
            return ("node", nid)

        self.root = rec(lo, L)
        self.n_nodes = len(self.nodes)

    def _iv(self, ref):
        if ref[0] == "leaf":
            return ref[1], ref[1]
        nd = self.nodes[ref[1]]
        return nd[0], nd[1]

    def mark(self, k, K, on_node, on_leaf):
        """Activate the window's classes: on_node(nid, 'A'|'P'), on_leaf(j)."""

        def rec(ref):
            if ref[0] == "leaf":
                if k <= ref[1] <= K:
                    on_leaf(ref[1])
                return
            _, _, _, z, o = self.nodes[ref[1]]
            zlo, zhi = self._iv(z)
            olo, ohi = self._iv(o)
            zin = not (zhi < k or zlo > K)
            oin = not (ohi < k or olo > K)
            if not (zin or oin):
                return
            if zin and oin:
                on_node(ref[1], "A")
                rec(z)
                rec(o)
            else:
                on_node(ref[1], "P")
                rec(z if zin else o)

        lo, hi = self._iv(self.root)
        if not (hi < k or lo > K):
            rec(self.root)


def tpl_unary(shape, idx, root, path, leaf_fn, order, node_cls):
    """Classed template of the full-shape unary iteration.  leaf_fn(j, e)
    returns fully-classed items; node_cls(nid, 'A'|'P') gives node classes.
    Mirrors unary_tree exactly on the AND skeleton.
    """
    out = []

    def rec(ref, ctrl, depth):
        if ref[0] == "leaf":
            out.extend(leaf_fn(ref[1], ctrl))
            return
        nid = ref[1]
        _, _, bit, z, o = shape.nodes[nid]
        h = path[depth]
        first, second = (z, o) if order == "inc" else (o, z)
        neg1 = 1 if first is z else 0
        neg2 = 1 if second is z else 0
        A = node_cls(nid, "A")
        P = node_cls(nid, "P")
        out.append((ANDP(ctrl, idx[bit], h, 0, neg1), A))
        out.append((CX_(ctrl, h), P))
        rec(first, h, depth + 1)
        out.append((CX_(ctrl, h), A))
        rec(second, h, depth + 1)
        out.append((ANDM(ctrl, idx[bit], h, 0, neg2), A))
        out.append((CX_(ctrl, h), P))

    if shape.root[0] == "leaf":
        out.extend(leaf_fn(shape.root[1], root))
    else:
        rec(shape.root, root, 0)
    return out


def tpl_dual(shape, idxA, idxB, root, pathA, pathB, leaf_fn, order, node_cls):
    """Classed template of dual_unary_tree (lockstep A/B scans)."""
    out = []

    def rec(ref, ca, cb, depth):
        if ref[0] == "leaf":
            out.extend(leaf_fn(ref[1], ca, cb))
            return
        nid = ref[1]
        _, _, bit, z, o = shape.nodes[nid]
        ha, hb = pathA[depth], pathB[depth]
        first, second = (z, o) if order == "inc" else (o, z)
        neg1 = 1 if first is z else 0
        neg2 = 1 if second is z else 0
        A = node_cls(nid, "A")
        P = node_cls(nid, "P")
        out.append((ANDP(ca, idxA[bit], ha, 0, neg1), A))
        out.append((ANDP(cb, idxB[bit], hb, 0, neg1), A))
        out.append((CX_(ca, ha), P))
        out.append((CX_(cb, hb), P))
        rec(first, ha, hb, depth + 1)
        out.append((CX_(ca, ha), A))
        out.append((CX_(cb, hb), A))
        rec(second, ha, hb, depth + 1)
        out.append((ANDM(cb, idxB[bit], hb, 0, neg2), A))
        out.append((ANDM(ca, idxA[bit], ha, 0, neg2), A))
        out.append((CX_(cb, hb), P))
        out.append((CX_(ca, ha), P))

    if shape.root[0] == "leaf":
        out.extend(leaf_fn(shape.root[1], root, root))
    else:
        rec(shape.root, root, root, 0)
    return out


def tpl_inv(items):
    """Exact inverse of a classed template: reversed order, and+ <-> and-
    (the same rule as inv_list), classes unchanged.
    """
    return [(((_INV[g[0]],) + g[1:]) if g[0] in _INV else g, cls) for (g, cls) in reversed(items)]


def tpl_emit_steps(b, items, nsteps, inv=False, chunk=45000):
    """Emit the step loop: b.repeat(nsteps) around the template, chunked into
    nested repeat(1) groups to respect the op-stream u16 body-item cap.  The
    inverse loop runs the mirrored template with the step index reversed.
    """
    seq = tpl_prune(tpl_inv(items) if inv else items)
    with b.repeat(nsteps) as s:
        sexpr = (nsteps - 1 - s) if inv else s
        if len(seq) <= chunk:
            tpl_emit_body(b, seq, sexpr)
        else:
            for c0 in range(0, len(seq), chunk):
                with b.repeat(1):
                    tpl_emit_body(b, seq[c0 : c0 + chunk], sexpr)


_STCACHE = {}


def step_tables(n, p):
    key = (n, p)
    if key not in _STCACHE:
        _STCACHE[key] = StepTables(n, p)
    return _STCACHE[key]


def _affine_spans(ids):
    """Split an id sequence into maximal position-affine runs:
    [(t0, length, base, stride)] with ids[t] = base + stride*(t - t0).
    """
    spans = []
    i, m = 0, len(ids)
    while i < m:
        if i + 1 == m:
            spans.append((i, 1, ids[i], 0))
            break
        stride = ids[i + 1] - ids[i]
        j = i + 1
        while j + 1 < m and ids[j + 1] - ids[j] == stride:
            j += 1
        spans.append((i, j - i + 1, ids[i], stride))
        i = j + 1
    return spans


# ===========================================================================
# gidney carry helpers
# ===========================================================================


def _cb_x(b, q, spec_bit, ctrl=None):
    """X/CX on q gated by a cbspec bit.  A compile-time int folds at
    build time (0 -> skip); a per-shot classical Bit rides as a when=
    condition -- the X stays Clifford-class (CX), the CX promotes to a
    full Toffoli (CCX).
    """
    if isinstance(spec_bit, int):
        if spec_bit == 0:
            return
        if ctrl is None:
            b.x(q)
        else:
            b.cx(ctrl, q)
    elif ctrl is None:
        b.x(q, when=spec_bit)
    else:
        b.cx(ctrl, q, when=spec_bit)


def _ccx_cond_const(b, c1, c2, target, s0, s1, ctrl=None):
    """target ^= (c1^b0)&(c2^b1) with b0/b1 cbspec bits, optionally gated so
    the constant is zero when ctrl=0 (the controlled Gidney carry trick).
    """
    _cb_x(b, c1, s0, ctrl)
    _cb_x(b, c2, s1, ctrl)
    b.ccx(c1, c2, target)
    _cb_x(b, c2, s1, ctrl)
    _cb_x(b, c1, s0, ctrl)


def _xor_carries(b, a, cb, out, cin, ctrl=None, include_top=False):
    """out[i] ^= carry_{i+1} of (a + ctrl*C); a = complemented final sum."""
    n = len(a)
    hi = n if include_top else n - 1
    if hi <= 0:
        return
    for i in range(hi - 1, 0, -1):
        _ccx_cond_const(b, a[i], out[i - 1], out[i], cb[i], 0, ctrl=ctrl)
    for i in range(hi):
        _cb_x(b, out[i], cb[i], ctrl)
    _ccx_cond_const(b, cin, a[0], out[0], cb[0], cb[0], ctrl=ctrl)
    for i in range(1, hi):
        _ccx_cond_const(b, a[i], out[i - 1], out[i], cb[i], cb[i], ctrl=ctrl)


def _gidney_pos(b, a_i, cb_i, dirty_i, cur, spare, anc, ctrl, mid=None):
    """One position of the borrowed-dirty Gidney ladder (the shared core
    of gidney_add_const and gidney_cmp_ge): dress anc with the constant
    bit, fold the incoming carry (cur) into anc / a_i, AND the pair into
    spare (the outgoing carry), write the dirty target, run the caller's
    mid section (top-bit out write / operand restore), then unfold the
    carry from anc and undress.
    """
    _cb_x(b, anc, cb_i, ctrl)
    b.cx(cur, anc)
    b.cx(cur, a_i)
    b.ccx(a_i, anc, spare)
    b.cx(cur, spare)
    b.cx(spare, dirty_i)
    if mid is not None:
        mid()
    b.cx(cur, anc)
    _cb_x(b, anc, cb_i, ctrl)


def gidney_add_const(b, target, cb, dirty, clean3, ctrl=None):
    """target += C mod 2^n (C as cbspec list), n-1 borrowed dirty qubits,
    3 clean.  Measurement-based (HMR + conditioned-Z phase repair).
    local: gadgets.arith.add takes a qubit operand — a constant costs an
    n-bit clean scratch (n-1 Toffoli, n-1 transient ANDs, +n width);
    this pays 3n-4 Toffoli to borrow dirty data wires at zero fresh
    allocs (the width frontier's trade).
    """
    n = len(target)
    a = list(target)
    if n == 0:
        return
    if n == 1:
        _cb_x(b, a[0], cb[0], ctrl)
        return
    assert len(dirty) >= n - 1 and len(clean3) >= 3
    cy, nxt, anc = clean3[0], clean3[1], clean3[2]
    cur, spare = cy, nxt
    mbits = []
    for i in range(n - 1):
        _gidney_pos(b, a[i], cb[i], dirty[i], cur, spare, anc, ctrl)
        _cb_x(b, a[i], cb[i], ctrl)  # sum kept: dress the sum bit
        if i > 0:
            mbits.append(b.hmr(cur))
        cur, spare = spare, cur
    _cb_x(b, a[n - 1], cb[n - 1], ctrl)
    b.cx(cur, a[n - 1])
    mbits.append(b.hmr(cur))
    for i in range(n - 1):
        b.z(dirty[i], cond=mbits[i])
    for q in a:
        b.x(q)
    _xor_carries(b, a, cb, dirty, clean3[0], ctrl=ctrl)
    for q in a:
        b.x(q)
    for i in range(n - 1):
        b.z(dirty[i], cond=mbits[i])


def gidney_cmp_ge(b, target, k_cb, out, dirty, clean3, ctrl=None):
    """out ^= ctrl & [target >= k]; k_cb = cbspec of (2^n - k) mod 2^n.
    Needs n dirty and 3 clean.  Self-inverse.  k must be in (0, 2^n).
    local: gadgets.arith has no comparator (add(cout=) leaves the sum
    in its target); this restores everything at 3n-1 Toffoli on the
    same borrowed-dirty profile.
    """
    n = len(target)
    a = list(target)
    assert len(dirty) >= n and len(clean3) >= 3
    cy, nxt, anc = clean3[0], clean3[1], clean3[2]
    cur, spare = cy, nxt
    mbits = []
    for i in range(n):

        def mid(i=i, cur=cur, spare=spare):
            # compare-only: top-bit out write, then restore the operand
            if i == n - 1:
                if ctrl is None:
                    b.cx(spare, out)
                else:
                    b.ccx(ctrl, spare, out)
            b.cx(cur, a[i])

        _gidney_pos(b, a[i], k_cb[i], dirty[i], cur, spare, anc, ctrl, mid=mid)
        if i > 0:
            mbits.append(b.hmr(cur))
        cur, spare = spare, cur
    mbits.append(b.hmr(cur))
    for i in range(n):
        b.z(dirty[i], cond=mbits[i])
    _xor_carries(b, a, k_cb, dirty, clean3[0], ctrl=ctrl, include_top=True)
    for i in range(n):
        b.z(dirty[i], cond=mbits[i])


def ctrl_qq_add_flag(b, ctrl, addend, acc, carry, flag, inv=False):
    """acc += ctrl*addend mod 2^n, flag ^= ctrl*carry_out (paper App. A
    specialized controlled ripple; 3n+1 Toffoli).  inv emits the exact
    inverse (controlled subtract, flag toggled by the same carry).
    local: gadgets.arith.add has no controlled form; the [CDKM] maj
    cell itself is the shared gadget.
    """
    bb, a = list(addend), list(acc)
    n = len(a)
    if n == 0:
        return
    if not inv:
        for i in range(n):
            maj(b, a[i], bb[i], carry)  # [CDKM] cell, shared gadget
        b.ccx(ctrl, carry, flag)
        for i in range(n - 1, -1, -1):
            b.ccx(a[i], bb[i], carry)
            b.cx(carry, a[i])
            b.ccx(ctrl, bb[i], a[i])
            b.cx(carry, bb[i])
    else:
        for i in range(n):
            b.cx(carry, bb[i])
            b.ccx(ctrl, bb[i], a[i])
            b.cx(carry, a[i])
            b.ccx(a[i], bb[i], carry)
        b.ccx(ctrl, carry, flag)
        for i in range(n - 1, -1, -1):
            maj_inv(b, a[i], bb[i], carry)  # [CDKM] cell, shared gadget


def cuccaro_lt_flag(b, left, right, flag, carry, ctrl=None):
    """flag ^= (ctrl &) [left < right]: high-bit-only Cuccaro comparator
    (2n+O(1) Toffoli); everything else restored.  Self-inverse.
    local: gadgets.arith.add(cout=) computes the carry but leaves the
    sum in its target; this restores both operands (2n+1 Toffoli
    controlled, 2n plain) on the shared maj/maj_inv cells.
    """
    lf, rt = list(left), list(right)
    n = len(lf)
    assert len(rt) == n and n >= 1
    if n == 1:
        b.x(lf[0])
        if ctrl is None:
            b.ccx(lf[0], rt[0], flag)
        else:
            b.ccx(lf[0], rt[0], carry)
            b.ccx(ctrl, carry, flag)
            b.ccx(lf[0], rt[0], carry)
        b.x(lf[0])
        return
    for q in rt:
        b.x(q)
    b.x(carry)
    for c, y, a in [(carry, rt[0], lf[0])] + [(lf[i - 1], rt[i], lf[i]) for i in range(1, n)]:
        maj(b, c, y, a)
    b.x(lf[n - 1])
    if ctrl is None:
        b.cx(lf[n - 1], flag)
    else:
        b.ccx(ctrl, lf[n - 1], flag)
    b.x(lf[n - 1])
    for c, y, a in reversed(
        [(carry, rt[0], lf[0])] + [(lf[i - 1], rt[i], lf[i]) for i in range(1, n)]
    ):
        maj_inv(b, c, y, a)
    b.x(carry)
    for q in rt:
        b.x(q)


# ===========================================================================
# k-block lazy modular multiplication: the k-block keeper gadgets, the
# lazy controlled mod-add, and the windowed Horner multiplier.  Each
# circuit keeps its own priced windows (FOLD_WIN / CMP_TOPG) and
# dispatch (_mul_cma / _mul_dbl) and passes them in.
# ===========================================================================


def _release(b, t, u, w):
    """Gidney AND release: t held AND(u, w); u/w must hold their
    compute-time values.  0 Toffoli.  local: Builder.and_release frees
    its wire (one R op); pool wires stay allocated.
    """
    mb = b.hmr(t)
    b.cz(u, w, cond=mb)


def _release_copy(b, t, u):
    """Release t == copy of u (CX-written).  0 Toffoli."""
    mb = b.hmr(t)
    b.z(u, cond=mb)


def _qq_fwd(b, a, e, lo, hi, cin, canc):
    d = cin
    for i in range(lo, hi):
        t = canc[i - lo]
        if d is not None:
            b.cx(d, a[i])
            b.cx(d, e[i])
        b.ccx(a[i], e[i], t)
        if d is not None:
            b.cx(d, t)
        d = t
    return d  # c_hi


def _qq_bwd(b, a, e, lo, hi, cin, canc, sum_ctrl=None, keep_top=False):
    for i in range(hi - 1, lo - 1, -1):
        t = canc[i - lo]
        d = cin if i == lo else canc[i - lo - 1]
        if not (keep_top and i == hi - 1):
            if d is not None:
                b.cx(d, t)  # t -> AND value
            _release(b, t, a[i], e[i])
        if sum_ctrl is not None:
            b.ccx(sum_ctrl, e[i], a[i])
        if d is not None:
            b.cx(d, a[i])
            b.cx(d, e[i])


def _cf_fwd(b, a, kb, F, lo, hi, cin, canc):
    cw = []
    d = cin
    for i in range(lo, hi):
        t = canc[i - lo]
        if kb[i]:
            if d is not None:
                b.cx(d, a[i])
                b.cx(d, F)
            b.ccx(a[i], F, t)
            if d is not None:
                b.cx(d, t)
                b.cx(d, F)  # F restored immediately
        else:
            if d is None:
                cw.append(None)
                d = None
                continue
            b.ccx(a[i], d, t)  # a[i] undressed for zero bits
        cw.append(t)
        d = t
    return cw


def _cf_bwd(b, a, kb, F, lo, hi, cin, cw, sums=True, keep_top=False):
    for i in range(hi - 1, lo - 1, -1):
        t = cw[i - lo]
        d = cin if i == lo else cw[i - lo - 1]
        keep = keep_top and i == hi - 1
        if kb[i]:
            if not keep:
                if d is not None:
                    b.cx(d, F)
                    b.cx(d, t)
                _release(b, t, a[i], F)
                if d is not None:
                    b.cx(d, F)
            if sums:
                b.cx(F, a[i])  # a = a ^ c_i ^ F  (dressing kept)
            elif d is not None:
                b.cx(d, a[i])  # release-only: undress
        else:
            if t is not None and not keep:
                _release(b, t, a[i], d)
            if sums and d is not None:
                b.cx(d, a[i])  # a = a ^ c_i


def _plan(nc, avail):
    """Blocks [(lo, hi)...]; single block when it fits.
    Constraints (0-indexed block j): forward runs with j keepers held
    => s_j <= avail - j; the ERASE of a keeper block runs with j+1
    keepers live (incl. the target) => s_j <= avail - (j+1) for
    j < k-1. Last block maximal to minimize the erase cost.
    """
    if avail >= nc:
        return [(0, nc)]
    for k in range(2, avail + 1):
        caps = [avail - j - 1 for j in range(k - 1)] + [avail - k + 1]
        if min(caps) < 2 or sum(caps) < nc:
            continue
        sizes = caps[:]  # start maximal
        excess = sum(caps) - nc
        for j in range(k - 1):  # shrink early blocks
            cut = min(excess, sizes[j] - 1)
            sizes[j] -= cut
            excess -= cut
        if excess:
            continue
        lo, out = 0, []
        for s in sizes:
            out.append((lo, lo + s))
            lo += s
        assert lo == nc
        return out
    raise AssertionError(("no k-block plan", nc, avail))


def _wires(anc, keepers, count):
    """count free pool wires excluding the live keeper wires."""
    ks = set(keepers)
    out = [q for q in anc if q not in ks][:count]
    assert len(out) == count, (len(anc), len(keepers), count)
    return out


def _clt_fwd_cin(b, ax, kb, lo, hi, canc, cin):
    cw = []
    d = cin
    for i in range(lo, hi):
        t = canc[i - lo]
        if kb[i]:
            if d is None:
                b.cx(ax[i], t)
            else:
                b.x(ax[i])
                b.x(d)
                b.ccx(ax[i], d, t)
                b.x(t)
                b.x(d)
                b.x(ax[i])
        else:
            if d is None:
                cw.append(None)
                d = None
                continue
            b.ccx(ax[i], d, t)
        cw.append(t)
        d = t
    return cw, d


def _clt_bwd_cin(b, ax, kb, lo, hi, cw, cin):
    for i in range(hi - 1, lo - 1, -1):
        t = cw[i - lo]
        if t is None:
            continue
        d = cin if i == lo else cw[i - lo - 1]
        if kb[i]:
            if d is None:
                _release_copy(b, t, ax[i])
            else:
                b.x(t)
                b.x(ax[i])
                b.x(d)
                mb = b.hmr(t)
                b.cz(ax[i], d, cond=mb)
                b.x(d)
                b.x(ax[i])
        else:
            _release(b, t, ax[i], d)


def kb_ctrl_add_flag(b, ctrl, y, a, flag, anc, inv=False):
    n = len(a)
    assert len(y) == n
    plan = _plan(n, len(anc))
    if inv:
        for i in range(n):
            b.cx(ctrl, y[i])  # e = y ^ ctrl (subtract spell)
    cin0 = ctrl if inv else None

    if len(plan) == 1:
        top = _qq_fwd(b, a, y, 0, n, cin0, anc)
        b.ccx(ctrl, top, flag)
        if inv:
            b.cx(ctrl, flag)
        _qq_bwd(b, a, y, 0, n, cin0, anc, sum_ctrl=ctrl)
        if inv:
            for i in range(n):
                b.cx(ctrl, y[i])
        return

    keepers = []  # K_1..K_{j} live wires
    for j, (lo, hi) in enumerate(plan):
        cin = cin0 if j == 0 else keepers[-1]
        cw = _wires(anc, keepers, hi - lo)
        last = j == len(plan) - 1
        top = _qq_fwd(b, a, y, lo, hi, cin, cw)
        if last:
            b.ccx(ctrl, top, flag)
            if inv:
                b.cx(ctrl, flag)
            _qq_bwd(b, a, y, lo, hi, cin, cw, sum_ctrl=ctrl)
        else:
            _qq_bwd(b, a, y, lo, hi, cin, cw, sum_ctrl=ctrl, keep_top=True)
            slot = _wires(anc, keepers + [cw[-1]], 1)[0]
            b.swap(cw[-1], slot)  # keeper -> reserved slot (free)
            keepers.append(slot)

    # ---- erases, descending; the keeper-erase carry identity:
    # carry(s + (e^ctrl) + (K_{j-1}^ctrl)) = K_j ^ ctrl, e as-run.
    for j in range(len(plan) - 2, -1, -1):
        lo, hi = plan[j]
        K = keepers[j]
        prev = (None if inv else ctrl) if j == 0 else keepers[j - 1]
        dress_prev = j > 0  # K_{j-1} ^ ctrl via CX sandwich
        for i in range(lo, hi):
            b.cx(ctrl, y[i])  # e ^ ctrl
        if dress_prev:
            b.cx(ctrl, prev)
        ke = _wires(anc, keepers[: j + 1], hi - lo)
        d = _qq_fwd(b, a, y, lo, hi, prev, ke)
        b.cx(d, K)
        b.cx(ctrl, K)
        _qq_bwd(b, a, y, lo, hi, prev, ke)
        if dress_prev:
            b.cx(ctrl, prev)
        for i in range(lo, hi):
            b.cx(ctrl, y[i])
        keepers.pop()  # K_j is |0>, back to the pool
    if inv:
        for i in range(n):
            b.cx(ctrl, y[i])  # restore y at the very end


def kb_lt_flag(b, ctrl, v, y, flag, anc, topg=None):
    # topg: priced top-window compare truncation (pricing: each
    # circuit's priced-mul-windows block at FOLD_WIN/CMP_TOPG).
    n = len(v)
    assert len(y) == n
    lo0 = 0 if (topg is None or topg >= n) else n - topg
    span = n - lo0
    plan = [(lo + lo0, hi + lo0) for lo, hi in _plan(span, len(anc))]
    for q in v[lo0:]:
        b.x(q)
    if len(plan) == 1:
        top = _qq_fwd(b, v, y, lo0, n, None, anc)
        b.ccx(ctrl, top, flag)
        _qq_bwd(b, v, y, lo0, n, None, anc)
        for q in v[lo0:]:
            b.x(q)
        return
    keepers = []
    for j, (lo, hi) in enumerate(plan):
        cin = None if j == 0 else keepers[-1]
        cw = _wires(anc, keepers, hi - lo)
        last = j == len(plan) - 1
        top = _qq_fwd(b, v, y, lo, hi, cin, cw)
        if last:
            b.ccx(ctrl, top, flag)
            _qq_bwd(b, v, y, lo, hi, cin, cw)
        else:
            _qq_bwd(b, v, y, lo, hi, cin, cw, keep_top=True)
            slot = _wires(anc, keepers + [cw[-1]], 1)[0]
            b.swap(cw[-1], slot)
            keepers.append(slot)
    for j in range(len(plan) - 2, -1, -1):  # data unmodified: recompute
        lo, hi = plan[j]
        K = keepers[j]
        prev = None if j == 0 else keepers[j - 1]
        ke = _wires(anc, keepers[: j + 1], hi - lo)
        d = _qq_fwd(b, v, y, lo, hi, prev, ke)
        b.cx(d, K)
        _qq_bwd(b, v, y, lo, hi, prev, ke)
        keepers.pop()
    for q in v[lo0:]:
        b.x(q)


def kb_fold_const(b, F, cval, a, anc, inv=False, win=None):
    # win: priced chain truncation (pricing: each circuit's
    # priced-mul-windows block at FOLD_WIN/CMP_TOPG — the pseudo-Mersenne
    # tail only propagates fwd / saturates-to-F inv, and the truncated
    # tail sums cancel; escape = priced value error).
    n = len(a)
    eff = cval if not inv else ((1 << n) - cval) % (1 << n)
    kb = int_bits(eff, n)
    nc = n - 1
    trunc = win is not None and win < nc
    if trunc:
        nc = win
    plan = _plan(nc, len(anc))
    if len(plan) == 1:
        cw = _cf_fwd(b, a, kb, F, 0, nc, None, anc)
        top = cw[nc - 1]
        if not trunc:
            if kb[n - 1]:
                b.cx(F, a[n - 1])
            if top is not None:
                b.cx(top, a[n - 1])
        _cf_bwd(b, a, kb, F, 0, nc, None, cw)
        return
    keepers = []  # entries may be None
    live = []  # non-None keeper wires
    for j, (lo, hi) in enumerate(plan):
        cin = None if j == 0 else keepers[-1]
        cw_anc = _wires(anc, live, hi - lo)
        last = j == len(plan) - 1
        cw = _cf_fwd(b, a, kb, F, lo, hi, cin, cw_anc)
        if last:
            top = cw[hi - lo - 1]
            if not trunc:
                if kb[n - 1]:
                    b.cx(F, a[n - 1])
                if top is not None:
                    b.cx(top, a[n - 1])
            _cf_bwd(b, a, kb, F, lo, hi, cin, cw)
        else:
            _cf_bwd(b, a, kb, F, lo, hi, cin, cw, keep_top=True)
            K = cw[hi - lo - 1]
            if K is None:  # no carry yet: nothing persists
                keepers.append(None)
            else:
                slot = _wires(anc, live + [K], 1)[0]
                b.swap(K, slot)
                keepers.append(slot)
                live.append(slot)
    for j in range(len(plan) - 2, -1, -1):
        lo, hi = plan[j]
        K = keepers[j]
        if K is None:
            keepers.pop()
            continue
        prev = None if j == 0 else keepers[j - 1]
        m = hi - lo
        ke = _wires(anc, [w for w in keepers[: j + 1] if w is not None], m)
        # keeper-erase borrow identity: borrow(s_blk - K_{j-1} - C_blk)
        # == K_j when F=1
        for i in range(lo, hi):
            b.x(a[i])
        cwk, d = _clt_fwd_cin(b, a, kb, lo, hi, ke, prev)
        assert d is not None
        b.ccx(F, d, K)
        _clt_bwd_cin(b, a, kb, lo, hi, cwk, prev)
        for i in range(lo, hi):
            b.x(a[i])
        keepers.pop()
        live.remove(K)


def kb_dbl_fold(b, reg, p, n, anc, inv=False, win=None):
    c = (1 << n) - p
    assert c & 1 and 1 < c < (1 << (n - 1)), ("not pseudo-Mersenne", p)
    half = (c - 1) >> 1
    if not inv:
        for i in range(n - 1, 0, -1):
            b.swap(reg[i], reg[i - 1])
        kb_fold_const(b, reg[0], half, reg[1:], anc, win=win)
    else:
        kb_fold_const(b, reg[0], half, reg[1:], anc, inv=True, win=win)
        for i in range(1, n):
            b.swap(reg[i], reg[i - 1])


def _span_tables(b, n, spans):
    """Per-Builder DATA tables selecting each affine span of a control
    sequence: table s has bit t = [t inside span s].
    """
    cache = _bcache(b, "_luo_span_tables")
    key = (n, tuple((t0, ln) for (t0, ln, _, _) in spans))
    if key not in cache:
        cache[key] = [
            b.declare_table([1 if t0 <= t < t0 + ln else 0 for t in range(n)])
            for (t0, ln, _, _) in spans
        ]
    return cache[key]


def _horner_edge_tables(b, n):
    """Per-Builder DATA tables for the Horner edges, as complement
    pairs: (tnot0, tnotl) gate the doubling ([t > 0] forward / [t <
    n-1] inverse), and (tis0, tisl) are their bitwise negations gating
    the zero-target rewrite's edge iterations (forward first / inverse
    last).  The declaration order (tnot0, tnotl, tis0, tisl) is part of
    the certified stream: table ids mint in declare order.
    """
    cache = _bcache(b, "_luo_horner_edge_tables")
    if n not in cache:
        not0 = [1 if t > 0 else 0 for t in range(n)]
        notl = [1 if t < n - 1 else 0 for t in range(n)]
        cache[n] = (
            b.declare_table(not0),
            b.declare_table(notl),
            b.declare_table([1 - v for v in not0]),
            b.declare_table([1 - v for v in notl]),
        )
    return cache[n]


def horner_mul(b, xc, y, out, p, n, ma, *, cma, dbl, square=False, inv=False):
    """Windowed Horner modular multiplier with the zero-target edge
    rewrite: every mul site in the point-add flow starts (fwd) with
    out = |0> / ends (inv) with out = the exact product, so the edge
    Horner step COPIES (CCX fan) instead of mod-adding. The edge copy
    computes the same map as the mod-add it replaces (adding into
    |0>), and the fold-based doubling is map-identical to the plain
    mod-p doubling — both are respellings, not approximations.
    cma / dbl are the circuit's controlled mod-add and doubling (its
    _mul_cma / _mul_dbl): each circuit passes its own priced windows.
    """
    tq = ma["tq"]
    ma = dict(ma)
    ma["dirty"] = list(y) + [tq]  # doubling borrows n+1 dirty qubits
    tnot0, tnotl, tis0, tisl = _horner_edge_tables(b, n)
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

        def zt_copy(cq):
            for i in range(n):
                b.ccx(cq, y[i], out[i])

        if not inv:
            with b.gated(tnot0, t):
                dbl(b, out, p, n, ma)
            if direct:
                with b.gated(tis0, t):
                    zt_copy(ctrl_expr())
                with b.gated(tnot0, t):
                    cma(b, ctrl_expr(), y, out, p, n, ma)
            else:
                copy_ctrl()
                with b.gated(tis0, t):
                    zt_copy(tq)
                with b.gated(tnot0, t):
                    cma(b, tq, y, out, p, n, ma)
                copy_ctrl()
        else:
            if direct:
                with b.gated(tnotl, t):
                    cma(b, ctrl_expr(), y, out, p, n, ma, inv=True)
                with b.gated(tisl, t):
                    zt_copy(ctrl_expr())
            else:
                copy_ctrl()
                with b.gated(tnotl, t):
                    cma(b, tq, y, out, p, n, ma, inv=True)
                with b.gated(tisl, t):
                    zt_copy(tq)
                copy_ctrl()
            with b.gated(tnotl, t):
                dbl(b, out, p, n, ma, inv=True)


def gc_lazy_cma(b, ctrl, addend, target, p, n, ma, anc, inv=False, win=None, topg=None):
    """Emit the lazy controlled mod-add on the k-block gadgets (the lazy
    mirror map: carry computed unconditionally, flag ^= ctrl*carry,
    fold +c on the flag alone, flag ^= ctrl*[v' < y]).
    """
    flag = ma["flag"]
    c = ((1 << n) - p) % (1 << n)
    if not inv:
        kb_ctrl_add_flag(b, ctrl, addend, target, flag, anc)
        kb_fold_const(b, flag, c, target, anc, win=win)
        kb_lt_flag(b, ctrl, target, addend, flag, anc, topg=topg)
    else:
        kb_lt_flag(b, ctrl, target, addend, flag, anc, topg=topg)
        kb_fold_const(b, flag, c, target, anc, inv=True, win=win)
        kb_ctrl_add_flag(b, ctrl, addend, target, flag, anc, inv=True)


# ===========================================================================
# classical-bit input layer (../_shared/clsbits.py): the point I/O
# registers and the per-shot cbspec networks, re-exported
# ===========================================================================

_CLSBITS = load_module(pathlib.Path(__file__).resolve().parent / "clsbits.py", "clsbits")
ClsBits = _CLSBITS.ClsBits
const_modp_bit_specs = _CLSBITS.const_modp_bit_specs
declare_point_io = _CLSBITS.declare_point_io
point_cbspecs = _CLSBITS.point_cbspecs


# ===========================================================================
# modular arithmetic + EEA flows
# ===========================================================================


def ctrl_mod_add(b, ctrl, addend, target, p, n, ma, inv=False):
    """target <- target + ctrl*addend mod p (paper App. A, 11n+O(1) Toffoli).
    ma: dict(carry, flag); the addend register doubles as the comparator /
    dirty workspace.  inv gives the exact inverse.
    """
    carry, flag = ma["carry"], ma["flag"]
    k_cb_p = int_bits(((1 << n) - p) % (1 << n), n)
    dirty = list(addend)

    def fwd_parts():
        ctrl_qq_add_flag(b, ctrl, addend, target, carry, flag)
        gidney_cmp_ge(b, target, k_cb_p, flag, dirty, ma["clean3"])
        gidney_add_const(b, target, k_cb_p, dirty[: n - 1], ma["clean3"], ctrl=flag)
        cuccaro_lt_flag(b, target, addend, flag, carry, ctrl=ctrl)

    def inv_parts():
        cuccaro_lt_flag(b, target, addend, flag, carry, ctrl=ctrl)
        gidney_add_const(b, target, int_bits(p, n), dirty[: n - 1], ma["clean3"], ctrl=flag)
        gidney_cmp_ge(b, target, k_cb_p, flag, dirty, ma["clean3"])
        ctrl_qq_add_flag(b, ctrl, addend, target, carry, flag, inv=True)

    inv_parts() if inv else fwd_parts()


def dbl_modp(b, reg, p, n, ma, inv=False):
    """reg <- 2*reg mod p (paper App. A doubling, 6n+O(1) Toffoli)."""
    top, f2 = ma["top"], ma["flag"]
    ext = list(reg) + [top]
    k_cb = int_bits(((1 << (n + 1)) - p) % (1 << (n + 1)), n + 1)
    dirty = ma["dirty"]

    def fwd():
        for i in range(n, 0, -1):  # cyclic left shift into top
            b.swap(ext[i], ext[i - 1])
        gidney_cmp_ge(b, ext, k_cb, f2, dirty[: n + 1], ma["clean3"])
        gidney_add_const(b, ext, k_cb, dirty[:n], ma["clean3"], ctrl=f2)
        b.cx(reg[0], f2)

    def bwd():
        b.cx(reg[0], f2)
        gidney_add_const(b, ext, int_bits(p, n + 1), dirty[:n], ma["clean3"], ctrl=f2)
        gidney_cmp_ge(b, ext, k_cb, f2, dirty[: n + 1], ma["clean3"])
        for i in range(1, n + 1):
            b.swap(ext[i], ext[i - 1])

    bwd() if inv else fwd()


def const_modp(b, reg, specs, flagq, dirty, clean3, p, n, inv=False):  # noqa: ARG001 - shared (p, n) gadget signature, ~25 callers
    """reg <- reg +/- C mod p (0 < C < p, reg < p), flag returned clean.
    specs: the four cbspec vectors k1..k4, compile-time ints or per-shot
    Bit handles (const_modp_bit_specs).
    """
    if not inv:
        gidney_cmp_ge(b, reg, specs["k1"], flagq, dirty, clean3)
        gidney_add_const(b, reg, specs["k1"], dirty[: n - 1], clean3, ctrl=flagq)
        b.x(flagq)
        gidney_add_const(b, reg, specs["k3"], dirty[: n - 1], clean3, ctrl=flagq)
        gidney_cmp_ge(b, reg, specs["k4"], flagq, dirty, clean3)
    else:
        gidney_cmp_ge(b, reg, specs["k4"], flagq, dirty, clean3)
        gidney_add_const(b, reg, specs["k4"], dirty[: n - 1], clean3, ctrl=flagq)
        b.x(flagq)
        gidney_add_const(b, reg, specs["k2"], dirty[: n - 1], clean3, ctrl=flagq)
        gidney_cmp_ge(b, reg, specs["k1"], flagq, dirty, clean3)


def neg_modp(b, reg, dirty, clean3, p, n):
    """reg <- p - reg (0 < reg < p): complement + add p+1 mod 2^n."""
    for q in reg:
        b.x(q)
    gidney_add_const(b, reg, int_bits(p + 1, n), dirty[: n - 1], clean3)


def emit_idiv(b, X, Y, A, Rws, Rwy, p, n, ma):
    """|x>_X |y>_Y |0>_A -> |x>_X |y/x>_Y |0>_A (Fig. 12, x != 0)."""
    emit_eea(b, Rws)  # X -> x^{-1}, workspace A
    mul_modp(b, Rws.tprime, Y, A, p, n, ma)  # A = y * x^{-1}
    mbits = []
    for q in Y:
        mbits.append(b.hmr(q))
    emit_eea(b, Rwy, inv=True)  # X restored, workspace Y
    mul_modp(b, X, A, Y, p, n, ma)  # recompute y
    for q, m in zip(Y, mbits):
        b.z(q, cond=m)  # Z^b phase fixup
    mul_modp(b, X, A, Y, p, n, ma, inv=True)
    for yq, aq in zip(Y, A):
        b.swap(yq, aq)


def emit_imul(b, X, Y, A, Rwy, p, n, ma):
    """|x>_X |y>_Y |0>_A -> |x>_X |xy>_Y |0>_A (x != 0)."""
    mul_modp(b, X, Y, A, p, n, ma)  # A = x*y
    mbits = []
    for q in Y:
        mbits.append(b.hmr(q))
    emit_eea(b, Rwy)  # X -> x^{-1}, workspace Y
    mul_modp(b, Rwy.tprime, A, Y, p, n, ma)  # recompute y = x^{-1} * xy
    for q, m in zip(Y, mbits):
        b.z(q, cond=m)
    mul_modp(b, Rwy.tprime, A, Y, p, n, ma, inv=True)
    emit_eea(b, Rwy, inv=True)  # X restored
    for yq, aq in zip(Y, A):
        b.swap(yq, aq)
