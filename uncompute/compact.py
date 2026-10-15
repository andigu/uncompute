"""uncompute.compact: the op-stream serializer (producer side).

The :mod:`uncompute.ops` docstring specifies the format. This module owns
the *segment model* that the :class:`~uncompute.Builder` records into (REGS,
DATA, FLAT, and LOOP trees), and it turns the model into the byte stream:

    MAGIC ‖ segment* ‖ END-segment

A stream means its **expansion**: the compact form changes the spelling,
never the meaning. Everything here is exact integer math (loop counts,
affine operand resolution, data-table popcounts), so the END segment's
``expanded_op_count`` is always the exact post-gating op count.

This module only produces streams; the evaluator (``evalcore``) parses them
with its own, separate code.

The model (the Builder constructs it; code can also build it by hand):

* :class:`Regs`      — the hoisted register declarations (REGS 0x01).
* :class:`Data`      — one bit table (DATA 0x02). It doubles as the table
  handle that ``Builder.declare_table`` returns.
* :class:`Flat`      — a run of depth-0 op records (FLAT 0x03).
* :class:`Loop`      — a symbolic loop (LOOP 0x04 at top level, item 0x01 when
  nested); ``items`` mixes :class:`OpRec` and nested :class:`Loop`.
* :class:`OpRec`     — one op record: kind, operand list in stream order
  (each operand ``(base, strides)`` affine in the enclosing loop indices),
  an optional outcome condition ``oc`` (sign kinds only), an optional bit
  condition ``cc`` (appended last), and an optional data-gate block
  ``(table_id, offset, strides)``.

:data:`KIND_OPERANDS` holds the operand order of every stream kind. The
condition rules are ``uncompute.ops.OC_ALLOWED_KINDS`` and
``CC_ALLOWED_KINDS``. The serializer, the expansion helpers, and the Builder
all read these tables.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .ops import (
    CC_ALLOWED_KINDS,
    NO_BIT,
    NO_OUT,
    NO_QUBIT,
    NO_REG,
    OC_ALLOWED_KINDS,
    OpKind,
    _is_fifo,
    resolve_ops_path,
)

MAGIC = b"OPSTRM01"  # the 8-byte stream magic

SEG_REGS = 0x01
SEG_DATA = 0x02
SEG_FLAT = 0x03
SEG_LOOP = 0x04
SEG_END = 0xFF

ITEM_OP = 0x00
ITEM_LOOP = 0x01

FLAG_OC = 0x01  # flags bit 0: an oc operand follows the required ones
FLAG_GATED = 0x02  # flags bit 1: has a data-gate block
FLAG_CC = 0x04  # flags bit 2: a cc operand follows, last (after oc)

# Register kinds in the REGS segment: one u8 per register.
WKIND_QUBIT = 0
WKIND_BIT = 1

MAX_LOOP_DEPTH = 4  # LOOP nesting limit
MAX_LOOP_ITEMS = 0xFFFF  # n_items is u16, and may not be 0

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_KIND_FLAGS = struct.Struct("<BB")  # op record header: kind byte + flags byte

_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1

# ---- the per-kind operand tables -------------------------------------------------
# Stream operand order per kind, named by the tuple slot each operand fills
# on expansion: (kind, qa, qb, qt, ot, oc, rt, ct, cc). OXOR's oc is a
# required operand (its outcome source). Flag bit 0 is legal only on the
# sign kinds (ops.OC_ALLOWED_KINDS: Z, CZ, NEG).
KIND_OPERANDS = {
    int(OpKind.NEG): (),
    int(OpKind.X): ("qt",),
    int(OpKind.Z): ("qt",),
    int(OpKind.R): ("qt",),
    int(OpKind.CX): ("qb", "qt"),
    int(OpKind.CZ): ("qb", "qt"),
    int(OpKind.SWAP): ("qb", "qt"),
    int(OpKind.HMR): ("qt", "ot"),
    int(OpKind.CCX): ("qa", "qb", "qt"),
    int(OpKind.CCZ): ("qa", "qb", "qt"),
    int(OpKind.OXOR): ("ot", "oc"),
    int(OpKind.BIT_INVERT): ("ct",),
    int(OpKind.BIT_STORE0): ("ct",),
    int(OpKind.BIT_STORE1): ("ct",),
}

# Flat 9-tuple slot per operand field name:
# (kind, qa, qb, qt, ot, oc, rt, ct, cc).
_FIELD_SLOT = {"qa": 1, "qb": 2, "qt": 3, "ot": 4, "oc": 5, "ct": 7, "cc": 8}

# Per-kind flat-tuple slots of the canonical operands, in canonical order
# (derived from KIND_OPERANDS; drives the fast tuple->OpRec conversion).
_KIND_SLOTS = {kind: tuple(_FIELD_SLOT[n] for n in names) for kind, names in KIND_OPERANDS.items()}


def _kind_name(kind):
    try:
        return OpKind(kind).name
    except ValueError:
        return f"kind {kind}"


def raise_oc_condition(kind):
    """Raise the error for an outcome condition on a kind that cannot carry
    one. Every producer path (Builder, OpRec, tuple conversion) raises this
    same error.
    """
    raise ValueError(f"{_kind_name(kind)} cannot carry an outcome condition (only Z, CZ, NEG can)")


def raise_cc_condition(kind):
    """Raise the error for a bit condition on a kind that cannot carry one.
    Only X, Z, NEG, CX, CZ, and the bit-only kinds can, because promotion
    past arity 3 does not exist.
    """
    raise ValueError(
        f"{_kind_name(kind)} cannot carry a bit condition "
        "(only X, Z, NEG, CX, CZ, and the bit-only kinds can)"
    )


# ---- segment / item model ------------------------------------------------------


class Regs:
    """The (single, hoisted) REGS segment: ``registers[r]`` is register ``r``'s
    ``(reg_kind, wires)`` pair — ``WKIND_QUBIT`` wires are qubit ids,
    ``WKIND_BIT`` wires are bit ids. A plain wire iterable is accepted as a
    Qubit register.
    """

    __slots__ = ("registers",)

    def __init__(self, registers):
        norm = []
        for reg in registers:
            if (
                isinstance(reg, tuple)
                and len(reg) == 2
                and reg[0] in (WKIND_QUBIT, WKIND_BIT)
                and isinstance(reg[1], (list, tuple))
            ):
                kind, wires = reg
            else:
                kind, wires = WKIND_QUBIT, reg
            norm.append((int(kind), tuple(int(i) for i in wires)))
        self.registers = norm


class Data:
    """One DATA segment / bit-table handle. ``bits`` is a tuple of 0/1; bit
    ``i`` lands in byte ``i // 8`` at bit ``i % 8`` on the wire.
    """

    __slots__ = ("bits", "table_id")

    def __init__(self, table_id, bits):
        self.table_id = int(table_id)
        self.bits = tuple(int(b) for b in bits)
        if any(b not in (0, 1) for b in self.bits):
            raise ValueError("table bits must all be 0 or 1")

    def __repr__(self):
        return f"Data(table_id={self.table_id}, n_bits={len(self.bits)})"


class OpRec:
    """One op record.

    ``operands``: tuple of ``(base, strides)`` in the kind's canonical order
    (:data:`KIND_OPERANDS`); ``strides`` has one i64 per enclosing loop,
    outermost first — empty at depth 0. ``oc``: the optional outcome
    condition in the same shape (legal only on Z, CZ, NEG —
    ``ops.OC_ALLOWED_KINDS``). ``cc``: the optional bit condition in
    the same shape (legal only on ``ops.CC_ALLOWED_KINDS``; encoded last).
    ``gate``: optional data-gate block ``(table_id, offset, strides)``.
    """

    __slots__ = ("cc", "gate", "kind", "oc", "operands")

    def __init__(self, kind, operands, oc=None, gate=None, cc=None):
        kind = int(kind)
        names = KIND_OPERANDS.get(kind)
        if names is None:
            raise ValueError(f"unknown op kind {kind}")
        operands = tuple((int(base), tuple(int(s) for s in strides)) for base, strides in operands)
        if len(operands) != len(names):
            raise ValueError(
                f"{_kind_name(kind)} takes {len(names)} operand(s) {names}, got {len(operands)}"
            )
        if oc is not None:
            if kind not in OC_ALLOWED_KINDS:
                raise_oc_condition(kind)
            oc = (int(oc[0]), tuple(int(s) for s in oc[1]))
        if cc is not None:
            if kind not in CC_ALLOWED_KINDS:
                raise_cc_condition(kind)
            cc = (int(cc[0]), tuple(int(s) for s in cc[1]))
        if gate is not None:
            gate = (int(gate[0]), int(gate[1]), tuple(int(s) for s in gate[2]))
        depths = {len(s) for _, s in operands}
        if oc is not None:
            depths.add(len(oc[1]))
        if cc is not None:
            depths.add(len(cc[1]))
        if gate is not None:
            depths.add(len(gate[2]))
        if len(depths) > 1:
            raise ValueError(
                f"{_kind_name(kind)}: inconsistent stride depths {sorted(depths)} "
                "within one op record"
            )
        self.kind = kind
        self.operands = operands
        self.oc = oc
        self.cc = cc
        self.gate = gate

    def __repr__(self):
        return (
            f"OpRec({_kind_name(self.kind)}, operands={self.operands}, "
            f"oc={self.oc}, cc={self.cc}, gate={self.gate})"
        )


class Flat:
    """A FLAT segment: a run of depth-0 op records.

    ``ops`` items are :class:`OpRec` s, or — the producer fast path for the
    Builder's flat recording — raw flat 9-tuples ``(kind, qa, qb, qt, ot,
    oc, rt, ct, cc)``, which serialize byte-identically to their
    :func:`oprec_from_tuple` conversion without materializing the OpRec.

    Sized non-list iterables — the Builder's zero-copy run views over its
    columnar record — are held as-is; lists, tuples, and plain iterables
    are materialized as a list.
    """

    __slots__ = ("ops",)

    def __init__(self, ops):
        if isinstance(ops, (list, tuple)) or not hasattr(ops, "__len__"):
            ops = list(ops)
        self.ops = ops


class Loop:
    """A LOOP: ``count`` iterations of ``items`` (OpRec or nested Loop)."""

    __slots__ = ("count", "items")

    def __init__(self, count, items):
        self.count = int(count)
        self.items = list(items)

    def __repr__(self):
        return f"Loop(count={self.count}, items={self.items!r})"


def oprec_from_tuple(op):
    """Convert a flat 9-tuple ``(kind, qa, qb, qt, ot, oc, rt, ct, cc)`` to
    a depth-0 :class:`OpRec`.

    Fast path: the tuple shape makes :class:`OpRec`'s generic argument
    validation redundant (operand count/order come straight from the kind's
    slot table, depths are uniformly 0), so the record is filled in directly.
    A set ``oc`` on a kind outside ``OC_ALLOWED_KINDS`` raises — unless the
    kind consumes slot 5 as a canonical operand (OXOR's source), in which
    case it is not a condition at all. A set ``cc`` on a kind outside
    ``CC_ALLOWED_KINDS`` raises; a set ``ct`` is legal only where it is a
    canonical operand (the bit-only kinds).
    """
    kind, _qa, _qb, _qt, _ot, oc, _rt, ct, cc = op
    kind = int(kind)
    slots = _KIND_SLOTS.get(kind)
    if slots is None:
        raise ValueError(f"cannot convert op kind {kind} to a stream op record")
    if ct != NO_BIT and 7 not in slots:
        raise ValueError(f"{_kind_name(kind)} cannot carry a ct operand")
    rec = OpRec.__new__(OpRec)
    rec.kind = kind
    rec.operands = tuple((int(op[s]), ()) for s in slots)
    if oc == NO_OUT or 5 in slots:
        rec.oc = None
    elif kind in OC_ALLOWED_KINDS:
        rec.oc = (int(oc), ())
    else:
        raise_oc_condition(kind)
    if cc == NO_BIT:
        rec.cc = None
    elif kind in CC_ALLOWED_KINDS:
        rec.cc = (int(cc), ())
    else:
        raise_cc_condition(kind)
    rec.gate = None
    return rec


# ---- exact gated popcounts (drives END counts and phase stats) -----------------


def box_bounds(base, strides, counts):
    """Return the attained lo/hi corners of the affine form ``base + Σ strides_j·i_j``
    over the loop box ``Π [0, counts_j)`` — each stride contributes its own
    extreme corner, so both bounds are reached by an actual iteration.
    """
    lo = base + sum(min(0, s * (c - 1)) for s, c in zip(strides, counts))
    hi = base + sum(max(0, s * (c - 1)) for s, c in zip(strides, counts))
    return lo, hi


_POPCOUNT_MEMO = {}


def gated_popcount(bits, offset, strides, counts):
    """Count the iterations of the loop box ``Π [0, counts_j)`` whose gate bit
    ``bits[offset + Σ strides_j · i_j]`` is 1 — i.e. how many expanded ops a
    data-gated op record contributes.

    Exact integer math. The multiplicity distribution of the bit index is
    built level by level; per-level cost is bounded by the *span* of reachable
    indices (≤ ``len(bits)`` after the range check), never by the box volume.
    Raises ``ValueError`` if any reachable index (the box corners, which are
    attained) falls outside ``[0, len(bits))`` — the same condition the
    evaluator reports per iteration.

    Memoized on ``(id(bits), offset, strides, counts)``: loop-form authoring
    re-emits identical gated shapes thousands of times (measured >50% of a
    large build). The cached entry keeps a strong reference to ``bits`` and
    validates it with ``is``, so id reuse after GC can never alias a stale
    entry. Pure cache — results are bit-identical.
    """
    key = (id(bits), offset, tuple(strides), tuple(counts))
    _e = _POPCOUNT_MEMO.get(key)
    if _e is not None and _e[0] is bits:
        return _e[1]
    if len(strides) != len(counts):
        raise ValueError("strides/counts length mismatch")
    for c in counts:
        if c == 0:
            return 0
    n_bits = len(bits)
    lo, hi = box_bounds(offset, strides, counts)
    if lo < 0 or hi >= n_bits:
        raise ValueError(
            f"gated bit index spans [{lo}, {hi}] over the loop box, but the table has {n_bits} bits"
        )
    dist = {offset: 1}
    for s, c in zip(strides, counts):
        if s == 0 or c == 1:
            if c > 1:
                dist = {v: m * c for v, m in dist.items()}
        else:
            dist = _convolve_progression(dist, s, c)
    r = sum(m for v, m in dist.items() if bits[v])
    _POPCOUNT_MEMO[key] = (bits, r)
    return r


def _convolve_progression(dist, s, n):
    """Convolve multiplicity map ``dist`` with the n-term progression of step
    ``s`` (each source value v spreads to v, v+s, …, v+s·(n-1)). Sparse
    difference-array sweep per residue class of |s|: cost O(reachable span).
    """
    if s < 0:
        dist = {v + s * (n - 1): m for v, m in dist.items()}
        s = -s
    by_res = {}
    for v, m in dist.items():
        d = by_res.setdefault(v % s, {})
        p = v // s
        d[p] = d.get(p, 0) + m
    out = {}
    for r, entries in by_res.items():
        diff = {}
        for p, m in entries.items():
            diff[p] = diff.get(p, 0) + m
            diff[p + n] = diff.get(p + n, 0) - m
        acc = 0
        prev = 0
        for p in sorted(diff):
            if acc:
                for q in range(prev, p):
                    out[r + s * q] = acc
            acc += diff[p]
            prev = p
    return out


# ---- expansion helpers (exact, shared with the Builder) -------------------------


def item_count(item, counts, tables):
    """Expanded-op contribution of one OpRec/Loop under enclosing loop counts
    ``counts`` (outermost first): Π counts for a plain op, the exact gated
    popcount for a data-gated op, recursive for loops.
    """
    if isinstance(item, Loop):
        inner = counts + (item.count,)
        return sum(item_count(x, inner, tables) for x in item.items)
    if item.gate is None:
        n = 1
        for c in counts:
            n *= c
        return n
    table_id, offset, strides = item.gate
    if table_id not in tables:
        raise ValueError(f"gated op references undeclared table {table_id}")
    return gated_popcount(tables[table_id], offset, strides, counts)


def expanded_count(segments):
    """Return the END segment's ``expanded_op_count``: exact number of expanded,
    post-gating ops the segment list denotes.
    """
    tables = {}
    total = 0
    for seg in segments:
        if isinstance(seg, Data):
            if seg.table_id in tables:
                raise ValueError(f"duplicate table_id {seg.table_id}")
            tables[seg.table_id] = seg.bits
        elif isinstance(seg, Flat):
            # A run that advertises its plain op count (every item expands
            # to exactly 1 — the Builder's all-tuple runs) skips the walk.
            n = getattr(seg.ops, "plain_op_count", None)
            if n is not None:
                total += n
                continue
            for op in seg.ops:
                # Ungated depth-0 op (raw flat tuples are always ungated):
                # contributes exactly 1 (the common case; skip the generic
                # item_count machinery per op).
                if type(op) is tuple or op.gate is None:
                    total += 1
                else:
                    total += item_count(op, (), tables)
        elif isinstance(seg, Loop):
            total += item_count(seg, (), tables)
        elif isinstance(seg, Regs):
            pass
        else:
            raise TypeError(f"not a stream segment: {seg!r}")
    return total


def _resolve(base, strides, idx):
    return base + sum(s * i for s, i in zip(strides, idx))


def iter_expand_item(item, idx, tables):
    """Yield flat-style 9-tuples for one OpRec/Loop under enclosing loop index
    vector ``idx`` (outermost first). Gated-off ops are skipped; resolved ids
    outside ``[0, 2^32)`` and out-of-range gate bits raise.
    """
    if isinstance(item, Loop):
        for t in range(item.count):
            inner = idx + (t,)
            for it in item.items:
                yield from iter_expand_item(it, inner, tables)
        return
    if item.gate is not None:
        table_id, offset, strides = item.gate
        bits = tables.get(table_id)
        if bits is None:
            raise ValueError(f"gated op references undeclared table {table_id}")
        b = _resolve(offset, strides, idx)
        if b < 0 or b >= len(bits):
            raise ValueError(
                f"{_kind_name(item.kind)} at iteration {idx}: gate bit index {b} "
                f"out of range [0, {len(bits)})"
            )
        if not bits[b]:
            return
    rec = [item.kind, NO_QUBIT, NO_QUBIT, NO_QUBIT, NO_OUT, NO_OUT, NO_REG, NO_BIT, NO_BIT]
    for name, (base, strides) in zip(KIND_OPERANDS[item.kind], item.operands):
        v = _resolve(base, strides, idx)
        if v < 0 or v > _U32_MAX:
            raise ValueError(
                f"{_kind_name(item.kind)} at iteration {idx}: operand {name} "
                f"resolves to {v}, outside [0, 2^32)"
            )
        rec[_FIELD_SLOT[name]] = v
    if item.oc is not None:
        v = _resolve(item.oc[0], item.oc[1], idx)
        if v < 0 or v > _U32_MAX:
            raise ValueError(
                f"{_kind_name(item.kind)} at iteration {idx}: outcome condition "
                f"resolves to {v}, outside [0, 2^32)"
            )
        rec[5] = v
    if item.cc is not None:
        v = _resolve(item.cc[0], item.cc[1], idx)
        if v < 0 or v > _U32_MAX:
            raise ValueError(
                f"{_kind_name(item.kind)} at iteration {idx}: bit condition "
                f"resolves to {v}, outside [0, 2^32)"
            )
        rec[8] = v
    yield tuple(rec)


# ---- serialization ---------------------------------------------------------------


def _pack_u32(buf, v, what):
    if v < 0 or v > _U32_MAX:
        raise ValueError(f"{what} {v} out of u32 range")
    buf += _U32.pack(v)


def _pack_u64(buf, v, what):
    if v < 0 or v > _U64_MAX:
        raise ValueError(f"{what} {v} out of u64 range")
    buf += _U64.pack(v)


def _pack_i64(buf, v, what):
    if v < _I64_MIN or v > _I64_MAX:
        raise ValueError(f"{what} {v} out of i64 range")
    buf += _I64.pack(v)


def _encode_regs(seg):
    buf = bytearray(_U8.pack(SEG_REGS))
    _pack_u32(buf, len(seg.registers), "n_regs")
    for r, (kind, wires) in enumerate(seg.registers):
        buf += _U8.pack(kind)
        _pack_u32(buf, len(wires), f"register {r} n_wires")
        for wid in wires:
            _pack_u32(buf, wid, f"register {r} wire id")
    return bytes(buf)


def _encode_data(seg):
    buf = bytearray(_U8.pack(SEG_DATA))
    _pack_u32(buf, seg.table_id, "table_id")
    _pack_u64(buf, len(seg.bits), "n_bits")
    packed = bytearray((len(seg.bits) + 7) // 8)
    for i, b in enumerate(seg.bits):
        if b:
            packed[i >> 3] |= 1 << (i & 7)
    buf += packed
    return bytes(buf)


def _encode_op(buf, op, depth, table_ids):
    # Hot path (one call per recorded op): header packed in one struct call,
    # error-message strings built only inside the raise branches.
    oc = op.oc
    cc = op.cc
    gate = op.gate
    buf += _KIND_FLAGS.pack(
        op.kind,
        (FLAG_OC if oc is not None else 0)
        | (FLAG_GATED if gate is not None else 0)
        | (FLAG_CC if cc is not None else 0),
    )
    if gate is not None:
        table_id, offset, strides = gate
        if table_id not in table_ids:
            raise ValueError(
                f"{_kind_name(op.kind)} references table {table_id} before its DATA segment"
            )
        if len(strides) != depth:
            raise ValueError(
                f"{_kind_name(op.kind)}: gate has {len(strides)} stride(s) at loop depth {depth}"
            )
        _pack_u32(buf, table_id, "gate table_id")
        _pack_u64(buf, offset, "gate offset")
        for s in strides:
            _pack_i64(buf, s, "gate stride")
    operands = op.operands if oc is None else op.operands + (oc,)
    if cc is not None:
        operands = operands + (cc,)
    for base, strides in operands:
        if len(strides) != depth:
            raise ValueError(
                f"{_kind_name(op.kind)}: operand has {len(strides)} stride(s) at loop depth {depth}"
            )
        if base < 0 or base > _U32_MAX:
            raise ValueError(f"{_kind_name(op.kind)} operand base {base} out of u32 range")
        buf += _U32.pack(base)
        for s in strides:
            if s < _I64_MIN or s > _I64_MAX:
                raise ValueError(f"{_kind_name(op.kind)} operand stride {s} out of i64 range")
            buf += _I64.pack(s)


# Per-kind one-shot packers for the flat-tuple fast path: header (kind +
# flags) followed by the canonical operands as u32 bases (depth 0, no
# strides), without or with the trailing oc operand.
_TUPLE_ENC = {
    kind: (
        slots,
        struct.Struct("<BB" + "I" * len(slots)),
        struct.Struct("<BB" + "I" * (len(slots) + 1)),
    )
    for kind, slots in _KIND_SLOTS.items()
}


def _encode_tuple_op(buf, op):
    """Encode one flat 9-tuple byte-identically to
    ``_encode_op(buf, oprec_from_tuple(op), 0, ...)`` without materializing
    the OpRec (the hot path for flat programs). Malformed tuples,
    out-of-range operands, and illegal outcome conditions fall back to the
    conversion path so they raise the canonical errors.
    """
    # The fast path covers cc-free ops (ct is a canonical operand for the
    # bit-only kinds and flows through their slot table); a set cc falls back
    # to the conversion path, which appends it after oc with FLAG_CC — as
    # does a stray ct on a kind without a ct slot (canonical error there).
    enc = _TUPLE_ENC.get(op[0]) if len(op) == 9 and op[8] == NO_BIT else None
    if enc is not None and op[7] != NO_BIT and 7 not in enc[0]:
        enc = None
    if enc is None:
        # Unknown kind / wrong shape: oprec_from_tuple raises the canonical
        # error (e.g. "cannot convert op kind ... to a stream op record").
        buf += _flat_op_bytes(op)
        return
    slots, st_plain, st_oc = enc
    kind, oc = op[0], op[5]
    try:
        if oc == NO_OUT or 5 in slots:
            buf += st_plain.pack(kind, 0, *(op[s] for s in slots))
        elif kind in OC_ALLOWED_KINDS:
            buf += st_oc.pack(kind, FLAG_OC, *(op[s] for s in slots), oc)
        else:
            # Stray oc on a non-phase kind: the conversion path raises the
            # canonical "cannot carry an outcome condition" error.
            buf += _flat_op_bytes(op)
            return
    except struct.error:
        # Out-of-range or non-int operand: re-encode via the generic path,
        # which raises the canonical ValueError (or, for exotic int-likes
        # struct rejects, still yields the identical bytes).
        buf += _flat_op_bytes(op)


def _flat_op_bytes(op):
    tmp = bytearray()
    _encode_op(tmp, oprec_from_tuple(op), 0, frozenset())
    return tmp


# Flush threshold for streamed FLAT encoding: bounds the encode buffer so a
# program-sized segment never doubles the record's footprint at emit time.
_FLAT_CHUNK = 1 << 20


def _write_flat(seg, table_ids, fh):
    """Encode one FLAT segment straight to ``fh`` in bounded chunks — the
    byte stream is identical to a single-buffer encode. Runs whose ops
    carry a ``_native_encode`` seam (the Builder's ``_TapeRun``) may
    serialize through the C kernel — byte-identical by contract
    (uncompute._tapefast), with this pure path the normative reference.
    """
    enc = getattr(seg.ops, "_native_encode", None)
    if enc is not None and enc(table_ids, fh):
        return
    buf = bytearray(_U8.pack(SEG_FLAT))
    _pack_u32(buf, len(seg.ops), "FLAT n_ops")
    for op in seg.ops:
        if type(op) is tuple:
            _encode_tuple_op(buf, op)
        else:
            _encode_op(buf, op, 0, table_ids)
        if len(buf) >= _FLAT_CHUNK:
            fh.write(bytes(buf))
            del buf[:]
    if buf:
        fh.write(bytes(buf))


def _encode_loop(buf, loop, depth, table_ids):
    if depth > MAX_LOOP_DEPTH:
        raise ValueError(f"LOOP nesting depth {depth} exceeds {MAX_LOOP_DEPTH}")
    n = len(loop.items)
    if n == 0:
        raise ValueError("LOOP body may not be empty (n_items >= 1)")
    if n > MAX_LOOP_ITEMS:
        raise ValueError(f"LOOP body has {n} items; the u16 cap is {MAX_LOOP_ITEMS}")
    _pack_u64(buf, loop.count, "LOOP count")
    buf += _U16.pack(n)
    for item in loop.items:
        if isinstance(item, Loop):
            buf += _U8.pack(ITEM_LOOP)
            _encode_loop(buf, item, depth + 1, table_ids)
        elif isinstance(item, OpRec):
            buf += _U8.pack(ITEM_OP)
            _encode_op(buf, item, depth, table_ids)
        else:
            raise TypeError(f"not a LOOP item: {item!r}")


def write_stream(segments, fh):
    """Serialize ``segments`` (Regs/Data/Flat/Loop, in stream order) as a
    byte stream to file object ``fh``, END segment included. The END
    count is computed here via :func:`expanded_count`, so it can never desync
    from the emitted segments.
    """
    n_expanded = expanded_count(segments)  # also validates tables + gate ranges
    fh.write(MAGIC)
    seen_regs = False
    seen_body = False
    table_ids = set()
    for seg in segments:
        if isinstance(seg, Regs):
            if seen_regs:
                raise ValueError("at most one REGS segment")
            if seen_body:
                raise ValueError("REGS must precede every FLAT/LOOP segment")
            seen_regs = True
            fh.write(_encode_regs(seg))
        elif isinstance(seg, Data):
            if seg.table_id in table_ids:
                raise ValueError(f"duplicate table_id {seg.table_id}")
            fh.write(_encode_data(seg))
            table_ids.add(seg.table_id)
        elif isinstance(seg, Flat):
            seen_body = True
            _write_flat(seg, table_ids, fh)
        elif isinstance(seg, Loop):
            seen_body = True
            buf = bytearray(_U8.pack(SEG_LOOP))
            _encode_loop(buf, seg, 1, table_ids)
            fh.write(bytes(buf))
        else:
            raise TypeError(f"not a stream segment: {seg!r}")
    end = bytearray(_U8.pack(SEG_END))
    _pack_u64(end, n_expanded, "expanded_op_count")
    fh.write(bytes(end))


class _HashSink:
    """File-shaped digest accumulator (a re-emit's payload check without a
    write — the FIFO's read end consumed the first send and cannot uncompute).
    """

    __slots__ = ("_h",)

    def __init__(self, h):
        self._h = h

    def write(self, data):
        self._h.update(data)


class _TeeSink:
    """Write-through wrapper that hashes what it forwards."""

    __slots__ = ("_fh", "_h")

    def __init__(self, fh, h):
        self._fh = fh
        self._h = h

    def write(self, data):
        self._h.update(data)
        self._fh.write(data)


def write_compact(segments, dest):
    """Write an op stream to ``dest``: a file object (anything with
    ``.write``), or a path. Paths honor the ``UNCOMPUTE_OPS_PATH`` override;
    FIFOs are written straight through (with the double-send guard), regular
    files atomically (tmp + rename).
    """
    if hasattr(dest, "write"):
        write_stream(segments, dest)
        return
    path = resolve_ops_path(dest)
    if _is_fifo(path):
        import hashlib

        from .ops import fifo_prev_digest, fifo_send_guard

        if fifo_prev_digest(path) is not None:
            # A stream already went through this FIFO: hash without writing,
            # then let the guard rule — no-op on an identical resend, raise
            # on a changed one.
            h = hashlib.sha256()
            write_stream(segments, _HashSink(h))
            fifo_send_guard(path, h.digest())
            return
        # First send: stream straight through (never buffering the whole
        # payload), hashing alongside, and record the digest for the guard.
        h = hashlib.sha256()
        with Path(path).open("wb") as fh:  # blocks until the evaluator opens the read end
            write_stream(segments, _TeeSink(fh, h))
        fifo_send_guard(path, h.digest())
        return
    tmp = Path(f"{path}.tmp")
    with tmp.open("wb") as fh:
        write_stream(segments, fh)
    tmp.replace(path)
