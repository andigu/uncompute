"""uncompute circuit builder — the Python authoring API.

Allocate and recycle qubits with a **LIFO free-list** (so peak-qubit counts, the
qubit factor of the TQ-cost, match the evaluator exactly), declare registers,
apply gates, then ``emit()`` the op stream for the evaluator.

Typical use::

    from uncompute import Builder
    b = Builder()
    x = b.qubits(16); b.qubit_register(x)   # register 0
    anc = b.qubit()
    b.ccx(x[0], x[1], anc)
    # ... circuit ...
    m = b.hmr(anc)              # X-basis erase -> Outcome handle
    b.cz(x[0], x[1], cond=m)    # phase fixup conditioned on the outcome
    b.recycle(anc)
    print("peak qubits:", b.peak_qubits)
    b.emit()

The classical layer has two value families. **Outcomes**: :meth:`Builder.hmr`
demolishes a qubit and returns an opaque :class:`Outcome`; outcomes combine
with ``^`` / ``^=`` (OXOR) and may condition exactly ``z``/``cz``/``neg``
via ``cond=``. **Bits**: :meth:`Builder.bit` returns an opaque
:class:`Bit` (per-shot classical input data; :meth:`Builder.bit_register`
declares an input register, and the free bit-only ops derive values); a Bit
may condition ``x``/``z``/``neg``/``cx``/``cz`` via ``when=``, billing the
gate at its promoted kind. Conditions narrow a gate's per-shot action, never
its billing, so the billed gate counts of an emitted circuit are constants
of the circuit.

Uncomputation is built in: :meth:`Builder.checkpoint` takes a :class:`Mark`
and :meth:`Builder.mirror` emits the exact Bennett inverse of everything
recorded since (``until=`` bounds the region for conjugation shapes). The
mirror maps each construct to its cheapest exact inverse: ``and_compute``
<-> ``and_release`` (the mirrored release costs 0 Toffoli), ``qubit``
<-> ``recycle`` (the LIFO free-list returns the same ids), and self-inverse
gates re-emit verbatim.

Symbolic loops (the op stream of :mod:`uncompute.ops`): a program that repeats a
body many times can record it **once** and serialize compactly::

    with b.repeat(1000) as i:          # i is a symbolic affine index
        b.cx(x[i], x[i + 1])           # operands affine in i (checked eagerly)

``qubits`` returns a :class:`WireList`; indexing one with
an affine expression of enclosing loop indices verifies that the underlying
ids are affine in the index (raising at authoring time otherwise) and yields
an affine operand. Data-gated emission (``declare_table`` + ``gated``) skips
ops on zero table bits without per-iteration code. ``emit`` always writes the
segmented op-stream framing (a loop-free program is REGS + FLAT + END).
"""

from __future__ import annotations

import json
import operator
from array import array
from bisect import bisect_left
from collections import Counter
from collections.abc import MutableSequence
from contextlib import contextmanager

from . import compact as _compact
from .compact import Data, Flat, Loop, OpRec, Regs
from .ops import CC_ALLOWED_KINDS, NO_BIT, NO_OUT, NO_QUBIT, NO_REG, OC_ALLOWED_KINDS, OpKind
from .ops import gate_class as _ops_gate_class

# Self-inverse gates: the only op kinds `mirror()` re-emits verbatim (and only
# when unconditioned). Everything else in a mirrored region must be covered by
# a journaled semantic event (and/andr/alloc/free) or it raises — a whitelist,
# so a future kind can never be silently mis-mirrored. NEG is included: a
# global phase flip is its own inverse.
_MIRROR_WHITELIST = frozenset(
    {OpKind.X, OpKind.Z, OpKind.CX, OpKind.CZ, OpKind.CCX, OpKind.CCZ, OpKind.SWAP, OpKind.NEG}
)

_QUBIT_FIELDS = ("qa", "qb", "qt")

# The last Builder that emitted a stream in this process — the handle
# tools/liveness_profile.py uses to reach the instance of a circuit run as
# __main__, without instrumenting or patching anything (name-independent).
LAST_EMITTED = None

_TAPEFAST = False  # cached _tapefast module; False = untried, None = unavailable


def _tape_engine():
    """Return the C serializer module, or None when it cannot be imported.

    Without the module, emit falls back to the pure-Python serializer.
    """
    global _TAPEFAST
    if _TAPEFAST is False:
        try:
            from . import _tapefast

            _TAPEFAST = _tapefast
        except (ImportError, OSError):
            _TAPEFAST = None
    return _TAPEFAST


def _as_int(v, what):
    try:
        return operator.index(v)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {type(v).__name__}") from None


def _gate_class(kind, conditioned=False):
    """Return the cost-model stat bucket for ``kind`` (billed at its promoted kind
    when ``conditioned`` — carrying a bit condition): ``"toffoli"`` (in the TQ-cost),
    ``"clifford"`` (counted, not in the TQ-cost), or ``None`` (free / metadata). The one
    classification every phase-stat attribution path shares; the authority is
    ``uncompute.ops.gate_class``, which the evaluator's cost model also bills
    by.
    """
    return _ops_gate_class(kind, conditioned)


# The recording hot path's classification tables, built once at import BY
# CALLING the authority (ops.gate_class) so they can never drift from it.
# _CLS_PLAIN classifies every known kind unconditioned (unknown kinds miss
# to None, exactly as gate_class returns); _CLS_COND holds exactly the
# promotable kinds — a conditioned miss re-raises gate_class's canonical
# ValueError at the call site (replay() depends on that error).
_CLS_PLAIN = {int(k): _ops_gate_class(int(k), False) for k in OpKind}
_CLS_COND = {}
for _k in OpKind:
    try:
        _CLS_COND[int(_k)] = _ops_gate_class(int(_k), True)
    except ValueError:
        pass
del _k


# ---- symbolic affine indices ----------------------------------------------------


class AffineExpr:
    """``base + Σ coeff·i`` over enclosing ``Builder.repeat`` indices.

    Instances come from ``with b.repeat(n) as i`` (the raw index) and from
    integer arithmetic on such indices: ``i + c``, ``c + i``, ``i - c``,
    ``c - i``, ``-i``, ``i * c``, ``c * i``, and sums/differences of affine
    expressions. Products of two indices are rejected (not affine).
    """

    __slots__ = ("base", "coeffs")

    def __init__(self, base=0, coeffs=None):
        self.base = operator.index(base)
        self.coeffs = {}
        if coeffs:
            for frame, g in coeffs.items():
                g = operator.index(g)
                if g:
                    self.coeffs[frame] = g

    # -- arithmetic --------------------------------------------------------------
    def _combine(self, other, sign):
        if isinstance(other, AffineExpr):
            co = dict(self.coeffs)
            for f, g in other.coeffs.items():
                v = co.get(f, 0) + sign * g
                if v:
                    co[f] = v
                else:
                    co.pop(f, None)
            return AffineExpr(self.base + sign * other.base, co)
        try:
            o = operator.index(other)
        except TypeError:
            return NotImplemented
        return AffineExpr(self.base + sign * o, self.coeffs)

    def __add__(self, other):
        return self._combine(other, 1)

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine(other, -1)

    def __rsub__(self, other):
        return (-self)._combine(other, 1)

    def __neg__(self):
        return AffineExpr(-self.base, {f: -g for f, g in self.coeffs.items()})

    def __mul__(self, other):
        if isinstance(other, AffineExpr):
            raise TypeError(
                "a product of two loop indices is not affine; only integer scaling is supported"
            )
        try:
            o = operator.index(other)
        except TypeError:
            return NotImplemented
        return AffineExpr(self.base * o, {f: g * o for f, g in self.coeffs.items()})

    __rmul__ = __mul__

    def __index__(self):
        raise TypeError(
            f"symbolic index {self!r} cannot be used as a plain integer; index a "
            "WireList (the qubits result) or pass it to a gate "
            "directly"
        )

    def __repr__(self):
        terms = []
        for f in sorted(self.coeffs, key=lambda fr: fr.level):
            g = self.coeffs[f]
            if g == 1:
                terms.append(f.name)
            elif g == -1:
                terms.append(f"-{f.name}")
            else:
                terms.append(f"{g}*{f.name}")
        if self.base or not terms:
            terms.append(str(self.base))
        return " + ".join(terms).replace("+ -", "- ")


class _LoopFrame:
    """Bookkeeping for one open ``repeat`` level."""

    __slots__ = (
        "allocated",
        "count",
        "freed",
        "items",
        "journal",
        "level",
        "name",
        "open",
    )

    def __init__(self, count, level):
        self.count = count
        self.level = level
        self.name = f"i{level}"
        self.items = []
        self.allocated = set()
        self.freed = set()
        self.open = True
        self.journal = []  # this frame's semantic-event overlay (see Builder)


class WireList(list):
    """A list of wire ids that additionally supports symbolic affine indexing.

    Plain-``list`` behavior is untouched (ints, slices, iteration, ``+`` with
    a list, …). Indexing with an :class:`AffineExpr` checks — at authoring
    time, over the index's full iteration range — that the addressed ids are
    an affine function of the index, and returns the resulting affine operand;
    non-affine id patterns raise immediately, naming the ids that broke.
    """

    def __getitem__(self, key):
        if isinstance(key, AffineExpr):
            return self._affine(key)
        r = list.__getitem__(self, key)
        return WireList(r) if isinstance(key, slice) else r

    def __add__(self, other):
        if isinstance(other, list):
            return WireList(list.__add__(self, other))
        return NotImplemented

    def _affine(self, expr):
        n = len(self)
        for f in expr.coeffs:
            if not f.open:
                raise ValueError(f"loop index {f.name} used after its repeat() block closed")
        # Reachable index values k (with a witness iteration each). Loop counts
        # are literals, so this is exact; a count-0 level is probed at its t=0
        # slice only (the body never runs, but stays type-checked).
        reach = {expr.base: ()}
        for f, g in expr.coeffs.items():
            new = {}
            for k, wit in reach.items():
                for t in range(max(1, f.count)):
                    kk = k + g * t
                    if kk < 0 or kk >= n:
                        at = ", ".join(f"{nm}={tv}" for nm, tv in wit + ((f.name, t),))
                        raise IndexError(
                            f"symbolic index {expr!r} reaches {kk} at {at}; "
                            f"the wire list has {n} entries"
                        )
                    if kk not in new:
                        new[kk] = wit + ((f.name, t),)
            reach = new
        ks = sorted(reach)
        ids = []
        for k in ks:
            try:
                ids.append(operator.index(list.__getitem__(self, k)))
            except TypeError:
                raise TypeError(f"wire list entry at index {k} is not an integer id") from None
        if len(ks) == 1:
            a, s = ids[0], 0
        else:
            step = ks[1] - ks[0]
            s, rem = divmod(ids[1] - ids[0], step)
            bad = []
            if rem:
                bad = list(zip(ks[:2], ids[:2]))
            else:
                a = ids[0] - s * ks[0]
                bad = [(k, w) for k, w in zip(ks, ids) if w != a + s * k]
            if bad:
                shown = "; ".join(f"id {w} at index {k}" for k, w in bad[:8])
                more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
                raise ValueError(
                    f"wire ids are not affine in {expr!r}: {shown}{more} break the "
                    f"stride fitted from indices {ks[0]},{ks[1]} "
                    f"(ids {ids[0]},{ids[1]})"
                )
        coeffs = {f: s * g for f, g in expr.coeffs.items()}
        return AffineExpr(a + s * expr.base, coeffs)


class Outcome:
    """An opaque measurement-outcome handle — a per-shot measurement result.

    Returned by :meth:`Builder.hmr` (which demolishes a qubit and writes the
    fresh fair bit into this handle's outcome id). Outcomes may:

    * condition exactly ``z`` / ``cz`` / ``neg`` via ``cond=`` (the sign
      applies only on shots where the outcome is 1);
    * accumulate in place: ``m1 ^= m2`` emits one OXOR (destructive on
      ``m1``, whose stored bit becomes ``m1 ^ m2``);
    * combine freshly at the top level: ``m1 ^ m2`` allocates a new outcome
      id and OXORs both in. Inside a ``repeat`` body this raises — a fresh
      id is zero only once, so per-iteration re-use would accumulate stale
      bits; use ``^=`` onto an outcome HMR rewrites each iteration instead;
    * be retired: :meth:`Builder.retire` returns the id to a reuse pool the
      next :meth:`Builder.hmr` draws from (HMR fully overwrites, so a
      recycled id needs no zero-init). A handle whose id has since been
      reused is *stale*: any further use of it raises.

    Outcome ids live in their own address space: an Outcome is never a qubit
    and never usable as one (and vice versa).
    """

    __slots__ = ("_b", "gen", "id")

    def __init__(self, builder, oid, gen=0):
        self._b = builder
        self.id = oid
        self.gen = gen

    def __ixor__(self, other):
        if not isinstance(other, Outcome):
            raise TypeError(f"Outcome ^= expects an Outcome, got {type(other).__name__}")
        if other._b is not self._b:
            raise ValueError("cannot mix Outcomes from different Builders")
        if other.id == self.id:
            raise ValueError("m ^= m self-assignment: OXOR operands must be distinct")
        self._b._check_out(self, "m ^= ... target")
        self._b._check_out(other, "^= source")
        self._b._emit(OpKind.OXOR, ot=self.id, oc=other.id)
        return self

    def __xor__(self, other):
        if not isinstance(other, Outcome):
            return NotImplemented
        if other._b is not self._b:
            raise ValueError("cannot mix Outcomes from different Builders")
        b = self._b
        if b._loop_stack:
            raise RuntimeError(
                "m1 ^ m2 inside a repeat() body: the fresh accumulator id is "
                "zero-initialized only once, not per iteration — accumulate "
                "in place (m1 ^= m2) onto an outcome written this iteration"
            )
        b._check_out(self, "^ operand")
        b._check_out(other, "^ operand")
        acc = Outcome(b, b._alloc_out())
        b._emit(OpKind.OXOR, ot=acc.id, oc=self.id)
        b._emit(OpKind.OXOR, ot=acc.id, oc=other.id)
        return acc

    def __bool__(self):
        raise TypeError(
            "an Outcome has no build-time truth value (it is a per-shot "
            "measurement result); pass it as cond= to z/cz/neg"
        )

    def __repr__(self):
        return f"Outcome(o{self.id})"


class Bit:
    """An opaque classical-bit handle — per-shot classical input data.

    Returned by :meth:`Builder.bit` / :meth:`Builder.bits`. Bit
    values are pure functions of the test inputs: `Bit`-kind registers
    (:meth:`Builder.bit_register`) receive per-shot values from the
    evaluator, and the free bit-only ops (:meth:`Builder.bit_invert` /
    ``bit_store0`` / ``bit_store1``, or ``b1 ^= b2``) derive from them.
    Nothing quantum and no measurement outcome ever writes a bit.

    A Bit may condition ``x``/``z``/``neg``/``cx``/``cz`` via ``when=``: the
    gate acts only on shots where the bit is 1 and is BILLED at its promoted
    kind (``when=`` on ``cx``/``cz`` costs a full Toffoli; on ``ccx``/``ccz``
    it is illegal). Bit ids live in their own address space — a Bit is never
    a qubit or an Outcome.
    """

    __slots__ = ("_b", "id")

    def __init__(self, builder, cid):
        self._b = builder
        self.id = cid

    def __ixor__(self, other):
        if not isinstance(other, Bit):
            raise TypeError(f"Bit ^= expects a Bit, got {type(other).__name__}")
        if other._b is not self._b:
            raise ValueError("cannot mix Bits from different Builders")
        if other.id == self.id:
            raise ValueError("c ^= c self-assignment: bit operands must be distinct")
        self._b._emit(OpKind.BIT_INVERT, ct=self.id, cc=other.id)
        return self

    def __add__(self, other):
        # Affine composition inside repeat(): Bit + loop-index expression
        # yields the bit-id affine form (c0 + i). Plain-int offsets are NOT
        # accepted — a raw bit id must never circulate as an int.
        if type(other) is AffineExpr:
            return self.id + other
        return NotImplemented

    __radd__ = __add__

    def __bool__(self):
        raise TypeError(
            "a Bit has no build-time truth value (it is per-shot classical "
            "input data); pass it as when= to x/z/neg/cx/cz or condition "
            "bit ops on it"
        )

    def __repr__(self):
        return f"Bit(c{self.id})"


class Mark:
    """An opaque recording-position token from :meth:`Builder.checkpoint`.

    Pass it to :meth:`Builder.mirror` to emit the Bennett inverse of
    everything recorded since. A Mark is bound to its builder and the
    ``repeat`` nesting level it was taken at.
    """

    __slots__ = (
        "_b",
        "_depth",
        "_journal",
        "_jpos",
        "_phase",
        "_pos",
        "_target",
    )

    def __repr__(self):
        return f"Mark(op {self._pos}, depth {self._depth})"


# ---- the op tape (columnar depth-0 record) -----------------------------------

# All four field sentinels (NO_QUBIT/NO_OUT/NO_REG/NO_BIT) share one value.
_TAPE_NO = NO_QUBIT
# Slot-0 marker: the item at this index lives in the side table (_objs).
_TAPE_OBJ = 0xFFFFFFFF
# Slot-0 bit: the cc field (tuple slot 8) holds the sentinel — clear means
# the op carries a bit condition (bills at its promoted kind).
_TAPE_CC = 0x8000
_MARKER_ROW = (_TAPE_OBJ, 0, 0, 0, 0, 0, 0, 0, 0)
_MARKER_ARR = array("I", _MARKER_ROW)


class _OpTape(MutableSequence):
    """The Builder's depth-0 record: list-like on the surface, columnar
    underneath.

    Flat 9-int op tuples — the bulk of any large program — pack into one
    contiguous ``array('I')``, 9 u32 slots per op: slot 0 carries the kind
    byte plus one bit per field marking the shared ``2**64 - 1`` sentinel,
    slots 1-8 the field values (the id range is already contract-bound to
    ``[0, 2**32)``). That is ~36 bytes per op against ~125 for a tuple of
    ints, which is what makes multi-million-op flat records affordable.
    Items a row cannot hold exactly — ``OpRec``/``Loop`` nodes, and tuples
    with out-of-range or non-int fields, whose canonical errors must keep
    surfacing at emit time — live in a side table under their index, behind
    a marker row.

    The record's documented contract is value-level, and this class keeps
    the full list surface: indexing, slices, in-place assignment, ``del``,
    ``insert``, ``append``/``extend``, iteration, equality and ``+`` with
    lists. Reads rebuild value-equal tuples rather than returning the
    originally appended objects (fields normalize to plain ints), and slice
    reads return plain lists.
    """

    __slots__ = ("_objs", "_rows")

    def __init__(self):
        self._rows = array("I")
        self._objs = {}  # index -> item a row cannot hold (OpRec/Loop/odd tuple)

    # -- packing -----------------------------------------------------------------
    @staticmethod
    def _pack(item):
        """``item`` as a 9-slot row tuple, or None for the side table. Range
        validation is left to the array store (C-speed), whose failure also
        routes to the side table.
        """
        if type(item) is not tuple or len(item) != 9:
            return None
        k, a, b, c, d, e, f, g, h = item
        if not (isinstance(k, int) and 0 <= k <= 0xFF):
            return None
        m = 0
        if a == _TAPE_NO:
            m = 1
            a = 0
        if b == _TAPE_NO:
            m |= 2
            b = 0
        if c == _TAPE_NO:
            m |= 4
            c = 0
        if d == _TAPE_NO:
            m |= 8
            d = 0
        if e == _TAPE_NO:
            m |= 16
            e = 0
        if f == _TAPE_NO:
            m |= 32
            f = 0
        if g == _TAPE_NO:
            m |= 64
            g = 0
        if h == _TAPE_NO:
            m |= 128
            h = 0
        return (k | m << 8, a, b, c, d, e, f, g, h)

    # -- core sequence protocol ---------------------------------------------------
    def __len__(self):
        return len(self._rows) // 9

    def _get(self, i):
        r = self._rows
        b = i * 9
        s0 = r[b]
        if s0 < 0x100:  # kind byte only: no sentinel fields
            return (
                s0,
                r[b + 1],
                r[b + 2],
                r[b + 3],
                r[b + 4],
                r[b + 5],
                r[b + 6],
                r[b + 7],
                r[b + 8],
            )
        if s0 == _TAPE_OBJ:
            return self._objs[i]
        m = s0 >> 8
        N = _TAPE_NO
        return (
            s0 & 0xFF,
            N if m & 1 else r[b + 1],
            N if m & 2 else r[b + 2],
            N if m & 4 else r[b + 3],
            N if m & 8 else r[b + 4],
            N if m & 16 else r[b + 5],
            N if m & 32 else r[b + 6],
            N if m & 64 else r[b + 7],
            N if m & 128 else r[b + 8],
        )

    def __getitem__(self, i):
        if type(i) is slice:
            n = len(self._rows) // 9
            return [self._get(j) for j in range(*i.indices(n))]
        i = operator.index(i)
        n = len(self._rows) // 9
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("op record index out of range")
        return self._get(i)

    def __iter__(self):
        r = self._rows
        objs = self._objs
        N = _TAPE_NO
        b = 0
        while b < len(r):  # length re-checked: append-during-iteration behaves
            s0 = r[b]
            if s0 < 0x100:
                yield (
                    s0,
                    r[b + 1],
                    r[b + 2],
                    r[b + 3],
                    r[b + 4],
                    r[b + 5],
                    r[b + 6],
                    r[b + 7],
                    r[b + 8],
                )
            elif s0 == _TAPE_OBJ:
                yield objs[b // 9]
            else:
                m = s0 >> 8
                yield (
                    s0 & 0xFF,
                    N if m & 1 else r[b + 1],
                    N if m & 2 else r[b + 2],
                    N if m & 4 else r[b + 3],
                    N if m & 8 else r[b + 4],
                    N if m & 16 else r[b + 5],
                    N if m & 32 else r[b + 6],
                    N if m & 64 else r[b + 7],
                    N if m & 128 else r[b + 8],
                )
            b += 9

    def append(self, item):
        # _pack, inlined: this is the recording hot path (one call per
        # emitted op), so the method call and the intermediate row tuple
        # are worth eliding.
        if type(item) is tuple and len(item) == 9:
            k, a, b, c, d, e, f, g, h = item
            if isinstance(k, int) and 0 <= k <= 0xFF:
                m = 0
                if a == _TAPE_NO:
                    m = 1
                    a = 0
                if b == _TAPE_NO:
                    m |= 2
                    b = 0
                if c == _TAPE_NO:
                    m |= 4
                    c = 0
                if d == _TAPE_NO:
                    m |= 8
                    d = 0
                if e == _TAPE_NO:
                    m |= 16
                    e = 0
                if f == _TAPE_NO:
                    m |= 32
                    f = 0
                if g == _TAPE_NO:
                    m |= 64
                    g = 0
                if h == _TAPE_NO:
                    m |= 128
                    h = 0
                rows = self._rows
                n = len(rows)
                try:
                    rows.extend((k | m << 8, a, b, c, d, e, f, g, h))
                except (OverflowError, TypeError, ValueError):
                    del rows[n:]  # partial extend: roll the row back
                else:
                    return
        self._objs[len(self._rows) // 9] = item
        self._rows.extend(_MARKER_ROW)

    def _append_row(self, k, a, b, c, d, e, f, g, h):
        """``append((k, a, ..., h))`` minus the tuple: the recording hot
        path is ``_emit``'s tail, which already holds the nine fields
        unpacked — the 9-tuple is built only on the failure path (a field
        the row cannot hold), where it lands in the side table exactly as
        ``append`` would store it (original field values reconstructed
        from the sentinel mask).
        """
        if isinstance(k, int) and 0 <= k <= 0xFF:
            m = 0
            if a == _TAPE_NO:
                m = 1
                a = 0
            if b == _TAPE_NO:
                m |= 2
                b = 0
            if c == _TAPE_NO:
                m |= 4
                c = 0
            if d == _TAPE_NO:
                m |= 8
                d = 0
            if e == _TAPE_NO:
                m |= 16
                e = 0
            if f == _TAPE_NO:
                m |= 32
                f = 0
            if g == _TAPE_NO:
                m |= 64
                g = 0
            if h == _TAPE_NO:
                m |= 128
                h = 0
            rows = self._rows
            n = len(rows)
            try:
                rows.extend((k | m << 8, a, b, c, d, e, f, g, h))
            except (OverflowError, TypeError, ValueError):
                del rows[n:]  # partial extend: roll the row back
            else:
                return
            N = _TAPE_NO
            item = (
                k,
                N if m & 1 else a,
                N if m & 2 else b,
                N if m & 4 else c,
                N if m & 8 else d,
                N if m & 16 else e,
                N if m & 32 else f,
                N if m & 64 else g,
                N if m & 128 else h,
            )
        else:
            item = (k, a, b, c, d, e, f, g, h)
        self._objs[len(self._rows) // 9] = item
        self._rows.extend(_MARKER_ROW)

    def extend(self, values):
        if values is self:
            values = list(values)
        for v in values:
            self.append(v)

    def _forget(self, i):
        """Drop index ``i``'s side-table entry (the row itself is the
        caller's to overwrite or delete).
        """
        if self._rows[i * 9] == _TAPE_OBJ:
            del self._objs[i]

    def _put(self, i, item):
        """Overwrite row ``i`` with ``item`` (the old item is already
        forgotten).
        """
        row = self._pack(item)
        b = i * 9
        if row is not None:
            try:
                self._rows[b : b + 9] = array("I", row)
            except (OverflowError, TypeError, ValueError):
                pass
            else:
                return
        self._rows[b : b + 9] = _MARKER_ARR
        self._objs[i] = item

    def _shift_objs(self, from_idx, delta):
        if self._objs and delta:
            self._objs = {(k + delta if k >= from_idx else k): v for k, v in self._objs.items()}

    def __setitem__(self, i, value):
        if type(i) is slice:
            self._set_slice(i, value)
            return
        i = operator.index(i)
        n = len(self._rows) // 9
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("op record assignment index out of range")
        self._forget(i)
        self._put(i, value)

    def _set_slice(self, s, value):
        n = len(self._rows) // 9
        start, stop, step = s.indices(n)
        items = list(value)
        if step != 1:
            idxs = range(start, stop, step)
            if len(items) != len(idxs):
                raise ValueError(
                    f"attempt to assign sequence of size {len(items)} to "
                    f"extended slice of size {len(idxs)}"
                )
            for j, it in zip(idxs, items):
                self._forget(j)
                self._put(j, it)
            return
        stop = max(start, stop)
        for j in range(start, stop):
            self._forget(j)
        self._shift_objs(stop, len(items) - (stop - start))
        block = array("I")
        pend = []  # (offset within the splice, item) for the side table
        for off, it in enumerate(items):
            row = self._pack(it)
            if row is not None:
                m = len(block)
                try:
                    block.extend(row)
                except (OverflowError, TypeError, ValueError):
                    del block[m:]
                    row = None
            if row is None:
                block.extend(_MARKER_ROW)
                pend.append((off, it))
        self._rows[start * 9 : stop * 9] = block
        for off, it in pend:
            self._objs[start + off] = it

    def __delitem__(self, i):
        if type(i) is slice:
            n = len(self._rows) // 9
            idxs = range(*i.indices(n))
            if len(idxs) == 0:
                return
            if idxs.step == 1:
                start, stop = idxs[0], idxs[-1] + 1
                for j in range(start, stop):
                    self._forget(j)
                del self._rows[start * 9 : stop * 9]
                self._shift_objs(stop, start - stop)
            else:
                for j in sorted(idxs, reverse=True):
                    del self[j]
            return
        i = operator.index(i)
        n = len(self._rows) // 9
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("op record index out of range")
        self._forget(i)
        del self._rows[i * 9 : (i + 1) * 9]
        self._shift_objs(i + 1, -1)

    def insert(self, i, item):
        i = operator.index(i)
        n = len(self._rows) // 9
        if i < 0:
            i = max(0, i + n)
        elif i > n:
            i = n
        self._shift_objs(i, 1)
        b = i * 9
        row = self._pack(item)
        if row is not None:
            try:
                self._rows[b:b] = array("I", row)
            except (OverflowError, TypeError, ValueError):
                row = None
        if row is None:
            self._rows[b:b] = _MARKER_ARR
            self._objs[i] = item

    def clear(self):
        del self._rows[:]
        self._objs.clear()

    # -- list interface ----------------------------------------------------------
    def copy(self):
        return list(self)

    def __add__(self, other):
        if isinstance(other, (list, _OpTape)):
            return list(self) + list(other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return other + list(self)
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, (list, _OpTape)):
            return NotImplemented
        if len(other) != len(self):
            return False
        return all(a == b for a, b in zip(self, other))
        # (defining __eq__ leaves the class unhashable, matching list)

    def __repr__(self):
        return f"<_OpTape of {len(self)} items>"

    # -- raw views (Builder-internal fast paths) ----------------------------------
    def _scan_slot0(self):
        """Yield ``(index, slot 0)`` per item — kind byte + sentinel mask for
        packed rows, the marker for side-table items — without building
        tuples.
        """
        r = self._rows
        b = i = 0
        while b < len(r):
            yield i, r[b]
            i += 1
            b += 9

    def _iter_run(self, start, stop):
        """Yield the items of ``[start, stop)`` — the FLAT-run expansion."""
        r = self._rows
        objs = self._objs
        N = _TAPE_NO
        for i in range(start, stop):
            b = i * 9
            s0 = r[b]
            if s0 == _TAPE_OBJ:
                yield objs[i]
                continue
            if s0 < 0x100:
                yield (
                    s0,
                    r[b + 1],
                    r[b + 2],
                    r[b + 3],
                    r[b + 4],
                    r[b + 5],
                    r[b + 6],
                    r[b + 7],
                    r[b + 8],
                )
            else:
                m = s0 >> 8
                yield (
                    s0 & 0xFF,
                    N if m & 1 else r[b + 1],
                    N if m & 2 else r[b + 2],
                    N if m & 4 else r[b + 3],
                    N if m & 8 else r[b + 4],
                    N if m & 16 else r[b + 5],
                    N if m & 32 else r[b + 6],
                    N if m & 64 else r[b + 7],
                    N if m & 128 else r[b + 8],
                )


class _TapeRun:
    """One zero-copy FLAT run over ``tape[start:stop]``: the sized iterable
    :class:`~uncompute.compact.Flat` serializes without materializing the run
    as a list.
    """

    __slots__ = ("_n", "_okeys", "_start", "_stop", "_tape")

    def __init__(self, tape, start, stop, n, okeys=None):
        self._tape = tape
        self._start = start
        self._stop = stop
        self._n = n
        self._okeys = okeys  # sorted OpRec indices (emit-time snapshot)

    def __len__(self):
        return self._n

    def __iter__(self):
        return self._tape._iter_run(self._start, self._stop)

    def _native_encode(self, table_ids, fh):
        """Serialize this run natively (uncompute._tapefast) when the C
        kernel is available and UNCOMPUTE_KERNEL allows it: byte-identical
        stream, reference errors at reference positions, Python keeping
        every ``fh.write``. The duck-typed seam ``compact._write_flat``
        dispatches on. False = not taken, nothing written — the caller
        runs the pure encoder.
        """
        eng = _tape_engine()
        if eng is None or not eng.enabled():
            return False
        return eng.write_run(self, table_ids, fh)

    @property
    def plain_op_count(self):
        """``len()`` when every item expands to exactly one op — raw tuples
        are always ungated — letting ``expanded_count`` skip the walk. None
        when the run holds an ``OpRec`` (possibly data-gated).

        With ``okeys`` (the sorted OpRec indices, computed once per emit in
        ``_build_segments``) the range test is two bisect probes; the full
        ``_objs`` scan is the fallback for runs built without a snapshot —
        per-run scans of a program-sized dict are quadratic across a large
        emit.
        """
        ok = self._okeys
        if ok is not None:
            i = bisect_left(ok, self._start)
            return None if i < len(ok) and ok[i] < self._stop else self._n
        objs = self._tape._objs
        if objs:
            for k, v in objs.items():
                if self._start <= k < self._stop and isinstance(v, OpRec):
                    return None
        return self._n


class Builder:
    """Record a reversible-circuit program, then serialize it with :meth:`emit`.

    A Builder records gates, allocations, and register declarations in
    program order. :meth:`emit` writes the recorded program as an op
    stream.

    Allocation model: :meth:`recycle` returns a qubit id to a LIFO free-list.
    The next allocation reuses the most recently freed id. Qubit-id reuse
    therefore governs ``peak_qubits``. Free ancilla qubits promptly to keep
    the peak low.

    Public attributes:

    * ``peak_qubits`` — the maximum number of simultaneously live qubits so far.
    * ``peak_phase`` — the phase label active when ``peak_qubits`` was reached.
    * ``active_qubits`` — the number of currently live qubits.
    * ``current_phase`` — the phase label attributed to subsequent gates.

    Conditioning: only ``z``, ``cz``, and ``neg`` accept ``cond=`` and it
    must be an :class:`Outcome` (from :meth:`hmr`) — the sign applies only
    on shots where the outcome is 1. A :class:`Bit` conditions ``x``,
    ``z``, ``neg``, ``cx``, and ``cz`` through ``when=``. A condition
    narrows the action of a gate, never its billing, so every emitted gate
    counts on every shot.

    Example::

        b = Builder()
        a, c = b.qubit(), b.qubit()
        t = b.and_compute(a, c)    # t = a AND c; 1 executed Toffoli
        b.cz(t, a)                 # use the AND result
        b.and_release(t, a, c)     # uncompute and free t; 0 executed Toffoli
        b.emit()
    """

    def __init__(self, recycle_outcomes=False):
        # Depth-0 recording. For programs using no loop-gated feature this is
        # a list-like record of flat 9-int op tuples (kind, qa, qb, qt, ot,
        # oc, rt, ct, cc) — columnar underneath (_OpTape), value-identical to
        # a plain list on every documented surface — and `self.ops`
        # exposes it directly. Programs using repeat()/gated()/declare_table()
        # interleave compact.OpRec / Loop nodes into it.
        self._items = _OpTape()
        self._loop_stack = []  # open _LoopFrame s, outermost first
        self._stats_stack = []  # per open loop: {phase: {"toffoli", "clifford"}}
        self._registers = []  # hoisted REGS payload: per register, (qubit_id, ...)
        self._tables = []  # declared compact.Data tables, index == table_id
        self._gate_ctx = None  # (Data, AffineExpr) inside a gated() block
        self._uses_loops = False  # any repeat()/gated()/declare_table() call
        # ---- semantic-event journal (the mirror()'s overlay on _items) --------
        # _items is already a complete gate tape; the journal records only the
        # events whose pairing is LOST at lowering (and_compute -> a bare CCX,
        # and_release -> HMR+CZ+R, alloc -> no op at all), plus set_phase marks
        # for reflected mirror labels. Entries carry the position in the ACTIVE
        # recording list they refer to; each open repeat frame keeps its own
        # journal (discarded at close — the balance rule guarantees a closed
        # loop body never needs one). Pure metadata: journaling changes no
        # emitted byte.
        self._journal = []  # depth-0 events
        self._journal_suspend = False  # True inside and_compute/and_release
        self.next_qubit = 0
        self.next_out = 0  # outcome-id mint frontier (HMR targets / fresh accumulators)
        self.next_bit = 0  # classical bit ids; never freed or reused
        # ---- outcome-id reuse pool (the evaluator sizes outcome storage by
        # max id, so retiring dead ids bounds its footprint by peak-LIVE
        # outcomes). Only hmr() draws from the pool — HMR fully overwrites
        # live lanes, so a recycled id needs no zero-init; fresh XOR
        # accumulators always mint (they rely on zero-init-once). _out_gen
        # maps a recycled id to its current generation: handles from an
        # earlier tenancy are stale, and every handle use checks.
        self.recycle_outcomes = bool(recycle_outcomes)
        self.free_outs = []  # LIFO pool of retired outcome ids
        self._free_out_set = set()  # mirror for O(1) double-retire detection
        self._out_gen = {}  # id -> current generation (absent = 0, never recycled)
        self.next_register = 0
        self.free_qubits = []  # LIFO free-list of qubit ids
        self._free_set = set()  # mirror of free_qubits for O(1) double-free detection
        self.active_qubits = 0
        self.peak_qubits = 0
        # ---- phase attribution (metadata-only; never emitted to ops.bin) ------
        # set_phase / peak-attribution bookkeeping: a per-phase
        # breakdown of emitted Toffoli/Clifford + the phase's local qubit peak, plus
        # which phase was active when the global peak was hit. The breakdown
        # points at the phase responsible for the cost, not just a scalar.
        self.current_phase = "init"
        self._phase_order = []  # phase names, first-seen order
        self._phase_stats = {}  # name -> {"toffoli", "clifford", "peak"}
        self.peak_phase = "init"  # phase active when global peak_qubits was reached
        # The latest set_phase: (name, peak before it, position, wire-event
        # count). A span with no op and no allocation or free leaves its
        # phase peak as it was (see _idle_peak).
        self._phase_entry = None
        self._wire_events = 0
        # ---- wire log (metadata-only; never emitted to ops.bin) --------------
        # The native chronological record of the wire axis: every entry is
        # (kind, x, token, live) with kind in {"mark", "alloc", "free",
        # "claim", "release", "mirror-begin", "mirror-end"} — phase marks,
        # allocation events, ownership events ("x" = (name, wires)), and
        # mirror brackets, position-tokened and live-counted at record
        # time. One list, so chronology is by construction. Profiling and
        # ownership-validation tools are pure consumers of a finished
        # Builder; recording is pure bookkeeping (no ops, byte-identical
        # streams with or without consumers).
        self.wire_log: list[tuple] = [("mark", "init", (0,), 0)]
        # The current phase's stats dict, cached for the depth-0 counting
        # hot path; maintained here and at set_phase only (current_phase
        # has no other writer).
        self._cur_stats = self._phase("init")

    def _phase(self, name):
        s = self._phase_stats.get(name)
        if s is None:
            s = {"toffoli": 0, "clifford": 0, "peak": 0}
            self._phase_stats[name] = s
            self._phase_order.append(name)
        return s

    def _count_gate(self, kind, n=1, conditioned=False):
        """Attribute ``n`` emitted gates to the current phase (metadata ops are
        ignored; bit-conditioned gates land in their PROMOTED class). Inside a
        ``repeat`` body the count lands on the loop's stats level and is
        multiplied by the loop count at exit, so phase totals are always
        *expanded* emitted counts.

        One call per recorded op: classification is the import-time
        ``_CLS_*`` tables (built from ops.gate_class, the authority — a
        conditioned miss re-raises its canonical error), and at depth 0
        the count lands on ``_cur_stats``, the current phase's stats dict
        cached at ``set_phase`` — value-identical to the ``_merge_stats``
        spelling.
        """
        if conditioned:
            try:
                key = _CLS_COND[kind]
            except KeyError:
                key = _gate_class(kind, conditioned)  # canonical ValueError
        else:
            key = _CLS_PLAIN.get(kind)
        if key is None:
            return
        if self._stats_stack:
            self._merge_stats(
                self.current_phase,
                n if key == "toffoli" else 0,
                n if key == "clifford" else 0,
            )
        else:
            self._cur_stats[key] += n

    def _count_gate_root(self, kind, n, conditioned=False):
        """Attribute ``n`` expanded gates directly to the root phase stats —
        used for data-gated ops, whose exact expanded count (the gate-table
        popcount over the full enclosing loop box) is known at emission.
        """
        key = _gate_class(kind, conditioned)
        if key is not None:
            self._phase(self.current_phase)[key] += n

    def _touch_phase_peak(self, name):
        """Raise phase `name`'s local qubit peak to the current live count if higher."""
        ps = self._phase(name)
        ps["peak"] = max(ps["peak"], self.active_qubits)

    # ---- recorded-program views -----------------------------------------------
    @property
    def ops(self):
        """The flat record: the recorded ops as the live, mutable, list-like
        record of 9-int tuples — flat programs only. In-place edits are legal
        under the emit-time recount contract (record what you emit; the
        sanctioned append is :meth:`replay`). Reads rebuild value-equal
        tuples (the record is columnar underneath); slice reads return plain
        lists. Programs using ``repeat``/``gated``/``declare_table`` have no
        flat record — iterate :meth:`iter_expanded`.
        """
        if self._uses_loops:
            raise RuntimeError(
                "Builder.ops is the flat record; this program uses repeat()/"
                "gated()/declare_table() — iterate b.iter_expanded()"
            )
        return self._items

    def replay(self, op):
        """Append one already-validated flat 9-int op tuple to the recording
        and count it — the sanctioned spelling of tape replay (re-emitting
        ops captured from a flat record). Replayed ops enter the same
        accounting as recorded ones, so the emit-time recount contract
        holds. Flat programs only.
        """
        if self._uses_loops or self._loop_stack or self._gate_ctx is not None:
            raise RuntimeError(
                "replay() appends to the flat record; this program uses "
                "repeat()/gated()/declare_table()"
            )
        op = tuple(op)
        self._count_gate(op[0], conditioned=op[8] != NO_BIT)
        self._items.append(op)

    def iter_expanded(self):
        """Yield the full expansion as flat-style 9-int op tuples.

        This is the flat *spelling* of the program: symbolic loops unrolled
        (affine operands resolved per iteration), data-gated ops included only
        where their table bit is 1. Feed it to :class:`~uncompute.Simulator` (its ``run()``
        does so automatically) or iterate it for debugging without ever
        materializing the whole list. Its length is the op-stream END count.
        """
        if self._loop_stack:
            raise RuntimeError("cannot expand while a repeat() block is open")
        tables = {t.table_id: t.bits for t in self._tables}
        for item in self._items:
            if isinstance(item, tuple):
                yield item
            else:
                yield from _compact.iter_expand_item(item, (), tables)

    def expanded_size(self):
        """Return the exact expanded post-gating op count — the ``len()`` of
        :meth:`iter_expanded` as pure arithmetic over the recorded loops
        and tables, without iterating.
        """
        if self._loop_stack:
            raise RuntimeError("cannot size the expansion while a repeat() block is open")
        tables = {t.table_id: t.bits for t in self._tables}
        n = 0
        for item in self._items:
            n += 1 if isinstance(item, tuple) else _compact.item_count(item, (), tables)
        return n

    def _target(self):
        """Return the list new items are recorded into (innermost open loop body,
        else the depth-0 stream).
        """
        return self._loop_stack[-1].items if self._loop_stack else self._items

    def _jlog(self, ev):
        """Append a semantic event to the active journal (innermost open loop
        frame's, else depth-0). Suppressed inside and_compute/and_release,
        whose composite events subsume their internal alloc/free.
        """
        if not self._journal_suspend:
            j = self._loop_stack[-1].journal if self._loop_stack else self._journal
            j.append(ev)

    # ---- low-level emit (mirrors the evaluator's aliasing check) -------------
    def _emit(
        self,
        kind,
        qa=NO_QUBIT,
        qb=NO_QUBIT,
        qt=NO_QUBIT,
        ot=NO_OUT,
        oc=NO_OUT,
        rt=NO_REG,
        ct=NO_BIT,
        cc=NO_BIT,
    ):
        if (
            self._loop_stack
            or self._gate_ctx is not None
            or type(qa) is AffineExpr
            or type(qb) is AffineExpr
            or type(qt) is AffineExpr
            or type(ot) is AffineExpr
            or type(oc) is AffineExpr
            or type(ct) is AffineExpr
            or type(cc) is AffineExpr
        ):
            self._emit_compact(int(kind), qa, qb, qt, ot, oc, rt, ct, cc)
            return
        if (qt == qb != NO_QUBIT) or (qt == qa != NO_QUBIT) or (qb == qa != NO_QUBIT):
            present = [q for q in (qt, qb, qa) if q != NO_QUBIT]
            raise ValueError(f"{OpKind(kind).name}: qubit operands must be distinct, got {present}")
        if ot == oc != NO_OUT:
            raise ValueError(f"{OpKind(kind).name}: outcome operands must be distinct, got o{ot}")
        if ct == cc != NO_BIT:
            raise ValueError(f"{OpKind(kind).name}: bit operands must be distinct, got c{ct}")
        k = int(kind)
        if (
            oc != NO_OUT
            and k not in OC_ALLOWED_KINDS
            and "oc" not in (_compact.KIND_OPERANDS.get(k) or ())
        ):
            # Eager twin of the serializer's check (compact.py) and of
            # _emit_compact's in-loop check — the private _emit surface fails
            # identically everywhere, at authoring time. (OXOR is exempt: its
            # oc slot is a named source operand, not an outcome condition.)
            _compact.raise_oc_condition(k)
        if cc != NO_BIT and k not in CC_ALLOWED_KINDS:
            _compact.raise_cc_condition(k)
        self._count_gate(k, conditioned=cc != NO_BIT)
        self._items._append_row(k, qa, qb, qt, ot, oc, rt, ct, cc)

    def _emit_compact(self, k, qa, qb, qt, ot, oc, rt, ct=NO_BIT, cc=NO_BIT):
        """Record one op as a compact OpRec (inside a repeat body, in a gated
        block, or with affine operands).
        """
        if rt != NO_REG:
            raise ValueError(f"{OpKind(k).name}: rt is not a compact-record operand")
        names = _compact.KIND_OPERANDS.get(k)
        if names is None:
            raise ValueError(f"unknown op kind {k}")
        vals = {"qa": qa, "qb": qb, "qt": qt, "ot": ot, "oc": oc, "ct": ct}
        operands = tuple(self._operand(vals[n], f"{OpKind(k).name} {n}") for n in names)
        # Aliasing, eagerly: identical affine forms collide on *every* iteration.
        # (Partial overlaps — equal at some iterations only — surface via
        # iter_expanded()/Simulator and at evaluation.)
        qops = [(n, o) for n, o in zip(names, operands) if n in _QUBIT_FIELDS]
        for i in range(len(qops)):
            for j in range(i + 1, len(qops)):
                if qops[i][1] == qops[j][1]:
                    raise ValueError(
                        f"{OpKind(k).name}: qubit operands {qops[i][0]} and "
                        f"{qops[j][0]} are identical ({qops[i][1]}) on every "
                        "iteration"
                    )
        if "ot" in names and "oc" in names:  # OXOR: outcome operands distinct
            i_ot, i_oc = names.index("ot"), names.index("oc")
            if operands[i_ot] == operands[i_oc]:
                raise ValueError(
                    f"{OpKind(k).name}: outcome operands ot and oc are "
                    f"identical ({operands[i_ot]}) on every iteration"
                )
        ocrec = None
        if k in OC_ALLOWED_KINDS:
            if type(oc) is AffineExpr or oc != NO_OUT:
                ocrec = self._operand(oc, f"{OpKind(k).name} oc")
        elif "oc" not in names and (type(oc) is AffineExpr or oc != NO_OUT):
            # For OXOR, oc is a named source operand (consumed above); for
            # every other non-phase kind a supplied oc is an illegal
            # outcome condition. Same rule and wording as the serializer's flat
            # path — the private _emit surface fails identically at depth 0
            # and inside repeat()/gated(), instead of silently dropping it.
            _compact.raise_oc_condition(k)
        ccrec = None
        if k in CC_ALLOWED_KINDS:
            if type(cc) is AffineExpr or cc != NO_BIT:
                ccrec = self._operand(cc, f"{OpKind(k).name} cc")
        elif type(cc) is AffineExpr or cc != NO_BIT:
            _compact.raise_cc_condition(k)
        # Bit-op target vs condition: identical affine forms collide on
        # every iteration (the eager twin of the ct==cc aliasing rule).
        if ccrec is not None and "ct" in names and operands[names.index("ct")] == ccrec:
            raise ValueError(
                f"{OpKind(k).name}: bit operands ct and cc are "
                f"identical ({ccrec}) on every iteration"
            )
        gate = None
        if self._gate_ctx is not None:
            table, expr = self._gate_ctx
            gate = self._gate_block(table, expr)
        rec = OpRec(k, operands, ocrec, gate, ccrec)
        conditioned = ccrec is not None
        if gate is None:
            self._count_gate(k, conditioned=conditioned)
        elif _gate_class(k, conditioned) is not None:
            counts = tuple(f.count for f in self._loop_stack)
            table = self._tables[gate[0]]
            self._count_gate_root(
                k,
                _compact.gated_popcount(table.bits, gate[1], gate[2], counts),
                conditioned=conditioned,
            )
        self._target().append(rec)

    def _resolve_affine(self, base, coeffs, hi_limit, frame_msg, range_msg):
        """Shared affine-expression resolution at the current loop depth:
        frame-membership check over ``coeffs``, strides derived from the loop
        stack, and the eager range check over the full enclosing loop box (box
        corners are attained; a count-0 level skips the check — the body never
        runs). ``frame_msg(frame)`` / ``range_msg(lo, hi)`` supply the caller's
        error wording. Returns ``(base, strides)``.
        """
        for f in coeffs:
            if f not in self._loop_stack:
                raise ValueError(frame_msg(f))
        strides = tuple(coeffs.get(f, 0) for f in self._loop_stack)
        counts = [f.count for f in self._loop_stack]
        if all(c > 0 for c in counts):
            lo, hi = _compact.box_bounds(base, strides, counts)
            if lo < 0 or hi >= hi_limit:
                raise ValueError(range_msg(lo, hi))
        return (base, strides)

    def _operand(self, v, what):
        """Convert an int or AffineExpr wire operand to compact ``(base,
        strides)`` at the current loop depth, eagerly checking the id range
        over the full enclosing loop box (box corners are attained).
        """
        if isinstance(v, AffineExpr):
            if not self._loop_stack:
                raise ValueError(f"{what}: affine operand used outside any repeat() block")
            base, coeffs = v.base, v.coeffs
        else:
            base, coeffs = _as_int(v, what), {}

        def frame_msg(f):
            where = (
                "after its repeat() block closed" if not f.open else "outside its repeat() block"
            )
            return f"{what}: loop index {f.name} used {where}"

        def range_msg(lo, hi):
            return (
                f"{what}: resolved id spans [{lo}, {hi}] over the loop box; "
                "ids must stay in [0, 2^32)"
            )

        return self._resolve_affine(base, coeffs, 1 << 32, frame_msg, range_msg)

    def _gate_block(self, table, expr):
        """Return the (table_id, offset, strides) gate block for the current gated()
        context at the current loop depth, with the bit-index range checked
        eagerly over the full enclosing box.
        """

        def frame_msg(f):
            return (
                f"gated(): loop index {f.name} is not an enclosing repeat() at this emission point"
            )

        def range_msg(lo, hi):
            return (
                f"gated(): bit index spans [{lo}, {hi}] over the loop box, "
                f"but table {table.table_id} has {len(table.bits)} bits"
            )

        offset, strides = self._resolve_affine(
            expr.base, expr.coeffs, len(table.bits), frame_msg, range_msg
        )
        return (table.table_id, offset, strides)

    def _oc(self, cond):
        """Resolve a ``cond=`` argument to an outcome-id operand. Only
        :class:`Outcome` handles from THIS builder are accepted — raw ints are
        rejected so the qubit and outcome address spaces can never be silently
        crossed, and foreign handles are rejected so two builders' outcome id
        spaces cannot alias (the same rule ``^``/``^=`` enforce).
        """
        if cond is None:
            return NO_OUT
        if isinstance(cond, Outcome):
            if cond._b is not self:
                raise ValueError("cannot mix Outcomes from different Builders")
            self._check_out(cond, "cond=")
            return cond.id
        raise TypeError(f"cond= must be an Outcome (from Builder.hmr), got {type(cond).__name__}")

    def _cc(self, when):
        """Resolve a ``when=`` argument to a bit-id operand (or an affine
        form inside ``repeat``). Only :class:`Bit` handles from THIS builder
        (or an :class:`AffineExpr` over bit ids) are accepted — the same
        address-space discipline as :meth:`_oc`.
        """
        if when is None:
            return NO_BIT
        if isinstance(when, Bit):
            if when._b is not self:
                raise ValueError("cannot mix Bits from different Builders")
            return when.id
        if type(when) is AffineExpr:
            return when
        raise TypeError(f"when= must be a Bit (from Builder.bit), got {type(when).__name__}")

    def _ct(self, t, what):
        """Resolve a bit-op target to a bit-id operand (or an affine form)."""
        if isinstance(t, Bit):
            if t._b is not self:
                raise ValueError("cannot mix Bits from different Builders")
            return t.id
        if type(t) is AffineExpr:
            return t
        raise TypeError(f"{what} must be a Bit (from Builder.bit), got {type(t).__name__}")

    # ---- allocation (LIFO free-list, matches the evaluator's qubit census) ----
    def _record_alloc(self, q):
        """Record an allocation — the one accounting path: active/peak counters, phase
        peaks, loop-frame membership, journal and wire_log events. Every
        way a wire becomes live (qubit, reclaim, contiguous_qubits's
        mint) funnels through here.
        """
        self.active_qubits += 1
        self._wire_events += 1
        if self.active_qubits > self.peak_qubits:
            self.peak_qubits = self.active_qubits
            self.peak_phase = self.current_phase
        self._touch_phase_peak(self.current_phase)
        if self._loop_stack:
            self._loop_stack[-1].allocated.add(q)
        self._jlog(("alloc", len(self._target()), q, self._gate_ctx is not None))
        self.wire_log.append(("alloc", q, self._pos_token(), self.active_qubits))

    def qubit(self):
        """Allocate one qubit id, reusing the most recently freed id first.

        Updates ``active_qubits`` and, if a new high, ``peak_qubits``.
        """
        if self.free_qubits:
            q = self.free_qubits.pop()
            self._free_set.discard(q)
        else:
            q = self._bump_qubit()
        self._record_alloc(q)
        return q

    def _bump_qubit(self):
        q = self.next_qubit
        self.next_qubit += 1
        return q

    def qubits(self, n):
        """Allocate ``n`` qubit ids. Returns them as a :class:`WireList`."""
        return WireList(self.qubit() for _ in range(n))

    def reclaim(self, ids):
        """Reclaim exactly ``ids`` from the free-list, in order.

        The named-id allocation: recorded regions (loop bodies, tapes,
        mirrored choreography) address fixed wire ids, so re-materializing
        their scratch requires those ids back regardless of LIFO order.
        Every id must currently be free; the rest of the free-list keeps its
        relative order. Emits no ops — accounting and events are exactly
        those of :meth:`qubit` per id. Returns a :class:`WireList` in
        the requested order.
        """
        want = [_as_int(q, "reclaim() id") for q in ids]
        ws = set(want)
        if len(ws) != len(want):
            raise ValueError("reclaim(): duplicate ids requested")
        missing = ws - self._free_set
        if missing:
            raise ValueError(f"reclaim(): ids not on the free-list: {sorted(missing)}")
        self.free_qubits[:] = [q for q in self.free_qubits if q not in ws]
        self._free_set.difference_update(ws)
        for q in want:
            self._record_alloc(q)
        return WireList(want)

    def _alloc_out(self):
        """Mint one brand-new outcome id (guaranteed zero-initialized — the
        fresh-XOR-accumulator requirement). Never draws from the reuse pool;
        :meth:`hmr` goes through :meth:`_pop_out` for that.
        """
        o = self.next_out
        self.next_out += 1
        return o

    def _pop_out(self):
        """Pick an outcome id for an HMR target: the most recently retired id if
        any (HMR fully overwrites live lanes, so recycled ids are safe
        there), else a fresh mint. Returns ``(id, generation)``.
        """
        if self.free_outs:
            o = self.free_outs.pop()
            self._free_out_set.discard(o)
            g = self._out_gen.get(o, 0) + 1
            self._out_gen[o] = g
            return o, g
        return self._alloc_out(), 0

    def _check_out(self, m, what):
        """Reject a stale :class:`Outcome` handle — one whose id was retired
        and has since been handed to a newer HMR. Until the id is actually
        reused, old handles keep reading the same (dead but intact) value,
        so they stay legal; after reuse they alias the new tenant.
        """
        if m.gen != self._out_gen.get(m.id, 0):
            raise ValueError(
                f"stale Outcome handle o{m.id} ({what}): the id was retired "
                "and has been reused by a later hmr()"
            )

    def retire(self, m):
        """Return ``m``'s outcome id to the reuse pool for a later
        :meth:`hmr`.

        Emits no op. Retire an outcome only once its value can never be
        read again — every conditioned fixup and OXOR that consumes it is
        already recorded. The record's order is the execution order, so any
        use recorded *before* the reusing HMR still reads the old value
        (correct); a retained handle used *after* the id is reused raises
        as stale. Retiring an outcome whose measurement fixup is still
        unresolved surfaces in the strict :class:`~uncompute.Simulator` as a
        cleanliness error at the overwriting HMR.

        Recycling keeps the outcome-id census — and with it the
        evaluator's outcome-storage footprint, which scales with the max
        id — bounded by peak simultaneously-live outcomes.
        """
        if not isinstance(m, Outcome):
            raise TypeError(f"retire() takes an Outcome (from Builder.hmr), got {type(m).__name__}")
        if m._b is not self:
            raise ValueError("cannot retire an Outcome from a different Builder")
        self._check_out(m, "retire()")
        if m.id in self._free_out_set:
            raise ValueError(f"double retire of outcome o{m.id}")
        self.free_outs.append(m.id)
        self._free_out_set.add(m.id)

    def bit(self):
        """Allocate one classical bit id; returns a :class:`Bit` handle.

        Bits are free real estate: they never enter ``active_qubits`` /
        ``peak_qubits`` (the TQ-cost counts qubit space only) and
        their ids are never freed or reused.
        """
        c = Bit(self, self.next_bit)
        self.next_bit += 1
        return c

    def bits(self, n):
        """Allocate ``n`` classical bit ids; returns a list of :class:`Bit`."""
        return [self.bit() for _ in range(n)]

    def recycle(self, q, reset_recorded=False):
        """Reset ``q`` and return its id to the free-list for LIFO reuse.

        ``reset_recorded=True`` emits no R: the caller asserts the stream
        ALREADY contains this wire's reset (e.g. an R recorded inside a loop
        body whose replay covers this wire) — the free is bookkeeping only,
        keeping the record equal to the stream. Such a free cannot be
        mirrored (its R lives elsewhere; ``mirror()`` fails closed on it).

        Raises on double-free and on never-allocated ids: either would put
        an id on the free-list that no live wire backs, hand it to two
        callers, and silently corrupt the live/peak accounting that
        certifies ancilla cleanliness.
        """
        if isinstance(q, AffineExpr):
            raise TypeError(
                "recycle() takes a concrete qubit id; allocations inside a repeat "
                "body reuse the same ids every iteration, so recycle what qubit() "
                "returned"
            )
        if q in self._free_set:
            raise ValueError(f"double recycle of qubit {q}")
        if not 0 <= q < self.next_qubit:
            raise ValueError(f"recycle of never-allocated qubit {q}")
        if reset_recorded:
            self._jlog(("freeq", len(self._target()), q, self._gate_ctx is not None))
        else:
            self.r(q)
            # pos = the index of the R this free just emitted
            self._jlog(("free", len(self._target()) - 1, q, self._gate_ctx is not None))
        self.free_qubits.append(q)
        self._free_set.add(q)
        self.active_qubits -= 1
        self._wire_events += 1
        if self._loop_stack:
            self._loop_stack[-1].freed.add(q)
        self.wire_log.append(("free", q, self._pos_token(), self.active_qubits))

    def recycle_all(self, qs, reset_recorded=False):
        """Recycle every qubit id in ``qs``, in order. See :meth:`recycle`.

        The default path calls ``self.recycle(q)`` per id, so subclasses
        that override ``recycle(self, q)`` keep working.
        """
        if reset_recorded:
            for q in qs:
                self.recycle(q, reset_recorded=True)
        else:
            for q in qs:
                self.recycle(q)

    # ---- free-list shaping (pure bookkeeping; emits no ops) -------------------
    def contiguous_qubits(self, n, mint=True):
        """Allocate ``n`` qubits as one contiguous ascending run.

        Scans the free ids' maximal contiguous runs and carves the first
        ``n`` ids of the best-fit (smallest adequate) run via
        :meth:`reclaim` — preserving the longer runs for later
        carves. When no free run is long enough, ``mint=True`` (the
        default) mints the block as brand-new consecutive ids past the
        frontier; ``mint=False`` raises instead. The rest of the pool
        keeps its order. Emits no ops — accounting and events per id are
        exactly :meth:`qubit`'s. Returns a :class:`WireList`.
        """
        n = _as_int(n, "contiguous_qubits() count")
        if n < 0:
            raise ValueError(f"contiguous_qubits() count must be >= 0, got {n}")
        if n == 0:
            return WireList()
        runs, cur = [], []
        for q in sorted(self._free_set):
            if cur and q == cur[-1] + 1:
                cur.append(q)
            else:
                if cur:
                    runs.append(cur)
                cur = [q]
        if cur:
            runs.append(cur)
        best = min((r for r in runs if len(r) >= n), key=len, default=None)
        if best is None:
            if not mint:
                raise ValueError(
                    f"contiguous_qubits(): no contiguous free run of {n} ids (mint=False)"
                )
            out = []
            for _ in range(n):
                q = self._bump_qubit()
                self._record_alloc(q)
                out.append(q)
            return WireList(out)
        return self.reclaim(best[:n])

    # ---- register declaration ------------------------------------------------
    def _register_guard(self):
        if self._loop_stack:
            raise RuntimeError(
                "registers must be declared outside repeat() blocks "
                "(the op stream hoists them into the REGS segment)"
            )
        if self._gate_ctx is not None:
            raise RuntimeError("register declarations cannot be data-gated")

    def qubit_register(self, qubits):
        """Declare ``qubits`` as one Qubit-kind register; returns the register
        index.

        Register qubits may hold outputs at the end; all other qubits must
        end in |0>. Declare registers outside :meth:`repeat` blocks.
        """
        self._register_guard()
        ids = [_as_int(w, "register wire id") for w in qubits]
        r = self.next_register
        self.next_register += 1
        self._registers.append(tuple(ids))
        return r

    def bit_register(self, bits):
        """Declare ``bits`` (:class:`Bit` handles) as one Bit-kind register;
        returns the register index.

        Bit registers are classical INPUTS: the evaluator injects each shot's
        drawn value into the bits before the first gate. They carry no
        expected outputs and are never read back; wire ``i`` receives bit
        ``i`` of the drawn value.
        """
        self._register_guard()
        ids = []
        for w in bits:
            if not isinstance(w, Bit):
                raise TypeError(
                    f"bit_register() takes Bit handles (from Builder.bit), got {type(w).__name__}"
                )
            if w._b is not self:
                raise ValueError("cannot mix Bits from different Builders")
            ids.append(w.id)
        r = self.next_register
        self.next_register += 1
        self._registers.append((_compact.WKIND_BIT, tuple(ids)))
        return r

    # ---- gates ---------------------------------------------------------------
    def x(self, q, when=None):
        """Flip qubit ``q``. Free: not counted.

        ``when`` (a :class:`Bit`) restricts the flip to shots where the bit
        is 1 — billed as a CX (Clifford).
        """
        self._emit(OpKind.X, qt=q, cc=self._cc(when))

    def z(self, q, cond=None, when=None):
        """Phase-flip qubit ``q`` where it is 1. Free: not counted.

        ``cond`` (an :class:`Outcome`) restricts the sign flip to shots where
        the outcome is 1; ``when`` (a :class:`Bit`) to shots where the bit is
        1 — billed as a CZ (Clifford). Both compose.
        """
        self._emit(OpKind.Z, qt=q, oc=self._oc(cond), cc=self._cc(when))

    def r(self, q):
        """Reset qubit ``q`` to |0>."""
        self._emit(OpKind.R, qt=q)

    def cx(self, ctrl, tgt, when=None):
        """Flip qubit ``tgt`` where qubit ``ctrl`` is 1. Clifford: counted, not in the TQ-cost.

        ``when`` (a :class:`Bit`) restricts the flip to shots where the bit
        is 1 — billed as a full Toffoli (the promoted CCX).
        """
        self._emit(OpKind.CX, qb=ctrl, qt=tgt, cc=self._cc(when))

    def cz(self, a, b, cond=None, when=None):
        """Phase-flip where qubits ``a`` and ``b`` are both 1.

        Clifford: counted, not in the TQ-cost.

        ``cond`` (an :class:`Outcome`) restricts the sign flip to shots where
        the outcome is 1 — the measurement-based-uncomputation fixup.
        ``when`` (a :class:`Bit`) restricts it to shots where the bit is 1 —
        billed as a full Toffoli (the promoted CCZ).
        """
        self._emit(OpKind.CZ, qb=a, qt=b, oc=self._oc(cond), cc=self._cc(when))

    def swap(self, a, b):
        """Exchange the states of qubits ``a`` and ``b``. Clifford: counted, not in the TQ-cost."""
        self._emit(OpKind.SWAP, qb=a, qt=b)

    def cswap(self, c, a, b):
        """Fredkin: exchange qubits ``a`` and ``b`` where qubit ``c`` is 1, as
        the standard 1-Toffoli expansion with ``a`` as the CX pivot —
        CX(b,a); CCX(c,a,b); CX(b,a).

        Costs 1 executed Toffoli + 2 Cliffords. The swap is symmetric;
        argument order only fixes which wire carries the conjugating CXes
        (the byte pattern). Affine operands compose per constituent gate.
        """
        if not (
            isinstance(c, AffineExpr) or isinstance(a, AffineExpr) or isinstance(b, AffineExpr)
        ):
            ids = (_as_int(c, "cswap c"), _as_int(a, "cswap a"), _as_int(b, "cswap b"))
            if len(set(ids)) != 3:
                raise ValueError(f"cswap: qubit operands must be distinct, got {list(ids)}")
        self.cx(b, a)
        self.ccx(c, a, b)
        self.cx(b, a)

    def ccx(self, c1, c2, tgt):
        """Flip qubit ``tgt`` where qubits ``c1`` and ``c2`` are both 1.

        Costs 1 executed Toffoli. Operand mapping: qa=c1, qb=c2, qt=tgt.
        """
        self._emit(OpKind.CCX, qa=c1, qb=c2, qt=tgt)

    def ccz(self, c1, c2, tgt):
        """Phase-flip where qubits ``c1``, ``c2``, and ``tgt`` are all 1.

        Costs 1 executed Toffoli. Operand mapping: qa=c1, qb=c2, qt=tgt.
        """
        self._emit(OpKind.CCZ, qa=c1, qb=c2, qt=tgt)

    def hmr(self, q):
        r"""X-basis measurement of ``q``, demolishing it to |0>.

        Returns an :class:`Outcome` holding the fresh fair bit — pass it as
        ``cond=`` to :meth:`z`/:meth:`cz`/:meth:`neg` for the phase fixup, or
        accumulate outcomes with ``^=``. Inside a ``repeat`` body one call
        records one op: every iteration overwrites the same outcome id, so
        measure-then-fixup within the iteration is exact.

        The target id is the most recently :meth:`retire`\ d one when the
        reuse pool is non-empty (HMR overwrites, so recycled ids are safe
        here), else a fresh mint.
        """
        oid, gen = self._pop_out()
        m = Outcome(self, oid, gen)
        self._emit(OpKind.HMR, qt=q, ot=m.id)
        return m

    # ── Gidney temporary-AND pair ───────────────────────────────────────────
    # Canonical form of the idiom for the common case. (Frames the pair
    # cannot express — a CX-conjugated fixup, an ancilla outliving the
    # release across episodes — are legitimately hand-rolled.)

    def and_compute(self, c1, c2, pol=(True, True)):
        """Gidney temporary AND: allocate t, CCX(c1, c2 -> t). 1 executed Toffoli.

        Returns t. Release with and_release(t, c1, c2) — c1/c2 must still hold
        their compute-time values at release (the conditioned CZ corrects the
        measurement phase against those values; releasing after mutating a
        control silently leaves phase garbage on half the shots).

        ``pol`` gives the control polarities: a False entry ANDs the negated
        control, as sugar for the hand-rolled X sandwich — an X on that
        control before and after the AND, recorded as ordinary gates, so the
        stream equals the hand spelling and :meth:`mirror` needs no special
        casing. Release with the same ``pol``.
        """
        p1, p2 = pol
        if not p1:
            self.x(c1)
        if not p2:
            self.x(c2)
        prev = self._journal_suspend
        self._journal_suspend = True
        try:
            t = self.qubit()
            self.ccx(c1, c2, t)
        finally:
            self._journal_suspend = prev
        # composite event at the CCX's index; subsumes t's alloc
        self._jlog(("and", len(self._target()) - 1, c1, c2, t, self._gate_ctx is not None))
        if not p1:
            self.x(c1)
        if not p2:
            self.x(c2)
        return t

    def and_release(self, t, c1, c2, pol=(True, True)):
        """Measurement-based release of an and_compute ancilla: hmr + CZ(c1,c2)
        conditioned on the measured outcome. 0 executed Toffoli, frees t.

        ``pol`` gives the control polarities exactly as in
        :meth:`and_compute` (the same X sandwich, recorded as ordinary
        gates); pass the compute-time polarities.

        An ancilla that must outlive its release as bookkeeping (say it
        stays in an ownership set through the demolition) spells the same
        bytes explicitly: ``m = hmr(t); cz(c1, c2, cond=m); r(t)``, then
        the :meth:`release` and a ``recycle(t, reset_recorded=True)``.

        Under ``Builder(recycle_outcomes=True)`` the outcome is retired
        here — the conditioned CZ is its one consumer, so the id returns to
        the reuse pool and the returned handle is spent (using it after a
        later hmr reuses the id raises as stale).
        """
        p1, p2 = pol
        if not p1:
            self.x(c1)
        if not p2:
            self.x(c2)
        prev = self._journal_suspend
        self._journal_suspend = True
        try:
            m = self.hmr(t)
            self.cz(c1, c2, cond=m)
            self.recycle(t)
        finally:
            self._journal_suspend = prev
        # composite event at the HMR's index; covers the HMR/CZ/R triple and
        # subsumes t's free
        self._jlog(("andr", len(self._target()) - 3, c1, c2, t, self._gate_ctx is not None))
        if not p1:
            self.x(c1)
        if not p2:
            self.x(c2)
        if self.recycle_outcomes:
            self.retire(m)
        return m

    def neg(self, cond=None, when=None):
        """Global phase flip. ``cond`` (an :class:`Outcome`) restricts it to
        shots where the outcome is 1; ``when`` (a :class:`Bit`) to shots
        where the bit is 1 (billed as a Z — free). Both compose.
        """
        self._emit(OpKind.NEG, oc=self._oc(cond), cc=self._cc(when))

    # ---- bit ops (free classical networks over per-shot input bits) -----------
    def bit_invert(self, t, when=None):
        """Bit flip: ``t ^= 1`` per shot, on shots where ``when`` is 1 (all
        shots when unconditioned). Free — bit-only ops are never counted.
        ``c1 ^= c2`` is sugar for ``bit_invert(c1, when=c2)``.
        """
        self._emit(OpKind.BIT_INVERT, ct=self._ct(t, "bit_invert() target"), cc=self._cc(when))

    def bit_store0(self, t, when=None):
        """Bit clear: ``t = 0`` per shot, on shots where ``when`` is 1. Free."""
        self._emit(OpKind.BIT_STORE0, ct=self._ct(t, "bit_store0() target"), cc=self._cc(when))

    def bit_store1(self, t, when=None):
        """Bit set: ``t = 1`` per shot, on shots where ``when`` is 1. Free."""
        self._emit(OpKind.BIT_STORE1, ct=self._ct(t, "bit_store1() target"), cc=self._cc(when))

    # ---- control-flow helpers ------------------------------------------------
    @contextmanager
    def repeat(self, count):
        """Record the block once, execute it ``count`` times (an op-stream LOOP).

        Yields the symbolic loop index ``i`` (an :class:`AffineExpr`, values
        0..count-1). Inside the body, gate operands may be plain ints (same
        wire every iteration) or affine expressions of enclosing indices —
        typically ``wirelist[i * s + c]``. Nesting composes, up to depth 4.

        Balanced-allocation rule: every qubit allocated inside the
        body must be freed inside it (and the body may not free outer qubits)
        — the LIFO free-list then hands every iteration the same ids, keeping
        peak accounting exact. Enforced here at block exit.

        ``count`` must be a literal non-negative int; a count of 0 records
        nothing. An empty body records nothing. Register declarations are not
        allowed inside the body.
        """
        count = _as_int(count, "repeat() count")
        if count < 0 or count >= (1 << 64):
            raise ValueError(f"repeat() count must be in [0, 2^64), got {count}")
        if len(self._loop_stack) >= _compact.MAX_LOOP_DEPTH:
            raise ValueError(
                f"repeat() nesting deeper than {_compact.MAX_LOOP_DEPTH} (the op-stream cap)"
            )
        self._uses_loops = True
        frame = _LoopFrame(count, len(self._loop_stack))
        snap = None
        if count == 0:
            snap = (
                self.peak_qubits,
                self.peak_phase,
                {n: s["peak"] for n, s in self._phase_stats.items()},
            )
        self._loop_stack.append(frame)
        self._stats_stack.append({})
        ok = False
        try:
            yield AffineExpr(0, {frame: 1})
            ok = True
        finally:
            # A loop close records no op of the current phase, so a phase
            # span idle before the close stays idle after it.
            idle = self._idle_peak() is not None
            frame.open = False
            self._loop_stack.pop()
            stats = self._stats_stack.pop()
            if ok:
                self._close_frame(frame, stats, snap)
            if idle:
                e = self._phase_entry
                assert e is not None
                self._phase_entry = (e[0], e[1], self._pos_token(), self._wire_events)

    def _close_frame(self, frame, stats, snap):
        leaked = sorted(frame.allocated - frame.freed)
        if leaked:
            raise ValueError(
                f"unbalanced repeat() body: qubit(s) {leaked} allocated inside "
                "the loop were not freed before loop exit — every iteration "
                "must free what it allocates so ids repeat exactly"
            )
        foreign = sorted(frame.freed - frame.allocated)
        if foreign:
            raise ValueError(
                f"unbalanced repeat() body: qubit(s) {foreign} freed inside the "
                "loop were allocated outside it"
            )
        if frame.count == 0:
            # The body never executes: contributes no ops, no stats, no peaks.
            self.peak_qubits, self.peak_phase = snap[0], snap[1]
            for name, s in self._phase_stats.items():
                s["peak"] = snap[2].get(name, 0)
            return
        for name, d in stats.items():
            self._merge_stats(name, d["toffoli"] * frame.count, d["clifford"] * frame.count)
        if not frame.items:
            return  # empty body: the loop is the identity, record nothing
        if len(frame.items) > _compact.MAX_LOOP_ITEMS:
            raise ValueError(
                f"repeat() body has {len(frame.items)} items; the op stream caps a "
                f"loop body at {_compact.MAX_LOOP_ITEMS} (u16 n_items) — "
                "restructure into nested or consecutive loops"
            )
        self._target().append(Loop(frame.count, frame.items))

    def _merge_stats(self, name, toffoli, clifford):
        if self._stats_stack:
            level = self._stats_stack[-1]
            d = level.get(name)
            if d is None:
                d = level[name] = {"toffoli": 0, "clifford": 0}
            d["toffoli"] += toffoli
            d["clifford"] += clifford
        else:
            s = self._phase(name)
            s["toffoli"] += toffoli
            s["clifford"] += clifford

    # ---- data-gated emission (op-stream DATA tables) --------------------------
    def declare_table(self, bits):
        """Declare a constant bit table (an op-stream DATA segment); returns the
        table handle for :meth:`gated`. ``bits`` is an iterable of 0/1.
        """
        vals = list(bits)
        if not vals:
            raise ValueError("declare_table: a table needs at least one bit")
        table = Data(len(self._tables), vals)  # Data validates the 0/1 bits
        self._uses_loops = True
        self._tables.append(table)
        return table

    @contextmanager
    def gated(self, table, index):
        """Data-gate every op emitted in the block on ``table`` bit ``index``.

        ``index`` is an int or an affine expression of enclosing ``repeat``
        indices. On iterations where the bit is 1 the ops exist as usual; where
        it is 0 they do not exist in the expansion at all — not validated, not
        counted, and (for hmr/r) not consuming a measurement-counter value.
        This is the value-dependent specialization channel: skip work on zero
        bits of pinned constants without per-iteration code.

        Blocks do not nest (an op record carries exactly one gate); register
        declarations are not allowed inside. The bit index is range-checked
        against the table eagerly, over the full enclosing loop box.
        """
        if not isinstance(table, Data) or not (
            table.table_id < len(self._tables) and self._tables[table.table_id] is table
        ):
            raise ValueError("gated(): table must be a handle from this builder's declare_table()")
        if self._gate_ctx is not None:
            raise RuntimeError("gated() blocks do not nest: an op record carries exactly one gate")
        expr = (
            index if isinstance(index, AffineExpr) else AffineExpr(_as_int(index, "gated() index"))
        )
        self._uses_loops = True
        self._gate_ctx = (table, expr)
        try:
            yield
        finally:
            self._gate_ctx = None

    # ---- Bennett mirror (checkpoint / mirror) --------------------------------
    def checkpoint(self):
        """Snapshot the current recording position for a later :meth:`mirror`.

        Legal at any ``repeat`` nesting level; the matching ``mirror(mark)``
        must be issued at the same level (with the same frame still open, for
        in-loop marks). Returns an opaque :class:`Mark`.
        """
        m = Mark()
        m._b = self
        m._target = self._target()
        m._pos = len(m._target)
        m._journal = self._loop_stack[-1].journal if self._loop_stack else self._journal
        m._jpos = len(m._journal)
        m._depth = len(self._loop_stack)
        m._phase = self.current_phase
        return m

    def mirror(self, since, until=None, phase_prefix="un-"):
        """Emit the exact Bennett inverse of a region — see :meth:`_mirror`
        for the full contract. Brackets the emission in the wire log so
        consumers can group the mirrored spans.
        """
        self.wire_log.append(("mirror-begin", None, self._pos_token(), self.active_qubits))
        try:
            return self._mirror(since, until, phase_prefix)
        finally:
            self.wire_log.append(("mirror-end", None, self._pos_token(), self.active_qubits))

    def _mirror(self, since, until=None, phase_prefix="un-"):
        """Emit the exact Bennett inverse of the region between two marks.

        The region is everything recorded from ``since`` to ``until``
        (default: now). The inverse is emitted in reverse order. Each
        construct maps to its cheapest exact inverse:

        * ``and_compute`` <-> ``and_release``. The mirrored release is
          measurement-based: 0 Toffoli. The mirrored compute costs 1
          Toffoli and must land on the same qubit id — the LIFO free-list
          guarantees this for a well-formed region, and violations raise.
        * ``qubit`` <-> ``recycle``, with the same id-stability
          guarantee.
        * X/Z/CX/CZ/CCX/CCZ/SWAP/NEG (unconditioned) re-emit verbatim.
        * Closed ``repeat`` loops of such gates reverse with the index
          remap i -> count-1-i.

        Anything else raises and names the op: a bare ``hmr``/``r``/OXOR,
        a conditioned ``z``/``cz``/``neg``, a
        data-gated alloc/free/AND event, a ``reset_recorded`` free, or a
        loop containing any of those. Nothing is skipped silently.

        ``until`` bounds the region for the conjugation shape F·A·F⁻¹:
        checkpoint before F, checkpoint after F, emit an un-mirrored stage
        A, then ``mirror(f_mark, until=a_mark)`` inverts F alone. The
        stage in between must be alloc/free-balanced.

        ``phase_prefix``: when the region contains ``set_phase`` marks,
        the mirrored spans get the reflected labels — forward phases A, B,
        C mirror as ``un-C``, ``un-B``, ``un-A`` — and the phase active at
        the call is restored afterwards. Pass ``None`` to disable
        relabeling.

        The mirror journals itself, so an outer ``mirror()`` over a region
        that contains inner mirrors replays correctly.
        """
        for name, m in (("since", since), ("until", until)):
            if m is None:
                continue
            if not isinstance(m, Mark):
                raise TypeError(
                    f"mirror() {name}= takes a Mark from checkpoint(), got {type(m).__name__}"
                )
            if m._b is not self:
                raise ValueError(f"mirror(): {name}= Mark from a different Builder")
            if len(self._loop_stack) != m._depth or self._target() is not m._target:
                raise RuntimeError(
                    "mirror() must be called at the same repeat() nesting level "
                    "as its checkpoint (close any repeat() opened since; an "
                    "in-loop mark is unusable after its frame closes)"
                )
        if self._gate_ctx is not None:
            raise RuntimeError(
                "mirror() inside a gated() block: mirrored ops must not "
                "silently acquire a data gate"
            )

        target = since._target
        journal = since._journal
        if until is not None:
            if until._pos < since._pos:
                raise ValueError("mirror(): until= Mark precedes since=")
            end, jend = until._pos, until._jpos
        else:
            end, jend = len(target), len(journal)
        events = journal[since._jpos : jend]

        trigger, allocs, marks = self._mirror_index_events(events)
        segs = self._mirror_phase_segments(since, marks, phase_prefix)
        self._mirror_replay(target, since, end, trigger, allocs, segs, phase_prefix)

    @staticmethod
    def _mirror_index_events(events):
        """Event index: item-anchored triggers + alloc flushes + phase marks."""
        trigger = {}  # item index -> ("and"|"andr"|"free") event
        allocs = {}  # item index -> [alloc events at that position]
        marks = []  # (pos, name) set_phase marks, in order
        for ev in events:
            k = ev[0]
            if k == "phase":
                marks.append((ev[1], ev[2]))
                continue
            if ev[-1]:  # gated flag
                raise ValueError(
                    f"mirror(): cannot mirror a data-gated {k} event "
                    "(its ops exist only on some iterations)"
                )
            if k == "alloc":
                allocs.setdefault(ev[1], []).append(ev)
            elif k == "and" or k == "free":
                trigger[ev[1]] = ev
            elif k == "freeq":
                raise ValueError(
                    "mirror(): cannot mirror a reset_recorded free (its R "
                    "was recorded elsewhere in the stream)"
                )
            elif k == "andr":
                trigger[ev[1] + 2] = ev
        return trigger, allocs, marks

    @staticmethod
    def _mirror_phase_segments(since, marks, phase_prefix):
        """Reflected phase segmentation: segment 0 = [since._pos, first mark)
        under the mark's entry phase, then one segment per set_phase mark.
        """
        segs = []
        if phase_prefix is not None and marks:
            segs = [(since._pos, since._phase)]
            for pos, name in marks:
                if segs and segs[-1][0] == pos:
                    segs[-1] = (pos, name)  # mark at the same pos overrides
                else:
                    segs.append((pos, name))
        return segs

    def _mirror_replay(self, target, since, end, trigger, allocs, segs, phase_prefix):
        """Reverse Bennett-inverse replay: walk items back to front, flushing
        alloc mirrors and phase-label switches at each position, dispatching
        journaled and/andr/free triggers or falling back to _mirror_item.
        """
        entry_phase = self.current_phase
        cur_seg = len(segs) - 1

        def flush_allocs(pos):
            # allocs at `pos` sat between items pos-1 and pos: their mirror
            # (free) is emitted after items >= pos are mirrored. Multiple
            # allocs at one pos mirror in reverse journal order (LIFO).
            for ev in reversed(allocs.get(pos, ())):
                self.recycle(ev[2])

        i = end - 1
        while i >= since._pos:
            flush_allocs(i + 1)
            if segs:
                while cur_seg > 0 and i < segs[cur_seg][0]:
                    cur_seg -= 1
                label = phase_prefix + segs[cur_seg][1]
                if self.current_phase != label:
                    self.set_phase(label)
            ev = trigger.get(i)
            if ev is None:
                self._mirror_item(target, i)
                i -= 1
            elif ev[0] == "and":
                _, p, c1, c2, t, _g = ev
                self._verify_journal(target, p, OpKind.CCX, t)
                self.and_release(t, c1, c2)
                i -= 1
            elif ev[0] == "free":
                self._verify_journal(target, ev[1], OpKind.R, ev[2])
                q = self.qubit()
                if q != ev[2]:
                    raise RuntimeError(
                        f"mirror(): LIFO free-list diverged from the journal "
                        f"(alloc returned qubit {q}, expected {ev[2]}) — the "
                        "region's alloc/free structure is not mirrorable"
                    )
                i -= 1
            elif ev[0] == "andr":
                _, p, c1, c2, t, _g = ev
                self._verify_journal(target, p, OpKind.HMR, t)
                self._verify_journal(target, p + 2, OpKind.R, t)
                t2 = self.and_compute(c1, c2)
                if t2 != t:
                    raise RuntimeError(
                        f"mirror(): LIFO free-list diverged from the journal "
                        f"(and_compute returned qubit {t2}, expected {t})"
                    )
                i = p - 1  # the whole HMR/CZ/R triple is consumed
        flush_allocs(since._pos)
        if segs and self.current_phase != entry_phase:
            self.set_phase(entry_phase)

    def _mirror_item(self, target, i):
        """Re-emit ``target[i]`` (an item with no covering journal event) into
        the recording, or raise if it is not whitelisted-self-inverse.
        """
        item = target[i]
        if isinstance(item, Loop):
            rev = self._mirror_loop(item, len(self._loop_stack))
            if rev.items:
                self._account_item(rev, ())
                target.append(rev)
            return
        if isinstance(item, tuple):
            kind = item[0]
            conditioned = item[5] != NO_OUT or item[8] != NO_BIT
        else:  # OpRec (affine and/or data-gated record)
            kind = item.kind
            conditioned = item.oc is not None or item.cc is not None
        if kind in _MIRROR_WHITELIST and not conditioned:
            if isinstance(item, tuple):
                self._count_gate(kind)
            else:
                self._account_item(item, ())
            target.append(item)
            return
        name = OpKind(kind).name if kind in OpKind._value2member_map_ else kind
        cond = " carrying a condition" if conditioned else ""
        raise ValueError(
            f"mirror(): op #{i} {name}{cond} is not Bennett-invertible (bare "
            "hmr/r/oxor/bit ops and conditioned ops cannot appear in "
            "a mirrored region; and_compute/and_release pairs and alloc/free "
            "mirror through their journaled events — if b.ops was mutated in "
            "place after the checkpoint, the journal is out of sync)"
        )

    def _mirror_loop(self, loop, pos):
        """Return the exact inverse of a closed ``Loop`` subtree: reversed item
        order, index at stride position ``pos`` remapped i → count−1−i.
        Restricted to whitelisted unconditioned gates — anything else in the
        body raises (a loop body's own alloc/free/and pairs necessarily put
        R/HMR ops in it, so nothing mirrorable is ever rejected here).
        """
        items = []
        for item in reversed(loop.items):
            if isinstance(item, Loop):
                sub = self._mirror_loop(item, pos + 1)
                if sub.items:
                    items.append(self._remap_item(sub, pos, loop.count))
            elif item.kind in _MIRROR_WHITELIST and item.oc is None:
                items.append(self._remap_item(item, pos, loop.count))
            else:
                name = (
                    OpKind(item.kind).name if item.kind in OpKind._value2member_map_ else item.kind
                )
                raise ValueError(
                    f"mirror(): a loop in the region contains {name}"
                    f"{' with an outcome condition' if item.oc is not None else ''}"
                    " — only closed loops of unconditioned self-inverse gates "
                    "reverse; mirror measurement-based structure inside the "
                    "loop body instead"
                )
        return Loop(loop.count, items)

    def _verify_journal(self, target, pos, kind, qt):
        """Desync defense: the item a journal event points at must still be
        the op the event was recorded for (kind + concrete target qubit).
        """
        ok = False
        if 0 <= pos < len(target):
            item = target[pos]
            if isinstance(item, tuple):
                ok = item[0] == kind and item[3] == qt
            elif isinstance(item, OpRec) and item.kind == kind:
                names = _compact.KIND_OPERANDS.get(int(kind)) or ()
                if "qt" in names:
                    base, strides = item.operands[names.index("qt")]
                    ok = base == qt and not any(strides)
        if not ok:
            raise RuntimeError(
                "mirror(): journal/op-stream desync — b.ops was mutated in "
                f"place after the checkpoint (op #{pos} is no longer the "
                f"{OpKind(kind).name} its journal event recorded)"
            )

    @classmethod
    def _remap_item(cls, item, pos, count):
        """Apply the index remap i_pos → count−1−i_pos to a subtree (pure)."""
        if isinstance(item, Loop):
            return Loop(item.count, [cls._remap_item(x, pos, count) for x in item.items])

        def rm(operand):
            base, strides = operand
            s = strides[pos]
            if s == 0:
                return operand
            st = list(strides)
            st[pos] = -s
            return (base + s * (count - 1), tuple(st))

        gate = item.gate
        if gate is not None:
            # The gate's (offset, strides) pair remaps exactly like an operand.
            gate = (gate[0],) + rm((gate[1], gate[2]))
        return OpRec(
            item.kind,
            tuple(rm(o) for o in item.operands),
            rm(item.oc) if item.oc is not None else None,
            gate,
        )

    @staticmethod
    def _walk_items(item, inner_counts, sink):
        """Walk a subtree — the one traversal every accounting pass shares: call
        ``sink(kind, gate, inner_counts, conditioned)`` per op leaf of the
        subtree — ``conditioned`` is the leaf's bit-condition flag (billing
        promotes on it) — with the loop counts INSIDE the subtree (enclosing
        open loops are the sink's concern), so each sink prices plain and
        gated leaves under its own count base.
        """
        if isinstance(item, Loop):
            inner = inner_counts + (item.count,)
            for x in item.items:
                Builder._walk_items(x, inner, sink)
        elif isinstance(item, tuple):
            sink(item[0], None, inner_counts, item[8] != NO_BIT)
        else:
            sink(item.kind, item.gate, inner_counts, item.cc is not None)

    def _account_item(self, item, inner_counts):
        """Phase-stat accounting for a re-emitted OpRec/Loop subtree: plain ops
        multiply by the subtree's inner loop counts (enclosing open loops
        multiply out at their exit), gated ops add their exact expanded
        popcount at the root.
        """

        def sink(kind, gate, inner, conditioned):
            if _gate_class(kind, conditioned) is None:
                return
            if gate is None:
                mult = 1
                for c in inner:
                    mult *= c
                self._count_gate(kind, mult, conditioned=conditioned)
            else:
                table_id, offset, strides = gate
                counts = tuple(f.count for f in self._loop_stack) + inner
                self._count_gate_root(
                    kind,
                    _compact.gated_popcount(self._tables[table_id].bits, offset, strides, counts),
                    conditioned=conditioned,
                )

        self._walk_items(item, inner_counts, sink)

    # ---- phase labels (metadata-only; drives phase_summary) --------------------
    def _pos_token(self):
        """Return the current recording position as a path token: the item-list lengths
        along the open-frame chain, outermost first. Loop bodies are recorded
        once, so a flat op index does not exist mid-build; item lists only
        append, so tokens order chronologically (tuple comparison) and resolve
        to executed coordinates after the run. Flat programs degenerate to
        ``(op_index,)``.
        """
        return (len(self._items), *(len(f.items) for f in self._loop_stack))

    def claim(self, name, wires):
        """Wires join the named wire-set — the wire-axis annotation
        (sibling of the time-axis :meth:`set_phase`). Pure bookkeeping:
        emits no op; ``ops.bin`` is byte-identical with or without it.

        Names are hierarchical (``"tape/w3"``), coarsened by ``/`` depth
        like phase names. Two modes by prefix:

        * ownership (default): exclusive — a wire belongs to at most one
          ownership set at a time, releases must precede :meth:`recycle`,
          and ``live − Σ owned`` is the anonymous scratch pool. The
          occupancy tooling validates these rules; the Builder only
          records.
        * decorative (``~`` prefix, e.g. ``"~syn02/3"``): non-exclusive
          labels that may overlap anything; membership ends implicitly
          when the wire is freed.
        """
        self.wire_log.append(
            (
                "claim",
                (name, tuple(_as_int(w, "claim wire id") for w in wires)),
                self._pos_token(),
                self.active_qubits,
            )
        )

    def release(self, name, wires):
        """Wires leave the named wire-set. See :meth:`claim`; decorative
        (``~``) sets rarely need explicit releases (free implies them).
        """
        self.wire_log.append(
            (
                "release",
                (name, tuple(_as_int(w, "release wire id") for w in wires)),
                self._pos_token(),
                self.active_qubits,
            )
        )

    def release_recycle(self, name, wires):
        """Release ``wires`` from the named wire-set, then recycle them in
        reverse order. The call equals ``release(name, wires)`` followed by
        ``recycle(w)`` for each ``w`` in ``reversed(wires)``: the same ops and
        the same record.
        """
        wires = list(wires)
        self.release(name, wires)
        for w in reversed(wires):
            self.recycle(w)

    def set_phase(self, name):
        """Label subsequent gates/allocations as belonging to phase ``name``.

        Pure bookkeeping — emits no op, so ``ops.bin`` is byte-identical whether or
        not you label phases. Drives the per-phase cost breakdown in ``phase_summary``.
        A span that records no op and no allocation or free leaves its phase's
        peak unchanged, so the restore at the end of a :meth:`phase` block
        does not inflate the outer phase's peak.
        """
        idle = self._idle_peak()
        if idle is not None:
            self._phase(idle[0])["peak"] = idle[1]
        self.current_phase = name
        self._cur_stats = self._phase(name)
        self._phase_entry = (name, self._cur_stats["peak"], self._pos_token(), self._wire_events)
        self._touch_phase_peak(name)
        self._jlog(("phase", len(self._target()), name))
        self.wire_log.append(("mark", name, self._pos_token(), self.active_qubits))

    def _idle_peak(self):
        """Return (name, peak before entry) when the current phase span is idle.

        A span is idle while it has recorded no op and no allocation or free
        since its set_phase. Its entry then must not count toward the peak.
        """
        e = self._phase_entry
        if not e or e[0] != self.current_phase or e[3] != self._wire_events:
            return None
        # A loop opened since the entry adds an empty frame (a trailing 0)
        # to the position, but records nothing.
        pos = self._pos_token()
        while len(pos) > len(e[2]) and pos[-1] == 0:
            pos = pos[:-1]
        return (e[0], e[1]) if pos == e[2] else None

    @contextmanager
    def phase(self, name):
        """Run a block under phase ``name``, restoring the previous phase after."""
        prev = self.current_phase
        self.set_phase(name)
        try:
            yield
        finally:
            self.set_phase(prev)

    def phase_summary(self):
        """Return the per-phase breakdown of emitted gates and local peaks as a dict.

        ``toffoli`` counts are *emitted* CCX/CCZ (the TQ-cost gate) — for loop
        programs, expanded counts (loop bodies multiply by their counts, gated
        ops by their table popcounts). Emitted equals the evaluator's
        executed-Toffoli total for every producer: the op stream forbids
        outcome conditions on Toffoli kinds. ``peak`` is the max
        simultaneously-live qubit count reached while that phase was active.
        ``peak_phase`` names the phase holding the qubit id when the global
        peak was hit.
        """
        idle = self._idle_peak()
        return {
            "peak_qubits": self.peak_qubits,
            "peak_phase": self.peak_phase,
            "total_emitted_toffoli": sum(s["toffoli"] for s in self._phase_stats.values()),
            "total_emitted_clifford": sum(s["clifford"] for s in self._phase_stats.values()),
            "phases": [
                {"name": name, **self._phase_stats[name]}
                | ({"peak": idle[1]} if idle and idle[0] == name else {})
                for name in self._phase_order
            ],
        }

    def _stream_totals(self):
        """Recount the expanded toffoli/clifford totals from the recorded
        items — the emit-time record-integrity check's independent source
        (the same traversal as ``_account_item``, accumulated instead of
        attributed to phases; called with the loop stack empty, so inner
        counts are the full expansion counts).
        """
        tables = {t.table_id: t.bits for t in self._tables}
        totals = {"toffoli": 0, "clifford": 0}

        def sink(kind, gate, inner, conditioned):
            key = _gate_class(kind, conditioned)
            if key is None:
                return
            if gate is not None:
                table_id, offset, strides = gate
                n = _compact.gated_popcount(tables[table_id], offset, strides, inner)
            else:
                n = 1
                for c in inner:
                    n *= c
            totals[key] += n

        # Packed rows (the common bulk): the distinct slot-0 values — kind
        # byte + sentinel mask — number a few dozen, so count them at C speed
        # and classify each once. Bit 0x8000 clear = the op carries a bit
        # condition (bills at its promoted kind), matching op[8] != NO_BIT.
        tape = self._items
        for s0, n in Counter(tape._rows[0::9]).items():
            if s0 == _TAPE_OBJ:
                continue
            key = _gate_class(s0 & 0xFF, not s0 & _TAPE_CC)
            if key is not None:
                totals[key] += n
        for item in tape._objs.values():
            if isinstance(item, tuple):
                key = _gate_class(item[0], item[8] != NO_BIT)
                if key is not None:
                    totals[key] += 1
            else:
                self._walk_items(item, (), sink)
        return totals["toffoli"], totals["clifford"]

    # ---- output --------------------------------------------------------------
    def _build_segments(self):
        """Build the op-stream segment list: REGS, all DATA tables, then the
        recorded FLAT runs / LOOP trees in program order.

        FLAT runs are zero-copy ``_TapeRun`` views over the columnar record —
        the serializer decodes each op once, straight into its byte encoding
        (the same bytes as a plain op list), so large flat programs
        never materialize a parallel tuple list per op.
        """
        segments = []
        if self._registers:
            segments.append(Regs(self._registers))
        segments.extend(self._tables)
        tape = self._items
        objs = tape._objs
        if not objs:
            # Pure packed-row program (no Loop/OpRec nodes): one run, O(1).
            n = len(tape)
            if n:
                segments.append(Flat(_TapeRun(tape, 0, len(tape), n)))
            return segments
        # One emit-time snapshot of the OpRec indices (already ascending —
        # dict insertion order — but sorted() guards the splice paths): every
        # run's plain_op_count answers by bisect instead of scanning _objs.
        okeys = sorted(k for k, v in objs.items() if isinstance(v, OpRec))
        start = n_run = 0  # current run: first index, item count
        for i, s0 in tape._scan_slot0():
            if s0 == _TAPE_OBJ and isinstance(objs[i], Loop):
                if n_run:
                    segments.append(Flat(_TapeRun(tape, start, i, n_run, okeys)))
                segments.append(objs[i])
                start, n_run = i + 1, 0
            else:
                n_run += 1
        if n_run:
            segments.append(Flat(_TapeRun(tape, start, len(tape), n_run, okeys)))
        return segments

    def emit(self, meta=True, dest=None):
        """Serialize the recorded program as an op stream.

        Transport is fixed: ``UNCOMPUTE_OPS_PATH`` when set (the streamed
        evaluator's FIFO), else ``./ops.bin``. There is no path or format
        choice — every program, loop-free ones included, is emitted in the
        compact framing (a loop-free program is just REGS + FLAT + END). The
        FIFO double-send guard (``ops.fifo_send_guard``) applies: an identical
        re-emit is a no-op, a changed re-emit raises.

        ``dest`` (a binary file object) captures the stream in-process
        instead. It is a capture hook, not a transport choice: paths are
        rejected.

        With ``meta=True`` (the default), after streaming one line

            UNCOMPUTE-META {"peak_qubits": ..., ...}

        is printed to stdout — compact JSON of :meth:`phase_summary`. It is
        advisory feedback only (the op stream is the interface to the
        evaluator, so nothing on stdout changes the result): the evaluator picks
        it up as ``report["circuit_meta"]``.
        """
        if self._loop_stack:
            raise RuntimeError("emit() called inside an open repeat() block")
        if dest is not None and not hasattr(dest, "write"):
            raise TypeError(
                "emit() dest= takes a binary file object; the stream path is "
                "fixed (UNCOMPUTE_OPS_PATH, else ./ops.bin)"
            )
        # Record-integrity contract: the phase stats (the record) must equal
        # what the recorded items expand to (the stream). Post-hoc surgery on
        # Builder.ops that does not keep the counters exact fails here — the
        # record must describe the stream it ships.
        stats = self.phase_summary()
        tof, cliff = self._stream_totals()
        if (tof, cliff) != (stats["total_emitted_toffoli"], stats["total_emitted_clifford"]):
            raise RuntimeError(
                f"emit(): recorded phase stats diverge from the op stream — "
                f"stats say toffoli={stats['total_emitted_toffoli']:,} "
                f"clifford={stats['total_emitted_clifford']:,} but the "
                f"recorded items expand to toffoli={tof:,} "
                f"clifford={cliff:,}. Record what you emit: rebuild with the "
                f"final ops recorded in place (post-hoc Builder.ops surgery "
                f"must keep the counters exact)"
            )
        global LAST_EMITTED
        LAST_EMITTED = self
        _compact.write_compact(self._build_segments(), "ops.bin" if dest is None else dest)
        if meta:
            print("UNCOMPUTE-META " + json.dumps(stats, separators=(",", ":")))
