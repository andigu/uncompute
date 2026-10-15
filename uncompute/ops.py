"""The uncompute op model: op kinds, operand rules, cost classes, transport.

The op set and its per-shot semantics follow the kickmix circuit model of
Babbush et al. (2026), arXiv:2603.28846.

The serializer (:mod:`uncompute.compact`) and the evaluator
(:mod:`uncompute.evalcore`) share this module. It holds what both sides need:

* :class:`OpKind`, the op kinds. The stream bytes carry their values.
* The in-memory op form: a 9-tuple
  ``(kind, qa, qb, qt, ot, oc, rt, ct, cc)``. An empty slot holds the
  sentinel ``2**64 - 1`` (``NO_QUBIT`` = ``NO_OUT`` = ``NO_REG`` =
  ``NO_BIT``). No kind uses ``rt``, so it is always empty.
* The operand rules: :data:`SIGNATURES` lists the slots each kind requires
  and accepts, and :func:`op_violation` checks one op against them.
* The cost classes: :func:`promoted_kind` and :func:`gate_class` price an
  op, including a bit-conditioned op at its promoted kind.
* The transport helpers: ``UNCOMPUTE_OPS_PATH`` resolution, FIFO detection,
  and the guard against a second, different stream on one FIFO.

Classical model. Measurement **outcomes** hold the results of ``HMR``
(which stores a fresh random bit) and ``OXOR`` (which accumulates
``ot ^= oc``). An outcome may condition ``Z``, ``CZ``, and ``NEG`` through
the ``oc`` slot; the condition narrows only the sign flip. Classical
**bits** hold per-shot input data (Bit registers) and the values that the
free bit-only kinds derive from it. A bit may condition ``X``, ``Z``,
``NEG``, ``CX``, ``CZ``, and the bit-only kinds through the ``cc`` slot. A
condition of either family narrows which shots an op acts on, never its
billing, so the billed counts of a valid stream are constants of the
stream. Every outcome and bit starts at 0. The operands of one op are
pairwise distinct within each space (qubits, outcomes, bits).

Op stream. All integers are little-endian. A stream means its flat
expansion; loops and data gates change only the spelling. The layout::

    "OPSTRM01" | segment* | END
    REGS  0x01  u32 n_regs, per register: u8 kind (0 Qubit, 1 Bit),
                u32 n_wires, n_wires x u32 wire id
    DATA  0x02  u32 table_id, u64 n_bits, ceil(n_bits/8) bytes (bit i:
                byte i/8, bit i mod 8)
    FLAT  0x03  u32 n_ops, n_ops op records at depth 0
    LOOP  0x04  u64 count, u16 n_items (> 0), per item: u8 (0 op record,
                1 nested LOOP) and its body; depth at most 4
    END   0xFF  u64 expanded op count (2**64 - 1 = unknown)
    op record at depth d: u8 kind, u8 flags,
        [u32 table_id, u64 offset, d x i64 stride]  if flags bit 1,
        operands: each u32 base + d x i64 stride,
        in order qa, qb, qt, ot, oc, ct (required slots only),
        then oc if flags bit 0, then cc if flags bit 2.

A stream has at most one REGS segment, and it precedes every FLAT and
LOOP segment. The evaluator loads the test inputs into the registers when
it reads REGS. A DATA segment precedes every op that names its table. An
operand resolves to ``base + sum(stride_j * i_j)`` over the enclosing loop
indices, and must lie in ``[0, 2**32)``. A data-gated op exists only on
iterations where its table bit ``offset + sum(stride_j * i_j)`` is 1. An
op that does not exist is not checked, counted, or billed, and takes no
measurement draw. The qubit count is the largest qubit id in the
expansion or in a Qubit register, plus 1.
:mod:`uncompute.evalcore.compact_reader` is the reference parser; its
docstring lists the error codes and the order of the per-op checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# The sentinel of an empty slot (2^64 - 1). All four names share one value.
NO_QUBIT = (1 << 64) - 1
NO_OUT = (1 << 64) - 1
NO_REG = (1 << 64) - 1
NO_BIT = (1 << 64) - 1


class OpKind(IntEnum):
    """Op kinds. The stream bytes carry these values."""

    X = 0
    Z = 1
    NEG = 2
    CX = 3
    CZ = 4
    SWAP = 5
    CCX = 6
    CCZ = 7
    R = 8
    HMR = 9
    OXOR = 10
    BIT_INVERT = 11
    BIT_STORE0 = 12
    BIT_STORE1 = 13


# ---- operand rules ------------------------------------------------------------
# Slot names of the in-memory op tuple, at tuple positions 1..8.
SLOTS = ("qa", "qb", "qt", "ot", "oc", "rt", "ct", "cc")
SLOT_POS = {name: i for i, name in enumerate(SLOTS, 1)}


@dataclass(frozen=True)
class Signature:
    """The slots one op kind fills: ``required`` slots must hold an id,
    ``optional`` slots may, and every other slot stays empty. Both tuples
    list slots in tuple order, which is also the operand order of a stream
    op record.
    """

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


_K = OpKind
_BIT_OP = Signature(("ct",), ("cc",))
SIGNATURES = {
    _K.NEG: Signature((), ("oc", "cc")),
    _K.X: Signature(("qt",), ("cc",)),
    _K.Z: Signature(("qt",), ("oc", "cc")),
    _K.CX: Signature(("qb", "qt"), ("cc",)),
    _K.CZ: Signature(("qb", "qt"), ("oc", "cc")),
    _K.SWAP: Signature(("qb", "qt")),
    _K.R: Signature(("qt",)),
    _K.HMR: Signature(("qt", "ot")),
    _K.CCX: Signature(("qa", "qb", "qt")),
    _K.CCZ: Signature(("qa", "qb", "qt")),
    _K.OXOR: Signature(("ot", "oc")),
    _K.BIT_INVERT: _BIT_OP,
    _K.BIT_STORE0: _BIT_OP,
    _K.BIT_STORE1: _BIT_OP,
}
assert set(SIGNATURES) == set(OpKind)

# Kinds that may carry an outcome condition: the sign-flip kinds.
OC_ALLOWED_KINDS = frozenset(int(k) for k, sig in SIGNATURES.items() if "oc" in sig.optional)


def _slot_mask(names):
    """Bit ``SLOT_POS[name] - 1`` set for every slot in ``names``."""
    m = 0
    for name in names:
        m |= 1 << (SLOT_POS[name] - 1)
    return m


def operand_mask_table() -> bytes:
    """Return the operand rules as one byte pair ``(required, optional)``
    per kind, indexed by kind. Bit ``s - 1`` of a mask stands for tuple slot ``s``.
    The C kernel's Simulator fast path checks ops against this table.
    """
    out = bytearray()
    for kind in OpKind:
        sig = SIGNATURES[kind]
        out += bytes((_slot_mask(sig.required), _slot_mask(sig.optional)))
    return bytes(out)


# Slot groups whose ids must differ within one op.
_DISTINCT_GROUPS = (("qa", "qb", "qt"), ("ot", "oc"), ("ct", "cc"))


def op_violation(op) -> str | None:
    """Check one in-memory op against the operand rules.

    Returns None for a valid op, else a one-line reason. The checks run in
    a fixed order: known kind, distinct operands, required slots, and empty
    slots. The evaluator reports the distinctness reason as
    ``validate/aliasing``.
    """
    kind = op[0]
    if kind not in _KIND_VALUES:
        return f"unknown op kind {kind}"
    name = OpKind(kind).name
    for group in _DISTINCT_GROUPS:
        seen = {}
        for slot in group:
            v = op[SLOT_POS[slot]]
            if v == NO_QUBIT:
                continue
            if v in seen:
                return f"{name}: slots {seen[v]} and {slot} hold the same id {v}"
            seen[v] = slot
    sig = SIGNATURES[kind]
    for slot in sig.required:
        if op[SLOT_POS[slot]] == NO_QUBIT:
            return f"{name}: slot {slot} is required"
    for slot in SLOTS:
        if op[SLOT_POS[slot]] != NO_QUBIT and slot not in sig.required and slot not in sig.optional:
            return f"{name}: slot {slot} must be empty"
    return None


_KIND_VALUES = frozenset(int(k) for k in OpKind)


# ---- cost classes -----------------------------------------------------------------
# CCX/CCZ are the costed Toffoli count. CX/CZ/SWAP/R/HMR are Clifford gates:
# counted, not costed. X/Z/NEG are free, because a classical control
# system tracks them. OXOR and the bit-only kinds are classical
# bookkeeping and also free. The sets hold raw ints, as ops are int tuples.
TOFFOLI_KINDS = frozenset({int(_K.CCX), int(_K.CCZ)})
CLIFFORD_KINDS = frozenset({int(_K.CX), int(_K.CZ), int(_K.SWAP), int(_K.R), int(_K.HMR)})

# The billed kind of a bit-conditioned op: the kind it would have if the
# condition bit were a qubit control. The bit-only kinds bill as
# themselves. A kind outside this table cannot carry a bit condition.
_CC_PROMOTION = {
    int(_K.X): int(_K.CX),
    int(_K.CX): int(_K.CCX),
    int(_K.Z): int(_K.CZ),
    int(_K.CZ): int(_K.CCZ),
    int(_K.NEG): int(_K.Z),
    int(_K.BIT_INVERT): int(_K.BIT_INVERT),
    int(_K.BIT_STORE0): int(_K.BIT_STORE0),
    int(_K.BIT_STORE1): int(_K.BIT_STORE1),
}

# Kinds that may carry a bit condition: exactly the promotable kinds.
CC_ALLOWED_KINDS = frozenset(_CC_PROMOTION)
assert CC_ALLOWED_KINDS == frozenset(int(k) for k, s in SIGNATURES.items() if "cc" in s.optional)


def promoted_kind(kind, conditioned):
    """Return the billed kind of an op: its own kind, or its promoted kind
    under a bit condition. Raises ValueError for a conditioned kind that
    cannot carry a bit condition.
    """
    k = int(kind)
    if not conditioned:
        return k
    p = _CC_PROMOTION.get(k)
    if p is None:
        raise ValueError(f"kind {k} cannot carry a bit condition")
    return p


def gate_class(kind, conditioned=False):
    """Return ``"toffoli"``, ``"clifford"``, or None for an op, priced at its
    billed kind. The Builder's census, the Simulator's counters, and the
    offline profilers all price ops through this function.
    """
    k = promoted_kind(kind, conditioned)
    if k in TOFFOLI_KINDS:
        return "toffoli"
    if k in CLIFFORD_KINDS:
        return "clifford"
    return None


# Qubit slots per kind (tuple positions), and the subset whose basis value
# the op can change. Controls and sign kinds read without writing. HMR and
# R force their target to 0, which counts as a reset, not a data write.
# The liveness analysis in tools/liveness_profile.py reads these tables.
QUBIT_FIELDS = {
    int(k): tuple(SLOT_POS[s] for s in sig.required if s in ("qa", "qb", "qt"))
    for k, sig in SIGNATURES.items()
}
WRITE_FIELDS = {
    int(_K.X): (3,),
    int(_K.CX): (3,),
    int(_K.CCX): (3,),
    int(_K.SWAP): (2, 3),
}
# Kinds that force their qt to 0 (measurement or reset).
ZEROING_KINDS = frozenset({int(_K.HMR), int(_K.R)})

_popcount = int.bit_count


def resolve_ops_path(path):
    """Honor the ``UNCOMPUTE_OPS_PATH`` override (streamed evaluation: the
    evaluator points the circuit at a FIFO so the stream never touches disk).
    """
    return os.environ.get("UNCOMPUTE_OPS_PATH", path)


def _is_fifo(path):
    try:
        import stat

        return stat.S_ISFIFO(Path(path).stat().st_mode)
    except OSError:
        return False


# One stream per FIFO per process. Circuit code may call ``b.emit()`` more
# than once; on a regular file the second emit truncates and rewrites identical bytes (harmless), but
# on a FIFO sequential writes would concatenate and corrupt the stream. This
# guard reproduces the file semantics: an identical re-send is a no-op, a
# changed re-send is an error.
_FIFO_SENT = {}  # resolved fifo path -> sha256 digest of the stream already sent


def fifo_prev_digest(path):
    """Return the digest of the stream already sent through ``path`` (None before
    the first send) — lets the serializer stream a first send straight
    through and take a hash-only pass on re-emits.
    """
    return _FIFO_SENT.get(path)


def fifo_send_guard(path, payload_digest):
    """Return True if the payload must be sent, False for an identical resend.

    Raises ValueError when a different stream was already sent through
    ``path`` — the reader consumed it and cannot be rewound.
    """
    prev = _FIFO_SENT.get(path)
    if prev is None:
        _FIFO_SENT[path] = payload_digest
        return True
    if prev == payload_digest:
        return False
    raise ValueError(
        f"a different op stream was already sent through the FIFO {path!r}; "
        "the evaluator consumed it and cannot rewind. Emit the final circuit "
        "exactly once (or make repeat emits byte-identical)."
    )
