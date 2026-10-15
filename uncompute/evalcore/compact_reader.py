"""The reference interpreter of the op stream (normative).

The :mod:`uncompute.ops` docstring specifies the format. This module reads
the segments, the op records with affine operands, the LOOP items, and the
data gates, and it executes the expansion. The compact form changes only
the spelling: the interpreter produces the same results as an op-by-op
evaluation of the flat expansion. It runs loops directly and never
materializes the expansion.

The parser follows the specification alone and shares no code with the
serializer in :mod:`uncompute.compact`, so the two implementations check each
other.

Errors (codes from :mod:`uncompute.evalcore.report`):

* ``CompactParseError`` (a ``LoadError``): a structural rejection while
  parsing. Framing and segment rules give ``load/segment``, truncation
  gives ``load/truncated``, a wrong END count gives
  ``load/count-mismatch``, and an illegal op record (unknown kind,
  illegal flags, unknown table) gives ``load/op-invalid``. Kind and
  flag checks run once, at parse.
* ``_ValidationError`` (internal): a per-expanded-op check failed. The
  checks are the data-gate index range, the resolved-id range, operand
  distinctness, and the address caps as the census grows. The error
  carries the **expanded op index**: the position the op has, or would
  have, in a flat spelling, counted post-gating.

Per-op check order (it decides which error a run reports): data-gate index,
then the gate bit (a gated-off op does not exist: no further checks, no
census, no billing, no measurement-counter step), then the resolved-id
range, then distinctness, then the census caps.

The interpreter resolves affine operands incrementally. Each op record
keeps its current resolved values. Each LOOP adds its stride once per
iteration and subtracts the total on exit, so no iteration multiplies.
"""

from __future__ import annotations

import struct
from typing import Any

from ..ops import (
    CC_ALLOWED_KINDS,
    NO_BIT,
    NO_OUT,
    NO_QUBIT,
    NO_REG,
    OC_ALLOWED_KINDS,
    SIGNATURES,
    op_violation,
)
from .executor import (
    UNKNOWN_COUNT,
    LaneSim,
    begin_batch,
    fail_batch,
    finish_batch,
)
from .loader import CompactParseError
from .report import (
    CAPS_ADDRESS,
    LOAD_COUNT_MISMATCH,
    LOAD_OP_INVALID,
    LOAD_SEGMENT,
    LOAD_TRUNCATED,
    VALIDATE_ALIASING,
    VALIDATE_ID_RANGE,
    VALIDATE_TABLE_INDEX,
)
from .util import MAX_BIT_ADDR, MAX_OUT_ADDR, MAX_QUBIT_ADDR, WKIND_BIT, WKIND_QUBIT, _set_register

SEG_REGS = 0x01
SEG_DATA = 0x02
SEG_FLAT = 0x03
SEG_LOOP = 0x04
SEG_END = 0xFF

MAX_LOOP_DEPTH = 4
ID_LIMIT = 1 << 32  # a resolved operand id lies in [0, 2^32)

_CHUNK_OPS = 8192

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")

# The required operands of each stream kind, in stream order; any other
# kind is unknown. OXOR's oc is a required operand. The flag-selected oc and
# cc follow the required ones.
_REQUIRED_OPERANDS = {int(k): sig.required for k, sig in SIGNATURES.items()}
# Position of each operand field in the flat op 9-tuple
# (kind, qa, qb, qt, ot, oc, rt, ct, cc).
_FIELD_POS = {"qa": 1, "qb": 2, "qt": 3, "ot": 4, "oc": 5, "ct": 7, "cc": 8}
_QUBIT_FIELDS = frozenset({"qa", "qb", "qt"})
_BIT_FIELDS = frozenset({"ct", "cc"})
# Census space per operand field: 0 = qubit, 1 = outcome, 2 = bit.
_SPACE_QUBIT, _SPACE_OUT, _SPACE_BIT = 0, 1, 2


def _field_space(name):
    if name in _QUBIT_FIELDS:
        return _SPACE_QUBIT
    if name in _BIT_FIELDS:
        return _SPACE_BIT
    return _SPACE_OUT


class _ValidationError(Exception):
    """A per-expanded-op validation error."""

    def __init__(self, code: str, detail: str, op_index: int | None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.op_index = op_index


class _OpRec:
    """One parsed op record with its current resolved operand values.

    ``cur`` holds the mutable affine values: slot 0 is the data-gate bit
    index when ``gated``, then one slot per operand (required order, then oc
    if flagged, then cc if flagged). ``slot_strides[k]`` is slot k's stride
    tuple (length = loop depth d, outermost first); enclosing loops register
    (cur, k, stride) adjust entries from it. ``fields`` is precomputed per
    operand as ``(name, flat-tuple position, census space)`` — the field
    list is fixed at parse time, so the per-op execution loop does no dict
    lookups.
    """

    __slots__ = (
        "cur",
        "fields",
        "gated",
        "kind",
        "slot_strides",
        "table_bits",
        "table_id",
        "table_n_bits",
    )

    def __init__(self, kind, fields, gated, table_id, table_bits, table_n_bits, cur, slot_strides):
        self.kind = kind
        self.fields = fields
        self.gated = gated
        self.table_id = table_id
        self.table_bits = table_bits
        self.table_n_bits = table_n_bits
        self.cur = cur
        self.slot_strides = slot_strides


class _Loop:
    """A parsed LOOP item: count, body items, and this level's adjust list."""

    __slots__ = ("adjust", "count", "items")

    def __init__(self, count, items):
        self.count = count
        self.items = items
        self.adjust: tuple = ()


def _wire_adjust(node: _Loop, pos: int) -> list[_OpRec]:
    """Assign each loop its incremental-add list (cur, slot, stride) over the
    op records of its subtree; returns those records for the caller's level.
    """
    recs: list[_OpRec] = []
    for item in node.items:
        if isinstance(item, _Loop):
            recs.extend(_wire_adjust(item, pos + 1))
        else:
            recs.append(item)
    node.adjust = tuple(
        (rec.cur, k, strides[pos])
        for rec in recs
        for k, strides in enumerate(rec.slot_strides)
        if strides[pos] != 0
    )
    return recs


class _CompactVM:
    """Single-pass segment walker and direct loop interpreter."""

    def __init__(self, stream, tests, sim: Any):
        self._f = stream
        self._tests = tests
        self.sim = sim
        # registers[r] = (reg_kind, wires): WKIND_QUBIT wires are qubit ids,
        # WKIND_BIT wires are bit ids.
        self.registers: list[tuple[int, list[int]]] = []
        self.tables = {}  # table_id -> (bits, n_bits)
        self.num_qubits = 0
        self.num_outs = 0
        self.num_bits = 0
        self.n_expanded = 0
        self._chunk: list[tuple] = []
        self._saw_regs = False
        self._saw_program = False  # any FLAT/LOOP segment yet

    # -- low-level reads ---------------------------------------------------------

    def _read_exact(self, n: int, what: str) -> bytes:
        buf = b""
        while len(buf) < n:
            got = self._f.read(n - len(buf))
            if not got:
                raise CompactParseError(
                    f"stream: truncated reading {what} (wanted {n} bytes, got {len(buf)})",
                    LOAD_TRUNCATED,
                )
            buf += got
        return buf

    def _u8(self, what):
        return _U8.unpack(self._read_exact(1, what))[0]

    def _u16(self, what):
        return _U16.unpack(self._read_exact(2, what))[0]

    def _u32(self, what):
        return _U32.unpack(self._read_exact(4, what))[0]

    def _u64(self, what):
        return _U64.unpack(self._read_exact(8, what))[0]

    def _i64(self, what):
        return _I64.unpack(self._read_exact(8, what))[0]

    # -- segment walk ------------------------------------------------------------

    def run(self) -> None:
        """Walk segments until END (mandatory). Raises CompactParseError /
        _ValidationError; on clean return the whole stream was consumed.
        """
        while True:
            head = self._f.read(1)
            if not head:
                raise CompactParseError("stream: EOF before END segment", LOAD_TRUNCATED)
            seg = head[0]
            if seg == SEG_REGS:
                self._seg_regs()
            elif seg == SEG_DATA:
                self._seg_data()
            elif seg == SEG_FLAT:
                self._saw_program = True
                self._seg_flat()
            elif seg == SEG_LOOP:
                self._saw_program = True
                self._seg_loop()
            elif seg == SEG_END:
                self._seg_end()
                return
            else:
                raise CompactParseError(f"stream: unknown segment type 0x{seg:02x}", LOAD_SEGMENT)

    def _seg_regs(self) -> None:
        if self._saw_regs:
            raise CompactParseError("stream: duplicate REGS segment (at most one)", LOAD_SEGMENT)
        if self._saw_program:
            raise CompactParseError(
                "stream: REGS segment must precede every FLAT/LOOP segment", LOAD_SEGMENT
            )
        self._saw_regs = True
        n_regs = self._u32("REGS n_regs")
        for r in range(n_regs):
            kind = self._u8(f"REGS reg {r} reg_kind")
            if kind not in (WKIND_QUBIT, WKIND_BIT):
                raise CompactParseError(
                    f"stream: unknown REGS reg_kind {kind} for register {r}", LOAD_SEGMENT
                )
            n_wires = self._u32(f"REGS reg {r} n_wires")
            wires = []
            for w in range(n_wires):
                wid = self._u32(f"REGS reg {r} wire {w} id")
                wires.append(wid)
                if kind == WKIND_QUBIT:
                    self.num_qubits = max(self.num_qubits, wid + 1)
                elif wid + 1 > self.num_bits:
                    self.num_bits = wid + 1
            self.registers.append((kind, wires))
        self._check_caps(op_index=None)
        # Registers are declared up front, so inputs load before the first
        # gate with zero buffering (the point of this framing).
        for shot, tc in enumerate(self._tests):
            if tc is None:
                continue  # dead lane (skipped draw): loads nothing
            for ri, val in enumerate(tc.inputs):
                if val is not None and ri < len(self.registers):
                    _set_register(self.sim, self.registers[ri], val, shot)

    def _seg_data(self) -> None:
        table_id = self._u32("DATA table_id")
        if table_id in self.tables:
            raise CompactParseError(f"stream: duplicate DATA table_id {table_id}", LOAD_SEGMENT)
        n_bits = self._u64("DATA n_bits")
        bits = self._read_exact((n_bits + 7) // 8, f"DATA table {table_id} bits")
        self.tables[table_id] = (bits, n_bits)

    def _seg_flat(self) -> None:
        n_ops = self._u32("FLAT n_ops")
        for _ in range(n_ops):
            rec = self._parse_op(0)
            self._exec_op(rec)

    def _seg_loop(self) -> None:
        node = self._parse_loop(enclosing=0)
        _wire_adjust(node, 0)
        self._run_loop(node)

    def _seg_end(self) -> None:
        declared = self._u64("END expanded_op_count")
        extra = self._f.read(1)
        if extra:
            raise CompactParseError("stream: trailing bytes after END segment", LOAD_SEGMENT)
        if declared != UNKNOWN_COUNT and declared != self.n_expanded:
            raise CompactParseError(
                f"stream: expanded op count mismatch: got {self.n_expanded}"
                f" ops, END declared {declared}",
                LOAD_COUNT_MISMATCH,
            )

    # -- op and loop parsing (structural checks, once per record) -----------------

    def _parse_op(self, depth: int) -> _OpRec:
        kind = self._u8("op kind")
        flags = self._u8("op flags")
        if kind not in _REQUIRED_OPERANDS:
            raise CompactParseError(f"op: unknown kind {kind}", LOAD_OP_INVALID)
        if flags & ~0x07:
            raise CompactParseError(
                f"op: reserved flag bits set (flags=0x{flags:02x})", LOAD_OP_INVALID
            )
        has_oc = bool(flags & 0x01)
        gated = bool(flags & 0x02)
        has_cc = bool(flags & 0x04)
        if has_oc and kind not in OC_ALLOWED_KINDS:
            raise CompactParseError(
                f"op: kind {kind} sets flag bit 0 (outcome condition), "
                "which only Z, CZ, and NEG may carry",
                LOAD_OP_INVALID,
            )
        if has_cc and kind not in CC_ALLOWED_KINDS:
            raise CompactParseError(
                f"op: kind {kind} sets flag bit 2 (bit condition), which only "
                "X, Z, NEG, CX, CZ, and the bit-only kinds may carry",
                LOAD_OP_INVALID,
            )

        cur: list[int] = []
        slot_strides: list[tuple[int, ...]] = []
        table_id = table_bits = table_n_bits = None
        if gated:
            table_id = self._u32("gate table_id")
            offset = self._u64("gate offset")
            strides = tuple(self._i64("gate stride") for _ in range(depth))
            tbl = self.tables.get(table_id)
            if tbl is None:
                raise CompactParseError(
                    f"op: gate references table {table_id} with no preceding DATA segment",
                    LOAD_OP_INVALID,
                )
            table_bits, table_n_bits = tbl
            cur.append(offset)
            slot_strides.append(strides)

        names = _REQUIRED_OPERANDS[kind]
        if has_oc:
            names = names + ("oc",)
        if has_cc:
            names = names + ("cc",)
        fields = []
        for name in names:
            base = self._u32("operand base")
            strides = tuple(self._i64("operand stride") for _ in range(depth))
            cur.append(base)
            slot_strides.append(strides)
            fields.append((name, _FIELD_POS[name], _field_space(name)))

        return _OpRec(
            kind, tuple(fields), gated, table_id, table_bits, table_n_bits, cur, slot_strides
        )

    def _parse_loop(self, enclosing: int) -> _Loop:
        if enclosing + 1 > MAX_LOOP_DEPTH:
            raise CompactParseError(
                f"stream: LOOP nesting depth exceeds {MAX_LOOP_DEPTH}", LOAD_SEGMENT
            )
        count = self._u64("LOOP count")
        n_items = self._u16("LOOP n_items")
        if n_items == 0:
            raise CompactParseError("stream: LOOP with n_items == 0", LOAD_SEGMENT)
        items = []
        for _ in range(n_items):
            item_kind = self._u8("LOOP item kind")
            if item_kind == 0x00:
                items.append(self._parse_op(enclosing + 1))
            elif item_kind == 0x01:
                items.append(self._parse_loop(enclosing + 1))
            else:
                raise CompactParseError(
                    f"stream: unknown LOOP item kind 0x{item_kind:02x}", LOAD_SEGMENT
                )
        return _Loop(count, items)

    # -- direct interpretation --------------------------------------------------------

    def _run_loop(self, node: _Loop) -> None:
        count = node.count
        if count == 0:
            return  # contributes nothing
        items = node.items
        adjust = node.adjust
        for _ in range(count):
            for item in items:
                if isinstance(item, _Loop):
                    self._run_loop(item)
                else:
                    self._exec_op(item)
            for cur, k, s in adjust:
                cur[k] += s
        # One multiply per loop *exit* undoes the accumulation for the
        # parent's next iteration (per-iteration cost stays pure adds).
        for cur, k, s in adjust:
            cur[k] -= count * s

    def _exec_op(self, rec: _OpRec) -> None:
        cur = rec.cur
        i0 = 0
        if rec.gated:
            b = cur[0]
            if b < 0 or b >= rec.table_n_bits:
                raise _ValidationError(
                    VALIDATE_TABLE_INDEX,
                    f"data-gate bit index {b} out of [0, {rec.table_n_bits})"
                    f" (table {rec.table_id})",
                    self.n_expanded,
                )
            if not (rec.table_bits[b >> 3] >> (b & 7)) & 1:
                return  # a gated-off op does not exist in the expansion
            i0 = 1

        tup = [rec.kind, NO_QUBIT, NO_QUBIT, NO_QUBIT, NO_OUT, NO_OUT, NO_REG, NO_BIT, NO_BIT]
        num_qubits = self.num_qubits
        num_outs = self.num_outs
        num_bits = self.num_bits
        for j, (name, pos, space) in enumerate(rec.fields):
            v = cur[i0 + j]
            if v < 0 or v >= ID_LIMIT:
                raise _ValidationError(
                    VALIDATE_ID_RANGE,
                    f"operand {name} resolves to {v}, outside [0, 2^32)",
                    self.n_expanded,
                )
            tup[pos] = v
            if space == _SPACE_QUBIT:
                num_qubits = max(num_qubits, v + 1)
            elif space == _SPACE_BIT:
                num_bits = max(num_bits, v + 1)
            elif v + 1 > num_outs:
                num_outs = v + 1

        # Distinct operands. A parsed record always fills the right slots,
        # so the only operand rule an expanded op can break is distinctness.
        # The quick test below finds every such op; op_violation words the
        # reason, the same text the C kernel path reports.
        _, qa, qb, qt, ot, oc, _, ct, cc = tup
        if (
            (qt != NO_QUBIT and (qt == qb or qt == qa))
            or (qb != NO_QUBIT and qb == qa)
            or (ot != NO_OUT and ot == oc)
            or (ct != NO_BIT and ct == cc)
        ):
            reason = op_violation(tup)
            assert reason is not None  # the quick test implies a violation
            raise _ValidationError(VALIDATE_ALIASING, reason, self.n_expanded)

        # Caps track monotone census maxima, so only a growing op can breach
        # them — the failing op index is unchanged (the op that grew past the
        # cap is always a growth op).
        if num_qubits > self.num_qubits or num_outs > self.num_outs or num_bits > self.num_bits:
            self.num_qubits = num_qubits
            self.num_outs = num_outs
            self.num_bits = num_bits
            self._check_caps(op_index=self.n_expanded)

        self.n_expanded += 1
        self._chunk.append(tuple(tup))
        if len(self._chunk) >= _CHUNK_OPS:
            self.flush()

    def _check_caps(self, op_index: int | None) -> None:
        if (
            self.num_qubits > MAX_QUBIT_ADDR
            or self.num_outs > MAX_OUT_ADDR
            or self.num_bits > MAX_BIT_ADDR
        ):
            raise _ValidationError(
                CAPS_ADDRESS,
                f"address space too large: {self.num_qubits} qubits"
                f" (cap {MAX_QUBIT_ADDR}) / {self.num_outs} outcomes"
                f" (cap {MAX_OUT_ADDR}) / {self.num_bits} bits"
                f" (cap {MAX_BIT_ADDR})",
                op_index,
            )

    def flush(self) -> None:
        if self._chunk:
            # Every op in the chunk already passed the operand checks
            # above; skip the simulator's redundant per-op re-validation.
            self.sim.apply(self._chunk, validate=False)
            self._chunk = []


def run_compact_batch(ch, stream, tests, seed_rng: bytes, lane0: int = 0):
    """Evaluate one lane batch of an op stream (magic already consumed).

    Returns ``(EvalReport, registers, state)`` where ``state`` is the
    extracted per-shot mask triple (``executor.extract_batch_state``), or
    ``None`` when the report already failed or the register schema is
    invalid. Raises CompactParseError (a LoadError) on structural rejection.

    ``tests`` is positional: entry ``i`` is the shot at lane ``lane0 + i``,
    ``None`` marking a dead lane (skipped draw — protocol.py's dead-lane
    convention). ``n_shots`` and the stat denominators count live lanes only.
    """
    r, n, live_mask, sim_live = begin_batch(tests)

    vm = _CompactVM(stream, tests, LaneSim(n, seed_rng, lane0, live_mask=sim_live))
    try:
        vm.run()
    except _ValidationError as e:
        return fail_batch(
            r,
            vm.registers,
            e.code,
            e.detail,
            e.op_index,
            ops_len=vm.n_expanded,
            num_qubits=vm.num_qubits,
            num_outs=vm.num_outs,
            num_bits=vm.num_bits,
        )
    vm.flush()

    return finish_batch(
        ch,
        r,
        tests,
        vm.registers,
        vm.sim,
        live_mask,
        ops_len=vm.n_expanded,
        num_qubits=vm.num_qubits,
        num_outs=vm.num_outs,
        tot_tof=vm.sim.stats.toffoli,
        tot_cliff=vm.sim.stats.clifford,
        num_bits=vm.num_bits,
    )
