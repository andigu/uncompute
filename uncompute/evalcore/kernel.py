"""ctypes wrapper for the uncompute C kernel.

Loads ``_kernel.dylib`` / ``_kernel.so`` (built by ``make kernel``)
from this directory and exposes:

* :class:`KernelExecutor` — the native wide-lane simulator state, including
  the bulk register load/readout paths (``kern_load_register`` /
  ``kern_read_register``) and the sharded-evaluation lane offset
  (``kern_set_lane0``);
* :func:`run_compact_kernel_batch` — the kernel-backed twin of
  ``compact_reader.run_compact_batch``.

The pure-Python reference stays normative, and this path gives identical
results. The two paths share the end-of-stream verdict code:
the kernel state is snapshotted into Python lane ints and fed through
``executor.extract_batch_state`` / ``finish_from_batch``.
"""

from __future__ import annotations

import ctypes
import io
import mmap
import os
import stat as _stat
import struct
import sys
from pathlib import Path

from ..ops import NO_REG, op_violation
from .executor import (
    begin_batch,
    fail_batch,
    finish_batch,
    needed_registers,
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
from .util import WKIND_BIT, WKIND_QUBIT

# The platform library name/path, derived once at import with no side
# effects (importing this module never builds or loads the library; loading
# is lazy via load_library). The Makefile reads LIB_PATH from here.
LIB_NAME = "_kernel.dylib" if sys.platform == "darwin" else "_kernel.so"
LIB_PATH = str(Path(__file__).resolve().parent / LIB_NAME)

# The exported ABI version (must match _kernel.c's UNCOMPUTE_KERNEL_ABI).
# _bind checks it before it touches any other symbol: a stale library may
# export every kern_* name below with different signatures, so the
# presence of the symbols proves nothing.
KERNEL_ABI = 1

# stream-interpreter status and error codes (must match _kernel.c)
CST_PAUSED = 0
CST_DONE = 1
CERR_TRUNCATED = 10
CERR_SEGMENT = 11
CERR_OP_INVALID = 12
CERR_COUNT_MISMATCH = 13
CERR_TABLE_INDEX = 14
CERR_ID_RANGE = 15
CERR_ALIASING = 16
CERR_ADDR_CAP = 17
CERR_OOM = 19

# CERR_* -> stable report codes: parse-time codes raise CompactParseError,
# validation codes come back as a failed report (compact_reader behavior).
_CPARSE_CODES = {
    CERR_TRUNCATED: LOAD_TRUNCATED,
    CERR_SEGMENT: LOAD_SEGMENT,
    CERR_OP_INVALID: LOAD_OP_INVALID,
    CERR_COUNT_MISMATCH: LOAD_COUNT_MISMATCH,
}
_CVALIDATE_CODES = {
    CERR_TABLE_INDEX: VALIDATE_TABLE_INDEX,
    CERR_ID_RANGE: VALIDATE_ID_RANGE,
    CERR_ALIASING: VALIDATE_ALIASING,
    CERR_ADDR_CAP: CAPS_ADDRESS,
}

_U64 = ctypes.c_uint64
_I64 = ctypes.c_int64

_lib = None
_lib_err: str | None = None


def _bind(lib) -> None:
    P = ctypes.POINTER
    # ABI handshake FIRST: a wrong value means a stale dylib built from an
    # older _kernel.c — fail the bind atomically so load_library converts it
    # into the clean pure-Python fallback (kernel_available() False) instead
    # of a mid-run mis-read of arguments. A missing symbol raises
    # AttributeError, which load_library reports the same way.
    abi_fn = lib.uncompute_kernel_abi_version
    abi_fn.argtypes = []
    abi_fn.restype = ctypes.c_long
    got = abi_fn()
    if got != KERNEL_ABI:
        raise ImportError(
            f"uncompute kernel library at {LIB_PATH} is stale (ABI {got}, "
            f"expected {KERNEL_ABI}); rebuild it with: "
            "make -B kernel"
        )
    lib.kern_create.argtypes = [_I64, ctypes.c_char_p]
    lib.kern_create.restype = ctypes.c_void_p
    lib.kern_free.argtypes = [ctypes.c_void_p]
    lib.kern_free.restype = None
    lib.kern_set_lane0.argtypes = [ctypes.c_void_p, _I64]
    lib.kern_set_lane0.restype = None
    lib.kern_set_live.argtypes = [ctypes.c_void_p, P(_U64)]
    lib.kern_set_live.restype = None
    for name in ("kern_n_words", "kern_alloc_qubits", "kern_reg_event_count"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = _I64
    lib.kern_counters.argtypes = [ctypes.c_void_p, P(_U64), P(_U64)]
    lib.kern_counters.restype = None
    lib.kern_read_qubit.argtypes = [ctypes.c_void_p, _U64, P(_U64)]
    lib.kern_read_qubit.restype = None
    lib.kern_read_phase.argtypes = [ctypes.c_void_p, P(_U64)]
    lib.kern_read_phase.restype = None
    lib.kern_reg_events.argtypes = [ctypes.c_void_p]
    lib.kern_reg_events.restype = P(_U64)
    lib.kern_load_register.argtypes = [
        ctypes.c_void_p,
        P(ctypes.c_uint32),
        _I64,
        ctypes.c_char_p,
        _I64,
        ctypes.c_char_p,
        _I64,
        _I64,
    ]
    lib.kern_load_register.restype = _I64
    lib.kern_read_register.argtypes = [
        ctypes.c_void_p,
        P(ctypes.c_uint32),
        _I64,
        ctypes.c_char_p,
        _I64,
        _I64,
        _I64,
    ]
    lib.kern_read_register.restype = None
    # stream interpreter
    lib.kern_compact_create.argtypes = [ctypes.c_void_p]
    lib.kern_compact_create.restype = ctypes.c_void_p
    lib.kern_compact_free.argtypes = [ctypes.c_void_p]
    lib.kern_compact_free.restype = None
    lib.kern_compact_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _I64]
    lib.kern_compact_start.restype = _I64
    lib.kern_compact_resume.argtypes = [ctypes.c_void_p]
    lib.kern_compact_resume.restype = _I64
    for name in (
        "kern_compact_error_code",
        "kern_compact_error_op_index",
        "kern_compact_ops",
        "kern_compact_num_qubits",
        "kern_compact_num_outs",
        "kern_compact_num_bits",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = _I64
    lib.kern_compact_error_msg.argtypes = [ctypes.c_void_p]
    lib.kern_compact_error_msg.restype = ctypes.c_char_p
    lib.kern_compact_fail_op.argtypes = [ctypes.c_void_p, P(_U64)]
    lib.kern_compact_fail_op.restype = None
    # uncompute.Simulator fast path (kern_sim_*; consumed by uncompute._simfast)
    lib.kern_sim_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _I64, _I64, ctypes.c_char_p]
    lib.kern_sim_init.restype = _I64
    lib.kern_sim_load_state.argtypes = [
        ctypes.c_void_p,
        P(_U64),
        _I64,
        P(_U64),
        _I64,
        P(_U64),
        _I64,
        P(_U64),
        _U64,
        _U64,
        _U64,
        _U64,
    ]
    lib.kern_sim_load_state.restype = _I64
    lib.kern_sim_load_ledger.argtypes = [ctypes.c_void_p, P(_U64), _I64, P(_U64), P(_U64), _I64]
    lib.kern_sim_load_ledger.restype = _I64
    lib.kern_sim_apply.argtypes = [ctypes.c_void_p, P(_U64), _I64]
    lib.kern_sim_apply.restype = _I64
    lib.kern_sim_run_rec.argtypes = [ctypes.c_void_p, P(_U64), _I64, P(_U64), _I64]
    lib.kern_sim_run_rec.restype = _I64
    lib.kern_sim_sizes.argtypes = [ctypes.c_void_p, P(_U64)]
    lib.kern_sim_sizes.restype = None
    lib.kern_sim_dump_state.argtypes = [ctypes.c_void_p, P(_U64), P(_U64), P(_U64), P(_U64)]
    lib.kern_sim_dump_state.restype = None
    lib.kern_sim_dump_ledger.argtypes = [ctypes.c_void_p, P(_U64), P(_U64), P(_U64)]
    lib.kern_sim_dump_ledger.restype = None
    # producer-side FLAT tape encoder (consumed by uncompute._tapefast)
    lib.kern_tape_encode_flat.argtypes = [
        P(ctypes.c_uint32),
        _I64,
        _I64,
        P(_I64),
        P(ctypes.c_ubyte),
        _I64,
        P(_I64),
    ]
    lib.kern_tape_encode_flat.restype = _I64


def load_library():
    """Load (and cache) the kernel shared library; ImportError if absent.

    Binding is strict: every symbol in :func:`_bind` must resolve, so a stale
    dylib fails the load cleanly here (``kernel_available()`` then reports
    False and evaluation falls back to the normative Python reference)
    instead of surfacing mid-run.
    """
    global _lib, _lib_err
    if _lib is not None:
        return _lib
    if _lib_err is not None:
        raise ImportError(_lib_err)
    if not Path(LIB_PATH).exists():
        _lib_err = f"uncompute kernel library not found at {LIB_PATH}; build it with: make kernel"
        raise ImportError(_lib_err)
    try:
        lib = ctypes.CDLL(LIB_PATH)
        _bind(lib)
    except OSError as e:
        _lib_err = f"could not load uncompute kernel {LIB_PATH}: {e}"
        raise ImportError(_lib_err) from e
    except AttributeError as e:
        _lib_err = f"uncompute kernel library at {LIB_PATH} is stale ({e}); rebuild it with: make -B kernel"
        raise ImportError(_lib_err) from e
    except ImportError as e:
        # _bind's ABI handshake already produced the operator message;
        # cache it so repeat loads fail identically without re-binding.
        _lib_err = str(e)
        raise
    _lib = lib
    return lib


def kernel_available() -> bool:
    try:
        load_library()
        return True
    except ImportError:
        return False


class KernelExecutor:
    """One native evaluation context: wide-lane wires + counters + RNG.

    ``live_mask`` (default: all ``n_shots`` lanes) implements dead-lane
    batches (protocol.py's dead-lane convention): it replaces the kernel's
    all-live lane mask via ``kern_set_live``, so dead lanes never bill
    a gate, take a state update, or receive a measurement store — the
    KernelExecutor twin of ``LaneSim(live_mask=...)``.
    """

    def __init__(self, n_shots: int, seed_rng: bytes, lane0: int = 0, live_mask: int | None = None):
        if len(seed_rng) != 32:
            raise ValueError("seed_rng must be 32 bytes")
        if lane0 % 8:
            raise ValueError("lane0 must be a multiple of 8")
        self._lib = load_library()
        self._ctx = self._lib.kern_create(n_shots, seed_rng)
        if not self._ctx:
            raise MemoryError("kern_create failed")
        if lane0:
            self._lib.kern_set_lane0(self._ctx, lane0)
        self.n_shots = n_shots
        self.n_words = self._lib.kern_n_words(self._ctx)
        self._word_buf = (_U64 * self.n_words)()
        if live_mask is not None:
            words = (_U64 * self.n_words).from_buffer_copy(
                live_mask.to_bytes(self.n_words * 8, "little")
            )
            self._lib.kern_set_live(self._ctx, words)

    def close(self) -> None:
        if self._ctx:
            self._lib.kern_free(self._ctx)
            self._ctx = None

    def __del__(self):  # best-effort; close() is called explicitly
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110  (__del__ must never raise)
            pass

    def drain_reg_events(self, registers: list[tuple]) -> None:
        """Fold REGS-segment register events into ``registers`` exactly like
        the reference census (compact_reader REGS handling): registers are
        ``(reg_kind, wires)`` pairs — WKIND_QUBIT wires are qubit ids,
        WKIND_BIT wires are bit ids. Events are quadruples
        ``(rt, type, wire, reg_kind)``; the declare event (type 0) fixes the
        register's kind, appends (type 1) grow its wire list.
        """
        cnt = self._lib.kern_reg_event_count(self._ctx)
        if not cnt:
            return
        ev = self._lib.kern_reg_events(self._ctx)
        for e in range(cnt):
            rt = ev[4 * e]
            typ = ev[4 * e + 1]
            wire = ev[4 * e + 2]
            kind = ev[4 * e + 3]
            while len(registers) <= rt:
                registers.append((WKIND_QUBIT, []))
            if typ == 0:
                registers[rt] = (kind, registers[rt][1])
            else:
                registers[rt][1].append(wire)

    # -- counters / lane I/O -----------------------------------------------------
    def alloc_qubits(self) -> int:
        return self._lib.kern_alloc_qubits(self._ctx)

    def counters(self):
        tof = _U64()
        cliff = _U64()
        self._lib.kern_counters(self._ctx, ctypes.byref(tof), ctypes.byref(cliff))
        return tof.value, cliff.value

    def _buf_to_int(self) -> int:
        return int.from_bytes(bytes(self._word_buf), "little")

    def read_qubit_lanes(self, q: int) -> int:
        self._lib.kern_read_qubit(self._ctx, q, self._word_buf)
        return self._buf_to_int()

    def read_phase_lanes(self) -> int:
        self._lib.kern_read_phase(self._ctx, self._word_buf)
        return self._buf_to_int()

    # -- bulk register I/O (normative twin: util._set_register/_get_register) ----
    @staticmethod
    def _wire_ids(wires):
        return (ctypes.c_uint32 * len(wires))(*wires)

    @staticmethod
    def _row_code(nw: int) -> str:
        """struct code for one packed row of an ``nw``-wide register
        (``nw`` <= 64). Rows may be wider than ``ceil(nw/8)`` bytes: the C
        side takes ``row_bytes`` as a caller-chosen stride and reads exactly
        ``n_wires`` bits per row, so widening to the struct sizes is
        bit-identical.
        """
        return "B" if nw <= 8 else ("H" if nw <= 16 else ("I" if nw <= 32 else "Q"))

    def load_register(self, reg, values) -> None:
        """Bulk-load per-shot register values (``values[s]`` int or None).

        ``reg`` is a ``(reg_kind, wires)`` pair: Qubit registers load into
        lane state, Bit registers into bit space (the C ``is_bit`` routing —
        util._set_register's normative semantics).

        Fast path (every shot carries a value, register fits a struct row):
        one count-prefixed ``struct.pack`` — C-speed, vs a per-shot
        ``int.to_bytes`` loop that dominated batch wall time at
        O(shots x registers). A ``None`` hole (dead lane / missing input)
        raises TypeError out of the pack and falls back to the per-shot
        loop, which is also the path for >64-wire registers (there
        ``to_bytes`` outruns any chunked scheme).
        """
        kind, wires = reg
        is_bit = 1 if kind == WKIND_BIT else 0
        nw = len(wires)
        n = len(values)
        if nw == 0 or n == 0:
            return
        mask = (1 << nw) - 1
        if nw <= 64:
            code = self._row_code(nw)
            try:
                packed = struct.pack(f"<{n}{code}", *((v & mask) for v in values))
            except TypeError:
                packed = None  # None hole (or junk value): per-shot path
            if packed is not None:
                full, rem = divmod(n, 8)
                all_valid = b"\xff" * full + (bytes([(1 << rem) - 1]) if rem else b"")
                rc = self._lib.kern_load_register(
                    self._ctx,
                    self._wire_ids(wires),
                    nw,
                    packed,
                    struct.calcsize(code),
                    all_valid,
                    n,
                    is_bit,
                )
                if rc != 0:
                    raise MemoryError("kern_load_register failed")
                return
        row_bytes = (nw + 7) // 8
        rows = bytearray(n * row_bytes)
        valid = bytearray((n + 7) // 8)
        any_valid = False
        for s, v in enumerate(values):
            if v is None:
                continue
            any_valid = True
            valid[s >> 3] |= 1 << (s & 7)
            rows[s * row_bytes : (s + 1) * row_bytes] = (v & mask).to_bytes(row_bytes, "little")
        if not any_valid:
            return
        rc = self._lib.kern_load_register(
            self._ctx, self._wire_ids(wires), nw, bytes(rows), row_bytes, bytes(valid), n, is_bit
        )
        if rc != 0:
            raise MemoryError("kern_load_register failed")

    def read_register_values(self, reg) -> list[int]:
        """Bulk-read one value per shot from a ``(reg_kind, wires)`` register
        (util._get_register's normative semantics; the verdict path only
        ever reads Qubit registers — Bit registers are input-only).

        Registers up to 64 wires unpack in one count-prefixed
        ``struct.unpack`` (same widened-stride convention as
        :meth:`load_register`); wider registers keep the per-shot
        ``int.from_bytes`` loop.
        """
        kind, wires = reg
        is_bit = 1 if kind == WKIND_BIT else 0
        n = self.n_shots
        nw = len(wires)
        if nw == 0:
            return [0] * n
        if nw <= 64:
            code = self._row_code(nw)
            row_bytes = struct.calcsize(code)
            out = ctypes.create_string_buffer(n * row_bytes)  # zero-init
            self._lib.kern_read_register(
                self._ctx, self._wire_ids(wires), nw, out, row_bytes, n, is_bit
            )
            return list(struct.unpack(f"<{n}{code}", out.raw))
        row_bytes = (nw + 7) // 8
        out = ctypes.create_string_buffer(n * row_bytes)  # zero-initialized
        self._lib.kern_read_register(
            self._ctx, self._wire_ids(wires), nw, out, row_bytes, n, is_bit
        )
        raw = out.raw
        return [
            int.from_bytes(raw[s * row_bytes : (s + 1) * row_bytes], "little") for s in range(n)
        ]


class _Snapshot:
    """End-of-stream state in the shape the shared verdict extraction
    consumes: phase/qubits lane ints plus the pre-read bulk register values
    (the native context is freed before verdicts run). Outcome lanes are
    never snapshotted: verdicts consume registers (qubits) and phase only.
    """

    def __init__(self, phase: int, qubits: list[int], regvals: dict[int, list[int]]):
        self.phase = phase
        self.qubits = qubits
        self._regvals = regvals

    def read_register_values(self, reg) -> list[int]:
        return self._regvals[id(reg)]


def _load_inputs(ex: KernelExecutor, registers, tests) -> None:
    """Bulk twin of the reference input loading (shot-outer, register-inner):
    register-outer with full per-shot overwrite gives the identical
    last-write-wins result on aliased wires. ``registers`` entries are
    ``(reg_kind, wires)`` pairs; :meth:`KernelExecutor.load_register` routes
    Qubit registers into lane state and Bit registers into bit space.
    ``None`` test entries are dead lanes and load nothing (their lanes stay
    out of the valid mask).

    Common case (no dead lanes, every shot the same input count — checked
    explicitly, since ``zip`` would silently truncate ragged rows): one
    C-speed ``zip(*)`` transpose instead of one O(shots) column-building
    pass per register.
    """
    live = [tc for tc in tests if tc is not None]
    if not live:
        return
    n_inputs = max(len(tc.inputs) for tc in live)
    n_regs = min(n_inputs, len(registers))
    if len(live) == len(tests) and all(len(tc.inputs) == n_inputs for tc in tests):
        cols = list(zip(*(tc.inputs for tc in tests))) if n_inputs else []
        for ri in range(n_regs):
            ex.load_register(registers[ri], cols[ri])
        return
    for ri in range(n_regs):
        values = [
            None if tc is None else (tc.inputs[ri] if ri < len(tc.inputs) else None) for tc in tests
        ]
        ex.load_register(registers[ri], values)


def run_compact_kernel_batch(ch, stream, tests, seed_rng: bytes, lane0: int = 0):
    """Kernel-backed twin of ``compact_reader.run_compact_batch`` (the
    dispatcher has already consumed the magic).

    The C interpreter owns the segment walk, direct loop interpretation, and
    every per-expanded-op check; register inputs are loaded at the
    reference's point (after the REGS/DATA prelude, before the first
    executable segment) via the bulk loader. Structural rejections raise
    ``CompactParseError`` with the reference code and message text;
    validation failures come back as a failed report. The reason text of a
    distinctness failure comes from :func:`uncompute.ops.op_violation` on the
    resolved op, so every failure text matches the reference path.
    """
    # Positional tests: None entries are dead lanes (protocol.py's dead-lane
    # convention); n_shots and the stat denominators count live lanes only.
    r, n, live_mask, sim_live = begin_batch(tests)

    # The C DATA tables alias the body buffer, so it must outlive the run
    # (both paths are held until after kern_compact_free). Regular files are
    # MMAP'd rather than read: RAM drops from O(stream) to O(working set)
    # with sequential readahead, and parallel shard workers mapping the same
    # spool file share ONE page-cache copy (the largest task streams are
    # multi-GB). ACCESS_COPY keeps the buffer writable-typed for
    # from_buffer without ever writing it (the kernel treats the body as
    # const), so no private pages materialize. Pipes/FIFOs (the unsharded
    # circuit path) fall back to a full read — bit-identical
    # either way, this is buffering strategy only.
    data, mm = None, None
    try:
        if os.environ.get("UNCOMPUTE_NO_MMAP") == "1":
            raise io.UnsupportedOperation("mmap disabled by UNCOMPUTE_NO_MMAP")
        st = os.fstat(
            stream.fileno()
        )  # AttributeError → fallback: tee/wrapper streams have no fileno
        if _stat.S_ISREG(st.st_mode) and st.st_size > 0:
            pos = stream.tell()
            if 0 <= pos <= st.st_size:
                mm = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
                if hasattr(mm, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
                    try:
                        mm.madvise(mmap.MADV_SEQUENTIAL)
                    except (OSError, ValueError):
                        pass
                data = (ctypes.c_char * (st.st_size - pos)).from_buffer(mm, pos)
                stream.seek(st.st_size)  # mimic read-to-EOF
    except (OSError, ValueError, io.UnsupportedOperation, AttributeError):
        data, mm = None, None
    if data is None:
        chunks = []
        while True:
            got = stream.read(1 << 20)
            if not got:
                break
            chunks.append(got)
        data = b"".join(chunks)

    ex = KernelExecutor(n, seed_rng, lane0, live_mask=sim_live)
    lib = ex._lib
    cctx = lib.kern_compact_create(ex._ctx)
    if not cctx:
        ex.close()
        raise MemoryError("kern_compact_create failed")
    try:
        registers: list[tuple] = []
        status = lib.kern_compact_start(cctx, data, len(data))
        ex.drain_reg_events(registers)
        if status in (CST_PAUSED, CST_DONE):
            # Registers are declared up front (REGS), so inputs load before
            # the first gate — compact_reader._seg_regs behavior.
            _load_inputs(ex, registers, tests)
            if status == CST_PAUSED:
                status = lib.kern_compact_resume(cctx)

        if status != CST_DONE:
            code = lib.kern_compact_error_code(cctx)
            msg = lib.kern_compact_error_msg(cctx).decode("utf-8")
            idx = lib.kern_compact_error_op_index(cctx)
            if code in _CPARSE_CODES:
                raise CompactParseError(msg, _CPARSE_CODES[code])
            if code in _CVALIDATE_CODES:
                if code == CERR_ALIASING:
                    buf = (_U64 * 8)()
                    lib.kern_compact_fail_op(cctx, buf)
                    kind, qa, qb, qt, ot, oc, ct, cc = (int(x) for x in buf)
                    reason = op_violation((kind, qa, qb, qt, ot, oc, NO_REG, ct, cc))
                    if reason is None:
                        raise RuntimeError(
                            "the C kernel rejected an op for repeated operands "
                            "that the reference accepts (kernel and reference "
                            "disagree)"
                        )
                    msg = reason
                return fail_batch(
                    r,
                    registers,
                    _CVALIDATE_CODES[code],
                    msg,
                    None if idx < 0 else idx,
                    ops_len=lib.kern_compact_ops(cctx),
                    num_qubits=lib.kern_compact_num_qubits(cctx),
                    num_outs=lib.kern_compact_num_outs(cctx),
                    num_bits=lib.kern_compact_num_bits(cctx),
                )
            raise RuntimeError(f"uncompute kernel compact error code {code}: {msg}")

        ops_len = lib.kern_compact_ops(cctx)
        num_qubits = lib.kern_compact_num_qubits(cctx)
        num_outs = lib.kern_compact_num_outs(cctx)
        num_bits = lib.kern_compact_num_bits(cctx)
        tot_tof, tot_cliff = ex.counters()

        # Bulk-read the verdict-consumed registers' per-shot values while the
        # native context is alive; verdict extraction consumes them from the
        # snapshot. Only ``needed_registers`` are read back (the set is
        # computed once and forwarded to finish_batch, so extraction queries
        # exactly these) — reading every declared register is
        # O(shots x registers) Python for values nothing consumes. The bounds
        # guard covers schema-invalid streams (fewer registers than the
        # grader expects): there the pre-read is dead — finish_batch fails
        # the schema before verdicts run. Bit registers are never pre-read:
        # they are input-only, so ``needed`` cannot legitimately contain one
        # — if a grader puts an expected output on a Bit register anyway,
        # extract_batch_state raises the coded grader error before it would
        # consult the snapshot.
        needed = needed_registers(tests)
        regvals = {
            id(registers[ri]): ex.read_register_values(registers[ri])
            for ri in needed
            if ri < len(registers) and registers[ri][0] != WKIND_BIT
        }
        snap = _Snapshot(
            phase=ex.read_phase_lanes(),
            qubits=[ex.read_qubit_lanes(q) for q in range(ex.alloc_qubits())],
            regvals=regvals,
        )
    finally:
        lib.kern_compact_free(cctx)
        ex.close()
        if mm is not None:
            data = None  # drop the exported buffer ref
            mm.close()

    return finish_batch(
        ch,
        r,
        tests,
        registers,
        snap,
        live_mask,
        ops_len=ops_len,
        num_qubits=num_qubits,
        num_outs=num_outs,
        tot_tof=tot_tof,
        tot_cliff=tot_cliff,
        num_bits=num_bits,
        needed=needed,
    )
