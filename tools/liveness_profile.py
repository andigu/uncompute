#!/usr/bin/env python3
"""Register-liveness profiler for the certified circuits.

Runs the discovered circuit of a task (the first existing entry of
``PRIMARY_SOURCES`` in tools/eval.py, e.g.
``tasks/<name>/circuits/discovered/circuit.py``) and writes ``liveness.json``
beside it. The file describes the register liveness of the circuit: a
per-phase Toffoli and peak-qubit breakdown plus the exact liveness skyline.

Phase marks may be hierarchical (``stage/sub``, e.g. ``squarer/a0g1``).
Circuits mark phases at the finest useful level, and consumers coarsen them
by name prefix (``render_liveness.py --depth``).

The circuit calls ``Builder.set_phase`` itself, so the profiler runs the
circuit file unchanged.  The Builder records the whole wire axis
(``Builder.wire_log``: phase marks, alloc/free, ownership
events, mirror brackets — no instrumentation, no patching), and
``uncompute.builder.LAST_EMITTED`` names the emitting instance; the
profiler reconstructs spans/events from that record.  Spans emitted
inside a ``Builder.mirror()`` call are grouped as ``"mirror"`` (with the
reflected ``un-*`` names the mirror labels itself).  Both the profiled and
any pristine run capture their stream via ``UNCOMPUTE_OPS_PATH`` (the
Builder's fixed transport).

Verification before anything is written:
  1. byte-identity — the profiled run's op stream equals the reference
     stream byte for byte. The reference is the circuit's certified stream
     ``<circuit-stem>.ops.xz`` (lzma-decompressed), else a fresh subprocess
     run of the discovered circuit. The stream's SHA-256 must also
     equal the ``h_ops`` of the results.jsonl row.
  2. cost parity — the profiled totals equal the Toffoli count and qubits
     of the results.jsonl row.

The **standalone variant** flow (``--profile-dir``, e.g.
``--profile-dir circuits/haner-piecewise`` for a baseline) profiles a
certified circuit other than the discovered circuit, so check 1 does not
apply. Instead the evaluator certifies the produced stream (its
draw is seeded by the stream's own hash, ``h_ops``), and the expected metrics
come from ``--expect-toffoli``/``--expect-qubits`` or, by default, the
circuit's ``meta.json`` pin (``uncompute.testing.pinned_cost``).
Standalone variants may churn hundreds of thousands of scratch ancillas, so
their allocation intervals are max-pooled to ``INTERVAL_CAP`` (untagged
same-slot episodes merge across small idle gaps; tagged and persistent
intervals are never pooled).

Usage:
    uv run python tools/liveness_profile.py tasks/<name>
    uv run python tools/liveness_profile.py tasks/<name> --profile-dir circuits/<variant>

Large flat streams replay much faster with the optional C replay kernel:
``make replay-kernel``.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

import click
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from alias_map import sha16
from eval import PRIMARY_SOURCES

from uncompute.ops import NO_BIT, QUBIT_FIELDS, ZEROING_KINDS, OpKind, gate_class
from uncompute.ops import WRITE_FIELDS as OP_WRITE_FIELDS
from uncompute.testing import pinned_cost

SKYLINE_CAP = 4000  # max change-points kept (max-pooled above this)
INTERVAL_CAP = 4000  # max intervals kept for standalone variants (pooled above)
XZ_THRESHOLD = 50_000_000  # payloads above this many bytes track as .json.xz
STANDALONE_EVAL_TIMEOUT = 7200.0  # evaluator budget for big baseline streams


def _billed_toffoli(op):
    """Toffoli test at PROMOTED arity (ops.py's single authority):
    CCX/CCZ, plus bit-conditioned CX/CZ — a bit condition narrows the
    per-shot action, never the billing (slot 8 = the cc operand).
    """
    return gate_class(op[0], op[8] != NO_BIT) == "toffoli"


# Display names for liveness.json phase "counts" keys.
KIND_NAMES = {int(k): k.name.lower() for k in OpKind}


def fail(msg: str) -> NoReturn:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def run_annotated(path: Path, cwd: Path):
    """Run the variant as the program it is and return the Builder that
    emitted its stream.

    No instrumentation, no patching: the vanilla Builder natively
    records the whole wire axis (``Builder.wire_log`` — phase marks,
    alloc/free with position tokens and live counts, ownership events,
    mirror brackets) and ``uncompute.builder.LAST_EMITTED`` names the
    instance that last emitted, so a __main__-run variant needs no
    cooperation beyond emitting.
    """
    from uncompute import builder as _bmod

    old = Path.cwd()
    os.chdir(cwd)
    try:
        # "__main__": variants that emit under a main guard (the documented
        # circuit convention) must emit here too — the profiler runs
        # the variant as the program it is.
        runpy.run_path(str(path), run_name="__main__")
    finally:
        os.chdir(old)
    return _bmod.LAST_EMITTED


# --------------------------------------------------------------------------
# shared activity mechanics
# --------------------------------------------------------------------------


def _state_segments(phases, ep_born, touch_parts, tl_parts, queries):
    """Merge [phase_start, phase_end, state] runs per queried allocation
    episode, over phases ep_born[e]..k1 of each. touch_parts = arrays
    of (episode, phase) gate-operand touches; tl_parts = arrays of
    (episode, op index, is_zero) timeline entries in chronological order,
    each episode's first entry its allocation; queries = [(episode,
    k1)]. State 2 = active, 1 = idle/may hold data, 0 = |0> at the phase
    start (the episode's last entry at or before it, else its first).
    """
    S = len(phases) + 1
    i64 = np.int64
    te = np.concatenate([np.zeros(0, dtype=i64)] + [p[0] for p in touch_parts])
    ts = np.concatenate([np.zeros(0, dtype=i64)] + [p[1] for p in touch_parts])
    tkey = np.sort(te * S + ts)
    le = np.concatenate([np.zeros(0, dtype=i64)] + [p[0] for p in tl_parts])
    lx = np.concatenate([np.zeros(0, dtype=i64)] + [p[1] for p in tl_parts])
    lz = np.concatenate([np.zeros(0, dtype=np.int8)] + [p[2] for p in tl_parts])
    order = np.argsort(le, kind="stable")  # per episode, chronological
    le, lx, lz = le[order], lx[order], lz[order]
    first = np.searchsorted(le, np.arange(len(ep_born)))
    starts = np.array([p["span"][0] for p in phases], dtype=i64)
    big = int(max(lx.max(initial=0), starts.max(initial=0))) + 2
    tl_key = le * big + lx

    qe = np.array([e for e, _ in queries], dtype=i64)
    born = np.asarray(ep_born, dtype=i64)[qe] if len(qe) else qe
    k1 = np.array([k for _, k in queries], dtype=i64)
    lens = np.maximum(k1 - born + 1, 0)
    off = np.zeros(len(queries) + 1, dtype=i64)
    np.cumsum(lens, out=off[1:])
    L = int(off[-1])
    qi = np.repeat(np.arange(len(queries)), lens)
    k = born[qi] + np.arange(L) - off[qi]
    e = qe[qi]
    v = e * S + k
    j = np.minimum(np.searchsorted(tkey, v), max(len(tkey) - 1, 0))
    active = (tkey[j] == v) if len(tkey) else np.zeros(L, dtype=bool)
    ti = np.maximum(np.searchsorted(tl_key, e * big + starts[k], side="right") - 1, first[e])
    st = np.where(active, 2, np.where(lz[ti] != 0, 0, 1))
    brk = np.flatnonzero((qi[1:] != qi[:-1]) | (st[1:] != st[:-1])) + 1
    rs = np.concatenate([[0], brk]).astype(np.intp) if L else brk
    re = np.concatenate([brk, [L]]).astype(np.intp) if L else brk
    runs = np.stack([k[rs], k[re - 1] + 1, st[rs]], axis=1).tolist()
    roff = np.zeros(len(queries) + 1, dtype=i64)
    np.cumsum(np.bincount(qi[rs], minlength=len(queries)), out=roff[1:])
    roff = roff.tolist()
    return [runs[roff[i] : roff[i + 1]] for i in range(len(queries))]


def activity_section(phases: list[dict], intervals: list[dict]) -> dict:
    """Build the liveness.json ``activity`` section from the state_segments
    the replay attached to its intervals: per-phase [zero, idle, active]
    counts (each SLOT counted once per phase, at the max state across its
    episodes — a churning slot can host several in one span) and the
    Toffoli-weighted qubit-time fractions.
    """
    nq = max(iv["q"] for iv in intervals) + 1
    grid = bytearray(b"\xff") * (len(phases) * nq)  # 255 = not allocated
    for iv in intervals:
        base = iv["q"]
        for a, b, st in iv["state_segments"]:
            for k in range(a, b):
                gi = k * nq + base
                if grid[gi] == 255 or st > grid[gi]:
                    grid[gi] = st
    per_phase = []
    for k in range(len(phases)):
        row = grid[k * nq : (k + 1) * nq]
        per_phase.append([row.count(0), row.count(1), row.count(2)])
    tot = [0.0, 0.0, 0.0]
    for k, p in enumerate(phases):
        for st in range(3):
            tot[st] += per_phase[k][st] * p["toffoli"]
    denom = max(sum(tot), 1.0)
    return {
        "states": ["zero", "idle", "active"],
        "semantics": {
            "active": "operand of >=1 gate in the span",
            "idle": "untouched; not known empty (may hold data)",
            "zero": "|0>: op-semantics-forced (alloc, and HMR/R "
            "demolition; only X/CX/CCX/Swap targets write "
            "values); never sampled or inferred",
            "io": "declared-register wires hold evaluator-loaded data "
            "from birth; their allocation never marks them zero",
            "coords": "interval state_segments are [phase_start, "
            "phase_end, state] in fine phase indices",
        },
        "per_phase": per_phase,
        "derived": {
            "active_fraction": round(tot[2] / denom, 4),
            "idle_fraction": round(tot[1] / denom, 4),
            "zero_fraction": round(tot[0] / denom, 4),
        },
    }


# --------------------------------------------------------------------------
# Builder mode
# --------------------------------------------------------------------------
#
# The replay walks the RECORDED program (flat items + closed Loop records),
# never the expansion: loop-form circuits record ~10^6 items that expand to
# 10^9+ executed ops, so every coordinate is computed in closed form via
# compact.item_count. Conventions:
#   * The x-axis is LOOP-COMPRESSED toffoli: a repeat(n) occupies the width
#     of one representative iteration (the balanced-allocation rule makes
#     every iteration's liveness identical, so the shown iteration IS each
#     iteration); the region and trip count land in ``loops`` for the
#     renderer's x-annotation. Flat programs degenerate to plain executed
#     toffoli with no ``loops`` section.
#   * MASS is never compressed: per-phase toffoli/clifford/counts, area and
#     totals carry full multiplicities (recording-order attribution — the
#     Builder's own semantics, cross-checked against its telemetry).
#   * Activity touches inside loop bodies expand strided operands over the
#     full enclosing loop box, honoring the record's data gate (gated-off
#     iterations touch nothing): every wire an op can address is credited,
#     once per record. Touched-id sets are memoized per (strides, counts,
#     gate) shape, so identical bodies re-emitted across slots resolve from
#     cache. Episodes allocated in a body carry an ``episodes``
#     multiplicity.
#   * Declared qubit-register wires (the schema io) hold evaluator-loaded
#     data from birth: their allocation event does NOT mark them |0>, so
#     "zero" is never claimed for an input the record cannot see.


def _leaf7(item):
    """Normalize a recorded leaf (flat 9-int tuple or compact.OpRec) to the
    flat 9-slot spelling with iteration-0 operand bases, so QUBIT_FIELDS /
    WRITE_FIELDS / ZEROING_KINDS indexing applies uniformly.
    """
    if isinstance(item, tuple):
        return item
    from uncompute.compact import _FIELD_SLOT, KIND_OPERANDS
    from uncompute.ops import NO_BIT, NO_OUT, NO_QUBIT, NO_REG

    slots = [item.kind, NO_QUBIT, NO_QUBIT, NO_QUBIT, NO_OUT, NO_OUT, NO_REG, NO_BIT, NO_BIT]
    for name, operand in zip(KIND_OPERANDS[item.kind], item.operands):
        slots[_FIELD_SLOT[name]] = operand[0]
    if item.oc is not None:
        slots[5] = item.oc[0]
    return tuple(slots)


def _slot_operands(item):
    """Slot -> (base, strides) map for an OpRec's qubit-capable operands."""
    from uncompute.compact import _FIELD_SLOT, KIND_OPERANDS

    m = {}
    for name, (base, strs) in zip(KIND_OPERANDS[item.kind], item.operands):
        m[_FIELD_SLOT[name]] = (base, tuple(strs))
    return m


_TOUCH_MEMO: dict = {}
_BITS_NP: dict = {}


def _touched_offsets(strides, counts, gate, tables):
    """Sorted unique operand offsets (Σ strides_j·i_j) over the loop box,
    keeping only iterations whose data-gate bit is 1. Memoized on the
    (strides, counts, gate) shape — loop-form bodies repeat identically
    across call sites, only their bases shift.
    """
    gk = None
    if gate is not None:
        gk = (gate[0], gate[1], tuple(gate[2]))
    key = (tuple(strides), counts, gk)
    hit = _TOUCH_MEMO.get(key)
    if hit is not None:
        return hit
    box = 1
    for c in counts:
        box *= c
    if box > (1 << 26):
        fail(f"activity expansion box too large: {counts}")
    idx = np.indices(counts, dtype=np.int64).reshape(len(counts), -1)
    off = np.zeros(idx.shape[1], dtype=np.int64)
    for s, row in zip(strides, idx):
        if s:
            off += s * row
    if gate is not None:
        tid, goff, gstr = gate
        bits = _BITS_NP.get(tid)
        if bits is None or len(bits) != len(tables[tid]):
            bits = np.asarray(tables[tid], dtype=np.uint8)
            _BITS_NP[tid] = bits
        gi = np.full(idx.shape[1], goff, dtype=np.int64)
        for s, row in zip(gstr, idx):
            if s:
                gi += s * row
        off = off[bits[gi] != 0]
    out = [int(v) for v in np.unique(off)]
    _TOUCH_MEMO[key] = out
    return out


def _leaf_field_ids(item, op, f, counts, tables):
    """Every wire id operand slot ``f`` of this leaf can address over its
    enclosing loop box (post-gating). Flat tuples and constant operands
    yield their single id; strided operands expand via the memoized box
    walk. The caller guards on the record's post-gating multiplicity, so
    a fully gated-off record touches nothing.
    """
    if isinstance(item, tuple):
        return (op[f],)
    so = _slot_operands(item).get(f)
    if so is None:
        return ()
    base, strs = so
    if not any(strs):
        return (base,)
    return [base + o for o in _touched_offsets(strs, tuple(counts), item.gate, tables)]


def _walk_recorded(items, tables, path=(), start=0):
    """Document-order stream over the closed recording.

    Yields, with lexicographic position keys comparable to event tokens
    (on equal prefixes an event sorts first — its token names the count of
    items already recorded, i.e. the position BEFORE the item of that index):
      ("leaf",  key, item)       -- a recorded op (flat tuple or OpRec)
      ("enter", key, count)      -- a Loop opens (count = its trip count)
      ("exit",  key, loop_item)  -- the Loop closes (full span squared up
                                    by the consumer via item_count).
    ``start`` numbers the first of ``items`` (a sub-walk of one record).
    """
    from uncompute.compact import Loop

    for i, item in enumerate(items, start):
        key = path + (i,)
        if isinstance(item, Loop):
            yield ("enter", key, item.count)
            yield from _walk_recorded(item.items, tables, key)
            yield ("exit", key, item)
        else:
            yield ("leaf", key, item)


def _event_before(token, key):
    """Return True when a log event with ``token`` precedes the stream item at
    ``key`` in recording order (equal prefix -> event first).
    """
    for a, b in zip(token, key):
        if a != b:
            return a < b
    return len(token) <= len(key)


# Flat-run replay: packed rows (the Builder's columnar _OpTape) replay in
# chunks. numpy bills the mass; the optional C kernel (tools/_replay_kernel.c,
# ``make replay-kernel``) records the activity. Without the kernel the rows
# replay per item, like loop records.
REPLAY_CHUNK = 1 << 20  # max rows per kernel chunk
REPLAY_SCALAR_RUN = 64  # shorter runs replay per item (call overhead)
_ROW_CC = 0x8000  # slot-0 bit: the cc field holds the sentinel
_EV_ALLOC, _EV_FREE, _EV_MARK = 1, 2, 3  # state-changing wire_log kinds
_EV_CODE = {"alloc": _EV_ALLOC, "free": _EV_FREE, "mark": _EV_MARK}
REPLAY_KERNEL_ABI = 1  # must match _replay_kernel.c
REPLAY_LIB_PATH = (
    Path(__file__)
    .resolve()
    .with_name("_replay_kernel.dylib" if sys.platform == "darwin" else "_replay_kernel.so")
)


# Per-row operand flags, indexed by a packed row's slot 0 (kind byte + the
# sentinel mask of slots 1..8; a masked slot holds no wire). Bits 0-2: slots
# 1-3 are qubit operands; bits 3-5: slots 1-3 are written; bit 6: slot 3 is
# the zeroing (demolition) target.
def _row_info():
    v = np.arange(1 << 16)
    k = v & 0xFF
    info = np.zeros(1 << 16, dtype=np.uint8)
    for f in (1, 2, 3):
        held = (v >> (7 + f)) & 1 == 0  # slot f holds a wire, not the sentinel
        for bit, table in ((1, QUBIT_FIELDS), (8, OP_WRITE_FIELDS)):
            by_kind = np.array([f in table.get(kk, ()) for kk in range(256)])
            info[held & by_kind[k]] |= bit << (f - 1)
    zeroing = np.array([kk in ZEROING_KINDS for kk in range(256)])
    info[((v >> 10) & 1 == 0) & zeroing[k]] |= 64
    return info


_ROW_INFO = _row_info()
# Billed flag per slot-0 value, filled lazily per value seen so an illegal
# conditioned kind raises in gate_class like the per-item path.
_BILLED = np.zeros(1 << 16, dtype=bool)
_BILLED_SEEN = np.zeros(1 << 16, dtype=bool)


def _billed_rows(s0):
    """Billed-Toffoli flag per packed row, from slot 0 alone."""
    for v in np.flatnonzero(np.bincount(s0, minlength=1 << 16)):
        if not _BILLED_SEEN[v]:
            _BILLED[v] = gate_class(int(v) & 0xFF, not v & _ROW_CC) == "toffoli"
            _BILLED_SEEN[v] = True
    return _BILLED[s0]


_KERNEL: list = []  # [ctypes function or None], loaded once


def _replay_kernel():
    """Load the C row replay, or None when the library is not built (or stale:
    older than its source, or another ABI).
    """
    if not _KERNEL:
        fn = None
        src = REPLAY_LIB_PATH.with_name("_replay_kernel.c")
        if REPLAY_LIB_PATH.exists() and (
            not src.exists() or REPLAY_LIB_PATH.stat().st_mtime >= src.stat().st_mtime
        ):
            import ctypes

            lib = ctypes.CDLL(str(REPLAY_LIB_PATH))
            lib.replay_abi.restype = ctypes.c_int64
            if lib.replay_abi() == REPLAY_KERNEL_ABI:
                fn = lib.replay_rows
                fn.restype = None
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int64,
                    ctypes.c_void_p,
                    ctypes.c_int64,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int64,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int64,
                    ctypes.c_void_p,
                    ctypes.c_int64,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
        if fn is None:
            print("note: C replay kernel not built (make replay-kernel); flat rows replay per item")
        _KERNEL.append(fn)
    return _KERNEL[0]


def _ptr(a):
    return a.ctypes.data


def builder_replay(writer) -> tuple[list[dict], dict]:
    """:func:`_replay` with the cyclic garbage collector paused: the replay
    allocates millions of acyclic containers (intervals, skyline points),
    and collection passes over them cost more than the replay itself.
    """
    import gc

    enabled = gc.isenabled()
    gc.disable()
    try:
        return _replay(writer)
    finally:
        if enabled:
            gc.enable()


def _replay(writer) -> tuple[list[dict], dict]:  # noqa: C901, PLR0915 - one-pass merge; splitting obscures it
    """One chronological merge of the profile log against a document-order
    walk of the closed recording. Phase spans (recording-order attribution,
    exactly the Builder's), skyline, intervals, activity touches, area and
    totals all come out of the single pass — O(recorded items), never
    O(expansion).

    With the C kernel, packed flat rows replay in chunks: the log events
    inside a chunk run first (their cursors from the chunk's billed prefix
    sums), numpy bills the mass per state segment, and the kernel walks
    the rows for the activity. Loop/OpRec records and short runs replay
    per item.
    """
    from math import prod

    from uncompute.compact import item_count

    log = writer.wire_log
    tables = {t.table_id: t.bits for t in writer._tables}
    _TOUCH_MEMO.clear()
    _BITS_NP.clear()

    # The schema io wires: declared Qubit registers (the one authority is
    # the Builder's own hoisted REGS payload). They hold evaluator-loaded
    # data from birth.
    from uncompute.compact import WKIND_BIT

    io_ids: set = set()
    for reg in writer._registers:
        if len(reg) == 2 and reg[0] == WKIND_BIT and isinstance(reg[1], tuple):
            continue  # Bit register: classical inputs, not qubit wires
        io_ids.update(reg)
    if io_ids and io_ids != set(range(len(io_ids))):
        fail(
            "declared qubit registers are not the wire-id prefix "
            f"0..{len(io_ids) - 1}; the io floor needs contiguous "
            "schema wires"
        )
    n_io = len(io_ids)

    spans: list[dict] = []
    skyline: list[list[int]] = []
    intervals: list[dict] = []
    loops_out: list[dict] = []
    open_q: dict = {}
    # Allocation episodes, by id in allocation order: acts maps an open
    # wire to its episode; ep_born is each episode's birth span; ep_of /
    # ep_state (1 = |0>) / ep_last (last touched span) are the arrays the
    # kernel shares. Touches and timeline entries (op index, is_zero) go
    # to chronological logs: Python lists, flushed as arrays around each
    # kernel chunk.
    tape = writer._items
    buf = getattr(tape, "_rows", None)
    kernel = _replay_kernel() if buf is not None and len(tape) else None
    acts: dict = {}
    ep_born: list = []
    if kernel is None:  # Python lists index faster per item
        ep_of: list | np.ndarray = []
        ep_state: list | np.ndarray = []
        ep_last: list | np.ndarray = []
    else:
        ep_of = np.full(64, -1, dtype=np.int64)
        ep_state = np.zeros(64, dtype=np.int8)
        ep_last = np.zeros(64, dtype=np.int64)
    touch_e: list = []
    touch_sp: list = []
    tl_e: list = []
    tl_x: list = []
    tl_z: list = []
    touch_parts: list = []
    tl_parts: list = []
    direct = True  # False while a kernel chunk defers episode effects
    deferred: list = []  # (interval, episode id, last span): segments at the end
    cur_phase = "init"
    mdepth = 0
    # LOOP-COMPRESSED geometry: the cursors advance by iteration-0
    # contributions only, so a repeat(n) occupies the width of ONE
    # representative iteration on the axis (its region + trip count land
    # in ``loops``). Mass (per-phase toffoli, area, totals) stays TRUE
    # executed values via per-leaf multiplicities.
    X = T = 0  # compressed cursors: recorded-op / toffoli axis
    live = last_ev_live = peak_all = 0
    area = tot_tof = 0
    counts_stack: list[int] = []  # trip counts of the open loops
    loop_ctx: list[list] = []  # per open loop: [T0, count]
    li = 0

    def _grow(a, n, fill):
        if n <= len(a):
            return a
        if isinstance(a, list):
            a.extend([fill] * (n - len(a)))
            return a
        b = np.full(max(n, 2 * len(a)), fill, dtype=a.dtype)
        b[: len(a)] = a
        return b

    def _flush_logs():
        if touch_e:
            touch_parts.append(
                (np.array(touch_e, dtype=np.int64), np.array(touch_sp, dtype=np.int64))
            )
            touch_e.clear()
            touch_sp.clear()
        if tl_e:
            tl_parts.append(
                (
                    np.array(tl_e, dtype=np.int64),
                    np.array(tl_x, dtype=np.int64),
                    np.array(tl_z, dtype=np.int8),
                )
            )
            tl_e.clear()
            tl_x.clear()
            tl_z.clear()

    def _sky(v):
        if not skyline or skyline[-1][1] != v:
            skyline.append([T, v])

    def _close_span(at):
        if spans:
            spans[-1]["span"][1] = at
            if spans[-1]["span"][0] == at and not spans[-1]["_touched"]:
                # Empty phase: no ops, no liveness events — it occupies no
                # axis width, so it leaves the chart. The Builder skips an
                # idle span's entry peak too.
                spans.pop()

    def _event():
        """Apply log event ``li`` at the current cursors. Under a kernel
        chunk (``direct`` False) the episode arrays and logs are left to
        the kernel, which replays the chunk's allocs/frees in row order.
        """
        nonlocal li, cur_phase, mdepth, live, last_ev_live, peak_all
        nonlocal ep_of, ep_state, ep_last
        kind, x, _token, ev_live = log[li]
        if kind == "alloc":
            live = last_ev_live = ev_live
            peak_all = max(peak_all, ev_live)
            _sky(ev_live)
            if spans:
                spans[-1]["peak"] = max(spans[-1]["peak"], ev_live)
                spans[-1]["_touched"] = True
            open_q[x] = (T, cur_phase, li)
            e = len(ep_born)
            ep_born.append(max(len(spans) - 1, 0))
            acts[x] = e
            if direct:
                ep_of = _grow(ep_of, x + 1, -1)
                ep_state = _grow(ep_state, e + 1, 0)
                ep_last = _grow(ep_last, e + 1, 0)
                ep_of[x] = e
                ep_state[e] = z = x >= n_io
                ep_last[e] = -1
                tl_e.append(e)
                tl_x.append(X)
                tl_z.append(z)
        elif kind == "free":
            live = last_ev_live = ev_live
            _sky(ev_live)
            if spans:
                spans[-1]["_touched"] = True
            born, bp, bs = open_q.pop(x)
            iv = {
                "q": x,
                "born": born,
                "died": T,
                "born_phase": bp,
                "died_phase": cur_phase,
                "born_seq": bs,
                "died_seq": li,
                "state_segments": None,
            }
            deferred.append((iv, acts.pop(x), max(len(spans) - 1, 0)))
            if direct:
                ep_of[x] = -1
            if counts_stack:
                iv["episodes"] = prod(counts_stack)
            intervals.append(iv)
        elif kind == "mark":
            _close_span(X)
            spans.append(
                {
                    "name": x,
                    "group": "mirror" if mdepth else "forward",
                    "span": [X, None],
                    "peak": ev_live,
                    "toffoli": 0,
                    "span_toffoli": 0,
                    "counts": {},
                    "_touched": False,
                    "_seq": li,
                }
            )
            cur_phase = x
        elif kind == "mirror-begin":
            mdepth += 1
        elif kind == "mirror-end":
            mdepth -= 1
            if spans and spans[-1]["span"][0] == X and not spans[-1]["_touched"]:
                spans[-1]["group"] = "mirror" if mdepth else "forward"
        li += 1

    def _scalar(walk):
        """Per-item replay of a document-order walk, each item after the
        log events that precede it.
        """
        nonlocal X, T, area, tot_tof
        for tag, key, item in walk:
            while li < len(log) and _event_before(log[li][2], key):
                _event()
            if tag == "enter":
                loop_ctx.append([T, item])
                counts_stack.append(item)
                continue
            if tag == "exit":
                T0, count = loop_ctx.pop()
                counts_stack.pop()
                # one representative iteration stays on the axis; the
                # region and trip count are disclosed for the renderer's
                # x-annotation (balanced allocation makes every iteration's
                # liveness identical, so the shown iteration IS each one)
                if T > T0 and count > 1:
                    loops_out.append({"span": [T0, T], "count": count, "depth": len(counts_stack)})
                continue
            op = _leaf7(item)
            k0 = op[0]
            if isinstance(item, tuple):
                first = total = 1
            else:
                # iteration-0 contribution: enclosing dims pinned to one
                # iteration (gated strides/counts lengths must match)
                first = item_count(item, (1,) * len(counts_stack), tables)
                total = item_count(item, tuple(counts_stack), tables)
            billed = _billed_toffoli(op)
            tof_total = total if billed else 0
            tot_tof += tof_total
            area += live * tof_total
            if spans:
                spans[-1]["_touched"] = True
                spans[-1]["toffoli"] += tof_total
                if billed:
                    spans[-1]["span_toffoli"] += first
                cd = spans[-1]["counts"]
                kn = KIND_NAMES.get(k0, str(k0))
                cd[kn] = cd.get(kn, 0) + total
            wf = OP_WRITE_FIELDS.get(k0, ())
            sp_i = max(len(spans) - 1, 0)
            if total:  # a fully gated-off record touches nothing
                for f in QUBIT_FIELDS.get(k0, ()):
                    for q in _leaf_field_ids(item, op, f, counts_stack, tables):
                        e = acts.get(q)
                        if e is None:
                            continue  # episode closed (recycle()'s own R)
                        if ep_last[e] != sp_i:
                            ep_last[e] = sp_i
                            touch_e.append(e)
                            touch_sp.append(sp_i)
                        if f in wf and ep_state[e]:
                            ep_state[e] = 0  # written: zero lapses
                            tl_e.append(e)
                            tl_x.append(X)
                            tl_z.append(0)
                if k0 in ZEROING_KINDS:
                    for q in _leaf_field_ids(item, op, 3, counts_stack, tables):
                        e = acts.get(q)
                        if e is not None and not ep_state[e]:
                            ep_state[e] = 1  # demolition leaves |0>
                            tl_e.append(e)
                            tl_x.append(X)
                            tl_z.append(1)
            X += first
            if billed:
                T += first

    def _chunk(kernel, rows, r0, r1):
        """Replay packed flat rows [r0, r1) — the same per-leaf semantics
        as _scalar (every row: first = total = 1).
        """
        nonlocal X, T, area, tot_tof, direct, ep_of, ep_state, ep_last
        R = rows[r0:r1]
        n = r1 - r0
        s0 = R[:, 0]
        billed = _billed_rows(s0)
        cb = np.zeros(n + 1, dtype=np.int64)  # billed rows before each row
        np.cumsum(billed, out=cb[1:])
        X0, T0 = X, T
        eps0, live0, nsp0 = len(ep_born), live, len(spans)
        cur = 0
        # the chunk's events: token[0] < r1 (a token past a loop's close
        # names the row after it)
        j0, j1 = li, int(np.searchsorted(ev_first, r1))
        ps = np.maximum(ev_first[j0:j1], r0) - r0
        mark_nsp = []  # len(spans) after each mark
        direct = False
        for p, t in zip(ps.tolist(), (T0 + cb[ps]).tolist()):
            if p > cur:
                if spans:
                    spans[-1]["_touched"] = True  # rows [cur, p) replayed
                cur = p
            X = X0 + p
            T = t
            if log[li][0] == "mark":
                _event()
                mark_nsp.append(len(spans))
            else:
                _event()
        direct = True
        if n > cur and spans:
            spans[-1]["_touched"] = True
        X = X0 + n
        T = T0 + int(cb[n])

        # State segments: segment s runs from row cp_pos[s] until the next
        # state change (alloc/free/mark event).
        ek = ev_kind[j0:j1]
        ci = np.flatnonzero(ek)
        eck = ek[ci]
        af = eck != _EV_MARK
        cp_pos = [0] + ps[ci].tolist()
        nseg = len(cp_pos)
        sidx = np.arange(nseg)
        lv = np.concatenate([[live0], np.where(af, ev_nlive[j0:j1][ci], -1)])
        cp_live = lv[np.maximum.accumulate(np.where(lv >= 0, sidx, 0))]
        nv = np.full(nseg, -1, dtype=np.int64)
        nv[0] = nsp0
        nv[1:][~af] = mark_nsp
        nv = nv[np.maximum.accumulate(np.where(nv >= 0, sidx, 0))]
        cp_nsp = nv.tolist()

        # ---- mass: totals, area, per-span toffoli and kind counts --------
        pos = np.array(cp_pos + [n], dtype=np.intp)
        tof_seg = np.diff(cb[pos])
        tot_tof += int(cb[n])
        area += int(tof_seg @ cp_live)
        kind = s0 & 0xFF
        s = 0
        while s < nseg:  # runs of segments sharing one span
            e = s + 1
            while e < nseg and cp_nsp[e] == cp_nsp[s]:
                e += 1
            ra, rb = cp_pos[s], (cp_pos[e] if e < nseg else n)
            if cp_nsp[s] and rb > ra:
                sp = spans[cp_nsp[s] - 1]
                tf = int(cb[rb] - cb[ra])
                sp["toffoli"] += tf
                sp["span_toffoli"] += tf
                cd = sp["counts"]
                kseg = kind[ra:rb]
                cnt = np.bincount(kseg, minlength=256)
                ks = [int(k) for k in np.flatnonzero(cnt)]
                new = [k for k in ks if KIND_NAMES.get(k, str(k)) not in cd]
                if len(new) > 1:  # new keys enter in first-occurrence order
                    new.sort(key=lambda k: int(np.argmax(kseg == k)))
                for k in ks:
                    kn = KIND_NAMES.get(k, str(k))
                    if kn in cd:
                        cd[kn] += int(cnt[k])
                for k in new:
                    cd[KIND_NAMES.get(k, str(k))] = int(cnt[k])
            s = e

        # ---- activity (the kernel) ----------------------------------------
        # episode changes: allocs take the next episode ids in order and
        # start at |0> unless they are io wires
        mci = ci[af]
        mod_q = np.ascontiguousarray(ev_wire[j0:j1][mci])
        is_alloc = eck[af] == _EV_ALLOC
        mod_e = np.where(is_alloc, eps0 + np.cumsum(is_alloc) - 1, -1)
        mod_init = (is_alloc & (mod_q >= n_io)).astype(np.int8)
        mod_pos = np.ascontiguousarray(ps[mci])
        ep_of = _grow(ep_of, int(mod_q.max(initial=-1)) + 1, -1)
        ep_state = _grow(ep_state, len(ep_born), 0)
        ep_last = _grow(ep_last, len(ep_born), 0)
        seg_pos = np.array(cp_pos, dtype=np.int64)
        seg_sp = np.maximum(nv - 1, 0)
        _flush_logs()
        cap = 3 * n + len(mod_q)
        o_te = np.empty(3 * n, dtype=np.int64)
        o_ts = np.empty(3 * n, dtype=np.int64)
        o_le = np.empty(cap, dtype=np.int64)
        o_lx = np.empty(cap, dtype=np.int64)
        o_lz = np.empty(cap, dtype=np.int8)
        n_out = np.zeros(2, dtype=np.int64)
        kernel(
            _ptr(R),
            n,
            _ptr(_ROW_INFO),
            X0,
            _ptr(seg_pos),
            _ptr(seg_sp),
            nseg,
            _ptr(mod_pos),
            _ptr(mod_q),
            _ptr(mod_e),
            _ptr(mod_init),
            len(mod_q),
            _ptr(ep_of),
            len(ep_of),
            _ptr(ep_state),
            _ptr(ep_last),
            _ptr(o_te),
            _ptr(o_ts),
            _ptr(n_out[0:]),
            _ptr(o_le),
            _ptr(o_lx),
            _ptr(o_lz),
            _ptr(n_out[1:]),
        )
        nt, nl = int(n_out[0]), int(n_out[1])
        touch_parts.append((o_te[:nt].copy(), o_ts[:nt].copy()))
        tl_parts.append((o_le[:nl].copy(), o_lx[:nl].copy(), o_lz[:nl].copy()))

    if kernel is None:
        _scalar(_walk_recorded(tape, tables))
    else:
        assert buf is not None
        rows = np.frombuffer(buf, dtype=np.uint32).reshape(-1, 9)
        ev_first = np.array([e[2][0] for e in log], dtype=np.int64)
        ev_kind = np.array([_EV_CODE.get(e[0], 0) for e in log], dtype=np.int8)
        ev_wire = np.array([e[1] if e[0] in ("alloc", "free") else -1 for e in log], dtype=np.int64)
        ev_nlive = np.array([e[3] for e in log], dtype=np.int64)
        start = 0
        for j in sorted(tape._objs) + [len(tape)]:
            r0 = start
            while r0 < j:
                r1 = min(j, r0 + REPLAY_CHUNK)
                if r1 - r0 < REPLAY_SCALAR_RUN:
                    _scalar(("leaf", (i,), tape[i]) for i in range(r0, r1))
                else:
                    _chunk(kernel, rows, r0, r1)
                r0 = r1
            if j < len(tape):
                _scalar(_walk_recorded([tape._objs[j]], tables, (), j))
            start = j + 1
        del rows
    while li < len(log):
        _event()
    _flush_logs()

    _close_span(X)
    if spans and spans[-1]["span"][1] == spans[-1]["span"][0] and not spans[-1]["_touched"]:
        spans.pop()
    for p in spans:
        if p["span"][1] is None:
            p["span"][1] = X
        del p["_touched"]
    if len(open_q) != last_ev_live:
        fail(f"interval accounting broken: {len(open_q)} unclosed vs persistent {last_ev_live}")
    last_pi = len(spans) - 1
    for q, (born, bp, bs) in sorted(open_q.items()):
        iv = {
            "q": q,
            "born": born,
            "died": None,
            "born_phase": bp,
            "died_phase": None,
            "born_seq": bs,
            "died_seq": None,
            "state_segments": None,
        }
        intervals.append(iv)
        deferred.append((iv, acts[q], last_pi))
    segs = _state_segments(
        spans, ep_born, touch_parts, tl_parts, [(e, k1) for _, e, k1 in deferred]
    )
    for (iv, _, _), sg in zip(deferred, segs):
        iv["state_segments"] = sg
    for p in spans:
        del p["_seq"]
    loops_out.sort(key=lambda l: (l["span"][0], l["depth"]))
    if not loops_out:
        # flat program: geometry == mass; drop the redundant field so flat
        # profiles stay byte-identical with the pre-loop-form format
        for p in spans:
            del p["span_toffoli"]
    return spans, {
        "toffoli": tot_tof,
        "peak": peak_all,
        "area": area,
        "skyline": skyline,
        "persistent": last_ev_live,
        "io_qubits": n_io,
        "intervals": intervals,
        "loops": loops_out,
        "span_toffoli": T,
    }


def attach_builder_sets(writer, intervals: list[dict]) -> int:
    """Builder-mode wire-set memberships: ``b.claim(name, wires)`` /
    ``b.release(name, wires)`` record (op, position token, name, wires)
    in ``b.wire_log``. Each interval gains ``sets``: the ordered, deduped
    set names the episode was claimed into (ownership names first as
    claimed; decorative ``~`` names as claimed). Qubit ids are
    LIFO-reused, so a claim resolves to the latest allocation of the
    wire at-or-before the recorded position. Releases end ownership for
    the checkers; for episode labeling the membership name suffices.
    """
    allocs: dict = {}
    claims = []
    for seq, (kind, x, token, _live) in enumerate(writer.wire_log):
        if kind == "alloc":
            allocs.setdefault(x, []).append((token, seq))
        elif kind == "claim":
            claims.append((x[0], x[1], token))
    by_key = {(iv["q"], iv["born_seq"]): iv for iv in intervals}
    n = 0
    for name, wires, pos in claims:
        for q in wires:
            cands = [sq for i2, sq in allocs.get(q, []) if i2 <= pos]
            if not cands:
                fail(f"claim('{name}') for qubit {q} precedes any allocation of it")
            iv = by_key.get((q, cands[-1]))
            if iv is None:
                fail(f"claim('{name}') for qubit {q} matches no interval")
            ss = iv.setdefault("sets", [])
            if name not in ss:
                ss.append(name)
            n += 1
    return n


def crosscheck_phase_summary(writer, phases) -> None:
    """Our span-based stats must agree with the Builder's own telemetry."""
    stats = writer.phase_summary()
    ours_tof: dict[str, int] = {}
    ours_peak: dict[str, int] = {}
    for p in phases:
        ours_tof[p["name"]] = ours_tof.get(p["name"], 0) + p["toffoli"]
        ours_peak[p["name"]] = max(ours_peak.get(p["name"], 0), p["peak"])
    for s in stats["phases"]:
        nm = s["name"]
        if nm in ours_tof and ours_tof[nm] != s["toffoli"]:
            fail(f"phase '{nm}' toffoli {ours_tof[nm]} != builder telemetry {s['toffoli']}")
        if nm in ours_peak and ours_peak[nm] != s["peak"]:
            fail(f"phase '{nm}' peak {ours_peak[nm]} != builder telemetry {s['peak']}")


# --------------------------------------------------------------------------


def pool_intervals(intervals, total_tof=None, cap=INTERVAL_CAP):
    """Max-pool allocation intervals down to ~``cap``.

    Lean variants (count at or under the cap) keep every interval. Above
    it, two stages apply. First, SUB-RESOLUTION untagged temporaries are
    dropped: episodes spanning less than ``total_tof/cap`` executed
    Toffoli occupy under one output pixel at the skyline's own resolution
    — the live-count silhouette (stored separately in ``skyline``) is
    unaffected, only invisible bars go, and the drop is disclosed in the
    pooling note. Then the classic stage merges consecutive UNTAGGED
    episodes of the same physical qubit
    whose idle gap (executed Toffoli) is at or below the smallest threshold
    that brings the count under ``cap`` — the interval analogue of the
    skyline max-pool: a merged bar claims its slot across the pooled gaps
    (an upper envelope), spanning first-birth to last-death, with a
    ``pooled`` episode count on every merged record.  Tagged and persistent
    (died=None) intervals are never pooled.  Returns (intervals, note);
    note is None when nothing was pooled.
    """
    if len(intervals) <= cap:
        return intervals, None
    open_iv = [iv for iv in intervals if iv["died"] is None]
    temps = [iv for iv in intervals if iv["died"] is not None]
    dropped = None
    if total_tof and len(intervals) > cap:
        thr_px = max(total_tof // cap, 1)
        keep = [iv for iv in temps if "sets" in iv or iv["died"] - iv["born"] >= thr_px]
        if len(keep) < len(temps):
            dropped = {"dropped_subpixel": len(temps) - len(keep), "span_below_toffoli": thr_px}
            print(
                f"intervals: dropped {dropped['dropped_subpixel']} "
                f"sub-resolution temps (span < {thr_px} executed Toffoli "
                f"= one pixel at skyline resolution)"
            )
            temps = keep
    if len(temps) + len(open_iv) <= cap:
        out = sorted(temps, key=lambda iv: iv["died_seq"]) + sorted(open_iv, key=lambda iv: iv["q"])
        return out, dropped
    by_q: dict = {}
    for iv in sorted(temps, key=lambda iv: iv["born_seq"]):
        by_q.setdefault(iv["q"], []).append(iv)
    gaps = []
    for chain in by_q.values():
        for a, nxt in zip(chain, chain[1:]):
            if "sets" not in a and "sets" not in nxt:
                gaps.append(nxt["born"] - a["died"])
    need = min(len(intervals) - cap, len(gaps))
    gaps.sort()
    thr = gaps[need - 1] if need > 0 else -1
    pooled = []
    for chain in by_q.values():
        cur, n = dict(chain[0]), 1
        for nxt in chain[1:]:
            if "sets" not in cur and "sets" not in nxt and nxt["born"] - cur["died"] <= thr:
                cur["died"] = nxt["died"]
                cur["died_phase"] = nxt["died_phase"]
                cur["died_seq"] = nxt["died_seq"]
                # a pooled bar spans several episodes: per-episode state
                # segments no longer apply — render it uniform
                cur.pop("state_segments", None)
                n += 1
            else:
                if n > 1:
                    cur["pooled"] = n
                pooled.append(cur)
                cur, n = dict(nxt), 1
        if n > 1:
            cur["pooled"] = n
        pooled.append(cur)
    pooled.sort(key=lambda iv: iv["died_seq"])
    out = pooled + sorted(open_iv, key=lambda iv: iv["q"])
    print(
        f"intervals pooled {len(intervals)} -> {len(out)} (untagged "
        f"same-slot episodes, idle gap <= {thr} executed Toffoli)"
    )
    note = {"before": len(intervals), "after": len(out), "gap_threshold_toffoli": thr}
    if dropped:
        note.update(dropped)
    return out, note


def downsample(skyline, cap=SKYLINE_CAP):
    if len(skyline) <= cap:
        return skyline
    total = skyline[-1][0]
    bins: dict[int, list] = {}
    for t, v in skyline:
        b = min(cap - 1, int(t * cap / max(total, 1)))
        if b not in bins or v > bins[b][1]:
            bins[b] = [t, v]
    out = [bins[b] for b in sorted(bins)]
    print(f"skyline downsampled {len(skyline)} -> {len(out)} points")
    return out


def recorded_sections(writer):
    """Run the recorded-data pipeline shared by main() and eval.py
    --liveness: replay, activity, set membership, pooling. Returns
    (phases, r, activity, intervals_out, pooling).
    """
    if sum(1 for e in writer.wire_log if e[0] == "mark") < 2:
        fail("circuit never called set_phase; the profiler needs native phase labels")
    phases, r = builder_replay(writer)
    crosscheck_phase_summary(writer, phases)
    activity = activity_section(phases, r["intervals"])
    af = activity["derived"]
    print(
        f"activity (toffoli-weighted): active {af['active_fraction']:.1%}"
        f"  idle {af['idle_fraction']:.1%}"
        f"  zero {af['zero_fraction']:.1%}"
        f"  ({len(r['intervals'])} episodes, op-semantics zeros only)"
    )
    n_mem = attach_builder_sets(writer, r["intervals"])
    print(f"mode=builder: {len(phases)} phases, {n_mem} set memberships")
    if sum(p["toffoli"] for p in phases) != r["toffoli"]:
        fail("per-phase toffoli does not sum to total")
    intervals_out, pooling = pool_intervals(r["intervals"], total_tof=r["span_toffoli"])
    return phases, r, activity, intervals_out, pooling


def register_bands(meta, writer, persistent):
    """Build the persistent-wire bands of the liveness plot, bottom up.

    Each meta.json ``registers`` entry names one band (``name``,
    ``label``). A band spans ``regs`` declared Qubit registers (default
    1), in declaration order, and takes its width from the Builder. A
    band of undeclared resident wires states its width as ``size``.
    Without ``registers``, one ``io`` band covers every persistent wire.
    """
    if "registers" not in meta:
        return [{"name": "io", "size": persistent}]
    from uncompute.compact import WKIND_QUBIT, Regs

    widths = [
        len(wires) for kind, wires in Regs(writer._registers).registers if kind == WKIND_QUBIT
    ]
    bands, i = [], 0
    for band in meta["registers"]:
        out = {k: band[k] for k in ("name", "label") if k in band}
        if "size" in band:
            out["size"] = band["size"]
        else:
            n = band.get("regs", 1)
            if i + n > len(widths):
                fail(
                    f"meta.json registers span more than the {len(widths)} declared qubit registers"
                )
            out["size"] = sum(widths[i : i + n])
            i += n
        bands.append(out)
    if i != len(widths):
        fail(f"meta.json registers span {i} of the {len(widths)} declared qubit registers")
    return bands


def assemble_payload(task, prov, registers, phases, r, activity, intervals_out, pooling):
    """Build the liveness.json payload — one construction for every producer."""
    if sum(rr["size"] for rr in registers) != r["persistent"]:
        fail(f"register band widths must sum to persistent width {r['persistent']}")
    out = {
        "task": task,
        "generated_by": "tools/liveness_profile.py",
        "provenance": prov,
        "toffoli_basis": "emitted==executed",
        "totals": {
            "toffoli": r["toffoli"],
            "peak_qubits": r["peak"],
            "tq_cost": r["toffoli"] * r["peak"],
        },
        "persistent_qubits": r["persistent"],
        "io_qubits": r["io_qubits"],
        "registers": registers,
        "phases": phases,
        "skyline": downsample(r["skyline"]),
        "intervals": intervals_out,
        "derived": {
            "mean_live": round(r["area"] / max(r["toffoli"], 1), 2),
            "spacetime_area": r["area"],
            "utilization": round(r["area"] / (r["toffoli"] * r["peak"]), 4),
        },
    }
    if activity is not None:
        out["activity"] = activity
    if pooling:
        out["interval_pooling"] = pooling
    if r["loops"]:
        # Loop-form only (flat profiles stay byte-identical): geometry is
        # loop-compressed, mass is true — see the builder_replay header.
        out["time_axis"] = (
            "loop-compressed: each repeat() shows one "
            "representative iteration (regions + trip "
            "counts in 'loops'); toffoli/area/phase totals "
            "are true executed values"
        )
        out["loops"] = r["loops"]
    return out


def write_liveness(path, payload):
    """Write a liveness payload at ``path``, compressing oversized ones.

    Payloads whose serialization exceeds XZ_THRESHOLD are written to
    ``<path>.xz`` (GitHub rejects blobs over 100 MB; the big profiles
    compress ~80:1). The stale twin is removed either way, so a variant
    never carries both forms. Returns the path actually written.
    """
    raw = (json.dumps(payload, indent=1) + "\n").encode()
    xz = path.with_name(path.name + ".xz")
    if len(raw) > XZ_THRESHOLD:
        xz.write_bytes(lzma.compress(raw, preset=6))
        path.unlink(missing_ok=True)
        return xz
    path.write_bytes(raw)
    xz.unlink(missing_ok=True)
    return path


def _locate_variant(task_dir: Path, profile_dir, circuit_name) -> tuple[Path, str]:
    """Return the variant's directory and filename.

    Standalone mode profiles ``profile_dir``. Record mode profiles the
    discovered circuit (the first existing entry of ``PRIMARY_SOURCES``).
    """
    if profile_dir is not None:
        return task_dir / profile_dir, circuit_name
    src = next(s for s in PRIMARY_SOURCES if (task_dir / s).exists())
    if circuit_name == "circuit.py":
        circuit_name = Path(src).name
    return (task_dir / src).parent, circuit_name


def _output_suffix(circuit_name: str) -> str:
    """Suffix for every output of a non-default variant: circuit_lowq.py -> _lowq."""
    if circuit_name == "circuit.py":
        return ""
    stem = Path(circuit_name).stem
    return stem[len("circuit") :] if stem.startswith("circuit") else "_" + stem


def _certified_record(results: list[dict]) -> dict:
    """Return the results.jsonl row of the discovered circuit ({} if absent)."""
    return next((e for e in results if e["source"] in PRIMARY_SOURCES), {})


def _run_profiled(annotated: Path, prof_dir: Path, sfx: str):
    """Run the variant under the profiler; return its Builder and stream path."""
    # The Builder's fixed transport is UNCOMPUTE_OPS_PATH (else ./ops.bin in
    # cwd): capture the profiled stream at an explicit path so an ambient
    # value can never misroute it.
    ops_path = prof_dir / f"ops{sfx}.bin"
    os.environ["UNCOMPUTE_OPS_PATH"] = str(ops_path)
    print("running the circuit under the profiler ...")
    writer = run_annotated(annotated, prof_dir)
    return writer, ops_path


def _eval_cert(task_dir: Path, ops_path: Path, label: str, jobs, batch_shots, **eval_kw):
    """Certify the produced stream with the evaluator; fail unless valid.

    Report artifacts go to a scratch dir so nothing tracked is touched.
    Returns the evaluator result and the cert block for the provenance.
    """
    print(f"running the evaluator (tools/eval.py --ops) on {label} ops.bin ...")
    from eval import evaluate

    with tempfile.TemporaryDirectory() as td:
        res = evaluate(
            task_dir.name,
            ops=str(ops_path),
            out_dir=td,
            jobs=jobs,
            batch_shots=batch_shots,
            **eval_kw,
        )
    if not res["valid"]:
        fail(f"evaluator certification FAILED: {res.get('fail_reason')}")
    cert = {
        "tq_cost": res["tq_cost"],
        "metrics": {"toffoli": res["toffoli"], "qubits": res["qubits"]},
    }
    print(f"cert OK: {cert['metrics']['toffoli']}x{cert['metrics']['qubits']} = {cert['tq_cost']}")
    return res, cert


def _reference_stream(discovered: Path) -> tuple[bytes, str]:
    """Return the stream a discovered circuit copy must reproduce byte for byte, and its description."""
    artifact = discovered.with_suffix(".ops.xz")
    if artifact.exists():
        # The circuit's .ops.xz is the exact byte stream that tools/certify.py
        # certified: the canonical reference.
        return lzma.decompress(artifact.read_bytes()), f"certified stream ({artifact.name})"
    print("running pristine discovered circuit (vanilla subprocess) ...")
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        env["UNCOMPUTE_OPS_PATH"] = str(Path(td) / "ops.bin")
        r = subprocess.run(
            [sys.executable, str(discovered)],
            cwd=td,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            fail(f"pristine discovered circuit run failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return (Path(td) / "ops.bin").read_bytes(), "pristine discovered circuit run"


def _check_provenance(annotated, discovered, profile_dir, standalone):
    """Anchor the variant to its source; return (variant sha, discovered circuit sha)."""
    if standalone:
        # A standalone variant is its own circuit: the results row and the
        # source of the discovered circuit are not its provenance anchors.
        sha = sha16(annotated.read_bytes())
        print(f"standalone variant '{profile_dir}' sha256[:16]={sha}")
        return sha, None
    if not discovered.exists():
        fail(f"missing certified source {discovered}")
    discovered_sha = sha16(discovered.read_bytes())
    print(f"discovered circuit sha256[:16]={discovered_sha}")
    return discovered_sha, discovered_sha


def _certify_standalone(
    task_dir, annotated, prof_dir, ops_path, profile_dir, expect, jobs, batch_shots
):
    """Certify a standalone variant's stream against its expected metrics.

    The evaluator seeds the draw by the stream's own h_ops (the results row
    belongs to the discovered circuit). The certified metrics must equal
    the expected totals: --expect-*, else the circuit's meta.json pin.
    Returns the evaluator result, the cert block and the expected totals.
    """
    expect_toffoli, expect_qubits = expect
    pin = pinned_cost(annotated) if (prof_dir / "meta.json").exists() else {}
    want_tof = expect_toffoli if expect_toffoli is not None else pin.get("toffoli")
    want_q = expect_qubits if expect_qubits is not None else pin.get("qubits")
    if want_tof is None or want_q is None:
        fail(
            "standalone variant needs expected metrics: pass "
            "--expect-toffoli/--expect-qubits or record the entry "
            "pin in meta.json"
        )
    res, cert = _eval_cert(
        task_dir, ops_path, profile_dir, jobs, batch_shots, timeout=STANDALONE_EVAL_TIMEOUT
    )
    if (cert["metrics"]["toffoli"], cert["metrics"]["qubits"]) != (want_tof, want_q):
        fail(f"certified metrics {cert['metrics']} != expected {want_tof}x{want_q}")
    return res, cert, want_tof, want_q


def _check_cost_parity(r: dict, want_tof, want_q, standalone: bool) -> None:
    """Fail unless the profiled totals equal the expected ones."""
    if want_tof is not None and (r["toffoli"], r["peak"]) != (want_tof, want_q):
        fail(
            f"profiled {r['toffoli']}x{r['peak']} != "
            f"{'expected' if standalone else 'discovered circuit record'} "
            f"{want_tof}x{want_q}"
        )
    print(
        f"cost parity OK: toffoli={r['toffoli']} peak={r['peak']} "
        f"TQ-cost={r['toffoli'] * r['peak']}"
    )


@click.command()
@click.argument("task_dir", type=Path)
@click.option(
    "--profile-dir",
    default=None,
    help="task subdirectory holding the variant to profile "
    "(default: the discovered circuit, e.g. circuits/discovered). An explicit "
    "value selects the STANDALONE flow: evaluator certification "
    "of the produced stream, expected metrics "
    "from --expect-* or the entry's meta.json pin",
)
@click.option(
    "--expect-toffoli",
    type=int,
    default=None,
    help="standalone variants: required certified/profiled "
    "Toffoli total (default: the meta.json pin)",
)
@click.option(
    "--expect-qubits",
    type=int,
    default=None,
    help="standalone variants: required certified/profiled "
    "peak qubits (default: the meta.json pin)",
)
@click.option(
    "--circuit-name",
    default="circuit.py",
    help="variant filename inside the profile dir (default "
    "circuit.py). A non-default name suffixes every "
    "output: circuit_lowq.py -> ops_lowq.bin, "
    "liveness_lowq.json",
)
@click.option(
    "--jobs", type=int, default=None, help="parallel workers for the evaluator certification"
)
@click.option(
    "--batch-shots",
    type=int,
    default=None,
    help="lane-batch size for the evaluator certification "
    "(sharding engages only when shots exceed this)",
)
def main(
    task_dir,
    profile_dir,
    expect_toffoli,
    expect_qubits,
    circuit_name,
    jobs,
    batch_shots,
) -> None:
    task_dir = task_dir.resolve()
    results_p = task_dir / "results.jsonl"
    results = (
        [json.loads(ln) for ln in results_p.read_text().splitlines()] if results_p.exists() else []
    )
    standalone = profile_dir is not None
    prof_dir, circuit_name = _locate_variant(task_dir, profile_dir, circuit_name)
    annotated = prof_dir / circuit_name
    sfx = _output_suffix(circuit_name)
    if not annotated.exists():
        fail(f"missing {annotated}")

    # ---- 1. provenance checks -------------------------------------------------
    record = _certified_record(results)
    # The record names its certified source (task-relative).
    discovered = task_dir / (record.get("source") or PRIMARY_SOURCES[0])
    sha, discovered_sha = _check_provenance(annotated, discovered, profile_dir, standalone)

    # ---- 2. profiled run + stream certification ------------------------------
    writer, ops_path = _run_profiled(annotated, prof_dir, sfx)
    cert = None
    if standalone:
        res, cert, want_tof, want_q = _certify_standalone(
            task_dir,
            annotated,
            prof_dir,
            ops_path,
            profile_dir,
            (expect_toffoli, expect_qubits),
            jobs,
            batch_shots,
        )
        byte_desc = None
    else:
        got_bytes = ops_path.read_bytes()
        ref_bytes, byte_desc = _reference_stream(discovered)
        if got_bytes != ref_bytes:
            fail(
                f"profiled ops.bin differs from {byte_desc} "
                f"({len(got_bytes)} vs {len(ref_bytes)} bytes)"
            )
        print(f"ops.bin byte-identical OK vs {byte_desc} ({len(got_bytes)} bytes)")
        h_ops = hashlib.sha256(got_bytes).hexdigest()
        if record.get("h_ops") and record["h_ops"] != h_ops:
            fail(f"profiled stream h_ops {h_ops} != record h_ops {record['h_ops']}")
        want_tof, want_q = record.get("toffoli"), record.get("qubits")

    # ---- 3. phases + replay ---------------------------------------------------
    if writer is None:
        fail("no Builder emitted ops.bin (uncompute.builder.LAST_EMITTED unset)")
    phases, r, activity, intervals_out, pooling = recorded_sections(writer)

    # ---- 4. cost parity (results.jsonl / the standalone expected metrics) -----
    _check_cost_parity(r, want_tof, want_q, standalone)

    # ---- 5. toffoli basis: "emitted==executed" (in assemble_payload) ----------
    # The emitted census equals the billed counts (at promoted arity) for
    # every producer: conditions — outcome (Z/CZ/NEG only) or bit — narrow a
    # gate's per-shot action, never its billing, enforced at emit AND at
    # parse. The stamp string is a fixed data-format field of liveness.json;
    # it names the invariant.

    # ---- 6. register bands (meta.json names them, the Builder sizes them) --------
    meta_p = prof_dir / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    registers = register_bands(meta, writer, r["persistent"])

    # ---- 7. emit ------------------------------------------------------------------
    prov = {
        "circuit_sha256": sha,
        "discovered_sha256": discovered_sha,
        "mode": "builder",
        "variant": profile_dir if standalone else "discovered",
        "verified": {
            "ops_bin_byte_identical": byte_desc,
            "eval_cert": cert,
            "cost_parity": want_tof is not None,
        },
    }
    if standalone:
        prov["h_ops"] = res["h_ops"]  # the variant's own seed input
    out = assemble_payload(
        task_dir.name, prov, registers, phases, r, activity, intervals_out, pooling
    )
    out_p = write_liveness(prof_dir / f"liveness{sfx}.json", out)
    print(f"wrote {out_p}")
    print(
        f"phases: {len(phases)}  skyline points: {len(out['skyline'])}  "
        f"utilization: {out['derived']['utilization']}"
    )


if __name__ == "__main__":
    main()
