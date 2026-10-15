"""C-engine fast path for :class:`uncompute.Simulator`.

Dispatch target of ``Simulator.apply()`` (sim.py): packs the op chunk into
flat u64 rows, bulk-loads the resting Python state into a per-call kernel
context (``kern_sim_*`` in ``evalcore/_kernel.c``), executes natively, and
syncs everything back — ``sim.qubits``/``outs``/``bits``/``phase``/``stats``
and the strict-mode ledger stay ordinary Python objects between calls.

sim.py remains the normative reference. Any op the native walk cannot run
with certain bit-identity (validation reject, strict-mode violation,
sentinel/garbage operands, OOM) makes it HALT and discard the context; the
whole chunk is then replayed through ``Simulator._apply_py`` from the
untouched resting state, so exceptions, their text, and partial state on a
raise are always produced by the reference engine. ``UNCOMPUTE_KERNEL=0``
disables this path (the engine convention of evalcore.protocol); without a
built dylib it is silently unavailable and apply() runs pure Python.
"""

from __future__ import annotations

import ctypes
import os
from array import array
from itertools import chain, islice

from .ops import operand_mask_table

_U64 = ctypes.c_uint64

# Ops per packed chunk when the source is a streaming iterable (bounds
# transient memory; lists pack in one piece).
_CHUNK_OPS = 1 << 18

_available = None


def enabled():
    """Return True when the dylib binds and UNCOMPUTE_KERNEL does not force the
    pure-Python engine.
    """
    global _available
    if os.environ.get("UNCOMPUTE_KERNEL") == "0":
        return False
    if _available is None:
        try:
            from .evalcore.kernel import kernel_available

            _available = kernel_available()
        except (ImportError, OSError):
            # evalcore or its kernel library cannot load
            _available = False
    return _available


def _u64ptr(buf):
    """Zero-copy u64 pointer over a writable buffer (array('Q')/bytearray);
    None for empty buffers (the C side skips on zero counts).
    """
    n = len(buf) if isinstance(buf, array) else len(buf) // 8
    if n == 0:
        return None
    return (_U64 * n).from_buffer(buf)


def _pack_wires(wires, w8):
    return b"".join(v.to_bytes(w8, "little") for v in wires)


def _unpack_ops(chunks):
    for arr in chunks:
        for i in range(0, len(arr), 9):
            yield tuple(arr[i : i + 9])


def apply(sim, ops, validate):
    """Run one ``apply`` chunk natively; fall back to sim._apply_py whenever
    packing or the native walk cannot guarantee bit-identity.
    """
    # Sized containers (lists/tuples, the Builder's list-like op record) are
    # re-iterable: pack in one shot. Anything else streams in chunks below.
    if hasattr(ops, "__len__"):
        try:
            packed = array("Q", chain.from_iterable(ops))
            if len(packed) != 9 * len(ops):
                raise ValueError("ragged op tuples")
        except (OverflowError, TypeError, ValueError):
            return sim._apply_py(ops, validate)
        chunks = [packed] if packed else []
        return _run(sim, chunks, ops, validate)

    # Streaming iterable: pack in bounded chunks, retaining enough to hand
    # the reference engine an equivalent replay stream on any failure.
    it = iter(ops)
    chunks = []
    while True:
        chunk = list(islice(it, _CHUNK_OPS))
        if not chunk:
            break
        try:
            arr = array("Q", chain.from_iterable(chunk))
            if len(arr) != 9 * len(chunk):
                raise ValueError("ragged op tuples")
        except (OverflowError, TypeError, ValueError):
            return sim._apply_py(chain(_unpack_ops(chunks), chunk, it), validate)
        chunks.append(arr)
    return _run(sim, chunks, None, validate)


def _run(sim, chunks, replay_ops, validate):
    """Execute packed flat chunks in one native context; sync back on
    success, replay through the reference engine on halt (resuming from
    the halt position when the native state can be trusted).
    """
    if not chunks:
        return sim
    if replay_ops is None:
        replay_ops = _unpack_ops(chunks)

    def replay(skip=0):
        return sim._apply_py(replay_ops[skip:] if skip else replay_ops, validate)

    def exec_fn(lib, ctx):
        for arr in chunks:
            if lib.kern_sim_apply(ctx, _u64ptr(arr), len(arr) // 9) >= 0:
                return False
        return True

    return _native(sim, replay, exec_fn)


_NO_RESUME = object()


def _rest_snapshot(sim):
    """Snapshot the mutable resting state _sync_back / _apply_py touch."""
    return (
        list(sim.qubits),
        list(sim.outs),
        list(sim.bits),
        sim.phase,
        sim.stats.toffoli,
        sim.stats.clifford,
        sim._ctr,
        sim._op_n,
        dict(sim.ledger),
        dict(sim._draw_refs),
        dict(sim._draw_src),
        {k: set(v) for k, v in sim._slot_draws.items()},
        set(sim.register_qubits),
    )


def _rest_restore(sim, snap):
    (q, o, bt, ph, tof, cl, ctr, opn, led, refs, src, slots, regs) = snap
    sim.qubits[:] = q
    sim.outs[:] = o
    sim.bits[:] = bt
    sim.phase = ph
    sim.stats.toffoli = tof
    sim.stats.clifford = cl
    sim._ctr = ctr
    sim._op_n = opn
    sim.ledger.clear()
    sim.ledger.update(led)
    sim._draw_refs.clear()
    sim._draw_refs.update(refs)
    sim._draw_src.clear()
    sim._draw_src.update(src)
    sim._slot_draws.clear()
    sim._slot_draws.update(slots)
    sim.register_qubits.clear()
    sim.register_qubits.update(regs)


def _resume_after_halt(sim, lib, ctx, w8, strict, replay):
    """Diagnose a native halt without re-simulating the prefix.

    The kernel halts BEFORE applying the offending op (validate and
    strict-pre checks precede exec; op_n is not yet counted), and its
    prefix state is bit-identical to the reference engine's, so syncing
    the context back and replaying only from the halt position raises
    the normative exception from the exact partial state a full replay
    would reach.  A resume that completes cleanly means the halt was not
    a reference-visible defect (e.g. a kernel-internal allocation
    failure mid-op): restore the resting state and signal the caller to
    run the full reference replay from scratch.
    """
    op_n0 = sim._op_n
    snap = _rest_snapshot(sim)
    try:
        _sync_back(sim, lib, ctx, w8, strict)
        skip = sim._op_n - op_n0
    except Exception:  # noqa: BLE001  (any sync-back failure falls back to the full reference replay)
        _rest_restore(sim, snap)
        return _NO_RESUME
    if skip < 0:
        _rest_restore(sim, snap)
        return _NO_RESUME
    try:
        replay(skip=skip)
    except TypeError:
        # a replay closure without resume support
        _rest_restore(sim, snap)
        return _NO_RESUME
    except BaseException:
        raise  # the normative diagnostic
    _rest_restore(sim, snap)
    return _NO_RESUME


def _native(sim, replay, exec_fn):
    """Shared native-context lifecycle: create/init, bulk state+ledger load,
    exec_fn(lib, ctx) -> completed?, sync back on success. On a halt, the
    reference engine resumes from the synced halt position (raising the
    normative exception where one is due); anything the resume cannot
    reproduce falls back to the full replay from the untouched resting
    state.
    """
    from .evalcore.kernel import load_library

    lib = load_library()
    nw = (sim.batch + 63) // 64
    w8 = nw * 8
    strict = sim.strict

    try:
        state = (
            _pack_wires(sim.qubits, w8),
            _pack_wires(sim.outs, w8),
            _pack_wires(sim.bits, w8),
            sim.phase.to_bytes(w8, "little"),
        )
        if strict:
            draws4 = array("Q")
            resid = bytearray(len(sim.ledger) * w8)
            for i, (d, r) in enumerate(sim.ledger.items()):
                opn, oid = sim._draw_src[d]
                draws4.extend((d, sim._draw_refs[d], opn, oid))
                resid[i * w8 : (i + 1) * w8] = r.to_bytes(w8, "little")
            slots = array("Q")
            for ot, ids in sim._slot_draws.items():
                slots.append(ot)
                slots.append(len(ids))
                slots.extend(sorted(ids))
    except (OverflowError, ValueError, KeyError, AttributeError):
        # poked/inconsistent resting state the flat format cannot carry —
        # the reference engine defines what it means
        return replay()

    ctx = lib.kern_create(sim.batch, bytes(32))
    if not ctx:
        return replay()
    try:
        if (
            lib.kern_sim_init(
                ctx, sim._seed, len(sim._seed), 1 if strict else 0, operand_mask_table()
            )
            != 0
        ):
            return replay()
        qb, ob, bb, pb = state
        rc = lib.kern_sim_load_state(
            ctx,
            _u64ptr(bytearray(qb)),
            len(sim.qubits),
            _u64ptr(bytearray(ob)),
            len(sim.outs),
            _u64ptr(bytearray(bb)),
            len(sim.bits),
            _u64ptr(bytearray(pb)),
            sim.stats.toffoli,
            sim.stats.clifford,
            sim._ctr,
            sim._op_n,
        )
        if rc != 0:
            return replay()
        if (
            strict
            and lib.kern_sim_load_ledger(
                ctx, _u64ptr(draws4), len(sim.ledger), _u64ptr(resid), _u64ptr(slots), len(slots)
            )
            != 0
        ):
            return replay()
        if not exec_fn(lib, ctx):
            res = _resume_after_halt(sim, lib, ctx, w8, strict, replay)
            if res is _NO_RESUME:
                return replay()
            return res
        _sync_back(sim, lib, ctx, w8, strict)
        return sim
    finally:
        lib.kern_free(ctx)


def _sync_back(sim, lib, ctx, w8, strict):
    sizes = (_U64 * 10)()
    lib.kern_sim_sizes(ctx, sizes)
    nq, no, nb, tof, cliff, ctr, op_n, n_draws, slots_len, _ev_len = (int(x) for x in sizes)

    qbuf = bytearray(max(nq, 1) * w8)
    obuf = bytearray(max(no, 1) * w8)
    bbuf = bytearray(max(nb, 1) * w8)
    pbuf = bytearray(w8)
    lib.kern_sim_dump_state(ctx, _u64ptr(qbuf), _u64ptr(obuf), _u64ptr(bbuf), _u64ptr(pbuf))
    # in-place mutation: external aliases of the state lists stay live
    sim.qubits[:] = [int.from_bytes(qbuf[i * w8 : (i + 1) * w8], "little") for i in range(nq)]
    sim.outs[:] = [int.from_bytes(obuf[i * w8 : (i + 1) * w8], "little") for i in range(no)]
    sim.bits[:] = [int.from_bytes(bbuf[i * w8 : (i + 1) * w8], "little") for i in range(nb)]
    sim.phase = int.from_bytes(pbuf, "little")
    sim.stats.toffoli = tof
    sim.stats.clifford = cliff
    sim._ctr = ctr
    sim._op_n = op_n

    if strict:
        draws4 = (_U64 * max(4 * n_draws, 1))()
        resid = bytearray(max(n_draws, 1) * w8)
        slots = (_U64 * max(slots_len, 1))()
        lib.kern_sim_dump_ledger(ctx, draws4, _u64ptr(resid), slots)
        sim.ledger.clear()
        sim._draw_refs.clear()
        sim._draw_src.clear()
        sim._slot_draws.clear()
        for i in range(n_draws):  # ascending id = creation order
            d = int(draws4[4 * i])
            sim.ledger[d] = int.from_bytes(resid[i * w8 : (i + 1) * w8], "little")
            sim._draw_refs[d] = int(draws4[4 * i + 1])
            sim._draw_src[d] = (int(draws4[4 * i + 2]), int(draws4[4 * i + 3]))
        p = 0
        while p < slots_len:
            ot = int(slots[p])
            n = int(slots[p + 1])
            p += 2
            sim._slot_draws[ot] = {int(slots[p + j]) for j in range(n)}
            p += n


# ---- the run() fast path: recorded (loop-form) programs -------------------------

_MASK64 = (1 << 64) - 1


def _pack_recorded(items, tables):
    """Pack the Builder's recorded items into the C walker's program +
    tables buffer (format: the 'recorded walk' comment in _kernel.c).
    Python cost scales with the DESCRIPTION (loops stay loops); the C side
    expands, mirroring compact.iter_expand_item. Raises ValueError on
    anything unpackable — the caller then takes the normal path.
    """
    from . import compact as _c

    prog = array("Q")

    def emit_affine(base, strides, depth):
        if len(strides) != depth:
            raise ValueError("stride depth mismatch")
        prog.append(int(base) & _MASK64)
        for s in strides:
            prog.append(int(s) & _MASK64)

    def pack(seq, depth):
        for item in seq:
            if isinstance(item, tuple):
                if depth != 0 or len(item) != 9:
                    raise ValueError("raw tuple outside depth 0")
                prog.append(0)
                prog.extend(int(v) & _MASK64 for v in item)
            elif isinstance(item, _c.Loop):
                prog.append(2)
                prog.append(int(item.count) & _MASK64)
                hole = len(prog)
                prog.append(0)
                pack(item.items, depth + 1)
                prog[hole] = len(prog) - hole - 1
            elif isinstance(item, _c.OpRec):
                names = _c.KIND_OPERANDS[item.kind]
                slots = {}
                for name, opnd in zip(names, item.operands):
                    slots[_c._FIELD_SLOT[name]] = opnd
                if item.oc is not None:
                    slots[5] = item.oc
                if item.cc is not None:
                    slots[8] = item.cc
                amask = 0
                for sl in slots:
                    amask |= 1 << (sl - 1)
                prog.append(1)
                prog.append(int(item.kind))
                prog.append(amask)
                prog.append(0 if item.gate is None else 1)
                if item.gate is not None:
                    tid, off, gstrides = item.gate
                    prog.append(int(tid))
                    emit_affine(off, gstrides, depth)
                for sl in sorted(slots):
                    base, strides = slots[sl]
                    emit_affine(base, strides, depth)
            else:
                raise ValueError(f"unknown recorded item {item!r}")  # noqa: TRY004  (internal invariant; keep the raised type)

    pack(items, 0)

    n = len(tables)
    tbuf = array("Q", bytes(8 * (1 + 2 * n)))
    tbuf[0] = n
    data = array("Q")
    base_off = 1 + 2 * n
    for i, t in enumerate(tables):
        if t.table_id != i:
            raise ValueError("table_id/index mismatch")
        bits = t.bits
        tbuf[1 + 2 * i] = len(bits)
        tbuf[2 + 2 * i] = base_off + len(data)
        words = [0] * ((len(bits) + 63) // 64)
        for j, b in enumerate(bits):
            if b:
                words[j >> 6] |= 1 << (j & 63)
        data.extend(words)
    tbuf.extend(data)
    return prog, tbuf


def run_source(sim, src):
    """Run the fast path of run() for loop-form Builders: pack the RECORDED items
    (loops stay compact; C expands and executes). Returns sim when handled,
    None to let run() take its normal path — including flat Builders (their
    op list goes through apply()'s flat fast path) and any state the packer
    cannot carry. A native halt replays through iter_expanded(), the
    normative expansion.
    """
    if not getattr(src, "_uses_loops", False):
        return None
    if getattr(src, "_loop_stack", None) or getattr(src, "_gate_ctx", None) is not None:
        return None
    items = getattr(src, "_items", None)
    tables = getattr(src, "_tables", None)
    if items is None or tables is None:
        return None
    try:
        prog, tbuf = _pack_recorded(items, tables)
    except (ValueError, TypeError, OverflowError, KeyError, AttributeError):
        return None

    def replay(skip=0):
        it = _expand_from(src, skip) if skip else src.iter_expanded()
        return sim._apply_py(it, True)

    def exec_fn(lib, ctx):
        return lib.kern_sim_run_rec(ctx, _u64ptr(prog), len(prog), _u64ptr(tbuf), len(tbuf)) == -1

    return _native(sim, replay, exec_fn)


def _expand_from(src, skip):
    """iter_expanded() from an op offset: an arithmetic seek over the
    recorded items (item_count is pure arithmetic), expanding only from
    the item containing the offset — the resume path never re-yields
    the prefix.
    """
    from . import compact as _bc

    tables = {t.table_id: t.bits for t in src._tables}
    items = src._items
    i, n_items = 0, len(items)
    while i < n_items and skip:
        item = items[i]
        n = 1 if isinstance(item, tuple) else _bc.item_count(item, (), tables)
        if n > skip:
            break
        skip -= n
        i += 1

    def gen(j=i, rem=skip):
        while j < n_items:
            item = items[j]
            if isinstance(item, tuple):
                yield item  # rem is 0 here (n=1 never > rem>0)
            else:
                it = _bc.iter_expand_item(item, (), tables)
                if rem:
                    it = islice(it, rem, None)
                    rem = 0
                yield from it
            j += 1

    return gen()
