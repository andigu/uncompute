"""C fast path for the op-stream FLAT serializer (:mod:`uncompute.compact`).

Dispatch target of ``compact._write_flat`` via ``_TapeRun._native_encode``
(builder.py): the packed rows of the Builder's columnar op tape are encoded
natively (``kern_tape_encode_flat`` in ``evalcore/_kernel.c``) in bounded
chunks, while Python keeps every ``fh.write`` — transport (FIFO / tee /
hash sinks, ``emit(dest=)``) and the ``_FLAT_CHUNK`` flush boundaries are
byte-for-byte the reference's.

compact.py remains the normative reference. The C encoder never raises and
never emits bytes the reference would reject: any item it cannot serialize
with certain byte-identity — side-table items (OpRec / odd tuples), rows
carrying a bit condition, stray or missing operands whose canonical errors
must surface at the exact byte position — is handed back one item at a
time and encoded (or raised) by the reference path, so bytes, exceptions,
their text, and the written prefix at a raise are always the reference's.
``UNCOMPUTE_KERNEL=0`` disables this path (the engine convention of
evalcore.protocol); without a built dylib it is silently unavailable and
serialization runs pure Python.
"""

from __future__ import annotations

import ctypes
import os

_U32 = ctypes.c_uint32
_I64 = ctypes.c_int64

# kern_tape_encode_flat status codes (must match _kernel.c).
TAPE_DONE = 0
TAPE_FLUSH = 1
TAPE_PYOP = 2
TAPE_BADARG = 3

# Storage slack past the flush threshold: the encoder appends whole op
# records (<= 14 bytes) and returns TAPE_FLUSH at the first record that
# reaches the threshold, so the chunk buffer must be threshold + 16 bytes.
_OUT_SLACK = 16

# The side-table marker in tape slot 0 (builder._TAPE_OBJ).
_OBJ_ROW = 0xFFFFFFFF

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


def write_run(run, table_ids, fh):
    """Encode one ``_TapeRun`` as a FLAT segment straight to ``fh`` — the
    native twin of ``compact._write_flat``, byte-identical including flush
    boundaries and error positions. Returns False (nothing written) when
    the fast path cannot be used at all; True when the segment was fully
    written. Once the first byte is out there is no bailing: every item
    the kernel hands back is finished by the reference encoder.
    """
    from .evalcore.kernel import load_library

    try:
        lib = load_library()
    except ImportError:
        return False
    from . import compact as _c

    tape = run._tape
    rows = tape._rows
    n_rows = len(rows)
    if n_rows == 0:
        return False  # empty tape: nothing to accelerate
    objs = tape._objs
    chunk = _c._FLAT_CHUNK
    stop = run._stop
    # The segment header, the reference spelling — including the canonical
    # n_ops range error, raised before any byte reaches fh.
    buf = bytearray(_c._U8.pack(_c.SEG_FLAT))
    _c._pack_u32(buf, len(run), "FLAT n_ops")
    out_store = bytearray(chunk + _OUT_SLACK)
    out = (ctypes.c_ubyte * len(out_store)).from_buffer(out_store)
    view = (_U32 * n_rows).from_buffer(rows)
    idx = _I64(run._start)
    got = _I64(0)
    try:
        while True:
            rc = lib.kern_tape_encode_flat(
                view, n_rows // 9, stop, ctypes.byref(idx), out, chunk - len(buf), ctypes.byref(got)
            )
            if got.value:
                buf += out_store[: got.value]
            if rc == TAPE_PYOP:
                # One item the C encoder cannot certify: the reference
                # produces its bytes or raises its canonical error, with
                # the flushed prefix already byte-identical.
                i = idx.value
                if rows[i * 9] == _OBJ_ROW:
                    item = objs[i]
                    if not tape._obj_is_reg(item):
                        if type(item) is tuple:
                            _c._encode_tuple_op(buf, item)
                        else:
                            _c._encode_op(buf, item, 0, table_ids)
                else:
                    _c._encode_tuple_op(buf, tape._get(i))
                idx.value = i + 1
            elif rc == TAPE_DONE:
                break
            elif rc != TAPE_FLUSH:
                raise RuntimeError(
                    f"kern_tape_encode_flat returned {rc} at item "
                    f"{idx.value} (kernel/reference divergence — report "
                    "this)"
                )
            if len(buf) >= chunk:
                fh.write(bytes(buf))
                del buf[:]
        if buf:
            fh.write(bytes(buf))
    finally:
        # Release the exported buffers even when a canonical encode error
        # propagates (the traceback pins this frame): the tape's array
        # must stay resizable after a raise.
        del out, view
    return True
