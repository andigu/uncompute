"""Small shared helpers for the evaluator.

Register bit-packing over the wide-lane simulator (the normative per-shot
register load/readout semantics) and the address-space caps.
"""

from __future__ import annotations

U64_MAX = (1 << 64) - 1
MAX_QUBIT_ADDR = 1 << 20
MAX_OUT_ADDR = 1 << 24
MAX_BIT_ADDR = 1 << 24

# REGS register kinds (the uncompute.ops docstring); uncompute.compact uses the same values.
WKIND_QUBIT = 0
WKIND_BIT = 1


def _reg_arr(sim, kind):
    return sim.bits if kind == WKIND_BIT else sim.qubits


def _set_register(sim, reg, val: int, shot: int) -> None:
    """Write ``val`` into a ``(reg_kind, wires)`` register at lane ``shot``.

    The normative per-shot semantics: wire ``i`` gets bit ``i`` of ``val``
    (set or cleared), in wire order — last write wins on aliased wires;
    Qubit registers land in lane state, Bit registers in bit space. The
    C kernel's bulk loader (``kern_load_register``) must match this exactly.
    """
    bit = 1 << shot
    kind, wires = reg
    arr = _reg_arr(sim, kind)
    for i, wid in enumerate(wires):
        sim._grow(arr, wid)
        if (val >> i) & 1:
            arr[wid] |= bit
        else:
            arr[wid] &= ~bit


def _get_register(sim, reg, shot: int) -> int:
    """Read a ``(reg_kind, wires)`` register at lane ``shot`` (normative twin
    of the C kernel's bulk ``kern_read_register``).
    """
    v = 0
    kind, wires = reg
    arr = _reg_arr(sim, kind)
    for i, wid in enumerate(wires):
        if wid < len(arr) and (arr[wid] >> shot) & 1:
            v |= 1 << i
    return v
