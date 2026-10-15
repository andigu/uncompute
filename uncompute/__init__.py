"""uncompute — Python library for the reversible circuits of the uncompute tasks.

Build a circuit with :class:`Builder`, then call ``.emit()`` to write its op
stream for the evaluator. ``emit()`` writes to ``UNCOMPUTE_OPS_PATH`` when that
variable is set (the evaluator's FIFO), else to ``./ops.bin``.
The ``uncompute.ops`` docstring specifies the op-stream format.
"""

from .builder import Builder
from .ops import NO_OUT, NO_QUBIT, NO_REG, OpKind
from .sim import CleanlinessError, SimStats, Simulator

__all__ = [
    "NO_OUT",
    "NO_QUBIT",
    "NO_REG",
    "Builder",
    "CleanlinessError",
    "OpKind",
    "SimStats",
    "Simulator",
]
