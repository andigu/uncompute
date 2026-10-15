"""Load errors of the evaluator.

The stream interpreters raise :class:`CompactParseError` (a
:class:`LoadError`) when they reject a stream while parsing it. A stream op
always has the right slots by construction, so the only per-op operand
check is distinctness (``validate/aliasing``). The interpreters word that
failure with :func:`uncompute.ops.op_violation`, which holds the operand rules
and which the local Simulator also uses.
"""

from __future__ import annotations

from .report import LOAD_OP_INVALID


class LoadError(Exception):
    """A rejected op stream. ``code`` is the stable failure code from
    :mod:`uncompute.evalcore.report`; ``op_index`` is the expanded op index for
    a per-op failure, else None.
    """

    def __init__(self, msg, code: str = LOAD_OP_INVALID, op_index=None):
        super().__init__(msg)
        self.code = code
        self.op_index = op_index


class CompactParseError(LoadError):
    """A stream rejected at parse time. The constructor matches
    :class:`LoadError`, so instances survive a pickle round trip out of a
    shard worker process.
    """
