"""The uncompute evaluator.

Modules:

* ``task``           — the ``Task`` grader interface and failure policies.
* ``protocol``       — seed derivation (per-shot test XOFs), sharded
                       evaluation, and the ``run_task`` entry point; it
                       re-exports the executor layer.
* ``executor``       — the batch-executor layer: the counter-mode
                       measurement-RNG simulator, the per-batch report
                       steps, and the verdicts.
* ``compact_reader`` — the normative op-stream interpreter (pure Python).
* ``kernel``         — the C kernel, used when built; identical results.
* ``loader``         — load errors.
* ``report``         — the structured report and its writers.
"""

from .loader import LoadError  # noqa: F401
from .protocol import Xof, derive_seeds, ops_hash, run_task  # noqa: F401
from .task import (  # noqa: F401
    BIT,
    QUBIT,
    STRICT,
    FailurePolicy,
    GraderConfigError,
    RegSpec,
    Task,
    TestCase,
    Unified,
    resolve_failure_policy,
)
