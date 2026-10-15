"""The ``Task`` grader interface.

A grader subclasses :class:`Task`, sets the constant attributes, and
implements ``schema()`` and ``gen_test()``. The rules:

* ``inputs`` and ``expected`` have length ``len(schema())``.
* values are plain ints; a task masks them to its own widths, and the
  harness only moves bits.
* the test XOF is per-shot (``SHAKE256(seed_tests ‖ u64le(shot))``,
  see ``protocol.py``): ``gen_test_indexed(shot, xof)``
  need only be deterministic in ``(shot, xof)`` — there is no cross-shot
  stream, so byte-consumption per call is unconstrained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Register kinds. Qubit registers carry quantum inputs and outputs. Bit
# registers carry classical inputs: per-shot values written into bit space,
# input-only, never read back.
QUBIT = "Qubit"
BIT = "Bit"


@dataclass
class RegSpec:
    width: int
    kind: str = QUBIT


@dataclass
class TestCase:
    """One randomized test case. ``inputs[i]``/``expected[i]`` are ints or None (skip)."""

    inputs: list[int | None] = field(default_factory=list)
    expected: list[int | None] = field(default_factory=list)
    # Expected global sign: False = +1 (default), True = -1.
    expected_phase: bool = False


# --- Failure policy -------------------------------------------------------------
#
# One attribute, ``failure_policy``, takes one of two shapes:
#
#   STRICT (the default): no failing shot, no dirty qubit.
#   Unified(budget): one verdict per shot; at most `budget` shots may fail
#       (wrong output, wrong sign, or a dirty qubit).


class FailurePolicy:
    """Base marker for failure policies (STRICT / Unified)."""

    __slots__ = ()


class _Strict(FailurePolicy):
    __slots__ = ()

    def __repr__(self) -> str:
        return "STRICT"


STRICT = _Strict()


@dataclass(frozen=True)
class Unified(FailurePolicy):
    """Whole-run budget over per-shot verdicts (wrong output, wrong sign, or a dirty qubit)."""

    budget: int

    def __post_init__(self):
        if self.budget < 0:
            raise ValueError("Unified budget must be >= 0")


class GraderConfigError(ValueError):
    """A grader declared an invalid failure policy."""


def resolve_failure_policy(ch: Task) -> FailurePolicy:
    """Return the task's :class:`FailurePolicy`; raise
    :class:`GraderConfigError` when ``failure_policy`` is not one.
    """
    policy = getattr(ch, "failure_policy", STRICT)
    if not isinstance(policy, FailurePolicy):
        raise GraderConfigError(
            f"{ch.name}: failure_policy must be STRICT or Unified(budget), got {policy!r}"
        )
    return policy


class Task:
    """Base grader. Subclass and set the attributes / override the methods."""

    name: str = "?"
    seed_domain: bytes = b"?"
    num_tests: int = 0
    failure_policy: FailurePolicy = STRICT

    def schema(self) -> list[RegSpec]:
        raise NotImplementedError

    def gen_test(self, xof) -> TestCase | None:
        raise NotImplementedError

    def gen_test_indexed(self, shot: int, xof) -> TestCase | None:  # noqa: ARG002  (overrides use shot)
        """Generate the test case for ``shot``; the harness calls this method.

        The default delegates to ``gen_test``.
        """
        return self.gen_test(xof)

    def check_shot(self, tc: TestCase, read: list[int]) -> bool:
        """Default per-shot check: every register with an expected value matches it."""
        return all(e is None or e == r for e, r in zip(tc.expected, read))
