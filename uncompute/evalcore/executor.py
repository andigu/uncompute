"""The batch-executor layer of the evaluator.

The two executors, the normative Python reference (``compact_reader``) and
the C kernel wrapper (``kernel``), share everything in this module:

* :class:`LaneSim` — the counter-mode measurement-RNG simulator;
* :class:`EvalReport` and the per-batch prelude/epilogues
  (:func:`begin_batch`, :func:`fail_batch`, :func:`finish_batch`) — the one
  place the report fields and the cap-on-fail invariant are spelled;
* the end-of-stream verdict machinery (:func:`extract_batch_state`,
  :func:`digest_batch_state`, :func:`apply_verdicts`,
  :func:`finish_from_batch`);
* :data:`UNKNOWN_COUNT` — the END-segment value for an unknown op count
  (the op-stream format in the :mod:`uncompute.ops` docstring).

``protocol`` (seeding, sharding, ``run_task``) sits above the executors
and re-exports these names, so ``uncompute.evalcore.protocol`` stays the public
surface.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from ..ops import _popcount
from ..sim import Simulator
from .report import (
    SCHEMA_COUNT,
    SCHEMA_KIND,
    SCHEMA_WIDTH,
    VERDICT_ANCILLA,
    VERDICT_CLASSICAL,
    VERDICT_PHASE,
    VERDICT_UNIFIED,
)
from .task import (
    BIT,
    Unified,
    resolve_failure_policy,
)
from .util import MAX_BIT_ADDR, MAX_QUBIT_ADDR, WKIND_BIT, WKIND_QUBIT, _get_register

# The END segment's "expanded op count unknown" sentinel (2^64 - 1; the
# op-stream format in the uncompute.ops docstring).
UNKNOWN_COUNT = (1 << 64) - 1


def live_lane_mask(tests) -> int:
    """OR of ``1 << lane`` over the non-``None`` entries of a positional test
    list — the batch's live mask under the dead-lane convention.

    Set bit-per-lane in a bytearray and convert once: a per-lane ``|=`` on a
    growing int recopies the whole mask every lane (quadratic in the lane
    span — measurable at 262144-lane batches). The no-dead-lanes common case
    is one shift.
    """
    if None not in tests:
        return (1 << len(tests)) - 1
    buf = bytearray((len(tests) + 7) // 8)
    for lane, tc in enumerate(tests):
        if tc is not None:
            buf[lane >> 3] |= 1 << (lane & 7)
    return int.from_bytes(buf, "little")


# --- Counter-mode RNG simulator ---------------------------------------------------


class LaneSim(Simulator):
    """Wide-lane simulator drawing measurement bits in block-counter mode.

    ``lane0`` is the batch's global lane offset (a multiple of 8): the k-th
    draw takes bytes ``[lane0/8, lane0/8 + ceil(batch/8))`` of counter k's
    stream, whose 128-byte block i is
    ``SHAKE256(seed_rng ‖ u64le(k) ‖ u64le(i))`` — so a batch sees exactly
    the lane bits an unsharded run would assign it, at O(1) cost in
    ``lane0`` (a sequential squeeze would pay an O(lane0) discarded prefix
    per draw, making sharded evaluation quadratic in the total shot count).

    ``live_mask`` (default: all ``n_shots`` lanes) implements dead-lane
    batches: it narrows the simulator's live-lane mask (so a dead lane never
    bills a gate or takes a state update — dead-lane wires stay |0⟩
    throughout) and the RNG mask.
    """

    def __init__(self, n_shots: int, seed_rng: bytes, lane0: int = 0, live_mask: int | None = None):
        # strict=False: the evaluator reports a dirty circuit through the
        # verdict counters (wrong_sign_shots, dirty_qubits) and the failure
        # codes, never through a mid-run exception. The C kernel implements
        # the same sampled semantics, so the strict ledger stays off here.
        super().__init__(None, batch=max(n_shots, 1), strict=False)
        if lane0 % 8:
            raise ValueError("lane0 must be a multiple of 8")
        if live_mask is not None:
            self.live_mask = live_mask
        self._seed_rng = seed_rng
        self._k = 0
        self._skip = lane0 // 8

    def _draw(self):
        prefix = self._seed_rng + self._k.to_bytes(8, "little")
        self._k += 1
        first, off = divmod(self._skip, 128)
        nblocks = (off + self._nbytes + 127) // 128
        buf = b"".join(
            hashlib.shake_256(prefix + b.to_bytes(8, "little")).digest(128)
            for b in range(first, first + nblocks)
        )
        return int.from_bytes(buf[off : off + self._nbytes], "little") & self.live_mask


# --- Run report --------------------------------------------------------------------


@dataclass
class EvalReport:
    ok: bool = True
    fail_reason: str | None = None
    fail_code: str | None = None
    fail_op_index: int | None = None
    n_shots: int = 0
    ops_len: int = 0
    total_qubits: int = 0
    num_outs: int = 0
    num_bits: int = 0
    avg_tof: float = 0.0
    avg_cliff: float = 0.0
    tot_tof: int = 0
    tot_cliff: int = 0
    wrong_output_shots: int = 0
    wrong_sign_shots: int = 0
    dirty_qubits: int = 0
    failed_shots: int = 0
    # First failing shot (global index) when the failure kind has one
    # (verdict/classical); surfaced as report fail["shot"].
    fail_shot: int | None = None
    # The coded register-schema failure (code, reason) computed once by
    # finish_batch, or None when the schema validated (or the run failed
    # before schema validation). Consumed by finish_from_batch; the verdict
    # is applied there, not per batch.
    schema_fail: tuple[str, str] | None = None
    # Failing shot indices from the verdict digest, set only when shot
    # indices are enabled; surfaced as report["failed_shot_indices"].
    failed_shot_indices: list[int] | None = None


def begin_batch(tests) -> tuple[EvalReport, int, int, int | None]:
    """Shared batch prelude for both executors.

    ``tests`` is positional (``None`` = dead lane). Returns
    ``(report, lane_span, live_mask, sim_live_mask)`` where ``sim_live_mask``
    is ``None`` when every lane is live (the executors then skip masking) and
    the report's ``n_shots`` counts live lanes only.
    """
    n = len(tests)
    live_mask = live_lane_mask(tests)
    r = EvalReport(n_shots=_popcount(live_mask))
    full = (1 << n) - 1 if n else 0
    return r, n, live_mask, (live_mask if live_mask != full else None)


def fail_batch(
    r: EvalReport,
    registers,
    code: str,
    detail: str,
    op_index: int | None,
    *,
    ops_len: int,
    num_qubits: int,
    num_outs: int,
    num_bits: int = 0,
):
    """Shared validation-failure epilogue: fill the failed report and return
    the executors' ``(EvalReport, registers, state)`` outcome triple.

    Invariant (shared by both executors, spelled once): on the failure path
    ``total_qubits`` is capped at ``MAX_QUBIT_ADDR`` — a ``caps/address``
    rejection reports the clamped census — while the success path
    (:func:`finish_batch`) reports the true census uncapped.
    """
    r.ok = False
    r.fail_reason = detail
    r.fail_code = code
    r.fail_op_index = op_index
    r.ops_len = ops_len
    r.total_qubits = min(num_qubits, MAX_QUBIT_ADDR)
    r.num_outs = num_outs
    r.num_bits = min(num_bits, MAX_BIT_ADDR)
    return r, registers, None


def finish_batch_averages(r: EvalReport) -> EvalReport:
    """Fill the per-live-shot averages from the summed counters."""
    denom = float(max(r.n_shots, 1))
    r.avg_tof = float(r.tot_tof) / denom
    r.avg_cliff = float(r.tot_cliff) / denom
    return r


def finish_batch(
    ch,
    r: EvalReport,
    tests,
    registers,
    snap,
    live_mask: int,
    *,
    ops_len: int,
    num_qubits: int,
    num_outs: int,
    tot_tof: int,
    tot_cliff: int,
    num_bits: int = 0,
    needed=None,
):
    """Shared success epilogue: census/counter fill, averages, schema
    pre-check, state extraction.

    Returns the executors' ``(EvalReport, registers, state)`` outcome triple;
    ``state`` is ``None`` when the register schema is invalid, in which case
    the coded schema failure rides on ``r.schema_fail`` for
    :func:`finish_from_batch` to apply. ``needed`` forwards a caller's
    already-computed :func:`needed_registers` set (the kernel path shares it
    with its bulk pre-read; deriving it is an O(shots x registers) pass).
    """
    r.ops_len = ops_len
    r.total_qubits = num_qubits
    r.num_outs = num_outs
    r.num_bits = num_bits
    r.tot_tof = tot_tof
    r.tot_cliff = tot_cliff
    finish_batch_averages(r)
    r.schema_fail = _validate_registers_coded(ch.schema(), registers)
    state = None
    if r.schema_fail is None:
        state = extract_batch_state(
            ch, tests, registers, snap, r.total_qubits, live_mask, needed=needed
        )
    return r, registers, state


def _expected_phase_mask(tests) -> int:
    # bytearray build for the same linear-vs-quadratic reason as
    # live_lane_mask.
    buf = bytearray((len(tests) + 7) // 8)
    for shot, tc in enumerate(tests):
        if tc is not None and tc.expected_phase:
            buf[shot >> 3] |= 1 << (shot & 7)
    return int.from_bytes(buf, "little")


def needed_registers(tests) -> set:
    """Return the indices of the registers some live shot's verdict actually reads
    (an ``expected`` entry that is not ``None``) — the only registers whose
    end-state values the verdicts consume. Shared by the kernel path's bulk
    pre-read (which must not read back every declared register) and
    :func:`extract_batch_state`, so the two can never disagree.
    """
    needed = set()
    for tc in tests:
        if tc is None:
            continue
        for ri, exp in enumerate(tc.expected):
            if exp is not None:
                needed.add(ri)
    return needed


# --- Schema validation + verdicts ---------------------------------------------------


def _validate_registers_coded(schema_regs, regs):
    """Run the register-schema checks with their stable ``schema/*`` codes.
    Returns ``(code, reason)`` or ``None``. Registers are ``(reg_kind,
    wires)`` pairs; the stream declares exactly the schema's registers, and
    each must match its schema entry's kind and width.
    """
    if len(regs) != len(schema_regs):
        return SCHEMA_COUNT, f"expected {len(schema_regs)} registers, got {len(regs)}"
    for i, ((kind, wires), spec) in enumerate(zip(regs, schema_regs)):
        want = WKIND_BIT if spec.kind == BIT else WKIND_QUBIT
        if kind != want:
            names = {WKIND_QUBIT: "Qubit", WKIND_BIT: "Bit"}
            return SCHEMA_KIND, (
                f"register {i} should be {names[want]}-kind, got {names.get(kind, kind)}"
            )
        if len(wires) != spec.width:
            return SCHEMA_WIDTH, (f"register {i} should be {spec.width} wide, got {len(wires)}")
    return None


def extract_batch_state(
    ch, tests, registers, snap, num_qubits: int, live_mask: int, *, needed=None
) -> tuple[int, int, dict[int, int]]:
    """Distill one batch's end-of-stream state into per-shot failure masks.

    Returns ``(wrong_output_mask, wrong_sign_mask, dirty)`` over the batch's local
    lanes: shots whose outputs fail the grader's check, shots with a wrong global
    sign, and per-qubit lane masks of dirty non-register qubits. ``tests`` is
    positional: ``None`` entries are dead lanes (skipped draws) and
    contribute to no mask — ``live_mask`` must already exclude them.

    ``snap`` needs ``phase`` (lane int) and ``qubits`` (list of lane ints);
    when it provides ``read_register_values(reg) -> [value per shot]`` (the C
    kernel's bulk readout) that path is used, otherwise the normative
    per-shot ``_get_register``. Both give identical values.
    ``needed`` (optional) forwards the caller's :func:`needed_registers` set
    so it is derived at most once per batch.
    """
    needed_set = needed_registers(tests) if needed is None else needed
    for ri in needed_set:
        if ri < len(registers) and registers[ri][0] == WKIND_BIT:
            # Grader misuse, not a circuit verdict: Bit registers are
            # input-only and carry no expected outputs.
            raise RuntimeError(
                f"grader error: register {ri} is Bit-kind but a test case "
                "carries an expected output for it (Bit registers are "
                "input-only)"
            )
    reader = getattr(snap, "read_register_values", None)
    vals = {}
    if reader is not None:
        for ri in needed_set:
            vals[ri] = reader(registers[ri])

    wrong_output_mask = 0
    for shot, tc in enumerate(tests):
        if tc is None:
            continue  # dead lane: no classical verdict
        read = [0] * len(registers)
        for ri, exp in enumerate(tc.expected):
            if exp is not None:
                if reader is not None:
                    read[ri] = vals[ri][shot]
                else:
                    read[ri] = _get_register(snap, registers[ri], shot)
        if not ch.check_shot(tc, read):
            wrong_output_mask |= 1 << shot

    wrong_sign_mask = (snap.phase ^ _expected_phase_mask(tests)) & live_mask

    # Only Qubit-kind registers exempt their wires from the ancilla scan —
    # bit ids live in a different space and must never shadow same-numbered
    # qubits.
    reg_qubits = set()
    for kind, wires in registers:
        if kind == WKIND_QUBIT:
            reg_qubits.update(wires)
    dirty: dict[int, int] = {}
    for q in range(min(num_qubits, len(snap.qubits))):
        if q in reg_qubits:
            continue
        v = snap.qubits[q] & live_mask
        if v:
            dirty[q] = v
    return wrong_output_mask, wrong_sign_mask, dirty


def _shot_indices_enabled() -> bool:
    """Return True when ``UNCOMPUTE_DEBUG_SHOT_INDICES=1`` asks for the
    indices of failed shots in the report. The default report omits them.
    """
    return os.environ.get("UNCOMPUTE_DEBUG_SHOT_INDICES") == "1"


_SHOT_INDICES_CAP = 512


def digest_batch_state(state: tuple[int, int, dict[int, int]], lane0: int) -> dict:
    """Distill one batch's local mask triple into an O(1)-size verdict digest.

    Under the dead-lane convention each shot belongs to exactly one batch, so
    end-of-run verdicts need only per-batch totals — no global-lane bigint is
    ever materialized (shifting per-batch masks to global lane positions and
    OR-merging them would be O(n_shots) bits per batch, quadratic across
    ~1000 batches). Fields:

    * ``wrong_output`` / ``wrong_sign`` — failed-shot counts (popcounts of
      the local masks);
    * ``failed`` — popcount of the per-shot union (wrong output, wrong sign,
      or a dirty qubit), the contribution to a ``Unified`` budget;
    * ``first_wrong_output`` — the **global** shot index of the batch's first
      wrong-output shot (``lane0`` + local lsb), or None;
    * ``dirty`` — per-qubit counts of live shots where the qubit is 1;
    * ``failed_shot_indices`` — only when debug shot indices are on.

    The masks stay batch-local; ``lane0`` applies only to the scalar
    ``first_wrong_output`` index.
    """
    wrong_output_mask, wrong_sign_mask, dirty = state
    failed = wrong_output_mask | wrong_sign_mask
    for v in dirty.values():
        failed |= v
    first = None
    if wrong_output_mask:
        first = lane0 + (wrong_output_mask & -wrong_output_mask).bit_length() - 1
    d = {
        "wrong_output": _popcount(wrong_output_mask),
        "wrong_sign": _popcount(wrong_sign_mask),
        "failed": _popcount(failed),
        "first_wrong_output": first,
        "dirty": {q: _popcount(v) for q, v in dirty.items()},
    }
    if _shot_indices_enabled():
        idx, m = [], failed
        while m and len(idx) < _SHOT_INDICES_CAP:
            b = m & -m
            idx.append(lane0 + b.bit_length() - 1)
            m ^= b
        d["failed_shot_indices"] = idx
    return d


def apply_verdicts(ch, r: EvalReport, digest: dict) -> EvalReport:
    """Apply the end-of-run verdict to a verdict digest (one batch or merged).

    Every executor and the shard merge call this one function, and budgets
    apply to the summed counts of the digest. Under ``Unified`` one budget
    covers the failed shots. Under ``STRICT`` the checks run in a fixed
    order (output, sign, ancilla), and the first failing check sets the
    verdict.
    """
    policy = resolve_failure_policy(ch)
    if digest.get("failed_shot_indices") is not None:
        r.failed_shot_indices = digest["failed_shot_indices"]
    r.wrong_output_shots = digest["wrong_output"]
    r.wrong_sign_shots = digest["wrong_sign"]
    dirty = digest["dirty"]
    r.dirty_qubits = len(dirty)

    def fail(code, reason, shot=None):
        r.ok = False
        r.fail_code = code
        r.fail_reason = reason
        r.fail_shot = shot
        return r

    if isinstance(policy, Unified):
        r.failed_shots = digest["failed"]
        if r.failed_shots > policy.budget:
            return fail(
                VERDICT_UNIFIED,
                f"{r.failed_shots} shots fail (wrong output, wrong sign, or a "
                f"dirty qubit); the task allows {policy.budget}",
            )
        return r

    if r.wrong_output_shots:
        first = digest["first_wrong_output"]
        return fail(
            VERDICT_CLASSICAL,
            f"wrong output on {r.wrong_output_shots} shots; the first is shot {first}",
            shot=first,
        )
    if r.wrong_sign_shots:
        return fail(
            VERDICT_PHASE,
            f"wrong global sign on {r.wrong_sign_shots} shots; the task allows 0",
        )
    if dirty:
        q = min(dirty)
        return fail(
            VERDICT_ANCILLA,
            f"qubit {q} is outside every register and is not 0 at the end of "
            f"the stream on {dirty[q]} shots",
        )
    return r


def finish_from_batch(ch, r: EvalReport, state) -> EvalReport:
    """Compose the final verdict from one batch outcome (or a merged digest).

    ``state`` is either a batch-local mask triple (single-batch runs, the
    executors' output) or an already-merged verdict digest (the shard merge).
    A schema-invalid stream arrives as ``state=None`` with the coded failure
    on ``r.schema_fail`` (computed once, by :func:`finish_batch`).
    """
    if not r.ok:
        return r
    if r.schema_fail is not None:
        r.ok = False
        r.fail_code, r.fail_reason = r.schema_fail
        return r
    digest = state if isinstance(state, dict) else digest_batch_state(state, 0)
    return apply_verdicts(ch, r, digest)
