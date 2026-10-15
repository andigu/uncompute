"""The evaluation protocol (normative).

A circuit program emits one op stream, and the evaluator interprets it
directly without materializing the expansion. Two executors do the work:
``evalcore.compact_reader`` (the normative Python reference) and the C
kernel (``kernel.run_compact_kernel_batch``), which gives identical results
and runs when built. ``UNCOMPUTE_KERNEL`` selects between them. A stream with
any magic other than ``MAGIC`` fails with ``load/bad-magic``.

Seed derivation. ``h_ops`` is the SHA256 of the raw op-stream bytes; the
caller hashes the whole stream before evaluation starts.

* ``seed_tests = SHAKE256(domain ‖ b"/tests/" ‖ h_ops).digest(32)``. Test
  shot ``i`` draws from its own XOF, ``SHAKE256(seed_tests ‖ u64le(i))``
  (:class:`ShotXof`), so any contiguous shot range can be generated alone.
* ``seed_rng = SHAKE256(domain ‖ b"/rng/" ‖ h_ops).digest(32)``. The k-th
  ``HMR`` or ``R`` in stream order reads counter k's lane-bit stream, whose
  128-byte block i is ``SHAKE256(seed_rng ‖ u64le(k) ‖ u64le(i))``. A batch
  seeks to its lanes at O(1) cost, so results are identical under any lane
  sharding.

Sharded evaluation. When ``n_shots`` exceeds the batch size
(:data:`DEFAULT_BATCH_SHOTS`, or ``--batch-shots``), the evaluator runs the
stream once per batch of lanes. Each batch generates its own tests and
slices the measurement RNG at its **global** lane offset. The merge sums the
counters and merges the per-batch verdict **digests**: each shot belongs to
exactly one batch, so failure counts sum and the first failing shot is a
minimum. Budgets apply to the merged totals, and the merged report equals
an unsharded run. Batches may run in worker processes (``jobs``); results
never depend on it.

Dead-lane convention. A skipped draw (``gen_test_indexed`` returning
``None``) keeps its lane: shot ``i`` always occupies **global lane i**, so
the measurement-RNG slicing never depends on earlier skips or on the batch
split. Dead lanes take no inputs, bill nothing, and enter no verdict. The
shot population ``n_shots`` (the cost denominator and budget base) counts
the live lanes.

The result is a structured report (``evalcore.report``), written as
``report.json`` beside ``tq_cost.json``. Stdout text is presentation only.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

# The batch-executor layer (executor.py) sits below both interpreters. This
# module re-exports it, so ``uncompute.evalcore.protocol`` is the one public
# import path.
from .executor import (  # noqa: F401  (re-exports)
    UNKNOWN_COUNT,
    EvalReport,
    LaneSim,
    apply_verdicts,
    begin_batch,
    digest_batch_state,
    extract_batch_state,
    fail_batch,
    finish_batch,
    finish_batch_averages,
    finish_from_batch,
    live_lane_mask,
)
from .loader import LoadError
from .report import (
    LOAD_BAD_MAGIC,
    LOAD_OP_INVALID,
    LOAD_TRUNCATED,
    build_report,
    make_fail,
    write_report_json,
    write_tq_cost,
)
from .task import (
    GraderConfigError,
    Unified,
    resolve_failure_policy,
)

MAGIC = b"OPSTRM01"  # the 8-byte stream magic

# Default lane-batch size for sharded evaluation. A multiple of 64 so batch
# boundaries fall on whole lane words AND whole XOF bytes in both executors.
# 8192 shards every large run (for example 100K shots) by default, so
# ``--jobs`` workers take part; a much larger default would leave such a run
# as one batch on one core. The cost per batch is one more read and parse
# of the stream. The merged report is identical at any batch size.
DEFAULT_BATCH_SHOTS = 8192


def n_batches(num_tests: int, batch_shots: int) -> int:
    """Make the sharding decision: how many lane batches ``num_tests`` shots take
    at ``batch_shots`` lanes per batch (1 = unsharded). The single authority —
    callers must not re-derive it.
    """
    return max(1, -(-num_tests // batch_shots))


# --- Seeding ---------------------------------------------------------------------


def ops_hash(path) -> bytes:
    """``h_ops``: the SHA256 of the raw (uncompressed) op-stream file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.digest()


def derive_seeds(domain: bytes, h_ops: bytes) -> tuple[bytes, bytes]:
    """Return ``(seed_tests, seed_rng)``. ``h_ops`` is the 32-byte op-stream
    SHA256.
    """
    seed_tests = hashlib.shake_256(domain + b"/tests/" + h_ops).digest(32)
    seed_rng = hashlib.shake_256(domain + b"/rng/" + h_ops).digest(32)
    return seed_tests, seed_rng


class Xof:
    """Grow-on-demand SHAKE256 reader: absorb bytes, then squeeze sequentially.

    ``hashlib.shake_256`` has no incremental squeeze; SHAKE output is
    prefix-stable, so a cursor over a geometrically regrown ``digest`` buffer
    serves sequential ``read`` calls exactly. Graders may use this for
    shot-local expansions inside ``gen_test`` (absorb the shot's draw, read
    as needed); absorbing after the first ``read`` is an error.
    """

    __slots__ = ("_buf", "_h", "_pos")

    def __init__(self, data: bytes = b""):
        self._h = hashlib.shake_256(data)
        self._buf = b""
        self._pos = 0

    def update(self, data: bytes) -> None:
        if self._pos or self._buf:
            raise RuntimeError("cannot absorb after squeezing has begun")
        self._h.update(data)

    def read(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._buf):
            self._buf = self._h.digest(max(end, 2 * len(self._buf), 256))
        out = self._buf[self._pos : end]
        self._pos = end
        return out


class ShotXof(Xof):
    """The per-shot test reader: ``SHAKE256(seed_tests ‖ u64le(shot))``."""

    __slots__ = ()

    def __init__(self, seed_tests: bytes, shot: int):
        Xof.__init__(self, seed_tests + shot.to_bytes(8, "little"))


def gen_batch_tests(ch, seed_tests: bytes, lo: int, hi: int) -> list:
    """Generate the test cases for shots ``[lo, hi)``, positionally.

    The returned list has exactly ``hi - lo`` entries: entry ``i`` is shot
    ``lo + i``'s test case, or ``None`` when the grader skipped that draw
    (dead-lane convention — see the module docstring). Every consumer of a
    positional test list must treat ``None`` entries as dead lanes.
    """
    return [ch.gen_test_indexed(shot, ShotXof(seed_tests, shot)) for shot in range(lo, hi)]


# --- Executor selection ------------------------------------------------------------


def _select_executor() -> str:
    """Return "kernel" or "reference", honoring ``UNCOMPUTE_KERNEL``.

    Default: the C kernel when its shared library is built, else the pure-
    Python reference. ``UNCOMPUTE_KERNEL=0`` forces the reference;
    ``UNCOMPUTE_KERNEL=1`` requires the kernel (RuntimeError if not built).
    Both produce identical results.
    """
    pref = os.environ.get("UNCOMPUTE_KERNEL")
    if pref == "0":
        return "reference"
    try:
        from .kernel import kernel_available

        available = kernel_available()
    except ImportError:
        available = False
    if pref == "1" and not available:
        raise RuntimeError("UNCOMPUTE_KERNEL=1 but the C kernel is not built; run: make kernel")
    return "kernel" if available else "reference"


def _read_magic(stream) -> bytes:
    buf = b""
    while len(buf) < 8:
        got = stream.read(8 - len(buf))
        if not got:
            break
        buf += got
    return buf


def _eval_batch(ch, body_stream, tests, seed_rng: bytes, lane0: int, executor: str):
    """Evaluate one lane batch of the (post-magic) stream body.

    Returns ``(EvalReport, registers, state)`` — ``state`` is the extracted
    mask triple, ``None`` when the report already failed or the register
    schema is invalid. Raises LoadError on structural rejection.
    """
    if executor == "kernel":
        from .kernel import run_compact_kernel_batch

        return run_compact_kernel_batch(ch, body_stream, tests, seed_rng, lane0)
    from .compact_reader import run_compact_batch

    return run_compact_batch(ch, body_stream, tests, seed_rng, lane0)


# --- Sharded evaluation --------------------------------------------------------------

# Fork-inherited context for worker processes (set immediately before the
# pool is created; graders are dynamically loaded modules, so they ride the
# fork instead of a pickle).
_SHARD_CTX: Any = None


def _init_shard_ctx(
    grader_file=None,
    stream_path=None,
    seed_tests=None,
    seed_rng=None,
    batch_shots=None,
    n_total=None,
    executor=None,
):
    """Worker initializer. Under the fork start method ``grader_file`` is
    None, because the context rides the fork. Under spawn
    (``UNCOMPUTE_MP=spawn``, for hosts where forked workers are unreliable)
    each fresh interpreter loads the grader module again from its file.
    """
    global _SHARD_CTX
    if grader_file is None:
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("uncompute_shard_grader", grader_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SHARD_CTX = (mod.TASK, stream_path, seed_tests, seed_rng, batch_shots, n_total, executor)


def _run_one_shard(b: int):
    """Evaluate batch ``b`` against the spooled stream file. Returns the
    batch outcome ``(report, registers, digest)``; LoadErrors propagate
    (they pickle intact out of worker processes).
    """
    ch, stream_path, seed_tests, seed_rng, batch_shots, n_total, executor = _SHARD_CTX
    lo = b * batch_shots
    hi = min(n_total, lo + batch_shots)
    # Positional test list: skipped draws stay as None (dead lanes), so the
    # batch always covers lanes [lo, hi) and len(tests) is the lane span.
    tests = gen_batch_tests(ch, seed_tests, lo, hi)
    with Path(stream_path).open("rb") as f:
        magic = _read_magic(f)
        if magic != MAGIC:
            raise LoadError("stream: bad magic", LOAD_BAD_MAGIC)
        r, registers, state = _eval_batch(ch, f, tests, seed_rng, lo, executor)
    # Digest the batch-local masks here, in the worker: the merge consumes
    # only O(1)-size totals (and the pickle back to the parent stays small).
    # Registers ride along only for batch 0 — every batch of the same stream
    # declares identical registers, so the merge needs just one copy.
    if state is not None:
        state = digest_batch_state(state, lo)
    return r, registers if b == 0 else None, state


def _merge_shards(outcomes):
    """Merge ordered per-batch outcomes into (report, registers, digest).

    Counters are summed; census/ops/registers come from batch 0 (the stream
    is the same; later batches carry ``None`` registers). Verdict inputs
    arrive as **digests** (``digest_batch_state``, applied at each batch's
    global lane offset before merging): each shot belongs to exactly one
    batch, so failure counts sum, per-qubit dirty counts sum, and the first
    wrong-output shot is the min over batches — no global-lane mask is
    ever materialized (that would be O(n_shots) bits per batch, quadratic
    across batches). ``n_shots`` accumulates only live lanes, so the shot
    population stays the accepted count. A failed batch report (validation
    failure) wins as-is — every batch of the same stream fails identically,
    so the first one is authoritative. A schema-invalid batch (``digest`` is
    ``None``) carries the coded failure on ``report.schema_fail``; the merge
    propagates it and ``finish_from_batch`` applies it.
    """
    merged = EvalReport()
    digest = {
        "wrong_output": 0,
        "wrong_sign": 0,
        "failed": 0,
        "first_wrong_output": None,
        "dirty": {},
        "failed_shot_indices": None,
    }
    registers = None
    for r, regs, d in outcomes:
        if not r.ok:
            return r, regs, None
        if registers is None:
            registers = regs
            merged.ops_len = r.ops_len
            merged.total_qubits = r.total_qubits
            merged.num_outs = r.num_outs
            merged.num_bits = r.num_bits
        merged.tot_tof += r.tot_tof
        merged.tot_cliff += r.tot_cliff
        merged.n_shots += r.n_shots
        if d is None:
            if merged.schema_fail is None:
                merged.schema_fail = r.schema_fail
        else:
            digest["wrong_output"] += d["wrong_output"]
            digest["wrong_sign"] += d["wrong_sign"]
            digest["failed"] += d["failed"]
            if d["first_wrong_output"] is not None and (
                digest["first_wrong_output"] is None
                or d["first_wrong_output"] < digest["first_wrong_output"]
            ):
                digest["first_wrong_output"] = d["first_wrong_output"]
            for q, c in d["dirty"].items():
                digest["dirty"][q] = digest["dirty"].get(q, 0) + c
            if d.get("failed_shot_indices") is not None:
                if digest["failed_shot_indices"] is None:
                    digest["failed_shot_indices"] = []
                room = 512 - len(digest["failed_shot_indices"])
                if room > 0:
                    digest["failed_shot_indices"].extend(d["failed_shot_indices"][:room])
    finish_batch_averages(merged)
    return merged, registers, digest if merged.schema_fail is None else None


def _scan_max_outcome(stream_path):
    """Largest outcome id a stream can address: parse-only O(stream) affine
    maxima over full loop ranges (zero-count loops skipped; data gating
    ignored — a safe overestimate). Never runs an op.
    """
    from .compact_reader import _SPACE_OUT, _CompactVM, _Loop

    class _Scan(_CompactVM):
        hi = -1

        def _note(self, rec, counts=()):
            for j, (_name, _pos, space) in enumerate(rec.fields):
                if space == _SPACE_OUT:
                    k = j + (1 if rec.gated else 0)
                    self.hi = max(
                        self.hi,
                        rec.cur[k]
                        + sum(max(s, 0) * (c - 1) for s, c in zip(rec.slot_strides[k], counts)),
                    )

        _exec_op = _note  # FLAT path: the base _seg_flat parses, we note

        def _walk(self, node, counts):
            for it in node.items if node.count else ():
                (self._walk if isinstance(it, _Loop) else self._note)(it, counts + (node.count,))

        def _seg_loop(self):
            self._walk(self._parse_loop(0), ())

        def _seg_end(self):
            self._u64("END expanded_op_count")  # no count/trailing checks

    with Path(stream_path).open("rb") as f:
        f.read(len(MAGIC))
        vm = _Scan(f, (), None)
        vm.run()
    return vm.hi


def _evaluate(ch, stream, seed_tests, seed_rng, *, batch_shots, jobs, stream_path, executor):
    """Evaluate the whole shot population, sharding when needed.

    ``stream`` is positioned right after the magic. Returns
    ``(EvalReport, registers, state)``; raises LoadError on structural
    rejection and RuntimeError on operator errors (unspooled pipe).

    Multi-worker runs get an outcome-lane RAM preflight: a parse-only scan
    bounds the kernel's per-worker outcome-lane allocation and may lower
    ``jobs`` (never below 1) to fit ``UNCOMPUTE_EVAL_RAM_MB`` (default 6144).
    Advisory only — verdicts, seeds, batching, and costs are untouched;
    a failed scan proceeds uncapped (guard, not gate).
    """
    n_total = ch.num_tests
    batches = n_batches(n_total, batch_shots)

    if batches == 1:
        tests = gen_batch_tests(ch, seed_tests, 0, n_total)
        return _eval_batch(ch, stream, tests, seed_rng, 0, executor)

    if stream_path is None:
        raise RuntimeError(
            "sharded evaluation (n_shots > batch size) needs a re-readable "
            "stream file; the CLI spools pipe streams automatically"
        )
    global _SHARD_CTX
    _SHARD_CTX = (ch, str(stream_path), seed_tests, seed_rng, batch_shots, n_total, executor)
    if jobs > 1:
        # Outcome-lane RAM preflight: each worker's kernel allocates
        # next_pow2(max_out_id+1) x (batch_shots/64) x 8 bytes of outcome
        # lanes; cap jobs so all workers fit UNCOMPUTE_EVAL_RAM_MB.
        try:
            n_out = _scan_max_outcome(stream_path) + 1
            lane_bytes = (1 << max(n_out - 1, 0).bit_length()) * (batch_shots // 64) * 8
            mb = os.environ.get("UNCOMPUTE_EVAL_RAM_MB", "6144")
            if lane_bytes * jobs > int(mb) << 20:
                jobs = max(1, (int(mb) << 20) // lane_bytes)
                print(
                    f"** outcome-lane preflight: {n_out} outcomes -> "
                    f"{lane_bytes >> 20} MB/worker; capping jobs at "
                    f"{jobs} (UNCOMPUTE_EVAL_RAM_MB={mb})",
                    file=sys.stderr,
                )
        except Exception as e:  # noqa: BLE001  (advisory preflight; the eval proceeds uncapped)
            print(
                f"warning: outcome-lane preflight failed ({e}); proceeding uncapped",
                file=sys.stderr,
            )
    try:
        outcomes = []
        if jobs > 1:
            import multiprocessing

            mp_method = os.environ.get("UNCOMPUTE_MP", "fork")
            try:
                mp = multiprocessing.get_context(mp_method)
            except ValueError:
                mp = None
            init_args = (None,)
            if mp is not None and mp_method != "fork":
                # spawn/forkserver: the ctx cannot ride a fork — workers
                # re-load the grader from its source file. sys.modules may
                # not know the grader (spec_from_file_location loaders don't
                # register), so fall back to a grader method's co_filename,
                # which always names the defining file.
                mod = sys.modules.get(type(ch).__module__)
                grader_file = getattr(mod, "__file__", None)
                if grader_file is None:
                    fn = getattr(ch, "gen_test", None)
                    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
                    grader_file = getattr(code, "co_filename", None)
                if grader_file is not None and not Path(grader_file).exists():
                    grader_file = None
                if grader_file is None:
                    print(
                        f"warning: UNCOMPUTE_MP={mp_method} needs a "
                        "file-loadable grader; running sequentially",
                        file=sys.stderr,
                    )
                    mp = None
                else:
                    init_args = (
                        grader_file,
                        str(stream_path),
                        seed_tests,
                        seed_rng,
                        batch_shots,
                        n_total,
                        executor,
                    )
            if mp is None:
                print(
                    f"warning: {mp_method} start method unavailable; running batches sequentially",
                    file=sys.stderr,
                )
                jobs = 1
            else:
                # ProcessPoolExecutor, NOT multiprocessing.Pool: Pool.map
                # blocks FOREVER when a worker dies (a crashed kernel worker
                # would leave the whole eval hung at 0% CPU).
                # BrokenProcessPool turns a dead worker into a loud failure
                # that names the unfinished shards.
                from concurrent.futures import ProcessPoolExecutor, as_completed
                from concurrent.futures.process import BrokenProcessPool

                with ProcessPoolExecutor(  # ty: ignore[no-matching-overload]
                    max_workers=min(jobs, batches),
                    mp_context=mp,
                    initializer=_init_shard_ctx,
                    initargs=init_args,
                ) as pool:
                    futs = {pool.submit(_run_one_shard, b): b for b in range(batches)}
                    got = {}
                    try:
                        for fut in as_completed(futs):
                            got[futs[fut]] = fut.result()
                    except BrokenProcessPool as e:
                        missing = sorted(b for b in range(batches) if b not in got)
                        raise RuntimeError(
                            "shard worker process died (batches not "
                            f"completed: {missing}); a crashed worker must "
                            "fail the run, not hang it"
                        ) from e
                    outcomes = [got[b] for b in range(batches)]
        if jobs <= 1:
            outcomes = [_run_one_shard(b) for b in range(batches)]
    finally:
        _SHARD_CTX = None

    return _merge_shards(outcomes)


# --- Entry point -----------------------------------------------------------------------


def run_task(
    ch,
    manifest_dir: str,
    stream,
    h_ops: bytes,
    *,
    out_dir: str | None = None,
    emit_json: bool = False,
    batch_shots: int = DEFAULT_BATCH_SHOTS,
    jobs: int = 1,
    stream_path=None,
    circuit_meta_fn=None,
) -> tuple[int, dict | None]:
    """Evaluate an op stream. Returns ``(exit_code, report_dict)``.

    ``stream`` is a binary file object; ``h_ops`` is the SHA256 of its bytes
    (the seed derivation input). ``stream_path`` names a re-readable copy of the
    stream on disk — required whenever ``ch.num_tests > batch_shots``
    (sharded evaluation re-reads the stream per batch); ``jobs`` runs batches
    in forked worker processes. ``circuit_meta_fn`` (optional) is called once
    after evaluation and may return an advisory dict embedded as
    ``report["circuit_meta"]`` — the CLI uses it to collect the circuit's
    ``UNCOMPUTE-META`` stdout line.

    Human text goes to stdout, or to stderr when ``emit_json`` (the CLI then
    prints the report object on stdout). Exit codes: 0 valid, 1 rejected,
    2 operator error (report is None).
    """
    human = sys.stderr if emit_json else sys.stdout

    def p(msg: str = "") -> None:
        print(msg, file=human)

    try:
        policy = resolve_failure_policy(ch)
    except GraderConfigError as e:
        print(f"!! grader config error: {e}", file=sys.stderr)
        return 2, None
    if batch_shots <= 0 or batch_shots % 8:
        print(f"!! batch size must be a positive multiple of 8, got {batch_shots}", file=sys.stderr)
        return 2, None
    try:
        executor = _select_executor()
    except RuntimeError as e:
        print(f"!! {e}", file=sys.stderr)
        return 2, None

    dir_ = out_dir if out_dir is not None else manifest_dir

    p(f"=== uncompute evaluator: {ch.name} ===")
    # Loud executor line: the pure-Python fallback is bit-identical but
    # ~1000x slower — silent selection would let a many-core eval crawl
    # unnoticed. UNCOMPUTE_KERNEL=1 hard-requires the
    # kernel; this line makes the default path auditable at a glance.
    p(
        f"executor: {executor}"
        + (
            "  (WARNING: pure-Python reference — orders of magnitude slower; "
            "build the C kernel with `make kernel`)"
            if executor == "reference"
            else ""
        )
        + "\n"
    )
    seed_tests, seed_rng = derive_seeds(ch.seed_domain, h_ops)

    batches = n_batches(ch.num_tests, batch_shots)
    t0 = time.time()
    p(f"-- randomized tests ({ch.num_tests} shots) --")
    if batches > 1:
        p(
            f"   sharded: {batches} batches of {batch_shots} lanes"
            + (f", {jobs} worker(s)" if jobs > 1 else "")
        )

    def _finish_fail(load_fail):
        detail = str(load_fail)
        total_s = time.time() - t0
        circuit_meta = circuit_meta_fn() if circuit_meta_fn is not None else None
        report = build_report(
            "FAIL",
            h_ops,
            fail=make_fail(load_fail.code, detail, load_fail.op_index),
            shots=ch.num_tests,
            total_s=total_s,
            circuit_meta=circuit_meta,
        )
        p(f"!! could not load stream: {detail}")
        write_report_json(dir_, report)
        return 1, report

    magic = _read_magic(stream)
    if len(magic) < 8:
        return _finish_fail(LoadError(f"stream: too short ({len(magic)} bytes)", LOAD_TRUNCATED))
    if magic != MAGIC:
        return _finish_fail(LoadError("stream: bad magic", LOAD_BAD_MAGIC))

    try:
        r, _registers, state = _evaluate(
            ch,
            stream,
            seed_tests,
            seed_rng,
            batch_shots=batch_shots,
            jobs=jobs,
            stream_path=stream_path,
            executor=executor,
        )
    except LoadError as e:
        return _finish_fail(e)
    except RuntimeError as e:
        print(f"!! {e}", file=sys.stderr)
        return 2, None

    r = finish_from_batch(ch, r, state)
    total_s = time.time() - t0
    circuit_meta = circuit_meta_fn() if circuit_meta_fn is not None else None

    fail = None
    if not r.ok:
        fail = make_fail(
            r.fail_code or LOAD_OP_INVALID,
            r.fail_reason or "(no detail)",
            r.fail_op_index,
            shot=r.fail_shot,
        )
    report = build_report(
        "OK" if r.ok else "FAIL",
        h_ops,
        fail=fail,
        avg_tof=r.avg_tof,
        avg_cliff=r.avg_cliff,
        qubits=r.total_qubits,
        outcomes=r.num_outs,
        bits=r.num_bits,
        ops=r.ops_len,
        shots=r.n_shots,
        wrong_output_shots=r.wrong_output_shots,
        wrong_sign_shots=r.wrong_sign_shots,
        dirty_qubits=r.dirty_qubits,
        failed_shots=r.failed_shots,
        total_s=total_s,
        circuit_meta=circuit_meta,
    )
    fsi = getattr(r, "failed_shot_indices", None)
    if fsi is not None:
        # debug aid (UNCOMPUTE_DEBUG_SHOT_INDICES=1): global indices of failed shots
        report["failed_shot_indices"] = sorted(fsi)

    def table(rows) -> None:
        for label, value in rows:
            p(f"  {label:<22}: {value}")

    rows = [
        ("expanded ops", r.ops_len),
        ("qubits", r.total_qubits),
        ("outcomes", r.num_outs),
        ("live shots", r.n_shots),
        ("wrong-output shots", r.wrong_output_shots),
        ("wrong-sign shots", r.wrong_sign_shots),
        ("dirty qubits", r.dirty_qubits),
    ]
    if isinstance(policy, Unified):
        rows.append(("failed shots", f"{r.failed_shots} (budget {policy.budget})"))
    table(rows)

    write_report_json(dir_, report)

    if not r.ok:
        if report.get("failed_shot_indices"):
            p(f"  failed shot indices   : {report['failed_shot_indices']}")
        p(f"\n!! verdict FAIL: {report['fail']['detail']}")
        return 1, report

    p(f"\n=== cost ({ch.name}) ===")
    table(
        [
            ("avg Toffoli count", f"{r.avg_tof:.3f}"),
            ("avg Clifford count", f"{r.avg_cliff:.3f}"),
            ("total Toffoli count", f"{r.tot_tof} over {r.n_shots} shots"),
            ("total Clifford count", r.tot_cliff),
            ("peak qubits", r.total_qubits),
            ("TQ-cost", report["metrics"]["tq_cost"]),
        ]
    )

    write_tq_cost(dir_, report)
    p("\n=== verdict OK ===")
    return 0, report
