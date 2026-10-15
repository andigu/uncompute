#!/usr/bin/env python3
"""uncompute eval CLI — streamed, single-pass evaluation.

Default (pipe mode): runs the task's discovered circuit as a subprocess
with ``UNCOMPUTE_OPS_PATH`` pointed at a FIFO, spools the op stream to a file
(the ``--capture`` path when given, else a temp file) while hashing it, then
evaluates the file. Without ``--circuit`` the CLI runs
``circuits/discovered/circuit.py`` (``circuits/divsteps/circuit.py`` for
modinv-point-add). File mode (``--ops``) audits a stored stream: a raw
stream, or an xz-compressed one such as a circuit's ``<entry-stem>.ops.xz``.
The CLI decompresses a ``.xz`` file to a temp file before the evaluation.

Seeds: the seed input ``h_ops`` is the SHA256 of the raw op-stream
bytes, so the evaluator must see the whole stream before the first shot. The
same stream gives the same draw in pipe mode and file mode.

    python tools/eval.py <task> [--circuit PATH] [--out DIR]
    python tools/eval.py <task> --ops circuit.ops.xz
    python tools/eval.py <task> --json          # report on stdout

Outputs land in ``--out DIR`` when given: ``report.json`` (the structured
verdict) and ``tq_cost.json`` on a valid run. Without ``--out`` they go to a
temp dir that is removed afterwards, so nothing lands in the task dir. ``--json`` prints the report
object to stdout (plus ``circuit_stdout`` in pipe mode) and moves the human
text to stderr — nothing downstream may parse evaluator stdout text.
``--capture PATH`` stores the raw op-stream bytes at PATH while evaluating —
tools/certify.py uses it to store each entry's ``.ops.xz``.

Sharding: when the task's shot count exceeds ``--batch-shots`` (default
8192), the stream is evaluated in independent lane batches and the merged
report is identical to an unsharded run: each batch re-reads the spooled
stream file. ``--jobs N`` runs batches in worker processes (default:
min(16, cpu count) when not given explicitly).

Circuit meta channel: the Builder prints one ``UNCOMPUTE-META {...}`` line
(compact JSON of its phase summary) to stdout after streaming; the last such
line is embedded as ``report["circuit_meta"]`` (advisory; absent if none).

``UNCOMPUTE_KERNEL`` selects the executor (1 = require the C kernel, 0 = force
the pure-Python reference).

Importable: ``from eval import evaluate`` is the single structured wrapper —
it runs this CLI with ``--json`` in a subprocess and maps the report onto one
flat result dict (TQ-cost, metrics, failure reason, and per-phase counts
from the circuit's UNCOMPUTE-META line). That function alone decides validity
from the exit code and recovers a failure reason from the stderr tail.

Exit codes: 0 OK, 1 validation failure, 2 operator error.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import lzma
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace

import click

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The task-relative entry file of each task's discovered circuit. The
# modinv-point-add task names its discovered circuits by method instead.
PRIMARY_SOURCES = ("circuits/discovered/circuit.py", "circuits/divsteps/circuit.py")


# Circuit subprocesses run with the cycle collector OFF (refcounting still
# reclaims everything the Builder drops). The recorded op list is millions of
# acyclic objects; CPython's gen-2 GC re-traverses all of them on every
# collection, turning large builds quadratic. Harness-side so the authoring
# surface never has to mention it, and so emission budgets mean the same
# thing on every path.
# argv is sanitized to [circuit_path] before run_path: the circuit must see
# the same argv as a bare `python circuit.py` invocation. Leaving the path
# in sys.argv[1] is not cosmetic — a circuit that reads a positional
# out-path argument would read its OWN SOURCE PATH as the ops destination
# and overwrite itself at emission.
GC_OFF_SHIM = (
    "import gc, runpy, sys; gc.disable(); "
    "p = sys.argv[1]; sys.argv[:] = [p]; "
    "runpy.run_path(p, run_name='__main__')"
)

from uncompute.evalcore.protocol import (
    DEFAULT_BATCH_SHOTS,
    ops_hash,
    run_task,
)
from uncompute.evalcore.task import (
    GraderConfigError,
    resolve_failure_policy,
)


def _tail(text: str, n: int = 8) -> str:
    """Return the last ``n`` lines of ``text`` (whitespace-trimmed) — the one
    stderr-tail helper for circuit/eval failure diagnostics.
    """
    return "\n".join(text.strip().splitlines()[-n:])


def _default_circuit(task_dir):
    """Default circuit resolution when ``--circuit`` is absent: the
    first of ``PRIMARY_SOURCES`` that exists, else None.
    """
    for src in PRIMARY_SOURCES:
        if (task_dir / src).exists():
            return (task_dir / src).resolve()
    return None


def load_grader(task: str):
    """Load ``tasks/<name>/grader.py`` and return its TASK object."""
    grader_path = REPO_ROOT / "tasks" / task / "grader.py"
    if not grader_path.exists():
        print(f"!! no grader for '{task}' ({grader_path} missing)", file=sys.stderr)
        raise SystemExit(2)
    spec = importlib.util.spec_from_file_location(f"uncompute_grader_{task}", grader_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "TASK"):
        print(f"!! {grader_path} does not expose TASK", file=sys.stderr)
        raise SystemExit(2)
    return mod.TASK


def _open_fifo_reader(fifo: pathlib.Path, proc: subprocess.Popen):
    """Open the FIFO read end, unblocking if the circuit dies before writing."""
    holder = {}

    def _open():
        try:
            holder["f"] = fifo.open("rb")  # the caller owns and closes it
        except OSError as e:
            holder["err"] = e

    t = threading.Thread(target=_open, daemon=True)
    t.start()
    while t.is_alive():
        t.join(timeout=0.05)
        if t.is_alive() and proc.poll() is not None:
            # Circuit exited without opening the pipe: connect a dummy writer
            # so the blocked open() returns, then report the circuit failure.
            try:
                fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                os.close(fd)
            except OSError:
                pass
            t.join()
            return None
    return holder.get("f")


def parse_meta(stdout_text: str):
    """Parse the LAST ``UNCOMPUTE-META {...}`` line of circuit stdout, or None.

    This is the one UNCOMPUTE-META parser; keep the semantics here, in one place.
    """
    meta = None
    for line in stdout_text.splitlines():
        if line.startswith("UNCOMPUTE-META "):
            try:
                parsed = json.loads(line[len("UNCOMPUTE-META ") :])
            except ValueError:
                continue
            if isinstance(parsed, dict):
                meta = parsed
    return meta


class _CircuitHandle:
    """The running circuit subprocess + its spooled stdout/stderr."""

    def __init__(self, proc, out_path, err_path):
        self.proc = proc
        self._out_path = out_path
        self._err_path = err_path
        self.rc = None
        self.timed_out = False
        self.stdout = ""
        self.stderr = ""
        self._done = False

    def finish(self, timeout: float) -> None:
        if self._done:
            return
        self._done = True
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            self.timed_out = True
        self.rc = self.proc.returncode
        for attr, path in (("stdout", self._out_path), ("stderr", self._err_path)):
            try:
                with path.open(errors="replace") as f:
                    setattr(self, attr, f.read())
            except OSError:
                pass

    def stderr_tail(self, n=8) -> str:
        return _tail(self.stderr, n)


def add_options(options):
    """Apply a list of ``click.option(...)`` decorators to a command
    function.
    """

    def _apply(f):
        for opt in reversed(options):
            f = opt(f)
        return f

    return _apply


# The circuit-run and output flags of this CLI.
FORWARDED_OPTIONS = [
    click.option(
        "--circuit",
        default=None,
        help="circuit file to run (default: circuits/discovered/circuit.py, "
        "else circuits/divsteps/circuit.py)",
    ),
    click.option(
        "--out",
        default=None,
        help="write report.json/tq_cost.json to DIR (default: a temp dir, removed afterwards)",
    ),
    click.option(
        "--capture",
        default=None,
        help="store the raw op-stream bytes (exactly as received) at PATH while evaluating",
    ),
    click.option(
        "--batch-shots",
        type=int,
        default=DEFAULT_BATCH_SHOTS,
        help="lane-batch size for sharded evaluation (multiple of 8; "
        f"default {DEFAULT_BATCH_SHOTS})",
    ),
    click.option(
        "--jobs",
        type=int,
        default=None,
        help="worker processes for sharded batches (default: "
        "min(16, cpu count); results never depend on this)",
    ),
    click.option(
        "--timeout", type=float, default=3600.0, help="circuit wall-clock limit (pipe mode)"
    ),
]


@click.command()
@click.argument("task")
@add_options(FORWARDED_OPTIONS)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="print the report object (JSON) to stdout; human text goes to stderr",
)
@click.option(
    "--ops",
    default=None,
    help="evaluate a stored stream (raw, or .xz) instead of running the circuit",
)
@click.option(
    "--liveness",
    is_flag=True,
    help="also write liveness.json beside the circuit from "
    "this run's evaluation (one evaluation, two outputs). "
    "Runs the circuit in this process under the "
    "profiler's recording Builder, as tools/liveness_profile.py does. "
    "Pipe mode only",
)
def _cli(
    task,
    circuit,
    out,
    capture,
    batch_shots,
    jobs,
    timeout,
    as_json,
    ops,
    liveness,
):
    """Evaluate a circuit or a stored op stream for TASK."""
    args = SimpleNamespace(
        task=task,
        circuit=circuit,
        out=out,
        json=as_json,
        capture=capture,
        batch_shots=batch_shots,
        jobs=jobs,
        timeout=timeout,
        ops=ops,
        liveness=liveness,
    )
    return _run(args)


def main(argv=None) -> int:
    """Plain, non-exception-raising entry point: returns an int, so click's
    default sys.exit() on invocation is bypassed via standalone_mode=False.
    """
    try:
        return _cli.main(args=argv, standalone_mode=False, prog_name="eval.py")
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except click.exceptions.Exit as e:
        return e.exit_code


def _run(args) -> int:
    if args.jobs is None:
        args.jobs = min(16, os.cpu_count() or 1)

    out_dir = pathlib.Path(args.out).resolve() if args.out is not None else None
    if out_dir is None:
        # No --out: the run artifacts land in a temp dir removed afterwards,
        # so nothing is written into the tracked task dir.
        with tempfile.TemporaryDirectory(prefix=f"uncompute-eval-{args.task}-out-") as td:
            args.out = td
            return _run(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    task = load_grader(args.task)
    try:
        resolve_failure_policy(task)  # contradictions fail at grader load
    except GraderConfigError as e:
        print(f"!! grader config error: {e}", file=sys.stderr)
        return 2
    task_dir = REPO_ROOT / "tasks" / args.task
    manifest_dir = str(task_dir)
    run_kwargs = {
        "out_dir": str(out_dir) if out_dir is not None else None,
        "emit_json": args.json,
        "batch_shots": args.batch_shots,
        "jobs": args.jobs,
    }

    def emit_report(report, circuit_stdout=None, stage=None):
        if args.json and report is not None:
            obj = dict(report)
            if circuit_stdout is not None:
                obj["circuit_stdout"] = circuit_stdout[-65536:]
            if stage is not None:
                obj["stage"] = stage  # which stage failed, as data (evaluate())
            print(json.dumps(obj))

    if args.ops is not None:
        return _run_file_mode(args, task, manifest_dir, run_kwargs, emit_report)

    # ---- pipe/liveness mode: resolve the circuit (default: the first
    # existing PRIMARY_SOURCES entry) ------------------------------------------
    if args.circuit:
        circuit_path = pathlib.Path(args.circuit).resolve()
    else:
        circuit_path = _default_circuit(task_dir)
        if circuit_path is None:
            print(
                f"!! no circuit given and no default found for "
                f"{args.task}: pass --circuit, or provide "
                f"one of {', '.join(PRIMARY_SOURCES)}",
                file=sys.stderr,
            )
            return 2
    if not circuit_path.exists():
        print(f"!! circuit not found: {circuit_path}", file=sys.stderr)
        return 2

    if args.liveness:
        return _run_liveness_mode(
            args, task, task_dir, manifest_dir, run_kwargs, emit_report, circuit_path
        )

    return _run_pipe_mode(args, task, manifest_dir, run_kwargs, emit_report, circuit_path)


def _run_file_mode(args, task, manifest_dir, run_kwargs, emit_report):
    """``--ops``: audit a stored stream instead of running the circuit."""
    if args.liveness:
        print(
            "!! --liveness needs circuit mode (the profile is read "
            "from the built Builder's wire_log; a stored stream "
            "carries no such record)",
            file=sys.stderr,
        )
        return 2
    ops = pathlib.Path(args.ops)
    with tempfile.TemporaryDirectory(prefix=f"uncompute-eval-{args.task}-") as td:
        if ops.suffix == ".xz":
            raw = pathlib.Path(td) / ops.stem
            with lzma.open(ops) as fin, raw.open("wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
            ops = raw
        if args.capture is not None:
            shutil.copyfile(ops, args.capture)
        with ops.open("rb") as f:
            rc, report = run_task(
                task, manifest_dir, f, ops_hash(ops), stream_path=ops, **run_kwargs
            )
    emit_report(report)
    return rc


def _run_liveness_mode(args, task, task_dir, manifest_dir, run_kwargs, emit_report, circuit_path):
    """``--liveness``: one evaluation, two outputs.

    The circuit runs in this process, because liveness.json comes from the
    Builder's wire_log, which lives on the Builder instance and would not
    cross a subprocess boundary. The emitted bytes then go through the
    normal file-mode evaluation (the same seeds, verdict, and results row as
    pipe mode for the same stream), and liveness.json comes from that one
    evaluation.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import liveness_profile

    with tempfile.TemporaryDirectory(prefix=f"uncompute-eval-{args.task}-") as td:
        ops_path = pathlib.Path(td) / "ops.bin"
        prev_ops = os.environ.get("UNCOMPUTE_OPS_PATH")
        os.environ["UNCOMPUTE_OPS_PATH"] = str(ops_path)
        try:
            writer = liveness_profile.run_annotated(circuit_path, pathlib.Path(td))
        finally:
            if prev_ops is None:
                os.environ.pop("UNCOMPUTE_OPS_PATH", None)
            else:
                os.environ["UNCOMPUTE_OPS_PATH"] = prev_ops
        if writer is None:
            print("!! --liveness: circuit emitted no stream", file=sys.stderr)
            return 2
        if args.capture is not None:
            shutil.copyfile(ops_path, args.capture)
        h_ops = ops_hash(ops_path)
        with ops_path.open("rb") as f:
            rc, report = run_task(task, manifest_dir, f, h_ops, stream_path=ops_path, **run_kwargs)
        emit_report(report)
        if rc != 0:
            return rc  # invalid runs write no liveness artifact
        phases, r, activity, intervals_out, pooling = liveness_profile.recorded_sections(writer)
        m = (report or {}).get("metrics") or {}
        got = (int(m.get("toffoli") or 0), m.get("qubits"))
        if (r["toffoli"], r["peak"]) != got:
            print(
                f"!! --liveness: recorded {r['toffoli']}x{r['peak']} "
                f"!= certified {got[0]}x{got[1]}",
                file=sys.stderr,
            )
            return 1
        meta_p = circuit_path.parent / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        registers = liveness_profile.register_bands(meta, writer, r["persistent"])
        prov = {
            "circuit_sha256": hashlib.sha256(circuit_path.read_bytes()).hexdigest()[:16],
            "mode": "builder",
            "variant": str(circuit_path.parent.relative_to(task_dir))
            if circuit_path.is_relative_to(task_dir)
            else str(circuit_path.parent),
            "verified": {
                "eval_cert": {
                    "tq_cost": m.get("tq_cost"),
                    "metrics": {"toffoli": int(m["toffoli"]), "qubits": m["qubits"]},
                },
            },
            "h_ops": h_ops.hex(),
        }
        out = liveness_profile.assemble_payload(
            args.task, prov, registers, phases, r, activity, intervals_out, pooling
        )
        lp = liveness_profile.write_liveness(circuit_path.parent / "liveness.json", out)
        print(f"wrote {lp}")
    return rc


def _run_pipe_mode(args, task, manifest_dir, run_kwargs, emit_report, circuit_path):
    """Default mode: run the circuit as a subprocess, stream its op stream
    through a FIFO into a spool file, then evaluate the spool.
    """
    with tempfile.TemporaryDirectory(prefix=f"uncompute-eval-{args.task}-") as td:
        tdp = pathlib.Path(td)
        fifo = tdp / "ops.fifo"
        os.mkfifo(fifo)
        env = dict(os.environ)
        env["UNCOMPUTE_OPS_PATH"] = str(fifo)
        env["PYTHONPATH"] = str(REPO_ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        out_path = tdp / "circuit.stdout"
        err_path = tdp / "circuit.stderr"
        circuit_argv = [sys.executable, "-c", GC_OFF_SHIM, str(circuit_path)]
        with out_path.open("w") as out_f, err_path.open("w") as err_f:
            proc = subprocess.Popen(
                circuit_argv,
                cwd=td,  # circuit scratch: any stray files land here, not the repo
                env=env,
                stdout=out_f,
                stderr=err_f,
            )
        handle = _CircuitHandle(proc, out_path, err_path)
        try:
            f = _open_fifo_reader(fifo, proc)
            if f is None:
                handle.finish(args.timeout)
                print(
                    f"!! circuit exited (rc={handle.rc}) before emitting ops:\n"
                    f"{handle.stderr_tail()}",
                    file=sys.stderr,
                )
                emit_report({}, handle.stdout, stage="circuit")
                return 1

            def meta_fn():
                # Stream is at EOF here: the circuit has closed the FIFO write
                # end; wait for it and read its stdout for the meta line.
                handle.finish(args.timeout)
                return parse_meta(handle.stdout)

            # h_ops seeds the draw, so the whole stream must arrive before
            # evaluation: spool the pipe to a file (the --capture path
            # doubles as spool), hashing it on the way.
            spool = pathlib.Path(args.capture or tdp / "stream.ops")
            try:
                sp = spool.open("wb")  # closed by the `with f, sp` below
            except OSError as e:
                f.close()
                print(f"!! cannot write --capture {spool}: {e}", file=sys.stderr)
                return 2
            h = hashlib.sha256()
            with f, sp:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
                    sp.write(chunk)
            with spool.open("rb") as sf:
                rc, report = run_task(
                    task,
                    manifest_dir,
                    sf,
                    h.digest(),
                    stream_path=spool,
                    circuit_meta_fn=meta_fn,
                    **run_kwargs,
                )

            handle.finish(args.timeout)
            if handle.timed_out:
                print("!! circuit did not exit after stream EOF; killed", file=sys.stderr)
                return 2
            if handle.rc != 0:
                print(
                    f"!! circuit exited nonzero (rc={handle.rc}) after streaming:\n"
                    f"{handle.stderr_tail()}",
                    file=sys.stderr,
                )
                emit_report(report if report is not None else {}, handle.stdout, stage="circuit")
                return 1
            emit_report(report, handle.stdout)
            return rc
        finally:
            if proc.poll() is None:
                proc.kill()


# ---- importable structured wrapper ------------------------------------------

DEFAULT_EVALUATE_TIMEOUT = 600.0
# Grace added to the eval.py subprocess hard kill on top of the circuit budget.
EVALUATE_KILL_GRACE = 120.0


def evaluate(
    task: str,
    *,
    circuit=None,
    ops=None,
    capture=None,
    out_dir=None,
    timeout: float = DEFAULT_EVALUATE_TIMEOUT,
    batch_shots=None,
    jobs=None,
    env=None,
) -> dict:
    """Build and evaluate ``task`` through this CLI (``--json``); return one result dict.

    ``ops`` is the file-mode passthrough (CLI ``--ops``): evaluate a stored
    op stream instead of running the circuit.

    The result dict has the fields (valid, tq_cost, toffoli,
    qubits, clifford, ops, n_shots, avg_toffoli, fail_reason, fail_code,
    stage, phases, peak_phase, circuit_stdout, first_failing_shot, h_ops,
    report). ``tq_cost`` is the TQ-cost. ``toffoli`` is the floored average;
    ``avg_toffoli`` keeps full precision. ``phases``/``peak_phase`` come from
    the circuit's UNCOMPUTE-META stdout line (``report["circuit_meta"]``).

    ``capture`` (a path) stores the raw op stream — tools/certify.py
    compresses it into the entry's ``.ops.xz``. ``out_dir`` routes the
    report artifacts (report.json/tq_cost.json) to that
    directory; without it they go to a temp dir that is removed afterwards.
    The circuit itself always runs in a scratch cwd.

    ``env`` (a mapping) replaces the environment of the eval.py subprocess —
    and therefore of the circuit subprocess it spawns, which inherits it.
    Default ``None`` inherits
    ``os.environ`` unchanged; set task-local eval overrides there
    before calling (or include them in ``env``).
    """
    import math
    import time

    result: dict = {
        "task": task,
        "valid": False,
        "tq_cost": None,
        "toffoli": None,
        "qubits": None,
        "clifford": None,
        "ops": None,
        "avg_toffoli": None,
        "n_shots": None,
        "fail_reason": None,
        "fail_code": None,
        "first_failing_shot": None,
        "stage": None,
        "report": None,
        "h_ops": None,
        "phases": None,
        "peak_phase": None,
        "circuit_stdout": "",
    }
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        task,
        "--json",
        "--timeout",
        str(timeout),
    ]
    if circuit is not None:
        cmd += ["--circuit", str(circuit)]
    if ops is not None:
        cmd += ["--ops", str(ops)]
    if capture is not None:
        cmd += ["--capture", str(capture)]
    if batch_shots is not None:
        cmd += ["--batch-shots", str(batch_shots)]
    if jobs is not None:
        cmd += ["--jobs", str(jobs)]

    if out_dir is not None:
        cmd += ["--out", str(out_dir)]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=(dict(env) if env is not None else None),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout + EVALUATE_KILL_GRACE,
        )
    except subprocess.TimeoutExpired:
        result["stage"] = "eval"
        result["fail_reason"] = f"eval.py timed out after {timeout + EVALUATE_KILL_GRACE}s"
        return result
    result["elapsed_s"] = round(time.time() - t0, 2)

    # --json contract: stdout is the report object (circuit/human text never
    # lands there). A run that died before the evaluator reported prints
    # either nothing (operator error) or a stage-only stub — the failing
    # stage travels as data, never parsed back out of the human prose.
    report = None
    raw = proc.stdout.strip()
    if raw:
        try:
            report = json.loads(raw)
        except ValueError:
            report = None
    if report is None:
        result["stage"] = "eval"
        result["fail_reason"] = (
            f"eval.py rc={proc.returncode}, no structured report:\n{_tail(proc.stderr or '')}"
        )
        return result

    result["circuit_stdout"] = report.pop("circuit_stdout", "") or ""
    stage = report.pop("stage", None)
    if not report:
        # Stage-only stub: the run died before the evaluator produced a report
        # (e.g. the circuit exited before emitting ops).
        result["stage"] = stage or "eval"
        result["fail_reason"] = (
            f"eval.py rc={proc.returncode}, no structured report:\n{_tail(proc.stderr or '')}"
        )
        return result
    result["report"] = report
    result["h_ops"] = report.get("h_ops")
    m = report.get("metrics") or {}
    result["qubits"] = m.get("qubits")
    result["clifford"] = m.get("clifford")
    result["ops"] = m.get("ops")
    result["avg_toffoli"] = m.get("toffoli")
    result["n_shots"] = m.get("shots")
    meta = report.get("circuit_meta") or {}
    result["phases"] = meta.get("phases")
    result["peak_phase"] = meta.get("peak_phase")

    # The process exit code stays authoritative: a circuit that exits nonzero
    # after streaming makes eval.py return 1 even when the stream verdict was
    # OK — such a run must not certify.
    if proc.returncode == 0 and report.get("verdict") == "OK":
        result["valid"] = True
        result["tq_cost"] = m.get("tq_cost")
        result["toffoli"] = math.floor(m.get("toffoli") or 0.0)
    else:
        result["stage"] = stage or "eval"
        fail = report.get("fail") or {}
        result["fail_code"] = fail.get("code")
        if fail.get("shot") is not None:
            result["first_failing_shot"] = fail["shot"]
        if fail.get("detail"):
            result["fail_reason"] = fail["detail"]
        else:
            result["fail_reason"] = (
                f"eval.py rc={proc.returncode} without a fail record "
                f"(circuit exit after streaming?):\n{_tail(proc.stderr or '')}"
            )
    return result


if __name__ == "__main__":
    sys.exit(main())
