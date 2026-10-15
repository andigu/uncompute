#!/usr/bin/env python3
"""Certify every circuit entry of a task and rewrite its results file.

    python tools/certify.py <task>... [--jobs N]
    python tools/certify.py --all [--jobs N]

A task is a top-level ``tasks/<name>/`` with a ``grader.py``.
Its entries are, per circuit folder
``circuits/<name>/``, ``circuit.py`` plus the files named as keys of the
optional ``meta.json`` object ``"entries"``. Each entry runs through the
evaluator (``eval.evaluate``). A valid entry stores its op stream
xz-compressed beside it as ``<entry-stem>.ops.xz``. A failed entry stores
no stream (a stale one is removed) but still gets a results row.

``tasks/<name>/results.jsonl`` is rewritten in full: one JSON row per
entry, sorted by ``source``. The discovered circuit is the row whose ``source`` is
``circuits/discovered/circuit.py`` (``circuits/divsteps/circuit.py`` for
modinv-point-add). Re-check a stored stream with
``python tools/eval.py <name> --ops <entry-stem>.ops.xz``. Each row's ``h_ops``
is the SHA256 of its decompressed stream and the seed input of its draw.

Exit codes: 0 every entry OK, 1 some entry failed or a bad input, 2 usage
error.
"""

from __future__ import annotations

import json
import lzma
import shutil
import sys
import tempfile
import time
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval import evaluate

TASKS = REPO_ROOT / "tasks"
# Circuit wall-clock budget per entry: the big baseline streams take far
# longer than the evaluate() default.
CERTIFY_TIMEOUT = 7200.0


def all_tasks() -> list[str]:
    return sorted(p.parent.name for p in TASKS.glob("*/grader.py"))


def entries(task_dir: Path) -> list[Path]:
    """Every entry file of every circuit folder, in path order."""
    out = []
    for sdir in sorted((task_dir / "circuits").iterdir()):
        if not (sdir / "circuit.py").exists():
            continue  # shared helpers, caches
        meta_p = sdir / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        for name in ["circuit.py", *meta.get("entries", {})]:
            if not (sdir / name).exists():
                raise click.ClickException(f"{meta_p}: missing entry {name}")
            out.append(sdir / name)
    return out


def certify_entry(task: str, task_dir: Path, src: Path, jobs) -> dict:
    ops_xz = src.with_suffix(".ops.xz")
    with tempfile.TemporaryDirectory(prefix=f"uncompute-certify-{task}-") as td:
        capture = Path(td) / "stream.ops"
        res = evaluate(
            task,
            circuit=src,
            capture=capture,
            out_dir=Path(td) / "out",
            jobs=jobs,
            timeout=CERTIFY_TIMEOUT,
        )
        if res["valid"]:
            with capture.open("rb") as fin, lzma.open(ops_xz, "wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
        else:
            ops_xz.unlink(missing_ok=True)
    row = {
        "source": str(src.relative_to(task_dir)),
        "h_ops": res["h_ops"],
        "verdict": "OK" if res["valid"] else "FAIL",
        "toffoli": res["toffoli"],
        "qubits": res["qubits"],
        "tq_cost": res["tq_cost"],
        "clifford": res["clifford"],
        "ts": int(time.time()),
    }
    if not res["valid"]:
        row["fail_code"] = res["fail_code"]
    return row


def certify(task: str, jobs) -> bool:
    task_dir = TASKS / task
    if not (task_dir / "grader.py").exists():
        raise click.ClickException(f"not a task: {task}")
    rows = []
    for src in entries(task_dir):
        row = certify_entry(task, task_dir, src, jobs)
        print(
            f"{task}/{row['source']}: {row['verdict']} "
            f"{row['toffoli']} x {row['qubits']}"
            + (f" ({row['fail_code']})" if row["verdict"] != "OK" else "")
        )
        rows.append(row)
    rows.sort(key=lambda r: r["source"])
    (task_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return all(r["verdict"] == "OK" for r in rows)


@click.command()
@click.argument("tasks", nargs=-1)
@click.option("--all", "all_", is_flag=True, help="certify every task")
@click.option(
    "--jobs",
    type=int,
    default=None,
    help="worker processes for sharded evaluation (passed to tools/eval.py --jobs)",
)
def main(tasks, all_, jobs) -> None:
    """Certify circuit entries and rewrite tasks/<name>/results.jsonl."""
    if all_ == bool(tasks):
        raise click.UsageError("pass task names or --all (exactly one)")
    ok = True
    for task in all_tasks() if all_ else tasks:
        ok = certify(task, jobs) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
