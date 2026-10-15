"""uncompute.testing — the shared circuit verification scaffolding.

Everything here is verification-only: it checks emitted circuits and never
shapes them. A bug in this package can hide a failure but cannot change a
stream. ``uncompute.gadgets`` emits ops, so the two live in separate packages.

The helpers hold the steps that every circuit shares: deterministic
lane generation (``cases``), the cost check
(``assert_metrics``, ``pinned_cost``), the digest pin (``load_pinned``),
by-path module loading (``load_module``, ``load_sibling``),
task-root discovery (``task_dir``, ``load_contract``), the
emit-plus-summary tail (``emit_and_report``), and the cached C-library build that
``gen_constants.py`` searchers use (``build_clib``). The evaluator
(``tools/eval.py``) checks circuit correctness, so emission epilogues run
no Simulator replay: sampled certification over the full shot population
is statistically exhaustive for every failure class (outcomes
condition only phase gates, so the phase is affine in the draws), and a
replay battery would cost O(expanded ops) in Python on every build.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile


def cases(width, n=None, seed=1, edges=()):
    """Deterministic lane values for a ``width``-bit input register.

    Exhaustive ``range(2**width)`` when the domain fits in ``n`` lanes (or
    ``n`` is None); otherwise the ``edges`` cases first, padded to ``n``
    lanes with ``random.Random(seed)`` draws reduced mod ``2**width``.
    """
    dom = 1 << width
    if n is None or dom <= n:
        return list(range(dom))
    vals = [e % dom for e in edges]
    assert len(vals) <= n, "more edge cases than lanes"
    rng = random.Random(seed)
    while len(vals) < n:
        vals.append(rng.getrandbits(width))
    return vals


def pinned_cost(circuit_file):
    """Return the pinned cost of one circuit file, from the ``meta.json`` beside it.

    ``circuit.py`` reads ``meta["expected"]``. Any other circuit file reads
    ``meta["entries"][<file name>]``. Both have the shape
    ``{"toffoli": T, "qubits": Q}``. A missing pin raises (fail closed).
    """
    path = pathlib.Path(circuit_file).resolve()
    meta = _meta(path)
    if path.name == "circuit.py":
        return meta["expected"]
    return meta["entries"][path.name]


def _meta(path):
    return json.loads((path.parent / "meta.json").read_text())


def _io_qubits(b):
    """Count the wires of the declared Qubit registers. Bit registers hold
    classical inputs and have no qubit wires.
    """
    from ..compact import WKIND_QUBIT, Regs

    return sum(len(wires) for kind, wires in Regs(b._registers).registers if kind == WKIND_QUBIT)


def assert_metrics(b, circuit_file):
    """Run the cost check: the cost pin plus the
    ancilla-leak guard. Returns the emitted Toffoli count.

    The emitted Toffoli count and the peak qubits must equal
    ``pinned_cost(circuit_file)``; pass the circuit file's ``__file__``. At the
    end of the build only the declared Qubit-register wires may be live,
    plus any resident wires that a ``meta.json`` register band declares
    with an explicit ``"size"``.
    """
    want = pinned_cost(circuit_file)
    tof = b.phase_summary()["total_emitted_toffoli"]
    assert (tof, b.peak_qubits) == (want["toffoli"], want["qubits"]), (
        f"metrics drifted: emitted ({tof}, {b.peak_qubits}), "
        f"pinned ({want['toffoli']}, {want['qubits']})"
    )
    resident = sum(
        band.get("size", 0)
        for band in _meta(pathlib.Path(circuit_file).resolve()).get("registers", [])
    )
    io = _io_qubits(b) + resident
    assert b.active_qubits == io, f"ancilla leak: {b.active_qubits} wires live, registers hold {io}"
    return tof


def load_pinned(words, digest, width=4, algo="sha256", msg=None):
    """Assert the grader pin over a word table and return it: ``algo``
    (default sha256) of the words serialized little-endian ``width`` bytes
    each must equal ``digest`` (hex), else AssertionError(``msg`` or a
    default naming the computed digest).
    """
    data = b"".join(int(w).to_bytes(width, "little") for w in words)
    got = hashlib.new(algo, data).hexdigest()
    assert got == digest, msg or f"pinned table drifted: {algo}={got}"
    return words


def load_module(path, name, *, register=False):
    """Load the Python module at ``path`` under ``name`` — the
    spec_from_file_location dance, once.

    ``register=True`` puts the module in ``sys.modules[name]`` BEFORE
    executing it (removed again if execution raises), so self-references
    resolve during exec; pick a name unique in the process, and never
    ``"__main__"`` — loading under a plain name is what skips a circuit's
    emit epilogue. The repo-root ``sys.path`` bootstrap stays an inline
    preamble at the call site: it runs before ``uncompute`` is importable,
    so it structurally cannot live here.
    """
    path = pathlib.Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a module from {path}")
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        if register:
            sys.modules.pop(name, None)
        raise
    return mod


def load_sibling(anchor_file, fname="circuit.py", name="circuit", *, register=True):
    """Load ``fname`` from the directory of ``anchor_file`` (typically
    ``__file__``) — the checks-file ritual, anchored so it stays
    cwd-robust under the evaluator's tempdir runs.
    """
    return load_module(pathlib.Path(anchor_file).resolve().parent / fname, name, register=register)


def task_dir(anchor):
    """Return the task root: the nearest ancestor of ``anchor`` (typically
    ``__file__``) carrying ``grader.py``. Raises LookupError outside a
    task tree — fail closed, never a guessed directory.
    """
    p = pathlib.Path(anchor).resolve()
    for parent in p.parents:
        if (parent / "grader.py").exists():
            return parent
    raise LookupError(f"no grader.py ancestor above {p}")


def load_contract(anchor):
    """Load the task's ``contract.json`` as a dict (a missing file raises
    FileNotFoundError — fail closed). Destructuring and the cross-asserts
    against inline literals stay at the call site: they are the payload.
    """
    return json.loads((task_dir(anchor) / "contract.json").read_text())


def emit_and_report(b, extra=""):
    """Emit the stream and print the one canonical summary line, naming the
    path ``emit()`` actually writes ($UNCOMPUTE_OPS_PATH, else ./ops.bin).
    """
    tof = b.phase_summary()["total_emitted_toffoli"]
    peak = b.peak_qubits
    path = os.environ.get("UNCOMPUTE_OPS_PATH", "ops.bin")
    b.emit()
    print(
        f"emitted {path}: toffoli={tof} peak={peak} "
        f"tq_cost={tof * peak}" + (f" {extra}" if extra else "")
    )


def build_clib(src, lib, cflags=("-O2",)):
    """Compile the C file ``src`` into the shared library ``lib``; return ``lib``.

    A sidecar file ``<lib>.srchash`` holds the sha256 of the source that
    built ``lib``. The call recompiles only when ``lib`` or the sidecar is
    missing or the hash differs. The compiler is ``$CC`` (default ``cc``)
    with ``cflags``, ``-shared`` and ``-fPIC``. The build writes a temp
    file beside ``lib`` and renames it into place, so concurrent callers
    converge on one good binary. A missing compiler raises OSError and a
    failed compile raises CalledProcessError. A caller that falls back to
    Python catches both.
    """
    src, lib = pathlib.Path(src), pathlib.Path(lib)
    hash_path = lib.with_name(lib.name + ".srchash")
    src_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    if lib.exists() and hash_path.exists() and hash_path.read_text().strip() == src_hash:
        return lib
    lib.parent.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "cc")
    fd, tmp_name = tempfile.mkstemp(dir=lib.parent, suffix=".tmp")
    os.close(fd)
    tmp = pathlib.Path(tmp_name)
    try:
        subprocess.run(
            [cc, *cflags, "-shared", "-fPIC", "-o", str(tmp), str(src)],
            check=True,
            capture_output=True,
        )
        tmp.replace(lib)
    finally:
        tmp.unlink(missing_ok=True)
    hash_path.write_text(src_hash + "\n")
    return lib
