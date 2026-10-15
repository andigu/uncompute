"""Structured eval results.

The evaluator's verdict is one JSON-able dict — the *report* — with stable
failure **codes**; message text is presentation and may change freely.
Nothing downstream may parse evaluator stdout.

Report shape::

    {
      "verdict":  "OK" | "FAIL",
      "fail":     null | {"code": str, "expanded_op_index": int|null,
                          "detail": str,
                          "shot": int},         # present only when the failure
                                                # has a known first-failing shot
      "metrics":  {"toffoli": float, "clifford": float, "qubits": int,
                   "outcomes": int, "bits": int, "ops": int, "shots": int,
                   "tq_cost": int|null},     # null unless verdict == OK
      "counters": {"wrong_output_shots": int, "wrong_sign_shots": int,
                   "dirty_qubits": int, "failed_shots": int},
      "h_ops":    hex str,                   # bytes fed to the seed derivation
      "timing":   {"total_s": float},
      "circuit_meta": {...},                 # present only when the circuit
                                             # printed a UNCOMPUTE-META line;
                                             # advisory
    }

``metrics.ops`` is the **expanded, post-gating** op count
(the :mod:`uncompute.ops` docstring). ``metrics.toffoli`` /
``metrics.clifford`` are the executed averages (full-precision floats;
gates bill at their promoted kind, so both are constants of the stream).
``tq_cost`` is the TQ-cost ``floor(toffoli) * qubits``; it is ``null`` on
FAIL, because an invalid run has no cost.

This module also owns the output writers: ``report.json`` (always) and
``tq_cost.json`` (verdict OK only; ``json.dumps(..., indent=2)`` canonical
form).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from .util import U64_MAX

# --- Stable failure codes ------------------------------------------------------
#
# load/*     : the stream was rejected while parsing (framing / structure).
# validate/* : a per-expanded-op check failed (carries expanded_op_index).
# caps/*     : resource caps breached while the census maxima grew.
# schema/*   : end-of-stream register-schema validation failed.
# verdict/*  : the circuit ran but failed a validity check or budget.

LOAD_BAD_MAGIC = "load/bad-magic"
LOAD_TRUNCATED = "load/truncated"
LOAD_COUNT_MISMATCH = "load/count-mismatch"
LOAD_OP_INVALID = "load/op-invalid"  # structural op-shape rejection
LOAD_SEGMENT = "load/segment"  # segment-level parse error

VALIDATE_ALIASING = "validate/aliasing"
VALIDATE_ID_RANGE = "validate/id-range"  # resolved id < 0 or >= 2^32
VALIDATE_TABLE_INDEX = "validate/table-index"  # data-gate bit index out of range

CAPS_ADDRESS = "caps/address"

SCHEMA_COUNT = "schema/count"
SCHEMA_WIDTH = "schema/width"
SCHEMA_KIND = "schema/kind"

VERDICT_CLASSICAL = "verdict/classical"
VERDICT_PHASE = "verdict/phase"
VERDICT_ANCILLA = "verdict/ancilla"
VERDICT_UNIFIED = "verdict/unified-budget"


def tq_cost_from(avg_tof: float, qubits: int) -> int:
    """floor of the f64 average times peak qubits, saturating."""
    return min(math.floor(avg_tof) * qubits, U64_MAX)


def build_report(
    verdict: str,
    h_ops: bytes,
    *,
    fail: dict | None = None,
    avg_tof: float = 0.0,
    avg_cliff: float = 0.0,
    qubits: int = 0,
    outcomes: int = 0,
    bits: int = 0,
    ops: int = 0,
    shots: int = 0,
    wrong_output_shots: int = 0,
    wrong_sign_shots: int = 0,
    dirty_qubits: int = 0,
    failed_shots: int = 0,
    total_s: float = 0.0,
    circuit_meta: dict | None = None,
) -> dict:
    """Assemble the report dict. Key order is the documented shape."""
    report = {
        "verdict": verdict,
    }
    report["fail"] = fail
    report["metrics"] = {
        "toffoli": avg_tof,
        "clifford": avg_cliff,
        "qubits": qubits,
        "outcomes": outcomes,
        "bits": bits,
        "ops": ops,
        "shots": shots,
        "tq_cost": tq_cost_from(avg_tof, qubits) if verdict == "OK" else None,
    }
    report["counters"] = {
        "wrong_output_shots": wrong_output_shots,
        "wrong_sign_shots": wrong_sign_shots,
        "dirty_qubits": dirty_qubits,
        "failed_shots": failed_shots,
    }
    report["h_ops"] = h_ops.hex()
    report["timing"] = {"total_s": total_s}
    if circuit_meta is not None:
        report["circuit_meta"] = circuit_meta
    return report


def make_fail(
    code: str, detail: str, expanded_op_index: int | None = None, shot: int | None = None
) -> dict:
    """Build the report's fail object. ``shot`` is
    the first failing shot's global index and is present only when the
    failure kind has one (``verdict/classical``). Consumers must not parse it
    out of ``detail``.
    """
    fail = {
        "code": code,
        "expanded_op_index": expanded_op_index,
        "detail": detail,
    }
    if shot is not None:
        fail["shot"] = shot
    return fail


# --- Writers -------------------------------------------------------------------


def _safe_write(dir_: str, name: str, text: str) -> None:
    """Best-effort file write: writing report.json or tq_cost.json must never
    turn a finished evaluation into a crash — failures degrade to a stderr
    warning.
    """
    try:
        with (Path(dir_) / name).open("w") as f:
            f.write(text)
            f.write("\n")
    except OSError as e:
        print(f"warning: failed to write {name}: {e}", file=sys.stderr)


def write_report_json(dir_: str, report: dict) -> None:
    _safe_write(dir_, "report.json", json.dumps(report, indent=2))


def write_tq_cost(dir_: str, report: dict) -> None:
    """tq_cost.json for a VALID run (canonical ``json.dumps(..., indent=2)``
    byte form).
    """
    m = report["metrics"]
    body = {"tq_cost": m["tq_cost"]}
    body["metrics"] = {
        "toffoli": math.floor(m["toffoli"]),
        "qubits": m["qubits"],
    }
    _safe_write(dir_, "tq_cost.json", json.dumps(body, indent=2))
