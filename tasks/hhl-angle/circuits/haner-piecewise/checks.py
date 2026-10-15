#!/usr/bin/env python3
"""Verification battery for the haner-piecewise circuit.

Evidence only: nothing here is needed to build or emit the circuit.
circuit.py never imports this file, so editing a check does not change
the circuit or its test draw.  This file holds the end-to-end check
``exhaustive_fixed_point_error``: it runs the quantized datapath
(Horner ladder + round) against the float arcsin(1/(2x)) reference on
every accepted input.  The datapath model is gen_constants.datapath, the
model the search accepts each interval with.

The standalone run checks constants.json, the tables that
gen_constants.py fits and circuit.build() loads, so the tables this check
certifies are the tables the build emits from.  It evaluates the quantized datapath on all 2,031,616 accepted inputs,
asserting worst == circuit.WORST_ULP at X = 66,097 and worst <= the
contract band.

RUN: uv run python checks.py
"""

import json
import pathlib

import numpy as np

from uncompute.testing import load_sibling

G = load_sibling(__file__, "gen_constants.py", name="hhl_haner_gen", register=False)
S = G.S

# The datapath surface under test, loaded from the sibling circuit.py by
# path (robust to any cwd).
DOMAIN_RAW_LO, DOMAIN_RAW_HI = S.DOMAIN_RAW_LO, S.DOMAIN_RAW_HI
INPUT_FRAC_BITS, OUTPUT_FRAC_BITS = S.INPUT_FRAC_BITS, S.OUTPUT_FRAC_BITS


def target_function(x: np.ndarray) -> np.ndarray:
    """arcsin(1/(2x)), with x = X/2^16."""
    return np.arcsin(0.5 / x)


def exhaustive_fixed_point_error(qtab, stops):
    """Evaluate the quantized datapath on every accepted input; return the
    worst |actual - ideal| in output units and its argmax X.
    """
    raw = np.arange(DOMAIN_RAW_LO, DOMAIN_RAW_HI, dtype=np.uint64)
    real = raw.astype(np.float64) / (1 << INPUT_FRAC_BITS)
    # The ideal reference uses the grader's pinned rounding rule: half
    # away from zero (C ``round`` semantics).  The values are positive, so the
    # rule is floor(v) + [frac(v) >= 1/2].
    v = target_function(real) * (1 << OUTPUT_FRAC_BITS)
    f = np.floor(v)
    ideal = (f + (v - f >= 0.5)).astype(np.int64)
    actual = np.empty(len(raw), dtype=np.int64)

    start = 0
    for quantized, stop in zip(qtab, stops):
        out = G.datapath(quantized, raw[start:stop])
        if out is None:
            raise AssertionError("negative final output")
        actual[start:stop] = out
        start = stop

    assert start == len(raw)
    residual = actual - ideal
    index = int(np.argmax(np.abs(residual)))
    return int(abs(residual[index])), int(raw[index])


def main():
    # Per-interval exclusive end offsets into the accepted-input range.
    tables = json.loads((pathlib.Path(__file__).resolve().parent / "constants.json").read_text())
    qtab, bounds = tables["qtab"], tables["bounds"]
    stops = [bound - DOMAIN_RAW_LO for bound in bounds] + [DOMAIN_RAW_HI - DOMAIN_RAW_LO]
    worst, worst_x = exhaustive_fixed_point_error(qtab, stops)
    print(f"exhaustive fixed-point check: worst={worst} ULP at X={worst_x} (band {S.ALLOWED_ULP})")
    assert (worst, worst_x) == (S.WORST_ULP, 66097)
    assert worst <= S.ALLOWED_ULP
    print(
        f"checks OK: exhaustive check of the fitted tables over "
        f"{DOMAIN_RAW_HI - DOMAIN_RAW_LO:,} accepted inputs"
    )


if __name__ == "__main__":
    main()
