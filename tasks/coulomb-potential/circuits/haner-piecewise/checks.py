#!/usr/bin/env python3
"""Evidence battery for the haner-piecewise baseline.

Evidence only: nothing here is needed to build or emit the circuit,
and circuit.py never imports this file, so editing a check does not
change the circuit or its test draw.  This file holds the classical datapath
model and its exhaustive check:

  * achievable_squared_distances — the task domain (all 139,637
    sums of three squares of |d| <= 255), taken from the task-level
    reference.py and re-matched against the grader's DOM_DIGEST pin;
  * exhaustive_fixed_point_error — the quantized datapath on every
    achievable n2: selects each input's interval by the integer
    thresholds, as the circuit does, proves the final Horner word
    positive (V0 < 2^25, so the 25-bit accumulate window is exact), matches the ideal table against the
    grader's TBL_DIGEST pin, and returns the worst error + argmax.
    Each Horner step is gen_constants.horner_step, the model of the HRS
    signed pre-truncated multiplier the search accepts each interval with.

main() checks the tables the circuit loads from constants.json, so the
checked tables are the tables the emitted circuit loads.  It pins the
check result: worst = WORST_ULP output units at n2 = WORST_N2, within
the task band of 32.

RUN: uv run python checks.py
"""

import numpy as np

from uncompute.testing import load_module, load_pinned, load_sibling, task_dir

S = load_sibling(__file__)
G = load_sibling(__file__, "gen_constants.py", name="coulomb_haner_gen", register=False)
R = load_module(task_dir(__file__) / "reference.py", "coulomb_potential_reference")

WORD_BITS = S.WORD_BITS
FRAC_BITS = S.FRAC_BITS
assert (G.WORD_BITS, G.FRAC_BITS) == (WORD_BITS, FRAC_BITS)
INTERVALS = S.INTERVALS
OUT_SCALE_BITS = S.OUT_SCALE_BITS
N2_MAX = S.N2_MAX
N_ACHIEVABLE = S.N_ACHIEVABLE
DOM_DIGEST = S.DOM_DIGEST
TBL_DIGEST = S.TBL_DIGEST

WORST_ULP = 32  # exhaustive-check pin
WORST_N2 = 3  # exhaustive-check argmax pin


def achievable_squared_distances() -> np.ndarray:
    """Return the task domain (all 139,637 sums of three squares of
    |d| <= 255, ascending), taken from the task-level reference.py
    and re-matched against the grader's DOM_DIGEST pin.
    """
    values = np.asarray(R.DOMAIN, dtype=np.uint64)
    assert len(values) == N_ACHIEVABLE
    load_pinned(values, DOM_DIGEST, msg="achievable domain drifted from grader")
    return values


def exhaustive_fixed_point_error(qtab, thresholds):
    """Evaluate the quantized datapath on every achievable n2; return the
    worst |actual - ideal| in output units and its argmax n2.  Also matches
    the ideal table against the grader's TBL_DIGEST pin.
    """
    mask_int = (1 << WORD_BITS) - 1
    sign = np.uint64(1 << (WORD_BITS - 1))
    n2 = achievable_squared_distances()
    # frexp is exact on integers < 2^53: value = m * 2^exp with m in [0.5, 1)
    exponent = (np.frexp(n2.astype(np.float64))[1] - 1).astype(np.int64)
    assert np.all(n2 >> exponent.astype(np.uint64) == 1)
    u_word = n2 << (FRAC_BITS - exponent).astype(np.uint64)
    assert np.all(u_word >> np.uint64(FRAC_BITS) == 1)

    # the circuit's label: the count of thresholds at or below u_word
    label = np.searchsorted(np.asarray(thresholds, dtype=np.uint64), u_word, side="right")

    parity = exponent & 1
    h = exponent >> 1
    actual = np.full(len(n2), -1, dtype=np.int64)
    for p in (0, 1):
        for i in range(INTERVALS):
            selected = (parity == p) & (label == i)
            if not np.any(selected):
                continue
            quantized = qtab[(p, i)]
            words = u_word[selected]
            value = np.full(len(words), np.uint64(quantized[-1] & mask_int), dtype=np.uint64)
            for coefficient in reversed(quantized[:-1]):
                value = G.horner_step(value, words, coefficient)
            if np.any(value >= sign):
                raise AssertionError("negative final output")
            actual[selected] = (value >> h[selected].astype(np.uint64)).astype(np.int64)

    assert np.all(actual >= 0)
    # The ideal table is reference.py's oracle (ideal_f); the scale
    # cross-assert ties the circuit's OUT_SCALE_BITS to the reference's
    # C_SCALE, and the whole table must reproduce the grader's SHA-256 pin.
    assert R.C_SCALE == 1 << OUT_SCALE_BITS
    ideal = np.fromiter((R.ideal_f(int(v)) for v in n2), dtype=np.int64, count=len(n2))
    load_pinned(ideal, TBL_DIGEST, msg="ideal table drifted from grader")

    residual = actual - ideal
    index = int(np.argmax(np.abs(residual)))
    return int(abs(residual[index])), int(n2[index])


def main():
    worst, worst_n2 = exhaustive_fixed_point_error(*S.load_tables())
    print(
        f"exhaustive fixed-point check: worst={worst} ULP at n2={worst_n2} (band {S.ALLOWED_ULP})"
    )
    assert (worst, worst_n2) == (WORST_ULP, WORST_N2)
    assert worst <= S.ALLOWED_ULP
    print("checks: PASS")


if __name__ == "__main__":
    main()
