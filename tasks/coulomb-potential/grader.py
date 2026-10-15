"""Grader for `coulomb-potential`.

Spec: |r_i>|r_j>|k> -> |r_i>|r_j>|k + F(r_i,r_j) mod 2^25> with
F = round(2^24/||r_i - r_j||) = round(2^24 * rsqrt(n2)),
n2 = sum_c (r_ic - r_jc)^2 in [1, 195075], r_j = r_i domain-rejected.
Six 8-qubit coordinate registers (preserved exactly) + a 25-qubit additive
accumulator k (randomized per shot).

Banded check, zero budget: a shot passes iff registers 0-5 round-trip
exactly AND |centered_25((k_out - k_in) - F(n2))| <= EPS = 32; every tested
shot must pass (failure_policy STRICT). The band is the whole approximation
allowance: an error enters the application's phase in the worst case, so no
allowance for failed shots sits on top of it.

Test draws: pinned three-mode distribution, 11 bytes/shot (see decode_draw).
Grader oracle: one f64 sqrt per shot (round-half floor 6.56e-7 — exhaustive).

Run `python3 grader.py` for the self-test (the pinned unit-test anchors;
~1 min: three exhaustive 139,637-value sweeps).
"""

from __future__ import annotations

import json
import math
import pathlib

from uncompute.evalcore import (
    QUBIT,
    RegSpec,
    Task,
    TestCase,
)

# Bits per coordinate (fixed; see README §3.2).
B = 8
# Accumulator width: 25 bits, so F(1) = 2^24 fits without saturation.
M = 25
# Output scale: F(n2) = round(C * rsqrt(n2)) with C = 2^24; F(1) = 2^24 exactly.
C_SCALE = 1 << 24
# Joint correctness band (centered residue, inclusive): the only error allowance.
EPS = 32
# Maximum achievable norm-squared: 3 * 255^2.
N2_MAX = 195_075
# Number of achievable n2 values (sums of three squares of |d| <= 255).
N_ACHIEVABLE = 139_637

MOD = 1 << M
MASK25 = MOD - 1

# Instance pins (SHA-256, LE u32 streams).
TBL_DIGEST = "3de8b2b0b3063d0f46aa4cbcf528173b73d4cf0d28d0a2fac27491238d804439"
DOM_DIGEST = "f52f7e1b110369374daea81136a21c36eca11364ccb2653b725d4f6ba6b1e8cc"

# contract.json is the circuit-facing copy of the pins above; the inline
# derivation stays authoritative and this assert (nothing more) keeps the
# shared file honest.
_CONTRACT = json.loads((pathlib.Path(__file__).resolve().parent / "contract.json").read_text())
assert {k: v for k, v in _CONTRACT.items() if not k.startswith("_")} == {
    "B": B,
    "M": M,
    "C_SCALE": C_SCALE,
    "EPS": EPS,
    "N2_MAX": N2_MAX,
    "N_ACHIEVABLE": N_ACHIEVABLE,
    "TBL_DIGEST": TBL_DIGEST,
    "DOM_DIGEST": DOM_DIGEST,
}, "contract.json drifted from the grader's inline pins"


def _round_half_away(v):
    """Round half away from zero (C ``round`` semantics); exact for v >= 0 (floor and
    v - floor(v) are exact in f64). Python's round() is half-to-even.
    """
    f = math.floor(v)
    return f + 1 if v - f >= 0.5 else f


def ideal_f(n2):
    """F(n2) = round(2^24 / sqrt(n2)) — one f64 sqrt, the grader oracle."""
    assert 1 <= n2 <= N2_MAX
    return _round_half_away(C_SCALE / math.sqrt(n2))


def norm2(ri, rj):
    """Squared distance of two points of the 256^3 grid."""
    return sum((a - b) * (a - b) for a, b in zip(ri, rj))


def ideal_k_out(ri, rj, k_in):
    """(k + F(n2)) mod 2^25. Callers guarantee ri != rj."""
    return (k_in + ideal_f(norm2(ri, rj))) % MOD


def decode_draw(blk):
    """One randomized-test draw from an 11-byte block.

    Bytes 0-2 -> r_i; 3 -> mode; 4-6 -> material; 7-10 LE, low 25 -> k.
    Returns None on the r_j = r_i skip.
    """
    ri = [blk[0], blk[1], blk[2]]
    mode = blk[3]
    mat = blk[4:7]
    k = int.from_bytes(blk[7:11], "little") & MASK25
    if mode & 1 == 0:
        # uniform: r_j = material (bulk coverage, full carry activity).
        rj = list(mat)
    else:
        e = (mode >> 1) & 7
        if e <= 6:
            # near: r_j,c = r_i,c + delta_c mod 256, delta_c = sign-extension
            # of the low e+2 bits of material_c.
            w = e + 2
            rj = []
            for c in range(3):
                v = mat[c] & ((1 << w) - 1)
                if v >= 1 << (w - 1):
                    v -= 1 << w
                rj.append((ri[c] + v) % 256)
        else:
            # far: opposite 32-wide corner bands per coordinate.
            ri = [224 | (x & 0x1F) for x in ri]
            rj = [x & 0x1F for x in mat]
    if rj == ri:
        return None  # singularity: redraw-free skip, deterministic indexing
    return ri, rj, k


class CoulombPotential(Task):
    name = "coulomb-potential"
    seed_domain = b"coulomb-potential"
    num_tests = 9024
    # Zero tolerance (STRICT default): the ±EPS band in check_shot is the
    # whole approximation contract; any shot outside it fails the run.

    def schema(self):
        return [
            RegSpec(width=B, kind=QUBIT),  # 0: rix
            RegSpec(width=B, kind=QUBIT),  # 1: riy
            RegSpec(width=B, kind=QUBIT),  # 2: riz
            RegSpec(width=B, kind=QUBIT),  # 3: rjx
            RegSpec(width=B, kind=QUBIT),  # 4: rjy
            RegSpec(width=B, kind=QUBIT),  # 5: rjz
            RegSpec(width=M, kind=QUBIT),  # 6: k accumulator
        ]

    def gen_test(self, xof):
        # Constant 11-byte draw per call (skip included).
        blk = xof.read(11)
        dec = decode_draw(blk)
        if dec is None:
            return None
        ri, rj, k = dec
        inputs = [ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], k]
        expected = inputs[:6] + [ideal_k_out(ri, rj, k)]
        return TestCase(inputs=inputs, expected=expected, expected_phase=False)

    def check_shot(self, tc, read):
        # Per-shot verdict: coords exact, k within the centered mod-2^25 band.
        if any(read[i] != tc.inputs[i] for i in range(6)):
            return False
        k_in = tc.inputs[6]
        ri = tc.inputs[0:3]
        rj = tc.inputs[3:6]
        f = ideal_f(norm2(ri, rj))
        d = (read[6] - k_in - f) % MOD
        if d > MOD // 2:
            d -= MOD
        return abs(d) <= EPS


TASK = CoulombPotential()


if __name__ == "__main__":
    import hashlib

    # 1. domain_and_table_digests (exhaustive).
    ach = bytearray(N2_MAX + 1)
    for dx in range(256):
        x2 = dx * dx
        for dy in range(dx, 256):
            s2 = x2 + dy * dy
            for dz in range(dy, 256):
                ach[s2 + dz * dz] = 1
    ach[0] = 0
    dom = [v for v in range(1, N2_MAX + 1) if ach[v]]
    assert len(dom) == N_ACHIEVABLE
    h_dom, h_tbl = hashlib.sha256(), hashlib.sha256()
    for n2 in dom:
        h_dom.update(n2.to_bytes(4, "little"))
        v = ideal_f(n2)
        assert v <= C_SCALE, f"F({n2}) = {v} exceeds 2^24"
        h_tbl.update(v.to_bytes(4, "little"))
    assert h_dom.hexdigest() == DOM_DIGEST, "domain digest drifted"
    assert h_tbl.hexdigest() == TBL_DIGEST, "table digest drifted"
    gaps = [v for v in range(1, N2_MAX + 1) if not ach[v]][:8]
    assert gaps == [7, 15, 23, 28, 31, 39, 47, 55]
    assert not ach[65_536] and ach[1] and ach[N2_MAX]

    # 2. rounding_floor_and_output_extremes (exhaustive).
    min_d, arg = 2.0, 0
    fmax, fmin = 0, 1 << 60
    for n2 in dom:
        t = C_SCALE / math.sqrt(n2)
        d = abs((t - math.floor(t)) - 0.5)
        if d < min_d:
            min_d, arg = d, n2
        v = ideal_f(n2)
        fmax, fmin = max(fmax, v), min(fmin, v)
    assert min_d > 1e-7 and arg == 24_459, (min_d, arg)
    assert abs(min_d - 6.558e-7) < 2e-8
    assert fmax == C_SCALE and fmin == 37_986

    # 3. kat_anchors.
    for n2, want in [
        (1, 16_777_216),
        (2, 11_863_283),
        (3, 9_686_330),
        (4, 8_388_608),
        (5, 7_502_999),
        (16_384, 131_072),
        (16_385, 131_068),
        (195_075, 37_986),
    ]:
        assert ideal_f(n2) == want, f"F({n2})"
    assert norm2([0, 0, 0], [255, 255, 255]) == 195_075
    assert norm2([7, 9, 200], [7, 9, 213]) == 13 * 13
    assert ideal_k_out([0, 0, 0], [1, 0, 0], 0) == C_SCALE
    k = MOD - 5
    assert ideal_k_out([0, 0, 0], [1, 0, 0], k) == (k + C_SCALE) % MOD

    # 4. check_shot_band_semantics (per-shot verdict; STRICT — any False
    #    fails the run).
    task = TASK
    for ri, rj, k in [
        ([0, 0, 0], [1, 0, 0], MOD - 3),
        ([10, 20, 30], [200, 100, 50], 0x0F0F0F0),
        ([255, 255, 255], [0, 0, 0], 0x1FFFFFE),
        ([100, 100, 100], [100, 100, 101], 12345),
    ]:
        inputs: list[int | None] = ri + rj + [k]
        tc = TestCase(inputs=inputs, expected=ri + rj + [ideal_k_out(ri, rj, k)])
        ideal = ideal_k_out(ri, rj, k)
        for d in (-EPS, -1, 0, 1, EPS):
            assert task.check_shot(tc, ri + rj + [(ideal + d) % MOD]), d
        for d in (-(EPS + 1), EPS + 1):
            assert not task.check_shot(tc, ri + rj + [(ideal + d) % MOD]), d
        bad = ri + rj + [ideal]
        bad[5] ^= 1
        assert not task.check_shot(tc, bad)  # coordinate corruption

    # 5. testvec_distribution_pins (pinned SHAKE-256 stream).
    stream = hashlib.shake_256(b"coulomb-potential/testvec").digest(9024 * 11)
    live = skips = n2_one = seam = 0
    modes = [0, 0, 0]
    oct_counts = [0] * 18
    shots = []
    for s in range(9024):
        blk = stream[s * 11 : (s + 1) * 11]
        dec = decode_draw(blk)
        if dec is None:
            skips += 1
            continue
        ri, rj, k = dec
        live += 1
        mode = blk[3]
        modes[0 if mode & 1 == 0 else (1 if (mode >> 1) & 7 <= 6 else 2)] += 1
        n2 = norm2(ri, rj)
        assert 1 <= n2 <= N2_MAX
        oct_counts[n2.bit_length() - 1] += 1
        n2_one += n2 == 1
        seam += k + ideal_f(n2) >= MOD
        shots.append((n2, k))
    assert (live, skips) == (9_017, 7)
    assert modes == [4_495, 3_989, 533]
    assert n2_one == 64 and oct_counts[17] == 527
    assert min(oct_counts) >= 64, oct_counts
    assert seam == 363, seam

    # 6. degenerate_circuits_fail: a constant output, an echo of n2, XOR instead
    #    of addition, and assignment instead of addition (pinned counts).
    def centered(v):
        r = v % MOD
        return r - MOD if r > MOD // 2 else r

    best_const = ideal_f(5)
    const_ok = echo_ok = xor_ok = set_ok = 0
    for n2, k in shots:
        f = ideal_f(n2)
        const_ok += abs(centered(best_const - f)) <= EPS
        echo_ok += abs(centered(n2 - f)) <= EPS
        want = (k + f) % MOD
        xor_ok += abs(centered(((k ^ f) & MASK25) - want)) <= EPS
        set_ok += abs(centered((f % MOD) - want)) <= EPS
    assert const_ok == 136 and echo_ok == 0 and xor_ok == 209 and set_ok == 0, (
        const_ok,
        echo_ok,
        xor_ok,
        set_ok,
    )
    for ok in (const_ok, echo_ok, xor_ok, set_ok):
        # Each degenerate circuit fails thousands of shots; STRICT fails on the first.
        assert live - ok > 4_000

    print("coulomb-potential grader.py: all self-checks OK")
