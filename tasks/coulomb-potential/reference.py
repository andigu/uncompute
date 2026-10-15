#!/usr/bin/env python3
"""coulomb-potential — self-verifying reference (classical spec document).

Pins, from first principles, everything the grader asserts:
  1. the task map F(r_i, r_j) = round(2^24/||r_i - r_j||) and its domain
     (139,637 achievable n2 values; digests byte-exact);
  2. KAT anchors and output extremes (F(1) = 2^24 exactly — no clamp);
  3. the f64-sqrt oracle's soundness (round-half floor 6.56e-7, exhaustive);
  4. band semantics (centered mod 2^25, +/-32 in, +/-33 out, seam-checked);
  5. the pinned three-mode test distribution on the pinned test-vector stream
     (live/skips, mode counts, per-octave coverage floor, seam coverage);
  6. the error of a simple Newton reference construction (L_inf = 14 <=
     EPS/2 = 16, exhaustive over the domain): the band admits a plain
     arithmetic circuit with a 2x margin;
  7. degenerate circuits (constant output, echo of n2, XOR instead of
     addition, assignment instead of addition) each fail thousands of shots.

Exit 0 <=> every assertion passed. Pure stdlib; no circuit here (the
circuits live under circuits/). Runtime ~40 s (three exhaustive
139,637-value sweeps).

Every exhaustive check below sweeps the full achievable-n2 domain. This is
why b = 8: at b >= 9 the domain is too large to sweep in a self-test, and
these checks would degrade to sampling.
"""

import hashlib
import math
from fractions import Fraction

_SELFCHECK = __name__ == "__main__"  # self-tests run script-mode only; import is instant

# ---- pinned constants (must match grader.py) ----
B = 8
M = 25
C_SCALE = 1 << 24
EPS = 32
N2_MAX = 3 * 255 * 255  # 195,075
N_ACHIEVABLE = 139_637
TESTVEC_TAG = b"coulomb-potential/testvec"
TBL_DIGEST = "3de8b2b0b3063d0f46aa4cbcf528173b73d4cf0d28d0a2fac27491238d804439"
DOM_DIGEST = "f52f7e1b110369374daea81136a21c36eca11364ccb2653b725d4f6ba6b1e8cc"

# ---- 0. hash-function anchor (NIST FIPS 180-4 "abc") ----
if _SELFCHECK:
    assert (
        hashlib.sha256(b"abc").hexdigest()
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def round_half_away(v: float) -> int:
    """Round v >= 0 to the nearest integer, with ties away from zero.

    This is the grader's rule. Python round() sends ties to the even
    integer instead. floor(v) and v - floor(v) are exact in f64.
    """
    f = math.floor(v)
    return f + 1 if v - f >= 0.5 else f


def ideal_f(n2: int) -> int:
    """F(n2) = round(2^24 / sqrt(n2)) with ONE f64 sqrt, ties away from zero.

    Section 3 shows that no tie occurs on the domain.
    """
    return round_half_away(C_SCALE / math.sqrt(n2))


def centered(v: int) -> int:
    v %= 1 << M
    return v - (1 << M) if v > 1 << (M - 1) else v


def check_shot(k_in: int, k_out: int, n2: int) -> bool:
    return abs(centered((k_out - k_in) - ideal_f(n2))) <= EPS


# ---- 1. domain + digests ----
_ach = bytearray(N2_MAX + 1)
for dx in range(256):
    x2 = dx * dx
    for dy in range(dx, 256):
        s2 = x2 + dy * dy
        for dz in range(dy, 256):
            _ach[s2 + dz * dz] = 1
_ach[0] = 0
DOMAIN = [v for v in range(1, N2_MAX + 1) if _ach[v]]
if _SELFCHECK:
    assert len(DOMAIN) == N_ACHIEVABLE
# Legendre three-square exclusions lead the gap list; 256^2 unreachable (|d|<=255)
gaps = [v for v in range(1, N2_MAX + 1) if not _ach[v]]
if _SELFCHECK:
    assert gaps[:8] == [7, 15, 23, 28, 31, 39, 47, 55]
    assert not _ach[65_536]
h_dom, h_tbl = hashlib.sha256(), hashlib.sha256()
for n2 in DOMAIN:
    h_dom.update(n2.to_bytes(4, "little"))
    v = ideal_f(n2)
    assert v <= C_SCALE  # 25 bits, no clamp anywhere
    h_tbl.update(v.to_bytes(4, "little"))
if _SELFCHECK:
    assert h_dom.hexdigest() == DOM_DIGEST, "domain digest drifted"
    assert h_tbl.hexdigest() == TBL_DIGEST, "table digest drifted"

# ---- 2. KATs + extremes ----
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
    assert ideal_f(n2) == want, (n2, ideal_f(n2), want)
if _SELFCHECK:
    assert max(ideal_f(v) for v in (1, 2, 3)) == C_SCALE
    assert ideal_f(N2_MAX) == 37_986

# ---- 3. oracle soundness: exhaustive round-half floor ----
min_d, min_arg = 2.0, 0
for n2 in DOMAIN:
    t = C_SCALE / math.sqrt(n2)
    d = abs((t - math.floor(t)) - 0.5)
    if d < min_d:
        min_d, min_arg = d, n2
if _SELFCHECK:
    assert min_d > 1e-7 and min_arg == 24_459, (min_d, min_arg)
    assert abs(min_d - 6.558e-7) < 2e-8

# ---- 4. band semantics incl. the mod-2^25 seam ----
for n2 in (1, 169, 24_459, 111_747, 195_075):
    f = ideal_f(n2)
    for k_in in (0, 12_345, (1 << M) - 3, (1 << M) - 1):  # incl. wrap cases
        ideal_out = (k_in + f) % (1 << M)
        for d in (-EPS, -1, 0, 1, EPS):
            assert check_shot(k_in, (ideal_out + d) % (1 << M), n2), (n2, k_in, d)
        for d in (-(EPS + 1), EPS + 1):
            assert not check_shot(k_in, (ideal_out + d) % (1 << M), n2), (n2, k_in, d)


# ---- 5. the pinned three-mode test distribution ----
def decode_draw(blk: bytes):
    """Exactly gen_test's decode (grader.py decode_draw). None <=> skip."""
    ri = list(blk[0:3])
    mode = blk[3]
    mat = blk[4:7]
    k = int.from_bytes(blk[7:11], "little") & ((1 << M) - 1)
    if mode & 1 == 0:
        rj = list(mat)
    else:
        e = (mode >> 1) & 7
        if e <= 6:
            w = e + 2
            rj = []
            for c in range(3):
                v = mat[c] & ((1 << w) - 1)
                if v >= 1 << (w - 1):
                    v -= 1 << w
                rj.append((ri[c] + v) % 256)
        else:
            ri = [224 | (x & 0x1F) for x in ri]
            rj = [x & 0x1F for x in mat]
    if rj == ri:
        return None
    return tuple(ri), tuple(rj), k


def mode_of(blk: bytes) -> int:
    """Sampling-mode class of a draw block (grader parity): 0 = uniform,
    1 = near, 2 = far.
    """
    mode = blk[3]
    return 0 if mode & 1 == 0 else (1 if (mode >> 1) & 7 <= 6 else 2)


def coverage_walk(n_blocks: int):
    """Decode the pinned test-vector stream: yield
    ``(blk, decode_draw(blk))`` for the first ``n_blocks`` 11-byte draws
    (decode None <=> skip).
    """
    stream = hashlib.shake_256(TESTVEC_TAG).digest(n_blocks * 11)
    for s in range(n_blocks):
        blk = stream[s * 11 : (s + 1) * 11]
        yield blk, decode_draw(blk)


def norm2(ri, rj):
    return sum((a - b) ** 2 for a, b in zip(ri, rj))


live = skips = n2_one = seam = 0
modes = [0, 0, 0]
oct_counts = [0] * 18
shots = []
for blk, dec in coverage_walk(9024):
    if dec is None:
        skips += 1
        continue
    ri, rj, k = dec
    live += 1
    modes[mode_of(blk)] += 1
    n2 = norm2(ri, rj)
    assert 1 <= n2 <= N2_MAX
    oct_counts[n2.bit_length() - 1] += 1
    n2_one += n2 == 1
    seam += k + ideal_f(n2) >= 1 << M
    shots.append((n2, k))
if _SELFCHECK:
    assert (live, skips) == (9_017, 7)
    assert modes == [4_495, 3_989, 533]
    assert n2_one == 64
    assert oct_counts[17] == 527
    assert min(oct_counts) >= 64, oct_counts  # every octave exercised
    assert seam == 363  # wrap seam never starved
# exact skip probability (mixture): 1/2*2^-24 + 7/16 near-mode collisions
p_skip = Fraction(1, 2) * Fraction(1, 1 << 24)
for e in range(7):
    p_skip += Fraction(1, 16) * Fraction(1, 1 << (3 * (e + 2)))
if _SELFCHECK:
    assert abs(float(p_skip) - 1.116e-3) < 1e-6
    assert abs(9024 * (1 - float(p_skip)) - 9013.9) < 0.1  # E[live]

# ---- 6. Newton reference construction: exhaustive envelope ----
PW = 28


def newton_reference(n2: int) -> int:
    """Find-first-one guess (constants TRUNCATED at pw = 28) + 3 truncating
    Newton iterations directly on n2; returns output on the 2^-24 grid
    (rounded). It shows that the band admits a simple arithmetic circuit;
    it is not a circuit requirement (any circuit inside the band passes).
    """
    i = n2.bit_length() - 1
    k = (i + 1) // 2
    pat = int((1.5 if k == 0 else 1.613) * 2.0 ** (PW - k))
    sh = 27 - 3 * k
    y = pat - (n2 << sh if sh >= 0 else n2 >> -sh)
    for _ in range(3):
        s = (y * y) >> PW
        e = 3 * (1 << PW) - n2 * s
        assert e > 0
        y = (y * e) >> (PW + 1)
    return (y + (1 << PW - 25)) >> (PW - 24)


linf = arg = 0
per_oct = [0] * 18
for n2 in DOMAIN:
    err = abs(newton_reference(n2) - ideal_f(n2))
    o = n2.bit_length() - 1
    per_oct[o] = max(per_oct[o], err)
    if err > linf:
        linf, arg = err, n2
if _SELFCHECK:
    assert (linf, arg) == (14, 176_261), (linf, arg)
    assert EPS >= 2 * linf  # >= 2x margin
    assert per_oct[2:] == [4, 1, 3, 1, 2, 1, 1, 1, 2, 2, 3, 4, 6, 8, 12, 14]

# ---- 7. degenerate circuits fail on the pinned stream ----
best_const = ideal_f(5)  # 7,502,999: max-mass output
const_ok = echo_ok = xor_ok = set_ok = 0
for n2, k in shots:
    f = ideal_f(n2)
    const_ok += abs(centered(best_const - f)) <= EPS
    echo_ok += abs(centered(n2 - f)) <= EPS
    want = (k + f) % (1 << M)
    xor_ok += abs(centered((k ^ f & ((1 << M) - 1)) - want)) <= EPS
    set_ok += abs(centered((f % (1 << M)) - want)) <= EPS
if _SELFCHECK:
    assert const_ok == 136, const_ok  # ~8,900 failed shots
    assert echo_ok == 0, echo_ok
    assert xor_ok == 209, xor_ok  # carries fire ~everywhere
    assert set_ok == 0, set_ok
    for ok in (const_ok, echo_ok, xor_ok, set_ok):
        assert live - ok > 4_000

    print("coulomb-potential reference: ALL CHECKS PASSED")
    print(f"  domain          : {len(DOMAIN):,} achievable n2 in [1, {N2_MAX:,}]")
    print(f"  oracle floor    : {min_d:.3e} at n2 = {min_arg:,}")
    print(f"  Newton ref L_inf: {linf} ULP at n2 = 176,261 (band {EPS}, margin {EPS / linf:.2f}x)")
    print(
        f"  testvec         : {live} live / {skips} skips; modes {modes}; octave floor {min(oct_counts)}"
    )
    print(f"  degenerate      : const {const_ok}, echo {echo_ok}, xor {xor_ok}, set {set_ok}")
