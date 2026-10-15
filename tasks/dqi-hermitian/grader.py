"""Grader for `dqi-hermitian`.

The task is coherent Hermitian-code syndrome decoding for DQI at the
q₀ = 8 Hermitian optimal polynomial intersection (HOPI) instance of Gu and
Jordan (2025), arXiv:2510.06603. The code is the one-point Hermitian code
on y⁸ + y = x⁹ over F₆₄. H is the 123×512 evaluation matrix of the
L(150·P∞) monomial basis, with parameters [512, 389, ≥ 96] (Goppa bound
d ≥ A − 2g + 2, g = 28). For a syndrome s = H·e with wt(e) ≤ 47 =
⌊(96−1)/2⌋ (unique decoding), the circuit implements
|s⟩ ⟼ (−1)^⟨c, e(s)⟩ |s⟩. Here c is the fixed 3072-bit mask and ⟨·,·⟩ is
the F₂ inner product with the 3072-bit packing of e (symbol j at bits
6j..6j+5). There are 123 registers of 6 qubits (738 total): register r
holds syndrome symbol r (H row order), with the coefficient of xᵗ in
qubit t. All 123 must return unchanged. There is no output register.

Instance choice: at this size, DQI's expected satisfied fraction is 0.7145
and a single Prange trial reaches 0.5726. The gap of 0.1419 is +7.4σ, so
matching DQI with repeated single-trial Prange costs about 2⁴⁴ work
(Chernoff estimate).

Failure policy: a shot fails if its registers are wrong, its sign is
wrong, or any workspace qubit is nonzero at the end. Each shot gets one
verdict. The run accepts at most 100 failed shots out of 100,000
(``failure_policy = Unified(100)``).

Test inputs: the weight k comes from the dyadically quantized DQI weight
law (WEIGHT_CUM_2P16: the principal-eigenvector law of Jordan et al.
(2025), arXiv:2408.08292, at p = 28/64). The law is supported on
k ∈ [29, 47] with mode 43; weights k ≤ 28 round to zero and the fixed
anchors cover them. The support comes from a Fisher-Yates shuffle of the
512 points, and values are uniform on F₆₄*. Both use exact rejection
sampling from a shot-local SHAKE256 expansion of the shot's fixed 64-byte
block, so the realized law equals the specified law. Shots 0–3 are fixed
anchors; they read and discard their block.

The grader rebuilds all instance data from first principles at import and
checks it: curve, point, and monomial counts, the rank-738 linear-collapse
check on c, and five syndrome/phase known-answer vectors. Run this file
directly for the self-test.

Field and curve conventions (cross-derived by instance.py and reference.py
in this directory): GF(64) = GF(2)[x]/(x⁶+x⁴+x³+x+1) (integer 91, the
`galois` package default; x primitive), elements 0..63 in the polynomial
basis, bit j = coefficient of xʲ. The 512 affine curve points are ordered
lexicographically by the integer pair (x, y), x major. The rows of H are
the monomials xᵃyᵇ with 0 ≤ b ≤ 7 and 8a + 9b ≤ 150, b-major, then a
ascending.
"""

import hashlib
import json
import pathlib
from bisect import bisect_right

import numpy as np

from uncompute.evalcore import (
    QUBIT,
    RegSpec,
    Task,
    TestCase,
    Unified,
    Xof,
)

# Number of curve points / error-word symbols.
N_POINTS = 512
# Number of parity checks (monomials of L(150·P∞)).
N_CHECKS = 123
# Bits per F₆₄ symbol.
B = 6
# Total syndrome qubits (123 registers × 6 qubits).
SYNDROME_QUBITS = N_CHECKS * B
# Unique-decoding radius ⌊(d−1)/2⌋, d ≥ 96 (Goppa bound).
ELL = 47
# Randomized tests.
NUM_TESTS = 100_000
# Tolerated failed shots (one shared allowance).
BUDGET = 100
# Full field polynomial x⁶+x⁴+x³+x+1 (bit 6 = the x⁶ term).
GF_POLY = 91 | 64
# Cumulative weight law on a /2¹⁶ grid: the exact DQI input measure |w_k|²
# for this instance (m = 512, ℓ = 47, p = 28/64), where w is the principal
# eigenvector of the Jordan et al. (arXiv:2408.08292) tridiagonal
#   A[k][k] = k·(1−2p)/√(p(1−p)),  A[k][k+1] = √((k+1)(m−k)),
# quantized by largest-remainder rounding (total variation vs the real
# eigenvector ≈ 4.1×10⁻⁵; classes k ≤ 28 round to zero). Entry k =
# P(weight ≤ k)·2¹⁶; a u16 XOF draw < CUM[k] with draw ≥ CUM[k−1] selects
# weight k. Derivation: instance.py in this directory.
WEIGHT_CUM_2P16 = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    3,
    9,
    26,
    72,
    184,
    440,
    987,
    2076,
    4085,
    7504,
    12842,
    20416,
    30068,
    40915,
    51358,
    59532,
    64180,
    65536,
)

# Fixed instance constant: the tag of the fixup-vector XOF stream
# (counter-mode SHA-256, 384 bytes).
C_INSTANCE_TAG = b"kickmix/hopi-512/fixup-c/v1"
# c stream length in bytes (3072 bits = 512 symbols × 6).
C_LEN = N_POINTS * B // 8

# Fixed XOF block per shot: 2 weight-draw bytes + 62 seed bytes for the
# shot-local rejection-sampling expander. Anchor shots consume the same
# block (and discard it) so the shot-indexed byte stream is uniform.
XOF_BLOCK = 64


# ─── GF(64) = GF(2)[x]/(x⁶+x⁴+x³+x+1), elements 0..63 in the poly basis ─────


def _gf_mul_slow(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
    for k in range(11, 5, -1):  # reduce by x⁶+x⁴+x³+x+1 (0b1011011)
        if (r >> k) & 1:
            r ^= GF_POLY << (k - 6)
    return r


# 64×64 multiplication table (numpy — the instance build and the per-shot
# syndrome path below are fully vectorized off this table).
_MUL = np.array(
    [[_gf_mul_slow(a, b) for b in range(64)] for a in range(64)],
    dtype=np.uint8,
)


def gf_mul(a, b):
    return int(_MUL[a, b])


def gf_pow(a, e):
    acc = 1
    for _ in range(e):
        acc = int(_MUL[acc, a])
    return acc


def c_stream():
    """Counter-mode SHA-256 XOF over C_INSTANCE_TAG: block i = SHA-256(tag ‖ u32_le(i)),
    concatenated, truncated to C_LEN bytes.
    """
    out = b""
    i = 0
    while len(out) < C_LEN:
        out += hashlib.sha256(C_INSTANCE_TAG + i.to_bytes(4, "little")).digest()
        i += 1
    return out[:C_LEN]


# ─── F₂ elimination on int bitmasks (row-mask helpers) ──────────────────────


def _f2_reduce(v, basis):
    """Reduce v against a {leading bit: row} basis; returns the residual."""
    while v:
        lead = v.bit_length() - 1
        row = basis.get(lead)
        if row is None:
            break
        v ^= row
    return v


def _f2_rank(rows):
    """F₂ elimination; returns (rank, {leading bit: row})."""
    basis = {}
    for r in rows:
        v = _f2_reduce(r, basis)
        if v:
            basis[v.bit_length() - 1] = v
    return len(basis), basis


# ─── Instance build (the pinned instance construction) ──────────────────────


def _build_instance():
    # 512 affine points of y⁸ + y = x⁹, lex order by (x, y).
    pts = [(x, y) for x in range(64) for y in range(64) if gf_pow(y, 8) ^ y == gf_pow(x, 9)]
    assert len(pts) == N_POINTS, "curve must have exactly 512 points"

    # L(150·P∞) monomial basis xᵃyᵇ, 0 ≤ b ≤ 7, 8a + 9b ≤ 150, b-major.
    h = []
    for b in range(8):
        for a in range(19):
            if 8 * a + 9 * b <= 150:
                h.append([gf_mul(gf_pow(x, a), gf_pow(y, b)) for (x, y) in pts])
    assert len(h) == N_CHECKS, "L(150) basis must have exactly 123 monomials"
    h = np.array(h, dtype=np.uint8)  # (123, 512)

    c = c_stream()
    c_int = int.from_bytes(c, "little")

    # Linear-collapse guard: c must be outside the F₂-rowspace of the
    # 738×3072 binary expansion of H, else ⟨c,e(s)⟩ would be linear in s and
    # the task would collapse to CNOTs. Hhat[(r,t),(i,u)] = bit t of
    # h[r,i]·xᵘ — built vectorized, packed to ints, eliminated.
    hx = _MUL[h[:, :, None], (1 << np.arange(B, dtype=np.uint8))[None, None, :]]
    # (123, 512, 6): hx[r, i, u] = h[r,i]·xᵘ; bit-t planes → (123, 6, 3072).
    hhat = (hx[:, None, :, :] >> np.arange(B, dtype=np.uint8)[None, :, None, None]) & 1
    hhat = hhat.reshape(N_CHECKS * B, N_POINTS * B)  # row 6r+t, col 6i+u
    packed = np.packbits(hhat, axis=1, bitorder="little")
    rows = [int.from_bytes(rb.tobytes(), "little") for rb in packed]
    rank, basis = _f2_rank(rows)
    assert rank == SYNDROME_QUBITS, "Hhat must have full F₂ rank 738"
    assert _f2_reduce(c_int, basis) != 0, (
        "fixup vector c must lie OUTSIDE rowspace_F2(H) (collapse guard)"
    )

    # Per-(position, value) syndrome columns: SCOL[i, v] is the 123-symbol
    # contribution of e_i = v (symbol XOR ≡ elementwise XOR), and PBIT[i, v]
    # the parity of ⟨c, v at symbol i⟩ — the whole oracle reduces to XORs
    # over the ≤ 47 nonzero symbols.
    scol = _MUL[h[:, :, None], np.arange(64, dtype=np.uint8)[None, None, :]]
    scol = np.ascontiguousarray(scol.transpose(1, 2, 0))  # (512, 64, 123)
    csym = np.frombuffer(
        np.packbits(
            np.unpackbits(np.frombuffer(c, dtype=np.uint8), bitorder="little").reshape(N_POINTS, B),
            axis=1,
            bitorder="little",
        ),
        dtype=np.uint8,
    )  # (512,): the 6-bit chunk of c at symbol i
    pop6 = np.array([bin(v).count("1") for v in range(64)], dtype=np.uint8)
    pbit = pop6[csym[:, None] & np.arange(64, dtype=np.uint8)[None, :]] & 1
    return pts, h, c, scol, pbit


PTS, H, C_BYTES, _SCOL, _PBIT = _build_instance()

# ─── Contract pin (assert-only) ──────────────────────────────────────────────
# tasks/dqi-hermitian/contract.json carries the task-defining derived
# constants for the circuits under circuits/.  The inline derivation above stays
# the authority; this block only proves the shared file matches it.
_CONTRACT = json.loads((pathlib.Path(__file__).resolve().parent / "contract.json").read_text())
assert _CONTRACT["gf_poly"] == GF_POLY, "contract.json: gf_poly"
assert _CONTRACT["c_tag"].encode() == C_INSTANCE_TAG, "contract.json: c_tag"
assert bytes.fromhex(_CONTRACT["c_hex"]) == C_BYTES, "contract.json: c_hex"
assert _CONTRACT["c_sha256"] == hashlib.sha256(C_BYTES).hexdigest(), "contract.json: c_sha256"
assert (
    _CONTRACT["points_sha256"] == hashlib.sha256(bytes(v for xy in PTS for v in xy)).hexdigest()
), "contract.json: points"
assert _CONTRACT["h_sha256"] == hashlib.sha256(H.tobytes()).hexdigest(), "contract.json: h_sha256"
assert tuple(_CONTRACT["weight_cum_2p16"]) == WEIGHT_CUM_2P16, "contract.json: weight law"
assert (_CONTRACT["n_points"], _CONTRACT["n_checks"], _CONTRACT["bits"], _CONTRACT["ell"]) == (
    N_POINTS,
    N_CHECKS,
    B,
    ELL,
), "contract.json: instance scalars"
assert (_CONTRACT["num_tests"], _CONTRACT["unified_budget"]) == (NUM_TESTS, BUDGET), (
    "contract.json: protocol scalars"
)
assert _CONTRACT["seed_domain"].encode() == b"dqi-hermitian", "contract.json: seed_domain"


def syndrome(e):
    """s = H·e as 123 F₆₄ symbols (H row order), numpy uint8."""
    nz = [i for i in range(N_POINTS) if e[i]]
    if not nz:
        return np.zeros(N_CHECKS, dtype=np.uint8)
    return np.bitwise_xor.reduce(_SCOL[nz, [e[i] for i in nz]], axis=0)


def phase_bit(e):
    """⟨c, e⟩ over F₂ on the 3072-bit packing of e (symbol i at bits 6i..)."""
    p = 0
    for i in range(N_POINTS):
        if e[i]:
            p ^= int(_PBIT[i, e[i]])
    return p == 1


# Syndrome/phase known-answer vectors (pinned at construction;
# digests = SHA-256 over the 123 syndrome symbol bytes; cross-derived by
# instance.py — numpy path — and reference.py — decoder path).
def _kat(e_sparse, digest, phase):
    e = [0] * N_POINTS
    for j, v in e_sparse:
        e[j] = v
    s = syndrome(e)
    assert hashlib.sha256(s.tobytes()).hexdigest() == digest, f"KAT syndrome {e_sparse}"
    assert phase_bit(e) == phase, f"KAT phase {e_sparse}"


_kat([], "409a7f83ac6b31dc8c77e3ec18038f209bd2f545e0f4177c2e2381aa4e067b49", False)
_kat([(0, 1)], "962a4d330a15318eb80cc16fa4d940645581c4f1c8f1e92b14a84e167684e971", True)
_kat([(511, 63)], "72ae9a9988605abcbd1a79a8d4a336d44c42769f9c609983f1d79c9ddcbbced6", True)
_kat([(137, 42)], "40637d230a06183e4280dc9f4c8dbba04cd029e1d8ca3532616f0ddadd7f440e", False)
_kat(
    [(3, 7), (100, 1), (200, 33), (511, 62)],
    "598d97b386f7f7e598622602c4f1b0a20f1e534f104a8f5a07e577c49fc9319f",
    False,
)


def _uniform(rng, n):
    """Exactly uniform draw on 0..n by rejection (the pinned uniform rule):
    accept a u32 below the largest multiple of n, redraw otherwise. No
    modulo bias — the realized law equals the pinned law.
    """
    zone = ((1 << 32) // n) * n
    while True:
        x = int.from_bytes(rng.read(4), "little")
        if x < zone:
            return x % n


def _tc_for(e):
    vals = syndrome(e).tolist()
    return TestCase(
        expected_phase=phase_bit(e),
        inputs=vals,
        expected=list(vals),  # all 123 syndrome registers must round-trip
    )


class DqiHermitian(Task):
    name = "dqi-hermitian"
    seed_domain = b"dqi-hermitian"
    num_tests = NUM_TESTS
    # Wrong registers, wrong sign, and dirty workspace share one per-shot
    # verdict and one allowance of 100 failed shots.
    failure_policy = Unified(BUDGET)

    def schema(self):
        return [RegSpec(width=B, kind=QUBIT) for _ in range(N_CHECKS)]

    def gen_test(self, xof):
        buf = xof.read(XOF_BLOCK)
        return self._tc_from_block(buf)

    def gen_test_indexed(self, shot, xof):
        buf = xof.read(XOF_BLOCK)  # constant consumption, anchors included

        # Fixed anchors: the zero error and three single-symbol errors.
        # Weights ≤ 28 have zero dyadic mass, so these anchors cover that
        # range on every run.
        if shot == 0:
            return _tc_for([0] * N_POINTS)  # e = 0: s = 0, sign +1
        if shot == 1:
            e = [0] * N_POINTS
            e[0] = 1  # point (0,0), the s = 0x…001 fixed point
            return _tc_for(e)
        if shot == 2:
            e = [0] * N_POINTS
            e[511] = 63
            return _tc_for(e)
        if shot == 3:
            e = [0] * N_POINTS
            e[137] = 42
            return _tc_for(e)
        return self._tc_from_block(buf)

    def _tc_from_block(self, buf):
        """Decode one fixed-size XOF block into a randomized test error vector (the pinned
        tc_from_block rule): weight k from WEIGHT_CUM_2P16 via the
        block's first two bytes as u16-LE; positions by Fisher-Yates over
        the 512 points and values on F₆₄*, both via exact rejection sampling
        from a shot-local SHAKE256 expansion of the whole block. The master
        XOF supplies exactly one block per shot; the unbounded rejection
        draws come only from the local expander.
        """
        wdraw = buf[0] | (buf[1] << 8)
        # First k with wdraw < CUM[k] (wdraw ≤ 65535 < CUM[47]).
        w = bisect_right(WEIGHT_CUM_2P16, wdraw)

        rng = Xof(bytes(buf))

        pool = list(range(N_POINTS))
        pos = [0] * w
        val = [0] * w
        for slot in range(w):
            j = slot + _uniform(rng, N_POINTS - slot)
            pool[slot], pool[j] = pool[j], pool[slot]
            pos[slot] = pool[slot]
            val[slot] = 1 + _uniform(rng, 63)

        if w == 0:
            s = np.zeros(N_CHECKS, dtype=np.uint8)
            ph = 0
        else:
            s = np.bitwise_xor.reduce(_SCOL[pos, val], axis=0)
            ph = int(_PBIT[pos, val].sum()) & 1
        vals: list[int | None] = s.tolist()
        return TestCase(expected_phase=ph == 1, inputs=vals, expected=list(vals))


TASK = DqiHermitian()


# ---------------------------------------------------------------------------
# Self-test: rechecks the pinned instance anchors.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    M64 = (1 << 64) - 1

    # points_provenance: SHA-256 of the lex point order (x,y byte pairs).
    pb = bytes(v for xy in PTS for v in xy)
    assert (
        hashlib.sha256(pb).hexdigest()
        == "35212139940f58ff3ec5ab82599affbaf002657a1d12b05eee4d887dfc14d95e"
    )
    print("PASS  point-order digest (instance provenance)")

    # h_provenance: SHA-256 of H (row-major symbol bytes).
    assert (
        hashlib.sha256(H.tobytes()).hexdigest()
        == "d667062e02b9e1918ac918ac6a251663ed3fd20c46a0b9872cd50a0f4e14a796"
    )
    print("PASS  H digest (instance provenance)")

    # c_provenance: stream digest + popcount.
    assert len(C_BYTES) == C_LEN
    assert (
        hashlib.sha256(C_BYTES).hexdigest()
        == "10fe6addc16595734e774145bfb571af2f11870903306a5f047aaebd8a7b3181"
    )
    assert sum(bin(b).count("1") for b in C_BYTES) == 1545
    print("PASS  c provenance (counter-mode SHA-256 stream, popcount 1545)")

    # KATs re-checked (already asserted at import).
    _kat([], "409a7f83ac6b31dc8c77e3ec18038f209bd2f545e0f4177c2e2381aa4e067b49", False)
    print("PASS  5 syndrome/phase known-answer vectors")

    # weight_law_shape: monotone cumulative, support [29, 47], mode 43.
    assert len(WEIGHT_CUM_2P16) == ELL + 1
    assert WEIGHT_CUM_2P16[-1] == 65536
    masses = [b - a for a, b in zip((0,) + WEIGHT_CUM_2P16, WEIGHT_CUM_2P16)]
    assert all(m >= 0 for m in masses)
    assert all(m == 0 for m in masses[:29])
    assert all(m > 0 for m in masses[29:])
    assert masses.index(max(masses)) == 43
    print("PASS  weight-law shape (support [29,47], mode 43)")

    # sampler_respects_promise: spot-check over pseudorandom blocks (the
    # LCG: state ← state·6364136223846793005 + 1, byte = state ≫ 24).
    state = 0x9E3779B97F4A7C15

    def _next_byte():
        global state
        state = (state * 6364136223846793005 + 1) & M64
        return (state >> 24) & 0xFF

    task = TASK
    for _ in range(500):
        buf = bytes(_next_byte() for _ in range(XOF_BLOCK))
        wdraw = buf[0] | (buf[1] << 8)
        w = bisect_right(WEIGHT_CUM_2P16, wdraw)
        assert w <= ELL
        tc = task._tc_from_block(buf)
        assert len(tc.inputs) == N_CHECKS
        assert tc.inputs == tc.expected
    print("PASS  sampler respects the promise (500 pseudorandom blocks)")

    # gen_test profile: anchors + round-trip invariant on indexed shots.
    class _FakeXof:
        def __init__(self, seed):
            self.state = seed

        def read(self, n):
            out = bytearray(n)
            for i in range(n):
                self.state = (self.state * 6364136223846793005 + 1) & M64
                out[i] = (self.state >> 24) & 0xFF
            return bytes(out)

    xof = _FakeXof(42)
    tc0 = task.gen_test_indexed(0, xof)
    assert all(v == 0 for v in tc0.inputs) and not tc0.expected_phase
    for shot in range(4, 2004):
        tc = task.gen_test_indexed(shot, xof)
        assert len(tc.inputs) == N_CHECKS
        assert tc.inputs == tc.expected
    print("PASS  gen_test profile (anchor 0 zero-syndrome; round-trip 4..2004)")

    # Failure policy pin.
    from uncompute.evalcore import resolve_failure_policy

    pol = resolve_failure_policy(task)
    assert isinstance(pol, Unified) and pol.budget == BUDGET
    print(f"PASS  failure policy Unified({BUDGET})")

    print("dqi-hermitian grader self-test: ALL PASS")
