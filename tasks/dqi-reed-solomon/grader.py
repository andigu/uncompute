"""Grader for `dqi-reed-solomon`.

The task is coherent Reed-Solomon syndrome decoding at the DQI optimal
polynomial intersection (OPI) instance of Khattar et al. (2025),
arXiv:2510.10967: m = 4095, n = 70, b = 12.  An error word e in
GF(2^12)^4095 has the 70 syndromes S_i = e(gamma^i) = sum_j e_j gamma^(i*j),
i = 1..70 (no zeroth power).  For a syndrome of an error with symbol weight
at most 35, the circuit implements the phase |s> -> (-1)^<c, e(s)> |s>.
Here c is the fixed 49140-bit mask (counter-mode SHA-256 over the instance
tag b"kickmix/rs-decode-4095/fixup-c/v1") and <.,.> is the F2 inner product
with the 49140-bit packing of e (symbol j at bits 12j..12j+11).  There is no
output register.  All 70 syndrome registers (12 qubits each, little-endian)
must return unchanged.

Field conventions: GF(2^12) = GF(2)[x]/(x^12+x^7+x^6+x^5+x^3+x+1)
(integer 4331, primitive), gamma = x (integer 2).

Failure policy: a shot fails if its registers are wrong, its sign is wrong,
or any workspace qubit is nonzero at the end.  Each shot gets one verdict.
The run accepts at most 100 failed shots out of 100,000.  check_shot is the
default exact register comparison.

Test inputs: the weight k comes from the dyadically quantized DQI weight
law WEIGHT_CUM_2P16.  Positions come from a Fisher-Yates shuffle of the
4095 points, and values are uniform on GF(2^12)*.  Both use exact rejection
sampling from a shot-local SHAKE256 expansion of the shot's fixed 212-byte
block.  Shots 0-3 are fixed anchors (e = 0 and three single-symbol
errors).  They read and discard their block.

The grader plants e and never decodes.  Two distinct errors of weight at
most 35 differ by a word of weight at most 70 < 71 = d.  That word cannot
be a nonzero codeword of the dual RS code (BCH bound on 70 consecutive
powers; RS is MDS).  Every syndrome therefore has a unique planted answer,
weight 35 included.  A circuit that applies no phase gets about half of the
nonzero-error shots wrong (49,855 of 100,000 measured), far over the
allowance.

Run `python3 grader.py` for the self-test (fixed anchors and the
F2 linear-collapse check; about 2 min).
"""

import hashlib

from uncompute.evalcore import QUBIT, RegSpec, Task, TestCase, Unified

# Code length: number of error-word symbols / nonzero field points.
M = 4095
# Number of syndrome symbols (powers gamma^1..gamma^70).
N_SYND = 70
# Bits per GF(2^12) symbol.
B = 12
# Total syndrome qubits (70 registers x 12 qubits).
SYNDROME_QUBITS = N_SYND * B
# Unique-decoding radius n/2 (designed distance d = 71 > 2*35).
W_MAX = 35
# Randomized test shots.
NUM_TESTS = 100_000
# Tolerated failed shots (one shared allowance; see module doc).
BUDGET = 100
# Field polynomial x^12+x^7+x^6+x^5+x^3+x+1 (primitive; `galois` default).
GF_POLY = 4331

# Cumulative weight law on a /2^16 grid: the exact DQI input measure |w_k|^2
# for this instance, quantized by largest-remainder rounding. Entry k =
# P(weight <= k) * 2^16; a u16 draw < CUM[k] with draw >= CUM[k-1] selects k.
WEIGHT_CUM_2P16 = [
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
    4,
    13,
    41,
    121,
    333,
    843,
    1963,
    4199,
    8230,
    14737,
    24027,
    35550,
    47605,
    57694,
    63723,
    65536,
]

# F2 length of the packed error word / fixup vector c: m*b = 49140 bits.
C_BITS = M * B
# c stream length in bytes (49140 bits -> 6143 bytes).
C_STREAM_BYTES = (C_BITS + 7) // 8
# Fixed instance constant: the SHA-256 tag that defines the fixup vector c.
C_INSTANCE_TAG = b"kickmix/rs-decode-4095/fixup-c/v1"

# Fixed master-XOF block per shot: 2 weight bytes + 35 x (4 position bytes +
# 2 value bytes). Anchor shots consume the same block (and discard it). Only
# the two weight bytes are decoded in place; positions and values are u32
# draws from the shot-local SHAKE256 seeded by the whole block, so 12-bit
# positions (m = 4095 > 2^10 - 1) fit without widening the layout.
XOF_BLOCK = 2 + W_MAX * 6


# --- GF(2^12) tables ---------------------------------------------------------


def gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
    for k in range(2 * B - 1, B - 1, -1):
        if (r >> k) & 1:
            r ^= GF_POLY << (k - B)
    return r


def c_stream():
    """Fixup-vector byte stream: block i = SHA-256(tag || u32le(i)),
    concatenated, truncated to 6143 bytes. Bit k of c = bit k%8 of byte k/8.
    """
    out = b""
    i = 0
    while len(out) < C_STREAM_BYTES:
        out += hashlib.sha256(C_INSTANCE_TAG + i.to_bytes(4, "little")).digest()
        i += 1
    return out[:C_STREAM_BYTES]


def _build_tables():
    # Log/antilog tables over GF(2^12).  The log table fills completely
    # only if gamma = 2 is primitive; the asserts check this.
    exp = [0] * M
    log = [0] * (M + 1)
    acc = 1
    for i in range(M):
        exp[i] = acc
        assert acc != 0 and acc <= M
        assert i == 0 or log[acc] == 0, "gamma = 2 must be primitive"
        log[acc] = i
        acc = gf_mul(acc, 2)
    assert acc == 1, "gamma^4095 must return to 1"

    stream = c_stream()
    c_sym = [0] * M
    for k in range(C_BITS):
        if (stream[k >> 3] >> (k & 7)) & 1:
            c_sym[k // B] |= 1 << (k % B)
    return exp, log, c_sym


EXP, LOG, C_SYM = _build_tables()
# Doubled power table: EXP2[a] = gamma^(a mod 4095) for a in [0, 8190) —
# lets the syndrome inner loop skip the % 4095.
EXP2 = EXP + EXP
# STEP[p][i] = ((i+1)*p) % 4095 — hoists the index arithmetic out of the
# 100k-shot syndrome loop.
STEP = [[((i + 1) * p) % M for i in range(N_SYND)] for p in range(M)]


def syndrome(e):
    """S_i = sum_j e_j gamma^{i*j} for i = 1..70, from a sparse error word
    (distinct positions 0..4094, values 1..4095).
    """
    s = [0] * N_SYND
    for p, v in e:
        lv = LOG[v]
        st = STEP[p]
        s = [a ^ EXP2[lv + b] for a, b in zip(s, st)]
    return s


def phase_bit(e):
    """<c, e> over F2 on the 49140-bit packing of e."""
    par = 0
    for p, v in e:
        par ^= bin(C_SYM[p] & v).count("1")
    return par & 1 == 1


def _tc_for(e):
    s = syndrome(e)
    return TestCase(inputs=list(s), expected=list(s), expected_phase=phase_bit(e))


class _LocalXof:
    """Shot-local SHAKE256 expander (prefix-stable digest + cursor)."""

    __slots__ = ("_buf", "_h", "_pos")

    def __init__(self, seed):
        self._h = hashlib.shake_256(seed)
        self._buf = self._h.digest(320)  # >= the typical 8*w draw
        self._pos = 0

    def read_u32(self):
        pos = self._pos
        if pos + 4 > len(self._buf):
            self._buf = self._h.digest(2 * len(self._buf))
        out = self._buf[pos : pos + 4]
        self._pos = pos + 4
        return int.from_bytes(out, "little")


def _uniform(rng, n):
    """Exactly uniform draw on 0..n by rejection (the pinned uniform rule):
    accept a u32 below the largest multiple of n, redraw otherwise.
    """
    zone = ((1 << 32) // n) * n
    while True:
        x = rng.read_u32()
        if x < zone:
            return x % n


def tc_from_block(buf):
    """Decode one fixed-size XOF block into a random test error word (the pinned
    tc_from_block rule).
    """
    wdraw = buf[0] | (buf[1] << 8)
    w = 0
    while wdraw >= WEIGHT_CUM_2P16[w]:
        w += 1
    rng = _LocalXof(buf)
    pool = list(range(M))
    e = []
    for slot in range(w):
        j = slot + _uniform(rng, M - slot)
        pool[slot], pool[j] = pool[j], pool[slot]
        e.append((pool[slot], 1 + _uniform(rng, M)))
    return _tc_for(e)


class DqiReedSolomon(Task):
    name = "dqi-reed-solomon"
    seed_domain = b"dqi-reed-solomon"
    num_tests = NUM_TESTS
    # Wrong registers, wrong sign, and dirty workspace share one per-shot
    # verdict and one allowance; check_shot is the default exact compare.
    failure_policy = Unified(BUDGET)

    def schema(self):
        return [RegSpec(width=B, kind=QUBIT) for _ in range(N_SYND)]

    def gen_test(self, xof):
        return tc_from_block(xof.read(XOF_BLOCK))

    def gen_test_indexed(self, shot, xof):
        buf = xof.read(XOF_BLOCK)  # constant consumption, anchors included

        # Fixed anchors: the zero error and three single-symbol errors.
        if shot == 0:
            return _tc_for([])  # e = 0: s = 0, sign +1
        if shot == 1:
            return _tc_for([(0, 1)])  # the all-ones-syndrome fixed point
        if shot == 2:
            return _tc_for([(4094, 4095)])
        if shot == 3:
            return _tc_for([(2047, 2049)])
        return tc_from_block(buf)


TASK = DqiReedSolomon()


# --- Import-time KATs (the cheap construction asserts) -----------------------


def _import_kats():
    def syn_digest(s):
        h = hashlib.sha256()
        for v in s:
            h.update(v.to_bytes(2, "little"))
        return h.hexdigest()

    s0 = syndrome([(0, 1)])
    assert all(x == 1 for x in s0), "KAT (p=0,v=1): S_i = 1 for all i"
    assert not phase_bit([(0, 1)]), "KAT (p=0,v=1) phase"
    s1 = syndrome([(4094, 4095)])
    assert syn_digest(s1) == "c86ef34677e887c9f6c42852da560d92d320f0720fe7f9919aeb65a29d22449b"
    assert s1[:3] == [3978, 1989, 2967], "KAT (p=4094,v=4095) head"
    assert phase_bit([(4094, 4095)]), "KAT (p=4094,v=4095) phase"
    s2 = syndrome([(2047, 2049)])
    assert syn_digest(s2) == "0a82bb2a327af9ccf35b4a3cab63b3ac8d2e54f017fa9df5e9ea56ada2cb3c7c"
    assert s2[:3] == [1555, 3189, 2940], "KAT (p=2047,v=2049) head"
    assert phase_bit([(2047, 2049)]), "KAT (p=2047,v=2049) phase"


_import_kats()


# --- contract.json parity (assert-only; the inline derivations above stay
# authoritative — this pins the sidecar every circuit loads to them) -----------


def _assert_contract():
    import json
    from pathlib import Path

    c = json.loads((Path(__file__).resolve().parent / "contract.json").read_text())
    assert (c["gf_poly"], c["m"], c["n_synd"], c["b"], c["w_max"]) == (
        GF_POLY,
        M,
        N_SYND,
        B,
        W_MAX,
    ), "contract.json instance drifted"
    assert c["c_tag"].encode() == C_INSTANCE_TAG, "contract.json c_tag drifted"
    stream = c_stream()
    assert bytes.fromhex(c["c_stream_hex"]) == stream, "contract.json c_stream drifted"
    assert c["c_popcount"] == sum((stream[k >> 3] >> (k & 7)) & 1 for k in range(C_BITS)), (
        "contract.json c_popcount drifted"
    )
    assert c["c_stream_sha256"] == hashlib.sha256(stream).hexdigest(), (
        "contract.json c_stream_sha256 drifted"
    )
    assert c["weight_cum_2p16"] == WEIGHT_CUM_2P16, "contract.json weight law drifted"


_assert_contract()


if __name__ == "__main__":
    # Self-test: the remaining construction checks.

    # 1. instance_provenance: SHA-256 of the gamma power table (LE u16).
    h = hashlib.sha256()
    for v in EXP:
        h.update(v.to_bytes(2, "little"))
    assert h.hexdigest() == "5624513c7b9c1e263a85830bb71877508e6026fa4efc3284e0139e0909bdcec5"

    # 2. c_provenance: stream digest + popcount, recomputed independently.
    stream = b"".join(
        hashlib.sha256(C_INSTANCE_TAG + i.to_bytes(4, "little")).digest() for i in range(192)
    )[:6143]
    assert stream == c_stream()
    assert (
        hashlib.sha256(stream).hexdigest()
        == "8fcd44e5a419ea29d84bb443d75f10dc9ac85dde2085b7851d92cd198eb3785a"
    )
    # popcount over the 49140 USED bits only (the last byte's top 4 bits are
    # outside c and excluded, exactly as pinned).
    assert sum((stream[k >> 3] >> (k & 7)) & 1 for k in range(C_BITS)) == 24492

    # 3. weight law table shape.
    assert all(a <= b for a, b in zip(WEIGHT_CUM_2P16, WEIGHT_CUM_2P16[1:]))
    assert WEIGHT_CUM_2P16[W_MAX] == 65536

    # 4. Linear-collapse guard (the expensive construction assert): the
    #    840x49140 F2 syndrome matrix has full rank 840 and c lies OUTSIDE
    #    its rowspace. Rows/vectors as Python ints (bit k = F2 coordinate k).
    c_vec = 0
    for k in range(C_BITS):
        if (stream[k >> 3] >> (k & 7)) & 1:
            c_vec |= 1 << k
    basis = {}  # pivot -> row

    def reduce(v):
        while v:
            p = v.bit_length() - 1
            row = basis.get(p)
            if row is None:
                return v, p
            v ^= row
        return 0, -1

    rank = 0
    for i in range(1, N_SYND + 1):
        syms = []
        for jpos in range(M):
            base = EXP[(i * jpos) % M]
            syms.append([gf_mul(base, 1 << t) for t in range(B)])
        for jbit in range(B):
            row = 0
            for jpos, s in enumerate(syms):
                for t, st in enumerate(s):
                    if (st >> jbit) & 1:
                        row |= 1 << (B * jpos + t)
            v, p = reduce(row)
            if v:
                basis[p] = v
                rank += 1
    assert rank == SYNDROME_QUBITS, f"syndrome map rank {rank} != 840"
    v, _ = reduce(c_vec)
    assert v != 0, "fixup c must lie OUTSIDE rowspace_F2 (linear-collapse guard)"

    # 5. gen_test_profile (anchors + round-trip structure) with the
    #    pinned LCG FakeXof, byte-exact.
    class FakeXof:
        def __init__(self, state):
            self.state = state

        def read(self, n):
            out = bytearray(n)
            s = self.state
            for i in range(n):
                s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
                out[i] = (s >> 33) & 0xFF
            self.state = s
            return bytes(out)

    task = TASK
    xof = FakeXof(42)
    tc0 = task.gen_test_indexed(0, xof)
    assert len(tc0.inputs) == N_SYND
    assert all(v == 0 for v in tc0.inputs) and not tc0.expected_phase
    tc1 = task.gen_test_indexed(1, xof)
    assert all(v == 1 for v in tc1.inputs)
    for shot in range(4, 1004):
        tc = task.gen_test_indexed(shot, xof)
        assert len(tc.inputs) == N_SYND
        assert tc.inputs == tc.expected
        assert all(0 <= v < (1 << B) for v in tc.inputs)

    # 6. weight_profile_direct (law-shape statistics on tc_from_block) with
    #    the second pinned LCG, byte-exact.
    _lcg = [7]

    def next_byte():
        _lcg[0] = (_lcg[0] * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (_lcg[0] >> 33) & 0xFF

    w35 = 0
    wlow = [0] * W_MAX
    total = 20000
    for _ in range(total):
        buf = bytes(next_byte() for _ in range(XOF_BLOCK))
        wdraw = buf[0] | (buf[1] << 8)
        w = next(i for i, cum in enumerate(WEIGHT_CUM_2P16) if wdraw < cum)
        if w == W_MAX:
            w35 += 1
        else:
            wlow[w] += 1
        tc = tc_from_block(buf)
        assert tc.inputs == tc.expected
    frac35 = w35 / total
    assert 0.02 < frac35 < 0.04, frac35
    frac32 = wlow[32] / total
    assert 0.15 < frac32 < 0.22, frac32
    assert all(n == 0 for n in wlow[:19])
    assert all(n > 0 for n in wlow[22:])

    # 7. syndrome_injectivity_sample (splitmix64 over a pi seed, byte-exact
    #    as pinned).
    seen = {}
    _sm = [0x243F6A8885A308D3]

    def next64():
        _sm[0] = (_sm[0] + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = _sm[0]
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    for _ in range(5000):
        w = next64() % (W_MAX + 1)
        pool = list(range(M))
        e = []
        for slot in range(w):
            j = slot + next64() % (M - slot)
            pool[slot], pool[j] = pool[j], pool[slot]
            e.append((pool[slot], 1 + next64() % M))
        e.sort()
        s = tuple(syndrome(e))
        prev = seen.get(s)
        assert prev is None or prev == e, "syndrome collision"
        seen[s] = e

    print("dqi-reed-solomon grader.py: all self-checks OK")
