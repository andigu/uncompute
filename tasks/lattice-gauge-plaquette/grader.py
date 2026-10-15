"""Python grader for `lattice-gauge-plaquette`.

Operation (the full Symanzik-improved magnetic Trotter layer on a 4³ periodic
lattice, d = 3, for Σ(72×3), |G| = 216, in the 9-qubit normal-form encoding
of Osorio Perez et al., arXiv:2511.17437, verbatim; PARAMS-DERIVATION.md §7
fixes every convention):

    |g₀ … g₁₉₁⟩|k⟩ → |g₀ … g₁₉₁⟩|k + a_P·S_P + a_R·S_R mod 2^51⟩

    S_P = Σ over 192 plaquettes f(P),  S_R = Σ over 384 rectangles f(R),
    f   = 2·Re tr ∈ {−3, −2, −1, 0, 1, 2, 6}
    a_P = 0x8EBA84840FB (SHA-256(INSTANCE_TAG)[0:51] | 1)
    a_R = −round(a_P/20) mod 2^51 = 2^51 − 490,411,682,214

Geometry: site s = x + 4y + 16z, link ℓ = 3s + μ; registers 0..191 the links
in ℓ-order (9 qubits each), register 192 is k (51 qubits). Loop products put
inverses on the return links; enumeration sites ascending, plaquettes over
(i,j) ∈ {(0,1),(0,2),(1,2)}, rectangles over ordered (i,j), i ≠ j, lex.

The task is exact: a shot is CORRECT iff all 192 link registers round-trip
exactly and read_k equals the specified modular sum. Every tested shot must
be correct (failure_policy STRICT).

Group + trace data are rebuilt at import from the generators of Osorio Perez
et al. (2025), with
the pinned construction-time asserts (relations, bijectivity, |409⟩,
Latin square, trace multiplicities, class-function property, separation
floor, constant derivation) and the three pinned SHA3-256 digests
(MUL/TRACE/INV) shared with reference.py. Run this file directly to
replicate the pinned unit-test anchors.
"""

import hashlib

from uncompute.evalcore import QUBIT, RegSpec, Task, TestCase

# Link register width (qubits per Σ(72×3) code).
N_LINK = 9
# Number of link registers (4³ sites × 3 directions).
N_LINKS = 192
# Plaquette and rectangle counts.
N_P = 192
N_R = 384
# Accumulator width (qubits of k).
M = 51
MOD = 1 << M
# Pinned plaquette multiplier: top 51 bits (MSB-first) of SHA-256(INSTANCE_TAG),
# forced odd (the raw prefix 0x8EBA84840FA is EVEN — the |1 fires).
A_P = 0x8EBA84840FB  # 9,808,233,644,283
# Magnitude of the rectangle multiplier: round(a_P/20); signed a_R = −A_R_MAG.
A_R_MAG = 490_411_682_214
# Exact modular accumulation: no output error is allowed.
EPS = 0
# Test shots (gen_test never skips, so this is exactly the checked count).
# Baseline-sized circuits (~10M ops, ~1.27M measurement-based releases) make the
# evaluator's pre-drawn RNG stream about 15 GB at this count, so their
# evaluation needs a machine with about 64 GB of memory.
NUM_TESTS = 100_000

# Fixed instance constant: SHA-256 of this tag defines a_P.
INSTANCE_TAG = b"kickmix-plaquette-trace-accumulate-v2"

# contract.json (task root) is the machine-readable sidecar of these
# contract constants that circuits load. The inline pins above stay the
# authority — this block only ASSERTS the sidecar matches them.
import json as _json
from pathlib import Path as _Path

_cj = _json.loads((_Path(__file__).resolve().parent / "contract.json").read_text())
assert _cj["instance_tag"].encode() == INSTANCE_TAG, "contract.json instance_tag drifted"
assert _cj["m"] == M and _cj["eps"] == EPS, "contract.json m/eps drifted"
assert int(_cj["a_P"], 16) == A_P, "contract.json a_P drifted"
assert int(_cj["a_R"], 16) == MOD - A_R_MAG, "contract.json a_R drifted"
del _cj

# Pinned SHA3-256 digests (cross-implementation contract with reference.py).
MUL_DIGEST = "a9ef423e2669f722b1b93c0c6a77911318b46f40f57dd0c89a33b9c051d551e3"
TRACE_DIGEST = "4a1ff65e31ae491e6ef73a7f3219e6cc112a7fa6de897c2b09f977dc4ca3ccca"
INV_DIGEST = "394fbe2d07f1078df6ea88d7b50e0489b5dd2999312c9cf8384cf193a154bf39"

# ---------------------------------------------------------------------------
# Exact arithmetic in Z[ω][1/3], ω = e^{2πi/3}, ω² = −1 − ω. An entry
# (a, b, k) represents (a + b·ω)/3^k.
# ---------------------------------------------------------------------------


def _norm(a, b, k):
    while k > 0 and a % 3 == 0 and b % 3 == 0:
        a //= 3
        b //= 3
        k -= 1
    return (a, b, k)


def _cadd(x, y):
    k = max(x[2], y[2])
    sx = 3 ** (k - x[2])
    sy = 3 ** (k - y[2])
    return _norm(x[0] * sx + y[0] * sy, x[1] * sx + y[1] * sy, k)


def _cmul(x, y):
    return _norm(
        x[0] * y[0] - x[1] * y[1],
        x[0] * y[1] + x[1] * y[0] - x[1] * y[1],
        x[2] + y[2],
    )


ZERO = (0, 0, 0)
ONE = (1, 0, 0)
W = (0, 1, 0)  # ω
W2 = (-1, -1, 0)  # ω²

I3 = ((ONE, ZERO, ZERO), (ZERO, ONE, ZERO), (ZERO, ZERO, ONE))


def _mmul(p, q):
    return tuple(
        tuple(
            _cadd(
                _cadd(_cmul(p[i][0], q[0][j]), _cmul(p[i][1], q[1][j])),
                _cmul(p[i][2], q[2][j]),
            )
            for j in range(3)
        )
        for i in range(3)
    )


def _smul(s, p):
    return tuple(tuple(_cmul(s, p[i][j]) for j in range(3)) for i in range(3))


def _mpow(p, n):
    r = I3
    for _ in range(n):
        r = _mmul(r, p)
    return r


def _trace2re(m):
    """2·Re tr(M), exact: entry (a, b, k) has 2·Re = (2a − b)/3^k."""
    t = _cadd(_cadd(m[0][0], m[1][1]), m[2][2])
    a, b, k = t
    num = 2 * a - b
    den = 3**k
    assert num % den == 0, "trace not integral at L = 2"
    v = num // den
    assert -3 <= v <= 6, "trace out of SU(3) range"
    return v


def _encode(p, q, r, s, t, u):
    return (
        (u << 8)
        | (t << 7)
        | (s << 6)
        | ((r >> 1) << 5)
        | ((r & 1) << 4)
        | ((q >> 1) << 3)
        | ((q & 1) << 2)
        | ((p >> 1) << 1)
        | (p & 1)
    )


def _centered(v):
    """Centered residue mod 2^51 in (−2^50, 2^50]."""
    r = v % MOD
    if r > MOD // 2:
        r -= MOD
    return r


# ---------------------------------------------------------------------------
# Group + trace data: built once at import; the build is the verification.
# ---------------------------------------------------------------------------


def _build_data():
    # Pinned constants: re-derive and assert (SHA-256 instance-tag pin).
    digest = hashlib.sha256(INSTANCE_TAG).digest()
    assert digest.hex() == "011d7509081f5efa37139519357a698a9b53bbe990fd36521ef37f31fe9e544d", (
        "instance-tag digest drifted — the tag string was edited"
    )
    top64 = int.from_bytes(digest[:8], "big")
    raw51 = top64 >> 13
    assert raw51 & 1 == 0, "raw 51-bit prefix is EVEN for this instance tag"
    assert raw51 == 0x8EBA84840FA
    assert raw51 | 1 == A_P, "a_P must equal the pinned derivation"
    assert A_P & 1 == 1, "a_P must be odd"
    assert A_P % 20 == 3, "rounding direction pin (0.15 → down)"
    assert (A_P + 10) // 20 == A_R_MAG, "a_R magnitude must be round(a_P/20)"

    # Generators, arXiv:2511.17437 Eq. (1).
    inv_sqrt3i = (-1, -2, 1)
    assert _cmul((1, 2, 0), inv_sqrt3i) == ONE

    c = ((ONE, ZERO, ZERO), (ZERO, W, ZERO), (ZERO, ZERO, W2))
    e = ((ZERO, ONE, ZERO), (ZERO, ZERO, ONE), (ONE, ZERO, ZERO))
    v = _smul(inv_sqrt3i, ((ONE, ONE, ONE), (ONE, W, W2), (ONE, W2, W)))
    x = _smul(inv_sqrt3i, ((ONE, ONE, W2), (ONE, W, W), (W, ONE, W)))

    # Relations, Eq. (2).
    assert _mmul(x, c) == _mmul(_mmul(_mpow(c, 2), e), x), "X C = C² E X"
    assert _mmul(x, e) == _smul(W2, _mmul(_mmul(c, e), x)), "X E = ω² C E X"
    assert _mmul(x, v) == _smul(W2, _mmul(_mmul(e, _mpow(v, 3)), x)), "X V = ω² E V³ X"
    assert _mpow(c, 3) == I3
    assert _mpow(e, 3) == I3
    assert _mpow(v, 4) == I3
    assert _mpow(x, 4) == I3

    # Normal-form enumeration: 216 codes → 216 DISTINCT matrices.
    sc = (ONE, W, W2)
    mat_of = {}
    code_of = {}
    for p in range(3):
        for q in range(3):
            for r in range(3):
                for s in range(2):
                    for t in range(2):
                        for u in range(2):
                            m = _smul(
                                sc[p],
                                _mmul(
                                    _mmul(_mpow(c, q), _mpow(e, r)),
                                    _mmul(_mpow(v, 2 * s + t), _mpow(x, u)),
                                ),
                            )
                            code = _encode(p, q, r, s, t, u)
                            mat_of[code] = m
                            code_of[m] = code
    assert len(mat_of) == 216, "216 codes"
    assert len(code_of) == 216, "normal form must be a bijection"

    # Paper worked example |409⟩ (Eq. (4)).
    i_sqrt3 = (1, 2, 1)
    m409 = _smul(i_sqrt3, ((W2, W, W), (ONE, W, ONE), (ONE, ONE, W)))
    assert code_of[m409] == 409, "paper Eq. (4) anchor |409⟩"

    valid = sorted(mat_of)
    idx = [0xFF] * 512
    for i, code in enumerate(valid):
        idx[code] = i

    # Multiplication table (closure) + Latin-square check.
    mul = []
    for a in valid:
        ma = mat_of[a]
        mul.append([code_of[_mmul(ma, mat_of[b])] for b in valid])
    assert idx[0] == 0, "identity code 0 is dense index 0"
    full = set(valid)
    for i in range(216):
        assert set(mul[i]) == full, f"Latin row {i}"
    for j in range(216):
        assert {mul[i][j] for i in range(216)} == full, f"Latin col {j}"

    # Inverses (two-sided), as codes.
    inv = []
    for i in range(216):
        j = next(j for j in range(216) if mul[i][j] == 0)
        assert mul[j][i] == 0, "inverse must be two-sided"
        inv.append(valid[j])

    # Trace table: f = 2·Re tr, exact.
    trace = [_trace2re(mat_of[code]) for code in valid]

    # Trace-layer asserts: 7-value set + class multiplicities.
    counts = {}
    for f in trace:
        counts[f] = counts.get(f, 0) + 1
    assert counts == {6: 1, -3: 2, -2: 9, 1: 18, 0: 24, 2: 54, -1: 108}, "trace multiplicity vector"
    assert trace[0] == 6, "identity trace 2·Re tr(I) = 6"
    assert (trace[idx[1]], trace[idx[2]]) == (-3, -3), "nontrivial center ωI, ω²I have f = −3"

    # Class-function property: f(g·x·g⁻¹) = f(x), exhaustive 216 × 216.
    for xi in range(216):
        fx = trace[xi]
        for gi in range(216):
            gx = mul[gi][xi]
            gxg = mul[idx[gx]][idx[inv[gi]]]
            assert trace[idx[gxg]] == fx, "class function"
    # f(g⁻¹) = f(g).
    for i in range(216):
        assert trace[idx[inv[i]]] == trace[i]

    # Separation: a single wrong trace class shifts k by a_P·d or a_R·d,
    # d ∈ ±{1..9}; minima are the constants themselves and are nonzero.
    min_p = min(abs(_centered(A_P * d)) for d in range(1, 10))
    min_r = min(abs(_centered(-A_R_MAG * d)) for d in range(1, 10))
    assert min_p == A_P, "plaquette separation floor drifted"
    assert min_r == A_R_MAG, "rectangle separation floor drifted"
    assert min_p > 0, "a_P must separate distinct trace values"
    assert min_r > 0, "a_R must separate distinct trace values"

    # Digests: the cross-implementation contract with reference.py.
    blob = bytearray()
    for i in range(216):
        for j in range(216):
            blob += mul[i][j].to_bytes(2, "little")
    assert hashlib.sha3_256(bytes(blob)).hexdigest() == MUL_DIGEST, "MUL digest"
    tb = bytes(f & 0xFF for f in trace)  # i8 two's complement
    assert hashlib.sha3_256(tb).hexdigest() == TRACE_DIGEST, "TRACE digest"
    ib = bytearray()
    for i in range(216):
        ib += inv[i].to_bytes(2, "little")
    assert hashlib.sha3_256(bytes(ib)).hexdigest() == INV_DIGEST, "INV digest"

    return valid, idx, mul, inv, trace


VALID, IDX, MUL, INV, TRACE = _build_data()

# Dense-index fast tables for the oracle: MUL_D[i][j] and INV_D[i] are dense
# indices into VALID; TRACE[i] is already dense. Dense identity index is 0.
MUL_D = [[IDX[code] for code in row] for row in MUL]
INV_D = [IDX[code] for code in INV]

# ---------------------------------------------------------------------------
# Lattice geometry and loop tables (PARAMS §3): 4³ periodic, d = 3.
# ---------------------------------------------------------------------------

LAT = 4


def _coords(s):
    return (s % LAT, (s // LAT) % LAT, s // (LAT * LAT))


def _shift(s, mu):
    x, y, z = _coords(s)
    if mu == 0:
        x = (x + 1) % LAT
    elif mu == 1:
        y = (y + 1) % LAT
    else:
        z = (z + 1) % LAT
    return x + LAT * y + LAT * LAT * z


def _link(s, mu):
    return 3 * s + mu


def _build_loops():
    plaq = []
    for s in range(LAT**3):
        for i, j in ((0, 1), (0, 2), (1, 2)):
            plaq.append(
                (
                    (_link(s, i), False),
                    (_link(_shift(s, i), j), False),
                    (_link(_shift(s, j), i), True),
                    (_link(s, j), True),
                )
            )
    rect = []
    for s in range(LAT**3):
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                si = _shift(s, i)
                sii = _shift(si, i)
                sj = _shift(s, j)
                sij = _shift(si, j)
                rect.append(
                    (
                        (_link(s, i), False),
                        (_link(si, i), False),
                        (_link(sii, j), False),
                        (_link(sij, i), True),
                        (_link(sj, i), True),
                        (_link(s, j), True),
                    )
                )
    assert len(plaq) == N_P
    assert len(rect) == N_R

    # Displacement closure: every loop returns to its base site.
    def walk(loop_links):
        start = loop_links[0][0] // 3
        cur = start
        for l, dag in loop_links:
            s, mu = l // 3, l % 3
            if not dag:
                assert s == cur, "forward link site mismatch"
                cur = _shift(s, mu)
            else:
                assert _shift(s, mu) == cur, "daggered link site mismatch"
                cur = s
        assert cur == start, "loop does not close"

    in_plaq = [0] * N_LINKS
    in_rect = [0] * N_LINKS
    for p in plaq:
        walk(p)
        for l, _ in p:
            in_plaq[l] += 1
    for r in rect:
        walk(r)
        for l, _ in r:
            in_rect[l] += 1
    assert all(cnt == 4 for cnt in in_plaq), "plaquette incidence must be 4"
    assert all(cnt == 12 for cnt in in_rect), "rectangle incidence must be 12"
    return plaq, rect


PLAQ, RECT = _build_loops()

# Flat loop tables for the fast oracle path (link indices only; the dagger
# pattern is fixed per loop shape: plaquette f,f,d,d — rectangle f,f,f,d,d,d).
_PLAQ_FLAT = [(p[0][0], p[1][0], p[2][0], p[3][0]) for p in PLAQ]
_RECT_FLAT = [(r[0][0], r[1][0], r[2][0], r[3][0], r[4][0], r[5][0]) for r in RECT]


# ---------------------------------------------------------------------------
# The graded map.
# ---------------------------------------------------------------------------


def loop_f(links, loop_links):
    """f of one Wilson loop over CODES (the pinned loop_f rule)."""
    p = 0  # dense identity index
    for l, dag in loop_links:
        g = IDX[links[l]]
        assert g != 0xFF, "invalid code"
        if dag:
            g = INV_D[g]
        p = MUL_D[p][g]
    return TRACE[p]


def _sums_dense(dis):
    """(S_P, S_R) for a full link assignment given as DENSE indices."""
    md = MUL_D
    invd = INV_D
    tr = TRACE
    s_p = 0
    for l0, l1, l2, l3 in _PLAQ_FLAT:
        s_p += tr[md[md[md[dis[l0]][dis[l1]]][invd[dis[l2]]]][invd[dis[l3]]]]
    s_r = 0
    for l0, l1, l2, l3, l4, l5 in _RECT_FLAT:
        s_r += tr[
            md[md[md[md[md[dis[l0]][dis[l1]]][dis[l2]]][invd[dis[l3]]]][invd[dis[l4]]]][
                invd[dis[l5]]
            ]
        ]
    return s_p, s_r


def sums(links):
    """(S_P, S_R) for a full link assignment (192 codes, ℓ-order)."""
    assert len(links) == N_LINKS
    dis = [IDX[g] for g in links]
    assert all(d != 0xFF for d in dis), "invalid code"
    return _sums_dense(dis)


def increment(s_p, s_r):
    """(a_P·S_P − A_R_MAG·S_R) mod 2^51 (S_P, S_R can be negative)."""
    return (A_P * s_p - A_R_MAG * s_r) % MOD


def ideal_k_out(links, k_in):
    s_p, s_r = sums(links)
    return (k_in + increment(s_p, s_r)) & (MOD - 1)


class LatticeGaugePlaquette(Task):
    name = "lattice-gauge-plaquette"
    seed_domain = b"lattice-gauge-plaquette"
    num_tests = NUM_TESTS
    # Exact output and strict run-level correctness.

    def schema(self):
        regs = [RegSpec(width=N_LINK, kind=QUBIT) for _ in range(N_LINKS)]
        regs.append(RegSpec(width=M, kind=QUBIT))
        return regs

    def gen_test(self, xof):
        # Constant 1,544 bytes/shot (192 × 8 for the links + 8 for k); never
        # skips. Each link: u64 LE → dense index ⌊u·216/2^64⌋ (wide-word
        # reduction, bias ≤ 2^−56) → sorted valid-code table. k: u64 LE
        # masked to 51 bits — exactly uniform, so the randomized k_in
        # exercises the mod-2^51 wraparound.  Both sums concentrate
        # (σ(S_P) = 19.6, σ(S_R) = 27.7), so random tests never draw the
        # identity/single-link configurations; the self-test pins those
        # as anchors.
        buf = xof.read(1544)
        dis = [
            (int.from_bytes(buf[8 * l : 8 * l + 8], "little") * 216) >> 64 for l in range(N_LINKS)
        ]
        k = int.from_bytes(buf[1536:1544], "little") & (MOD - 1)
        s_p, s_r = _sums_dense(dis)
        k_out = (k + increment(s_p, s_r)) & (MOD - 1)
        inputs = [VALID[d] for d in dis]
        inputs.append(k)
        expected = list(inputs)  # links preserved, checked exactly
        expected[N_LINKS] = k_out
        return TestCase(expected_phase=False, inputs=inputs, expected=expected)

    def check_shot(self, tc, read):
        """Exact acceptance: all links must round-trip and the accumulator
        must equal the specified modular sum. Any wrong shot fails the run.
        """
        inputs = tc.inputs
        for i in range(N_LINKS):
            if read[i] != inputs[i]:
                return False
        delta = _centered(read[N_LINKS] - tc.expected[N_LINKS])
        return -EPS <= delta <= EPS


TASK = LatticeGaugePlaquette()


# ---------------------------------------------------------------------------
# Self-test: replicates the pinned unit-test anchors.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class _FixedXof:
        def __init__(self, data):
            self.data = bytes(data)
            self.pos = 0

        def read(self, n):
            out = self.data[self.pos : self.pos + n]
            assert len(out) == n
            self.pos += n
            return out

    def make_tc(links, k_in):
        inputs = list(links) + [k_in]
        expected = list(inputs)
        expected[N_LINKS] = ideal_k_out(links, k_in)
        return TestCase(expected_phase=False, inputs=inputs, expected=expected)

    def make_read(links, k):
        return list(links) + [k % MOD]

    # pinned_constant_derivation (mostly asserted at import).
    assert A_P == 9_808_233_644_283
    assert (A_P // 20, A_P % 20) == (490_411_682_214, 3)
    assert MOD - A_R_MAG == 0x7FF8DD12F965A  # = 2_251_309_402_003_034
    print("PASS  pinned constant derivation (a_P, a_R, |1 fired)")

    # loop_table_integrity: distinctness, dagger patterns, incidence split,
    # pinned first/last entries.
    in_plaq = [0] * N_LINKS
    long_c = [0] * N_LINKS
    short_c = [0] * N_LINKS
    for p in PLAQ:
        ls = sorted({l for (l, _) in p})
        assert len(ls) == 4, "plaquette links must be distinct"
        assert [d for (_, d) in p] == [False, False, True, True]
        for l, _ in p:
            in_plaq[l] += 1
    for r in RECT:
        ls = sorted({l for (l, _) in r})
        assert len(ls) == 6, "rectangle links must be distinct"
        assert [d for (_, d) in r] == [False, False, False, True, True, True]
        for slot, (l, _) in enumerate(r):
            if slot in (2, 5):
                short_c[l] += 1
            else:
                long_c[l] += 1
    for l in range(N_LINKS):
        assert in_plaq[l] == 4
        assert long_c[l] == 8 and short_c[l] == 4
    assert PLAQ[0] == ((0, False), (4, False), (12, True), (1, True))
    assert PLAQ[191] == ((190, False), (155, False), (46, True), (191, True))
    assert RECT[0] == ((0, False), (3, False), (7, False), (15, True), (12, True), (1, True))
    assert RECT[5] == ((2, False), (50, False), (97, False), (62, True), (14, True), (1, True))
    assert RECT[383] == (
        (191, False),
        (47, False),
        (94, False),
        (11, True),
        (155, True),
        (190, True),
    )
    print("PASS  loop table integrity (incidence 4/8+4, pinned entries)")

    # Cross-check the specialized dense chains against the generic loop_f.
    probe = [VALID[(l * 37 + 5) % 216] for l in range(N_LINKS)]
    sp_gen = sum(loop_f(probe, p) for p in PLAQ)
    sr_gen = sum(loop_f(probe, r) for r in RECT)
    assert (sp_gen, sr_gen) == sums(probe), "dense chain vs generic loop_f"
    print("PASS  dense-chain oracle == generic loop_f oracle")

    # anchors: (a) all-identity, (b) single code 9, (c) asymmetric spread.
    ident = [0] * N_LINKS
    assert sums(ident) == (1152, 2304)
    assert increment(1152, 2304) == 1_161_977_387_651_968
    assert ideal_k_out(ident, 0) == 1_161_977_387_651_968
    k_hi = MOD - 7
    assert ideal_k_out(ident, k_hi) == (k_hi + 1_161_977_387_651_968) % MOD
    one = [0] * N_LINKS
    one[0] = 9
    assert sums(one) == (1128, 2232)
    assert increment(1128, 2232) == 961_889_421_308_584
    for pos in (1, 47, 100, 191):
        ls = [0] * N_LINKS
        ls[pos] = 9
        assert sums(ls) == (1128, 2232), f"link-choice invariance at {pos}"
    asym = [VALID[l % 216] for l in range(N_LINKS)]
    assert sums(asym) == (-100, 31)
    assert increment(-100, 31) == 1_255_773_687_108_314
    assert ideal_k_out(asym, 5) == 1_255_773_687_108_319
    print("PASS  3 anchor vectors (identity / single-code9 / spread)")

    # Exact check_shot semantics.
    task = LatticeGaugePlaquette()
    cases = [
        (ident, MOD - 3),
        (asym, 0),
        (asym, 0x4D2C6A7EF9DB2),
    ]
    for links, k_in in cases:
        tc = make_tc(links, k_in)
        ideal = ideal_k_out(links, k_in)
        assert task.check_shot(tc, make_read(links, ideal)), "exact output rejected"
        for d in (-1, 1):
            assert not task.check_shot(tc, make_read(links, ideal + d)), (
                f"inexact output accepted at δ={d}"
            )
        for l in (0, 95, 191):
            bad = list(links)
            bad[l] = VALID[(IDX[bad[l]] + 1) % 216]
            assert not task.check_shot(tc, make_read(bad, ideal)), f"corrupt link {l}"
    print("PASS  exact-output semantics incl. modular seam; link corruption")

    # zero_budget_semantics: check_shot is stateless; every wrong shot is
    # simply False (STRICT fails the run on the first one).
    task = LatticeGaugePlaquette()
    links = [0] * N_LINKS
    tc = make_tc(links, 12_345)
    ideal = ideal_k_out(links, 12_345)
    good = make_read(links, ideal)
    bad_k = make_read(links, ideal + 1)
    corrupt = list(links)
    corrupt[17] = VALID[1]
    bad_link = make_read(corrupt, ideal)
    assert task.check_shot(tc, good)
    for read in (bad_k, bad_link, bad_k):
        assert not task.check_shot(tc, read)
    assert task.check_shot(tc, good)  # verdicts stay independent across shots
    print("PASS  zero budget: wrong shots are False, verdicts stateless")

    # separation_floor (both signs).
    min_p = min(abs(_centered(A_P * d * sg)) for d in range(1, 10) for sg in (1, -1))
    min_r = min(abs(_centered(-A_R_MAG * d * sg)) for d in range(1, 10) for sg in (1, -1))
    assert min_p == A_P and min_r == A_R_MAG
    assert min_p > 0 and min_r > 0
    print("PASS  separation floor (distinct trace values remain distinct)")

    # gen_test_byte_budget_and_boundaries.
    task = TASK
    xof = _FixedXof(b"\xff" * 1544)
    tc = task.gen_test(xof)
    assert tc is not None, "gen_test never skips"
    assert xof.pos == 1544, "must draw exactly 1,544 bytes"
    assert len(tc.inputs) == 193 and len(tc.expected) == 193
    assert VALID[215] == 490
    for i in range(N_LINKS):
        assert tc.inputs[i] == 490
        assert tc.expected[i] == tc.inputs[i], f"link {i} preservation"
    assert tc.inputs[192] == (1 << 51) - 1
    xof = _FixedXof(b"\x00" * 1544)
    tc = task.gen_test(xof)
    assert xof.pos == 1544
    assert all(tc.inputs[i] == 0 for i in range(N_LINKS))
    assert tc.expected[192] == 1_161_977_387_651_968
    q = ((1 << 64) - 1) // 216  # = ⌊2^64/216⌋ (216 ∤ 2^64)
    assert (q * 216) >> 64 == 0
    assert ((q + 1) * 216) >> 64 == 1
    for u, want_code in ((q, VALID[0]), (q + 1, VALID[1])):
        data = bytearray(1544)
        data[:8] = u.to_bytes(8, "little")
        xof = _FixedXof(bytes(data))
        tc = task.gen_test(xof)
        assert xof.pos == 1544
        assert tc.inputs[0] == want_code
    print("PASS  gen_test byte budget, wide-word boundaries, k mask")

    # seed_domain_stream_sanity (1,000 shots on the raw seed-domain SHAKE stream,
    # not the per-shot seed derivation).
    from uncompute.evalcore import Xof

    sx = Xof()
    sx.update(task.seed_domain)
    tot_p = tot_r = 0
    shots = 1000
    for shot in range(shots):
        tc = task.gen_test(sx)
        assert tc is not None, "gen_test never skips"
        links = []
        for i in range(N_LINKS):
            code = tc.inputs[i]
            assert IDX[code] != 0xFF, "invalid code drawn"
            assert tc.expected[i] == tc.inputs[i]
            links.append(code)
        k = tc.inputs[192]
        assert k < MOD
        s_p, s_r = sums(links)
        assert -576 <= s_p <= 1152, f"S_P out of range at shot {shot}"
        assert -1152 <= s_r <= 2304, f"S_R out of range at shot {shot}"
        assert tc.expected[192] == (k + increment(s_p, s_r)) & (MOD - 1)
        tot_p += s_p
        tot_r += s_r
    mean_p = tot_p / shots
    mean_r = tot_r / shots
    assert abs(mean_p) < 4.0, f"S_P sample mean {mean_p} far from 0"
    assert abs(mean_r) < 6.0, f"S_R sample mean {mean_r} far from 0"
    print(
        f"PASS  seed-domain stream sanity (S_P mean {mean_p:+.3f}, "
        f"S_R mean {mean_r:+.3f} over {shots} shots)"
    )

    print("lattice-gauge-plaquette grader self-test: ALL PASS")
