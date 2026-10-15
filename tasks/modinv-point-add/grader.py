"""Grader for `modinv-point-add`: point addition with a per-shot classical addend.

The task map is affine point addition |P⟩ → |P+Q⟩ on the curve y² = x³ + 7
over F_p, p = 2²⁵⁶ − 2³² − 977. Registers 0 and 1 are the 256-qubit target
coordinates (P.x, P.y → (P+Q).x, (P+Q).y). Registers 2 and 3 are 256-wide
Bit-kind input registers that carry Q.x and Q.y. The evaluator writes Q into
bit space for each shot and never reads those registers back. Coordinates are
little-endian canonical residues mod p (README §1.1). The map has no control.

The addend is input data, not a constant. A circuit must be correct for any
addend Q, not for one fixed point. A circuit may compute any classical
function of Q at no cost. Every coupling of Q into the quantum state bills at
the promoted kind: a bit-conditioned CX counts as a Toffoli.
Bit wires do not
count toward peak qubits.

Failure policy: a shot fails if either output coordinate is wrong, the global
sign is wrong, or any workspace qubit is nonzero at the end. Each shot gets
one verdict. A run passes with at most 20 failed shots of 100,000
(``failure_policy = Unified(20)``; README §2).

Test distribution: s and t are two raw 256-bit little-endian draws (64 XOF
bytes per shot, a constant consumption). The fixed-base multiply reduces them
modulo the group order n, with a modulo bias of about 2⁻¹²⁷. P = s·G and
Q = t·G are therefore independent, nearly uniform points of the prime-order
subgroup. The grader skips a draw if either operand is the identity, if
P = −Q, or if P = Q (one x-equality test covers the last two). These skips
have measure about 2⁻²⁵⁵. They are the cases where the affine formulas are
incomplete. Graders must not add shots from the excluded cases, because that
would make the task exact on a measure-zero set. No fixed anchor shots exist.
Every shot runs the full field arithmetic with a fresh addend, so a circuit
specialized to one Q cannot pass.

Curve arithmetic: fixed-base 12-bit windows over G (22 affine tables, built
at import and normalized with one Montgomery batch inversion), Jacobian
additions, and one batched modular inversion per shot for the affine outputs.
`gmpy2` is an optional accelerator (about 2.3×). The plain-int path is
normative and gives identical results.

Run this file directly for the self-test: group-law anchors, the policy pin,
and the gen_test pins.
"""

import hashlib
import json
import pathlib

from uncompute.evalcore import (
    BIT,
    QUBIT,
    RegSpec,
    Task,
    TestCase,
    Unified,
)

try:  # optional accelerator; the int path below is normative and identical
    from gmpy2 import invert as _invert
    from gmpy2 import mpz as _mpz
except ImportError:  # pragma: no cover - exercised on zero-dep installs
    _mpz = int

    def _invert(a, m):
        return pow(a, -1, m)


# Curve parameters (README §1.2): y² = x³ + 7 over F_p.
P_MOD = _mpz(2**256 - 2**32 - 977)
N_ORD = _mpz(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141)
GX = _mpz(0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798)
GY = _mpz(0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
CURVE_B = 7
_ZERO, _ONE = _mpz(0), _mpz(1)

# Test shots and the failure budget (20 of 100,000, a 2e-4 failure rate).
# At a failure rate near 1e-4, 100,000 shots resolve the budget with about
# a 4x margin.
NUM_TESTS = 100_000
BUDGET = 20
# Fixed XOF consumption per shot: two raw 256-bit LE scalar draws (P and Q).
XOF_BLOCK = 64
# Fixed-base window width. 12 bits ⇒ 22 windows × 4,095 affine entries,
# built in ≈ 0.2 s (gmpy2) / ≈ 0.6 s (int) at import.
WBITS = 12
_WMASK = (1 << WBITS) - 1


# ─── Jacobian arithmetic on y² = x³ + 7 (identity = Z = 0) ──────────────────


def _jdbl(X1, Y1, Z1):
    A = X1 * X1 % P_MOD
    B = Y1 * Y1 % P_MOD
    C = B * B % P_MOD
    D = 2 * ((X1 + B) * (X1 + B) - A - C) % P_MOD
    E = 3 * A % P_MOD
    X3 = (E * E - 2 * D) % P_MOD
    Y3 = (E * (D - X3) - 8 * C) % P_MOD
    Z3 = 2 * Y1 * Z1 % P_MOD
    return X3, Y3, Z3


def _jadd(X1, Y1, Z1, X2, Y2, Z2):
    """General Jacobian add; handles identity, doubling, and cancellation."""
    if Z1 == 0:
        return X2, Y2, Z2
    if Z2 == 0:
        return X1, Y1, Z1
    Z1Z1 = Z1 * Z1 % P_MOD
    Z2Z2 = Z2 * Z2 % P_MOD
    U1 = X1 * Z2Z2 % P_MOD
    U2 = X2 * Z1Z1 % P_MOD
    S1 = Y1 * Z2 * Z2Z2 % P_MOD
    S2 = Y2 * Z1 * Z1Z1 % P_MOD
    if U1 == U2:
        if S1 != S2:
            return _ZERO, _ONE, _ZERO
        return _jdbl(X1, Y1, Z1)
    H = (U2 - U1) % P_MOD
    R = (S2 - S1) % P_MOD
    HH = H * H % P_MOD
    HHH = H * HH % P_MOD
    V = U1 * HH % P_MOD
    X3 = (R * R - HHH - 2 * V) % P_MOD
    Y3 = (R * (V - X3) - S1 * HHH) % P_MOD
    Z3 = H * Z1 * Z2 % P_MOD
    return X3, Y3, Z3


def _madd(X1, Y1, Z1, x2, y2):
    """Mixed add: Jacobian + affine point (Z2 = 1)."""
    if Z1 == 0:
        return x2, y2, _ONE
    Z1Z1 = Z1 * Z1 % P_MOD
    U2 = x2 * Z1Z1 % P_MOD
    S2 = y2 * Z1 * Z1Z1 % P_MOD
    if X1 == U2:
        if Y1 != S2:
            return _ZERO, _ONE, _ZERO
        return _jdbl(X1, Y1, Z1)
    H = (U2 - X1) % P_MOD
    R = (S2 - Y1) % P_MOD
    HH = H * H % P_MOD
    HHH = H * HH % P_MOD
    V = X1 * HH % P_MOD
    X3 = (R * R - HHH - 2 * V) % P_MOD
    Y3 = (R * (V - X3) - Y1 * HHH) % P_MOD
    Z3 = H * Z1 % P_MOD
    return X3, Y3, Z3


def _batch_affine(points):
    """Montgomery batch inversion: Jacobian points → affine (x, y) tuples.

    Callers must exclude identity points (Z = 0) — one zero Z would zero the
    whole prefix-product chain.
    """
    zs = [pt[2] for pt in points]
    prefix = [_ONE] * (len(zs) + 1)
    for i, z in enumerate(zs):
        prefix[i + 1] = prefix[i] * z % P_MOD
    inv = _invert(prefix[-1], P_MOD)
    out = [None] * len(points)
    for i in range(len(zs) - 1, -1, -1):
        zi = inv * prefix[i] % P_MOD
        inv = inv * zs[i] % P_MOD
        X, Y, _ = points[i]
        zi2 = zi * zi % P_MOD
        out[i] = (X * zi2 % P_MOD, Y * zi2 * zi % P_MOD)
    return out


def _build_tables():
    """TBL[w][d] = affine (d · 2^(WBITS·w)) · G for d in 1..2^WBITS−1.

    Rows are accumulated in Jacobian and normalized to affine with a single
    whole-table batch inversion, so wide windows are cheap to build.
    """
    nwin = -(-256 // WBITS)
    tbl = []
    base = (GX, GY, _ONE)  # 2^(WBITS·w) · G
    flat = []
    for _ in range(nwin):
        row = [None] * (1 << WBITS)
        acc = base
        row[1] = acc
        for d in range(2, 1 << WBITS):
            acc = _jadd(*acc, *base)
            row[d] = acc
        tbl.append(row)
        base = _jadd(*acc, *base)
        flat.extend(row[1:])
    aff = _batch_affine(flat)
    k = 0
    for row in tbl:
        for d in range(1, 1 << WBITS):
            row[d] = aff[k]
            k += 1
    return tbl


_TBL = _build_tables()


def mul_g(k):
    """k·G in Jacobian, k reduced mod the group order (identity for k ≡ 0)."""
    k = k % N_ORD
    acc = (_ZERO, _ONE, _ZERO)
    w = 0
    while k:
        d = k & _WMASK
        if d:
            acc = _madd(*acc, *_TBL[w][int(d)])
        k >>= WBITS
        w += 1
    return acc


# ─── contract.json cross-assert (fail-closed) ────────────────────────────────
# The task-defining constants are published in ./contract.json for
# circuits to load. This grader remains authoritative: it refuses to load
# when the file is missing or disagrees with its own derivations.
_contract = json.loads((pathlib.Path(__file__).resolve().parent / "contract.json").read_text())
assert (
    int(_contract["p"], 16) == int(P_MOD)
    and int(_contract["n_order"], 16) == int(N_ORD)
    and _contract["curve_b"] == CURVE_B
    and int(_contract["gx"], 16) == int(GX)
    and int(_contract["gy"], 16) == int(GY)
    and _contract["num_tests"] == NUM_TESTS
    and _contract["budget"] == BUDGET
), "contract.json disagrees with the grader's derived constants"
del _contract


class ModinvPointAdd(Task):
    name = "modinv-point-add"
    seed_domain = b"modinv-point-add"
    num_tests = NUM_TESTS
    # A wrong output, a wrong sign, and a dirty ancilla share one per-shot
    # verdict and one run budget.
    failure_policy = Unified(BUDGET)

    def schema(self):
        return [
            RegSpec(width=256, kind=QUBIT),  # 0: target_x  P.x → (P+Q).x
            RegSpec(width=256, kind=QUBIT),  # 1: target_y  P.y → (P+Q).y
            RegSpec(width=256, kind=BIT),  # 2: offset_x  Q.x (classical)
            RegSpec(width=256, kind=BIT),  # 3: offset_y  Q.y (classical)
        ]

    def gen_test(self, xof):
        buf = xof.read(XOF_BLOCK)  # constant consumption, skips included
        s = int.from_bytes(buf[:32], "little")
        t = int.from_bytes(buf[32:], "little")
        pj = mul_g(s)
        qj = mul_g(t)
        if pj[2] == 0 or qj[2] == 0:
            return None  # either operand the identity (measure ~2⁻²⁵⁵)
        ej = _jadd(*pj, *qj)
        if ej[2] == 0:
            return None  # cancellation P = −Q (P.x = Q.x)
        (px, py), (qx, qy), (ex, ey) = _batch_affine([pj, qj, ej])
        if px == qx:
            return None  # doubling P = Q (P.x = Q.x)
        return TestCase(
            inputs=[int(px), int(py), int(qx), int(qy)],
            expected=[int(ex), int(ey), None, None],
            expected_phase=False,
        )


TASK = ModinvPointAdd()


# ---------------------------------------------------------------------------
# Self-test: group-law anchors, point digests, policy pin, and gen_test pins.
# ---------------------------------------------------------------------------


def _affine(pj):
    return _batch_affine([pj])[0]


def _pt_digest(pt):
    x, y = pt
    return hashlib.sha256(int(x).to_bytes(32, "little") + int(y).to_bytes(32, "little")).hexdigest()


if __name__ == "__main__":
    M64 = (1 << 64) - 1

    # Curve membership: G and 500 pseudorandom multiples satisfy y² = x³ + 7.
    def on_curve(pt):
        x, y = _mpz(pt[0]), _mpz(pt[1])
        return (y * y - (x * x % P_MOD) * x - CURVE_B) % P_MOD == 0

    assert on_curve((GX, GY))
    state = 0x9E3779B97F4A7C15
    for _ in range(500):
        state = (state * 6364136223846793005 + 1) & M64
        pj = mul_g(state)
        assert on_curve(_affine(pj))
    print("PASS  curve membership (G + 500 pseudorandom multiples)")

    # Subgroup order: n·G = identity, (n+1)·G = G.
    assert mul_g(int(N_ORD))[2] == 0
    assert _affine(mul_g(int(N_ORD) + 1)) == (GX, GY)
    print("PASS  subgroup order (n·G = O, (n+1)·G = G)")

    # Point digests: 2G and 3G (SHA-256 over LE x ‖ LE y).
    g2 = _affine(mul_g(2))
    g3 = _affine(mul_g(3))
    assert _pt_digest(g2) == "4f32d9a2d201c0ee10da8ea869ba6ab0e9cf4557a7438e5ff70f097cde9fe188"
    assert _pt_digest(g3) == "eb231731c08a42537e7af8928f01961e65683c733635374f6804b162f21426e4"
    print("PASS  2G/3G point digests")

    # Group-law anchors (README §1.3): G + 2G = 3G, 2G + G = 3G, and
    # s·G + t·G = (s+t)·G for 200 pseudorandom scalar pairs, via the
    # affine-add path gen_test uses (jadd on affine-lifted operands).
    def affine_add(a, b):
        return _affine(_jadd(a[0], a[1], _ONE, b[0], b[1], _ONE))

    g1 = (GX, GY)
    assert affine_add(g1, g2) == g3
    assert affine_add(g2, g1) == g3
    for _ in range(200):
        state = (state * 6364136223846793005 + 1) & M64
        s = state
        state = (state * 6364136223846793005 + 1) & M64
        t = state
        lhs = affine_add(_affine(mul_g(s)), _affine(mul_g(t)))
        assert lhs == _affine(mul_g(s + t))
    print("PASS  group-law anchors (G+2G=3G, 200 pseudorandom identities)")

    # gen_test: determinism, constant 64-byte consumption, in-domain draws,
    # and the drawn triple being a genuine group identity P + Q = E.
    class _FakeXof:
        def __init__(self, seed):
            self.state = seed
            self.consumed = 0

        def read(self, n):
            self.consumed += n
            out = bytearray(n)
            for i in range(n):
                self.state = (self.state * 6364136223846793005 + 1) & M64
                out[i] = (self.state >> 24) & 0xFF
            return bytes(out)

    task = TASK
    for seed in range(64):
        a = task.gen_test(_FakeXof(seed))
        b = task.gen_test(_FakeXof(seed))
        assert a is not None  # skips have measure ~2⁻²⁵⁵
        assert a.inputs == b.inputs and a.expected == b.expected
        assert not a.expected_phase
        assert len(a.inputs) == 4 and len(a.expected) == 4
        assert a.expected[2] is None and a.expected[3] is None
        p_pt, q_pt = (a.inputs[0], a.inputs[1]), (a.inputs[2], a.inputs[3])
        e_pt = (a.expected[0], a.expected[1])
        assert on_curve(p_pt) and on_curve(q_pt) and on_curve(e_pt)
        assert a.inputs[0] != a.inputs[2]  # P.x ≠ Q.x in-domain
        # the drawn triple is a genuine group identity
        assert affine_add(p_pt, q_pt) == e_pt
        assert all(0 <= v < int(P_MOD) for v in a.inputs)
    x = _FakeXof(7)
    task.gen_test(x)
    assert x.consumed == XOF_BLOCK
    print("PASS  gen_test (deterministic, 64-byte draws, in-domain, exact)")

    # Failure-policy pin: Unified(20) at 100,000 shots and the seed domain.
    from uncompute.evalcore import resolve_failure_policy

    pol = resolve_failure_policy(task)
    assert isinstance(pol, Unified) and pol.budget == BUDGET
    assert task.num_tests == NUM_TESTS
    assert task.seed_domain == b"modinv-point-add"
    print(f"PASS  failure policy Unified({BUDGET}) @ {NUM_TESTS} shots")

    # check_shot: default exact equality on both target registers; the Bit
    # registers carry no expectation.
    tc = task.gen_test(_FakeXof(123))
    good = [tc.expected[0], tc.expected[1], 0, 0]
    assert task.check_shot(tc, good)
    assert not task.check_shot(tc, [tc.expected[0] ^ 1, tc.expected[1], 0, 0])
    assert not task.check_shot(tc, [tc.expected[0], tc.expected[1] ^ 1, 0, 0])
    print("PASS  check_shot (exact on both output registers)")

    print("modinv-point-add grader self-test: ALL PASS")
