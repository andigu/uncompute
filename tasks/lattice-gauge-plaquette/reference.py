#!/usr/bin/env python3
"""Reference implementation for uncompute task `lattice-gauge-plaquette`
(the full Symanzik-improved magnetic Trotter layer on a 4^3 lattice).

Target map (one diagonal magnetic layer for the SU(3) discrete subgroup
Sigma(72x3), |G| = 216, over a 4^3 periodic lattice, d = 3):

    |g_0 .. g_191>|k>  ->  |g_0 .. g_191>|k + a_P*S_P + a_R*S_R mod 2^51>

    S_P = sum over the 192 plaquettes  P_ij(x) of f(P)
    S_R = sum over the 384 rectangles  R_ij(x) of f(R)
    f   = 2*Re tr  (the 7-value Sigma(72x3) class function)

  g_l : 192 nine-qubit link registers, normal-form encoding of Osorio Perez
        et al., arXiv:2511.17437, verbatim (216 valid codes of 512).
        Link indexing (pinned): site s = x + 4y + 16z, link l = 3s + mu,
        mu in {0,1,2}; registers declared l = 0..191 in order, then k (192).
  k   : 51-bit modular accumulator, randomized every shot. A phase-gradient
        eigenstate is a downstream use, not part of this interface.
  a_P : pinned odd 51-bit hash-derived constant,
        a_P = (first 51 bits, MSB-first, of SHA-256(INSTANCE_TAG)) | 1
            = 0x8EBA84840FB = 9,808,233,644,283      (raw prefix EVEN; |1 fired)
  a_R : ratio-pinned to the tree-level improvement couplings
        beta_V1/beta_V0 = -1/20 exactly (physics, not hash):
        a_R = -round(a_P/20) mod 2^51 = 2^51 - 490,411,682,214
            = 0x7FF8DD12F965A = 2,251,309,402,003,034

The 51-bit constants define the exact modular map. Loop products (inverses
on the return links):

    P_ij(x) = U_i(x) U_j(x+i) U_i(x+j)^-1 U_j(x)^-1
    R_ij(x) = U_i(x) U_i(x+i) U_j(x+2i) U_i(x+i+j)^-1 U_i(x+j)^-1 U_j(x)^-1

Enumeration order: sites ascending; per site, plaquettes over
(i,j) in {(0,1),(0,2),(1,2)}, then rectangles over ordered (i,j), i != j,
lexicographic. S_P sums 192 terms, S_R sums 384.

The task is exact: a shot is correct iff all 192 links round-trip and the
accumulator equals the specified modular sum. Every tested shot must be
correct; ancilla cleanup and phase cleanliness are also exact.

The group machinery (exact Z[w][1/3] arithmetic, generators, relations,
encoding, |409> anchor, group axioms) is built inline in Sec. 2 from the
generator matrices of arXiv:2511.17437 Sec. II (Sigma(36x3) companion
arXiv:2405.05973). Everything here is stdlib-only and self-checking. Run this
file: exit status 0 means all asserts passed. It prints PASS lines and the
three ANCHOR lines (the fixed cross-validation contract). These asserts are
the ground truth that the grader must encode.

Exposes:
  VALID            - sorted list of the 216 valid codes
  MUL, INV         - group tables (from the inlined group build)
  TRACE            - dict[code] -> f = 2*Re tr(M_code)
  A_P, A_R, M, EPS, NUM_TESTS - pinned constants
  LOOP_P, LOOP_R   - loop tables: tuples of (link_index, invert_flag)
  layer_b(links), task_map(links, k), check_shot(...)
"""

import hashlib
import itertools
import math
import statistics
from collections import Counter
from fractions import Fraction

# ----------------------------------------------------------------------------
# 0. Anchor hashlib against the NIST FIPS 180-2 "abc" known-answer vector, so
#    the constant derivation below is grounded in an external test vector.
# ----------------------------------------------------------------------------
assert (
    hashlib.sha256(b"abc").hexdigest()
    == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
), "hashlib SHA-256 fails the NIST 'abc' KAT"
print("PASS  SHA-256 NIST 'abc' known-answer vector")

# ----------------------------------------------------------------------------
# 1. Pinned constants: a_P hash-derived (generic compilation), a_R ratio-pinned
#    (-1/20 is definitional tree-level improvement, PARAMS Sec. 2).
# ----------------------------------------------------------------------------
N_LINK = 9
N_LINKS = 192
M = 51
MOD = 1 << M
EPS = 0
NUM_TESTS = 100_000
INSTANCE_TAG = b"kickmix-plaquette-trace-accumulate-v2"  # fixed instance constant

PINNED_INSTANCE_TAG_SHA256 = "011d7509081f5efa37139519357a698a9b53bbe990fd36521ef37f31fe9e544d"
PINNED_A_P = 0x8EBA84840FB  # = 9,808,233,644,283
PINNED_RAW = 0x8EBA84840FA  # raw 51-bit prefix (EVEN; |1 does real work)
PINNED_ROUND = 490_411_682_214  # round(a_P/20), unambiguous since a_P odd
PINNED_A_R = 0x7FF8DD12F965A  # = 2^51 - round(a_P/20) = 2,251,309,402,003,034


def derive_constants():
    digest = hashlib.sha256(INSTANCE_TAG).digest()
    raw = int.from_bytes(digest, "big") >> (256 - M)  # first 51 bits, MSB-first
    a_p = raw | 1  # force odd (total derivation)
    q, r = divmod(a_p, 20)
    rnd = q + (1 if r > 10 else 0)  # round-half never occurs
    return a_p, (MOD - rnd) % MOD


A_P, A_R = derive_constants()
assert hashlib.sha256(INSTANCE_TAG).hexdigest() == PINNED_INSTANCE_TAG_SHA256, (
    "instance-tag hash drift"
)
_raw = int.from_bytes(hashlib.sha256(INSTANCE_TAG).digest(), "big") >> (256 - M)
assert _raw == PINNED_RAW and _raw & 1 == 0, "raw prefix expected even 0x8EBA84840FA"
assert A_P == PINNED_A_P and A_P & 1 == 1, f"constant drift: a_P = {A_P:#x}"
# round unambiguity: a_P odd => a_P/20 is never a half-integer (a_P % 20 is odd):
assert A_P % 20 == 3 and A_P % 20 != 10
_rnd = (MOD - A_R) % MOD
assert _rnd == PINNED_ROUND, f"round drift: {_rnd}"
# relation error |a_R + a_P/20| (signed reading) <= 0.5 count per unit S_R:
_rel_err = abs(PINNED_ROUND - A_P / 20)
assert _rel_err <= 0.5 and abs(_rel_err - 0.15) < 0.001, _rel_err
assert A_R == PINNED_A_R, f"constant drift: a_R = {A_R:#x}"
print(
    f"PASS  a_P = {A_P:#x} = {A_P} (raw prefix even, |1 fired); "
    f"a_R = 2^51 - {PINNED_ROUND} = {A_R} (ratio -1/20, rel err {_rel_err:.4f})"
)

# ----------------------------------------------------------------------------
# 2. Group build + trace table: the Sigma(72x3)
#    construction of Osorio Perez et al., arXiv:2511.17437 Sec. II (generators,
#    relations, encoding; Sigma(36x3) companion arXiv:2405.05973), in exact
#    Z[w][1/3] arithmetic. The build IS its verification (generator relations,
#    |409> anchor, bijectivity, group axioms, Latin square, center, order
#    spectrum, 16 classes, subgroup tower, pinned triples) -- every assert
#    runs on import.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Exact arithmetic in the ring Z[w][1/3],  w = e^{2 pi i/3},  w^2 = -1 - w.
# An entry is a normalized tuple (A, B, k) representing (A + B*w) / 3^k with
# not (A % 3 == 0 and B % 3 == 0) unless k == 0 (canonical form => hashable
# equality is exact equality of complex numbers).
# ----------------------------------------------------------------------------


def _norm(A, B, k):
    while k > 0 and A % 3 == 0 and B % 3 == 0:
        A //= 3
        B //= 3
        k -= 1
    return (A, B, k)


def cadd(x, y):
    A1, B1, k1 = x
    A2, B2, k2 = y
    k = max(k1, k2)
    A = A1 * 3 ** (k - k1) + A2 * 3 ** (k - k2)
    B = B1 * 3 ** (k - k1) + B2 * 3 ** (k - k2)
    return _norm(A, B, k)


def cmul(x, y):
    A1, B1, k1 = x
    A2, B2, k2 = y
    # (A1 + B1 w)(A2 + B2 w) = A1A2 + (A1B2 + B1A2) w + B1B2 w^2,  w^2 = -1-w
    return _norm(A1 * A2 - B1 * B2, A1 * B2 + B1 * A2 - B1 * B2, k1 + k2)


ZERO = (0, 0, 0)
ONE = (1, 0, 0)
W = (0, 1, 0)
W2 = (-1, -1, 0)
SC = {0: ONE, 1: W, 2: W2}  # w^p


def mmul(P, Q):
    return tuple(
        tuple(
            cadd(cadd(cmul(P[i][0], Q[0][j]), cmul(P[i][1], Q[1][j])), cmul(P[i][2], Q[2][j]))
            for j in range(3)
        )
        for i in range(3)
    )


def smul(s, P):
    return tuple(tuple(cmul(s, P[i][j]) for j in range(3)) for i in range(3))


def mpow(P, n):
    R = I3
    for _ in range(n):
        R = mmul(R, P)
    return R


I3 = ((ONE, ZERO, ZERO), (ZERO, ONE, ZERO), (ZERO, ZERO, ONE))

# ----------------------------------------------------------------------------
# Generators, arXiv:2511.17437 Eq. (1).  1/(sqrt(3) i) = -(1 + 2w)/3 exactly
# (check: (1+2w)^2 = 1 + 4w + 4w^2 = -3).
# ----------------------------------------------------------------------------

INV_SQRT3I = (-1, -2, 1)  # (-1 - 2w)/3
assert cmul((1, 2, 0), INV_SQRT3I) == ONE

C = ((ONE, ZERO, ZERO), (ZERO, W, ZERO), (ZERO, ZERO, W2))
E = ((ZERO, ONE, ZERO), (ZERO, ZERO, ONE), (ONE, ZERO, ZERO))
V = smul(INV_SQRT3I, ((ONE, ONE, ONE), (ONE, W, W2), (ONE, W2, W)))
X = smul(INV_SQRT3I, ((ONE, ONE, W2), (ONE, W, W), (W, ONE, W)))

# Relations, arXiv:2511.17437 Eq. (2)
assert mmul(X, C) == mmul(mmul(mpow(C, 2), E), X), "X C = C^2 E X"
assert mmul(X, E) == smul(W2, mmul(mmul(C, E), X)), "X E = w^2 C E X"
assert mmul(X, V) == smul(W2, mmul(mmul(E, mpow(V, 3)), X)), "X V = w^2 E V^3 X"
assert mpow(C, 3) == I3 and mpow(E, 3) == I3
assert mpow(V, 4) == I3 and mpow(X, 4) == I3

# ----------------------------------------------------------------------------
# Encoding, arXiv:2511.17437 Eqs. (3)-(4):  |u t s r1 r0 q1 q0 p1 p0>
# ----------------------------------------------------------------------------


def encode(p, q, r, s, t, u):
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


def decode(n):
    p = ((n >> 1) & 1) * 2 + (n & 1)
    q = ((n >> 3) & 1) * 2 + ((n >> 2) & 1)
    r = ((n >> 5) & 1) * 2 + ((n >> 4) & 1)
    return (p, q, r, (n >> 6) & 1, (n >> 7) & 1, (n >> 8) & 1)


def is_valid(n):
    if not 0 <= n < 512:
        return False
    p, q, r, _, _, _ = decode(n)
    return p != 3 and q != 3 and r != 3


# Normal-form enumeration: 216 tuples -> 216 DISTINCT matrices (bijection).
MAT = {}  # code -> matrix
for p, q, r in itertools.product(range(3), repeat=3):
    for s, t, u in itertools.product(range(2), repeat=3):
        _Mg = smul(SC[p], mmul(mmul(mpow(C, q), mpow(E, r)), mmul(mpow(V, 2 * s + t), mpow(X, u))))
        MAT[encode(p, q, r, s, t, u)] = _Mg

VALID = sorted(MAT)
assert len(VALID) == 216 and len(set(MAT.values())) == 216, (
    "normal form is not a bijection onto 216 distinct matrices"
)
assert sum(is_valid(n) for n in range(512)) == 216
CODE = {M: n for n, M in MAT.items()}  # matrix -> code

# Paper worked example, arXiv:2511.17437 Eq. (4):
#   (i/sqrt3) [[w^2, w, w], [1, w, 1], [1, 1, w]]  # noqa: ERA001 - paper equation
#   = w^1 C^2 E^1 V^1 X^1  ->  |110011001> = |409>
I_SQRT3 = (1, 2, 1)  # i/sqrt3 = (1 + 2w)/3
M409 = smul(I_SQRT3, ((W2, W, W), (ONE, W, ONE), (ONE, ONE, W)))
assert encode(1, 2, 1, 0, 1, 1) == 409 == int("110011001", 2)
assert CODE[M409] == 409, "paper Eq. (4) anchor |409> failed"

# BFS closure from generators reproduces exactly the same 216 matrices.
seen = {I3}
frontier = [I3]
while frontier:
    g = frontier.pop()
    for h in (C, E, V, X):
        pr = mmul(g, h)
        if pr not in seen:
            seen.add(pr)
            frontier.append(pr)
assert seen == set(MAT.values()), "closure(C,E,V,X) != normal-form image"

# ----------------------------------------------------------------------------
# Multiplication table (the grader's oracle), inverses, group axioms.
# ----------------------------------------------------------------------------

MUL = {}
for a in VALID:
    Ma = MAT[a]
    for b in VALID:
        MUL[(a, b)] = CODE[mmul(Ma, MAT[b])]  # closure: KeyError if it fails

IDENT = CODE[I3]
assert IDENT == 0
assert all(MUL[(0, a)] == a and MUL[(a, 0)] == a for a in VALID)

INV = {}
for a in VALID:
    for b in VALID:
        if MUL[(a, b)] == 0:
            INV[a] = b
            break
assert len(INV) == 216
assert all(MUL[(INV[a], a)] == 0 for a in VALID), "two-sided inverse"

# Latin-square property (each row/col of the table is a permutation).
for a in VALID:
    assert len({MUL[(a, b)] for b in VALID}) == 216
    assert len({MUL[(b, a)] for b in VALID}) == 216

# Associativity: inherited from matrix multiplication by construction; verify
# on a deterministic pseudo-random sample as a self-check of the table build.
rng_state = 0x6B69636B6D6978  # arbitrary fixed LCG seed


def _lcg():
    global rng_state
    rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) % 2**64
    return rng_state


for _ in range(20000):
    a = VALID[_lcg() % 216]
    b = VALID[_lcg() % 216]
    c = VALID[_lcg() % 216]
    assert MUL[(MUL[(a, b)], c)] == MUL[(a, MUL[(b, c)])]

# ----------------------------------------------------------------------------
# Group identification: this really is Sigma(72x3).
# ----------------------------------------------------------------------------

# Center = Z_3 = {w^p I}: codes 0, 1, 2.
center = [a for a in VALID if all(MUL[(a, b)] == MUL[(b, a)] for b in VALID)]
assert center == [0, 1, 2], "center must be Z_3 = {I, wI, w^2 I}"


# Element-order spectrum (matches the 16-class structure of the character
# table of arXiv:2511.17437: classes of orders 1,2,3,4,6,12).
def order_of(a):
    x, n = a, 1
    while x != 0:
        x = MUL[(x, a)]
        n += 1
    return n


spec = {}
for a in VALID:
    spec[order_of(a)] = spec.get(order_of(a), 0) + 1
assert spec == {1: 1, 2: 9, 3: 26, 4: 54, 6: 18, 12: 108}, spec

# 16 conjugacy classes == 16 irreps (arXiv:2511.17437 Sec. II: 1^(a), 2,
# 3^(a), 3^(a)*, 6, 6*, 8;  4*1 + 4 + 4*9 + 4*9 + 36 + 36 + 64 = 216).
unassigned, nclasses = set(VALID), 0
while unassigned:
    a = next(iter(unassigned))
    cls = {MUL[(MUL[(g, a)], INV[g])] for g in VALID}
    unassigned -= cls
    nclasses += 1
assert nclasses == 16

# Abelianization G/[G,G] has order 4 (=> in particular NO homomorphism onto
# Z_3, so the central Z_3 does NOT split off: G != Z_3 x H).
comm = {MUL[(MUL[(a, b)], MUL[(INV[a], INV[b])])] for a in VALID for b in VALID}
D, frontier = set(comm), list(comm)
while frontier:
    x = frontier.pop()
    for y in list(D):
        for z in (MUL[(x, y)], MUL[(y, x)]):
            if z not in D:
                D.add(z)
                frontier.append(z)
assert len(D) == 54 and 216 // len(D) == 4

# Subgroup tower:  <C,E,V> = Sigma(36x3), order 108, exactly the u=0 sector
# (arXiv:2511.17437 Eq. (26): U_x^{S72x3} = U_x^{S36x3} . U_{x,X}).
cC, cE, cV, cX = CODE[C], CODE[E], CODE[V], CODE[X]
assert (cC, cE, cV, cX) == (4, 16, 128, 256)
H = {0, cC, cE, cV}
frontier = list(H)
while frontier:
    x = frontier.pop()
    for g in (cC, cE, cV):
        z = MUL[(x, g)]
        if z not in H:
            H.add(z)
            frontier.append(z)
assert len(H) == 108 and H == {n for n in VALID if not (n >> 8) & 1}

# N = <wI, C, E> = codes < 64 (s=t=u=0): normal, order 27 (Heisenberg/Delta(27):
# E C = w C E), quotient of order 8 isomorphic to Q8 (orders 1,2,4^6).
N27 = {n for n in VALID if n < 64}
assert len(N27) == 27
assert all(MUL[(MUL[(g, n)], INV[g])] in N27 for g in VALID for n in N27)
assert MUL[(cE, cC)] == MUL[(1, MUL[(cC, cE)])]  # E C = w (C E)
qmul = {}
for a in VALID:
    for b in VALID:
        k = (a >> 6, b >> 6)
        v = MUL[(a, b)] >> 6
        assert qmul.setdefault(k, v) == v, "quotient by N27 ill-defined"


def qorder(x):
    y, n = x, 1
    while y != 0:
        y = qmul[(y, x)]
        n += 1
    return n


assert sorted(qorder(x) for x in range(8)) == [1, 2, 4, 4, 4, 4, 4, 4]  # Q8

# ----------------------------------------------------------------------------
# Pinned product triples (g, h, g*h) - the grader must reproduce ALL of these.
# First four are paper-derived identities; the rest pin generic entries.
# ----------------------------------------------------------------------------

PINNED_TRIPLES = [
    # paper Eq. (2) relation X C = C^2 E X  -> code (0,2,1,0,0,1) = 280
    (256, 4, 280),
    # Heisenberg core: C*E = 20 = (0,1,1,0,0,0);  E*C = 21 = w C E
    (4, 16, 20),
    (16, 4, 21),
    # X^2 = C^2 V^2 = (0,2,0,1,0,0) = 72: the C^{2 u_g u_h} V^{2 u_g u_h}
    # correction of arXiv:2511.17437 Eq. (28) at u_g = u_h = 1
    (256, 256, 72),
    # V^2 = (0,0,0,1,0,0) = 64  (s-bit is the V^2 flag)
    (128, 128, 64),
    # identity behaviour on the paper's worked example |409>
    (409, 0, 409),
    (0, 409, 409),
    # generic noncommuting pair
    (409, 280, 233),
    (280, 409, 134),
    # inverse: 409^-1 = 484 = (0,1,2,1,1,1)
    (409, 484, 0),
    (484, 409, 0),
]
for g, h, gh in PINNED_TRIPLES:
    assert MUL[(g, h)] == gh, (g, h, gh, MUL[(g, h)])
assert INV[409] == 484

print(
    "PASS  group build verified (relations, |409> anchor, axioms, "
    "Latin square, center, spectrum, classes, tower, pinned triples)"
)


# f(code) = 2*Re tr(M_code), exact. Entry (A_, B_, k) represents
# (A_ + B_*w)/3^k with w = e^{2 pi i/3}; Re(w) = -1/2, so 2*Re = (2A_ - B_)/3^k.
def _trace2re(code):
    Mm = MAT[code]
    tot = Fraction(0)
    for i in range(3):
        A_, B_, k = Mm[i][i]
        tot += Fraction(2 * A_ - B_, 3**k)
    assert tot.denominator == 1, f"trace not integral at L=2 for code {code}"
    return int(tot)


TRACE = {c: _trace2re(c) for c in VALID}

PINNED_TRACE_VALUES = (-3, -2, -1, 0, 1, 2, 6)
PINNED_MULTIPLICITY = {6: 1, -3: 2, -2: 9, 1: 18, 0: 24, 2: 54, -1: 108}
assert tuple(sorted(set(TRACE.values()))) == PINNED_TRACE_VALUES, sorted(set(TRACE.values()))
assert dict(Counter(TRACE.values())) == PINNED_MULTIPLICITY, Counter(TRACE.values())
print(f"PASS  trace set {PINNED_TRACE_VALUES}, multiplicities {PINNED_MULTIPLICITY}")

# Class function: f is constant on conjugacy classes (exhaustive 216 x 216).
for x in VALID:
    fx = TRACE[x]
    for g in VALID:
        assert TRACE[MUL[(MUL[(g, x)], INV[g])]] == fx, (g, x)
print("PASS  f is a class function (exhaustive 216 x 216 conjugation check)")

# f(g^-1) = f(g) (SU(3): tr(U+) = conj(tr U), Re equal); identity/center pins.
assert all(TRACE[INV[c]] == TRACE[c] for c in VALID)
assert TRACE[0] == 6 and [c for c in VALID if TRACE[c] == 6] == [0]
assert sorted(c for c in VALID if TRACE[c] == -3) == [1, 2]
print("PASS  f(g^-1) = f(g); f = 6 only at identity; f = -3 at nontrivial center")

# Table digests (fixed cross-implementation contract).
_mul_bytes = bytearray()
for a_ in VALID:
    for b_ in VALID:
        _mul_bytes += int(MUL[(a_, b_)]).to_bytes(2, "little")
assert (
    hashlib.sha3_256(bytes(_mul_bytes)).hexdigest()
    == "a9ef423e2669f722b1b93c0c6a77911318b46f40f57dd0c89a33b9c051d551e3"
)
_trace_bytes = bytes((TRACE[c] & 0xFF) for c in VALID)  # i8 two's complement
assert (
    hashlib.sha3_256(_trace_bytes).hexdigest()
    == "4a1ff65e31ae491e6ef73a7f3219e6cc112a7fa6de897c2b09f977dc4ca3ccca"
)
_inv_bytes = bytearray()
for c in VALID:
    _inv_bytes += int(INV[c]).to_bytes(2, "little")
assert (
    hashlib.sha3_256(bytes(_inv_bytes)).hexdigest()
    == "394fbe2d07f1078df6ea88d7b50e0489b5dd2999312c9cf8384cf193a154bf39"
)
print("PASS  digests: MUL (shared with component), TRACE (i8), INV (u16-LE)")

# ----------------------------------------------------------------------------
# 3. Lattice geometry (pinned, PARAMS Sec. 3): 4^3 periodic, d = 3.
#    site s = x + 4y + 16z; link l = 3s + mu. Loop tables are lists of
#    (link_index, invert_flag) in product order, inverses on return links.
# ----------------------------------------------------------------------------
L = 4


def site(x, y, z):
    return (x % L) + L * (y % L) + L * L * (z % L)


def link(p, mu):
    return 3 * site(*p) + mu


_E = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _add(p, e, n=1):
    return tuple((p[a] + n * e[a]) % L for a in range(3))


def _plaq(p, i, j):
    """P_ij(x) = U_i(x) U_j(x+i) U_i(x+j)^-1 U_j(x)^-1."""
    return (
        (link(p, i), 0),
        (link(_add(p, _E[i]), j), 0),
        (link(_add(p, _E[j]), i), 1),
        (link(p, j), 1),
    )


def _rect(p, i, j):
    """R_ij(x) = U_i(x) U_i(x+i) U_j(x+2i) U_i(x+i+j)^-1 U_i(x+j)^-1 U_j(x)^-1."""
    return (
        (link(p, i), 0),
        (link(_add(p, _E[i]), i), 0),
        (link(_add(p, _E[i], 2), j), 0),
        (link(_add(_add(p, _E[i]), _E[j]), i), 1),
        (link(_add(p, _E[j]), i), 1),
        (link(p, j), 1),
    )


# Sites ascending in s = x + 4y + 16z (x fastest):
_SITES = [(s % 4, (s // 4) % 4, s // 16) for s in range(64)]
assert [site(*p) for p in _SITES] == list(range(64))

LOOP_P = tuple(_plaq(p, i, j) for p in _SITES for (i, j) in ((0, 1), (0, 2), (1, 2)))
LOOP_R = tuple(
    _rect(p, i, j) for p in _SITES for (i, j) in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
)
assert len(LOOP_P) == 192 and len(LOOP_R) == 384

# Shape: plaquettes 4 links (2 inverted), rectangles 6 links (3 inverted);
# within a loop all link indices are distinct (L = 4 > 2 prevents coincidences).
for lp in LOOP_P:
    assert len(lp) == 4 and sum(f for _, f in lp) == 2
    assert len({l for l, _ in lp}) == 4
for lr in LOOP_R:
    assert len(lr) == 6 and sum(f for _, f in lr) == 3
    assert len({l for l, _ in lr}) == 6
print(
    "PASS  loop tables: 192 plaquettes (4 links, 2 inverted), "
    "384 rectangles (6 links, 3 inverted), distinct links per loop"
)


# Loop closure, verified geometrically as a connected walk: a forward link
# must be based at the cursor (cursor += e_mu); an inverted link is traversed
# backward (cursor -= e_mu, link based at the new cursor). Every loop returns
# to its start.
def _closes(loop):
    s0 = loop[0][0] // 3
    start = pos = (s0 % 4, (s0 // 4) % 4, s0 // 16)
    for l, f in loop:
        mu, s = l % 3, l // 3
        base = (s % 4, (s // 4) % 4, s // 16)
        if not f:
            assert base == pos, (loop, l)
            pos = _add(pos, _E[mu])
        else:
            pos = _add(pos, _E[mu], -1)
            assert base == pos, (loop, l)
    return pos == start


assert all(_closes(lp) for lp in LOOP_P)
assert all(_closes(lr) for lr in LOOP_R)
print("PASS  loop closure: all 576 loops are connected walks returning to start")

# Link membership, counted empirically. Plaquettes: exactly 2(d-1) = 4 per
# link. Rectangles: exactly 12 per link (384*6/192 = 12 by counting slots) --
# split 8 as a LONG side (direction-i slots) + 4 as a SHORT side (direction-j
# slots), as PARAMS Sec. 3 states. The single-code-9 anchor below
# corroborates the total (S_R drops by exactly 12 * 6).
_cnt_p, _cnt_r, _cnt_long, _cnt_short = Counter(), Counter(), Counter(), Counter()
for lp in LOOP_P:
    for l, _ in lp:
        _cnt_p[l] += 1
for lr in LOOP_R:
    _long_dir = lr[0][0] % 3  # direction i of this rectangle
    for l, _ in lr:
        _cnt_r[l] += 1
        (_cnt_long if l % 3 == _long_dir else _cnt_short)[l] += 1
assert set(_cnt_p) == set(range(192)) and set(_cnt_p.values()) == {4}
assert set(_cnt_r) == set(range(192)) and set(_cnt_r.values()) == {12}
assert set(_cnt_long.values()) == {8} and set(_cnt_short.values()) == {4}
print(
    "PASS  link membership: every link in exactly 4 plaquettes and "
    "12 rectangles (8 long-side + 4 short-side)"
)

assert N_LINK * N_LINKS == 1728  # code qubits; k is register 192
print("PASS  registers: 192 links x 9 qubits = 1728 code qubits, then k (51)")


# ----------------------------------------------------------------------------
# 4. Layer oracle: the graded map.
# ----------------------------------------------------------------------------
def _loop_f(loop, links):
    prod = 0  # identity code
    for l, f in loop:
        g = links[l]
        prod = MUL[(prod, INV[g] if f else g)]
    return TRACE[prod]


def layer_b(links):
    """links[192] -> (S_P, S_R, k-increment mod 2^51)."""
    assert len(links) == N_LINKS and all(g in TRACE for g in links)
    S_P = sum(_loop_f(lp, links) for lp in LOOP_P)
    S_R = sum(_loop_f(lr, links) for lr in LOOP_R)
    return S_P, S_R, (A_P * S_P + A_R * S_R) % MOD


def task_map(links, k):
    """Exact reference: (links, k) -> (links, k + a_P*S_P + a_R*S_R mod 2^51)."""
    assert 0 <= k < MOD
    return tuple(links), (k + layer_b(links)[2]) % MOD


# ----------------------------------------------------------------------------
# 5. Anchor vectors (the fixed cross-validation contract).
#    (a) all links identity: every loop product = I, f = +6 everywhere, so
#        S_P = 192*6 = 1152, S_R = 384*6 = 2304.
#    (b) links[0] = 9 (a valid code, = VALID[7]), all others identity: every
#        containing loop's product is conjugate to g^{+-1}, so its f is
#        TRACE[9] = 0; S_P = 1152 - 4*6, S_R = 2304 - 12*6. Link-choice
#        invariant (uniform membership + class function) -- asserted.
#    (c) links[l] = VALID[l % 216] (the l-th valid code, sorted).
# ----------------------------------------------------------------------------
PINNED_ANCHORS = [
    # fields: name, links, S_P, S_R, increment
    ("identity", [0] * N_LINKS, 1152, 2304, 1_161_977_387_651_968),
    ("single-code9", [0] * N_LINKS, 1128, 2232, 961_889_421_308_584),
    ("spread", [VALID[l % 216] for l in range(N_LINKS)], -100, 31, 1_255_773_687_108_314),
]
PINNED_ANCHORS[1][1][0] = 9
assert VALID[7] == 9 and TRACE[9] == 0

for name, links, sp, sr, inc in PINNED_ANCHORS:
    got = layer_b(links)
    assert got == (sp, sr, inc), (name, got)
    assert (A_P * sp + A_R * sr) % MOD == inc
    # through the full map at k = 0 and at a wraparound k:
    assert task_map(links, 0)[1] == inc
    k_hi = MOD - 5
    assert task_map(links, k_hi)[1] == (k_hi + inc) % MOD
    print(f"ANCHOR {name}: S_P={sp} S_R={sr} inc={inc}")

# identity-anchor arithmetic double-check + (b) link-choice invariance:
assert (A_P * 1152 + A_R * 2304) % MOD == 1_161_977_387_651_968
for _j in (1, 95, 191):
    _lk = [0] * N_LINKS
    _lk[_j] = 9
    assert layer_b(_lk) == (1128, 2232, 961_889_421_308_584), _j
print(
    "PASS  3 anchor vectors (incl. wraparound k); single-code9 anchor is "
    "link-choice invariant (checked at links 1, 95, 191)"
)


# ----------------------------------------------------------------------------
# 6. Exact output semantics.
# ----------------------------------------------------------------------------
def centered(v):
    """Centered residue mod 2^51 in (-2^50, 2^50]."""
    v %= MOD
    return v - MOD if v > MOD >> 1 else v


def check_shot(links_in, k_in, links_out, k_out):
    """Per-shot correctness: all 192 links and the modular sum are exact."""
    if tuple(links_out) != tuple(links_in):
        return False
    return abs(centered(k_out - k_in - layer_b(links_in)[2])) <= EPS


for _name, links, _sp, _sr, inc in PINNED_ANCHORS:
    gs = tuple(links)
    for k_in in (0, MOD - 1, 0x5A5A5_A5A5A5A5):
        exact = (k_in + inc) % MOD
        assert check_shot(gs, k_in, gs, exact)
        for delta in (-1, 1, 1 << 40):
            assert not check_shot(gs, k_in, gs, (exact + delta) % MOD)
# corrupting any single link register is a wrong shot even with a perfect k:
_gs = tuple(PINNED_ANCHORS[2][1])
_inc = PINNED_ANCHORS[2][4]
for i in (0, 91, 191):
    bad = list(_gs)
    bad[i] = VALID[(VALID.index(bad[i]) + 1) % 216]
    assert not check_shot(_gs, 0, tuple(bad), _inc)
print("PASS  exact-output check incl. mod-2^51 seam; link corruption rejected")

# ----------------------------------------------------------------------------
# 7. Separation theorem: a single wrong trace class on
#    one loop shifts the increment by a_{P/R}*d, d in +-{1..9} (the nonzero
#    ordered trace differences). The exhaustive minima are both nonzero.
# ----------------------------------------------------------------------------
DIFFS = sorted({x - y for x in PINNED_TRACE_VALUES for y in PINNED_TRACE_VALUES if x != y})
assert DIFFS == [d for d in range(-9, 10) if d != 0]
PINNED_SEP_P = 9_808_233_644_283  # = a_P itself (argmin d = +-1)
PINNED_SEP_R = 490_411_682_214  # = round(a_P/20) (argmin d = +-1)
sep_p = min(abs(centered(A_P * d)) for d in DIFFS)
sep_r = min(abs(centered(A_R * d)) for d in DIFFS)
assert sep_p == PINNED_SEP_P == A_P, sep_p
assert sep_r == PINNED_SEP_R == MOD - A_R, sep_r
assert sep_p > 0 and sep_r > 0
print(
    f"PASS  separation: min_d |centered(a_P*d)| = {sep_p} "
    f"and min_d |centered(a_R*d)| = {sep_r}; both nonzero"
)

# ----------------------------------------------------------------------------
# 8. Concentration: exact single-loop moments from the class table, additive
#    variance across loops via the pairwise-uncorrelated lemma
#    (condition on shared links; a product with a free uniform factor is
#    uniform, so conditional means vanish) -- verified empirically below.
# ----------------------------------------------------------------------------
_classes = [(f, n) for f, n in PINNED_MULTIPLICITY.items()]
assert sum(n for _, n in _classes) == 216
assert sum(f * n for f, n in _classes) == 0  # E[f] = 0 exact
assert sum(f * f * n for f, n in _classes) == 432  # Var[f] = 432/216 = 2 exact
SIGMA_SP = math.sqrt(2 * 192)  # = sqrt(384) = 19.60
SIGMA_SR = math.sqrt(2 * 384)  # = sqrt(768) = 27.71
assert 19.5 < SIGMA_SP < 19.7 and 27.6 < SIGMA_SR < 27.8
print(
    f"PASS  moments: E[f] = 0, Var[f] = 2 exact; sigma(S_P) = sqrt(384) = "
    f"{SIGMA_SP:.4f}, sigma(S_R) = sqrt(768) = {SIGMA_SR:.4f}"
)

# Empirical check of the lemma: two plaquettes sharing exactly one link
# (P_01(0) and P_02(0) share link 0). Seeded Monte Carlo over the 7-link
# union; sample covariance of (f, f') must be 0 within 4 standard errors.
_p1, _p2 = LOOP_P[0], LOOP_P[1]
_shared = {l for l, _ in _p1} & {l for l, _ in _p2}
assert _shared == {0}, _shared
_union = sorted({l for l, _ in _p1} | {l for l, _ in _p2})
assert len(_union) == 7
_NMC = 200_000
_stream = hashlib.shake_256(b"lattice-gauge-plaquette/covariance-selftest").digest(
    8 * len(_union) * _NMC
)
_links = [0] * N_LINKS
_t1 = _t2 = 0
_prods = []
_off = 0
for _ in range(_NMC):
    for l in _union:
        u = int.from_bytes(_stream[_off : _off + 8], "little")
        _off += 8
        _links[l] = VALID[(u * 216) >> 64]
    f1, f2 = _loop_f(_p1, _links), _loop_f(_p2, _links)
    _t1 += f1
    _t2 += f2
    _prods.append(f1 * f2)
_cov = sum(_prods) / _NMC - (_t1 / _NMC) * (_t2 / _NMC)
_se = statistics.pstdev(_prods) / math.sqrt(_NMC)
assert abs(_cov) <= 4 * _se, (_cov, _se)
print(
    f"PASS  pairwise-uncorrelated lemma: shared-link plaquette pair, "
    f"{_NMC:,} samples, cov = {_cov:+.5f} (|cov| <= 4*SE, SE = {_se:.5f})"
)

# ----------------------------------------------------------------------------
# 9. Test-model sanity (gen_test recipe): each link drawn uniform over the 216
#    valid codes via the wide-word reduction floor(u64 * 216 / 2^64) on
#    constant XOF bytes (draw bias <= 2^-56, documented and negligible); k
#    uniform over 2^51. Seeded model: 4096 simulated shots of S_P.
# ----------------------------------------------------------------------------
_NS = 4096
_st = hashlib.shake_256(b"lattice-gauge-plaquette/fuzzmodel-selftest").digest(8 * N_LINKS * _NS)
_vals = []
_off = 0
for s in range(_NS):
    lk = [
        VALID[(int.from_bytes(_st[_off + 8 * i : _off + 8 * i + 8], "little") * 216) >> 64]
        for i in range(N_LINKS)
    ]
    _off += 8 * N_LINKS
    _vals.append(sum(_loop_f(lp, lk) for lp in LOOP_P))
    if s < 4:  # full-map round trip on a few shots: exact output passes, +1 fails
        _sp, _sr, _inc = layer_b(lk)
        _k = (s * 0x9E3779B97F4A7 + 12345) % MOD
        assert check_shot(lk, _k, tuple(lk), (_k + _inc) % MOD)
        assert not check_shot(lk, _k, tuple(lk), (_k + _inc + 1) % MOD)
_mean = sum(_vals) / _NS
_std = statistics.pstdev(_vals)
assert abs(_mean) <= 5 * SIGMA_SP / math.sqrt(_NS), _mean  # |mean| <= 1.53
assert 18.5 < _std < 21.0, _std  # spread ~ 19.6
print(
    f"PASS  fuzz model: {_NS} simulated shots, S_P mean = {_mean:+.3f} "
    f"(bound {5 * SIGMA_SP / math.sqrt(_NS):.3f}), std = {_std:.2f} "
    f"(sigma = {SIGMA_SP:.1f}); exact check green on sampled full shots"
)

if __name__ == "__main__":
    print()
    print(f"a_P        = {A_P:#x} ({A_P}), odd (|1 fired on even prefix)")
    print(f"a_R        = {A_R:#x} ({A_R}) = 2^51 - round(a_P/20), ratio -1/20")
    print(f"map        : |g_0..g_191>|k> -> |g_0..g_191>|k + a_P*S_P + a_R*S_R mod 2^{M}>")
    print("lattice    : 4^3 periodic, d = 3; 192 plaquettes, 384 rectangles")
    print("accuracy   : exact modular sum; links exact")
    print("contract   : STRICT, zero wrong shots")
    print(f"separation : a_P-class {PINNED_SEP_P}, a_R-class {PINNED_SEP_R} (both nonzero)")
    print("all asserts passed")
