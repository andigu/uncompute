#!/usr/bin/env python3
"""dqi-hermitian pinned instance: generation + self-verification.

Everything the grader and circuits pin is derived here from first principles
and checked: field, curve points, parity-check matrix H, fixup vector c,
weight law, and known-answer triples. Run:  python instance.py
Exits nonzero on any failed check; prints the SHA-256 provenance anchors
that the grader asserts.

Pinned conventions:
  - Field: GF(64) = GF(2)[x]/(x^6 + x^4 + x^3 + x + 1)  (int 91, the
    `galois` package default / Conway polynomial; x is primitive).
    Elements are ints 0..63 in the polynomial basis, bit j = coeff of x^j.
  - Curve: y^8 + y = x^9 over GF(64); its 512 affine points ordered
    lexicographically by the integer pair (x, y), x major.
  - H (123 x 512): rows = monomials x^a y^b with 0 <= b <= 7 and
    8a + 9b <= 150 (pole order at P_inf <= A = 150), ordered b-major then
    a ascending; entry = monomial evaluated at the point.
  - Code: ker H = [512, 389, >= 96] (Goppa bound d >= A - 2g + 2, g = 28);
    unique decoding radius ell = 47.
  - Error packing: e in GF(64)^512 packs to 3072 bits, symbol i at bits
    6i..6i+5.
  - Fixup vector c: 3072 bits from SHA-256 counter-XOF over the tag
    b"kickmix/hopi-512/fixup-c/v1" (block i = SHA-256(tag || u32_le(i)),
    concatenated, truncated to 384 bytes), bytes packed little-endian
    (bit k of c = bit k%8 of byte k//8). Must lie outside the F2-rowspace
    of the binary expansion of H (else the phase would be linear in s).
  - Weight law: |w_k|^2 of the principal eigenvector of the DQI tridiagonal
    (m = 512, ell = 47, p = r/q = 28/64 -- bent target sets over GF(64),
    size q/2 - sqrt(q)/2 = 28), quantized to a /2^16 grid by
    largest-remainder rounding; reported as the cumulative CUM array.
"""

import hashlib
import sys

import numpy as np

Q0 = 8
Q = 64  # q0^2
M = 512  # q0^3 affine points
GENUS = Q0 * (Q0 - 1) // 2  # 28
A_POLE = 150
N_CHECKS = 123  # dim L(150) = A - g + 1
D_GOPPA = A_POLE - 2 * GENUS + 2  # 96
ELL = (D_GOPPA - 1) // 2  # 47
R_BENT = Q // 2 - Q0 // 2  # 28
POLY = 91  # x^6 + x^4 + x^3 + x + 1 (galois default)
C_INSTANCE_TAG = b"kickmix/hopi-512/fixup-c/v1"  # fixed instance constant
C_BYTES_LEN = 3 * M // 4  # 384 bytes = 3072 bits = 512 symbols x 6

FAIL = []


def check(name, ok):
    print(f"  {'OK ' if ok else 'FAIL'} {name}")
    if not ok:
        FAIL.append(name)


# ---------------- GF(64) = GF(2)[x]/(x^6+x^4+x^3+x+1) ----------------
def _mul(a, b):
    r = 0
    for i in range(6):
        if (b >> i) & 1:
            r ^= a << i
    for i in range(11, 5, -1):
        if (r >> i) & 1:
            r ^= (POLY | 64) << (i - 6)
    return r


MUL = [[_mul(a, b) for b in range(Q)] for a in range(Q)]
MULn = np.array(MUL, dtype=np.uint8)


def gpow(a, k):
    r = 1
    for _ in range(k):
        r = MUL[r][a]
    return r


# primitivity of x (= 2): order must be exactly 63
orders = {k for k in range(1, 64) if gpow(2, k) == 1}
check("x is primitive (order 63)", min(orders) == 63)

INV = [0] * Q
for a in range(1, Q):
    for b in range(1, Q):
        if MUL[a][b] == 1:
            INV[a] = b
check("inverses total", all(MUL[a][INV[a]] == 1 for a in range(1, Q)))

# independent cross-derivation of the arithmetic via the galois package
import galois

GFg = galois.GF(2**6)
check("galois default poly == 91", int(GFg.irreducible_poly) == POLY | 64)
ag = GFg(np.arange(Q, dtype=np.uint8))
outer = ag.reshape(-1, 1) * ag.reshape(1, -1)
check("MUL table matches galois", np.array_equal(np.asarray(outer, dtype=np.uint8), MULn))

# ---------------- curve points ----------------
points = [(x, y) for x in range(Q) for y in range(Q) if (gpow(y, Q0) ^ y) == gpow(x, Q0 + 1)]
check(f"affine point count == {M}", len(points) == M)

# ---------------- monomial basis of L(150 P_inf) ----------------
MONOMIALS = []
for b in range(Q0):
    a = 0
    while Q0 * a + (Q0 + 1) * b <= A_POLE:
        MONOMIALS.append((a, b))
        a += 1
check(f"monomial count == {N_CHECKS}", len(MONOMIALS) == N_CHECKS)
# pole orders distinct (semigroup <8,9> representation unique in this range)
poles = [Q0 * a + (Q0 + 1) * b for a, b in MONOMIALS]
check("pole orders distinct", len(set(poles)) == len(poles))

H = np.zeros((N_CHECKS, M), dtype=np.uint8)
for r_i, (a, b) in enumerate(MONOMIALS):
    for c_i, (x, y) in enumerate(points):
        H[r_i, c_i] = MUL[gpow(x, a)][gpow(y, b)]


def gf_rank(mat):
    """Rank over GF(64) by elimination."""
    mat = mat.astype(np.uint8).copy()
    rank, rows, cols = 0, mat.shape[0], mat.shape[1]
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if mat[i, c]), None)
        if piv is None:
            continue
        mat[[r, piv]] = mat[[piv, r]]
        inv = INV[mat[r, c]]
        mat[r] = MULn[mat[r], np.full(cols, inv, dtype=np.uint8)]
        for i in range(rows):
            if i != r and mat[i, c]:
                f = mat[i, c]
                mat[i] ^= MULn[mat[r], np.full(cols, f, dtype=np.uint8)]
        r += 1
        rank += 1
        if r == rows:
            break
    return rank


check("rank_F64(H) == 123 (full)", gf_rank(H) == N_CHECKS)

# ---------------- fixup vector c ----------------
stream = b"".join(
    hashlib.sha256(C_INSTANCE_TAG + i.to_bytes(4, "little")).digest()
    for i in range((C_BYTES_LEN + 31) // 32)
)[:C_BYTES_LEN]
C_BITS = np.array([(stream[k // 8] >> (k % 8)) & 1 for k in range(8 * C_BYTES_LEN)], dtype=np.uint8)
print(f"  c popcount: {int(C_BITS.sum())} / {8 * C_BYTES_LEN}")

# binary expansion of H: syndrome bit rows as F2 vectors over the 3072 e-bits.
# bit t of syndrome symbol r_i = sum_j Tr-like F2-linear map; build by symbol:
# syndrome_r = sum_j H[r,j] * e_j  (GF(64)); bit t of that is F2-linear in the
# 6 bits of each e_j with coefficient bit t of (H[r,j] * x^u) for e-bit u.
B2 = np.zeros((N_CHECKS * 6, 8 * C_BYTES_LEN), dtype=np.uint8)
for r_i in range(N_CHECKS):
    for j in range(M):
        h = H[r_i, j]
        if h == 0:
            continue
        for u in range(6):
            prod = MUL[h][1 << u]
            for t in range(6):
                if (prod >> t) & 1:
                    B2[6 * r_i + t, 6 * j + u] = 1


def f2_rank(mat):
    mat = np.packbits(mat, axis=1)
    rows = [int.from_bytes(row.tobytes(), "big") for row in mat]
    rank = 0
    basis = []
    for v in rows:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
            rank += 1
    return rank


rk = f2_rank(B2)
check("rank_F2(binary H) == 738", rk == N_CHECKS * 6)
rk_with_c = f2_rank(np.vstack([B2, C_BITS[None, :]]))
check("c outside F2-rowspace of H", rk_with_c == N_CHECKS * 6 + 1)

# ---------------- weight law ----------------
p_frac = R_BENT / Q
d_tri = (Q - 2 * R_BENT) / np.sqrt(R_BENT * (Q - R_BENT))
T = np.zeros((ELL + 1, ELL + 1))
for k in range(ELL + 1):
    T[k, k] = k * d_tri
    if k < ELL:
        T[k, k + 1] = T[k + 1, k] = np.sqrt((k + 1) * (M - k))
evals, evecs = np.linalg.eigh(T)
lam = evals[-1]
w = evecs[:, -1]
w = w * np.sign(w[np.argmax(np.abs(w))])
check("principal eigenvector nonnegative", (w >= -1e-12).all())
w = np.clip(w, 0, None)
mass = w * w
mass /= mass.sum()

# largest-remainder quantization to /2^16
scaled = mass * 65536
base = np.floor(scaled).astype(np.int64)
rem = scaled - base
need = 65536 - int(base.sum())
order = np.argsort(-rem)
base[order[:need]] += 1
check("dyadic masses sum to 65536", int(base.sum()) == 65536)
tv = float(np.abs(base / 65536 - mass).sum()) / 2
print(f"  quantization TV vs real eigenvector: {tv:.3e}")
CUM = np.cumsum(base)
nz = np.nonzero(base)[0]
print(
    f"  law support: k in [{nz.min()}, {nz.max()}], mode k={int(np.argmax(base))} "
    f"({base.max() / 65536:.3f})"
)

# instance numbers (cross-check the design table)
c1 = np.sqrt(R_BENT * (Q - R_BENT)) / Q
fbar = (2 * R_BENT - Q) / Q
s_m = 0.5 * (1 + fbar + 2 * c1 * lam / M)
prange = (N_CHECKS + p_frac * (M - N_CHECKS)) / M
print(f"  DQI <s>/m = {s_m:.6f}, single-trial Prange = {prange:.6f}, gap = {s_m - prange:.6f}")
check("gap matches design (~0.142)", abs((s_m - prange) - 0.1419) < 5e-3)


# ---------------- syndrome + phase KATs ----------------
def syndrome(e_sparse):
    s = np.zeros(N_CHECKS, dtype=np.uint8)
    for j, v in e_sparse:
        s ^= MULn[H[:, j], np.full(N_CHECKS, v, dtype=np.uint8)]
    return s


def phase(e_sparse):
    par = 0
    for j, v in e_sparse:
        for u in range(6):
            if (v >> u) & 1:
                par ^= int(C_BITS[6 * j + u])
    return par


kats = [[], [(0, 1)], [(511, 63)], [(137, 42)], [(3, 7), (100, 1), (200, 33), (511, 62)]]
print("  KAT (grader test vectors):")
for e in kats:
    s = syndrome(e)
    print(
        f"    e={e} -> sha256(synd)={hashlib.sha256(s.tobytes()).hexdigest()[:16]} phase={phase(e)}"
    )

# ---------------- provenance anchors ----------------
pt_bytes = b"".join(bytes([x, y]) for x, y in points)
print("  provenance:")
print(f"    points  sha256 = {hashlib.sha256(pt_bytes).hexdigest()}")
print(f"    H       sha256 = {hashlib.sha256(H.tobytes()).hexdigest()}")
print(f"    c       sha256 = {hashlib.sha256(stream).hexdigest()}")
print(f"    CUM[{len(CUM)}]  = {CUM.tolist()}")

if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("all instance checks pass")
