"""Shared classical instance scaffold for the dqi-hermitian circuits.

The instance every circuit rebuilds from first principles before
emitting a circuit: GF(64) tables (exp/log, MUL64, POW62, SQT, TR,
PARB, gpow, gf64_rref), the curve points and monomial bases (PTS, KN/KIDX, FPORD,
ORDS/KNORDS/UNK, XPW/YPW), the syndrome map H, the pinned c stream
with its gamma / pay-vector solve (C_INSTANCE_TAG, buf, cb, csym, delta, gamma,
W), the semigroup <8,9> structure constants (O48/PHI48, red_mon, BSRC,
SHIFT_A/SHIFT_B, TAUORD/TIDX/NTAUP), and gen_triples —
the 15-AND Toom GF(64) bilinear decomposition (the cost basis). No
circuit emission and no contract.json I/O: each circuit keeps its own
asserts beside its build.

Each circuit loads this file BY PATH (uncompute.testing.load_module).
"""

import hashlib

import numpy as np

from uncompute.classical import GF2n

# ---------------- GF(64) tables ----------------
MODP = 0b1011011  # x^6+x^4+x^3+x+1
F64 = GF2n(6, MODP | 1 << 6)
assert F64.gen == 2, "x not primitive with this modulus"
exp_t, log_t = F64.exp, F64.log
gpow = F64.pow
MUL64 = np.array([[F64.mul(a, c) for c in range(64)] for a in range(64)], np.uint8)
POW62 = np.array([F64.inv(a) for a in range(64)], np.uint8)
SQT = MUL64.diagonal().copy()
TR = np.array([F64.trace(a) for a in range(64)], np.uint8)
PARB = np.array([bin(i).count("1") & 1 for i in range(64)], np.uint8)


# ---------------- curve, monomials ----------------
PTS = [(x, y) for x in range(64) for y in range(64) if (gpow(y, 8) ^ y) == gpow(x, 9)]
assert len(PTS) == 512
KN = []
for bb in range(8):
    aa = 0
    while 8 * aa + 9 * bb <= 150:
        KN.append((aa, bb))
        aa += 1
assert len(KN) == 123
KIDX = {m: i for i, m in enumerate(KN)}
FPORD = {}
for aa in range(64):
    for bb in range(8):
        N = 8 * aa + 9 * bb
        assert N not in FPORD
        FPORD[N] = (aa, bb)
ORDS = sorted(FPORD)
KNORDS = [N for N in ORDS if N <= 150]
UNK = [N for N in ORDS if N > 150]
assert len(KNORDS) == 123 and len(UNK) == 389
assert all(FPORD[N] in KIDX for N in KNORDS)

XPW = np.zeros((512, 80), np.uint8)
YPW = np.zeros((512, 16), np.uint8)
for i, (x, y) in enumerate(PTS):
    for a in range(80):
        XPW[i, a] = gpow(x, a)
    for bb in range(16):
        YPW[i, bb] = gpow(y, bb)
H = np.zeros((123, 512), np.uint8)
for r, (a, bb) in enumerate(KN):
    H[r, :] = MUL64[XPW[:, a], YPW[:, bb]]

# ---------------- c, gamma, w ----------------
C_INSTANCE_TAG = b"kickmix/hopi-512/fixup-c/v1"  # fixed instance constant
buf = b"".join(
    hashlib.sha256(C_INSTANCE_TAG + int(i).to_bytes(4, "little")).digest() for i in range(12)
)
assert len(buf) == 384
cb = [(buf[t >> 3] >> (t & 7)) & 1 for t in range(3072)]
csym = np.array([sum(cb[6 * i + j] << j for j in range(6)) for i in range(512)], np.uint8)


def gf2_inv(M):
    """Invert a GF(2) matrix by augmented Gauss-Jordan (asserts full rank)."""
    n = M.shape[0]
    Aug = np.concatenate([M, np.eye(n, dtype=np.uint8)], 1)
    for col in range(n):
        piv = next(rr for rr in range(col, n) if Aug[rr, col])
        if piv != col:
            Aug[[col, piv]] = Aug[[piv, col]]
        for rr in range(n):
            if rr != col and Aug[rr, col]:
                Aug[rr] ^= Aug[col]
    return Aug[:, n:]


G6 = np.zeros((6, 6), np.uint8)
for j in range(6):
    for k2 in range(6):
        G6[j, k2] = TR[MUL64[1 << j, 1 << k2]]
Ginv = gf2_inv(G6)
delta = []
for j in range(6):
    d = 0
    for k2 in range(6):
        if Ginv[j, k2]:
            d ^= 1 << k2
    delta.append(d)
for j in range(6):
    for k2 in range(6):
        assert TR[MUL64[delta[j], 1 << k2]] == (1 if j == k2 else 0)
gamma = np.zeros(512, np.uint8)
for i in range(512):
    g = 0
    for j in range(6):
        if cb[6 * i + j]:
            g ^= delta[j]
    gamma[i] = g


def gf64_rref(A, ncols):
    """Gauss-Jordan over GF(64) on columns [0, ncols) of a copy of A. The
    pivot of each column is its first nonzero entry at or below the
    current row. Returns (reduced matrix, pivot columns).
    """
    A = A.copy()
    n = A.shape[0]
    piv = []
    r = 0
    for c in range(ncols):
        if r >= n:
            break
        nz = np.nonzero(A[r:, c])[0]
        if not len(nz):
            continue
        p = r + int(nz[0])
        if p != r:
            A[[r, p]] = A[[p, r]]
        A[r] = MUL64[int(POW62[A[r, c]]), A[r]]
        col = A[:, c].copy()
        col[r] = 0
        rows = np.nonzero(col)[0]
        if len(rows):
            A[rows] ^= MUL64[col[rows][:, None], A[r][None, :]]
        piv.append(c)
        r += 1
    return A, piv


def _solve_w():
    V = np.zeros((512, 513), np.uint8)
    for n2, N in enumerate(ORDS):
        a, bb = FPORD[N]
        V[:, n2] = MUL64[XPW[:, a], YPW[:, bb]]
    V[:, 512] = gamma
    R, piv = gf64_rref(V, 512)
    assert len(piv) == 512, "V singular?!"
    return R[:, 512]


W = _solve_w()
# verify w on random sparse errors
_rngv = np.random.default_rng(0)
for _ in range(4):
    e = np.zeros(512, np.uint8)
    pos = _rngv.choice(512, 47, replace=False)
    e[pos] = _rngv.integers(1, 64, 47)
    par = 0
    for i in pos:
        par ^= int(PARB[csym[i] & e[i]])
    p2 = 0
    for n2, N in enumerate(ORDS):
        a, bb = FPORD[N]
        SN = 0
        for i in pos:
            SN ^= int(MUL64[e[i], MUL64[XPW[i, a], YPW[i, bb]]])
        p2 ^= int(TR[MUL64[int(W[n2]), SN]])
    assert par == p2, "w vector wrong"

# ---------------- structure constants ----------------
O48 = [N for N in ORDS if N <= 75]
assert len(O48) == 48 and O48[47] == 75 and O48[46] == 74
PHI48 = [FPORD[N] for N in O48]


def red_mon(A_, B_):
    ts = [(A_, B_)] if B_ <= 7 else [(A_ + 9, B_ - 8), (A_, B_ - 7)]
    out = []
    for Aa2, Bb2 in ts:
        assert Bb2 <= 7
        while Aa2 >= 64:
            Aa2 -= 63
        out.append((Aa2, Bb2))
    return out


BSRC = [[None] * 48 for _ in range(48)]
for m in range(48):
    am, bm = PHI48[m]
    for l in range(48):
        al, bl = PHI48[l]
        ts = red_mon(am + al, bm + bl)
        rs = []
        for t2 in ts:
            assert 8 * t2[0] + 9 * t2[1] <= 150 and t2 in KIDX
            rs.append(KIDX[t2])
        BSRC[m][l] = rs

SHIFT_A = [red_mon(PHI48[l][0] + 1, PHI48[l][1]) for l in range(47)]
SHIFT_B = [red_mon(PHI48[l][0], PHI48[l][1] + 1) for l in range(46)]
_tau_set = set()
for ts_ in SHIFT_A + SHIFT_B:
    for a2, b2 in ts_:
        _tau_set.add(8 * a2 + 9 * b2)
TAUORD = sorted(_tau_set)
assert 83 not in TAUORD and all(o in FPORD and o <= 82 for o in TAUORD)
NTAU = len(TAUORD)
assert red_mon(PHI48[47][0] + 1, PHI48[47][1]) == [(7, 3)]
assert red_mon(PHI48[46][0], PHI48[46][1] + 1) == [(7, 3)]

# --- PADDED tauN layout: group by residue class rho = o mod 8 (== y-degree
# of the canonical monomial), slots indexed by x-exponent a_o with gap slots
# kept at |0>. Keeps tap addresses affine for loop-form emitters. ---
CLS = {}
for o in TAUORD:
    rho_ = o % 8
    ao_ = (o - 9 * rho_) // 8
    assert 8 * ao_ + 9 * rho_ == o and ao_ >= 0
    CLS.setdefault(rho_, []).append(ao_)
for aos_ in CLS.values():
    aos_.sort()
CBASE = {}
CAMIN = {}
CLEN = {}
_nb = 0
for rho_ in sorted(CLS):
    aos_ = CLS[rho_]
    CAMIN[rho_] = aos_[0]
    CLEN[rho_] = aos_[-1] - aos_[0] + 1
    CBASE[rho_] = _nb
    _nb += CLEN[rho_]
NTAUP = _nb
assert NTAUP <= 90
TIDX = {}
for o in TAUORD:
    rho_ = o % 8
    ao_ = (o - 9 * rho_) // 8
    TIDX[o] = CBASE[rho_] + (ao_ - CAMIN[rho_])
assert len(set(TIDX.values())) == NTAU and max(TIDX.values()) < NTAUP


# --- 15-AND bilinear GF(64) multiplier (Toom over GF(4)) — cost basis ---
# local: gadgets.gf_tower.GF mul_xor is an 18-AND Karatsuba; this
# Toom decomposition is 15 ANDs per GF(64) product.
def gen_triples():
    w4 = exp_t[21]
    w4sq = int(MUL64[w4, w4])
    assert int(MUL64[w4, w4sq]) == 1 and w4 != 1
    assert w4sq == (w4 ^ 1)
    X = 2
    X2 = int(MUL64[X, X])
    cols = []
    for xp in (1, X, X2):
        for wp in (1, w4):
            cols.append(int(MUL64[wp, xp]))
    Mb = np.zeros((6, 6), np.uint8)
    for cidx, v in enumerate(cols):
        for bit in range(6):
            Mb[bit, cidx] = (v >> bit) & 1
    Mbi = gf2_inv(Mb)

    gmul = F64.mul

    def coords(v):
        bits = np.array([(v >> bit) & 1 for bit in range(6)], np.uint8)
        cbits = (Mbi @ bits) % 2
        a = []
        for j3 in range(3):
            g2 = 0
            if cbits[2 * j3]:
                g2 ^= 1
            if cbits[2 * j3 + 1]:
                g2 ^= w4
            a.append(g2)
        return a

    pts = [0, 1, w4, w4sq]
    lags = []
    for i4, p in enumerate(pts):
        num = 1
        den = 1
        for j4, q in enumerate(pts):
            if j4 == i4:
                continue
            num = gmul(num, X ^ q)
            den = gmul(den, p ^ q)
        lags.append(gmul(num, int(POW62[den])))
    X4 = gmul(X2, X2)
    kinf = X4
    for i4, p in enumerate(pts):
        p4 = gmul(gmul(p, p), gmul(p, p))
        kinf ^= gmul(p4, lags[i4])
    Klist = lags + [kinf]
    tri = []
    for t4, K in enumerate(Klist):
        mask_lo = 0
        mask_hi = 0
        for bit in range(6):
            a = coords(1 << bit)
            if t4 < 4:
                p = pts[t4]
                val = a[0] ^ gmul(a[1], p) ^ gmul(a[2], gmul(p, p))
            else:
                val = a[2]
            if val == 0:
                lo, hi = 0, 0
            elif val == 1:
                lo, hi = 1, 0
            elif val == w4:
                lo, hi = 0, 1
            elif val == w4sq:
                lo, hi = 1, 1
            else:
                raise AssertionError(val)
            mask_lo |= lo << bit
            mask_hi |= hi << bit
        Kw = gmul(K, w4)
        tri.append((mask_lo, K ^ Kw))
        tri.append((mask_hi, K))
        tri.append((mask_lo ^ mask_hi, Kw))
    idx = np.arange(64)
    accv = np.zeros((64, 64), np.uint8)
    for ma, om in tri:
        pa = PARB[idx & ma]
        accv ^= np.where((pa[:, None] & pa[None, :]) == 1, om, 0).astype(np.uint8)
    assert (accv == MUL64).all(), "bilinear decomposition wrong"
    rem = tri[1:]
    ordd = [tri[0]]
    while rem:
        cur = ordd[-1][0]
        j2 = min(range(len(rem)), key=lambda i3: bin(cur ^ rem[i3][0]).count("1"))
        ordd.append(rem.pop(j2))
    return ordd


TRIPLES = gen_triples()
NT = len(TRIPLES)
assert NT == 15  # 15 ANDs per variable GF(64) product
