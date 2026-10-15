#!/usr/bin/env python3
"""Generator for the sweep constants of the discovered circuit.

Derives, from the shared instance scaffold alone, the fixed tables the
circuit loads from constants.json:

  * OMB_ORDS   — the omega basis: semigroup pole orders <= 138;
  * VMAP_KEYS/VMAP_ROWS — the omega MAC schedule: for locator slot l
    and basis order N, the F64-linear form V_{N,l} of the 123 known
    syndromes with  omega_N = sum_l lam_l * V_{N,l}(S)  (the truncated
    key equation: eta extracted triangularly from the residue pairing
    Res(phi_nu * eta) = S_nu at P_inf, omega = lam-series * eta / dx,
    coefficients peeled top-down against the monomial basis);
  * LAMH_MASK / OMH_MASK — per-point Hasse gather coefficients of the
    locator and omega bases, levels 0..3 / 0..2.  On the curve
    y^8 + y = x^9 in char 2, dy = x^8 dx and y(x_P + t) = y_P +
    x_P^8 t (mod t^9), so every level is a mod-2-binomial pattern:
      D^(k)(x^a y^b)(P) = sum_{i+j=k, i in a, j in b}
                          x_P^(a-i+8j) * y_P^(b-j);
  * KN_SORT — the known syndrome orders, ascending.
  * LAM_EVAL / G_EVAL / INTERP — the omega evaluation-interpolation
    tables.  The same bilinear map factors through one GF(64) product
    per curve point: q = prod_{i<13}(x - c_i) * prod_{j<3}(y - d_j)
    has pole order 131 and 131 explicit rational simple zeros; E =
    the first 103 of them with L(130Pinf - E) = 0 (greedy evaluation
    rank on L(130Pinf)); G(S) = the unique member of L(55Pinf + E)
    window-equivalent to F (the window-match system solves uniquely
    and G stays F64-linear in S); lam * G(S) lies in the 206-dim
    L(130Pinf + E), so the bank is one product per point at 206
    chosen points: LAM_EVAL / G_EVAL are the two per-point gather
    masks and INTERP = window-read . eval^-1 (rank 103; the 8 basis
    orders above 130 are structurally zero rows).  G_EVAL columns
    follow KN_SORT order; the circuit permutes to register order via
    KIDX[FPORD[.]].  The derivation asserts fail-closed: the MAC
    schedule (2,452 nonzero entries), existence + uniqueness of G,
    the exhaustive 48x123 basis-pair bilinear identity, the keyless
    zero rows, and the 206-dim product span.

The derivation is deterministic.  generate() returns the canonical
constants.json text; main() writes it.

RUN: uv run python gen_constants.py   (writes constants.json)
"""

import base64
import json
import pathlib

import numpy as np

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent
inst = load_module(HERE.parent / "_instance64.py", "dqi_hermitian_instance64")

MUL64, POW62 = inst.MUL64, inst.POW62
KNORDS, FPORD, O48 = inst.KNORDS, inst.FPORD, inst.O48
XPW, YPW = inst.XPW, inst.YPW
gpow = inst.gpow
gf64_rref = inst.gf64_rref

DEPTH = 420
NPTS = 512
OMB_BOUND = 138
MLEV = 3
NXF, NYF = 13, 3  # fiber counts behind q: pole order 8*13 + 9*3 = 131
DEGE = 103  # deg E; Z' keeps the other 28 fiber zeros
NEV = 206  # evaluation points = dim L(130Pinf + E)


gmul = inst.F64.mul


def ginv(a):
    assert a
    return inst.F64.inv(a)


def gmm(A, B):
    """GF(64) matmul: out[i,k] = XOR_j MUL64[A[i,j], B[j,k]]."""
    A = np.asarray(A, np.uint8)
    B = np.asarray(B, np.uint8)
    out = np.zeros((A.shape[0], B.shape[1]), np.uint8)
    for j in range(A.shape[1]):
        col = A[:, j]
        if col.any():
            out ^= MUL64[col[:, None], B[j][None, :]]
    return out


def gf64_kernel(A):
    """Right-kernel basis of A over GF(64), rows = basis vectors."""
    A = np.asarray(A, np.uint8)
    R, piv = gf64_rref(A, A.shape[1])
    free = [c for c in range(A.shape[1]) if c not in piv]
    ker = np.zeros((len(free), A.shape[1]), np.uint8)
    for i, fc in enumerate(free):
        ker[i, fc] = 1
        for r, pc in enumerate(piv):
            ker[i, pc] = R[r, fc]
    return ker


class GFRank:
    """Incremental GF(64) row-echelon rank tracker (greedy selections)."""

    def __init__(self):
        self.rows = {}

    def try_add(self, v):
        v = np.asarray(v, np.uint8).copy()
        for p in sorted(self.rows):
            c = int(v[p])
            if c:
                v ^= MUL64[c, self.rows[p]]
        nz = np.nonzero(v)[0]
        if not len(nz):
            return False
        p = int(nz[0])
        self.rows[p] = MUL64[ginv(int(v[p])), v]
        return True


# ---------------- P_inf series machinery (uniformizer u = x/y) ------------
class Ser:
    """Truncated Laurent series over F64: coeffs c[i] at exponent
    lo + i.
    """

    __slots__ = ("c", "lo")

    def __init__(self, lo, c):
        self.lo = lo
        self.c = np.asarray(c, np.uint8)

    @staticmethod
    def mono(exp, coeff, depth):
        c = np.zeros(depth, np.uint8)
        c[0] = coeff
        return Ser(exp, c)

    def __mul__(self, o):
        n = min(len(self.c), len(o.c))
        out = np.zeros(n, np.uint8)
        for i in range(n):
            a = int(self.c[i])
            if a:
                out[i:] ^= MUL64[a, o.c[: n - i]]
        return Ser(self.lo + o.lo, out)

    def __add__(self, o):
        lo = min(self.lo, o.lo)
        hi = min(self.lo + len(self.c), o.lo + len(o.c))
        n = hi - lo
        out = np.zeros(n, np.uint8)
        s0 = self.lo - lo
        out[s0 : s0 + min(len(self.c), n - s0)] ^= self.c[: min(len(self.c), n - s0)]
        o0 = o.lo - lo
        out[o0 : o0 + min(len(o.c), n - o0)] ^= o.c[: min(len(o.c), n - o0)]
        return Ser(lo, out)

    def coeff(self, exp):
        i = exp - self.lo
        return int(self.c[i]) if 0 <= i < len(self.c) else 0

    def norm(self):
        nz = np.nonzero(self.c)[0]
        k = int(nz[0])
        return Ser(self.lo + k, self.c[k:])

    def inv(self, depth):
        s = self.norm() if self.c[0] == 0 else self
        l0 = ginv(int(s.c[0]))
        out = np.zeros(depth, np.uint8)
        out[0] = l0
        for k in range(1, depth):
            acc = 0
            for i in range(1, min(k + 1, len(s.c))):
                acc ^= gmul(int(s.c[i]), int(out[k - i]))
            out[k] = gmul(l0, acc)
        return Ser(-s.lo, out)


def build_xy(depth):
    """x = u^-8 X(u), y = u^-9 X(u) with X the unit solution of
    X^8 = X^7 + u^63 (from y = x/u on the curve).
    """
    X = np.zeros(depth, np.uint8)
    X[0] = 1
    for _ in range(depth + 2):
        X7 = Ser(0, X)
        p = X7
        for _ in range(6):
            p = p * X7
        corr_c = np.zeros(depth, np.uint8)
        if 63 < depth:
            corr_c[63] = 1
        corr = Ser(0, corr_c) * p.inv(depth)
        Xn = np.zeros(depth, np.uint8)
        Xn[0] = 1
        n = min(depth, len(corr.c))
        Xn[:n] ^= corr.c[:n]
        if np.array_equal(Xn, X):
            break
        X = Xn
    return Ser(-8, X), Ser(-9, X.copy())


def basis_upto(bound):
    out = []
    for a in range(bound // 8 + 1):
        for b in range(8):
            o = 8 * a + 9 * b
            if o <= bound:
                out.append((o, a, b))
    out.sort()
    return [(a, b) for (_, a, b) in out]


# Monomial bases of L(130Pinf), L(186Pinf) = L((55 + 131)Pinf) and
# L(261Pinf) = L((130 + 131)Pinf); 131 is the pole order of q.
B130, B186, B261 = basis_upto(130), basis_upto(186), basis_upto(261)
assert (len(B130), len(B186), len(B261)) == (103, 159, 234)


def _monomial_series(Xs, Ys):
    """Return mono(a, b): the cached P_inf series of x^a y^b."""
    cache = {}

    def mono(a, b):
        if (a, b) not in cache:
            s = Ser.mono(0, 1, DEPTH)
            for _ in range(a):
                s = s * Xs
            for _ in range(b):
                s = s * Ys
            cache[(a, b)] = s
        return cache[(a, b)]

    return mono


def _f_map(Xs, mono, kn_sort):
    """F = eta / dx as F64-linear rows over the 123 known syndromes.

    Returns (FMAP, flo): FMAP[i] is the row of F's coefficient at u^(flo + i).
    """
    # dx/du: odd-exponent coefficients of x
    dc = np.zeros(len(Xs.c), np.uint8)
    for i in range(len(Xs.c)):
        if (Xs.lo + i) % 2 != 0:
            dc[i] = Xs.c[i]
    dxinv = Ser(Xs.lo - 1, dc).inv(DEPTH)

    # eta extraction table: eta[exp] as F64-linear rows over the 123
    # known syndromes, pinned triangularly by Res(phi_nu * eta) = S_nu.
    eta_lo = -56
    EMAP = np.zeros((DEPTH, 123), np.uint8)
    for nu in kn_sort:
        phi = mono(*FPORD[nu])
        acc = np.zeros(123, np.uint8)
        for k in range(1, len(phi.c)):
            pk = int(phi.c[k])
            if pk:
                i = (nu - 1 - k) - eta_lo
                if 0 <= i < DEPTH:
                    acc ^= MUL64[pk, EMAP[i]]
        acc[kn_sort.index(nu)] ^= 1
        EMAP[nu - 1 - eta_lo] = MUL64[ginv(int(phi.c[0])), acc]
    # F is eta / dx
    flo = eta_lo + dxinv.lo
    FMAP = np.zeros((DEPTH, 123), np.uint8)
    for i in range(DEPTH):
        for j in range(i + 1):
            dk = i - j
            if dk < len(dxinv.c):
                dcoef = int(dxinv.c[dk])
                if dcoef:
                    FMAP[i] ^= MUL64[dcoef, EMAP[j]]
    return FMAP, flo


def _omega_schedule(mono, FMAP, flo, omb_ords, ord2ab):
    """Build the omega MAC schedule, sparse and dense.

    Returns (vmap, vfull): vmap[(sl, N)] is the nonzero row V_{N,sl};
    vfull[sl, oi] is the row for basis order omb_ords[oi] (zero if absent).
    """
    # raw coefficient rows of mono_l * F at u^-N, peeled top-down
    # against the monomial basis (units with tails).
    vmap = {}
    for sl in range(48):
        msl = mono(*FPORD[O48[sl]])
        raw = {}
        for N in range(OMB_BOUND + 2):
            row = np.zeros(123, np.uint8)
            for k in range(len(msl.c)):
                mc = int(msl.c[k])
                if not mc:
                    continue
                i = (-N - (msl.lo + k)) - flo
                if 0 <= i < DEPTH:
                    row ^= MUL64[mc, FMAP[i]]
            raw[N] = row
        omrow = {}
        for M in sorted(omb_ords, reverse=True):
            acc = raw[M].copy()
            for Mp in sorted(omb_ords, reverse=True):
                if Mp <= M:
                    break
                tc = mono(*ord2ab[Mp]).coeff(-M)
                if tc:
                    acc ^= MUL64[tc, omrow[Mp]]
            omrow[M] = acc
        for N in omb_ords:
            if omrow[N].any():
                vmap[(sl, N)] = omrow[N]

    vfull = np.zeros((48, len(omb_ords), 123), np.uint8)
    for sl in range(48):
        for oi, N in enumerate(omb_ords):
            if (sl, N) in vmap:
                vfull[sl, oi] = vmap[(sl, N)]
    keyless = [oi for oi, N in enumerate(omb_ords) if N > 130]
    assert len(keyless) == 8 and not vfull[:, keyless].any()
    return vmap, vfull


def _fiber_divisor(Xs, Ys):
    """Build the fiber divisor q and locate its zeros.

    Returns (zq, qinv, qv): the 131 zero points of q, the P_inf series
    of 1/q, and q's value at every point.
    """
    xs = XPW[:, 1].copy()
    ys = YPW[:, 1].copy()
    cset = list(range(NXF))
    c9 = {int(gpow(c, 9)) for c in cset}
    dset = []
    for d in range(64):
        if len(dset) == NYF:
            break
        v = int(gpow(d, 8)) ^ d
        if v == 0 or v in c9:  # F8 fiber (1 point) or incidence
            continue
        dset.append(d)
    assert len(dset) == NYF
    for c in cset:
        assert int((xs == c).sum()) == 8
    for d in dset:
        assert int((ys == d).sum()) == 9
    zmask = np.zeros(NPTS, bool)
    for c in cset:
        zmask |= xs == c
    for d in dset:
        zmask |= ys == d
    zq = [int(pp) for pp in np.nonzero(zmask)[0]]
    assert len(zq) == 8 * NXF + 9 * NYF == 131

    qs = Ser.mono(0, 1, DEPTH)
    for c in cset:
        qs = qs * (Xs + Ser.mono(0, c, DEPTH))
    for d in dset:
        qs = qs * (Ys + Ser.mono(0, d, DEPTH))
    assert qs.lo == -131 and qs.c[0], "q must have exact pole order 131"
    qinv = qs.inv(DEPTH)
    qv = np.ones(NPTS, np.uint8)
    for c in cset:
        qv = MUL64[qv, xs ^ c]
    for d in dset:
        qv = MUL64[qv, ys ^ d]
    assert sorted(int(pp) for pp in np.nonzero(qv == 0)[0]) == zq
    return zq, qinv, qv


def _mvals(basis):
    """Values of each basis monomial at every point, one row per monomial."""
    return np.stack([MUL64[XPW[:, a], YPW[:, b]] for (a, b) in basis])


def _divisor_kernels(zq):
    """Split q's zeros into E and Z' and return the bases of L(55Pinf + E)
    and L(130Pinf + E).

    Returns (kg, kv, numg, numv): each kernel as coefficient rows over
    B186 / B261, and each kernel's numerator values at every point.
    """
    mv130, mv186, mv261 = _mvals(B130), _mvals(B186), _mvals(B261)
    rk = GFRank()
    ept, zp = [], []
    for pp in zq:
        if len(ept) < DEGE and rk.try_add(mv130[:, pp]):
            ept.append(pp)
        else:
            zp.append(pp)
    assert len(ept) == DEGE, "L(130Pinf-E)=0 unreachable on the fiber zeros"
    assert len(zp) == 28

    kg = gf64_kernel(mv186[:, zp].T.astype(np.uint8))
    kv = gf64_kernel(mv261[:, zp].T.astype(np.uint8))
    assert kg.shape == (131, 159)  # dim L(55Pinf + E)
    assert kv.shape == (NEV, 234)  # dim L(130Pinf + E)
    return kg, kv, gmm(kg, mv186), gmm(kv, mv261)


def _window_reader(mono, qinv, omb_ords, ord2ab):
    """Return wread(a, b): the cached, peeled window read of mono(a, b) / q."""
    ords_desc = sorted(omb_ords, reverse=True)
    tc_tab = {Mp: {M: mono(*ord2ab[Mp]).coeff(-M) for M in omb_ords if M < Mp} for Mp in omb_ords}

    def peel_scalar(raw):
        om = {}
        for M in ords_desc:
            acc = raw[M]
            for Mp in ords_desc:
                if Mp <= M:
                    break
                tc = tc_tab[Mp][M]
                if tc:
                    acc = acc ^ gmul(tc, om[Mp])
            om[M] = acc
        return om

    # window reads of mono(a,b)/q: raw coefficients at u^-N, peeled.
    wr_cache = {}

    def wread(a, b):
        key = (a, b)
        hit = wr_cache.get(key)
        if hit is not None:
            return hit
        T = mono(a, b) * qinv
        raw = {N: T.coeff(-N) for N in omb_ords}
        om = peel_scalar(raw)
        v = np.array([om[N] for N in omb_ords], np.uint8)
        wr_cache[key] = v
        return v

    return wread


def _solve_g(kg, wread, vfull):
    """Solve the window-match system for G(S), asserting existence and uniqueness.

    Returns gam (131 x 123): G's coordinates over kg, F64-linear in S.
    """
    nsw = vfull.shape[1]
    a_sys = np.zeros((48 * nsw, 131), np.uint8)
    for sl in range(48):
        a1, b1 = FPORD[O48[sl]]
        wrm = np.stack([wread(a1 + a2, b1 + b2) for (a2, b2) in B186])
        a_sys[sl * nsw : (sl + 1) * nsw] = gmm(kg, wrm).T
    aug = np.concatenate([a_sys, vfull.reshape(48 * nsw, 123)], axis=1)
    red, piv = gf64_rref(aug, 131)
    assert len(piv) == 131, "G not unique in L(55Pinf+E)"
    assert not red[131:].any(), "window class of F misses L(55Pinf+E)"
    return red[:131, 131:].copy()  # (131, 123)


def _eval_interp(kv, numg, numv, qv, gam, wread):
    """Choose 206 evaluation points, invertible on L(130Pinf + E).

    Returns (lam_eval, g_eval, interp): the two per-point gather masks
    and the window-read . eval^-1 interpolation matrix.
    """
    rk2 = GFRank()
    pts = []
    for pp in range(NPTS):
        if len(pts) == NEV:
            break
        if qv[pp] and rk2.try_add(numv[:, pp]):
            pts.append(pp)
    assert len(pts) == NEV, "evaluation points: rank stalled"
    pts_a = np.array(pts)
    qiv = POW62[qv[pts_a]]
    vmat = MUL64[numv[:, pts_a], qiv[None, :]].T.copy()
    vi = np.concatenate([vmat, np.eye(NEV, dtype=np.uint8)], axis=1)
    rv, pv = gf64_rref(vi, NEV)
    assert len(pv) == NEV
    vinv = rv[:, NEV:].copy()

    wrv = np.stack([wread(a, b) for (a, b) in B261])  # (234, 111)
    wv = gmm(kv, wrv).T  # (111, 206)
    interp = gmm(wv, vinv)  # (111, 206)

    lam_eval = np.zeros((NEV, 48), np.uint8)
    for sl in range(48):
        a, b = FPORD[O48[sl]]
        lam_eval[:, sl] = MUL64[XPW[pts_a, a], YPW[pts_a, b]]
    g_eval = gmm(MUL64[numg[:, pts_a], qiv[None, :]].T.copy(), gam)
    return lam_eval, g_eval, interp


def _check_identities(interp, lam_eval, g_eval, vfull, omb_ords):
    """Fail-closed identity gates on the evaluation-interpolation tables."""
    nsw = len(omb_ords)
    keyless = [oi for oi, N in enumerate(omb_ords) if N > 130]
    assert not interp[keyless].any(), "keyless rows must vanish"
    # exhaustive basis-pair bilinearity
    for sl in range(48):
        ra = MUL64[interp, lam_eval[:, sl][None, :]]
        assert np.array_equal(gmm(ra, g_eval), vfull[sl]), f"bilinear identity fails at slot {sl}"
    rk3 = GFRank()
    assert sum(rk3.try_add(interp[i]) for i in range(nsw)) == 103
    rk4 = GFRank()
    span = 0
    for sl in range(48):
        for k in range(123):
            if span < NEV and rk4.try_add(MUL64[lam_eval[:, sl], g_eval[:, k]]):
                span += 1
    assert span == NEV, "product span must fill L(130Pinf+E)"
    # seeded spot checks incl. the all-zero edges
    rng = np.random.default_rng(0x686F7069)
    ntr = 512
    lam_t = rng.integers(0, 64, (48, ntr)).astype(np.uint8)
    s_t = rng.integers(0, 64, (123, ntr)).astype(np.uint8)
    lam_t[:, 0] = 0
    s_t[:, 0] = 0
    lam_t[:, 1] = 0
    s_t[:, 2] = 0
    lam_t[47, ntr // 2 :] = 1
    old = np.zeros((nsw, ntr), np.uint8)
    for sl in range(48):
        old ^= MUL64[lam_t[sl][None, :], gmm(vfull[sl], s_t)]
    new = gmm(interp, MUL64[gmm(lam_eval, lam_t), gmm(g_eval, s_t)])
    assert np.array_equal(old, new), "spot-check mismatch"


def _hasse_row(a, b, k):
    """Level-k Hasse derivative of x^a y^b at every point (binomial pattern)."""
    out = np.zeros(NPTS, np.uint8)
    for i in range(k + 1):
        j = k - i
        if i <= a and j <= b and (i & a) == i and (j & b) == j:
            out ^= MUL64[XPW[:, a - i + 8 * j], YPW[:, b - j]]
    return out


def _hasse_masks(omb_ords, ord2ab):
    """Hasse gather masks (levels 0..MLEV) of the locator and omega bases.

    Binomial patterns from the affine local series x = x_P + t,
    y = y_P + x_P^8 t (mod t^9).
    """
    lamh = np.zeros((MLEV + 1, 48, NPTS), np.uint8)
    for lvl in range(MLEV + 1):
        for sl, o in enumerate(O48):
            lamh[lvl, sl] = _hasse_row(*FPORD[o], lvl)
    omh = np.zeros((MLEV, len(omb_ords), NPTS), np.uint8)
    for lvl in range(MLEV):
        for oi, o in enumerate(omb_ords):
            omh[lvl, oi] = _hasse_row(*ord2ab[o], lvl)
    return lamh, omh


def derive():
    Xs, Ys = build_xy(DEPTH)
    mono = _monomial_series(Xs, Ys)
    kn_sort = sorted(int(n) for n in KNORDS)
    omb = basis_upto(OMB_BOUND)
    omb_ords = sorted({8 * a + 9 * b for (a, b) in omb})
    ord2ab = {8 * a + 9 * b: (a, b) for (a, b) in omb}

    FMAP, flo = _f_map(Xs, mono, kn_sort)
    vmap, vfull = _omega_schedule(mono, FMAP, flo, omb_ords, ord2ab)
    keys = sorted(vmap.keys())

    zq, qinv, qv = _fiber_divisor(Xs, Ys)
    kg, kv, numg, numv = _divisor_kernels(zq)
    wread = _window_reader(mono, qinv, omb_ords, ord2ab)
    gam = _solve_g(kg, wread, vfull)
    lam_eval, g_eval, interp = _eval_interp(kv, numg, numv, qv, gam, wread)
    _check_identities(interp, lam_eval, g_eval, vfull, omb_ords)
    lamh, omh = _hasse_masks(omb_ords, ord2ab)

    return {
        "OMB_ORDS": np.array(omb_ords, np.int64),
        "VMAP_KEYS": np.array(keys, np.int64),
        "VMAP_ROWS": np.stack([vmap[k] for k in keys]),
        "LAMH_MASK": lamh,
        "OMH_MASK": omh,
        "KN_SORT": np.array(kn_sort, np.int64),
        "LAM_EVAL": lam_eval,
        "G_EVAL": g_eval,
        "INTERP": interp,
    }


def pack(tables):
    out = {}
    for k, v in tables.items():
        out[k] = {
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "b64": base64.b64encode(np.ascontiguousarray(v).tobytes()).decode(),
        }
    return json.dumps(out, indent=1, sort_keys=True) + "\n"


def generate():
    """Run the derivation and return the canonical constants.json text."""
    return pack(derive())


def main():
    new = generate()
    tgt = HERE / "constants.json"
    old = tgt.read_text() if tgt.exists() else None
    tgt.write_text(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
