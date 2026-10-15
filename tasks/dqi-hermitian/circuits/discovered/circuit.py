"""dqi-hermitian discovered circuit: the scheduled block-Lanczos locator
walk.  Lanes that hit a breakdown freeze through zero scalars until a
compile-time tiered service slot repairs them; a swap journal enables
the full Bennett mirror; a stripe-direct evaluation-interpolation tail
applies the phase.

Pipeline (the b.phase stages, in emission order; section headers
below carry the same numbers):

  1. init   -- tables, schedule, state file, y0 seed
  2. P1     -- 52 forward atomic slots (P1/sNN)
  3. mux    -- the fused kind sandwich, forward
  4. tail   -- stripe-direct EI tail, bit-collapsed tower-core sweep
  5. unmux  -- the kind sandwich reversed (same emitter, rev=True)
  6. xseam  -- the X-strip measurement seam (in the assembly)
  7. P2     -- 52 mirror slots (P2/sNN) + the P2/init teardown

This file holds the whole build.  It loads two artifacts: the shared
instance scaffold ../_instance64.py and the sweep tables in
constants.json (written by gen_constants.py).

RUN: uv run python circuit.py -- builds, asserts the pinned
metrics, and streams to $UNCOMPUTE_OPS_PATH (else ./ops.bin).
build() is the raw-build entry for the profilers (no emission).

README.md in this folder describes the construction.
"""

import base64
import gc
import json
import pathlib

import numpy as np

from uncompute.gadgets import GF
from uncompute.testing import load_module, task_dir

HERE = pathlib.Path(__file__).resolve().parent

# The recorded op list is large and acyclic: refcounting alone
# reclaims everything the Builder drops, so the GC stays off for the
# whole build.
gc.disable()

# ============================================================================
# S1. Classical scaffold: pinned instance, GF(64^2)
# ============================================================================
N, NSTEPS = 48, 48
_INST_PATH = task_dir(__file__) / "circuits" / "_instance64.py"
inst = load_module(_INST_PATH, "inst64_bm")

MUL64, POW62, SQT = inst.MUL64, inst.POW62, inst.SQT
TRIPLES = inst.TRIPLES
BSRC, O48, FPORD = inst.BSRC, inst.O48, inst.FPORD
XPW, YPW, H = inst.XPW, inst.YPW, inst.H
AB = [FPORD[o] for o in O48]

# GF(64^2) = GF(64)[t] / (t^2 + t + EXT_C); 8 is the smallest c for
# which t^2 + t + c is irreducible over the pinned GF(64).
EXT_C = 8

# ============================================================================
# Table bank + gadget emitters
# ============================================================================
PARB = inst.PARB
NPT, NT16 = 123, 16


def _curve_points():
    _, sel = inst.gf64_rref(H, 512)
    assert len(sel) == 123, f"curve-point greedy: rank {len(sel)} != 123"
    return sel


def gf_inv_mat(M):
    h = M.shape[0]
    R, piv = inst.gf64_rref(np.concatenate([M, np.eye(h, dtype=np.uint8)], 1), h)
    assert len(piv) == h, "singular matrix"
    return R[:, h:]


_SEL = _curve_points()
TMAP = gf_inv_mat(H[:, _SEL].copy().astype(np.uint8))
VMAP = np.zeros((NPT, 48), np.uint8)
for _t, _pt in enumerate(_SEL):
    for _j, (_a, _b) in enumerate(AB):
        VMAP[_t, _j] = MUL64[XPW[_pt, _a], YPW[_pt, _b]]
OMAP = VMAP.T.copy()


def _maskb(m):
    cols = MUL64[:, [1 << j for j in range(6)]]
    return ((PARB[cols & m].astype(np.int64)) << np.arange(6)).sum(1)


def _unpack6(rows, width):
    return ((rows[:, :, None] >> np.arange(6)) & 1).astype(np.uint8).reshape(rows.shape[0], width)


def build_scatter_tables():
    SB = np.zeros((NPT, NT16, 738), np.uint8)
    VB = np.zeros((NPT, NT16, 288), np.uint8)
    TG = np.zeros((NPT, NT16, 288), np.uint8)
    deltas = []
    prev_m = 0
    for m_, om_ in TRIPLES:
        deltas.append((m_ ^ prev_m, om_))
        prev_m = m_
    deltas.append((prev_m, None))
    for t, (dm, om_) in enumerate(deltas):
        mb = _maskb(dm)
        SB[:, t, :] = _unpack6(mb[TMAP], 738)
        VB[:, t, :] = _unpack6(mb[VMAP], 288)
        if om_ is not None:
            w = MUL64[OMAP, om_]
            TG[:, t, :] = _unpack6(w.T.astype(np.int64), 288)
    return SB, VB, TG


def diagmap():
    """DM[48][123]: B_mm = XOR of s[srcs] (qform coefficients)."""
    DM = np.zeros((N, NPT), np.uint8)
    for i in range(N):
        for src in BSRC[i][i]:
            DM[i, src] ^= 1
    return DM


def _adj_masks(k):
    out = {}
    for m_, _o in TRIPLES:
        mp = 0
        for i in range(6):
            v = 1 << i
            for _ in range(k):
                v = int(SQT[v])
            if bin(v & m_).count("1") & 1:
                mp |= 1 << i
        out[m_] = mp
    return out


ADJ1, ADJ2, ADJ3 = _adj_masks(1), _adj_masks(2), _adj_masks(3)


# ---------------- loop-form tables ----------------
def build_loop_tables(b, SB, VB, TG):
    LT = {}
    LT["SB"] = b.declare_table(SB.reshape(-1).tolist())
    LT["VB"] = b.declare_table(VB.reshape(-1).tolist())
    LT["TG"] = b.declare_table(TG.reshape(-1).tolist())
    LT["T15"] = b.declare_table([1] * 15 + [0])
    return LT


def build_mac_tables(b):
    global _MT
    TBA = np.zeros((15, 6), np.uint8)
    TBO = np.zeros((15, 6), np.uint8)
    TBO8 = np.zeros((15, 6), np.uint8)
    for t, (m_, om_) in enumerate(TRIPLES):
        for j in range(6):
            TBA[t, j] = (m_ >> j) & 1
            TBO[t, j] = (om_ >> j) & 1
            TBO8[t, j] = (int(MUL64[EXT_C, om_]) >> j) & 1
    _MT = {
        "TBA": b.declare_table(TBA.reshape(-1).tolist()),
        "TBO": b.declare_table(TBO.reshape(-1).tolist()),
        "TBO8": b.declare_table(TBO8.reshape(-1).tolist()),
    }
    return _MT


def build_sigb_tables(b):
    """sigma(B*x) scan tables: sigma(B*x) = sq(sum_m (Bx)_m) =
    sum_l sq(c_l(s)) * sq(x_l) with c_l(s) = XOR_m B[m][l] -- the
    column sums are s-LINEAR (each B[m][l] is an XOR of <= 2
    syndrome symbols), so the form is a standard 48x15 bilinear
    scan; both Frobenius squares fold into ADJ1 masks.  720 Tof per
    pass, no hosted B*x.
    """
    # per-column odd-multiplicity symbol sets
    syms = []
    for l in range(N):
        cnt: dict = {}
        for m in range(N):
            for src in BSRC[m][l]:
                cnt[src] = cnt.get(src, 0) ^ 1
        syms.append([s_ for s_, odd in cnt.items() if odd])
    SM3 = np.zeros((N, 15, 738), np.uint8)
    VM3 = np.zeros((N, 15, 288), np.uint8)
    for t, (m_, _om) in enumerate(TRIPLES):
        am = ADJ1[m_]
        for l in range(N):
            for s_ in syms[l]:
                for jq in range(6):
                    if (am >> jq) & 1:
                        SM3[l, t, 6 * s_ + jq] ^= 1
            for jq in range(6):
                if (am >> jq) & 1:
                    VM3[l, t, 6 * l + jq] = 1
    return {
        "SM": b.declare_table(SM3.reshape(-1).tolist()),
        "VM": b.declare_table(VM3.reshape(-1).tolist()),
    }


def build_qform_tables(b):
    """qform vscan tables: g = sum_m B_mm * (A_m)^2 over GF(64^2).
    Per-slice passes: slice-0 -> g0 via OM; slice-1 -> g1 via OM AND
    g0 via OM8 (ext square: (a0+a1 t)^2 = (a0^2 + 8 a1^2) + a1^2 t).
    The squaring folds into the V-side masks (ADJ1 adjoint: par(A^2&m)
    = par(A & ADJ1[m])).
    """
    DM = diagmap()  # [48 x 123] F2 selection
    SM = np.zeros((48, 15, 738), np.uint8)
    VM = np.zeros((48, 15, 288), np.uint8)
    OM = np.zeros((48, 15, 6), np.uint8)
    OM8 = np.zeros((48, 15, 6), np.uint8)
    for t, (m_, om_) in enumerate(TRIPLES):
        # s-side: B_mm = XOR of selected s-symbols (coefficients all 1)
        # -> par(B_mm & m_) = XOR over selected src of par(s_src & m_):
        # mask m_ replicated on each selected symbol's 6 wires
        rows = DM.astype(np.int64) * m_  # [48 x 123] masks
        SM[:, t, :] = _unpack6(rows, 738)
        # V-side: qform reads the SQUARED operand -- ADJ1 adjoint mask
        vmask = ADJ1[m_]
        for i in range(48):
            for j in range(6):
                if (vmask >> j) & 1:
                    VM[i, t, 6 * i + j] = 1
        OM[:, t, :] = (om_ >> np.arange(6)) & 1
        om8 = int(MUL64[EXT_C, om_])
        OM8[:, t, :] = (om8 >> np.arange(6)) & 1
    return {
        "SM": b.declare_table(SM.reshape(-1).tolist()),
        "VM": b.declare_table(VM.reshape(-1).tolist()),
        "OM": b.declare_table(OM.reshape(-1).tolist()),
        "OM8": b.declare_table(OM8.reshape(-1).tolist()),
    }


# ---------------- flat-form core (maskpair family) ----------------
def flat_base(regs):
    ids = [q for r in regs for q in r]
    assert ids == list(range(ids[0], ids[0] + len(ids))), "non-contiguous"
    return ids[0]


_MT = None  # MAC tables; set by build_mac_tables


# ---------------- loop-form emitters ----------------
def _matvec_core(b, s_base, v_base, scr, LT, middle):
    ua, vb = scr[0], scr[1]
    with b.repeat(NPT) as p, b.repeat(NT16) as t:
        with b.repeat(738) as pos, b.gated(LT["SB"], p * (NT16 * 738) + t * 738 + pos):
            b.cx(s_base + pos, ua)
        with b.repeat(288) as pos, b.gated(LT["VB"], p * (NT16 * 288) + t * 288 + pos):
            b.cx(v_base + pos, vb)
        middle(p, t, ua, vb)


def emit_matvec_proj_loop(b, s_base, v_base, out_base, scr, LT):
    def middle(p, t, ua, vb):
        with b.gated(LT["T15"], t):
            tq = b.and_compute(ua, vb)
        with b.repeat(288) as pos, b.gated(LT["TG"], p * (NT16 * 288) + t * 288 + pos):
            b.cx(tq, out_base + pos)
        with b.gated(LT["T15"], t):
            b.and_release(tq, ua, vb)

    _matvec_core(b, s_base, v_base, scr, LT, middle)


def emit_matvec_mbu_loop(b, s_base, v_base, out_base, scr, LT):
    ms = [b.hmr(out_base + j) for j in range(288)]

    def middle(p, t, ua, vb):
        e0 = p * (NT16 * 288) + t * 288
        for j in range(288):
            with b.gated(LT["TG"], e0 + j):
                b.cz(ua, vb, cond=ms[j])

    _matvec_core(b, s_base, v_base, scr, LT, middle)


def _qscan_core(b, T, s_base, v_base, scr, middle):
    """48x15 masked-mult scan core for the qform vscan (absolute masks:
    per-iteration ungathers).
    """
    ua, vb = scr[0], scr[1]
    with b.repeat(48) as i, b.repeat(15) as t:
        with b.repeat(738) as pos, b.gated(T["SM"], i * (15 * 738) + t * 738 + pos):
            b.cx(s_base + pos, ua)
        with b.repeat(288) as pos, b.gated(T["VM"], i * (15 * 288) + t * 288 + pos):
            b.cx(v_base + pos, vb)
        middle(i, t, ua, vb)
        with b.repeat(288) as pos, b.gated(T["VM"], i * (15 * 288) + t * 288 + pos):
            b.cx(v_base + pos, vb)
        with b.repeat(738) as pos, b.gated(T["SM"], i * (15 * 738) + t * 738 + pos):
            b.cx(s_base + pos, ua)


def emit_qscan_loop(b, T, s_base, v_base, g12, scr, sl):
    """g ^= sum_m B_mm * (V_m)^2 for ONE slice of the 576q strip:
    slice 0 -> scatter OM into g0; slice 1 -> OM into g1 AND OM8 into
    g0 (ext square).  720 Tof per slice.
    """

    def middle(i, t, ua, vb):
        tq = b.and_compute(ua, vb)
        with b.repeat(6) as j, b.gated(T["OM"], i * (15 * 6) + t * 6 + j):
            b.cx(tq, g12[6 + j] if sl else g12[j])
        if sl:
            with b.repeat(6) as j, b.gated(T["OM8"], i * (15 * 6) + t * 6 + j):
                b.cx(tq, g12[j])
        b.and_release(tq, ua, vb)

    _qscan_core(b, T, s_base, v_base + sl * 288, scr, middle)


# ---------------- flag/scalar gadgets ----------------
def _and_tree(b, wires, mid):
    """Build a balanced AND tree over wires; mid(top) runs at the apex.

    (local: gadgets.arith.mcx is a linear ladder without the mid
    callback -- a swap would change the stream.)
    """
    lvl = list(wires)
    stack = []
    while len(lvl) > 1:
        nxt = []
        for k in range(0, len(lvl) - 1, 2):
            t = b.and_compute(lvl[k], lvl[k + 1])
            stack.append((t, lvl[k], lvl[k + 1]))
            nxt.append(t)
        if len(lvl) & 1:
            nxt.append(lvl[-1])
        lvl = nxt
    mid(lvl[0])
    for t, c1, c2 in reversed(stack):
        b.and_release(t, c1, c2)


def emit_ztest(b, W, flag, scr):
    """flag ^= [W == 0] (len(W)-1 ANDs)."""
    n = len(W)
    for j in range(n):
        b.cx(W[j], scr[j])
        b.x(scr[j])
    _and_tree(b, [scr[j] for j in range(n)], lambda top: b.cx(top, flag))
    for j in range(n):
        b.x(scr[j])
        b.cx(W[j], scr[j])


def _maskpair_core(b, scr, Aq, Am, Bq, Bm, middle):
    """Shared masked-product scaffold: per-triple interleaved gather of
    (Aq masked by Am, Bq masked by Bm) onto scr -> middle(om_, ua, vb)
    -> ungather.
    """
    ua, vb = scr[0], scr[1]
    for m_, om_ in TRIPLES:
        ma = Am[m_] if Am is not None else m_
        mb = Bm[m_] if Bm is not None else m_
        for i in range(6):
            if (ma >> i) & 1:
                b.cx(Aq[i], ua)
            if (mb >> i) & 1:
                b.cx(Bq[i], vb)
        middle(om_, ua, vb)
        for i in range(6):
            if (ma >> i) & 1:
                b.cx(Aq[i], ua)
            if (mb >> i) & 1:
                b.cx(Bq[i], vb)


def emit_inv64(b, A6, OUT6, t9, t6, scr):
    """OUT ^= A^-1 over GF(64) (45 Tof; INV(0)=0)."""
    emit_mac6(b, A6, A6, t9, scr, Bm=ADJ3)
    emit_mac6(b, t9, t9, t6, scr, Am=ADJ1, Bm=ADJ2)
    emit_mac6(b, t6, A6, OUT6, scr, Bm=ADJ3)


def emit_inv64_unwind(b, A6, t9, t6, scr):
    """MBU-clear t6 then t9 (reverse of emit_inv64's scratch builds)."""
    emit_mac6_mbu(b, t9, t9, t6, scr, Am=ADJ1, Bm=ADJ2)
    emit_mac6_mbu(b, A6, A6, t9, scr, Bm=ADJ3)


# ---------------- register-vs-constant compare ----------------
def _const_eq_gather(b, WJ, i):
    """X-conjugate WJ to compare against constant i (self-inverse;
    any register width).
    """
    for j in range(len(WJ)):
        if not ((i >> j) & 1):
            b.x(WJ[j])


# ============================================================================
# The sweep table bank + fixed-mask walk core
# ============================================================================
# Table conventions:
#   * AFIX(sl) -> (lam, 6*sl): lam is one contiguous 288q register,
#     slot 47 a real variable -- no af47 split, no monic X (the sweep
#     is scale-invariant, so raw unmonicized lambdas pay the correct
#     parity).
#   * s is the 738q flat syndrome register.
# Constants: constants.json (written by gen_constants.py from
# _instance64);
# kill-set/dual/pay tables re-derived identically.


def _load_sweep_tables():
    raw = json.loads((HERE / "constants.json").read_bytes())
    out = {}
    for k, d in raw.items():
        out[k] = np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"])
    return out


_TBL = _load_sweep_tables()
SW_ORDS = [int(x) for x in _TBL["OMB_ORDS"]]
NSW = len(SW_ORDS)
_SW_ORD_IDX = {o: i for i, o in enumerate(SW_ORDS)}
SW_KN = [int(x) for x in _TBL["KN_SORT"]]
SW_LAMH = _TBL["LAMH_MASK"]  # (4, 48, 512) F64 gather values
SW_OMH = _TBL["OMH_MASK"]  # (3, NSW, 512)
KIDX = inst.KIDX
SW_MACS = []  # (slot, om-index, syn-kidx, c)
for _k, _row in zip(_TBL["VMAP_KEYS"], _TBL["VMAP_ROWS"]):
    _sl = int(_k[0])
    _N2 = int(_k[1])
    for _i in np.nonzero(_row)[0]:
        SW_MACS.append((_sl, _SW_ORD_IDX[_N2], int(KIDX[FPORD[SW_KN[int(_i)]]]), int(_row[_i])))
SW_MU = [int(x) for x in inst.gamma]
TR = inst.TR

_H64 = np.array(H, np.uint8)
SW_KILL = _SEL
_Asl = _H64[:, SW_KILL].T.copy()
_rhs = np.array([SW_MU[p] for p in SW_KILL], np.uint8)
_Aug = np.concatenate([_Asl, _rhs[:, None]], 1).astype(np.uint8)
_Aug, _piv = inst.gf64_rref(_Aug, 123)
assert len(_piv) == 123, "kill-set dual singular"
SW_UDUAL = _Aug[:, -1].copy()
_MU_T = np.zeros(512, np.uint8)
for _r in range(123):
    _u = int(SW_UDUAL[_r])
    if _u:
        _MU_T ^= MUL64[_u, _H64[_r]]
SW_MU_RES = [int(_MU_T[p]) ^ SW_MU[p] for p in range(512)]
for _p in SW_KILL:
    assert SW_MU_RES[_p] == 0, "kill-set residual must vanish"
SW_SYNZ = [(r, j) for r in range(123) for j in range(6) if int(TR[MUL64[int(SW_UDUAL[r]), 1 << j]])]
SW_PAYT = []
for _p in range(512):
    _mu = SW_MU_RES[_p]
    _trm = 0
    for j in range(6):
        if int(TR[MUL64[_mu, 1 << j]]):
            _trm |= 1 << j
    # triple m contributes to Tr(mu~_p * prod) iff parity(trm & om_m)
    # (om = TRIPLES[m][1])
    SW_PAYT.append(tuple(mi for mi in range(15) if bin(_trm & TRIPLES[mi][1]).count("1") & 1))
SW_SKIP = [SW_MU_RES[p] == 0 for p in range(512)]

# ---- fixed-mask walk core (_walk_g family) ----
I6M = [[1 if i3 == j3 else 0 for j3 in range(6)] for i3 in range(6)]

_MT_CACHE: dict = {}


def maskT(M, mask):
    """Return the mask m with parity((Mx)&mask) == parity(x & m); memoized."""
    key = (id(M), mask)
    hit = _MT_CACHE.get(key)
    if hit is not None and hit[0] is M:
        return hit[1]
    nm = 0
    for j in range(6):
        v = 0
        for i3 in range(6):
            if (mask >> i3) & 1:
                v ^= M[i3][j]
        nm |= (v & 1) << j
    _MT_CACHE[key] = (M, nm)
    return nm


def wref(ref, j):
    wl, base = ref
    return wl[base + j]


def _walk_g(b, s1, M1, s2, M2, mid):
    """15-triple delta-mask walk core."""
    ta = b.qubit()
    tb = b.qubit()
    p1 = 0
    p2 = 0
    for mi, (ma, om) in enumerate(TRIPLES):
        n1 = maskT(M1, ma)
        n2 = maskT(M2, ma)
        d1 = p1 ^ n1
        d2 = p2 ^ n2
        for j in range(6):
            if (d1 >> j) & 1:
                b.cx(wref(s1, j), ta)
            if (d2 >> j) & 1:
                b.cx(wref(s2, j), tb)
        mid(mi, om, ta, tb)
        p1, p2 = n1, n2
    for j in range(6):
        if (p1 >> j) & 1:
            b.cx(wref(s1, j), ta)
        if (p2 >> j) & 1:
            b.cx(wref(s2, j), tb)
    b.recycle(tb)
    b.recycle(ta)


def emit_mac_g(b, s1, M1=I6M, s2=None, M2=I6M, macc=None):
    """acc ^= (M1 s1)*(M2 s2), maps folded free. 15 Tof."""

    def mid(_mi, om, ta, tb):
        tt = b.and_compute(ta, tb)
        for j in range(6):
            if (om >> j) & 1:
                b.cx(tt, wref(macc, j))
        b.and_release(tt, ta, tb)

    _walk_g(b, s1, M1, s2, M2, mid)


# ---------------- evaluation-interpolation tables ----------------
_EI: dict = {"loaded": False}


def _ei_tables():
    """Build the evaluation-interpolation tables for the omega bank.

    The tables come from the constants bank, re-based to this file's
    register conventions:

      LEVT[t, sl] = LAM_EVAL           (206 x 48; lam slot = O48 index)
      GEVT[t, kx] = G_EVAL, columns permuted KN_SORT -> register
                    order (kx = KIDX[FPORD[N]])            (206 x 123)
      LMAP[oi, t] = INTERP             (111 x 206; oi = OMB_ORDS index)

    Fail-closed: the EI fold LMAP o (eval x eval) must reproduce the
    SW_MACS bank tensor on every (oi, sl, kx) entry (bilinearity
    closes the argument per slot); any mismatch raises.  Cached.
    """
    if _EI["loaded"]:
        return _EI
    LEVT = np.ascontiguousarray(_TBL["LAM_EVAL"], np.uint8)
    LMAP = np.ascontiguousarray(_TBL["INTERP"], np.uint8)
    GEVT = np.zeros((LEVT.shape[0], 123), np.uint8)
    for i, Npt in enumerate(_TBL["KN_SORT"]):
        GEVT[:, int(KIDX[FPORD[int(Npt)]])] = _TBL["G_EVAL"][:, i]
    assert LEVT.shape == (206, 48) and GEVT.shape == (206, 123)
    assert LMAP.shape == (NSW, 206)
    T = np.zeros((NSW, 48, 123), np.uint8)
    for sl, oi, kx, c in SW_MACS:
        T[oi, sl, kx] ^= c
    for sl in range(48):
        ra = MUL64[LMAP, LEVT[:, sl][None, :]]  # (NSW, 206)
        out = np.zeros((NSW, 123), np.uint8)
        for t in range(206):
            nz = ra[:, t] != 0
            if nz.any():
                out[nz] ^= MUL64[ra[nz, t][:, None], GEVT[t][None, :]]
        assert np.array_equal(out, T[:, sl, :]), f"EI fold != SW_MACS tensor at slot {sl}"
    _EI.update(loaded=True, LEVT=LEVT, GEVT=GEVT, LMAP=LMAP)
    return _EI


# GF(64) word gadgets over the pinned poly (mw_xor: dst ^= flag*src;
# w_pow2_xor(.., 1): dst ^= sq(src), the free Frobenius CX gather)
_G8 = GF(8)
assert _G8.poly == inst.MODP, "gadget field != instance field"

# The walk seed y0: column 0 of a seeded (N, 2) draw, with y0[0] made
# odd.  sigma(y0) != 0 on every lane, so no lane can trigger at slot 0
# (the slot-0 service elision and the first-block kappa seed both
# rest on this).
_y = np.random.default_rng(20260809).integers(0, 64, (N, 2))
Y0 = np.ascontiguousarray(_y[:, 0].astype(np.uint8))
Y0[0] |= 1
assert int(np.bitwise_xor.reduce(Y0)) != 0, "sigma(y0) == 0"


# ---- word gadgets: GF(64) word MACs, taps, scans, cswap -------------------
def emit_mac6(b, A6, B6, R6, scr, Am=None, Bm=None):
    """R6 ^= A6 * B6 over GF(64) (15 Tof, Toom triples). Am/Bm remap
    the operand gather masks (see _maskpair_core).
    """

    def middle(om_, ua, vb):
        t = b.and_compute(ua, vb)
        for j in range(6):
            if (om_ >> j) & 1:
                b.cx(t, R6[j])
        b.and_release(t, ua, vb)

    _maskpair_core(b, scr, A6, Am, B6, Bm, middle)


def emit_mac6_mbu(b, A6, B6, R6, scr, Am=None, Bm=None):
    """MBU-clear of R6 holding exactly the emit_mac6 product, operands
    live (0 Tof).
    """
    ms = [b.hmr(R6[j]) for j in range(6)]

    def middle(om_, ua, vb):
        for j in range(6):
            if (om_ >> j) & 1:
                b.cz(ua, vb, cond=ms[j])

    _maskpair_core(b, scr, A6, Am, B6, Bm, middle)
    for m_ in ms:
        b.retire(m_)


def emit_macv48(b, MT, coef6, v_base, dst_base, scr):
    """dst[m] ^= coef * v[m] over 48 GF(64) symbols (48 x 15 = 720
    Tof); loop form on the shared TBA/TBO triple tables.
    """
    ua, vb = scr[0], scr[1]
    with b.repeat(48) as m, b.repeat(15) as t:
        with b.repeat(6) as j:
            with b.gated(MT["TBA"], t * 6 + j):
                b.cx(coef6[j], ua)
            with b.gated(MT["TBA"], t * 6 + j):
                b.cx(v_base + m * 6 + j, vb)
        tq = b.and_compute(ua, vb)
        with b.repeat(6) as j, b.gated(MT["TBO"], t * 6 + j):
            b.cx(tq, dst_base + m * 6 + j)
        b.and_release(tq, ua, vb)
        with b.repeat(6) as j:
            with b.gated(MT["TBA"], t * 6 + j):
                b.cx(v_base + m * 6 + j, vb)
            with b.gated(MT["TBA"], t * 6 + j):
                b.cx(coef6[j], ua)


def emit_mac6_gated(b, g1, A6, B6, R6, scr):
    """R6 ^= A6 * B6, additionally conditioned on flag g1 (30 Tof:
    each Toom AND extended by the flag via a second and_compute).
    """

    def middle(om_, ua, vb):
        t = b.and_compute(ua, vb)
        t2 = b.and_compute(t, g1)
        for j in range(6):
            if (om_ >> j) & 1:
                b.cx(t2, R6[j])
        b.and_release(t2, t, g1)
        b.and_release(t, ua, vb)

    _maskpair_core(b, scr, A6, None, B6, None, middle)


def emit_mac6_gated_mbu(b, g1, A6, B6, R6, scr):
    """MBU-clear of R6 holding exactly g1*(A6*B6) per lane, with A6/
    B6/g1 live (15 Tof: one Toom AND per triple; the flag rides the
    outcome-conditioned CZ, so lanes with g1 = 0 contribute no
    phase).
    """
    ms = [b.hmr(R6[j]) for j in range(6)]

    def middle(om_, ua, vb):
        t = b.and_compute(ua, vb)
        for j in range(6):
            if (om_ >> j) & 1:
                b.cz(t, g1, cond=ms[j])
        b.and_release(t, ua, vb)

    _maskpair_core(b, scr, A6, None, B6, None, middle)
    for m_ in ms:
        b.retire(m_)


def emit_macv48_cz(b, coef6, v_base, ms, scr):
    """Emit the mirror-side conjugate of emit_macv48 into the measured X
    strip: the same Toom gather mesh, but each dst write becomes an
    outcome-conditioned CZ vs the seam X-hmr outcome for that strip
    position (0 Tof, unscored).  ms: 288 Outcomes in strip order.
    Flat emission (outcome ids vary per iteration).
    """
    ua, vb = scr[0], scr[1]
    for m in range(48):
        for m_, om_ in TRIPLES:
            for j in range(6):
                if (m_ >> j) & 1:
                    b.cx(coef6[j], ua)
                    b.cx(v_base + m * 6 + j, vb)
            for j in range(6):
                if (om_ >> j) & 1:
                    b.cz(ua, vb, cond=ms[m * 6 + j])
            for j in range(6):
                if (m_ >> j) & 1:
                    b.cx(v_base + m * 6 + j, vb)
                    b.cx(coef6[j], ua)


def emit_sigma_tap(b, v_base, t6):
    """t6 ^= sq(sum_m v[m]) (free CX): the sigma tap is the SQUARED
    sum (the mu-moment form); Frobenius linearity folds the square
    into the gather wiring (_G8.w_pow2_xor per summand).
    """
    with b.repeat(48) as m:
        _G8.w_pow2_xor(b, [v_base + m * 6 + j for j in range(6)], t6, 1)


def _cswap288(b, flag, a_base, b_base):
    """cswap the two 288q strips at a_base/b_base under flag
    (288 Toffoli; loop form).
    """
    with b.repeat(288) as pos:
        b.cswap(flag, a_base + pos, b_base + pos)


def _qscan1(b, T, s_base, v_base, out6, scr):
    """out6 ^= sum_m B_mm(s) * v_m^2 -- the qscan slice-0 pass
    (720 Tof); the slice-0 scatter only touches out6[0:6].
    """
    emit_qscan_loop(b, T["QT"], s_base, v_base, out6, scr, 0)


def _qscan1_mbu(b, T, s_base, v_base, out6, scr):
    """MBU-clear of out6 holding exactly qform(v): hmr 6 + the
    slice-0 mesh walk (0 Tof).
    """
    ms = [b.hmr(out6[j]) for j in range(6)]

    def mid0(i, t, ua, vb):
        for j in range(6):
            with b.gated(T["QT"]["OM"], i * (15 * 6) + t * 6 + j):
                b.cz(ua, vb, cond=ms[j])

    _qscan_core(b, T["QT"], s_base, v_base, scr, mid0)
    for m_ in ms:
        b.retire(m_)


# ---- 1. init: tables, schedule, state file, y0 seed -----------------------
def build_tables(b):
    SB, VB, TG = build_scatter_tables()
    LT = build_loop_tables(b, SB, VB, TG)
    # the kind-test sketch compressor: K = R.(B*v) via the folded
    # 48-bit spray table TGR = R.TG -- the loop-form matvec bills per
    # (p, t) term, so the compressed scatter costs the same 1,845 Tof
    # as the full 288-bit matvec.  R is a fixed pseudorandom GF(2)
    # 48x288 matrix (deterministic seed); the test is one-sided:
    # B*v = 0 => K = 0 exactly.
    R = np.random.default_rng(20260810).integers(0, 2, (48, 288), dtype=np.uint8)
    TGR = (TG.astype(np.int64) @ R.T.astype(np.int64)) & 1
    LT["TGR"] = b.declare_table(TGR.astype(np.uint8).reshape(-1).tolist())
    QT = build_qform_tables(b)
    BT = build_sigb_tables(b)
    MT = build_mac_tables(b)
    return {"LT": LT, "QT": QT, "BT": BT, "MT": MT}


def _y0_init(b, st):
    """WA ^= y0 (free X mask) + Cu = 1: the walk's initial state."""
    for m in range(N):
        v = int(Y0[m])
        for q in range(6):
            if (v >> q) & 1:
                b.x(st["WA"][6 * m + q])
    b.x(st["Cu"][0])


def _y0_teardown(b, st):
    """Exact reverse of _y0_init (Cu first)."""
    b.x(st["Cu"][0])
    for m in range(N):
        v = int(Y0[m])
        for q in range(6):
            if (v >> q) & 1:
                b.x(st["WA"][6 * m + q])


K, V3K, MEXT = 3, 6, 4
NSLOTS = NSTEPS + MEXT
# Slot 0 carries no service machinery: sigma(y0) != 0 on every lane
# (asserted by the emitter layer), so no lane can trigger -- let alone
# be serviced -- at slot 0.  The schedule is unchanged (a slot-0
# service could never fire); only the dead machinery is elided.
SERVICE = frozenset(
    {s for s in range(NSTEPS) if s % K == 0 and s} | {NSTEPS + i for i in range(MEXT) if i % 2 == 0}
)
V3CAP = frozenset(
    {s for s in range(NSTEPS) if s % V3K == 0 and s}
    | {NSTEPS + i for i in range(MEXT) if i % 2 == 0}
)
assert V3CAP <= SERVICE


# Episodic-register taxonomy. A register is parked (its wires provably
# |0>, its ids back on the free-list) outside its working window and
# re-materialized at the window open; the resident core alone survives
# to the end of the circuit. Members whose window close is a
# measurement demolition park with their reset already recorded (H);
# the rest park through recycle()'s explicit reset (R).
_W_BASE_H = ("QF", "alpha", "Ca")  # every slot; HMR-cleared
_W_BASE_R = ("SG", "DLT", "t6a", "t6b", "t6c", "Z6", "XC")
_F_BASE = ("trig", "capf", "pj", "zq", "zd")
_W_SVC_H = ("QU",)  # service slots
_W_SVC_R = ("BA0", "BA1", "BKG", "BFC", "XH")
_F_SVC = ("zqf",)
_W_V3_H = ("QV",)  # v3-capable slots
_W_V3_R = ("SV", "BA2")
_F_V3 = ("zdd",)
_K48_W = ("SG", "QF", "DLT", "QU", "QV", "SV", "alpha", "Ca")
# One tier per schedule level, (HMR-words, R-words, flags): _slot_open
# walks the applicable prefix forward, _slot_close walks it reversed,
# and the initial parks flatten it -- one authority for the cascade.
_TIERS = (
    (_W_BASE_H, _W_BASE_R, ("scr", "zt") + _F_BASE),
    (_W_SVC_H, _W_SVC_R, _F_SVC),
    (_W_V3_H, _W_V3_R, _F_V3),
)

# The walk-state word file (22 six-qubit scalars) and the ten 1q
# flags: the resident trio, the seam-hosting K48 octet, and the
# slot-local words, concatenated in allocation order; also the
# membership lists of the "scal" and "flag" wire-sets.
_W_CORE = ("Cu", "KP", "KM")
_W_AUX = ("t6a", "t6b", "t6c", "Z6", "XC", "BA0", "BA1", "BA2", "BKG", "BFC", "XH")
_SCAL = _W_CORE + _K48_W + _W_AUX
_FLAGS = ("trig", "capf", "exh", "pj", "zq", "zd", "zqf", "zdd", "capk1", "capk2")
# Partition tripwires: every episodic word sits in the tier table
# exactly, and the flag universe is the resident trio plus the tier
# flags (scr/zt are separate registers claimed alongside them).
assert set(_K48_W + _W_AUX) == {n for t in _TIERS for n in t[0] + t[1]}
assert set(_FLAGS) == (
    ({"exh", "capk1", "capk2"} | {n for t in _TIERS for n in t[2]}) - {"scr", "zt"}
)


def alloc_state(b):
    """Walk-state allocation in residency order: the syndrome register
    (the io floor) first, then the resident walk core -- the wires live
    from init to teardown -- and every episodically-held register above
    it, so parked spans read as reusable ids, never as resident file.
    """
    st = {}
    sregs = b.qubits(738)
    for r in range(123):
        b.qubit_register(sregs[6 * r : 6 * r + 6])
    st["s"] = sregs
    # resident core (live init..teardown): the vector swap pair, the
    # recurrence scalars, the freeze record, and the survival flags
    for nm in ("WA", "WB"):
        st[nm] = b.qubits(288)
    for nm in _W_CORE:
        st[nm] = b.qubits(6)
    st["FZT"] = b.qubits(6)
    for nm in ("exh", "capk1", "capk2"):
        st[nm] = b.qubit()
    # episodic registers, in window order: the x-carrier (init..seam),
    # the swap journal (bit j: slot j..mirror slot j), the per-slot
    # word file and flags, and the U/V2 strips
    st["X"] = b.qubits(288)
    st["SWJ"] = b.qubits(NSLOTS)
    for nm in _K48_W + _W_AUX:
        st[nm] = b.qubits(6)
    st["scr"] = b.qubits(2)
    st["zt"] = b.qubits(6)
    for nm in _F_BASE + _F_SVC + _F_V3:
        st[nm] = b.qubit()
    for nm in ("U", "V2"):
        st[nm] = b.qubits(288)
    st["fix"] = []
    # wire-set claims (metadata; emit nothing): the ownership names the
    # meta.json viz roles anchor to.  Mid-build frees of claimed wires
    # (the _inv_p park, the sandwich hosts, the episodic parks) release
    # first and re-claim after re-allocation.
    b.claim("s", sregs)
    for nm in ("WA", "WB", "U", "V2", "X"):
        b.claim(nm, st[nm])
    b.claim("scal", [q for nm in _SCAL for q in st[nm]])
    b.claim("jrn", list(st["FZT"]) + list(st["SWJ"]))
    b.claim("flag", list(st["scr"]) + list(st["zt"]) + [st[nm] for nm in _FLAGS])
    # contiguous 48-id block (SG..Ca) that is |0> at the seam: hosts
    # the sandwich's 48-bit sketch register K
    st["K48"] = [q for nm in _K48_W for q in st[nm]]
    assert st["K48"] == list(range(st["K48"][0], st["K48"][0] + 48))
    # initial parks: every episodic register is |0> before slot 0
    # (freshly allocated wires are |0>, so the parks emit nothing)
    _park(b, st, ("V2", "U"))
    _park(b, st, tuple(n for t in _TIERS for n in t[0] + t[1]), owner="scal")
    _park(b, st, tuple(n for t in _TIERS for n in t[2]), owner="flag")
    _park(b, st, ("SWJ",), owner="jrn")
    return st


def _reg_ids(st, nm):
    v = st[nm]
    return [v] if isinstance(v, int) else list(v)


def _park(b, st, names, owner=None, recorded=True):
    """Park registers whose wires are |0>: release ownership (under
    ``owner`` when the wires belong to a shared claim set), return the
    ids to the free-list. ``recorded=True`` asserts each wire's reset is
    already in the stream (fresh allocation, or the HMR demolition that
    cleared it) and emits nothing; ``recorded=False`` resets explicitly
    through recycle()'s R.
    """
    for nm in names:
        ids = _reg_ids(st, nm)
        b.release(owner or nm, ids)
        b.recycle_all(list(reversed(ids)), reset_recorded=recorded)


def _unpark(b, st, names, owner=None):
    """Re-materialize parked registers: the exact ids back, ownership
    restored. Emits nothing.
    """
    for nm in names:
        ids = _reg_ids(st, nm)
        b.reclaim(ids)
        b.claim(owner or nm, ids)


def _slot_tiers(service, v3cap):
    return _TIERS[: 1 + bool(service) + bool(service and v3cap)]


def _slot_open(b, st, j, service, v3cap, journal=True):
    """Re-materialize the slot's working set. ``journal=True`` also opens
    journal bit j's window, which closes at the mirrored slot (the
    mirror's own open finds the bit already live).
    """
    _unpark(b, st, ("U",))
    if journal:
        b.reclaim([st["SWJ"][j]])
        b.claim("jrn", [st["SWJ"][j]])
    for wh, wr, fl in _slot_tiers(service, v3cap):
        _unpark(b, st, wh + wr, owner="scal")
        _unpark(b, st, fl, owner="flag")


def _slot_close(b, st, j, service, v3cap, journal=False):
    """Park the slot's working set, tier by tier in reverse (every
    member |0> here: HMR-cleared words with their reset recorded, the
    rest through recycle()'s explicit reset). ``journal=True`` -- the
    mirror pass -- also parks journal bit j, zeroed at its underive.
    """
    for wh, wr, fl in reversed(_slot_tiers(service, v3cap)):
        _park(b, st, wh, owner="scal")
        _park(b, st, wr, owner="scal", recorded=False)
        _park(b, st, fl, owner="flag", recorded=False)
    _park(b, st, ("U",))
    if journal:
        b.release_recycle("jrn", [st["SWJ"][j]])


# ---- 2. P1: the forward atomic slot ---------------------------------------
def _inv_p(b, st, src, dst, scr):
    park_ids = list(st["alpha"]) + list(st["Ca"])
    b.release("scal", park_ids)
    b.recycle_all(park_ids)
    t9 = b.qubits(6)
    t6 = b.qubits(6)
    emit_inv64(b, src, dst, t9, t6, scr)
    emit_inv64_unwind(b, src, t9, t6, scr)
    b.recycle_all(t6)
    b.recycle_all(t9)
    b.reclaim(park_ids)
    b.claim("scal", park_ids)


def _fzt_cmp(b, st, j):
    _const_eq_gather(b, st["FZT"], j + 1)
    _and_tree(b, list(st["FZT"]), lambda top: b.cx(top, st["pj"]))
    _const_eq_gather(b, st["FZT"], j + 1)


def _capf_tree(b, st, zt):
    zq, zd = st["zq"], st["zd"]
    b.x(st["exh"])
    emit_ztest(b, list(st["QF"]), zq, zt)
    emit_ztest(b, list(st["DLT"]), zd, zt)
    _and_tree(b, [st["trig"], st["exh"], zq, zd], lambda top: b.cx(top, st["capf"]))
    emit_ztest(b, list(st["DLT"]), zd, zt)
    emit_ztest(b, list(st["QF"]), zq, zt)
    b.x(st["exh"])


def _freeze_fwd(b, st, j):
    b.cx(st["capf"], st["pj"])
    for q in range(6):
        if ((j + 1) >> q) & 1:
            b.cx(st["pj"], st["FZT"][q])
    b.cx(st["pj"], st["exh"])
    _fzt_cmp(b, st, j)


def _freeze_rev(b, st, j):
    _fzt_cmp(b, st, j)
    b.cx(st["pj"], st["exh"])
    for q in range(6):
        if ((j + 1) >> q) & 1:
            b.cx(st["pj"], st["FZT"][q])
    b.cx(st["capf"], st["pj"])


def _capf_conj(b, st, j, zt):
    _fzt_cmp(b, st, j)
    b.cx(st["pj"], st["exh"])
    _capf_tree(b, st, zt)
    b.cx(st["pj"], st["exh"])
    _fzt_cmp(b, st, j)


def _coeffs(b, st, d2a, d3a, rev=False):
    """Block close coefficients BA0/BA1/BA2 (scale-free)."""
    scr = st["scr"]
    giv = st["t6c"]
    ops = [
        lambda: _G8.w_pow2_xor(b, giv, st["t6b"], 1),
        lambda: emit_mac6_gated(b, d2a, st["DLT"], giv, st["BA1"], scr),
        lambda: _G8.w_pow2_xor(b, st["DLT"], st["t6a"], 1),
        lambda: emit_mac6(b, st["SG"], st["QU"], st["t6a"], scr),
        lambda: emit_mac6_gated(b, d2a, st["t6a"], st["t6b"], st["BA0"], scr),
        lambda: emit_mac6(b, st["SG"], st["QU"], st["t6a"], scr),
        lambda: _G8.w_pow2_xor(b, st["DLT"], st["t6a"], 1),
    ]
    if d3a is not None:
        ops += [
            lambda: emit_mac6_gated(b, d3a, st["QU"], giv, st["BA2"], scr),
            lambda: _G8.w_pow2_xor(b, st["QU"], st["t6a"], 1),
            lambda: emit_mac6(b, st["SG"], st["SV"], st["t6a"], scr),
            lambda: emit_mac6_gated(b, d3a, st["t6a"], st["t6b"], st["BA1"], scr),
            lambda: emit_mac6(b, st["SG"], st["SV"], st["t6a"], scr),
            lambda: _G8.w_pow2_xor(b, st["QU"], st["t6a"], 1),
            lambda: _G8.w_pow2_xor(b, st["QU"], st["XC"], 1),
            lambda: emit_mac6(b, st["QU"], st["XC"], st["t6a"], scr),
            lambda: _G8.w_pow2_xor(b, st["QU"], st["XC"], 1),
            lambda: _G8.w_pow2_xor(b, st["SG"], st["XC"], 1),
            lambda: emit_mac6(b, st["XC"], st["QV"], st["t6a"], scr),
            lambda: _G8.w_pow2_xor(b, st["SG"], st["XC"], 1),
            lambda: emit_mac6(b, st["t6b"], giv, st["XC"], scr),
            lambda: emit_mac6_gated(b, d3a, st["t6a"], st["XC"], st["BA0"], scr),
            lambda: emit_mac6(b, st["t6b"], giv, st["XC"], scr),
            lambda: _G8.w_pow2_xor(b, st["SG"], st["XC"], 1),
            lambda: emit_mac6(b, st["XC"], st["QV"], st["t6a"], scr),
            lambda: _G8.w_pow2_xor(b, st["SG"], st["XC"], 1),
            lambda: _G8.w_pow2_xor(b, st["QU"], st["XC"], 1),
            lambda: emit_mac6(b, st["QU"], st["XC"], st["t6a"], scr),
            lambda: _G8.w_pow2_xor(b, st["QU"], st["XC"], 1),
        ]
    for f in reversed(ops) if rev else ops:
        f()


def _xmacs(b, st, T, d2a, d3a, sx=None):
    """Emit the block X MACs with a single staged coefficient word XH:
    per column, stage the coefficient (pre-kappa KP), fire the MAC
    (real, or the CZ mesh vs sx in the mirror), unstage.  BKG = KP*giv
    brackets the whole sequence.
    """
    MT = T["MT"]
    scr = st["scr"]
    giv = st["t6c"]
    hb = flat_base([st["WA"]])
    Ub, Vb = flat_base([st["U"]]), flat_base([st["V2"]])
    Xb = flat_base([st["X"]])

    def mac(coef, vb):
        if sx is None:
            emit_macv48(b, MT, coef, vb, Xb, scr)
        else:
            emit_macv48_cz(b, coef, vb, sx, scr)

    emit_mac6_gated(b, d2a, st["KP"], giv, st["BKG"], scr)
    if d3a is not None:
        emit_mac6_gated(b, d3a, st["KP"], giv, st["BKG"], scr)
    # column 0 (head): coeff = BKG*av[1]
    emit_mac6(b, st["BKG"], st["BA1"], st["XH"], scr)
    mac(st["XH"], hb)
    emit_mac6(b, st["BKG"], st["BA1"], st["XH"], scr)
    # column 1 (U): pair kg, triple kg*c2
    _G8.mw_xor(b, d2a, st["BKG"], st["XH"])
    if d3a is not None:
        emit_mac6(b, st["BKG"], st["BA2"], st["XH"], scr)
    mac(st["XH"], Ub)
    if d3a is not None:
        emit_mac6(b, st["BKG"], st["BA2"], st["XH"], scr)
    _G8.mw_xor(b, d2a, st["BKG"], st["XH"])
    # column 2 (V2): triple kg
    if d3a is not None:
        _G8.mw_xor(b, d3a, st["BKG"], st["XH"])
        mac(st["XH"], Vb)
        _G8.mw_xor(b, d3a, st["BKG"], st["XH"])
    if d3a is not None:
        emit_mac6_gated(b, d3a, st["KP"], giv, st["BKG"], scr)
    emit_mac6_gated(b, d2a, st["KP"], giv, st["BKG"], scr)


def _kappa_block(b, st, d2a, d3a, rev=False):
    scr = st["scr"]

    def _swaps():
        for q in range(6):
            b.cswap(d2a, st["KP"][q], st["KM"][q])
            if d3a is not None:
                b.cswap(d3a, st["KP"][q], st["KM"][q])

    ops = [
        lambda: emit_mac6(b, st["Cu"], st["BA0"], st["BFC"], scr),
        lambda: emit_mac6(b, st["BFC"], st["KP"], st["KM"], scr),
        _swaps,
    ]
    for f in reversed(ops) if rev else ops:
        f()


def _assembly(b, st, T, d2a, d3a, rev=False):
    """wm ^= FC0*head ^ (Cu*av1)*U ^ (Cu | Cu*av2)*V2 (class-fused)."""
    MT = T["MT"]
    scr = st["scr"]
    hb, mb_ = flat_base([st["WA"]]), flat_base([st["WB"]])
    Ub, Vb = flat_base([st["U"]]), flat_base([st["V2"]])
    ops = [
        lambda: emit_macv48(b, MT, st["BFC"], hb, mb_, scr),
        lambda: emit_mac6(b, st["Cu"], st["BA1"], st["t6a"], scr),
        lambda: emit_macv48(b, MT, st["t6a"], Ub, mb_, scr),
        lambda: emit_mac6(b, st["Cu"], st["BA1"], st["t6a"], scr),
        lambda: _G8.mw_xor(b, d2a, st["Cu"], st["t6a"]),
        lambda: emit_mac6(b, st["Cu"], st["BA2"], st["t6a"], scr) if d3a is not None else None,
        lambda: emit_macv48(b, MT, st["t6a"], Vb, mb_, scr),
        lambda: emit_mac6(b, st["Cu"], st["BA2"], st["t6a"], scr) if d3a is not None else None,
        lambda: _G8.mw_xor(b, d2a, st["Cu"], st["t6a"]),
    ]
    for f in reversed(ops) if rev else ops:
        f()


def emit_slot(b, st, T, j, phase="P1"):  # noqa: C901 - straight-line circuit emitter
    """Forward walk slot (mirrorable)."""
    LT, MT = T["LT"], T["MT"]
    service = j in SERVICE
    v3cap = j in V3CAP
    sb = flat_base([st["s"]])
    hb, mb_ = flat_base([st["WA"]]), flat_base([st["WB"]])
    Ub, Vb = flat_base([st["U"]]), flat_base([st["V2"]])
    Xb = flat_base([st["X"]])
    scr, zt = st["scr"], st["zt"]
    swj = st["SWJ"][j]
    with b.phase(f"{phase}/s{j:02d}"):
        _slot_open(b, st, j, service, v3cap)
        # B01 natives
        emit_matvec_proj_loop(b, sb, hb, Ub, scr, LT)
        emit_sigma_tap(b, hb, st["SG"])
        _qscan1(b, T, sb, hb, st["QF"], scr)
        emit_sigma_tap(b, Ub, st["DLT"])
        # B02 trigger
        emit_ztest(b, list(st["SG"]), st["trig"], zt)
        _capf_tree(b, st, zt)
        # B03 freeze
        _freeze_fwd(b, st, j)
        # B04 service natives
        if service:
            _unpark(b, st, ("V2",))
            emit_matvec_proj_loop(b, sb, Ub, Vb, scr, LT)
            _qscan1(b, T, sb, Ub, st["QU"], scr)
            emit_ztest(b, list(st["QF"]), st["zqf"], zt)
            if v3cap:
                _qscan1(b, T, sb, Vb, st["QV"], scr)
                emit_sigma_tap(b, Vb, st["SV"])
                emit_ztest(b, list(st["DLT"]), st["zdd"], zt)
        # B05 held gates + ptap routes
        d2a = d3a = t_qz = None
        t_nrm = b.and_compute(st["exh"], st["trig"], pol=(False, False))
        if service:
            t_tl = b.and_compute(st["exh"], st["trig"], pol=(False, True))
            d2a = b.and_compute(t_tl, st["zqf"], pol=(True, False))
            if v3cap:
                t_qz = b.and_compute(t_tl, st["zqf"])
                d3a = b.and_compute(t_qz, st["zdd"], pol=(True, False))
            for q in range(6):
                b.cswap(d2a, st["SG"][q], st["QF"][q])
                if v3cap:
                    b.cswap(d3a, st["SG"][q], st["DLT"][q])
        # B06 swap journal: SWJ[j] = t_nrm ^ d2a ^ d3a
        b.cx(t_nrm, swj)
        if service:
            b.cx(d2a, swj)
            if v3cap:
                b.cx(d3a, swj)
        # B07 engine + entering rotation
        _inv_p(b, st, st["SG"], st["t6c"], scr)
        if j == 0:
            for q in range(6):
                b.cx(st["SG"][q], st["KP"][q])
        else:
            emit_mac6(b, st["Cu"], st["t6c"], st["Z6"], scr)
            for q in range(6):
                b.cswap(swj, st["Cu"][q], st["Z6"][q])
            emit_mac6(b, st["Cu"], st["SG"], st["Z6"], scr)
        # B08 normal path
        emit_mac6_gated(b, t_nrm, st["KP"], st["t6c"], st["XC"], scr)
        emit_macv48(b, MT, st["XC"], hb, Xb, scr)
        emit_mac6_gated(b, t_nrm, st["KP"], st["t6c"], st["XC"], scr)
        emit_mac6_gated(b, t_nrm, st["QF"], st["t6c"], st["alpha"], scr)
        emit_mac6(b, st["Cu"], st["alpha"], st["Ca"], scr)
        emit_mac6(b, st["Ca"], st["KP"], st["KM"], scr)
        for q in range(6):
            b.cswap(t_nrm, st["KP"][q], st["KM"][q])
        _G8.mw_xor(b, t_nrm, st["Cu"], st["t6b"])
        emit_macv48(b, MT, st["t6b"], Ub, mb_, scr)
        _G8.mw_xor(b, t_nrm, st["Cu"], st["t6b"])
        emit_macv48(b, MT, st["Ca"], hb, mb_, scr)
        # B09 service block
        if service:
            _coeffs(b, st, d2a, d3a)
            _xmacs(b, st, T, d2a, d3a)
            _kappa_block(b, st, d2a, d3a)
            _assembly(b, st, T, d2a, d3a)
            emit_mac6(b, st["Cu"], st["BA0"], st["BFC"], scr)  # BFC clear
            _coeffs(b, st, d2a, d3a, rev=True)
            if v3cap:
                _G8.mw_xor(b, d3a, st["Cu"], st["XC"])  # CV3 stage
        # B10 clears + leaving rotation
        emit_mac6_mbu(b, st["Cu"], st["alpha"], st["Ca"], scr)
        emit_mac6_gated_mbu(b, t_nrm, st["QF"], st["t6c"], st["alpha"], scr)
        emit_mac6(b, st["Cu"], st["SG"], st["Z6"], scr)
        for q in range(6):
            b.cswap(swj, st["Cu"][q], st["Z6"][q])
        emit_mac6(b, st["Cu"], st["t6c"], st["Z6"], scr)
        _inv_p(b, st, st["SG"], st["t6c"], scr)
        if service:
            for q in range(6):
                if v3cap:
                    b.cswap(d3a, st["SG"][q], st["DLT"][q])
                b.cswap(d2a, st["SG"][q], st["QF"][q])
        # B11 capf underive
        _capf_conj(b, st, j, zt)
        # B12 held-gate releases (zqf/zdd/trig/exh live)
        if service:
            if v3cap:
                b.and_release(d3a, t_qz, st["zdd"], pol=(True, False))
                b.and_release(t_qz, t_tl, st["zqf"])
            b.and_release(d2a, t_tl, st["zqf"], pol=(True, False))
            b.and_release(t_tl, st["exh"], st["trig"], pol=(False, True))
        b.and_release(t_nrm, st["exh"], st["trig"], pol=(False, False))
        # B13 zqf/zdd underives (QF/DLT live)
        if service:
            emit_ztest(b, list(st["QF"]), st["zqf"], zt)
            if v3cap:
                emit_ztest(b, list(st["DLT"]), st["zdd"], zt)
        # B14 trig underive (SG live)
        emit_ztest(b, list(st["SG"]), st["trig"], zt)
        # B15 tap clears (strip sources live)
        if service:
            _qscan1_mbu(b, T, sb, Ub, st["QU"], scr)
            if v3cap:
                _qscan1_mbu(b, T, sb, Vb, st["QV"], scr)
                emit_sigma_tap(b, Vb, st["SV"])
        _qscan1_mbu(b, T, sb, hb, st["QF"], scr)
        emit_sigma_tap(b, Ub, st["DLT"])
        emit_sigma_tap(b, hb, st["SG"])
        # B16 dances + disposals
        if service:
            emit_matvec_mbu_loop(b, sb, hb, Ub, scr, LT)  # U clear #1
            if v3cap:
                emit_matvec_proj_loop(b, sb, Vb, Ub, scr, LT)  # U = v3
                emit_macv48(b, MT, st["XC"], Ub, mb_, scr)  # Cu*v3
                st["fix"].append(("cv3", j, [b.hmr(st["XC"][q]) for q in range(6)]))
                emit_matvec_mbu_loop(b, sb, Vb, Ub, scr, LT)  # U clear
            st["fix"].append(("v2d", j, [b.hmr(st["V2"][q]) for q in range(288)]))
            _park(b, st, ("V2",))
        else:
            emit_matvec_mbu_loop(b, sb, hb, Ub, scr, LT)
        # B17 role swap on the journal bit
        _cswap288(b, swj, hb, mb_)
        _slot_close(b, st, j, service, v3cap)


# ---- 7. P2: the mirror slot ------------------------------------------------
def emit_slot_mirror(b, st, T, j, sx, fixmap, phase="P2"):  # noqa: C901 - straight-line circuit emitter
    """Exact conjugate of emit_slot."""
    LT, MT = T["LT"], T["MT"]
    service = j in SERVICE
    v3cap = j in V3CAP
    sb = flat_base([st["s"]])
    hb, mb_ = flat_base([st["WA"]]), flat_base([st["WB"]])
    Ub, Vb = flat_base([st["U"]]), flat_base([st["V2"]])
    scr, zt = st["scr"], st["zt"]
    swj = st["SWJ"][j]
    with b.phase(f"{phase}/s{j:02d}"):
        _slot_open(b, st, j, service, v3cap, journal=False)
        # B17' unswap on the live journal bit
        _cswap288(b, swj, hb, mb_)
        # B16' re-materializations + v2d repair
        emit_matvec_proj_loop(b, sb, hb, Ub, scr, LT)
        if service:
            _unpark(b, st, ("V2",))
            emit_matvec_proj_loop(b, sb, Ub, Vb, scr, LT)
            ids = fixmap.pop(("v2d", j))
            for q, o in enumerate(ids):
                b.z(st["V2"][q], cond=o)
            for o in ids:
                b.retire(o)
        # B15' tap re-creates
        emit_sigma_tap(b, hb, st["SG"])
        emit_sigma_tap(b, Ub, st["DLT"])
        _qscan1(b, T, sb, hb, st["QF"], scr)
        if service:
            if v3cap:
                emit_sigma_tap(b, Vb, st["SV"])
                _qscan1(b, T, sb, Vb, st["QV"], scr)
            _qscan1(b, T, sb, Ub, st["QU"], scr)
        # B14'/B13' trig + zqf/zdd re-derives
        emit_ztest(b, list(st["SG"]), st["trig"], zt)
        if service:
            if v3cap:
                emit_ztest(b, list(st["DLT"]), st["zdd"], zt)
            emit_ztest(b, list(st["QF"]), st["zqf"], zt)
        # B12' held gates re-derive
        t_nrm = b.and_compute(st["exh"], st["trig"], pol=(False, False))
        d2a = d3a = t_qz = t_tl = None
        if service:
            t_tl = b.and_compute(st["exh"], st["trig"], pol=(False, True))
            d2a = b.and_compute(t_tl, st["zqf"], pol=(True, False))
            if v3cap:
                t_qz = b.and_compute(t_tl, st["zqf"])
                d3a = b.and_compute(t_qz, st["zdd"], pol=(True, False))
        # B11' capf re-derive
        _capf_conj(b, st, j, zt)
        # B10' ptap routes re-open + leaving-rotation inverse (restores
        # the mid-slot Cu) + alpha/Ca re-create
        if service:
            for q in range(6):
                b.cswap(d2a, st["SG"][q], st["QF"][q])
                if v3cap:
                    b.cswap(d3a, st["SG"][q], st["DLT"][q])
        _inv_p(b, st, st["SG"], st["t6c"], scr)
        emit_mac6(b, st["Cu"], st["t6c"], st["Z6"], scr)
        for q in range(6):
            b.cswap(swj, st["Cu"][q], st["Z6"][q])
        emit_mac6(b, st["Cu"], st["SG"], st["Z6"], scr)
        emit_mac6_gated(b, t_nrm, st["QF"], st["t6c"], st["alpha"], scr)
        emit_mac6(b, st["Cu"], st["alpha"], st["Ca"], scr)
        # B09' service block inverse
        if service:
            if v3cap:
                # v3 dance inverse (cv3 stage/unstage inside; mid Cu live)
                _G8.mw_xor(b, d3a, st["Cu"], st["XC"])
                emit_matvec_mbu_loop(b, sb, hb, Ub, scr, LT)  # drop B*head
                emit_matvec_proj_loop(b, sb, Vb, Ub, scr, LT)  # U = v3
                emit_macv48(b, MT, st["XC"], Ub, mb_, scr)  # unwrite
                ids = fixmap.pop(("cv3", j))
                for q, o in enumerate(ids):
                    b.z(st["XC"][q], cond=o)
                for o in ids:
                    b.retire(o)
                emit_matvec_mbu_loop(b, sb, Vb, Ub, scr, LT)  # drop v3
                _G8.mw_xor(b, d3a, st["Cu"], st["XC"])
                emit_matvec_proj_loop(b, sb, hb, Ub, scr, LT)  # U back
            _coeffs(b, st, d2a, d3a)
            emit_mac6(b, st["Cu"], st["BA0"], st["BFC"], scr)  # BFC re-make
            _assembly(b, st, T, d2a, d3a, rev=True)
            _kappa_block(b, st, d2a, d3a, rev=True)
            _xmacs(b, st, T, d2a, d3a, sx=sx)
            _coeffs(b, st, d2a, d3a, rev=True)
        # B08' normal path inverse
        emit_macv48(b, MT, st["Ca"], hb, mb_, scr)
        _G8.mw_xor(b, t_nrm, st["Cu"], st["t6b"])
        emit_macv48(b, MT, st["t6b"], Ub, mb_, scr)
        _G8.mw_xor(b, t_nrm, st["Cu"], st["t6b"])
        for q in range(6):
            b.cswap(t_nrm, st["KP"][q], st["KM"][q])
        emit_mac6(b, st["Ca"], st["KP"], st["KM"], scr)
        emit_mac6_mbu(b, st["Cu"], st["alpha"], st["Ca"], scr)
        emit_mac6_gated_mbu(b, t_nrm, st["QF"], st["t6c"], st["alpha"], scr)
        emit_mac6_gated(b, t_nrm, st["KP"], st["t6c"], st["XC"], scr)
        emit_macv48_cz(b, st["XC"], hb, sx, scr)
        emit_mac6_gated(b, t_nrm, st["KP"], st["t6c"], st["XC"], scr)
        # B07' engine inverse
        if j == 0:
            for q in range(6):
                b.cx(st["SG"][q], st["KP"][q])
        else:
            emit_mac6(b, st["Cu"], st["SG"], st["Z6"], scr)
            for q in range(6):
                b.cswap(swj, st["Cu"][q], st["Z6"][q])
            emit_mac6(b, st["Cu"], st["t6c"], st["Z6"], scr)
        _inv_p(b, st, st["SG"], st["t6c"], scr)
        # B06' swap-journal underive (gates live)
        if service:
            if v3cap:
                b.cx(d3a, swj)
            b.cx(d2a, swj)
        b.cx(t_nrm, swj)
        # B05' ptap unroute + gate releases
        if service:
            for q in range(6):
                if v3cap:
                    b.cswap(d3a, st["SG"][q], st["DLT"][q])
                b.cswap(d2a, st["SG"][q], st["QF"][q])
            if v3cap:
                b.and_release(d3a, t_qz, st["zdd"], pol=(True, False))
                b.and_release(t_qz, t_tl, st["zqf"])
            b.and_release(d2a, t_tl, st["zqf"], pol=(True, False))
            b.and_release(t_tl, st["exh"], st["trig"], pol=(False, True))
        b.and_release(t_nrm, st["exh"], st["trig"], pol=(False, False))
        # B04' service natives un-create (zq underives + tap/strip mbus)
        if service:
            if v3cap:
                emit_ztest(b, list(st["DLT"]), st["zdd"], zt)
                emit_sigma_tap(b, Vb, st["SV"])
                _qscan1_mbu(b, T, sb, Vb, st["QV"], scr)
            emit_ztest(b, list(st["QF"]), st["zqf"], zt)
            _qscan1_mbu(b, T, sb, Ub, st["QU"], scr)
            emit_matvec_mbu_loop(b, sb, Ub, Vb, scr, LT)
            _park(b, st, ("V2",))
        # B03' freeze inverse
        _freeze_rev(b, st, j)
        # B02' trigger inverse
        _capf_tree(b, st, zt)
        emit_ztest(b, list(st["SG"]), st["trig"], zt)
        # B01' natives un-create
        emit_sigma_tap(b, Ub, st["DLT"])
        _qscan1_mbu(b, T, sb, hb, st["QF"], scr)
        emit_sigma_tap(b, hb, st["SG"])
        emit_matvec_mbu_loop(b, sb, hb, Ub, scr, LT)
        _slot_close(b, st, j, service, v3cap, journal=True)


# ---- 3. mux + 5. unmux: the fused kind sandwich ---------------------------
def emit_sketch_proj(b, s_base, v_base, K48, scr, LT):
    """K ^= R.(B*v) via the TGR spray table (1,845 Tof): the loop-
    form matvec with the 48-bit compressed scatter.
    """

    def middle(p, t, ua, vb):
        with b.gated(LT["T15"], t):
            tq = b.and_compute(ua, vb)
        with b.repeat(48) as k, b.gated(LT["TGR"], p * (NT16 * 48) + t * 48 + k):
            b.cx(tq, K48 + k)
        with b.gated(LT["T15"], t):
            b.and_release(tq, ua, vb)

    _matvec_core(b, s_base, v_base, scr, LT, middle)


def emit_sketch_mbu(b, s_base, v_base, K48ids, scr, LT):
    """MBU-clear of K holding exactly R.(B*v), operands live (0
    Tof).
    """
    ms = [b.hmr(q) for q in K48ids]

    def middle(p, t, ua, vb):
        e0 = p * (NT16 * 48) + t * 48
        for k in range(48):
            with b.gated(LT["TGR"], e0 + k):
                b.cz(ua, vb, cond=ms[k])

    _matvec_core(b, s_base, v_base, scr, LT, middle)
    for m_ in ms:
        b.retire(m_)


def emit_ztest_strip_flat(b, strip_ids, flag, host_ids):
    """flag ^= [288q strip == 0]: one in-place 288-leaf AND tree whose
    287 nodes host in the (provably |0>) host_ids.  287 Tof; the free
    X-conjugation runs on the strip wires themselves.
    """
    for q in strip_ids:
        b.x(q)
    b.recycle_all(list(host_ids))
    _and_tree(b, list(strip_ids), lambda top: b.cx(top, flag))
    b.reclaim(list(host_ids))
    for q in strip_ids:
        b.x(q)


def _ktree_flat(b, st, tP):
    """capk2 ^= tP & [K == 0], K = the 48-bit sketch hosted on the
    coefficient words; the 47-node tree hosts in V2's dead ids.
    """
    host = list(st["V2"])
    for q in st["K48"]:
        b.x(q)
    b.recycle_all(host)

    def mid(top):
        t2 = b.and_compute(tP, top)
        b.cx(t2, st["capk2"])
        b.and_release(t2, tP, top)

    _and_tree(b, list(st["K48"]), mid)
    b.reclaim(host)
    for q in st["K48"]:
        b.x(q)


def emit_sandwich(b, st, T, rev=False):
    """Fused kind tests + k2 lam staging.  capk1 ^= [WA != 0] &
    [B*WA == 0]; capk2 ^= [WA != 0] & [B*WA != 0] & [K == 0], with
    K = R.(B*U) the one-sided 48-bit sketch of B^2*WA.  The staged
    U = B*WA is the k2 lam park (U is NOT poolable through the tail).
    Test trees host in V2's dead ids; K hosts on the coefficient
    words -- both provably |0> here (the battery gates it).
    """
    LT = T["LT"]
    sb = flat_base([st["s"]])
    hb, Ub = flat_base([st["WA"]]), flat_base([st["U"]])
    xb = flat_base([st["X"]])
    scr = st["scr"]
    pnz, z1 = st["zqf"], st["zdd"]
    borrow = ("scr", "zqf", "zdd")  # scr + the wires pnz/z1 borrow
    host = list(st["V2"])
    # V2 is the host pool for the strip ztest trees and the K tree in
    # this span: re-materialized from its park, ownership released for
    # hosting, re-claimed and re-parked at exit. The sketch hosts and
    # the borrowed test flags re-materialize alongside it.
    _unpark(b, st, ("V2",))
    _unpark(b, st, _K48_W, owner="scal")
    _unpark(b, st, borrow, owner="flag")
    b.release("V2", host)
    if not rev:
        _unpark(b, st, ("U",))
        emit_ztest_strip_flat(b, list(st["WA"]), pnz, host)
        b.x(pnz)  # pnz = [WA != 0]
        emit_matvec_proj_loop(b, sb, hb, Ub, scr, LT)  # U = B*WA
        emit_ztest_strip_flat(b, list(st["U"]), z1, host)
        t = b.and_compute(pnz, z1)
        b.cx(t, st["capk1"])
        b.and_release(t, pnz, z1)
        b.x(z1)
        tP = b.and_compute(pnz, z1)  # held across the sketch
        emit_sketch_proj(b, sb, Ub, st["K48"][0], scr, LT)
        _ktree_flat(b, st, tP)
        emit_sketch_mbu(b, sb, Ub, list(st["K48"]), scr, LT)
        b.and_release(tP, pnz, z1)
        b.x(z1)
        emit_ztest_strip_flat(b, list(st["U"]), z1, host)
        b.x(pnz)
        emit_ztest_strip_flat(b, list(st["WA"]), pnz, host)
        _cswap288(b, st["capk2"], xb, Ub)
        _cswap288(b, st["capk1"], xb, hb)
    else:
        _cswap288(b, st["capk1"], xb, hb)
        _cswap288(b, st["capk2"], xb, Ub)
        emit_ztest_strip_flat(b, list(st["WA"]), pnz, host)
        b.x(pnz)
        emit_ztest_strip_flat(b, list(st["U"]), z1, host)
        b.x(z1)
        tP = b.and_compute(pnz, z1)
        emit_sketch_proj(b, sb, Ub, st["K48"][0], scr, LT)
        _ktree_flat(b, st, tP)
        emit_sketch_mbu(b, sb, Ub, list(st["K48"]), scr, LT)
        b.and_release(tP, pnz, z1)
        b.x(z1)
        t = b.and_compute(pnz, z1)
        b.cx(t, st["capk1"])
        b.and_release(t, pnz, z1)
        emit_ztest_strip_flat(b, list(st["U"]), z1, host)
        emit_matvec_mbu_loop(b, sb, hb, Ub, scr, LT)  # U -> 0, free
        b.x(pnz)
        emit_ztest_strip_flat(b, list(st["WA"]), pnz, host)
    b.claim("V2", host)
    _park(b, st, _K48_W, owner="scal")
    _park(b, st, borrow, owner="flag", recorded=False)
    _park(b, st, ("V2",))
    if rev:
        _park(b, st, ("U",))


# ---- 4. tail: the stripe-direct EI tail, tower-core sweep ----------------
def _stripe_tables():
    """M = OMH_pt . LMAP over the live points, point-major rows
    (3*i + lvl): row applied to the 206 EI products v gives the exact
    Wh value the sweep reads at live point i, level lvl.
    """
    ei = _ei_tables()
    LMAP = ei["LMAP"]
    LIVE = [p for p in range(512) if not SW_SKIP[p]]
    OMH_pt = np.ascontiguousarray(
        SW_OMH[:, :, LIVE].transpose(2, 0, 1).reshape(3 * len(LIVE), NSW)
    ).astype(np.uint8)
    M = np.zeros((OMH_pt.shape[0], LMAP.shape[1]), np.uint8)
    for oi in range(NSW):
        col = OMH_pt[:, oi]
        nz = col != 0
        if nz.any():
            M[nz] ^= MUL64[col[nz][:, None], LMAP[oi][None, :]]
    return M, LIVE


def emit_bank_stripe(b, BQ, si, TT, lam, s, rev, acc6):  # noqa: C901 - straight-line circuit emitter
    """Fill (rev=False) / release (rev=True) a stripe bank: one
    repeat(206) whose body carries the dense gather/fold meshes,
    gated on the LEVT/GEVT/fold bit tables (TT) at t-affine indices
    (the MUL64 bit meshes of the evaluation pairs, scanned in
    (row, r, j) order).  Fill: gated LT/GT gathers, one 15-Tof MAC
    into vq, gated fold vq -> bank words, MBU vq on in-body recycled
    outcomes.  Release is the unomq pattern: hmr the bank, then per
    product -- gated gathers, gated OXOR accumulation of the product
    mask mu_t onto the six zero-valued acc6 handles, the mesh walk
    vs acc6, gated OXOR un-accumulation restoring acc6 to zero (the
    OXOR lane work is free and unscored, and the six shared handles
    keep the outcome-id space small).  acc6 is minted zero once, at
    the first release stripe, through the public algebra (fresh
    combine, then XOR both sources back out).
    """
    W = len(BQ) // 6
    FST = 206 * W * 36
    LT6 = b.qubits(6)
    GT6 = b.qubits(6)

    def _lt(t):
        for l in range(48):
            for r in range(6):
                for j in range(6):
                    with b.gated(TT["LT"], t * 1728 + l * 36 + r * 6 + j):
                        b.cx(lam[6 * l + j], LT6[r])

    def _gt(t):
        for kx in range(123):
            for r in range(6):
                for j in range(6):
                    with b.gated(TT["GT"], t * 4428 + kx * 36 + r * 6 + j):
                        b.cx(s[6 * kx + j], GT6[r])

    def _fbit(t, Wi, r, j):
        return si * FST + t * (W * 36) + Wi * 36 + r * 6 + j

    if not rev:
        vq = b.qubits(6)
        with b.repeat(206) as t:
            _lt(t)
            _gt(t)
            emit_mac_g(b, (LT6, 0), I6M, (GT6, 0), I6M, macc=(vq, 0))
            for Wi in range(W):
                for r in range(6):
                    for j in range(6):
                        with b.gated(TT["FOLD"], _fbit(t, Wi, r, j)):
                            b.cx(vq[j], BQ[6 * Wi + r])
            ms = [b.hmr(vq[j]) for j in range(6)]

            def mid(_mi, om, ta, tb):
                for j in range(6):
                    if (om >> j) & 1:
                        b.cz(ta, tb, cond=ms[j])

            _walk_g(b, (LT6, 0), I6M, (GT6, 0), I6M, mid)
            for m_ in ms:
                b.retire(m_)
            _gt(t)
            _lt(t)
    else:
        mW = [[b.hmr(BQ[6 * Wi + r]) for r in range(6)] for Wi in range(W)]
        if not acc6:
            for _ in range(6):
                a = mW[0][0] ^ mW[0][1]
                a ^= mW[0][0]
                a ^= mW[0][1]
                acc6.append(a)

        def _mus(t):
            # mu_t accumulation onto acc6: dense (j, Wi, r) scan of
            # the fold table's column for product t
            for j in range(6):
                for Wi in range(W):
                    for r in range(6):
                        with b.gated(TT["FOLD"], _fbit(t, Wi, r, j)):
                            acc6[j] ^= mW[Wi][r]

        with b.repeat(206) as t:
            _lt(t)
            _gt(t)
            _mus(t)

            def mid(_mi, om, ta, tb):
                for j in range(6):
                    if (om >> j) & 1:
                        b.cz(ta, tb, cond=acc6[j])

            _walk_g(b, (LT6, 0), I6M, (GT6, 0), I6M, mid)
            _mus(t)
            _gt(t)
            _lt(t)
        for row in mW:
            for m_ in row:
                b.retire(m_)
    if not rev:
        b.recycle_all(list(reversed(list(vq))))
    b.recycle_all(list(reversed(list(GT6))))
    b.recycle_all(list(reversed(list(LT6))))


# -- the bit-collapsed tower-core sweep
# The sweep's per-point pay Tr2(mu * w / c) collapses to the tower
# form Tr8_2( inv8(N(c)) * Tr64_8(mu * w * c^8) ), N(c) = c^9 in the
# S8 subfield (exhaustively asserted below): 9 + 3 + 18 + 3 = 33 ANDs
# per point, against about 52 for a 3-product inversion ladder plus a
# masked pay walk.  inv8(0) = 0 pays 0 at c = 0.

_S8 = sorted({z for z in range(64) if int(SQT[SQT[SQT[z]]]) == z})
_S8B = []
_sp = {0}
for _z in _S8:
    if _z not in _sp:
        _S8B.append(_z)
        _sp = {a ^ z2 for a in _sp for z2 in (0, _z)}
assert len(_S8B) == 3 and len(_sp) == 8
_B8M = np.zeros((6, 3), np.uint8)
for _j, _z in enumerate(_S8B):
    for _r in range(6):
        _B8M[_r, _j] = (_z >> _r) & 1


def _f2_solve(Amat, bvec):
    A = np.concatenate([Amat.astype(np.uint8) & 1, (bvec[:, None] & 1)], 1).copy()
    rows, cols = A.shape
    piv = []
    r = 0
    for c in range(cols - 1):
        nz = [i for i in range(r, rows) if A[i, c]]
        if not nz:
            continue
        A[[r, nz[0]]] = A[[nz[0], r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c)
        r += 1
    x = np.zeros(cols - 1, np.uint8)
    for i, c in enumerate(piv):
        x[c] = A[i, -1]
    assert np.array_equal((Amat.astype(int) @ x) % 2, bvec & 1)
    return x


def _s8c(z):
    return _f2_solve(_B8M, np.array([(int(z) >> r) & 1 for r in range(6)], np.uint8))


def _quad_dec(nin, table):
    """table(x) = sum (u.x)(v.x) + lin.x (F2); polar peeling with an
    exhaustive residual assert; constant term must vanish.
    """
    xs = np.arange(1 << nin)
    bits = ((xs[:, None] >> np.arange(nin)) & 1).astype(np.uint8)
    tab = np.asarray(table, np.uint8) & 1
    idx = {tuple(b): k for k, b in enumerate(bits)}
    f0 = int(tab[0])
    P = np.zeros((nin, nin), np.uint8)
    for i in range(nin):
        for j in range(nin):
            if i != j:
                xi = np.zeros(nin, np.uint8)
                xi[i] = 1
                xj = np.zeros(nin, np.uint8)
                xj[j] = 1
                P[i, j] = (
                    int(tab[idx[tuple(xi ^ xj)]])
                    ^ int(tab[idx[tuple(xi)]])
                    ^ int(tab[idx[tuple(xj)]])
                    ^ f0
                )
    pairs = []
    Pc = P.copy()
    while Pc.any():
        ii, jj = np.argwhere(Pc)[0]
        u = Pc[ii].copy()
        v = Pc[jj].copy()
        pairs.append((v, u))
        Pc = (Pc ^ (np.outer(v, u) ^ np.outer(u, v))) & 1
    acc = np.zeros(len(xs), np.uint8)
    for u, v in pairs:
        acc ^= ((bits @ u) & 1) & ((bits @ v) & 1)
    diff = tab ^ acc
    lin = np.array(
        [int(diff[idx[tuple(np.eye(nin, dtype=np.uint8)[i])]]) ^ int(diff[0]) for i in range(nin)],
        np.uint8,
    )
    assert int(diff[0]) == 0, "constant term"
    assert np.array_equal(diff, ((bits @ lin) & 1).astype(np.uint8)), "quad residual not linear"
    return pairs, lin


def _bilin_dec(Bm):
    A = (Bm.copy() & 1).astype(np.uint8)
    pairs = []
    while A.any():
        i, j = np.argwhere(A)[0]
        pairs.append((A[:, j].copy(), A[i, :].copy()))
        A = (A ^ np.outer(pairs[-1][0], pairs[-1][1])) & 1
    return pairs


def _tower_core_tables():
    """Build the fixed decs (N, inv8, final) + per-live-point b' gather tables
    WM/CM (w/c-side masks) and OM (output-bit mask), 18 AND slots per
    point; exhaustively asserted against the pay truth per point.
    """
    C8 = [int(SQT[SQT[SQT[c]]]) for c in range(64)]
    NTAB = [int(MUL64[c, C8[c]]) for c in range(64)]
    IN8 = {z: (int(POW62[z]) if z else 0) for z in _S8}

    def tr64_8(z):
        return int(z) ^ int(SQT[SQT[SQT[int(z)]]])

    TR82 = {z: (z ^ int(SQT[z]) ^ int(SQT[SQT[z]])) for z in _S8}
    ndec = [_quad_dec(6, [int(_s8c(NTAB[c])[i]) for c in range(64)]) for i in range(3)]
    idec = []
    for i in range(3):
        tab = []
        for x in range(8):
            z = 0
            for j in range(3):
                if (x >> j) & 1:
                    z ^= _S8B[j]
            tab.append(int(_s8c(IN8[z])[i]))
        idec.append(_quad_dec(3, tab))
    FB = np.zeros((3, 3), np.uint8)
    for i in range(3):
        for j in range(3):
            FB[i, j] = TR82[int(MUL64[_S8B[i], _S8B[j]])]
    fdec = _bilin_dec(FB)
    LIVE = [p for p in range(512) if not SW_SKIP[p]]
    NL = len(LIVE)
    WM = np.zeros((NL, 18, 6), np.uint8)
    CM = np.zeros((NL, 18, 6), np.uint8)
    OM = np.zeros((NL, 18, 3), np.uint8)
    wg, cg = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    for i, p in enumerate(LIVE):
        mu = int(SW_MU_RES[p])
        slot = 0
        for bi in range(3):
            Bm = np.zeros((6, 6), np.uint8)
            for wb in range(6):
                for cb in range(6):
                    Bm[wb, cb] = _s8c(tr64_8(MUL64[int(MUL64[mu, 1 << wb]), C8[1 << cb]]))[bi]
            for u, v in _bilin_dec(Bm):
                WM[i, slot] = u
                CM[i, slot] = v
                OM[i, slot, bi] = 1
                slot += 1
        assert slot == 18, (p, slot)
        # exhaustive pay-truth assert for this point (vectorized)
        wv = ((wg[:, :, None] >> np.arange(6)) & 1).astype(np.uint8)
        cv = ((cg[:, :, None] >> np.arange(6)) & 1).astype(np.uint8)
        bb = np.zeros((64, 64, 3), np.uint8)
        for t in range(18):
            a = (wv @ WM[i, t]) & 1
            b_ = (cv @ CM[i, t]) & 1
            bb ^= (a & b_)[:, :, None] * OM[i, t][None, None, :]
        # N-bits of c
        nn = np.zeros((64, 64, 3), np.uint8)
        for bi in range(3):
            pr, lin = ndec[bi]
            acc = (cv @ lin) & 1
            for u, v in pr:
                acc ^= ((cv @ u) & 1) & ((cv @ v) & 1)
            nn[:, :, bi] = acc
        # m = inv8(N): recompute mm from nn (b' path used above is the
        # w-c bilinear; inv8 acts on N-bits)
        m2 = np.zeros((64, 64, 3), np.uint8)
        for bi in range(3):
            pr, lin = idec[bi]
            acc = (nn @ lin) & 1
            for u, v in pr:
                acc ^= ((nn @ u) & 1) & ((nn @ v) & 1)
            m2[:, :, bi] = acc
        pay = np.zeros((64, 64), np.uint8)
        for u, v in fdec:
            pay ^= ((bb @ u) & 1) & ((m2 @ v) & 1)
        exp = TR[MUL64[MUL64[mu, wg], POW62[cg]]].astype(np.uint8)
        exp[:, 0] = 0
        assert np.array_equal(pay, exp), f"tower-core pay truth at point {p}"
    return {"ndec": ndec, "idec": idec, "fdec": fdec, "WM": WM, "CM": CM, "OM": OM, "NL": NL}


def emit_tail_stripe(b, st, lam, k):  # noqa: C901, PLR0915 - straight-line circuit emitter
    """Emit the stripe-direct tail with the bit-collapsed sweep: fill and
    release each stripe bank (emit_bank_stripe); the sweep pays each point
    with the 33-AND tower core (_tower_core_tables).
    """
    s = st["s"]
    M, LIVE = _stripe_tables()
    NL = len(LIVE)
    assert NL % k == 0, (NL, k)
    TC = _tower_core_tables()
    assert TC["NL"] == NL
    # Contiguous runs: the sweep's loop bodies index these registers
    # with affine expressions, so each must be one ascending id run;
    # the parked strips guarantee adequate runs on the free-list.
    BQ = b.contiguous_qubits(18 * k, mint=False)
    Lh = b.contiguous_qubits(24, mint=False)
    Wh = b.contiguous_qubits(18, mint=False)
    cw = b.contiguous_qubits(12, mint=False)
    FC = b.contiguous_qubits(18, mint=False)
    DQ = b.contiguous_qubits(24, mint=False)
    NB = b.contiguous_qubits(3, mint=False)
    MB = b.contiguous_qubits(3, mint=False)
    BB = b.contiguous_qubits(3, mint=False)
    MULBITS = np.zeros((64, 6, 6), np.uint8)
    for c0 in range(1, 64):
        for j in range(6):
            v = int(MUL64[c0, 1 << j])
            for r in range(6):
                MULBITS[c0, r, j] = (v >> r) & 1
    lam_bits = MULBITS[SW_LAMH[:, :, LIVE]]
    TBL = b.declare_table(lam_bits.transpose(2, 0, 1, 3, 4).reshape(-1).tolist())
    TWM = b.declare_table(TC["WM"].reshape(-1).tolist())
    TCM = b.declare_table(TC["CM"].reshape(-1).tolist())
    TOM = b.declare_table(TC["OM"].reshape(-1).tolist())
    sel = np.zeros((k, 3 * k), np.uint8)
    for il in range(k):
        sel[il, 3 * il : 3 * il + 3] = 1
    TSEL = b.declare_table(sel.reshape(-1).tolist())
    # bank-stripe tables: MULBITS meshes of the EI evaluation pairs
    # (LEVT/GEVT) and of the per-stripe fold matrix M, in the
    # emitter's dense (row, r, j) scan order
    eit = _ei_tables()
    TT = {
        "LT": b.declare_table(MULBITS[eit["LEVT"]].reshape(-1).tolist()),
        "GT": b.declare_table(MULBITS[eit["GEVT"]].reshape(-1).tolist()),
        "FOLD": b.declare_table(
            MULBITS[M.reshape(NL // k, 3 * k, 206).transpose(0, 2, 1)].reshape(-1).tolist()
        ),
    }
    acc6 = []
    LSTRIDE = 4 * 48 * 36
    scr = b.contiguous_qubits(2, mint=False)
    ua, vb = scr[0], scr[1]

    def _fc_copies(rv):
        for lvl in range(3):
            for j in range(6):
                q = FC[6 * lvl + j]
                if rv:
                    b.x(q)
                    b.cx(Lh[6 * lvl + j], q)
                else:
                    b.cx(Lh[6 * lvl + j], q)
                    b.x(q)

    def _flag(base):
        ch = []
        for i in range(5):
            ch.append(b.and_compute(ch[-1] if ch else FC[base], FC[base + 1 + i]))
        return ch

    def _unflag(ch, base):
        for i in range(4, -1, -1):
            b.and_release(ch[i], ch[i - 1] if i else FC[base], FC[base + 1 + i])

    def _dq_diffs():
        for j in range(6):
            b.cx(Lh[6 + j], DQ[j])
            b.cx(Lh[12 + j], DQ[j])
            b.cx(Lh[12 + j], DQ[6 + j])
            b.cx(Lh[18 + j], DQ[6 + j])
            b.cx(Wh[j], DQ[12 + j])
            b.cx(Wh[6 + j], DQ[12 + j])
            b.cx(Wh[6 + j], DQ[18 + j])
            b.cx(Wh[12 + j], DQ[18 + j])

    MUX = ((Lh, 6, 0, 6, 0), (Wh, 0, 12, 18, 6))

    def _gfix(dec, wires, dst, mbu):
        """XOR the fixed quadratic form dec into/off dst (wires live)."""
        prs, lin = dec
        if not mbu:
            for u, v in prs:
                for kk in range(6 if len(u) == 6 else 3):
                    if u[kk]:
                        b.cx(wires[kk], ua)
                    if v[kk]:
                        b.cx(wires[kk], vb)
                t = b.and_compute(ua, vb)
                b.cx(t, dst)
                b.and_release(t, ua, vb)
                for kk in range(6 if len(u) == 6 else 3):
                    if v[kk]:
                        b.cx(wires[kk], vb)
                    if u[kk]:
                        b.cx(wires[kk], ua)
            for kk, l_ in enumerate(lin):
                if l_:
                    b.cx(wires[kk], dst)
        else:
            m = b.hmr(dst)
            for u, v in prs:
                for kk in range(6 if len(u) == 6 else 3):
                    if u[kk]:
                        b.cx(wires[kk], ua)
                    if v[kk]:
                        b.cx(wires[kk], vb)
                b.cz(ua, vb, cond=m)
                for kk in range(6 if len(u) == 6 else 3):
                    if v[kk]:
                        b.cx(wires[kk], vb)
                    if u[kk]:
                        b.cx(wires[kk], ua)
            for kk, l_ in enumerate(lin):
                if l_:
                    b.z(wires[kk], cond=m)
            b.retire(m)

    for r, j in SW_SYNZ:
        b.z(s[6 * r + j])
    for si in range(NL // k):
        with b.phase("tail/fill"):
            emit_bank_stripe(b, BQ, si, TT, lam, s, rev=False, acc6=acc6)
        with b.phase("tail/sweep"):
            l_any = lam_bits[:, :, k * si : k * (si + 1)].any(axis=2)
            with b.repeat(k) as ip:

                def gathers():
                    for lvl in range(4):
                        for sl in range(48):
                            cell = l_any[lvl, sl]
                            base = si * k * LSTRIDE + (lvl * 48 + sl) * 36
                            for r in range(6):
                                for j in range(6):
                                    if cell[r, j]:
                                        with b.gated(TBL, ip * LSTRIDE + base + r * 6 + j):
                                            b.cx(lam[6 * sl + j], Lh[6 * lvl + r])
                    for w in range(3 * k):
                        for r in range(6):
                            with b.gated(TSEL, ip * (3 * k) + w):
                                b.cx(BQ[6 * w + r], Wh[6 * (w % 3) + r])

                gathers()
                _fc_copies(False)
                ch_fr = _flag(0)
                ch_z1 = _flag(6)
                ch_z2 = _flag(12)
                fr, z1, z2 = ch_fr[-1], ch_z1[-1], ch_z2[-1]
                t12 = b.and_compute(z1, z2)
                _dq_diffs()
                for src, so, da, db, co in MUX:
                    for j in range(6):
                        b.cx(src[so + j], cw[co + j])
                        b.ccx(z1, DQ[da + j], cw[co + j])
                        b.ccx(t12, DQ[db + j], cw[co + j])
                # ---- tower core: N(c), inv8, b'(mu; w, c), pay ----
                for bi in range(3):
                    _gfix(TC["ndec"][bi], [cw[q] for q in range(6)], NB[bi], mbu=False)
                for bi in range(3):
                    _gfix(TC["idec"][bi], list(NB), MB[bi], mbu=False)
                pbase = si * k * 18
                for t in range(18):
                    with b.repeat(6) as jq:
                        with b.gated(TWM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[6 + jq], ua)
                        with b.gated(TCM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[jq], vb)
                    tq = b.and_compute(ua, vb)
                    with b.repeat(3) as jq, b.gated(TOM, ip * 18 * 3 + pbase * 3 + t * 3 + jq):
                        b.cx(tq, BB[jq])
                    b.and_release(tq, ua, vb)
                    with b.repeat(6) as jq:
                        with b.gated(TCM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[jq], vb)
                        with b.gated(TWM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[6 + jq], ua)
                for u, v in TC["fdec"]:
                    for kk in range(3):
                        if u[kk]:
                            b.cx(BB[kk], ua)
                        if v[kk]:
                            b.cx(MB[kk], vb)
                    tq = b.and_compute(ua, vb)
                    b.cz(fr, tq)
                    b.and_release(tq, ua, vb)
                    for kk in range(3):
                        if v[kk]:
                            b.cx(MB[kk], vb)
                        if u[kk]:
                            b.cx(BB[kk], ua)
                # ---- tower-core uncompute: b' mesh, inv8, N ----
                msb = [b.hmr(BB[jq]) for jq in range(3)]
                for t in range(18):
                    with b.repeat(6) as jq:
                        with b.gated(TWM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[6 + jq], ua)
                        with b.gated(TCM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[jq], vb)
                    for jq in range(3):
                        with b.gated(TOM, ip * 18 * 3 + pbase * 3 + t * 3 + jq):
                            b.cz(ua, vb, cond=msb[jq])
                    with b.repeat(6) as jq:
                        with b.gated(TCM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[jq], vb)
                        with b.gated(TWM, ip * 18 * 6 + pbase * 6 + t * 6 + jq):
                            b.cx(cw[6 + jq], ua)
                for m_ in msb:
                    b.retire(m_)
                for bi in (2, 1, 0):
                    _gfix(TC["idec"][bi], list(NB), MB[bi], mbu=True)
                for bi in (2, 1, 0):
                    _gfix(TC["ndec"][bi], [cw[q] for q in range(6)], NB[bi], mbu=True)
                # ---- MUX unwind ----
                for src, so, da, db, co in MUX:
                    for j in range(6):
                        m = b.hmr(cw[co + j])
                        b.z(src[so + j], cond=m)
                        b.cz(z1, DQ[da + j], cond=m)
                        b.cz(t12, DQ[db + j], cond=m)
                        b.retire(m)
                _dq_diffs()
                b.and_release(t12, z1, z2)
                for chain, base in ((ch_z2, 12), (ch_z1, 6), (ch_fr, 0)):
                    _unflag(chain, base)
                _fc_copies(True)
                gathers()
        with b.phase("tail/unfill"):
            emit_bank_stripe(b, BQ, si, TT, lam, s, rev=True, acc6=acc6)
    for a in acc6:
        b.retire(a)
    for q in (BB, MB, NB, DQ, FC, cw, Wh, Lh):
        b.recycle_all(list(reversed(list(q))))
    b.recycle_all(list(reversed(list(BQ))))
    b.recycle_all(list(reversed(list(scr))))


# ---- 6. xseam + assembly: the full palindrome ------------------------------
def build_full(nsteps=NSLOTS, with_tail=True):
    """Init + P1 + mux + tail + unmux + xseam + P2 + teardown."""
    from uncompute import Builder

    b = Builder(recycle_outcomes=True)
    T = build_tables(b)
    st = alloc_state(b)
    with b.phase("init"):
        _y0_init(b, st)
    for j in range(nsteps):
        emit_slot(b, st, T, j)
    with b.phase("mux"):
        emit_sandwich(b, st, T)
    if with_tail:
        # Every |0>-at-seam word is already parked by the slot and
        # sandwich closes (NOT U: the k2 staging; NOT the resident
        # core), so the tail allocates its pools straight from the
        # free-list.
        with b.phase("tail"):
            emit_tail_stripe(b, st, st["X"], k=16)
    with b.phase("unmux"):
        emit_sandwich(b, st, T, rev=True)
    with b.phase("xseam"):
        sx = [b.hmr(st["X"][q]) for q in range(288)]
        _park(b, st, ("X",))
    fixmap = {(tag, jj): ids for (tag, jj, ids) in st["fix"]}
    for j in range(nsteps - 1, -1, -1):
        emit_slot_mirror(b, st, T, j, sx, fixmap)
    assert not fixmap, f"unconsumed fixups: {sorted(fixmap)}"
    for o in sx:
        b.retire(o)
    with b.phase("P2/init"):
        _y0_teardown(b, st)
    return b, st


# ---- epilogue --------------------------------------------------------------
def build():
    """Run the one raw build: the full palindrome, without emission.  The
    liveness/occupancy profilers import this module, call it, and read
    the Builder's native wire_log; no post pass touches the allocator.
    """
    b, st = build_full()
    return b, st


if __name__ == "__main__":
    from uncompute.testing import assert_metrics, emit_and_report

    b, _st = build()
    # The 603-wire resident walk core stays live at |0> to the end; the
    # meta.json "size" bands declare it to the leak guard.
    assert_metrics(b, __file__)
    emit_and_report(b)
