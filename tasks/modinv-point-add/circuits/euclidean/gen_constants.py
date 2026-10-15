#!/usr/bin/env python3
"""Regenerate constants.json from scratch: the three priced sections.

This is the full provenance of every fitted constant the circuit loads:

  wl_windows   per-step window schedule (10 stats, 5 windows
                           + the newK truncation curve), eps 8e-5/lane
  tcore        t-block certainty core: RLO(T), certified
                           LOWER bound on the t-op right edge, eps 5e-5
  rcore        r-block certainty core: L_max(T)/R_min(T)
                           bounds, eps 5e-5 per side

Method (the fit pipeline):

  1. The value model is FastEEA — an integer-image twin of the baseline
     circuit's EEABitState (bit registers packed into ints), validated
     STEP-EXACT against it on 40 full-width lanes before any sampling.
     It records, per step T, the positions each block actually needs
     (window "needs"); ~2.7 ms per full-width lane.
  2. Base pass, 60,000 lanes (seeds "p1:0".."p1:59999"): per-(stat, T)
     need histograms -> quantile curves at BASE_DELTA = 1e-3 on the
     needing side; rows that fired < MINFIRE = 400 lanes stay at the
     worst-case window (sentinel -1).
  3. Dev pass, 300,000 fresh lanes ("p2:{i}"): per-lane worst deviation
     beyond each base curve (K-side high / k-side low, per stat; the
     t-core low side; the r-core L-high and R-low sides).
  4. Margins = max(empirical quantile at the stat's eps share, an
     exponential tail fit), clamped to [0, 48] — no safety decade; the
     margins are priced directly against the Unified(20) budget.
  5. Tables = base -/+ margin clipped inside the worst-case curves
     (window rows beyond the nmax cut, sentinel rows, and inverted
     rows fall back to worst-case).  The worst-case curves are this
     circuit's own (_active_windows_wc in circuit.py, the curves the
     circuit clips every table row into).  The baseline in
     ../luo-eea uses its own, tighter worst-case curves.
Every lane is seeded by a fixed string, so a fresh run reproduces
constants.json byte for byte.  generate() returns the file text;
main() writes it.  constants.json holds only the tables circuit.py
reads.

The passes run on a thread pool.  The C walk model beside this file
(fasteea.c, compiled on demand) releases the GIL, so the threads run
in parallel; it is used when it compiles and passes a bit-exactness
gate against the Python model (see _backend below).  Without a
compiler the pure-Python model runs, serialized by the GIL (tens of
minutes).  A thread pool, not a process pool, keeps generate()
callable from a module loaded by path.

RUN: uv run python gen_constants.py   (writes constants.json;
     review the change with git diff)

The eps ledger these tables price into is in circuit.py's header; the
window sensitivities (SENS: executed Toffoli per row-position of each
stat over the schedule) allocate the lane budget across stats.
"""

import ctypes
import json
import os
import pathlib
import random
import subprocess
import sys
import types
from multiprocessing.pool import ThreadPool

import numpy as np

from uncompute.testing import build_clib, load_contract, load_module

HERE = pathlib.Path(__file__).resolve().parent

BASELINE = HERE.parent / "luo-eea" / "circuit.py"
OWN = HERE / "circuit.py"
EEA_COMMON = HERE.parent / "_shared" / "eea_common.py"

P_MOD = int(load_contract(__file__)["p"], 16)
NMAX, CUT = 1616, 1456  # fit horizon / schedule cut
BASE_LANES, FIT_LANES = 60_000, 300_000
BASE_DELTA, MINFIRE = 1e-3, 400
EPS_WL, EPS_CORE = 8e-5, 5e-5  # per-lane budgets: windows / each core
NEEDMAX = 264
SENT = -1
NAMES = ("r", "swap", "t", "len_t", "len_rp")
STATS = (
    "r_k",
    "r_K",
    "swap_k",
    "swap_K",
    "t_K",
    "len_t_k",
    "len_t_K",
    "len_rp_k",
    "len_rp_K",
    "newK",
)
KSIDE = {s: s.endswith("_k") for s in STATS}
# executed-Toffoli sensitivity per row position (measured census of
# the emitted schedule); allocates EPS_WL across stats.
SENS = {
    "r_k": 39,
    "r_K": 39,
    "swap_k": 2,
    "swap_K": 2,
    "t_K": 47,
    "len_t_k": 8,
    "len_t_K": 8,
    "len_rp_k": 8,
    "len_rp_K": 8,
    "newK": 12,
}


def load_baseline():
    """Load the baseline surface as one namespace: the shared bit model
    (../eea_common.py, direct frame) bound to the baseline circuit.py's
    schedule (the build-path names nmax_steps, active_windows,
    ls_width).
    """
    E = load_module(EEA_COMMON, "_eea_common")
    S = load_module(BASELINE, "luo_eea_circuit", register=True)

    def bitstate(p, x):
        return E.EEABitState(
            p, x, active_windows=S.active_windows, nmax_steps=S.nmax_steps, ls_width=S.ls_width
        )

    return types.SimpleNamespace(
        eea_reference=E.eea_reference, EEABitState=bitstate, nmax_steps=S.nmax_steps
    )


def _rev(x, w):
    return int(format(x, f"0{w}b")[::-1], 2) if w else 0


class FastEEA:
    """Integer-image twin of the baseline EEABitState + need recorder.

    Registers are ints with bit (j-1) = position j (1-based, as in
    Luo et al.);
    big-endian slice ops go through a string bit-reverse. Semantics
    transliterated from EEABitState.step and validated step-exactly by
    validate_twin() below.
    """

    __slots__ = (
        "T",
        "W",
        "lq",
        "lrp",
        "ls",
        "lt",
        "maskW",
        "n",
        "needs",
        "nmax",
        "p",
        "p1",
        "p2",
        "sign",
        "w1",
        "w2",
    )

    def __init__(self, L, p, x, nmax):
        self.p = p
        self.n = n = p.bit_length()
        self.W = W = n + 3
        self.maskW = (1 << W) - 1
        ref = L.eea_reference(p, x)
        x_used = ref["x_used"]
        w1 = 1  # position 1 bit set (t = 1)
        for i in range(n):
            if (p >> (n - 1 - i)) & 1:
                w1 |= 1 << (3 + i)
        w2 = 0
        for i in range(n):
            if (x_used >> (n - 1 - i)) & 1:
                w2 |= 1 << (3 + i)
        self.w1, self.w2 = w1, w2
        self.lt, self.lq, self.ls = 1, 0, 0
        self.lrp = x_used.bit_length()
        self.p1 = self.p2 = self.sign = 0
        self.T = 0
        self.nmax = nmax
        self.needs = {s: [0] * nmax for s in STATS}

    def _be(self, reg, Lp, R):
        w = R - Lp + 1
        return _rev((reg >> (Lp - 1)) & ((1 << w) - 1), w)

    def _be_set(self, reg, Lp, R, val):
        w = R - Lp + 1
        m = ((1 << w) - 1) << (Lp - 1)
        return (reg & ~m) | (_rev(val, w) << (Lp - 1))

    def step(self, T):
        n, W = self.n, self.W
        nd = self.needs
        i = T - 1
        alive = self.lrp != 0
        p1, p2 = self.p1, self.p2

        if p1 == 0:
            if alive:
                w2 = self.w2
                self.w2 = (w2 >> 1) | ((w2 & 1) << (W - 1))
            self.ls += 1
            if p2 == 1:
                if alive:
                    w2 = self.w2
                    self.w2 = ((w2 << 2) | (w2 >> (W - 2))) & self.maskW
                self.ls -= 2
        assert self.ls >= 0

        if alive and p1 == 0:
            Lp = self.lt + self.lq + 2
            R = W - self.ls
            nd["r_k"][i] = Lp
            nd["r_K"][i] = R
            if Lp <= R:
                wd = R - Lp + 1
                u = self._be(self.w1, Lp, R)
                v = self._be(self.w2, Lp, R)
                self.w1 = self._be_set(self.w1, Lp, R, (u - v) % (1 << wd))
                self.sign ^= 1 if u < v else 0
            if p2 == 1:
                self.sign ^= 1
            if (p2 == 0 or self.sign == 0) and Lp <= R:
                wd = R - Lp + 1
                u = self._be(self.w1, Lp, R)
                v = self._be(self.w2, Lp, R)
                self.w1 = self._be_set(self.w1, Lp, R, (u + v) % (1 << wd))

        if (p1 ^ p2) == 1:
            J = self.lt + self.lq + 1 + (1 if p1 == 0 else 0)
            nd["swap_k"][i] = J
            nd["swap_K"][i] = J
            bJ = (self.w1 >> (J - 1)) & 1
            if self.sign != bJ:
                self.w1 ^= 1 << (J - 1)
            self.sign = bJ
            self.lq += -1 if p1 == 1 else 1
            assert self.lq >= 0

        if p1 == 1:
            if p2 == 0:
                R = self.lt + 1
            else:
                R = W - self.lrp - self.ls
            nd["t_K"][i] = R
            mR = (1 << R) - 1
            if p2 == 1 or self.sign == 0:
                tp, tt = self.w2 & mR, self.w1 & mR
                self.w2 = (self.w2 & ~mR) | ((tp - tt) % (1 << R))
            self.sign ^= 1
            tp, tt = self.w2 & mR, self.w1 & mR
            s = tp + tt
            self.sign ^= (s >> R) & 1
            self.w2 = (self.w2 & ~mR) | (s & mR)
            w2 = self.w2
            self.w2 = (w2 >> 1) | ((w2 & 1) << (W - 1))
            self.ls += 1
            if p2 == 1:
                w2 = self.w2
                self.w2 = ((w2 << 2) | (w2 >> (W - 2))) & self.maskW
                self.ls -= 2
        assert self.ls >= 0

        if self.lq == 0 and self.lrp > 0:
            self.p2 ^= self.sign ^ self.p1
            self.sign ^= self.p2
        if self.ls == 0:
            self.p1 ^= 1
            self.p2 ^= 1

        if T % 4 == 0 and self.lq == 0 and self.ls == 0:
            self.w1, self.w2 = self.w2, self.w1
            B = W - self.lrp
            mB = (1 << B) - 1
            old_lt = (self.w2 & mB).bit_length()
            new_lt = (self.w1 & mB).bit_length()
            assert old_lt == self.lt
            nd["len_t_k"][i] = min(self.lt, new_lt) if new_lt else self.lt
            nd["len_t_K"][i] = B
            lrp_old = self.lrp
            self.lt = new_lt
            A = self.lt + 2
            hi1 = self.w1 >> (A - 1)
            m_old = A + ((hi1 & -hi1).bit_length() - 1) if hi1 else 0
            old_r = (n + 4 - m_old) if m_old else 0
            assert old_r == lrp_old, (T, old_r, lrp_old)
            nd["len_rp_k"][i] = A
            if lrp_old:
                nd["len_rp_K"][i] = W - lrp_old
            hi2 = self.w2 >> (A - 1)
            m_new = A + ((hi2 & -hi2).bit_length() - 1) if hi2 else 0
            new_r = (n + 4 - m_new) if m_new else 0
            if new_r:
                nd["newK"][i] = m_new
            self.lrp = new_r

    def run(self):
        for T in range(1, self.nmax + 1):
            self.T = T
            self.step(T)
        return self


def validate_twin(L, lanes=40, seed=7):
    """Step-exact equivalence vs the baseline EEABitState at n=256."""
    nmax = L.nmax_steps(256)
    rng = random.Random(seed)
    for k in range(lanes):
        x = rng.randrange(1, P_MOD)
        ref = L.EEABitState(P_MOD, x)
        fast = FastEEA(L, P_MOD, x, nmax)
        for T in range(1, nmax + 1):
            ref.step(T, check_windows=False)
            fast.T = T
            fast.step(T)
            rw1 = sum(b << j for j, b in enumerate(ref.w1))
            rw2 = sum(b << j for j, b in enumerate(ref.w2))
            assert (rw1, rw2) == (fast.w1, fast.w2), (k, T, "regs")
            assert (ref.lt, ref.lq, ref.ls, ref.lrp) == (fast.lt, fast.lq, fast.ls, fast.lrp), (
                k,
                T,
                "lens",
            )
            assert (ref.p1, ref.p2, ref.sign) == (fast.p1, fast.p2, fast.sign), (k, T, "phase")
        ref.finish_checks()
    return lanes


def _needs_arr(st):
    """Return the lane's needs as one (n_stats, NMAX) int16 array (0 = idle)."""
    return np.array([st.needs[nm] for nm in STATS], np.int16)


# ---------------------------------------------------------------- workers
_L = None


def _init():
    global _L
    if _L is None:
        _L = load_baseline()
    return _L


# --- optional C backend (fasteea.c): the same walks, native speed ---
#
# FastEEA above stays the authority.  The backend is used only after a
# per-process bit-exactness gate — full needs arrays plus the final
# register/counter state on GATE_LANES walks against the Python model —
# and a backend that loads but disagrees RAISES; it never silently
# falls back.  With no compiler, everything runs pure-Python.
#
# Layout facts shared with the C side: registers are 259-bit values as
# 5 x uint64 LSB-first (bit i = paper position i+1); needs arrays are
# stat-major (NSTATS, nmax) int16 in STATS order; SENT = -1 in base
# curves; dev sentinel -1000.  The C entry points keep all lane scratch
# on the stack, so batch calls may run from concurrent threads.

_C_NW = 5
_C_SRC = HERE / "fasteea.c"
_C_LIB = HERE / ("fasteea.dylib" if sys.platform == "darwin" else "fasteea.so")
_KSIDE_U8 = np.array([1 if KSIDE[s] else 0 for s in STATS], np.uint8)


def _c_build():
    """Compile fasteea.c if the binary is missing or stale
    (build_clib: source sha256 sidecar, atomic temp file + rename).
    Returns the library path, or None when no compiler is available.
    """
    try:
        return build_clib(_C_SRC, _C_LIB)
    except (OSError, subprocess.CalledProcessError):
        return None


def _c_load():
    """ctypes handle with argtypes bound, or None (no compiler)."""
    path = _c_build()
    if path is None:
        return None
    lib = ctypes.CDLL(str(path))
    u64p = np.ctypeslib.ndpointer(np.uint64, flags="C")
    i16p = np.ctypeslib.ndpointer(np.int16, flags="C")
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C")
    u32p = np.ctypeslib.ndpointer(np.uint32, flags="C")
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C")
    ci = ctypes.c_int
    lib.fasteea_run.argtypes = [u64p, u64p, ci, ci, i16p, i64p]
    lib.fasteea_hist.argtypes = [u64p, u64p, i32p, ci, ci, u32p]
    lib.fasteea_devs.argtypes = [u64p, u64p, i32p, ci, ci, ci, u8p, i16p, i16p, i16p, i16p, i16p]
    return lib


def _c_words(v):
    return np.frombuffer(v.to_bytes(8 * _C_NW, "little"), np.uint64).copy()


def _c_lane_init(x):
    """(Work2 words, lrp0) for input x — FastEEA.__init__'s layout:
    x_used big-endian at positions 4..W, lrp0 = bitlen(x_used).
    """
    x_used = P_MOD - x if x > P_MOD // 2 else x
    w2 = 0
    for i in range(256):
        if (x_used >> (255 - i)) & 1:
            w2 |= 1 << (3 + i)
    return _c_words(w2), x_used.bit_length()


def _c_w1_init():
    # Work1 = |t=1 at position 1, p big-endian at positions 4..W>
    v = 1
    for i in range(256):
        if (P_MOD >> (255 - i)) & 1:
            v |= 1 << (3 + i)
    return _c_words(v)


_C_W1 = _c_w1_init()


def _c_lanes_from_seeds(prefix, lo, hi):
    """Init vectors for a seed stream: lane k draws
    x = Random(f"{prefix}:{lo + k}").randrange(1, P_MOD).
    """
    L = hi - lo
    w2s = np.empty((L, _C_NW), np.uint64)
    lrps = np.empty(L, np.int32)
    for k, seed in enumerate(range(lo, hi)):
        x = random.Random(f"{prefix}:{seed}").randrange(1, P_MOD)
        w2s[k], lrps[k] = _c_lane_init(x)
    return np.ascontiguousarray(w2s.reshape(-1)), lrps


def _c_run(lib, x, nmax):
    """One lane: needs (NSTATS, nmax) int16 + final-state dict."""
    w2, lrp0 = _c_lane_init(x)
    needs = np.zeros(len(STATS) * nmax, np.int16)
    state = np.zeros(2 * _C_NW + 7, np.int64)
    rc = lib.fasteea_run(_C_W1, w2, lrp0, nmax, needs, state)
    assert rc == 0, f"model invariant broke at T={rc}"
    w1 = int.from_bytes(state[:_C_NW].astype(np.uint64).tobytes(), "little")
    w2f = int.from_bytes(state[_C_NW : 2 * _C_NW].astype(np.uint64).tobytes(), "little")
    fin = dict(
        zip(("lt", "lq", "ls", "lrp", "p1", "p2", "sign"), (int(v) for v in state[2 * _C_NW :]))
    )
    fin["w1"], fin["w2"] = w1, w2f
    return needs.reshape(len(STATS), nmax), fin


def _c_hist(lib, w2s, lrps, nmax):
    """Base-pass histograms H (NSTATS, nmax, NEEDMAX) uint32."""
    H = np.zeros(len(STATS) * nmax * NEEDMAX, np.uint32)
    rc = lib.fasteea_hist(_C_W1, w2s, lrps, len(lrps), nmax, H)
    assert rc == 0, rc
    return H.reshape(len(STATS), nmax, NEEDMAX)


def _c_devs(lib, w2s, lrps, nmax, cut, wl_base, t_base, r_baseL, r_baseR):
    """Dev-pass rows D (L, NSTATS + 3) int16, sentinel -1000."""
    L = len(lrps)
    D = np.zeros(L * (len(STATS) + 3), np.int16)
    rc = lib.fasteea_devs(
        _C_W1,
        w2s,
        lrps,
        L,
        nmax,
        cut,
        _KSIDE_U8,
        np.ascontiguousarray(wl_base, np.int16).reshape(-1),
        np.ascontiguousarray(t_base, np.int16),
        np.ascontiguousarray(r_baseL, np.int16),
        np.ascontiguousarray(r_baseR, np.int16),
        D,
    )
    assert rc == 0, rc
    return D.reshape(L, len(STATS) + 3)


_BE = 0  # 0 = untried, None = unavailable, ctypes lib = gated OK
GATE_LANES = 16


def _backend():
    global _BE
    if _BE != 0:
        return _BE
    try:
        lib = _c_load()
    except OSError:
        lib = None
    if lib is None:
        _BE = None
        return None
    L = _init()
    rng = random.Random("fasteea-gate")
    for x in [1, P_MOD - 1] + [rng.randrange(1, P_MOD) for _ in range(GATE_LANES - 2)]:
        needs_c, fin = _c_run(lib, x, NMAX)
        st = FastEEA(L, P_MOD, x, NMAX).run()
        ok = (
            np.array_equal(needs_c, _needs_arr(st))
            and fin["w1"] == st.w1
            and fin["w2"] == st.w2
            and (fin["lt"], fin["lq"], fin["ls"], fin["lrp"], fin["p1"], fin["p2"], fin["sign"])
            == (st.lt, st.lq, st.ls, st.lrp, st.p1, st.p2, st.sign)
        )
        if not ok:
            raise RuntimeError(
                "fasteea C backend diverged from the Python model -- refusing to fit"
            )
    _BE = lib
    return lib


def w_hist(args):
    """Run the base pass: per-(stat, T) need histograms over BASE_LANES."""
    lo, hi = args
    L = _init()
    be = _backend()
    if be is not None:
        w2s, lrps = _c_lanes_from_seeds("p1", lo, hi)
        return _c_hist(be, w2s, lrps, NMAX)
    H = np.zeros((len(STATS), NMAX, NEEDMAX), np.uint32)
    for seed in range(lo, hi):
        x = random.Random(f"p1:{seed}").randrange(1, P_MOD)
        A = _needs_arr(FastEEA(L, P_MOD, x, NMAX).run())
        si, T = np.nonzero(A)
        np.add.at(H, (si, T, A[si, T]), 1)
    return H


_BASES = None  # (wl_base, tcore_base, rcore_baseL, rcore_baseR)


def w_devs(args):
    """Dev pass: per-lane worst deviations beyond every base curve —
    the 10 window stats (side per KSIDE), the t-core low side, and the
    r-core L-high / R-low sides — in one model run per lane.
    """
    lo, hi = args
    L = _init()
    assert _BASES is not None, "main sets _BASES before the dev pass"
    wl_base, t_base, r_baseL, r_baseR = _BASES
    be = _backend()
    if be is not None:
        w2s, lrps = _c_lanes_from_seeds("p2", lo, hi)
        return _c_devs(be, w2s, lrps, NMAX, CUT, wl_base, t_base, r_baseL, r_baseR)
    D = np.full((hi - lo, len(STATS) + 3), -1000, np.int16)
    for k, seed in enumerate(range(lo, hi)):
        x = random.Random(f"p2:{seed}").randrange(1, P_MOD)
        st = FastEEA(L, P_MOD, x, NMAX).run()
        for si, s in enumerate(STATS):
            arr = st.needs[s]
            b = wl_base[si]
            best = -1000
            if KSIDE[s]:
                for T in range(CUT):
                    v = arr[T]
                    if v and b[T] != SENT and b[T] - v > best:
                        best = b[T] - v
            else:
                for T in range(CUT):
                    v = arr[T]
                    if v and b[T] != SENT and v - b[T] > best:
                        best = v - b[T]
            D[k, si] = best
        arr = st.needs["t_K"]
        best = -1000
        for T in range(CUT):
            v = arr[T]
            if v and t_base[T] >= 0 and t_base[T] - v > best:
                best = t_base[T] - v
        D[k, len(STATS)] = best
        aL, aR = st.needs["r_k"], st.needs["r_K"]
        bl, br = -1000, -1000
        for T in range(CUT):
            v = aL[T]
            if v and r_baseL[T] >= 0 and v - r_baseL[T] > bl:
                bl = v - r_baseL[T]
            v = aR[T]
            if v and r_baseR[T] >= 0 and r_baseR[T] - v > br:
                br = r_baseR[T] - v
        D[k, len(STATS) + 1] = bl
        D[k, len(STATS) + 2] = br
    return D


def _pool_init(bases):
    global _BASES
    _BASES = bases


# ------------------------------------------------------------- reductions
def margin_of(devs, eps):
    """max(empirical quantile, exponential tail fit), clamped [0, 48]."""
    d = np.sort(devs.astype(float))
    n = len(d)
    m_emp = int(d[min(n - 1, int(np.ceil((1 - eps) * n)))])
    vals, cnts = np.unique(d[d > d[int(0.9 * n)] - 1], return_counts=True)
    exc = np.cumsum(cnts[::-1])[::-1] / n
    ok = exc > 0
    m_fit = m_emp
    if ok.sum() >= 3:
        A = np.polyfit(vals[ok], np.log(exc[ok]), 1)
        if A[0] < -1e-6:
            m_fit = int(np.ceil((np.log(eps) - A[1]) / A[0]))
    return int(np.clip(max(m_emp, m_fit), 0, 48)), m_emp, m_fit, int(d[-1])


def quantile_from_hist(c, delta, side):
    """Index of the delta-quantile (side='lo') or (1-delta) (side='hi')."""
    cum = np.cumsum(c)
    q = delta if side == "lo" else (1 - delta)
    return int(np.searchsorted(cum, q * int(c.sum())))


def generate():
    """Run every pass and return the canonical constants.json text."""
    L = _init()
    own = load_module(OWN, "_luo_lowwidth_own")
    assert NMAX == L.nmax_steps(256) == own._nmax_steps_wc(256), "fit horizon != worst-case budget"
    print(f"twin validation: {validate_twin(L)} lanes step-exact vs EEABitState", flush=True)
    # Resolve the model engine in the parent: builds the C backend
    # once (atomic) so pool workers only load, and runs its gate.
    eng = "C (fasteea.c, gated)" if _backend() is not None else "pure Python"
    print(f"model engine: {eng}", flush=True)

    nproc = min(8, os.cpu_count() or 8)

    def chunks(total, size=2500):
        return [(i, min(i + size, total)) for i in range(0, total, size)]

    # ---- pass 1: base histograms + quantile curves ----------------------
    # (one chunk per worker: each returns a 17 MB histogram, so fewer,
    # larger chunks cut the memory; sums are order-independent)
    print(f"pass 1: base histograms ({BASE_LANES:,} lanes) ...", flush=True)
    with ThreadPool(nproc) as p:
        parts = p.map(w_hist, chunks(BASE_LANES, -(-BASE_LANES // nproc)))
    H = parts[0]
    for pt in parts[1:]:
        H += pt

    wl_base = np.full((len(STATS), NMAX), SENT, np.int16)
    for si, s in enumerate(STATS):
        for T in range(NMAX):
            c = H[si, T]
            if int(c.sum()) < MINFIRE:
                continue
            wl_base[si, T] = quantile_from_hist(c, BASE_DELTA, "lo" if KSIDE[s] else "hi")
    t_base = np.full(NMAX, -1, np.int16)
    ti = STATS.index("t_K")
    for T in range(NMAX):
        c = H[ti, T]
        if int(c.sum()) >= MINFIRE:
            t_base[T] = quantile_from_hist(c, BASE_DELTA, "lo")
    r_baseL = np.full(NMAX, -1, np.int16)
    r_baseR = np.full(NMAX, -1, np.int16)
    li, ri = STATS.index("r_k"), STATS.index("r_K")
    for T in range(NMAX):
        cL, cR = H[li, T], H[ri, T]
        if int(cL.sum()) >= MINFIRE:
            r_baseL[T] = quantile_from_hist(cL, BASE_DELTA, "hi")
        if int(cR.sum()) >= MINFIRE:
            r_baseR[T] = quantile_from_hist(cR, BASE_DELTA, "lo")
    print(
        f"  window rows fitted (worst fallback/stat over cut): "
        f"{int((wl_base[:, :CUT] == SENT).sum(1).max())}; "
        f"t rows {(t_base >= 0).sum()}, "
        f"r rows L {(r_baseL >= 0).sum()} / R {(r_baseR >= 0).sum()}",
        flush=True,
    )

    # NMAX_SCHED sanity: lanes still walking past the circuit's schedule
    # cut (1432). Max per-(stat, T) fire count over the tail == lanes
    # alive at the earliest tail step; the cut is priced at
    # P(shot needs more) ~ 1.3e-5 (a shot runs ~2 lanes).
    alive = int(H[:, 1432:CUT, :].sum(axis=2).max())
    print(
        f"  schedule-cut check: {alive}/{BASE_LANES:,} base lanes "
        f"alive past T=1432 (P_lane ~ {alive / BASE_LANES:.1e})",
        flush=True,
    )

    # ---- pass 2: dev tails ----------------------------------------------
    print(f"pass 2: dev tails ({FIT_LANES:,} lanes) ...", flush=True)
    bases = (wl_base, t_base, r_baseL, r_baseR)
    with ThreadPool(nproc, initializer=_pool_init, initargs=(bases,)) as p:
        parts = p.map(w_devs, chunks(FIT_LANES))
    D = np.concatenate(parts)

    # windows margins: EPS_WL split across stats by Toffoli sensitivity
    wts = np.array([SENS[s] for s in STATS], float)
    eps_s = EPS_WL * wts / wts.sum()
    wl_margins = {}
    for si, s in enumerate(STATS):
        m, m_emp, m_fit, mx = margin_of(D[:, si], eps_s[si])
        wl_margins[s] = {
            "m": m,
            "m_emp": m_emp,
            "m_fit": m_fit,
            "max_obs": mx,
            "eps": float(eps_s[si]),
        }
    mT, eT, fT, xT = margin_of(D[:, len(STATS)], EPS_CORE)
    mL, _eL, _fL, _xL = margin_of(D[:, len(STATS) + 1], EPS_CORE)
    mR, _eR, _fR, _xR = margin_of(D[:, len(STATS) + 2], EPS_CORE)
    print(
        f"  margins: windows "
        f"{ {s: v['m'] for s, v in wl_margins.items()} }; "
        f"t-core {mT} (emp {eT} fit {fT} max {xT}); "
        f"r-core L {mL} / R {mR}",
        flush=True,
    )

    # ---- tables -----------------------------------------------------------
    b_of = {s: wl_base[STATS.index(s)] for s in STATS}
    win = {nm: [] for nm in NAMES}
    newK = []
    for T in range(1, NMAX + 1):
        wc = dict(own._active_windows_wc(256, T))
        i = T - 1
        for nm in NAMES:
            wck, wcK = wc[nm]
            k, K = wck, wcK
            if T <= CUT:
                ks = nm + "_k"
                if ks in wl_margins and b_of[ks][i] != SENT:
                    k = int(np.clip(b_of[ks][i] - wl_margins[ks]["m"], wck, wcK))
                Ks = nm + "_K"
                if Ks in wl_margins and b_of[Ks][i] != SENT:
                    K = int(np.clip(b_of[Ks][i] + wl_margins[Ks]["m"], wck, wcK))
                if k > K:
                    k, K = wck, wcK
            win[nm].append([int(k), int(K)])
        if T <= CUT and b_of["newK"][i] != SENT:
            newK.append(int(np.clip(b_of["newK"][i] + wl_margins["newK"]["m"], 1, 259)))
        else:
            newK.append(259)

    rlo = np.zeros(NMAX, np.int16)
    for T in range(CUT):
        if t_base[T] >= 0:
            rlo[T] = max(0, t_base[T] - mT)
    lmax = np.zeros(NMAX, np.int16)
    rmin = np.zeros(NMAX, np.int16)
    for T in range(CUT):
        if r_baseL[T] >= 0:
            lmax[T] = r_baseL[T] + mL
        if r_baseR[T] >= 0:
            rmin[T] = max(0, r_baseR[T] - mR)

    # ---- emit: only the keys circuit.py reads ------------------------------
    outs = {}
    outs["wl_windows"] = {"nmax": NMAX, "nmax_cut": CUT, "windows": win, "len_rp_new_K": newK}
    outs["tcore"] = {"nmax_cut": CUT, "rlo": [int(v) for v in rlo]}
    outs["rcore"] = {
        "nmax_cut": CUT,
        "lmax": [int(v) for v in lmax],
        "rmin": [int(v) for v in rmin],
    }

    return json.dumps(outs)


def main():
    new = generate()
    path = HERE / "constants.json"
    old = path.read_text() if path.exists() else None
    path.write_text(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
