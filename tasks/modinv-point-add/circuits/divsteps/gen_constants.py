#!/usr/bin/env python3
"""Regenerate constants.json for the divstep family.

Running this file reruns every search and census and writes
constants.json.  It has no flags.  constants.json holds only the tables the engine and
circuit.py read.  Every census runs on the C sampler window_census.c
(xoshiro256** with fixed seeds), so a rerun is deterministic; the
sampler is gated per process on bit-exact agreement with the Python
trajectory models below, and without a compiler the same draw streams
run on those models (bit-identical output, much slower).

Sections (consumer in brackets):

  j34_forms  DERIVED [engine.py FORMS_P].  The 16 per-stratum decode
             forms follow from the anchored-ladder theorem, form 2: at
             a class-3 post-state with j = v2(fv'), q = fv' >> j,
             b = bit_{j-1}(p * (f' q)^{-1} mod 2^j).  For j in {3, 4}
             b is affine in the anchored window bits within each
             (j, f' mod 16) stratum; the script solves each stratum.
             A stratum whose solve carries constant 1 is spelled
             through ag4[0] = gv' bit j-1, which the determinant pin
             forces to 1 on exactly those strata (asserted).
  tstop      DERIVED [engine.py].  The widths census also counts
             the stopping time tstop of every walk over the full
             SMAX = 590-step schedule (every walk stops by then).  S is
             the shortest schedule with P(tstop > S) <= 1e-5/walk.
             Tmin is the longest untested prefix with
             P(tstop <= Tmin) <= 3e-6/walk.
  windows_q  SEARCHED [circuit.py, engine.py].  Per-step row-width
             tail histograms over 1e8 walks; the tables are a
             deterministic greedy quantile at a 2.85e-5/walk any-step
             budget plus one backward emission-consistency sweep.
  spans_q    SEARCHED [circuit.py, hmr_main, staged j = 5].  Per-step entry-fill
             tails over 1e8 walks at store threshold v2(fv') >= 5; the
             reach is a Laplace-smoothed greedy quantile at 1e-6/walk,
             clipped at the record cap 60.  The cap 60 is hand-set,
             not searched.
  z          DERIVED [engine.py].  The j = 5 fill census also records
             each walk's widest z counter.  Wz is the narrowest z
             register with P(z needs more bits) <= 1e-6/walk.
  spans_q_j<j>, j = 6..9
             SEARCHED [staged_main, j<j>.py].  The same census and
             greedy at store threshold v2(fv') >= j.  A cap sweep
             chooses the record cap: the first cap C, stepping by 2
             from a hand-set start (CAP_START), whose flat table
             [C-1]*S escapes at <= 2.5e-6/walk.
  qcap       SEARCHED [circuit.py, wide.py, j5.py, j6.py, j8.py,
             j9.py].  Each of these points runs at the lowest arena
             ceiling its build attains: the smallest ceiling q whose
             build peaks at or below q (a bisection over full builds).

The rates above are this circuit's allotments of the task's failure
budget (20 failed tests per 100,000).  The censuses turn each
allotment into a table or a bound.

RUN: uv run python gen_constants.py   (about 30 minutes on 8
     cores for the censuses, plus about 15 minutes for the qcap
     bisections)
"""

import ctypes
import heapq
import json
import multiprocessing
import pathlib
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

from uncompute.testing import build_clib, load_module

HERE = pathlib.Path(__file__).resolve().parent

P_MOD = (1 << 256) - (1 << 32) - 977
SCHED_RATE = 1e-5  # schedule overrun budget per walk: P(tstop > S)
EARLY_RATE = 3e-6  # untested-prefix budget per walk: P(tstop <= Tmin)
Z_RATE = 1e-6  # z-counter overflow budget per walk
QW_R_TOTAL = 2.85e-5  # windows_q any-step width-escape budget per walk
SPQ_R_TOTAL = 1e-6  # spans_q* any-step fill-escape budget per walk
CAP_RATE = 2.5e-6  # cap-sweep target escape rate per walk
R_CAP_COMPOSED = 60  # record cap at store threshold j = 5 (hand-set)
LADDER = (6, 7, 8, 9)  # deployed deeper store thresholds (j<j>.py)
CAP_START = {6: 36, 7: 22, 8: 14, 9: 20}  # cap-sweep starts (hand-set)

N_CENSUS = 100_000_000
N_CAP = 40_000_000  # walks per cap-sweep candidate
NTHREADS = 14


def seeds(base):
    return [base + i for i in range(NTHREADS)]


# ------------------------------------------------------------- j34 forms


def _affine_solve3(cells):
    """Solve b = c XOR sum a_i q_i over GF(2) for cells {q(3b): b}.
    Asserts the stratum is affine (the theorem guarantees it).
    """
    c = cells[0]
    coef = [cells[1 << i] ^ c for i in range(3)]
    for qh in range(8):
        want = c
        for i in range(3):
            if (qh >> i) & 1 and coef[i]:
                want ^= 1
        assert cells[qh] == want, "stratum not affine"
    return c, [f"af4[{i + 1}]" for i in range(3) if coef[i]]


def derive_j34_forms(p):
    """Derive the 16 (j, f3) stratum forms from ladder-theorem form 2 + the
    determinant pin. Window naming: ag4[i] = gv' bit j-1+i, af4[i] =
    fv' bit j+i (af4[0] = 1 always: q is odd).
    """
    forms = {}
    for j in (3, 4):
        M = 1 << j
        for f3 in range(8):
            fp = (1 + 2 * f3) % M  # f' mod 2^j (f' odd)
            cells = {}
            for qh in range(8):  # q bits 1..3 = af4[1..3]
                q = 1 + 2 * qh
                cells[qh] = ((p * pow((fp * (q % M)) % M, -1, M)) % M) >> (j - 1) & 1
            const, terms = _affine_solve3(cells)
            if const:
                # pin spelling: gv' bit j-1 = bit_{j-1}(p f'^{-1}) = 1 here
                ag0 = ((p * pow(fp, -1, M)) % M) >> (j - 1) & 1
                assert ag0 == 1, (j, f3)
                terms = ["ag4[0]"] + terms
                const = 0
            forms[f"j{j}_f3={f3}"] = {"const": const, "terms": terms}
    return forms


# ------------------------------------------------------------- census sampler

SMAX = 590
ROWS = 4
WBINS = 320
FILL_MAX = 96
ZW_MAX = 24

_SRC = HERE / "window_census.c"
_LIBNAME = "window_census.dylib" if sys.platform == "darwin" else "window_census.so"
_LIB = HERE / _LIBNAME


def _build():
    """Compile window_census.c if the binary is missing or stale
    (build_clib: source sha256 sidecar, atomic temp file + rename).
    Returns the library path, or None when no compiler is available.
    """
    try:
        return build_clib(_SRC, _LIB, cflags=["-O3"])
    except (OSError, subprocess.CalledProcessError):
        return None


_lib = None


def _load():
    """ctypes handle with restypes bound, or None (no compiler)."""
    global _lib
    if _lib is not None:
        return _lib
    path = _build()
    if path is None:
        return None
    lib = ctypes.CDLL(str(path))
    lib.run_fill_holdout_j.restype = ctypes.c_uint64
    _lib = lib
    return lib


def plimbs(v, n=4):
    return (ctypes.c_uint64 * n)(*[(v >> (64 * i)) & ((1 << 64) - 1) for i in range(n)])


# ===========================================================================
# Python trajectory models (normative; the C side mirrors these)
# ===========================================================================


def py_walk_widths(p, x, S):
    """Per-step |value| bit-length rows of the hd walk on (p, x)
    (post-step values; a frozen lane holds F=1, G=0 and its stop
    fv/gv widths).  Returns (tstop, [S][4]); tstop = S + 1 when the
    walk does not freeze inside the schedule.
    """
    z, f, g, fv, gv = 0, p, x, 0, 1
    w = [[0, 0, 0, 0] for _ in range(S)]
    for t in range(S):
        s = 1 if (z >= 0 and (g & 1)) else 0
        if s:
            z = -z - 1
            f, g = g, f
            fv, gv = gv, fv
        if g & 1:
            if s:
                g, gv = f - g, fv - gv
            else:
                g, gv = g + f, gv + fv
        z += 1
        g >>= 1
        fv <<= 1
        w[t] = [
            abs(f).bit_length(),
            abs(g).bit_length(),
            abs(fv).bit_length(),
            abs(gv).bit_length(),
        ]
        if g == 0:
            bfv, bgv = w[t][2], w[t][3]
            for tt in range(t + 1, S):
                w[tt] = [1, 0, bfv, bgv]
            return t + 1, w
    return S + 1, w


def py_fill_walk_j(p, x, S, jmin):
    """Per-step R-stack ENTRY fill of the hd walk on (p, x)
    (frozen-inclusive, freeze-push-suppressed, push on class-3 steps
    with v2(fv') >= jmin) plus the per-walk max z width
    (post-increment two's-complement values).  Returns (zw, [S]).
    """
    z, f, g, fv, gv = 0, p, x, 0, 1
    fills = [0] * S
    R, zw = 0, 1
    for t in range(S):
        fills[t] = R
        if g == 0:
            continue
        s = 1 if (z >= 0 and (g & 1)) else 0
        if s:
            z = -z - 1
            f, g = g, f
            fv, gv = gv, fv
        if g & 1:
            if s:
                g, gv = f - g, fv - gv
            else:
                g, gv = g + f, gv + fv
        z += 1
        m = z if z >= 0 else -z - 1
        zw = max(zw, m.bit_length() + 1)
        g >>= 1
        fv <<= 1
        if s == 0 and z <= 0 and g != 0 and ((fv & -fv).bit_length() - 1) >= jmin:
            R += 1
    return zw, fills


# ===========================================================================
# the draw stream (xoshiro256**, splitmix64 seed expansion) — the
# pure-Python twin of the C sampler's generator
# ===========================================================================

_M64 = (1 << 64) - 1


class Xoshiro:
    def __init__(self, seed):
        self.s = []
        z = seed & _M64
        for _ in range(4):
            z = (z + 0x9E3779B97F4A7C15) & _M64
            w = z
            w = ((w ^ (w >> 30)) * 0xBF58476D1CE4E5B9) & _M64
            w = ((w ^ (w >> 27)) * 0x94D9736C719C2DC5) & _M64
            self.s.append(w ^ (w >> 31))

    def next(self):
        s = self.s
        res = ((((s[1] * 5) & _M64) << 7 | ((s[1] * 5) & _M64) >> 57) * 9) & _M64
        t = (s[1] << 17) & _M64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = ((s[3] << 45) | (s[3] >> 19)) & _M64
        return res

    def draw_x(self, p):
        """Rejection sample into [1, p) from four 64-bit limbs."""
        while True:
            x = 0
            for i in range(4):
                x |= self.next() << (64 * i)
            if 0 < x < p:
                return x


# ===========================================================================
# census entry points: C backend when available (validated), else the
# pure-Python samplers on the same draw stream
# ===========================================================================

_validated = False


def _backend():
    """Return the validated C library, or None (pure-Python fallback)."""
    global _validated
    lib = _load()
    if lib is not None and not _validated:
        validate(lib)
        _validated = True
    return lib


def validate(lib, n=400, seed=20260807, p=None):
    """Run the per-process backend gate: n fresh trajectories through the
    C walk_widths and fill_walk_trace_j, compared bit-exactly against
    the Python models (widths, tstop, fills, z width).
    """
    if p is None:
        p = (1 << 256) - (1 << 32) - 977
    rng = random.Random(f"window-census-gate:{seed}")
    wout = (ctypes.c_uint16 * (SMAX * ROWS))()
    fout = (ctypes.c_uint16 * SMAX)()
    pl = plimbs(p)
    for _ in range(n):
        x = rng.randrange(1, p)
        S = SMAX
        ts = lib.walk_widths(pl, plimbs(x), S, wout)
        ts_py, w_py = py_walk_widths(p, x, S)
        assert ts == ts_py, ("census backend: tstop mismatch", x)
        for t in range(S):
            for r in range(ROWS):
                assert wout[t * ROWS + r] == w_py[t][r], ("census backend: width mismatch", x, t, r)
        for jmin in (5, 6, 7):
            zw = lib.fill_walk_trace_j(pl, plimbs(x), S, fout, jmin)
            zw_py, fills_py = py_fill_walk_j(p, x, S, jmin)
            assert zw == zw_py, ("census backend: z-width mismatch (j)", x, jmin)
            for t in range(S):
                assert fout[t] == fills_py[t], ("census backend: fill mismatch (j)", x, jmin, t)
    return n


def census_widths_c(p, seeds, per, S):
    """Width histograms [S][4][320] + tstop histogram [S+2], summed
    over one xoshiro stream of `per` walks per seed (threaded on the
    C backend; the pure-Python fallback runs the same streams).
    """
    lib = _backend()
    if lib is not None:
        hists = [(ctypes.c_uint64 * (S * ROWS * WBINS))() for _ in seeds]
        ths = [(ctypes.c_uint64 * (S + 2))() for _ in seeds]
        pl = plimbs(p)

        def worker(i):
            lib.run_census(pl, seeds[i], per, S, hists[i], ths[i])

        tt = [threading.Thread(target=worker, args=(i,)) for i in range(len(seeds))]
        for x in tt:
            x.start()
        for x in tt:
            x.join()
        hist = [
            [
                [sum(h[(t * ROWS + r) * WBINS + w] for h in hists) for w in range(WBINS)]
                for r in range(ROWS)
            ]
            for t in range(S)
        ]
        tstop = [sum(h[k] for h in ths) for k in range(S + 2)]
        return hist, tstop
    hist = [[[0] * WBINS for _ in range(ROWS)] for _ in range(S)]
    tstop = [0] * (S + 2)
    for seed in seeds:
        rng = Xoshiro(seed)
        for _ in range(per):
            ts, w = py_walk_widths(p, rng.draw_x(p), S)
            tstop[min(ts, S + 1)] += 1
            for t in range(S):
                for r in range(ROWS):
                    hist[t][r][w[t][r]] += 1
    return hist, tstop


def census_fill_j(p, seeds, per, S, jmin):
    """Entry-fill histograms [S][96] + per-walk max z-width histogram
    [24] at store threshold jmin, summed over one stream of `per`
    walks per seed.
    """
    lib = _backend()
    if lib is not None:
        hists = [(ctypes.c_uint64 * (S * FILL_MAX))() for _ in seeds]
        zws = [(ctypes.c_uint64 * ZW_MAX)() for _ in seeds]
        pl = plimbs(p)

        def worker(i):
            lib.run_fill_census_j(pl, seeds[i], per, S, hists[i], zws[i], jmin)

        tt = [threading.Thread(target=worker, args=(i,)) for i in range(len(seeds))]
        for x in tt:
            x.start()
        for x in tt:
            x.join()
        hist = [
            [sum(h[t * FILL_MAX + w] for h in hists) for w in range(FILL_MAX)] for t in range(S)
        ]
        zw = [sum(h[k] for h in zws) for k in range(ZW_MAX)]
        return hist, zw
    hist = [[0] * FILL_MAX for _ in range(S)]
    zw = [0] * ZW_MAX
    for seed in seeds:
        rng = Xoshiro(seed)
        for _ in range(per):
            z, fills = py_fill_walk_j(p, rng.draw_x(p), S, jmin)
            zw[min(z, ZW_MAX - 1)] += 1
            for t in range(S):
                hist[t][min(fills[t], FILL_MAX - 1)] += 1
    return hist, zw


def holdout_fill_j(p, seeds, per, S, reach, ref, jmin):
    """INCREMENTAL any-step entry-fill escapes at store threshold
    jmin: exceed the candidate reach without exceeding the reference
    table (walks past the reference already fail through the
    pre-existing channel).
    """
    lib = _backend()
    if lib is not None:
        tabc = (ctypes.c_uint16 * S)(*reach)
        refc = (ctypes.c_uint16 * S)(*ref)
        pl = plimbs(p)
        res = [0] * len(seeds)

        def worker(i):
            res[i] = lib.run_fill_holdout_j(pl, seeds[i], per, S, tabc, refc, jmin)

        tt = [threading.Thread(target=worker, args=(i,)) for i in range(len(seeds))]
        for x in tt:
            x.start()
        for x in tt:
            x.join()
        return int(sum(res))
    exceeds = 0
    for seed in seeds:
        rng = Xoshiro(seed)
        for _ in range(per):
            _, fills = py_fill_walk_j(p, rng.draw_x(p), S, jmin)
            new_esc = any(fills[t] > reach[t] for t in range(S))
            old_esc = any(fills[t] > ref[t] for t in range(S))
            if new_esc and not old_esc:
                exceeds += 1
    return exceeds


# ------------------------------------------------------------- censuses


def _tails(row, nbins):
    """Return the top 32 bins of one histogram row: the greedy's input."""
    mx = max((w for w in range(nbins) if row[w]), default=0)
    lo = max(0, mx - 31)
    return {"lo": lo, "counts": row[lo : mx + 1]}


def census_widths():
    """Run the widths census over N_CENSUS walks and the full SMAX-step
    schedule.  Returns (n, hist, tstop): the per-step per-row
    |value| bit-length histograms and the stopping-time histogram.
    """
    per = N_CENSUS // NTHREADS
    hist, tstop = census_widths_c(P_MOD, seeds(900_000), per, SMAX)
    return per * NTHREADS, hist, tstop


def width_tails(hist, S):
    """Return the greedy's input: the tails of the first S steps.  A
    step's widths do not depend on the schedule length, so these
    equal the tails of a census run over S steps.
    """
    return {(t, r): _tails(hist[t][r], WBINS) for t in range(S) for r in range(4)}


def derive_schedule(n, tstop):
    """Return (S, Tmin) from the stopping-time histogram.  S is the
    shortest schedule whose overrun rate P(tstop > S) is at most
    SCHED_RATE.  Tmin is the longest prefix of untested steps whose
    early-stop rate P(tstop <= Tmin) is at most EARLY_RATE: a walk
    that stops inside the prefix fails.
    """
    assert tstop[SMAX + 1] == 0, "a walk ran past the 590-step bound"
    S, late = SMAX, 0  # late = walks with tstop > S
    while late + tstop[S] <= SCHED_RATE * n:
        late += tstop[S]
        S -= 1
    tmin, early = 0, tstop[0]  # early = walks with tstop <= tmin
    while early + tstop[tmin + 1] <= EARLY_RATE * n:
        early += tstop[tmin + 1]
        tmin += 1
    return S, tmin


def census_fill(jmin, base, S):
    """Run the entry-fill census over N_CENSUS walks at store threshold
    v2(fv') >= jmin.  Returns (n, per-step fill tails, z-width
    histogram).
    """
    per = N_CENSUS // NTHREADS
    hist, zw = census_fill_j(P_MOD, seeds(base), per, S, jmin)
    return per * NTHREADS, {t: _tails(hist[t], FILL_MAX) for t in range(S)}, zw


def derive_wz(n, zw):
    """Return the narrowest z register whose overflow rate (walks whose
    z counter needs more bits) is at most Z_RATE.
    """
    w = 1
    while sum(zw[w + 1 :]) > Z_RATE * n:
        w += 1
    return w


def cap_sweep(jmin, S):
    """Return the first record cap C (stepping by 2 from CAP_START) whose
    flat table [C-1]*S escapes at <= CAP_RATE per walk.
    """
    per = N_CAP // NTHREADS
    ref = [FILL_MAX - 1] * S
    for cap in range(CAP_START[jmin], 61, 2):
        ex = holdout_fill_j(P_MOD, seeds(944_000), per, S, [cap - 1] * S, ref, jmin)
        if ex / (per * NTHREADS) <= CAP_RATE:
            return cap
    raise AssertionError(f"no cap under {CAP_RATE:g} for j = {jmin}")


# ------------------------------------------------------------- greedies


def derive_width_tables(n, tails):
    """Deterministic greedy: repeatedly lower the site whose next
    reduction costs the least exceed rate (ties by (rate, t, r))
    until the per-walk any-step budget is spent; then one backward
    emission-consistency sweep (cell <= prior-pair-max + 1).
    """
    S = len(tails) // 4

    def exceed(t, r, W):
        rec = tails[t, r]
        lo, cnts = rec["lo"], rec["counts"]
        if W < lo:
            return float("inf")
        return sum(c for i, c in enumerate(cnts) if lo + i > W)

    tab = [[0] * 4 for _ in range(S)]
    heap = []
    for t in range(S):
        for r in range(4):
            rec = tails[t, r]
            tab[t][r] = rec["lo"] + max((i for i, c in enumerate(rec["counts"]) if c), default=0)
            heapq.heappush(heap, (exceed(t, r, tab[t][r] - 1) / n, t, r))
    budget = QW_R_TOTAL
    while heap:
        d, t, r = heapq.heappop(heap)
        cur = exceed(t, r, tab[t][r] - 1) / n
        if cur > d + 1e-18:
            heapq.heappush(heap, (cur, t, r))
            continue
        if cur > budget:
            break
        tab[t][r] -= 1
        budget -= cur
        heapq.heappush(heap, (exceed(t, r, tab[t][r] - 1) / n, t, r))
    for t in range(S - 1, 0, -1):
        for a, b in ((0, 1), (2, 3)):
            need = max(tab[t][a], tab[t][b]) - 1
            if max(tab[t - 1][a], tab[t - 1][b]) < need:
                big = a if tab[t - 1][a] >= tab[t - 1][b] else b
                tab[t - 1][big] = need
    for r in (2, 3):
        tab[0][r] = max(tab[0][r], 1)
    return {k: [tab[t][r] for t in range(S)] for r, k in enumerate(("WF", "WG", "WFV", "WGV"))}


def derive_reach(n, tails, clip):
    """Laplace-smoothed greedy on the fill tails: start every step at
    census max + 1 and charge each reduction its in-sample exceed
    count PLUS one add-one unit whenever the cell drops to or below
    the census max.  The smoothing prices the unseen-tail union across
    the S sites that a bare in-sample estimate prices at zero.
    """
    S = len(tails)
    mxs = {
        t: tails[t]["lo"] + max((i for i, c in enumerate(tails[t]["counts"]) if c), default=0)
        for t in range(S)
    }

    def cost(t, W):
        lo, cnts = tails[t]["lo"], tails[t]["counts"]
        if W < lo - 1:
            return float("inf")
        raw = sum(c for i, c in enumerate(cnts) if lo + i > W)
        return (raw + (1 if W <= mxs[t] else 0)) / n

    tab = [min(mxs[t] + 1, clip, t) for t in range(S)]
    heap = [(cost(t, tab[t] - 1), t) for t in range(S)]
    heapq.heapify(heap)
    budget = SPQ_R_TOTAL
    while heap:
        d, t = heapq.heappop(heap)
        if tab[t] <= 0:
            continue
        cur = cost(t, tab[t] - 1)
        if cur > d + 1e-18:
            heapq.heappush(heap, (cur, t))
            continue
        if cur > budget:
            break
        tab[t] -= 1
        budget -= cur
        heapq.heappush(heap, (cost(t, tab[t] - 1), t))
    return tab


# ------------------------------------------------------------- generate


def _step(name, fn, *args):
    t0 = time.time()
    out = fn(*args)
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    return out


def generate():
    """Run every census and table search; return the table sections.
    main() adds the qcap section, whose builds read these tables.
    """
    n, hist, tstop = _step("widths census", census_widths)
    S, tmin = derive_schedule(n, tstop)
    out = {
        "tstop": {"S": S, "Tmin": tmin},
        "j34_forms": {"forms": derive_j34_forms(P_MOD)},
        "windows_q": derive_width_tables(n, width_tails(hist, S)),
    }
    n, tails, zw = _step("fill census j5", census_fill, 5, 910_000, S)
    out["spans_q"] = {"reach": derive_reach(n, tails, R_CAP_COMPOSED - 1)}
    out["z"] = {"Wz": derive_wz(n, zw)}
    for j in LADDER:
        n, tails, _ = _step(f"fill census j{j}", census_fill, j, 930_000, S)
        cap = _step(f"cap sweep j{j}", cap_sweep, j, S)
        out[f"spans_q_j{j}"] = {"cap": cap, "reach": derive_reach(n, tails, cap - 1)}
    return out


# ------------------------------------------------------------- qcap

# The points that run at their floor, with the engine dials that
# define each one.  circuit.py spells the Hmr engine's build flat, so
# its floor is the Hmr engine's floor without the wide variant.
FLOOR_POINTS = {
    "circuit": {"wide": False},
    "wide": {"wide": True},
    "j5": {"jstore": 5},
    "j6": {"jstore": 6},
    "j8": {"jstore": 8},
    "j9": {"jstore": 9},
}


def attains(name, q):
    """Build point `name` at arena ceiling q and return True when the
    build peaks at or below q.  A build that runs out of room at q
    (a ceiling assertion) does not attain q.
    """
    E = load_module(HERE / "engine.py", f"qcap_{name}_{q}")
    cfg = dict(FLOOR_POINTS[name], qcap=q)
    try:
        if "jstore" in cfg:
            E.staged_configure(cfg)
            b, _ = E.build_point(E.staged_build_circuit, E.P_MOD, E.N)
        else:
            E.hmr_configure(cfg)
            b, _ = E.build_point(E.hmr_build_circuit, E.P_MOD, E.N)
    except AssertionError as e:
        if "ceiling too low" in str(e) or "pool empty" in str(e):
            return False
        raise
    return b.peak_qubits <= q


def qcap_floor(name, lo=None, hi=None):
    """Return the lowest arena ceiling point `name` attains.  Below
    that floor a build still peaks at the floor, and at or above it
    the build peaks at the ceiling, so attainment grows with q and a
    bisection finds the floor.  The default bracket starts at
    MUL_BASE, where the multiplier pool is empty, and ends 2n wires
    higher.
    """
    E = load_module(HERE / "engine.py", "qcap_bracket")
    lo = E.MUL_BASE if lo is None else lo
    hi = E.MUL_BASE + 2 * E.N if hi is None else hi
    assert attains(name, hi), (name, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if attains(name, mid):
            hi = mid
        else:
            lo = mid
    return hi


def search_qcaps():
    """Run the qcap bisections, one process per point."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(len(FLOOR_POINTS), mp_context=ctx) as pool:
        return dict(zip(FLOOR_POINTS, pool.map(qcap_floor, FLOOR_POINTS)))


def _write(path, sections):
    blob = (json.dumps(sections, indent=1) + "\n").encode()
    path.write_bytes(blob)
    return blob


def main():
    path = HERE / "constants.json"
    old = path.read_bytes() if path.exists() else None
    sections = generate()
    _write(path, sections)
    sections["qcap"] = _step("qcap search", search_qcaps)
    blob = _write(path, sections)
    print(f"constants.json {'unchanged' if blob == old else 'WRITTEN'}")


if __name__ == "__main__":
    main()
