#!/usr/bin/env python3
"""Regenerate constants.json for the gaussian discovered circuit by running the
config search.

The candidate space is the full grid SPACE of knob tuples
(M, CAPW, A0SH, EQ, RB, I0, CLAMP, FLOW, MSKIP); the comment at SPACE
gives each range.  The objective is the TQ-cost, executed Toffoli x peak
qubits, subject to an exact worst-case error <= EXACT_GATE.  The rule
is: rank every candidate that passes the sampled gate by TQ-cost, then
Toffoli count, then grid order, and accept the first one in that order
whose exact scan passes.  The accepted tuple is therefore the cheapest
grid point that the exact scan certifies, among the sampled-gate passers.

The search has three stages.

1. Table sweep (build_tables).  For each knob tuple in SPACE, the
   sweep fits a
   per-segment quadratic model of 2^32 exp(-y^2) on 129 uniform nodes
   (least squares).  It quantizes the negated slope d to EQ-bit cells.
   A dither search over five dq offsets and four FLOW-aligned B1
   offsets picks the best of 20 variants per segment.  constify()
   picks the tail split s0 and the constant runs.  The sweep scores the
   sampled sup-error on the certificate set Dcert: cell-boundary
   triples, 2048 draws from default_rng(20240915), and one draw per
   cell.  Candidates with sampled error <= PRE_GATE pass.
2. Circuit cost ranking (count_build).  The search builds each
   passer's full circuit with circuit.build and runs it on the one-lane
   Simulator.  A tuple that circuit.build rejects is not a candidate.
   Stages 1 and 2 run in a process pool, one task per tuple.
3. Exact scan and acceptance (exact_scan).  In rank order, the search
   computes the exact per-segment sup over all 2^25 D points.  The
   Booth partial sum is constant on aligned D stretches, and the ideal
   Gaussian is monotone per segment, so the stretch endpoints bound the
   band.  The search recenters A0 per segment to the exact mid-band.
   It accepts the first candidate whose exact worst error is
   <= EXACT_GATE (428.98 < 429, the task's eps band).

booth_psum() evaluates the Booth partial sums as one small integer
matrix product per D cell.  The digit coefficients depend only on the
(segment, cell) slope word F, and the truncated D terms depend only on
D.  The arithmetic is exact int64, so the result equals an
element-wise evaluation of the same sums.

The search is deterministic.  build_tables draws its sample from the
fixed seed default_rng(20240915), and the Simulator run uses seed 1.  Floating point enters only through numpy double arithmetic
and one least-squares solve, which is bit-stable for a fixed numpy
build.

The exact scan is the worst-case proof of the error bound: it covers
every 25-bit coordinate of every quadratic segment.  main() prints the
exact supremum of the quadratic segments and the sup of the constant
tail, the two numbers the README states.

constants.json holds only the values circuit.py reads: the knob
scalars, Ncut and s0, the per-segment tables A0p/B1p/E, and the packed
55-bit QROM words.

The full grid has 15,552 tuples.  Stages 1 and 2 take about 10-90 s of
CPU per tuple (M = 13 is the slowest), roughly 100-200 CPU-hours in
total, so run the search on a many-core machine.  The search never
emits ops.

RUN: uv run python gen_constants.py   (writes constants.json;
     review the change with git diff)
"""

import itertools
import json
import math
import os
import pathlib
from multiprocessing import Pool

import numpy as np

from uncompute import Simulator
from uncompute.testing import load_sibling

HERE = pathlib.Path(__file__).resolve().parent
CONSTANTS = HERE / "constants.json"
CIRCUIT = load_sibling(__file__, name="gaussian_discovered_circuit", register=False)

# ---- search constants ----
L = 7
H = 2.0**-L
NADDR = 1024
FBITS = 21
TAILZERO = 428.0  # gfun cutoff defining Ncut
CONSTMAX = 428.0  # cap for the constant-run tail segments
PRE_GATE = 560.0  # sampled sup gate to enter the circuit-cost ranking
EXACT_GATE = 428.98  # exact-scan acceptance band (task eps is 429)

# Candidate space: the grid of knob tuples
# (M, CAPW, A0SH, EQ, RB, I0, CLAMP, FLOW, MSKIP).
#   M     bits of D in the mid multiply; odd, because circuit.build
#         indexes past the end of D for even M
#   CAPW  column cap of the final Booth rows
#   A0SH  shift of the stored constant a0 (grid step 2^A0SH)
#   EQ    quantization step 2^EQ of the slope correction
#   RB    minimum number of dropped low bits per final row
#   I0    first retained final row
#   CLAMP 1: segment 0 saturates at 2^A0BITS - 1, which saves one word
#         bit; circuit.load_cfg asserts this clamp, so CLAMP is fixed
#   FLOW  dropped low bits of the stored slope
#   MSKIP 0: build_tables implements no other value
SPACE = tuple(
    itertools.product(
        (9, 11, 13),  # M
        range(16, 22),  # CAPW
        (6, 7, 8),  # A0SH
        range(5, 11),  # EQ
        range(5, 9),  # RB
        (1, 2, 3),  # I0
        (1,),  # CLAMP
        range(4),  # FLOW
        (0,),  # MSKIP
    )
)


def gfun(y):
    return math.exp(-y * y) * 2.0**32


def constify(Ncut, A0SH):
    feas = []
    for s in range(Ncut):
        hi = gfun(s * H)
        lo = gfun((s + 1) * H)
        A0 = round((hi + lo) / 2.0 / (1 << A0SH))
        v = A0 * (1 << A0SH)
        feas.append(max(abs(v - hi), abs(v - lo)) <= CONSTMAX)
    s0 = Ncut
    for s in range(Ncut - 1, -1, -1):
        if feas[s]:
            s0 = s
        else:
            break
    runs = []
    pos = s0
    while pos < Ncut:
        end = pos + 1
        best = None
        while end <= Ncut:
            hi = gfun(pos * H)
            lo = gfun(end * H)
            A0 = round((hi + lo) / 2.0 / (1 << A0SH))
            v = A0 * (1 << A0SH)
            if A0 >= 0 and max(abs(v - hi), abs(v - lo)) <= CONSTMAX:
                best = (end, A0)
                end += 1
            else:
                break
        if best is None:
            return Ncut, []
        runs.append((pos, best[0], best[1]))
        pos = best[0]
    return s0, runs


SEG_CHUNK = 64  # segments per matrix product (bounds the coefficient array)


def booth_psum(Fcell, Dvec, cb, CAPW, RB, ilo, ihi):
    """Booth partial sum P[s, j] of final rows ilo..ihi-1 at D = Dvec[j].

    Fcell[s, u] is the slope word of segment s in D cell u = D >> cb.
    Row i contributes sign * (m1 * t1 + m2 * t2 + (m1 | m2) * bias).
    The digit terms sign * m1, sign * m2 and the bias term depend only
    on (s, u); t1 and t2 depend only on D.  So each cell is one integer
    matrix product: coefficients (segments x (2 rows + 1)) times D terms
    ((2 rows + 1) x points in the cell).  The last coefficient column
    is the summed bias; it multiplies a D term of 1.  Cells hold unequal
    point counts, so the D terms are zero-padded to the largest count.
    Dvec must be sorted.
    """
    assert np.all(Dvec[1:] >= Dvec[:-1]), "Dvec not sorted"
    nseg, ncell = Fcell.shape
    cell = Dvec >> cb
    counts = np.bincount(cell, minlength=ncell)
    kmax = int(counts.max())
    pos = np.arange(Dvec.size) - np.repeat(np.cumsum(counts) - counts, counts)
    col = cell * kmax + pos  # column of each D point in the padded layout
    padded = col.size != ncell * kmax
    nrow = ihi - ilo
    T = np.zeros((ncell, 2 * nrow + 1, kmax), dtype=np.int64)
    T[cell, 2 * nrow, pos] = 1
    for r, i in enumerate(range(ilo, ihi)):
        sh = FBITS - 2 * i
        drop = max(RB, (2 * i + 5) - CAPW)
        T[cell, 2 * r, pos] = ((Dvec >> sh) >> drop) << drop
        T[cell, 2 * r + 1, pos] = ((Dvec >> (sh - 1)) >> drop) << drop
    P = np.empty((nseg, Dvec.size), dtype=np.int64)
    for a in range(0, nseg, SEG_CHUNK):
        Fc = Fcell[a : a + SEG_CHUNK].T  # (ncell, segments)
        # One contiguous plane per coefficient; digits in int8.
        C = np.zeros((2 * nrow + 1, ncell, Fc.shape[1]), dtype=np.int64)
        for r, i in enumerate(range(ilo, ihi)):
            f1 = ((Fc >> (2 * i)) & 1).astype(np.int8)
            f2 = ((Fc >> (2 * i + 1)) & 1).astype(np.int8)
            f0 = ((Fc >> (2 * i - 1)) & 1).astype(np.int8) if i > 0 else 0
            m1 = f0 ^ f1
            m2 = (1 - m1) & (f2 ^ f1)
            sign = 1 - 2 * f2
            drop = max(RB, (2 * i + 5) - CAPW)
            C[2 * r] = sign * m1
            C[2 * r + 1] = sign * m2
            C[2 * nrow] += (sign * (m1 | m2)).astype(np.int64) << (drop - 1)
        C = np.ascontiguousarray(C.transpose(1, 2, 0))
        out = np.matmul(C, T).transpose(1, 0, 2).reshape(Fc.shape[1], -1)
        P[a : a + SEG_CHUNK] = out[:, col] if padded else out
    return P


def rebuild_words(t):
    EQ = t["EQ"]
    FLOW = t["FLOW"]
    FW = t["FW"]
    A0OFF = t["A0OFF"]
    words = [0] * NADDR
    for s in range(t["Ncut"]):
        Es = (int(t["E"][s]) >> EQ) ^ (1 << (18 - EQ))
        words[s] = (int(t["B1p"][s]) >> FLOW) | (Es << FW) | (int(t["A0p"][s]) << A0OFF)
    t["words"] = words


def build_tables(combo):
    M, CAPW, A0SH, EQ, RB, I0, CLAMP, FLOW, MSKIP = combo
    assert MSKIP == 0
    SOFF = 14 - M
    EWID = 19 - EQ
    FW = FBITS - FLOW
    NT = M // 2 + 1
    n = 1
    while gfun(n * H) > TAILZERO:
        n += 1
    Ncut = n
    segs = np.arange(Ncut)
    U = np.arange(1 << M, dtype=np.int64)

    def s_of(Earr):
        """Booth radix-4 mid multiply model with SIGNED d = E - 2^18:
        rows apply arithmetic-shifted truncated d, so base = S directly.
        Numerically identical to the unsigned S - (U<<SOFF).
        """
        Darr = Earr - (1 << 18)
        S = np.zeros((Ncut, 1 << M), dtype=np.int64)
        for t in range(NT):
            j0, j1, j2 = 2 * t - 1, 2 * t, 2 * t + 1
            b0v = ((U >> j0) & 1) if j0 >= 0 else np.zeros(1 << M, dtype=np.int64)
            b1v = ((U >> j1) & 1) if j1 < M else np.zeros(1 << M, dtype=np.int64)
            b2v = ((U >> j2) & 1) if j2 < M else np.zeros(1 << M, dtype=np.int64)
            m1 = b0v ^ b1v
            m2 = (1 - m1) & (b1v ^ b2v)
            sign = 1 - 2 * b2v
            SH1 = M + 4 - 2 * t
            qlo = max(FLOW, EQ - SH1 + 1)
            v1 = ((Darr >> SH1) >> qlo) << qlo
            v2 = ((Darr >> (SH1 - 1)) >> qlo) << qlo
            S += (sign * m1)[None, :] * v1[:, None] + (sign * m2)[None, :] * v2[:, None]
        return S

    def base_of(S):
        return S

    def finish(E, B1p, A0p, s0, maxerr, cmaxerr):
        A0BITS = int(A0p.max()).bit_length()
        if A0BITS < 1 or A0BITS > 26:
            return None
        S = s_of(E)
        base = base_of(S)
        F = B1p[:, None] + base
        if F.min() < 0 or F.max() >= (1 << FBITS):
            return None
        t = {
            "Ncut": Ncut,
            "B1p": B1p,
            "E": E,
            "A0p": A0p,
            "F": F,
            "maxerr": maxerr,
            "cmaxerr": cmaxerr,
            "s0": s0,
            "combo": combo,
            "NT": NT,
            "M": M,
            "CAPW": CAPW,
            "A0SH": A0SH,
            "A0BITS": A0BITS,
            "EQ": EQ,
            "RB": RB,
            "I0": I0,
            "CLAMP": CLAMP,
            "FLOW": FLOW,
            "MSKIP": MSKIP,
            "FW": FW,
            "SOFF": SOFF,
            "EWID": EWID,
            "A0OFF": FW + EWID,
        }
        rebuild_words(t)
        return t

    y0 = segs * H
    ug = (np.arange(129) + 0.5) / 129.0
    Y = y0[:, None] + ug[None, :] * H
    V = np.exp(-Y * Y) * 2.0**32
    Mdes = np.stack([np.ones_like(ug), ug, ug * ug], 1)
    Mp = np.linalg.solve(Mdes.T @ Mdes, Mdes.T)
    coef = V @ Mp.T
    c1 = coef[:, 1]
    c2 = coef[:, 2]
    d = -c2
    dq0 = np.round(d / (1 << EQ)).astype(np.int64)
    b1 = np.maximum(-c1, 0.0)

    dcell = 1 << (25 - M)
    rng = np.random.default_rng(20240915)
    dl_fit = np.array([0, dcell // 2, dcell - 1], dtype=np.int64)
    Dfit = (U[:, None] * dcell + dl_fit[None, :]).ravel()
    Dc_list = [
        Dfit,
        rng.integers(0, 1 << 25, size=2048).astype(np.int64),
        U * dcell + rng.integers(0, dcell, size=1 << M).astype(np.int64),
    ]
    Dcert = np.unique(np.concatenate(Dc_list))

    yDf = (segs[:, None] * (1 << 25) + Dfit[None, :]).astype(np.float64) * 2.0**-32
    idealf = np.exp(-yDf * yDf) * 2.0**32
    yDc = (segs[:, None] * (1 << 25) + Dcert[None, :]).astype(np.float64) * 2.0**-32
    idealc = np.exp(-yDc * yDc) * 2.0**32
    tid = (b1[:, None] + d[:, None] * ((U[None, :] + 0.5) / (1 << M))) / 16.0

    DQCAP = (1 << (18 - EQ)) - 1
    A0CAPV = (1 << (32 - A0SH)) - 1
    offs = tuple(o << FLOW for o in range(4)) if FLOW >= 2 else (0,)
    variants = []
    for delta in (-2, -1, 0, 1, 2):
        dqv = np.clip(dq0 + delta, -DQCAP, DQCAP)
        E = dqv * (1 << EQ) + (1 << 18)
        S = s_of(E)
        base = base_of(S)
        B1b = np.round((tid - base).mean(axis=1)).astype(np.int64)
        B1b = np.maximum(np.maximum(B1b, -base.min(axis=1)), 0)
        if FLOW:
            B1b = ((B1b + (1 << FLOW) - 1) >> FLOW) << FLOW
        for b1off in offs:
            B1p = B1b + b1off
            F = B1p[:, None] + base
            valid = (F.min(axis=1) >= 0) & (F.max(axis=1) < (1 << FBITS))
            Pfit = booth_psum(B1p[:, None] + base, Dfit, 25 - M, CAPW, RB, I0, 11)
            q = idealf + Pfit
            A0p = np.round((q.max(axis=1) + q.min(axis=1)) / 2.0 / 2.0**A0SH).astype(np.int64)
            A0p = np.maximum(A0p, 0)
            if CLAMP:
                A0p = np.minimum(A0p, A0CAPV)
            eseg = np.abs(A0p[:, None] * (2.0**A0SH) - Pfit - idealf).max(axis=1)
            eseg = np.where(valid, eseg, 1e18)
            variants.append({"E": E, "B1p": B1p, "A0p": A0p, "eseg": eseg})

    esegs = np.stack([v["eseg"] for v in variants], 0)
    pick = np.argmin(esegs, axis=0)
    if float(esegs.min(axis=0).max()) >= 1e17:
        return None
    E = np.empty(Ncut, dtype=np.int64)
    B1p = np.empty(Ncut, dtype=np.int64)
    A0p = np.empty(Ncut, dtype=np.int64)
    for vi, v in enumerate(variants):
        sel = pick == vi
        E[sel] = v["E"][sel]
        B1p[sel] = v["B1p"][sel]
        A0p[sel] = v["A0p"][sel]

    s0, runs = constify(Ncut, A0SH)
    for a, e, A0c in runs:
        B1p[a:e] = 0
        E[a:e] = 1 << 18
        A0p[a:e] = A0c
    S = s_of(E)
    base = base_of(S)
    F = B1p[:, None] + base
    if F.min() < 0 or F.max() >= (1 << FBITS):
        return None

    Pc = booth_psum(B1p[:, None] + base, Dcert, 25 - M, CAPW, RB, I0, 11)
    err = np.abs(A0p[:, None] * (2.0**A0SH) - Pc - idealc)
    maxerr = float(err[:s0].max()) if s0 > 0 else 0.0
    cmaxerr = float(err[s0:].max()) if s0 < Ncut else 0.0
    if cmaxerr > CONSTMAX + 1.0:
        return None
    return finish(E, B1p, A0p, s0, maxerr, cmaxerr)


def exact_scan(t):
    """Exact sup ingredients over the quadratic region (P const on aligned
    2^(sh-1+drop) D-stretches; ideal monotone -> both stretch endpoints).
    """
    s0 = t["s0"]
    F = t["F"]
    Mv = t["M"]
    CAPW = t["CAPW"]
    RB = t["RB"]
    I0 = t["I0"]
    cb = 25 - Mv
    rows = []
    gbits = cb
    for i in range(I0, 11):
        sh = FBITS - 2 * i
        drop = max(RB, (2 * i + 5) - CAPW)
        per = sh - 1 + drop
        rows.append((i, per))
        gbits = min(gbits, per)
    GD = np.arange(0, 1 << 25, 1 << gbits, dtype=np.int64)
    cellG = GD >> cb
    Dcell = np.arange(1 << Mv, dtype=np.int64) << cb
    NP = GD.size
    yg1 = np.arange(0, (1 << 25) + 1, 1 << gbits, dtype=np.int64)
    hiA = np.empty(s0)
    loA = np.empty(s0)
    CHs = 8
    for a in range(0, s0, CHs):
        e = min(a + CHs, s0)
        Fc = F[a:e]
        Pcell = np.zeros((e - a, 1 << Mv), dtype=np.int64)
        Pf = np.zeros((e - a, NP), dtype=np.int64)
        for i, per in rows:
            if per >= cb:
                Pcell += booth_psum(Fc, Dcell, cb, CAPW, RB, i, i + 1)
            else:
                Dg = np.arange(0, 1 << 25, 1 << per, dtype=np.int64)
                Pi = booth_psum(Fc, Dg, cb, CAPW, RB, i, i + 1)
                rep = 1 << (per - gbits)
                if rep > 1:
                    Pi = np.repeat(Pi, rep, axis=1)
                Pf += Pi
        P = (Pcell[:, cellG] + Pf).astype(np.float64)
        yy = ((np.arange(a, e, dtype=np.int64) << 25)[:, None] + yg1[None, :]).astype(
            np.float64
        ) * 2.0**-32
        idl = np.exp(-yy * yy) * 2.0**32
        G1 = P + idl[:, :-1]
        G2 = P + idl[:, 1:]
        hiA[a:e] = np.maximum(G1.max(axis=1), G2.max(axis=1))
        loA[a:e] = np.minimum(G1.min(axis=1), G2.min(axis=1))
    return hiA, loA


# ---------------- config sweep + selection ----------------------------------


def count_build(t):
    """Count executed Toffoli and peak qubits of the full discovered circuit for
    config ``t`` (one all-zero lane; executed == emitted for this circuit).
    """
    bb, regs = CIRCUIT.build(t)
    sim = Simulator(bb, batch=1, seed=1)
    sim.write(regs["x"], [0])
    sim.write(regs["k"], [0])
    sim.run()
    return int(sim.stats.toffoli), int(bb.peak_qubits)


def rank_entry(combo):
    """Stages 1 and 2 for one tuple: return (combo, sampled, tof, peak).

    tof and peak are None when the sampled error fails PRE_GATE, and
    sampled is None when build_tables finds no valid table.  The entry
    holds no tables, so the pool returns little data per tuple.
    """
    t = build_tables(combo)
    if t is None:
        return combo, None, None, None
    if t["maxerr"] > PRE_GATE:
        return combo, t["maxerr"], None, None
    try:
        tof, pk = count_build(t)
    except (AssertionError, IndexError):
        return combo, t["maxerr"], None, None  # circuit.build rejects the tuple
    return combo, t["maxerr"], tof, pk


def search(space=SPACE, verbose=True):
    """Run the three-stage search over ``space``; return the chosen table dict.

    The returned dict is the accepted config: ``combo`` + derived
    scalars + ``A0p``/``B1p``/``E`` (exact-recentered) + packed ``words`` +
    ``maxerr``/``cmaxerr``/``s0``/``Ncut``, plus ``tof``/``peak`` (the
    circuit cost) and ``exact_worst`` (the accepted exact-scan sup).
    """
    log = print if verbose else (lambda *_a, **_k: None)
    order = {c: n for n, c in enumerate(space)}
    with Pool(os.cpu_count()) as pool:
        entries = pool.map(rank_entry, space, chunksize=1)
    ranked = sorted(
        ((tof * pk, tof, order[c], c) for c, _, tof, pk in entries if tof is not None),
    )
    log(f"{len(space)} tuples, {len(ranked)} pass the sampled gate and build")

    for prod, tof, _, combo in ranked:
        t = build_tables(combo)
        if max(t["RB"], 25 - t["CAPW"]) < 6:
            continue  # exact-scan model needs >= 6 dropped low bits
        hi, lo = exact_scan(t)
        A0CAP = (1 << t["A0BITS"]) - 1
        sc = float(1 << t["A0SH"])
        A0n = np.clip(np.round((hi + lo) / 2.0 / sc), 0, A0CAP).astype(np.int64)
        wo = float(np.maximum(A0n * sc - lo, hi - A0n * sc).max()) if t["s0"] else 0.0
        log(f"cfg {combo} -> tof {tof:d} prod {prod:d} exact {wo:.2f}")
        if wo <= EXACT_GATE:
            t["A0p"][: t["s0"]] = A0n
            rebuild_words(t)
            t["tof"], t["peak"] = count_build(t)
            t["exact_worst"] = wo
            log(f"CHOSEN cfg={combo} tof={t['tof']:d} peak={t['peak']:d}")
            return t
    raise AssertionError("no tuple in the space passes the exact scan")


# ---------------- constants.json ---------------------------------------------

# The keys circuit.py reads, in file order.
SCALARS = (
    "Ncut",
    "s0",
    "NT",
    "M",
    "CAPW",
    "A0SH",
    "A0BITS",
    "EQ",
    "RB",
    "I0",
    "FLOW",
    "FW",
    "EWID",
)
TABLES = ("A0p", "B1p", "E")


def render(t):
    """Return the constants.json text for the chosen config ``t``."""
    doc: dict = {k: int(t[k]) for k in SCALARS}
    for k in TABLES:
        doc[k] = [int(x) for x in t[k]]
    doc["words"] = [f"0x{w:x}" for w in t["words"]]
    return json.dumps(doc, indent=1) + "\n"


def main():
    t = search(verbose=False)
    print(
        f"exact sup: quadratic segments {t['exact_worst']:.2f}, "
        f"constant tail {t['cmaxerr']:.2f} (band 429)"
    )
    new = render(t)
    old = CONSTANTS.read_text() if CONSTANTS.exists() else None
    CONSTANTS.write_text(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
