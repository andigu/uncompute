#!/usr/bin/env python3
"""Regenerate constants.json for the hhl-angle discovered circuit by running
its three searches: the configuration search, the Booth-refit leaf
fitter, and the cap grid.

constants.json holds the chosen configuration (cfg) and the 345-leaf
dyadic coefficient table (leaves: keys lo, bb, g0, g1, g2, sh, one
entry per leaf, sorted by lo).

The configuration search
------------------------
The search chooses C0DROP, VORD, and the seven B4 truncations.  Its
objective is the TQ-cost floor(T) x q of the built circuit.  To score
one configuration, the search runs the leaf fitter and then the cap
grid.  A configuration fails if the fitter cannot meet ERRLIM or the
circuit cannot carry the table.

The search is a coordinate search.  It starts at START, the
configuration of the discovery run.  For each coordinate in turn, it
scores every value in the coordinate's range, with the other
coordinates held.  It moves only to a strictly lower TQ-cost; among
equal costs it takes the first value in range order.  It stops after a
full pass that makes no move.  The ranges hold every value the
datapath admits:

* VORD: all six curvature-row orders.
* C0DROP: 0 to 9.  The intercept rounding error reaches 2^(C0DROP-1)
  output units, which must stay under the 429-unit band.
* B4[t]: from the smallest value with sig_t >= 1 (the |d|=2 operand
  reads v bit sig_t - 1) to 8.  One row's truncation error reaches
  2^B4[t] output units, which must stay under the band.

One pass from START scores 65 configurations: START and its 64
single-coordinate changes.
The search does not vary the datapath widths NDN, GUARD, RPROD, G1W,
TAU, RQ, G2W, J0, and QCAP.  circuit.py lays out its registers and
digit rows for these widths.

The leaf fitter
---------------
Both of the circuit's multiplies run as signed Booth radix-4 digit rows.
The fitter prices the table against exactly that arithmetic:

* product: Y = dn >> J0 is recoded into 7 digits d_t = y_{2t-1} + y_{2t}
  - 2 y_{2t+1} (boundary bits zero).  Row t adds d_t (v >> sig_t) << B4[t]
  with sig_t = B4[t] + RPROD - J0 - 2t.  The |d|=2 candidate sits one v
  bit deeper: the magnitude folds into the shift before truncation.
* curvature: g2 (G2W = 5 bits) is recoded into 3 digits e_s.  Row s adds
  e_s (dq << 2s) >> RQ on dq = dn >> TAU, |e|=2 one dq bit deeper.  The
  circuit emits the rows in the order VORD.

fit_leaf accepts a candidate only if the two-sided window bound
0 <= (W mod 2^17) - S < 2^17 holds after each curvature row, walked in
the VORD order.  The final row's bound is the load-bearing one: it keeps
the slope subtraction inside W's 2^17 block, which licenses the
circuit's 17-bit conjugated chains.  Signed digits make its lower edge
non-trivial (Q can be negative).  The per-prefix form is stronger than
the circuit needs, and it covers the uncompute direction, whose row
order is VORD reversed.

The fitter is an adaptive splitter.  It seeds each leaf with numpy
polyfit on a fixed decimated sample grid, then runs a bounded local
integer search.  It halves a leaf until a candidate meets ERRLIM at
every input.  verify() then checks all 2,031,616 in-domain X.  At
C0DROP = 6, ERRLIM = 428.5 it lands 345 leaves, worst centered error
427.15.

The cap grid
------------
PCAP (the live-qubit cap) and C0GL (the c0add [GID]-prefix length) are
emit-time dials; the leaf table does not depend on them.  The grid
builds circuit.py at every (PCAP, C0GL) pair in PCAPS x [0, C0W] and
takes the floor(T) x q argmin.  Forked worker processes share the
builds.  The argmin is unique: (117, 11), TQ-cost 155,020.  The grid
loads circuit.py from a temporary copy of the task layout that
carries the freshly fitted table, so the output never reads the
committed constants.json.

Determinism
-----------
There is no RNG and no time dependence.  Every searched quantity is
rounded to an integer and refined in exact int64 arithmetic.  Ties in
the grid break toward the smaller cap, then the shorter prefix.  The
configuration search scores candidates in forked worker processes and
keeps their range order.
checks.py proves the worst centered error of the table that circuit.py
loads.

Field meanings (circuit.py has the full datapath model):

    NDN      local-coordinate width: dn is the 15-bit dyadic coordinate
    GUARD    guard bits under the stored slope word (v = W - Q headroom)
    RPROD    product right-shift base (digit t shifts by
             sig_t = B4[t] + RPROD - J0 - 2t)
    G1W      stored slope-word width (W rides on G1W wires, msb-aligned)
    TAU/RQ   quadratic coordinate drop / quadratic right shift
    G2W      quadratic coefficient width (5: three Booth digits)
    J0       lowest dn bit entering the product (Y = dn >> J0 is the
             recoded multiplier)
    C0DROP   intercept bits dropped (g0 is a multiple of 2^C0DROP;
             searched: the configuration search)
    QCAP     bias exponent: W = (g1 << GUARD) + 2^QCAP; the certified
             window bound keeps the slope subtraction inside W's
             2^(QCAP+1) block, capping every curvature-row carry chain
             at QCAP+1 bits
    B4       per-digit product truncations: row t lands at offset B4[t]
             and keeps v down to bit sig_t (|d|=2 one bit deeper — the
             magnitude folds into the shift before truncation;
             searched: the configuration search)
    VORD     the curvature-row emission order; the window invariant is
             certified prefix-by-prefix in exactly this order (the
             uncompute retraces it reversed; searched: the
             configuration search)
    PCAP     live-qubit cap fed to the product rows' [GID]-prefix dial
             (searched: the cap grid)
    C0GL     c0add [GID]-prefix length (the rest of the chain rides
             ancilla-free [CDKM] steps; searched: the cap grid)
    ERRLIM   fit-time error budget (output units); the run-time gate is
             the task band (contract.json eps_ulp = 429).  A search
             parameter only: circuit.py does not read it, so it stays out
             of constants.json.

RUN: uv run python gen_constants.py   (writes constants.json;
     review the change with git diff)
"""

import itertools
import json
import multiprocessing
import os
import pathlib
import shutil
import tempfile

import numpy as np

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent
CONSTANTS = HERE / "constants.json"

LO, HI = 1 << 16, 1 << 21
MOD = 1 << 33
Xs = np.arange(LO, HI, dtype=np.int64)
Ff = np.arcsin((2.0**15) / Xs.astype(np.float64)) * (2.0**32)

# The datapath widths and the fit-time error budget ERRLIM.  The
# configuration search holds these fixed.
DESIGN = {
    "NDN": 15,
    "GUARD": 4,
    "RPROD": 11,
    "G1W": 16,
    "TAU": 3,
    "RQ": 4,
    "G2W": 5,
    "J0": 2,
    "QCAP": 16,
    "ERRLIM": 428.5,
}
# The start of the configuration search: the configuration of the
# discovery run.
START = {"C0DROP": 6, "B4": [4, 5, 5, 5, 6, 6, 6], "VORD": [2, 1, 0]}

NDN = DESIGN["NDN"]
GUARD = DESIGN["GUARD"]
RPROD = DESIGN["RPROD"]
G1W = DESIGN["G1W"]
TAU = DESIGN["TAU"]
RQ = DESIGN["RQ"]
G2W = DESIGN["G2W"]
J0 = DESIGN["J0"]
QCAP = DESIGN["QCAP"]
ERRLIM = DESIGN["ERRLIM"]
VW = G1W + GUARD
NDQ = NDN - TAU
G1SC = RPROD - GUARD  # slope seed scale: 2^7 per output unit
G2SC = RPROD + TAU + RQ  # curvature seed scale: 2^18
BIAS = 1 << QCAP
GLIM = (1 << G1W) - (BIAS >> GUARD)
WIN = 1 << (QCAP + 1)

CFG_KEYS = [
    "NDN",
    "GUARD",
    "RPROD",
    "G1W",
    "TAU",
    "RQ",
    "G2W",
    "J0",
    "C0DROP",
    "QCAP",
    "PCAP",
    "B4",
    "VORD",
    "C0GL",
]
LEAF_KEYS = ["lo", "bb", "g0", "g1", "g2", "sh"]


def sig_of(b4):
    """Return the per-row v truncation bits sig_t = B4[t] + RPROD - J0 - 2t."""
    return [b4[t] + RPROD - J0 - 2 * t for t in range(7)]


class Leaf:
    __slots__ = ("bb", "coef", "lo")

    def __init__(self, lo, bb, coef):
        self.lo, self.bb, self.coef = lo, bb, coef


def booth_digits(y, nd):
    """Radix-4 Booth digits of integer ``y``, boundary bits zero."""
    out = []
    for t in range(nd):
        ya = (y >> (2 * t - 1)) & 1 if t > 0 else 0
        yb = (y >> (2 * t)) & 1
        yc = (y >> (2 * t + 1)) & 1
        out.append(ya + yb - 2 * yc)
    assert sum(d * (4**t) for t, d in enumerate(out)) == y
    return out


def leaf_dn(lo, bb):
    w = 1 << bb
    sh = NDN - bb
    i0 = lo - LO
    xs = Xs[i0 : i0 + w]
    Hh = (lo & ((1 << NDN) - 1)) >> bb
    dn = (((xs & (w - 1)) << sh) | Hh).astype(np.int64)
    return dn, sh, i0, w


def q_row(dn, e, s):
    """Curvature digit row s: e (dq << 2s) >> RQ, |e|=2 one bit deeper."""
    if e == 0:
        return np.zeros_like(dn)
    dq = dn >> TAU
    return np.sign(e) * ((dq << (2 * s + (abs(e) == 2))) >> RQ)


def model_Q(dn, g2):
    ee = booth_digits(g2, 3)
    return q_row(dn, ee[0], 0) + q_row(dn, ee[1], 1) + q_row(dn, ee[2], 2)


def window_ok(dn, g2, g1, vord):
    """Check the per-prefix window invariant in the row order ``vord``:
    after each curvature row, 0 <= wlow - S < 2^17 on every dn.
    """
    wlow = ((g1 << GUARD) + BIAS) % WIN
    ee = booth_digits(g2, 3)
    S = np.zeros_like(dn)
    for s in vord:
        S = S + q_row(dn, ee[s], s)
        if int(S.max()) > wlow or int(S.min()) <= wlow - WIN:
            return False
    return True


def model_P(dn, v, b4):
    """Product: 7 signed Booth digit rows of Y = dn >> J0."""
    sig = sig_of(b4)
    y = dn >> J0
    out = np.zeros_like(dn)
    for t in range(7):
        ya = (y >> (2 * t - 1)) & 1 if t > 0 else np.zeros_like(y)
        yb = (y >> (2 * t)) & 1
        yc = (y >> (2 * t + 1)) & 1
        d = ya + yb - 2 * yc
        a1 = (v >> sig[t]) << b4[t]
        a2 = (v >> (sig[t] - 1)) << b4[t]
        out = out + np.sign(d) * np.where(np.abs(d) == 2, a2, a1)
    return out


def comp_vec(dn):
    return ((dn >> J0) << J0) * (1 << (QCAP - RPROD))


def fit_leaf(lo, bb, cfg):
    sh = NDN - bb
    if sh < 0 or sh > 7:
        return None
    b4, vord = cfg["B4"], cfg["VORD"]
    dn, sh, i0, w = leaf_dn(lo, bb)
    Fs = Ff[i0 : i0 + w]
    compv = comp_vec(dn)
    step = max(1, w // 2048)
    dnf = dn[::step].astype(np.float64)
    try:
        c2, c1, _c0f = np.polyfit(dnf, Fs[::step], 2)
    except Exception:  # noqa: BLE001  (a failed fit rejects this leaf; the search moves on)
        return None
    g2_0 = round(c2 * (1 << G2SC))
    if g2_0 > (1 << G2W) + 60:
        return None
    g2_0 = min(max(g2_0, 0), (1 << G2W) - 1)
    g1_0 = round(-c1 * (1 << G1SC))
    step0 = 1 << cfg["C0DROP"]
    best = None
    seen = set()
    for wide in (False, True):
        rg2 = 2 if wide else 1
        rg1 = range(-3, 4) if wide else range(-2, 3)
        for g2 in range(max(0, g2_0 - rg2), min((1 << G2W) - 1, g2_0 + rg2) + 1):
            Q = model_Q(dn, g2)
            g1c = g1_0
            if 1 <= g1c < GLIM and window_ok(dn, g2, g1c, vord):
                vb = (np.int64(g1c) << GUARD) + BIAS - Q
                if int(vb.min()) >= 0 and int(vb.max()) < (1 << VW):
                    r = Fs + model_P(dn, vb, b4) - compv
                    slope = np.polyfit(dnf, r[::step], 1)[0]
                    g1c = g1c + round(-slope * (1 << G1SC))
            for dg1 in rg1:
                g1 = g1c + dg1
                if g1 < 1 or g1 >= GLIM:
                    continue
                if not window_ok(dn, g2, g1, vord):
                    continue
                if (g2, g1) in seen:
                    continue
                seen.add((g2, g1))
                vb = (np.int64(g1) << GUARD) + BIAS - Q
                if int(vb.min()) < 0 or int(vb.max()) >= (1 << VW):
                    continue
                r = Fs + model_P(dn, vb, b4) - compv
                mid = (float(r.max()) + float(r.min())) / 2.0
                g0f = int(np.floor(mid / step0)) * step0
                for g0 in (g0f, g0f + step0):
                    if g0 < 0 or g0 >= (1 << 32):
                        continue
                    err = float(np.abs(np.float64(g0) - r).max())
                    if err <= ERRLIM and (best is None or err < best[0]):
                        best = (err, (g0, g1, g2, sh))
        if best is not None:
            break
    return None if best is None else best[1]


def build_tree(cfg):
    def build(lo, bb):
        if bb <= NDN:
            t = fit_leaf(lo, bb, cfg)
            if t is not None:
                return Leaf(lo, bb, t)
        if bb <= 7:
            raise RuntimeError(f"segmentation failure at lo={lo}")
        h = 1 << (bb - 1)
        return Node(bb - 1, build(lo, bb - 1), build(lo + h, bb - 1))

    Ts = [build(1 << (16 + e), 16 + e) for e in range(5)]
    return Node(20, Node(19, Node(18, Node(17, Ts[0], Ts[1]), Ts[2]), Ts[3]), Ts[4])


class Node:
    __slots__ = ("bit", "left", "right")

    def __init__(self, bit, left, right):
        self.bit, self.left, self.right = bit, left, right


def collect(nd):
    out = []

    def go(n):
        if isinstance(n, Leaf):
            out.append(n)
        else:
            go(n.left)
            go(n.right)

    go(nd)
    return out


def verify(leaves, cfg):
    c0drop, b4, vord = cfg["C0DROP"], cfg["B4"], cfg["VORD"]
    addend = np.zeros(HI - LO, dtype=np.int64)
    worst = 0.0
    for lf in leaves:
        dn, sh, i0, w = leaf_dn(lf.lo, lf.bb)
        g0, g1, g2, shv = lf.coef
        assert shv == sh
        assert g0 % (1 << c0drop) == 0
        assert 1 <= g1 < GLIM and 0 <= g2 < (1 << G2W)
        assert window_ok(dn, g2, g1, vord)
        vb = (np.int64(g1) << GUARD) + BIAS - model_Q(dn, g2)
        assert int(vb.min()) >= 0 and int(vb.max()) < (1 << VW)
        add = (np.int64(g0) - model_P(dn, vb, b4) + comp_vec(dn)) % MOD
        addend[i0 : i0 + w] = add
        e = add.astype(np.float64) - Ff[i0 : i0 + w]
        e = np.mod(e + 2.0**32, 2.0**33) - 2.0**32
        m = float(np.abs(e).max())
        worst = max(worst, m)
    return addend, worst


def fit(cfg):
    """Run the deterministic Booth-refit leaf fitter at configuration ``cfg``.

    ``cfg`` holds C0DROP, B4, and VORD.  Returns ``(leaves, worst)``:
    ``leaves`` is a dict in the exact shape and order that
    ``constants.json`` uses, and ``worst`` is the exhaustively verified
    worst centered error in output units.  Raises RuntimeError if no
    table meets ERRLIM.
    """
    root = build_tree(cfg)
    leaf_objs = collect(root)
    _addend, worst = verify(leaf_objs, cfg)
    if worst > ERRLIM + 0.25:
        raise RuntimeError(f"fit rejected: worst={worst:.3f} > ERRLIM")
    leaf_objs = sorted(leaf_objs, key=lambda lf: lf.lo)
    leaves = {
        "lo": [lf.lo for lf in leaf_objs],
        "bb": [lf.bb for lf in leaf_objs],
        "g0": [lf.coef[0] for lf in leaf_objs],
        "g1": [lf.coef[1] for lf in leaf_objs],
        "g2": [lf.coef[2] for lf in leaf_objs],
        "sh": [lf.coef[3] for lf in leaf_objs],
    }
    return leaves, worst


def dump(data):
    """Canonical serialization: top-level keys one per line, ``cfg`` inline,
    each leaf array on its own (long) line — inspectable, diff-stable.
    """
    assert list(data["cfg"].keys()) == CFG_KEYS, list(data["cfg"].keys())
    assert list(data["leaves"].keys()) == LEAF_KEYS, list(data["leaves"].keys())
    nleaves = len(data["leaves"]["lo"])
    out = ["{"]
    out.append(f'  "cfg": {json.dumps(data["cfg"])},')
    out.append('  "leaves": {')
    for i, k in enumerate(LEAF_KEYS):
        arr = data["leaves"][k]
        assert len(arr) == nleaves and all(isinstance(v, int) for v in arr)
        sep = "," if i < len(LEAF_KEYS) - 1 else ""
        out.append(f'    "{k}": {json.dumps(arr)}{sep}')
    out.append("  }")
    out.append("}")
    return "\n".join(out) + "\n"


def full_cfg(cfg, pcap, c0gl):
    """Return the constants.json "cfg" dict for a searched configuration."""
    vals = {**DESIGN, **cfg, "PCAP": pcap, "C0GL": c0gl}
    return {k: vals[k] for k in CFG_KEYS}


# --------------------------------------------------------------------------
# The cap grid.
# --------------------------------------------------------------------------

PCAPS = [None, 133, 131, 129, 127, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113]
UNCAPPED = 1 << 30  # a live-qubit cap no schedule can reach

_CIRCUIT = None  # the loaded circuit; forked workers inherit it


def _score(point):
    """floor(T) x q of the circuit built at one (PCAP, C0GL) point."""
    pcap, c0gl = point
    assert _CIRCUIT is not None, "the caller loads the circuit before scoring"
    _CIRCUIT.PCAP = UNCAPPED if pcap is None else pcap
    _CIRCUIT.C0GL = c0gl
    b, _ = _CIRCUIT.build()
    s = b.phase_summary()
    return int(s["total_emitted_toffoli"]) * s["peak_qubits"]


def _map_forked(fn, items):
    """Return [fn(x) for x in items], computed across forked workers.

    Each worker inherits this module by fork, so fn needs no pickling
    (this module loads by path and has no importable name).  Only the
    results cross the pipes.
    """
    ctx = multiprocessing.get_context("fork")
    nproc = min(os.cpu_count() or 1, len(items))
    chunks = [items[i::nproc] for i in range(nproc)]
    jobs = []
    for chunk in chunks:
        rd, wr = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=lambda c=chunk, w=wr: w.send([fn(x) for x in c]))
        proc.start()
        wr.close()
        jobs.append((proc, rd))
    parts = []
    for proc, rd in jobs:
        parts.append(rd.recv())
        proc.join()
        assert proc.exitcode == 0, proc.exitcode
    out = [None] * len(items)
    for i, part in enumerate(parts):
        out[i::nproc] = part
    return out


def select_caps(cfg, leaves):
    """Return (TQ-cost, PCAP, C0GL, uncapped TQ-cost), or None.

    The first three values come from the argmin over the capped grid
    points.  The result is None if circuit.py rejects the table.  The
    circuit rejects a table that leaves a coefficient wire unwritten.

    circuit.py finds contract.json at the nearest ancestor with grader.py
    and reads constants.json beside itself.  The temporary layout gives
    it both, with the fitted table.  The table carries placeholder dials
    (PCAP = -1, C0GL = 0), and _score sets both dials before each build.
    """
    global _CIRCUIT
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "grader.py").write_text("")
        shutil.copy(HERE.parents[1] / "contract.json", tmp)
        sol = tmp / "circuits" / "discovered"
        sol.mkdir(parents=True)
        shutil.copy(HERE / "circuit.py", sol)
        (sol / "constants.json").write_text(dump({"cfg": full_cfg(cfg, -1, 0), "leaves": leaves}))
        _CIRCUIT = load_module(sol / "circuit.py", "hhl_grid_circuit")
    try:
        _score((None, 0))
    except AssertionError:
        return None
    points = [(p, g) for p in PCAPS for g in range(_CIRCUIT.C0W + 1)]
    scores = _map_forked(_score, points)
    tq, pcap, c0gl = min((s, p, g) for s, (p, g) in zip(scores, points) if p is not None)
    return tq, pcap, c0gl, min(s for s, (p, _) in zip(scores, points) if p is None)


# --------------------------------------------------------------------------
# The configuration search.
# --------------------------------------------------------------------------


def coordinates():
    """Return the search coordinates as (key, index, values) triples.

    ``index`` selects one B4 entry and is None for the scalar keys.
    The module docstring states the reason for each range.
    """
    coords = [
        ("VORD", None, [list(p) for p in itertools.permutations(range(3))]),
        ("C0DROP", None, list(range(10))),
    ]
    for t in range(7):
        coords.append(("B4", t, list(range(max(0, 2 * t + J0 + 1 - RPROD), 9))))
    return coords


def _fit_or_none(cfg):
    try:
        return fit(cfg)
    except RuntimeError:
        return None


def search():
    """Run the coordinate search from START.

    Returns (cfg, leaves, pcap, c0gl, tq, uncapped tq) at the
    configuration the search stops at.
    """
    scored = {}  # json key -> None or (tq, leaves, pcap, c0gl, uncapped tq)

    def key(cfg):
        return json.dumps(cfg, sort_keys=True)

    def score_all(cfgs):
        todo = []
        for cfg in cfgs:
            if key(cfg) not in scored and cfg not in todo:
                todo.append(cfg)
        fits = _map_forked(_fit_or_none, todo)
        for cfg, fitted in zip(todo, fits):
            res = None if fitted is None else select_caps(cfg, fitted[0])
            scored[key(cfg)] = None if res is None else (res[0], fitted[0], *res[1:])

    cur = {"C0DROP": START["C0DROP"], "B4": list(START["B4"]), "VORD": list(START["VORD"])}
    score_all([cur])
    assert scored[key(cur)] is not None, "the start configuration fails"
    moved = True
    while moved:
        moved = False
        for name, idx, values in coordinates():
            cands = []
            for v in values:
                cand = {**cur, "B4": list(cur["B4"])}
                if idx is None:
                    cand[name] = v
                else:
                    cand["B4"][idx] = v
                cands.append(cand)
            score_all(cands)
            ok = [(scored[key(c)][0], i) for i, c in enumerate(cands) if scored[key(c)]]
            tq, i = min(ok)
            if tq < scored[key(cur)][0]:
                cur, moved = cands[i], True
    tq, leaves, pcap, c0gl, tq_uncapped = scored[key(cur)]
    return cur, leaves, pcap, c0gl, tq, tq_uncapped


def generate():
    """Run the searches and return the canonical constants.json text."""
    cfg, leaves, pcap, c0gl, tq, tq_uncapped = search()
    assert tq_uncapped >= tq, "the uncapped schedule has the lowest cost on the grid"
    # Width check: the fitted table must fill exactly the G2W the wires
    # carry.  A narrower table would leave a provably dead wire.
    assert max(int(v).bit_length() for v in leaves["g2"]) == G2W
    return dump({"cfg": full_cfg(cfg, pcap, c0gl), "leaves": leaves})


def main():
    new = generate()
    old = CONSTANTS.read_text() if CONSTANTS.exists() else None
    CONSTANTS.write_text(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
