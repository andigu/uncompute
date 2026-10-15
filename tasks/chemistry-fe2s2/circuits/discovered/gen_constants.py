#!/usr/bin/env python3
"""Regenerate constants.json for chemistry-fe2s2 by running its two
searches.

constants.json holds two searched constant sets.  Both minimize one
objective, the emitted Toffoli cost of the sign-monomial channel, and
both come from the pinned table alone:

  GAUGE_C     the free Z-gauge masks dressing the permutation frames
              (CZ layers on the flag wires; the sign shift gains
              parity(c AND f) on the pre-image flip of every frame
              application);
  NODE_MONOS  the sign monomials hosted at select-walk nodes instead
              of being emitted per block.

A degree-d monomial prices at d-1 Toffolis: its CCZ plus the
temporary ANDs building its control prefix.  (Same-leading-bit
monomials of one block could share a prefix AND and price lower;
the constants in constants.json contain no such pair, so the price
is exact where it decides anything.)

(a) GAUGE_C: for each permutation frame the candidate Z-masks are
    supported on the frame's moved modes — every subset when the
    frame moves at most 10 modes, otherwise every union of whole
    orbital pairs.  A greedy pass across frames picks, per frame,
    the mask minimizing the exact channel cost with the other frames
    held fixed, and repeats to a fixpoint.  The census replicates
    the circuit's: per-block sign residuals, Möbius-transformed to
    monomials; a gauge term is GF(2)-linear in its mask, so every
    candidate evaluates through a precomputed table.

(b) NODE_MONOS: the greedy single-toggle scan over walk-aligned
    spans, one degree class at a time (pairs, then triples).  A
    toggle (lo, span, monomial) is legal when a block starts at lo,
    blocks tile [lo, lo+span) exactly, and every covered block's
    rank exceeds the monomial's top branch bit.  Its gain is the
    priced difference between the per-block copies it removes and
    the copies it adds plus the node's own emission; toggles apply
    best-gain-first while a strictly positive gain exists, ties
    preferring a node that already hosts a toggle, then the deeper
    node, then the lower address.

The node scan runs on the monomial state that the searched gauge
leaves.  Both searches are deterministic: fixed iteration order, no
randomness, no clock.  The channel cost falls from 769 Toffolis
without a gauge to 117 after (a) and to 108 after (b).

The script loads circuit.py for the table and the frame machinery.
Importing circuit.py does not read constants.json, so the script runs
without it.  The search does not read GAUGE_C or NODE_MONOS.

RUN: uv run python gen_constants.py   (writes constants.json and
     reports whether its text changed)
"""

import json
import pathlib

import numpy as np

from uncompute.classical import parity
from uncompute.testing import load_sibling

HERE = pathlib.Path(__file__).resolve().parent
CONSTANTS = HERE / "constants.json"

S = load_sibling(__file__)

X = S.X
PERM_FRAMES = sorted(S.PERM_MACH)  # gauge-capable frames
MODE_CAP = 10  # full-subset enumeration cap


def _sgn_nogauge(g, f):
    """Return the frame's bare sign shift (circuit._sgn minus the gauge)."""
    if g == 0:
        return S._g(f)
    if g == 1:
        return S._s(f)
    perm, _dimg, scols, _b, _l = S.PERM_MACH[g]
    return S._qs_graph(scols, S._apply_perm(perm, f))


def block_census():
    """Per block: the gauge-free residual Möbius spectrum and, per
    permutation frame, the Möbius spectrum of the XORed pre-image
    flips its applications contribute.  Returns
    [(base, k, mob0, {g: mobU})] over all rank >= 1 blocks.
    """
    out = []
    for t, lo, hi in S.REGIONS:
        k = len(t)
        if k == 0:
            continue
        n = 1 << k
        for base in range(lo, hi, n):
            resid = [0] * n
            umaps = {g: [0] * n for g in set(t) if g in S.PERM_MACH}
            sacc = [0] * n
            for bm in range(1, n):
                i = bm.bit_length() - 1
                bm0 = bm ^ (1 << i)
                f0, _z0 = S._fz(base + bm0)
                g = t[i]
                sacc[bm] = sacc[bm0] ^ _sgn_nogauge(g, f0)
                for u in umaps.values():
                    u[bm] = u[bm0]
                if g in umaps:
                    umaps[g][bm] ^= f0
                resid[bm] = S.SIGN[base] ^ sacc[bm] ^ S.SIGN[base + bm]
            for vec in [resid] + list(umaps.values()):
                for i in range(k):
                    st = 1 << i
                    for lo2 in range(0, n, st << 1):
                        for j in range(st):
                            vec[lo2 + st + j] ^= vec[lo2 + j]
            out.append((base, k, resid, umaps))
    return out


def monomial_entries(census):
    """(weight, mob0, {g: W}) for every degree >= 2 monomial slot;
    weight = degree - 1, the slot's emitted Toffoli price.
    """
    entries = []
    for _base, k, mob0, umaps in census:
        for m in range(1, 1 << k):
            deg = bin(m).count("1")
            if deg < 2:
                continue
            ws = {g: u[m] for g, u in umaps.items() if u[m]}
            if mob0[m] or ws:
                entries.append((deg - 1, mob0[m], ws))
    return entries


def channel_cost(entries, gauge):
    """Exact channel Toffoli cost under a Z-gauge assignment."""
    total = 0
    for wt, mob0, ws in entries:
        v = mob0
        for g, w in ws.items():
            c = gauge.get(g, 0)
            if c:
                v ^= parity(c & w)
        total += wt * v
    return total


def _frame_candidates(g):
    """(slot bit -> mode mask) generators for frame g's search space."""
    perm = S.PERM_MACH[g][0]
    moved = [k for k in range(X) if perm[k] != k]
    if len(moved) <= MODE_CAP:
        return [1 << k for k in moved]
    pairs = sorted({k >> 1 for k in moved})
    return [3 << (2 * p) for p in pairs]


def _best_mask(g, entries, gauge):
    """argmin over frame g's candidate masks, others held at
    ``gauge``; ties resolve to the smallest slot index (np.argmin).
    """
    slots = _frame_candidates(g)
    nb = len(slots)
    base_cost = 0
    diffs = {}
    for wt, mob0, ws in entries:
        v = mob0
        for gg, w in ws.items():
            if gg == g:
                continue
            c = gauge.get(gg, 0)
            if c:
                v ^= parity(c & w)
        w = ws.get(g, 0)
        wk = 0
        for j, sm in enumerate(slots):
            if parity(w & sm):
                wk |= 1 << j
        base_cost += wt * v
        if wk:
            diffs[wk] = diffs.get(wk, 0) + wt * (1 - 2 * v)
    cost = np.full(1 << nb, base_cost, dtype=np.int64)
    sel = np.arange(1 << nb, dtype=np.uint32)
    for wk, d in diffs.items():
        if d:
            par = (np.bitwise_count(sel & np.uint32(wk)) & 1).astype(np.int64)
            cost += d * par
    best = int(np.argmin(cost))
    mask = 0
    for j in range(nb):
        if (best >> j) & 1:
            mask |= slots[j]
    return mask, int(cost[best])


def search_gauge(entries):
    """Greedy across frames to a fixpoint; returns (gauge, cost)."""
    gauge = {}
    for _round in range(8):
        changed = False
        for g in PERM_FRAMES:
            mask, _cost = _best_mask(g, entries, gauge)
            if mask != gauge.get(g, 0):
                changed = True
            if mask:
                gauge[g] = mask
            else:
                gauge.pop(g, None)
        if not changed:
            break
    return gauge, channel_cost(entries, gauge)


def block_monomials(census, gauge):
    """Return {base: set(monomial tuples)} under a Z-gauge assignment."""
    mono = {}
    for base, k, mob0, umaps in census:
        ms = set()
        for m in range(1, 1 << k):
            if bin(m).count("1") < 2:
                continue
            v = mob0[m]
            for g, u in umaps.items():
                c = gauge.get(g, 0)
                if c and u[m]:
                    v ^= parity(c & u[m])
            if v:
                ms.add(tuple(i for i in range(k) if (m >> i) & 1))
        if ms:
            mono[base] = ms
    return mono


def _walk_spans():
    """Every walk-aligned span [lo, lo+span) a select-walk node can
    host: a block starts at lo and blocks tile the span exactly.
    """
    spans = []
    for s in range(16):
        sp = 1 << s
        for lo in range(0, S.L, sp):
            if lo not in S.SPAN_AT:
                continue
            covered = []
            b = lo
            ok = True
            while b < lo + sp:
                nb = S.SPAN_AT.get(b)
                if nb is None or b + nb > lo + sp:
                    ok = False
                    break
                covered.append(b)
                b += nb
            if ok and len(covered) > 1:
                spans.append((lo, sp, covered))
    return spans


def search_nodes(mono):
    """Greedy single-toggle scan; mutates ``mono``; returns the
    chosen toggle set {(lo, hi, monomial)}.  A toggle's gain is
    weighted at the slot price (degree - 1): it removes the copies
    present, adds the copies absent, and adds one node emission.
    Degree classes scan in ascending order; equal gains resolve to a
    node already hosting a toggle, then the deeper node, then the
    lower address.
    """
    spans = _walk_spans()
    toggles = set()
    degrees = sorted({len(mn) for v in mono.values() for mn in v})
    for deg in degrees:
        while True:
            best = None
            for lo, sp, covered in spans:
                min_rank = min(S.SPAN_AT[b].bit_length() - 1 for b in covered)
                cands = set()
                for b in covered:
                    cands |= {mn for mn in mono.get(b, ()) if len(mn) == deg and max(mn) < min_rank}
                hosted = any(t[0] == lo and t[1] == lo + sp for t in toggles)
                for mn in cands:
                    present = sum(1 for b in covered if mn in mono.get(b, ()))
                    gain = (deg - 1) * (2 * present - len(covered) - 1)
                    key = (-gain, not hosted, -sp, lo, mn)
                    if gain >= 1 and (best is None or key < best):
                        best = key
            if best is None:
                break
            _ng, _nh, nsp, lo, mn = best
            sp = -nsp
            for b in [bb for bb in range(lo, lo + sp) if bb in S.SPAN_AT]:
                ms = mono.get(b, set())
                ms ^= {mn}
                if ms:
                    mono[b] = ms
                else:
                    mono.pop(b, None)
            toggles.add((lo, lo + sp, mn))
    return toggles


def search():
    """Run both searches; return (gauge, toggles)."""
    census = block_census()
    gauge, _cost = search_gauge(monomial_entries(census))
    return gauge, search_nodes(block_monomials(census, gauge))


def encode_doc(gauge, toggles):
    """Canonical constants.json text, one NODE_MONOS row per line.

    GAUGE_C maps a frame index (GEN_NAMES order: 3 = D2, 4 = s_thio) to
    a 40-bit Z mask c as a hex string; the frame's sign shift gains
    parity(c AND f).  Each NODE_MONOS row [lo, hi, monomial] hosts one
    sign monomial at the select-walk node covering blocks [lo, hi); the
    monomial lists branch-bit positions.
    """
    rows = sorted([lo, hi, list(mn)] for lo, hi, mn in toggles)
    out = ["{"]
    gauge_c = json.dumps({str(g): hex(m) for g, m in sorted(gauge.items())})
    out.append(f'  "GAUGE_C": {gauge_c},')
    out.append('  "NODE_MONOS": [')
    for i, r in enumerate(rows):
        sep = "," if i < len(rows) - 1 else ""
        out.append(f"    {json.dumps(r)}{sep}")
    out.append("  ]")
    out.append("}")
    return "\n".join(out) + "\n"


def generate():
    """Run both searches and return the canonical constants.json text."""
    return encode_doc(*search())


def main():
    new = generate()
    old = CONSTANTS.read_text() if CONSTANTS.exists() else None
    CONSTANTS.write_text(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
