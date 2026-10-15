#!/usr/bin/env python3
"""Exhaustive worst-case proof for the hhl-angle discovered circuit.

circuit.py never imports this file, so an edit here does not change the
circuit or its test draw.  certify_table walks all 2,031,616 in-domain X through
the exact integer datapath model (the signed Booth digit rows of model_Q
and model_P plus the comp term).  It proves the invariants the circuit
relies on:

  - partition: every X is covered exactly once; each leaf is dyadically
    aligned (lo = 0 mod 2^bb) with sh = 15 - bb in [0, 7].
  - window: after each curvature row, walked in the frozen VORD order,
    0 <= (W mod 2^17) - S < 2^17 on every input.  The final row's bound
    keeps the slope subtraction inside W's 2^17 block, which licenses
    the circuit's 17-bit conjugated chains.  The uncompute runs the rows
    in reverse, so the per-prefix form covers it too.
  - range: 0 <= v < 2^20, so no complement-subtract wraps.
  - quantize: g0 = 0 mod 2^6, 1 <= g1 < GLIM, g2 < 2^5.

It then measures the worst centered error against float arcsin.  The
README states that error: 427.15 output units at X = 1,886,892, inside
the 429-unit band.  The 0.25-unit slop covers float evaluation of the
graded ideal.

RUN: uv run python checks.py
"""

import numpy as np

from uncompute.testing import load_sibling

# Runs the contract/config asserts.
S = load_sibling(__file__, name="hhl_discovered_circuit", register=False)
G = load_sibling(__file__, "gen_constants.py", name="hhl_discovered_gen", register=False)
ERRLIM = G.ERRLIM
# The exact radix-4 recode both row stages build in place.
booth_digits = G.booth_digits

# The construction surface the battery consumes, loaded from the sibling
# circuit.py by path (robust to any cwd).
LO, HI, MOD, EPS = S.LO, S.HI, S.MOD, S.EPS
CFG, LEAVES = S.CFG, S.LEAVES

XS = np.arange(LO, HI, dtype=np.int64)
FF = np.arcsin((2.0**15) / XS.astype(np.float64)) * (2.0**32)


def certify_table(cfg, leaves):
    """Certify the leaf table exhaustively.  ``cfg`` is the frozen
    configuration dict (constants.json "cfg"); ``leaves`` iterates (lo, bb, g0, g1, g2, sh) rows in any
    order — the partition asserts make the order irrelevant.  Every
    cfg-derived parameter (and the config consistency asserts) is
    recomputed from ``cfg`` here.  Dyadic alignment (lo = 0 mod 2^bb) is
    asserted explicitly; circuit.py enforces it structurally instead
    (tree_from_leafset descends only aligned (lo, bb) pairs).  Returns
    (addend, worst): the certified addend over [LO, HI) and the worst
    centered error.
    """
    ndn, guard, rprod = cfg["NDN"], cfg["GUARD"], cfg["RPROD"]
    g1w, tau, rq, g2w = cfg["G1W"], cfg["TAU"], cfg["RQ"], cfg["G2W"]
    j0, c0d, qcap = cfg["J0"], cfg["C0DROP"], cfg["QCAP"]
    b4, vord = cfg["B4"], cfg["VORD"]
    vw = g1w + guard
    bias = 1 << qcap
    glim = (1 << g1w) - (bias >> guard)
    win = 1 << (qcap + 1)
    sig = [b4[t] + rprod - j0 - 2 * t for t in range(7)]
    assert qcap + 1 <= vw and qcap >= rprod and len(b4) == 7
    assert sorted(vord) == [0, 1, 2]

    def leaf_dn(lo, bb):
        w = 1 << bb
        sh = ndn - bb
        i0 = lo - LO
        xs = XS[i0 : i0 + w]
        fixed = (lo & ((1 << ndn) - 1)) >> bb
        dn = (((xs & (w - 1)) << sh) | fixed).astype(np.int64)
        return dn, sh, i0, w

    def q_row(dn, e, s):
        """Curvature digit row s: e (dq << 2s) >> RQ on dq = dn >> TAU,
        the |e|=2 candidate one dq bit deeper (magnitude folded into the
        shift before truncation — the pair-select the circuit wires).
        """
        if e == 0:
            return np.zeros_like(dn)
        dq = dn >> tau
        return np.sign(e) * ((dq << (2 * s + (abs(e) == 2))) >> rq)

    def model_P(dn, v):
        """Product: 7 signed Booth digit rows of Y = dn >> J0; row t adds
        d_t (v >> sig_t) << B4[t], the |d|=2 candidate one v bit deeper.
        """
        y = dn >> j0
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

    addend = np.zeros(HI - LO, dtype=np.int64)
    cover = np.zeros(HI - LO, dtype=bool)
    worst = 0.0
    for lo, bb, g0, g1, g2, shv in leaves:
        lo, bb, g0, g1, g2, shv = map(int, (lo, bb, g0, g1, g2, shv))
        assert lo % (1 << bb) == 0, (lo, bb)  # dyadic alignment
        dn, sh, i0, w = leaf_dn(lo, bb)
        assert shv == sh and 0 <= sh <= 7
        assert g0 % (1 << c0d) == 0 and 0 <= g0 < (1 << 32)
        assert 1 <= g1 < glim and 0 <= g2 < (1 << g2w)
        assert not cover[i0 : i0 + w].any(), "leaf overlap"
        cover[i0 : i0 + w] = True
        # The window invariant, walked in the frozen VORD emission order:
        # after each curvature row, 0 <= wlow - S < 2^17 on every dn.
        # The final row's bound licenses the circuit's 17-bit chains;
        # the per-prefix form is the stronger invariant the frozen table
        # satisfies.  W mod 2^17, not just 2^16 — bit 12 of g1 can wrap
        # the window, so the exact modular form is required.
        ee = booth_digits(g2, 3)
        wlow = ((g1 << guard) + bias) % win
        Sp = np.zeros_like(dn)
        for s in vord:
            Sp = Sp + q_row(dn, ee[s], s)
            assert int(Sp.max()) <= wlow and int(Sp.min()) > wlow - win, (lo, bb, s)
        Q = Sp
        v = (np.int64(g1) << guard) + bias - Q
        assert int(v.min()) >= 0 and int(v.max()) < (1 << vw)
        compv = ((dn >> j0) << j0) * (1 << (qcap - rprod))
        addend[i0 : i0 + w] = (np.int64(g0) - model_P(dn, v) + compv) % MOD
        e = addend[i0 : i0 + w].astype(np.float64) - FF[i0 : i0 + w]
        e = np.mod(e + 2.0**32, 2.0**33) - 2.0**32
        worst = max(worst, float(np.abs(e).max()))
    assert cover.all(), "partition gap"
    assert worst <= ERRLIM + 0.25, worst
    return addend, worst


def main():
    addend, worst = certify_table(CFG, ((lf.lo, lf.bb, *lf.coef) for lf in LEAVES))
    e = np.abs(np.mod(addend.astype(np.float64) - FF + 2.0**32, 2.0**33) - 2.0**32)
    assert int(LO + int(np.argmax(e))) == 1886892
    assert abs(worst - 427.15) <= 0.25, worst
    print(
        f"table certified: leaves={len(LEAVES)} worst={worst:.2f} "
        f"at X={LO + int(np.argmax(e))} (band {EPS})"
    )


if __name__ == "__main__":
    main()
