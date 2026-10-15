#!/usr/bin/env python3
"""modinv-point-add: the classical reference oracle.

This module is the classical model of the task map |P⟩ → |P+Q⟩: affine point
addition on y² = x³ + 7 mod p (the a = 0 short-Weierstrass case). The addend
Q is a per-shot classical input (the Bit-kind registers 2 and 3). Circuit
checks may load this module by path and compare their models against it.
grader.py stays the authority: it derives every value here independently
with its own Jacobian oracle, and it asserts contract.json at import. The
group arithmetic is the affine group law below (ec_add, ec_mul). Every
modular inverse is ``pow(x, -1, p)``.

Exports:

  P_MOD, ORDER, CURVE_B, GX, GY    the curve domain (contract.json)
  CONTRACT                         the raw contract.json dict
  curve_add(x1, y1, x2, y2, p)     the grader's expected-output map,
                                   restricted to the chord case
                                   x1 != x2 (the only case the
                                   protocol draws; asserted)
  curve_mul(k, p)                  k·G by double-and-add
  point_add_model(x1, y1, x2, y2, p)
                                   value-level chord-addition schedule
                                   (no control): the group law
                                   computed through
                                   λ = (y1−y2)/(x1−x2), written as the
                                   in-place register walk that the
                                   circuits follow; asserts the two
                                   exceptional chains (x1 == x2 at
                                   the IDiv, x3 == x2 at the IMul)
                                   instead of modeling them

point_add_model == curve_add is a pointwise identity wherever both are
defined. The identity holds for arbitrary residue tuples, not only for
points on the curve, so reduced-width checks may use either at any odd
prime p.

Registers (README §1.1): coordinates are 256-bit little-endian canonical
residues mod p. The map has no control. The protocol never draws either
operand as the identity, P = −Q, or P.x = Q.x.

Run this file directly for the self-check: contract round-trip, curve
membership and group-order anchors, and sweeps of model == group law (full
width and reduced primes).
"""

import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve()

CONTRACT = json.loads((HERE.parent / "contract.json").read_text())
P_MOD = int(CONTRACT["p"], 16)
ORDER = int(CONTRACT["n_order"], 16)
CURVE_B = CONTRACT["curve_b"]
GX = int(CONTRACT["gx"], 16)
GY = int(CONTRACT["gy"], 16)


# ---- affine group law: points (x, y), None for the identity ----------------


def ec_add(p1, p2, a_coef, p):
    """Group law: p1 + p2 on y^2 = x^3 + a_coef*x + b mod p."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if x1 == x2 and y1 == y2:
        lam = (3 * x1 * x1 + a_coef) * pow(2 * y1, -1, p) % p
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)


def ec_mul(pt, k, a_coef, p):
    """Scalar multiple k * pt by double-and-add."""
    res = None
    acc = pt
    while k > 0:
        if k & 1:
            res = ec_add(res, acc, a_coef, p)
        acc = ec_add(acc, acc, a_coef, p)
        k >>= 1
    return res


def curve_add(x1, y1, x2, y2, p=P_MOD):
    """Apply the grader's expected-output map restricted to the chord case
    x1 != x2 (the only case the protocol draws): the shared group law
    (ec_add with a = 0), domain restriction asserted.
    """
    assert x1 != x2
    return ec_add((x1, y1), (x2, y2), 0, p)


def curve_mul(k, p=P_MOD):
    """k·G (for generating protocol-like test points)."""
    return ec_mul((GX, GY), k, 0, p)


def point_add_model(x1, y1, x2, y2, p=P_MOD):
    """Value-level model of the chord schedule (ctrl dropped)."""
    x = (x1 - x2) % p
    y = (y1 - y2) % p
    assert x != 0
    lam = (y * pow(x, -1, p)) % p  # IDiv
    A = (lam * lam) % p  # squ
    x = (x - A) % p
    x = (x + 3 * x2) % p  # now -(x3 - x2)
    assert x != 0, "x3 == x2 exceptional chain"
    y = (x * lam) % p  # IMul -> y3 + y2
    x = (p - x) % p
    x = (x + x2) % p
    y = (y - y2) % p
    return x, y


def selfcheck():
    # -- contract round-trip: the derived constants re-derive ------------
    assert P_MOD == (1 << 256) - (1 << 32) - 977
    # -- curve membership + subgroup order -------------------------------
    x, y = GX, GY
    assert (y * y - (x * x * x + CURVE_B)) % P_MOD == 0
    assert curve_mul(ORDER) is None  # n*G = identity
    # -- group-law anchors + model == law on protocol-like points --------
    assert point_add_model(*curve_mul(1), *curve_mul(2)) == curve_mul(3)
    assert point_add_model(*curve_mul(2), *curve_mul(1)) == curve_mul(3)
    rng = random.Random(11)
    pairs = 0
    while pairs < 25:
        s, t = rng.randrange(1, ORDER), rng.randrange(1, ORDER)
        Pa, Pb = curve_mul(s), curve_mul(t)
        if Pa[0] == Pb[0]:
            continue
        want = curve_mul((s + t) % ORDER)
        assert point_add_model(*Pa, *Pb) == want
        assert curve_add(*Pa, *Pb) == want
        pairs += 1
    # -- pointwise identity on arbitrary residue tuples, reduced p -------
    checked = 0
    for p in (13, 61, 251):
        for _ in range(600):
            x1, y1, x2, y2 = (rng.randrange(p) for _ in range(4))
            try:
                want = point_add_model(x1, y1, x2, y2, p)
            except AssertionError:  # an exceptional chain
                continue
            assert want == curve_add(x1, y1, x2, y2, p)
            checked += 1
    assert checked > 1000, checked
    print(
        f"modinv-point-add reference: ALL CHECKS PASSED\n"
        f"  anchors         : G on curve; n*G = identity; "
        f"G+2G / 2G+G KATs\n"
        f"  model == law    : {pairs} random s*G + t*G pairs at "
        f"n = 256; {checked} arbitrary tuples at p = 13/61/251"
    )


if __name__ == "__main__":
    selfcheck()
