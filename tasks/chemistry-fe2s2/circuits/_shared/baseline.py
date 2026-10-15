"""Circuit stages of the chemistry-fe2s2 baseline circuits/babbush-composed.
circuits/discovered reads only the pinned table (DATA, MASKS).

The baseline adds the alias-word lookup and the SELECT walk. This module
holds the other stages: the table loader and its shape asserts, the
[BAB] Sec. III.D comparator and multiplexor, the [BER] App. C word
unlookup, the dense SELECT leaf, and the erasure tail.
[BAB] is Babbush et al. (2018), arXiv:1805.03662; [BER] is Berry et al.
(2019), arXiv:1902.02134.

Loading contract: each circuit loads this file by path from its own
``__file__`` with ``uncompute.testing.load_module``. Every load is a
private instance; nothing registers in ``sys.modules``.
"""

import json
import lzma

from uncompute.gadgets import erase_register, walk, xor_word
from uncompute.testing import task_dir

L, A, S, X, B = 41052, 16, 16, 40, 34
XMASK = (1 << X) - 1
TABLE_XZ = task_dir(__file__) / "data" / "table.json.xz"


def load_tables():
    t = json.loads(lzma.decompress(TABLE_XZ.read_bytes()))
    assert t["L"] == L and t["n_qubits"] == X
    words = [int(r["word"], 16) for r in t["terms"]]
    masks = []
    for r in t["terms"]:
        flip = zy = ny = 0
        for tok in r["pauli"].split():
            p, q = tok[0], int(tok[1:])
            if p in "XY":
                flip |= 1 << q
            if p in "ZY":
                zy |= 1 << q
            if p == "Y":
                ny += 1
        assert ny % 2 == 0
        masks.append(flip | (zy << X) | (((ny % 4) // 2) << 80))
    return words, masks


DATA, MASKS = load_tables()


# Table shape — asserted, not just measured: the lone keep=0 row
# aliases itself; every alt stays inside the graded domain, so sel
# stays in the SELECT walk's window; the minimum word popcount (every
# fixup fold gets >= 2 outcomes, so no fix entry is None); the keep
# ceiling (sigma = 2^16 - 1 yields f = 0 on every row); the theta_alt
# prefetch identity (the f=0 branch reads row i's bit 32 as theta_sel,
# so it must equal row alt_i's own sign bit 33).
# Packed fields: DATA[i] packs keep | alt<<16 | theta_alt<<32 | theta<<33;
# MASKS[i] packs flip | zy<<40 | ny2<<80.
_zk = [i for i in range(L) if (DATA[i] & 0xFFFF) == 0]
assert len(_zk) == 1 and ((DATA[_zk[0]] >> 16) & 0xFFFF) == _zk[0]
assert max((DATA[i] >> 16) & 0xFFFF for i in range(L)) < L
assert min(bin(w).count("1") for w in DATA) == 5
assert max(DATA[i] & 0xFFFF for i in range(L)) == 65535
assert all(((DATA[i] >> 32) & 1) == ((DATA[(DATA[i] >> 16) & 0xFFFF] >> 33) & 1) for i in range(L))


# ---- [BAB] Sec. III.D: strict borrow-recurrence comparator (cmp/fwd) ------
def compare(b, sigma, keep_bits):
    carries = []
    for k in range(S):
        prev = carries[-1] if carries else None
        if prev is not None:
            b.cx(prev, sigma[k])
            b.cx(prev, keep_bits[k])
        b.x(sigma[k])
        t = b.and_compute(sigma[k], keep_bits[k])
        b.claim("carry", [t])
        if prev is not None:
            b.cx(prev, t)
        carries.append(t)
    return carries


# ---- [BAB] Sec. III.D: keep ? i : alt multiplexor; theta into the phase ---
def select(b, idx, word, sel_bits, flag):
    for j in range(A):
        b.cx(word[16 + j], sel_bits[j])
        b.cx(idx[j], word[16 + j])
        b.ccx(flag, word[16 + j], sel_bits[j])
        b.cx(idx[j], word[16 + j])
    # theta_sel is not an output bit: the same mux into a scratch
    # ancilla, bare Z folds (-1)^theta_sel into the phase, exact uncompute.
    th = b.qubit()
    b.claim("theta", [th])
    b.cx(word[32], th)
    b.cx(word[33], word[32])
    b.ccx(flag, word[32], th)
    b.cx(word[33], word[32])
    b.z(th)
    b.cx(word[33], word[32])
    b.ccx(flag, word[32], th)
    b.cx(word[33], word[32])
    b.cx(word[32], th)
    b.release_recycle("theta", [th])


# ---- [BAB] Sec. III.D: comparator unwind, measurement-based (cmp/rev) -----
def uncompare(b, sigma, keep_bits, carries):
    for k in range(S - 1, -1, -1):
        prev = carries[k - 1] if k else None
        if prev is not None:
            b.cx(prev, carries[k])
        b.release("carry", [carries[k]])
        b.and_release(carries[k], sigma[k], keep_bits[k])
        b.x(sigma[k])
        if prev is not None:
            b.cx(prev, keep_bits[k])
            b.cx(prev, sigma[k])


# ---- [BER] App. C: measurement-based unlookup, window k' = ku -------------
def unlookup(b, idx, word, ku):
    # The pinned table's minimum word popcount is 5 (asserted at load),
    # so every row folds >= 2 fresh Outcomes and no fix entry is None —
    # every in-promise address gets its conditioned CZ.
    b.release("word", word)
    erase_register(
        b,
        word,
        idx,
        L,
        DATA,
        ku,
        range(L),
        "unlookup/measure",
        free=True,
        claim_set="window",
        phase="unlookup/window",
    )


# ---- [BAB] Secs. III.B/IV.A: SELECT leaf (all Cliffords) ------------------
def select_pauli(b, sel_bits, x):
    def leaf(ctrl, iv):
        m = MASKS[iv]
        for j in range(X):  # Z-part first: CZ sees the INPUT x values
            if (m >> (X + j)) & 1:
                b.cz(ctrl, x[j])
        xor_word(b, ctrl, m & XMASK, x)  # then the X-part flips
        if (m >> 80) & 1:  # constant sign (-1)^ny2 of the string
            b.z(ctrl)

    walk(b, sel_bits, None, A - 1, 0, 1 << A, 0, L, leaf)


def erase(b, idx, sigma, word, sel, flag, ku):
    """Round-trip erasure: sel by demolition against the live alias
    word ([BER] App. C kickback, resolved wire by wire: <m, sel> =
    <m, alt> ^ f.<m, i^alt>, all outcome-conditioned Z/CZ), then the
    f copy against the keep word by a second comparator pass, then the
    word by :func:`unlookup` with window ``ku``.
    """
    b.set_phase("erase/sel")
    for t in range(A):
        m = b.hmr(sel[t])
        b.release_recycle("sel", [sel[t]])
        b.z(word[16 + t], cond=m)
        b.cz(flag, idx[t], cond=m)
        b.cz(flag, word[16 + t], cond=m)
    b.set_phase("erase/flag")
    carries = compare(b, sigma, word[:16])
    b.cx(carries[-1], flag)
    uncompare(b, sigma, word[:16], carries)
    b.release_recycle("flag", [flag])
    unlookup(b, idx, word, ku)
