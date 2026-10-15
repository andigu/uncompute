"""dqi-reed-solomon reference oracle: syndrome (70 GF(2^12) symbols) -> expected sign.

This file transcribes the synchronized-EEA syndrome decoder in the code
that Khattar et al. (2025), arXiv:2510.10967, released on Zenodo
(doi:10.5281/zenodo.17301475) under the CC BY 4.0 license
(https://creativecommons.org/licenses/by/4.0/).  The authors are
T. Khattar, N. Shutty, C. Gidney, A. Zalcman, N. Yosri, D. Maslov,
R. Babbush, and S. P. Jordan.  The transcription follows the released
`_poly_eea_zalka_gcd_helper` machine and the direct Chien/Forney sweep over
gamma^0 .. gamma^4094.  File:line references point into that release.
Changes from the release: division uses FLT semantics (x/0 = 0), the stop
rule is n_B < n/2 + 1 (so the decoder also handles weight exactly n/2), and
the code uses the Python standard library only.  Nothing here imports from
circuits/.

API:
  decode(s70)  -> list of 4095 error symbols, or None if off-promise
  phase(s70)   -> <c, e(s)> in {0, 1}

Import-time self-checks: plant-and-decode on the grader's anchor shapes
(zero error, the three fixed single-symbol errors) plus random weights.
"""

import hashlib
import random

M = 4095  # error positions / evaluation points
N_SYND = 70  # syndromes S_1 .. S_70
B = 12  # bits per GF(2^12) symbol
GF_POLY = 4331  # x^12+x^7+x^6+x^5+x^3+x+1, gamma = x = 2
C_INSTANCE_TAG = b"kickmix/rs-decode-4095/fixup-c/v1"  # fixed instance constant (fixup vector c)
N_CYCLES = 3 * (N_SYND + 1) + 9  # 222 machine cycles
MPRIME = N_SYND + 3  # 73 slots per array


# ---------------------------------------------------------------------------
# GF(2^12): exp/log tables, FLT inversion (0 -> 0)
# ---------------------------------------------------------------------------


def _build_tables():
    exp = [0] * M
    log = [0] * (M + 1)
    acc = 1
    for i in range(M):
        assert i == 0 or acc != 1, "gamma = 2 must be primitive"
        exp[i], log[acc] = acc, i
        acc <<= 1
        if acc >> B:
            acc ^= GF_POLY
    assert acc == 1
    return exp, log


_EXP, _LOG = _build_tables()


def _mul(x, y):
    if x == 0 or y == 0:
        return 0
    return _EXP[(_LOG[x] + _LOG[y]) % M]


def _div(x, y):
    """x / y with FLT semantics: x/0 = 0."""
    if x == 0 or y == 0:
        return 0
    return _EXP[(_LOG[x] - _LOG[y]) % M]


def _gamma_pow(k):
    return _EXP[k % M]


# ---------------------------------------------------------------------------
# The synchronized EEA machine (poly_eea_zalka.py:141-252)
# ---------------------------------------------------------------------------


class _State:
    __slots__ = (
        "a",
        "b",
        "counter",
        "flag",
        "hc",
        "n_A",
        "n_B",
        "n_a",
        "n_b",
        "n_q",
    )


def _init(s):
    """Build the initial state: A = syndrome poly top-aligned in a, B = z^n in b (a = 1, b = 0)."""
    deg = max((k for k in range(N_SYND) if s[k]), default=0)
    st = _State()
    st.a = [0] * MPRIME
    st.b = [0] * MPRIME
    st.a[0] = 1
    for t in range(deg + 1):
        st.a[MPRIME - 1 - deg + t] = s[t]
    st.b[MPRIME - 1] = 1
    st.n_A, st.n_B, st.n_a, st.n_b, st.n_q = deg + 1, N_SYND + 1, 1, 0, 0
    st.flag, st.counter, st.hc = 1, 0, 0
    return st


def _cycle(st):
    m = MPRIME
    live = st.hc == 0
    # 1. UpdateFlag
    c = st.counter
    if c == 0:
        first, last = st.n_q == 0 and st.n_A < st.n_B, st.n_A == st.n_B
    elif c == 1:
        first, last = st.n_A == st.n_B, st.n_B == 1 or st.b[m - 2] != 0
    elif c == 2:
        first, last = st.b[0] != 0 or st.n_b == 0, st.n_a == st.n_b + 1
    else:
        first, last = st.n_a == st.n_b, st.n_q == 1
    if live:
        st.flag ^= int(first) ^ int(last)
    # 2. MultiplyAndAddOrSub (add == sub == XOR)
    q = _div(st.b[m - 1], st.a[m - 1]) if c == 0 and live else 0
    for i in range(m - 1, -1, -1):
        if live and ((c == 0 and i == st.n_a + st.n_q) or (c == 3 and i + 1 == st.n_a + st.n_q)):
            q, st.a[i] = st.a[i], q
        if (c == 3 and i < st.n_a) or (c == 0 and i >= m - st.n_A):
            st.b[i] ^= _mul(q, st.a[i])
    # (a nonzero q residue after counter 3 stays in q; the machine state
    # carries it forward)
    # 3. ShiftBArrayRight
    ctrl_nB = c == 1 or (c == 0 and not st.flag)
    for i in range(m - 1, -1, -1):
        do = (st.n_B >= m - i + 1) and ctrl_nB
        do ^= (0 < i <= st.n_b) and c == 2
        do ^= (0 < i < m - st.n_B) and not st.flag and c == 3
        if do and live:
            st.b[i - 1], st.b[i] = st.b[i], st.b[i - 1]
    # 4. UpdateMetadata
    if live:
        if c == 0:
            st.n_q += 1
            st.n_B -= not st.flag
        elif c == 1:
            st.n_B -= 1
        elif c == 2:
            st.n_b += 1
        else:
            st.n_b += not st.flag
            st.n_q -= 1
    # 5. SwapAandB
    if c == 3 and st.flag and live:
        st.n_A, st.n_B = st.n_B, st.n_A
        st.n_a, st.n_b = st.n_b, st.n_a
        st.a, st.b = st.b, st.a
    # 6. UpdateHaltingCounter (not gated on hc == 0; fixed stop rule)
    if c == 3 and st.flag and (st.n_A == 0 or st.n_B < N_SYND // 2 + 1):
        st.hc += 1
    st.counter = (c + st.flag) % 4


# ---------------------------------------------------------------------------
# Chien/Forney sweep (rs_code_egcd_for_paper.py:53-78)
# ---------------------------------------------------------------------------


def _sweep(st):
    """Point i evaluates at gamma^i and writes error position (m - i) % m;
    Forney constant gamma^-(m'-1 + n_b - n_B - 1) per point.
    """
    expn = MPRIME - 1 + st.n_b - st.n_B - 1
    half = MPRIME // 2
    errors = [0] * M
    for i in range(M):
        om = sg = sgp = 0
        for k in range(MPRIME - 1, half, -1):
            om ^= _mul(st.b[k], _gamma_pow(i * k))
        for k in range(half, -1, -1):
            t = _mul(st.b[k], _gamma_pow(-i * k))
            sg ^= t
            if k & 1:
                sgp ^= t
        if sg == 0:
            errors[(M - i) % M] = _mul(_gamma_pow(-i * expn), _div(om, sgp))
    return errors


# ---------------------------------------------------------------------------
# The fixup mask and the public API
# ---------------------------------------------------------------------------


def _derive_mask_bits(nbits):
    """Counter-mode SHA-256 XOF: block k = SHA256(tag || u32le(k)); bit t
    of the stream = bit t % 8 of byte t // 8.
    """
    nbytes = (nbits + 7) // 8
    stream = b""
    k = 0
    while len(stream) < nbytes:
        stream += hashlib.sha256(C_INSTANCE_TAG + k.to_bytes(4, "little")).digest()
        k += 1
    return [(stream[t >> 3] >> (t & 7)) & 1 for t in range(nbits)]


_MASK_BITS = _derive_mask_bits(M * B)
_MASK_INT = sum(b << i for i, b in enumerate(_MASK_BITS))
_MASK_SYM = [(_MASK_INT >> (B * j)) & ((1 << B) - 1) for j in range(M)]  # c_j


def decode(s70):
    """70 syndrome symbols -> 4095 error symbols (or None if off-promise)."""
    if len(s70) != N_SYND or any(not 0 <= v < 4096 for v in s70):
        raise ValueError("need 70 GF(2^12) symbols")
    st = _init(list(s70))
    for _ in range(N_CYCLES):
        _cycle(st)
    if st.hc == 0:  # the machine did not halt within its cycle budget
        return None
    return _sweep(st)


def phase(s70) -> int:
    e = decode(s70)
    if e is None:
        raise ValueError("off-promise syndrome: no error of symbol weight <= 35")
    return sum((c & v).bit_count() for c, v in zip(_MASK_SYM, e)) & 1


# ---------------------------------------------------------------------------
# Import-time self-asserts (fast: 6 decodes)
# ---------------------------------------------------------------------------


def _syndromes(e):
    """S_i = sum_j e_j gamma^{i j}, i = 1..70 (list index k holds S_{k+1})."""
    s = [0] * N_SYND
    for j, v in enumerate(e):
        if v:
            for i in range(1, N_SYND + 1):
                s[i - 1] ^= _EXP[(_LOG[v] + i * j) % M]
    return s


def _planted(posns, vals):
    e = [0] * M
    for j, v in zip(posns, vals):
        e[j] = v
    return e


def _selfcheck():
    # zero error (grader anchor shot 0)
    assert phase([0] * N_SYND) == 0
    # the grader's three pinned single-symbol anchors
    for pos, val in ((0, 1), (4094, 4095), (2047, 2049)):
        e = _planted([pos], [val])
        s = _syndromes(e)
        assert decode(s) == e, f"anchor ({pos},{val}) decode mismatch"
        assert phase(s) == (_MASK_SYM[pos] & val).bit_count() & 1
    # random weights (dominant law region + boundary)
    rng = random.Random(20260716)
    for w in (32, 35):
        posns = rng.sample(range(M), w)
        vals = [rng.randrange(1, 4096) for _ in range(w)]
        e = _planted(posns, vals)
        assert decode(_syndromes(e)) == e, f"weight-{w} decode mismatch"


_selfcheck()
