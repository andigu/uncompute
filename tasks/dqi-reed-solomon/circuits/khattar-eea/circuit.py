"""dqi-reed-solomon khattar-eea: in-place forward EEA -> sweep -> adjoint.

Executable reconstruction of the Reed-Solomon DQI phase-fixup decoder of
Khattar et al. (2025), arXiv:2510.10967, at their instance
(m, n, b) = (4095, 70, 12).  The circuit transliterates the decoder in the
authors' code release (Zenodo, doi:10.5281/zenodo.17301475), which is
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
README.md gives the attribution; comments below mark each adaptation.  The
instance is loaded from the task contract.json and pinned below.
Certified cost: the meta.json pin, asserted by build().
The stream is loop-form: the 222 cycles, the 4,095 sweep points and the
222 reversed cycles are one op-stream repeat body each.

Layout (flat):
  circuit.py         the circuit, in sections: the instance and field, the
                    GF(2^12) gadgets (the extracted multiplier loaded from
                    constants.json and digest-checked), one synchronized
                    EEA cycle, the forward pass, the Chien/Forney sweep,
                    and the tape adjoint with build().
  constants.json    the extracted gate list of the released 51-Toffoli
                    GF(2^12) multiplier.
  gen_constants.py  extracts the gate list from the release and writes
                    constants.json.

RUN: uv run python circuit.py -- builds, asserts its cost, and
streams to $UNCOMPUTE_OPS_PATH (else ./ops.bin).  Importing is cheap and
builds nothing; build() feeds the occupancy/liveness profilers.
Evaluate:
python tools/eval.py dqi-reed-solomon --circuit
tasks/dqi-reed-solomon/circuits/khattar-eea/circuit.py.

README.md in this folder describes the construction.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

from uncompute import Builder
from uncompute.classical import GF2n, f2_rows_of, f2_transpose, parity
from uncompute.gadgets.arith import eq_const, gidney_add, gidney_sub, release_chain
from uncompute.gadgets.linear import linear_xor
from uncompute.gadgets.wires import declare_registers
from uncompute.ops import NO_BIT, NO_OUT, NO_QUBIT, NO_REG, OpKind
from uncompute.testing import (
    assert_metrics,
    emit_and_report,
    load_contract,
)

_HERE = pathlib.Path(__file__).resolve().parent

# =========================================================================
# Instance and field (the classical specification the circuit needs)
# =========================================================================
# The task instance (single source, read by every circuit section):
# the contract.json the grader asserts, loaded fail-closed.
_CONTRACT = load_contract(__file__)
GF_B = int(_CONTRACT["b"])  # extension degree
GF_MOD = int(_CONTRACT["gf_poly"])  # galois default irreducible polynomial
N = int(_CONTRACT["n_synd"])  # syndrome symbols S_1 .. S_n
M = int(_CONTRACT["m"])  # error positions / evaluation points, 2^b - 1
W_MAX = int(_CONTRACT["w_max"])  # graded symbol weight, n/2
assert M == (1 << GF_B) - 1 and W_MAX == N // 2, "contract instance shape"

GF = GF2n(GF_B, GF_MOD)  # the task field, gamma = x = 2 (galois default)
# The fixup vector c: bit t of the contract's c stream (little-endian
# within each byte). The grader asserts the stream against its SHA-256
# XOF, and indexing past its end raises.
_C_STREAM = bytes.fromhex(_CONTRACT["c_stream_hex"])
C_BITS = [(_C_STREAM[t >> 3] >> (t & 7)) & 1 for t in range(M * GF_B)]


def n_eea_steps(n: int) -> int:
    """3(n+1) + 5 + 4 cycles (rs_code_egcd_for_paper.py:49); 222 for n = 70."""
    return 3 * (n + 1) + 9


# =========================================================================
# GF(2^12) gadgets
# =========================================================================

# A field element is a little-endian list of GF_B qubits: reg[i] holds the
# coefficient of x^i.  Addition is XOR (CX).  Every gadget takes a Builder
# first and operand registers as lists of qubit ids.  The multiplier
# gf_mult_opt replays the release's own gate list from constants.json and
# is pure CX/CCX, so it is phase-clean.  gf_mult_uncompute erases a product
# with 0 Toffoli and pays its ANF as measurement-conditioned CZ phases.
# The inversion pair gf_inverse_keep / gf_inverse_unkeep has FLT semantics
# (0 -> 0).
#
# local: uncompute.gadgets.gf_tower covers only the tower fields GF(q0^2),
# q0 in {4, 8}, so the GF(2^12) word gadgets live here.


# The released decoder's quantum-quantum multiply: GF2MulViaOptimizedPolyMul,
# built from the Maslov-optimized circuits shipped in the Zenodo release
# (b = 12: circuits/poly_mul_6.txt, 3 x 17 = 51 Toffoli).  Its flattened
# gate list is constants.json beside this file (extracted by
# gen_constants.py).  The gate
# list must be the one extracted for THIS field width: its result
# allocation names exactly the out wires 2b .. 3b-1 (fail closed on a list
# extracted for another b).
OPTMUL = json.loads((_HERE / "constants.json").read_bytes())
# Gate tuples: ("cx", c, t) | ("ccx", c0, c1, t) | ("alloc_result", wires).
OPTMUL_GATES: tuple[tuple, ...] = tuple(
    (g[0], tuple(g[1])) if g[0] == "alloc_result" else tuple(g) for g in OPTMUL["gates"]
)
OPT_MULT_TOFFOLI = sum(g[0] == "ccx" for g in OPTMUL_GATES)
assert any(
    g[0] == "alloc_result" and g[1] == tuple(range(2 * GF_B, 3 * GF_B)) for g in OPTMUL_GATES
), f"constants.json is not a GF(2^{GF_B}) gate list"


# ---------------------------------------------------------------------------
# Classical helpers (ground truth for the CX/CCX wiring below)
# ---------------------------------------------------------------------------


def _bits(v: int) -> list[int]:
    """Return the indices of set bits of v, ascending."""
    return [i for i in range(v.bit_length()) if (v >> i) & 1]


# MULT_PAIRS[k] = ordered pairs (i, j) such that the ANF of output bit k of the
# product x*y contains the quadratic term x_i * y_j.  Because x and y are
# distinct registers, (i, j) and (j, i) are distinct terms and both appear.
MULT_PAIRS: list[list[tuple]] = [[] for _ in range(GF_B)]
for _i in range(GF_B):
    for _j in range(GF_B):
        for _k in _bits(GF.pow(2, _i + _j)):  # x^(i+j) mod p
            MULT_PAIRS[_k].append((_i, _j))
del _i, _j, _k

# _FROB_ROWS[k] = the rows of the x -> x^(2^k) matrix, memoized per
# Frobenius power so the emission does no field arithmetic.
_FROB_ROWS: dict[int, list[int]] = {}

# Operand-register validation is O(register size) of set bookkeeping per call
# — measurable across the ~10^5 gadget calls of a full build — so it is gated
# behind a module flag (set it to True to check operands; emission is
# identical either way).
VALIDATE_REGS = False


def _check_regs(*regs: Sequence[int]) -> None:
    """Each register must be GF_B distinct qubits; registers must be disjoint."""
    if not VALIDATE_REGS:
        return
    seen = set()
    for reg in regs:
        if len(reg) != GF_B:
            raise ValueError(f"register must have {GF_B} qubits, got {len(reg)}")
        for q in reg:
            if q in seen:
                raise ValueError(f"registers must be disjoint, qubit {q} repeats")
            seen.add(q)


def _macro(b, *token):
    """Builder seam for macro-level recording of a gadget call.

    A builder that provides ``macro(*token)`` (a context manager — see
    TapeBuilder) gets the whole gadget as ONE structural token; plain
    builders get a no-op context.  Emission is identical either way.
    """
    macro = getattr(b, "macro", None)
    return macro(*token) if macro is not None else nullcontext()


# Liveness metadata: phase marks and Builder.claim/release never emit an
# op.  Every non-AND allocation in this section joins a named wire set
# (gfw/frob Frobenius operands, gfw/junk kept chain products) and is
# released before its free.


# ---------------------------------------------------------------------------
# Quantum-quantum multiply (the released decoder's optimized replay) and its
# measured uncompute
# ---------------------------------------------------------------------------

# OPTMUL_GATES preprocessed once at import into integer-indexed rows
# over the flat 3b-slot operand list x[0..b-1] | y[0..b-1] | out[0..b-1]:
# (0, c, t) for CX and (1, c1, c2, t) for CCX, in emission order.  The
# alloc_result event binds the caller's |0> out register and emits nothing.


def _opt_rows():
    rows = []
    for g in OPTMUL_GATES:
        if g[0] == "cx":
            rows.append((0, g[1], g[2]))
        elif g[0] == "ccx":
            rows.append((1, g[1], g[2], g[3]))
        else:
            assert g[0] == "alloc_result"
    return tuple(rows)


_OPT_ROWS = _opt_rows()


def gf_mult_opt(b, x: Sequence[int], y: Sequence[int], out: Sequence[int]) -> None:
    """out = x * y over GF(2^b) — the released decoder's multiplier
    (OPT_MULT_TOFFOLI = 51 Toffoli at b = 12).

    Replays the flattened ``dqi...GF2MulViaOptimizedPolyMul(QGF(2, b))`` gate
    list from constants.json (Maslov-optimized poly_mul_6 circuits inside
    one-level Karatsuba; see its docstring for provenance and the
    Adjoint(GF2MulK) expansion used during extraction).  Zero workspace
    ancillas.

    PRECONDITION: ``out`` must be |0> (result wires may double as
    workspace, so the list does not accumulate; callers never rely on
    accumulation).  x and y are preserved; x, y, out must be
    pairwise-disjoint b-qubit registers.  Entirely CX/CCX: phase-clean by
    construction, every emitted Toffoli executes on every shot.  For x == y
    use :func:`gf_frobenius` with k=1.
    """
    _check_regs(x, y, out)
    with _macro(b, "mult", tuple(x), tuple(y), tuple(out)):
        w = list(x) + list(y) + list(out)
        for row in _OPT_ROWS:
            if row[0]:
                b.ccx(w[row[1]], w[row[2]], w[row[3]])
            else:
                b.cx(w[row[1]], w[row[2]])


def gf_mult_uncompute(
    b, out: Sequence[int], x: Sequence[int], y: Sequence[int], free_out: bool = True
) -> None:
    """Erase ``out`` holding exactly x*y, for **0 Toffoli** — measurement-
    based uncompute, the released bloq's declared GF2MulMBUCOpt adjoint.

    Each out[k] is hmr-measured into a classical bit m_k; conditioned on m_k,
    the ANF of that output bit — the pure-quadratic sum of x_i*y_j over
    MULT_PAIRS[k] — is paid back as CZ(x[i], y[j]) phases (Cliffords, free).
    x and y must still hold the compute-time operand values; sloppy timing
    leaves phase garbage on ~half the shots and fails the run.

    Circuit-independence: the CZ fixup is a function of the *state* being
    erased (out == x*y), not of the circuit that produced it — the hmr
    kickback on |out_k> = |f_k(x, y)> is Z^{m_k} on that function value, and
    paying f_k's ANF cancels it regardless of how f_k was computed.

    With ``free_out`` (default) the out qubits are freed in reverse order
    (LIFO-friendly); pass False to keep ownership of the (now |0>) qubits.
    """
    _check_regs(x, y, out)
    with _macro(b, "mule", tuple(out), tuple(x), tuple(y), free_out):
        for k in range(GF_B):
            m = b.hmr(out[k])
            for i, j in MULT_PAIRS[k]:
                b.cz(x[i], y[j], cond=m)
        if free_out:
            b.recycle_all(reversed(list(out)))


# ---------------------------------------------------------------------------
# Linear maps: constant multiply, Frobenius / squaring (pure CX, 0 Toffoli)
# ---------------------------------------------------------------------------


_FOLD_COLS_CACHE: dict[int, list[int]] = {}


def _fold_cols(c: int) -> list[int]:
    """Column i of the multiply-by-c matrix M_c (c * x^i mod p) as a
    bitmask over rows, memoized per constant.
    """
    cols = _FOLD_COLS_CACHE.get(c)
    if cols is None:
        cols = [GF.mul(c, 1 << i) for i in range(GF_B)]
        _FOLD_COLS_CACHE[c] = cols
    return cols


def gf_frobenius(b, x: Sequence[int], out: Sequence[int], k: int = 1) -> None:
    """out ^= x^(2^k).  The Frobenius map is GF(2)-linear: pure CX, 0 Toffoli.

    x^(2^k) = sum_i x_i * (x^i)^(2^k) = sum_i x_i * x^(i*2^k) mod p, so column
    i of the matrix is x^(i*2^k) mod p.  Calling twice uncomputes.
    """
    _check_regs(x, out)
    if k < 0:
        raise ValueError("k must be >= 0")
    rows = _FROB_ROWS.get(k)
    if rows is None:
        rows = _FROB_ROWS[k] = f2_rows_of(lambda v: GF.pow(v, 1 << k), GF_B)
    linear_xor(b, x, out, rows)


# ---------------------------------------------------------------------------
# Inversion via Itoh-Tsujii (FLT semantics: 0 -> 0), keep-junk realization
# ---------------------------------------------------------------------------
# The released GF2Inverse KEEPS its Itoh-Tsujii chain intermediates as RIGHT
# ``junk`` registers, and GF2DivisionUsingFLT threads them to its adjoint —
# which can then erase every product by measurement-based uncompute against
# the still-live operands (bilinear CZ fixups): a LEGITIMATELY 0-Toffoli
# adjoint, at the price of holding the junk (5 x 12 = 60 qubits here)
# between compute and uncompute.

# Addition chain on k for e_k = 2^k - 1 exponents: e_{a+b} = (e_a)^(2^b) * e_b,
# built by the released bloq's rule (qualtran GF2Inverse.build_composite_bloq,
# pinned commit; GF2InverseOpt only swaps in the optimized multiplier):
# k1 = bit_length(b-1) - 1 doublings 1 -> 2 -> 4 -> .. -> 2^k1, then one
# addition per remaining set bit of b-1, high to low.  The final free
# squaring yields x^(2^b - 2) = x^{-1} (and 0 -> 0).  At b = 12 (b-1 = 11 =
# 0b1011): 1 -> 2 -> 4 -> 8 -> 10 -> 11, five multiplications
# (k1 + popcount(b-1) - 1, the bloq's mul_count).


def itoh_tsujii_steps(b: int):
    """Return the released chain as (src, k, other, new) tuples over exponent indices
    (index j stands for x^(2^j - 1)): t_new = frob(t_src, k) * t_other with
    new = src + other and k = other (e_{a+c} = (e_a)^(2^c) * e_c).
    """
    e = b - 1
    k1 = e.bit_length() - 1
    steps = []
    idx = 1
    for _ in range(k1):  # doublings: e_{2i} = (e_i)^(2^i) * e_i
        steps.append((idx, idx, idx, 2 * idx))
        idx *= 2
    for j in range(k1 - 1, -1, -1):  # the remaining set bits of b-1, high to low
        if (e >> j) & 1:
            steps.append((idx, 1 << j, 1 << j, idx + (1 << j)))
            idx += 1 << j
    assert idx == e
    return tuple(steps)


ITOH_TSUJII_STEPS = itoh_tsujii_steps(GF_B)
ITOH_TSUJII_CHAIN = (1,) + tuple(s[3] for s in ITOH_TSUJII_STEPS)
ITOH_TSUJII_QQ_MULTS = len(ITOH_TSUJII_STEPS)
assert ITOH_TSUJII_QQ_MULTS == len(ITOH_TSUJII_CHAIN) - 1
# Chain validity: every step's operand indices are already available and
# the final free squaring doubles e_{b-1} to the inverse exponent 2^b - 2.
for _src, _k, _other, _new in ITOH_TSUJII_STEPS:
    assert _k == _other and _new == _src + _other
    assert _src in ITOH_TSUJII_CHAIN and _other in ITOH_TSUJII_CHAIN
del _src, _k, _other, _new
assert 2 * ((1 << ITOH_TSUJII_CHAIN[-1]) - 1) == (1 << GF_B) - 2


@contextmanager
def _frob_operand(b, src, k):
    """Yield f = frob(src, k) on a fresh ``gfw/frob`` register (pure CX),
    then replay it back to |0> and free it.
    """
    f = b.qubits(GF_B)
    b.claim("gfw/frob", f)
    gf_frobenius(b, src, f, k)
    yield f
    gf_frobenius(b, src, f, k)  # linear replay -> f back to |0>
    b.release_recycle("gfw/frob", f)


def _chain_step(b, src, k, other):
    """t = frob(src, k) * other on a fresh |0> register (one chain step).
    Returns the product register t (owned by the ``gfw/junk`` wire-set
    until :func:`_chain_release`).
    """
    with _frob_operand(b, src, k) as f:
        t = b.qubits(GF_B)
        b.claim("gfw/junk", t)
        gf_mult_opt(b, f, other, t)
    return t


def _chain_release(b, t, src, k, other):
    """Measured release of a chain product t == frob(src, k) * other.

    The Frobenius operand is recomputed so the CZ phase fixups see the
    compute-time values.  0 Toffoli; frees t.
    """
    with _frob_operand(b, src, k) as f:
        b.release("gfw/junk", t)
        gf_mult_uncompute(b, t, f, other)  # 0 Toffoli, frees t


def gf_inverse_keep(b, x: Sequence[int], out: Sequence[int]) -> list[list[int]]:
    """out ^= inv(x), KEEPING the chain intermediates.
    ITOH_TSUJII_QQ_MULTS x OPT_MULT_TOFFOLI Toffoli (5 x 51 = 255 at b = 12).

    Matches the released GF2InverseOpt (x^(2^b - 2), 0 -> 0):
    Itoh-Tsujii with the released chain ITOH_TSUJII_CHAIN (b = 12:
    1, 2, 4, 8, 10, 11 — t2 = x^3, t4 = x^15, t8 = x^255, t10 = x^1023,
    t11 = x^2047), one quantum-quantum multiply per step, each preceded by
    a free linear Frobenius, then out ^= t_{b-1}^2 (free); 0 -> 0 throughout.

    Returns junk = [t_k for each chain step] (ITOH_TSUJII_QQ_MULTS live
    b-qubit registers) which MUST be passed unchanged to
    :func:`gf_inverse_unkeep` while x and out still hold their compute-time
    values.
    """
    _check_regs(x, out)
    regs = {1: list(x)}
    junk = []
    for src, k, other, new in ITOH_TSUJII_STEPS:
        t = _chain_step(b, regs[src], k, regs[other])  # x^(2^k (2^src-1)) * x^(2^other-1)
        regs[new] = t
        junk.append(t)
    gf_frobenius(b, regs[GF_B - 1], out, 1)  # out ^= x^(2^b - 2) = x^{-1}
    return junk


def gf_inverse_unkeep(b, x: Sequence[int], out: Sequence[int], junk: list[list[int]]) -> None:
    """Erase out and the kept junk of :func:`gf_inverse_keep`.  0 Toffoli.

    out is un-XORed from t_{b-1} (linear replay); each intermediate is erased
    by measured uncompute against its still-live operands, newest first
    (LIFO-friendly).
    """
    assert len(junk) == ITOH_TSUJII_QQ_MULTS
    regs = {1: list(x)}
    for (_src, _k, _other, new), t in zip(ITOH_TSUJII_STEPS, junk):
        regs[new] = t
    gf_frobenius(b, regs[GF_B - 1], out, 1)  # out back to pre-compute value
    for (src, k, other, _new), t in reversed(list(zip(ITOH_TSUJII_STEPS, junk))):
        _chain_release(b, t, regs[src], k, regs[other])


# =========================================================================
# One synchronized EEA cycle
# =========================================================================

# One cycle of the Khattar et al. synchronized reversible polynomial EEA
# (_PolyEEAZalkaImplStep, poly_eea_zalka_bloqs.py:760-890 of the Zenodo
# release), transcribed stage by stage on their register layout with their
# leaf realizations.  The release's classical code is the semantic
# ground truth; the bloq wiring (file:line refs inline) is the structural one.
#
# Layout (in place on the syndrome): the n = 70 grader syndrome registers
# ARE a_array[3..72], the release's top-aligned remainder (the init
# stage right-shifts smaller degrees into place).  a_array[0..2] are fresh
# pads; the Bezout cofactor a packs up from a[0].  All m' = 73 b_array
# symbols are fresh (bb[72] ^= 1 seeds B = z^n).  The pass adjoint later
# restores the syndrome.  Counters n_A/n_B/n_a/n_b/n_q are QUInt(7) each
# (their quint_logm), plus flag 1, counter 2, halting_counter 8 (their
# log_nsteps), a 12-qubit quotient scratch q, and the Forney flag d2:
# 2*73*12 + 5*7 + 1 + 2 + 8 + 12 + 1 = 1,811 persistent qubits.  q is |0>
# at cycle boundaries except on a rare quotient residue.
#
# Every per-slot comparison against a classical slot index comes from a
# unary-iteration walker: a DFS over the 7-bit counter's trie that visits
# leaves 0 .. m' in order (1 Toffoli per left descent, 1 CX per sibling
# switch; subtrees above m' are pruned): 76 Toffoli per uncontrolled sweep,
# 77 with a control folded into the root.  Threshold windows ([r > i] etc.)
# are XOR-accumulations of the walker's exclusive leaf pulses (Cliffords
# only).  The window collapses rely on n_A, n_B, n_a, n_b, n_a + n_q <= m'
# at every cycle entry, a machine invariant that a seeded census of the
# classical decoder measured.
#
# UpdateFlag / ShiftBArrayRight / UpdateMetadata carry only a call graph in
# the release; they are realized here per their on_classical_vals, with
# per-stage counts pinned in STAGE_TOFFOLI.  UpdateFlag relies on the
# measured invariant counter == 0 implies n_A <= n_B
# (_stage1_update_flag).
#
# The StopEEA_RS constant is n/2 + 1 = 36, so the circuit also decodes
# weight exactly n/2 = 35 (2.8% of test shots); at weight <= 34 it agrees
# bit for bit with the constant n/2 = 35.


MPRIME = N + 3  # 73 array symbols (their m' = n + 3)
CTR_W = MPRIME.bit_length()  # 7: QUInt(bit_length(m')) — their quint_logm
N_EEA_CYCLES = n_eea_steps(N)  # 222 = 3(n+1) + 9 (rs_code_egcd_for_paper.py:49)
HC_W = (N_EEA_CYCLES - 1).bit_length()  # 8: ceil(log2 n_steps) — their log_nsteps
STOP_LT_CONST = N // 2 + 1  # 36: decodes weight exactly n/2 = 35 (section header)
# The unary walkers' leaf domain: the counters' reachable values 0 .. m'.
WALK_LEAVES = MPRIME + 1  # 74 addresses (the full 2^w trie iff m' = 2^w - 1)

PERSISTENT_QUBITS = 2 * MPRIME * GF_B + 5 * CTR_W + 1 + 2 + HC_W + GF_B + 1  # 1811 incl. d2
assert HC_W == len(bin(N_EEA_CYCLES - 1)) - 2 and (1 << HC_W) >= N_EEA_CYCLES
assert WALK_LEAVES <= (1 << CTR_W)


# ---------------------------------------------------------------------------
# Register file
# ---------------------------------------------------------------------------


@dataclass
class EEALayout:
    """Qubit ids of the persistent EEA machine registers.

    The composed build aliases the syndromes into a_array[3..m'-1]
    (section header); the release's b_array[i] <-> bb[i] and a_array[i] <-> a[i]
    (``st.b`` and ``st.a`` in reference.py), all symbols little-endian.  q is the persistent quotient scratch of stage 2.
    """

    a: list[list[int]]
    bb: list[list[int]]
    n_A: list[int]  # noqa: N815 - dqi counter name; n_a is a different counter
    n_B: list[int]  # noqa: N815 - dqi counter name; n_b is a different counter
    n_a: list[int]
    n_b: list[int]
    n_q: list[int]
    flag: int
    counter: list[int]  # 2 qubits LE (value = c0 + 2*c1)
    hc: list[int]
    q: list[int]
    d2: int  # [n_b - n_B == 2]: the Forney-exponent flag (emit_forney_flag)


# ---------------------------------------------------------------------------
# Gidney-AND micro-gadgets (negated literals ride Builder pol=: a False
# polarity is the X-conjugated control, recorded as ordinary X gates)
# ---------------------------------------------------------------------------


class _EqQQ:
    """[v == u] for equal-width registers: XOR wires + multi-AND of negations.

    len(v)-1 Toffoli, free release (the XOR wires are Clifford).
    """

    def __init__(self, b, v, u):
        self.b = b
        self.v, self.u = list(v), list(u)
        self.ds = []
        for i in range(len(v)):
            d = b.qubit()
            b.cx(v[i], d)
            b.cx(u[i], d)
            self.ds.append(d)
        b.claim("scr/eq", self.ds)
        self.res, self.trail = eq_const(b, self.ds, 0)

    def release(self):
        b = self.b
        release_chain(b, self.trail)
        b.release("scr/eq", self.ds)
        for i in range(len(self.ds) - 1, -1, -1):
            b.cx(self.u[i], self.ds[i])
            b.cx(self.v[i], self.ds[i])
            b.recycle(self.ds[i])


# Stage 6's constant comparator, stage 2's adder and stage 1's flag
# scratch ride a small replay tape.


class _Rev:
    """Records plain involutive ops + allocations for exact undo.

    ``own`` (a wire-set name) claims every allocation into that ownership
    set; :meth:`undo` releases them all before replaying the frees.

    local: Builder.checkpoint/mirror inverts the same regions and emits the
    same ops, but it appends the mirrored ops without ``_emit``, so the
    recording TapeBuilder would miss them and emit_adjoint would break.
    uncompute.bennett.Journal would need one thunk per gate.  The undo goes
    through ``b.x``/``b.cx``/``b.ccx``/``b.recycle``, so the TapeBuilder
    records it.
    """

    def __init__(self, b, own=None):
        self.b = b
        self.own = own
        self.tape = []

    def alloc(self):
        q = self.b.qubit()
        if self.own is not None:
            self.b.claim(self.own, [q])
        self.tape.append(("a", q))
        return q

    def x(self, q):
        self.b.x(q)
        self.tape.append(("x", q))

    def cx(self, c, t):
        self.b.cx(c, t)
        self.tape.append(("c", c, t))

    def ccx(self, c1, c2, t):
        self.b.ccx(c1, c2, t)
        self.tape.append(("cc", c1, c2, t))

    def undo(self):
        b = self.b
        if self.own is not None:
            allocs = [e[1] for e in self.tape if e[0] == "a"]
            if allocs:
                b.release(self.own, allocs)
        for e in reversed(self.tape):
            if e[0] == "x":
                b.x(e[1])
            elif e[0] == "c":
                b.cx(e[1], e[2])
            elif e[0] == "cc":
                b.ccx(e[1], e[2], e[3])
            else:
                b.recycle(e[1])
        self.tape = []


def _ctrl_increment(b, ctrl, v):
    """v += 1 iff ctrl (LE register, mod 2^len(v)).  len(v)-1 Toffoli:
    the shared all-[GID] adder corner with the control as the one-bit
    operand, ``v += [ctrl]``.
    """
    gidney_add(b, [ctrl], v)


def _ctrl_decrement(b, ctrl, v):
    """v -= 1 iff ctrl (X-conjugated controlled increment: the shared
    complement-subtract corner, ``v -= [ctrl]``).
    """
    gidney_sub(b, [ctrl], v)


def _cswap_regs(b, ctrl, r1, r2):
    """Emit a controlled swap of two equal registers: 1 Toffoli + 2 CX per bit
    (their CSWAP(dtype) = num_qubits x TwoBitCSwap accounting), each pair
    the Builder's native Fredkin expansion.
    """
    for x, y in zip(r1, r2):
        b.cswap(ctrl, x, y)


# ---------------------------------------------------------------------------
# Unary-iteration walker (their QROM / IsEquals-per-slot accounting)
# ---------------------------------------------------------------------------


def walker_toffoli(w: int, n_leaves: int, ctrl: bool) -> int:
    """Toffoli of one full UnaryWalker sweep over leaves 0 .. n_leaves-1 of
    a w-bit trie: one Gidney AND per visited internal node (its first child;
    the sibling is a CX switch), i.e. sum over depths d < w of the number of
    depth-d subtrees that intersect the leaf range, minus the uncontrolled
    root (a CX alias).  Full trie (n_leaves = 2^w): 2^w - 2 / 2^w - 1.
    """
    assert 1 <= n_leaves <= (1 << w)
    nodes = sum(-(-n_leaves // (1 << (w - d))) for d in range(w))
    return nodes if ctrl else nodes - 1


class UnaryWalker:
    """DFS over a little-endian register's bit-trie, visiting the leaf values
    0 .. n_leaves-1 in strictly descending (or ascending) address order.

    At each leaf the walker exposes ``leaf`` — an ancilla (or literal) holding
    [reg == value], or [ctrl AND reg == value] when a control qubit is given.
    Subtrees entirely above n_leaves-1 are never entered (the pruned trie
    of unary iteration over a non-power-of-two address range; with
    n_leaves = 2^w the walk is the full trie).  Amortized cost: one Gidney
    AND per left-descent (walker_toffoli: 76 Toffoli per uncontrolled sweep
    of the 74 addresses 0 .. m' in the 7-bit trie; 77 controlled — the
    control rides the walk root, so it costs +1 Toffoli per sweep, not per
    leaf), sibling switches are a single CX (t_right = parent XOR t_left for
    path ANDs), releases are measured (free).  This is the realization
    family priced by their QROM(m, 1, num_controls=1) ~= m-Toffoli
    call-graph terms.

    local: shared qrom.walk prices identically (1 Toffoli per two-sided
    node, 2^n - 2 per full uncontrolled window) but is a one-shot ascending
    recursive sweep with a body callback — the cycle interleaves four
    walkers (three descending, one ascending) stepping in lockstep per
    slot, which walk cannot express without reordering the stream.

    The register (and the control) must not be modified while the walker is
    alive.
    """

    def __init__(
        self,
        b,
        reg: Sequence[int],
        descending: bool = True,
        ctrl: int | None = None,
        n_leaves: int | None = None,
    ):
        self.b = b
        self.reg = list(reg)
        self.w = len(reg)
        self.descending = descending
        # Optional control qubit folded into the walk root (their
        # QROM(..., num_controls=1) idiom): every leaf then holds
        # [ctrl AND reg == value] for +1 Toffoli per full sweep.
        self.ctrl = ctrl
        # Leaf domain 0 .. n_leaves-1 (default: the full trie).
        self.n_leaves = (1 << self.w) if n_leaves is None else n_leaves
        assert 1 <= self.n_leaves <= (1 << self.w)
        # DFS stack of (bit_index, branch_bit, ancilla or None-at-root-level)
        # path[i] covers bit (w-1-i); ancilla None only transiently.
        self.path: list[tuple[int, int, int | None]] = []
        self._prefix = 0  # value of the branch bits on the current path
        self.value: int | None = None
        self.leaf: int | None = None

    # -- internals ----------------------------------------------------------

    def _in_range(self, prefix_above: int, bit_idx: int, branch: int) -> bool:
        """Return whether the subtree (prefix_above, bit_idx = branch) meets the domain."""
        return prefix_above + (branch << bit_idx) <= self.n_leaves - 1

    def _descend(self, bit_idx: int, branch: int):
        """Push one trie level: path AND gains literal [reg_bit == branch]."""
        b = self.b
        qbit = self.reg[bit_idx]
        neg = branch == 0
        if not self.path:
            parent = self.ctrl
        else:
            parent = self.path[-1][2]
        if parent is None:
            # depth-1 node, uncontrolled: the path condition is a single
            # literal; alias it via a fresh ancilla (CX/X copy) so sibling
            # switching is uniform.
            t = b.qubit()
            b.cx(qbit, t)
            if neg:
                b.x(t)
            self.path.append((bit_idx, branch, t))
        else:
            t = b.and_compute(parent, qbit, pol=(True, not neg))
            self.path.append((bit_idx, branch, t))
        self._prefix += branch << bit_idx

    def _release_top(self):
        b = self.b
        bit_idx, branch, t = self.path.pop()
        self._prefix -= branch << bit_idx
        qbit = self.reg[bit_idx]
        neg = branch == 0
        if not self.path:
            if self.ctrl is None:
                if neg:
                    b.x(t)
                b.cx(qbit, t)
                b.recycle(t)
            else:
                b.and_release(t, self.ctrl, qbit, pol=(True, not neg))
        else:
            parent = self.path[-1][2]
            b.and_release(t, parent, qbit, pol=(True, not neg))

    def _switch_top_sibling(self):
        """t := parent AND (other branch) via t ^= parent (or ^= 1 at root)."""
        b = self.b
        bit_idx, branch, t = self.path.pop()
        if not self.path:
            if self.ctrl is None:
                b.x(t)  # single-literal path: NOT flips the branch
            else:
                b.cx(self.ctrl, t)
        else:
            b.cx(self.path[-1][2], t)
        self.path.append((bit_idx, 1 - branch, t))
        self._prefix += (1 - 2 * branch) << bit_idx

    def _sibling_pending(self) -> bool:
        """Return whether the top level still has an unvisited in-range sibling."""
        bit_idx, branch, _t = self.path[-1]
        if self.descending:
            return branch == 1  # branch 0 (smaller values) is always in range
        return branch == 0 and self._in_range(self._prefix, bit_idx, 1)

    def _descend_all(self):
        while len(self.path) < self.w:
            bit_idx = self.w - 1 - len(self.path)
            if self.descending:
                branch = 1 if self._in_range(self._prefix, bit_idx, 1) else 0
            else:
                branch = 0
            self._descend(bit_idx, branch)
        self.leaf = self.path[-1][2]
        self.value = self._prefix

    # -- public -------------------------------------------------------------

    def start(self) -> int:
        """Descend to the first leaf (n_leaves-1 for descending, 0 ascending)."""
        assert not self.path
        self._descend_all()
        assert self.value is not None  # _descend_all sets value
        return self.value

    def step(self) -> int | None:
        """Advance to the next leaf in order; None when exhausted."""
        # backtrack: pop levels whose remaining sibling is visited or pruned
        while self.path and not self._sibling_pending():
            self._release_top()
        if not self.path:
            self.leaf = None
            self.value = None
            return None
        self._switch_top_sibling()
        self._descend_all()
        return self.value

    def finish(self):
        """Release whatever remains of the path (call after the last leaf).

        A walker stepped to exhaustion has nothing left; a walker abandoned
        mid-sweep releases its remaining path here.
        """
        while self.path:
            self._release_top()
        self.leaf = None
        self.value = None


# ---------------------------------------------------------------------------
# Per-cycle shared context: is_hc_zero + counter case decode
# ---------------------------------------------------------------------------


_CASE_POL = [((k & 1) == 1, (k & 2) == 2) for k in range(4)]  # [counter == k]


class _CycleCtx:
    """Amortized per-cycle control bits.

    hc0 = [halting_counter == 0] (their per-step IsEquals(hc, 0), HC_W - 1
    = 7 Toffoli),
    case_k = [counter == k] (4 Gidney ANDs; their bloqs re-derive these inside
    every sub-bloq via X-conjugated MCX controls — decoding once per cycle is
    the same leaf family, applied once), and case_h[k] = case_k AND hc0
    (4 more ANDs) — the hc-gated case bits every stage conditions on, hoisted
    so no stage pays a per-use AND for them.
    """

    def __init__(self, b, lay: EEALayout):
        self.b = b
        self.lay = lay
        self.hc0, self.hc0_trail = eq_const(b, lay.hc, 0)
        c0, c1 = lay.counter
        self.case: list[int] | None = [b.and_compute(c0, c1, pol=_CASE_POL[k]) for k in range(4)]
        self.case_h: list[int] | None = [b.and_compute(self.case[k], self.hc0) for k in range(4)]

    def release_case_h(self):
        b = self.b
        assert self.case is not None and self.case_h is not None
        for k in (3, 2, 1, 0):
            b.and_release(self.case_h[k], self.case[k], self.hc0)
        self.case_h = None

    def release_hc0(self):
        release_chain(self.b, self.hc0_trail)
        self.hc0 = None

    def release_cases(self):
        b = self.b
        c0, c1 = self.lay.counter
        assert self.case is not None
        for k in (3, 2, 1, 0):
            b.and_release(self.case[k], c0, c1, pol=_CASE_POL[k])
        self.case = None


def _or2(b, q1, q2):
    """t = q1 OR q2 = NOT AND(NOT q1, NOT q2).  1 Toffoli; release with
    _or2_release while operands hold compute-time values.
    """
    t = b.and_compute(q1, q2, pol=(False, False))
    b.x(t)
    return t


def _or2_release(b, t, q1, q2):
    b.x(t)
    b.and_release(t, q1, q2, pol=(False, False))


@contextmanager
def _and_not_or(b, case, flag, case_other):
    """Yield u = (case AND NOT flag) OR case_other for the block, then
    release it.  2 Toffoli compute, 0 release (measured); the operands
    must hold their compute-time values at the release.
    """
    t = b.and_compute(case, flag, pol=(True, False))
    u = _or2(b, t, case_other)
    yield u
    _or2_release(b, u, t, case_other)
    b.and_release(t, case, flag, pol=(True, False))


# ---------------------------------------------------------------------------
# Stage 1 — UpdateFlag (estimate-only bloq; semantics: its on_classical_vals)
# ---------------------------------------------------------------------------


def _stage1_update_flag(b, lay: EEALayout, ctx: _CycleCtx) -> None:
    m = MPRIME

    # MEASURED INVARIANT (a seeded census of the classical decoder:
    # 690 runs — all weights 0-35, all leading-zero degrees,
    # monomials, non-genuine random syndromes — covering every counter==0
    # stage-1 evaluation, zero violations): at every reachable counter==0
    # evaluation n_A <= n_B, so (n_A < n_B) == NOT (n_A == n_B) and case 0
    # needs no LessThan comparator — matching the released call graph,
    # which books IsEquals bloqs only.
    #
    # Case 0: first = [n_q==0] AND NOT eqAB;  last = eqAB
    #         first XOR last == [n_q==0] OR eqAB              (1 Toffoli)

    # shared condition: [n_A == n_B]  (case0 first-xor-last, case1 is_first)
    eqAB = _EqQQ(b, lay.n_A, lay.n_B)
    # shared n_q suffix: [n_q[1..] == 0], specialized to [n_q==0] / [n_q==1]
    nq_hi0, nq_hi0_trail = eq_const(b, lay.n_q[1:], 0)
    eq_nq0 = b.and_compute(nq_hi0, lay.n_q[0], pol=(True, False))

    d0 = _or2(b, eq_nq0, eqAB.res)
    _flag_xor_case(b, ctx, 0, d0)
    _or2_release(b, d0, eq_nq0, eqAB.res)

    # Case 1: first = eqAB;  last = [n_B == 1] OR [b_array[m-2] != 0]
    #                             = NOT (z_top AND NOT eq_nB1)
    eq_nB1, eq_nB1_trail = eq_const(b, lay.n_B, 1)
    z_top, z_top_trail = eq_const(b, lay.bb[m - 2], 0)  # [sym == 0]
    c1l = b.and_compute(z_top, eq_nB1, pol=(True, False))
    rev = _Rev(b, own="scr/f")
    f = rev.alloc()  # first XOR last = eqAB XOR NOT c1l
    rev.cx(eqAB.res, f)
    rev.cx(c1l, f)
    rev.x(f)
    _flag_xor_case(b, ctx, 1, f)
    rev.undo()
    b.and_release(c1l, z_top, eq_nB1, pol=(True, False))
    release_chain(b, z_top_trail)
    release_chain(b, eq_nB1_trail)
    eqAB.release()

    # Case 2: first = [b_array[0] != 0] OR [n_b == 0] = NOT (z_b0 AND NOT eq_nb0)
    #         and last = [n_a == n_b + 1]
    z_b0, z_b0_trail = eq_const(b, lay.bb[0], 0)
    eq_nb0, eq_nb0_trail = eq_const(b, lay.n_b, 0)
    c2f = b.and_compute(z_b0, eq_nb0, pol=(True, False))  # == NOT first
    # n_b + 1 into a scratch copy (prefix-AND carry chain, then compare).
    with _plus_pow2_copy(b, lay.n_b, 0, "scr/nb1") as nb1:
        eq_c2l = _EqQQ(b, lay.n_a, nb1)
        rev = _Rev(b, own="scr/f")
        f = rev.alloc()  # first XOR last = NOT c2f XOR eq_c2l
        rev.cx(c2f, f)
        rev.x(f)
        rev.cx(eq_c2l.res, f)
        _flag_xor_case(b, ctx, 2, f)
        rev.undo()
        eq_c2l.release()
    b.and_release(c2f, z_b0, eq_nb0, pol=(True, False))
    release_chain(b, eq_nb0_trail)
    release_chain(b, z_b0_trail)

    # Case 3: first = [n_a == n_b];  last = [n_q == 1]
    eq_ab = _EqQQ(b, lay.n_a, lay.n_b)
    eq_nq1 = b.and_compute(nq_hi0, lay.n_q[0])
    rev = _Rev(b, own="scr/f")
    f = rev.alloc()  # first XOR last
    rev.cx(eq_ab.res, f)
    rev.cx(eq_nq1, f)
    _flag_xor_case(b, ctx, 3, f)
    rev.undo()
    b.and_release(eq_nq1, nq_hi0, lay.n_q[0])
    eq_ab.release()

    b.and_release(eq_nq0, nq_hi0, lay.n_q[0], pol=(True, False))
    release_chain(b, nq_hi0_trail)


@contextmanager
def _plus_pow2_copy(b, src, s, name):
    """Yield a scratch copy of src + 2^s (prefix-AND carry chain of
    w-s-2 ANDs), then undo it and free the scratch (measured, free).
    """
    w = len(src)
    out = b.qubits(w)
    b.claim(name, out)
    for q, d in zip(src, out):
        b.cx(q, d)
    carry = {s + 1: b.and_compute(src[s], src[s + 1])}
    for i in range(s + 2, w - 1):
        carry[i] = b.and_compute(carry[i - 1], src[i])
    b.x(out[s])
    b.cx(src[s], out[s + 1])
    for i in range(s + 2, w):
        b.cx(carry[i - 1], out[i])
    yield out
    for i in range(w - 1, s + 1, -1):
        b.cx(carry[i - 1], out[i])
    b.cx(src[s], out[s + 1])
    b.x(out[s])
    for i in range(w - 2, s + 1, -1):
        b.and_release(carry[i], carry[i - 1], src[i])
    b.and_release(carry[s + 1], src[s], src[s + 1])
    for q, d in zip(src, out):
        b.cx(q, d)
    b.release_recycle(name, out)


def _flag_xor_case(b, ctx: _CycleCtx, k: int, d: int) -> None:
    """flag ^= case_h[k] AND d  (case_h = case_k AND hc0, hoisted in the ctx)."""
    assert ctx.case_h is not None
    t = b.and_compute(ctx.case_h[k], d)
    b.cx(t, ctx.lay.flag)
    b.and_release(t, ctx.case_h[k], d)


# ---------------------------------------------------------------------------
# Stage 2 — MultiplyAndAddOrSub (their circuit, poly_eea_zalka_bloqs.py:359-441)
# ---------------------------------------------------------------------------


@contextmanager
def _quotient(b, num_reg, den_reg, name):
    """Yield q = num / den (keep-junk inverse + one multiply, 306 Toffoli),
    then erase q, the inverse and its junk by measurement (0 Toffoli).
    """
    inv = b.qubits(GF_B)
    b.claim("gfw/inv", inv)
    junk = gf_inverse_keep(b, den_reg, inv)
    q = b.qubits(GF_B)
    b.claim(name, q)
    gf_mult_opt(b, num_reg, inv, q)
    yield q
    b.release(name, q)
    gf_mult_uncompute(b, q, num_reg, inv)
    gf_inverse_unkeep(b, den_reg, inv, junk)
    b.release_recycle("gfw/inv", inv)


def _division_xor_into(b, num_reg, den_reg, ctrl, dst_reg) -> None:
    """dst ^= (num / den) when ctrl, via their GF2DivisionUsingFLT structure
    WITH the released keep-junk adjoint pattern: y_inv = inverse(den) keeping
    the Itoh-Tsujii chain intermediates as junk [255], t = num * y_inv [51],
    controlled-XOR into dst [12 CCX], then MBUC-erase t [0] and erase y_inv +
    junk by measured uncompute against live operands [0 — their
    Adjoint(GF2DivisionUsingFLT) = 0 Toffoli accounting is LEGITIMATE under
    this pattern; the price is 60 junk qubits held across the ctrl-add,
    matching their transient-width headroom].  DIVISION_TOFFOLI = 318 total
    (their FLT division books 306: the controlled add is a separate CADD).
    """
    with _quotient(b, num_reg, den_reg, "gfw/prod") as t:
        for tj, dj in zip(t, dst_reg):
            b.ccx(ctrl, tj, dj)  # 12


def _body_frame(b):
    """Return a count-1 sub-loop around one balanced block, inside a repeat body.

    The op stream caps a loop body at 65,535 items (u16 n_items), and one cycle
    records ~200k ops, most of them in the 73 slot products.  Inside a loop
    body each slot product is its own repeat(1), one item of the cycle body;
    the stream expands to the same gates.  Flat builds record no framing.
    """
    return b.repeat(1) if getattr(b, "_loop_stack", None) else nullcontext()


def _stage2_multiply_add_sub(b, lay: EEALayout, ctx: _CycleCtx) -> None:
    assert ctx.case is not None and ctx.case_h is not None
    m = MPRIME

    # nq' = n_a + n_q (their in-place Add / Add-adjoint pair around the loop)
    rev = _Rev(b, own="scr/add")
    nqp = _add_out(rev, lay.n_a, lay.n_q)

    # head division: q ^= b[-1]/a[-1] iff counter == 0 and hc == 0 (the ctrl
    # is the hoisted case_h[0]; the 12 CCX inside are their head seed CADD)
    _division_xor_into(b, lay.bb[m - 1], lay.a[m - 1], ctx.case_h[0], lay.q)

    # Walkers with the case controls FOLDED INTO the walk root (their
    # QROM(m,1,num_controls=1) accounting), so every per-slot predicate is a
    # ready leaf/window bit and the slot body pays only CSWAP + mult + CADD:
    #   wa  leaf = case0 AND hc0 AND [nq' == i]      (deposit one-hot)
    #   wb  leaf = case3 AND hc0 AND [nq' == i+1]    (extract one-hot)
    #   wna pulses accumulate  case3 AND [n_a > i]   (add window)
    #   wnA pulses accumulate  case0 AND [n_A >= m-i] (sub window, via the
    #       case0-seeded complement: case0 XOR (case0 AND [n_A <= m-i-1]))
    # The add/sub windows are deliberately NOT hc-gated, exactly like the
    # classical cycle's add/sub windows (reference.py ``_cycle``, step 2):
    # on halted paths q == 0 makes them no-ops.
    # Every walk covers the leaf domain 0 .. m (WALK_LEAVES addresses): the
    # counters and n_a + n_q never exceed m on any reachable path (cycle
    # section header), which is what the window collapses below rely on.
    wa = UnaryWalker(b, nqp, descending=True, ctrl=ctx.case_h[0], n_leaves=WALK_LEAVES)
    wb = UnaryWalker(b, nqp, descending=True, ctrl=ctx.case_h[3], n_leaves=WALK_LEAVES)
    wna = UnaryWalker(b, lay.n_a, descending=True, ctrl=ctx.case[3], n_leaves=WALK_LEAVES)
    wnA = UnaryWalker(b, lay.n_A, descending=False, ctrl=ctx.case[0], n_leaves=WALK_LEAVES)

    swp = b.qubit()  # per-slot swap control: wa.leaf XOR wb.leaf
    aos = b.qubit()  # add-or-sub window: disjoint XOR of both windows
    b.claim("scr/win", [swp, aos])
    b.cx(ctx.case[0], aos)  # sub-window complement seed

    wb.start()  # at m == leaf i+1 for the first slot (i = m-1)
    wa.start()
    wa.step()  # at m-1 == leaf i for the first slot
    wna.start()  # pulse v=m: add window [n_a > m-1]
    b.cx(wna.leaf, aos)
    wnA.start()  # pulse v=0: sub window [n_A >= 1] for the first slot
    b.cx(wnA.leaf, aos)

    for i in range(m - 1, -1, -1):
        # swap_ctrl = wa.leaf XOR wb.leaf (disjoint: case0 vs case3)
        b.cx(wa.leaf, swp)
        b.cx(wb.leaf, swp)
        _cswap_regs(b, swp, lay.q, lay.a[i])  # GF_B
        b.cx(wb.leaf, swp)
        b.cx(wa.leaf, swp)

        # t = q * a[i]  (their GF2MulViaOptimizedPolyMul)
        with _body_frame(b):
            t = b.qubits(GF_B)
            b.claim("gfw/prod", t)
            gf_mult_opt(b, lay.q, lay.a[i], t)  # OPT_MULT_TOFFOLI
            for tj, bj in zip(t, lay.bb[i]):
                b.ccx(aos, tj, bj)  # GF_B (their controlled GF2Addition)
            b.release("gfw/prod", t)
            gf_mult_uncompute(b, t, lay.q, lay.a[i])  # 0 (their MBUC adjoint)

        if i > 0:
            wb.step()  # -> leaf i   (== i+1 for the next slot)
            wa.step()  # -> leaf i-1
            wna.step()  # pulse v=i: add window extends to [n_a > i-1]
            b.cx(wna.leaf, aos)
            wnA.step()  # pulse v=m-i: sub window extends to [n_A >= m-i+1]
            b.cx(wnA.leaf, aos)

    # Window teardown: the remaining leaf pulses collapse each window to its
    # bare control, which a single CX removes — 0-Toffoli cleanup.
    wna.step()  # pulse v=0: add part -> case3 AND [n_a >= 0] == case3
    b.cx(wna.leaf, aos)
    b.cx(ctx.case[3], aos)
    wnA.step()  # pulse v=m: sub accum -> case0 AND [n_A <= m] == case0,
    b.cx(wnA.leaf, aos)  # cancelling the case0 seed
    wa.finish()
    wb.finish()
    wna.finish()
    wnA.finish()
    b.release("scr/win", [swp, aos])
    b.recycle(aos)
    b.recycle(swp)

    rev.undo()  # their Add.adjoint()

    # tail division: q ^= b[0]/a[0] iff counter == 3 and hc == 0
    _division_xor_into(b, lay.bb[0], lay.a[0], ctx.case_h[3], lay.q)


def _add_out(rev: _Rev, x, y):
    """s = x + y (out-of-place ripple on the tape; their Add(QUInt(7))).

    local: a shared-gadget realization is cheaper — CX-copy one operand to
    a fresh register, then the shared in-place adder (arith.add, w - 1
    Toffoli, measured carry erase) holds the sum with both operands
    live — but this form keeps the released Add/Add-adjoint pair
    accounting verbatim: ADD_TOFFOLI = 1 + 3(w - 2) Toffoli forward (16 at
    w = 7) + the same in the exact replay undo.
    """
    w = len(x)
    s = [rev.alloc() for _ in range(w)]
    carry = None
    for i in range(w):
        rev.cx(x[i], s[i])
        rev.cx(y[i], s[i])
        if carry is not None:
            rev.cx(carry, s[i])
        if i < w - 1:
            c = rev.alloc()
            rev.ccx(x[i], y[i], c)
            if carry is not None:
                rev.ccx(carry, x[i], c)
                rev.ccx(carry, y[i], c)
            carry = c
    return s


# ---------------------------------------------------------------------------
# Stage 3 — ShiftBArrayRight (estimate-only; semantics: its on_classical_vals)
# ---------------------------------------------------------------------------


def _stage3_shift_b_right(b, lay: EEALayout, ctx: _CycleCtx) -> None:
    assert ctx.case_h is not None
    m = MPRIME

    # per-cycle window controls, hc-gated via the hoisted case_h bits:
    #   cnB  = hc0 AND ((counter==1) OR (counter==0 AND !flag))  [windows 1]
    #   c2h  = hc0 AND (counter==2)                              [window 2]
    #   c3nf = hc0 AND (counter==3) AND !flag                    [window 3]
    with _and_not_or(b, ctx.case_h[0], lay.flag, ctx.case_h[1]) as cnB:
        c3nf = b.and_compute(ctx.case_h[3], lay.flag, pol=(True, False))
        c2h = ctx.case_h[2]

        # Three controlled walkers accumulate the three (mutually exclusive)
        # shift windows on ONE wire, so the slot body is just the CSWAP:
        #   w1 (asc on n_B, ctrl cnB, complement seed):
        #       dsh += cnB AND [n_B >= m-i+1]
        #   w2 (desc on n_b, ctrl c2h):   dsh += c2h AND [n_b >= i]
        #   w3 (asc on n_B, ctrl c3nf):   dsh += c3nf AND [n_B <= m-1-i]
        w1 = UnaryWalker(b, lay.n_B, descending=False, ctrl=cnB, n_leaves=WALK_LEAVES)
        w2 = UnaryWalker(b, lay.n_b, descending=True, ctrl=c2h, n_leaves=WALK_LEAVES)
        w3 = UnaryWalker(b, lay.n_B, descending=False, ctrl=c3nf, n_leaves=WALK_LEAVES)

        dsh = b.qubit()  # do_shift: XOR of the three windows
        b.claim("scr/win", [dsh])
        b.cx(cnB, dsh)  # window-1 complement seed
        w1.start()  # pulse v=0
        b.cx(w1.leaf, dsh)
        w2.start()  # pulse v=m
        b.cx(w2.leaf, dsh)
        w3.start()  # pulse v=0: window3 = c3nf AND [n_B <= 0] for slot m-1
        b.cx(w3.leaf, dsh)

        for i in range(m - 1, 0, -1):
            w1.step()  # pulse v = m-i: window1 = cnB AND [n_B >= m-i+1]
            b.cx(w1.leaf, dsh)
            w2.step()  # pulse v = i:   window2 = c2h AND [n_b >= i]
            b.cx(w2.leaf, dsh)
            if i < m - 1:
                w3.step()  # pulse v = m-1-i: window3 = c3nf AND [n_B <= m-1-i]
                b.cx(w3.leaf, dsh)
            _cswap_regs(b, dsh, lay.bb[i - 1], lay.bb[i])  # GF_B

        # Teardown: remaining pulses collapse each window to its bare control
        # (n_B, n_b <= m on every reachable path — section header).
        w1.step()  # v=m: accum -> cnB AND [n_B <= m] == cnB, cancels the seed
        b.cx(w1.leaf, dsh)
        w2.step()  # v=0: window2 -> c2h AND [n_b >= 0] == c2h
        b.cx(w2.leaf, dsh)
        b.cx(c2h, dsh)
        w3.step()  # v=m-1
        b.cx(w3.leaf, dsh)
        w3.step()  # v=m: window3 -> c3nf AND [n_B <= m] == c3nf
        b.cx(w3.leaf, dsh)
        b.cx(c3nf, dsh)
        w1.finish()
        w2.finish()
        w3.finish()
        b.release_recycle("scr/win", [dsh])

        b.and_release(c3nf, ctx.case_h[3], lay.flag, pol=(True, False))


# ---------------------------------------------------------------------------
# Stage 4 — UpdateMetadata (estimate-only; semantics: its on_classical_vals)
# ---------------------------------------------------------------------------


def _stage4_update_metadata(b, lay: EEALayout, ctx: _CycleCtx) -> None:
    assert ctx.case_h is not None
    # The four counter cases are mutually exclusive, so the two n_B
    # decrements (case0 AND !flag; case1) merge into ONE decrement on the
    # disjoint-OR control, and likewise the two n_b increments (case2;
    # case3 AND !flag) — action-identical to the separate branches of
    # UpdateMetadata (reference.py ``_cycle``, step 4).
    # All case controls are the hoisted case_h.

    # Case 0 (hc0): n_q += 1
    _ctrl_increment(b, ctx.case_h[0], lay.n_q)

    # n_B -= 1 iff (case0 AND !flag AND hc0) OR (case1 AND hc0)
    with _and_not_or(b, ctx.case_h[0], lay.flag, ctx.case_h[1]) as u:
        _ctrl_decrement(b, u, lay.n_B)

    # n_b += 1 iff (case2 AND hc0) OR (case3 AND !flag AND hc0)
    with _and_not_or(b, ctx.case_h[3], lay.flag, ctx.case_h[2]) as u:
        _ctrl_increment(b, u, lay.n_b)

    # Case 3 (hc0): n_q -= 1
    _ctrl_decrement(b, ctx.case_h[3], lay.n_q)


# ---------------------------------------------------------------------------
# Stage 5 — SwapAandB (their circuit, poly_eea_zalka_bloqs.py:652-683)
# ---------------------------------------------------------------------------


def _stage5_swap_a_b(b, lay: EEALayout, ctx: _CycleCtx) -> None:
    assert ctx.case is not None
    # sw = counter == 3 AND flag AND hc0  (their MCX(1,1,1,1))
    t = b.and_compute(ctx.case[3], lay.flag)
    sw = b.and_compute(t, ctx.hc0)
    _cswap_regs(b, sw, lay.n_a, lay.n_b)  # their CSWAP(quint_logm)
    _cswap_regs(b, sw, lay.n_A, lay.n_B)
    for i in range(MPRIME):
        _cswap_regs(b, sw, lay.a[i], lay.bb[i])  # their CSWAP(qgf) x m
    b.and_release(sw, t, ctx.hc0)
    b.and_release(t, ctx.case[3], lay.flag)


# ---------------------------------------------------------------------------
# Stage 6 — UpdateHaltingCounter (their circuit; stop constant per section header)
# ---------------------------------------------------------------------------


def _stage6_update_halting_counter(b, lay: EEALayout, ctx: _CycleCtx) -> None:
    assert ctx.case is not None
    # their wiring releases is_hc_zero BEFORE this stage (caller does that);
    # stop = [n_A == 0] OR [n_B < STOP_LT_CONST]   (36, see section header)
    eqA0, eqA0_trail = eq_const(b, lay.n_A, 0)
    rev = _Rev(b, own="scr/stop")
    lt = _lt_const(rev, lay.n_B, STOP_LT_CONST)
    is_stop = _or2(b, eqA0, lt)
    t = b.and_compute(ctx.case[3], lay.flag)
    upd = b.and_compute(t, is_stop)
    _ctrl_increment(b, upd, lay.hc)
    b.and_release(upd, t, is_stop)
    b.and_release(t, ctx.case[3], lay.flag)
    _or2_release(b, is_stop, eqA0, lt)
    rev.undo()
    release_chain(b, eqA0_trail)


def _lt_const(rev: _Rev, v, k):
    """[v < k] for a classical constant k > 0 (borrow chain on the tape)."""
    borrow = None
    for i in range(len(v)):
        kb = (k >> i) & 1
        if kb:
            t = rev.alloc()
            rev.cx(v[i], t)
            rev.x(t)
            if borrow is not None:
                rev.ccx(v[i], borrow, t)
            borrow = t
        else:
            if borrow is None:
                continue
            t = rev.alloc()
            rev.x(v[i])
            rev.ccx(v[i], borrow, t)
            rev.x(v[i])
            borrow = t
    assert borrow is not None, "k == 0 gives constant false"
    return borrow


def lt_const_toffoli(w: int, k: int) -> int:
    """Toffoli of _lt_const on a w-bit register (one CCX per link past the
    lowest set bit of k).
    """
    cost = 0
    borrow = False
    for i in range(w):
        if (k >> i) & 1:
            cost += 1 if borrow else 0
            borrow = True
        elif borrow:
            cost += 1
    return cost


# ---------------------------------------------------------------------------
# One full cycle (their _PolyEEAZalkaImplStep wiring order)
# ---------------------------------------------------------------------------


def emit_eea_cycle(b, lay: EEALayout, phase: str | None = None) -> None:
    """One synchronized EEA cycle on the persistent register file.

    Stage order and is_hc_zero bracketing exactly as the step bloq's
    build_composite_bloq (poly_eea_zalka_bloqs.py:835-890): compute
    is_hc_zero; UpdateFlag; MultiplyAndAddOrSub; ShiftBArrayRight;
    UpdateMetadata; SwapAandB; release is_hc_zero; UpdateHaltingCounter;
    counter += flag.

    ``phase`` (profile-only, metadata): when given, per-stage phase marks
    ``{phase}/s0..s7`` are recorded for tools/liveness_profile.py —
    s0 = ctx decode, s1..s5 = the five stage bloqs, s6 = ctx release +
    UpdateHaltingCounter, s7 = the counter increment.  Marks emit nothing.
    The forward pass emits the cycle once, as a repeat(222) body, under
    ``phase="eea/cycle"``.
    """

    def mark(stage: str) -> None:
        if phase is not None:
            b.set_phase(f"{phase}/{stage}")

    mark("s0")
    ctx = _CycleCtx(b, lay)
    mark("s1")
    _stage1_update_flag(b, lay, ctx)
    mark("s2")
    _stage2_multiply_add_sub(b, lay, ctx)
    mark("s3")
    _stage3_shift_b_right(b, lay, ctx)
    mark("s4")
    _stage4_update_metadata(b, lay, ctx)
    mark("s5")
    _stage5_swap_a_b(b, lay, ctx)
    mark("s6")
    ctx.release_case_h()  # while counter and hc are still unchanged
    ctx.release_hc0()  # their IsEquals(hc, 0).adjoint()
    _stage6_update_halting_counter(b, lay, ctx)
    ctx.release_cases()
    mark("s7")
    _ctrl_increment(b, lay.flag, lay.counter)  # counter = (counter+flag) % 4


# Emitted Toffoli per stage of one cycle block, derived from the leaf
# realizations above and pinned against the emission by build()'s cost
# assert (the released call-graph budgets for comparison: at this
# instance MultiplyAndAddOrSub 6,543 / ShiftBArrayRight 1,096 / SwapAandB
# 896 / UpdateFlag 74 / UpdateMetadata 36 / UpdateHaltingCounter 94, per
# cycle 8,754; the estimate-only stages are realized below their budgets).
INV_TOFFOLI = ITOH_TSUJII_QQ_MULTS * OPT_MULT_TOFFOLI  # 255
DIVISION_TOFFOLI = INV_TOFFOLI + OPT_MULT_TOFFOLI + GF_B  # 318 incl. the CADD
WALK_TOFFOLI = walker_toffoli(CTR_W, WALK_LEAVES, ctrl=True)  # 77
ADD_TOFFOLI = 1 + 3 * (CTR_W - 2)  # 16: _add_out ripple
EQ_TOFFOLI = CTR_W - 1  # 6: _EqQQ / IsEquals on a counter
ZERO_SYM_TOFFOLI = GF_B - 1  # 11: [symbol == 0]
STAGE_TOFFOLI = {
    # IsEquals(hc, 0) + the four case ANDs + the four hc-gated case ANDs
    "s0_ctx": (HC_W - 1) + 4 + 4,
    # Case 0: eqAB + nq_hi0 + eq_nq0 + or2 + flag_xor
    # Case 1: eq_nB1 + z_top + c1l + flag_xor
    # Case 2: z_b0 + eq_nb0 + c2f + carries + eq_c2l + flag_xor
    # Case 3: eq_ab + eq_nq1 + flag_xor
    "s1_update_flag": (EQ_TOFFOLI + (CTR_W - 2) + 3)
    + (EQ_TOFFOLI + ZERO_SYM_TOFFOLI + 2)
    + (ZERO_SYM_TOFFOLI + EQ_TOFFOLI + 1 + (CTR_W - 2) + EQ_TOFFOLI + 1)
    + (EQ_TOFFOLI + 2),
    "s2_multiply_add_sub": MPRIME * OPT_MULT_TOFFOLI  # m' slot multiplies
    + MPRIME * GF_B  # m' slot CSWAPs
    + MPRIME * GF_B  # m' slot CADDs
    + 2 * DIVISION_TOFFOLI  # head/tail division + CADD
    + 4 * WALK_TOFFOLI  # the four controlled walkers
    + 2 * ADD_TOFFOLI,  # the Add pair (forward + undo)
    "s3_shift_b_right": 3 * WALK_TOFFOLI + (MPRIME - 1) * GF_B + 3,
    # four controlled counter steps + the two AND/OR control pairs
    "s4_update_metadata": 4 * (CTR_W - 1) + 4,
    "s5_swap_a_and_b": 2 + 2 * CTR_W + MPRIME * GF_B,
    # StopEEA_RS: eqA0 + the lt chain (forward + undo) + or2 + 2 ANDs + hc step
    "s6_check_stop": EQ_TOFFOLI + 2 * lt_const_toffoli(CTR_W, STOP_LT_CONST) + 3 + (HC_W - 1),
    "s7_counter_inc": 1,  # controlled AddK(QUInt(2), 1)
}
CYCLE_TOFFOLI = sum(STAGE_TOFFOLI.values())


# =========================================================================
# The forward pass: in-place init + 222 cycles + the Forney flag
# =========================================================================

# The 222 cycles are one repeat(222) loop.  The init (the release's
# classical eea_init) runs in place on the caller's syndrome registers
# (n = 70, m' = 73, w = 7):
#
#   * the n syndrome registers statically alias a_array[3..m'-1] — for the
#     generic case deg = n-1 this IS reference.py ``_init``'s top-aligned placement
#     a[m'-1-deg+t] = S[t]; smaller degrees need a right-shift by s = n-1 - deg;
#   * a suffix-zero Gidney thermometer over the syndrome symbols yields
#     z_k = [S[k..n-1] all zero] (k = n-1..1); s (the ``shift`` register) and
#     n_A = n - s are accumulated by controlled increments/decrements; the
#     thermometer is then uncomputed (measured releases; syndrome untouched);
#   * the shift is applied as w binary controlled block-shifts (conveyor of
#     CSWAPs over a_array, top-down, controlled on bit k of s, one per
#     2^k < m'): sum_k (m' - 2^k) = 384 register CSWAPs = 4,608 Toffoli, in
#     place;
#   * b_array[m'-1] ^= 1 (B = z^n), a_array[0] ^= 1 (a = 1), n_B = n + 1,
#     n_a = 1, flag = 1 (Cliffords).
#
# The w-qubit ``shift`` register persists for the whole pass — the tape-driven
# adjoint replays the init backwards, un-shifting the array and returning
# ``shift`` to |0> before the syndrome registers revert to the caller.
# Persistent total: PERSISTENT_QUBITS + w = 1,818 (incl. the
# Forney-exponent flag d2, below).
#
# After the last cycle the pass computes ONE flag qubit d2 = [n_b - n_B == 2]
# from the final degree counters (emit_forney_flag, FLAG_TOFFOLI = 10, on
# the tape so the adjoint recomputes and clears it): the sweep's Forney
# constant is exact iff n_b - n_B == 1, and d2 selects the corrected mask
# at every root (the sweep section, step 4).  The data-dependent exponent
# ``expn`` in reference.py ``_sweep`` is exact.


FORWARD_PERSISTENT = PERSISTENT_QUBITS + CTR_W  # 1,818 incl. shift register


def alloc_eea_layout(
    b,
    syndrome_regs: Sequence[Sequence[int]] | None = None,
) -> EEALayout:
    """Allocate the persistent EEA register file (the ONE allocator).

    ``syndrome_regs`` (the caller's n x b-qubit registers) alias into
    a_array[3..m'-1]: the top-aligned remainder of reference.py ``_init`` (the
    cycle section's header).  With ``syndrome_regs=None`` every symbol is
    fresh (a standalone build of the sweep or one cycle, without the
    init stage).  The a_array allocates first,
    then b_array, then the counters/flags/quotient scratch.
    """
    if syndrome_regs is not None:
        assert len(syndrome_regs) == N and all(len(r) == GF_B for r in syndrome_regs)
        a = [b.qubits(GF_B) for _ in range(MPRIME - N)] + [list(r) for r in syndrome_regs]
    else:
        a = [b.qubits(GF_B) for _ in range(MPRIME)]
    bb = [b.qubits(GF_B) for _ in range(MPRIME)]
    return EEALayout(
        a=a,
        bb=bb,
        n_A=b.qubits(CTR_W),
        n_B=b.qubits(CTR_W),
        n_a=b.qubits(CTR_W),
        n_b=b.qubits(CTR_W),
        n_q=b.qubits(CTR_W),
        flag=b.qubit(),
        counter=b.qubits(2),
        hc=b.qubits(HC_W),
        q=b.qubits(GF_B),
        d2=b.qubit(),
    )


FLAG_TOFFOLI = (CTR_W - 3) + (CTR_W - 1)  # 10: the +2 carries + the equality


def emit_forney_flag(b, lay: EEALayout) -> None:
    """d2 ^= [n_b == n_B + 2] on the final counters (FLAG_TOFFOLI Toffoli).

    n_B + 2 is formed in a scratch copy (bit 1 flips; bit i >= 2 flips iff
    n_B[1..i-1] are all set — a prefix-AND carry chain of w-3 ANDs), compared
    with n_b by _EqQQ (w-1 ANDs), copied out with one CX, and everything is
    released (measured, free) — the flag alone persists through the sweep.
    n_B + 2 < 2^w on every reachable path (n_B <= n+1).
    """
    with _plus_pow2_copy(b, lay.n_B, 1, "scr/nb2") as nb2:
        eq = _EqQQ(b, lay.n_b, nb2)
        b.cx(eq.res, lay.d2)
        eq.release()


_CHUNK = 6  # symbols per thermometer chunk (bounds the transient width)


def _chunk_scan(b, lay: EEALayout, ks: Sequence[int], base, shift=None):
    """Build the suffix-zero z-chain for symbol indices ``ks`` (descending),
    with optional base literal (the carry from the previous chunk).

    If ``shift`` is given, s += z_k is accumulated per link.  Returns
    (parts, z_last) where parts is the LIFO release list.
    """
    parts = []
    z_prev = base
    for k in ks:
        y, y_trail = eq_const(b, lay.a[MPRIME - N + k], 0)
        parts.append(("y", y_trail))
        if z_prev is None:
            z_k = y
        else:
            z_k = b.and_compute(y, z_prev)
            parts.append(("z", z_k, y, z_prev))
        if shift is not None:
            _ctrl_increment(b, z_k, shift)  # s += z_k
        z_prev = z_k
    return parts, z_prev


def _chunk_release(b, parts) -> None:
    for p in reversed(parts):
        if p[0] == "z":
            _, z, yres, prev = p
            b.and_release(z, yres, prev)
        else:
            release_chain(b, p[1])


def emit_eea_init(b, lay: EEALayout, shift: Sequence[int]) -> None:
    """Run the release's classical eea_init (reference.py ``_init``) in place.

    PRECONDITION: ``shift`` and n_A fresh |0>.  Syndrome symbol k sits at
    a[3 + k].  Transient width is bounded by the chunked thermometer
    (~_CHUNK*(b+1) live ancillas), matching the cycle block's transient
    scale; the per-chunk carry bits are erased at the end by RECOMPUTING
    each chunk's chain (syndrome data is untouched until the shift), trading
    ~800 Toffoli for ~700 qubits of transient width — the right trade under
    the Toffoli x qubits metric.
    """
    all_ks = list(range(N - 1, 0, -1))  # n-1..1
    chunks = [all_ks[i : i + _CHUNK] for i in range(0, len(all_ks), _CHUNK)]

    # --- forward: per-chunk scan + count, carry copies between chunks -------
    b.set_phase("init/scan")  # profile mark (metadata-only)
    carries: list[int] = []
    base = None
    for ci, ks in enumerate(chunks):
        parts, z_last = _chunk_scan(b, lay, ks, base, shift=shift)
        if ci < len(chunks) - 1:
            c = b.qubit()
            b.claim("scr/carry", [c])
            b.cx(z_last, c)
            carries.append(c)
            _chunk_release(b, parts)
            base = c
        else:
            _chunk_release(b, parts)

    # --- erase carries by recompute (reverse order; syndromes unchanged) ----
    for ci in range(len(carries) - 1, -1, -1):
        base = carries[ci - 1] if ci > 0 else None
        parts, z_last = _chunk_scan(b, lay, chunks[ci], base, shift=None)
        b.cx(z_last, carries[ci])
        _chunk_release(b, parts)
        b.release_recycle("scr/carry", [carries[ci]])

    # --- n_A = n - s: load n, subtract 2^k iff shift[k] ----------------------
    b.set_phase("init/count")  # profile mark (metadata-only)
    for i in range(CTR_W):
        if (N >> i) & 1:
            b.x(lay.n_A[i])
    for k in range(CTR_W):
        # subtract 2^k == controlled decrement of the bit-slice n_A[k..]
        _ctrl_decrement(b, shift[k], lay.n_A[k:])

    # --- binary controlled block-shifts (right by s, in place) --------------
    b.set_phase("init/shift")  # profile mark (metadata-only)
    for k in range(CTR_W - 1, -1, -1):
        step = 1 << k
        if step >= MPRIME:
            continue
        for i in range(MPRIME - 1, step - 1, -1):
            _cswap_regs(b, shift[k], lay.a[i], lay.a[i - step])

    # --- constants -----------------------------------------------------------
    b.set_phase("init/const")  # profile mark (metadata-only)
    b.x(lay.a[0][0])  # a = 1
    b.x(lay.bb[MPRIME - 1][0])  # B = z^n
    for i in range(CTR_W):
        if ((N + 1) >> i) & 1:
            b.x(lay.n_B[i])  # n_B = n + 1
    b.x(lay.n_a[0])  # n_a = 1
    b.x(lay.flag)  # flag = 1


def _claim_workspace(b, lay: EEALayout, shift: Sequence[int]) -> dict:
    """Ownership wire-set claims (metadata-only) on the persistent EEA
    workspace: every long-lived wire joins exactly one named set, full id
    list per claim.

    Returns the owner map {wire: set} that drives release-before-free on
    the teardown paths: q is released ahead of the eea/qfree frees (and
    re-claimed at its adjoint revival); every other workspace wire is
    released at its final free inside the tape adjoint
    (emit_adjoint).  The n*b syndrome qubits (lay.a[3..m'-1]) are the
    declared io registers, claimed as "syn" by the top level and owned to
    the end; AND scratch is the only anonymous pool.
    """
    sets = {
        "a": [q for reg in lay.a[: MPRIME - N] for q in reg],
        "bb": [q for reg in lay.bb for q in reg],
        "ctr": [
            *lay.n_A,
            *lay.n_B,
            *lay.n_a,
            *lay.n_b,
            *lay.n_q,
            *lay.counter,
            *lay.hc,
            lay.flag,
            lay.d2,
        ],
        "q": list(lay.q),
        "shift": list(shift),
    }
    owner = {}
    for name, wires in sets.items():
        b.claim(name, wires)
        for q in wires:
            owner[q] = name
    return owner


def emit_eea_forward(b, syndrome_regs: Sequence[Sequence[int]]) -> tuple[EEALayout, list[int]]:
    """Full forward pass.  Returns (layout, shift_register)."""
    b.set_phase("init/alloc")  # profile mark (metadata-only)
    lay = alloc_eea_layout(b, syndrome_regs)
    shift = b.qubits(CTR_W)
    # Owner map stash (metadata-only, the _profile_tape_cuts pattern): the
    # tape adjoint reads it back for release-before-free at each final free.
    b._wire_owner = _claim_workspace(b, lay, shift)
    # Profile-only tape cuts: when the caller records a structural tape,
    # remember where each segment (init, the cycle loop, flag, qfree)
    # starts on it so the adjoint replay can mirror the segment marks
    # (emit_adjoint).
    tape = getattr(b, "tape", None)
    cuts = [("init", len(tape))] if tape is not None else None
    emit_eea_init(b, lay, shift)  # loads n_A = n - s itself
    # The 222 cycles are one op-stream loop: the cycle is identical every
    # iteration and balanced (its ctx scratch is allocated and released
    # inside it), so it is recorded once.  A TapeBuilder records the body
    # as one ("loop", ...) tape token, reversed under its own repeat by the
    # adjoint (emit_adjoint).
    if cuts is not None:
        cuts.append(("cycle", len(tape)))  # ty: ignore[invalid-argument-type]  # cuts set => tape set
    with b.repeat(N_EEA_CYCLES):
        emit_eea_cycle(b, lay, phase="eea/cycle")
    # The Forney-exponent flag from the final counters (on the tape: the
    # adjoint recomputes the chain and clears d2 for FLAG_TOFFOLI more).
    b.set_phase("eea/flag")  # profile mark (metadata-only)
    if cuts is not None:
        cuts.append(("flag", len(tape)))  # ty: ignore[invalid-argument-type]  # cuts set => tape set
    emit_forney_flag(b, lay)
    # The quotient scratch is |0> at every cycle boundary (the release's own
    # `assert q_i == 0`; README "Accuracy" gives the measured residue rate), so it is
    # freed here; the sweep's transients reuse the slots (-b peak) and the
    # tape adjoint revives it before replaying the reversed cycles.
    b.set_phase("eea/qfree")  # profile mark (metadata-only)
    if cuts is not None:
        cuts.append(("qfree", len(tape)))  # ty: ignore[invalid-argument-type]  # cuts set => tape set
        b._profile_tape_cuts = cuts
    b.release_recycle("q", lay.q)  # release-before-free (qfree teardown path)
    return lay, shift


# Emitted init cost, derived (pinned against the emission by build()'s
# cost assert): chunked suffix-zero scan + counting + carry
# recompute-erase + binary block-shifts.


def _scan_toffoli() -> int:
    """Return the thermometer's Toffoli count: per symbol a [sym == 0] chain (b-1), a z AND (all
    but the first symbol) and s += z_k (w-1); then every chunk but the last
    is recomputed without the counting to erase its carry.
    """
    nsym = N - 1
    nchunks = -(-nsym // _CHUNK)
    fwd = nsym * (GF_B - 1) + (nsym - 1) + nsym * (CTR_W - 1)
    redo_syms = min(nsym, (nchunks - 1) * _CHUNK)
    redo = redo_syms * (GF_B - 1) + max(redo_syms - 1, 0)
    return fwd + redo


SCAN_TOFFOLI = _scan_toffoli()  # 2,032
COUNT_TOFFOLI = sum(CTR_W - k - 1 for k in range(CTR_W))  # 21: slice decrements
SHIFT_TOFFOLI = GF_B * sum(
    MPRIME - (1 << k) for k in range(CTR_W) if (1 << k) < MPRIME
)  # 4,608 = 12 x 384 CSWAPs
INIT_TOFFOLI = SCAN_TOFFOLI + COUNT_TOFFOLI + SHIFT_TOFFOLI  # 6,661

FORWARD_TOFFOLI = INIT_TOFFOLI + N_EEA_CYCLES * CYCLE_TOFFOLI + FLAG_TOFFOLI


# =========================================================================
# The Chien/Forney sweep with streamed phase payment
# =========================================================================

# An oblivious reversible transcription of the classical sweep in
# reference.py ``_sweep`` (the incremental form the released ConstructOrPhaseErrors / ConstructError
# realizes, rs_code_egcd_bloqs_zalka_eea.py:16-93) over all m = 4095 points
# gamma^0 .. gamma^4094, paying the phase (-1)^<c, e> as it goes.  The op
# sequence is fixed, so emitted Toffoli == executed Toffoli on every shot.
# Only lay.bb (the shared sigma|omega array) is touched, and it is THRU:
# slot k is multiplied by gamma^{+k} (omega, k > m'//2) or gamma^{-k}
# (sigma) after every point, and gamma^m = 1 restores it after the sweep.
#
# Per point i (error position j = (m - i) % m,
# rs_code_egcd_for_paper.py:76):
#   1. Evaluate (pure CX): XOR-accumulate om = omega slots 72..37,
#      sg = sigma slots 36..0, sgp = the odd-k sigma terms (the char-2
#      formal derivative).
#   2. Zero flag: root = [sg == 0] by a Gidney AND chain (b-1 = 11 Toffoli).
#   3. Forney quotient q1 = om / sgp: keep-junk inverse (255) + multiply
#      (51).  The circuit includes the multiply, as the released classical
#      decoder (rs_code_egcd_for_paper.py) does.  The Forney constant
#      gamma^{-expn i} is classical given the exponent, so step 4 folds it
#      into the phase mask.  The exponent m'-1 + n_b - n_B - 1
#      (rs_code_egcd_for_paper.py:75) is n + 2, or n + 3 on the 2^-b tail
#      where the error values XOR to zero; lay.d2 selects the tail.
#   4. Phase: over GF(2), <c_j, const*q1> = <M_const^T c_j, q1>, so the
#      payment is CZ(root, q1[t]) per set bit t of the folded mask c'_j.
#      This is exact on every basis state; at non-root points root = 0.
#      The tail correction d_j = c'_j XOR c''_j is paid as CZ(rf, q1[t])
#      with rf = root AND d2 (1 Toffoli), skipped where d_j = 0.
#   5. Erase q1, inv and its junk by measured uncompute (0 Toffoli),
#      release the root chain, and replay the step-1 accumulations.
#   6. Advance: slot k *= gamma^{+/-k} in place (CNOT lists; k = 0 elided).
#
# POINT_TOFFOLI = 317 per point, plus 1 at each of the N_CORR_POINTS
# corrected points.  SWEEP_TRANSIENT = 132 qubits exceeds the forward
# pass's transient, so the composed peak lands in the Forney step of
# point 1.


# ---------------------------------------------------------------------------
# Pinned instance constants
# ---------------------------------------------------------------------------

HALF = MPRIME // 2  # 36: sigma slots are k <= 36, omega slots are k > 36

# The Forney exponent m'-1 + n_b - n_B - 1 realized as classical constants:
# n_b - n_B == 1 (exponent n + 2) on all but the 2^-b tail of genuine
# weight >= 1 syndromes under the fixed stop rule, n_b - n_B == 2
# (exponent n + 3) on that tail, selected by the flag qubit lay.d2; the
# weight-0 machine yields sigma = 1 (no roots), so no constant is consumed
# there.
FORNEY_EXPN = MPRIME - 1  # 72 = n + 2: the common case
FORNEY_EXPN_TAIL = FORNEY_EXPN + 1  # 73 = n + 3: the n_b - n_B = 2 tail
assert FORNEY_EXPN == N + 2

# Slot -> per-point step constant: gamma^k or gamma^-k, the ratio between
# consecutive points of the slot's term in reference.py ``_sweep``.
STEP_CONST: list[int] = [GF.pow(2, k) if k > HALF else GF.pow(2, -k) for k in range(MPRIME)]
assert STEP_CONST[0] == 1  # the k = 0 step is the identity (elided below)

# Evaluation slot index sets (the omega and sigma loops of reference.py
# ``_sweep``, descending order).
OMEGA_SLOTS = list(range(MPRIME - 1, HALF, -1))  # 72 .. 37  (36 slots)
SIGMA_SLOTS = list(range(HALF, -1, -1))  # 36 .. 0   (37 slots)

# Per-point Toffoli profile (section header):
# the root AND chain (GF_B - 1) + the quantum-quantum multiplies of one
# keep-junk inversion plus the Forney product, + 1 (root AND d2) at every
# corrected point.
POINT_TOFFOLI = (GF_B - 1) + (ITOH_TSUJII_QQ_MULTS + 1) * OPT_MULT_TOFFOLI  # 317
CORR_TOFFOLI = 1
# Per-point transient width: om/sg/sgp + the root chain + inv + the kept
# inversion junk + q1 + the root AND d2 wire.
SWEEP_TRANSIENT = 3 * GF_B + (GF_B - 1) + GF_B + (ITOH_TSUJII_QQ_MULTS * GF_B) + GF_B + 1  # 132


# ---------------------------------------------------------------------------
# In-place constant multiply: CNOT synthesis (GF2MulK semantics)
# ---------------------------------------------------------------------------

_MULK_CACHE: dict[int, list[tuple[int, int]]] = {}


def _mulk_cnots(c: int) -> list[tuple[int, int]]:
    """CNOT list [(ctrl_idx, tgt_idx), ...] realizing reg <- c * reg IN PLACE.

    The multiply-by-c matrix M (M[j][i] = coeff j of c * x^i mod p) is
    invertible for c != 0; over GF(2) every invertible matrix is a product of
    transvections (row additions), i.e. of CNOTs.  Gauss-Jordan reduce M to I
    recording row ops; the circuit is the recorded ops in reverse (each is
    self-inverse).  Verified on all basis vectors before caching.

    local: shared linear.linear_in_place applies the same invertible GF(2)
    map at the same cost (0 Toffoli, 0 ancillas); kept local for the
    CX-only netlist (no SWAPs), the per-constant memo cache, and that
    per-basis-vector replay self-check.
    """
    if c in _MULK_CACHE:
        return _MULK_CACHE[c]
    if not 0 < c < (1 << GF_B):
        raise ValueError(f"in-place multiply needs invertible constant, got {c}")
    rows = f2_transpose(_fold_cols(c), GF_B)  # rows[j] = bitmask of M[j][i]
    ops: list[tuple[int, int]] = []  # (target_row, ctrl_row): row_t ^= row_c
    for col in range(GF_B):
        if not (rows[col] >> col) & 1:
            r = next(r for r in range(col + 1, GF_B) if (rows[r] >> col) & 1)
            rows[col] ^= rows[r]
            ops.append((col, r))
        for r in range(GF_B):
            if r != col and (rows[r] >> col) & 1:
                rows[r] ^= rows[col]
                ops.append((r, col))
    assert rows == [1 << j for j in range(GF_B)]
    # (E_k ... E_1) M = I  =>  M = E_1 ... E_k; apply E_k first => reversed.
    cnots = [(cr, tr) for tr, cr in reversed(ops)]
    # Self-check: replay on every basis vector against the classical multiply.
    for i in range(GF_B):
        v = 1 << i
        for ctrl, tgt in cnots:
            if (v >> ctrl) & 1:
                v ^= 1 << tgt
        assert v == GF.mul(c, 1 << i), f"mulk synthesis broken for c={c}"
    _MULK_CACHE[c] = cnots
    return cnots


def _emit_inplace_mulk(b, reg: Sequence[int], c: int) -> None:
    """reg <- c * reg, pure CX (0 Toffoli)."""
    for ctrl, tgt in _mulk_cnots(c):
        b.cx(reg[ctrl], reg[tgt])


# ---------------------------------------------------------------------------
# Classical mask folding: c' = M_c^T c so that <c', v> == <c, c*v> over GF(2)
# ---------------------------------------------------------------------------


def fold_mask_const(cvec: int, c: int) -> int:
    """Return the folded mask slice c' = M_c^T c (b-bit ints, little-endian).

    Bit t' of c' is <c, column t' of M_c> = parity(cvec & (c * x^t')),
    which gives the exact identity <c', v> == <c, c*v> for every v — the
    XOR-linearity fact the direct-CZ phase payment rests on.  Self-checked on
    deterministic probe values per (cvec, c) pair via _fold_selfcheck.
    """
    cols = _fold_cols(c)
    out = 0
    for tp in range(GF_B):
        if parity(cvec & cols[tp]):
            out |= 1 << tp
    return out


def _fold_selfcheck(cvec: int, c: int, cprime: int) -> None:
    """assert <c', v> == <c, c*v> on deterministic probe values (build-time)."""
    full = (1 << GF_B) - 1
    for v in (full, 0x555 & full, 0xAAA & full, 0x1E7 & full, 1, 1 << (GF_B - 1)):
        lhs = parity(cprime & v)
        rhs = parity(cvec & GF.mul(c, v))
        assert lhs == rhs, f"mask fold broken: cvec={cvec:#x} c={c:#x} v={v:#x}"


def fold_masks(mask: Sequence[int]) -> tuple[list[int], list[int]]:
    """Per point i: (c'_i, d_i) — the mask slice at error position j =
    (m - i) % m folded by the common-case constant gamma^{-(n+2) i}, and
    the XOR difference to the tail fold at gamma^{-(n+3) i}.  Every
    distinct (slice, constant) pair is self-checked once.
    """
    if len(mask) != M * GF_B:
        raise ValueError(f"mask must have {M * GF_B} bits, got {len(mask)}")
    folded: list[int] = []
    diff: list[int] = []
    checked = set()
    for i in range(M):
        j = (M - i) % M
        cvec = 0
        for t in range(GF_B):
            if mask[GF_B * j + t]:
                cvec |= 1 << t
        cps = []
        for expn in (FORNEY_EXPN, FORNEY_EXPN_TAIL):
            const_i = GF.pow(2, -expn * i)
            cp = fold_mask_const(cvec, const_i)
            if (cvec, const_i) not in checked:
                _fold_selfcheck(cvec, const_i, cp)
                checked.add((cvec, const_i))
            cps.append(cp)
        folded.append(cps[0])
        diff.append(cps[0] ^ cps[1])
    return folded, diff


# The pinned mask's folds (the contract's c): N_CORR_POINTS is the number
# of points paying the d2-gated correction (d_i != 0: every point but
# i = 0, where both constants are 1, and the positions whose mask slice is
# zero).
FOLDED_MASK, DIFF_MASK = fold_masks(C_BITS)
N_CORR_POINTS = sum(1 for d in DIFF_MASK if d)
SWEEP_TOFFOLI = M * POINT_TOFFOLI + N_CORR_POINTS * CORR_TOFFOLI
assert DIFF_MASK[0] == 0


# ---------------------------------------------------------------------------
# The sweep block
# ---------------------------------------------------------------------------


def _pay_phase(b, i, root, q1, d2, t_fold, t_diff, t_corr):
    # ---- 4. direct phase payment: CZ(root, q1) under the folded mask ---
    # (-1)^{root*<c'_j, q1>} == (-1)^<c_j, root*(const_i*q1)> exactly
    # (XOR-linearity; const_i folded classically — no error register).
    # Inert when root = 0 (non-root points).
    for t in range(GF_B):
        with b.gated(t_fold, i * GF_B + t):
            b.cz(root, q1[t])
    # the n_b - n_B = 2 tail: (-1)^{root*d2*<d_j, q1>} on the fold
    # difference, gated by rf = root AND d2 (one temporary AND, only
    # where d_i != 0; q1 is garbage at non-root points, so root must
    # stay in the gate)
    with b.gated(t_corr, i):
        rf = b.and_compute(root, d2)
    for t in range(GF_B):
        with b.gated(t_diff, i * GF_B + t):
            b.cz(rf, q1[t])
    with b.gated(t_corr, i):
        b.and_release(rf, root, d2)


def emit_sweep(b, lay: EEALayout) -> None:
    """Emit the full m-point Chien/Forney sweep with streamed phase payment.

    Reads and steps ONLY lay.bb (THRU: restored after the last step) and
    reads lay.d2 (the Forney-exponent flag); pays the phase
    (-1)^<c, e(state)> where e is the error vector that reference.py
    ``_sweep`` decodes from the machine state held in the register file,
    under the grader-pinned c-mask (C_BITS, little-endian bit t of e_j at
    mask index b*j + t), folded once at import (FOLDED_MASK, DIFF_MASK).

    The points are ONE repeat(M) body.  Every per-point difference
    is a pinned constant bit — the folded mask c'_i, the correction slice
    d_i, and [d_i != 0] for the root-AND-d2 pair — so those gates are
    data-gated on declared tables.  Phase marks: sweep/point/{eval,forney,
    step}.
    """
    folded, diff = FOLDED_MASK, DIFF_MASK
    t_fold = b.declare_table([(folded[i] >> t) & 1 for i in range(M) for t in range(GF_B)])
    t_diff = b.declare_table([(diff[i] >> t) & 1 for i in range(M) for t in range(GF_B)])
    t_corr = b.declare_table([int(d != 0) for d in diff])

    with b.repeat(M) as i:
        # ---- 1. running evaluation: om / sg / sgp (pure CX) ----------------
        b.set_phase("sweep/point/eval")  # profile mark (metadata-only)
        om = b.qubits(GF_B)
        sg = b.qubits(GF_B)
        sgp = b.qubits(GF_B)
        b.claim("ev/om", om)
        b.claim("ev/sg", sg)
        b.claim("ev/sgp", sgp)
        acc = [(lay.bb[k], om) for k in OMEGA_SLOTS] + [
            p
            for k in SIGMA_SLOTS
            for p in [(lay.bb[k], sg)] + ([(lay.bb[k], sgp)] if k & 1 else [])
        ]
        for src, dst in acc:
            for t in range(GF_B):
                b.cx(src[t], dst[t])

        # ---- 2. zero flag: root = [sigma_eval == 0] (b-1 Toffoli) ----------
        root, root_trail = eq_const(b, sg, 0)

        # ---- 3. Forney quotient q1 = om / sgp (unconditional) --------------
        b.set_phase("sweep/point/forney")  # profile mark (metadata-only)
        with _quotient(b, om, sgp, "gfw/q1") as q1:
            _pay_phase(b, i, root, q1, lay.d2, t_fold, t_diff, t_corr)

        release_chain(b, root_trail)  # measured, 0 Toffoli; sg unchanged since compute
        for src, dst in reversed(acc):
            for t in range(GF_B):
                b.cx(src[t], dst[t])
        b.release_recycle("ev/sgp", sgp)
        b.release_recycle("ev/sg", sg)
        b.release_recycle("ev/om", om)

        # ---- 6. advance the running registers (k = 0 step is identity) -----
        b.set_phase("sweep/point/step")  # profile mark (metadata-only)
        for k in range(1, MPRIME):
            _emit_inplace_mulk(b, lay.bb[k], STEP_CONST[k])


# =========================================================================
# The tape adjoint and build()
# =========================================================================

# FORWARD_TOFFOLI (the forward-pass section): 1,911,431 = init 6,661 + 222 x 8,580 + flag 10
ADJOINT_TOFFOLI = FORWARD_TOFFOLI

_INV_OK = {int(OpKind.X), int(OpKind.CX), int(OpKind.CCX), int(OpKind.SWAP)}


class TapeBuilder(Builder):
    """Builder recording the structural tape of the forward pass.

    Recording is on while ``tape`` is a list.  Gadgets that provide their own
    adjoint are recorded as one macro token via :meth:`macro` (the seam
    _macro calls into); everything else is recorded op by op.

    local: Builder.mirror cannot express this adjoint.  A multiplier
    macro inverts through its own gadget pair: gf_mult_uncompute erases a
    product at 0 Toffoli, where a mirror of the CCX list would pay the
    product again, and a recorded erase holds HMR ops that mirror
    rejects.  The adjoint also revives freed wires with reclaim and
    re-claims their owners.
    """

    def __init__(self):
        self.tape: list | None = None
        self._suppress = 0
        super().__init__()

    def _record(self, tok) -> None:
        if self.tape is not None and self._suppress == 0:
            self.tape.append(tok)

    @contextmanager
    def _suppressed(self):
        """Emit the enclosed ops without recording them on the tape."""
        self._suppress += 1
        try:
            yield
        finally:
            self._suppress -= 1

    @contextmanager
    def macro(self, *token):
        """Record a gadget call as ONE structural token: suppress recording
        of the inner ops, then append the token on exit (op emission is
        unaffected — the tape only shapes the adjoint).
        """
        if self.tape is None:
            yield
            return
        with self._suppressed():
            yield
        self._record(token)

    def _emit(
        self,
        kind,
        qa=NO_QUBIT,
        qb=NO_QUBIT,
        qt=NO_QUBIT,
        ot=NO_OUT,
        oc=NO_OUT,
        rt=NO_REG,
        ct=NO_BIT,
        cc=NO_BIT,
    ):
        super()._emit(kind, qa, qb, qt, ot, oc, rt, ct, cc)
        # the flat 9-int record spelling, also inside a repeat body
        # (where the Builder records a compact OpRec instead)
        self._record(("op", (int(kind), qa, qb, qt, ot, oc, rt, ct, cc)))

    @contextmanager
    def repeat(self, count):
        """Builder.repeat, recording the body as ONE ("loop", count, body)
        token on its own sub-tape: emit_adjoint reverses the body once and
        re-emits it under its own repeat(count).  The body must address
        fixed wires (no affine operands) — the reversed body replays the
        recorded ids.
        """
        if self.tape is None or self._suppress:
            with super().repeat(count) as i:
                yield i
            return
        outer, self.tape = self.tape, []
        try:
            with super().repeat(count) as i:
                yield i
        finally:
            body, self.tape = self.tape, outer
        self._record(("loop", count, body))

    def qubit(self):
        q = super().qubit()
        self._record(("alloc", q))
        return q

    def recycle(self, q, reset_recorded=False):
        # The tape replays every free with its own R, so a free whose R lives
        # elsewhere has no tape spelling.
        if reset_recorded:
            raise ValueError("TapeBuilder cannot record a reset_recorded free")
        with self._suppressed():
            super().recycle(q)
        self._record(("free", q))

    # The pol= X-conjugation stays OUTSIDE the suppressed window in both
    # AND overrides: the tape then carries the X gates as plain
    # self-inverse ops around the pol-less and-token, so the reversed walk
    # re-emits them in reversed order with no pol bookkeeping of its own.

    def _pol_x(self, c1, c2, pol):
        """X each control whose polarity is False (self-inverse)."""
        if not pol[0]:
            self.x(c1)
        if not pol[1]:
            self.x(c2)

    def and_compute(self, c1, c2, pol=(True, True)):
        self._pol_x(c1, c2, pol)
        with self._suppressed():
            t = super().and_compute(c1, c2)
        self._record(("andc", c1, c2, t))
        self._pol_x(c1, c2, pol)
        return t

    def and_release(self, t, c1, c2, pol=(True, True), cond=None):
        assert cond is None, "baseline never conditions a release"
        self._pol_x(c1, c2, pol)
        with self._suppressed():
            m = super().and_release(t, c1, c2)
        self._record(("andr", t, c1, c2))
        self._pol_x(c1, c2, pol)
        return m


# -- adjoint emitter ------------------------------------------------------------


def emit_adjoint(b: TapeBuilder, tape: list) -> None:
    """Replay the recorded tape backwards, swapping compute/release roles.

    Multiplier macro tokens invert via the gadget pair itself: a recorded
    "mult" is erased by gf_mult_uncompute, a recorded "mule" is recomputed by
    gf_mult_opt (recording is off here, so the calls emit plainly).  A
    recorded ("loop", count, body) token — a TapeBuilder.repeat body — is
    re-emitted as the reversed body under its own repeat(count): the
    reversed body is balanced because the forward body is, so the adjoint
    stream stays loop-compressed and the peak does not move.

    Profile marks (metadata-only): when emit_eea_forward stashed tape cuts on the
    builder, the reversed walk mirrors them as ``adjoint/<segment>`` phase
    marks (qfree, flag, cycle, init) at each segment entry.

    Ownership (metadata-only): when emit_eea_forward stashed its owner map
    ({wire: set}) on the builder, every owned wire is released right before
    its final free here — the tape adjoint is the teardown path of the
    persistent register file — and q re-joins its set at its revival (the
    replay of the eea/qfree frees).  Revived transients stay anonymous: the
    tape carries no set identity, and they mirror the forward transients'
    lifetimes exactly.
    """
    assert b.tape is None, "stop recording before the adjoint"
    cuts = getattr(b, "_profile_tape_cuts", None)  # profile-only tape cuts
    owner = getattr(b, "_wire_owner", None)  # {wire: set} release-before-free
    ci = len(cuts) - 1 if cuts else -1
    cur_seg = None
    for pos in range(len(tape) - 1, -1, -1):
        if cuts:
            while ci > 0 and cuts[ci][1] > pos:
                ci -= 1
            if cuts[ci][0] != cur_seg:
                cur_seg = cuts[ci][0]
                b.set_phase(f"adjoint/{cur_seg}")
        _emit_reversed_token(b, tape[pos], owner)


def _emit_reversed_token(b: TapeBuilder, tok: tuple, owner) -> None:
    """Emit the adjoint of one tape token.  The adjoint replays a recorded
    free/erase backwards as a re-activation: reclaim([q]) pulls the
    named id off the free-list with stock allocation accounting and
    events.
    """
    kind = tok[0]
    if kind == "op":
        op = tok[1]
        if op[0] not in _INV_OK or op[5] != NO_OUT:
            raise AssertionError(f"non-self-inverse plain op in tape: {op}")
        if not all(type(v) is int for v in op):
            raise AssertionError(f"affine operand in a taped loop body: {op}")
        b._emit(*op)  # TapeBuilder._emit: recording is off, so it just emits
    elif kind == "loop":
        _, count, body = tok
        with b.repeat(count):
            for sub in reversed(body):
                _emit_reversed_token(b, sub, owner)
    elif kind == "alloc":
        q = tok[1]
        if owner is not None and q in owner:
            b.release(owner[q], [q])
        b.recycle(q)
    elif kind == "free":
        q = tok[1]
        b.reclaim([q])
        if owner is not None and q in owner:
            b.claim(owner[q], [q])
    elif kind == "andc":
        _, c1, c2, t = tok
        b.and_release(t, c1, c2)
    elif kind == "andr":
        _, t, c1, c2 = tok
        b.reclaim([t])
        b.ccx(c1, c2, t)
    elif kind == "mult":
        _, x, y, out = tok
        gf_mult_uncompute(b, list(out), list(x), list(y), free_out=False)
    elif kind == "mule":
        _, out, x, y, free_out = tok
        assert free_out, "baseline erases always free their product"
        b.reclaim(list(out))
        gf_mult_opt(b, list(x), list(y), list(out))
    else:  # pragma: no cover
        raise AssertionError(f"unknown tape token {tok!r}")


# -- full build ----------------------------------------------------------------


def build() -> tuple[TapeBuilder, int]:
    """Run the full build at the certified pins; return (builder, emitted
    Toffoli) — the raw-build accessor shape the profilers take.
    """
    b = TapeBuilder()
    # The grader's 70 x 12-qubit THRU syndrome registers (ids 0..839),
    # allocated and declared through the shared pairing.
    syn = list(declare_registers(b, [(f"S{r + 1}", GF_B) for r in range(N)]).values())
    # The declared I/O registers own their wires to the end of the build;
    # every other set is claimed in _claim_workspace (persistent
    # file) or at its allocation site (non-AND transients).
    b.claim("syn", [w for reg in syn for w in reg])

    # Phase labels are the profiler's marks (metadata-only, hierarchical):
    # init/* and eea/cycle/* inside emit_eea_forward, sweep/point/* inside
    # emit_sweep, adjoint/* mirrored from the tape cuts in emit_adjoint.
    b.tape = []
    lay, _shift = emit_eea_forward(b, syn)
    tape, b.tape = b.tape, None
    emit_sweep(b, lay)
    emit_adjoint(b, tape)

    tof = assert_metrics(b, __file__)
    return b, tof


if __name__ == "__main__":
    b, tof = build()
    print(f"ops {b.expanded_size():,}  outcomes {b.next_out:,}")
    print(
        f"emitted Toffoli {tof:,} "
        f"(forward {FORWARD_TOFFOLI:,} + sweep {SWEEP_TOFFOLI:,} "
        f"+ adjoint {ADJOINT_TOFFOLI:,})"
    )
    # q is freed after the forward pass (exact, on the release's own q==0
    # invariant); the sweep transient includes the kept inversion junk.
    print(
        f"peak {b.peak_qubits:,} (forward persistent {FORWARD_PERSISTENT:,} "
        f"- q {GF_B} + sweep transient {SWEEP_TRANSIENT})"
    )
    emit_and_report(b)
