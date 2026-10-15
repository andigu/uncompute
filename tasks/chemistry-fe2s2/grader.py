"""Grader for `chemistry-fe2s2`.  reference.py derives the same table
from the FCIDUMP file and checks it.

The task is the complete data-dependent content of one qubitization walk
step for the pinned [2Fe-2S] (30e,20o) iron-sulfur Hamiltonian, BOTH
directions: the alias-sampling decision FUSED with the SELECT application
of the chosen Pauli string to the 40-qubit system register, the
coefficient sign folded into the shot's global phase, and the selection
ERASED again before the boundary — no output register survives:

    |i⟩₁₆ |σ⟩₁₆ |x⟩₄₀  →  (−1)^phase |i⟩ |σ⟩ |x ⊕ flip(sel)⟩

    f     = [σ < keep_i]                     (1 bit — strict comparator)
    sel   = f ? i : alt_i                    (16 bits — the sampled term)
    phase = θ_sel ⊕ ny2(sel) ⊕ par(x ∧ zy(sel))

f and sel are interior values: the clean-workspace rule (every ancilla
back to |0⟩ before free) grades their erasure, and the exact global sign
grades the erasure's phase fixups on the sampled measurement records.
Distinct rows are distinct Jordan-Wigner strings — (flip, zy) determines
a Pauli string uniquely — so grading x over the full 40-bit space pins
zy(sel) through the sign and flip(sel) through the output, hence sel
itself on every (i, σ).

per row of the pinned table (L = 41,052, μ = 16): flip = the string's X∪Y
support, zy = its Z∪Y support, ny2 = (Y-count mod 4)/2.  The table is the
Jordan-Wigner term list of the Li-Chan [2Fe-2S] (30e,20o) active space,
truncated at |w| ≥ 1e-4 Ha (the sparse-qubitization threshold practice),
ordered in frame blocks — over the seven-frame commuting family
{F, C, D, D2, s_thio, s_d45, x_muS} (D2 = the composed hybrid
reflection), every row taking the largest of 13 fixed frame sets whose
mask orbit survives in the table, blocks size-descending with duplicate
blocks contiguous at aligned slots (the 14 contract regions), branch
bit p of a block keyed to the p-th frame of its set — and converted to
a deterministic credit-aligned melt alias table (heavy larges share keep
words through exact whole-entry subset fills with class-aligning
donation routing, near-threshold larges close through credit chains;
reference.py steps 2-9).

Correctness: ZERO tolerance — check_shot is the Task default (exact
equality) and the failure policy is STRICT.  The only freedom: behavior
on i ≥ 41,052 is unconstrained (the grader never generates it).  x is
graded over the FULL 40-bit space — the fermionic even-parity sector is
physical context, not a test-domain restriction.

Test scheme:

1. STRUCTURED PROBES, 41,052·5·8 = 1,642,080 deterministic shots: every
   row × σ ∈ {0, 2¹⁶−1, keep−1, keep, keep+1} × 8 x probes
   {0, all-ones, HF 0x3fffffff, flip(i), flip(alt_i), 3 SHA-derived
   values, the first forced to odd parity so every row exercises both
   parity sectors}.
2. SEEDED RANDOM SAMPLE, 2²⁴ shots over the joint (i, σ, x)
   domain: i by 32-bit rejection (unbiased; reject probability
   24,952/2³²), σ uniform, x uniform over all 40 bits.

Every gen_test_indexed call consumes exactly 16 XOF bytes (structured
shots discard them) so the shot-indexed byte stream is constant-rate.

Data provenance: the alias words are loaded from data/table.json.xz
(emitted by reference.py --emit from the pinned FCIDUMP) and pinned by
DATA_SHA256; the Pauli-action masks are re-derived here from the same
table's Pauli strings — the identical derivation as reference.py — and
pinned by MASKS_SHA256 (the blob materialized in data/masks.json.xz).
A stale or edited data file can never grade silently.

Run this file directly to run its self-test.
"""

import hashlib
import json
import lzma
import pathlib

from uncompute.evalcore import QUBIT, RegSpec, Task, TestCase

HERE = pathlib.Path(__file__).resolve().parent
TABLE_XZ = HERE / "data" / "table.json.xz"

# Number of table rows ([2Fe-2S] (30e,20o) JW terms at the 1e-4 cutoff).
L = 41052
# Index register width: ⌈log₂ 41052⌉.
IDX_BITS = 16
# Comparand register width (μ of the alias quantization).
SIGMA_BITS = 16
# System register width (spin-orbitals of the (30e,20o) active space).
X_BITS = 40
# Data word width: theta(1) + alt_theta(1) + alt(16) + keep(16).
B_WORD = 34
# The Hartree-Fock determinant: 30 electrons in the first 30 spin-orbitals.
HF_STATE = (1 << 30) - 1

# σ probe slots per row / x probe slots per row (structured stratum).
SIGMA_SLOTS = 5
X_PROBES = 8
# Structured shots: every row × every σ slot × every x probe.
STRUCT_SHOTS = L * SIGMA_SLOTS * X_PROBES  # 1,642,080
# Seeded random shots over the joint (i, σ, x) domain.
RANDOM_SHOTS = 1 << 24  # 16,777,216
# Total shots.
NUM_SHOTS = STRUCT_SHOTS + RANDOM_SHOTS  # 18,419,296

# Unbiased row rejection bound: the largest multiple of L below 2³².
REJECT_LIMIT = 4_294_942_344  # (2^32 // 41052) * 41052

# Seed domain: separates the seeded draw and the SHA-derived x probes.
SEED_DOMAIN = b"chemistry-fe2s2"

# SHA-256 of the alias words as little-endian u64 (reference.py pin).
DATA_SHA256 = "1fd83441ba295e3dace47ee25224bfb84a539d1c63c14e0a1ce02c2fb8c03e45"
# SHA-256 of the MASKS array as 16-byte little-endian blocks
# (reference.py pin, the blob materialized in data/masks.json.xz).
MASKS_SHA256 = "b970bec6c3689c53de7738c7f6a7454cdf07e641e27e4b048fc1b5eb52a02751"

# Contract pin: contract.json is the shared statement of these derived
# values (circuits load it instead of hardcoding the digests). The inline
# derivation above stays the authority — this check is assert-only.
_CONTRACT = json.loads((HERE / "contract.json").read_text())
assert (
    _CONTRACT["L"],
    _CONTRACT["idx_bits"],
    _CONTRACT["sigma_bits"],
    _CONTRACT["x_bits"],
    _CONTRACT["b_word"],
    _CONTRACT["data_sha256"],
    _CONTRACT["masks_sha256"],
) == (L, IDX_BITS, SIGMA_BITS, X_BITS, B_WORD, DATA_SHA256, MASKS_SHA256), (
    "contract.json drifted from the grader's inline derivation"
)
assert "out_bits" not in _CONTRACT, "the round-trip contract has no output register"


def _parse_pauli(pauli):
    """flip / zy / ny of one Pauli string (identical to reference.py)."""
    flip = zy = 0
    ny = 0
    for tok in pauli.split():
        p, q = tok[0], int(tok[1:])
        assert p in "XYZ" and 0 <= q < X_BITS, tok
        if p in "XY":
            flip |= 1 << q
        if p in "ZY":
            zy |= 1 << q
        if p == "Y":
            ny += 1
    assert ny % 2 == 0, "odd Y-count would give a ±i phase: out of model"
    return flip, zy, ny


def _load_tables():
    t = json.loads(lzma.decompress(TABLE_XZ.read_bytes()))
    assert t["L"] == L and t["n_qubits"] == X_BITS
    data = [int(r["word"], 16) for r in t["terms"]]
    assert len(data) == L, "pinned table length drifted"
    blob = b"".join(w.to_bytes(8, "little") for w in data)
    assert hashlib.sha256(blob).hexdigest() == DATA_SHA256, (
        "pinned table drifted — regenerate ONLY via reference.py --emit"
    )
    masks = []
    for r in t["terms"]:
        flip, zy, ny = _parse_pauli(r["pauli"])
        masks.append(flip | (zy << X_BITS) | (((ny % 4) // 2) << 80))
    mblob = b"".join(m.to_bytes(16, "little") for m in masks)
    assert hashlib.sha256(mblob).hexdigest() == MASKS_SHA256, (
        "mask table drifted — regenerate with reference.py --emit and re-audit"
    )
    return data, masks


DATA, MASKS = _load_tables()

# Per-row field tables.
_XMASK = (1 << X_BITS) - 1
KEEP = [w & 0xFFFF for w in DATA]  # keep field (low 16 bits)
ALT = [(w >> 16) & 0xFFFF for w in DATA]  # alt field (bits 16..31)
THETA = [(w >> 33) & 1 for w in DATA]  # θ (bit 33)
ALT_THETA = [(w >> 32) & 1 for w in DATA]  # prefetched θ[alt_i] (bit 32)
FLIP = [m & _XMASK for m in MASKS]  # X∪Y support
ZY = [(m >> X_BITS) & _XMASK for m in MASKS]  # Z∪Y support
NY2 = [(m >> 80) & 1 for m in MASKS]  # (Y-count mod 4)/2


def oracle(i, sigma, x):
    """(17-bit word f‖sel — interior, self-test use only —
    x ⊕ flip(sel), expected phase bit).
    """
    if sigma < KEEP[i]:
        f, sel, th = 1, i, THETA[i]
    else:
        f, sel, th = 0, ALT[i], ALT_THETA[i]
    word = (f << IDX_BITS) | sel
    x_out = x ^ FLIP[sel]
    phase = th ^ NY2[sel] ^ ((x & ZY[sel]).bit_count() & 1)
    return word, x_out, phase == 1


def probe_sigma(i, j):
    """σ probe slot j for row i: both boundaries and both extremes."""
    k = KEEP[i]
    if j == 0:
        return 0
    if j == 1:
        return 0xFFFF
    if j == 2:
        return k - 1 if k > 0 else 0  # saturating_sub
    if j == 3:
        return k
    return min(k + 1, 0xFFFF)


def probe_x(i, k):
    """x probe slot k for row i: the named states (0, all-ones, HF,
    flip(i), flip(alt_i)) plus three SHA-derived values; slot 5 is
    forced to odd parity so every row deterministically exercises both
    parity sectors.
    """
    if k == 0:
        return 0
    if k == 1:
        return _XMASK  # all-ones (weight 40)
    if k == 2:
        return HF_STATE  # Hartree-Fock determinant (weight 30)
    if k == 3:
        return FLIP[i]  # even weight (superselection structure)
    if k == 4:
        return FLIP[ALT[i]]
    h = hashlib.sha256()
    h.update(SEED_DOMAIN)
    h.update(b"x-probe")
    h.update(i.to_bytes(4, "little"))
    h.update(bytes([k - 5]))
    d = h.digest()
    x = int.from_bytes(d[:5], "little") & _XMASK
    if k == 5 and not x.bit_count() & 1:
        x ^= 1  # slot 5: force odd parity (an exact 2-to-1 map)
    return x


# Structured-probe tables, precomputed once (SIGMA_SLOTS/X_PROBES per row).
_PSIG = [[probe_sigma(i, j) for j in range(SIGMA_SLOTS)] for i in range(L)]
_PX = [[probe_x(i, k) for k in range(X_PROBES)] for i in range(L)]


class ChemistryFe2S2(Task):
    name = "chemistry-fe2s2"
    seed_domain = SEED_DOMAIN
    num_tests = NUM_SHOTS
    # Zero tolerance is the contract: STRICT failure policy (default),
    # Task-default check_shot — deliberately no overrides, no budgets.

    def schema(self):
        return [
            RegSpec(width=IDX_BITS, kind=QUBIT),  # 0: i
            RegSpec(width=SIGMA_BITS, kind=QUBIT),  # 1: σ
            RegSpec(width=X_BITS, kind=QUBIT),  # 2: x → x⊕flip
        ]

    def gen_test(self, xof):  # noqa: ARG002 - overrides Task.gen_test
        raise AssertionError(
            "chemistry-fe2s2 is shot-indexed; the evaluator calls gen_test_indexed"
        )

    def gen_test_indexed(self, shot, xof):
        """Shots [0, STRUCT_SHOTS) are the deterministic structured probes
        (shot = row·40 + σ_slot·8 + x_slot); shots [STRUCT_SHOTS,
        NUM_SHOTS) are seeded random draws.  EVERY call consumes exactly
        16 XOF bytes (structured shots discard them); the only skip is
        the unbiased-row rejection (probability 24,952/2³² per random
        shot).
        """
        assert shot < NUM_SHOTS, "num_tests bounds the shot index"
        buf = xof.read(16)
        if shot < STRUCT_SHOTS:
            row, rest = divmod(shot, SIGMA_SLOTS * X_PROBES)
            i = row
            sigma = _PSIG[row][rest // X_PROBES]
            x = _PX[row][rest % X_PROBES]
        else:
            r = int.from_bytes(buf, "little")
            r32 = r & 0xFFFFFFFF
            if r32 >= REJECT_LIMIT:
                return None  # unbiased rejection; consumption already constant
            i = r32 % L
            sigma = (r >> 32) & 0xFFFF
            x = (r >> 48) & _XMASK  # uniform over the full 40-bit space
        _, x_out, phase = oracle(i, sigma, x)
        return TestCase(
            expected_phase=phase,
            inputs=[i, sigma, x],
            expected=[i, sigma, x_out],
        )


TASK = ChemistryFe2S2()


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # pinned_digests (asserted at load; re-checked here).
    assert len(DATA) == L and len(MASKS) == L and L == 41052
    blob = b"".join(w.to_bytes(8, "little") for w in DATA)
    assert hashlib.sha256(blob).hexdigest() == DATA_SHA256
    mblob = b"".join(m.to_bytes(16, "little") for m in MASKS)
    assert hashlib.sha256(mblob).hexdigest() == MASKS_SHA256
    print("PASS  pinned digests (DATA, MASKS)")

    # mask/table structure invariants.
    fw = [0] * (X_BITS + 1)
    ny2_sum = 0
    for i in range(L):
        assert MASKS[i] >> 81 == 0, f"row {i}: stray bits above ny2"
        w = FLIP[i].bit_count()
        assert w % 2 == 0, f"row {i}: odd flip weight breaks superselection"
        fw[w] += 1
        ny2_sum += NY2[i]
        y_count = (FLIP[i] & ZY[i]).bit_count()
        assert y_count % 2 == 0, f"row {i}: odd Y-count is out of model"
        assert NY2[i] == (y_count % 4) // 2, f"row {i}: ny2 mismatch"
        assert ALT_THETA[i] == THETA[ALT[i]], f"row {i}: prefetch identity"
    assert (fw[0], fw[2], fw[4]) == (820, 23304, 16928)
    assert sum(fw) == L
    assert ny2_sum == 23438
    assert sum(THETA) == 19614
    # the zero-keep stratum: exactly one row, self-aliasing (the
    # conservation-closure large; it lives in the F|D region)
    assert [i for i in range(L) if KEEP[i] == 0] == [40608] and ALT[40608] == 40608
    assert max(KEEP) == 65535
    # derivation censuses (asserted end-to-end by reference.py)
    n_pairs = _CONTRACT["n_pairs"]
    assert n_pairs == 20400 and _CONTRACT["n_selfsym"] == 252
    assert (_CONTRACT["n_tokens"], _CONTRACT["n_groups"], _CONTRACT["n_residual_pairs"]) == (
        9942,
        4152,
        516,
    )
    assert (
        _CONTRACT["n_sigma_groups"],
        _CONTRACT["n_sigma_tokens"],
        _CONTRACT["n_residual_quads"],
        _CONTRACT["n_tail_doubles"],
        _CONTRACT["n_spans"],
    ) == (612, 66, 245, 123, 3221)
    # region sanity: the 14 regions tile [0, L), every length is a
    # multiple of the block span, and the span count matches n_spans
    regions = _CONTRACT["regions"]
    assert len(regions) == 14
    prev = nsp = 0
    gen_names = {"F", "C", "D", "D2", "s_thio", "s_d45", "x_muS"}
    for rg in regions:
        assert rg["start"] == prev and set(rg["gens"]) <= gen_names
        span = 1 << len(rg["gens"])
        assert (rg["end"] - rg["start"]) % span == 0
        nsp += (rg["end"] - rg["start"]) // span
        prev = rg["end"]
    assert prev == L and nsp == _CONTRACT["n_spans"]
    # pair alignment: F-partners sit at (2j, 2j+1) inside the F-keyed
    # domain [0, 40748) and mostly carry byte-identical words
    f_end = max(rg["end"] for rg in regions if rg["gens"][:1] == ["F"])
    assert f_end == 40748
    assert sum(1 for r in range(0, f_end, 2) if DATA[r] == DATA[r + 1]) == 18463
    assert min(w.bit_count() for w in DATA) == 5
    print(
        "PASS  structure invariants (flip-weight {820,23304,16928}, "
        "Σny2=23438, zero-keep={40608}, 14 regions tile, "
        "pair words 18,463/20,374)"
    )

    # oracle_anchors: pinned (word, x_out, phase) triples —
    # both comparator branches, both boundary cells, both parity sectors,
    # the zero-keep row, and the top x bit.
    anchors = [
        (0, 0, 0x0, 0x10000, 0x1824000000, False),
        (0, 8249, 0xFFFFFFFFFF, 0x10000, 0xE7DBFFFFFF, False),
        (0, 8250, 0xFFFFFFFFFF, 0x08F2D, 0xFFFFFFFFFF, True),
        (500, 22428, 0x3FFFFFFF, 0x101F4, 0x3FFFFFEB, True),
        (500, 22429, 0x3FFFFFFF, 0x08FE2, 0x3FFFFFFF, True),
        (41051, 0, 0x3, 0x1A05B, 0x3, False),
        (41051, 65535, 0x3, 0x04BCF, 0x808003, True),
        (40608, 0, 0x0, 0x09EA0, 0x0, False),  # zero-keep: f=0 at σ=0
        (40608, 0, 0x1, 0x09EA0, 0x1, False),
        (0, 0, 0x1, 0x10000, 0x1824000001, False),  # odd-parity anchor
        (500, 22429, 0x8000000000, 0x08FE2, 0x8000000000, False),
    ]
    for i, sigma, x, word, x_out, phase in anchors:
        assert oracle(i, sigma, x) == (word, x_out, phase), f"oracle({i},{sigma},0x{x:x})"
    assert probe_x(42, 6) == 0xA841F9A18E, "SHA probe literal"
    assert probe_x(0, 5) == 0x72614377A8 and probe_x(0, 5).bit_count() & 1
    print("PASS  11 oracle anchors (both sectors) + SHA probe literals")

    # probe_coverage.
    zero_keep = []
    for i in range(L):
        for k in range(X_PROBES):
            px = _PX[i][k]
            assert px >> X_BITS == 0, f"probe_x({i},{k}) out of range"
            if k == 5:
                assert px.bit_count() & 1, f"probe_x({i},5) must be odd-parity"
        kp = KEEP[i]
        if kp == 0:
            zero_keep.append(i)
            continue
        assert _PSIG[i][2] < kp, f"row {i}: keep−1 slot on the keep side"
        assert _PSIG[i][3] >= kp, f"row {i}: keep slot on the alt side"
        w_keep, _, _ = oracle(i, _PSIG[i][2], 0)
        w_alt, _, _ = oracle(i, _PSIG[i][3], 0)
        assert w_keep >> IDX_BITS == 1
        assert w_alt >> IDX_BITS == 0
    assert zero_keep == [40608]
    print("PASS  probe coverage (both branches every row; zero-keep = {40608})")

    # gen_test_indexed_contract (counting SHAKE256 xof).
    from uncompute.evalcore import Xof

    class _CountingXof:
        def __init__(self, seed):
            self._x = Xof()
            self._x.update(seed)
            self.read_bytes = 0

        def read(self, n):
            self.read_bytes += n
            return self._x.read(n)

    xof = _CountingXof(b"test")
    structured = [
        (0, 0, 0, 0x0),  # row 0, σ slot 0, x probe 0
        (0 * 40 + 3 * 8 + 1, 0, 8250, _XMASK),  # row 0, σ_slot 3 (=keep), x 1
        (41051 * 40 + 1 * 8 + 0, 41051, 0xFFFF, 0x0),  # last row, σ_slot 1
    ]
    consumed = 0
    for shot, i, sigma, x in structured:
        tc = TASK.gen_test_indexed(shot, xof)
        assert tc is not None, "structured shots never skip"
        consumed += 16
        assert xof.read_bytes == consumed, "exactly 16 XOF bytes per call"
        assert tc.inputs == [i, sigma, x]
        _, x_out, phase = oracle(i, sigma, x)
        assert tc.expected == [i, sigma, x_out]
        assert tc.expected_phase == phase
    skips = 0
    for s in range(2000):
        tc = TASK.gen_test_indexed(STRUCT_SHOTS + s, xof)
        if tc is None:
            skips += 1
            continue
        i, sigma, x = tc.inputs
        assert i < L and sigma <= 0xFFFF and x >> X_BITS == 0
        _, x_out, phase = oracle(i, sigma, x)
        assert tc.expected == [i, sigma, x_out]
        assert tc.expected_phase == phase
    assert xof.read_bytes == consumed + 2000 * 16, "constant consumption incl. skips"
    assert skips <= 1, "rejection probability is 24,952/2^32; >1 skip is a bug"
    print("PASS  gen_test_indexed contract (constant 16 B/call, domain, skips)")

    # scheme_arithmetic + STRICT policy pin.
    assert STRUCT_SHOTS == 1_642_080
    assert RANDOM_SHOTS == 16_777_216
    assert NUM_SHOTS == 18_419_296
    assert TASK.num_tests == NUM_SHOTS
    assert REJECT_LIMIT % L == 0
    assert (1 << 32) - REJECT_LIMIT == 24_952
    assert 24_952 < L
    from uncompute.evalcore import STRICT, resolve_failure_policy

    assert resolve_failure_policy(TASK) is STRICT, (
        "chemistry-fe2s2 is zero-tolerance by construction"
    )
    print("PASS  scheme arithmetic + STRICT policy pin")

    # Cross-check the oracle against literal Pauli application on a
    # deterministic row sample (the closed-form sign validation).
    table = json.loads(lzma.decompress(TABLE_XZ.read_bytes()))

    def brute_apply(pauli, x):
        phc = complex(1, 0)
        xo = x
        for tok in pauli.split():
            p, q = tok[0], int(tok[1:])
            bit = (xo >> q) & 1
            if p == "Z":
                phc *= (-1) ** bit
            elif p == "X":
                xo ^= 1 << q
            elif p == "Y":
                phc *= 1j if bit == 0 else -1j
                xo ^= 1 << q
        assert abs(phc.imag) < 1e-12
        return xo, int(phc.real)

    for i in (0, 1, 500, 40608, 20000, 41051):
        pauli = table["terms"][i]["pauli"]
        for x in (0, _XMASK, HF_STATE, FLIP[i], (i * 2654435761) & _XMASK):
            xo, ph = brute_apply(pauli, x)
            assert xo == x ^ FLIP[i]
            assert ph == (-1) ** (NY2[i] ^ ((x & ZY[i]).bit_count() & 1)), (i, x)
    print("PASS  closed-form sign vs literal Pauli application (sampled rows)")

    print("chemistry-fe2s2 grader self-test: ALL PASS")
