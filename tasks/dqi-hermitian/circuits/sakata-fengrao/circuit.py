#!/usr/bin/env python3
"""dqi-hermitian sakata-fengrao: the oblivious lazy pivot-slot Feng-Rao
decode of the [512, 389, >=96] Hermitian code over F64 — the parametric
stage pipeline in this file, run at the task instance (q0 = 8,
K = 48 pivot slots, rerun schedule).  It is a reconstruction of the
Feng-Rao majority-voting decoder (Feng and Rao (1993),
doi:10.1109/18.179340; Sakata et al. (1995), doi:10.1109/18.476248).

README.md in this folder gives the main ideas, the cost decomposition,
and the accuracy argument.

RUN: uv run python circuit.py — streams to $UNCOMPUTE_OPS_PATH (else
./ops.bin), then prints the one-line advisory cost summary. Build
~95 s. build() is the raw-build entry for the occupancy/liveness
profilers (same circuit, metrics asserted, no emission).
"""

import gc

from uncompute import Builder
from uncompute.gadgets.gf_tower import GF
from uncompute.gadgets.wires import declare_registers
from uncompute.testing import assert_metrics, emit_and_report, load_module, task_dir

# The recorded op list is large and acyclic: CPython's generational GC
# re-traverses it on every gen-2 collection, turning the build quadratic,
# while refcounting alone reclaims everything the Builder drops — so the
# GC stays off for the whole build.
gc.disable()

GRADER_DIR = task_dir(__file__)

REF = load_module(GRADER_DIR / "reference.py", "dqi_hermitian_ref")


# ---------------------------------------------------------------------------
# The stage machinery: the oblivious lazy pivot-slot Feng-Rao pipeline.
#
# Uncompute discipline (MBU — measurement-based uncomputation):
#   - val words release by X-measurement with the three-term fixup
#     syn-Z + CZ(E,Pn) pairs + CZ(colhot, E): correct at every cell branch
#     (at factor cells the first two terms cancel and the third reads val's
#     own live E copy; elsewhere the third term is dead). pred words release
#     with the first two terms only (nothing was stored from them).
#     0 Toffoli either way.
#   - AND-tree flags release via and_release; controlled-increment carries
#     release against their mutated control with the conditioned
#     CZ(t_prev, ctr) + Z(t_prev) identity. 0 Toffoli.
#   - Long-lived stores (E, Pn, pinv, hots, c3 bits, vote chains) unwind by
#     a structured reverse pass (the x2 in the cost model); votes clean up
#     per round, cells at voting-stage end.
#
# Claim machinery: nexthot one-hot h[0..K] advances by the 2-pass masked
# shift — BOTH passes descending (desc CCX(c3,h[s],h[s+1]) copies the hot
# up; desc CCX(c3,h[s+1],h[s]) clears the old — an ascending second pass
# re-lights slots below any hot above 0). Inverse: the two passes
# transposed, each ascending. c3 persists per cell until the stage reverse
# (releasing it in-pass would race the used/colused mutations it drives).
#
# Ownership wire-sets (pure bookkeeping — no ops emitted; the stream is
# byte-identical with or without them): every wire a stage allocates joins
# a named ownership set at birth and is released just before its free,
# through every teardown path. The anonymous pool is exactly the
# shared-gadget scratch (AND ancillas and gf_tower-internal
# words). tools/occupancy_profile.py validates the
# discipline fail-closed: claims need live-and-unowned wires, releases
# precede frees, and the build ends owning only the declared syndrome
# register. Set names match meta.json roles one-to-one; "claim" alone
# always means the pivot-slot claim of the decode algorithm, never wire
# ownership.
#
# Loop emission (op-stream repeat + DATA tables; streams are O(program), not
# O(gate count)). The recording rules this file relies on:
#   - persistent mid-churn state (c3, sigma, z, mflags) is PRE-ALLOCATED
#     below the temp band, or it fragments the id pool;
#   - free_qubits.sort(reverse=True) before any contiguous matrix allocation (allocs
#     then pop ascending); affine operands are
#     built arithmetically off a verified base id (_cbase), never through
#     WireList re-validation;
#   - word frees run in reverse-allocation order (stack discipline) or the
#     pool poisons later growing allocations;
#   - persistent slots release by manual hmr+CZ (no free; an and_release
#     would return pool ids, a CCX release would cost counted Toffoli) —
#     gadgets.wires.WirePool's pooled release inlined, because these slots
#     are indexed (claim cell, selector position), never popped LIFO.
# ---------------------------------------------------------------------------


class Oblivious:
    """The fixed voting schedule of the oblivious pipeline at the task
    instance (q0 = 8): row/column pole orders, voted orders, and the cell bands
    keyed by the vote they precede.  K = provisioned pivot slots.
    """

    def __init__(self, inst, K):
        self.I = inst
        self.K = K
        ng = [o for o in inst.NONGAPS if o < inst.w_max]
        self.ROWS = ng  # row/col labels are pole orders
        self.COLS = ng
        self.ngset = set(inst.NONGAPS)
        # voted orders: every integer in (A, w_max] is a nongap (>= 2g)
        self.VOTED = list(range(inst.A + 1, inst.w_max + 1))
        assert all(w in inst.MONO_OF_ORDER for w in self.VOTED)
        # fixed cell schedule: bands keyed by the round they precede
        self.BANDS = {}  # w -> [(u, v), ...] v asc, u asc
        lo = 0
        for w in self.VOTED + [None]:
            hi = inst.w_max if w is None else w
            cells = [(u, v) for v in self.COLS for u in self.ROWS if lo <= u + v < hi]
            cells.sort(key=lambda uv: (uv[1], uv[0]))
            self.BANDS[w] = cells
            lo = hi
        # Tripwire only: VOTED ends at w_max, so this band is the empty
        # range [w_max, w_max) by construction — it fires only if a future
        # edit truncates VOTED. Forward cell coverage is asserted for real
        # in VotingStage.run() (cell_idx == n_cells).
        assert not self.BANDS[None]  # nothing beyond the last vote
        del self.BANDS[None]
        self.n_cells = sum(len(c) for c in self.BANDS.values())


class VotingStage:
    """The elimination + voting front-end: consumes the syndrome register,
    produces sigma words (voted syndromes); reverse() erases all internal
    state, leaving syn + sigma.
    """

    def __init__(self, b: Builder, inst, K, syn):
        """``syn``: {pole order -> Bw-qubit word} for the input syndromes
        (slices of the declared register; this stage does not allocate them).
        """
        self.b = b
        self.I = inst
        self.K = K
        self.gf = GF(inst.q0)
        self.obl = Oblivious(inst, K)
        self.Bw = inst.bits
        self.CBITS = {8: 8}[inst.q0]  # vote counter width
        ng = self.obl.ROWS
        self.ridx = {u: i for i, u in enumerate(ng)}
        self.nng = len(ng)
        # Counter-width invariant: every counter and the voters counter top
        # out at the worst round's candidate-pair count (gt2 is a shifted
        # borrow compare — no headroom bit beyond the count itself).
        assert max(len(self._pairs(w)) for w in self.obl.VOTED) < 2**self.CBITS
        # ---- persistent registers ------------------------------------------
        # All long-lived state lives in FLAT contiguous WireLists, allocated
        # here while the free-list is pristine (fresh counter bumps) — the
        # ids are then affine in the slot index s, which is what lets every
        # per-slot loop below be recorded ONCE as an op-stream repeat() instead
        # of K flat copies (streams are O(program), not O(gate count)).
        Bw = self.Bw
        self.syn = syn  # order -> word (input syndromes)
        self.sigma = {}  # voted orders (filled in run)
        # ctr/vcnt are NOT allocated here: alloc_counters() places them at
        # the TOP of the persistent bank (after every other persistent
        # register) so free_counters/revive_counters can drop them across
        # the kernel..solve span with the hole merging into the temp band.
        self.ctrf = None
        self.vcnt = None
        self.Ef = {u: b.qubits(K * Bw) for u in ng}
        self.Pnf = b.qubits(K * self.nng * Bw)
        self.pinvf = b.qubits(K * Bw)
        self.rowhotf = b.qubits(K * self.nng)
        self.colhotf = b.qubits(K * self.nng)
        self.used = {u: b.qubit() for u in ng}
        self.colused = {v: b.qubit() for v in ng}
        self.h = b.qubits(K + 1)
        b.x(self.h[0])  # next slot = 0
        # c3 claim bits and sigma words PERSIST across the stage but are
        # written mid-churn — pre-allocating them here keeps the id space
        # below the temp band solid (scattered persistent ids would
        # fragment the pool and break the sort+contiguous-matrix scheme
        # of the kernel/solve stages).
        self.c3f = b.qubits(self.obl.n_cells)
        self.sigmaf = b.qubits(len(self.obl.VOTED) * Bw)
        self._sig_of = {w: i for i, w in enumerate(self.obl.VOTED)}
        self.c3s = {}  # (u, v) -> persistent claim bit
        self.cell_idx = 0
        # bit j of value x at index x*Bw+j — drives the gated X patterns of
        # the vote equality tests and the sigma winner writes
        self.xbits = b.declare_table([(x >> j) & 1 for x in range(self.gf.q) for j in range(Bw)])
        # contiguity certificates for the arithmetic affine accessors
        self._Ebase = {u: _cbase(self.Ef[u]) for u in ng}
        self._Pnbase = _cbase(self.Pnf)
        self._pinvbase = _cbase(self.pinvf)
        self._rhbase = _cbase(self.rowhotf)
        self._chbase = _cbase(self.colhotf)
        _cbase(self.h)
        # Ownership wire-sets (see the ownership note in the module
        # docstring): the whole
        # persistent bank, full id lists; reverse() releases before free.
        b.claim("E", [q for u in ng for q in self.Ef[u]])  # sub-pivot stores
        b.claim("Pn", self.Pnf)  # normalized pivot rows
        b.claim("pinv", self.pinvf)  # pivot inverses
        b.claim("rowhot", self.rowhotf)  # slot->row one-hots
        b.claim("colhot", self.colhotf)  # slot->col one-hots
        b.claim("used", [self.used[u] for u in ng])  # row-used flags
        b.claim("colused", [self.colused[v] for v in ng])  # col-used flags
        b.claim("nexthot", self.h)  # next-slot one-hot
        b.claim("c3", self.c3f)  # persistent claim bits
        b.claim("sigma", self.sigmaf)  # voted syndrome words

    # ---- vote counters: top-of-bank allocation, freed across the peak ----
    def alloc_counters(self):
        """Allocate ctr/vcnt LAST in the persistent bank (the top ids —
        build_pipeline calls this after z/mflags/mchain). Counters are |0>
        from the end of each vote round (count-up/scan/count-down is an
        exact inverse: pred skips the just-written sigma[w]), so
        free_counters/revive_counters drop them across the kernel..solve
        span and the hole merges with the temp band.
        """
        b = self.b
        self.ctrf = b.qubits(self.gf.q * self.CBITS)
        self.vcnt = b.qubits(self.CBITS)
        self._ctrbase = _cbase(self.ctrf)
        b.claim("ctr", self.ctrf)  # 64 per-value vote counters
        b.claim("vcnt", self.vcnt)  # voters counter

    def free_counters(self):
        """Free the provably-|0> counters (release before free,
        reverse-allocation order); revive_counters restores the exact ids
        before the un-rounds.
        """
        b = self.b
        b.release("vcnt", self.vcnt)
        b.recycle_all(sorted(self.vcnt, reverse=True))  # ty: ignore[invalid-argument-type]  # set by alloc_counters
        b.release("ctr", self.ctrf)
        b.recycle_all(sorted(self.ctrf, reverse=True))  # ty: ignore[invalid-argument-type]  # set by alloc_counters

    def revive_counters(self):
        """Exact-id revival: the affine _ctrbase is baked into ctr_w."""
        self.b.reclaim(list(self.ctrf) + list(self.vcnt))  # ty: ignore[invalid-argument-type]  # set by alloc_counters
        self.b.claim("ctr", self.ctrf)
        self.b.claim("vcnt", self.vcnt)

    # ---- affine accessors (s may be an int or a repeat() index) -----------
    # The flat arrays are verified CONTIGUOUS once (below), so operands are
    # constructed arithmetically: id = base + index. WireList's affine
    # indexing would re-validate the whole iteration range on every access,
    # which puts the O(ns) per-op cost right back — the entire point of
    # recording loops is O(1) per recorded op.

    def E_w(self, u, s):
        b0, B = self._Ebase[u], self.Bw
        return [s * B + (b0 + j) for j in range(B)]

    def Pn_w(self, s, v):
        b0 = self._Pnbase + self.ridx[v] * self.Bw
        st = self.nng * self.Bw
        return [s * st + (b0 + j) for j in range(self.Bw)]

    def pinv_w(self, s):
        b0, B = self._pinvbase, self.Bw
        return [s * B + (b0 + j) for j in range(B)]

    def rowhot_q(self, s, u):
        return s * self.nng + (self._rhbase + self.ridx[u])

    def colhot_q(self, s, v):
        return s * self.nng + (self._chbase + self.ridx[v])

    def ctr_w(self, x):
        b0, C = self._ctrbase, self.CBITS
        return [x * C + (b0 + c) for c in range(C)]

    # ---- syndrome access -------------------------------------------------
    def syn_word(self, o):
        return self.syn[o] if o <= self.I.A else self.sigma[o]

    def raw_cx(self, dst, u, v, skip=None):
        for o in self.I.PROD[(u, v)]:
            if o != skip:
                self.gf.w_cx(self.b, self.syn_word(o), dst)

    def n_slots(self):
        return min(self.K, self.cell_idx)

    # ---- val words ---------------------------------------------------------
    def val_compute(self, u, v):
        b = self.b
        val = b.qubits(self.Bw)
        b.claim("val", val)  # cell value word
        self.raw_cx(val, u, v)
        ns = self.n_slots()
        if ns:
            with b.repeat(ns) as s:
                self.gf.mul_xor(b, self.E_w(u, s), self.Pn_w(s, v), val)
        return val

    def release_val_word(self, val, u, v):
        b, gf = self.b, self.gf
        b.release("val", val)  # ownership ends before free
        ns = self.n_slots()
        for j in range(self.Bw - 1, -1, -1):
            m = b.hmr(val[j])
            # raw part: Z on source syndrome bits
            for o in self.I.PROD[(u, v)]:
                b.z(self.syn_word(o)[j], cond=m)
            if ns:
                with b.repeat(ns) as s:
                    gf.cz_product(b, self.E_w(u, s), self.Pn_w(s, v), j, m)
                    b.cz(self.colhot_q(s, v), self.E_w(u, s)[j], cond=m)
            b.recycle(val[j])

    # ---- one cell, forward or reverse ---------------------------------------
    def cell(self, u, v, fwd=True):
        if fwd:
            self.cell_idx += 1
        ns = self.n_slots()
        val = self.val_compute(u, v)

        if fwd:
            # branch masks are mutually exclusive; claim runs LAST because it
            # mutates colhot/used/colused that the other masks read.
            self._branch_factor(u, v, val, ns)
            self._branch_pext(u, v, val, ns)
            self._branch_claim(u, v, val, ns, fwd=True)
        else:
            # post-cell state: at factor cells val_compute yields 0 (the sum
            # now includes the stored E copy); adding the colhot-masked E
            # extract restores val_fwd at every cell type.
            if ns:
                with self.b.repeat(ns) as s:
                    self.gf.mw_xor(self.b, self.colhot_q(s, v), self.E_w(u, s), val)
            self._branch_claim(u, v, val, ns, fwd=False)
            self._branch_pext(u, v, val, ns)  # involution given same env
            self._branch_factor(u, v, val, ns)  # un-store (mw self-inverse)

        self.release_val_word(val, u, v)
        if not fwd:
            self.cell_idx -= 1

    def _branch_factor(self, u, v, val, ns):
        # E[u][s] ^= colhot[s][v] * val  (order-free vs the other branches:
        # colhot/rowhot/claim masks are mutually exclusive at a cell)
        if ns:
            with self.b.repeat(ns) as s:
                self.gf.mw_xor(self.b, self.colhot_q(s, v), val, self.E_w(u, s))

    def _branch_pext(self, u, v, val, ns):
        b, gf = self.b, self.gf
        # gather pinv_sel under rowhot, t = val * pinv_sel, store masked by
        # rowhot & ~colused, release t vs (val, pinv_sel), ungather.
        psel = b.qubits(self.Bw)
        b.claim("pext", psel)  # gathered pivot inverse
        if ns:
            with b.repeat(ns) as s:
                gf.mw_xor(b, self.rowhot_q(s, u), self.pinv_w(s), psel)
        t = b.qubits(self.Bw)
        b.claim("pext", t)  # val * pinv product
        gf.mul_xor(b, val, psel, t)
        b.x(self.colused[v])
        if ns:
            with b.repeat(ns) as s:
                m2 = b.and_compute(self.rowhot_q(s, u), self.colused[v])
                gf.mw_xor(b, m2, t, self.Pn_w(s, v))
                b.and_release(m2, self.rowhot_q(s, u), self.colused[v])
        b.x(self.colused[v])
        b.release("pext", t)  # mbu_release_product frees t
        gf.mbu_release_product(b, t, val, psel)
        if ns:
            with b.repeat(ns) as s:
                r = ns - 1 - s
                gf.mw_xor(b, self.rowhot_q(r, u), self.pinv_w(r), psel)
        b.release_recycle("pext", psel)

    def _branch_claim(self, u, v, val, ns, fwd):
        b, gf = self.b, self.gf
        if fwd:
            nzf, tree = gf.nonzero_flag(b, val)
            av = b.and_compute(nzf, self.used[u], pol=(True, False))
            b.x(self.colused[v])
            c3 = self.c3f[self.cell_idx - 1]  # pre-allocated claim slot
            b.ccx(av, self.colused[v], c3)
            b.x(self.colused[v])
            self.c3s[(u, v)] = c3  # persists to the reverse pass
            b.and_release(av, nzf, self.used[u], pol=(True, False))
            gf.release_nonzero(b, val, nzf, tree)
            self._claim_stores(u, v, val, ns, c3)
            b.cx(c3, self.used[u])
            b.cx(c3, self.colused[v])
            _hot_shift(b, c3, self.h, True)
        else:
            c3 = self.c3s.pop((u, v))
            _hot_shift(b, c3, self.h, False)
            b.cx(c3, self.colused[v])
            b.cx(c3, self.used[u])
            self._claim_stores(u, v, val, ns, c3)
            # release c3 against recomputed controls (state now pre-claim);
            # manual hmr+CZ — the slot belongs to the pre-allocated c3f array
            # and must NOT go back to the free-list mid-run
            nzf, tree = gf.nonzero_flag(b, val)
            av = b.and_compute(nzf, self.used[u], pol=(True, False))
            b.x(self.colused[v])
            m = b.hmr(c3)
            b.cz(av, self.colused[v], cond=m)
            b.x(self.colused[v])
            b.and_release(av, nzf, self.used[u], pol=(True, False))
            gf.release_nonzero(b, val, nzf, tree)

    def _claim_stores(self, u, v, val, ns, c3):
        b, gf = self.b, self.gf
        ival = b.qubits(self.Bw)
        b.claim("ival", ival)  # inverse of the claimed pivot
        gf.inv_xor(b, val, ival)
        if ns:
            with b.repeat(ns) as s:
                ch = b.and_compute(c3, self.h[s])
                gf.mw_xor(b, ch, ival, self.pinv_w(s))
                b.cx(ch, self.Pn_w(s, v)[0])  # Pn = 1 at the pivot col
                b.cx(ch, self.rowhot_q(s, u))
                b.cx(ch, self.colhot_q(s, v))
                b.and_release(ch, c3, self.h[s])
        gf.inv_xor(b, val, ival)  # ival back to |0>
        b.release_recycle("ival", ival)

    # ---- vote machinery ------------------------------------------------
    # Plurality realized as STRICT MAJORITY (valid on-promise: Feng-Rao
    # nu(w) > 2*ell):
    # 64 counters + a voters counter, per value x a gt2 threshold flag
    # (2*votes[x] > voters) CX'd into sigma[w]. No chained argmax state —
    # every threshold flag releases against the still-live counters, and the
    # counters themselves clean up per ROUND (count up / scan / count down),
    # a measured trade: ~+6% Toffoli for ~-25% qubits vs holding argmax
    # chain state.

    def _pairs(self, w):
        I = self.I
        return [
            (u, w - u) for u in self.obl.ROWS if (w - u) in self.obl.ngset and (w - u) < I.w_max
        ]

    def pred_compute(self, u, v, w):
        """Vote candidate word cand(u,v) = lower(u,v) XOR slot_sum, with
        slot_sum = sum_s E[u][s]*Pn[s][v] (raw part minus the unknown
        syn[w]). 'pred' always names this FULL candidate.
        """
        b = self.b
        word = b.qubits(self.Bw)
        b.claim("pred", word)  # vote candidate word
        self.raw_cx(word, u, v, skip=w)
        ns = self.n_slots()
        if ns:
            with b.repeat(ns) as s:
                self.gf.mul_xor(b, self.E_w(u, s), self.Pn_w(s, v), word)
        return word

    def pred_release(self, word, u, v, w):
        """MBU release: Z on the lower syn taps + CZ(E, Pn) bilinears (no
        colhot term — nothing was stored from this word). 0 Toffoli.
        """
        b, gf = self.b, self.gf
        b.release("pred", word)  # ownership ends before free
        ns = self.n_slots()
        for j in range(self.Bw - 1, -1, -1):
            m = b.hmr(word[j])
            for o in self.I.PROD[(u, v)]:
                if o != w:
                    b.z(self.syn_word(o)[j], cond=m)
            if ns:
                with b.repeat(ns) as s:
                    gf.cz_product(b, self.E_w(u, s), self.Pn_w(s, v), j, m)
            b.recycle(word[j])

    def _count_pass(self, w, pairs, direction):
        """Counter sweep over the round's pairs: +1 = count up, -1 = down.
        The 64-value equality/count loop is ONE recorded repeat: the
        XOR-with-x patterns are data-gated X ops on the xbits table.
        """
        b, gf = self.b, self.gf
        for u, v in pairs:
            word = self.pred_compute(u, v, w)
            # candidacy: NOT used[u] AND NOT colused[v]
            b.x(self.used[u])
            b.x(self.colused[v])
            av = b.and_compute(self.used[u], self.colused[v])
            b.x(self.colused[v])
            b.x(self.used[u])
            step = gf.ctr_inc if direction > 0 else gf.ctr_dec
            step(b, av, self.vcnt)
            with b.repeat(gf.q) as x:
                for j in range(self.Bw):
                    with b.gated(self.xbits, x * self.Bw + j):
                        b.x(word[j])
                zf, tree = gf.zero_flag(b, word)
                g = b.and_compute(zf, av)
                step(b, g, self.ctr_w(x))
                b.and_release(g, zf, av)
                gf.release_zero(b, word, zf, tree)
                for j in range(self.Bw):
                    with b.gated(self.xbits, x * self.Bw + j):
                        b.x(word[j])
            b.x(self.used[u])
            b.x(self.colused[v])
            b.and_release(av, self.used[u], self.colused[v])
            b.x(self.colused[v])
            b.x(self.used[u])
            self.pred_release(word, u, v, w)

    def vote_round(self, w, fwd=True):
        """Count up, threshold-scan into sigma[w] (XOR — the same scan
        un-writes it on the reverse pass), count down. Involution up to the
        sigma alloc/free bookkeeping.
        """
        b, gf = self.b, self.gf
        pairs = self._pairs(w)
        if fwd:
            base = self._sig_of[w] * self.Bw
            self.sigma[w] = self.sigmaf[base : base + self.Bw]
        self._count_pass(w, pairs, +1)
        sig = self.sigma[w]
        with b.repeat(gf.q) as x:
            flag, chain = gf.gt2_flag(b, self.ctr_w(x), self.vcnt)
            for j in range(self.Bw):
                with b.gated(self.xbits, x * self.Bw + j):
                    b.cx(flag, sig[j])
            gf.gt2_release(b, self.ctr_w(x), self.vcnt, flag, chain)
        self._count_pass(w, pairs, -1)
        if not fwd:
            self.sigma.pop(w)  # slot zeroed; freed at teardown

    # ---- stage driver ----------------------------------------------------
    def run(self):
        for w in self.obl.VOTED:
            self.b.set_phase("votes/cells")
            for u, v in self.obl.BANDS[w]:
                self.cell(u, v, fwd=True)
            self.b.set_phase("votes/round")
            self.vote_round(w, fwd=True)
        # forward schedule completeness: every pre-allocated c3 claim bit
        # got its cell (the reverse-side twin is cell_idx == 0 in reverse())
        assert self.cell_idx == self.obl.n_cells

    def reverse(self):
        """Erase all internal state; leaves syn restored (sigma words are
        un-written and freed round by round, last round first).
        """
        b = self.b
        for w in reversed(self.obl.VOTED):
            b.set_phase("un-votes/round")
            self.vote_round(w, fwd=False)
            b.set_phase("un-votes/cells")
            for u, v in reversed(self.obl.BANDS[w]):
                self.cell(u, v, fwd=False)
        assert self.cell_idx == 0 and not self.c3s and not self.sigma
        b.set_phase("teardown")
        b.x(self.h[0])
        ng = self.obl.ROWS
        # ownership ends before free, set by set (the occupancy discipline);
        # counters first — they top the persistent bank (alloc_counters)
        b.release_recycle("vcnt", self.vcnt)
        b.release_recycle("ctr", self.ctrf)
        b.release("sigma", self.sigmaf)
        b.recycle_all(sorted(self.sigmaf, reverse=True))
        b.release("c3", self.c3f)
        b.recycle_all(sorted(self.c3f, reverse=True))
        b.release_recycle("nexthot", self.h)
        b.release("colused", [self.colused[v] for v in ng])
        for v in reversed(ng):
            b.recycle(self.colused[v])
        b.release("used", [self.used[u] for u in ng])
        for u in reversed(ng):
            b.recycle(self.used[u])
        b.release_recycle("colhot", self.colhotf)
        b.release_recycle("rowhot", self.rowhotf)
        b.release_recycle("pinv", self.pinvf)
        b.release_recycle("Pn", self.Pnf)
        b.release("E", [q for u in ng for q in self.Ef[u]])
        for u in reversed(ng):
            b.recycle_all(list(reversed(self.Ef[u])))


def _hot_shift(b, ctl, hots, forward):
    """Masked one-hot advance (both passes DESCENDING; see the claim-
    machinery note in the module docstring).
    """
    n = len(hots) - 1
    if forward:
        with b.repeat(n) as i:
            b.ccx(ctl, hots[n - 1 - i], hots[n - i])
        with b.repeat(n) as i:
            b.ccx(ctl, hots[n - i], hots[n - 1 - i])
    else:
        with b.repeat(n) as i:
            b.ccx(ctl, hots[i + 1], hots[i])
        with b.repeat(n) as i:
            b.ccx(ctl, hots[i], hots[i + 1])


def _cbase(wl):
    """Contiguity certificate: returns the base id of a flat WireList."""
    base = wl[0]
    assert all(wl[k] == base + k for k in range(len(wl))), (
        "flat array not contiguous (allocated after free-list churn?)"
    )
    return base


class GEBlock:
    """Column-major masked full-Jordan GE (shared by kernel and solve).

    ``rows``: ONE flat contiguous WireList of R*ncols*Bw qubits (the caller
    allocates it right after the pool sort), addressed affinely — every per-row
    and per-column loop below records once as an op-stream repeat. Per data
    column c: one-hot select the first unused nonzero row, gather the pivot
    row, invert, normalize into the COLUMN-INDEXED piv copy (consumers
    address pivot rows classically — CX-free), un-gather, then update every
    physical row and every prior piv row with IN-PLACE multipliers (no f
    copies). Column c stays dirty at non-pivot rows; consumers mask
    processed pivot columns out (kernel: membership free-flags; solve: only
    the never-pivoted RHS column is read).

    ``full_span``: piv is ONE flat npiv x ncols matrix (kernel — membership
    reads entries left of the pivot, and the flat layout is affine across
    columns for its data-gated tables) vs per-column c..ncols-1 tails
    (solve — only the RHS tail is ever read; tail storage is the qubit
    saving that matters at S=75).
    """

    def __init__(self, b, gf, rows, R, ncols, npiv, full_span):
        self.b, self.gf = b, gf
        B = gf.bits
        self.rows = rows
        self.R = R
        self.ncols = ncols
        self.npiv = npiv
        self.full_span = full_span
        assert len(rows) == R * ncols * B
        self._rbase = _cbase(rows)
        self.usedf = b.qubits(R)
        self._ubase = _cbase(self.usedf)
        self.self_sel = b.qubits(npiv * R)
        self._selbase = _cbase(self.self_sel)
        self.pivotedf = b.qubits(npiv)
        self._pivotedbase = _cbase(self.pivotedf)
        self.pinvf = b.qubits(npiv * B)
        self._pinvbase = _cbase(self.pinvf)
        b.claim("ge-used", self.usedf)  # row-used flags
        b.claim("ge-sel", self.self_sel)  # pivot-row selectors
        b.claim("ge-pivoted", self.pivotedf)  # column-pivoted flags
        b.claim("ge-pinv", self.pinvf)  # pivot inverses
        if full_span:
            self.pivf = b.qubits(npiv * ncols * B)
            self._pivbase = _cbase(self.pivf)
            b.claim("ge-piv", self.pivf)  # normalized pivot rows
        else:
            # tail layout: all per-column tails in ONE flat block allocated
            # up front (a mid-column allocation would pop gadget-temp churn
            # from the pool and come out non-contiguous; total qubits are
            # unchanged — every tail is live by the end of the GE anyway)
            self._tail_off = []
            off = 0
            for c in range(npiv):
                self._tail_off.append(off)
                off += (ncols - c) * B
            self.pivf = b.qubits(off)
            self._pivbase = _cbase(self.pivf)
            b.claim("ge-piv", self.pivf)  # normalized pivot tails

    # ---- affine accessors (indices may be ints or repeat() exprs) ---------
    def row_w(self, r, k):
        B = self.gf.bits
        st = self.ncols * B
        return [r * st + k * B + (self._rbase + j) for j in range(B)]

    def used_q(self, r):
        return r + self._ubase

    def sel_q(self, c, r):
        return r + (self._selbase + c * self.R)

    def pinv_w(self, c):
        B = self.gf.bits
        return [c * B + (self._pinvbase + j) for j in range(B)]

    def piv_w(self, c, k):
        """Return the normalized pivot-row word for column c at position k. For the
        tail layout c must be a plain int and k >= c.
        """
        B = self.gf.bits
        if self.full_span:
            st = self.ncols * B
            return [c * st + k * B + (self._pivbase + j) for j in range(B)]
        base = self._pivbase + self._tail_off[c] - c * B
        return [k * B + (base + j) for j in range(B)]

    def _span(self, c):
        return range(0 if self.full_span else c, self.ncols)

    def _selection(self, c, forward):
        b, gf = self.b, self.gf
        R = self.R
        found = self.pivotedf[c]
        if forward:
            with b.repeat(R) as r:
                nzf, tree = gf.nonzero_flag(b, self.row_w(r, c))
                av = b.and_compute(nzf, self.used_q(r), pol=(True, False))
                b.x(found)
                b.ccx(av, found, self.sel_q(c, r))
                b.x(found)
                b.cx(self.sel_q(c, r), found)
                b.and_release(av, nzf, self.used_q(r), pol=(True, False))
                gf.release_nonzero(b, self.row_w(r, c), nzf, tree)
            with b.repeat(R) as r:
                b.cx(self.sel_q(c, r), self.used_q(r))
        else:
            with b.repeat(R) as r:
                b.cx(self.sel_q(c, r), self.used_q(r))
            with b.repeat(R) as i:
                r = R - 1 - i
                b.cx(self.sel_q(c, r), found)  # found -> pre-row-r state
                nzf, tree = gf.nonzero_flag(b, self.row_w(r, c))
                av = b.and_compute(nzf, self.used_q(r), pol=(True, False))
                # sel released by MEASUREMENT (0 Toffoli — a CCX here would
                # execute R*npiv counted Toffoli); manual hmr+CZ because the
                # slot belongs to the pre-allocated sel matrix and must not
                # go back to the free-list mid-run.
                b.x(found)
                m = b.hmr(self.sel_q(c, r))
                b.cz(av, found, cond=m)
                b.x(found)
                b.and_release(av, nzf, self.used_q(r), pol=(True, False))
                gf.release_nonzero(b, self.row_w(r, c), nzf, tree)

    def _column(self, c, forward):
        b, gf = self.b, self.gf
        B = gf.bits
        NC, R = self.ncols, self.R
        span = list(self._span(c))
        ns = len(span)
        c0 = span[0]
        nk = NC - c - 1

        def pv_w(pvb, k):
            return [k * B + (pvb + j) for j in range(B)]

        def gather(pvb):
            with b.repeat(ns) as k, b.repeat(R) as r:
                gf.mw_xor(b, self.sel_q(c, r), self.row_w(r, c0 + k), pv_w(pvb, k))

        def updates():
            if nk <= 0:
                return
            if c > 0:
                if self.full_span:
                    with b.repeat(c) as c2, b.repeat(nk) as k:
                        gf.mul_xor(
                            b,
                            self.piv_w(c2, c),
                            self.piv_w(c, c + 1 + k),
                            self.piv_w(c2, c + 1 + k),
                        )
                else:
                    for c2 in range(c):
                        with b.repeat(nk) as k:
                            gf.mul_xor(
                                b,
                                self.piv_w(c2, c),
                                self.piv_w(c, c + 1 + k),
                                self.piv_w(c2, c + 1 + k),
                            )
            with b.repeat(R) as r, b.repeat(nk) as k:
                gf.mul_xor(b, self.row_w(r, c), self.piv_w(c, c + 1 + k), self.row_w(r, c + 1 + k))

        if forward:
            self._selection(c, True)
            pvl = b.qubits(ns * B)
            b.claim("ge-gather", pvl)  # per-column pivot gather
            pvb = _cbase(pvl)
            gather(pvb)
            gf.inv_xor(b, pv_w(pvb, c - c0), self.pinv_w(c))
            with b.repeat(ns) as k:
                gf.mul_xor(b, self.pinv_w(c), pv_w(pvb, k), self.piv_w(c, c0 + k))
            gather(pvb)  # un-gather (XOR involution)
            b.release("ge-gather", pvl)
            b.recycle_all(sorted(pvl, reverse=True))
            updates()
        else:
            updates()
            pvl = b.qubits(ns * B)
            b.claim("ge-gather", pvl)  # per-column pivot gather
            pvb = _cbase(pvl)
            gather(pvb)
            with b.repeat(ns) as k:
                gf.mul_xor(b, self.pinv_w(c), pv_w(pvb, k), self.piv_w(c, c0 + k))  # piv words -> 0
            gf.inv_xor(b, pv_w(pvb, c - c0), self.pinv_w(c))  # -> 0
            gather(pvb)
            b.release("ge-gather", pvl)
            b.recycle_all(sorted(pvl, reverse=True))
            self._selection(c, False)

    def run(self):
        self.b.free_qubits.sort(reverse=True)  # pv gathers pop a clean band
        for c in range(self.npiv):
            self._column(c, True)

    def reverse(self):
        self.b.free_qubits.sort(reverse=True)  # pool churned since run() (payout etc.)
        for c in range(self.npiv - 1, -1, -1):
            self._column(c, False)
        # ownership ends before free, set by set (the occupancy discipline)
        self.b.release("ge-piv", self.pivf)
        self.b.recycle_all(sorted(self.pivf, reverse=True))
        self.b.release("ge-pinv", self.pinvf)
        self.b.recycle_all(sorted(self.pinvf, reverse=True))
        self.b.release("ge-pivoted", self.pivotedf)
        self.b.recycle_all(sorted(self.pivotedf, reverse=True))
        self.b.release("ge-sel", self.self_sel)
        self.b.recycle_all(sorted(self.self_sel, reverse=True))
        self.b.release("ge-used", self.usedf)
        self.b.recycle_all(sorted(self.usedf, reverse=True))


class KernelStage:
    """RREF of the loc x test evaluation matrix (entries CX-read off
    syn/sigma). Rows = test-space orders, columns = locator-space orders.
    The raw fill is data-gated on the PROD tap table: per syndrome word one
    repeat over (row, col) with the membership bit from the table.
    """

    def __init__(self, b, inst, vs):
        self.b, self.I, self.vs = b, inst, vs
        self.gf = vs.gf
        self.loc = [inst.ORDER[mn] for mn in inst.L_LOC]
        self.test = [inst.ORDER[mn] for mn in inst.L_TEST]
        self.R = len(self.test)
        self.NC = len(self.loc)
        # taps: every order contributing to some cell, with its (ri, ci) mask
        taps = sorted({o for vo in self.test for uo in self.loc for o in inst.PROD[(uo, vo)]})
        self.taps = taps
        bits = []
        for o in taps:
            for vo in self.test:
                for uo in self.loc:
                    bits.append(1 if o in inst.PROD[(uo, vo)] else 0)
        self.ftbl = b.declare_table(bits)
        self.A = None
        self.ge = None

    def _fill(self):
        b, B = self.b, self.I.bits
        RN = self.R * self.NC
        for oi, o in enumerate(self.taps):
            sw = self.vs.syn_word(o)
            with b.repeat(self.R) as ri, b.repeat(self.NC) as ci:
                idx = oi * RN + ri * self.NC + ci
                for j in range(B):
                    with b.gated(self.ftbl, idx):
                        b.cx(sw[j], self.ge.row_w(ri, ci)[j])  # ty: ignore[unresolved-attribute]  # set by run

    def run(self):
        # allocate at run time from a COMPACTED pool: the rerun schedule
        # tears the stage down and rebuilds after solve, and the flat matrix
        # must be contiguous for affine addressing.
        b = self.b
        b.free_qubits.sort(reverse=True)
        self.A = b.qubits(self.R * self.NC * self.I.bits)
        b.claim("kernelA", self.A)  # test-space matrix
        self.ge = GEBlock(b, self.gf, self.A, self.R, self.NC, self.NC, full_span=True)
        self._fill()
        self.ge.run()

    def reverse(self):
        self.ge.reverse()  # ty: ignore[unresolved-attribute]  # set by run
        self._fill()  # CX involution: A -> 0
        self.b.release("kernelA", self.A)  # ownership ends before free
        self.b.recycle_all(sorted(self.A, reverse=True))  # ty: ignore[invalid-argument-type]  # set by run
        self.A = None
        self.ge = None


class MembershipStage:
    """z_i = all kernel basis vectors vanish at point i. Basis vector of a
    free column fc is read CX-free off the piv rows (entry piv[c2][fc]) plus
    the classical +1 at fc; processed pivot columns are masked out via the
    pivoted flags (their acc words are multiplier-column dirt).

    Recorded as ONE repeat over the m points: the per-point LOC_EVAL
    constant patterns are data-gated tables indexed affinely in (point,
    column); the per-fc flags and the NOR chain live in pre-allocated
    contiguous arrays (pipeline prologue, like ``z``) so the fc loops are
    repeats too. The whole stage body is an involution up to the z write,
    so reverse() replays it.

    ``z``/``mflags``/``mchain``: pre-allocated flat arrays (pipeline
    prologue — they persist across stages, so they must live below the
    temp band).
    """

    def __init__(self, b, inst, ks, z, mflags, mchain):
        self.b, self.I, self.ks = b, inst, ks
        self.gf = ks.gf
        self.z = z
        self.mflags = mflags
        self._mfb = _cbase(mflags)
        self.mchain = mchain
        self._mcb = _cbase(mchain)
        C = len(ks.loc)
        B = inst.bits
        m = inst.m
        # +1-at-fc term: bit j of LOC_EVAL[fc][i] at ((fc*m)+i)*B + j
        self.t_unit = b.declare_table(
            [
                (int(inst.LOC_EVAL[fc][i]) >> j) & 1
                for fc in range(C)
                for i in range(m)
                for j in range(B)
            ]
        )
        # const-mult pattern: bit (j -> jj) of mul(LOC_EVAL[c2][i], x^j)
        # sits at index (((c2*m)+i)*B + j)*B + jj
        gf = self.gf
        bits = []
        for c2 in range(C):
            for i in range(m):
                k = int(inst.LOC_EVAL[c2][i])
                for j in range(B):
                    pat = gf.mul(k, 1 << j)
                    bits += [(pat >> jj) & 1 for jj in range(B)]
        self.t_cmul = b.declare_table(bits)

    def _sweep(self):
        """One full membership pass: z[i] ^= all-vanish flag, for every i.
        Applying it twice is the identity (the mflag/chain passes are
        replayed involutions).
        """
        b, gf, ge = self.b, self.gf, self.ks.ge
        I = self.I
        C = ge.npiv
        B = I.bits
        m = I.m

        def mflag_pass(i):
            with b.repeat(C) as fc:
                acc = b.qubits(B)
                b.claim("acc", acc)  # basis-vector evaluation word
                for j in range(B):
                    with b.gated(self.t_unit, (fc * m + i) * B + j):
                        b.x(acc[j])
                with b.repeat(C) as c2:
                    pw = ge.piv_w(c2, fc)
                    for j in range(B):
                        for jj in range(B):
                            with b.gated(self.t_cmul, ((c2 * m + i) * B + j) * B + jj):
                                b.cx(pw[j], acc[jj])
                nzf, tree = gf.nonzero_flag(b, acc)
                pvq = fc + ge._pivotedbase
                b.x(pvq)
                b.ccx(pvq, nzf, fc + self._mfb)  # mflag ^= free & nonzero
                b.x(pvq)
                gf.release_nonzero(b, acc, nzf, tree)
                with b.repeat(C) as c2:
                    pw = ge.piv_w(c2, fc)
                    for j in range(B):
                        for jj in range(B):
                            with b.gated(self.t_cmul, ((c2 * m + i) * B + j) * B + jj):
                                b.cx(pw[j], acc[jj])
                for j in range(B):
                    with b.gated(self.t_unit, (fc * m + i) * B + j):
                        b.x(acc[j])
                b.release_recycle("acc", acc)

        def chain(up):
            # chain[k] = AND over fc<=k+1 of ~mflag[fc]
            # local: gadgets.arith.mcx (open ctrls) matches the C-1 Toffoli
            # but ladders through transient pool ANDs; this ladder writes
            # the pre-allocated mchain array (repeat-affine ids, hmr+CZ
            # release), spending C-1 persistent wires for it. Swapping in
            # mcx is a straight width win that moves the stream, so it
            # needs a fresh certification.
            mf, ch = self._mfb, self._mcb
            if up:
                b.x(self.mflags[0])
                b.x(self.mflags[1])
                b.ccx(self.mflags[0], self.mflags[1], self.mchain[0])
                b.x(self.mflags[1])
                b.x(self.mflags[0])
                if C > 2:
                    with b.repeat(C - 2) as k:
                        b.x(k + 2 + mf)
                        b.ccx(k + ch, k + 2 + mf, k + 1 + ch)
                        b.x(k + 2 + mf)
            else:
                if C > 2:
                    with b.repeat(C - 2) as kk:
                        k = C - 3 - kk
                        b.x(k + 2 + mf)
                        m2 = b.hmr(k + 1 + ch)
                        b.cz(k + ch, k + 2 + mf, cond=m2)
                        b.x(k + 2 + mf)
                b.x(self.mflags[0])
                b.x(self.mflags[1])
                m2 = b.hmr(self.mchain[0])
                b.cz(self.mflags[0], self.mflags[1], cond=m2)
                b.x(self.mflags[1])
                b.x(self.mflags[0])

        with b.repeat(m) as i:
            mflag_pass(i)
            chain(True)
            b.cx(self.mchain[C - 2], self.z[i])
            chain(False)
            mflag_pass(i)  # mflags -> 0 (replay involution)

    def run(self):
        self._sweep()

    def reverse(self):
        self._sweep()  # z toggles back; state was clean


class SolveStage:
    """Compact the z-flagged classical H columns into u_loc slots under a
    walking one-hot (CX-only gathers), then masked full-Jordan GE on the
    compacted n_checks x (u_loc+1) system. Slot value piv[s][S] is the
    decoded symbol of the point assigned to slot s.
    """

    def __init__(self, b, inst, gf, syn_words, z):
        self.b, self.I, self.gf = b, inst, gf
        self.syn_words = syn_words  # list of n_checks input words
        self.z = z
        self.S = inst.u_loc
        b.free_qubits.sort(reverse=True)
        self.M = b.qubits(inst.n_checks * (self.S + 1) * inst.bits)
        self.h2 = b.qubits(self.S + 2)
        b.claim("solveM", self.M)  # compacted solve system
        b.claim("hot2", self.h2)  # compaction one-hot
        self.ge = GEBlock(b, gf, self.M, inst.n_checks, self.S + 1, self.S, full_span=False)
        # H column bits: bit j of H[r][i] at (i*n_checks + r)*bits + j
        self.t_h = b.declare_table(
            [
                (int(inst.H[r][i]) >> j) & 1
                for i in range(inst.m)
                for r in range(inst.n_checks)
                for j in range(inst.bits)
            ]
        )

    def _rhs(self):
        for r in range(self.I.n_checks):
            self.gf.w_cx(self.b, self.syn_words[r], self.ge.row_w(r, self.S))

    def _compact(self, forward):
        """One repeat over the m points: walk the one-hot, gather the
        CLASSICAL H columns under chot via the H bit table (gated CX,
        index affine in the point), advance. Reverse = same body with the
        point index and the shift direction flipped.
        """
        b, I = self.b, self.I
        m, S, B = I.m, self.S, I.bits
        RN = I.n_checks
        with b.repeat(m) as ii:
            i = ii if forward else (m - 1 - ii)
            if not forward:
                _hot_shift(b, self.z[i], self.h2, False)
            for s in range(S):
                ch = b.and_compute(self.z[i], self.h2[s])
                with b.repeat(RN) as r:
                    for j in range(B):
                        with b.gated(self.t_h, (i * RN + r) * B + j):
                            b.cx(ch, self.ge.row_w(r, s)[j])
                b.and_release(ch, self.z[i], self.h2[s])
            if forward:
                _hot_shift(b, self.z[i], self.h2, True)

    def run(self):
        self.b.x(self.h2[0])
        self._rhs()
        self._compact(True)
        self.ge.run()

    def reverse(self):
        self.ge.reverse()
        self._compact(False)
        self._rhs()
        self.b.x(self.h2[0])
        # ownership ends before free (the occupancy discipline)
        self.b.release("hot2", self.h2)
        self.b.recycle_all(sorted(self.h2, reverse=True))
        self.b.release("solveM", self.M)
        self.b.recycle_all(sorted(self.M, reverse=True))


def payout(b, inst, z, solve):
    """(-1)^<c, e>: walk a fresh payout one-hot (hot3) along the compaction
    schedule again (hot2 stays live inside the solve block); per (point i,
    slot s) the fixup is CZ(chot, V_s bits masked by the classical c_i
    chunk).
    """
    S = solve.S
    B = inst.bits
    m = inst.m
    b.free_qubits.sort(reverse=True)  # h3 must be contiguous for the hot shifts
    h3 = b.qubits(S + 2)
    b.claim("hot3", h3)  # payout walking one-hot
    b.x(h3[0])
    # fixup-vector chunk bits: bit t of c_i at i*bits + t
    t_c = b.declare_table([(inst.C_INT >> (B * i + t)) & 1 for i in range(m) for t in range(B)])
    with b.repeat(m) as i:
        for s in range(S):
            ch = b.and_compute(z[i], h3[s])
            V = solve.ge.piv_w(s, S)
            for t in range(B):
                with b.gated(t_c, i * B + t):
                    b.cz(ch, V[t])
            b.and_release(ch, z[i], h3[s])
        _hot_shift(b, z[i], h3, True)
    with b.repeat(m) as ii:
        _hot_shift(b, z[m - 1 - ii], h3, False)
    b.x(h3[0])
    b.release("hot3", h3)  # ownership ends before free
    b.recycle_all(sorted(h3, reverse=True))


def build_pipeline(b, inst, K, syn_map, do_payout=True, schedule="rerun"):
    """Full decode pipeline on a declared syndrome register.

    schedule="lifo": votes -> kernel -> member -> solve -> payout ->
    reverses in LIFO order (kernel matrix coexists with the solve block).

    schedule="rerun" (default): reverse the kernel BEFORE the solve block
    and re-run it after — membership reverse needs the piv rows, and the
    kernel forward is deterministic in syn/sigma (untouched by solve).
    Costs one extra kernel fwd+rev; frees the kernel block during the
    solve peak (measured at q0=8: 7,950,384 of the 104,263,342 executed
    Toffoli buy the 39,435-wire kernel block off the 194,053-qubit solve
    peak). The rerun window also frees the four provably-|0> persistent
    registers across the peak: ctr/vcnt after the last vote round and
    mflags/mchain after un-kernel, each revived at its exact ids
    (reclaim + claim)
    before its reverse consumer.
    """
    b.set_phase("alloc")  # names the persistent-bank birth cohort
    b.claim("syn", [q for w in syn_map.values() for q in w])  # declared I/O
    vs = VotingStage(b, inst, K, syn_map)
    # persistent cross-stage arrays: pre-churn allocation keeps them below
    # the temp band (contiguous, and no pool fragmentation for the stage
    # matrices)
    z = b.qubits(inst.m)
    b.claim("z", z)  # membership flags
    C = len(inst.L_LOC)
    mflags = b.qubits(C)
    b.claim("mflags", mflags)  # per-column membership flags
    mchain = b.qubits(C - 1)
    b.claim("mchain", mchain)  # membership NOR chain
    vs.alloc_counters()  # ctr/vcnt: TOP of the persistent bank
    vs.run()  # phases: votes/cells, votes/round
    if schedule == "rerun":
        # ctr/vcnt are |0> between the last vote round and the first
        # un-round; they top the persistent bank, so the hole merges with
        # the temp band and the kernel/solve matrices still pop contiguous
        # ascending ids. Revived exact-id before the un-rounds. Rerun-only
        # bracket, matching the mflags/mchain one below.
        vs.free_counters()
    b.set_phase("kernel")
    ks = KernelStage(b, inst, vs)
    ks.run()
    b.set_phase("member")
    ms = MembershipStage(b, inst, ks, z, mflags, mchain)
    ms.run()
    if schedule == "rerun":
        b.set_phase("un-kernel")
        ks.reverse()
        # mflags/mchain are |0> outside each member point iteration (every
        # sweep pass is a replayed involution). Freed here — after the
        # kernel block above them is gone — they top the live bank, the
        # hole merges with the freed band, and the solve matrix still pops
        # contiguous ascending ids while the peak drops by their width;
        # reclaim restores the exact ids before un-member. Rerun-only:
        # under lifo the still-live kernel block would strand the hole.
        b.release("mchain", mchain)
        b.recycle_all(sorted(mchain, reverse=True))
        b.release("mflags", mflags)
        b.recycle_all(sorted(mflags, reverse=True))
    syn_words = [syn_map[inst.ORDER[mn]] for mn in inst.H_MONOS]
    b.set_phase("solve")
    ss = SolveStage(b, inst, vs.gf, syn_words, z)
    ss.run()
    if do_payout:
        b.set_phase("pay")
        payout(b, inst, z, ss)
    b.set_phase("un-solve")
    ss.reverse()
    if schedule == "rerun":
        # un-member needs mflags/mchain back at their construction-time
        # base ids (MembershipStage baked _cbase at __init__).
        b.reclaim(list(mflags) + list(mchain))
        b.claim("mflags", mflags)
        b.claim("mchain", mchain)
        b.set_phase("re-kernel")
        ks.run()  # identical state: syn/sigma unchanged
    b.set_phase("un-member")
    ms.reverse()
    b.set_phase("un-kernel")
    ks.reverse()
    if schedule == "rerun":
        vs.revive_counters()  # the un-rounds need ctr/vcnt back
    b.set_phase("un-votes")  # refined to un-votes/* inside reverse()
    # ownership ends before free ("syn" is never released: the declared
    # register survives the build as the only owned wires)
    b.release("mchain", mchain)
    b.recycle_all(sorted(mchain, reverse=True))
    b.release("mflags", mflags)
    b.recycle_all(sorted(mflags, reverse=True))
    b.release("z", z)
    b.recycle_all(sorted(z, reverse=True))
    vs.reverse()
    return vs


Q0 = 8
inst = REF.Hermitian(Q0)
K = 48  # slots: MC max claims = 47 = ell
# Provisioning invariants (corners C1/C5): K = ell + 1 slots hold every
# on-promise claim sequence (rank <= wt(e) <= ell; Monte Carlo max claims
# = ell), and u_loc = ell + g compaction slots hold every
# on-promise vanishing-point set (|Z| <= ell + g).
assert K == inst.ell + 1 and inst.u_loc == inst.ell + inst.g


def build():
    """Run the one raw build (~95 s): the full pipeline plus the pinned-metrics
    epilogue, without emission. tools/occupancy_profile.py and the liveness
    profiler call this and consume the Builder's wire_log directly — no post
    pass touches the allocator.
    """
    b = Builder()
    # 123 registers of 6 qubits, row order
    regs = list(declare_registers(b, ((f"s{r}", inst.bits) for r in range(inst.n_checks))).values())
    syn_map = {inst.ORDER[mn]: regs[r] for r, mn in enumerate(inst.H_MONOS)}

    build_pipeline(b, inst, K, syn_map, schedule="rerun")

    # -------------------------------------------------------------- accounting
    # The meta.json cost pin, measured on the q0 = 8 configuration (K = 48,
    # rerun schedule): 104,263,342 Toffoli x 194,053 qubits.  The evaluator
    # at 100,000 shots certifies the circuit (results.jsonl).
    assert_metrics(b, __file__)
    return b, regs


if __name__ == "__main__":
    b, _regs = build()
    # ---------------------------------------------------------------- emission
    emit_and_report(b)
