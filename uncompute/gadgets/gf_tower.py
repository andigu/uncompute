"""Field-parametric GF(q0^2) tower-word gadgets (MBU).

The module implements word gadgets for GF(q0^2), q0 in {4, 8}; the
dqi-hermitian circuits use q0 = 8. Multiplication is Karatsuba AND-pair
multiplication with measurement-based uncomputation, Gidney (2018),
arXiv:1709.06648. Inversion follows Itoh and Tsujii (1988),
doi:10.1016/0890-5401(88)90024-7.

Words are `bits`-qubit little-endian F2-polynomial coefficient vectors:
  q0=4: GF(16), poly 19 (x^4+x+1),         mult =  9 AND (2x2 Karatsuba)
  q0=8: GF(64), poly 91 (x^6+x^4+x^3+x+1), mult = 18 AND (2x3 Karatsuba)
(q0=8 is the dqi-hermitian instance field; the self-test below also
covers q0=4).

Costs (executed Toffoli, q0=8 / q0=4):
  mul_xor           out ^= a*b            18 / 9   (AND-pair MBU inside)
  mbu_release_product(out, a, b)           0       (hmr + conditioned CZ)
  inv_xor           out ^= a^-1 (0->0)    54 / 18  (Itoh-Tsujii free squarings)
  nonzero_flag                             5 / 3   (AND tree); release: 0
  mw_xor            dst ^= flag*src        6 / 4
  const-mult / Frobenius powers            0       (CX patterns)

Every temporary a gadget allocates is released before return (AND ancillas
by and_release, product words by MBU, linear words by CX-undo), so callers
see net-zero ancilla and zero phase residue.
"""

from uncompute import Builder
from uncompute.classical import GF2n, f2_rows_of
from uncompute.gadgets.linear import linear_xor


def _k2_contrib():
    # 2-term Karatsuba: r0 = m0; r1 = m01+m0+m1; r2 = m1
    return {(0,): [0, 1], (1,): [1, 2], (0, 1): [1]}


def _k3_contrib():
    # 3-term pairwise: r0=m0; r1=m01+m0+m1; r2=m02+m0+m1+m2; r3=m12+m1+m2; r4=m2
    return {
        (0,): [0, 1, 2],
        (1,): [1, 2, 3],
        (2,): [2, 3, 4],
        (0, 1): [1],
        (1, 2): [3],
        (0, 2): [2],
    }


class GF:
    def __init__(self, q0):
        assert q0 in (4, 8)
        self.q0 = q0
        self.bits = {4: 4, 8: 6}[q0]
        self.poly = {4: 19, 8: 91}[q0]
        self.q = 1 << self.bits
        self.field = GF2n(self.bits, self.poly | self.q)
        B = self.bits
        # RED[n] = x^n mod poly
        self.RED = []
        for n in range(2 * B - 1):
            r = 1 << n
            for i in range(2 * B - 2, B - 1, -1):
                if (r >> i) & 1:
                    r ^= (self.poly | self.q) << (i - B)
            self.RED.append(r)
        self.MUL_ANDS = self._build_mul_ands()
        # FROB_ROWS[e] = the rows of the Frobenius map a -> a^(2^e)
        self.FROB_ROWS = [f2_rows_of(lambda v, e=e: self.pow(v, 1 << e), B) for e in range(B)]
        # exhaustive verification of the descriptor wiring
        for a in range(self.q):
            for b in range(self.q):
                out = 0
                for am, bm, pat in self.MUL_ANDS:
                    if (bin(a & am).count("1") & 1) and (bin(b & bm).count("1") & 1):
                        out ^= pat
                assert out == self.mul(a, b), (q0, a, b)

    # ---- classical arithmetic (exp/log tables) ----
    def mul(self, a, b):
        return self.field.mul(a, b)

    def pow(self, a, e):
        return self.field.pow(a, e)

    def inv(self, a):
        return self.field.inv(a)

    # ---- Karatsuba descriptors: (amask, bmask, outpat) per AND ----
    def _half_ands(self, pa, qa, coeff_sets, contrib):
        ands = []
        for key, rs in contrib.items():
            amask = bmask = 0
            for t in key:
                amask ^= pa[t]
                bmask ^= qa[t]
            coeffs = set()
            for r in rs:
                coeffs ^= coeff_sets[r]
            outpat = 0
            for n in coeffs:
                outpat ^= self.RED[n]
            ands.append((amask, bmask, outpat))
        return ands

    def _build_mul_ands(self):
        h = self.bits // 2  # half width (2 or 3)
        contrib = _k2_contrib() if h == 2 else _k3_contrib()
        lo = [1 << j for j in range(h)]
        hi = [1 << (j + h) for j in range(h)]
        su = [lo[j] | hi[j] for j in range(h)]
        nr = 2 * h - 1
        ands = []
        # c(x) = P0 (1 + x^h) + P1 x^h + P2 (x^h + x^2h)
        ands += self._half_ands(lo, lo, [{i, i + h} for i in range(nr)], contrib)
        ands += self._half_ands(su, su, [{i + h} for i in range(nr)], contrib)
        ands += self._half_ands(hi, hi, [{i + h, i + 2 * h} for i in range(nr)], contrib)
        return ands

    # ---- linear (CX-only) word ops ----
    def w_cx(self, b: Builder, src, dst):
        for s, d in zip(src, dst):
            b.cx(s, d)

    def w_const_mul_xor(self, b: Builder, src, dst, k):
        """dst ^= k*src for classical k (CX pattern)."""
        linear_xor(b, src, dst, f2_rows_of(lambda v: self.mul(k, v), self.bits))

    def w_pow2_xor(self, b: Builder, src, dst, e):
        """dst ^= src^(2^e) — Frobenius, F2-linear (CX pattern)."""
        linear_xor(b, src, dst, self.FROB_ROWS[e % self.bits])

    def mw_xor(self, b: Builder, flag, src, dst):
        """dst ^= flag * src (single mask qubit)."""
        for j in range(self.bits):
            b.ccx(flag, src[j], dst[j])

    # ---- loop-emission tables (per Builder; op-stream DATA segments) ------
    # The Karatsuba AND masks and the schoolbook CZ pattern are pinned
    # constants: recording them as data-GATED ops inside a repeat() collapses
    # each mul from ~540 recorded items to ~34 (expansion order and executed
    # gate counts are identical to the flat emission — gated-out ops do not
    # exist in the expansion).
    def _emit_tables(self, b: Builder):
        h = getattr(b, "_gf_tables", None)
        if h is None:
            h = b._gf_tables = {}  # ty: ignore[unresolved-attribute]
        key = self.q0
        if key not in h:
            B = self.bits
            abits, bbits, obits = [], [], []
            for am, bm, op in self.MUL_ANDS:
                abits += [(am >> j) & 1 for j in range(B)]
                bbits += [(bm >> j) & 1 for j in range(B)]
                obits += [(op >> j) & 1 for j in range(B)]
            pbits = [
                (self.RED[i + k] >> j) & 1 for j in range(B) for i in range(B) for k in range(B)
            ]
            h[key] = (
                b.declare_table(abits),
                b.declare_table(bbits),
                b.declare_table(obits),
                b.declare_table(pbits),
            )
        return h[key]

    # ---- variable*variable multiply ----
    def mul_xor(self, b: Builder, a, c, out):
        """out ^= a*c; all ancillas released before return. Recorded as ONE
        repeat over the AND terms with data-gated CX patterns.
        """
        B = self.bits
        ta, tb, to, _ = self._emit_tables(b)
        with b.repeat(len(self.MUL_ANDS)) as t:
            u = b.qubit()
            v = b.qubit()
            for j in range(B):
                with b.gated(ta, t * B + j):
                    b.cx(a[j], u)
                with b.gated(tb, t * B + j):
                    b.cx(c[j], v)
            tq = b.and_compute(u, v)
            for i in range(B):
                with b.gated(to, t * B + i):
                    b.cx(tq, out[i])
            b.and_release(tq, u, v)
            for j in range(B - 1, -1, -1):
                with b.gated(tb, t * B + j):
                    b.cx(c[j], v)
                with b.gated(ta, t * B + j):
                    b.cx(a[j], u)
            b.recycle(v)
            b.recycle(u)

    def cz_product(self, b: Builder, a, c, j, cond):
        """Phase (-1)^{bit_j(a*c)} under classical condition: the schoolbook
        bilinear CZ network (the MBU fixup primitive). Deliberately FLAT:
        the pattern is ~50% dense, so data-gating would double the recorded
        ops and pay the gated-record overhead on every one (profiled 7x
        slower).
        """
        for i in range(self.bits):
            for k in range(self.bits):
                if (self.RED[i + k] >> j) & 1:
                    b.cz(a[i], c[k], cond=cond)

    def mbu_release_product(self, b: Builder, out, a, c):
        """Release a word holding EXACTLY a*c (a, c still live). 0 Toffoli."""
        for j in range(self.bits - 1, -1, -1):
            m = b.hmr(out[j])
            self.cz_product(b, a, c, j, m)
            b.recycle(out[j])

    # ---- inverse: Itoh-Tsujii with free Frobenius squarings (0 -> 0) ----
    # q0=8: a^62 = ((a^3)^4 * a^3)^4 * a^2   (3 muls)
    # q0=4: a^14 = (a^3)^4 * a^2             (2 muls)
    def inv_xor(self, b: Builder, a, out):
        B = self.bits
        a2 = b.qubits(B)
        self.w_pow2_xor(b, a, a2, 1)
        a3 = b.qubits(B)
        self.mul_xor(b, a, a2, a3)
        stack = [(a3, None)]  # (word, product parents)
        cur = a3
        if self.q0 == 8:
            a12 = b.qubits(B)
            self.w_pow2_xor(b, a3, a12, 2)
            a15 = b.qubits(B)
            self.mul_xor(b, a3, a12, a15)
            stack += [(a12, "pow2"), (a15, (a3, a12))]
            cur = a15
        top = b.qubits(B)
        self.w_pow2_xor(b, cur, top, 2)
        self.mul_xor(b, top, a2, out)
        # releases, reverse dependency order
        self.w_pow2_xor(b, cur, top, 2)
        b.recycle_all(list(reversed(top)))
        if self.q0 == 8:
            a15, a12 = stack[2][0], stack[1][0]
            self.mbu_release_product(b, a15, a3, a12)
            self.w_pow2_xor(b, a3, a12, 2)
            b.recycle_all(list(reversed(a12)))
        self.mbu_release_product(b, a3, a, a2)
        self.w_pow2_xor(b, a, a2, 1)
        b.recycle_all(list(reversed(a2)))

    # ---- zero / nonzero flags ----
    def zero_flag(self, b: Builder, w):
        """Return (flag, tree): flag = (w == 0). 5 Tof at B=6, 3 at B=4."""
        B = self.bits
        for j in range(B):
            b.x(w[j])
        pairs = [b.and_compute(w[2 * i], w[2 * i + 1]) for i in range(B // 2)]
        if B == 4:
            top = b.and_compute(pairs[0], pairs[1])
            tree = (pairs, top)
        else:
            t4 = b.and_compute(pairs[0], pairs[1])
            top = b.and_compute(t4, pairs[2])
            tree = (pairs, t4, top)
        for j in range(B):
            b.x(w[j])
        return top, tree

    def release_zero(self, b: Builder, w, _flag, tree):
        """0 executed Toffoli (w unchanged since compute)."""
        B = self.bits
        for j in range(B):
            b.x(w[j])
        if B == 4:
            pairs, top = tree
            b.and_release(top, pairs[0], pairs[1])
        else:
            pairs, t4, top = tree
            b.and_release(top, t4, pairs[2])
            b.and_release(t4, pairs[0], pairs[1])
        for i in range(B // 2 - 1, -1, -1):
            b.and_release(pairs[i], w[2 * i], w[2 * i + 1])
        for j in range(B):
            b.x(w[j])

    def nonzero_flag(self, b: Builder, w):
        """Return (flag, tree): flag = (w != 0)."""
        top, tree = self.zero_flag(b, w)
        b.x(top)  # all-zero -> (w != 0)
        return top, tree

    def release_nonzero(self, b: Builder, w, flag, tree):
        """0 executed Toffoli (w unchanged since compute)."""
        b.x(flag)
        self.release_zero(b, w, flag, tree)

    # ---- counters (vote machinery) --------------------------------------
    # Note: increment carries are computed from PRE-flip bits,
    # so their release controls are mutated -> the CZ(t_prev, bit) +
    # Z(t_prev) measured identity ((-1)^{m*t*(bit^t)} = (-1)^{m*(t*bit ^ t)}).
    # Decrement borrows are computed from POST-flip bits (unmutated after),
    # so plain and_release applies. Both: n-1 executed Toffoli, free release.

    @staticmethod
    def ctr_inc(b: Builder, g, ctr):
        """ctr += g (g a control qubit, not consumed)."""
        n = len(ctr)
        ts = [g]
        for i in range(n - 1):
            t = b.and_compute(ts[-1], ctr[i])  # carry from OLD bit
            b.cx(ts[-1], ctr[i])
            ts.append(t)
        b.cx(ts[-1], ctr[n - 1])
        for i in range(n - 1, 0, -1):
            # release against the NEW bit needs the extra conditioned Z
            # (t was computed from the OLD bit; CZ+Z identity, see above)
            m = b.and_release(ts[i], ts[i - 1], ctr[i - 1])
            b.z(ts[i - 1], cond=m)

    @staticmethod
    def ctr_dec(b: Builder, g, ctr):
        """ctr -= g (exact inverse of ctr_inc for the same g)."""
        n = len(ctr)
        ts = [g]
        b.cx(g, ctr[0])
        for i in range(1, n):
            t = b.and_compute(ts[-1], ctr[i - 1])  # borrow from NEW bit
            ts.append(t)
            b.cx(t, ctr[i])
        for i in range(n - 1, 0, -1):
            b.and_release(ts[i], ts[i - 1], ctr[i - 1])

    # ---- strict-majority comparator: flag = (2*val(v) > val(V)) ----------
    # Borrow chain of V - 2v over n+1 positions (A = v shifted left one).
    # br_{i+1} = A_i*B'_i XOR br_i*~B'_i with B'_i = B_i^A_i (B mutated in
    # place, restored per step). ~2(n-1)+1 Tof compute; release = exact
    # replay-reverse (operands unmutated between compute and release).

    @staticmethod
    def _gt2_step(b: Builder, a, bq, br):
        """Return br_next given bits a (A_i), bq (B_i register qubit) and
        prior borrow br (or None = 0). Leaves a, bq unchanged.
        """
        nxt = b.qubit()
        b.cx(a, bq)  # bq := B'
        b.ccx(a, bq, nxt)
        if br is not None:
            b.x(bq)
            b.ccx(br, bq, nxt)
            b.x(bq)
        b.cx(a, bq)  # restore
        return nxt

    @staticmethod
    def _gt2_unstep(b: Builder, a, bq, br, nxt):
        b.cx(a, bq)
        if br is not None:
            b.x(bq)
            b.ccx(br, bq, nxt)
            b.x(bq)
        b.ccx(a, bq, nxt)
        b.cx(a, bq)
        b.recycle(nxt)

    def gt2_flag(self, b: Builder, v, V):
        """flag = (2*val(v) > val(V)); v, V same width n. Returns
        (flag, chain) for gt2_release.
        """
        n = len(v)
        chain = []
        br = None
        # position 0: A_0 = 0, B_0 = V[0] -> br_1 = 0 (skip).
        for i in range(1, n):
            br = self._gt2_step(b, v[i - 1], V[i], br)
            chain.append(br)
        # position n: A_n = v[n-1], B_n = 0 -> out = v[n-1] OR br.
        flag = b.qubit()
        b.cx(v[n - 1], flag)
        if br is not None:
            b.cx(br, flag)
            b.ccx(v[n - 1], br, flag)
        return flag, chain

    def gt2_release(self, b: Builder, v, V, flag, chain):
        """Exact replay-reverse of gt2_flag (v, V unchanged since compute)."""
        n = len(v)
        br = chain[-1] if chain else None
        if br is not None:
            b.ccx(v[n - 1], br, flag)
            b.cx(br, flag)
        b.cx(v[n - 1], flag)
        b.recycle(flag)
        for i in range(n - 1, 0, -1):
            prev = chain[i - 2] if i >= 2 else None
            self._gt2_unstep(b, v[i - 1], V[i], prev, chain[i - 1])


# ---- unit tests ---------------------------------------------------------------

if __name__ == "__main__":
    from uncompute import Simulator

    for q0 in (4, 8):
        gf = GF(q0)
        B, Q = gf.bits, gf.q

        # multiply + MBU release, exhaustive in 64-lane chunks
        for chunk in range(Q * Q // 64):
            pairs = [((chunk * 64 + i) // Q, (chunk * 64 + i) % Q) for i in range(64)]
            b = Builder()
            a = b.qubits(B)
            c = b.qubits(B)
            out = b.qubits(B)
            res = b.qubits(B)
            gf.mul_xor(b, a, c, out)
            gf.w_cx(b, out, res)
            gf.mbu_release_product(b, out, a, c)
            sim = Simulator(b, batch=64, seed=1)
            sim.write(a, [p[0] for p in pairs])
            sim.write(c, [p[1] for p in pairs])
            sim.run()
            assert sim.read(res) == [gf.mul(p[0], p[1]) for p in pairs]
            assert sim.read(out) == [0] * 64
            assert sim.phase == 0
        print(f"PASS q0={q0} mul_xor + mbu_release_product exhaustive, phase 0")

        # inverse, all values (chunked for q0=8)
        for chunk in range(Q // 64 or 1):
            vals = [(chunk * 64 + i) % Q for i in range(64)]
            b = Builder()
            a = b.qubits(B)
            out = b.qubits(B)
            gf.inv_xor(b, a, out)
            sim = Simulator(b, batch=64, seed=1)
            sim.write(a, vals)
            sim.run()
            assert sim.read(out) == [gf.inv(v) for v in vals]
            assert sim.phase == 0
        tof = b.phase_summary()["total_emitted_toffoli"]
        print(f"PASS q0={q0} inv_xor all values (emitted toffoli={tof}), phase 0")

        # nonzero flag + release
        b = Builder()
        w = b.qubits(B)
        res = b.qubit()
        flag, tree = gf.nonzero_flag(b, w)
        b.cx(flag, res)
        gf.release_nonzero(b, w, flag, tree)
        sim = Simulator(b, batch=64, seed=1)
        vals = [i % Q for i in range(64)]
        sim.write(w, vals)
        sim.run()
        assert sim.read([res]) == [int(v != 0) for v in vals]
        assert sim.phase == 0
        print(f"PASS q0={q0} nonzero_flag + release, phase 0")

        # zero flag + release
        b = Builder()
        w = b.qubits(B)
        res = b.qubit()
        flag, tree = gf.zero_flag(b, w)
        b.cx(flag, res)
        gf.release_zero(b, w, flag, tree)
        sim = Simulator(b, batch=64, seed=1)
        sim.write(w, vals)
        sim.run()
        assert sim.read([res]) == [int(v == 0) for v in vals]
        assert sim.phase == 0
        print(f"PASS q0={q0} zero_flag + release, phase 0")

    # counter inc/dec round-trip (the identities above), width 5 and 8
    for CB_ in (5, 8):
        b = Builder()
        g = b.qubit()
        ctr = b.qubits(CB_)
        b.x(g)
        reps = (1 << CB_) + 3  # wraps the counter at least once
        for _ in range(reps):
            GF.ctr_inc(b, g, ctr)
        snap = b.qubits(CB_)
        for j in range(CB_):
            b.cx(ctr[j], snap[j])
        for _ in range(reps):
            GF.ctr_dec(b, g, ctr)
        b.x(g)
        sim = Simulator(b, batch=8, seed=2)
        sim.run()
        assert sim.read(snap) == [reps % (1 << CB_)] * 8
        assert sim.read(ctr) == [0] * 8
        assert sim.read([g]) == [0] * 8
        assert sim.phase == 0
        print(f"PASS ctr_inc/ctr_dec width {CB_}: {reps} up, {reps} down, phase 0")

    # gt2 comparator: exhaustive over 4-bit (v, V)
    n_ = 4
    ok = 0
    for chunk in range(256 // 64):
        pairs = [((chunk * 64 + i) >> n_, (chunk * 64 + i) & (n_ * 0 + 15)) for i in range(64)]
        b = Builder()
        gfx = GF(4)
        v = b.qubits(n_)
        V = b.qubits(n_)
        res = b.qubit()
        flag, chain = gfx.gt2_flag(b, v, V)
        b.cx(flag, res)
        gfx.gt2_release(b, v, V, flag, chain)
        sim = Simulator(b, batch=64, seed=3)
        sim.write(v, [p[0] for p in pairs])
        sim.write(V, [p[1] for p in pairs])
        sim.run()
        want = [int(2 * p[0] > p[1]) for p in pairs]
        assert sim.read([res]) == want, (chunk, sim.read([res]), want)
        assert sim.read(v) == [p[0] for p in pairs]
        assert sim.read(V) == [p[1] for p in pairs]
        assert sim.phase == 0
        ok += 64
    print(f"PASS gt2_flag exhaustive {ok} pairs (4-bit), round-trip, phase 0")
