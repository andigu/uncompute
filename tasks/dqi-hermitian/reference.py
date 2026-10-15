"""Classical reference decoder for dqi-hermitian: Feng-Rao majority voting
for one-point Hermitian codes (Feng and Rao (1993), doi:10.1109/18.179340;
the form of Theorem 8.10 in the survey of Hoholdt, van Lint, and
Pellikaan (1998)).

The decoder is written for the curve y^q0 + y = x^(q0+1) over GF(q0^2) and
runs at the task instance q0 = 8.  It chooses its spaces by the general
formulas

  locator space  L(ell + g)         (q0=8: L(75), dim 48)
  test space     L(ell + 2g - 1)    (q0=8: L(102))
  syndromes through 2*ell + 3g - 1  (q0=8: 177, 27 voted)

Validity at q0 = 8: voting needs nu(w) > 2*ell for all voted w; the minimum
is nu(151) = 152 - 2g = 96 > 94.  Kernel elements vanish on supp(e) because
u_i = f(P_i) e_i lies in the residue code of L(ell+2g-1), whose Goppa
distance is ell + 1 > wt(e).  |Z| <= ell + g = 75 < d = 96 makes the value
solve unique (spurious positions receive 0).

Conventions match instance.py: GF(64) with the galois-default poly (91),
polynomial basis; points ordered lexicographically by (x, y) integer pairs,
x major; H rows = monomials x^a y^b (b <= q0-1, pole order 8a+9b <= A),
b-major, then a ascending.

Numpy vectorizes the GF(q) row operations (log/exp tables) only.
"""

import hashlib

import numpy as np

POLY_OF_Q0 = {8: 91}  # galois-package default for GF(64)
# Fixed instance constant: the SHA-256 tag that defines the fixup vector c.
C_INSTANCE_TAGS = {8: b"kickmix/hopi-512/fixup-c/v1"}


class Hermitian:
    def __init__(self, q0):
        self.q0 = q0
        self.q = q0 * q0
        self.m = q0**3
        self.g = q0 * (q0 - 1) // 2
        self.bits = {8: 6}[q0]
        self.A = {8: 150}[q0]
        self.d = self.A - 2 * self.g + 2
        self.ell = (self.d - 1) // 2
        self.n_checks = self.A - self.g + 1
        self.u_loc = self.ell + self.g
        self.v_test = self.ell + 2 * self.g - 1
        self.w_max = self.u_loc + self.v_test  # = 2*ell + 3g - 1
        poly = POLY_OF_Q0[q0]

        # ---- field tables (log/exp; x is primitive for the pinned poly) ----
        q = self.q
        exp = np.zeros(2 * q, dtype=np.int32)
        log = np.zeros(q, dtype=np.int32)
        v = 1
        for i in range(q - 1):
            exp[i] = v
            log[v] = i
            v <<= 1
            if v & q:
                v ^= poly | q
        for i in range(q - 1, 2 * q):
            exp[i] = exp[i - (q - 1)]
        self.EXP, self.LOG = exp, log

        # ---- curve points ----
        pts = [
            (x, y)
            for x in range(q)
            for y in range(q)
            if self.gpow(y, q0) ^ y == self.gpow(x, q0 + 1)
        ]
        assert len(pts) == self.m
        self.PTS = pts

        # ---- monomials / nongaps through w_max ----
        self.MONOS = [
            (a, b)
            for b in range(q0)
            for a in range((self.w_max // q0) + 1)
            if q0 * a + (q0 + 1) * b <= self.w_max
        ]
        self.ORDER = {mn: q0 * mn[0] + (q0 + 1) * mn[1] for mn in self.MONOS}
        self.NONGAPS = sorted(self.ORDER[mn] for mn in self.MONOS)
        assert len(set(self.NONGAPS)) == len(self.NONGAPS)
        self.MONO_OF_ORDER = {self.ORDER[mn]: mn for mn in self.MONOS}
        self.H_MONOS = [mn for mn in self.MONOS if self.ORDER[mn] <= self.A]
        assert len(self.H_MONOS) == self.n_checks

        # voting validity: nu(w) > 2*ell for every voted order, where nu is
        # the Feng-Rao count over ALL nongap pairs u + v = w (including
        # (0, w) and (w, 0)); decoding radius = floor((min nu - 1)/2),
        # survey Thm 8.10. Gaps all lie below 2g, so membership is decidable
        # from the <= w_max nongap list plus the "past 2g-1 everything is a
        # nongap" fact.
        ngset = set(self.NONGAPS)

        def is_nongap(o):
            return o >= 2 * self.g or o in ngset

        for w in range(self.A + 1, self.w_max + 1):
            nu = sum(1 for u in range(w + 1) if is_nongap(u) and is_nongap(w - u))
            assert nu > 2 * self.ell, (w, nu)

        # ---- H (n_checks x m) and monomial evaluation table ----
        self.H = np.array(
            [[self.mono_eval(mn, p) for p in pts] for mn in self.H_MONOS], dtype=np.uint8
        )

        # ---- product-reduction table (F2 XOR-multisets on the curve) ----
        self.PROD = {}
        for u in self.NONGAPS:
            au, bu = self.MONO_OF_ORDER[u]
            for v in self.NONGAPS:
                if v > u or u + v > self.w_max:
                    continue
                av, bv = self.MONO_OF_ORDER[v]
                terms = self._reduce(au + av, bu + bv)
                lead = [mn for mn in terms if self.ORDER[mn] == u + v]
                assert len(lead) == 1 and all(self.ORDER[mn] <= u + v for mn in terms)
                self.PROD[(u, v)] = self.PROD[(v, u)] = [self.ORDER[mn] for mn in terms]

        # locator/test monomial lists
        self.L_LOC = [mn for mn in self.MONOS if self.ORDER[mn] <= self.u_loc]
        self.L_TEST = [mn for mn in self.MONOS if self.ORDER[mn] <= self.v_test]
        assert len(self.L_LOC) == self.u_loc - self.g + 1
        # evaluation table of locator monomials at all points
        self.LOC_EVAL = np.array(
            [[self.mono_eval(mn, p) for p in pts] for mn in self.L_LOC], dtype=np.uint8
        )

        # fixup vector c (counter-mode SHA-256 XOF, little-endian bit packing)
        nbytes = self.m * self.bits // 8
        c_instance_tag = C_INSTANCE_TAGS[q0]
        stream = b"".join(
            hashlib.sha256(c_instance_tag + i.to_bytes(4, "little")).digest()
            for i in range((nbytes + 31) // 32)
        )[:nbytes]
        self.C_INT = int.from_bytes(stream, "little")

    # ---- scalar field ops ----
    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return int(self.EXP[self.LOG[a] + self.LOG[b]])

    def inv(self, a):
        return int(self.EXP[(self.q - 1 - self.LOG[a]) % (self.q - 1)])

    def gpow(self, a, e):
        if a == 0:
            return 0 if e else 1
        return int(self.EXP[(int(self.LOG[a]) * e) % (self.q - 1)])

    def mono_eval(self, mn, p):
        a, b = mn
        x, y = p
        return self.mul(self.gpow(x, a), self.gpow(y, b))

    # ---- vectorized field ops (uint8 arrays) ----
    def vmul(self, vec, s):
        """vec * scalar s (s != 0)."""
        out = self.EXP[self.LOG[vec] + self.LOG[s]].astype(np.uint8)
        out[vec == 0] = 0
        return out

    def _reduce(self, a, b):
        """x^a y^b -> canonical F2 multiset via y^q0 = x^(q0+1) + y."""
        out = {}
        stack = [(a, b)]
        while stack:
            aa, bb = stack.pop()
            if bb <= self.q0 - 1:
                out[(aa, bb)] = out.get((aa, bb), 0) ^ 1
            else:
                stack.append((aa + self.q0 + 1, bb - self.q0))
                stack.append((aa, bb - self.q0 + 1))
        return [mn for mn, c in out.items() if c]

    # ---- syndrome bookkeeping ----
    def _syn_pair(self, syn, u, v):
        t = 0
        for o in self.PROD[(u, v)]:
            t ^= syn[o]
        return t

    def _lower_part(self, syn, u, v):
        w = u + v
        t = 0
        for o in self.PROD[(u, v)]:
            if o != w:
                t ^= syn[o]
        return t

    # ---- Feng-Rao majority voting for one unknown pole order ----
    def _vote(self, syn, w):
        rows = [u for u in self.NONGAPS if u < w]
        cols = [v for v in self.NONGAPS if v < w]
        col_idx = {v: i for i, v in enumerate(cols)}
        # defined prefix of row u: cols v with u + v < w (cols ascending)
        dlen = {u: sum(1 for v in cols if u + v < w) for u in rows}

        pivot_col_of_row = {}
        pivot_row_of_col = {}
        reduced = {}
        lam = {}
        for u in rows:
            nd = dlen[u]
            r = np.array([self._syn_pair(syn, u, v) for v in cols[:nd]], dtype=np.uint8)
            coeffs = {}
            for pu, pv in pivot_col_of_row.items():
                j = col_idx[pv]
                if j < nd and r[j]:
                    f = self.mul(int(r[j]), self.inv(int(reduced[pu][j])))
                    coeffs[pu] = f
                    r ^= self.vmul(reduced[pu][:nd], f)
            lam[u] = coeffs
            reduced[u] = r
            nz = np.nonzero(r)[0]
            if nz.size:
                k = int(nz[0])
                pivot_col_of_row[u] = cols[k]
                pivot_row_of_col[cols[k]] = u

        votes = {}
        for u in rows:
            v = w - u
            if v not in col_idx:
                continue
            pv = pivot_col_of_row.get(u)
            if pv is not None and pv < v:
                continue
            ru = pivot_row_of_col.get(v)
            if ru is not None and ru < u:
                continue
            j = col_idx[v]
            pred = 0
            for pu, f in lam[u].items():
                # defined: pu < u  =>  pu + v < u + v = w
                pred ^= self.mul(f, int(reduced[pu][j]))
            cand = pred ^ self._lower_part(syn, u, v)
            votes[cand] = votes.get(cand, 0) + 1

        if not votes:
            return 0
        return max(votes.items(), key=lambda kv: kv[1])[0]

    # ---- locator kernel over L(u_loc) ----
    def _kernel(self, syn):
        ncols = len(self.L_LOC)
        rows = []
        for pm in self.L_TEST:
            vord = self.ORDER[pm]
            rows.append([self._syn_pair(syn, self.ORDER[fm], vord) for fm in self.L_LOC])
        A = np.array(rows, dtype=np.uint8)
        # RREF over GF(q)
        pivots = {}
        ech = []
        for r in A:
            r = r.copy()
            for c, ei in pivots.items():
                if r[c]:
                    r ^= self.vmul(ech[ei], int(r[c]))  # ech rows are monic
            nz = np.nonzero(r)[0]
            if nz.size:
                lead = int(nz[0])
                r = self.vmul(r, self.inv(int(r[lead])))
                pivots[lead] = len(ech)
                ech.append(r)
        for c, ei in pivots.items():
            for oi in range(len(ech)):
                if oi != ei and ech[oi][c]:
                    ech[oi] = ech[oi] ^ self.vmul(ech[ei], int(ech[oi][c]))
        free = [c for c in range(ncols) if c not in pivots]
        basis = []
        for fc in free:
            vec = np.zeros(ncols, dtype=np.uint8)
            vec[fc] = 1
            for c, ei in pivots.items():
                vec[c] = ech[ei][fc]
            basis.append(vec)
        return basis

    def _common_zeros(self, kbasis):
        vals = np.zeros((len(kbasis), self.m), dtype=np.uint8)
        for i, vec in enumerate(kbasis):
            acc = np.zeros(self.m, dtype=np.uint8)
            for c in np.nonzero(vec)[0]:
                acc ^= self.vmul(self.LOC_EVAL[c], int(vec[c]))
            vals[i] = acc
        return [int(i) for i in np.nonzero((vals != 0).sum(axis=0) == 0)[0]]

    def _solve_values(self, zs, s_arr):
        ncols = len(zs)
        aug = np.concatenate(
            [self.H[:, zs], np.asarray(s_arr, dtype=np.uint8)[:, None]], axis=1
        ).astype(np.uint8)
        nrows = aug.shape[0]
        rr = 0
        piv = []
        for c in range(ncols):
            sel = next((i for i in range(rr, nrows) if aug[i, c]), None)
            if sel is None:
                return None
            aug[[rr, sel]] = aug[[sel, rr]]
            aug[rr] = self.vmul(aug[rr], self.inv(int(aug[rr, c])))
            for i in range(nrows):
                if i != rr and aug[i, c]:
                    aug[i] ^= self.vmul(aug[rr], int(aug[i, c]))
            piv.append(c)
            rr += 1
        if rr < nrows and aug[rr:].any():
            return None
        e = [0] * self.m
        for k, c in enumerate(piv):
            e[zs[c]] = int(aug[k, ncols])
        return e

    # ---- top-level ----
    def decode(self, s_syms):
        """s_syms: n_checks field symbols (H row order), s = H e, wt(e) <= ell.
        Returns the unique e as a list of m ints (or None off-promise).
        """
        syn = {}
        for r, mn in enumerate(self.H_MONOS):
            syn[self.ORDER[mn]] = int(s_syms[r])
        for w in range(self.A + 1, self.w_max + 1):
            if w in self.MONO_OF_ORDER:
                syn[w] = self._vote(syn, w)
        kb = self._kernel(syn)
        if not kb:
            return None
        zs = self._common_zeros(kb)
        if len(zs) > self.u_loc:
            return None
        if not zs:
            return [0] * self.m
        return self._solve_values(zs, s_syms)

    def syndrome_of(self, e):
        ev = np.asarray(e, dtype=np.uint8)
        out = np.zeros(self.n_checks, dtype=np.uint8)
        nz = np.nonzero(ev)[0]
        for j in nz:
            out ^= self.vmul(self.H[:, j], int(ev[j]))
        return [int(x) for x in out]

    def phase_of(self, e):
        packed = 0
        for i, v in enumerate(e):
            packed |= int(v) << (self.bits * i)
        return bin(self.C_INT & packed).count("1") & 1

    def task_phase(self, s_syms):
        e = self.decode(s_syms)
        assert e is not None, "off-promise syndrome"
        return self.phase_of(e)


# ---- validation -----------------------------------------------------------


def _gauntlet(inst, per_weight, weights, rng, label):
    import time

    t0 = time.time()
    n = 0
    for w in weights:
        for _ in range(per_weight):
            e = [0] * inst.m
            for pos in rng.sample(range(inst.m), w):
                e[pos] = rng.randrange(1, inst.q)
            got = inst.decode(inst.syndrome_of(e))
            assert got == e, (label, w, e.index(max(e)))
            n += 1
        print(
            f"PASS  {per_weight} random errors at weight {w} ({(time.time() - t0):.1f}s cum)",
            flush=True,
        )
    return n


if __name__ == "__main__":
    import argparse
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--q0", type=int, default=8, choices=[8])
    ap.add_argument("--per-weight", type=int, default=20)
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--seed", type=int, default=0xFEB0512)
    ap.add_argument(
        "--structured",
        type=int,
        default=0,
        help="count of equal-value + clustered full-radius extras",
    )
    args = ap.parse_args()

    inst = Hermitian(args.q0)
    print(
        f"q0={args.q0}: m={inst.m} n={inst.n_checks} d>={inst.d} "
        f"ell={inst.ell} u_loc={inst.u_loc} v_test={inst.v_test} "
        f"w_max={inst.w_max} voted={inst.w_max - inst.A}"
    )
    rng = random.Random(args.seed)

    # zero + singles smoke
    assert inst.decode(inst.syndrome_of([0] * inst.m)) == [0] * inst.m
    for pos in (0, inst.m // 2, inst.m - 1):
        e = [0] * inst.m
        e[pos] = inst.q - 1
        assert inst.decode(inst.syndrome_of(e)) == e
    print("PASS  zero + boundary singles")

    weights = [int(x) for x in args.weights.split(",") if x] or sorted(
        {0, 1, 2, inst.ell // 2, inst.ell - 1, inst.ell}
    )
    n = _gauntlet(inst, args.per_weight, weights, rng, "rand")

    if args.structured:
        for t in range(args.structured):
            e = [0] * inst.m
            val = rng.randrange(1, inst.q)
            for pos in rng.sample(range(inst.m), inst.ell):
                e[pos] = val
            assert inst.decode(inst.syndrome_of(e)) == e, ("equal-values", t)
            n += 1
        print(f"PASS  {args.structured} weight-{inst.ell} equal-value errors", flush=True)
        for t in range(args.structured):
            start = rng.randrange(0, inst.m - inst.ell)
            e = [0] * inst.m
            for pos in range(start, start + inst.ell):
                e[pos] = rng.randrange(1, inst.q)
            assert inst.decode(inst.syndrome_of(e)) == e, ("clustered", t)
            n += 1
        print(f"PASS  {args.structured} weight-{inst.ell} clustered errors", flush=True)
        # phase spot-checks against direct computation
        for _ in range(200):
            w = rng.randrange(0, inst.ell + 1)
            e = [0] * inst.m
            for pos in rng.sample(range(inst.m), w):
                e[pos] = rng.randrange(1, inst.q)
            assert inst.task_phase(inst.syndrome_of(e)) == inst.phase_of(e)
            n += 1
        print("PASS  200 phase round-trips", flush=True)

    print(f"ALL PASS ({n} decodes)")
