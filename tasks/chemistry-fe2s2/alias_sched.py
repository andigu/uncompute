"""chemistry-fe2s2 alias scheduler: credit-aligned two-phase melt.

Contract step 7, called by reference.py: phase A melts heavy larges
onto shared keep words through exact whole-entry subset fills over
the run-atomic small stream; phase B closes the near-threshold
larges through credit-chained window closes.  Window-closed larges'
donations (value T - kappa, any residue) are banked and routed so
that row-adjacent melted larges share one keep-word class,
circumventing the coarse pool's mod-4 donation lattice (on the
pinned coefficients this alignment cuts the melt-word class flips
from 314 to 23).  Deterministic and seedless; the closing assertion
reproduces the integer weights bit-perfectly.  Every function here
is plain single-process Python (greedy + exact replay + bounded
repair).
"""

import bisect
from collections import Counter, defaultdict

T = 1 << 16
W = T - 1
# Scheduler knobs.  They belong to the task definition: contract.json's
# data_sha256 pins the table that they produce.
MELT_COARSE = 81_919  # phase-A pool floor: fills use entries that donate more
KEEP_BASE = T - 4  # a melted keep word is KEEP_BASE + its class
MELT_NEAR_T = 147_455  # phase-B membership: larges with m <= MELT_NEAR_T
MELT_MAX_UNDO = 8  # exact-fill repair depth; depths 2 to 32 give the same table
MELT_PROBE_CAP = 8_000  # two-entry probe bound; above the 6,492 donation values, so inactive


def _build_stream(m, dup_groups):
    L = len(m)
    smalls = {i for i in range(L) if m[i] < T}
    dup_master = {}
    dupb = set()
    for n, bs in dup_groups:
        for b in bs:
            dupb.add(b)
            dupb.add(b + n)
        mb = bs[0]
        if not all(m[c + o] == m[mb + o] for c in bs[1:] for o in range(n)):
            continue
        for c in bs[1:]:
            for o in range(n):
                if (mb + o) in smalls:
                    dup_master[c + o] = mb + o
    rev = defaultdict(list)
    for c, mstr in dup_master.items():
        rev[mstr].append(c)
    stream: list[list] = []
    i = 0
    while i < L:
        if i not in smalls or i in dup_master:
            i += 1
            continue
        j = i
        while (
            j < L
            and j in smalls
            and j not in dup_master
            and m[j] == m[i]
            and (j == i or j not in dupb)
        ):
            j += 1
        stream.append([i, j - i, m[i], None])
        i = j
    for e in stream:
        st, n_r, _mv, _ = e
        cands = defaultdict(int)
        for o in range(n_r):
            for cc in rev.get(st + o, ()):
                cands[cc - o] += 1
        e[3] = [st] + sorted(c for c, k in cands.items() if k == n_r)
    return smalls, stream


class _State:
    """One scheduling pass over a fresh copy of the stream."""

    def __init__(self, m, smalls, stream):
        self.m = m
        L = len(m)
        self.L = L
        self.keep = [None] * L
        self.alt = [None] * L
        for s in smalls:
            self.keep[s] = m[s]
        self.larges = [i for i in range(L) if m[i] > T]
        self.lg_sorted = sorted(self.larges, key=lambda i: (-m[i], i))
        self.x1 = next(i for i in self.lg_sorted if i != 0)
        self.stream = [list(e) for e in stream]
        NE = len(self.stream)
        self.NE = NE
        self.don = [(T - e[2]) * e[1] * len(e[3]) for e in self.stream]
        self.alive = [True] * NE
        self.pool = sorted(range(NE), key=lambda k: (-self.don[k], self.stream[k][0]))
        self.by_don = defaultdict(list)
        for k in self.pool:
            self.by_don[self.don[k]].append(k)
        self.vals = sorted(self.by_don)
        self.taken = set()
        self.credit = defaultdict(int)
        self.n_split = 0
        self.consumed_by = {}
        self.melted = []
        self.fill_fail = []
        self.phase_a = [q for q in self.lg_sorted if q != self.x1 and m[q] > MELT_NEAR_T]
        self.phase_b = sorted(
            (q for q in self.lg_sorted if q != self.x1 and m[q] <= MELT_NEAR_T), key=lambda q: -m[q]
        )
        self.open_b = set(self.phase_b)

    def consume(self, k, tgt):
        e = self.stream[k]
        for rs in e[3]:
            for o in range(e[1]):
                self.alt[rs + o] = tgt
        self.alive[k] = False
        self.consumed_by[k] = tgt

    def grab(self, v, lo_val):
        if v < lo_val:
            return None
        for k in self.by_don.get(v, ()):
            if self.alive[k] and k not in self.taken and self.don[k] == v:
                return k
        return None

    def solve2(self, g, lo_val):
        i1 = bisect.bisect_right(self.vals, g // 2)
        probes = 0
        for idx in range(i1 - 1, -1, -1):
            v1 = self.vals[idx]
            if v1 < lo_val:
                break
            probes += 1
            if probes > MELT_PROBE_CAP:
                break
            v2 = g - v1
            k1 = self.grab(v1, lo_val)
            if k1 is None:
                continue
            self.taken.add(k1)
            k2 = self.grab(v2, lo_val)
            self.taken.discard(k1)
            if k2 is not None:
                return k1, k2
        return None

    def exact_fill(self, V, lo_val):
        S = 0
        seq = []
        for k in self.pool:
            if not self.alive[k] or k in self.taken or self.don[k] <= lo_val:
                continue
            if S + self.don[k] <= V:
                seq.append(k)
                self.taken.add(k)
                S += self.don[k]
                if S == V:
                    return seq
        g = V - S
        for _undo in range(MELT_MAX_UNDO + 1):
            hit = self.grab(g, lo_val)
            if hit is not None:
                seq.append(hit)
                self.taken.add(hit)
                return seq
            for a in reversed(seq[-8:]):
                cand = self.grab(self.don[a] + g, lo_val)
                if cand is not None:
                    seq.remove(a)
                    self.taken.discard(a)
                    seq.append(cand)
                    self.taken.add(cand)
                    return seq
            pair = self.solve2(g, lo_val)
            if pair is not None:
                for k in pair:
                    seq.append(k)
                    self.taken.add(k)
                return seq
            if not seq:
                break
            a = seq.pop()
            self.taken.discard(a)
            g += self.don[a]
        for k in seq:
            self.taken.discard(k)
        return None

    def window_close(self, lq):
        m = self.m
        S = self.credit[lq]
        tgt_lo, tgt_hi = m[lq] - W, m[lq] - 1
        for k in self.pool:
            if not self.alive[k] or k in self.taken:
                continue
            if S >= tgt_lo:
                break
            if S + self.don[k] <= tgt_hi:
                self.consume(k, lq)
                S += self.don[k]
        if S < tgt_lo:
            best = None
            for k2 in self.pool:
                if not self.alive[k2] or k2 in self.taken:
                    continue
                st, n_r, mv, reps = self.stream[k2]
                d1 = (T - mv) * len(reps)
                for jj in range(1, n_r):
                    if tgt_lo <= S + jj * d1 <= tgt_hi:
                        p = st + jj
                        v2 = (p & -p).bit_length() - 1
                        c = (-v2, k2, jj)
                        if best is None or c < best:
                            best = c
            if best is None:
                raise AssertionError(f"no split can close {lq}")
            _nv, k2, jj = best
            e = self.stream[k2]
            st, n_r, mv, reps = e
            for rs in reps:
                for o in range(jj):
                    self.alt[rs + o] = lq
            e[0] = st + jj
            e[1] = n_r - jj
            e[3] = [r + jj for r in reps]
            dnew = (T - mv) * e[1] * len(reps)
            self.by_don[dnew].append(k2)
            self.don[k2] = dnew
            S += jj * (T - mv) * len(reps)
            self.n_split += 1
        kv = m[lq] - S
        assert 1 <= kv <= W
        self.keep[lq] = kv
        return kv

    route_asc = False

    def route(self, dn, fallback_pool):
        m = self.m
        best = None
        order = self.phase_b[::-1] if self.route_asc else self.phase_b
        for q in order:
            if q not in self.open_b:
                continue
            if m[q] - W <= self.credit[q] + dn <= m[q] - 1:
                best = q
                break
            if self.credit[q] + dn <= m[q] - 1 and best is None:
                best = q
                break
        if best is not None:
            self.credit[best] += dn
            return best
        hr_best = 0
        nxt = None
        for q in fallback_pool:
            hr = m[q] - self.credit[q] - dn
            if hr >= 1 and hr > hr_best:
                hr_best, nxt = hr, q
        if nxt is not None:
            self.credit[nxt] += dn
            return nxt
        return self.x1

    def finish(self):
        for k in range(self.NE):
            if self.alive[k]:
                self.consume(k, self.x1)
        self.keep[self.x1] = 0
        self.alt[self.x1] = self.x1
        m = self.m
        D = [0] * self.L
        for j in range(self.L):
            k, a = self.keep[j], self.alt[j]
            assert k is not None and a is not None, j
            D[j] += k
            D[a] += T - k
        assert D == m, "alias table does not reproduce integer weights"
        return self.keep, self.alt


def _phase_b_pass(st):
    m = st.m
    while st.open_b:
        lcur = None
        for q in st.phase_b:
            if q in st.open_b and m[q] - W <= st.credit[q] <= m[q] - 1:
                lcur = q
                break
        if lcur is None:
            lcur = max(st.open_b, key=lambda q: (m[q], -q))
        st.open_b.discard(lcur)
        kv = st.window_close(lcur)
        st.alt[lcur] = st.route(T - kv, [])


def exact_fill_pref(st, V, lo_val, pref):
    """exact_fill with a preferred entry ordering (dry set first)."""
    S = 0
    seq = []
    order = [k for k in pref if st.alive[k] and k not in st.taken]
    rest = [k for k in st.pool if k not in set(pref)]
    for k in order + rest:
        if not st.alive[k] or k in st.taken or st.don[k] <= lo_val:
            continue
        if S + st.don[k] <= V:
            seq.append(k)
            st.taken.add(k)
            S += st.don[k]
            if S == V:
                return seq
    g = V - S
    for _undo in range(8 + 1):
        hit = st.grab(g, lo_val)
        if hit is not None:
            seq.append(hit)
            st.taken.add(hit)
            return seq
        for a in reversed(seq[-8:]):
            cand = st.grab(st.don[a] + g, lo_val)
            if cand is not None:
                seq.remove(a)
                st.taken.discard(a)
                seq.append(cand)
                st.taken.add(cand)
                return seq
        pair = st.solve2(g, lo_val)
        if pair is not None:
            for k in pair:
                seq.append(k)
                st.taken.add(k)
            return seq
        if not seq:
            break
        a = seq.pop()
        st.taken.discard(a)
        g += st.don[a]
    for k in seq:
        st.taken.discard(k)
    return None


def dry_pass_recorded(m, smalls, stream):
    """Pinned pass recording fill sets and failure close sets."""
    dry = _State(m, smalls, stream)
    dry_fills = {}
    fail_close = {}
    for pos, lq in enumerate(dry.phase_a):
        cr = dry.credit[lq]
        assert cr == 0, f"phase-A credit nonzero in dry pass: {lq}"
        kv = KEEP_BASE + ((m[lq] - cr) & 3)
        V = m[lq] - cr - kv
        seq = dry.exact_fill(V, MELT_COARSE) if V >= 4 else None
        if seq is not None:
            dry_fills[lq] = list(seq)
            for k in seq:
                dry.consume(k, lq)
                dry.taken.discard(k)
            dry.keep[lq] = kv
            dry.alt[lq] = dry.x1
            dry.melted.append(lq)
            continue
        pre = dict(dry.consumed_by)
        kv = dry.window_close(lq)
        fail_close[lq] = [k for k in dry.consumed_by if k not in pre]
        assert dry.n_split == 0, "dry failure close split a run"
        dry.alt[lq] = dry.route(T - kv, dry.phase_a[pos + 1 :])
        dry.fill_fail.append(lq)
    _phase_b_pass(dry)
    dry.finish()
    return dry, dry_fills, fail_close


def window_close_dead_fine(st, lq, allowed):
    """Close lq from `allowed` entries only, smallest donation first;
    rolls back and returns None on failure.
    """
    m = st.m
    S = st.credit[lq]
    tgt_lo, tgt_hi = m[lq] - W, m[lq] - 1
    used = []
    order = sorted(
        (k for k in allowed if st.alive[k] and k not in st.taken and not (st.don[k] & 3)),
        key=lambda k: (st.don[k], st.stream[k][0]),
    )
    for k in order:
        if S >= tgt_lo:
            break
        if S + st.don[k] <= tgt_hi:
            st.consume(k, lq)
            used.append(k)
            S += st.don[k]
    if S < tgt_lo:
        for k in reversed(used):
            e = st.stream[k]
            for rs in e[3]:
                for o in range(e[1]):
                    st.alt[rs + o] = None
            st.alive[k] = True
            del st.consumed_by[k]
        return None
    kv = m[lq] - S
    assert 1 <= kv <= W
    st.keep[lq] = kv
    return kv


def _melted_runs(melted):
    """Maximal runs of consecutive melted rows, in ascending order."""
    runs = []
    for q in sorted(melted):
        if q - 1 in melted:
            continue
        run = [q]
        j = q + 1
        while j in melted:
            run.append(j)
            j += 1
        runs.append(run)
    return runs


def _pay_shifts(supply, quota, sh):
    """Pay a run's class shifts atomically: singles, then quota, then pairs.

    Returns the remaining (supply, quota), or None when the run is
    unaffordable.
    """
    s = Counter(supply)
    for r in (1, 2, 3):
        need = sh.get(r, 0)
        take = min(s[r], need)
        s[r] -= take
        need -= take
        tq = min(quota, need)
        quota -= tq
        need -= tq
        for _ in range(need):
            for r1 in (1, 2, 3):
                r2 = (r - r1) & 3
                if r2 == 0 or r1 == r or r2 == r:
                    continue
                if r1 == r2 and s[r1] >= 2:
                    s[r1] -= 2
                    break
                if r1 != r2 and s[r1] and s[r2]:
                    s[r1] -= 1
                    s[r2] -= 1
                    break
            else:
                return None
    return s, quota


def _plan_targets(m, dry, pull_budget):
    """Choose each melted row's target keep-word class.

    Returns (targets, com_rows): the target class per melted row, and
    the commit rank of each row in a committed (class-aligned) run.
    """
    melted0 = set(dry.melted)
    cand = []
    for run in _melted_runs(melted0):
        cls = [m[i] & 3 for i in run]
        flips = sum(1 for a, b in zip(cls, cls[1:]) if a != b)
        cn = Counter(cls)
        maj = max(cn.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        sh = Counter((m[i] - maj) & 3 for i in run if (m[i] & 3) != maj)
        nsh = sum(sh.values())
        if flips == 0 or nsh == 0:
            continue
        cand.append((flips / nsh, -flips, run[0], run, maj, sh))
    # total value first (the big runs carry the walk savings), then
    # flips-per-shift for the tail
    cand.sort(key=lambda t: (t[1], -t[0], t[2]))

    supply = Counter((T - dry.keep[q]) & 3 for q in dry.fill_fail)
    supply.pop(0, None)
    quota = pull_budget
    committed = []
    for _ratio, _nf, _start, run, maj, sh in cand:
        paid = _pay_shifts(supply, quota, sh)
        if paid is not None:
            supply, quota = paid
            committed.append((run, maj))

    targets = {}
    com_rows = {}
    for rank, (run, maj) in enumerate(committed):
        for i in run:
            targets[i] = maj
            com_rows[i] = rank
    for q in melted0:
        if q not in targets:
            targets[q] = m[q] & 3
    return targets, com_rows


def _pop_pair(bank, r1, i1, r2, i2):
    """Remove two banked donations; pops the higher index first in one bucket."""
    if r1 == r2:
        bank[r1].pop(max(i1, i2))
        bank[r1].pop(min(i1, i2))
    else:
        bank[r1].pop(i1)
        bank[r2].pop(i2)


def _pair_combos(bank, delta):
    """Banked donation pairs (r1, i1, r2, i2) whose residues sum to delta."""
    combos = []
    for r1 in (1, 2, 3):
        r2 = (delta - r1) & 3
        if r2 == 0 or r2 < r1:
            continue
        if r1 == r2:
            for j in range(min(4, len(bank[r1]) - 1)):
                combos.append((r1, j, r1, j + 1))
        else:
            for j in range(min(3, len(bank[r1]))):
                for j2 in range(min(3, len(bank[r2]))):
                    combos.append((r1, j, r2, j2))
    return combos


def melt_align(m, dup_groups, pull_budget, tries):  # noqa: C901, PLR0915 - execution closures share mutable state
    smalls, stream = _build_stream(m, dup_groups)
    dry, dry_fills, fail_close = dry_pass_recorded(m, smalls, stream)
    dead = {k for k in range(dry.NE) if dry.consumed_by.get(k) == dry.x1}
    dead_fine = sorted(
        (k for k in dead if dry.don[k] <= MELT_COARSE), key=lambda k: (dry.don[k], k)
    )
    fail_set = set(dry.fill_fail)
    dry_keep = list(dry.keep)
    targets, com_rows = _plan_targets(m, dry, pull_budget)

    # ---- execution ------------------------------------------------------
    st = _State(m, smalls, stream)
    st.route_asc = True
    bank = {0: [], 1: [], 2: [], 3: []}

    for q in dry.fill_fail:
        for k in fail_close[q]:
            assert st.alive[k], "failure-close replay collision"
            st.consume(k, q)
        st.keep[q] = dry_keep[q]
        d = T - dry_keep[q]
        bank[d & 3].append((q, d))

    pb_asc = {
        r: sorted((q for q in st.phase_b if ((-m[q]) & 3) == r), key=lambda q: (m[q], q))
        for r in (1, 2, 3)
    }
    pb_desc = {r: list(reversed(pb_asc[r])) for r in (1, 2, 3)}
    pulls_left = [pull_budget]
    freed = set()

    def lazy_pull(delta):
        if pulls_left[0] <= 0:
            return False
        # tier 1: a freed coarse tail closes a high-m member's window
        for q in pb_desc[delta]:
            if q not in st.open_b:
                continue
            lo, hi = m[q] - W - st.credit[q], m[q] - 1 - st.credit[q]
            # the largest fitting donation, lowest entry on ties
            best_e = max(
                (e for e in freed if st.alive[e] and lo <= st.don[e] <= hi),
                key=lambda e: (st.don[e], -e),
                default=None,
            )
            if best_e is None:
                continue
            freed.discard(best_e)
            st.consume(best_e, q)
            kv = m[q] - st.credit[q] - st.don[best_e]
            assert 1 <= kv <= W
            st.keep[q] = kv
            st.open_b.discard(q)
            d = T - kv
            pulls_left[0] -= 1
            bank[d & 3].append((q, d))
            return (d & 3) == delta
        # tier 2: dead fines close a low-m member
        for q in pb_asc[delta]:
            if q not in st.open_b:
                continue
            kv = window_close_dead_fine(st, q, dead_fine)
            if kv is None:
                continue
            st.open_b.discard(q)
            d = T - kv
            pulls_left[0] -= 1
            bank[d & 3].append((q, d))
            return (d & 3) == delta
        return False

    def fill(V, pref, floor=MELT_COARSE):
        if V < 4:
            return None
        return exact_fill_pref(st, V, floor, pref)

    def commit(lq, seq, kv, donors=(), pref=()):
        for k in seq:
            st.consume(k, lq)
            st.taken.discard(k)
        used = set(seq)
        for k in pref:
            if k not in used and st.alive[k]:
                freed.add(k)
        st.keep[lq] = kv
        st.alt[lq] = st.x1
        st.melted.append(lq)
        for dq, _dv in donors:
            st.alt[dq] = lq

    def try_shift(lq, delta, kv_t, pref):
        tried = 0
        i = 0
        pulled = False
        while tried < tries:
            if i >= len(bank[delta]):
                if not pulled:
                    pulled = True
                    if lazy_pull(delta):
                        continue
                break
            dq, d = bank[delta][i]
            seq = fill(m[lq] - d - kv_t, pref)
            tried += 1
            if seq is not None:
                bank[delta].pop(i)
                return seq, [(dq, d)]
            i += 1
        for r1, i1, r2, i2 in _pair_combos(bank, delta):
            if tried >= tries + 8:
                break
            d1 = bank[r1][i1]
            d2 = bank[r2][i2]
            seq = fill(m[lq] - d1[1] - d2[1] - kv_t, pref)
            tried += 1
            if seq is not None:
                _pop_pair(bank, r1, i1, r2, i2)
                return seq, [d1, d2]
        return None

    # stage 1: own-class rows, pinned order
    stage2 = []
    for lq in st.phase_a:
        if lq in fail_set:
            continue
        c_t = targets[lq]
        if ((m[lq] - c_t) & 3) == 0:
            kv_t = KEEP_BASE + c_t
            seq = fill(m[lq] - kv_t, dry_fills[lq])
            assert seq is not None, f"stage-1 replay failed for {lq}"
            commit(lq, seq, kv_t)
        else:
            stage2.append(lq)

    # stage 2: committed-run shifted rows, rank-major, m-desc within.
    stage2.sort(key=lambda q: (com_rows.get(q, 1 << 30), -m[q], q))
    for lq in stage2:
        pref = dry_fills[lq]
        c_t = targets[lq]
        delta = (m[lq] - c_t) & 3
        kv_t = KEEP_BASE + c_t
        got = try_shift(lq, delta, kv_t, pref)
        if got is not None:
            seq, donors = got
            commit(lq, seq, kv_t, donors, pref)
            continue
        seq = fill(m[lq] - kv_t, pref)
        if seq is not None:
            commit(lq, seq, kv_t, (), pref)
            continue
        # fine-pool exact fill AT THE TARGET CLASS (whole-entry, any
        # size): still a zero-boundary melt onto the shared word
        seq = fill(m[lq] - kv_t, pref, 4)
        if seq is not None:
            commit(lq, seq, kv_t, (), pref)
            continue
        c_o = m[lq] & 3
        seq = fill(m[lq] - (KEEP_BASE + c_o), pref)
        if seq is not None:
            commit(lq, seq, KEEP_BASE + c_o, (), pref)
            continue
        kv = st.window_close(lq)
        bank[(T - kv) & 3].append((lq, T - kv))
        st.fill_fail.append(lq)

    for r in (0, 1, 2, 3):
        for row, d in bank[r]:
            st.alt[row] = st.route(d, [])

    _phase_b_pass(st)
    keep, alt = st.finish()
    return keep, alt
