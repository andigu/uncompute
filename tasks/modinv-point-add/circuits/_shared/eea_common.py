"""Shared classical EEA models for the Luo-family circuits (verification only).

This module holds the two models that the Luo-family code verifies
against: ``eea_reference`` (the value-level classical EEA, the Alg-1
endpoint) and ``EEABitState`` (the bit-exact model of the register
state one Algorithm-3 step manipulates).  ``euclidean/gen_constants.py``
validates its fast walk model step-exactly against ``EEABitState``.
Callers load this module by path and bind their own schedule surface
onto it.  "The paper" below means Luo et al. (2026), arXiv:2607.13816.
No circuit.py file imports it, so nothing here touches an
emission path or a pinned cost.

The class is parametric where the circuits deliberately diverge:

* schedule — ``active_windows(n, T)``, ``nmax_steps(n)`` and
  ``ls_width(n)`` are injected callables (each caller passes its own,
  typically its circuit's module globals), so window widths and the
  step budget are the caller's;
* frame — ``virtual_rotation=False`` is the direct frame: rotations of
  Work2 are physical, gated on the termination-freeze literal
  [lrp != 0], and the t-block reads Work2 in place and applies its own
  post-shift.  ``virtual_rotation=True`` is the virtualized-rotation
  frame: one UNCONDITIONAL rot_left at the step top (dead lanes
  over-rotate; ``finish_checks`` un-rotates by the residual ls) and
  the t-block reads Work2 through the (j-2) mod W rotated view;
* D6 right edge — ``newK(n, T, k)``, when supplied, truncates the
  new-length scan's full-width right edge K == n+3 (the circuit scans
  no further); None scans to the physical edge.  The old-edge window
  soundness assert (``len_rp-K-old``) presumes worst-case right edges
  and is asserted exactly when ``newK`` is None.

Every other line is common to both circuits, including the per-step
window assertions (``check_windows``), the termination-freeze
invariants, and the endpoint checks against ``eea_reference``.
"""

import math


def eea_reference(p, x):
    """Return the expected Alg-1 endpoint for input x: (x_used, iter_start, exact_steps,
    tprime, iter_final, inverse).  tprime = t_{k-1} of the classical EEA run
    on (p, x_used); Iter accumulates one flip per Euclidean iteration on top
    of the initial [x > p/2] flip; inverse = tprime if Iter_final else p-tprime.
    """
    assert 1 <= x < p and math.gcd(x, p) == 1
    iter_bit = 0
    x_used = x
    if x > p // 2:
        iter_bit = 1
        x_used = p - x
    r_prev, r = p, x_used
    t_prev, t = 0, 1
    steps = 0
    while r != 0:
        q = r_prev // r
        steps += 4 * q.bit_length()
        r_prev, r = r, r_prev - q * r
        t_prev, t = t, t_prev + q * t
        iter_bit ^= 1
    assert r_prev == 1, "x not invertible"
    inv = t_prev % p if iter_bit else (p - t_prev) % p
    assert inv == pow(x, -1, p)
    return {
        "x_used": x_used,
        "iter_start": 1 if x > p // 2 else 0,
        "exact_steps": steps,
        "tprime": t_prev % p,
        "iter_final": iter_bit,
        "inverse": inv,
    }


class EEABitState:
    """Bit-exact model of the register state the circuit manipulates.

    Work1/Work2 are (n+3)-bit lists indexed 1..n+3 by paper convention
    (list index j-1).  Length registers hold plain values.  step(T)
    transliterates Algorithm 3 with the shared conventions:
      * merged swap position J = lt+lq+1+[Phase1=0]  (trace rows 3/5);
      * termination freeze: [lrp != 0] gates the rotations (direct
        frame) and the r-side arithmetic, while the ls counter keeps
        running — exactly the paper's value semantics (Table 4 rows
        33-36 show ls counting while everything else is inert).  The
        residual ls != 0 keeps the phase toggle and the 4|T swap block
        silent; that silence is the ls-monotonicity invariant,
        asserted below ("swap block refire in residual regime");
      * D-K3: the t-op right endpoint is phase-dependent.  Phase 3
        (P2=0): R = lt+1 (t'(h) provably fits; the addend is t).
        Phase 4 (P2=1): R = n+3-lrp-ls, the full t'(h) region — the
        comparison must see all of t'(h); Work1 positions above lt+1
        are the r-region's zero padding there, so the same cells act
        as pure carry propagators;
      * arithmetic acts on the physical windows exactly as the
        location-controlled blocks do, with per-step active-window
        assertions;
      * each step records the realized block endpoints in ``edges``.

    Frame and D6-edge parametrization: the class docstring in
    eea_common (module header) — the constructor keywords select them.
    """

    def __init__(
        self, p, x, *, active_windows, nmax_steps, ls_width, newK=None, virtual_rotation=False
    ):
        self.active_windows = active_windows
        self.nmax_steps = nmax_steps
        self.ls_width = ls_width
        self.newK = newK
        self.virtual = virtual_rotation
        self.p = p
        self.n = n = p.bit_length()
        ref = eea_reference(p, x)
        self.ref = ref
        x_used = ref["x_used"]
        W = n + 3
        self.w1 = [0] * W
        self.w2 = [0] * W
        # Work1 = |t=1 (little-endian, lt+1 slots), 0.., r=p (big-endian)>
        self.w1[0] = 1  # t=1 -> "10" at u_1,u_2
        for i in range(n):  # p big-endian at u_4..u_{n+3}
            self.w1[3 + i] = (p >> (n - 1 - i)) & 1
        # Work2 = |0.., x_used (big-endian at v_4..v_{n+3})>
        for i in range(n):
            self.w2[3 + i] = (x_used >> (n - 1 - i)) & 1
        self.lt, self.lq, self.ls = 1, 0, 0
        self.lrp = x_used.bit_length()
        self.p1 = self.p2 = self.sign = 0
        self.iter = ref["iter_start"]
        self.T = 0

    # -- register views -----------------------------------------------------
    def _win_val_be(self, reg, L, R):
        """Big-endian value of reg positions L..R (1-based, inclusive)."""
        v = 0
        for j in range(L, R + 1):
            v = (v << 1) | reg[j - 1]
        return v

    def _win_set_be(self, reg, L, R, v):
        for idx, j in enumerate(range(R, L - 1, -1)):
            reg[j - 1] = (v >> idx) & 1

    def _win_val_le(self, reg, L, R):
        v = 0
        for idx, j in enumerate(range(L, R + 1)):
            v |= reg[j - 1] << idx
        return v

    def _win_set_le(self, reg, L, R, v):
        for idx, j in enumerate(range(L, R + 1)):
            reg[j - 1] = (v >> idx) & 1

    def rot_left(self, k=1):
        """Circular left shift of Work2: v_i <- v_{i+1} (wrap v_last <- v_1)."""
        for _ in range(k):
            self.w2 = self.w2[1:] + self.w2[:1]

    def rot_right(self, k=1):
        for _ in range(k):
            self.w2 = self.w2[-1:] + self.w2[:-1]

    # -- the t-block's view of Work2: in place (direct frame) or through
    # the (j-2) mod W rotated view (virtualized frame)
    def _t_val(self, L, R):
        if not self.virtual:
            return self._win_val_le(self.w2, L, R)
        W = self.n + 3
        v = 0
        for idx, j in enumerate(range(L, R + 1)):
            v |= self.w2[(j - 2) % W] << idx
        return v

    def _t_set(self, L, R, v):
        if not self.virtual:
            self._win_set_le(self.w2, L, R, v)
            return
        W = self.n + 3
        for idx, j in enumerate(range(L, R + 1)):
            self.w2[(j - 2) % W] = (v >> idx) & 1

    # -- one Algorithm-3 step -------------------------------------------------
    def step(self, T, check_windows=True):  # noqa: C901 - one Algorithm-3 step, kept whole
        n, W = self.n, self.n + 3
        self.T = T
        self.edges = {"r": None, "t": None}  # realized block endpoints
        win = self.active_windows(n, T)
        alive = self.lrp != 0  # termination-freeze literal

        # ---- pre-shift ----
        if self.virtual:
            self.rot_left()  # UNCONDITIONAL (virtualized rotation)
        if self.p1 == 0:
            if not self.virtual and alive:
                self.rot_left()
            self.ls += 1
            if self.p2 == 1:
                if alive:
                    self.rot_right(2)
                self.ls -= 2
        assert self.ls >= 0

        # ---- arithmetic block 1: location-controlled ops on r's ----
        if alive and self.p1 == 0:
            L = self.lt + self.lq + 2
            R = W - self.ls
            self.edges["r"] = (L, R)
            k1, K1 = win["r"]
            if check_windows:
                assert k1 <= L, (T, "k1", k1, L)
                assert L <= R + 1, (T, "r-window", L, R)
                assert R <= K1
            # (Sign, r) -= 2^ls r'  == windowed big-endian subtract
            if L <= R:
                u = self._win_val_be(self.w1, L, R)
                v = self._win_val_be(self.w2, L, R)
                width = R - L + 1
                diff = (u - v) % (1 << width)
                borrow = 1 if u < v else 0
                self._win_set_be(self.w1, L, R, diff)
                self.sign ^= borrow
            if self.p2 == 1:
                self.sign ^= 1
            if (self.p2 == 0 or self.sign == 0) and L <= R:
                u = self._win_val_be(self.w1, L, R)
                v = self._win_val_be(self.w2, L, R)
                width = R - L + 1
                self._win_set_be(self.w1, L, R, (u + v) % (1 << width))

        # ---- arithmetic block 2: location-controlled swap ----
        if (self.p1 ^ self.p2) == 1:
            J = self.lt + self.lq + 1 + (1 if self.p1 == 0 else 0)
            k2, K2 = win["swap"]
            if check_windows:
                assert k2 <= J <= K2, (T, "swap", k2, J, K2)
            self.sign, self.w1[J - 1] = self.w1[J - 1], self.sign
            if self.p1 == 1:
                self.lq -= 1
            else:
                self.lq += 1
            assert self.lq >= 0

        # ---- arithmetic block 3: location-controlled ops on t's ----
        # (D-K3 phase-dependent right endpoint; class docstring)
        if self.p1 == 1:
            _k3, K3 = win["t"]
            L = 1
            if self.p2 == 0:
                R = self.lt + 1
            else:
                R = W - self.lrp - self.ls
                for j in range(self.lt + 2, R + 1):
                    assert self.w1[j - 1] == 0, (T, "t addend pad", j)
            self.edges["t"] = R
            if check_windows:
                assert R <= K3, (T, "t-window", R, K3, self.p2)
            width = R - L + 1
            if self.p2 == 1 or self.sign == 0:
                tp = self._t_val(L, R)
                tt = self._win_val_le(self.w1, L, R)
                self._t_set(L, R, (tp - tt) % (1 << width))
            self.sign ^= 1
            tp = self._t_val(L, R)
            tt = self._win_val_le(self.w1, L, R)
            s = tp + tt
            self.sign ^= (s >> width) & 1
            self._t_set(L, R, s % (1 << width))
            # ---- post-shift (in the virtualized frame it is the top one)
            if not self.virtual:
                self.rot_left()
            self.ls += 1
            if self.p2 == 1:
                self.rot_right(2)
                self.ls -= 2
        assert self.ls >= 0

        # ---- phase update (residual ls != 0 keeps it inert) ----
        if self.lq == 0 and self.lrp > 0:
            self.p2 ^= self.sign ^ self.p1
            self.sign ^= self.p2
        if self.ls == 0:
            self.p1 ^= 1
            self.p2 ^= 1

        # ---- end-of-iteration swap + length updates (4|T only) ----
        if T % 4 == 0 and self.lq == 0 and self.ls == 0:
            assert alive, "swap block refire in residual regime"
            self.w1, self.w2 = self.w2, self.w1
            k4, K4 = win["len_t"]
            k5, K5 = win["len_rp"]
            # lt update: XOR out old (highest nonzero of Work2's valid left
            # region), XOR in new (same scan on Work1).  Valid j <= B.
            B = W - self.lrp
            old = self._highest_nonzero(self.w2, k4, K4, B)
            new = self._highest_nonzero(self.w1, k4, K4, B)
            if check_windows:
                assert k4 <= self.lt <= B <= K4, (T, "len_t", k4, self.lt, B, K4)
            assert old == self.lt, (T, old, self.lt)
            self.lt ^= old ^ new
            assert self.lt == new
            # lrp update: XOR out old, XOR in new right length.  Right
            # length = n+4-m for the lowest valid nonzero position m
            # (valid j >= A).  The old scan is bounded by K5; the new
            # scan runs to the full right edge n+3 (D-K5), truncated by
            # ``newK`` when one is supplied.
            A = self.lt + 2
            old_r = self._right_length(self.w1, k5, K5, A)
            new_r = self._right_length(self.w2, k5, W, A)
            if check_windows:
                assert k5 <= A, (T, "len_rp-A", k5, A)
                if self.newK is None:
                    assert self.lrp == 0 or W + 1 - self.lrp <= K5 + 1, (
                        T,
                        "len_rp-K-old",
                        self.lrp,
                        K5,
                    )
            assert old_r == self.lrp, (T, old_r, self.lrp)
            self.lrp ^= old_r ^ new_r
            assert self.lrp == new_r
            self.iter ^= 1

    def _highest_nonzero(self, reg, k, K, B):
        """Highest j in [k, min(K,B)] with reg bit j set; 0 if none."""
        for j in range(min(K, B), k - 1, -1):
            if reg[j - 1]:
                return j
        return 0

    def _right_length(self, reg, k, K, A):
        """n+4-m for the lowest j=m in [max(k,A), K] with bit set; 0 if
        none.  With a ``newK`` schedule, a full-width right edge
        K == n+3 is truncated to newK(n, T, k) (the circuit scans no
        further).
        """
        if self.newK is not None and K == self.n + 3:
            K = self.newK(self.n, self.T, k)
        for j in range(max(k, A), K + 1):
            if reg[j - 1]:
                return self.n + 4 - j
        return 0

    # -- full runs -----------------------------------------------------------
    def run(self, check_windows=True):
        for T in range(1, self.nmax_steps(self.n) + 1):
            self.step(T, check_windows=check_windows)
        return self

    def finish_checks(self):
        """Assert the Alg-1 endpoint: t' little-endian across Work2, Work1 =
        (t=p, r=1) constants, lengths/phases at their constant values.  In
        the virtualized frame, un-rotates Work2 by the residual ls first.
        """
        if self.virtual:
            k_rot = self.ls % (self.n + 3)
            if k_rot:
                self.w2 = self.w2[-k_rot:] + self.w2[:-k_rot]
        n, W, p = self.n, self.n + 3, self.p
        ref = self.ref
        # Work2 = t' little-endian over the whole register (top 3 bits 0)
        tprime = self._win_val_le(self.w2, 1, W)
        if self.virtual:
            assert tprime < (1 << n), tprime
            assert tprime % p == ref["tprime"], (tprime, ref["tprime"])
        else:
            # Continuant range: t' = t_{k-1} = K(q_1..q_{k-1}) < K(q_1..q_k)
            # = p, and t' >= 1, so a p - t' negation never wraps.  Asserted
            # strictly -- no % p hedge.
            assert 0 < tprime < p, tprime
            assert tprime == ref["tprime"], (tprime, ref["tprime"])
        assert self.iter == ref["iter_final"]
        # Work1 = t=p little-endian at u_1..u_{n+1} (appended 0), r=1 at u_{n+3}
        expect_w1 = [0] * W
        for i in range(n):
            expect_w1[i] = (p >> i) & 1
        expect_w1[W - 1] = 1
        assert self.w1 == expect_w1, "Work1 terminal constants"
        # ls holds the residual step count (input-dependent, part of
        # Gamma(x) along with Iter); everything else is constant.
        assert (self.lt, self.lq, self.lrp) == (n, 0, 0)
        assert self.ls == self.T - ref["exact_steps"], (self.ls, self.T)
        assert self.ls < (1 << self.ls_width(n)), "ls register overflow"
        assert (self.p1, self.p2, self.sign) == (0, 0, 0)
        inv = tprime % p if self.iter else (p - tprime) % p
        assert inv == ref["inverse"]
        return tprime
