import Dqi.Basic

/-!
# Syndrome pullback at strict distance

Support conditions for pulling the objective back through the syndrome map.  The
objective connects a pattern `y` to `y' = y + Pi.single i a` — one constraint
touched, by one field element.  On the syndrome register this connection is
visible only as a syndrome relation, so a priori a dual codeword could alias a
different pair of patterns onto the same relation.

The results here say that in the strict-distance regime this cannot happen: if
`2 * l + 1 < dualDist B` (resp. `2 * l + 2 < dualDist B` when one endpoint sits
one Hamming step above the decodable set), any dual-codeword witness connecting
`y` to `y'` through coordinate `i` forces `y' = y + Pi.single i a` exactly — no
dual codeword is short enough to create an alias, so the only nonzero matrix
elements of the syndrome-basis operator are the obvious ones, and it agrees with
the pattern-space operator.
-/

namespace Dqi

variable {K : Type*} [Field K] [DecidableEq K]
variable {m n : ℕ}

/-- A single-coordinate pattern has Hamming weight at most one. -/
theorem hammingNorm_single_le (i : Fin m) (a : K) :
    hammingNorm (Pi.single i a : Pattern m K) ≤ 1 := by
  refine Finset.card_le_one.mpr fun j hj k hk => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, ne_eq,
    Pi.single_apply] at hj hk
  have hj' : j = i := by
    by_contra h
    rw [if_neg h] at hj
    exact hj rfl
  have hk' : k = i := by
    by_contra h
    rw [if_neg h] at hk
    exact hk rfl
  rw [hj', hk']

/-- Subadditivity of the Hamming weight for the three-term difference. -/
theorem hammingNorm_sub_sub_le (y z : Pattern m K) (i : Fin m) (a : K) :
    hammingNorm (y - z - Pi.single i a) ≤ hammingNorm y + hammingNorm z + 1 := by
  have h1 := hammingNorm_sub_le (y - z) (Pi.single i a)
  have h2 := hammingNorm_sub_le y z
  have h3 := hammingNorm_single_le i a
  omega

/-- **The strict-distance pullback.**  If both endpoints are decodable and
`2 * l + 1 < dualDist B`, the only dual codeword of the form
`y' - y - Pi.single i a` is zero: the syndrome relation determines `y'`. -/
theorem eq_add_single_of_dual_of_decodable {l : ℕ}
    {B : Matrix (Fin m) (Fin n) K} (hstrict : 2 * l + 1 < dualDist B)
    {y y' : Pattern m K} (hy : Decodable l y) (hy' : Decodable l y')
    (i : Fin m) (a : K)
    (hdual : IsDualCodeword B (y' - y - Pi.single i a)) :
    y' = y + Pi.single i a := by
  have hw := hammingNorm_sub_sub_le y' y i a
  have hzero : y' - y - Pi.single i a = 0 := by
    by_contra hne
    have hlow := dualDist_le_hammingNorm hne hdual
    have hyl : hammingNorm y ≤ l := hy
    have hy'l : hammingNorm y' ≤ l := hy'
    omega
  rw [sub_sub] at hzero
  exact sub_eq_zero.mp hzero

/-- The next-layer pullback: the target `y'` may sit one Hamming step above the
decodable set, at the price of one more unit of distance. -/
theorem eq_add_single_of_dual_of_succ {l : ℕ}
    {B : Matrix (Fin m) (Fin n) K} (hstrict : 2 * l + 2 < dualDist B)
    {y y' : Pattern m K} (hy : Decodable l y) (hy' : hammingNorm y' ≤ l + 1)
    (i : Fin m) (a : K)
    (hdual : IsDualCodeword B (y' - y - Pi.single i a)) :
    y' = y + Pi.single i a := by
  have hw := hammingNorm_sub_sub_le y' y i a
  have hzero : y' - y - Pi.single i a = 0 := by
    by_contra hne
    have hlow := dualDist_le_hammingNorm hne hdual
    have hyl : hammingNorm y ≤ l := hy
    omega
  rw [sub_sub] at hzero
  exact sub_eq_zero.mp hzero

/-- No aliasing in the next layer: at `2 * l + 2 < dualDist B`, even the
weight-`(l + 1)` patterns have pairwise distinct syndromes. -/
theorem eq_of_syndrome_eq_succ {l : ℕ} {B : Matrix (Fin m) (Fin n) K}
    (hd : 2 * l + 2 < dualDist B) {y y' : Pattern m K}
    (hy : hammingNorm y ≤ l + 1) (hy' : hammingNorm y' ≤ l + 1)
    (h : syndrome B y = syndrome B y') : y = y' := by
  by_contra hne
  have hsub : y - y' ≠ 0 := sub_ne_zero_of_ne hne
  have hlow := dualDist_le_hammingNorm hsub (isDualCodeword_sub_of_syndrome_eq h)
  have htri := hammingNorm_sub_le y y'
  omega

end Dqi
