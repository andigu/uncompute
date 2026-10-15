import Dqi.Basic
import Dqi.Packing
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Counting ingredients for the boundary correction

At the boundary decoding regime `dualDist B = 2 * l + 1`, the syndrome pullback
acquires a defect: a decodable weight-`l` pattern can alias with weight-`(l+1)`
patterns through the minimum-weight dual codewords compatible with it (decodable
patterns still never alias with each other — that needs only `2 * l < dualDist`),
and the resulting correction has norm `R_C`.  This file proves
the two counting facts that bound `R_C`; it does not construct the defect
operator itself.

* `card_compatible_mul_le`: at most `(m - l).choose 2 / (l + 1).choose 2`
  minimum-weight dual codewords are compatible with a fixed weight-`l` output
  pattern, by a tail-packing argument over `card_mul_choose_le_of_pairwise_inter_le`.
* `sum_sq_le_mul` and `norm_le_of_count_bound`: the abstract shape of the
  boundary-residual bound — if each output coordinate carries at most `cnt`
  terms of magnitude `Kc`, no more than `Npack` of them anywhere and `tot` in
  total, the norm is at most `Kc * √(Npack * tot)`.
-/

namespace Dqi

open Finset

/-- **Compatible minimum-weight codewords are few.**  Fix a weight-`l` pattern
`y'`.  A minimum-weight dual codeword compatible with `y'` (agreeing with `-y'`
on `supp y'`) contains `supp y'` and has a tail of size `l + 1` in the `m - l`
coordinates outside; distinct compatible codewords have distinct tails meeting
in at most one coordinate, so the packing bound applies with `s = l + 1`,
`t = 1`, `n = m - l`. -/
theorem card_compatible_mul_le
    {K : Type*} [Field K] [DecidableEq K] {m n l : ℕ}
    {B : Matrix (Fin m) (Fin n) K}
    (hbd : dualDist B = 2 * l + 1)
    (y' : Pattern m K) (hy' : hammingNorm y' = l)
    (Cs : Finset (Pattern m K))
    (hdual : ∀ c ∈ Cs, IsDualCodeword B c)
    (hwt : ∀ c ∈ Cs, hammingNorm c = 2 * l + 1)
    (hcompat : ∀ c ∈ Cs, ∀ i, y' i ≠ 0 → c i = -(y' i)) :
    Cs.card * (l + 1).choose 2 ≤ (m - l).choose 2 := by
  have hnorm : ∀ x : Pattern m K,
      hammingNorm x = (Finset.univ.filter fun i => x i ≠ 0).card := fun _ => rfl
  set S : Finset (Fin m) := Finset.univ.filter fun i => y' i ≠ 0 with hSdef
  have hScard : S.card = l := (hnorm y').symm.trans hy'
  set tl : Pattern m K → Finset (Fin m) :=
    fun c => (Finset.univ.filter fun i => c i ≠ 0) \ S with htldef
  -- Every compatible codeword contains `supp y'`.
  have hSsub : ∀ c ∈ Cs, S ⊆ Finset.univ.filter fun i => c i ≠ 0 := by
    intro c hc i hi
    have hyi : y' i ≠ 0 := (Finset.mem_filter.mp hi).2
    rw [Finset.mem_filter]
    exact ⟨Finset.mem_univ _, by rw [hcompat c hc i hyi]; exact neg_ne_zero.mpr hyi⟩
  -- Tails have size `l + 1`.
  have htl_card : ∀ c ∈ Cs, (tl c).card = l + 1 := by
    intro c hc
    rw [htldef]
    rw [Finset.card_sdiff_of_subset (hSsub c hc), ← hnorm c, hwt c hc, hScard]
    omega
  -- The difference of two compatible codewords is supported on the tails.
  have hdiff_supp : ∀ c ∈ Cs, ∀ c' ∈ Cs,
      (Finset.univ.filter fun i => (c - c') i ≠ 0) ⊆ tl c ∪ tl c' := by
    intro c hc c' hc' i hi
    have hne : c i ≠ c' i := by
      simpa [sub_ne_zero] using (Finset.mem_filter.mp hi).2
    have hiS : i ∉ S := fun hiS =>
      hne ((hcompat c hc i (Finset.mem_filter.mp hiS).2).trans
        (hcompat c' hc' i (Finset.mem_filter.mp hiS).2).symm)
    rcases ne_or_eq (c i) 0 with h0 | h0
    · exact Finset.mem_union_left _ (Finset.mem_sdiff.mpr
        ⟨Finset.mem_filter.mpr ⟨Finset.mem_univ _, h0⟩, hiS⟩)
    · have h0' : c' i ≠ 0 := fun h => hne (h0.trans h.symm)
      exact Finset.mem_union_right _ (Finset.mem_sdiff.mpr
        ⟨Finset.mem_filter.mpr ⟨Finset.mem_univ _, h0'⟩, hiS⟩)
  -- The difference of distinct compatible codewords has weight at least `2l + 1`.
  have hlow : ∀ c ∈ Cs, ∀ c' ∈ Cs, c ≠ c' → 2 * l + 1 ≤ hammingNorm (c - c') := by
    intro c hc c' hc' hne
    calc 2 * l + 1 = dualDist B := hbd.symm
      _ ≤ hammingNorm (c - c') :=
          dualDist_le_hammingNorm (sub_ne_zero_of_ne hne)
            (isDualCodeword_sub_of_syndrome_eq ((hdual c hc).trans (hdual c' hc').symm))
  -- Distinct tails meet in at most one coordinate.
  have hinter : ∀ c ∈ Cs, ∀ c' ∈ Cs, c ≠ c' → (tl c ∩ tl c').card ≤ 1 := by
    intro c hc c' hc' hne
    have h1 : 2 * l + 1 ≤ (tl c ∪ tl c').card := by
      calc 2 * l + 1 ≤ hammingNorm (c - c') := hlow c hc c' hc' hne
        _ = (Finset.univ.filter fun i => (c - c') i ≠ 0).card := hnorm _
        _ ≤ (tl c ∪ tl c').card := Finset.card_le_card (hdiff_supp c hc c' hc')
    have h2 := Finset.card_union_add_card_inter (tl c) (tl c')
    rw [htl_card c hc, htl_card c' hc'] at h2
    omega
  rcases Nat.eq_zero_or_pos l with hl0 | hlpos
  · subst hl0
    simp
  -- Distinct compatible codewords have distinct tails.
  have hinj : Set.InjOn tl Cs := by
    intro c hc c' hc' htleq
    by_contra hne
    have hc'' := Finset.mem_coe.mp hc
    have hc''' := Finset.mem_coe.mp hc'
    have h1 : 2 * l + 1 ≤ hammingNorm (c - c') := hlow c hc'' c' hc''' hne
    have h2 : hammingNorm (c - c') ≤ (tl c).card := by
      rw [hnorm]
      refine Finset.card_le_card ?_
      have h := hdiff_supp c hc'' c' hc'''
      rwa [← htleq, Finset.union_self] at h
    rw [htl_card c hc''] at h2
    omega
  -- Pack the tails outside `supp y'`.
  have hG : (Sᶜ : Finset (Fin m)).card = m - l := by
    rw [Finset.card_compl, hScard, Fintype.card_fin]
  have key : (Cs.image tl).card * (l + 1).choose (1 + 1) ≤ (m - l).choose (1 + 1) := by
    refine card_mul_choose_le_of_pairwise_inter_le (G := Sᶜ) hG ?_ ?_ ?_
    · intro A hA
      obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hA
      intro i hi
      rw [Finset.mem_compl]
      exact (Finset.mem_sdiff.mp hi).2
    · intro A hA
      obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hA
      exact htl_card c hc
    · intro A hA A' hA' hAB
      obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hA
      obtain ⟨c', hc', rfl⟩ := Finset.mem_image.mp hA'
      exact hinter c hc c' hc' fun h => hAB (by rw [h])
  rw [Finset.card_image_of_injOn hinj] at key
  exact key

/-- If each count is at most `Npack` and the counts sum to `tot`, the sum of
squares is at most `Npack * tot`. -/
theorem sum_sq_le_mul {α : Type*} (Y : Finset α) (cnt : α → ℕ) (Npack tot : ℕ)
    (hn : ∀ y ∈ Y, cnt y ≤ Npack) (htot : ∑ y ∈ Y, cnt y = tot) :
    ∑ y ∈ Y, cnt y ^ 2 ≤ Npack * tot := by
  calc ∑ y ∈ Y, cnt y ^ 2 ≤ ∑ y ∈ Y, Npack * cnt y :=
        Finset.sum_le_sum fun y hy => by rw [sq]; exact Nat.mul_le_mul (hn y hy) le_rfl
    _ = Npack * tot := by rw [← Finset.mul_sum, htot]

/-- **Abstract boundary-residual bound.**  A vector whose `i`-th coordinate is a
sum of at most `cnt i` terms of magnitude `Kc`, with each count at most `Npack`
and `tot` terms in total, has norm at most `Kc * √(Npack * tot)`. -/
theorem norm_le_of_count_bound {ι : Type*} [Fintype ι] [DecidableEq ι]
    (v : EuclideanSpace ℂ ι) (cnt : ι → ℕ) (Kc : ℝ) (hK : 0 ≤ Kc)
    (Npack tot : ℕ)
    (hv : ∀ i, ‖v i‖ ≤ (cnt i : ℝ) * Kc)
    (hn : ∀ i, cnt i ≤ Npack) (htot : ∑ i, cnt i = tot) :
    ‖v‖ ≤ Kc * Real.sqrt ((Npack : ℝ) * tot) := by
  have hsum : ∑ i, ‖v i‖ ^ 2 ≤ ((Npack : ℝ) * tot) * Kc ^ 2 := by
    have h1 : ∑ i, ‖v i‖ ^ 2 ≤ ∑ i, ((cnt i : ℝ) * Kc) ^ 2 :=
      Finset.sum_le_sum fun i _ => pow_le_pow_left₀ (norm_nonneg _) (hv i) 2
    have h2 : ∑ i, ((cnt i : ℝ) * Kc) ^ 2 = ((∑ i, cnt i ^ 2 : ℕ) : ℝ) * Kc ^ 2 := by
      push_cast
      rw [Finset.sum_mul]
      exact Finset.sum_congr rfl fun i _ => by ring
    have h3 : ((∑ i, cnt i ^ 2 : ℕ) : ℝ) ≤ (Npack : ℝ) * tot := by
      have := sum_sq_le_mul Finset.univ cnt Npack tot (fun i _ => hn i) htot
      exact_mod_cast this
    calc ∑ i, ‖v i‖ ^ 2 ≤ ((∑ i, cnt i ^ 2 : ℕ) : ℝ) * Kc ^ 2 := h1.trans_eq h2
      _ ≤ ((Npack : ℝ) * tot) * Kc ^ 2 :=
          mul_le_mul_of_nonneg_right h3 (sq_nonneg _)
  rw [EuclideanSpace.norm_eq]
  calc Real.sqrt (∑ i, ‖v i‖ ^ 2) ≤ Real.sqrt (((Npack : ℝ) * tot) * Kc ^ 2) :=
        Real.sqrt_le_sqrt hsum
    _ = Kc * Real.sqrt ((Npack : ℝ) * tot) := by
        rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hK, mul_comm]

end Dqi
