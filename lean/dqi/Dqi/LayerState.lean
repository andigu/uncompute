import Dqi.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Real.Sqrt

/-!
# Weight-layer states and their orthonormality

The DQI state decomposes along the Hamming weight of the error pattern: for each
weight `k` there is a normalized *layer state*, the uniform-over-supports
superposition of all patterns of weight exactly `k`, with coordinate amplitudes
supplied by an abstract family `g i : K → ℂ` (one amplitude profile per
constraint, vanishing at `0` and of unit `ℓ²`-mass on the nonzero field
elements).

This file defines these layer states (`layerState`) as elements of
`EuclideanSpace ℂ (Pattern m K)` and proves that they form an orthonormal family
(`layerState_inner`).  Orthonormality is the statement that lets the weight
profile of the DQI state be read off as a unit vector: the squared amplitudes
that the algorithm places on the individual weight layers sum to one, so the
weight profile itself is a unit vector in the layer basis.

The file also records the binomial-coefficient identity
`C(m, k+1) * (k+1) = C(m, k) * (m - k)` and its square-root consequence
`√C(m,k+1) / √C(m,k) * (k+1) = √((k+1) * (m-k))`, which supply the off-diagonal
coefficients of the tridiagonal weight-layer operator built on top of this
basis.
-/

namespace Dqi

/-! ### Binomial coefficient identities -/

/-- The ratio of adjacent binomial coefficients: `C(m, k+1) * (k+1) = C(m, k) * (m-k)`.
This is Mathlib's `Nat.choose_succ_right_eq`, restated here as the form used by the
shell formula. -/
theorem choose_succ_mul_succ_eq (m k : ℕ) :
    m.choose (k + 1) * (k + 1) = m.choose k * (m - k) :=
  Nat.choose_succ_right_eq m k

/-- The square-root consequence used by the shell formula:
`√C(m,k+1) / √C(m,k) * (k+1) = √((k+1) * (m-k))`, i.e. the off-diagonal
coefficient of the weight-layer operator. -/
theorem sqrt_choose_succ_ratio (m k : ℕ) (hk : k + 1 ≤ m) :
    Real.sqrt (m.choose (k + 1)) / Real.sqrt (m.choose k) * (k + 1) =
      Real.sqrt ((k + 1) * ((m : ℝ) - k)) := by
  have hkm : k ≤ m := Nat.le_of_succ_le hk
  have hb : (0 : ℝ) < m.choose k := by exact_mod_cast Nat.choose_pos hkm
  have key : (m.choose (k + 1) : ℝ) * ((k : ℝ) + 1) = m.choose k * ((m : ℝ) - k) := by
    have h := congrArg (fun n : ℕ => (n : ℝ)) (Nat.choose_succ_right_eq m k)
    push_cast [Nat.cast_sub hkm] at h
    linarith [h]
  have h2 : ((k : ℝ) + 1) * ((m : ℝ) - k) =
      (m.choose (k + 1) : ℝ) * ((k : ℝ) + 1) ^ 2 / (m.choose k) := by
    rw [eq_div_iff hb.ne']
    linear_combination (-((k : ℝ) + 1)) * key
  rw [h2, Real.sqrt_div (by positivity), Real.sqrt_mul (by positivity),
    Real.sqrt_sq (by positivity)]
  ring

/-! ### Weight-layer states -/

variable {m : ℕ} {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- The normalized weight-`k` layer state.  On a pattern `y` of Hamming weight `k`
its amplitude is `C(m,k)^(-1/2)` times the product of the coordinate amplitudes
`g i (y i)` over the support of `y`; on every other pattern it vanishes. -/
noncomputable def layerState (g : Fin m → K → ℂ) (k : ℕ) :
    EuclideanSpace ℂ (Pattern m K) :=
  WithLp.toLp 2 fun y =>
    if hammingNorm y = k then
      (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) * ∏ i : Fin m, if y i = 0 then 1 else g i (y i)
    else 0

omit [Fintype K] in
theorem layerState_apply (g : Fin m → K → ℂ) (k : ℕ) (y : Pattern m K) :
    layerState g k y =
      if hammingNorm y = k then
        (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) * ∏ i : Fin m, if y i = 0 then 1 else g i (y i)
      else 0 :=
  rfl

omit [Fintype K] in
/-- The Hamming weight of a pattern is the size of its support finset. -/
theorem hammingNorm_eq_card_support (y : Pattern m K) :
    hammingNorm y = (Finset.univ.filter fun i => y i ≠ 0).card :=
  rfl

/-- The key counting identity behind normalization: summing the product of the
per-coordinate squared amplitudes over all patterns of weight exactly `k` gives
`C(m,k)` — for each of the `C(m,k)` supports, the value sum factorizes into a
product of the unit masses `∑_{a ≠ 0} ‖g i a‖² = 1`. -/
theorem sum_layer_prod_normSq (g : Fin m → K → ℂ) (hg0 : ∀ i, g i 0 = 0)
    (hgnorm : ∀ i, ∑ a : K, ‖g i a‖ ^ 2 = 1) (k : ℕ) :
    ∑ y ∈ Finset.univ.filter (fun y : Pattern m K => hammingNorm y = k),
        ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2) =
      (m.choose k : ℝ) := by
  classical
  -- every weight-`k` pattern has a `k`-element support
  have hmaps : ∀ y ∈ Finset.univ.filter (fun y : Pattern m K => hammingNorm y = k),
      (Finset.univ.filter fun i => y i ≠ 0) ∈
        Finset.powersetCard k (Finset.univ : Finset (Fin m)) := by
    intro y hy
    rw [Finset.mem_powersetCard]
    refine ⟨Finset.subset_univ _, ?_⟩
    rw [← hammingNorm_eq_card_support y]
    exact (Finset.mem_filter.mp hy).2
  -- for each fixed support, the fiber sum factorizes and equals 1
  have hfiber : ∀ S ∈ Finset.powersetCard k (Finset.univ : Finset (Fin m)),
      (∑ y ∈ (Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).filter
          (fun y => (Finset.univ.filter fun i => y i ≠ 0) = S),
        ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2)) = 1 := by
    intro S hS
    obtain ⟨-, hScard⟩ := Finset.mem_powersetCard.mp hS
    -- the fiber over `S` is exactly the product finset "nonzero on S, zero off S"
    have hset : ((Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).filter
          (fun y => (Finset.univ.filter fun i => y i ≠ 0) = S)) =
        Fintype.piFinset fun i => if i ∈ S then (Finset.univ : Finset K).erase 0 else {0} := by
      ext y
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
      constructor
      · rintro ⟨-, hsupp⟩ i
        by_cases hi : i ∈ S
        · rw [if_pos hi]
          have hmem : i ∈ Finset.univ.filter fun i => y i ≠ 0 := by rw [hsupp]; exact hi
          exact Finset.mem_erase.mpr ⟨(Finset.mem_filter.mp hmem).2, Finset.mem_univ _⟩
        · rw [if_neg hi]
          have hyi : y i = 0 := by
            by_contra hne
            exact hi (hsupp ▸ Finset.mem_filter.mpr ⟨Finset.mem_univ _, hne⟩)
          simp [hyi]
      · intro h
        have hchar : ∀ i, y i ≠ 0 ↔ i ∈ S := by
          intro i
          by_cases hi : i ∈ S
          · have hyi := h i
            rw [if_pos hi] at hyi
            exact ⟨fun _ => hi, fun _ => (Finset.mem_erase.mp hyi).1⟩
          · have hyi := h i
            rw [if_neg hi, Finset.mem_singleton] at hyi
            exact ⟨fun hne => absurd hyi hne, fun hmem => absurd hmem hi⟩
        have hsupp : (Finset.univ.filter fun i => y i ≠ 0) = S := by
          ext i
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          exact hchar i
        exact ⟨by rw [hammingNorm_eq_card_support y, hsupp, hScard], hsupp⟩
    calc (∑ y ∈ (Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).filter
            (fun y => (Finset.univ.filter fun i => y i ≠ 0) = S),
          ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2))
        = ∑ y ∈ Fintype.piFinset
              (fun i => if i ∈ S then (Finset.univ : Finset K).erase 0 else {0}),
            ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2) := by rw [hset]
      _ = ∏ i : Fin m, ∑ a ∈ (if i ∈ S then (Finset.univ : Finset K).erase 0 else {0}),
            (if a = 0 then (1 : ℝ) else ‖g i a‖ ^ 2) :=
          (Finset.prod_univ_sum _ fun i a => if a = 0 then (1 : ℝ) else ‖g i a‖ ^ 2).symm
      _ = 1 := by
          refine Finset.prod_eq_one fun i _ => ?_
          by_cases hi : i ∈ S
          · rw [if_pos hi,
              Finset.sum_congr rfl fun a ha => if_neg (Finset.ne_of_mem_erase ha),
              Finset.sum_erase _ (by simp [hg0 i])]
            exact hgnorm i
          · rw [if_neg hi, Finset.sum_singleton, if_pos rfl]
  calc ∑ y ∈ Finset.univ.filter (fun y : Pattern m K => hammingNorm y = k),
        ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2)
      = ∑ S ∈ Finset.powersetCard k (Finset.univ : Finset (Fin m)),
          ∑ y ∈ (Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).filter
            (fun y => (Finset.univ.filter fun i => y i ≠ 0) = S),
          ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2) :=
        (Finset.sum_fiberwise_of_maps_to hmaps _).symm
    _ = ∑ _S ∈ Finset.powersetCard k (Finset.univ : Finset (Fin m)), (1 : ℝ) :=
        Finset.sum_congr rfl hfiber
    _ = (m.choose k : ℝ) := by
        simp [Finset.card_powersetCard]

/-! ### Orthonormality -/

/-- Layer states of distinct weights are orthogonal: no pattern has two Hamming
weights, so the coordinate-wise products vanish identically. -/
theorem layerState_inner_of_ne (g : Fin m → K → ℂ) {j k : ℕ} (hjk : j ≠ k) :
    inner ℂ (layerState g j) (layerState g k) = 0 := by
  rw [PiLp.inner_apply]
  refine Finset.sum_eq_zero fun y _ => ?_
  by_cases hy : hammingNorm y = j
  · have hzero : layerState g k y = 0 := by
      rw [layerState_apply, if_neg fun h => hjk (hy.symm.trans h)]
    rw [hzero, inner_zero_right]
  · have hzero : layerState g j y = 0 := by
      rw [layerState_apply, if_neg hy]
    rw [hzero, inner_zero_left]

/-- Each layer state is normalized, provided the layer is nonempty (`k ≤ m`). -/
theorem layerState_inner_self (g : Fin m → K → ℂ) (hg0 : ∀ i, g i 0 = 0)
    (hgnorm : ∀ i, ∑ a : K, ‖g i a‖ ^ 2 = 1) {k : ℕ} (hk : k ≤ m) :
    inner ℂ (layerState g k) (layerState g k) = 1 := by
  have hpos : (0 : ℝ) < m.choose k := by exact_mod_cast Nat.choose_pos hk
  -- the squared amplitudes sum to 1
  have hsq : ∑ y : Pattern m K, ‖layerState g k y‖ ^ 2 = 1 := by
    have hcomp : ∀ y : Pattern m K, ‖layerState g k y‖ ^ 2 =
        if hammingNorm y = k then
          (m.choose k : ℝ)⁻¹ * ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2)
        else 0 := by
      intro y
      rw [layerState_apply]
      split_ifs with hy
      · rw [norm_mul, mul_pow]
        congr 1
        · rw [Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (by positivity),
            ← Real.sqrt_inv, Real.sq_sqrt (by positivity)]
        · rw [norm_prod, ← Finset.prod_pow]
          refine Finset.prod_congr rfl fun i _ => ?_
          split_ifs with h0 <;> simp
      · simp
    calc ∑ y : Pattern m K, ‖layerState g k y‖ ^ 2
        = ∑ y : Pattern m K,
            if hammingNorm y = k then
              (m.choose k : ℝ)⁻¹ *
                ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2)
            else 0 :=
          Finset.sum_congr rfl fun y _ => hcomp y
      _ = ∑ y ∈ Finset.univ.filter (fun y : Pattern m K => hammingNorm y = k),
            (m.choose k : ℝ)⁻¹ *
              ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2) :=
          (Finset.sum_filter _ _).symm
      _ = (m.choose k : ℝ)⁻¹ *
            ∑ y ∈ Finset.univ.filter (fun y : Pattern m K => hammingNorm y = k),
              ∏ i : Fin m, (if y i = 0 then (1 : ℝ) else ‖g i (y i)‖ ^ 2) :=
          (Finset.mul_sum _ _ _).symm
      _ = (m.choose k : ℝ)⁻¹ * (m.choose k : ℝ) := by
          rw [sum_layer_prod_normSq g hg0 hgnorm k]
      _ = 1 := inv_mul_cancel₀ hpos.ne'
  have hn2 : ‖layerState g k‖ ^ 2 = 1 := by
    rw [EuclideanSpace.norm_sq_eq]
    exact hsq
  have hn : ‖layerState g k‖ = 1 := by
    have h := congrArg Real.sqrt hn2
    rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_one] at h
  rw [inner_self_eq_norm_sq_to_K, hn]
  simp

/-- **Orthonormality of the weight-layer states.**  This is what makes the DQI
weight profile a unit vector in the layer basis. -/
theorem layerState_inner (g : Fin m → K → ℂ) (hg0 : ∀ i, g i 0 = 0)
    (hgnorm : ∀ i, ∑ a : K, ‖g i a‖ ^ 2 = 1) {j k : ℕ} (hj : j ≤ m) (hk : k ≤ m) :
    inner ℂ (layerState g j) (layerState g k) = if j = k then 1 else 0 := by
  rcases eq_or_ne j k with h | h
  · subst h
    rw [if_pos rfl]
    exact layerState_inner_self g hg0 hgnorm ((le_min hj hk).trans (min_le_left m m))
  · rw [if_neg h]
    exact layerState_inner_of_ne g h

end Dqi
