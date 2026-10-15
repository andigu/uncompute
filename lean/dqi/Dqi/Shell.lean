import Dqi.Amplitudes
import Dqi.SiteOp

/-!
# The weight-shell (tridiagonal) structure of the DQI objective

The DQI objective operator is a sum of single-site constraint operators.  This
file shows that, on the orthonormal weight-layer states of `Dqi.LayerState`,
the (shifted) objective `∑ᵢ (2 Pᵢ - 1)` acts *tridiagonally*: applied to the
layer state of Hamming weight `k` it produces a combination of the layer
states of weights `k - 1`, `k`, and `k + 1` only.

The three coefficients are the entries of the tridiagonal shell matrix:
`m ⋅ f̄ + 2 c₁ k d` on the diagonal and `2 c₁ √(k (m - k + 1))`,
`2 c₁ √((k+1) (m - k))` on the two off-diagonals, where `f̄ = (2r - q)/q`,
`d = (q - 2r)/√(r (q - r))`, and `c₁ = √(r (q - r))/q` is `Dqi.cc1`.

This tridiagonal structure is what makes the ideal weight profile an
eigenvector of the objective within the decodable subspace: the operator never
leaves the span of neighbouring weight layers, so its spectral analysis
reduces to that of a tridiagonal matrix indexed by the Hamming weight.
-/

namespace Dqi

section Shell

variable {m : ℕ} {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- The mean objective value per constraint: `(2r - q)/q`. -/
noncomputable def fbar (r q : ℕ) : ℝ := (2 * r - q) / q

/-- The diagonal drift coefficient `(q - 2r)/√(r (q - r))` of the shell matrix. -/
noncomputable def dd (r q : ℕ) : ℝ :=
  ((q : ℝ) - 2 * r) / Real.sqrt (r * ((q : ℝ) - r))

/-- The single-site constraint operator at site `i`: convolve the `i`-th
coordinate with the (unnormalized) indicator character sum of `F`, leaving all
other coordinates untouched. -/
noncomputable def siteOp (χ : AddChar K ℂ) (F : Finset K) (i : Fin m)
    (ψ : EuclideanSpace ℂ (Pattern m K)) : EuclideanSpace ℂ (Pattern m K) :=
  WithLp.toLp 2 fun y =>
    ((Fintype.card K : ℝ)⁻¹ : ℂ) *
      ∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) * ψ (Function.update y i a)

omit [DecidableEq K] in
theorem siteOp_apply (χ : AddChar K ℂ) (F : Finset K) (i : Fin m)
    (ψ : EuclideanSpace ℂ (Pattern m K)) (y : Pattern m K) :
    siteOp χ F i ψ y =
      ((Fintype.card K : ℝ)⁻¹ : ℂ) *
        ∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) * ψ (Function.update y i a) :=
  rfl

/-- The shifted objective operator `∑ᵢ (2 Pᵢ - 1)`, acting site by site. -/
noncomputable def shellOp (χ : AddChar K ℂ) (Fs : Fin m → Finset K)
    (ψ : EuclideanSpace ℂ (Pattern m K)) : EuclideanSpace ℂ (Pattern m K) :=
  WithLp.toLp 2 fun y => ∑ i : Fin m, (2 * (siteOp χ (Fs i) i ψ) y - ψ y)

omit [DecidableEq K] in
theorem shellOp_apply (χ : AddChar K ℂ) (Fs : Fin m → Finset K)
    (ψ : EuclideanSpace ℂ (Pattern m K)) (y : Pattern m K) :
    shellOp χ Fs ψ y = ∑ i : Fin m, (2 * (siteOp χ (Fs i) i ψ) y - ψ y) :=
  rfl

omit [Field K] [DecidableEq K] in
/-- `cc1` depends only on the cardinality of the target set. -/
theorem cc1_congr {F G : Finset K} (h : F.card = G.card) : cc1 F = cc1 G := by
  unfold cc1
  rw [h]

omit [Fintype K] in
/-- How a single-site update moves the Hamming weight, stated additively so no
truncated subtraction appears: the weight of `Function.update y i a` plus the
indicator of the old entry equals the weight of `y` plus the indicator of the
new entry. -/
theorem hammingNorm_update (y : Pattern m K) (i : Fin m) (a : K) :
    hammingNorm (Function.update y i a) + (if y i = 0 then 0 else 1)
      = hammingNorm y + (if a = 0 then 0 else 1) := by
  classical
  have hcard : ∀ z : Pattern m K,
      hammingNorm z = ∑ j : Fin m, if z j ≠ 0 then 1 else 0 := by
    intro z
    rw [hammingNorm_eq_card_support, Finset.card_filter]
  rw [hcard, hcard, ← Finset.sum_erase_add _ _ (Finset.mem_univ i),
    ← Finset.sum_erase_add _ _ (Finset.mem_univ i)]
  have herase : ∑ j ∈ Finset.univ.erase i,
        (if Function.update y i a j ≠ 0 then 1 else 0)
      = ∑ j ∈ Finset.univ.erase i, (if y j ≠ 0 then 1 else 0) := by
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hj)]
  rw [herase, Function.update_self]
  by_cases hyi : y i = 0 <;> by_cases ha : a = 0 <;>
    simp [hyi, ha]

omit [Fintype K] in
/-- Splitting the layer product at site `i` after an update: the updated entry
contributes its own factor, the rest of the product is untouched. -/
theorem prod_update_eq (g : Fin m → K → ℂ) (y : Pattern m K) (i : Fin m) (a : K) :
    (∏ j : Fin m,
        if Function.update y i a j = 0 then 1 else g j (Function.update y i a j))
      = (if a = 0 then 1 else g i a) *
        ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else g j (y j)) := by
  classical
  rw [← Finset.mul_prod_erase Finset.univ _ (Finset.mem_univ i), Function.update_self]
  congr 1
  refine Finset.prod_congr rfl fun j hj => ?_
  rw [Function.update_of_ne (Finset.ne_of_mem_erase hj)]

omit [Fintype K] in
/-- Splitting the layer product at site `i` without an update. -/
theorem prod_split (g : Fin m → K → ℂ) (y : Pattern m K) (i : Fin m) :
    (∏ j : Fin m, if y j = 0 then 1 else g j (y j))
      = (if y i = 0 then 1 else g i (y i)) *
        ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else g j (y j)) :=
  (Finset.mul_prod_erase Finset.univ _ (Finset.mem_univ i)).symm

/-! ### The site operator on a layer state, pointwise -/

/-- The single-site operator applied to a weight-`k` layer state, evaluated at a
pattern `y`: the value is the full layer amplitude of `y` times a scalar that
depends only on whether `y i` vanishes and on how the weight of `y` compares to
`k`.  The four cases are the four entries of the 2×2 single-site shell matrix of
`Dqi.SiteOp`, and only the layers `k - 1`, `k`, `k + 1` can contribute. -/
theorem siteOp_layerState_apply (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (i : Fin m) (hr : 0 < (Fs i).card) (hrq : (Fs i).card < Fintype.card K)
    (k : ℕ) (y : Pattern m K) :
    siteOp χ (Fs i) i (layerState (amp χ Fs) k) y
      = (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
          (∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) *
        ((if hammingNorm y = k then
            (if y i = 0 then ((((Fs i).card : ℝ) / (Fintype.card K : ℝ) : ℝ) : ℂ)
             else ((((Fintype.card K : ℝ) - (Fs i).card) / (Fintype.card K : ℝ) : ℝ) : ℂ))
          else 0)
         + (if y i = 0 then
             (if hammingNorm y + 1 = k then ((cc1 (Fs i) : ℝ) : ℂ) else 0)
           else
             (if hammingNorm y = k + 1 then ((cc1 (Fs i) : ℝ) : ℂ) else 0))) := by
  classical
  have hz := siteRow_zero χ (Fs i) hr hrq (y i)
  have hgh : ((Fintype.card K : ℝ)⁻¹ : ℂ) *
      ∑ a : K, (∑ v ∈ Fs i, χ ((y i - a) * v)) * amp χ Fs i a
        = ((cc1 (Fs i) : ℝ) : ℂ) * (if y i = 0 then 1 else 0)
          + ((((Fintype.card K : ℝ) - (Fs i).card) / (Fintype.card K : ℝ) : ℝ) : ℂ) *
            ghat χ (Fs i) (y i) :=
    siteRow_ghat χ hχ (Fs i) hr hrq (y i)
  rw [siteOp_apply]
  by_cases hyi : y i = 0
  · -- The site reads `0`: the diagonal entry is `r/q`, the off-diagonal reaches
    -- weight `k` only from weight `k - 1`.
    have hL : ∀ a : K,
        layerState (amp χ Fs) k (Function.update y i a)
          = (if hammingNorm y + 1 = k then
               (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                 ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
             else 0) * amp χ Fs i a
            + (if a = 0 then
                (if hammingNorm y = k then
                  (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                    ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
                 else 0)
               else 0) := by
      intro a
      have hw := hammingNorm_update y i a
      rw [if_pos hyi, add_zero] at hw
      rw [layerState_apply, prod_update_eq]
      by_cases ha : a = 0
      · subst ha
        rw [if_pos rfl, add_zero] at hw
        rw [if_pos rfl, one_mul, if_pos rfl, amp_zero, mul_zero, zero_add, hw]
      · rw [if_neg ha] at hw
        rw [if_neg ha, if_neg ha, add_zero, hw]
        split_ifs with h
        · ring
        · rw [zero_mul]
    calc ((Fintype.card K : ℝ)⁻¹ : ℂ) *
          ∑ a : K, (∑ v ∈ Fs i, χ ((y i - a) * v)) *
            layerState (amp χ Fs) k (Function.update y i a)
        = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ∑ a : K,
              ((if hammingNorm y + 1 = k then
                  (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                    ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
                else 0) * ((∑ v ∈ Fs i, χ ((y i - a) * v)) * amp χ Fs i a)
               + (if a = 0 then
                   (∑ v ∈ Fs i, χ ((y i - a) * v)) *
                     (if hammingNorm y = k then
                       (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                         ∏ j ∈ Finset.univ.erase i,
                           (if y j = 0 then 1 else amp χ Fs j (y j))
                      else 0)
                  else 0)) := by
          congr 1
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [hL a, mul_add, mul_ite, mul_zero, mul_left_comm]
      _ = (if hammingNorm y + 1 = k then
              (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
            else 0) *
            (((Fintype.card K : ℝ)⁻¹ : ℂ) *
              ∑ a : K, (∑ v ∈ Fs i, χ ((y i - a) * v)) * amp χ Fs i a)
          + (((Fintype.card K : ℝ)⁻¹ : ℂ) * ∑ v ∈ Fs i, χ ((y i - 0) * v)) *
            (if hammingNorm y = k then
              (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
             else 0) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum,
            Finset.sum_ite_eq' Finset.univ (0 : K), if_pos (Finset.mem_univ (0 : K))]
          ring
      _ = (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
            (∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) *
          ((if hammingNorm y = k then
              (if y i = 0 then ((((Fs i).card : ℝ) / (Fintype.card K : ℝ) : ℝ) : ℂ)
               else ((((Fintype.card K : ℝ) - (Fs i).card) / (Fintype.card K : ℝ) : ℝ) : ℂ))
            else 0)
           + (if y i = 0 then
               (if hammingNorm y + 1 = k then ((cc1 (Fs i) : ℝ) : ℂ) else 0)
             else
               (if hammingNorm y = k + 1 then ((cc1 (Fs i) : ℝ) : ℂ) else 0))) := by
          rw [sub_zero, hgh, hz, prod_split (amp χ Fs) y i]
          simp only [hyi, if_true, ghat_zero, mul_zero, add_zero, mul_one, one_mul]
          split_ifs <;> ring
  · -- The site reads a nonzero value: the diagonal entry is `(q - r)/q`, the
    -- off-diagonal reaches weight `k` only from weight `k + 1`.
    have hL : ∀ a : K,
        layerState (amp χ Fs) k (Function.update y i a)
          = (if hammingNorm y = k then
               (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                 ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
             else 0) * amp χ Fs i a
            + (if a = 0 then
                (if hammingNorm y = k + 1 then
                  (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                    ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
                 else 0)
               else 0) := by
      intro a
      have hw := hammingNorm_update y i a
      rw [if_neg hyi] at hw
      rw [layerState_apply, prod_update_eq]
      by_cases ha : a = 0
      · subst ha
        rw [if_pos rfl, add_zero] at hw
        rw [if_pos rfl, one_mul, if_pos rfl, amp_zero, mul_zero, zero_add]
        by_cases hnk : hammingNorm y = k + 1
        · rw [if_pos hnk, if_pos (by omega)]
        · rw [if_neg hnk, if_neg (by omega)]
      · rw [if_neg ha] at hw
        have hww : hammingNorm (Function.update y i a) = hammingNorm y := by omega
        rw [if_neg ha, if_neg ha, add_zero, hww]
        split_ifs with h
        · ring
        · rw [zero_mul]
    calc ((Fintype.card K : ℝ)⁻¹ : ℂ) *
          ∑ a : K, (∑ v ∈ Fs i, χ ((y i - a) * v)) *
            layerState (amp χ Fs) k (Function.update y i a)
        = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ∑ a : K,
              ((if hammingNorm y = k then
                  (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                    ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
                else 0) * ((∑ v ∈ Fs i, χ ((y i - a) * v)) * amp χ Fs i a)
               + (if a = 0 then
                   (∑ v ∈ Fs i, χ ((y i - a) * v)) *
                     (if hammingNorm y = k + 1 then
                       (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                         ∏ j ∈ Finset.univ.erase i,
                           (if y j = 0 then 1 else amp χ Fs j (y j))
                      else 0)
                  else 0)) := by
          congr 1
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [hL a, mul_add, mul_ite, mul_zero, mul_left_comm]
      _ = (if hammingNorm y = k then
              (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
            else 0) *
            (((Fintype.card K : ℝ)⁻¹ : ℂ) *
              ∑ a : K, (∑ v ∈ Fs i, χ ((y i - a) * v)) * amp χ Fs i a)
          + (((Fintype.card K : ℝ)⁻¹ : ℂ) * ∑ v ∈ Fs i, χ ((y i - 0) * v)) *
            (if hammingNorm y = k + 1 then
              (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                ∏ j ∈ Finset.univ.erase i, (if y j = 0 then 1 else amp χ Fs j (y j))
             else 0) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum,
            Finset.sum_ite_eq' Finset.univ (0 : K), if_pos (Finset.mem_univ (0 : K))]
          ring
      _ = (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
            (∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) *
          ((if hammingNorm y = k then
              (if y i = 0 then ((((Fs i).card : ℝ) / (Fintype.card K : ℝ) : ℝ) : ℂ)
               else ((((Fintype.card K : ℝ) - (Fs i).card) / (Fintype.card K : ℝ) : ℝ) : ℂ))
            else 0)
           + (if y i = 0 then
               (if hammingNorm y + 1 = k then ((cc1 (Fs i) : ℝ) : ℂ) else 0)
             else
               (if hammingNorm y = k + 1 then ((cc1 (Fs i) : ℝ) : ℂ) else 0))) := by
          rw [sub_zero, hgh, hz, prod_split (amp χ Fs) y i]
          have hamp : amp χ Fs i (y i) = ghat χ (Fs i) (y i) := rfl
          simp only [hyi, if_false, mul_zero, zero_add, hamp]
          split_ifs <;> ring

/-! ### The scalar identities of the shell matrix -/

omit [Field K] [Fintype K] [DecidableEq K] in
/-- The lower off-diagonal matching: `(m - n) / √C(m, n+1) = √((n+1)(m-n)) / √C(m, n)`,
the form of `choose_succ_mul_succ_eq` used when the input layer sits one step above
the output layer. -/
theorem offdiag_low (m n : ℕ) (hn : n + 1 ≤ m) :
    ((m : ℝ) - n) * (Real.sqrt (m.choose (n + 1)))⁻¹
      = Real.sqrt (((n : ℝ) + 1) * ((m : ℝ) - n)) * (Real.sqrt (m.choose n))⁻¹ := by
  have hnm : n ≤ m := Nat.le_of_succ_le hn
  have hCn : (0 : ℝ) < m.choose n := by exact_mod_cast Nat.choose_pos hnm
  have hCn1 : (0 : ℝ) < m.choose (n + 1) := by exact_mod_cast Nat.choose_pos hn
  have hmn : (0 : ℝ) ≤ (m : ℝ) - n := by
    have h : (n : ℝ) ≤ m := by exact_mod_cast hnm
    linarith
  have key : (m.choose (n + 1) : ℝ) * ((n : ℝ) + 1)
      = (m.choose n : ℝ) * ((m : ℝ) - n) := by
    have h := congrArg (fun t : ℕ => (t : ℝ)) (choose_succ_mul_succ_eq m n)
    push_cast [Nat.cast_sub hnm] at h
    linarith [h]
  calc ((m : ℝ) - n) * (Real.sqrt (m.choose (n + 1)))⁻¹
      = Real.sqrt (((m : ℝ) - n) ^ 2 * (m.choose (n + 1) : ℝ)⁻¹) := by
        rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hmn, Real.sqrt_inv]
    _ = Real.sqrt ((((n : ℝ) + 1) * ((m : ℝ) - n)) * (m.choose n : ℝ)⁻¹) := by
        congr 1
        rw [inv_eq_one_div, inv_eq_one_div, mul_one_div, mul_one_div,
          div_eq_div_iff hCn1.ne' hCn.ne']
        linear_combination ((n : ℝ) - m) * key
    _ = Real.sqrt (((n : ℝ) + 1) * ((m : ℝ) - n)) * (Real.sqrt (m.choose n))⁻¹ := by
        rw [Real.sqrt_mul (by positivity), Real.sqrt_inv]

omit [Field K] [Fintype K] [DecidableEq K] in
/-- The upper off-diagonal matching: `(k + 1) / √C(m, k) = √((k+1)(m-k)) / √C(m, k+1)`,
the form of `choose_succ_mul_succ_eq` used when the input layer sits one step below
the output layer. -/
theorem offdiag_high (m k : ℕ) (hk : k + 1 ≤ m) :
    ((k : ℝ) + 1) * (Real.sqrt (m.choose k))⁻¹
      = Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) * (Real.sqrt (m.choose (k + 1)))⁻¹ := by
  have hkm : k ≤ m := Nat.le_of_succ_le hk
  have hCk : (0 : ℝ) < m.choose k := by exact_mod_cast Nat.choose_pos hkm
  have hCk1 : (0 : ℝ) < m.choose (k + 1) := by exact_mod_cast Nat.choose_pos hk
  have hmk : (0 : ℝ) ≤ (m : ℝ) - k := by
    have h : (k : ℝ) ≤ m := by exact_mod_cast hkm
    linarith
  have key : (m.choose (k + 1) : ℝ) * ((k : ℝ) + 1)
      = (m.choose k : ℝ) * ((m : ℝ) - k) := by
    have h := congrArg (fun t : ℕ => (t : ℝ)) (choose_succ_mul_succ_eq m k)
    push_cast [Nat.cast_sub hkm] at h
    linarith [h]
  calc ((k : ℝ) + 1) * (Real.sqrt (m.choose k))⁻¹
      = Real.sqrt (((k : ℝ) + 1) ^ 2 * (m.choose k : ℝ)⁻¹) := by
        rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity), Real.sqrt_inv]
    _ = Real.sqrt ((((k : ℝ) + 1) * ((m : ℝ) - k)) * (m.choose (k + 1) : ℝ)⁻¹) := by
        congr 1
        rw [inv_eq_one_div, inv_eq_one_div, mul_one_div, mul_one_div,
          div_eq_div_iff hCk.ne' hCk1.ne']
        linear_combination ((k : ℝ) + 1) * key
    _ = Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) * (Real.sqrt (m.choose (k + 1)))⁻¹ := by
        rw [Real.sqrt_mul (by positivity), Real.sqrt_inv]

/-- The off-diagonal entry `cc1` of the single-site shell matrix as a function
of the two cardinalities `r = |F|` and `q = |K|` alone.  With all target sets
of a common size `r`, `cc1 (Fs i) = cc1rq r q` for every site (`cc1_eq_cc1rq`),
which is the form the shell theorem uses — it does not depend on a site. -/
noncomputable def cc1rq (r q : ℕ) : ℝ := Real.sqrt ((r : ℝ) * ((q : ℝ) - r)) / q

omit [Field K] [DecidableEq K] in
/-- `cc1` computed from the cardinalities: the bridge from the per-site
constant of `Dqi.SiteOp` to the site-independent constant of the shell. -/
theorem cc1_eq_cc1rq (F : Finset K) : cc1 F = cc1rq F.card (Fintype.card K) := rfl

omit [Field K] [Fintype K] [DecidableEq K] in
/-- The diagonal matching: summing the per-site diagonal contributions and
subtracting the identity gives the drifted mean `m f̄ + 2 c₁ k d`. -/
theorem diag_identity (m k r q : ℕ) (hr : 0 < r) (hrq : r < q) :
    2 * (((m : ℝ) - k) * ((r : ℝ) / q) + (k : ℝ) * (((q : ℝ) - r) / q)) - m
      = m * fbar r q + 2 * cc1rq r q * ((k : ℝ) * dd r q) := by
  have hr0 : (0 : ℝ) < (r : ℝ) := by exact_mod_cast hr
  have hq0 : (0 : ℝ) < (q : ℝ) := by exact_mod_cast hr.trans hrq
  have hqr : (0 : ℝ) < (q : ℝ) - r := by
    have h : (r : ℝ) < q := by exact_mod_cast hrq
    linarith
  have hs : Real.sqrt ((r : ℝ) * ((q : ℝ) - r)) ≠ 0 := by positivity
  rw [fbar, dd, cc1rq]
  field_simp
  ring


/-! ### The shell theorem -/

/-- **The weight-shell (tridiagonal) structure of the DQI objective.**

The shifted objective `∑ᵢ (2 Pᵢ - 1)` maps the weight-`k` layer state into the
span of the layer states of weights `k - 1`, `k`, `k + 1`, with the tridiagonal
coefficients `m f̄ + 2 c₁ k d`, `2 c₁ √(k (m - k + 1))`, and
`2 c₁ √((k + 1) (m - k))`.  All target sets have the common size `r`, so the
single-site constant is the site-independent `c₁ = cc1rq r q = cc1 (Fs i)`
(see `cc1_eq_cc1rq`). -/
theorem shellOp_layerState (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (r : ℕ) (hcard : ∀ i, (Fs i).card = r) (hr : 0 < r) (hrq : r < Fintype.card K)
    (k : ℕ) (hk : k ≤ m) :
    shellOp χ Fs (layerState (amp χ Fs) k) =
      (((m : ℝ) * fbar r (Fintype.card K) : ℝ) : ℂ) • layerState (amp χ Fs) k
      + ((2 * cc1rq r (Fintype.card K) : ℝ) : ℂ) •
          (((Real.sqrt ((k : ℝ) * ((m : ℝ) - k + 1)) : ℝ) : ℂ) •
              layerState (amp χ Fs) (k - 1)
          + (((k : ℝ) * dd r (Fintype.card K) : ℝ) : ℂ) • layerState (amp χ Fs) k
          + ((Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) : ℝ) : ℂ) •
              layerState (amp χ Fs) (k + 1)) := by
  classical
  apply PiLp.ext
  intro y
  -- The per-site values, with the uniform cardinality `r` substituted in.
  have hsite : ∀ i : Fin m,
      siteOp χ (Fs i) i (layerState (amp χ Fs) k) y
        = (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
            (∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) *
          ((if hammingNorm y = k then
              (if y i = 0 then (((r : ℝ) / (Fintype.card K : ℝ) : ℝ) : ℂ)
               else ((((Fintype.card K : ℝ) - r) / (Fintype.card K : ℝ) : ℝ) : ℂ))
            else 0)
           + (if y i = 0 then
               (if hammingNorm y + 1 = k then ((cc1rq r (Fintype.card K) : ℝ) : ℂ) else 0)
             else
               (if hammingNorm y = k + 1 then
                 ((cc1rq r (Fintype.card K) : ℝ) : ℂ) else 0))) := by
    intro i
    have h := siteOp_layerState_apply χ hχ Fs i
      (by rw [hcard i]; exact hr) (by rw [hcard i]; exact hrq) k y
    rw [cc1_eq_cc1rq, hcard i] at h
    exact h
  -- Counting: `m - n` sites read zero and `n` sites read a nonzero value.
  have hn_le : hammingNorm y ≤ m := by
    rw [hammingNorm_eq_card_support]
    simpa using Finset.card_filter_le Finset.univ (fun j : Fin m => y j ≠ 0)
  have hcount : (Finset.univ.filter fun j : Fin m => ¬ y j = 0).card = hammingNorm y := by
    rw [hammingNorm_eq_card_support]
  have hc0 : (Finset.univ.filter fun j : Fin m => y j = 0).card + hammingNorm y = m := by
    rw [← hcount]
    simpa using Finset.card_filter_add_card_filter_not
      (s := (Finset.univ : Finset (Fin m))) (fun j : Fin m => y j = 0)
  -- Splitting the site sum by whether the site reads zero.
  have hgen : ∀ A B : ℂ, (∑ i : Fin m, if y i = 0 then A else B)
      = ((Finset.univ.filter fun j : Fin m => y j = 0).card : ℂ) * A
        + ((hammingNorm y : ℕ) : ℂ) * B := by
    intro A B
    rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, hcount,
      nsmul_eq_mul, nsmul_eq_mul]
  -- The pointwise summand, resolved by the value at site `i`.
  have hpt : ∀ i : Fin m,
      2 * siteOp χ (Fs i) i (layerState (amp χ Fs) k) y - layerState (amp χ Fs) k y
        = if y i = 0 then
            2 * ((((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                (∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) *
              ((if hammingNorm y = k then (((r : ℝ) / (Fintype.card K : ℝ) : ℝ) : ℂ) else 0)
               + (if hammingNorm y + 1 = k then
                   ((cc1rq r (Fintype.card K) : ℝ) : ℂ) else 0)))
             - (if hammingNorm y = k then
                 (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                   ∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)
                else 0)
          else
            2 * ((((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                (∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) *
              ((if hammingNorm y = k then
                  ((((Fintype.card K : ℝ) - r) / (Fintype.card K : ℝ) : ℝ) : ℂ) else 0)
               + (if hammingNorm y = k + 1 then
                   ((cc1rq r (Fintype.card K) : ℝ) : ℂ) else 0)))
             - (if hammingNorm y = k then
                 (((Real.sqrt (m.choose k))⁻¹ : ℝ) : ℂ) *
                   ∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)
                else 0) := by
    intro i
    rw [hsite i, layerState_apply]
    by_cases hyi : y i = 0
    · simp only [hyi, if_true]
    · simp only [hyi, if_false]
  rw [shellOp_apply, Finset.sum_congr rfl fun i _ => hpt i, hgen]
  simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul, layerState_apply]
  by_cases h1 : hammingNorm y = k
  · -- Diagonal case: the input layer is the output layer.
    by_cases hk0 : k = 0
    · -- At `k = 0` the `k - 1` layer is the `k` layer itself, but its
      -- coefficient `√(k (m - k + 1))` vanishes.
      subst hk0
      have e1 : (hammingNorm y = 0) = True := eq_true h1
      have e2 : (hammingNorm y + 1 = 0) = False := eq_false (by omega)
      have e3 : (hammingNorm y = 0 + 1) = False := eq_false (by omega)
      simp only [e1, e2, e3, if_true, if_false]
      rw [h1]
      simp only [Nat.cast_zero, zero_mul, Real.sqrt_zero, Complex.ofReal_zero,
        mul_zero, add_zero, zero_add]
      have hc0m : (Finset.univ.filter fun j : Fin m => y j = 0).card = m := by omega
      rw [hc0m]
      have hreal := diag_identity m 0 r (Fintype.card K) hr hrq
      have hcast := congrArg (fun t : ℝ => (t : ℂ)) hreal
      push_cast at hcast ⊢
      linear_combination ((((Real.sqrt (m.choose 0) : ℝ) : ℂ))⁻¹ *
        ∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) * hcast
    · have e1 : (hammingNorm y = k) = True := eq_true h1
      have e2 : (hammingNorm y + 1 = k) = False := eq_false (by omega)
      have e3 : (hammingNorm y = k + 1) = False := eq_false (by omega)
      have e4 : (hammingNorm y = k - 1) = False := eq_false (by omega)
      simp only [e1, e2, e3, e4, if_true, if_false, add_zero, zero_add, mul_zero]
      rw [h1]
      have hnat : (Finset.univ.filter fun j : Fin m => y j = 0).card + k = m := by omega
      have hmC : ((Finset.univ.filter fun j : Fin m => y j = 0).card : ℂ) + (k : ℂ)
          = (m : ℂ) := by exact_mod_cast hnat
      have hc0R : ((Finset.univ.filter fun j : Fin m => y j = 0).card : ℝ)
          = (m : ℝ) - k := by
        have h2 : ((Finset.univ.filter fun j : Fin m => y j = 0).card : ℝ) + (k : ℝ)
            = (m : ℝ) := by exact_mod_cast hnat
        linarith
      have hreal := diag_identity m k r (Fintype.card K) hr hrq
      rw [← hc0R] at hreal
      have hcast := congrArg (fun t : ℝ => (t : ℂ)) hreal
      push_cast at hcast ⊢
      linear_combination
        ((((Real.sqrt (m.choose k) : ℝ) : ℂ))⁻¹ *
          ∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) * hcast
        - ((((Real.sqrt (m.choose k) : ℝ) : ℂ))⁻¹ *
          ∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) * hmC
  by_cases h2 : hammingNorm y + 1 = k
  · -- Lower off-diagonal: the input layer sits one step above the output layer.
    subst h2
    have e1 : (hammingNorm y = hammingNorm y + 1) = False := eq_false (by omega)
    have e3 : (hammingNorm y = hammingNorm y + 1 + 1) = False := eq_false (by omega)
    simp only [e1, e3, if_true, if_false, add_zero, zero_add, mul_zero,
      sub_zero, Nat.add_sub_cancel]
    have hrad : Real.sqrt (((hammingNorm y + 1 : ℕ) : ℝ) *
          ((m : ℝ) - ((hammingNorm y + 1 : ℕ) : ℝ) + 1))
        = Real.sqrt ((((hammingNorm y : ℕ) : ℝ) + 1) *
          ((m : ℝ) - ((hammingNorm y : ℕ) : ℝ))) := by
      congr 1
      push_cast
      ring
    rw [hrad]
    have hreal := offdiag_low m (hammingNorm y) hk
    have hcast := congrArg (fun t : ℝ => (t : ℂ)) hreal
    have hmCC : ((Finset.univ.filter fun j : Fin m => y j = 0).card : ℂ)
        = (m : ℂ) - (hammingNorm y : ℂ) := by
      have h2 : ((Finset.univ.filter fun j : Fin m => y j = 0).card : ℂ)
          + (hammingNorm y : ℂ) = (m : ℂ) := by exact_mod_cast hc0
      linear_combination h2
    push_cast at hcast ⊢
    linear_combination
      (2 * ((cc1rq r (Fintype.card K) : ℝ) : ℂ) *
        ∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) * hcast
      + (2 * ((cc1rq r (Fintype.card K) : ℝ) : ℂ) *
        (∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) *
        (((Real.sqrt (m.choose (hammingNorm y + 1)) : ℝ) : ℂ))⁻¹) * hmCC
  by_cases h3 : hammingNorm y = k + 1
  · -- Upper off-diagonal: the input layer sits one step below the output layer.
    have e1 : (hammingNorm y = k) = False := eq_false h1
    have e2 : (hammingNorm y + 1 = k) = False := eq_false h2
    have e3 : (hammingNorm y = k + 1) = True := eq_true h3
    have e4 : (hammingNorm y = k - 1) = False := eq_false (by omega)
    simp only [e1, e2, e3, e4, if_true, if_false, add_zero, zero_add, mul_zero,
      sub_zero]
    rw [h3]
    have hreal := offdiag_high m k (by omega)
    have hcast := congrArg (fun t : ℝ => (t : ℂ)) hreal
    push_cast at hcast ⊢
    linear_combination
      (2 * ((cc1rq r (Fintype.card K) : ℝ) : ℂ) *
        ∏ j : Fin m, if y j = 0 then 1 else amp χ Fs j (y j)) * hcast
  · -- The input layer is more than one step away: both sides vanish.
    have e1 : (hammingNorm y = k) = False := eq_false h1
    have e2 : (hammingNorm y + 1 = k) = False := eq_false h2
    have e3 : (hammingNorm y = k + 1) = False := eq_false h3
    have e4 : (hammingNorm y = k - 1) = False := eq_false (by omega)
    simp only [e1, e2, e3, e4, if_false, add_zero, mul_zero, sub_zero]

end Shell

end Dqi
