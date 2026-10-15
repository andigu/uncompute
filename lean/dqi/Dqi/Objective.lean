import Dqi.Characters
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Topology.Algebra.Module.FiniteDimension

/-!
# Objective-operator ingredients: dual shifts and bounded multipliers

Two independent facts feeding the DQI objective analysis.

**Shift structure.**  A character-weighted multiplication operator in position
space acts as a *shift* in the dual (syndrome) index.  Concretely,
`Dqi.sum_char_shift` upgrades the orthogonality delta `Dqi.sum_char_dot` to the
statement that the character sum with argument `∑ j, (u j + v j - u' j) * x j`
vanishes unless `u' = u + v`, and `Dqi.matElt_eq_zero_of_not_mem_range` deduces
that the matrix element of the position-space multiplier
`x ↦ ∑ a, c a * χ (a * (b ⬝ x))` between dual indices `u` and `u'` vanishes
whenever `u' - u` is not a scalar multiple `a • b` of the row `b`.  This is the
support condition forced by the syndrome-pullback step.

**Bounded multipliers.**  Multiplication by a real-valued function `g` on
`EuclideanSpace ℂ ι` is a self-adjoint continuous linear operator
(`Dqi.isSelfAdjoint_mulOp`) whose operator norm is at most any uniform bound
`m` on `|g|` (`Dqi.norm_mulOp_le`; the hypothesis `0 ≤ m` is taken explicitly,
since an empty index type would otherwise make it underivable).

Main results:
* `Dqi.sum_char_shift`: `∑ x, χ (∑ j, (u j + v j - u' j) * x j)
    = if u' = u + v then #K ^ n else 0`.
* `Dqi.matElt_eq_zero_of_not_mem_range`: the multiplier matrix element
  `matElt χ c b u u'` vanishes when `u' - u ∉ {a • b | a : K}`.
* `Dqi.mulOp`, `Dqi.mulOp_apply`: the multiplication operator on
  `EuclideanSpace ℂ ι` and its pointwise action.
* `Dqi.isSelfAdjoint_mulOp`: `mulOp g` is self-adjoint for real-valued `g`.
* `Dqi.norm_mulOp_le`: `‖mulOp g‖ ≤ m` when `0 ≤ m` and `∀ i, |g i| ≤ m`.
-/

namespace Dqi

section Shift

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- Shift structure of the dual index: the character sum against the vector
`u + v - u'` is a delta at `u' = u + v`.  This is `sum_char_dot` applied to
`u + v - u'`, and it is what makes a character-weighted multiplication operator
a shift by `v` in the dual index. -/
theorem sum_char_shift (χ : AddChar K ℂ) (hχ : χ ≠ 1) {n : ℕ} (u u' v : Fin n → K) :
    ∑ x : Fin n → K, χ (∑ j, (u j + v j - u' j) * x j) =
      if u' = u + v then (Fintype.card K : ℂ) ^ n else 0 := by
  have h := sum_char_dot χ hχ (u + v - u')
  simp only [Pi.add_apply, Pi.sub_apply, sub_eq_zero] at h
  rw [h]
  exact if_congr eq_comm rfl rfl

omit [DecidableEq K] in
/-- The matrix element, between dual indices `u` and `u'`, of the position-space
multiplier `x ↦ ∑ a, c a * χ (a * ∑ j, b j * x j)` attached to the coefficient
family `c : K → ℂ` and the row `b : Fin n → K`. -/
noncomputable def matElt (χ : AddChar K ℂ) {n : ℕ} (c : K → ℂ) (b : Fin n → K)
    (u u' : Fin n → K) : ℂ :=
  ∑ x : Fin n → K,
    (∑ a : K, c a * χ (a * ∑ j, b j * x j)) * χ (∑ j, (u j - u' j) * x j)

/-- Support condition forced by the syndrome pullback: the multiplier matrix
element between dual indices `u` and `u'` vanishes unless the shift `u' - u` is
a scalar multiple `a • b` of the row `b`. -/
theorem matElt_eq_zero_of_not_mem_range
    (χ : AddChar K ℂ) (hχ : χ ≠ 1) {n : ℕ} (c : K → ℂ) (b : Fin n → K) (u u' : Fin n → K)
    (h : ∀ a : K, u' - u ≠ a • b) :
    matElt χ c b u u' = 0 := by
  -- Merge the two character factors into a single dot-product argument.
  have merge : ∀ x : Fin n → K,
      (∑ a : K, c a * χ (a * ∑ j, b j * x j)) * χ (∑ j, (u j - u' j) * x j) =
        ∑ a : K, c a * χ (∑ j, (a * b j + (u j - u' j)) * x j) := by
    intro x
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun a _ ↦ ?_
    rw [mul_assoc, ← AddChar.map_add_eq_mul]
    congr 1
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact congrArg χ (Finset.sum_congr rfl fun j _ ↦ by ring)
  -- Each inner sum over `x` is a `sum_char_dot` delta at `u' - u = a • b`,
  -- which the hypothesis rules out.
  have hzero : ∀ a : K,
      ∑ x : Fin n → K, χ (∑ j, (a * b j + (u j - u' j)) * x j) = 0 := by
    intro a
    have hv : (fun j : Fin n ↦ a * b j + (u j - u' j)) ≠ 0 := by
      intro h0
      refine h a ?_
      funext j
      have hj := congrFun h0 j
      simp only [Pi.zero_apply] at hj
      rw [add_eq_zero_iff_eq_neg, neg_sub] at hj
      simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
      exact hj.symm
    have hd := sum_char_dot χ hχ (fun j : Fin n ↦ a * b j + (u j - u' j))
    rw [if_neg hv] at hd
    exact hd
  calc matElt χ c b u u'
      = ∑ x : Fin n → K, ∑ a : K, c a * χ (∑ j, (a * b j + (u j - u' j)) * x j) :=
        Finset.sum_congr rfl fun x _ ↦ merge x
    _ = ∑ a : K, ∑ x : Fin n → K, c a * χ (∑ j, (a * b j + (u j - u' j)) * x j) :=
        Finset.sum_comm
    _ = 0 := Finset.sum_eq_zero fun a _ ↦ by rw [← Finset.mul_sum, hzero a, mul_zero]

end Shift

section MulOp

variable {ι : Type*} [Fintype ι]

/-- Multiplication by the real-valued function `g`, as a continuous linear
operator on `EuclideanSpace ℂ ι`: it sends `ψ` to `fun i ↦ (g i : ℂ) * ψ i`. -/
noncomputable def mulOp (g : ι → ℝ) : EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι :=
  LinearMap.toContinuousLinearMap
    { toFun := fun ψ ↦ WithLp.toLp 2 fun i ↦ (g i : ℂ) * ψ i
      map_add' := fun ψ φ ↦ PiLp.ext fun i ↦ by
        simp only [PiLp.add_apply, mul_add]
      map_smul' := fun z ψ ↦ PiLp.ext fun i ↦ by
        simp only [PiLp.smul_apply, smul_eq_mul, RingHom.id_apply]
        ring }

@[simp] theorem mulOp_apply (g : ι → ℝ) (ψ : EuclideanSpace ℂ ι) (i : ι) :
    mulOp g ψ i = (g i : ℂ) * ψ i :=
  rfl

/-- Multiplication by a real-valued function is self-adjoint. -/
theorem isSelfAdjoint_mulOp (g : ι → ℝ) : IsSelfAdjoint (mulOp g) := by
  refine LinearMap.IsSymmetric.isSelfAdjoint ?_
  intro ψ φ
  simp only [ContinuousLinearMap.coe_coe, PiLp.inner_apply, mulOp_apply, RCLike.inner_apply,
    map_mul, Complex.conj_ofReal]
  exact Finset.sum_congr rfl fun i _ ↦ by ring

/-- A uniform bound on `|g|` bounds the operator norm of `mulOp g`.  The
hypothesis `0 ≤ m` is taken explicitly: for an empty index type it does not
follow from `∀ i, |g i| ≤ m`. -/
theorem norm_mulOp_le (g : ι → ℝ) (m : ℝ) (hm : 0 ≤ m) (hg : ∀ i, |g i| ≤ m) :
    ‖mulOp g‖ ≤ m := by
  refine ContinuousLinearMap.opNorm_le_bound _ hm fun ψ ↦ ?_
  have hsum : ∑ i, ‖mulOp g ψ i‖ ^ 2 ≤ m ^ 2 * ∑ i, ‖ψ i‖ ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ ↦ ?_
    rw [mulOp_apply, norm_mul, Complex.norm_real, Real.norm_eq_abs, mul_pow]
    refine mul_le_mul_of_nonneg_right ?_ (sq_nonneg _)
    nlinarith [abs_nonneg (g i), hg i]
  calc ‖mulOp g ψ‖
      = Real.sqrt (∑ i, ‖mulOp g ψ i‖ ^ 2) := EuclideanSpace.norm_eq _
    _ ≤ Real.sqrt (m ^ 2 * ∑ i, ‖ψ i‖ ^ 2) := Real.sqrt_le_sqrt hsum
    _ = m * Real.sqrt (∑ i, ‖ψ i‖ ^ 2) := by
        rw [Real.sqrt_mul (sq_nonneg m), Real.sqrt_sq hm]
    _ = m * ‖ψ‖ := by rw [EuclideanSpace.norm_eq]

end MulOp

end Dqi
