import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# The reachable residual

A perturbation `D` can only see the part of a residual vector that lives on the
computational-basis outputs `D` actually populates.  `restr S` models the
projector onto a set `S` of basis outputs, and `supp D` is the set a
perturbation populates.  The adjointness step `inner_eq_inner_restr_supp` says
an inner product against `D` only sees the restriction of the other vector to
`supp D`, and `norm_inner_le_of_restr` is the step that lets a large residual
contribute to a bound only through its reachable part: the Cauchy–Schwarz
constant is the norm of the *restricted* residual, not of the residual itself.

`norm_restr_union_sq_le` is the splitting `Q_D ≼ Π_Z + Π_{Z_U}` of the
reachable projector into the two layer projectors, and `restr_eq_zero` is the
orthogonality input: a vector vanishing on `S` has zero restriction to `S`, the
formal content of "the ideal images have decodable syndromes, so they cannot
overlap the edge component".
-/

namespace Dqi

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [Fintype ι] in
/-- Coordinate restriction to a `Finset`: the projector onto the
computational-basis outputs in `S`. -/
noncomputable def restr (S : Finset ι) (v : EuclideanSpace ℂ ι) : EuclideanSpace ℂ ι :=
  WithLp.toLp 2 fun i => if i ∈ S then v i else 0

omit [Fintype ι] in
@[simp]
theorem restr_apply (S : Finset ι) (v : EuclideanSpace ℂ ι) (i : ι) :
    restr S v i = if i ∈ S then v i else 0 := rfl

open scoped Classical in
/-- The support of a vector: the basis outputs it populates. -/
noncomputable def supp (v : EuclideanSpace ℂ ι) : Finset ι :=
  Finset.univ.filter fun i => v i ≠ 0

omit [DecidableEq ι] in
theorem mem_supp {v : EuclideanSpace ℂ ι} {i : ι} : i ∈ supp v ↔ v i ≠ 0 := by
  simp [supp]

omit [Fintype ι] in
/-- Restriction is idempotent. -/
theorem restr_restr (S : Finset ι) (v : EuclideanSpace ℂ ι) :
    restr S (restr S v) = restr S v := by
  refine PiLp.ext fun i => ?_
  by_cases h : i ∈ S <;> simp [h]

/-- Restriction can only shrink the norm. -/
theorem norm_restr_le (S : Finset ι) (v : EuclideanSpace ℂ ι) : ‖restr S v‖ ≤ ‖v‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  refine Real.sqrt_le_sqrt (Finset.sum_le_sum fun i _ => ?_)
  by_cases h : i ∈ S <;> simp [h]

/-- Restriction is monotone in the norm. -/
theorem norm_restr_mono {S T : Finset ι} (h : S ⊆ T) (v : EuclideanSpace ℂ ι) :
    ‖restr S v‖ ≤ ‖restr T v‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  refine Real.sqrt_le_sqrt (Finset.sum_le_sum fun i _ => ?_)
  by_cases hi : i ∈ S
  · simp [hi, h hi]
  · simp [hi]

/-- A vector is unchanged by restriction to its own support. -/
theorem restr_supp (v : EuclideanSpace ℂ ι) : restr (supp v) v = v := by
  refine PiLp.ext fun i => ?_
  by_cases h : v i = 0 <;> simp [mem_supp, h]

/-- **Adjointness.**  The inner product of `D` with any vector only sees the
restriction of that vector to the support of `D`. -/
theorem inner_eq_inner_restr_supp (D v : EuclideanSpace ℂ ι) :
    inner ℂ D v = inner ℂ D (restr (supp D) v) := by
  rw [PiLp.inner_apply, PiLp.inner_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases h : D i = 0
  · simp [h]
  · simp [mem_supp, h]

/-- **Cauchy–Schwarz through the reachable part.**  A bound `rho` on the
restricted residual bounds the whole inner product against `D`. -/
theorem norm_inner_le_of_restr (D v : EuclideanSpace ℂ ι) (rho : ℝ)
    (h : ‖restr (supp D) v‖ ≤ rho) :
    ‖inner ℂ D v‖ ≤ ‖D‖ * rho := by
  rw [inner_eq_inner_restr_supp]
  calc ‖inner ℂ D (restr (supp D) v)‖
      ≤ ‖D‖ * ‖restr (supp D) v‖ := norm_inner_le_norm (𝕜 := ℂ) D (restr (supp D) v)
    _ ≤ ‖D‖ * rho := mul_le_mul_of_nonneg_left h (norm_nonneg D)

/-- Splitting a restriction over a union: the reachable projector is dominated
by the sum of the two layer projectors. -/
theorem norm_restr_union_sq_le (S T : Finset ι) (v : EuclideanSpace ℂ ι) :
    ‖restr (S ∪ T) v‖ ^ 2 ≤ ‖restr S v‖ ^ 2 + ‖restr T v‖ ^ 2 := by
  have hsq : ∀ R : Finset ι, ‖restr R v‖ ^ 2 = ∑ i, ‖restr R v i‖ ^ 2 := fun R => by
    rw [EuclideanSpace.norm_eq,
      Real.sq_sqrt (Finset.sum_nonneg fun i _ => by positivity)]
  rw [hsq, hsq, hsq, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  by_cases hS : i ∈ S
  · by_cases hT : i ∈ T
    · simp only [restr_apply, Finset.mem_union, hS, hT, or_self, if_true]
      exact le_add_of_nonneg_right (by positivity)
    · simp [hS, hT]
  · by_cases hT : i ∈ T <;> simp [hS, hT]

omit [Fintype ι] in
/-- Orthogonality kills a restriction: a vector vanishing on `S` restricts to
zero on `S`. -/
theorem restr_eq_zero (S : Finset ι) (v : EuclideanSpace ℂ ι)
    (h : ∀ i ∈ S, v i = 0) : restr S v = 0 := by
  refine PiLp.ext fun i => ?_
  by_cases hi : i ∈ S <;> simp [hi, h i]

end Dqi
