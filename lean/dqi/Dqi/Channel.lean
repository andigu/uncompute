import Dqi.Reachable
import Dqi.Failure

/-!
# Channel blindness

The abstract core of the reachable-residual proposition: **a perturbation can
only see the part of a residual that lives on coordinates the perturbation
actually populates.**  Splitting the residual into two pieces with disjoint
support, an error confined to the first piece's coordinates is blind to the
second — this is `norm_inner_le_of_avoids`, the "blindness" lemma: a sign or
leakage error keeps a decodable syndrome, so it cannot overlap the escaping
weight layer at all, and the bound sees only the decodable part of the
residual.

`norm_restr_sq_eq_mass` ties the failure mass to the perturbation norm,
`norm_inner_le_of_split` is the relocation version (the perturbation may touch
the second component, but only through its reachable part), and
`norm_inner_le_sum` splits a perturbation into channels bounded separately —
the step that puts `μ_cl` rather than the total `μ` on the relocation term.
-/

namespace Dqi

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Mass is a restricted norm.**  The squared norm of a restriction is the
failure mass of the restricting set. -/
theorem norm_restr_sq_eq_mass (Ψ : EuclideanSpace ℂ ι) (F : Finset ι) :
    ‖restr F Ψ‖ ^ 2 = mass Ψ F := by
  rw [EuclideanSpace.norm_sq_eq, mass]
  calc ∑ i, ‖restr F Ψ i‖ ^ 2
      = ∑ i, if i ∈ F then ‖Ψ i‖ ^ 2 else 0 := by
        refine Finset.sum_congr rfl fun i _ => ?_
        by_cases h : i ∈ F <;> simp [h]
    _ = ∑ i ∈ F, ‖Ψ i‖ ^ 2 := by rw [Finset.sum_ite_mem, Finset.univ_inter]

/-- The norm of a restriction is the square root of the mass. -/
theorem norm_restr_eq_sqrt_mass (Ψ : EuclideanSpace ℂ ι) (F : Finset ι) :
    ‖restr F Ψ‖ = Real.sqrt (mass Ψ F) := by
  rw [← norm_restr_sq_eq_mass, Real.sqrt_sq (norm_nonneg _)]

omit [Fintype ι] in
/-- Restricting to coordinates that avoid the second component drops it. -/
theorem restr_add_of_avoids (S : Finset ι) (X₁ X₂ : EuclideanSpace ℂ ι)
    (h : ∀ i ∈ S, X₂ i = 0) :
    restr S (X₁ + X₂) = restr S X₁ := by
  refine PiLp.ext fun i => ?_
  by_cases hi : i ∈ S
  · simp [hi, h i hi]
  · simp [hi]

/-- **Blindness, norm form.**  An error confined to `S` cannot see a component
vanishing on `S`: the restricted norm of the sum is bounded by the first piece
alone. -/
theorem norm_restr_add_le_of_avoids (S : Finset ι) (X₁ X₂ : EuclideanSpace ℂ ι)
    (h : ∀ i ∈ S, X₂ i = 0) :
    ‖restr S (X₁ + X₂)‖ ≤ ‖X₁‖ := by
  rw [restr_add_of_avoids S X₁ X₂ h]
  exact norm_restr_le S X₁

/-- **The blindness bound.**  A perturbation whose support avoids `X₂` sees
only `X₁`: its inner product with the split residual is bounded by the
decodable part alone. -/
theorem norm_inner_le_of_avoids (D X₁ X₂ : EuclideanSpace ℂ ι)
    (h : ∀ i ∈ supp D, X₂ i = 0) :
    ‖inner ℂ D (X₁ + X₂)‖ ≤ ‖D‖ * ‖X₁‖ :=
  norm_inner_le_of_restr D (X₁ + X₂) ‖X₁‖ (norm_restr_add_le_of_avoids _ _ _ h)

omit [Fintype ι] in
/-- Restriction is additive. -/
theorem restr_add (S : Finset ι) (v w : EuclideanSpace ℂ ι) :
    restr S (v + w) = restr S v + restr S w := by
  refine PiLp.ext fun i => ?_
  by_cases hi : i ∈ S <;> simp [hi]

/-- **The relocation version.**  When the perturbation can touch the second
component, the bound picks up only its reachable part. -/
theorem norm_inner_le_of_split (D X₁ X₂ : EuclideanSpace ℂ ι) (rho2 : ℝ)
    (h : ‖restr (supp D) X₂‖ ≤ rho2) :
    ‖inner ℂ D (X₁ + X₂)‖ ≤ ‖D‖ * (‖X₁‖ + rho2) := by
  refine norm_inner_le_of_restr D (X₁ + X₂) _ ?_
  calc ‖restr (supp D) (X₁ + X₂)‖
      = ‖restr (supp D) X₁ + restr (supp D) X₂‖ := by rw [restr_add]
    _ ≤ ‖restr (supp D) X₁‖ + ‖restr (supp D) X₂‖ := norm_add_le _ _
    _ ≤ ‖X₁‖ + rho2 := add_le_add (norm_restr_le _ _) h

omit [DecidableEq ι] in
/-- **Channel splitting.**  A perturbation decomposed into channels contributes
additively: each channel is bounded separately and the bounds add. -/
theorem norm_inner_le_sum {n : ℕ} (Ds : Fin n → EuclideanSpace ℂ ι)
    (X : EuclideanSpace ℂ ι) (bs : Fin n → ℝ)
    (h : ∀ c, ‖inner ℂ (Ds c) X‖ ≤ bs c) :
    ‖inner ℂ (∑ c, Ds c) X‖ ≤ ∑ c, bs c := by
  rw [sum_inner]
  exact (norm_sum_le _ _).trans (Finset.sum_le_sum fun c _ => h c)

end Dqi
