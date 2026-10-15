import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Complex.Norm

/-!
# The master degradation bound

The change in the expectation `⟪Ψ, M Ψ⟫` of a self-adjoint operator `M` under a
norm-preserving perturbation `Ψ ↦ Ψ + D` is second order in `‖D‖`, except for a
single first-order cross term controlled by how much of the residual
`(M - lam) Ψ` the perturbation can reach: if `‖⟪D, (M - lam) Ψ⟫‖ ≤ ‖D‖ * rho`
then

`|⟪Ψ + D, M (Ψ + D)⟫.re - lam| ≤ (|lam| + ‖M‖) * ‖D‖² + 2 * ‖D‖ * rho`,

where `lam = ⟪Ψ, M Ψ⟫.re` is the ideal expectation.  The point of the
recentring is that `rho` can be far smaller than `‖M‖`: the perturbation only
pays for the part of the residual vector it actually overlaps.

The companion lemma `norm_le_two_sqrt` is the triangle-inequality step
`‖D‖ ≤ 2 √μ` used when `D` is the difference of two failure components each of
mass at most `μ`.
-/

namespace Dqi

/-- **Master degradation bound.**  Let `M` be self-adjoint, `‖Ψ‖ = 1`, and
`‖Ψ + D‖ = 1` (the perturbed state is again normalized, as when the perturbation
comes from an isometry).  Write `lam = ⟪Ψ, M Ψ⟫.re`.  If `‖M‖ ≤ m` and the
recentred cross term satisfies `‖⟪D, M Ψ - lam • Ψ⟫‖ ≤ ‖D‖ * rho`, then the
expectation of `M` in the perturbed state differs from `lam` by at most
`(|lam| + m) * ‖D‖² + 2 * ‖D‖ * rho`. -/
theorem expectation_perturbation_bound
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [CompleteSpace E]
    (M : E →L[ℂ] E) (hMsa : IsSelfAdjoint M) (Ψ D : E) (m rho : ℝ)
    (hΨ : ‖Ψ‖ = 1) (hΨD : ‖Ψ + D‖ = 1) (hM : ‖M‖ ≤ m)
    (hcross :
      ‖inner ℂ D (M Ψ - (((inner ℂ Ψ (M Ψ)).re : ℝ) : ℂ) • Ψ)‖ ≤ ‖D‖ * rho) :
    |(inner ℂ (Ψ + D) (M (Ψ + D))).re - (inner ℂ Ψ (M Ψ)).re| ≤
      (|(inner ℂ Ψ (M Ψ)).re| + m) * ‖D‖ ^ 2 + 2 * ‖D‖ * rho := by
  set lam : ℝ := (inner ℂ Ψ (M Ψ)).re with hlam
  -- Step 1: normalization identity `2 * ⟪Ψ, D⟫.re = -‖D‖²`.
  have hnorm : 2 * (inner ℂ Ψ D).re = -‖D‖ ^ 2 := by
    have h := norm_add_sq (𝕜 := ℂ) Ψ D
    rw [hΨ, hΨD] at h
    simp only [one_pow, RCLike.re_to_complex] at h
    linarith
  -- Self-adjointness: the two cross terms are complex conjugates.
  have hsymm : (M : E →ₗ[ℂ] E).IsSymmetric := hMsa.isSymmetric
  have hsym : inner ℂ Ψ (M D) = starRingEnd ℂ (inner ℂ D (M Ψ)) := by
    have h1 : inner ℂ (M Ψ) D = inner ℂ Ψ (M D) := hsymm Ψ D
    have h2 : starRingEnd ℂ (inner ℂ D (M Ψ)) = inner ℂ (M Ψ) D :=
      inner_conj_symm (𝕜 := ℂ) (M Ψ) D
    rw [← h1, h2]
  -- Step 2: expand the perturbed expectation.
  have hexpand : (inner ℂ (Ψ + D) (M (Ψ + D))).re
      = lam + 2 * (inner ℂ D (M Ψ)).re + (inner ℂ D (M D)).re := by
    rw [map_add, inner_add_left, inner_add_right, inner_add_right, hsym]
    simp only [Complex.add_re, Complex.conj_re]
    rw [hlam]
    ring
  -- Step 1 again, conjugated: `⟪D, Ψ⟫.re = -‖D‖²/2`.
  have hDPsi : (inner ℂ D Ψ).re = -‖D‖ ^ 2 / 2 := by
    have h1 : (inner ℂ D Ψ).re = (inner ℂ Ψ D).re := by
      rw [← inner_conj_symm (𝕜 := ℂ) Ψ D, Complex.conj_re]
    rw [h1]
    linarith
  -- Step 3: recentre the cross term around `lam • Ψ`.
  have hrecenter : 2 * (inner ℂ D (M Ψ)).re
      = 2 * (inner ℂ D (M Ψ - (lam : ℂ) • Ψ)).re - lam * ‖D‖ ^ 2 := by
    have h1 : inner ℂ D (M Ψ)
        = inner ℂ D (M Ψ - (lam : ℂ) • Ψ) + (lam : ℂ) * inner ℂ D Ψ := by
      rw [inner_sub_right, inner_smul_right]
      ring
    rw [h1]
    simp only [Complex.add_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
      zero_mul, sub_zero]
    rw [hDPsi]
    ring
  -- The exact decomposition of the deviation.
  have hdiff : (inner ℂ (Ψ + D) (M (Ψ + D))).re - lam
      = 2 * (inner ℂ D (M Ψ - (lam : ℂ) • Ψ)).re - lam * ‖D‖ ^ 2
        + (inner ℂ D (M D)).re := by
    rw [hexpand]
    linarith [hrecenter]
  -- Step 4: bound each piece.
  have hc : |(inner ℂ D (M Ψ - (lam : ℂ) • Ψ)).re| ≤ ‖D‖ * rho :=
    (Complex.abs_re_le_norm _).trans hcross
  have hq : |(inner ℂ D (M D)).re| ≤ m * ‖D‖ ^ 2 := by
    have h2 : ‖M D‖ ≤ m * ‖D‖ :=
      (M.le_opNorm D).trans (mul_le_mul_of_nonneg_right hM (norm_nonneg D))
    calc |(inner ℂ D (M D)).re| ≤ ‖inner ℂ D (M D)‖ := Complex.abs_re_le_norm _
      _ ≤ ‖D‖ * ‖M D‖ := norm_inner_le_norm (𝕜 := ℂ) D (M D)
      _ ≤ ‖D‖ * (m * ‖D‖) := mul_le_mul_of_nonneg_left h2 (norm_nonneg D)
      _ = m * ‖D‖ ^ 2 := by ring
  have hl1 : lam * ‖D‖ ^ 2 ≤ |lam| * ‖D‖ ^ 2 :=
    mul_le_mul_of_nonneg_right (le_abs_self lam) (sq_nonneg ‖D‖)
  have hl2 : -(|lam| * ‖D‖ ^ 2) ≤ lam * ‖D‖ ^ 2 := by
    have h := mul_le_mul_of_nonneg_right (neg_abs_le lam) (sq_nonneg ‖D‖)
    linarith
  obtain ⟨hc1, hc2⟩ := abs_le.mp hc
  obtain ⟨hq1, hq2⟩ := abs_le.mp hq
  rw [hdiff, abs_le]
  constructor <;> nlinarith [hc1, hc2, hq1, hq2, hl1, hl2]

/-- If two vectors each have norm at most `√μ`, their difference has norm at
most `2 √μ`.  This is the `‖D‖ ≤ 2 √μ` step: the perturbation is the difference
of two failure components each of mass at most `μ`. -/
theorem norm_le_two_sqrt {E : Type*} [NormedAddCommGroup E] (a b : E) (mu : ℝ)
    (ha : ‖a‖ ≤ Real.sqrt mu) (hb : ‖b‖ ≤ Real.sqrt mu) :
    ‖a - b‖ ≤ 2 * Real.sqrt mu :=
  (norm_sub_le a b).trans (by linarith)

end Dqi
