import Dqi.Main

/-!
# The boundary-regime master theorem

The boundary variant of the master theorem, for the regime `2*l + 1 = dualDist B`
where the transfer to the physical syndrome register acquires a defect operator
whose norm is at most the paper's boundary correction `R_C` (its counting core is
`Dqi.Boundary`; the defect operator itself is deliberately not constructed — it
contributes below `10⁻¹⁷` for the `dqi-reed-solomon` task instance).

Accordingly, the residual split (`hsplit`) and the widened decodable bound
(`hX₁`, with `sigmaDec` enlarged by the cited constant `rc`) are **hypotheses**
here, discharged by citation of the paper's boundary-correction bound `R_C`;
everything downstream of those two hypotheses is proved.  The strict theorem
`dqi_degradation_strict` (in `Dqi.Main`) *derives* both instead of assuming them.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m : ℕ}
variable (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
variable (r : ℕ) (hcard : ∀ i, (Fs i).card = r) (hr : 0 < r) (hrq : r < Fintype.card K)

include hχ hcard hr hrq

/-- **The boundary-regime master theorem.**  The channel-resolved degradation
bound at the boundary `2*l + 1 = dualDist B`: identical in shape to
`dqi_degradation_strict`, except that the residual split `hsplit` and the
decodable bound `hX₁` — widened by the boundary correction `rc` (the paper's
`R_C`, counting core in `Dqi.Boundary`) — are taken as hypotheses, discharged
by citation.  Everything downstream of those two hypotheses is proved. -/
theorem dqi_degradation_boundary
    (w : ℕ → ℝ) (l : ℕ) (hl : l < m)
    (hw : ∑ k ∈ Finset.range (l + 1), w k ^ 2 = 1)
    (D Ds Dl Dr : EuclideanSpace ℂ (Pattern m K))
    (hchan : D = Ds + Dl + Dr)
    (hΨD : ‖dqiState (amp χ Fs) w l + D‖ = 1)
    (X₁ X₂ : EuclideanSpace ℂ (Pattern m K))
    (hsplit : shellOpL χ Fs (dqiState (amp χ Fs) w l)
        - ((lamF m r (Fintype.card K) w l : ℝ) : ℂ) • dqiState (amp χ Fs) w l
      = X₁ + X₂)
    (rc : ℝ) (hX₁ : ‖X₁‖ ≤ sigmaDec m r (Fintype.card K) w l + rc)
    (hblindS : ∀ y ∈ supp Ds, X₂ y = 0)
    (hblindL : ∀ y ∈ supp Dl, X₂ y = 0)
    (erel : ℝ) (hreach : ‖restr (supp Dr) X₂‖ ≤ erel)
    (mu mucl : ℝ) (hmu : 0 ≤ mu)
    (hDn : ‖D‖ ≤ 2 * Real.sqrt mu) (hDrn : ‖Dr‖ ≤ Real.sqrt mucl) :
    |expct (shellOpL χ Fs) (dqiState (amp χ Fs) w l + D)
        - lamF m r (Fintype.card K) w l|
      ≤ (|lamF m r (Fintype.card K) w l| + m) * (4 * mu)
        + 4 * Real.sqrt mu * (sigmaDec m r (Fintype.card K) w l + rc)
        + 2 * Real.sqrt mucl * erel := by
  have hexp : expct (shellOpL χ Fs) (dqiState (amp χ Fs) w l)
      = lamF m r (Fintype.card K) w l :=
    expct_shellOpL χ hχ Fs r hcard hr hrq w l hl hw
  rw [← hexp] at hsplit
  have hmain := degradation_channels (shellOpL χ Fs) (isSelfAdjoint_shellOpL χ Fs)
    (dqiState (amp χ Fs) w l) D Ds Dl Dr
    (norm_dqiState χ hχ Fs r hcard hr hrq w l hl hw) hΨD hchan
    (m : ℝ) (norm_shellOpL_le χ hχ Fs) X₁ X₂ hsplit
    (sigmaDec m r (Fintype.card K) w l + rc) erel hX₁
    hblindS hblindL hreach
    mu mucl hmu hDn hDrn
  rwa [hexp] at hmain

end Dqi
