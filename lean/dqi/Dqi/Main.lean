import Dqi.ShellOp
import Dqi.Degradation

/-!
# The strict-distance master theorem, assembled

This file plugs the finished pieces together: the abstract channel-resolved
degradation bound (`Dqi.Degradation.degradation_channels`) is instantiated at
the real objective operator (`Dqi.ShellOp.shellOpL`), the real DQI state
(`Dqi.Residual.dqiState`), and the real residual split
(`Dqi.Residual.shellOp_dqiState_residual`), which supplies the
`X₁ + X₂` decomposition as an *equality* — the decodable-layer combination
plus the single escaping edge component.

The result, `dqi_degradation_strict`, is the strict-distance master theorem:
the measured DQI objective moves from its ideal value `lamF` by at most

  `(|lamF| + m) · 4μ + 4√μ · σ_dec + 2√μ_cl · ε_rel`.

Its remaining hypotheses — `hΨD`, `hDn`, `hDrn`, `hreach` — are discharged
earlier in the development by the isometry, mass, and counting lemmas
(`norm_sub_le_two_sqrt_mass`, `norm_restr_imageBlock`,
`norm_le_of_count_bound`); the blindness hypotheses `hblindS` / `hblindL` say
exactly that sign and leakage errors do not populate weight-`(l+1)` patterns.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m : ℕ}
variable (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
variable (r : ℕ) (hcard : ∀ i, (Fs i).card = r) (hr : 0 < r) (hrq : r < Fintype.card K)

include hχ hcard hr hrq

/-! ### Parseval for layer combinations -/

/-- **Parseval for layer combinations.**  The norm of a real-weighted
combination of the orthonormal weight-layer states is the Euclidean norm of
its coefficient vector. -/
theorem norm_layer_combo (c : ℕ → ℝ) (l : ℕ) (hl : l < m) :
    ‖∑ j ∈ Finset.range (l + 1), ((c j : ℝ) : ℂ) • layerState (amp χ Fs) j‖
      = Real.sqrt (∑ j ∈ Finset.range (l + 1), c j ^ 2) := by
  have hr0 : ∀ i, 0 < (Fs i).card := fun i => by rw [hcard i]; exact hr
  have hrq0 : ∀ i, (Fs i).card < Fintype.card K := fun i => by rw [hcard i]; exact hrq
  have hlm : l ≤ m := le_of_lt hl
  rw [← dqiState_def]
  have hPP : inner ℂ (dqiState (amp χ Fs) c l) (dqiState (amp χ Fs) c l)
      = ((∑ j ∈ Finset.range (l + 1), c j ^ 2 : ℝ) : ℂ) := by
    nth_rewrite 2 [dqiState_def]
    rw [inner_sum]
    have hterm : ∀ k ∈ Finset.range (l + 1),
        inner ℂ (dqiState (amp χ Fs) c l) (((c k : ℝ) : ℂ) • layerState (amp χ Fs) k)
          = ((c k ^ 2 : ℝ) : ℂ) := by
      intro k hk
      have hkl : k ≤ l := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
      rw [inner_smul_right,
        inner_dqiState_layerState χ hχ Fs hr0 hrq0 c l hlm (le_trans hkl hlm),
        if_pos hkl, ← Complex.ofReal_mul, sq]
    rw [Finset.sum_congr rfl hterm, Complex.ofReal_sum]
  have hsq : ‖dqiState (amp χ Fs) c l‖ ^ 2 = ∑ j ∈ Finset.range (l + 1), c j ^ 2 := by
    rw [← inner_self_eq_norm_sq (𝕜 := ℂ), hPP]
    simp [-Complex.ofReal_pow]
  rw [← hsq, Real.sqrt_sq (norm_nonneg _)]

/-! ### The state is normalized -/

/-- The DQI state of a unit weight profile is a unit vector. -/
theorem norm_dqiState (w : ℕ → ℝ) (l : ℕ) (hl : l < m)
    (hw : ∑ k ∈ Finset.range (l + 1), w k ^ 2 = 1) :
    ‖dqiState (amp χ Fs) w l‖ = 1 := by
  rw [dqiState_def, norm_layer_combo χ hχ Fs r hcard hr hrq w l hl, hw, Real.sqrt_one]

/-! ### The ideal expectation is `lamF` -/

/-- The expectation of the objective operator in the DQI state is exactly the
finite-size objective value `lamF`. -/
theorem expct_shellOpL (w : ℕ → ℝ) (l : ℕ) (hl : l < m)
    (hw : ∑ k ∈ Finset.range (l + 1), w k ^ 2 = 1) :
    expct (shellOpL χ Fs) (dqiState (amp χ Fs) w l) = lamF m r (Fintype.card K) w l := by
  unfold expct
  rw [shellOpL_apply, inner_dqiState_shellOp χ hχ Fs r hcard hr hrq w l hl hw,
    Complex.ofReal_re]

/-! ### The decodable residual norm -/

/-- The norm of the decodable part of the residual: the Euclidean norm of the
tridiagonal-action coefficients on the layers `0, …, l`. -/
noncomputable def sigmaDec (m r q : ℕ) (w : ℕ → ℝ) (l : ℕ) : ℝ :=
  Real.sqrt (∑ j ∈ Finset.range (l + 1),
    (2 * cc1rq r q * (∑ k ∈ Finset.range (l + 1), Amat m r q j k * w k)
      + ((m : ℝ) * fbar r q - lamF m r q w l) * w j) ^ 2)

/-- The decodable component of the residual has norm exactly `sigmaDec`. -/
theorem norm_residual_dec (w : ℕ → ℝ) (l : ℕ) (hl : l < m) :
    ‖∑ j ∈ Finset.range (l + 1),
        ((2 * cc1rq r (Fintype.card K) *
              (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
            + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
          : ℝ) : ℂ) • layerState (amp χ Fs) j‖
      = sigmaDec m r (Fintype.card K) w l :=
  norm_layer_combo χ hχ Fs r hcard hr hrq
    (fun j => 2 * cc1rq r (Fintype.card K) *
        (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
      + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j) l hl

/-! ### The master theorem -/

/-- **The strict-distance master theorem.**  The channel-resolved degradation
bound, instantiated at the real objective, the real DQI state, and the real
residual split: the measured objective moves from `lamF` by at most a
second-order term in the total failure mass `mu`, a cross term against the
decodable residual `sigmaDec`, and a relocation term weighted by the clean
relocation mass `mucl` alone. -/
theorem dqi_degradation_strict
    (w : ℕ → ℝ) (l : ℕ) (hl : l < m)
    (hw : ∑ k ∈ Finset.range (l + 1), w k ^ 2 = 1)
    (D Ds Dl Dr : EuclideanSpace ℂ (Pattern m K))
    (hchan : D = Ds + Dl + Dr)
    (hΨD : ‖dqiState (amp χ Fs) w l + D‖ = 1)
    (hblindS : ∀ y ∈ supp Ds, hammingNorm y ≠ l + 1)
    (hblindL : ∀ y ∈ supp Dl, hammingNorm y ≠ l + 1)
    (erel : ℝ)
    (hreach : ‖restr (supp Dr)
        (((2 * cc1rq r (Fintype.card K) * w l *
            Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
          • layerState (amp χ Fs) (l + 1))‖ ≤ erel)
    (mu mucl : ℝ) (hmu : 0 ≤ mu)
    (hDn : ‖D‖ ≤ 2 * Real.sqrt mu) (hDrn : ‖Dr‖ ≤ Real.sqrt mucl) :
    |expct (shellOpL χ Fs) (dqiState (amp χ Fs) w l + D) - lamF m r (Fintype.card K) w l|
      ≤ (|lamF m r (Fintype.card K) w l| + m) * (4 * mu)
        + 4 * Real.sqrt mu * sigmaDec m r (Fintype.card K) w l
        + 2 * Real.sqrt mucl * erel := by
  have hexp : expct (shellOpL χ Fs) (dqiState (amp χ Fs) w l)
      = lamF m r (Fintype.card K) w l :=
    expct_shellOpL χ hχ Fs r hcard hr hrq w l hl hw
  -- The edge component vanishes on every pattern of Hamming weight other than `l + 1`.
  have hblind : ∀ E : EuclideanSpace ℂ (Pattern m K),
      (∀ y ∈ supp E, hammingNorm y ≠ l + 1) → ∀ y ∈ supp E,
        (((2 * cc1rq r (Fintype.card K) * w l *
            Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
          • layerState (amp χ Fs) (l + 1)) y = 0 := by
    intro E hE y hy
    simp [layerState_apply, hE y hy]
  -- The residual split, with the expectation rewritten to `lamF`.
  have hsplit : shellOpL χ Fs (dqiState (amp χ Fs) w l)
      - ((expct (shellOpL χ Fs) (dqiState (amp χ Fs) w l) : ℝ) : ℂ)
          • dqiState (amp χ Fs) w l
      = (∑ j ∈ Finset.range (l + 1),
          ((2 * cc1rq r (Fintype.card K) *
                (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
              + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
            : ℝ) : ℂ) • layerState (amp χ Fs) j)
        + ((2 * cc1rq r (Fintype.card K) * w l *
              Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
            • layerState (amp χ Fs) (l + 1) := by
    rw [shellOpL_apply, hexp]
    exact shellOp_dqiState_residual χ hχ Fs r hcard hr hrq w l hl
  have hmain := degradation_channels (shellOpL χ Fs) (isSelfAdjoint_shellOpL χ Fs)
    (dqiState (amp χ Fs) w l) D Ds Dl Dr
    (norm_dqiState χ hχ Fs r hcard hr hrq w l hl hw) hΨD hchan
    (m : ℝ) (norm_shellOpL_le χ hχ Fs) _ _ hsplit
    (sigmaDec m r (Fintype.card K) w l) erel
    (norm_residual_dec χ hχ Fs r hcard hr hrq w l hl).le
    (hblind Ds hblindS) (hblind Dl hblindL) hreach
    mu mucl hmu hDn hDrn
  rwa [hexp] at hmain

end Dqi
