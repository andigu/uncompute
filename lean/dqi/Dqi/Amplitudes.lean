import Dqi.Fourier
import Dqi.Flat
import Dqi.LayerState

/-!
# The DQI amplitude family

`LayerState` proves orthonormality of the weight-layer states for an *abstract*
family of coordinate amplitudes `g`, assuming only `g i 0 = 0` and
`∑ a, ‖g i a‖ ^ 2 = 1`.  `Fourier` proves exactly those two facts for the
concrete DQI amplitudes `ghat`, obtained by transporting the normalized centred
indicator through the character transform.

This file joins the two: the amplitude family of an actual instance is `amp`,
and `layerState_amp_inner` says its layer states are orthonormal.  Neither
normalization fact is a hypothesis of the development — both are theorems
about the real object.

What remains hypothesised about the amplitudes is only *flatness*
(`‖ghat χ F a‖ = (q-1)^(-1/2)` for `a ≠ 0`), which is a property of the bent
target-set construction rather than of amplitudes in general, and which enters
the final bound as a cited hypothesis.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]
variable {m : ℕ}

/-- The coordinate amplitude family of a DQI instance: one `ghat` per
constraint, determined by that constraint's target set. -/
noncomputable def amp (χ : AddChar K ℂ) (Fs : Fin m → Finset K) (i : Fin m) : K → ℂ :=
  ghat χ (Fs i)

/-- The amplitude at the zero frequency vanishes: the `hg0` hypothesis of
`layerState_inner`, discharged. -/
theorem amp_zero (χ : AddChar K ℂ) (Fs : Fin m → Finset K) (i : Fin m) :
    amp χ Fs i 0 = 0 :=
  ghat_zero χ (Fs i)

/-- Each coordinate's amplitudes are normalized: the `hgnorm` hypothesis of
`layerState_inner`, discharged. -/
theorem sum_normSq_amp (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (hr : ∀ i, 0 < (Fs i).card) (hrq : ∀ i, (Fs i).card < Fintype.card K) (i : Fin m) :
    ∑ a : K, ‖amp χ Fs i a‖ ^ 2 = 1 :=
  sum_normSq_ghat χ hχ (Fs i) (hr i) (hrq i)

/-- **The weight-layer states of an actual DQI instance are orthonormal.**

This is `layerState_inner` with its two abstract hypotheses supplied by
`Fourier`, so it is a statement about the real amplitude family rather than an
assumed one. -/
theorem layerState_amp_inner (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (hr : ∀ i, 0 < (Fs i).card) (hrq : ∀ i, (Fs i).card < Fintype.card K)
    {j k : ℕ} (hj : j ≤ m) (hk : k ≤ m) :
    inner ℂ (layerState (amp χ Fs) j) (layerState (amp χ Fs) k) = if j = k then 1 else 0 :=
  layerState_inner (amp χ Fs) (amp_zero χ Fs) (sum_normSq_amp χ hχ Fs hr hrq) hj hk

/-- **Flatness of the amplitude family, from constancy alone.**

`Flat.norm_ghat_of_const` derives the value `(q-1)^(-1/2)` from normalization, so
what the development assumes about the target sets is only that each one's
nonzero Fourier magnitudes are *equal* — the definition of a bent function. This
is hypothesis (iii) of the paper, in its weakest usable form. -/
theorem amp_flat_of_const (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (hr : ∀ i, 0 < (Fs i).card) (hrq : ∀ i, (Fs i).card < Fintype.card K)
    (hconst : ∀ i, ∀ a b : K, a ≠ 0 → b ≠ 0 →
      ‖ghat χ (Fs i) a‖ = ‖ghat χ (Fs i) b‖) :
    ∀ i, ∀ a : K, a ≠ 0 →
      ‖amp χ Fs i a‖ = (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹ :=
  fun i a ha => norm_ghat_of_const χ hχ (Fs i) (hr i) (hrq i) (hconst i) a ha

end Dqi
