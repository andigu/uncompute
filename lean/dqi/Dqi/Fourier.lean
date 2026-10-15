import Dqi.Characters

/-!
# The DQI coordinate amplitudes

The amplitude at frequency `a` is the normalized character sum
`(√(r(q-r)))⁻¹ ∑_{v ∈ F} χ(a v)`, guarded to vanish at `a = 0`.

The two results here are exactly the two hypotheses that
`Dqi.layerState_inner` (in `Dqi.LayerState`) requires of its abstract amplitude
family: `ghat_zero` supplies `hg0 : ∀ i, g i 0 = 0`, which the guard makes true
by definition, and `sum_normSq_ghat` supplies
`hgnorm : ∀ i, ∑ a, ‖g i a‖ ^ 2 = 1`.  So neither is an assumption of the
development — both are theorems about the real amplitude family, and
`Dqi.Amplitudes` joins them to the layer states.

`sum_normSq_ghat` is proved directly: after removing the guarded `a = 0` term
the double character sum over `F × F` collapses by orthogonality to
`q r - r² = r (q - r)`, cancelling the normalizer.  No unitarity of a Fourier
transform is needed, which is why none is developed here.
-/

namespace Dqi

section Fourier

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- The DQI coordinate amplitude at frequency `a`: the normalized character sum
`(√(r(q-r)))⁻¹ ∑_{v ∈ F} χ(a v)` on nonzero frequencies, guarded to vanish at
`a = 0`.  The guard is deliberate: it makes `ghat_zero` hold by definition,
which is one of the two hypotheses of `Dqi.layerState_inner`. -/
noncomputable def ghat (χ : AddChar K ℂ) (F : Finset K) (a : K) : ℂ :=
  if a = 0 then 0
  else (((Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹ : ℝ) : ℂ) *
    ∑ v ∈ F, χ (a * v)

/-- The guard of `ghat` in the form used by `Dqi.layerState_inner`: the
amplitude at frequency `0` vanishes. -/
theorem ghat_zero (χ : AddChar K ℂ) (F : Finset K) : ghat χ F 0 = 0 := by
  simp [ghat]

/-- The DQI coordinate amplitudes have unit `ℓ²`-mass: this is the `hgnorm`
hypothesis of `Dqi.layerState_inner`.  Proved directly: after removing the
guarded `a = 0` term, the double character sum over `F × F` collapses by
orthogonality to `q r - r² = r (q - r)`, cancelling the normalizer. -/
theorem sum_normSq_ghat (χ : AddChar K ℂ) (hχ : χ ≠ 1) (F : Finset K)
    (hr : 0 < F.card) (hrq : F.card < Fintype.card K) :
    ∑ a : K, ‖ghat χ F a‖ ^ 2 = 1 := by
  have hr0 : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hr
  have hqr : (0 : ℝ) < (Fintype.card K : ℝ) - (F.card : ℝ) := by
    have h : (F.card : ℝ) < (Fintype.card K : ℝ) := by exact_mod_cast hrq
    linarith
  have hX0 : (0 : ℝ) < (F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)) :=
    mul_pos hr0 hqr
  -- Pointwise: `conj (ghat a) * ghat a` is the guarded double character sum.
  have hpoint : ∀ a : K, (starRingEnd ℂ) (ghat χ F a) * ghat χ F a =
      if a = 0 then 0
      else ((((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)))⁻¹ : ℝ) : ℂ) *
        ∑ w ∈ F, ∑ v ∈ F, χ ((v - w) * a) := by
    intro a
    by_cases ha : a = 0
    · simp [ghat, ha]
    · rw [if_neg ha]
      simp only [ghat, if_neg ha]
      rw [map_mul, Complex.conj_ofReal, map_sum, mul_mul_mul_comm,
        ← Complex.ofReal_mul, ← mul_inv, Real.mul_self_sqrt hX0.le,
        Finset.sum_mul_sum]
      congr 1
      refine Finset.sum_congr rfl fun w _ => Finset.sum_congr rfl fun v _ => ?_
      rw [← AddChar.map_neg_eq_conj, ← AddChar.map_add_eq_mul]
      congr 1
      ring
  -- Trade the guard for subtracting the `a = 0` term of the unguarded sum.
  have hsplit : ∀ a : K,
      (if a = 0 then (0 : ℂ)
        else ((((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)))⁻¹ : ℝ) : ℂ) *
          ∑ w ∈ F, ∑ v ∈ F, χ ((v - w) * a)) =
      ((((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)))⁻¹ : ℝ) : ℂ) *
          ∑ w ∈ F, ∑ v ∈ F, χ ((v - w) * a) -
        (if a = 0 then
          ((((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)))⁻¹ : ℝ) : ℂ) *
            ∑ w ∈ F, ∑ v ∈ F, χ ((v - w) * a)
        else 0) := by
    intro a
    split_ifs <;> simp
  -- The unguarded double sum: orthogonality collapses the frequency sum to a
  -- delta on the diagonal of `F × F`, contributing `q` for each of the `r`
  -- diagonal pairs.
  have hDsum : ∑ a : K, ∑ w ∈ F, ∑ v ∈ F, χ ((v - w) * a) =
      (F.card : ℂ) * (Fintype.card K : ℂ) := by
    rw [Finset.sum_comm]
    have hw : ∀ w ∈ F, (∑ a : K, ∑ v ∈ F, χ ((v - w) * a)) = (Fintype.card K : ℂ) := by
      intro w hwF
      rw [Finset.sum_comm,
        Finset.sum_congr rfl fun v _ => sum_char_mul χ hχ (v - w)]
      simp_rw [sub_eq_zero]
      rw [Finset.sum_ite_eq' F w fun _ => (Fintype.card K : ℂ), if_pos hwF]
    rw [Finset.sum_congr rfl hw, Finset.sum_const, nsmul_eq_mul]
  -- The `a = 0` term of the unguarded sum is `r²`.
  have hD0 : ∑ w ∈ F, ∑ v ∈ F, χ ((v - w) * (0 : K)) = (F.card : ℂ) * (F.card : ℂ) := by
    simp [AddChar.map_zero_eq_one]
  -- Assemble the complex identity `∑ conj (ghat a) * ghat a = 1`.
  have key : ∑ a : K, (starRingEnd ℂ) (ghat χ F a) * ghat χ F a = 1 := by
    rw [Finset.sum_congr rfl fun a _ => hpoint a,
      Finset.sum_congr rfl fun a _ => hsplit a, Finset.sum_sub_distrib,
      Finset.sum_ite_eq' Finset.univ (0 : K), if_pos (Finset.mem_univ (0 : K)),
      ← Finset.mul_sum, hDsum, hD0]
    have hR : ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)))⁻¹ *
          ((F.card : ℝ) * (Fintype.card K : ℝ)) -
        ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)))⁻¹ *
          ((F.card : ℝ) * (F.card : ℝ)) = 1 := by
      rw [← mul_sub]
      have hX : (F.card : ℝ) * (Fintype.card K : ℝ) - (F.card : ℝ) * (F.card : ℝ) =
          (F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)) := by ring
      rw [hX, inv_mul_cancel₀ hX0.ne']
    exact_mod_cast hR
  -- Cast down to the real statement.
  have hcast : ∀ a : K, ((‖ghat χ F a‖ ^ 2 : ℝ) : ℂ) =
      (starRingEnd ℂ) (ghat χ F a) * ghat χ F a := fun a => by
    rw [Complex.sq_norm]
    exact Complex.normSq_eq_conj_mul_self
  have hsum : ((∑ a : K, ‖ghat χ F a‖ ^ 2 : ℝ) : ℂ) = 1 := by
    rw [Complex.ofReal_sum, Finset.sum_congr rfl fun a _ => hcast a, key]
  exact_mod_cast hsum

end Fourier

end Dqi
