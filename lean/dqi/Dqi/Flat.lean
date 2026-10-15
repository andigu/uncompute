import Dqi.Fourier

/-!
# Flatness: the constant is forced, not assumed

The relocation bound needs the DQI coordinate amplitudes to be *flat*: every
nonzero frequency carries magnitude `(q-1)^(-1/2)`.  For the bent target sets
used by the task instances this is true, and the appendix cites it.

This file observes that the numerical value need not be cited at all.  It is
forced by normalization: `Fourier.sum_normSq_ghat` already proves
`∑ a, ‖ghat a‖ ^ 2 = 1`, and `ghat` vanishes at the zero frequency, so the `q-1`
nonzero magnitudes sum in square to `1`.  If they are merely all *equal* — which
is precisely what "bent" means, with no reference to any particular
construction — each must be `(q-1)^(-1/2)`.

So the hypothesis the development carries is the weakest possible form,

  `∀ a b, a ≠ 0 → b ≠ 0 → ‖ghat χ F a‖ = ‖ghat χ F b‖`

and the constant is a theorem.  This matters for the paper: the assumption to be
discharged by citation is "the nonzero Fourier magnitudes of the target set
are equal", which is the definition of a bent function, rather than a specific
numerical claim that could be misquoted.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

omit [DecidableEq K] in
/-- A finite field has at least two elements. -/
theorem one_lt_card_field : 1 < Fintype.card K :=
  Fintype.one_lt_card

/-- **The flat magnitude is forced by normalization.**

Given only that the nonzero coordinate amplitudes all have the *same* magnitude,
each one equals `(q-1)^(-1/2)`.  No property of the target set beyond that
constancy is used. -/
theorem norm_ghat_of_const (χ : AddChar K ℂ) (hχ : χ ≠ 1) (F : Finset K)
    (hr : 0 < F.card) (hrq : F.card < Fintype.card K)
    (hconst : ∀ a b : K, a ≠ 0 → b ≠ 0 → ‖ghat χ F a‖ = ‖ghat χ F b‖)
    (a : K) (ha : a ≠ 0) :
    ‖ghat χ F a‖ = (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹ := by
  classical
  have hcard : (1 : ℕ) < Fintype.card K := one_lt_card_field
  have hq1 : (0 : ℝ) < (Fintype.card K : ℝ) - 1 := by
    have : (1 : ℝ) < (Fintype.card K : ℝ) := by exact_mod_cast hcard
    linarith
  -- Split the normalization sum at the zero frequency.
  have hsum := sum_normSq_ghat χ hχ F hr hrq
  rw [← Finset.sum_erase_add _ _ (Finset.mem_univ (0 : K)), ghat_zero] at hsum
  simp only [norm_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
    add_zero] at hsum
  -- Every surviving term equals ‖ghat a‖ ^ 2.
  have hconstsum : ∑ b ∈ Finset.univ.erase (0 : K), ‖ghat χ F b‖ ^ 2
      = (Finset.univ.erase (0 : K)).card • ‖ghat χ F a‖ ^ 2 := by
    refine Finset.sum_congr rfl ?_ |>.trans (Finset.sum_const _)
    intro b hb
    rw [hconst b a (Finset.ne_of_mem_erase hb) ha]
  rw [hconstsum, Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ,
    nsmul_eq_mul] at hsum
  -- Solve (q - 1) * ‖ghat a‖ ^ 2 = 1.
  have hcast : ((Fintype.card K - 1 : ℕ) : ℝ) = (Fintype.card K : ℝ) - 1 := by
    push_cast [Nat.cast_sub hcard.le]
    ring
  rw [hcast] at hsum
  have hsq : ‖ghat χ F a‖ ^ 2 = ((Fintype.card K : ℝ) - 1)⁻¹ := by
    field_simp at hsum ⊢
    linarith [hsum]
  have hnn : 0 ≤ ‖ghat χ F a‖ := norm_nonneg _
  rw [← Real.sqrt_inv]
  rw [← hsq, Real.sqrt_sq hnn]

end Dqi
