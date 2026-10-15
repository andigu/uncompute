import Dqi.Fourier

/-!
# The single-site constraint operator in the pattern basis

The two theorems of this file are the two rows of the single-site constraint
operator, written in the pattern (Fourier) basis spanned by the zero-frequency
vector and the DQI amplitude vector `ghat χ F`.  `siteRow_zero` is the row on
the zero-frequency component and `siteRow_ghat` the row on the `ghat`
component; their coefficients

* `r/q` and `cc1 F` (first row),
* `cc1 F` and `(q - r)/q` (second row),

with `q = |K|`, `r = |F|`, and `cc1 F = √(r(q-r))/q`, are exactly the entries
of the 2×2 shell matrix of the single-site operator.
-/

namespace Dqi

section SiteOp

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- The off-diagonal entry of the 2×2 shell matrix of the single-site
constraint operator: `√(r(q-r))/q` where `q = |K|` and `r = |F|`. -/
noncomputable def cc1 (F : Finset K) : ℝ :=
  Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - F.card)) / (Fintype.card K : ℝ)

/-- First row of the single-site constraint operator in the pattern basis:
the zero-frequency component of the constrained vector is `r/q` times the
zero-frequency input plus `cc1 F` times the `ghat` input.

Unlike `siteRow_ghat` this needs no nontriviality hypothesis on `χ`: it is a
rearrangement of the definition of `ghat`, with no character sum to collapse. -/
theorem siteRow_zero (χ : AddChar K ℂ) (F : Finset K)
    (hr : 0 < F.card) (hrq : F.card < Fintype.card K) (b : K) :
    ((Fintype.card K : ℝ)⁻¹ : ℂ) * ∑ v ∈ F, χ (b * v)
      = (((F.card : ℝ) / (Fintype.card K : ℝ) : ℝ) : ℂ) * (if b = 0 then 1 else 0)
        + ((cc1 F : ℝ) : ℂ) * ghat χ F b := by
  have hr0 : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hr
  have hqr : (0 : ℝ) < (Fintype.card K : ℝ) - (F.card : ℝ) := by
    have h : (F.card : ℝ) < (Fintype.card K : ℝ) := by exact_mod_cast hrq
    linarith
  have hX0 : (0 : ℝ) < (F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)) :=
    mul_pos hr0 hqr
  have hs0 : Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))) ≠ 0 :=
    (Real.sqrt_pos.mpr hX0).ne'
  by_cases hb : b = 0
  · -- At the zero frequency both sides are `r/q`.
    subst hb
    rw [ghat_zero, mul_zero, add_zero, if_pos (rfl : (0 : K) = 0), mul_one]
    have hsF : ∑ v ∈ F, χ ((0 : K) * v) = (F.card : ℂ) := by
      simp [AddChar.map_zero_eq_one]
    rw [hsF]
    push_cast
    ring
  · -- Away from zero the normalizers combine: `cc1 F * (√(r(q-r)))⁻¹ = q⁻¹`.
    rw [if_neg hb, mul_zero, zero_add]
    simp only [ghat, if_neg hb]
    have hcc : cc1 F *
        (Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹
        = (Fintype.card K : ℝ)⁻¹ := by
      unfold cc1
      rw [div_mul_eq_mul_div, mul_inv_cancel₀ hs0, one_div]
    rw [← mul_assoc, ← Complex.ofReal_mul, hcc, Complex.ofReal_inv, Complex.ofReal_natCast]

/-- Second row of the single-site constraint operator in the pattern basis:
the `ghat`-weighted component of the constrained vector is `cc1 F` times the
zero-frequency input plus `(q - r)/q` times the `ghat` input. -/
theorem siteRow_ghat (χ : AddChar K ℂ) (hχ : χ ≠ 1) (F : Finset K)
    (hr : 0 < F.card) (hrq : F.card < Fintype.card K) (b : K) :
    ((Fintype.card K : ℝ)⁻¹ : ℂ) * ∑ a : K, (∑ v ∈ F, χ ((b - a) * v)) * ghat χ F a
      = ((cc1 F : ℝ) : ℂ) * (if b = 0 then 1 else 0)
        + ((((Fintype.card K : ℝ) - F.card) / (Fintype.card K : ℝ) : ℝ) : ℂ) *
          ghat χ F b := by
  have hr0 : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hr
  have hqr : (0 : ℝ) < (Fintype.card K : ℝ) - (F.card : ℝ) := by
    have h : (F.card : ℝ) < (Fintype.card K : ℝ) := by exact_mod_cast hrq
    linarith
  have hX0 : (0 : ℝ) < (F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ)) :=
    mul_pos hr0 hqr
  have hs0 : Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))) ≠ 0 :=
    (Real.sqrt_pos.mpr hX0).ne'
  -- The full double character sum over all frequencies collapses by
  -- orthogonality to `q ∑_{v ∈ F} χ(b v)`.
  have hfull : ∑ a : K, (∑ v ∈ F, χ ((b - a) * v)) * ∑ w ∈ F, χ (a * w)
      = (Fintype.card K : ℂ) * ∑ v ∈ F, χ (b * v) := by
    have hterm : ∀ a : K, (∑ v ∈ F, χ ((b - a) * v)) * ∑ w ∈ F, χ (a * w)
        = ∑ v ∈ F, ∑ w ∈ F, χ (b * v) * χ ((w - v) * a) := by
      intro a
      rw [Finset.sum_mul_sum]
      refine Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun w _ => ?_
      rw [← AddChar.map_add_eq_mul, ← AddChar.map_add_eq_mul]
      congr 1
      ring
    have hv : ∀ v ∈ F, (∑ a : K, ∑ w ∈ F, χ (b * v) * χ ((w - v) * a))
        = (Fintype.card K : ℂ) * χ (b * v) := by
      intro v hvF
      rw [Finset.sum_comm]
      have hw : ∀ w ∈ F, (∑ a : K, χ (b * v) * χ ((w - v) * a))
          = χ (b * v) * (if w = v then (Fintype.card K : ℂ) else 0) := by
        intro w _
        rw [← Finset.mul_sum, sum_char_mul χ hχ (w - v)]
        simp only [sub_eq_zero]
      rw [Finset.sum_congr rfl hw, ← Finset.mul_sum,
        Finset.sum_ite_eq' F v fun _ => (Fintype.card K : ℂ), if_pos hvF,
        mul_comm (χ (b * v)) (Fintype.card K : ℂ)]
    rw [Finset.sum_congr rfl fun a _ => hterm a, Finset.sum_comm,
      Finset.sum_congr rfl hv, ← Finset.mul_sum]
  -- Trade the guard of `ghat` for subtracting the `a = 0` term.
  have hpoint : ∀ a : K, (∑ v ∈ F, χ ((b - a) * v)) * ghat χ F a
      = (((Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹ : ℝ) : ℂ) *
          ((∑ v ∈ F, χ ((b - a) * v)) * ∑ w ∈ F, χ (a * w))
        - (if a = 0 then
            (((Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹ : ℝ) : ℂ) *
              ((∑ v ∈ F, χ ((b - a) * v)) * ∑ w ∈ F, χ (a * w))
          else 0) := by
    intro a
    by_cases ha : a = 0
    · simp [ghat, ha]
    · simp only [ghat, if_neg ha]
      ring
  -- The `a = 0` term of the full sum is `r ∑_{v ∈ F} χ(b v)`.
  have hzero : (∑ v ∈ F, χ ((b - 0) * v)) * ∑ w ∈ F, χ ((0 : K) * w)
      = (∑ v ∈ F, χ (b * v)) * (F.card : ℂ) := by
    simp [AddChar.map_zero_eq_one]
  have hcore : ∑ a : K, (∑ v ∈ F, χ ((b - a) * v)) * ghat χ F a
      = (((Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹ : ℝ) : ℂ) *
          ((Fintype.card K : ℂ) * ∑ v ∈ F, χ (b * v))
        - (((Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹ : ℝ) : ℂ) *
          ((∑ v ∈ F, χ (b * v)) * (F.card : ℂ)) := by
    rw [Finset.sum_congr rfl fun a _ => hpoint a, Finset.sum_sub_distrib,
      Finset.sum_ite_eq' Finset.univ (0 : K), if_pos (Finset.mem_univ (0 : K)),
      ← Finset.mul_sum, hfull, hzero]
  rw [hcore]
  by_cases hb : b = 0
  · -- At the zero frequency: `q⁻¹ (q - r) r / √(r(q-r)) = cc1 F`.
    subst hb
    rw [if_pos (rfl : (0 : K) = 0), mul_one, ghat_zero, mul_zero, add_zero]
    have hS : ∑ v ∈ F, χ ((0 : K) * v) = (F.card : ℂ) := by
      simp [AddChar.map_zero_eq_one]
    rw [hS]
    have hR : (Fintype.card K : ℝ)⁻¹ *
        ((Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹ *
            ((Fintype.card K : ℝ) * (F.card : ℝ))
          - (Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))))⁻¹ *
            ((F.card : ℝ) * (F.card : ℝ))) = cc1 F := by
      have h2 : (Fintype.card K : ℝ) * (F.card : ℝ) - (F.card : ℝ) * (F.card : ℝ)
          = Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))) *
            Real.sqrt ((F.card : ℝ) * ((Fintype.card K : ℝ) - (F.card : ℝ))) := by
        rw [Real.mul_self_sqrt hX0.le]
        ring
      rw [← mul_sub, h2, inv_mul_cancel_left₀ hs0]
      unfold cc1
      rw [inv_mul_eq_div]
    exact_mod_cast hR
  · -- Away from zero: cancel the common character sum; a formal field identity.
    rw [if_neg hb, mul_zero, zero_add]
    simp only [ghat, if_neg hb]
    push_cast
    ring

end SiteOp

end Dqi
