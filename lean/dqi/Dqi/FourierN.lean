import Dqi.Characters
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# The `n`-variable character transform and its unitarity

`Dqi.ftN` is the unitarily normalized character transform on
`EuclideanSpace ℂ (Fin n → K)`.  It is the change of basis between the
measurement basis — where the DQI objective is physically measured, acting
diagonally as multiplication by the objective function in the `x` basis — and
the syndrome basis, where the decoder analysis works.  Conjugating the measured
objective into the syndrome basis is an application of this transform, so the
end-to-end master theorem needs the transform together with its norm theory.

Main results:
* `Dqi.ftNInv_ftN` / `Dqi.ftN_ftNInv`: the conjugate-character transform
  `ftNInv` inverts `ftN` on both sides.
* `Dqi.ftN_add` / `Dqi.ftN_smul`: linearity.
* `Dqi.ftNInv_add` / `Dqi.ftNInv_smul`: linearity of the inverse.
* `Dqi.ftN_inner` / `Dqi.norm_ftN`: unitarity (Parseval).
* `Dqi.ftN_single`: the transform of a measurement-basis vector is a flat
  character profile.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {n : ℕ}

/-- The n-variable character transform, unitarily normalized. -/
noncomputable def ftN (χ : AddChar K ℂ) (f : EuclideanSpace ℂ (Fin n → K)) :
    EuclideanSpace ℂ (Fin n → K) :=
  WithLp.toLp 2 fun u =>
    ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) *
      ∑ x : Fin n → K, χ (∑ j, u j * x j) * f x

/-- The inverse transform (conjugate character). -/
noncomputable def ftNInv (χ : AddChar K ℂ) (f : EuclideanSpace ℂ (Fin n → K)) :
    EuclideanSpace ℂ (Fin n → K) :=
  WithLp.toLp 2 fun x =>
    ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) *
      ∑ u : Fin n → K, χ (-(∑ j, u j * x j)) * f u

omit [DecidableEq K] in
/-- `ftN`, read off pointwise. -/
theorem ftN_apply (χ : AddChar K ℂ) (f : EuclideanSpace ℂ (Fin n → K)) (u : Fin n → K) :
    ftN χ f u =
      ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) *
        ∑ x : Fin n → K, χ (∑ j, u j * x j) * f x := rfl

omit [DecidableEq K] in
/-- `ftNInv`, read off pointwise. -/
theorem ftNInv_apply (χ : AddChar K ℂ) (f : EuclideanSpace ℂ (Fin n → K)) (x : Fin n → K) :
    ftNInv χ f x =
      ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) *
        ∑ u : Fin n → K, χ (-(∑ j, u j * x j)) * f u := rfl

omit [DecidableEq K] in
/-- The two unitary normalizers cancel the orthogonality mass `(#K)^n` exactly. -/
private theorem prefactor_mul_card :
    ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) *
        ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) * (Fintype.card K : ℂ) ^ n = 1 := by
  have hq : (0 : ℝ) < (Fintype.card K : ℝ) := by exact_mod_cast Fintype.card_pos
  have hr : (Real.sqrt (Fintype.card K))⁻¹ * (Real.sqrt (Fintype.card K))⁻¹ *
      (Fintype.card K : ℝ) = 1 := by
    rw [← mul_inv, Real.mul_self_sqrt hq.le]
    exact inv_mul_cancel₀ hq.ne'
  have hc : (((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) *
      (((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) * (Fintype.card K : ℂ) = 1 := by
    rw [← Complex.ofReal_mul, ← Complex.ofReal_natCast, ← Complex.ofReal_mul, hr,
      Complex.ofReal_one]
  rw [← mul_pow, ← mul_pow, hc, one_pow]

/-- Orthogonality in the paired form produced by inversion and Parseval: summing
the product of a conjugated and a plain character kernel over the frequency
register collapses to a delta between the two position arguments. -/
private theorem sum_char_delta (χ : AddChar K ℂ) (hχ : χ ≠ 1) (a b : Fin n → K) :
    ∑ x : Fin n → K, χ (-(∑ j, x j * a j)) * χ (∑ j, x j * b j) =
      if b = a then (Fintype.card K : ℂ) ^ n else 0 := by
  have h1 : ∀ x : Fin n → K,
      χ (-(∑ j, x j * a j)) * χ (∑ j, x j * b j) = χ (∑ j, (b j - a j) * x j) := by
    intro x
    rw [← AddChar.map_add_eq_mul]
    congr 1
    rw [neg_add_eq_sub, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [Finset.sum_congr rfl fun x _ => h1 x, sum_char_dot χ hχ]
  by_cases hab : b = a
  · subst hab
    rw [if_pos (show (fun j => b j - b j) = (0 : Fin n → K) from funext fun j => sub_self _),
      if_pos rfl]
  · rw [if_neg fun h => hab (funext fun j => sub_eq_zero.mp (by simpa using congrFun h j)),
      if_neg hab]

/-- Left inversion: the conjugate-character transform undoes `ftN`. -/
theorem ftNInv_ftN (χ : AddChar K ℂ) (hχ : χ ≠ 1) (f : EuclideanSpace ℂ (Fin n → K)) :
    ftNInv χ (ftN χ f) = f := by
  apply PiLp.ext
  intro x
  rw [ftNInv_apply]
  simp only [ftN_apply]
  set s : ℂ := (((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n with hs
  have hterm : ∀ u : Fin n → K,
      χ (-(∑ j, u j * x j)) * (s * ∑ x' : Fin n → K, χ (∑ j, u j * x' j) * f x') =
        ∑ x' : Fin n → K,
          s * (χ (-(∑ j, u j * x j)) * χ (∑ j, u j * x' j) * f x') := by
    intro u
    rw [Finset.mul_sum, Finset.mul_sum]
    exact Finset.sum_congr rfl fun x' _ => by ring
  rw [Finset.sum_congr rfl fun u _ => hterm u, Finset.sum_comm]
  have hinner : ∀ x' : Fin n → K,
      ∑ u : Fin n → K, s * (χ (-(∑ j, u j * x j)) * χ (∑ j, u j * x' j) * f x') =
        s * ((if x' = x then (Fintype.card K : ℂ) ^ n else 0) * f x') := by
    intro x'
    rw [← Finset.mul_sum]
    congr 1
    rw [← Finset.sum_mul, sum_char_delta χ hχ x x']
  rw [Finset.sum_congr rfl fun x' _ => hinner x']
  simp only [ite_mul, zero_mul, mul_ite, mul_zero]
  rw [Finset.sum_ite_eq' Finset.univ x fun x' => s * ((Fintype.card K : ℂ) ^ n * f x'),
    if_pos (Finset.mem_univ x)]
  rw [show s * (s * ((Fintype.card K : ℂ) ^ n * f x)) =
      s * s * (Fintype.card K : ℂ) ^ n * f x from by ring, hs, prefactor_mul_card, one_mul]

/-- Right inversion: `ftN` undoes the conjugate-character transform. -/
theorem ftN_ftNInv (χ : AddChar K ℂ) (hχ : χ ≠ 1) (f : EuclideanSpace ℂ (Fin n → K)) :
    ftN χ (ftNInv χ f) = f := by
  apply PiLp.ext
  intro u
  rw [ftN_apply]
  simp only [ftNInv_apply]
  set s : ℂ := (((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n with hs
  have hterm : ∀ x : Fin n → K,
      χ (∑ j, u j * x j) * (s * ∑ u' : Fin n → K, χ (-(∑ j, u' j * x j)) * f u') =
        ∑ u' : Fin n → K,
          s * (χ (-(∑ j, x j * u' j)) * χ (∑ j, x j * u j) * f u') := by
    intro x
    rw [Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun u' _ => ?_
    rw [show (∑ j, u j * x j) = ∑ j, x j * u j from
        Finset.sum_congr rfl fun j _ => mul_comm _ _,
      show (∑ j, u' j * x j) = ∑ j, x j * u' j from
        Finset.sum_congr rfl fun j _ => mul_comm _ _]
    ring
  rw [Finset.sum_congr rfl fun x _ => hterm x, Finset.sum_comm]
  have hinner : ∀ u' : Fin n → K,
      ∑ x : Fin n → K, s * (χ (-(∑ j, x j * u' j)) * χ (∑ j, x j * u j) * f u') =
        s * ((if u = u' then (Fintype.card K : ℂ) ^ n else 0) * f u') := by
    intro u'
    rw [← Finset.mul_sum]
    congr 1
    rw [← Finset.sum_mul, sum_char_delta χ hχ u' u]
  rw [Finset.sum_congr rfl fun u' _ => hinner u']
  simp only [ite_mul, zero_mul, mul_ite, mul_zero]
  rw [Finset.sum_ite_eq Finset.univ u fun u' => s * ((Fintype.card K : ℂ) ^ n * f u'),
    if_pos (Finset.mem_univ u)]
  rw [show s * (s * ((Fintype.card K : ℂ) ^ n * f u)) =
      s * s * (Fintype.card K : ℂ) ^ n * f u from by ring, hs, prefactor_mul_card, one_mul]

omit [DecidableEq K] in
/-- The transform is additive. -/
theorem ftN_add (χ : AddChar K ℂ) (f g : EuclideanSpace ℂ (Fin n → K)) :
    ftN χ (f + g) = ftN χ f + ftN χ g := by
  apply PiLp.ext
  intro u
  rw [PiLp.add_apply, ftN_apply, ftN_apply, ftN_apply, ← mul_add, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [PiLp.add_apply, mul_add]

omit [DecidableEq K] in
/-- The transform commutes with scalar multiplication. -/
theorem ftN_smul (χ : AddChar K ℂ) (a : ℂ) (f : EuclideanSpace ℂ (Fin n → K)) :
    ftN χ (a • f) = a • ftN χ f := by
  apply PiLp.ext
  intro u
  rw [PiLp.smul_apply, ftN_apply, ftN_apply, smul_eq_mul, Finset.mul_sum, Finset.mul_sum,
    Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [PiLp.smul_apply, smul_eq_mul]
  ring

omit [DecidableEq K] in
/-- The inverse transform is additive. -/
theorem ftNInv_add (χ : AddChar K ℂ) (f g : EuclideanSpace ℂ (Fin n → K)) :
    ftNInv χ (f + g) = ftNInv χ f + ftNInv χ g := by
  apply PiLp.ext
  intro x
  rw [PiLp.add_apply, ftNInv_apply, ftNInv_apply, ftNInv_apply, ← mul_add,
    ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [PiLp.add_apply, mul_add]

omit [DecidableEq K] in
/-- The inverse transform commutes with scalar multiplication. -/
theorem ftNInv_smul (χ : AddChar K ℂ) (a : ℂ) (f : EuclideanSpace ℂ (Fin n → K)) :
    ftNInv χ (a • f) = a • ftNInv χ f := by
  apply PiLp.ext
  intro x
  rw [PiLp.smul_apply, ftNInv_apply, ftNInv_apply, smul_eq_mul, Finset.mul_sum,
    Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [PiLp.smul_apply, smul_eq_mul]
  ring

/-- Parseval: the character transform preserves the inner product. -/
theorem ftN_inner (χ : AddChar K ℂ) (hχ : χ ≠ 1) (f g : EuclideanSpace ℂ (Fin n → K)) :
    inner ℂ (ftN χ f) (ftN χ g) = inner ℂ f g := by
  rw [PiLp.inner_apply, PiLp.inner_apply]
  simp only [RCLike.inner_apply]
  set s : ℂ := (((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n with hs
  have h3 : ∀ x : Fin n → K,
      (starRingEnd ℂ) (f x) =
        s * ∑ u : Fin n → K, χ (∑ j, u j * x j) * (starRingEnd ℂ) (ftN χ f u) := by
    intro x
    have h2 : s * ∑ u : Fin n → K, χ (-(∑ j, u j * x j)) * ftN χ f u = f x := by
      rw [hs, ← ftNInv_apply, ftNInv_ftN χ hχ f]
    rw [← h2, map_mul, map_sum]
    congr 1
    · rw [hs, map_pow, Complex.conj_ofReal]
    · exact Finset.sum_congr rfl fun u _ => by
        rw [map_mul, ← AddChar.map_neg_eq_conj, neg_neg]
  have hL : ∀ u : Fin n → K,
      ftN χ g u * (starRingEnd ℂ) (ftN χ f u) =
        ∑ x : Fin n → K,
          s * (χ (∑ j, u j * x j) * (g x * (starRingEnd ℂ) (ftN χ f u))) := by
    intro u
    rw [ftN_apply, ← hs, mul_assoc, Finset.sum_mul, Finset.mul_sum]
    exact Finset.sum_congr rfl fun x _ => by ring
  have hR : ∀ x : Fin n → K,
      g x * (starRingEnd ℂ) (f x) =
        ∑ u : Fin n → K,
          s * (χ (∑ j, u j * x j) * (g x * (starRingEnd ℂ) (ftN χ f u))) := by
    intro x
    rw [h3 x, Finset.mul_sum, Finset.mul_sum]
    exact Finset.sum_congr rfl fun u _ => by ring
  rw [Finset.sum_congr rfl fun u _ => hL u, Finset.sum_comm]
  exact Finset.sum_congr rfl fun x _ => (hR x).symm

/-- Parseval, norm form: the character transform is an isometry. -/
theorem norm_ftN (χ : AddChar K ℂ) (hχ : χ ≠ 1) (f : EuclideanSpace ℂ (Fin n → K)) :
    ‖ftN χ f‖ = ‖f‖ := by
  have h2 : ‖ftN χ f‖ ^ 2 = ‖f‖ ^ 2 := by
    rw [← inner_self_eq_norm_sq (𝕜 := ℂ), ← inner_self_eq_norm_sq (𝕜 := ℂ),
      ftN_inner χ hχ f f]
  have h3 := congrArg Real.sqrt h2
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at h3

/-- The transform of a measurement-basis vector is a flat character profile. -/
theorem ftN_single (χ : AddChar K ℂ) (x₀ : Fin n → K) :
    ∀ u, ftN χ (EuclideanSpace.single x₀ 1) u =
      ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) * χ (∑ j, u j * x₀ j) := by
  intro u
  rw [ftN_apply]
  congr 1
  have hterm : ∀ x : Fin n → K,
      χ (∑ j, u j * x j) * EuclideanSpace.single x₀ (1 : ℂ) x =
        if x = x₀ then χ (∑ j, u j * x j) else 0 := by
    intro x
    rw [PiLp.single_apply]
    by_cases h : x = x₀ <;> simp [h]
  rw [Finset.sum_congr rfl fun x _ => hterm x,
    Finset.sum_ite_eq' Finset.univ x₀ fun x => χ (∑ j, u j * x j),
    if_pos (Finset.mem_univ x₀)]

end Dqi
