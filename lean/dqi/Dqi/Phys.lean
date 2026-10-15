import Dqi.Basic
import Dqi.FourierN
import Dqi.Objective

/-!
# The physical objective and the pattern-to-syndrome pushforward

The DQI objective is physically measured in the assignment (`x`) basis, where it
acts diagonally: `Dqi.fObj` scores `+1` per satisfied constraint and `−1` per
violated one.  The decoder analysis lives in the syndrome basis, so `Dqi.synOp`
conjugates multiplication-by-`fObj` by the character transform `ftN`.  The
transport between the two pictures is `Dqi.toSyn`, the pushforward of a
pattern-space vector along the syndrome map.

The metric statements are the point of the file: `synOp` is a symmetric form
bounded by the constraint count `m`, and `toSyn` preserves inner products and
norms under a no-cross-collision hypothesis — distinct vectors in the supports
never share a syndrome.  That hypothesis is exactly the strict-distance
condition in the form the code lemmas (`syndrome_injOn_decodable`) supply.  The
intertwining of `synOp` with the pattern-space operator is proved in the next
file.

Main results:
* `Dqi.abs_fObj_le`: `|fObj B Fs x| ≤ m`.
* `Dqi.synOp_inner` / `Dqi.norm_synOp_le`: the syndrome-basis objective is
  symmetric and bounded by `m`.
* `Dqi.synOp_add` / `Dqi.synOp_smul`: linearity of the syndrome-basis
  objective.
* `Dqi.toSyn_add` / `Dqi.toSyn_smul`: linearity of the pushforward.
* `Dqi.inner_toSyn_toSyn` / `Dqi.norm_toSyn`: the pushforward preserves inner
  products and norms on collision-free supports.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m n : ℕ}

/-- The objective as a function of the measured assignment: +1 per satisfied
constraint, −1 per violated one. -/
noncomputable def fObj (B : Matrix (Fin m) (Fin n) K) (Fs : Fin m → Finset K)
    (x : Fin n → K) : ℝ :=
  ∑ i : Fin m, (if (B.mulVec x) i ∈ Fs i then (1 : ℝ) else -1)

/-- The objective in the syndrome basis: conjugate multiplication-by-`fObj`
(diagonal in the measurement basis) by the character transform. -/
noncomputable def synOp (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ : EuclideanSpace ℂ (Fin n → K)) :
    EuclideanSpace ℂ (Fin n → K) :=
  ftN χ (mulOp (fObj B Fs) (ftNInv χ ψ))

/-- Pushforward of a pattern-space vector along the syndrome map. -/
noncomputable def toSyn (B : Matrix (Fin m) (Fin n) K)
    (ψ : EuclideanSpace ℂ (Pattern m K)) : EuclideanSpace ℂ (Fin n → K) :=
  WithLp.toLp 2 fun u => ∑ y : Pattern m K, if syndrome B y = u then ψ y else 0

/-- `toSyn`, read off pointwise. -/
theorem toSyn_apply (B : Matrix (Fin m) (Fin n) K)
    (ψ : EuclideanSpace ℂ (Pattern m K)) (u : Fin n → K) :
    toSyn B ψ u = ∑ y : Pattern m K, if syndrome B y = u then ψ y else 0 := rfl

omit [Fintype K] in
/-- The objective is bounded by the constraint count. -/
theorem abs_fObj_le (B : Matrix (Fin m) (Fin n) K) (Fs : Fin m → Finset K)
    (x : Fin n → K) : |fObj B Fs x| ≤ (m : ℝ) := by
  have h : fObj B Fs x = ∑ i : Fin m, (if (B.mulVec x) i ∈ Fs i then (1 : ℝ) else -1) := rfl
  rw [h]
  calc |∑ i : Fin m, (if (B.mulVec x) i ∈ Fs i then (1 : ℝ) else -1)|
      ≤ ∑ i : Fin m, |if (B.mulVec x) i ∈ Fs i then (1 : ℝ) else -1| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ _i : Fin m, (1 : ℝ) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        by_cases hi : (B.mulVec x) i ∈ Fs i
        · rw [if_pos hi, abs_one]
        · rw [if_neg hi, abs_neg, abs_one]
    _ = (m : ℝ) := by
        rw [Finset.sum_const, Finset.card_fin, nsmul_eq_mul, mul_one]

/-- Multiplication by a real-valued function is a symmetric form; two-line
pointwise restatement of `isSelfAdjoint_mulOp` in the shape the transport
argument uses. -/
private theorem inner_mulOp_left {ι : Type*} [Fintype ι] (g : ι → ℝ)
    (a b : EuclideanSpace ℂ ι) :
    inner ℂ (mulOp g a) b = inner ℂ a (mulOp g b) := by
  simp only [PiLp.inner_apply, mulOp_apply, RCLike.inner_apply, map_mul,
    Complex.conj_ofReal]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- The syndrome-basis objective is a symmetric form. -/
theorem synOp_inner (χ : AddChar K ℂ) (hχ : χ ≠ 1) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ φ : EuclideanSpace ℂ (Fin n → K)) :
    inner ℂ (synOp χ B Fs ψ) φ = inner ℂ ψ (synOp χ B Fs φ) := by
  unfold synOp
  have h1 : inner ℂ (ftN χ (mulOp (fObj B Fs) (ftNInv χ ψ))) φ
      = inner ℂ (ftNInv χ ψ) (mulOp (fObj B Fs) (ftNInv χ φ)) := by
    conv_lhs => rw [← ftN_ftNInv χ hχ φ]
    rw [ftN_inner χ hχ, inner_mulOp_left]
  have h2 : inner ℂ ψ (ftN χ (mulOp (fObj B Fs) (ftNInv χ φ)))
      = inner ℂ (ftNInv χ ψ) (mulOp (fObj B Fs) (ftNInv χ φ)) := by
    conv_lhs => rw [← ftN_ftNInv χ hχ ψ]
    rw [ftN_inner χ hχ]
  rw [h1, h2]

/-- The syndrome-basis objective is norm-bounded by the constraint count. -/
theorem norm_synOp_le (χ : AddChar K ℂ) (hχ : χ ≠ 1) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ : EuclideanSpace ℂ (Fin n → K)) :
    ‖synOp χ B Fs ψ‖ ≤ (m : ℝ) * ‖ψ‖ := by
  unfold synOp
  have hinv : ‖ftNInv χ ψ‖ = ‖ψ‖ := by
    rw [← norm_ftN χ hχ (ftNInv χ ψ), ftN_ftNInv χ hχ]
  calc ‖ftN χ (mulOp (fObj B Fs) (ftNInv χ ψ))‖
      = ‖mulOp (fObj B Fs) (ftNInv χ ψ)‖ := norm_ftN χ hχ _
    _ ≤ ‖mulOp (fObj B Fs)‖ * ‖ftNInv χ ψ‖ := ContinuousLinearMap.le_opNorm _ _
    _ ≤ (m : ℝ) * ‖ψ‖ := by
        rw [hinv]
        exact mul_le_mul_of_nonneg_right
          (norm_mulOp_le _ _ (Nat.cast_nonneg m) (abs_fObj_le B Fs)) (norm_nonneg ψ)

/-- The syndrome-basis objective is additive. -/
theorem synOp_add (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ φ : EuclideanSpace ℂ (Fin n → K)) :
    synOp χ B Fs (ψ + φ) = synOp χ B Fs ψ + synOp χ B Fs φ := by
  unfold synOp
  rw [ftNInv_add, map_add, ftN_add]

/-- The syndrome-basis objective commutes with scalars. -/
theorem synOp_smul (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (a : ℂ) (ψ : EuclideanSpace ℂ (Fin n → K)) :
    synOp χ B Fs (a • ψ) = a • synOp χ B Fs ψ := by
  unfold synOp
  rw [ftNInv_smul, map_smul, ftN_smul]

/-- The pushforward is additive. -/
theorem toSyn_add (B : Matrix (Fin m) (Fin n) K)
    (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    toSyn B (ψ + φ) = toSyn B ψ + toSyn B φ := by
  apply PiLp.ext
  intro u
  rw [PiLp.add_apply, toSyn_apply, toSyn_apply, toSyn_apply, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  by_cases h : syndrome B y = u
  · rw [if_pos h, if_pos h, if_pos h, PiLp.add_apply]
  · rw [if_neg h, if_neg h, if_neg h, add_zero]

/-- The pushforward commutes with scalar multiplication. -/
theorem toSyn_smul (B : Matrix (Fin m) (Fin n) K) (a : ℂ)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    toSyn B (a • ψ) = a • toSyn B ψ := by
  apply PiLp.ext
  intro u
  rw [PiLp.smul_apply, smul_eq_mul, toSyn_apply, toSyn_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl fun y _ => ?_
  by_cases h : syndrome B y = u
  · rw [if_pos h, if_pos h, PiLp.smul_apply, smul_eq_mul]
  · rw [if_neg h, if_neg h, mul_zero]

/-- The metric transport: when the supports of `ψ` and `φ` never collide across
the syndrome map, the pushforward preserves the inner product. -/
theorem inner_toSyn_toSyn (B : Matrix (Fin m) (Fin n) K)
    (ψ φ : EuclideanSpace ℂ (Pattern m K))
    (hcross : ∀ y z : Pattern m K, ψ y ≠ 0 → φ z ≠ 0 →
      syndrome B y = syndrome B z → y = z) :
    inner ℂ (toSyn B ψ) (toSyn B φ) = inner ℂ ψ φ := by
  rw [PiLp.inner_apply, PiLp.inner_apply]
  simp only [RCLike.inner_apply, toSyn_apply]
  have hconj : ∀ u : Fin n → K,
      (starRingEnd ℂ) (∑ y : Pattern m K, if syndrome B y = u then ψ y else 0) =
        ∑ y : Pattern m K, if syndrome B y = u then (starRingEnd ℂ) (ψ y) else 0 := by
    intro u
    rw [map_sum]
    exact Finset.sum_congr rfl fun y _ => by rw [apply_ite (starRingEnd ℂ), map_zero]
  have hprod : ∀ u : Fin n → K,
      (∑ z : Pattern m K, if syndrome B z = u then φ z else 0) *
          (starRingEnd ℂ) (∑ y : Pattern m K, if syndrome B y = u then ψ y else 0) =
        ∑ z : Pattern m K, ∑ y : Pattern m K,
          (if syndrome B z = u then φ z else 0) *
            (if syndrome B y = u then (starRingEnd ℂ) (ψ y) else 0) := by
    intro u
    rw [hconj u, Finset.sum_mul_sum]
  rw [Finset.sum_congr rfl fun u _ => hprod u, Finset.sum_comm]
  have hz : ∀ z : Pattern m K,
      (∑ u : Fin n → K, ∑ y : Pattern m K,
          (if syndrome B z = u then φ z else 0) *
            (if syndrome B y = u then (starRingEnd ℂ) (ψ y) else 0)) =
        φ z * (starRingEnd ℂ) (ψ z) := by
    intro z
    rw [Finset.sum_comm]
    have hy : ∀ y : Pattern m K,
        (∑ u : Fin n → K,
            (if syndrome B z = u then φ z else 0) *
              (if syndrome B y = u then (starRingEnd ℂ) (ψ y) else 0)) =
          if y = z then φ z * (starRingEnd ℂ) (ψ z) else 0 := by
      intro y
      have hcollapse :
          (∑ u : Fin n → K,
              (if syndrome B z = u then φ z else 0) *
                (if syndrome B y = u then (starRingEnd ℂ) (ψ y) else 0)) =
            (if syndrome B y = syndrome B z then φ z * (starRingEnd ℂ) (ψ y) else 0) := by
        simp only [ite_mul, zero_mul]
        rw [Finset.sum_ite_eq Finset.univ (syndrome B z)
            (fun u => φ z * if syndrome B y = u then (starRingEnd ℂ) (ψ y) else 0),
          if_pos (Finset.mem_univ _), mul_ite, mul_zero]
      rw [hcollapse]
      by_cases hyz : y = z
      · subst hyz
        rw [if_pos rfl, if_pos rfl]
      · rw [if_neg hyz]
        by_cases hsyn : syndrome B y = syndrome B z
        · rw [if_pos hsyn]
          by_cases hψ : ψ y = 0
          · rw [hψ, map_zero, mul_zero]
          · by_cases hφ : φ z = 0
            · rw [hφ, zero_mul]
            · exact absurd (hcross y z hψ hφ hsyn) hyz
        · rw [if_neg hsyn]
    rw [Finset.sum_congr rfl fun y _ => hy y,
      Finset.sum_ite_eq' Finset.univ z (fun _ => φ z * (starRingEnd ℂ) (ψ z)),
      if_pos (Finset.mem_univ z)]
  exact Finset.sum_congr rfl fun z _ => hz z

/-- Norm form of the metric transport: on a collision-free support the
pushforward is an isometry. -/
theorem norm_toSyn (B : Matrix (Fin m) (Fin n) K)
    (ψ : EuclideanSpace ℂ (Pattern m K))
    (hinj : ∀ y z : Pattern m K, ψ y ≠ 0 → ψ z ≠ 0 →
      syndrome B y = syndrome B z → y = z) :
    ‖toSyn B ψ‖ = ‖ψ‖ := by
  have h2 : ‖toSyn B ψ‖ ^ 2 = ‖ψ‖ ^ 2 := by
    rw [← inner_self_eq_norm_sq (𝕜 := ℂ), ← inner_self_eq_norm_sq (𝕜 := ℂ),
      inner_toSyn_toSyn B ψ ψ hinj]
  have h3 := congrArg Real.sqrt h2
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at h3

end Dqi
