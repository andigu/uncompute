import Dqi.Failure
import Dqi.Intertwine

/-!
# The objective on the full register space

The master theorem must be stated on the full register space — syndrome ×
ancilla × record — because leakage errors (a correct syndrome paired with a
dirty ancilla or record) are distinguishable only there.  This file lifts the
syndrome-register objective `Dqi.synOp` to `Dqi.synOpFull`, acting trivially on
the ancilla and record registers ("the objective acts only on the syndrome
register"), embeds pattern-space states into the full space with a clean
ancilla value and a fixed record state (`Dqi.toPhys`), and lifts the
intertwining identity along the embedding.

Main results:
* `Dqi.norm_sq_eq_sum_slices`: the squared norm decomposes over ancilla-record
  slices.
* `Dqi.norm_synOpFull_le` / `Dqi.synOpFull_inner`: the lifted objective keeps
  the bound `m` and the symmetry of `synOp`.
* `Dqi.inner_toPhys_toPhys` / `Dqi.norm_toPhys`: the clean-register embedding
  is metric when the record state is a unit vector.
* `Dqi.toPhys_add` / `Dqi.toPhys_smul`: linearity of the embedding.
* `Dqi.synOpFull_toPhys`: the intertwining `synOp_toSyn`, lifted to the full
  register space.
* `Dqi.toPhys_eq_zero_of_anc` / `Dqi.toPhys_eq_zero_of_syn`: support control
  for the later blindness argument.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m n : ℕ}
variable {Anc Rec : Type*} [Fintype Anc] [DecidableEq Anc] [Fintype Rec]

/-- The objective on the full register space: `synOp` on the syndrome register,
identity on ancilla and record. -/
noncomputable def synOpFull (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ : Phys (Fin n → K) Anc Rec) : Phys (Fin n → K) Anc Rec :=
  WithLp.toLp 2 fun z =>
    synOp χ B Fs (WithLp.toLp 2 fun s => ψ (s, z.2)) z.1

/-- Embed a pattern-space vector into the full register space with a fixed clean
ancilla value and a fixed record state. -/
noncomputable def toPhys (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (ψ : EuclideanSpace ℂ (Pattern m K)) :
    Phys (Fin n → K) Anc Rec :=
  WithLp.toLp 2 fun z =>
    (if z.2.1 = a₀ then 1 else 0) * toSyn B ψ z.1 * wrec z.2.2

omit [Fintype Anc] [DecidableEq Anc] [Fintype Rec] in
/-- `synOpFull`, read off pointwise. -/
theorem synOpFull_apply (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ : Phys (Fin n → K) Anc Rec)
    (z : (Fin n → K) × Anc × Rec) :
    synOpFull χ B Fs ψ z = synOp χ B Fs (WithLp.toLp 2 fun s => ψ (s, z.2)) z.1 := rfl

omit [Fintype Anc] [Fintype Rec] in
/-- `toPhys`, read off pointwise. -/
theorem toPhys_apply (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (ψ : EuclideanSpace ℂ (Pattern m K))
    (z : (Fin n → K) × Anc × Rec) :
    toPhys B a₀ wrec ψ z = (if z.2.1 = a₀ then 1 else 0) * toSyn B ψ z.1 * wrec z.2.2 := rfl

omit [Fintype Anc] [DecidableEq Anc] [Fintype Rec] in
/-- The ancilla-record slice of `synOpFull ψ` is `synOp` of the slice of `ψ`. -/
theorem synOpFull_slice (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ : Phys (Fin n → K) Anc Rec) (w : Anc × Rec) :
    (WithLp.toLp 2 fun s => synOpFull χ B Fs ψ (s, w) : EuclideanSpace ℂ (Fin n → K))
      = synOp χ B Fs (WithLp.toLp 2 fun s => ψ (s, w)) := by
  apply PiLp.ext
  intro s
  rfl

omit [Fintype Anc] [Fintype Rec] in
/-- The ancilla-record slice of the embedding is a scalar multiple of the
syndrome pushforward. -/
theorem toPhys_slice (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (ψ : EuclideanSpace ℂ (Pattern m K)) (w : Anc × Rec) :
    (WithLp.toLp 2 fun s => toPhys B a₀ wrec ψ (s, w) : EuclideanSpace ℂ (Fin n → K))
      = ((if w.1 = a₀ then 1 else 0) * wrec w.2) • toSyn B ψ := by
  apply PiLp.ext
  intro s
  rw [PiLp.smul_apply, smul_eq_mul, toPhys_apply]
  ring

omit [Field K] [DecidableEq K] [DecidableEq Anc] in
/-- The slice decomposition of the squared norm: sum over ancilla-record pairs
of the squared norms of the syndrome-register slices. -/
theorem norm_sq_eq_sum_slices (ψ : Phys (Fin n → K) Anc Rec) :
    ‖ψ‖ ^ 2 = ∑ w : Anc × Rec,
      ‖(WithLp.toLp 2 fun s => ψ (s, w) : EuclideanSpace ℂ (Fin n → K))‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq,
    Fintype.sum_prod_type (fun z : (Fin n → K) × Anc × Rec => ‖ψ z‖ ^ 2),
    Finset.sum_comm]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [EuclideanSpace.norm_sq_eq]

omit [Field K] [DecidableEq K] [DecidableEq Anc] in
/-- The slice decomposition of the inner product. -/
theorem inner_eq_sum_slices (ψ φ : Phys (Fin n → K) Anc Rec) :
    inner ℂ ψ φ = ∑ w : Anc × Rec,
      inner ℂ (WithLp.toLp 2 fun s => ψ (s, w) : EuclideanSpace ℂ (Fin n → K))
        (WithLp.toLp 2 fun s => φ (s, w)) := by
  rw [PiLp.inner_apply,
    Fintype.sum_prod_type (fun z : (Fin n → K) × Anc × Rec => inner ℂ (ψ z) (φ z)),
    Finset.sum_comm]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [PiLp.inner_apply]

omit [DecidableEq Anc] in
/-- The lifted objective keeps the norm bound `m`. -/
theorem norm_synOpFull_le (χ : AddChar K ℂ) (hχ : χ ≠ 1) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ : Phys (Fin n → K) Anc Rec) :
    ‖synOpFull χ B Fs ψ‖ ≤ (m : ℝ) * ‖ψ‖ := by
  have hsq : ‖synOpFull χ B Fs ψ‖ ^ 2 ≤ ((m : ℝ) * ‖ψ‖) ^ 2 := by
    rw [norm_sq_eq_sum_slices, mul_pow, norm_sq_eq_sum_slices ψ, Finset.mul_sum]
    refine Finset.sum_le_sum fun w _ => ?_
    rw [synOpFull_slice, ← mul_pow]
    have h := norm_synOp_le χ hχ B Fs (WithLp.toLp 2 fun s => ψ (s, w))
    rw [pow_two, pow_two]
    exact mul_le_mul h h (norm_nonneg _) ((norm_nonneg _).trans h)
  have h1 := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (by positivity)] at h1

omit [DecidableEq Anc] in
/-- The lifted objective is a symmetric form. -/
theorem synOpFull_inner (χ : AddChar K ℂ) (hχ : χ ≠ 1) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ φ : Phys (Fin n → K) Anc Rec) :
    inner ℂ (synOpFull χ B Fs ψ) φ = inner ℂ ψ (synOpFull χ B Fs φ) := by
  rw [inner_eq_sum_slices, inner_eq_sum_slices]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [synOpFull_slice, synOpFull_slice, synOp_inner χ hχ]

/-- The clean-register embedding preserves inner products of pushforwards when
the record state is a unit vector. -/
theorem inner_toPhys_toPhys (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (hrec : ‖wrec‖ = 1)
    (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    inner ℂ (toPhys B a₀ wrec ψ) (toPhys B a₀ wrec φ)
      = inner ℂ (toSyn B ψ) (toSyn B φ) := by
  rw [inner_eq_sum_slices]
  have hterm : ∀ w : Anc × Rec,
      inner ℂ
          (WithLp.toLp 2 fun s => toPhys B a₀ wrec ψ (s, w) : EuclideanSpace ℂ (Fin n → K))
          (WithLp.toLp 2 fun s => toPhys B a₀ wrec φ (s, w))
        = (if w.1 = a₀ then wrec w.2 * (starRingEnd ℂ) (wrec w.2) else 0) *
            inner ℂ (toSyn B ψ) (toSyn B φ) := by
    intro w
    rw [toPhys_slice, toPhys_slice, inner_smul_left, inner_smul_right]
    by_cases h : w.1 = a₀
    · rw [if_pos h, if_pos h, one_mul]
      ring
    · rw [if_neg h, if_neg h]
      simp only [zero_mul, mul_zero, map_zero]
  rw [Finset.sum_congr rfl fun w _ => hterm w, ← Finset.sum_mul]
  have hsum : (∑ w : Anc × Rec,
      if w.1 = a₀ then wrec w.2 * (starRingEnd ℂ) (wrec w.2) else 0) = (1 : ℂ) := by
    rw [Fintype.sum_prod_type (fun w : Anc × Rec =>
      if w.1 = a₀ then wrec w.2 * (starRingEnd ℂ) (wrec w.2) else 0)]
    have ha : ∀ a : Anc,
        (∑ rc : Rec, if a = a₀ then wrec rc * (starRingEnd ℂ) (wrec rc) else 0)
          = if a = a₀ then ∑ rc : Rec, wrec rc * (starRingEnd ℂ) (wrec rc) else 0 := by
      intro a
      by_cases h : a = a₀
      · simp only [if_pos h]
      · simp only [if_neg h, Finset.sum_const_zero]
    rw [Finset.sum_congr rfl fun a _ => ha a,
      Finset.sum_ite_eq' Finset.univ a₀
        (fun _ => ∑ rc : Rec, wrec rc * (starRingEnd ℂ) (wrec rc)),
      if_pos (Finset.mem_univ a₀)]
    have hw : (∑ rc : Rec, wrec rc * (starRingEnd ℂ) (wrec rc)) = inner ℂ wrec wrec := by
      rw [PiLp.inner_apply]
      exact Finset.sum_congr rfl fun rc _ => (RCLike.inner_apply _ _).symm
    rw [hw, inner_self_eq_norm_sq_to_K, hrec]
    norm_num
  rw [hsum, one_mul]

/-- Norm form of the metric embedding. -/
theorem norm_toPhys (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (hrec : ‖wrec‖ = 1)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    ‖toPhys B a₀ wrec ψ‖ = ‖toSyn B ψ‖ := by
  have h2 : ‖toPhys B a₀ wrec ψ‖ ^ 2 = ‖toSyn B ψ‖ ^ 2 := by
    rw [← inner_self_eq_norm_sq (𝕜 := ℂ), ← inner_self_eq_norm_sq (𝕜 := ℂ),
      inner_toPhys_toPhys B a₀ wrec hrec ψ ψ]
  have h3 := congrArg Real.sqrt h2
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at h3

omit [Fintype Anc] [Fintype Rec] in
/-- The clean-register embedding is additive. -/
theorem toPhys_add (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    toPhys B a₀ wrec (ψ + φ) = toPhys B a₀ wrec ψ + toPhys B a₀ wrec φ := by
  apply PiLp.ext
  intro z
  rw [PiLp.add_apply, toPhys_apply, toPhys_apply, toPhys_apply, toSyn_add,
    PiLp.add_apply]
  ring

omit [Fintype Anc] [Fintype Rec] in
/-- The clean-register embedding commutes with scalars. -/
theorem toPhys_smul (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (a : ℂ) (ψ : EuclideanSpace ℂ (Pattern m K)) :
    toPhys B a₀ wrec (a • ψ) = a • toPhys B a₀ wrec ψ := by
  apply PiLp.ext
  intro z
  rw [PiLp.smul_apply, toPhys_apply, toPhys_apply, toSyn_smul, PiLp.smul_apply,
    smul_eq_mul, smul_eq_mul]
  ring

omit [Fintype Anc] [Fintype Rec] in
/-- **The lifted intertwining.**  The objective on the full register space,
applied to a clean-register embedding, is the embedding of the pattern-space
shell operator — the point of the file. -/
theorem synOpFull_toPhys (χ : AddChar K ℂ) (hχ : χ ≠ 1) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (a₀ : Anc) (wrec : EuclideanSpace ℂ Rec)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    synOpFull χ B Fs (toPhys B a₀ wrec ψ) = toPhys B a₀ wrec (shellOp χ Fs ψ) := by
  apply PiLp.ext
  rintro ⟨s, w⟩
  show synOp χ B Fs (WithLp.toLp 2 fun t => toPhys B a₀ wrec ψ (t, w)) s
      = (if w.1 = a₀ then 1 else 0) * toSyn B (shellOp χ Fs ψ) s * wrec w.2
  rw [toPhys_slice, synOp_smul, synOp_toSyn χ hχ, PiLp.smul_apply, smul_eq_mul]
  ring

omit [Fintype Anc] [Fintype Rec] in
/-- The embedding is supported on the clean ancilla value. -/
theorem toPhys_eq_zero_of_anc (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (ψ : EuclideanSpace ℂ (Pattern m K))
    (z : (Fin n → K) × Anc × Rec) (hz : z.2.1 ≠ a₀) :
    toPhys B a₀ wrec ψ z = 0 := by
  rw [toPhys_apply, if_neg hz, zero_mul, zero_mul]

omit [Fintype Anc] [Fintype Rec] in
/-- The embedding is supported on the syndrome support of the pushforward. -/
theorem toPhys_eq_zero_of_syn (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (ψ : EuclideanSpace ℂ (Pattern m K))
    (z : (Fin n → K) × Anc × Rec) (hz : toSyn B ψ z.1 = 0) :
    toPhys B a₀ wrec ψ z = 0 := by
  rw [toPhys_apply, hz, mul_zero, zero_mul]

end Dqi
