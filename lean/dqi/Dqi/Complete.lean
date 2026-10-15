import Dqi.Main
import Dqi.CrossPhys
import Dqi.Ewire

/-!
# The end-to-end master theorem

This file states and proves the terminal theorem of the development: the
degradation bound with **every internal hypothesis discharged**.  Its hypotheses are
only

* the instance data (character, constraint matrix, target sets, cardinality),
* the cited code facts (the strict-distance condition and the edge-layer
  collision count),
* bent constancy of the target sets,
* the truncated weight profile,
* two branch-diagonal oracles, isometric on the decodable subspace — the ideal
  one with ideal labels and a fixed unit record state, the implemented one
  leaving each branch's record either clean or orthogonal to the clean state,
  all asserted **on the decodable subspace only**
  (patterns of weight at most `l`) — agreeing off a failure set, and
* the two measured masses (total failure mass and clean-relocation mass).

Everything else — normalization of the DQI state, blindness of the sign and
leakage channels, the residual split, the relocation reach, and the sharp
cross term — is derived inside from the theorems of the development.

The file also packages the full-register objective `Dqi.synOpFull` as a
bounded self-adjoint continuous linear map `Dqi.synOpFullL`, mirroring
`Dqi.shellOpL`.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m n : ℕ}

/-! ### Pattern-space helpers -/

section Helpers

omit [Fintype K] in
/-- **The cross-collision discharge.**  At strict distance, a pattern of
weight at most `l` and a pattern of weight at most `l + 1` sharing a syndrome
are equal: their difference would be a nonzero dual codeword of weight at most
`2 * l + 1`. -/
private theorem eq_of_syndrome_eq_of_wt {B : Matrix (Fin m) (Fin n) K} {l : ℕ}
    (hstrict : 2 * l + 1 < dualDist B) {y z : Pattern m K}
    (hy : hammingNorm y ≤ l) (hz : hammingNorm z ≤ l + 1)
    (hsyn : syndrome B y = syndrome B z) : y = z := by
  by_contra hne
  have hsub : y - z ≠ 0 := sub_ne_zero_of_ne hne
  have hlow : dualDist B ≤ hammingNorm (y - z) :=
    dualDist_le_hammingNorm hsub (isDualCodeword_sub_of_syndrome_eq hsyn)
  have htri : hammingNorm (y - z) ≤ hammingNorm y + hammingNorm z :=
    hammingNorm_sub_le y z
  omega

omit [Fintype K] in
/-- A real-weighted combination of the layers `0, …, l` vanishes on every
pattern of weight above `l`. -/
private theorem layerCombo_apply_eq_zero (g : Fin m → K → ℂ) (c : ℕ → ℝ) (l : ℕ)
    (y : Pattern m K) (hy : l < hammingNorm y) :
    (∑ j ∈ Finset.range (l + 1), ((c j : ℝ) : ℂ) • layerState g j) y = 0 := by
  simp only [WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul, Pi.smul_apply,
    smul_eq_mul]
  refine Finset.sum_eq_zero fun j hj => ?_
  have hne : hammingNorm y ≠ j := by
    have := Finset.mem_range.mp hj
    omega
  rw [layerState_apply, if_neg hne, mul_zero]

end Helpers

/-! ### The full-register objective as a bounded self-adjoint operator -/

section FullOp

variable {Anc Rec : Type*} [Fintype Anc] [DecidableEq Anc] [Fintype Rec] [DecidableEq Rec]

omit [Fintype Anc] [DecidableEq Anc] [Fintype Rec] [DecidableEq Rec] in
/-- The lifted objective is additive: slices of sums are sums of slices. -/
theorem synOpFull_add (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ φ : Phys (Fin n → K) Anc Rec) :
    synOpFull χ B Fs (ψ + φ) = synOpFull χ B Fs ψ + synOpFull χ B Fs φ := by
  apply PiLp.ext
  intro z
  have hslice : (WithLp.toLp 2 fun s => (ψ + φ) (s, z.2) : EuclideanSpace ℂ (Fin n → K))
      = (WithLp.toLp 2 fun s => ψ (s, z.2)) + (WithLp.toLp 2 fun s => φ (s, z.2)) := by
    apply PiLp.ext
    intro s
    rw [PiLp.add_apply]
    rfl
  rw [synOpFull_apply, hslice, synOp_add, PiLp.add_apply, PiLp.add_apply]
  rfl

omit [Fintype Anc] [DecidableEq Anc] [Fintype Rec] [DecidableEq Rec] in
/-- The lifted objective commutes with scalars. -/
theorem synOpFull_smul (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (a : ℂ) (ψ : Phys (Fin n → K) Anc Rec) :
    synOpFull χ B Fs (a • ψ) = a • synOpFull χ B Fs ψ := by
  apply PiLp.ext
  intro z
  have hslice : (WithLp.toLp 2 fun s => (a • ψ) (s, z.2) : EuclideanSpace ℂ (Fin n → K))
      = a • (WithLp.toLp 2 fun s => ψ (s, z.2)) := by
    apply PiLp.ext
    intro s
    rw [PiLp.smul_apply]
    rfl
  rw [synOpFull_apply, hslice, synOp_smul, PiLp.smul_apply, PiLp.smul_apply]
  rfl

/-- The full-register DQI objective `Dqi.synOpFull`, packaged as a continuous
linear operator on the physical state space. -/
noncomputable def synOpFullL (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) :
    Phys (Fin n → K) Anc Rec →L[ℂ] Phys (Fin n → K) Anc Rec :=
  LinearMap.toContinuousLinearMap
    { toFun := synOpFull χ B Fs
      map_add' := synOpFull_add χ B Fs
      map_smul' := synOpFull_smul χ B Fs }

omit [DecidableEq Anc] [DecidableEq Rec] in
@[simp] theorem synOpFullL_apply (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (ψ : Phys (Fin n → K) Anc Rec) :
    synOpFullL χ B Fs ψ = synOpFull χ B Fs ψ :=
  rfl

omit [DecidableEq Anc] [DecidableEq Rec] in
/-- **The full-register objective operator is self-adjoint**: it inherits the
symmetry of `synOpFull`. -/
theorem isSelfAdjoint_synOpFullL (χ : AddChar K ℂ) (hχ : χ ≠ 1)
    (B : Matrix (Fin m) (Fin n) K) (Fs : Fin m → Finset K) :
    IsSelfAdjoint (synOpFullL (Anc := Anc) (Rec := Rec) χ B Fs) := by
  refine LinearMap.IsSymmetric.isSelfAdjoint ?_
  intro ψ φ
  simp only [ContinuousLinearMap.coe_coe, synOpFullL_apply]
  exact synOpFull_inner χ hχ B Fs ψ φ

omit [DecidableEq Anc] [DecidableEq Rec] in
/-- **The full-register objective operator has norm at most `m`**. -/
theorem norm_synOpFullL_le (χ : AddChar K ℂ) (hχ : χ ≠ 1)
    (B : Matrix (Fin m) (Fin n) K) (Fs : Fin m → Finset K) :
    ‖synOpFullL (Anc := Anc) (Rec := Rec) χ B Fs‖ ≤ (m : ℝ) := by
  refine ContinuousLinearMap.opNorm_le_bound _ (Nat.cast_nonneg m) fun ψ => ?_
  rw [synOpFullL_apply]
  exact norm_synOpFull_le χ hχ B Fs ψ

end FullOp

/-! ### The theorem -/

open scoped Classical in
/-- **The end-to-end master theorem** — every internal hypothesis discharged.  The
hypotheses are the instance data, the cited code facts (the strict-distance
condition and the edge-layer collision count), bent constancy of the target
sets, the truncated weight profile, two branch-diagonal oracles — the ideal
one with ideal labels and a fixed unit record state, the implemented one
leaving each branch's record either clean or orthogonal to the clean state —
with isometry, labels, record states, and
agreement off the failure set `F` all asserted **on the decodable subspace
only** (states supported on patterns of weight at most `l`), and the two
measured masses.  The restriction to the decodable subspace is forced, not a
convenience: the global versions of these hypotheses are contradictory, since
any nonzero dual codeword shares syndrome `0` with the zero pattern, so a
global isometry cannot send both basis vectors to the same labeled branch.
The conclusion bounds how far the measured full-register objective's
expectation on the implemented state can move from the ideal value `lamF`. -/
theorem dqi_end_to_end
    -- instance data
    (χ : AddChar K ℂ) (hχ : χ ≠ 1) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (r : ℕ) (hcard : ∀ i, (Fs i).card = r)
    (hr : 0 < r) (hrq : r < Fintype.card K)
    -- (i) code distance, strict regime
    (l : ℕ) (hl : l < m) (hstrict : 2 * l + 1 < dualDist B)
    -- code collision count (code-facts cluster, cited)
    (ncoll : ℕ)
    (hcoll : ∀ u : Fin n → K, (Finset.univ.filter fun y : Pattern m K =>
        hammingNorm y = l + 1 ∧ syndrome B y = u).card ≤ ncoll)
    -- (iii) bent constancy
    (hconst : ∀ i, ∀ a b : K, a ≠ 0 → b ≠ 0 → ‖ghat χ (Fs i) a‖ = ‖ghat χ (Fs i) b‖)
    -- (iv) weight profile
    (w : ℕ → ℝ) (hw : ∑ k ∈ Finset.range (l + 1), w k ^ 2 = 1)
    -- (v) the two oracles
    {Anc Rec : Type*} [Fintype Anc] [DecidableEq Anc] [Fintype Rec] [DecidableEq Rec]
    (a₀ : Anc) (wrec : EuclideanSpace ℂ Rec) (hrec : ‖wrec‖ = 1)
    {U V : EuclideanSpace ℂ (Pattern m K) →ₗ[ℂ] Phys (Fin n → K) Anc Rec}
    (hU : IsBranchDiagonal U) (hV : IsBranchDiagonal V)
    (hUiso : ∀ x : EuclideanSpace ℂ (Pattern m K),
      (∀ y, x y ≠ 0 → hammingNorm y ≤ l) → ‖U x‖ = ‖x‖)
    (hViso : ∀ x : EuclideanSpace ℂ (Pattern m K),
      (∀ y, x y ≠ 0 → hammingNorm y ≤ l) → ‖V x‖ = ‖x‖)
    (hUlbl : ∀ y, hammingNorm y ≤ l → hU.lbl y = (syndrome B y, a₀))
    (hUrec : ∀ y, hammingNorm y ≤ l → hU.recSt y = wrec)
    (horth : ∀ y, hammingNorm y ≤ l →
      (hV.recSt y = wrec ∨ inner ℂ (hV.recSt y) wrec = 0))
    -- the pre-oracle state and failure set
    (Ψpre : EuclideanSpace ℂ (Pattern m K))
    (hUΨ : U Ψpre = toPhys B a₀ wrec (dqiState (amp χ Fs) w l))
    (hsupp : ∀ y, Ψpre y ≠ 0 → hammingNorm y ≤ l)
    (F : Finset (Pattern m K))
    (hagree : ∀ x : EuclideanSpace ℂ (Pattern m K), (∀ y, x y ≠ 0 → hammingNorm y ≤ l) →
      (∀ y ∈ F, x y = 0) → V x = U x)
    -- (vi) measured masses
    (mu mucl : ℝ) (hmu0 : 0 ≤ mu) (hmu : mass Ψpre F ≤ mu)
    (hmucl : mass Ψpre (F.filter fun y => IsCleanRelocation hU hV y) ≤ mucl) :
    |expct (synOpFullL χ B Fs) (V Ψpre) - lamF m r (Fintype.card K) w l|
      ≤ (|lamF m r (Fintype.card K) w l| + m) * (4 * mu)
        + 4 * Real.sqrt mu * sigmaDec m r (Fintype.card K) w l
        + 2 * Real.sqrt mucl *
            (Real.sqrt (((F.filter fun y => IsCleanRelocation hU hV y).card : ℝ)) *
              (sigmaEdge m r (Fintype.card K) w l * ncoll *
                (Real.sqrt (shellCard m K (l + 1) : ℝ))⁻¹)) := by
  -- Step 1: support/collision facts and flatness.
  have hr' : ∀ i, 0 < (Fs i).card := fun i => by rw [hcard i]; exact hr
  have hrq' : ∀ i, (Fs i).card < Fintype.card K := fun i => by rw [hcard i]; exact hrq
  have hflat := amp_flat_of_const χ hχ Fs hr' hrq' hconst
  have hmucl0 : 0 ≤ mucl := le_trans (mass_nonneg Ψpre _) hmucl
  have hc1 : (0 : ℝ) ≤ cc1rq r (Fintype.card K) :=
    div_nonneg (Real.sqrt_nonneg _) (Nat.cast_nonneg _)
  have hsE : 0 ≤ sigmaEdge m r (Fintype.card K) w l := by
    unfold sigmaEdge
    exact mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) hc1) (Real.sqrt_nonneg _))
      (abs_nonneg _)
  have hbr0 : 0 ≤ sigmaEdge m r (Fintype.card K) w l * ncoll *
      (Real.sqrt (shellCard m K (l + 1) : ℝ))⁻¹ :=
    mul_nonneg (mul_nonneg hsE (Nat.cast_nonneg _)) (inv_nonneg.mpr (Real.sqrt_nonneg _))
  have herel0 : 0 ≤ Real.sqrt (((F.filter fun y => IsCleanRelocation hU hV y).card : ℝ)) *
      (sigmaEdge m r (Fintype.card K) w l * ncoll *
        (Real.sqrt (shellCard m K (l + 1) : ℝ))⁻¹) :=
    mul_nonneg (Real.sqrt_nonneg _) hbr0
  have hsdec0 : 0 ≤ sigmaDec m r (Fintype.card K) w l := Real.sqrt_nonneg _
  -- Step 2: supports of the DQI state and of its image under the objective.
  have hΨsupp : ∀ y : Pattern m K, dqiState (amp χ Fs) w l y ≠ 0 → hammingNorm y ≤ l := by
    intro y hy
    by_contra hgt
    refine hy ?_
    rw [dqiState_def]
    exact layerCombo_apply_eq_zero (amp χ Fs) w l y (by omega)
  have hres := shellOp_dqiState_residual χ hχ Fs r hcard hr hrq w l hl
  have hSsupp : ∀ z : Pattern m K,
      shellOp χ Fs (dqiState (amp χ Fs) w l) z ≠ 0 → hammingNorm z ≤ l + 1 := by
    intro z hz
    by_contra hgt
    refine hz ?_
    have h2 := sub_eq_iff_eq_add.mp hres
    have hz1 : (∑ j ∈ Finset.range (l + 1),
        ((2 * cc1rq r (Fintype.card K) *
              (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
            + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
          : ℝ) : ℂ) • layerState (amp χ Fs) j) z = 0 :=
      layerCombo_apply_eq_zero (amp χ Fs) _ l z (by omega)
    have hz2 : layerState (amp χ Fs) (l + 1) z = 0 := by
      rw [layerState_apply, if_neg (by omega)]
    have hz3 : dqiState (amp χ Fs) w l z = 0 := by
      rw [dqiState_def]
      exact layerCombo_apply_eq_zero (amp χ Fs) w l z (by omega)
    rw [h2, PiLp.add_apply, PiLp.add_apply, PiLp.smul_apply, PiLp.smul_apply,
      hz1, hz2, hz3]
    simp
  have hΨinj : ∀ y z : Pattern m K, dqiState (amp χ Fs) w l y ≠ 0 →
      dqiState (amp χ Fs) w l z ≠ 0 → syndrome B y = syndrome B z → y = z :=
    fun y z hy hz => eq_of_syndrome_eq_of_wt hstrict (hΨsupp y hy)
      ((hΨsupp z hz).trans (Nat.le_succ l))
  have hcrossΨS : ∀ y z : Pattern m K, dqiState (amp χ Fs) w l y ≠ 0 →
      shellOp χ Fs (dqiState (amp χ Fs) w l) z ≠ 0 →
      syndrome B y = syndrome B z → y = z :=
    fun y z hy hz => eq_of_syndrome_eq_of_wt hstrict (hΨsupp y hy) (hSsupp z hz)
  -- Step 3: normalization of the embedded state and of the implemented state.
  have hΦ : ‖toPhys B a₀ wrec (dqiState (amp χ Fs) w l)‖ = 1 := by
    rw [norm_toPhys B a₀ wrec hrec, norm_toSyn B _ hΨinj,
      norm_dqiState χ hχ Fs r hcard hr hrq w l hl hw]
  have hVeq : toPhys B a₀ wrec (dqiState (amp χ Fs) w l) + (V Ψpre - U Ψpre)
      = V Ψpre := by
    rw [← hUΨ]
    abel
  have hΦD : ‖toPhys B a₀ wrec (dqiState (amp χ Fs) w l) + (V Ψpre - U Ψpre)‖ = 1 := by
    rw [hVeq, hViso Ψpre hsupp, ← hΦ, ← hUΨ, hUiso Ψpre hsupp]
  -- Step 4: centering — the ideal expectation is `lamF`.
  have hexp : expct (synOpFullL χ B Fs) (toPhys B a₀ wrec (dqiState (amp χ Fs) w l))
      = lamF m r (Fintype.card K) w l := by
    unfold expct
    rw [synOpFullL_apply, synOpFull_toPhys χ hχ B Fs a₀ wrec,
      inner_toPhys_toPhys B a₀ wrec hrec, inner_toSyn_toSyn B _ _ hcrossΨS,
      inner_dqiState_shellOp χ hχ Fs r hcard hr hrq w l hl hw, Complex.ofReal_re]
  -- Step 5: the residual split, transported to the full register space.
  have hsplitPhys : synOpFullL χ B Fs (toPhys B a₀ wrec (dqiState (amp χ Fs) w l))
      - ((lamF m r (Fintype.card K) w l : ℝ) : ℂ)
          • toPhys B a₀ wrec (dqiState (amp χ Fs) w l)
      = toPhys B a₀ wrec (∑ j ∈ Finset.range (l + 1),
            ((2 * cc1rq r (Fintype.card K) *
                  (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
                + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l)
                    * w j
              : ℝ) : ℂ) • layerState (amp χ Fs) j)
        + toPhys B a₀ wrec (((2 * cc1rq r (Fintype.card K) * w l *
              Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
            • layerState (amp χ Fs) (l + 1)) := by
    rw [synOpFullL_apply, synOpFull_toPhys χ hχ B Fs a₀ wrec, ← toPhys_smul,
      ← toPhys_add, sub_eq_iff_eq_add, ← toPhys_add]
    congr 1
    rw [← hres]
    abel
  -- Step 6: the decodable residual size, through the metric embedding.
  have hX₁supp : ∀ y : Pattern m K, (∑ j ∈ Finset.range (l + 1),
      ((2 * cc1rq r (Fintype.card K) *
            (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
          + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
        : ℝ) : ℂ) • layerState (amp χ Fs) j) y ≠ 0 → hammingNorm y ≤ l := by
    intro y hy
    by_contra hgt
    exact hy (layerCombo_apply_eq_zero (amp χ Fs) _ l y (by omega))
  have hX₁inj : ∀ y z : Pattern m K, (∑ j ∈ Finset.range (l + 1),
      ((2 * cc1rq r (Fintype.card K) *
            (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
          + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
        : ℝ) : ℂ) • layerState (amp χ Fs) j) y ≠ 0 → (∑ j ∈ Finset.range (l + 1),
      ((2 * cc1rq r (Fintype.card K) *
            (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
          + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
        : ℝ) : ℂ) • layerState (amp χ Fs) j) z ≠ 0 →
      syndrome B y = syndrome B z → y = z :=
    fun y z hy hz => eq_of_syndrome_eq_of_wt hstrict (hX₁supp y hy)
      ((hX₁supp z hz).trans (Nat.le_succ l))
  have hdec : ‖inner ℂ (V Ψpre - U Ψpre) (toPhys B a₀ wrec
      (∑ j ∈ Finset.range (l + 1),
        ((2 * cc1rq r (Fintype.card K) *
              (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
            + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
          : ℝ) : ℂ) • layerState (amp χ Fs) j))‖
      ≤ ‖V Ψpre - U Ψpre‖ * sigmaDec m r (Fintype.card K) w l := by
    have hn : ‖toPhys B a₀ wrec (∑ j ∈ Finset.range (l + 1),
        ((2 * cc1rq r (Fintype.card K) *
              (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
            + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
          : ℝ) : ℂ) • layerState (amp χ Fs) j)‖
        = sigmaDec m r (Fintype.card K) w l := by
      rw [norm_toPhys B a₀ wrec hrec, norm_toSyn B _ hX₁inj,
        norm_residual_dec χ hχ Fs r hcard hr hrq w l hl]
    rw [← hn]
    exact norm_inner_le_norm (𝕜 := ℂ) _ _
  -- Step 7: perturbation size from the measured total mass.
  have hD : ‖V Ψpre - U Ψpre‖ ≤ 2 * Real.sqrt mu := by
    refine le_trans (norm_sub_le_two_sqrt_mass U V
      (S := {y : Pattern m K | hammingNorm y ≤ l}) hUiso hViso F hagree Ψpre hsupp) ?_
    have h := Real.sqrt_le_sqrt hmu
    linarith
  -- Steps 8–9: the sharp relocation cross term, with the counting reach value.
  have hξ : ∀ y : Pattern m K, (((2 * cc1rq r (Fintype.card K) * w l *
      Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
        • layerState (amp χ Fs) (l + 1)) y ≠ 0 → hammingNorm y = l + 1 := by
    intro y hy
    by_contra hne
    exact hy (by simp [layerState_apply, hne])
  have hcoefN : ‖((2 * cc1rq r (Fintype.card K) * w l *
      Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)‖
      = sigmaEdge m r (Fintype.card K) w l := by
    rw [Complex.norm_real, Real.norm_eq_abs]
    unfold sigmaEdge
    rw [abs_mul, abs_mul, abs_mul, abs_two, abs_of_nonneg hc1,
      abs_of_nonneg (Real.sqrt_nonneg _)]
    ring
  have hsubset : supp (V (restr (F.filter fun y => IsCleanRelocation hU hV y) Ψpre))
      ⊆ ((F.filter fun y => IsCleanRelocation hU hV y).image hV.lbl).biUnion
          (fun lb => labelBlock lb) := by
    intro z hz
    have hmem : z ∈ imageBlock hV.lbl (F.filter fun y => IsCleanRelocation hU hV y) := by
      by_contra hnz
      exact mem_supp.mp hz (apply_restr_support hV _ Ψpre z hnz)
    rwa [Finset.image_biUnion]
  have hreach : ‖restr (supp (V (restr (F.filter fun y => IsCleanRelocation hU hV y) Ψpre)))
      (toPhys B a₀ wrec (((2 * cc1rq r (Fintype.card K) * w l *
          Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
        • layerState (amp χ Fs) (l + 1)))‖
      ≤ Real.sqrt (((F.filter fun y => IsCleanRelocation hU hV y).card : ℝ)) *
          (sigmaEdge m r (Fintype.card K) w l * ncoll *
            (Real.sqrt (shellCard m K (l + 1) : ℝ))⁻¹) := by
    refine le_trans (norm_restr_mono hsubset _) (le_trans
      (norm_restr_labels_toPhys_le B a₀ wrec hrec (amp χ Fs) (l + 1) _ hflat ncoll
        hcoll _) ?_)
    rw [hcoefN]
    exact mul_le_mul_of_nonneg_right
      (Real.sqrt_le_sqrt (Nat.cast_le.mpr Finset.card_image_le)) hbr0
  have hDeq : V Ψpre - U Ψpre = V (restr F Ψpre) - U (restr F Ψpre) := by
    have hcompl : ∀ i ∈ F, (restr Fᶜ Ψpre) i = 0 := by
      intro i hi
      simp [restr_apply, Finset.mem_compl, hi]
    conv_lhs => rw [← restr_add_restr_compl F Ψpre]
    rw [map_add, map_add,
      hagree _ (fun y hy => hsupp y fun h0 => hy (by simp [restr_apply, h0])) hcompl]
    abel
  have hedge : ‖inner ℂ (V Ψpre - U Ψpre) (toPhys B a₀ wrec
      (((2 * cc1rq r (Fintype.card K) * w l *
          Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
        • layerState (amp χ Fs) (l + 1)))‖
      ≤ Real.sqrt mucl *
          (Real.sqrt (((F.filter fun y => IsCleanRelocation hU hV y).card : ℝ)) *
            (sigmaEdge m r (Fintype.card K) w l * ncoll *
              (Real.sqrt (shellCard m K (l + 1) : ℝ))⁻¹)) := by
    rw [hDeq]
    refine le_trans (inner_diff_toPhys_le hU hV l hstrict hUlbl hUrec horth hViso Ψpre
      hsupp F _ hξ _ hreach) ?_
    exact mul_le_mul_of_nonneg_right (Real.sqrt_le_sqrt hmucl) herel0
  -- Step 10: assemble through the master perturbation bound.
  have hcrossC : ‖inner ℂ (V Ψpre - U Ψpre)
      (synOpFullL χ B Fs (toPhys B a₀ wrec (dqiState (amp χ Fs) w l))
        - ((expct (synOpFullL χ B Fs) (toPhys B a₀ wrec (dqiState (amp χ Fs) w l))
            : ℝ) : ℂ) • toPhys B a₀ wrec (dqiState (amp χ Fs) w l))‖
      ≤ ‖V Ψpre - U Ψpre‖ * sigmaDec m r (Fintype.card K) w l
        + Real.sqrt mucl *
            (Real.sqrt (((F.filter fun y => IsCleanRelocation hU hV y).card : ℝ)) *
              (sigmaEdge m r (Fintype.card K) w l * ncoll *
                (Real.sqrt (shellCard m K (l + 1) : ℝ))⁻¹)) := by
    rw [hexp, hsplitPhys, inner_add_right]
    exact le_trans (norm_add_le _ _) (add_le_add hdec hedge)
  have hmain := expectation_perturbation_bound_of_cross (synOpFullL χ B Fs)
    (isSelfAdjoint_synOpFullL χ hχ B Fs)
    (toPhys B a₀ wrec (dqiState (amp χ Fs) w l)) (V Ψpre - U Ψpre) hΦ hΦD
    (m : ℝ) _ (norm_synOpFullL_le χ hχ B Fs) hcrossC
  rw [hVeq, hexp] at hmain
  have hDnn : (0 : ℝ) ≤ ‖V Ψpre - U Ψpre‖ := norm_nonneg _
  have hsq : ‖V Ψpre - U Ψpre‖ ^ 2 ≤ 4 * mu := by
    calc ‖V Ψpre - U Ψpre‖ ^ 2 ≤ (2 * Real.sqrt mu) ^ 2 := pow_le_pow_left₀ hDnn hD 2
      _ = 4 * mu := by rw [mul_pow, Real.sq_sqrt hmu0]; norm_num
  have hlin : ‖V Ψpre - U Ψpre‖ * sigmaDec m r (Fintype.card K) w l
      ≤ 2 * Real.sqrt mu * sigmaDec m r (Fintype.card K) w l :=
    mul_le_mul_of_nonneg_right hD hsdec0
  have hcoefnn : (0 : ℝ) ≤ |lamF m r (Fintype.card K) w l| + m :=
    add_nonneg (abs_nonneg _) (Nat.cast_nonneg m)
  have h4 := mul_le_mul_of_nonneg_left hsq hcoefnn
  refine le_trans hmain ?_
  linarith

end Dqi
