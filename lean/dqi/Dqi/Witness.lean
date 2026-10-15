import Dqi.Complete

/-!
# Satisfiability witness for `Dqi.dqi_end_to_end`

A concrete instantiation discharging every hypothesis of the terminal theorem,
in the non-degenerate regime `l = 1`:

* `K = ZMod 2`, `m = 7`, `n = 4`, `B` = the transpose of a generator matrix of
  the `[7,4]` Hamming code, so the dual code is the `[7,3]` simplex code —
  every nonzero dual codeword has weight exactly `4`, whence
  `2 * 1 + 1 = 3 < dualDist B`.
* `Fs i = {1}`: a single nonzero frequency, `r = 1`.
* `w` = the `(3/5, 4/5)` profile on layers `0, 1`.
* `Anc = Rec = Unit`, the ideal oracle `U = V = toPhys B () wrec`, which is
  branch-diagonal with ideal labels, fixed unit record state, and unit phase.
* `F = ∅`, with the paper's budget numbers `mu = 2434/1000000`,
  `mucl = 37/1000000`.

The only free data is a nontrivial additive character `χ` on `ZMod 2`, whose
existence is `Dqi.exists_addChar_ne_one`.
-/

namespace DqiWitness

open Dqi

/-! ### Instance data -/

/-- Transpose of a `[7,4]` Hamming-code generator matrix: its column space is
the Hamming code, so the dual code `{y ≠ 0 | Bᵀ y = 0}` is the `[7,3]` simplex
code (all nonzero weights `4`). -/
def B : Matrix (Fin 7) (Fin 4) (ZMod 2) :=
  Matrix.of ![![1,0,0,0], ![0,1,0,0], ![0,0,1,0], ![0,0,0,1],
    ![1,1,0,1], ![1,0,1,1], ![0,1,1,1]]

/-- One nonzero target frequency per constraint. -/
def Fs : Fin 7 → Finset (ZMod 2) := fun _ => {1}

/-- The truncated weight profile `(3/5, 4/5)` on layers `0, 1`. -/
noncomputable def w : ℕ → ℝ :=
  fun k => if k = 0 then 3 / 5 else if k = 1 then 4 / 5 else 0

/-- Total failure-mass budget (the paper's number; any nonnegative bound works
since `F = ∅`). -/
noncomputable def mu : ℝ := 2434 / 1000000

/-- Clean-relocation mass budget. -/
noncomputable def mucl : ℝ := 37 / 1000000

/-! ### Code facts -/

instance (y : Pattern 7 (ZMod 2)) : Decidable (IsDualCodeword B y) :=
  inferInstanceAs (Decidable (syndrome B y = 0))

/-- No nonzero dual codeword of weight at most `3` exists: a finite check over
the `2 ^ 7` patterns. -/
theorem low_weight_dual_empty :
    (Finset.univ.filter fun y : Pattern 7 (ZMod 2) =>
      y ≠ 0 ∧ syndrome B y = 0 ∧ hammingNorm y ≤ 3) = ∅ := by decide

/-- Every nonzero dual codeword has weight at least `4`. -/
theorem dual_weight_lb (y : Pattern 7 (ZMod 2)) (hy0 : y ≠ 0)
    (hyd : IsDualCodeword B y) : 4 ≤ hammingNorm y := by
  by_contra h
  have hy : y ∈ Finset.univ.filter fun y : Pattern 7 (ZMod 2) =>
      y ≠ 0 ∧ syndrome B y = 0 ∧ hammingNorm y ≤ 3 :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ y, hy0, hyd, by omega⟩
  rw [low_weight_dual_empty] at hy
  simp at hy

/-- The dual distance is at least `4`: the weight set is nonempty (the simplex
codeword `1101100` has weight `4`), and every member weight is at least `4`. -/
theorem four_le_dualDist : 4 ≤ dualDist B := by
  unfold dualDist
  have hne : {k | ∃ y : Pattern 7 (ZMod 2),
      y ≠ 0 ∧ IsDualCodeword B y ∧ hammingNorm y = k}.Nonempty :=
    ⟨4, ![1,1,0,1,1,0,0], by decide, by decide, by decide⟩
  obtain ⟨y, hy0, hyd, hyw⟩ := Nat.sInf_mem hne
  rw [← hyw]
  exact dual_weight_lb y hy0 hyd

/-- The strict-distance condition at decoding radius `l = 1`. -/
theorem hstrict : 2 * 1 + 1 < dualDist B := by
  have := four_le_dualDist
  omega

/-- The edge-layer collision count: every weight-`2` fiber of the syndrome map
has at most `3` elements (three pair-splits of a weight-`4` simplex codeword
can share a syndrome, so `2` would not suffice). -/
theorem hcoll : ∀ u : Fin 4 → ZMod 2,
    (Finset.univ.filter fun y : Pattern 7 (ZMod 2) =>
      hammingNorm y = 1 + 1 ∧ syndrome B y = u).card ≤ 3 := by decide

/-! ### Small instance-data facts -/

theorem hcard : ∀ i : Fin 7, (Fs i).card = 1 := fun _ => rfl

theorem hw : ∑ k ∈ Finset.range (1 + 1), w k ^ 2 = 1 := by
  rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num [w]

/-- Bent constancy is trivial over `ZMod 2`: there is only one nonzero
frequency. -/
theorem hFsconst (χ : AddChar (ZMod 2) ℂ) :
    ∀ i, ∀ a b : ZMod 2, a ≠ 0 → b ≠ 0 → ‖ghat χ (Fs i) a‖ = ‖ghat χ (Fs i) b‖ := by
  intro i a b ha hb
  have h1 : ∀ x : ZMod 2, x ≠ 0 → x = 1 := by decide
  rw [h1 a ha, h1 b hb]

/-! ### The record state and the (ideal = implemented) oracle -/

/-- The unit record state on the one-point record register. -/
noncomputable def wrec : EuclideanSpace ℂ Unit := WithLp.toLp 2 fun _ => 1

theorem wrec_apply (u : Unit) : wrec u = 1 := rfl

theorem norm_wrec : ‖wrec‖ = 1 := by
  rw [EuclideanSpace.norm_eq]
  simp [wrec_apply]

/-- The oracle: the clean-register embedding, as a linear map. -/
noncomputable def Umap :
    EuclideanSpace ℂ (Pattern 7 (ZMod 2)) →ₗ[ℂ] Phys (Fin 4 → ZMod 2) Unit Unit where
  toFun := toPhys B () wrec
  map_add' := toPhys_add B () wrec
  map_smul' := toPhys_smul B () wrec

theorem Umap_apply (ψ : EuclideanSpace ℂ (Pattern 7 (ZMod 2))) :
    Umap ψ = toPhys B () wrec ψ := rfl

/-- The pushforward of a basis vector is a delta at its syndrome (local mirror
of the private `toSyn_single_apply` of `Dqi.Intertwine`). -/
theorem toSynB_single_apply (y₀ : Pattern 7 (ZMod 2)) (u : Fin 4 → ZMod 2) :
    toSyn B (EuclideanSpace.single y₀ (1 : ℂ)) u
      = if u = syndrome B y₀ then 1 else 0 := by
  rw [toSyn_apply]
  have hterm : ∀ y : Pattern 7 (ZMod 2),
      (if syndrome B y = u then (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)
        = if y = y₀ then (if syndrome B y = u then (1 : ℂ) else 0) else 0 := by
    intro y
    rw [PiLp.single_apply]
    by_cases hy : y = y₀ <;> by_cases hs : syndrome B y = u <;> simp [hy, hs]
  rw [Finset.sum_congr rfl fun y _ => hterm y,
    Finset.sum_ite_eq' Finset.univ y₀ fun y => if syndrome B y = u then (1 : ℂ) else 0,
    if_pos (Finset.mem_univ y₀)]
  by_cases h : syndrome B y₀ = u
  · rw [if_pos h, if_pos h.symm]
  · rw [if_neg h, if_neg fun hu => h hu.symm]

/-- `Umap` is branch-diagonal: ideal labels, the fixed unit record state, unit
phase. -/
noncomputable def hUbd : IsBranchDiagonal Umap where
  lbl y := (syndrome B y, ())
  recSt _ := wrec
  phase _ := 1
  recSt_norm _ := norm_wrec
  phase_norm _ := norm_one
  apply_single y := by
    apply PiLp.ext
    intro z
    rw [Umap_apply, toPhys_apply, toSynB_single_apply, atLabel_apply]
    by_cases h : z.1 = syndrome B y <;> simp [h, wrec_apply]

/-- Restricted isometry on the decodable subspace: two weight-`≤ 1` patterns
with equal syndromes coincide, since `2 * 1 < dualDist B`. -/
theorem hUiso : ∀ x : EuclideanSpace ℂ (Pattern 7 (ZMod 2)),
    (∀ y, x y ≠ 0 → hammingNorm y ≤ 1) → ‖Umap x‖ = ‖x‖ := by
  intro x hx
  rw [Umap_apply, norm_toPhys B () wrec norm_wrec]
  exact norm_toSyn B x fun y z hy hz hsyn =>
    syndrome_injOn_decodable (B := B) (l := 1)
      (by have := four_le_dualDist; omega) (hx y hy) (hx z hz) hsyn

/-! ### The pre-oracle state -/

/-- A weighted combination of the layers `0, …, l` vanishes above weight `l`
(local mirror of the private `layerCombo_apply_eq_zero` of `Dqi.Complete`). -/
theorem layerCombo_zero (g : Fin 7 → ZMod 2 → ℂ) (c : ℕ → ℝ) (l : ℕ)
    (y : Pattern 7 (ZMod 2)) (hy : l < hammingNorm y) :
    (∑ j ∈ Finset.range (l + 1), ((c j : ℝ) : ℂ) • layerState g j) y = 0 := by
  simp only [WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul, Pi.smul_apply,
    smul_eq_mul]
  refine Finset.sum_eq_zero fun j hj => ?_
  have hne : hammingNorm y ≠ j := by
    have := Finset.mem_range.mp hj
    omega
  rw [layerState_apply, if_neg hne, mul_zero]

/-- The pre-oracle state: the DQI state of the profile `w`, truncated at
`l = 1`. -/
noncomputable def Ψ (χ : AddChar (ZMod 2) ℂ) : EuclideanSpace ℂ (Pattern 7 (ZMod 2)) :=
  dqiState (amp χ Fs) w 1

/-- The pre-oracle state is supported on weights at most `1`. -/
theorem Ψsupp (χ : AddChar (ZMod 2) ℂ) :
    ∀ y, Ψ χ y ≠ 0 → hammingNorm y ≤ 1 := by
  intro y hy
  by_contra hgt
  refine hy ?_
  show dqiState (amp χ Fs) w 1 y = 0
  rw [dqiState_def]
  exact layerCombo_zero (amp χ Fs) w 1 y (by omega)

/-! ### The witness -/

open scoped Classical in
/-- **The hypothesis cluster of `Dqi.dqi_end_to_end` is jointly satisfiable in
the non-degenerate regime `l = 1`**: the proof is `dqi_end_to_end` with every
argument concrete, except the nontrivial character, which is universally
quantified (and exists by `Dqi.exists_addChar_ne_one`). -/
theorem witness_end_to_end (χ : AddChar (ZMod 2) ℂ) (hχ : χ ≠ 1) :
    |expct (synOpFullL χ B Fs) (Umap (Ψ χ)) - lamF 7 1 (Fintype.card (ZMod 2)) w 1|
      ≤ (|lamF 7 1 (Fintype.card (ZMod 2)) w 1| + ((7 : ℕ) : ℝ)) * (4 * mu)
        + 4 * Real.sqrt mu * sigmaDec 7 1 (Fintype.card (ZMod 2)) w 1
        + 2 * Real.sqrt mucl *
            (Real.sqrt ((((∅ : Finset (Pattern 7 (ZMod 2))).filter
                  fun y => IsCleanRelocation hUbd hUbd y).card : ℝ)) *
              (sigmaEdge 7 1 (Fintype.card (ZMod 2)) w 1 * ((3 : ℕ) : ℝ) *
                (Real.sqrt ((shellCard 7 (ZMod 2) (1 + 1) : ℕ) : ℝ))⁻¹)) := by
  refine dqi_end_to_end χ hχ B Fs 1 hcard Nat.one_pos (by decide) 1 (by norm_num)
    hstrict 3 hcoll (hFsconst χ) w hw () wrec norm_wrec hUbd hUbd hUiso hUiso
    (fun y _ => rfl) (fun y _ => rfl) (fun y _ => Or.inl rfl) (Ψ χ) rfl (Ψsupp χ) ∅
    (fun x _ _ => rfl) mu mucl (by norm_num [mu]) ?_ ?_
  · have h0 : mass (Ψ χ) (∅ : Finset (Pattern 7 (ZMod 2))) = 0 := by simp [mass]
    rw [h0]
    norm_num [mu]
  · rw [Finset.filter_empty]
    have h0 : mass (Ψ χ) (∅ : Finset (Pattern 7 (ZMod 2))) = 0 := by simp [mass]
    rw [h0]
    norm_num [mucl]

/-- The remaining free datum exists: `ZMod 2` carries a nontrivial additive
character. -/
example : ∃ χ : AddChar (ZMod 2) ℂ, χ ≠ 1 :=
  exists_addChar_ne_one (ZMod 2)

end DqiWitness

#print axioms DqiWitness.witness_end_to_end
