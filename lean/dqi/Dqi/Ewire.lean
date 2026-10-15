import Dqi.Bridge
import Dqi.Counting
import Dqi.PhysFull

/-!
# The counting value of the relocation reach

The relocation term of the degradation bound is, in the end, a counting
statement: an error that reaches at most `N` syndrome-ancilla labels sees at
most `√N` times the per-label amplitude of the escaping layer, and by flatness
the per-label amplitude of the physically embedded edge component is
`(collision count) × ‖c‖ / √(shell size)`.  This file proves that bound in the
physical register space — labels × flat amplitude × collisions.

Main results:
* `Dqi.labelBlock_disjoint`: distinct labels index disjoint register blocks.
* `Dqi.abs_toSyn_layer_le`: each syndrome value of the pushed-forward layer
  state carries amplitude at most `(collision count) / √(shell size)`.
* `Dqi.norm_restr_labels_toPhys_le`: the reach bound — restricting the
  physically embedded edge component to the blocks of a label set `S` yields
  norm at most `√|S| * ‖c‖ * (collision count) / √(shell size)`.
-/

namespace Dqi

/-! ### Distinct labels have disjoint blocks -/

variable {Syn Anc Rec : Type*}
variable [Fintype Syn] [DecidableEq Syn]
variable [Fintype Anc] [DecidableEq Anc]
variable [Fintype Rec] [DecidableEq Rec]

omit [DecidableEq Rec] in
/-- **Distinct labels index disjoint register blocks.** -/
theorem labelBlock_disjoint {lb lb' : Label Syn Anc} (h : lb ≠ lb') :
    Disjoint (labelBlock lb : Finset (Syn × Anc × Rec)) (labelBlock lb') :=
  Finset.disjoint_left.2 fun _z hz hz' => h (label_eq_of_mem_labelBlock hz hz')

/-! ### The per-syndrome amplitude of the pushed-forward layer -/

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m n : ℕ}

/-- **Per-syndrome amplitude of the pushed-forward edge layer.**  Under the
flatness hypothesis each weight-`k` pattern carries amplitude
`1/√(shellCard)`, and at most `ncoll` patterns of the shell collide onto any
one syndrome value, so each syndrome amplitude of the pushforward is at most
`ncoll / √(shellCard)`. -/
theorem abs_toSyn_layer_le (B : Matrix (Fin m) (Fin n) K) (g : Fin m → K → ℂ) (k : ℕ)
    (hflat : ∀ i, ∀ a : K, a ≠ 0 → ‖g i a‖ = (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹)
    (ncoll : ℕ)
    (hcoll : ∀ u : Fin n → K,
      (Finset.univ.filter fun y : Pattern m K =>
        hammingNorm y = k ∧ syndrome B y = u).card ≤ ncoll)
    (u : Fin n → K) :
    ‖toSyn B (layerState g k) u‖ ≤ ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹ := by
  have hval : toSyn B (layerState g k) u
      = ∑ y ∈ Finset.univ.filter (fun y : Pattern m K =>
          hammingNorm y = k ∧ syndrome B y = u), layerState g k y := by
    rw [toSyn_apply, ← Finset.sum_filter]
    refine (Finset.sum_subset (fun y hy => ?_) (fun y hy hyT => ?_)).symm
    · exact Finset.mem_filter.2 ⟨Finset.mem_univ _, (Finset.mem_filter.1 hy).2.2⟩
    · have hk : hammingNorm y ≠ k := fun hk =>
        hyT (Finset.mem_filter.2 ⟨Finset.mem_univ _, hk, (Finset.mem_filter.1 hy).2⟩)
      rw [layerState_apply, if_neg hk]
  rw [hval]
  calc ‖∑ y ∈ Finset.univ.filter (fun y : Pattern m K =>
          hammingNorm y = k ∧ syndrome B y = u), layerState g k y‖
      ≤ ∑ y ∈ Finset.univ.filter (fun y : Pattern m K =>
          hammingNorm y = k ∧ syndrome B y = u), ‖layerState g k y‖ :=
        norm_sum_le _ _
    _ = ∑ _y ∈ Finset.univ.filter (fun y : Pattern m K =>
          hammingNorm y = k ∧ syndrome B y = u),
          (Real.sqrt (shellCard m K k : ℝ))⁻¹ :=
        Finset.sum_congr rfl fun y hy =>
          norm_layerState_of_flat g k hflat y (Finset.mem_filter.1 hy).2.1
    _ = ((Finset.univ.filter fun y : Pattern m K =>
          hammingNorm y = k ∧ syndrome B y = u).card : ℝ) *
          (Real.sqrt (shellCard m K k : ℝ))⁻¹ := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹ :=
        mul_le_mul_of_nonneg_right (by exact_mod_cast hcoll u) (by positivity)

/-! ### The reach bound -/

/-- **The reach bound.**  Restricting the physically embedded edge component
`c • layerState g k` to the register blocks of a label set `S` yields norm at
most `√|S|` times the per-label amplitude `‖c‖ * ncoll / √(shellCard)`: within
one block the amplitude is the per-syndrome pushforward amplitude spread over
the unit record state, and the blocks of distinct labels are disjoint. -/
theorem norm_restr_labels_toPhys_le (B : Matrix (Fin m) (Fin n) K) (a₀ : Anc)
    (wrec : EuclideanSpace ℂ Rec) (hrec : ‖wrec‖ = 1)
    (g : Fin m → K → ℂ) (k : ℕ) (c : ℂ)
    (hflat : ∀ i, ∀ a : K, a ≠ 0 → ‖g i a‖ = (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹)
    (ncoll : ℕ)
    (hcoll : ∀ u : Fin n → K,
      (Finset.univ.filter fun y : Pattern m K =>
        hammingNorm y = k ∧ syndrome B y = u).card ≤ ncoll)
    (S : Finset (Label (Fin n → K) Anc)) :
    ‖restr (S.biUnion fun lb => labelBlock lb)
        (toPhys B a₀ wrec (c • layerState g k))‖
      ≤ Real.sqrt (S.card : ℝ) * (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) := by
  have hdisj : (S : Set (Label (Fin n → K) Anc)).PairwiseDisjoint
      fun lb => (labelBlock lb : Finset ((Fin n → K) × Anc × Rec)) :=
    fun _lb _ _lb' _ hne => labelBlock_disjoint hne
  -- each block carries at most the squared per-label amplitude
  have hblock : ∀ lb ∈ S,
      (∑ z ∈ (labelBlock lb : Finset ((Fin n → K) × Anc × Rec)),
        ‖toPhys B a₀ wrec (c • layerState g k) z‖ ^ 2)
      ≤ (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) ^ 2 := by
    intro lb _
    -- the block is a product: syndrome and ancilla fixed, record free
    have hbe : (labelBlock lb : Finset ((Fin n → K) × Anc × Rec))
        = {lb.1} ×ˢ ({lb.2} ×ˢ (Finset.univ : Finset Rec)) := by
      ext z
      simp only [mem_labelBlock, Finset.mem_product, Finset.mem_singleton,
        Finset.mem_univ, and_true]
    rw [hbe, Finset.sum_product, Finset.sum_singleton, Finset.sum_product,
      Finset.sum_singleton]
    by_cases ha : lb.2 = a₀
    · -- clean ancilla: the block sum factors through the unit record state
      have hterm : ∀ rc : Rec,
          ‖toPhys B a₀ wrec (c • layerState g k) (lb.1, lb.2, rc)‖ ^ 2
            = ‖toSyn B (c • layerState g k) lb.1‖ ^ 2 * ‖wrec rc‖ ^ 2 := by
        intro rc
        have hz : toPhys B a₀ wrec (c • layerState g k) (lb.1, lb.2, rc)
            = (if lb.2 = a₀ then 1 else 0) * toSyn B (c • layerState g k) lb.1
                * wrec rc := rfl
        rw [hz, if_pos ha, one_mul, norm_mul, mul_pow]
      have hsum : ∑ rc : Rec, ‖wrec rc‖ ^ 2 = 1 := by
        rw [← EuclideanSpace.norm_sq_eq, hrec, one_pow]
      have hle : ‖toSyn B (c • layerState g k) lb.1‖
          ≤ ‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹ := by
        rw [toSyn_smul, PiLp.smul_apply, smul_eq_mul, norm_mul, mul_assoc]
        exact mul_le_mul_of_nonneg_left
          (abs_toSyn_layer_le B g k hflat ncoll hcoll lb.1) (norm_nonneg c)
      calc ∑ rc : Rec, ‖toPhys B a₀ wrec (c • layerState g k) (lb.1, lb.2, rc)‖ ^ 2
          = ∑ rc : Rec, ‖toSyn B (c • layerState g k) lb.1‖ ^ 2 * ‖wrec rc‖ ^ 2 :=
            Finset.sum_congr rfl fun rc _ => hterm rc
        _ = ‖toSyn B (c • layerState g k) lb.1‖ ^ 2 * ∑ rc : Rec, ‖wrec rc‖ ^ 2 := by
            rw [← Finset.mul_sum]
        _ = ‖toSyn B (c • layerState g k) lb.1‖ ^ 2 := by rw [hsum, mul_one]
        _ ≤ (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) ^ 2 :=
            pow_le_pow_left₀ (norm_nonneg _) hle 2
    · -- dirty ancilla: the block carries nothing
      have hterm : ∀ rc : Rec,
          ‖toPhys B a₀ wrec (c • layerState g k) (lb.1, lb.2, rc)‖ ^ 2 = 0 := by
        intro rc
        have hz : toPhys B a₀ wrec (c • layerState g k) (lb.1, lb.2, rc)
            = (if lb.2 = a₀ then 1 else 0) * toSyn B (c • layerState g k) lb.1
                * wrec rc := rfl
        rw [hz, if_neg ha, zero_mul, zero_mul, norm_zero]
        norm_num
      calc ∑ rc : Rec, ‖toPhys B a₀ wrec (c • layerState g k) (lb.1, lb.2, rc)‖ ^ 2
          = ∑ _rc : Rec, (0 : ℝ) := Finset.sum_congr rfl fun rc _ => hterm rc
        _ = 0 := Finset.sum_const_zero
        _ ≤ (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) ^ 2 := by positivity
  -- sum the disjoint blocks and take the square root
  have hmass : mass (toPhys B a₀ wrec (c • layerState g k))
      (S.biUnion fun lb => labelBlock lb)
      ≤ (S.card : ℝ) * (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) ^ 2 := by
    rw [mass, Finset.sum_biUnion hdisj]
    calc ∑ lb ∈ S, ∑ z ∈ (labelBlock lb : Finset ((Fin n → K) × Anc × Rec)),
          ‖toPhys B a₀ wrec (c • layerState g k) z‖ ^ 2
        ≤ ∑ _lb ∈ S, (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) ^ 2 :=
          Finset.sum_le_sum hblock
      _ = (S.card : ℝ) * (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) ^ 2 := by
          rw [Finset.sum_const, nsmul_eq_mul]
  calc ‖restr (S.biUnion fun lb => labelBlock lb)
        (toPhys B a₀ wrec (c • layerState g k))‖
      = Real.sqrt (mass (toPhys B a₀ wrec (c • layerState g k))
          (S.biUnion fun lb => labelBlock lb)) :=
        norm_restr_eq_sqrt_mass _ _
    _ ≤ Real.sqrt ((S.card : ℝ) *
          (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) ^ 2) :=
        Real.sqrt_le_sqrt hmass
    _ = Real.sqrt (S.card : ℝ) *
          (‖c‖ * ncoll * (Real.sqrt (shellCard m K k : ℝ))⁻¹) := by
        rw [Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq (by positivity)]

end Dqi
