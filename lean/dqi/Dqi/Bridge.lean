import Dqi.Degradation

/-!
# The bridge from branch indexing to physical-register indexing

The degradation bound of `Dqi.Degradation` consumes probability mass measured
on the *branch* index `Pat`, while any physical observation lives on the
*register* index `Syn × Anc × Rec`.  This file connects the two: a
branch-diagonal oracle sends the branches in a set `F` into the block of
register basis states carrying their labels (`imageBlock`), and — this is the
point — the amplitude mass found on that block after the oracle equals the mass
the branches carried before it (`norm_restr_imageBlock`).

That last identity is the formal content of "the ideal decoder only applies
phases, so it moves no probability between branches": it is what lets the
failure mass consumed by the degradation bound be identified with the quantity
the validation protocol estimates by sampling branches.
-/

namespace Dqi

variable {Pat Syn Anc Rec : Type*}
variable [Fintype Pat] [DecidableEq Pat]
variable [Fintype Syn] [DecidableEq Syn]
variable [Fintype Anc] [DecidableEq Anc]
variable [Fintype Rec] [DecidableEq Rec]

/-- The computational-basis indices carrying a given syndrome-and-ancilla label. -/
def labelBlock (lb : Label Syn Anc) : Finset (Syn × Anc × Rec) :=
  Finset.univ.filter fun z => z.1 = lb.1 ∧ z.2.1 = lb.2

/-- The indices reached by a set of branches. -/
def imageBlock (lbl : Pat → Label Syn Anc) (F : Finset Pat) : Finset (Syn × Anc × Rec) :=
  F.biUnion fun y => labelBlock (lbl y)

omit [Fintype Pat] [DecidableEq Pat] [DecidableEq Rec] in
theorem mem_labelBlock {lb : Label Syn Anc} {z : Syn × Anc × Rec} :
    z ∈ labelBlock lb ↔ z.1 = lb.1 ∧ z.2.1 = lb.2 := by
  simp [labelBlock]

omit [Fintype Pat] [DecidableEq Pat] [DecidableEq Rec] in
/-- A register index determines the label of any block containing it. -/
theorem label_eq_of_mem_labelBlock {lb lb' : Label Syn Anc} {z : Syn × Anc × Rec}
    (h : z ∈ labelBlock lb) (h' : z ∈ labelBlock lb') : lb = lb' := by
  rw [mem_labelBlock] at h h'
  exact Prod.ext (h.1.symm.trans h'.1) (h.2.symm.trans h'.2)

omit [Fintype Pat] [Fintype Syn] [DecidableEq Syn] [Fintype Anc] [DecidableEq Anc]
  [Fintype Rec] [DecidableEq Rec] in
/-- **Basis expansion of a restriction.** -/
theorem restr_eq_sum (F : Finset Pat) (Ψ : EuclideanSpace ℂ Pat) :
    restr F Ψ = ∑ y ∈ F, Ψ y • EuclideanSpace.single y (1 : ℂ) := by
  refine PiLp.ext fun i => ?_
  simp only [restr_apply, WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul,
    Pi.smul_apply, PiLp.single_apply, smul_eq_mul, mul_ite, mul_one, mul_zero,
    Finset.sum_ite_eq, EuclideanSpace.single]

omit [Fintype Pat] in
/-- **A branch-diagonal image lands in the image block.** -/
theorem apply_restr_support {V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec}
    (hbd : IsBranchDiagonal V) (F : Finset Pat) (Ψ : EuclideanSpace ℂ Pat)
    (z : Syn × Anc × Rec) (hz : z ∉ imageBlock hbd.lbl F) :
    (V (restr F Ψ)) z = 0 := by
  rw [restr_eq_sum, map_sum]
  simp only [WithLp.ofLp_sum, Finset.sum_apply]
  refine Finset.sum_eq_zero fun y hy => ?_
  have hzy : z ∉ labelBlock (hbd.lbl y) := fun hmem =>
    hz (Finset.mem_biUnion.2 ⟨y, hy, hmem⟩)
  rw [mem_labelBlock] at hzy
  rw [map_smul, hbd.apply_single]
  simp [hzy]

/-- **Restriction to the image block recovers the restricted image.**  Injectivity
of the labelling on a set `S` containing `F` and the support of `Ψ` is what keeps
the block of `F` disjoint from the blocks of the populated complement branches. -/
theorem restr_imageBlock_apply {V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec}
    (hbd : IsBranchDiagonal V) {S : Set Pat} (hinj : Set.InjOn hbd.lbl S)
    (F : Finset Pat) (hFS : ∀ y ∈ F, y ∈ S)
    (Ψ : EuclideanSpace ℂ Pat) (hsupp : ∀ z : Pat, Ψ z ≠ 0 → z ∈ S) :
    restr (imageBlock hbd.lbl F) (V Ψ) = V (restr F Ψ) := by
  classical
  set G : Finset Pat := Fᶜ.filter fun z => Ψ z ≠ 0
  have hrestr : restr Fᶜ Ψ = restr G Ψ := by
    refine PiLp.ext fun z => ?_
    by_cases hzF : z ∈ Fᶜ
    · by_cases hzΨ : Ψ z = 0
      · simp [hzΨ]
      · have hzG : z ∈ G := Finset.mem_filter.2 ⟨hzF, hzΨ⟩
        rw [restr_apply, restr_apply, if_pos hzF, if_pos hzG]
    · have hzG : z ∉ G := fun hz => hzF (Finset.mem_filter.1 hz).1
      rw [restr_apply, restr_apply, if_neg hzF, if_neg hzG]
  have hsplit : V Ψ = V (restr F Ψ) + V (restr G Ψ) := by
    conv_lhs => rw [← restr_add_restr_compl F Ψ]
    rw [hrestr, map_add]
  refine PiLp.ext fun z => ?_
  by_cases hz : z ∈ imageBlock hbd.lbl F
  · have hz' : z ∉ imageBlock hbd.lbl G := by
      intro hz'
      obtain ⟨y, hyF, hy⟩ := Finset.mem_biUnion.1 hz
      obtain ⟨y', hy'G, hy'⟩ := Finset.mem_biUnion.1 hz'
      obtain ⟨hy'F, hy'Ψ⟩ := Finset.mem_filter.1 hy'G
      have heq : y = y' := hinj (hFS y hyF) (hsupp y' hy'Ψ)
        (label_eq_of_mem_labelBlock hy hy')
      exact Finset.mem_compl.1 hy'F (heq ▸ hyF)
    rw [restr_apply, if_pos hz, hsplit, PiLp.add_apply,
      apply_restr_support hbd G Ψ z hz', add_zero]
  · rw [restr_apply, if_neg hz, apply_restr_support hbd F Ψ z hz]

/-- **The bridge.**  The amplitude mass found on the physical registers reached
by a set of branches equals the mass those branches carried beforehand: a
branch-diagonal isometry moves no probability between branches. -/
theorem norm_restr_imageBlock {V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec}
    (hbd : IsBranchDiagonal V) {S : Set Pat} (hinj : Set.InjOn hbd.lbl S)
    (hV : ∀ x, ‖V x‖ = ‖x‖)
    (F : Finset Pat) (hFS : ∀ y ∈ F, y ∈ S)
    (Ψ : EuclideanSpace ℂ Pat) (hsupp : ∀ z : Pat, Ψ z ≠ 0 → z ∈ S) :
    ‖restr (imageBlock hbd.lbl F) (V Ψ)‖ = Real.sqrt (mass Ψ F) := by
  rw [restr_imageBlock_apply hbd hinj F hFS Ψ hsupp, hV, norm_restr_eq_sqrt_mass]

end Dqi
