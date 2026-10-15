import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# The failure model

This file fixes what it means for the implemented decoder to differ from the
ideal one.  It is the interface between a physical circuit and the perturbation
bound, and it is where all the modelling risk of the development sits: the
theorems downstream are only as meaningful as these definitions are faithful.

The registers are kept as three abstract finite types rather than a tensor
product: `Syn` (the syndrome register), `Anc` (the ancilla register), and `Rec`
(the retained measurement record, the *junk register* of the analysis).  Working
in `EuclideanSpace ℂ (Syn × Anc × Rec)` avoids `PiTensorProduct`, which is
painful to compute with, at no loss of generality.

A *label* is a syndrome-and-ancilla pair.  The record is deliberately not part of
a label: record states are `X`-basis states, hence superpositions in the
computational basis, and the arguments downstream group the basis by label while
treating the record factor opaquely.

`IsBranchDiagonal` is the structural hypothesis satisfied by every uncompute
circuit: a reversible gate permutes the basis string independently of the
measurement record, and every outcome-conditioned operation is diagonal, so each
input branch produces one definite label, one record state, and one phase.  It is
an assumption about the circuit class, not about the mathematics, and is
discharged outside this development by the uncompute semantics.
-/

namespace Dqi

/-- A syndrome-and-ancilla label.  Two states with distinct labels have disjoint
computational-basis support whatever their record factors — the fact the
reachable-residual argument turns on. -/
abbrev Label (Syn Anc : Type*) : Type _ := Syn × Anc

/-- The physical state space: syndrome, ancilla, and retained record. -/
abbrev Phys (Syn Anc Rec : Type*) : Type _ := EuclideanSpace ℂ (Syn × Anc × Rec)

section Mass

variable {Pat : Type*}

/-- The DQI probability mass carried by a set of branches, measured in the
amplitude distribution of the state `Ψ`.

This is the quantity the whole bound depends on, and it is exactly what the fuzz
test estimates: sample a branch from `‖Ψ y‖ ^ 2`, ask whether it lies in the
failure set.  That is why the DQI amplitude distribution, and no other, is the
correct input distribution for validation. -/
noncomputable def mass (Ψ : EuclideanSpace ℂ Pat) (F : Finset Pat) : ℝ :=
  ∑ y ∈ F, ‖Ψ y‖ ^ 2

theorem mass_nonneg (Ψ : EuclideanSpace ℂ Pat) (F : Finset Pat) : 0 ≤ mass Ψ F :=
  Finset.sum_nonneg fun _ _ => by positivity

theorem mass_mono (Ψ : EuclideanSpace ℂ Pat) {F G : Finset Pat} (h : F ⊆ G) :
    mass Ψ F ≤ mass Ψ G :=
  Finset.sum_le_sum_of_subset_of_nonneg h fun _ _ _ => by positivity

end Mass

section Labels

variable {Syn Anc Rec : Type*}
variable [Fintype Syn] [DecidableEq Syn]
variable [Fintype Anc] [DecidableEq Anc]
variable [Fintype Rec]

/-- The physical state with label `lb`, record state `w`, and phase `c`. -/
noncomputable def atLabel (lb : Label Syn Anc) (w : EuclideanSpace ℂ Rec) (c : ℂ) :
    Phys Syn Anc Rec :=
  WithLp.toLp 2 fun z => if z.1 = lb.1 ∧ z.2.1 = lb.2 then c * w z.2.2 else 0

omit [Fintype Syn] [Fintype Anc] [Fintype Rec] in
@[simp]
theorem atLabel_apply (lb : Label Syn Anc) (w : EuclideanSpace ℂ Rec) (c : ℂ)
    (z : Syn × Anc × Rec) :
    atLabel lb w c z = if z.1 = lb.1 ∧ z.2.1 = lb.2 then c * w z.2.2 else 0 := rfl

/-- States carrying distinct labels are orthogonal, whatever their record
factors.  This is the workhorse behind "a sign or leakage error keeps a decodable
syndrome, so it cannot overlap the next-layer residual". -/
theorem atLabel_inner_eq_zero {lb lb' : Label Syn Anc} (h : lb ≠ lb')
    (w w' : EuclideanSpace ℂ Rec) (c c' : ℂ) :
    inner ℂ (atLabel lb w c) (atLabel lb' w' c') = (0 : ℂ) := by
  classical
  rw [PiLp.inner_apply]
  refine Finset.sum_eq_zero ?_
  rintro ⟨s, a, r⟩ -
  by_cases hs : s = lb.1 ∧ a = lb.2
  · have hne : ¬ (s = lb'.1 ∧ a = lb'.2) := by
      rintro ⟨h1, h2⟩
      exact h (Prod.ext (hs.1.symm.trans h1) (hs.2.symm.trans h2))
    simp [hne]
  · simp [hs]

end Labels

section BranchDiagonal

variable {Pat Syn Anc Rec : Type*}
variable [Fintype Pat] [DecidableEq Pat]
variable [Fintype Syn] [DecidableEq Syn]
variable [Fintype Anc] [DecidableEq Anc]
variable [Fintype Rec]

/-- **Branch-diagonality.**  The data witnessing that `V` sends each pattern
basis vector to a phase times a single-label state. -/
structure IsBranchDiagonal (V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec) where
  /-- The label reached by each branch. -/
  lbl : Pat → Label Syn Anc
  /-- The record state left by each branch (a unit vector, per `recSt_norm`). -/
  recSt : Pat → EuclideanSpace ℂ Rec
  /-- The phase applied to each branch. -/
  phase : Pat → ℂ
  /-- Record states are unit vectors. -/
  recSt_norm : ∀ y, ‖recSt y‖ = 1
  /-- Phases have unit modulus. -/
  phase_norm : ∀ y, ‖phase y‖ = 1
  /-- The defining action on basis vectors. -/
  apply_single : ∀ y, V (EuclideanSpace.single y (1 : ℂ)) =
      atLabel (lbl y) (recSt y) (phase y)

omit [Fintype Pat] in
/-- Branches sent to distinct labels have orthogonal images. -/
theorem IsBranchDiagonal.inner_eq_zero
    {V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec} (hV : IsBranchDiagonal V)
    {y z : Pat} (h : hV.lbl y ≠ hV.lbl z) :
    inner ℂ (V (EuclideanSpace.single y (1 : ℂ))) (V (EuclideanSpace.single z (1 : ℂ)))
      = (0 : ℂ) := by
  rw [hV.apply_single, hV.apply_single]
  exact atLabel_inner_eq_zero h _ _ _ _

/-- The failure set of an implementation `V` relative to the ideal `U`: the
branches on which the two disagree.  No structure is assumed of it — it may be
arbitrary and correlated with the instance. -/
def failureSet (U V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec) : Set Pat :=
  {y | V (EuclideanSpace.single y (1 : ℂ)) ≠ U (EuclideanSpace.single y (1 : ℂ))}

omit [Fintype Pat] [DecidableEq Syn] [DecidableEq Anc] in
theorem notMem_failureSet_iff {U V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec}
    {y : Pat} :
    y ∉ failureSet U V ↔
      V (EuclideanSpace.single y (1 : ℂ)) = U (EuclideanSpace.single y (1 : ℂ)) := by
  simp [failureSet]

end BranchDiagonal

end Dqi
