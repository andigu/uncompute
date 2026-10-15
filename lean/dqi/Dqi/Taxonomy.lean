import Dqi.Failure

/-!
# The failure-channel taxonomy

The paper classifies the branches on which the implemented decoder differs from
the ideal one into three exclusive channels, by a priority rule:

* **relocation** — the implementation lands on a wrong syndrome.  This takes
  priority: a branch with a wrong syndrome is a relocation even when its
  ancilla or record are also dirty.
* **leakage** — the syndrome is right, but the ancilla is dirty or the record
  state is not the clean one.
* **sign** — every register is ideal; at worst the phase differs.

This file defines the three channels from the branch-diagonal data of the ideal
oracle `U` and the implemented oracle `V`, proves that they are an exclusive
and exhaustive partition of the branches (`channel_trichotomy`) and hence of
the failure mass (`mass_partition`), and proves the support fact the blindness
argument needs: on a non-relocating branch the error vector `V|y⟩ - U|y⟩` is
supported at the ideal syndrome (`sub_single_syn_of_not_reloc`).  That is what
"sign and leakage errors keep the ideal syndrome" means formally, and it is
what makes those two channels blind to the escaping weight layer.
-/

namespace Dqi

section Taxonomy

variable {Pat Syn Anc Rec : Type*}
variable [Fintype Pat] [DecidableEq Pat]
variable [Fintype Syn] [DecidableEq Syn]
variable [Fintype Anc] [DecidableEq Anc]
variable [Fintype Rec] [DecidableEq Rec]
variable {U V : EuclideanSpace ℂ Pat →ₗ[ℂ] Phys Syn Anc Rec}

/-- Relocation: the implementation lands on a wrong syndrome (priority: this
holds even when ancilla or record are also wrong). -/
def IsRelocation (hU : IsBranchDiagonal U) (hV : IsBranchDiagonal V) (y : Pat) : Prop :=
  (hV.lbl y).1 ≠ (hU.lbl y).1

/-- Clean relocation: wrong syndrome, but ancilla and record ideal. -/
def IsCleanRelocation (hU : IsBranchDiagonal U) (hV : IsBranchDiagonal V) (y : Pat) : Prop :=
  IsRelocation hU hV y ∧ (hV.lbl y).2 = (hU.lbl y).2 ∧ hV.recSt y = hU.recSt y

/-- Leakage: right syndrome, but dirty ancilla or nonclean record. -/
def IsLeakage (hU : IsBranchDiagonal U) (hV : IsBranchDiagonal V) (y : Pat) : Prop :=
  (hV.lbl y).1 = (hU.lbl y).1 ∧
    ((hV.lbl y).2 ≠ (hU.lbl y).2 ∨ hV.recSt y ≠ hU.recSt y)

/-- Sign-only: all registers ideal; at worst the phase differs. -/
def IsSignOnly (hU : IsBranchDiagonal U) (hV : IsBranchDiagonal V) (y : Pat) : Prop :=
  hV.lbl y = hU.lbl y ∧ hV.recSt y = hU.recSt y

variable (hU : IsBranchDiagonal U) (hV : IsBranchDiagonal V)

omit [Fintype Pat] [Fintype Syn] [Fintype Anc] [DecidableEq Rec] in
/-- **Exclusive-exhaustive partition.**  Every branch falls in exactly one of
the three channels: the priority rule relocation ≻ leakage ≻ sign is a genuine
trichotomy. -/
theorem channel_trichotomy (y : Pat) :
    (IsRelocation hU hV y ∧ ¬IsLeakage hU hV y ∧ ¬IsSignOnly hU hV y) ∨
    (IsLeakage hU hV y ∧ ¬IsRelocation hU hV y ∧ ¬IsSignOnly hU hV y) ∨
    (IsSignOnly hU hV y ∧ ¬IsRelocation hU hV y ∧ ¬IsLeakage hU hV y) := by
  simp only [IsRelocation, IsLeakage, IsSignOnly, Prod.ext_iff, ne_eq]
  tauto

omit [Fintype Pat] [Fintype Syn] [Fintype Anc] [DecidableEq Rec] in
/-- On a sign-only branch, the error vector is a phase multiple of the ideal
single-label state. -/
theorem sub_single_of_signOnly (y : Pat) (h : IsSignOnly hU hV y) :
    V (EuclideanSpace.single y (1 : ℂ)) - U (EuclideanSpace.single y (1 : ℂ))
      = atLabel (hU.lbl y) (hU.recSt y) (hV.phase y - hU.phase y) := by
  rw [hV.apply_single, hU.apply_single, h.1, h.2]
  refine PiLp.ext fun z => ?_
  simp only [PiLp.sub_apply, atLabel_apply]
  by_cases hz : z.1 = (hU.lbl y).1 ∧ z.2.1 = (hU.lbl y).2
  · simp [hz, sub_mul]
  · simp [hz]

omit [Fintype Pat] [Fintype Syn] [Fintype Anc] [DecidableEq Rec] in
/-- **The blindness support fact.**  On a non-relocating branch — a leakage or
sign branch — the error vector is supported at the ideal syndrome. -/
theorem sub_single_syn_of_not_reloc (y : Pat) (h : ¬IsRelocation hU hV y)
    (z : Syn × Anc × Rec)
    (hz : (V (EuclideanSpace.single y (1 : ℂ)) - U (EuclideanSpace.single y (1 : ℂ))) z ≠ 0) :
    z.1 = (hU.lbl y).1 := by
  have hsyn : (hV.lbl y).1 = (hU.lbl y).1 := not_not.mp h
  by_contra hne
  refine hz ?_
  have h1 : ¬(z.1 = (hV.lbl y).1 ∧ z.2.1 = (hV.lbl y).2) := fun hc => hne (hc.1.trans hsyn)
  have h2 : ¬(z.1 = (hU.lbl y).1 ∧ z.2.1 = (hU.lbl y).2) := fun hc => hne hc.1
  simp [PiLp.sub_apply, hV.apply_single, hU.apply_single, h1, h2]

omit [Fintype Pat] [Fintype Syn] [Fintype Anc] [DecidableEq Rec] in
open scoped Classical in
/-- **Channel masses partition the failure mass.**  For any state and any set
of branches, the mass splits exactly into the relocation, leakage, and
sign-only channel masses. -/
theorem mass_partition (Ψ : EuclideanSpace ℂ Pat) (F : Finset Pat) :
    mass Ψ F
      = mass Ψ (F.filter fun y => IsRelocation hU hV y)
      + mass Ψ (F.filter fun y => IsLeakage hU hV y)
      + mass Ψ (F.filter fun y => IsSignOnly hU hV y) := by
  simp only [mass, Finset.sum_filter, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  rcases channel_trichotomy hU hV y with ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ | ⟨h1, h2, h3⟩ <;>
    simp [h1, h2, h3]

end Taxonomy

end Dqi
