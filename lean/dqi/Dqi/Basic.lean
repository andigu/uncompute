import Mathlib.InformationTheory.Hamming
import Mathlib.Data.Matrix.Mul
import Mathlib.Order.ConditionallyCompleteLattice.Finset

/-!
# Patterns, syndromes, and the decodable set

This file fixes the combinatorial setting shared by the whole development: error
patterns over a finite field, the syndrome map determined by a constraint matrix,
the dual code and its minimum distance, and the decodable set.

The substantive result is `syndrome_injOn_decodable`: when the decoding radius
satisfies `2 * l < dualDist B`, distinct decodable patterns have distinct
syndromes.  That is what lets the rest of the development label a branch by its
error pattern even though the physical register stores only the syndrome, and it
is what makes the DQI state normalized.

Notation follows the DQI decoder analysis: `B` is the constraint matrix, a
pattern is `y : Fin m → K`, and the syndrome is `Bᵀ *ᵥ y`.
-/

namespace Dqi

open Matrix

variable {K : Type*} [Field K]
variable {m n : ℕ}

/-- An error pattern: one field element per constraint. -/
abbrev Pattern (m : ℕ) (K : Type*) : Type _ := Fin m → K

/-- The syndrome stored by the circuit for the pattern `y`. -/
def syndrome (B : Matrix (Fin m) (Fin n) K) (y : Pattern m K) : Fin n → K :=
  B.transpose *ᵥ y

/-- Membership in the dual code `C⊥ = {y : Bᵀ y = 0}`. -/
def IsDualCodeword (B : Matrix (Fin m) (Fin n) K) (y : Pattern m K) : Prop :=
  syndrome B y = 0

variable {B : Matrix (Fin m) (Fin n) K}

/-- The syndrome map is additive, so a syndrome collision produces a dual codeword. -/
theorem syndrome_sub (y z : Pattern m K) :
    syndrome B (y - z) = syndrome B y - syndrome B z := by
  simp [syndrome, Matrix.mulVec_sub]

/-- Two patterns with a common syndrome differ by a dual codeword. -/
theorem isDualCodeword_sub_of_syndrome_eq {y z : Pattern m K}
    (h : syndrome B y = syndrome B z) : IsDualCodeword B (y - z) := by
  simp [IsDualCodeword, syndrome_sub, h]

variable [DecidableEq K]

/-- The minimum distance of the dual code: the least Hamming weight of a nonzero
pattern that the syndrome cannot distinguish from zero.

If the dual code is trivial the defining set is empty and this is `0` by
`Nat.sInf_empty`.  Every hypothesis below has the form `_ < dualDist B`, so the
degenerate case makes the hypothesis false rather than vacuously true: the
convention fails closed. -/
noncomputable def dualDist (B : Matrix (Fin m) (Fin n) K) : ℕ :=
  sInf {w | ∃ y : Pattern m K, y ≠ 0 ∧ IsDualCodeword B y ∧ hammingNorm y = w}

/-- A nonzero dual codeword has weight at least the dual distance. -/
theorem dualDist_le_hammingNorm {y : Pattern m K}
    (hy : y ≠ 0) (hdual : IsDualCodeword B y) :
    dualDist B ≤ hammingNorm y :=
  Nat.sInf_le ⟨y, hy, hdual, rfl⟩

/-- The decodable set: patterns of weight at most the decoding radius `l`. -/
def Decodable (l : ℕ) (y : Pattern m K) : Prop := hammingNorm y ≤ l

/-- Hamming weight is subadditive on differences. -/
theorem hammingNorm_sub_le (y z : Pattern m K) :
    hammingNorm (y - z) ≤ hammingNorm y + hammingNorm z := by
  have h := hammingDist_triangle_right z y (0 : Pattern m K)
  rw [hammingDist_eq_hammingNorm, hammingDist_zero_right, hammingDist_zero_right] at h
  rwa [neg_add_eq_sub, Nat.add_comm] at h

/-- **Distinct decodable patterns have distinct syndromes.**

If two patterns of weight at most `l` shared a syndrome, their difference would
be a nonzero dual codeword of weight at most `2 * l`, contradicting
`2 * l < dualDist B`. -/
theorem syndrome_injOn_decodable {l : ℕ} (hl : 2 * l < dualDist B) :
    Set.InjOn (syndrome B) {y : Pattern m K | Decodable l y} := by
  intro y hy z hz h
  by_contra hne
  have hsub : y - z ≠ 0 := sub_ne_zero_of_ne hne
  have hlow : dualDist B ≤ hammingNorm (y - z) :=
    dualDist_le_hammingNorm hsub (isDualCodeword_sub_of_syndrome_eq h)
  have htri : hammingNorm (y - z) ≤ hammingNorm y + hammingNorm z := hammingNorm_sub_le y z
  have hyl : hammingNorm y ≤ l := hy
  have hzl : hammingNorm z ≤ l := hz
  omega

end Dqi
