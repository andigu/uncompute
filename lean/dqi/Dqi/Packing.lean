import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Nat.Choose.Basic

/-!
# A Fisher-type packing bound

If `F` is a family of `s`-element subsets of an `n`-element ground set `G` such that any
two distinct members of `F` intersect in at most `t` elements, then
`F.card * s.choose (t+1) ≤ n.choose (t+1)`, and consequently
`F.card ≤ n.choose (t+1) / s.choose (t+1)`.

The proof is a double count of `(t+1)`-element subsets: each member of `F` contains
exactly `s.choose (t+1)` of them, no two distinct members share one, and the ground set
contains `n.choose (t+1)` in total.

This lemma is used to bound how many error patterns can share a syndrome in a DQI
decoder analysis.
-/

namespace Dqi

open Finset

/-- **Packing bound, multiplicative form.** If every member of `F` is an `s`-subset of a
ground set `G` of size `n`, and distinct members intersect in at most `t` elements, then
`F.card * s.choose (t+1) ≤ n.choose (t+1)`. -/
theorem card_mul_choose_le_of_pairwise_inter_le
    {α : Type*} [DecidableEq α] {n s t : ℕ}
    {G : Finset α} (hG : G.card = n)
    {F : Finset (Finset α)}
    (hsub : ∀ A ∈ F, A ⊆ G)
    (hcard : ∀ A ∈ F, A.card = s)
    (hint : ∀ A ∈ F, ∀ B ∈ F, A ≠ B → (A ∩ B).card ≤ t) :
    F.card * s.choose (t + 1) ≤ n.choose (t + 1) := by
  classical
  -- Pairwise disjointness of the families of `(t+1)`-subsets.
  have hdisj : (F : Set (Finset α)).PairwiseDisjoint
      (fun A => A.powersetCard (t + 1)) := by
    intro A hA B hB hAB
    rw [Finset.mem_coe] at hA hB
    simp only [Function.onFun]
    rw [Finset.disjoint_left]
    intro S hSA hSB
    rw [Finset.mem_powersetCard] at hSA hSB
    have hSAB : S ⊆ A ∩ B := Finset.subset_inter hSA.1 hSB.1
    have : t + 1 ≤ t := hSA.2 ▸ (Finset.card_le_card hSAB).trans (hint A hA B hB hAB)
    omega
  -- The union of these families sits inside the `(t+1)`-subsets of `G`.
  have hunion : F.biUnion (fun A => A.powersetCard (t + 1)) ⊆ G.powersetCard (t + 1) := by
    intro S hS
    rw [Finset.mem_biUnion] at hS
    obtain ⟨A, hA, hSA⟩ := hS
    rw [Finset.mem_powersetCard] at hSA ⊢
    exact ⟨hSA.1.trans (hsub A hA), hSA.2⟩
  calc F.card * s.choose (t + 1)
      = ∑ A ∈ F, (A.powersetCard (t + 1)).card :=
        (Finset.sum_const_nat fun A hA => by
          rw [Finset.card_powersetCard, hcard A hA]).symm
    _ = (F.biUnion (fun A => A.powersetCard (t + 1))).card :=
        (Finset.card_biUnion hdisj).symm
    _ ≤ (G.powersetCard (t + 1)).card := Finset.card_le_card hunion
    _ = n.choose (t + 1) := by rw [Finset.card_powersetCard, hG]

/-- **Packing bound, division form.** Under the additional hypothesis `t + 1 ≤ s`
(so that `s.choose (t+1)` is positive), the family has at most
`n.choose (t+1) / s.choose (t+1)` members (`Nat` division). -/
theorem card_le_choose_div_choose_of_pairwise_inter_le
    {α : Type*} [DecidableEq α] {n s t : ℕ}
    {G : Finset α} (hG : G.card = n)
    {F : Finset (Finset α)}
    (hsub : ∀ A ∈ F, A ⊆ G)
    (hcard : ∀ A ∈ F, A.card = s)
    (hint : ∀ A ∈ F, ∀ B ∈ F, A ≠ B → (A ∩ B).card ≤ t)
    (hts : t + 1 ≤ s) :
    F.card ≤ n.choose (t + 1) / s.choose (t + 1) := by
  rw [Nat.le_div_iff_mul_le (Nat.choose_pos hts)]
  exact card_mul_choose_le_of_pairwise_inter_le hG hsub hcard hint

end Dqi
