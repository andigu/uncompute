import Dqi.Taxonomy
import Dqi.PhysFull
import Dqi.Embed
import Dqi.Bridge

/-!
# The sharp relocation cross-term

The error's overlap with the escaping component must be bounded by
`√(clean-relocation mass) × reach`, not by the full error norm.  The reason is
exact orthogonality in the physical register space: the escaping component
`toPhys B a₀ wrec ξ` lives on edge syndromes, the clean ancilla value, and the
clean record state, while

* every **ideal** image sits at a decodable syndrome
  (`inner_U_single_toPhys_eq_zero`), and
* every **non-clean** implemented image either keeps a decodable syndrome, or
  carries a dirty ancilla, or carries a record state orthogonal to the clean one
  (`inner_V_single_toPhys_eq_zero`),

so all of them pair to exactly zero against the escaping component.  Expanding
the error over branches, only the implemented images of the *clean relocations*
survive, and Cauchy–Schwarz through the reachable part gives the assembled
bound `inner_diff_toPhys_le`.  This is why the final bound carries `√μ_cl` with
no triangle-inequality factor of two.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m n : ℕ}
variable {Anc Rec : Type*} [Fintype Anc] [DecidableEq Anc] [Fintype Rec] [DecidableEq Rec]
variable {U V : EuclideanSpace ℂ (Pattern m K) →ₗ[ℂ] Phys (Fin n → K) Anc Rec}
variable (hU : IsBranchDiagonal U) (hV : IsBranchDiagonal V)
variable {B : Matrix (Fin m) (Fin n) K} {a₀ : Anc} {wrec : EuclideanSpace ℂ Rec}

omit [Fintype Anc] [Fintype Rec] [DecidableEq Rec] in
/-- The escaping component vanishes at every register index carrying a decodable
syndrome: its syndrome pushforward is supported on the edge layer, and the
strict-distance hypothesis keeps the edge disjoint from the decodable set. -/
theorem toPhys_eq_zero_of_mem_decodableSyn (l : ℕ) (hstrict : 2 * l + 1 < dualDist B)
    (ξ : EuclideanSpace ℂ (Pattern m K)) (hξ : ∀ y, ξ y ≠ 0 → hammingNorm y = l + 1)
    (z : (Fin n → K) × Anc × Rec) (hz : z.1 ∈ decodableSyn B l) :
    toPhys B a₀ wrec ξ z = 0 := by
  refine toPhys_eq_zero_of_syn B a₀ wrec ξ z ?_
  by_contra hne
  rw [toSyn_apply] at hne
  obtain ⟨y', -, hy'⟩ := Finset.exists_ne_zero_of_sum_ne_zero hne
  by_cases hsyn : syndrome B y' = z.1
  · rw [if_pos hsyn] at hy'
    exact Finset.disjoint_left.mp (disjoint_decodableSyn_edgeSyn hstrict) hz
      (mem_edgeSyn.mpr ⟨y', hξ y' hy', hsyn⟩)
  · rw [if_neg hsyn] at hy'
    exact hy' rfl

omit [DecidableEq Rec] in
/-- A single-label state over a decodable syndrome is exactly orthogonal to the
escaping component, whatever its ancilla, record factor, and phase. -/
theorem inner_atLabel_toPhys_eq_zero_of_mem_decodableSyn (l : ℕ)
    (hstrict : 2 * l + 1 < dualDist B)
    (ξ : EuclideanSpace ℂ (Pattern m K)) (hξ : ∀ y, ξ y ≠ 0 → hammingNorm y = l + 1)
    (lb : Label (Fin n → K) Anc) (w : EuclideanSpace ℂ Rec) (c : ℂ)
    (hlb : lb.1 ∈ decodableSyn B l) :
    inner ℂ (atLabel lb w c) (toPhys B a₀ wrec ξ) = 0 := by
  rw [PiLp.inner_apply]
  refine Finset.sum_eq_zero fun z _ => ?_
  by_cases hz : z.1 = lb.1 ∧ z.2.1 = lb.2
  · rw [toPhys_eq_zero_of_mem_decodableSyn l hstrict ξ hξ z (by rw [hz.1]; exact hlb),
      inner_zero_right]
  · rw [atLabel_apply, if_neg hz, inner_zero_left]

omit [DecidableEq Rec] in
/-- A single-label state whose record factor is orthogonal to the clean record
state is exactly orthogonal to the escaping component: for each fixed syndrome
and ancilla, the record sum factors out of the register inner product as
`⟪w, wrec⟫ = 0`. -/
theorem inner_atLabel_toPhys_eq_zero_of_rec_orth
    (ξ : EuclideanSpace ℂ (Pattern m K)) (lb : Label (Fin n → K) Anc)
    (w : EuclideanSpace ℂ Rec) (c : ℂ) (hw : inner ℂ w wrec = 0) :
    inner ℂ (atLabel lb w c) (toPhys B a₀ wrec ξ) = 0 := by
  rw [PiLp.inner_apply, Fintype.sum_prod_type (fun z : (Fin n → K) × Anc × Rec =>
    inner ℂ (atLabel lb w c z) (toPhys B a₀ wrec ξ z))]
  refine Finset.sum_eq_zero fun s _ => ?_
  rw [Fintype.sum_prod_type (fun p : Anc × Rec =>
    inner ℂ (atLabel lb w c (s, p)) (toPhys B a₀ wrec ξ (s, p)))]
  refine Finset.sum_eq_zero fun a _ => ?_
  by_cases hsa : s = lb.1 ∧ a = lb.2
  · have hw' : (∑ rc : Rec, wrec rc * (starRingEnd ℂ) (w rc)) = inner ℂ w wrec := by
      rw [PiLp.inner_apply]
      exact Finset.sum_congr rfl fun rc _ => (RCLike.inner_apply _ _).symm
    have hterm : ∀ rc : Rec,
        inner ℂ (atLabel lb w c (s, a, rc)) (toPhys B a₀ wrec ξ (s, a, rc))
          = ((if a = a₀ then 1 else 0) * toSyn B ξ s * (starRingEnd ℂ) c)
              * (wrec rc * (starRingEnd ℂ) (w rc)) := by
      intro rc
      have h1 : atLabel lb w c (s, a, rc) = c * w rc := by
        rw [atLabel_apply]
        exact if_pos hsa
      have h2 : toPhys B a₀ wrec ξ (s, a, rc)
          = (if a = a₀ then 1 else 0) * toSyn B ξ s * wrec rc := rfl
      rw [h1, h2, RCLike.inner_apply, map_mul]
      ring
    calc (∑ rc : Rec, inner ℂ (atLabel lb w c (s, a, rc)) (toPhys B a₀ wrec ξ (s, a, rc)))
        = ∑ rc : Rec, ((if a = a₀ then 1 else 0) * toSyn B ξ s * (starRingEnd ℂ) c)
            * (wrec rc * (starRingEnd ℂ) (w rc)) :=
          Finset.sum_congr rfl fun rc _ => hterm rc
      _ = ((if a = a₀ then 1 else 0) * toSyn B ξ s * (starRingEnd ℂ) c)
            * ∑ rc : Rec, wrec rc * (starRingEnd ℂ) (w rc) := by
          rw [← Finset.mul_sum]
      _ = 0 := by rw [hw', hw, mul_zero]
  · refine Finset.sum_eq_zero fun rc _ => ?_
    have h0 : atLabel lb w c (s, a, rc) = 0 := by
      rw [atLabel_apply]
      exact if_neg hsa
    rw [h0, inner_zero_left]

omit [DecidableEq Rec] in
/-- **Ideal images are blind to the edge.**  An ideal branch image sits at the
decodable syndrome `syndrome B y`, so it is exactly orthogonal to the escaping
component, which lives on edge syndromes only. -/
theorem inner_U_single_toPhys_eq_zero (l : ℕ) (hstrict : 2 * l + 1 < dualDist B)
    (hUlbl : ∀ y, hammingNorm y ≤ l → hU.lbl y = (syndrome B y, a₀))
    (ξ : EuclideanSpace ℂ (Pattern m K)) (hξ : ∀ y, ξ y ≠ 0 → hammingNorm y = l + 1)
    (y : Pattern m K) (hy : hammingNorm y ≤ l) :
    inner ℂ (U (EuclideanSpace.single y (1 : ℂ))) (toPhys B a₀ wrec ξ) = 0 := by
  rw [hU.apply_single]
  refine inner_atLabel_toPhys_eq_zero_of_mem_decodableSyn l hstrict ξ hξ _ _ _ ?_
  rw [hUlbl y hy]
  exact mem_decodableSyn.mpr ⟨y, hy, rfl⟩

omit [DecidableEq Rec] in
/-- **Non-clean implemented images are blind to the edge.**  On a branch that
is not a clean relocation, the implemented image either keeps a decodable
syndrome, or carries a dirty ancilla, or carries a record state orthogonal to
the clean one — and in every case it is exactly orthogonal to the escaping
component. -/
theorem inner_V_single_toPhys_eq_zero (l : ℕ) (hstrict : 2 * l + 1 < dualDist B)
    (hUlbl : ∀ y, hammingNorm y ≤ l → hU.lbl y = (syndrome B y, a₀))
    (hUrec : ∀ y, hammingNorm y ≤ l → hU.recSt y = wrec)
    (horth : ∀ y, hammingNorm y ≤ l →
      (hV.recSt y = wrec ∨ inner ℂ (hV.recSt y) wrec = 0))
    (ξ : EuclideanSpace ℂ (Pattern m K)) (hξ : ∀ y, ξ y ≠ 0 → hammingNorm y = l + 1)
    (y : Pattern m K) (hy : hammingNorm y ≤ l)
    (hnc : ¬IsCleanRelocation hU hV y) :
    inner ℂ (V (EuclideanSpace.single y (1 : ℂ))) (toPhys B a₀ wrec ξ) = 0 := by
  by_cases hrel : IsRelocation hU hV y
  · -- relocation, but not clean: the ancilla or the record must be dirty
    have hbad : ¬((hV.lbl y).2 = a₀ ∧ hV.recSt y = wrec) := by
      intro hcon
      refine hnc ⟨hrel, ?_, ?_⟩
      · rw [hUlbl y hy]
        exact hcon.1
      · rw [hUrec y hy]
        exact hcon.2
    rcases not_and_or.mp hbad with hanc | hrec
    · -- dirty ancilla: the supports are disjoint on the ancilla register
      rw [hV.apply_single, PiLp.inner_apply]
      refine Finset.sum_eq_zero fun z _ => ?_
      by_cases hz : z.1 = (hV.lbl y).1 ∧ z.2.1 = (hV.lbl y).2
      · rw [toPhys_eq_zero_of_anc B a₀ wrec ξ z (by rw [hz.2]; exact hanc),
          inner_zero_right]
      · rw [atLabel_apply, if_neg hz, inner_zero_left]
    · -- nonclean record: the record states are orthogonal
      rw [hV.apply_single]
      exact inner_atLabel_toPhys_eq_zero_of_rec_orth ξ (hV.lbl y) (hV.recSt y)
        (hV.phase y) ((horth y hy).resolve_left hrec)
  · -- not a relocation: the implemented image keeps the decodable syndrome
    have heq : (hV.lbl y).1 = syndrome B y :=
      (not_not.mp hrel).trans (by rw [hUlbl y hy])
    rw [hV.apply_single]
    refine inner_atLabel_toPhys_eq_zero_of_mem_decodableSyn l hstrict ξ hξ _ _ _ ?_
    rw [heq]
    exact mem_decodableSyn.mpr ⟨y, hy, rfl⟩

open scoped Classical in
/-- **The sharp relocation cross bound.**  Expanding the error `V − U` over the
branches of `F`, every ideal term and every non-clean implemented term is
exactly orthogonal to the escaping component, so the overlap reduces to the
implemented image of the clean relocations alone; Cauchy–Schwarz through its
reachable part then yields `√(clean-relocation mass) × reach` — the reason the
final bound carries `√μ_cl` with no factor of two. -/
theorem inner_diff_toPhys_le (l : ℕ) (hstrict : 2 * l + 1 < dualDist B)
    (hUlbl : ∀ y, hammingNorm y ≤ l → hU.lbl y = (syndrome B y, a₀))
    (hUrec : ∀ y, hammingNorm y ≤ l → hU.recSt y = wrec)
    (horth : ∀ y, hammingNorm y ≤ l →
      (hV.recSt y = wrec ∨ inner ℂ (hV.recSt y) wrec = 0))
    (hViso : ∀ x : EuclideanSpace ℂ (Pattern m K),
      (∀ y, x y ≠ 0 → hammingNorm y ≤ l) → ‖V x‖ = ‖x‖)
    (Ψpre : EuclideanSpace ℂ (Pattern m K))
    (hsupp : ∀ y, Ψpre y ≠ 0 → hammingNorm y ≤ l)
    (F : Finset (Pattern m K))
    (ξ : EuclideanSpace ℂ (Pattern m K)) (hξ : ∀ y, ξ y ≠ 0 → hammingNorm y = l + 1)
    (erel : ℝ)
    (hreach : ‖restr (supp (V (restr (F.filter fun y => IsCleanRelocation hU hV y) Ψpre)))
        (toPhys B a₀ wrec ξ)‖ ≤ erel) :
    ‖inner ℂ (V (restr F Ψpre) - U (restr F Ψpre)) (toPhys B a₀ wrec ξ)‖
      ≤ Real.sqrt (mass Ψpre (F.filter fun y => IsCleanRelocation hU hV y)) * erel := by
  set Fcl : Finset (Pattern m K) := F.filter (fun y => IsCleanRelocation hU hV y) with hFcl
  set X₂ : Phys (Fin n → K) Anc Rec := toPhys B a₀ wrec ξ with hX₂
  have hUzero : inner ℂ (U (restr F Ψpre)) X₂ = 0 := by
    rw [restr_eq_sum F Ψpre, map_sum, sum_inner]
    refine Finset.sum_eq_zero fun y _ => ?_
    rw [map_smul, inner_smul_left]
    by_cases hΨ : Ψpre y = 0
    · rw [hΨ, map_zero, zero_mul]
    · rw [hX₂, inner_U_single_toPhys_eq_zero hU l hstrict hUlbl ξ hξ y (hsupp y hΨ),
        mul_zero]
  have hsplit : inner ℂ (V (restr F Ψpre)) X₂ = inner ℂ (V (restr Fcl Ψpre)) X₂ := by
    rw [restr_eq_sum F Ψpre, restr_eq_sum Fcl Ψpre, map_sum, map_sum, sum_inner,
      sum_inner, hFcl, Finset.sum_filter]
    refine Finset.sum_congr rfl fun y _ => ?_
    by_cases hcl : IsCleanRelocation hU hV y
    · rw [if_pos hcl]
    · rw [if_neg hcl, map_smul, inner_smul_left]
      by_cases hΨ : Ψpre y = 0
      · rw [hΨ, map_zero, zero_mul]
      · rw [hX₂, inner_V_single_toPhys_eq_zero hU hV l hstrict hUlbl hUrec horth ξ hξ y
          (hsupp y hΨ) hcl, mul_zero]
  calc ‖inner ℂ (V (restr F Ψpre) - U (restr F Ψpre)) X₂‖
      = ‖inner ℂ (V (restr Fcl Ψpre)) X₂‖ := by
        rw [inner_sub_left, hUzero, sub_zero, hsplit]
    _ ≤ ‖V (restr Fcl Ψpre)‖ * erel :=
        norm_inner_le_of_restr (V (restr Fcl Ψpre)) X₂ erel hreach
    _ = Real.sqrt (mass Ψpre Fcl) * erel := by
        rw [hViso (restr Fcl Ψpre)
          (fun y hy => hsupp y fun h0 => hy (by simp [restr_apply, h0])),
          norm_restr_eq_sqrt_mass]

end Dqi
