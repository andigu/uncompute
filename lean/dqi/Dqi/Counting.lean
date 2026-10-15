import Dqi.LayerState
import Dqi.Reachable

/-!
# Counting ingredients of the relocation bound

Three counting facts underlying the relocation term of the DQI degradation
bound.  `card_hammingNorm_eq` computes the size of a Hamming shell: there are
`C(m,k) * (q-1)^k` patterns of weight exactly `k` (choose the support, then a
nonzero value at each support coordinate).  `norm_layerState_of_flat` shows
that under the flatness hypothesis satisfied by bent target sets — every
nonzero amplitude has magnitude `1/√(q-1)` — a layer state spreads uniformly
across its shell, each pattern carrying amplitude `1/√(shellCard)`.  Finally
`norm_restr_le_of_forall` converts a bound on the number of reachable labels
into a norm bound: if the error reaches at most `N` labels, each holding
amplitude at most `c`, the restricted vector has norm at most `√N * c` — the
shape of the relocation term `E_rel`.
-/

namespace Dqi

/-! ### The size of a Hamming shell -/

/-- The number of patterns of Hamming weight exactly `k`: choose a `k`-element
support, then a nonzero value at each support coordinate. -/
noncomputable def shellCard (m : ℕ) (K : Type*) [Fintype K] (k : ℕ) : ℕ :=
  m.choose k * (Fintype.card K - 1) ^ k

/-- **The Hamming shell has size `C(m,k) * (q-1)^k`.** -/
theorem card_hammingNorm_eq (m k : ℕ) (K : Type*) [Field K] [Fintype K] [DecidableEq K] :
    (Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).card = shellCard m K k := by
  classical
  -- every weight-`k` pattern has a `k`-element support
  have hmaps : ∀ y ∈ Finset.univ.filter (fun y : Pattern m K => hammingNorm y = k),
      (Finset.univ.filter fun i => y i ≠ 0) ∈
        Finset.powersetCard k (Finset.univ : Finset (Fin m)) := by
    intro y hy
    rw [Finset.mem_powersetCard]
    refine ⟨Finset.subset_univ _, ?_⟩
    rw [← hammingNorm_eq_card_support y]
    exact (Finset.mem_filter.mp hy).2
  -- for each fixed support, the fiber has `(q-1)^k` elements
  have hfiber : ∀ S ∈ Finset.powersetCard k (Finset.univ : Finset (Fin m)),
      ((Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).filter
          (fun y => (Finset.univ.filter fun i => y i ≠ 0) = S)).card =
        (Fintype.card K - 1) ^ k := by
    intro S hS
    obtain ⟨-, hScard⟩ := Finset.mem_powersetCard.mp hS
    -- the fiber over `S` is exactly the product finset "nonzero on S, zero off S"
    have hset : ((Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).filter
          (fun y => (Finset.univ.filter fun i => y i ≠ 0) = S)) =
        Fintype.piFinset fun i => if i ∈ S then (Finset.univ : Finset K).erase 0 else {0} := by
      ext y
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
      constructor
      · rintro ⟨-, hsupp⟩ i
        by_cases hi : i ∈ S
        · rw [if_pos hi]
          have hmem : i ∈ Finset.univ.filter fun i => y i ≠ 0 := by rw [hsupp]; exact hi
          exact Finset.mem_erase.mpr ⟨(Finset.mem_filter.mp hmem).2, Finset.mem_univ _⟩
        · rw [if_neg hi]
          have hyi : y i = 0 := by
            by_contra hne
            exact hi (hsupp ▸ Finset.mem_filter.mpr ⟨Finset.mem_univ _, hne⟩)
          simp [hyi]
      · intro h
        have hsupp : (Finset.univ.filter fun i => y i ≠ 0) = S := by
          ext i
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          by_cases hi : i ∈ S
          · have hyi := h i
            rw [if_pos hi] at hyi
            exact ⟨fun _ => hi, fun _ => (Finset.mem_erase.mp hyi).1⟩
          · have hyi := h i
            rw [if_neg hi, Finset.mem_singleton] at hyi
            exact ⟨fun hne => absurd hyi hne, fun hmem => absurd hmem hi⟩
        exact ⟨by rw [hammingNorm_eq_card_support y, hsupp, hScard], hsupp⟩
    calc ((Finset.univ.filter fun y : Pattern m K => hammingNorm y = k).filter
            (fun y => (Finset.univ.filter fun i => y i ≠ 0) = S)).card
        = (Fintype.piFinset fun i =>
            if i ∈ S then (Finset.univ : Finset K).erase 0 else {0}).card := by rw [hset]
      _ = ∏ i : Fin m,
            ((if i ∈ S then (Finset.univ : Finset K).erase 0 else {0}) : Finset K).card :=
          Fintype.card_piFinset _
      _ = ∏ i : Fin m, (if i ∈ S then Fintype.card K - 1 else 1) := by
          refine Finset.prod_congr rfl fun i _ => ?_
          by_cases hi : i ∈ S
          · rw [if_pos hi, if_pos hi, Finset.card_erase_of_mem (Finset.mem_univ 0),
              Finset.card_univ]
          · rw [if_neg hi, if_neg hi, Finset.card_singleton]
      _ = (Fintype.card K - 1) ^ k := by
          rw [Finset.prod_ite_mem, Finset.univ_inter, Finset.prod_const, hScard]
  rw [Finset.card_eq_sum_card_fiberwise hmaps, Finset.sum_congr rfl hfiber,
    Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin,
    smul_eq_mul]
  rfl

/-! ### Flat amplitudes give a flat shell -/

variable {m : ℕ} {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- **Flat amplitudes spread a layer state uniformly across its shell.**  Under
the flatness hypothesis satisfied by bent target sets — every nonzero amplitude
has magnitude `1/√(q-1)` — each weight-`k` pattern carries layer-state amplitude
of magnitude exactly `1/√(shellCard m K k)`. -/
theorem norm_layerState_of_flat (g : Fin m → K → ℂ) (k : ℕ)
    (hflat : ∀ i, ∀ a : K, a ≠ 0 → ‖g i a‖ = (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹)
    (y : Pattern m K) (hy : hammingNorm y = k) :
    ‖layerState g k y‖ = (Real.sqrt (shellCard m K k : ℝ))⁻¹ := by
  have hq : 1 < Fintype.card K := Fintype.one_lt_card
  have hq1 : (0 : ℝ) ≤ (Fintype.card K : ℝ) - 1 := by
    have h1 : (1 : ℝ) ≤ (Fintype.card K : ℝ) := by exact_mod_cast hq.le
    linarith
  have hsupp : (Finset.univ.filter fun i => y i ≠ 0).card = k := by
    rw [← hammingNorm_eq_card_support y]; exact hy
  have hcast : (shellCard m K k : ℝ) =
      (m.choose k : ℝ) * ((Fintype.card K : ℝ) - 1) ^ k := by
    simp only [shellCard]
    push_cast [Nat.cast_sub hq.le]
    ring
  -- the product of the coordinate magnitudes: one factor `1/√(q-1)` per
  -- support coordinate
  have hprod : (∏ i : Fin m, ‖if y i = 0 then (1 : ℂ) else g i (y i)‖) =
      (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹ ^ k := by
    calc (∏ i : Fin m, ‖if y i = 0 then (1 : ℂ) else g i (y i)‖)
        = ∏ i : Fin m,
            (if y i ≠ 0 then (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹ else 1) := by
          refine Finset.prod_congr rfl fun i _ => ?_
          by_cases h0 : y i = 0
          · simp [h0]
          · simp [h0, hflat i (y i) h0]
      _ = ∏ _i ∈ Finset.univ.filter fun i => y i ≠ 0,
            (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹ :=
          (Finset.prod_filter _ _).symm
      _ = (Real.sqrt ((Fintype.card K : ℝ) - 1))⁻¹ ^ k := by
          rw [Finset.prod_const, hsupp]
  -- pull the `k`-th power out of the radical
  have hpow : Real.sqrt (((Fintype.card K : ℝ) - 1) ^ k) =
      Real.sqrt ((Fintype.card K : ℝ) - 1) ^ k := by
    rw [show ((Fintype.card K : ℝ) - 1) ^ k =
          (Real.sqrt ((Fintype.card K : ℝ) - 1) ^ k) ^ 2 by
        rw [← pow_mul, mul_comm, pow_mul, Real.sq_sqrt hq1],
      Real.sqrt_sq (by positivity)]
  rw [layerState_apply, if_pos hy, norm_mul, Complex.norm_real, Real.norm_eq_abs,
    abs_of_nonneg (by positivity), norm_prod, hprod, hcast,
    Real.sqrt_mul (by positivity), mul_inv, hpow, inv_pow]

/-! ### From a label count to a norm bound -/

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **A label count converts to a norm bound.**  If a vector holds amplitude at
most `c` on each of the labels in `Z`, its restriction to `Z` has norm at most
`√|Z| * c` — the shape of the relocation term `E_rel`. -/
theorem norm_restr_le_of_forall (Z : Finset ι) (v : EuclideanSpace ℂ ι) (c : ℝ) (hc : 0 ≤ c)
    (h : ∀ i ∈ Z, ‖v i‖ ≤ c) :
    ‖restr Z v‖ ≤ Real.sqrt (Z.card) * c := by
  have hsum : ∑ i, ‖restr Z v i‖ ^ 2 ≤ (Z.card : ℝ) * c ^ 2 := by
    calc ∑ i, ‖restr Z v i‖ ^ 2
        = ∑ i ∈ Z, ‖restr Z v i‖ ^ 2 :=
          (Finset.sum_subset (Finset.subset_univ Z) fun i _ hi => by simp [hi]).symm
      _ = ∑ i ∈ Z, ‖v i‖ ^ 2 := Finset.sum_congr rfl fun i hi => by simp [hi]
      _ ≤ ∑ _i ∈ Z, c ^ 2 :=
          Finset.sum_le_sum fun i hi => pow_le_pow_left₀ (norm_nonneg _) (h i hi) 2
      _ = (Z.card : ℝ) * c ^ 2 := by rw [Finset.sum_const, nsmul_eq_mul]
  calc ‖restr Z v‖
      = Real.sqrt (∑ i, ‖restr Z v i‖ ^ 2) := EuclideanSpace.norm_eq _
    _ ≤ Real.sqrt ((Z.card : ℝ) * c ^ 2) := Real.sqrt_le_sqrt hsum
    _ = Real.sqrt (Z.card) * c := by
        rw [Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq hc]

end Dqi
