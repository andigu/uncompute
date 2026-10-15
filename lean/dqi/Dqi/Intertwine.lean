import Dqi.Phys
import Dqi.Shell
import Dqi.Residual
import Dqi.Bridge

/-!
# The syndrome-pullback intertwining

The measured objective, conjugated to the syndrome basis, equals the
pattern-space shell operator transported along the syndrome pushforward:
`synOp χ B Fs (toSyn B ψ) = toSyn B (shellOp χ Fs ψ)`, with **no distance
hypothesis**.  Both sides are computed on measurement-basis vectors and land on
a common shift normal form (`Dqi.shiftForm`), so the identity is an exact
operator equation.  Distance conditions enter the development only through the
metric transport of `toSyn` (`Dqi.inner_toSyn_toSyn`), never through this
identity.

Main result:
* `Dqi.synOp_toSyn`: the physical objective and the pattern-space objective are
  the same operator, transported along the syndrome pushforward.
-/

open Matrix

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K] {m n : ℕ}

/-- The shift normal form: what both the physical objective on a syndrome-basis
vector and the pushed-forward shell operator compute pointwise.  Per constraint
`i` and shift `a`, a character-weighted delta at the syndrome shifted by
`a • Bᵀ eᵢ`, minus `m` times the unshifted delta. -/
noncomputable def shiftForm (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (u₀ u : Fin n → K) : ℂ :=
  (∑ i : Fin m, ∑ a : K,
      2 * ((Fintype.card K : ℝ)⁻¹ : ℂ) * (∑ v ∈ Fs i, χ (a * v)) *
        (if u = u₀ + a • (B.transpose *ᵥ Pi.single i 1) then 1 else 0))
    - (m : ℂ) * (if u = u₀ then 1 else 0)

omit [Field K] [DecidableEq K] in
/-- The complex inverse-cardinality constant, written the way `siteOp` and
`shiftForm` spell it, is the inverse of the `ℕ → ℂ` cast. -/
private theorem card_inv_cast :
    ((Fintype.card K : ℝ)⁻¹ : ℂ) = (Fintype.card K : ℂ)⁻¹ := by
  rw [Complex.ofReal_natCast]

omit [DecidableEq K] in
/-- The square of the unitary normalizer of `ftN` cancels the character
orthogonality mass `(#K)^n` exactly. -/
private theorem prefactor_sq_mul_card :
    ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) *
        ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) * (Fintype.card K : ℂ) ^ n = 1 := by
  have hq : (0 : ℝ) < (Fintype.card K : ℝ) := by exact_mod_cast Fintype.card_pos
  have hr : (Real.sqrt (Fintype.card K))⁻¹ * (Real.sqrt (Fintype.card K))⁻¹ *
      (Fintype.card K : ℝ) = 1 := by
    rw [← mul_inv, Real.mul_self_sqrt hq.le]
    exact inv_mul_cancel₀ hq.ne'
  have hc : (((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) *
      (((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) * (Fintype.card K : ℂ) = 1 := by
    rw [← Complex.ofReal_mul, ← Complex.ofReal_natCast, ← Complex.ofReal_mul, hr,
      Complex.ofReal_one]
  rw [← mul_pow, ← mul_pow, hc, one_pow]

omit [Fintype K] [DecidableEq K] in
/-- Component formula for the syndrome direction of a single-site shift. -/
private theorem transpose_mulVec_single (B : Matrix (Fin m) (Fin n) K) (i : Fin m)
    (j : Fin n) : (B.transpose *ᵥ (Pi.single i (1 : K) : Pattern m K)) j = B i j := by
  have h : (B.transpose *ᵥ (Pi.single i (1 : K) : Pattern m K)) j
      = ∑ k : Fin m, B.transpose j k * (Pi.single i (1 : K) : Pattern m K) k := rfl
  have hterm : ∀ k : Fin m, B.transpose j k * (Pi.single i (1 : K) : Pattern m K) k
      = if k = i then B.transpose j k else 0 := by
    intro k
    by_cases hk : k = i
    · subst hk
      rw [if_pos rfl, Pi.single_eq_same, mul_one]
    · rw [if_neg hk, Pi.single_eq_of_ne hk, mul_zero]
  rw [h, Finset.sum_congr rfl fun k _ => hterm k,
    Finset.sum_ite_eq' Finset.univ i fun k => B.transpose j k,
    if_pos (Finset.mem_univ i), Matrix.transpose_apply]

/-- The scalar delta expanded as a character sum over all shifts. -/
private theorem indicator_eq_char_sum (χ : AddChar K ℂ) (hχ : χ ≠ 1) (w v : K) :
    (if w = v then (1 : ℂ) else 0) =
      (Fintype.card K : ℂ)⁻¹ * ∑ a : K, χ (a * (w - v)) := by
  have hcomm : ∀ a : K, χ (a * (w - v)) = χ ((w - v) * a) := fun a => by rw [mul_comm]
  rw [Finset.sum_congr rfl fun a _ => hcomm a, sum_char_mul χ hχ (w - v)]
  by_cases h : w = v
  · rw [if_pos h, if_pos (sub_eq_zero.mpr h),
      inv_mul_cancel₀ (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)]
  · rw [if_neg h, if_neg fun hz => h (sub_eq_zero.mp hz), mul_zero]

omit [Field K] [Fintype K] in
/-- The membership indicator expanded as a sum of scalar deltas. -/
private theorem indicator_mem_eq_sum (F : Finset K) (w : K) :
    (if w ∈ F then (1 : ℂ) else 0) = ∑ v ∈ F, if w = v then (1 : ℂ) else 0 :=
  (Finset.sum_ite_eq F w fun _ => (1 : ℂ)).symm

omit [Fintype K] [DecidableEq K] in
/-- Updating `y₀` at `i` to `y₀ i + a` is a single-site additive shift. -/
private theorem update_add_eq (y₀ : Pattern m K) (i : Fin m) (a : K) :
    Function.update y₀ i (y₀ i + a) = y₀ + a • (Pi.single i (1 : K) : Pattern m K) := by
  funext j
  by_cases hj : j = i
  · subst hj
    rw [Function.update_self, Pi.add_apply, Pi.smul_apply, Pi.single_eq_same, smul_eq_mul,
      mul_one]
  · rw [Function.update_of_ne hj, Pi.add_apply, Pi.smul_apply, Pi.single_eq_of_ne hj,
      smul_eq_mul, mul_zero, add_zero]

omit [Fintype K] [DecidableEq K] in
/-- The syndrome of a single-site additive shift. -/
private theorem syndrome_update_add (B : Matrix (Fin m) (Fin n) K) (y₀ : Pattern m K)
    (i : Fin m) (a : K) :
    syndrome B (Function.update y₀ i (y₀ i + a))
      = syndrome B y₀ + a • (B.transpose *ᵥ (Pi.single i (1 : K) : Pattern m K)) := by
  rw [update_add_eq]
  unfold syndrome
  rw [Matrix.mulVec_add, Matrix.mulVec_smul]

omit [Field K] in
/-- Summing a function gated on `Function.update y i (y₀ i) = y₀` over all
patterns reindexes to the single free coordinate at site `i`. -/
private theorem sum_update_support (y₀ : Pattern m K) (i : Fin m) (G : Pattern m K → ℂ) :
    (∑ y : Pattern m K, if Function.update y i (y₀ i) = y₀ then G y else 0)
      = ∑ t : K, G (Function.update y₀ i t) := by
  have hy : ∀ y : Pattern m K, (if Function.update y i (y₀ i) = y₀ then G y else 0)
      = ∑ t : K, if y = Function.update y₀ i t then G y else 0 := by
    intro y
    by_cases h : Function.update y i (y₀ i) = y₀
    · rw [if_pos h]
      have hcond : ∀ t : K, (y = Function.update y₀ i t) ↔ t = y i := by
        intro t
        constructor
        · intro hyt
          rw [hyt, Function.update_self]
        · intro ht
          subst ht
          funext j
          by_cases hj : j = i
          · subst hj
            rw [Function.update_self]
          · rw [Function.update_of_ne hj, ← h, Function.update_of_ne hj]
      have hsum : (∑ t : K, if y = Function.update y₀ i t then G y else 0)
          = ∑ t : K, if t = y i then G y else 0 :=
        Finset.sum_congr rfl fun t _ => if_congr (hcond t) rfl rfl
      rw [hsum, Finset.sum_ite_eq' Finset.univ (y i) fun _ => G y,
        if_pos (Finset.mem_univ _)]
    · rw [if_neg h]
      symm
      refine Finset.sum_eq_zero fun t _ => ?_
      have hne : ¬(y = Function.update y₀ i t) := by
        intro hyt
        apply h
        rw [hyt, Function.update_idem, Function.update_eq_self]
      rw [if_neg hne]
  rw [Finset.sum_congr rfl fun y _ => hy y, Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Finset.sum_ite_eq' Finset.univ (Function.update y₀ i t) G, if_pos (Finset.mem_univ _)]

/-- **Step A**, pointwise: the pushforward of a measurement-basis vector is a
delta at the syndrome of the pattern. -/
private theorem toSyn_single_apply (B : Matrix (Fin m) (Fin n) K) (y₀ : Pattern m K)
    (u : Fin n → K) :
    toSyn B (EuclideanSpace.single y₀ (1 : ℂ)) u = if u = syndrome B y₀ then 1 else 0 := by
  rw [toSyn_apply]
  have hterm : ∀ y : Pattern m K,
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

/-- **Step A**: the pushforward sends a measurement-basis vector to the
basis vector at its syndrome. -/
private theorem toSyn_single (B : Matrix (Fin m) (Fin n) K) (y₀ : Pattern m K) :
    toSyn B (EuclideanSpace.single y₀ (1 : ℂ))
      = EuclideanSpace.single (syndrome B y₀) (1 : ℂ) := by
  apply PiLp.ext
  intro u
  rw [toSyn_single_apply, PiLp.single_apply]

/-- The inverse transform of a syndrome-basis vector is a flat conjugate
character profile; mirror of `ftN_single`. -/
private theorem ftNInv_single (χ : AddChar K ℂ) (u₀ x : Fin n → K) :
    ftNInv χ (EuclideanSpace.single u₀ (1 : ℂ)) x
      = ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) * χ (-(∑ j, u₀ j * x j)) := by
  rw [ftNInv_apply]
  congr 1
  have hterm : ∀ u : Fin n → K,
      χ (-(∑ j, u j * x j)) * EuclideanSpace.single u₀ (1 : ℂ) u
        = if u = u₀ then χ (-(∑ j, u j * x j)) else 0 := by
    intro u
    rw [PiLp.single_apply]
    by_cases h : u = u₀ <;> simp [h]
  rw [Finset.sum_congr rfl fun u _ => hterm u,
    Finset.sum_ite_eq' Finset.univ u₀ fun u => χ (-(∑ j, u j * x j)),
    if_pos (Finset.mem_univ u₀)]

/-- The site operator on a measurement-basis vector, pointwise: the shift sum
collapses on the update condition. -/
private theorem siteOp_single_apply (χ : AddChar K ℂ) (F : Finset K) (i : Fin m)
    (y₀ y : Pattern m K) :
    siteOp χ F i (EuclideanSpace.single y₀ (1 : ℂ)) y
      = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
          (if Function.update y i (y₀ i) = y₀ then ∑ v ∈ F, χ ((y i - y₀ i) * v) else 0) := by
  rw [siteOp_apply]
  congr 1
  have hterm : ∀ a : K,
      (∑ v ∈ F, χ ((y i - a) * v)) *
          (EuclideanSpace.single y₀ (1 : ℂ)) (Function.update y i a)
        = if a = y₀ i then
            (if Function.update y i (y₀ i) = y₀ then ∑ v ∈ F, χ ((y i - a) * v) else 0)
          else 0 := by
    intro a
    rw [PiLp.single_apply]
    by_cases ha : a = y₀ i
    · subst ha
      rw [if_pos rfl]
      by_cases h2 : Function.update y i (y₀ i) = y₀
      · rw [if_pos h2, if_pos h2, mul_one]
      · rw [if_neg h2, if_neg h2, mul_zero]
    · have hupd : ¬(Function.update y i a = y₀) := fun h2 =>
        ha (by rw [← h2, Function.update_self])
      rw [if_neg ha, if_neg hupd, mul_zero]
  rw [Finset.sum_congr rfl fun a _ => hterm a,
    Finset.sum_ite_eq' Finset.univ (y₀ i)
      (fun a => if Function.update y i (y₀ i) = y₀ then ∑ v ∈ F, χ ((y i - a) * v) else 0),
    if_pos (Finset.mem_univ _)]

/-- **Step C**: the pushforward of the shell operator on a measurement-basis
vector, pointwise, is the shift normal form at the pattern's syndrome. -/
private theorem toSyn_shellOp_single (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (y₀ : Pattern m K) (u : Fin n → K) :
    toSyn B (shellOp χ Fs (EuclideanSpace.single y₀ (1 : ℂ))) u
      = shiftForm χ B Fs (syndrome B y₀) u := by
  unfold shiftForm
  rw [toSyn_apply]
  have hsplit : ∀ y : Pattern m K,
      (if syndrome B y = u then shellOp χ Fs (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)
        = ∑ i : Fin m,
            (2 * (if syndrome B y = u then
                siteOp χ (Fs i) i (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)
              - (if syndrome B y = u then (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)) := by
    intro y
    by_cases h : syndrome B y = u
    · rw [if_pos h, shellOp_apply]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [if_pos h, if_pos h]
    · rw [if_neg h]
      symm
      refine Finset.sum_eq_zero fun i _ => ?_
      rw [if_neg h, if_neg h, mul_zero, sub_zero]
  have hsite : ∀ i : Fin m,
      (∑ y : Pattern m K, if syndrome B y = u then
          siteOp χ (Fs i) i (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)
        = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ∑ a : K, (∑ v ∈ Fs i, χ (a * v)) *
              (if u = syndrome B y₀ + a • (B.transpose *ᵥ Pi.single i 1) then 1 else 0) := by
    intro i
    have h1 : ∀ y : Pattern m K,
        (if syndrome B y = u then
            siteOp χ (Fs i) i (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)
          = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
              (if Function.update y i (y₀ i) = y₀ then
                (if syndrome B y = u then ∑ v ∈ Fs i, χ ((y i - y₀ i) * v) else 0) else 0) := by
      intro y
      rw [siteOp_single_apply]
      by_cases hs : syndrome B y = u
      · rw [if_pos hs]
        by_cases hu : Function.update y i (y₀ i) = y₀
        · rw [if_pos hu, if_pos hu, if_pos hs]
        · rw [if_neg hu, if_neg hu, mul_zero]
      · rw [if_neg hs]
        by_cases hu : Function.update y i (y₀ i) = y₀
        · rw [if_pos hu, if_neg hs, mul_zero]
        · rw [if_neg hu, mul_zero]
    rw [Finset.sum_congr rfl fun y _ => h1 y, ← Finset.mul_sum,
      sum_update_support y₀ i
        (fun y => if syndrome B y = u then ∑ v ∈ Fs i, χ ((y i - y₀ i) * v) else 0)]
    congr 1
    refine (Fintype.sum_equiv (Equiv.addLeft (y₀ i)) _ _ fun a => ?_).symm
    show (∑ v ∈ Fs i, χ (a * v)) *
          (if u = syndrome B y₀ + a • (B.transpose *ᵥ Pi.single i 1) then 1 else 0)
        = if syndrome B (Function.update y₀ i (y₀ i + a)) = u then
            ∑ v ∈ Fs i, χ ((Function.update y₀ i (y₀ i + a) i - y₀ i) * v) else 0
    rw [Function.update_self, add_sub_cancel_left, syndrome_update_add]
    by_cases hcond : u = syndrome B y₀ + a • (B.transpose *ᵥ Pi.single i 1)
    · rw [if_pos hcond, if_pos hcond.symm, mul_one]
    · rw [if_neg hcond, if_neg fun hh => hcond hh.symm, mul_zero]
  have hterm : ∀ i : Fin m,
      (∑ y : Pattern m K,
          (2 * (if syndrome B y = u then
              siteOp χ (Fs i) i (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)
            - (if syndrome B y = u then (EuclideanSpace.single y₀ (1 : ℂ)) y else 0)))
        = (∑ a : K, 2 * ((Fintype.card K : ℝ)⁻¹ : ℂ) * (∑ v ∈ Fs i, χ (a * v)) *
              (if u = syndrome B y₀ + a • (B.transpose *ᵥ Pi.single i 1) then 1 else 0))
          - (if u = syndrome B y₀ then 1 else 0) := by
    intro i
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hsite i, ← toSyn_apply, toSyn_single_apply]
    congr 1
    rw [Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => ?_
    ring
  rw [Finset.sum_congr rfl fun y _ => hsplit y, Finset.sum_comm,
    Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_fin, nsmul_eq_mul]

omit [Fintype K] [DecidableEq K] in
/-- The character-orthogonality survivor condition of the physical side, solved
for `u`: vanishing of the shifted dot-product vector pins `u` to the shifted
syndrome. -/
private theorem shift_delta (B : Matrix (Fin m) (Fin n) K) (u₀ u : Fin n → K)
    (i : Fin m) (a : K) :
    ((fun j => a * B i j + (u j - u₀ j)) = (0 : Fin n → K))
      ↔ u = u₀ - a • (B.transpose *ᵥ (Pi.single i (1 : K) : Pattern m K)) := by
  rw [funext_iff, funext_iff]
  refine forall_congr' fun j => ?_
  rw [Pi.zero_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul, transpose_mulVec_single]
  constructor
  · intro h
    have h2 : u j - (u₀ j - a * B i j) = 0 := by rw [← h]; ring
    exact sub_eq_zero.mp h2
  · intro h
    rw [h]
    ring

/-- The constraint-indicator part of the physical side, per constraint: expand
the membership indicator into character deltas and collapse the `x`-sum by
orthogonality, leaving one shifted delta per shift `a`. -/
private theorem synOp_main_sum (χ : AddChar K ℂ) (hχ : χ ≠ 1)
    (B : Matrix (Fin m) (Fin n) K) (Fs : Fin m → Finset K) (u₀ u : Fin n → K)
    (i : Fin m) :
    (∑ x : Fin n → K,
        (if (B.mulVec x) i ∈ Fs i then (1 : ℂ) else 0) * χ (∑ j, (u j - u₀ j) * x j))
      = ∑ a : K, (Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(a * v))) *
          (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
            (Fintype.card K : ℂ) ^ n else 0) := by
  have hfac : ∀ (v a : K) (x : Fin n → K),
      χ (a * ((B.mulVec x) i - v)) * χ (∑ j, (u j - u₀ j) * x j)
        = χ (-(a * v)) * χ (∑ j, (a * B i j + (u j - u₀ j)) * x j) := by
    intro v a x
    rw [← AddChar.map_add_eq_mul, ← AddChar.map_add_eq_mul]
    congr 1
    have hw : (B.mulVec x) i = ∑ j, B i j * x j := rfl
    have hexp : (∑ j, (a * B i j + (u j - u₀ j)) * x j)
        = a * (∑ j, B i j * x j) + ∑ j, (u j - u₀ j) * x j := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      ring
    rw [hw, hexp]
    ring
  have hind : ∀ x : Fin n → K, (if (B.mulVec x) i ∈ Fs i then (1 : ℂ) else 0)
      = ∑ v ∈ Fs i, (Fintype.card K : ℂ)⁻¹ * ∑ a : K, χ (a * ((B.mulVec x) i - v)) := by
    intro x
    rw [indicator_mem_eq_sum]
    exact Finset.sum_congr rfl fun v _ => indicator_eq_char_sum χ hχ _ v
  have h2 : ∀ x : Fin n → K,
      (if (B.mulVec x) i ∈ Fs i then (1 : ℂ) else 0) * χ (∑ j, (u j - u₀ j) * x j)
        = ∑ v ∈ Fs i, ∑ a : K, (Fintype.card K : ℂ)⁻¹ *
            (χ (-(a * v)) * χ (∑ j, (a * B i j + (u j - u₀ j)) * x j)) := by
    intro x
    rw [hind x, Finset.sum_mul]
    refine Finset.sum_congr rfl fun v _ => ?_
    rw [mul_assoc, Finset.sum_mul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [hfac v a x]
  have h3 : ∀ v a : K,
      (∑ x : Fin n → K, (Fintype.card K : ℂ)⁻¹ *
          (χ (-(a * v)) * χ (∑ j, (a * B i j + (u j - u₀ j)) * x j)))
        = (Fintype.card K : ℂ)⁻¹ * (χ (-(a * v)) *
            (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
              (Fintype.card K : ℂ) ^ n else 0)) := by
    intro v a
    rw [← Finset.mul_sum, ← Finset.mul_sum,
      sum_char_dot χ hχ fun j => a * B i j + (u j - u₀ j),
      if_congr (shift_delta B u₀ u i a) rfl rfl]
  rw [Finset.sum_congr rfl fun x _ => h2 x, Finset.sum_comm]
  refine Eq.trans (Finset.sum_congr rfl fun v _ => Finset.sum_comm) ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => ?_
  refine Eq.trans (Finset.sum_congr rfl fun v _ => h3 v a) ?_
  rw [← Finset.mul_sum, ← Finset.sum_mul, ← mul_assoc]

/-- **Step B**: the physical objective on a syndrome-basis vector, pointwise,
is the shift normal form. -/
private theorem synOp_single (χ : AddChar K ℂ) (hχ : χ ≠ 1) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (u₀ u : Fin n → K) :
    synOp χ B Fs (EuclideanSpace.single u₀ (1 : ℂ)) u = shiftForm χ B Fs u₀ u := by
  have hmerge : ∀ x : Fin n → K,
      χ (∑ j, u j * x j) * χ (-(∑ j, u₀ j * x j)) = χ (∑ j, (u j - u₀ j) * x j) := by
    intro x
    rw [← AddChar.map_add_eq_mul]
    congr 1
    rw [← sub_eq_add_neg, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => (sub_mul _ _ _).symm
  have hfObj : ∀ x : Fin n → K, ((fObj B Fs x : ℝ) : ℂ)
      = ∑ i : Fin m, (2 * (if (B.mulVec x) i ∈ Fs i then (1 : ℂ) else 0) - 1) := by
    intro x
    unfold fObj
    rw [Complex.ofReal_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases h : (B.mulVec x) i ∈ Fs i
    · rw [if_pos h, if_pos h, Complex.ofReal_one]
      ring
    · rw [if_neg h, if_neg h, Complex.ofReal_neg, Complex.ofReal_one]
      ring
  have hDsum : (∑ x : Fin n → K, χ (∑ j, (u j - u₀ j) * x j))
      = if u = u₀ then (Fintype.card K : ℂ) ^ n else 0 := by
    rw [sum_char_dot χ hχ fun j => u j - u₀ j]
    refine if_congr ?_ rfl rfl
    constructor
    · intro h
      funext j
      exact sub_eq_zero.mp (by simpa using congrFun h j)
    · intro h
      subst h
      funext j
      simp
  unfold shiftForm
  rw [card_inv_cast]
  unfold synOp
  rw [ftN_apply]
  have hx : ∀ x : Fin n → K,
      χ (∑ j, u j * x j) *
          (mulOp (fObj B Fs) (ftNInv χ (EuclideanSpace.single u₀ (1 : ℂ)))) x
        = ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) *
            (((fObj B Fs x : ℝ) : ℂ) * χ (∑ j, (u j - u₀ j) * x j)) := by
    intro x
    rw [mulOp_apply, ftNInv_single, ← hmerge x]
    ring
  rw [Finset.sum_congr rfl fun x _ => hx x, ← Finset.mul_sum]
  set s : ℂ := ((((Real.sqrt (Fintype.card K))⁻¹ : ℝ) : ℂ) ^ n) with hs
  have hss : s * s * (Fintype.card K : ℂ) ^ n = 1 := by
    rw [hs]
    exact prefactor_sq_mul_card
  have habs : ∀ (P : Prop) [inst : Decidable P],
      s * s * (if P then (Fintype.card K : ℂ) ^ n else 0) = if P then 1 else 0 := by
    intro P inst
    by_cases hP : P
    · rw [if_pos hP, if_pos hP]
      exact hss
    · rw [if_neg hP, if_neg hP, mul_zero]
  have hswap : (∑ x : Fin n → K, ((fObj B Fs x : ℝ) : ℂ) * χ (∑ j, (u j - u₀ j) * x j))
      = ∑ i : Fin m,
          (2 * (∑ a : K, (Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(a * v))) *
              (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
                (Fintype.card K : ℂ) ^ n else 0))
            - (if u = u₀ then (Fintype.card K : ℂ) ^ n else 0)) := by
    have h1 : ∀ x : Fin n → K, ((fObj B Fs x : ℝ) : ℂ) * χ (∑ j, (u j - u₀ j) * x j)
        = ∑ i : Fin m,
            (2 * ((if (B.mulVec x) i ∈ Fs i then (1 : ℂ) else 0) *
                χ (∑ j, (u j - u₀ j) * x j))
              - χ (∑ j, (u j - u₀ j) * x j)) := by
      intro x
      rw [hfObj x, Finset.sum_mul]
      refine Finset.sum_congr rfl fun i _ => ?_
      ring
    rw [Finset.sum_congr rfl fun x _ => h1 x, Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hDsum, synOp_main_sum χ hχ B Fs u₀ u i]
  have hper : ∀ i : Fin m,
      s * (s * (2 * (∑ a : K, (Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(a * v))) *
              (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
                (Fintype.card K : ℂ) ^ n else 0))
            - (if u = u₀ then (Fintype.card K : ℂ) ^ n else 0)))
        = (∑ a : K, 2 * (Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (a * v)) *
              (if u = u₀ + a • (B.transpose *ᵥ Pi.single i 1) then 1 else 0))
          - (if u = u₀ then 1 else 0) := by
    intro i
    have e2 : ∀ a : K,
        s * s * ((Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(a * v))) *
            (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
              (Fintype.card K : ℂ) ^ n else 0))
          = (Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(a * v))) *
              (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then 1 else 0) := by
      intro a
      rw [show s * s * ((Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(a * v))) *
              (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
                (Fintype.card K : ℂ) ^ n else 0))
            = (Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(a * v))) *
                (s * s * (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
                  (Fintype.card K : ℂ) ^ n else 0)) from by ring,
        habs]
    have e1 : s * (s * (2 * (∑ a : K, (Fintype.card K : ℂ)⁻¹ *
            (∑ v ∈ Fs i, χ (-(a * v))) *
            (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
              (Fintype.card K : ℂ) ^ n else 0))
          - (if u = u₀ then (Fintype.card K : ℂ) ^ n else 0)))
        = 2 * (∑ a : K, s * s * ((Fintype.card K : ℂ)⁻¹ *
            (∑ v ∈ Fs i, χ (-(a * v))) *
            (if u = u₀ - a • (B.transpose *ᵥ Pi.single i 1) then
              (Fintype.card K : ℂ) ^ n else 0)))
          - s * s * (if u = u₀ then (Fintype.card K : ℂ) ^ n else 0) := by
      rw [← Finset.mul_sum]
      ring
    rw [e1, habs, Finset.mul_sum]
    congr 1
    refine (Fintype.sum_equiv (Equiv.neg K) _ _ fun a => ?_).symm
    show 2 * (Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (a * v)) *
          (if u = u₀ + a • (B.transpose *ᵥ Pi.single i 1) then 1 else 0)
        = 2 * (s * s * ((Fintype.card K : ℂ)⁻¹ * (∑ v ∈ Fs i, χ (-(-a * v))) *
            (if u = u₀ - -a • (B.transpose *ᵥ Pi.single i 1) then
              (Fintype.card K : ℂ) ^ n else 0)))
    rw [e2 (-a), neg_smul, sub_neg_eq_add]
    have hv : (∑ v ∈ Fs i, χ (-(-a * v))) = ∑ v ∈ Fs i, χ (a * v) :=
      Finset.sum_congr rfl fun v _ => by rw [neg_mul, neg_neg]
    rw [hv]
    ring
  rw [hswap, Finset.mul_sum, Finset.mul_sum,
    Finset.sum_congr rfl fun i _ => hper i, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_fin, nsmul_eq_mul]

omit [Field K] in
/-- Basis expansion of a pattern-space vector: `restr_eq_sum` at the full
index set. -/
private theorem eq_sum_single (ψ : EuclideanSpace ℂ (Pattern m K)) :
    ψ = ∑ y : Pattern m K, ψ y • EuclideanSpace.single y (1 : ℂ) := by
  have huniv : restr (Finset.univ : Finset (Pattern m K)) ψ = ψ := by
    apply PiLp.ext
    intro y
    rw [restr_apply, if_pos (Finset.mem_univ y)]
  have h := restr_eq_sum (Finset.univ : Finset (Pattern m K)) ψ
  rw [huniv] at h
  exact h

private theorem toSyn_zero (B : Matrix (Fin m) (Fin n) K) :
    toSyn B (0 : EuclideanSpace ℂ (Pattern m K)) = 0 := by
  have h := toSyn_smul B 0 0
  rw [zero_smul, zero_smul] at h
  exact h

private theorem synOp_zero (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) :
    synOp χ B Fs (0 : EuclideanSpace ℂ (Fin n → K)) = 0 := by
  have h := synOp_smul χ B Fs 0 0
  rw [zero_smul, zero_smul] at h
  exact h

private theorem toSyn_sum {ι : Type*} (B : Matrix (Fin m) (Fin n) K) (s : Finset ι)
    (f : ι → EuclideanSpace ℂ (Pattern m K)) :
    toSyn B (∑ y ∈ s, f y) = ∑ y ∈ s, toSyn B (f y) := by
  classical
  induction s using Finset.cons_induction with
  | empty => simpa using toSyn_zero B
  | cons a s ha ih => rw [Finset.sum_cons, Finset.sum_cons, toSyn_add, ih]

private theorem synOp_sum {ι : Type*} (χ : AddChar K ℂ) (B : Matrix (Fin m) (Fin n) K)
    (Fs : Fin m → Finset K) (s : Finset ι)
    (f : ι → EuclideanSpace ℂ (Fin n → K)) :
    synOp χ B Fs (∑ y ∈ s, f y) = ∑ y ∈ s, synOp χ B Fs (f y) := by
  classical
  induction s using Finset.cons_induction with
  | empty => simpa using synOp_zero χ B Fs
  | cons a s ha ih => rw [Finset.sum_cons, Finset.sum_cons, synOp_add, ih]

omit [DecidableEq K] in
private theorem shellOp_sum_pattern {ι : Type*} (χ : AddChar K ℂ)
    (Fs : Fin m → Finset K) (s : Finset ι)
    (f : ι → EuclideanSpace ℂ (Pattern m K)) :
    shellOp χ Fs (∑ y ∈ s, f y) = ∑ y ∈ s, shellOp χ Fs (f y) := by
  classical
  induction s using Finset.cons_induction with
  | empty => simpa using shellOp_zero χ Fs
  | cons a s ha ih => rw [Finset.sum_cons, Finset.sum_cons, shellOp_add, ih]

/-- **The syndrome-pullback intertwining.**  The measured objective, conjugated
to the syndrome basis, equals the pattern-space shell operator transported
along the syndrome pushforward — with **no distance hypothesis**.  Distance
conditions enter the development only through the metric transport of `toSyn`,
never through this identity. -/
theorem synOp_toSyn (χ : AddChar K ℂ) (hχ : χ ≠ 1)
    (B : Matrix (Fin m) (Fin n) K) (Fs : Fin m → Finset K)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    synOp χ B Fs (toSyn B ψ) = toSyn B (shellOp χ Fs ψ) := by
  have hsingle : ∀ y₀ : Pattern m K,
      synOp χ B Fs (toSyn B (EuclideanSpace.single y₀ (1 : ℂ)))
        = toSyn B (shellOp χ Fs (EuclideanSpace.single y₀ (1 : ℂ))) := by
    intro y₀
    rw [toSyn_single]
    apply PiLp.ext
    intro u
    rw [synOp_single χ hχ B Fs (syndrome B y₀) u]
    exact (toSyn_shellOp_single χ B Fs y₀ u).symm
  conv_lhs => rw [eq_sum_single ψ]
  conv_rhs => rw [eq_sum_single ψ]
  rw [toSyn_sum, synOp_sum, shellOp_sum_pattern, toSyn_sum]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [toSyn_smul, synOp_smul, shellOp_smul, toSyn_smul, hsingle y]

end Dqi
