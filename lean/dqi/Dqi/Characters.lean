import Mathlib.NumberTheory.LegendreSymbol.AddCharacter
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
import Mathlib.FieldTheory.Finite.Trace

/-!
# Additive-character orthogonality over a finite field

These are the orthogonality relations behind the syndrome-pullback step of the
DQI analysis.  Summing a nontrivial additive character `χ : AddChar K ℂ` over
the syndrome register collapses to a delta: `∑ x, χ (a * x)` vanishes unless
`a = 0`, and the vector version `∑ x, χ (∑ j, v j * x j)` vanishes unless
`v = 0`.  Applied with `v = Bᵀ (y' - y - a • eᵢ)`, this delta is exactly what
forces `Bᵀ (y' - y - a • eᵢ) = 0` in the pullback argument.

Main results:
* `Dqi.sum_char_mul`: `∑ x : K, χ (a * x) = if a = 0 then #K else 0`.
* `Dqi.sum_char_dot`: `∑ x : Fin n → K, χ (∑ j, v j * x j)
    = if v = 0 then #K ^ n else 0`.
* `Dqi.exists_addChar_ne_one`: a nontrivial `ℂ`-valued additive character
  exists on every finite field.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

omit [Fintype K] [DecidableEq K] in
/-- An additive character turns a finite sum in the source into a product of values. -/
theorem char_map_sum {ι : Type*} (χ : AddChar K ℂ) (s : Finset ι) (f : ι → K) :
    χ (∑ j ∈ s, f j) = ∏ j ∈ s, χ (f j) := by
  classical
  induction s using Finset.cons_induction with
  | empty => simp
  | cons a s ha ih => rw [Finset.sum_cons, Finset.prod_cons, AddChar.map_add_eq_mul, ih]

/-- Orthogonality, scaled form: summing a nontrivial additive character over all
scalings of a fixed `a : K` gives `#K` when `a = 0` and `0` otherwise. -/
theorem sum_char_mul (χ : AddChar K ℂ) (hχ : χ ≠ 1) (a : K) :
    ∑ x : K, χ (a * x) = if a = 0 then (Fintype.card K : ℂ) else 0 := by
  simp_rw [mul_comm a]
  rw [AddChar.sum_mulShift a (AddChar.IsPrimitive.of_ne_one hχ),
    apply_ite (Nat.cast : ℕ → ℂ), Nat.cast_zero]

/-- Orthogonality, vector form: summing a nontrivial additive character of the dot
product `∑ j, v j * x j` over all vectors `x` gives `#K ^ n` when `v = 0` and `0`
otherwise. -/
theorem sum_char_dot {n : ℕ} (χ : AddChar K ℂ) (hχ : χ ≠ 1) (v : Fin n → K) :
    ∑ x : Fin n → K, χ (∑ j, v j * x j) =
      if v = 0 then (Fintype.card K : ℂ) ^ n else 0 := by
  have hfactor : ∑ x : Fin n → K, χ (∑ j, v j * x j) = ∏ j, ∑ y : K, χ (v j * y) := by
    rw [← Finset.sum_prod_piFinset Finset.univ fun j y ↦ χ (v j * y),
      Fintype.piFinset_univ]
    exact Finset.sum_congr rfl fun x _ ↦ char_map_sum χ Finset.univ fun j ↦ v j * x j
  rw [hfactor]
  simp_rw [sum_char_mul χ hχ]
  by_cases hv : v = 0
  · simp [hv, Finset.prod_const]
  · rw [if_neg hv]
    obtain ⟨j, hj⟩ := Function.ne_iff.mp hv
    exact Finset.prod_eq_zero (Finset.mem_univ j) (if_neg hj)

/-- Every finite field carries a nontrivial `ℂ`-valued additive character: compose the
standard additive character of `ZMod p` (for `p` the characteristic) with the field
trace, which is nondegenerate. -/
theorem exists_addChar_ne_one (K : Type*) [Field K] [Fintype K] :
    ∃ χ : AddChar K ℂ, χ ≠ 1 := by
  have hp : Fact (ringChar K).Prime := ⟨CharP.char_is_prime K _⟩
  have : NeZero (ringChar K) := ⟨hp.out.ne_zero⟩
  let _ : Algebra (ZMod (ringChar K)) K := ZMod.algebra _ _
  refine ⟨(ZMod.stdAddChar (N := ringChar K)).compAddMonoidHom
      (Algebra.trace (ZMod (ringChar K)) K).toAddMonoidHom, ?_⟩
  obtain ⟨b, hb⟩ := FiniteField.trace_to_zmod_nondegenerate K (one_ne_zero (α := K))
  rw [one_mul] at hb
  refine AddChar.ne_one_iff.2 ⟨b, fun hf ↦ hb ?_⟩
  exact ((ZMod.isPrimitive_stdAddChar (ringChar K)).zmod_char_eq_one_iff _ _).mp hf

end Dqi
