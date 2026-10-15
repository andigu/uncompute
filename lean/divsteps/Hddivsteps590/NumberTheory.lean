/-
# Number theory of the integer iteration

Once `g` reaches zero, the carried rows deliver the results:
`gcd_hdRun` (`|f|` is the gcd), `inverse_hdRun` (the modular inverse),
`d_hdRun_of_g_zero` (the `g = 0` start), and `prime_coprime`. The
opening section holds the `hdStep`/`hdRun` bookkeeping shared with
`Invariant` and `Recovery`; this module depends only on `Defs`.

Coq counterparts: `gcd_step`, `gcd_spec`, `modulo_invariant`,
`modulo_step`, `modulo_spec`, `d_zero_spec` (hddivsteps_def.v).
-/
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Int.GCD
import Mathlib.Algebra.Ring.Int.Parity
import Mathlib.RingTheory.Int.Basic
import Hddivsteps590.Defs

namespace Hddivsteps

/-! ## `hdStep` and `hdRun` bookkeeping -/

/-- Unfold one step of the run. -/
theorem hdRun_succ (n : Nat) (f g : Int) :
    hdRun (n + 1) f g = hdStep (hdRun n f g) := rfl

/-- A step keeps `f` odd. -/
theorem hdStep_odd_f (s : HDState) (h : s.f % 2 = 1) : (hdStep s).f % 2 = 1 := by
  rw [hdStep]
  split
  · exact h
  · split
    · exact show s.g % 2 = 1 by omega
    · exact h

/-- A step keeps `g = 0`. -/
theorem hdStep_g_zero (s : HDState) (h : s.g = 0) : (hdStep s).g = 0 := by
  rw [hdStep]
  split
  next => exact show s.g / 2 = 0 by omega
  next hg => exact absurd (by omega : s.g % 2 = 0) hg

/-- A step from `g = 0` keeps `d` fixed. -/
theorem hdStep_d_of_g_zero (s : HDState) (h : s.g = 0) : (hdStep s).d = s.d := by
  rw [hdStep]
  split
  next => rfl
  next hg => exact absurd (by omega : s.g % 2 = 0) hg

/-! ## Private helpers: gcd is stable under the divstep branches -/

/-- Exact halving of an even argument does not change the gcd with an odd
`f` (the inner `Hgcd` of Coq `gcd_step`). -/
private theorem gcd_half (f g : Int) (hf : f % 2 = 1) (hg : g % 2 = 0) :
    Int.gcd f (g / 2) = Int.gcd f g := by
  obtain ⟨x, rfl⟩ : (2 : Int) ∣ g := by omega
  have hcop : Nat.Coprime 2 f.natAbs :=
    Nat.coprime_two_left.mpr (Int.natAbs_odd.mpr (Int.odd_iff.mpr hf))
  rw [Int.mul_ediv_cancel_left x two_ne_zero, Int.gcd_eq_natAbs_gcd_natAbs,
    Int.gcd_eq_natAbs_gcd_natAbs, Int.natAbs_mul]
  exact (hcop.gcd_mul_left_cancel_right x.natAbs).symm

/-- One divstep preserves the gcd (Coq `gcd_step`). -/
private theorem gcd_hdStep (s : HDState) (hf : s.f % 2 = 1) :
    Int.gcd (hdStep s).f (hdStep s).g = Int.gcd s.f s.g := by
  unfold hdStep
  split_ifs with h1 h2
  · exact gcd_half s.f s.g hf h1
  · show Int.gcd s.g ((s.g - s.f) / 2) = _
    rw [gcd_half s.g _ (by omega) (by omega), Int.gcd_self_sub_right, Int.gcd_comm]
  · show Int.gcd s.f ((s.g + s.f) / 2) = _
    rw [gcd_half s.f _ hf (by omega), Int.gcd_add_self_right]

private theorem hdRun_odd_f (N : Nat) (f g : Int) (hodd : f % 2 = 1) :
    (hdRun N f g).f % 2 = 1 := by
  induction N with
  | zero => exact hodd
  | succ n ih => exact hdStep_odd_f _ ih

/-- The run preserves `gcd(f, g)`. -/
theorem gcd_hdRun_eq (N : Nat) (f g : Int) (hodd : f % 2 = 1) :
    Int.gcd (hdRun N f g).f (hdRun N f g).g = Int.gcd f g := by
  induction N with
  | zero => rfl
  | succ n ih => rw [hdRun_succ, gcd_hdStep _ (hdRun_odd_f n f g hodd), ih]

/-- After `g` reaches 0, `|f|` is the gcd (Coq `gcd_spec`, specialized to
the run from `hdInit`). -/
theorem gcd_hdRun (N : Nat) (f g : Int) (hodd : f % 2 = 1)
    (h0 : (hdRun N f g).g = 0) :
    Int.gcd f g = (hdRun N f g).f.natAbs := by
  rw [← gcd_hdRun_eq N f g hodd, h0, Int.gcd_zero_right]

/-! ## Private helpers: the modular invariant (Coq `modulo_*`) -/

/-- Cancel a factor of 2 in a congruence with odd modulus. -/
private theorem modEq_of_two_mul (M a b : Int) (hM : M % 2 = 1)
    (h : 2 * a ≡ 2 * b [ZMOD M]) : a ≡ b [ZMOD M] := by
  have hcop : IsCoprime M 2 :=
    Int.isCoprime_two_right.mpr (Int.odd_iff.mpr hM)
  rw [Int.modEq_iff_dvd] at h ⊢
  exact hcop.dvd_of_dvd_mul_left (by rwa [← mul_sub] at h)

/-- `div2M` is division by 2 mod an odd `M` (Coq `eqm_div2M`, doubled). -/
private theorem two_mul_div2M (M x : Int) (hM : M % 2 = 1) :
    2 * div2M M x ≡ x [ZMOD M] := by
  unfold div2M
  split_ifs with hx
  · rw [Int.mul_ediv_cancel' (by omega : (2 : Int) ∣ x + M)]
    exact Int.modEq_iff_dvd.mpr ⟨-1, by ring⟩
  · rw [Int.mul_ediv_cancel' (by omega : (2 : Int) ∣ x)]

/-- The common step of the invariant: an even value congruent to `c·g`
halves to a value congruent to `div2M f c · g`. -/
private theorem div2M_step {f g a c : Int} (hodd : f % 2 = 1) (ha2 : a % 2 = 0)
    (ha : a ≡ c * g [ZMOD f]) : a / 2 ≡ div2M f c * g [ZMOD f] := by
  apply modEq_of_two_mul f _ _ hodd
  rw [Int.mul_ediv_cancel' (by omega : (2 : Int) ∣ a)]
  calc a ≡ c * g [ZMOD f] := ha
    _ ≡ 2 * div2M f c * g [ZMOD f] := ((two_mul_div2M f c hodd).mul_right g).symm
    _ = 2 * (div2M f c * g) := by ring

/-- The Coq `modulo_invariant` carried along the run (with `modulo_init`
and `modulo_step` folded into one induction; `x := g`, modulus `f`). -/
private theorem hdRun_modEq_inv (N : Nat) (f g : Int) (hodd : f % 2 = 1) :
    (hdRun N f g).modulus = f ∧
    ((hdRun N f g).f ≡ (hdRun N f g).d * g [ZMOD f] ∧
     (hdRun N f g).g ≡ (hdRun N f g).e * g [ZMOD f]) := by
  induction N with
  | zero =>
    exact ⟨rfl, show f ≡ 0 * g [ZMOD f] by simp [Int.ModEq],
      show g ≡ 1 * g [ZMOD f] by rw [one_mul]⟩
  | succ n ih =>
    obtain ⟨hm, hf, hg⟩ := ih
    rw [hdRun_succ]
    set s := hdRun n f g
    have hfo : s.f % 2 = 1 := hdRun_odd_f n f g hodd
    unfold hdStep
    rw [hm]
    split_ifs with h1 h2
    · -- even g: g ← g/2, e ← div2M M e
      exact ⟨rfl, hf, div2M_step hodd h1 hg⟩
    · -- odd g, delta > 0: f ← g, g ← (g−f)/2, d ← e, e ← div2M M (e−d)
      exact ⟨rfl, hg, div2M_step hodd (by omega) (by rw [sub_mul]; exact hg.sub hf)⟩
    · -- odd g, delta ≤ 0: g ← (g+f)/2, e ← div2M M (e+d)
      exact ⟨rfl, hf, div2M_step hodd (by omega) (by rw [add_mul]; exact hg.add hf)⟩

/-- After `g` reaches 0 on coprime inputs, the carried row is the modular
inverse (Coq `modulo_spec` at the initial state). -/
theorem inverse_hdRun (N : Nat) (f g : Int) (hodd : f % 2 = 1)
    (hcop : Int.gcd f g = 1) (h0 : (hdRun N f g).g = 0) :
    (hdRun N f g).f.natAbs = 1 ∧
    (hdRun N f g).d * (hdRun N f g).f * g ≡ 1 [ZMOD f] := by
  have habs : (hdRun N f g).f.natAbs = 1 := hcop ▸ (gcd_hdRun N f g hodd h0).symm
  obtain ⟨-, hf, -⟩ := hdRun_modEq_inv N f g hodd
  set s := hdRun N f g
  have hsq : s.f * s.f = 1 := by
    rcases Int.natAbs_eq_iff.mp habs with h | h <;> rw [h] <;> norm_num
  refine ⟨habs, ?_⟩
  calc s.d * s.f * g = s.d * g * s.f := by ring
    _ ≡ s.f * s.f [ZMOD f] := (hf.mul_right s.f).symm
    _ = 1 := hsq

/-- `g = 0` start: `g` stays 0 forever. -/
private theorem hdRun_g_zero (N : Nat) (f : Int) : (hdRun N f 0).g = 0 := by
  induction N with
  | zero => rfl
  | succ n ih => exact hdStep_g_zero _ ih

/-- `g = 0` start: `d` stays 0 forever (Coq `d_zero_spec`). -/
theorem d_hdRun_of_g_zero (N : Nat) (f : Int) : (hdRun N f 0).d = 0 := by
  induction N with
  | zero => rfl
  | succ n ih => rw [hdRun_succ, hdStep_d_of_g_zero _ (hdRun_g_zero n f), ih]

/-- A prime is coprime to everything strictly between 0 and itself. -/
theorem prime_coprime (f g : Int) (hp : Prime f) (h0 : 0 < g)
    (hgf : g < f) : Int.gcd f g = 1 :=
  (Int.prime_iff_natAbs_prime.mp hp).coprime_iff_not_dvd.mpr fun hdvd => by
    have := Nat.le_of_dvd (by omega) hdvd
    omega

end Hddivsteps
