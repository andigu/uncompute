/-
# Exact branch recovery (the accompanying paper's appendix, in Lean)

The `z`-form walk and branch-recovery theory of the appendix “Exact branch
recovery for divstep inversion” of the accompanying paper (arXiv:2610.22367).
Correspondence (paper → Lean). The Lean docstrings call the decoding rule
the *anchored ladder* and the cheap tests the *swap*, *nonnegative-counter*,
and *terminating classes*.

* Basic invariants, determinant      → `det_zWalk`
* Basic invariants, parity           → `parity_zWalk`
* Scaling and termination            → `scale_zWalk`, `endpoint_zWalk`,
  `inverse_zWalk`, `inverse_zWalk_unscaled`
* Cheap tests, part 1 (swap)         → `sigma_class`, `bbit_of_swap`
* Cheap tests, part 2 (`z' ≥ 1`)     → `bbit_of_ramp`
* Cheap tests, part 3 (`g' = 0`)     → `bbit_of_freeze`
* Window invariant                   → `invariant_zWalk` and `window_zWalk`
  (clause (i) is the `Inv.high` field; clause (ii) is `window_low`)
* Decoding rule for an ambiguous     → `ladder` (first form) and `ladder_det`
  decision                             (second form), stated exactly as in
  the paper: `v₂` is `padicValInt 2`, `bit_k(u)` is `bitk u k`, and
  `· ⁻¹ mod 2^j` is the inverse in `ZMod (2^j)` with `.val` reading off the
  representative in `[0, 2^j)`.
* Finite dependence                  → `finite_dependence`: for fixed `p`
  and `j`, one function of the two `j`-bit residues decodes every step.
* Depth-two rule                     → `j2_law`

The bridge theorem `zWalk_hdRun` proves that this frozen `z`-form walk and
the certificate's `delta`-form walk have identical `f,g` trajectories, with
`delta = 2z+1` before termination.

Naming: generic walk theorems are `property_zWalk`; concrete 590-step
instantiations (in `Main.lean`) are `zWalk590_property`. The proofs run
through private inverse-free engine lemmas (`ladder_witness`,
`ladder_exists`): the window witness `r < 2^j` with `g_v' ≡ q·r (mod 2^j)`
exists, and every such witness carries the branch bit at position `j - 1`.

The step bound (`hdRun`) lives in the sibling
modules; `Main.lean` transfers it across `zWalk_hdRun`.
-/
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Int.GCD
import Mathlib.Algebra.Ring.Int.Parity
import Mathlib.RingTheory.Int.Basic
import Mathlib.NumberTheory.Padics.PadicVal.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.ZMod.Units
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Hddivsteps590.NumberTheory

namespace Hddivsteps

/-! ## The z-form state and one step -/

/-- The `z`-form divstep state `(z, f, g, f_v, g_v)`. -/
structure ZState where
  z : Int
  f : Int
  g : Int
  fv : Int
  gv : Int

variable {p x : Int} {s : ZState}

/-- One active step.  The swap condition is `0 ≤ z ∧ g` odd; the branch bit
of a non-swap step is `g % 2`. -/
def zStep (s : ZState) : ZState :=
  if 0 ≤ s.z ∧ s.g % 2 = 1 then
    ⟨-s.z, s.g, (s.g - s.f) / 2, 2 * s.gv, s.gv - s.fv⟩
  else
    ⟨s.z + 1, s.f, (s.g + s.g % 2 * s.f) / 2, 2 * s.fv, s.gv + s.g % 2 * s.fv⟩

/-- The branch bit `b` of the step taken from `s`: the parity of the working
`g` in post-swap names. On a swap step that working value is the pre-swap
`f`, so `b = f mod 2` there — equal to 1 at every reachable state
(`bbit_of_swap`, the swap-class theorem's "moreover" clause). -/
def bbit (s : ZState) : Int :=
  if 0 ≤ s.z ∧ s.g % 2 = 1 then s.f % 2 else s.g % 2

theorem zStep_swap (hσ : 0 ≤ s.z ∧ s.g % 2 = 1) :
    zStep s = ⟨-s.z, s.g, (s.g - s.f) / 2, 2 * s.gv, s.gv - s.fv⟩ :=
  if_pos hσ

theorem zStep_nonswap (hσ : ¬(0 ≤ s.z ∧ s.g % 2 = 1)) :
    zStep s = ⟨s.z + 1, s.f, (s.g + s.g % 2 * s.f) / 2, 2 * s.fv,
               s.gv + s.g % 2 * s.fv⟩ :=
  if_neg hσ

theorem bbit_of_nonswap (hσ : ¬(0 ≤ s.z ∧ s.g % 2 = 1)) : bbit s = s.g % 2 :=
  if_neg hσ

/-- Seed: `f = p`, `g = x`, `f_v = 0`, `g_v = 1`, `z = 0`. -/
def zSeed (p x : Int) : ZState := ⟨0, p, x, 0, 1⟩

/-- The fixed-schedule walk: frozen (`g = 0`) steps execute nothing. -/
def zWalk (p x : Int) : Nat → ZState
  | 0 => zSeed p x
  | t + 1 =>
    let s := zWalk p x t
    if s.g = 0 then s else zStep s

theorem zWalk_succ (t : Nat) :
    zWalk p x (t + 1) =
      if (zWalk p x t).g = 0 then zWalk p x t else zStep (zWalk p x t) := rfl

/-! ## Correspondence with the certificate walk

The certificate uses `HDState`, with `delta = 2z+1` and coefficient rows
scaled in the opposite direction.  The two definitions nevertheless have
identical `f` and `g` trajectories.  Once `g` reaches zero, `zWalk` freezes;
`hdRun` continues changing bookkeeping fields, but its `f` and `g` fields
remain fixed. -/

/-- The `z`-form walk and the certificate's `delta`-form walk have identical
`f` and `g` fields.  Before termination their counters satisfy
`delta = 2z+1`. -/
theorem zWalk_hdRun (p x : Int) (t : Nat) :
    (zWalk p x t).f = (hdRun t p x).f ∧
    (zWalk p x t).g = (hdRun t p x).g ∧
    ((zWalk p x t).g ≠ 0 → (hdRun t p x).delta = 2 * (zWalk p x t).z + 1) := by
  induction t with
  | zero => norm_num [zWalk, zSeed, hdRun, hdInit, Nat.repeat]
  | succ t ih =>
    obtain ⟨hf, hg, hdelta⟩ := ih
    rw [zWalk_succ, hdRun_succ]
    by_cases hzero : (zWalk p x t).g = 0
    · have hzero' : (hdRun t p x).g = 0 := by rw [← hg, hzero]
      rw [if_pos hzero]
      simp [hdStep, hzero', hf, hg]
    · rw [if_neg hzero]
      have hd := hdelta hzero
      rcases (by omega : (zWalk p x t).g % 2 = 0 ∨
          (zWalk p x t).g % 2 = 1) with heven | hodd
      · -- even g: both walks only halve
        have heven' : (hdRun t p x).g % 2 = 0 := by rw [← hg]; exact heven
        rw [zStep_nonswap (by omega), hdStep, if_pos heven']
        exact ⟨hf, by dsimp only; rw [heven, zero_mul, add_zero, ← hg],
          fun _ => by dsimp only; omega⟩
      · have hodd' : (hdRun t p x).g % 2 ≠ 0 := by rw [← hg]; omega
        rw [hdStep, if_neg hodd']
        rcases (by omega : 0 ≤ (zWalk p x t).z ∨ (zWalk p x t).z < 0)
          with hz | hz
        · -- odd g, z ≥ 0: both walks swap
          rw [if_pos (by omega : 0 < (hdRun t p x).delta),
            zStep_swap ⟨hz, hodd⟩]
          exact ⟨hg, by dsimp only; rw [hf, hg], fun _ => by dsimp only; omega⟩
        · -- odd g, z < 0: both walks add then halve
          rw [if_neg (by omega : ¬ 0 < (hdRun t p x).delta),
            zStep_nonswap (by omega)]
          exact ⟨hf, by dsimp only; rw [← hg, hodd, one_mul, hf],
            fun _ => by dsimp only; omega⟩

/-! ## The reachable-state invariant

`det`/`fodd` are the determinant and parity invariants; `high`/`low` are the
two clauses of the window invariant. `low` is stated inverse-free:
`g_v ≡ q·r (mod 2^(j-z))` for a witness `r < 2^j`. -/

/-- The reachable-state invariant: the determinant identity, oddness of `f`,
and the two clauses of the window invariant. -/
structure Inv (p : Int) (s : ZState) : Prop where
  /-- Determinant invariant: `f·g_v − g·f_v = p`. -/
  det : s.f * s.gv - s.g * s.fv = p
  /-- Parity invariant: `f` is odd. -/
  fodd : s.f % 2 = 1
  /-- Window invariant, clause (i): while `z ≥ 0`, `2^(z+1) ∣ f_v`. -/
  high : 0 ≤ s.z → (2 : Int) ^ (s.z.toNat + 1) ∣ s.fv
  /-- Window invariant, clause (ii), inverse-free: while `z < 0`, a witness
  `r < 2^j` has `g_v ≡ q·r (mod 2^(j−z))` for `f_v = 2^j·q`, `q` odd. -/
  low : s.z < 0 → ∃ j : Nat, ∃ q r : Int, 1 ≤ j ∧ q % 2 = 1 ∧
    s.fv = 2 ^ j * q ∧ 0 ≤ r ∧ r < 2 ^ j ∧
    (2 : Int) ^ (j + (-s.z).toNat) ∣ s.gv - q * r

/-- Parity invariant: `f_v` is even (from either window clause). -/
theorem Inv.fveven (h : Inv p s) : s.fv % 2 = 0 := by
  rcases (by omega : s.z < 0 ∨ 0 ≤ s.z) with hz | hz
  · obtain ⟨j, q, r, hj, -, hfv, -, -, -⟩ := h.low hz
    have h2 : (2 : Int) ∣ s.fv :=
      hfv ▸ (dvd_pow_self 2 (by omega : j ≠ 0)).mul_right q
    omega
  · have h2 : (2 : Int) ∣ s.fv :=
      dvd_trans (dvd_pow_self 2 (by omega : s.z.toNat + 1 ≠ 0)) (h.high hz)
    omega

/-- Parity invariant: `g_v` is odd (from the determinant identity). -/
theorem Inv.gvodd (h : Inv p s) (hp : p % 2 = 1) : s.gv % 2 = 1 := by
  obtain ⟨k, hk⟩ : (2 : Int) ∣ s.fv := by have := h.fveven; omega
  have h2 : s.f * s.gv = p + 2 * (s.g * k) := by
    linear_combination h.det + s.g * hk
  have h1 : s.f * s.gv % 2 = 1 := by
    rw [h2, Int.add_mul_emod_self_left]; exact hp
  rw [Int.mul_emod, h.fodd] at h1
  omega

/-! ## Preservation: one step keeps the invariant -/

/-- The window invariant's induction step: one active step preserves it. -/
theorem Inv.zStep (h : Inv p s) (hp : p % 2 = 1) : Inv p (zStep s) := by
  have hfodd := h.fodd
  have hfv2 := h.fveven
  have hgv2 := h.gvodd hp
  by_cases hσ : 0 ≤ s.z ∧ s.g % 2 = 1
  · -- swap step, firing at z = D ≥ 0
    obtain ⟨hz0, hg2⟩ := hσ
    rw [zStep_swap ⟨hz0, hg2⟩]
    obtain ⟨k, hk⟩ : (2 : Int) ∣ s.g - s.f := by omega
    have hdiv : (s.g - s.f) / 2 = k := by
      rw [hk]; exact Int.mul_ediv_cancel_left k two_ne_zero
    refine ⟨?_, hg2, ?_, ?_⟩
    · show s.g * (s.gv - s.fv) - (s.g - s.f) / 2 * (2 * s.gv) = p
      rw [hdiv]; linear_combination h.det + s.gv * hk
    · -- post-counter −z is nonnegative only for z = 0 (case D = 0)
      show 0 ≤ -s.z → (2 : Int) ^ ((-s.z).toNat + 1) ∣ 2 * s.gv
      intro h0
      have hz : s.z = 0 := by omega
      rw [hz]
      norm_num
    · -- z' = −D < 0: the fresh window, witness r = 1 (case D > 0)
      show -s.z < 0 → ∃ j : Nat, ∃ q r : Int, 1 ≤ j ∧ q % 2 = 1 ∧
        2 * s.gv = 2 ^ j * q ∧ 0 ≤ r ∧ r < 2 ^ j ∧
        (2 : Int) ^ (j + (- -s.z).toNat) ∣ (s.gv - s.fv) - q * r
      intro hneg
      refine ⟨1, s.gv, 1, le_rfl, hgv2, by ring, by norm_num, by norm_num, ?_⟩
      have hexp : 1 + (- -s.z).toNat = s.z.toNat + 1 := by omega
      have hval : (s.gv - s.fv) - s.gv * 1 = -s.fv := by ring
      rw [hexp, hval]
      exact dvd_neg.mpr (h.high hz0)
  · -- non-swap step; the high clause is branch-independent
    rw [zStep_nonswap hσ]
    have hhigh : 0 ≤ s.z + 1 →
        (2 : Int) ^ ((s.z + 1).toNat + 1) ∣ 2 * s.fv := by
      intro h1
      rcases (by omega : 0 ≤ s.z ∨ s.z < 0) with hz0 | hz0
      · obtain ⟨c, hc⟩ := h.high hz0
        have hexp : (s.z + 1).toNat + 1 = (s.z.toNat + 1) + 1 := by omega
        rw [hexp, pow_succ, hc]; exact ⟨c, by ring⟩
      · have hz : s.z = -1 := by omega
        rw [hz]; norm_num
    rcases (by omega : s.g % 2 = 0 ∨ s.g % 2 = 1) with hb | hb
    · -- branch bit 0: only halve
      obtain ⟨k, hk⟩ : (2 : Int) ∣ s.g := by omega
      rw [hb]
      simp only [zero_mul, add_zero]
      have hdiv : s.g / 2 = k := by
        rw [hk]; exact Int.mul_ediv_cancel_left k two_ne_zero
      refine ⟨?_, hfodd, hhigh, ?_⟩
      · show s.f * s.gv - s.g / 2 * (2 * s.fv) = p
        rw [hdiv]; linear_combination h.det + s.fv * hk
      · show s.z + 1 < 0 → ∃ j : Nat, ∃ q r : Int, 1 ≤ j ∧ q % 2 = 1 ∧
          2 * s.fv = 2 ^ j * q ∧ 0 ≤ r ∧ r < 2 ^ j ∧
          (2 : Int) ^ (j + (-(s.z + 1)).toNat) ∣ s.gv - q * r
        intro h1
        obtain ⟨j, q, r, hj, hq, hfv, hr0, hr1, hdvd⟩ := h.low (by omega)
        have hpos : (0 : Int) < 2 ^ j := by positivity
        refine ⟨j + 1, q, r, by omega, hq, by rw [hfv, pow_succ]; ring,
          hr0, ?_, ?_⟩
        · rw [pow_succ]; linarith
        · have hexp : (j + 1) + (-(s.z + 1)).toNat = j + (-s.z).toNat := by
            omega
          rw [hexp]; exact hdvd
    · -- branch bit 1: add the rows, then halve
      obtain ⟨k, hk⟩ : (2 : Int) ∣ s.g + s.f := by omega
      rw [hb]
      simp only [one_mul]
      have hdiv : (s.g + s.f) / 2 = k := by
        rw [hk]; exact Int.mul_ediv_cancel_left k two_ne_zero
      refine ⟨?_, hfodd, hhigh, ?_⟩
      · show s.f * (s.gv + s.fv) - (s.g + s.f) / 2 * (2 * s.fv) = p
        rw [hdiv]; linear_combination h.det + s.fv * hk
      · show s.z + 1 < 0 → ∃ j : Nat, ∃ q r : Int, 1 ≤ j ∧ q % 2 = 1 ∧
          2 * s.fv = 2 ^ j * q ∧ 0 ≤ r ∧ r < 2 ^ j ∧
          (2 : Int) ^ (j + (-(s.z + 1)).toNat) ∣ (s.gv + s.fv) - q * r
        intro h1
        obtain ⟨j, q, r, hj, hq, hfv, hr0, hr1, hdvd⟩ := h.low (by omega)
        have hpos : (0 : Int) < 2 ^ j := by positivity
        refine ⟨j + 1, q, r + 2 ^ j, by omega, hq, by rw [hfv, pow_succ]; ring,
          by linarith, ?_, ?_⟩
        · rw [pow_succ]; linarith
        · have hexp : (j + 1) + (-(s.z + 1)).toNat = j + (-s.z).toNat := by
            omega
          have hval : (s.gv + s.fv) - q * (r + 2 ^ j) = s.gv - q * r := by
            rw [hfv]; ring
          rw [hexp, hval]; exact hdvd

/-- The seed satisfies the invariant. -/
theorem inv_zSeed (hp : p % 2 = 1) : Inv p (zSeed p x) := by
  refine ⟨?_, hp, ?_, ?_⟩
  · show p * 1 - x * 0 = p; ring
  · intro _; exact dvd_zero _
  · intro hlt; exact absurd hlt (by norm_num [zSeed])

/-- The window invariant: every reachable state satisfies `Inv`. -/
theorem invariant_zWalk (hp : p % 2 = 1) (x : Int) (t : Nat) :
    Inv p (zWalk p x t) := by
  induction t with
  | zero => exact inv_zSeed hp
  | succ t ih =>
    rw [zWalk_succ]
    split_ifs with h
    · exact ih
    · exact ih.zStep hp

/-- The determinant invariant, as stated: every reachable state satisfies
`f·g_v − g·f_v = p`. -/
theorem det_zWalk (hp : p % 2 = 1) (x : Int) (t : Nat) :
    (zWalk p x t).f * (zWalk p x t).gv
      - (zWalk p x t).g * (zWalk p x t).fv = p :=
  (invariant_zWalk hp x t).det

/-- The parity invariant, as stated: at every reachable state, `f` and
`g_v` are odd and `f_v` is even. -/
theorem parity_zWalk (hp : p % 2 = 1) (x : Int) (t : Nat) :
    (zWalk p x t).f % 2 = 1 ∧ (zWalk p x t).gv % 2 = 1 ∧
    (zWalk p x t).fv % 2 = 0 :=
  ⟨(invariant_zWalk hp x t).fodd, (invariant_zWalk hp x t).gvodd hp,
    (invariant_zWalk hp x t).fveven⟩

/-! ## Scaling: the congruences, the endpoint, and the inverse -/

private theorem modEq_congr {n a b a' b' : Int} (h : a ≡ b [ZMOD n])
    (ha : a = a') (hb : b = b') : a' ≡ b' [ZMOD n] := ha ▸ hb ▸ h

/-- Scaling congruences: after `t` active steps,
`f_v·x ≡ 2^t·f` and `g_v·x ≡ 2^t·g (mod p)`. -/
theorem scale_zWalk (hp : p % 2 = 1) (x : Int) (t : Nat)
    (hact : ∀ u < t, (zWalk p x u).g ≠ 0) :
    (zWalk p x t).fv * x ≡ 2 ^ t * (zWalk p x t).f [ZMOD p] ∧
    (zWalk p x t).gv * x ≡ 2 ^ t * (zWalk p x t).g [ZMOD p] := by
  induction t with
  | zero =>
    constructor
    · show (0 : Int) * x ≡ 2 ^ 0 * p [ZMOD p]
      simp [Int.ModEq]
    · show (1 : Int) * x ≡ 2 ^ 0 * x [ZMOD p]
      simp
  | succ t ih =>
    have hg := hact t (by omega)
    obtain ⟨ihf, ihg⟩ := ih (fun u hu => hact u (by omega))
    have hfodd := (invariant_zWalk hp x t).fodd
    rw [zWalk_succ, if_neg hg]
    set s := zWalk p x t with hs
    by_cases hσ : 0 ≤ s.z ∧ s.g % 2 = 1
    · rw [zStep_swap hσ]
      obtain ⟨m, hm⟩ : (2 : Int) ∣ s.g - s.f := by
        have hg2 := hσ.2
        omega
      have hdiv : (s.g - s.f) / 2 = m := by
        rw [hm]; exact Int.mul_ediv_cancel_left m two_ne_zero
      constructor
      · show 2 * s.gv * x ≡ 2 ^ (t + 1) * s.g [ZMOD p]
        refine modEq_congr (ihg.mul_left 2) (by ring) ?_
        rw [pow_succ]; ring
      · show (s.gv - s.fv) * x ≡ 2 ^ (t + 1) * ((s.g - s.f) / 2) [ZMOD p]
        rw [hdiv]
        refine modEq_congr (ihg.sub ihf) (by ring) ?_
        rw [pow_succ]; linear_combination ((2 : Int) ^ t) * hm
    · rw [zStep_nonswap hσ]
      -- the doubled f_v congruence is branch-independent
      have hfv2 : 2 * s.fv * x ≡ 2 ^ (t + 1) * s.f [ZMOD p] := by
        refine modEq_congr (ihf.mul_left 2) (by ring) ?_
        rw [pow_succ]; ring
      rcases (by omega : s.g % 2 = 0 ∨ s.g % 2 = 1) with hb | hb
      · obtain ⟨m, hm⟩ : (2 : Int) ∣ s.g := by omega
        rw [hb]; simp only [zero_mul, add_zero]
        have hdiv : s.g / 2 = m := by
          rw [hm]; exact Int.mul_ediv_cancel_left m two_ne_zero
        refine ⟨hfv2, ?_⟩
        show s.gv * x ≡ 2 ^ (t + 1) * (s.g / 2) [ZMOD p]
        rw [hdiv]
        refine modEq_congr ihg rfl ?_
        rw [pow_succ]; linear_combination ((2 : Int) ^ t) * hm
      · obtain ⟨m, hm⟩ : (2 : Int) ∣ s.g + s.f := by omega
        rw [hb]; simp only [one_mul]
        have hdiv : (s.g + s.f) / 2 = m := by
          rw [hm]; exact Int.mul_ediv_cancel_left m two_ne_zero
        refine ⟨hfv2, ?_⟩
        show (s.gv + s.fv) * x ≡ 2 ^ (t + 1) * ((s.g + s.f) / 2) [ZMOD p]
        rw [hdiv]
        refine modEq_congr (ihg.add ihf) (by ring) ?_
        rw [pow_succ]; linear_combination ((2 : Int) ^ t) * hm

/-- The walk preserves `gcd(f, g)`: transported from the certificate walk
(`gcd_hdRun_eq`) across the bridge `zWalk_hdRun`. -/
private theorem gcd_zWalk (hp : p % 2 = 1) (x : Int) (t : Nat) :
    Int.gcd (zWalk p x t).f (zWalk p x t).g = Int.gcd p x := by
  rw [(zWalk_hdRun p x t).1, (zWalk_hdRun p x t).2.1]
  exact gcd_hdRun_eq t p x hp

/-- At the freeze, `|f| = 1` and `g_v = f·p`. -/
theorem endpoint_zWalk (hp : p % 2 = 1) (hcop : Int.gcd p x = 1) (t : Nat)
    (h0 : (zWalk p x t).g = 0) :
    (zWalk p x t).f.natAbs = 1 ∧ (zWalk p x t).gv = (zWalk p x t).f * p := by
  have hgcd := gcd_zWalk hp x t
  rw [h0, Int.gcd_zero_right, hcop] at hgcd
  have hdet := (invariant_zWalk hp x t).det
  rw [h0] at hdet
  have hfg : (zWalk p x t).f * (zWalk p x t).gv = p := by
    linear_combination hdet
  refine ⟨hgcd, ?_⟩
  rcases Int.natAbs_eq_iff.mp hgcd with hf | hf <;> rw [hf] at hfg ⊢ <;> omega

/-- The delivered inverse: at the freeze,
`f·f_v·x ≡ 2^t (mod p)` — `f·f_v` is the `2^t`-scaled inverse of `x`. -/
theorem inverse_zWalk (hp : p % 2 = 1) (hcop : Int.gcd p x = 1) (t : Nat)
    (hact : ∀ u < t, (zWalk p x u).g ≠ 0) (h0 : (zWalk p x t).g = 0) :
    (zWalk p x t).f * (zWalk p x t).fv * x ≡ 2 ^ t [ZMOD p] := by
  obtain ⟨hf1, -⟩ := endpoint_zWalk hp hcop t h0
  have hsq : (zWalk p x t).f * (zWalk p x t).f = 1 := by
    rcases Int.natAbs_eq_iff.mp hf1 with hf | hf <;> rw [hf] <;> norm_num
  have hscale := (scale_zWalk hp x t hact).1
  refine modEq_congr (hscale.mul_left (zWalk p x t).f) (by ring) ?_
  calc (zWalk p x t).f * (2 ^ t * (zWalk p x t).f)
      = 2 ^ t * ((zWalk p x t).f * (zWalk p x t).f) := by ring
    _ = 2 ^ t := by rw [hsq, mul_one]

/-- The scaling invariant in explicit inverse form. If `a` represents
`2⁻ᵗ (mod p)`, then `f_v·a·f` represents `x⁻¹ (mod p)`. -/
theorem inverse_zWalk_unscaled (hp : p % 2 = 1) (hcop : Int.gcd p x = 1)
    (t : Nat) (hact : ∀ u < t, (zWalk p x u).g ≠ 0)
    (h0 : (zWalk p x t).g = 0) (a : Int) (ha : 2 ^ t * a ≡ 1 [ZMOD p]) :
    (zWalk p x t).fv * a * (zWalk p x t).f * x ≡ 1 [ZMOD p] := by
  calc
    (zWalk p x t).fv * a * (zWalk p x t).f * x
        = ((zWalk p x t).f * (zWalk p x t).fv * x) * a := by ring
    _ ≡ 2 ^ t * a [ZMOD p] := (inverse_zWalk hp hcop t hact h0).mul_right a
    _ ≡ 1 [ZMOD p] := ha

/-! ## The local tests -/

/-- Engine for the swap class: a step swapped iff `f_v' % 4 = 2`. -/
private theorem swap_iff_fv (h : Inv p s) (hp : p % 2 = 1) :
    (0 ≤ s.z ∧ s.g % 2 = 1) ↔ (zStep s).fv % 4 = 2 := by
  have hgv2 := h.gvodd hp
  have hfv2 := h.fveven
  by_cases hσ : 0 ≤ s.z ∧ s.g % 2 = 1
  · rw [zStep_swap hσ]
    exact iff_of_true hσ (by show 2 * s.gv % 4 = 2; omega)
  · rw [zStep_nonswap hσ]
    exact iff_of_false hσ (by show ¬ 2 * s.fv % 4 = 2; omega)

/-- The swap-class "moreover" clause: `σ = 1` implies `b = 1` — the post-swap
working `g` is the pre-swap `f`, odd at every reachable state
(by the parity invariant). -/
theorem bbit_of_swap (hf : s.f % 2 = 1) (hσ : 0 ≤ s.z ∧ s.g % 2 = 1) :
    bbit s = 1 := (if_pos hσ).trans hf

/-- The nonnegative-counter class: a nonswap landing at `z' ≥ 1` has `b = 0`. -/
theorem bbit_of_ramp (hσ : ¬(0 ≤ s.z ∧ s.g % 2 = 1))
    (hz : 1 ≤ (zStep s).z) : bbit s = 0 := by
  rw [zStep_nonswap hσ] at hz
  change 1 ≤ s.z + 1 at hz
  rw [bbit_of_nonswap hσ]
  by_contra hne
  exact hσ ⟨by omega, by omega⟩

/-- The terminating class: the terminating step has `b = 1`, whatever the
class (`f` is odd at every reachable state by the parity invariant). -/
theorem bbit_of_freeze (hf : s.f % 2 = 1) (hg : s.g ≠ 0)
    (h0 : (zStep s).g = 0) : bbit s = 1 := by
  by_cases hσ : 0 ≤ s.z ∧ s.g % 2 = 1
  · exact bbit_of_swap hf hσ
  · rw [zStep_nonswap hσ] at h0
    change (s.g + s.g % 2 * s.f) / 2 = 0 at h0
    rw [bbit_of_nonswap hσ]
    rcases (by omega : s.g % 2 = 0 ∨ s.g % 2 = 1) with hb | hb
    · rw [hb, zero_mul, add_zero] at h0
      omega
    · exact hb

/-! ## The anchored ladder -/

/-- `v₂` and the odd part, from an odd factorization (uniqueness). -/
private theorem padicValInt_of_odd_factor {n q : Int} {j : Nat}
    (hq : q % 2 = 1) (hn : n = 2 ^ j * q) :
    padicValInt 2 n = j ∧ n / 2 ^ j = q := by
  subst hn
  refine ⟨?_, Int.mul_ediv_cancel_left _ (by positivity)⟩
  rw [padicValInt.mul (by positivity) (by omega),
    padicValInt.eq_zero_of_not_dvd (p := 2) (z := q) (by push_cast; omega),
    add_zero, show ((2 : Int) ^ j) = ((2 ^ j : Nat) : Int) by push_cast; ring,
    padicValInt.of_nat, padicValNat.prime_pow]

/-- Odd-part factorizations are unique: `2^a·u = 2^b·v` with `u, v` odd
forces `a = b` and `u = v`. -/
private theorem two_pow_mul_odd_inj {a b : Nat} {u v : Int} (hu : u % 2 = 1)
    (hv : v % 2 = 1) (h : (2 : Int) ^ a * u = 2 ^ b * v) : a = b ∧ u = v := by
  have ha := padicValInt_of_odd_factor hu (rfl : (2 : Int) ^ a * u = 2 ^ a * u)
  have hb := padicValInt_of_odd_factor hv h
  obtain rfl : a = b := by have h1 := ha.1; have h2 := hb.1; omega
  exact ⟨rfl, ha.2.symm.trans hb.2⟩

/-- An odd integer is coprime to every power of two. -/
private theorem isCoprime_two_pow {k : Nat} {q : Int} (hq : q % 2 = 1) :
    IsCoprime ((2 : Int) ^ k) q :=
  (Int.isCoprime_two_left.mpr (Int.odd_iff.mpr hq)).pow_left

/-- Cancel an odd factor against a power of two. -/
private theorem pow_two_dvd_cancel {k : Nat} {q m : Int} (hq : q % 2 = 1)
    (h : (2 : Int) ^ k ∣ q * m) : (2 : Int) ^ k ∣ m :=
  (isCoprime_two_pow hq).dvd_of_dvd_mul_left h

/-- Two residues below the modulus that differ by a multiple of it agree. -/
private theorem eq_of_sub_dvd_of_bounds {n a b : Int} (h : n ∣ a - b)
    (ha0 : 0 ≤ a) (ha : a < n) (hb0 : 0 ≤ b) (hb : b < n) : a = b := by
  have hmod : b % n = a % n := Int.modEq_iff_dvd.mpr h
  rw [Int.emod_eq_of_lt hb0 hb, Int.emod_eq_of_lt ha0 ha] at hmod
  exact hmod.symm

/-- The engine behind the anchored ladder: for a negative-counter nonswap
(pre-state counter `z < 0`), write the post-state `f_v'` as `2^j·q` with
`q` odd; then any
window witness — any `r < 2^j` with `g_v' ≡ q·r (mod 2^j)` — has the
step's branch bit at its top: `b = r / 2^(j-1)`. -/
theorem ladder_witness (h : Inv p s) (hz : s.z < 0) {j : Nat} {q r : Int}
    (hfv : (zStep s).fv = 2 ^ j * q) (hq : q % 2 = 1)
    (hr0 : 0 ≤ r) (hr1 : r < 2 ^ j)
    (hdvd : (2 : Int) ^ j ∣ (zStep s).gv - q * r) :
    r / 2 ^ (j - 1) = bbit s := by
  have hσ : ¬(0 ≤ s.z ∧ s.g % 2 = 1) := fun hc => absurd hc.1 (by omega)
  obtain ⟨j₀, q₀, r₀, hj₀, hq₀, hfv₀, hr₀0, hr₀1, hdvd₀⟩ := h.low hz
  rw [zStep_nonswap hσ] at hfv hdvd
  change 2 * s.fv = 2 ^ j * q at hfv
  change (2 : Int) ^ j ∣ (s.gv + s.g % 2 * s.fv) - q * r at hdvd
  -- identify (j, q) with (j₀ + 1, q₀)
  rw [hfv₀] at hfv
  have hcomb : (2 : Int) ^ (j₀ + 1) * q₀ = 2 ^ j * q := by
    rw [pow_succ]; linear_combination hfv
  obtain ⟨hje, rfl⟩ := two_pow_mul_odd_inj hq₀ hq hcomb
  subst hje
  -- the deposited witness r₀ + b·2^{j₀} satisfies the same congruence as r
  have hstep : (2 : Int) ^ (j₀ + 1) ∣ s.gv - q₀ * r₀ :=
    dvd_trans (pow_dvd_pow 2 (by omega)) hdvd₀
  have hdvd' : (2 : Int) ^ (j₀ + 1) ∣
      (s.gv + s.g % 2 * s.fv) - q₀ * (r₀ + s.g % 2 * 2 ^ j₀) := by
    have hval : (s.gv + s.g % 2 * s.fv) - q₀ * (r₀ + s.g % 2 * 2 ^ j₀) =
        s.gv - q₀ * r₀ := by
      rw [hfv₀]; ring
    rw [hval]; exact hstep
  have hpos : (0 : Int) < 2 ^ j₀ := by positivity
  have hb01 : s.g % 2 = 0 ∨ s.g % 2 = 1 := by omega
  have hdiff : (2 : Int) ^ (j₀ + 1) ∣ r - (r₀ + s.g % 2 * 2 ^ j₀) :=
    pow_two_dvd_cancel hq₀ (by
      have hval : q₀ * (r - (r₀ + s.g % 2 * 2 ^ j₀)) =
          ((s.gv + s.g % 2 * s.fv) - q₀ * (r₀ + s.g % 2 * 2 ^ j₀)) -
            ((s.gv + s.g % 2 * s.fv) - q₀ * r) := by ring
      rw [hval]; exact dvd_sub hdvd' hdvd)
  have hreq : r = r₀ + s.g % 2 * 2 ^ j₀ := by
    refine eq_of_sub_dvd_of_bounds hdiff hr0 hr1 ?_ ?_
    · rcases hb01 with hb' | hb' <;> rw [hb'] <;> linarith
    · rw [pow_succ]; rcases hb01 with hb' | hb' <;> rw [hb'] <;> linarith
  rw [Nat.add_sub_cancel, hreq,
    Int.add_mul_ediv_right _ _ (by positivity : (2 : Int) ^ j₀ ≠ 0),
    Int.ediv_eq_zero_of_lt hr₀0 hr₀1, zero_add, bbit_of_nonswap hσ]

/-- Non-vacuity of `ladder_witness`: a negative-counter nonswap's post-state
always admits a window witness, with depth `j ≥ 2`. -/
private theorem ladder_exists (h : Inv p s) (hz : s.z < 0) :
    ∃ (j : Nat) (q r : Int), 2 ≤ j ∧ q % 2 = 1 ∧
      (zStep s).fv = 2 ^ j * q ∧ 0 ≤ r ∧ r < 2 ^ j ∧
      (2 : Int) ^ j ∣ (zStep s).gv - q * r := by
  have hσ : ¬(0 ≤ s.z ∧ s.g % 2 = 1) := fun hc => absurd hc.1 (by omega)
  obtain ⟨j₀, q₀, r₀, hj₀, hq₀, hfv₀, hr₀0, hr₀1, hdvd₀⟩ := h.low hz
  rw [zStep_nonswap hσ]
  have hpos : (0 : Int) < 2 ^ j₀ := by positivity
  have hb01 : s.g % 2 = 0 ∨ s.g % 2 = 1 := by omega
  refine ⟨j₀ + 1, q₀, r₀ + s.g % 2 * 2 ^ j₀, by omega, hq₀, ?_, ?_, ?_, ?_⟩
  · show 2 * s.fv = 2 ^ (j₀ + 1) * q₀
    rw [hfv₀, pow_succ]; ring
  · rcases hb01 with hb | hb <;> rw [hb] <;> linarith
  · rw [pow_succ]
    rcases hb01 with hb | hb <;> rw [hb] <;> linarith
  · show (2 : Int) ^ (j₀ + 1) ∣
      (s.gv + s.g % 2 * s.fv) - q₀ * (r₀ + s.g % 2 * 2 ^ j₀)
    have hval : (s.gv + s.g % 2 * s.fv) - q₀ * (r₀ + s.g % 2 * 2 ^ j₀) =
        s.gv - q₀ * r₀ := by
      rw [hfv₀]; ring
    rw [hval]
    exact dvd_trans (pow_dvd_pow 2 (by omega)) hdvd₀

/-! ## The depth-two law, engine form -/

/-- Engine for the depth-two law, phrased with `f_v' % 8 = 4` (⟺ `v₂ = 2`)
and explicit bit arithmetic. -/
private theorem j2_mod (h : Inv p s) (hp : p % 2 = 1) (hz : s.z < 0)
    (h4 : (zStep s).fv % 8 = 4) :
    bbit s = ((zStep s).gv % 4 / 2 + (zStep s).fv % 16 / 8) % 2 := by
  have hinv' : Inv p (zStep s) := h.zStep hp
  have hgv' : (zStep s).gv % 2 = 1 := hinv'.gvodd hp
  set q : Int := (zStep s).fv / 4 with hqdef
  have hfvq : (zStep s).fv = 2 ^ 2 * q := by omega
  have hqodd : q % 2 = 1 := by omega
  set r : Int := (zStep s).gv * q % 4 with hrdef
  obtain ⟨k, hk⟩ : ∃ k, q = 2 * k + 1 := ⟨q / 2, by omega⟩
  have hdvd : (2 : Int) ^ 2 ∣ (zStep s).gv - q * r := by
    have hr_eq : r = (zStep s).gv * q - 4 * ((zStep s).gv * q / 4) := by
      rw [hrdef, Int.emod_def]
    show (4 : Int) ∣ _
    exact ⟨-((zStep s).gv * (k * k + k)) + q * ((zStep s).gv * q / 4),
      by rw [hr_eq, hk]; ring⟩
  have hr0 : (0 : Int) ≤ r := by
    rw [hrdef]; exact Int.emod_nonneg _ (by norm_num)
  have hr1 : r < 2 ^ 2 := by
    rw [hrdef, show ((2 : Int) ^ 2) = 4 by norm_num]
    exact Int.emod_lt_of_pos _ (by norm_num)
  have hb := ladder_witness h hz hfvq hqodd hr0 hr1 hdvd
  rw [← hb]
  -- now a finite check over the odd residues mod 4
  have hfv4 : (zStep s).fv = 4 * q := by
    rw [hfvq]; norm_num
  have hq4 : q % 4 = 1 ∨ q % 4 = 3 := by omega
  have hgv4 : (zStep s).gv % 4 = 1 ∨ (zStep s).gv % 4 = 3 := by omega
  have hfv16 : (zStep s).fv % 16 = 4 * (q % 4) := by omega
  have hr4 : r = (zStep s).gv % 4 * (q % 4) % 4 := by
    rw [hrdef, Int.mul_emod]
  rcases hgv4 with hg | hg <;> rcases hq4 with hq' | hq' <;>
    rw [hg, hq'] at hr4 <;> rw [hfv16, hg, hq', hr4] <;> decide

/-! ## Paper-exact statements

The public theorems below restate the appendix results exactly:
`v₂` is `padicValInt 2`, the paper's `u[k]`/`bit_k(u)` is
`bitk u k`, and `y⁻¹ mod 2^j` is the inverse in `ZMod (2^j)`, with `.val`
the representative in `[0, 2^j)`. -/

/-- The paper's `u[k] = bit_k(u)`: bit `k` of `u mod 2^(k+1)`, i.e.
`⌊(u mod 2^(k+1)) / 2^k⌋`. -/
def bitk (u : Int) (k : Nat) : Int := u % 2 ^ (k + 1) / 2 ^ k

/-- Read a window witness through the `ZMod (2^j)` inverse: the paper's
`u·q⁻¹ mod 2^j` equals the witness `r`. -/
private theorem inv_val_eq_of_witness {j : Nat} {u q r : Int}
    (hq : q % 2 = 1) (hr0 : 0 ≤ r) (hr1 : r < 2 ^ j)
    (hdvd : (2 : Int) ^ j ∣ u - q * r) :
    (((u : ZMod (2 ^ j)) * (q : ZMod (2 ^ j))⁻¹).val : Int) = r := by
  haveI : NeZero (2 ^ j) := ⟨by positivity⟩
  -- q is a unit mod 2^j
  have hunit : IsUnit ((q : Int) : ZMod (2 ^ j)) :=
    (ZMod.coe_int_isUnit_iff_isCoprime q _).mpr
      (by push_cast; exact isCoprime_two_pow hq)
  -- the congruence, in ZMod
  have hcong : ((u : Int) : ZMod (2 ^ j)) = ((q * r : Int) : ZMod (2 ^ j)) := by
    rw [ZMod.intCast_eq_intCast_iff_dvd_sub]
    push_cast
    exact dvd_sub_comm.mp hdvd
  rw [hcong]
  push_cast
  rw [mul_comm ((q : Int) : ZMod (2 ^ j)) _, mul_assoc,
    ZMod.mul_inv_of_unit _ hunit, mul_one, ZMod.val_intCast]
  push_cast
  exact Int.emod_eq_of_lt hr0 hr1

/-- The swap class, as stated: at every active step,
`σ = 1 ⟺ f_v' ≡ 2 (mod 4) ⟺ v₂(f_v') = 1`; and if `σ = 0` then
`4 ∣ f_v'`.  (The `4 ∣ f_v'` clause also covers `f_v' = 0`, where the
paper's convention is `v₂(0) = +∞`.) -/
theorem sigma_class (h : Inv p s) (hp : p % 2 = 1) :
    ((0 ≤ s.z ∧ s.g % 2 = 1) ↔ (zStep s).fv ≡ 2 [ZMOD 4]) ∧
    ((0 ≤ s.z ∧ s.g % 2 = 1) ↔ padicValInt 2 (zStep s).fv = 1) ∧
    (¬(0 ≤ s.z ∧ s.g % 2 = 1) → 4 ∣ (zStep s).fv) := by
  have hiff := swap_iff_fv h hp
  have hgv2 := h.gvodd hp
  have hfv2 := h.fveven
  refine ⟨?_, ?_, ?_⟩
  · rw [hiff]
    show _ ↔ (zStep s).fv % 4 = 2 % 4
    omega
  · by_cases hσ : 0 ≤ s.z ∧ s.g % 2 = 1
    · rw [zStep_swap hσ]
      exact iff_of_true hσ
        (padicValInt_of_odd_factor hgv2
          (by show 2 * s.gv = 2 ^ 1 * s.gv; ring)).1
    · refine iff_of_false hσ ?_
      rw [zStep_nonswap hσ]
      show ¬ padicValInt 2 (2 * s.fv) = 1
      by_cases h0 : s.fv = 0
      · rw [h0, mul_zero]
        simp [padicValInt]
      · obtain ⟨k, hk⟩ : (2 : Int) ∣ s.fv := by omega
        intro h1
        have hdvd : ((2 : Nat) : Int) ^ 2 ∣ 2 * s.fv := by
          push_cast; exact ⟨k, by rw [hk]; ring⟩
        have h2 := (padicValInt_dvd_iff 2 (2 * s.fv)).mp hdvd
        have hne : 2 * s.fv ≠ 0 := by omega
        omega
  · intro hσ
    rw [zStep_nonswap hσ]
    show (4 : Int) ∣ 2 * s.fv
    omega

/-- The window invariant's second clause: for `z < 0`, `f_v ≠ 0` and, with
`j = v₂(f_v)` and `q = f_v/2^j` the odd part, there is `0 ≤ r < 2^j` with
`g_v ≡ q·r (mod 2^(j-z))`. -/
theorem window_low (h : Inv p s) (hz : s.z < 0) :
    s.fv ≠ 0 ∧ ∃ r : Int, 0 ≤ r ∧ r < 2 ^ padicValInt 2 s.fv ∧
      s.gv ≡ s.fv / 2 ^ padicValInt 2 s.fv * r
        [ZMOD (2 : Int) ^ (padicValInt 2 s.fv + (-s.z).toNat)] := by
  obtain ⟨j, q, r, hj, hq, hfv, hr0, hr1, hdvd⟩ := h.low hz
  obtain ⟨hv, hqe⟩ := padicValInt_of_odd_factor hq hfv
  have hne : s.fv ≠ 0 := by
    rw [hfv]; exact mul_ne_zero (by positivity) (by omega)
  refine ⟨hne, r, hr0, by rw [hv]; exact hr1, ?_⟩
  rw [hv, hqe]
  have hd : (2 : Int) ^ (j + (-s.z).toNat) ∣ q * r - s.gv := by
    have hval : q * r - s.gv = -(s.gv - q * r) := by ring
    rw [hval]; exact dvd_neg.mpr hdvd
  exact Int.modEq_iff_dvd.mpr hd

/-- The window invariant, as stated over the reachable states of
the walk: clause (i) — if `z ≥ 0` then `2^(z+1) ∣ f_v` — and clause (ii)
as in `window_low`. -/
theorem window_zWalk (hp : p % 2 = 1) (x : Int) (t : Nat) :
    (0 ≤ (zWalk p x t).z →
      (2 : Int) ^ ((zWalk p x t).z.toNat + 1) ∣ (zWalk p x t).fv) ∧
    ((zWalk p x t).z < 0 → (zWalk p x t).fv ≠ 0 ∧
      ∃ r : Int, 0 ≤ r ∧ r < 2 ^ padicValInt 2 (zWalk p x t).fv ∧
        (zWalk p x t).gv ≡
          (zWalk p x t).fv / 2 ^ padicValInt 2 (zWalk p x t).fv * r
          [ZMOD (2 : Int) ^ (padicValInt 2 (zWalk p x t).fv
            + (-(zWalk p x t).z).toNat)]) :=
  ⟨(invariant_zWalk hp x t).high, window_low (invariant_zWalk hp x t)⟩

/-- Shared prelude of the two ladder forms: the canonical window witness of
`ladder_exists`, identified with `(j, q)` through `v₂`, carries the branch
bit as `bit_(j-1)`. -/
private theorem ladder_core (h : Inv p s) (hz : s.z < 0) {j : Nat} {q : Int}
    (hj : j = padicValInt 2 (zStep s).fv) (hq : q = (zStep s).fv / 2 ^ j) :
    ∃ r : Int, q % 2 = 1 ∧ (zStep s).fv = 2 ^ j * q ∧ 0 ≤ r ∧ r < 2 ^ j ∧
      (2 : Int) ^ j ∣ (zStep s).gv - q * r ∧ bitk r (j - 1) = bbit s := by
  obtain ⟨j', q', r, hj2, hq', hfv, hr0, hr1, hdvd⟩ := ladder_exists h hz
  obtain ⟨hv, hqe⟩ := padicValInt_of_odd_factor hq' hfv
  have hjj : j = j' := by rw [hj, hv]
  subst hjj
  have hqq : q = q' := by rw [hq, hqe]
  subst hqq
  refine ⟨r, hq', hfv, hr0, hr1, hdvd, ?_⟩
  have hbit : bitk r (j - 1) = r / 2 ^ (j - 1) := by
    simp only [bitk]
    rw [(by omega : (j - 1) + 1 = j), Int.emod_eq_of_lt hr0 hr1]
  rw [hbit]
  exact ladder_witness h hz hfv hq' hr0 hr1 hdvd

/-- The anchored ladder, first form, exactly as stated: put
`j = v₂(f_v')` and `q = f_v'/2^j`; then
`b = bit_(j-1)(g_v'·q⁻¹ mod 2^j)`. -/
theorem ladder (h : Inv p s) (hz : s.z < 0) {j : Nat} {q : Int}
    (hj : j = padicValInt 2 (zStep s).fv) (hq : q = (zStep s).fv / 2 ^ j) :
    bbit s = bitk
      ((((zStep s).gv : ZMod (2 ^ j)) *
        ((q : Int) : ZMod (2 ^ j))⁻¹).val : Int)
      (j - 1) := by
  obtain ⟨r, hq', hfv, hr0, hr1, hdvd, hbit⟩ := ladder_core h hz hj hq
  rw [inv_val_eq_of_witness hq' hr0 hr1 hdvd]
  exact hbit.symm

/-- The anchored ladder, second form, exactly as stated:
`b = bit_(j-1)(p·(f'·q)⁻¹ mod 2^j)`.  Together with the first form this
also gives finite dependence: the branch bit is a
function of `(j, f' mod 2^j, q mod 2^j)` for each fixed `p`. -/
theorem ladder_det (h : Inv p s) (hp : p % 2 = 1) (hz : s.z < 0)
    {j : Nat} {q : Int}
    (hj : j = padicValInt 2 (zStep s).fv) (hq : q = (zStep s).fv / 2 ^ j) :
    bbit s = bitk
      (((p : ZMod (2 ^ j)) *
        (((zStep s).f * q : Int) : ZMod (2 ^ j))⁻¹).val : Int)
      (j - 1) := by
  obtain ⟨r, hq', hfv, hr0, hr1, hdvd, hbit⟩ := ladder_core h hz hj hq
  have hinv' := h.zStep hp
  -- f'·q is odd
  have hfq : ((zStep s).f * q) % 2 = 1 := by
    rw [Int.mul_emod, hinv'.fodd, hq']
    decide
  -- p ≡ (f'·q)·r (mod 2^j), by the determinant
  have hdvdp : (2 : Int) ^ j ∣ p - (zStep s).f * q * r := by
    have hdvdfv : (2 : Int) ^ j ∣ (zStep s).fv := ⟨q, hfv⟩
    have hval : p - (zStep s).f * q * r =
        (zStep s).f * ((zStep s).gv - q * r) -
          ((zStep s).f * (zStep s).gv - p) := by ring
    have hval2 : (zStep s).f * (zStep s).gv - p = (zStep s).g * (zStep s).fv := by
      linear_combination hinv'.det
    rw [hval, hval2]
    exact dvd_sub (hdvd.mul_left _) (hdvdfv.mul_left _)
  rw [inv_val_eq_of_witness hfq hr0 hr1 hdvdp]
  exact hbit.symm

/-- Finite dependence, stated as factorization through two
`j`-bit residues: for each fixed odd `p` and depth `j`, ONE function of
`ZMod (2^j) × ZMod (2^j)` recovers the branch bit of every negative-counter
nonswap at that depth from `f' mod 2^j` and `q mod 2^j` (the decoder is quantified
before the state, so it cannot depend on the step). -/
theorem finite_dependence (hp : p % 2 = 1) (j : Nat) :
    ∃ decode : ZMod (2 ^ j) → ZMod (2 ^ j) → Int,
      ∀ s : ZState, Inv p s → s.z < 0 →
        j = padicValInt 2 (zStep s).fv →
        bbit s = decode ((zStep s).f : ZMod (2 ^ j))
          (((zStep s).fv / 2 ^ j : Int) : ZMod (2 ^ j)) := by
  refine ⟨fun fbar qbar =>
    bitk ((((p : ZMod (2 ^ j)) * (fbar * qbar)⁻¹).val : Nat) : Int) (j - 1),
    fun s h hz hj => ?_⟩
  simpa only [Int.cast_mul] using ladder_det h hp hz hj rfl

/-- The depth-two law, exactly as stated: at every negative-counter nonswap
with `j = v₂(f_v') = 2`, the branch bit is `b = g_v'[1] ⊕ f_v'[3]`
(exclusive-or of bits = sum mod 2). -/
theorem j2_law (h : Inv p s) (hp : p % 2 = 1) (hz : s.z < 0)
    (h2 : padicValInt 2 (zStep s).fv = 2) :
    bbit s = (bitk (zStep s).gv 1 + bitk (zStep s).fv 3) % 2 := by
  -- v₂ = 2 forces f_v' ≡ 4 (mod 8)
  obtain ⟨j', q', r, hj2, hq', hfv, hr0, hr1, hdvd⟩ := ladder_exists h hz
  have hv := (padicValInt_of_odd_factor hq' hfv).1
  have hj4 : j' = 2 := by omega
  subst hj4
  have h4 : (zStep s).fv % 8 = 4 := by
    rw [hfv, (by norm_num : (2 : Int) ^ 2 = 4)]
    omega
  rw [j2_mod h hp hz h4]
  simp only [bitk]
  norm_num
