/-
# The final theorems and the axiom audit

Composition of the certificate (`example590`), reachability
(`g_reaches_zero`), the number theory, and the `delta`/`z` bridge
(`zWalk_hdRun`). The `#print axioms` commands verify the trust base of
every result in the paper's divstep appendix: the standard Lean axioms alone for the
walk invariants and the branch-recovery theory, plus the `native_decide`
axiom of the certificate for the step-bound theorems. See README.md.
-/
import Hddivsteps590.Invariant
import Hddivsteps590.NumberTheory
import Hddivsteps590.Certificate
import Hddivsteps590.Recovery

open Hddivsteps

/-! ## The delivered inverse -/

/-- For odd prime `f ≤ M*` and `g < f`, after
590 hddivsteps: if `g = 0` then `d = 0` (the seed starts frozen and `d` —
the paper's `f_v` — remains 0); if `0 < g` then `|f_end| = 1` and
`d_end · f_end · g ≡ 1 (mod f)`. -/
theorem hddivsteps590_prime_inverse (f g : Int)
    (hodd : f % 2 = 1) (hprime : Prime f) (hM : f ≤ Mstar) (hgf : g < f) :
    (g = 0 → (hdRun 590 f g).d = 0) ∧
    (0 < g → (hdRun 590 f g).f.natAbs = 1 ∧
      (hdRun 590 f g).d * (hdRun 590 f g).f * g ≡ 1 [ZMOD f]) := by
  constructor
  · rintro rfl
    exact d_hdRun_of_g_zero 590 f
  · intro hgpos
    exact inverse_hdRun 590 f g hodd (prime_coprime f g hprime hgpos hgf)
      (g_reaches_zero 590 Mstar f g hodd hM hgpos.le hgf.le example590)

/-! ## The step bound on the frozen z-form walk -/

/-- On the frozen `z`-form walk of the
branch-recovery development (via the bridge `zWalk_hdRun`): every odd
`f ≤ M*` and `0 ≤ g ≤ f` reaches `g = 0` within 590 steps. -/
theorem zWalk590_bound (f g : Int) (hodd : f % 2 = 1) (hM : f ≤ Mstar)
    (hg0 : 0 ≤ g) (hgf : g ≤ f) : (zWalk f g 590).g = 0 := by
  rw [(zWalk_hdRun f g 590).2.1]
  exact g_reaches_zero 590 Mstar f g hodd hM hg0 hgf example590

/-- The 256-bit prime `p = 2^256 - 2^32 - 977`. -/
def fieldPrime : Int := 2 ^ 256 - 2 ^ 32 - 977

/-- The certified threshold strictly exceeds `p`. -/
theorem fieldPrime_lt_Mstar : fieldPrime < Mstar := by
  norm_num [fieldPrime, Mstar]

/-- The concrete step bound: every canonical input `0 ≤ x < p` reaches
`g = 0` in the frozen `z`-form walk within 590 active slots. -/
theorem zWalk590 (x : Int) (hx0 : 0 ≤ x) (hxp : x < fieldPrime) :
    (zWalk fieldPrime x 590).g = 0 :=
  zWalk590_bound fieldPrime x (by norm_num [fieldPrime])
    fieldPrime_lt_Mstar.le hx0 hxp.le

/-! ## Axiom audit — one line per paper claim -/

-- Step bound (native_decide expected)
#print axioms hddivsteps590_prime_inverse
#print axioms zWalk590_bound
#print axioms zWalk590
-- standard axioms only from here on: the threshold and the delta/z bridge
#print axioms fieldPrime_lt_Mstar
#print axioms zWalk_hdRun
-- Determinant, parity, and scaling invariants
#print axioms det_zWalk
#print axioms parity_zWalk
#print axioms scale_zWalk
#print axioms endpoint_zWalk
#print axioms inverse_zWalk
#print axioms inverse_zWalk_unscaled
-- Local branch-recovery classes
#print axioms sigma_class
#print axioms bbit_of_swap
#print axioms bbit_of_ramp
#print axioms bbit_of_freeze
-- Window invariant
#print axioms invariant_zWalk
#print axioms window_zWalk
#print axioms window_low
-- Anchored ladder, finite dependence, and depth-two law
#print axioms ladder
#print axioms ladder_det
#print axioms finite_dependence
#print axioms j2_law
