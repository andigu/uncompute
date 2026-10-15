import Dqi.Residual
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Topology.Algebra.Module.FiniteDimension

/-!
# The DQI objective as a bounded self-adjoint operator

`Dqi.Shell` defines the shifted objective `Dqi.shellOp` as a bare function on
`EuclideanSpace ℂ (Pattern m K)`, and `Dqi.Residual` proves it is linear.  This
file packages it as a continuous linear map `Dqi.shellOpL` that is self-adjoint
with operator norm at most `m` — the exact shape of operator to which the
abstract degradation bound of `Dqi.Degradation` applies, which is what lets that
bound be instantiated at the real DQI objective.

The structure behind both facts is per-site: each single-site operator
`Dqi.siteOp` is an *orthogonal projection* — idempotent (`Dqi.siteOp_siteOp`,
by character orthogonality) and self-adjoint (`Dqi.siteOp_inner`, because its
matrix kernel at `(y', y)` is `(1/q) ∑ v ∈ F, χ ((y' i - y i) v)`, whose
conjugate is the kernel at `(y, y')`).  Each site therefore contributes the
reflection `2 P - 1` about a rank-restricted projection, an isometry
(`Dqi.norm_two_smul_siteOp_sub`), and the objective is the sum of the `m` site
reflections, so its operator norm is at most `m` (`Dqi.norm_shellOpL_le`).
-/

namespace Dqi

section ShellOp

variable {m : ℕ} {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- **The single-site operator is idempotent.**  Composing the site operator
with itself sums `χ (b (w - v))` over the intermediate letter `b`, which by
character orthogonality collapses the double sum over `F × F` to its diagonal
and cancels one factor of `1/q`. -/
theorem siteOp_siteOp (χ : AddChar K ℂ) (hχ : χ ≠ 1) (F : Finset K) (i : Fin m)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    siteOp χ F i (siteOp χ F i ψ) = siteOp χ F i ψ := by
  apply PiLp.ext
  intro y
  have hq : ((Fintype.card K : ℝ)⁻¹ : ℂ) * (Fintype.card K : ℂ) = 1 := by
    push_cast
    exact inv_mul_cancel₀ (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)
  have hinner : ∀ a : K,
      siteOp χ F i ψ (Function.update y i a)
        = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ∑ b : K, (∑ w ∈ F, χ ((a - b) * w)) * ψ (Function.update y i b) := by
    intro a
    rw [siteOp_apply]
    simp only [Function.update_self, Function.update_idem]
  have key : ∀ b : K,
      ∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) * (∑ w ∈ F, χ ((a - b) * w))
        = (Fintype.card K : ℂ) * ∑ v ∈ F, χ ((y i - b) * v) := by
    intro b
    calc ∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) * (∑ w ∈ F, χ ((a - b) * w))
        = ∑ a : K, ∑ v ∈ F, ∑ w ∈ F, χ ((w - v) * a) * χ (y i * v - b * w) := by
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [Finset.sum_mul_sum]
          refine Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun w _ => ?_
          rw [← AddChar.map_add_eq_mul, ← AddChar.map_add_eq_mul]
          congr 1
          ring
      _ = ∑ v ∈ F, ∑ a : K, ∑ w ∈ F, χ ((w - v) * a) * χ (y i * v - b * w) :=
          Finset.sum_comm
      _ = ∑ v ∈ F, ∑ w ∈ F, ∑ a : K, χ ((w - v) * a) * χ (y i * v - b * w) :=
          Finset.sum_congr rfl fun v _ => Finset.sum_comm
      _ = ∑ v ∈ F, ∑ w ∈ F, (∑ a : K, χ ((w - v) * a)) * χ (y i * v - b * w) :=
          Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun w _ => by
            rw [Finset.sum_mul]
      _ = ∑ v ∈ F, ∑ w ∈ F,
            (if w = v then (Fintype.card K : ℂ) * χ (y i * v - b * w) else 0) := by
          refine Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun w _ => ?_
          rw [sum_char_mul χ hχ (w - v)]
          simp only [sub_eq_zero, ite_mul, zero_mul]
      _ = (Fintype.card K : ℂ) * ∑ v ∈ F, χ ((y i - b) * v) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun v hv => ?_
          rw [Finset.sum_ite_eq' F v, if_pos hv, sub_mul]
  have hstep : ∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) *
        (((Fintype.card K : ℝ)⁻¹ : ℂ) *
          ∑ b : K, (∑ w ∈ F, χ ((a - b) * w)) * ψ (Function.update y i b))
      = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
          ∑ b : K, (∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) * (∑ w ∈ F, χ ((a - b) * w))) *
            ψ (Function.update y i b) := by
    calc ∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) *
          (((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ∑ b : K, (∑ w ∈ F, χ ((a - b) * w)) * ψ (Function.update y i b))
        = ∑ a : K, ∑ b : K, ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ((∑ v ∈ F, χ ((y i - a) * v)) *
              ((∑ w ∈ F, χ ((a - b) * w)) * ψ (Function.update y i b))) := by
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [Finset.mul_sum, Finset.mul_sum]
          refine Finset.sum_congr rfl fun b _ => ?_
          ring
      _ = ∑ b : K, ∑ a : K, ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ((∑ v ∈ F, χ ((y i - a) * v)) *
              ((∑ w ∈ F, χ ((a - b) * w)) * ψ (Function.update y i b))) :=
          Finset.sum_comm
      _ = ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ∑ b : K, (∑ a : K, (∑ v ∈ F, χ ((y i - a) * v)) *
              (∑ w ∈ F, χ ((a - b) * w))) * ψ (Function.update y i b) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun b _ => ?_
          rw [Finset.sum_mul, Finset.mul_sum]
          refine Finset.sum_congr rfl fun a _ => ?_
          ring
  rw [siteOp_apply, siteOp_apply]
  simp only [hinner]
  rw [hstep]
  simp only [key]
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [← mul_assoc, ← mul_assoc, hq, one_mul]

omit [DecidableEq K] in
/-- **The single-site operator is self-adjoint.**  Its matrix entry at
`(y', y)` is `(1/q) ∑ v ∈ F, χ ((y' i - y i) v)`, whose conjugate is the entry
at `(y, y')`; the double sum over `(y, a)` is reindexed by the involution
`(y, a) ↦ (update y i a, y i)`. -/
theorem siteOp_inner (χ : AddChar K ℂ) (F : Finset K) (i : Fin m)
    (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    inner ℂ (siteOp χ F i ψ) φ = inner ℂ ψ (siteOp χ F i φ) := by
  classical
  have hL : inner ℂ (siteOp χ F i ψ) φ
      = ∑ p : Pattern m K × K, ((Fintype.card K : ℝ)⁻¹ : ℂ) *
          ((∑ v ∈ F, χ ((p.2 - p.1 i) * v)) *
            (φ p.1 * (starRingEnd ℂ) (ψ (Function.update p.1 i p.2)))) := by
    have hy : ∀ y : Pattern m K, inner ℂ (siteOp χ F i ψ y) (φ y)
        = ∑ a : K, ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ((∑ v ∈ F, χ ((a - y i) * v)) *
              (φ y * (starRingEnd ℂ) (ψ (Function.update y i a)))) := by
      intro y
      rw [RCLike.inner_apply, siteOp_apply, map_mul, map_inv₀, Complex.conj_ofReal,
        map_sum, Finset.mul_sum, Finset.mul_sum]
      refine Finset.sum_congr rfl fun a _ => ?_
      have hc : (starRingEnd ℂ) (∑ v ∈ F, χ ((y i - a) * v))
          = ∑ v ∈ F, χ ((a - y i) * v) := by
        rw [map_sum]
        refine Finset.sum_congr rfl fun v _ => ?_
        rw [← AddChar.map_neg_eq_conj, neg_mul_eq_neg_mul, neg_sub]
      rw [map_mul, hc]
      ring
    rw [PiLp.inner_apply, Finset.sum_congr rfl fun y _ => hy y]
    exact (Fintype.sum_prod_type (fun p : Pattern m K × K =>
      ((Fintype.card K : ℝ)⁻¹ : ℂ) * ((∑ v ∈ F, χ ((p.2 - p.1 i) * v)) *
        (φ p.1 * (starRingEnd ℂ) (ψ (Function.update p.1 i p.2)))))).symm
  have hR : inner ℂ ψ (siteOp χ F i φ)
      = ∑ p : Pattern m K × K, ((Fintype.card K : ℝ)⁻¹ : ℂ) *
          ((∑ v ∈ F, χ ((p.1 i - p.2) * v)) *
            (φ (Function.update p.1 i p.2) * (starRingEnd ℂ) (ψ p.1))) := by
    have hy : ∀ y : Pattern m K, inner ℂ (ψ y) (siteOp χ F i φ y)
        = ∑ a : K, ((Fintype.card K : ℝ)⁻¹ : ℂ) *
            ((∑ v ∈ F, χ ((y i - a) * v)) *
              (φ (Function.update y i a) * (starRingEnd ℂ) (ψ y))) := by
      intro y
      rw [RCLike.inner_apply, siteOp_apply, Finset.mul_sum, Finset.sum_mul]
      exact Finset.sum_congr rfl fun a _ => by ring
    rw [PiLp.inner_apply, Finset.sum_congr rfl fun y _ => hy y]
    exact (Fintype.sum_prod_type (fun p : Pattern m K × K =>
      ((Fintype.card K : ℝ)⁻¹ : ℂ) * ((∑ v ∈ F, χ ((p.1 i - p.2) * v)) *
        (φ (Function.update p.1 i p.2) * (starRingEnd ℂ) (ψ p.1))))).symm
  have hinv : Function.Involutive
      (fun p : Pattern m K × K => (Function.update p.1 i p.2, p.1 i)) := by
    intro p
    simp only [Function.update_self, Function.update_idem, Function.update_eq_self,
      Prod.mk.eta]
  have hswap := hinv.bijective.sum_comp (fun p : Pattern m K × K =>
    ((Fintype.card K : ℝ)⁻¹ : ℂ) * ((∑ v ∈ F, χ ((p.1 i - p.2) * v)) *
      (φ (Function.update p.1 i p.2) * (starRingEnd ℂ) (ψ p.1))))
  simp only [] at hswap
  rw [hL, hR, ← hswap]
  refine Finset.sum_congr rfl fun p _ => ?_
  simp only [Function.update_self, Function.update_idem, Function.update_eq_self]

omit [DecidableEq K] in
/-- The inner product against the objective's output, resolved site by site. -/
theorem inner_shellOp_left (χ : AddChar K ℂ) (Fs : Fin m → Finset K)
    (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    inner ℂ (shellOp χ Fs ψ) φ
      = ∑ i : Fin m, (2 * inner ℂ (siteOp χ (Fs i) i ψ) φ - inner ℂ ψ φ) := by
  have h1 : ∀ y : Pattern m K, inner ℂ (shellOp χ Fs ψ y) (φ y)
      = ∑ i : Fin m, (2 * (φ y * (starRingEnd ℂ) (siteOp χ (Fs i) i ψ y))
          - φ y * (starRingEnd ℂ) (ψ y)) := by
    intro y
    rw [RCLike.inner_apply, shellOp_apply, map_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [map_sub, map_mul, map_ofNat]
    ring
  rw [PiLp.inner_apply, Finset.sum_congr rfl fun y _ => h1 y, Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.sum_sub_distrib, PiLp.inner_apply, PiLp.inner_apply]
  simp only [RCLike.inner_apply]
  rw [Finset.mul_sum]

omit [DecidableEq K] in
/-- The inner product with the objective's output, resolved site by site. -/
theorem inner_shellOp_right (χ : AddChar K ℂ) (Fs : Fin m → Finset K)
    (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    inner ℂ ψ (shellOp χ Fs φ)
      = ∑ i : Fin m, (2 * inner ℂ ψ (siteOp χ (Fs i) i φ) - inner ℂ ψ φ) := by
  have h1 : ∀ y : Pattern m K, inner ℂ (ψ y) (shellOp χ Fs φ y)
      = ∑ i : Fin m, (2 * (siteOp χ (Fs i) i φ y * (starRingEnd ℂ) (ψ y))
          - φ y * (starRingEnd ℂ) (ψ y)) := by
    intro y
    rw [RCLike.inner_apply, shellOp_apply, Finset.sum_mul]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [PiLp.inner_apply, Finset.sum_congr rfl fun y _ => h1 y, Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.sum_sub_distrib, PiLp.inner_apply, PiLp.inner_apply]
  simp only [RCLike.inner_apply]
  rw [Finset.mul_sum]

/-- Each site's shifted contribution `2 P - 1` is a reflection about the
projection `P = siteOp`: idempotence and self-adjointness give
`⟪(2P - 1) ψ, (2P - 1) ψ⟫ = ⟪ψ, ψ⟫`, so it preserves the norm. -/
theorem norm_two_smul_siteOp_sub (χ : AddChar K ℂ) (hχ : χ ≠ 1) (F : Finset K)
    (i : Fin m) (ψ : EuclideanSpace ℂ (Pattern m K)) :
    ‖(2 : ℂ) • siteOp χ F i ψ - ψ‖ = ‖ψ‖ := by
  have hPP : inner ℂ (siteOp χ F i ψ) (siteOp χ F i ψ)
      = inner ℂ ψ (siteOp χ F i ψ) := by
    rw [siteOp_inner χ F i ψ (siteOp χ F i ψ), siteOp_siteOp χ hχ]
  have hexp : inner ℂ ((2 : ℂ) • siteOp χ F i ψ - ψ) ((2 : ℂ) • siteOp χ F i ψ - ψ)
      = inner ℂ ψ ψ := by
    simp only [inner_sub_left, inner_sub_right, inner_smul_left, inner_smul_right,
      map_ofNat]
    rw [hPP, siteOp_inner χ F i ψ ψ]
    ring
  have hsq : ‖(2 : ℂ) • siteOp χ F i ψ - ψ‖ ^ 2 = ‖ψ‖ ^ 2 := by
    have h := hexp
    rw [inner_self_eq_norm_sq_to_K, inner_self_eq_norm_sq_to_K] at h
    exact_mod_cast h
  have h := congrArg Real.sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at h

omit [DecidableEq K] in
/-- The shifted objective as a sum of site reflections, at the vector level. -/
theorem shellOp_eq_sum_reflections (χ : AddChar K ℂ) (Fs : Fin m → Finset K)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    shellOp χ Fs ψ = ∑ i : Fin m, ((2 : ℂ) • siteOp χ (Fs i) i ψ - ψ) := by
  apply PiLp.ext
  intro y
  rw [shellOp_apply]
  simp only [WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_sub, WithLp.ofLp_smul,
    Pi.sub_apply, Pi.smul_apply, smul_eq_mul]

omit [DecidableEq K] in
/-- The DQI weight-shell objective `Dqi.shellOp`, packaged as a continuous
linear operator on `EuclideanSpace ℂ (Pattern m K)`. -/
noncomputable def shellOpL (χ : AddChar K ℂ) (Fs : Fin m → Finset K) :
    EuclideanSpace ℂ (Pattern m K) →L[ℂ] EuclideanSpace ℂ (Pattern m K) :=
  LinearMap.toContinuousLinearMap
    { toFun := shellOp χ Fs
      map_add' := shellOp_add χ Fs
      map_smul' := shellOp_smul χ Fs }

omit [DecidableEq K] in
@[simp] theorem shellOpL_apply (χ : AddChar K ℂ) (Fs : Fin m → Finset K)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    shellOpL χ Fs ψ = shellOp χ Fs ψ :=
  rfl

omit [DecidableEq K] in
/-- **The objective operator is self-adjoint**: it is a real combination of the
self-adjoint site projections. -/
theorem isSelfAdjoint_shellOpL (χ : AddChar K ℂ) (Fs : Fin m → Finset K) :
    IsSelfAdjoint (shellOpL χ Fs) := by
  refine LinearMap.IsSymmetric.isSelfAdjoint ?_
  intro ψ φ
  simp only [ContinuousLinearMap.coe_coe, shellOpL_apply]
  rw [inner_shellOp_left, inner_shellOp_right]
  exact Finset.sum_congr rfl fun j _ => by rw [siteOp_inner]

/-- **The objective operator has norm at most `m`**: it is the sum of `m` site
reflections, each of norm at most `1`. -/
theorem norm_shellOpL_le (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K) :
    ‖shellOpL χ Fs‖ ≤ m := by
  refine ContinuousLinearMap.opNorm_le_bound _ (Nat.cast_nonneg m) fun ψ => ?_
  rw [shellOpL_apply, shellOp_eq_sum_reflections]
  calc ‖∑ i : Fin m, ((2 : ℂ) • siteOp χ (Fs i) i ψ - ψ)‖
      ≤ ∑ i : Fin m, ‖(2 : ℂ) • siteOp χ (Fs i) i ψ - ψ‖ := norm_sum_le _ _
    _ = ∑ _i : Fin m, ‖ψ‖ := Finset.sum_congr rfl fun i _ =>
        norm_two_smul_siteOp_sub χ hχ (Fs i) i ψ
    _ = (m : ℝ) * ‖ψ‖ := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end ShellOp

end Dqi
