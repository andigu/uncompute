import Dqi.Shell

/-!
# The residual decomposition of the DQI objective

`Dqi.Shell` proves that the shifted objective acts tridiagonally on the
orthonormal weight-layer states.  This file turns that single-layer identity
into the *residual decomposition* of the full DQI state: writing the state as
a weight profile `w` over the layers `0, …, l` of the decodable range, the
objective acts on it as the truncated tridiagonal shell matrix `Amat` inside
the span of those layers, and produces **exactly one** escaping direction —
the single next weight layer `l + 1`, entered through the top raising
coefficient `√((l+1)(m-l))` weighted by `w l`.

Concretely:

* `dqiState` is the profile-weighted combination of layer states, and
  `shellOp_dqiState` computes its image under the objective: the tridiagonal
  action `Amat · w` on layers `0, …, l` plus the isolated `layerState (l+1)`
  component.
* `lamF` is the exact finite-size objective value: for a unit profile,
  `inner (dqiState …) (shellOp … (dqiState …)) = lamF` — the quadratic form
  of the shell matrix (`inner_dqiState_shellOp`).
* `shellOp_dqiState_residual` exhibits the residual
  `shellOp Ψ - lamF • Ψ` as an explicit combination of the layers
  `0, …, l` plus a single `layerState (l+1)` term, whose norm is exactly
  `sigmaEdge` (`norm_residual_edge`).

The mathematical point: the *only* direction through which the objective
leaks out of the decodable subspace is the single layer `l + 1`.  An error
supported on decodable syndromes lives in the span of the layers `0, …, l`
and therefore cannot see any of the (orthogonal) leakage — which is why the
spectral analysis of the DQI objective reduces to the `(l+1) × (l+1)`
tridiagonal matrix `Amat` up to the one edge term `sigmaEdge`.
-/

namespace Dqi

/-! ### The truncated tridiagonal shell matrix and its scalar invariants -/

/-- Entry `(j,k)` of the truncated tridiagonal shell matrix. -/
noncomputable def Amat (m : ℕ) (r q : ℕ) (j k : ℕ) : ℝ :=
  if j = k then (j : ℝ) * dd r q
  else if k = j + 1 then Real.sqrt (((j : ℝ) + 1) * ((m : ℝ) - j))
  else if j = k + 1 then Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k))
  else 0

/-- The shell matrix is symmetric. -/
theorem Amat_symm (m r q j k : ℕ) : Amat m r q j k = Amat m r q k j := by
  rcases eq_or_ne j k with h | h
  · subst h; rfl
  rcases eq_or_ne k (j + 1) with h1 | h1
  · have hj : j ≠ k + 1 := by omega
    simp only [Amat]
    rw [if_neg h, if_pos h1, if_neg (Ne.symm h), if_neg hj, if_pos h1]
  rcases eq_or_ne j (k + 1) with h2 | h2
  · have hk : k ≠ j + 1 := by omega
    simp only [Amat]
    rw [if_neg h, if_neg hk, if_pos h2, if_neg (Ne.symm h), if_pos h2]
  · simp only [Amat]
    rw [if_neg h, if_neg h1, if_neg h2, if_neg (Ne.symm h), if_neg h2, if_neg h1]

/-- The three bands of a shell-matrix entry, split into separate indicator
terms — the form in which a row or column sum can be evaluated band by
band. -/
theorem Amat_decomp (m r q j k : ℕ) :
    Amat m r q j k
      = (if j = k then (k : ℝ) * dd r q else 0)
        + (if j = k + 1 then Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) else 0)
        + (if k = j + 1 then Real.sqrt (((j : ℝ) + 1) * ((m : ℝ) - j)) else 0) := by
  unfold Amat
  split_ifs <;> first | omega | (subst_vars; ring)

/-- The exact finite-size DQI objective value of a weight profile `w` on the
layers `0, …, l`: the mean term `m f̄` plus `2 c₁` times the quadratic form of
the truncated shell matrix. -/
noncomputable def lamF (m r q : ℕ) (w : ℕ → ℝ) (l : ℕ) : ℝ :=
  (m : ℝ) * fbar r q + 2 * cc1rq r q *
    ∑ j ∈ Finset.range (l + 1), ∑ k ∈ Finset.range (l + 1), w j * Amat m r q j k * w k

/-- The norm of the single escaping component: the top raising coefficient
`√((l+1)(m-l))` weighted by the edge amplitude `|w l|`. -/
noncomputable def sigmaEdge (m r q : ℕ) (w : ℕ → ℝ) (l : ℕ) : ℝ :=
  2 * cc1rq r q * Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) * |w l|

/-! ### Regrouping linear combinations of layer vectors

The two lemmas below are pure module algebra over an abstract family
`L : ℕ → V` of vectors — no property of the layer states is used.  They carry
all the index bookkeeping of the residual decomposition. -/

section Regroup

variable {V : Type*} [AddCommGroup V] [Module ℂ V]

/-- Two real-weighted combinations of the same vectors merge into one. -/
theorem residual_regroup (L : ℕ → V) (n : ℕ) (u v : ℕ → ℝ) (a b : ℝ) :
    ((a : ℝ) : ℂ) • (∑ j ∈ Finset.range n, ((u j : ℝ) : ℂ) • L j)
      + ((b : ℝ) : ℂ) • (∑ j ∈ Finset.range n, ((v j : ℝ) : ℂ) • L j)
      = ∑ j ∈ Finset.range n, ((a * u j + b * v j : ℝ) : ℂ) • L j := by
  rw [Finset.smul_sum, Finset.smul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [smul_smul, smul_smul, ← Complex.ofReal_mul, ← Complex.ofReal_mul, ← add_smul,
    ← Complex.ofReal_add]

/-- **Regrouping the tridiagonal action by output layer.**  Summing the
three-term (lower/diagonal/upper) images of the layers `0, …, l` and
collecting by the *output* index turns the lowering term of layer `k` into
the upper band `Amat j (j+1)` at `j = k - 1` and the raising term into the
lower band `Amat j (j-1)` at `j = k + 1`; every contribution lands back in
`range (l+1)` except the single raising term of the top layer `k = l`. -/
theorem tridiag_regroup (m r q : ℕ) (w : ℕ → ℝ) (l : ℕ) (L : ℕ → V) :
    ∑ k ∈ Finset.range (l + 1), ((w k : ℝ) : ℂ) •
        (((Real.sqrt ((k : ℝ) * ((m : ℝ) - k + 1)) : ℝ) : ℂ) • L (k - 1)
          + (((k : ℝ) * dd r q : ℝ) : ℂ) • L k
          + ((Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) : ℝ) : ℂ) • L (k + 1))
      = ∑ j ∈ Finset.range (l + 1),
          ((∑ k ∈ Finset.range (l + 1), Amat m r q j k * w k : ℝ) : ℂ) • L j
        + ((w l * Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ) • L (l + 1) := by
  classical
  have key : ∀ k ∈ Finset.range (l + 1),
      (∑ j ∈ Finset.range (l + 1), ((Amat m r q j k * w k : ℝ) : ℂ) • L j)
        + (if k = l then
            ((w l * Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ) • L (l + 1)
          else 0)
        = ((w k : ℝ) : ℂ) •
            (((Real.sqrt ((k : ℝ) * ((m : ℝ) - k + 1)) : ℝ) : ℂ) • L (k - 1)
              + (((k : ℝ) * dd r q : ℝ) : ℂ) • L k
              + ((Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) : ℝ) : ℂ) • L (k + 1)) := by
    intro k hk
    have hkl : k ≤ l := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
    -- split the `Amat` column into its three bands
    have hsplit : (∑ j ∈ Finset.range (l + 1), ((Amat m r q j k * w k : ℝ) : ℂ) • L j)
        = (∑ j ∈ Finset.range (l + 1),
            (if j = k then (((k : ℝ) * dd r q * w k : ℝ) : ℂ) • L j else 0))
          + (∑ j ∈ Finset.range (l + 1),
              (if j = k + 1 then
                ((Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) * w k : ℝ) : ℂ) • L j
              else 0))
          + (∑ j ∈ Finset.range (l + 1),
              (if k = j + 1 then
                ((Real.sqrt (((j : ℝ) + 1) * ((m : ℝ) - j)) * w k : ℝ) : ℂ) • L j
              else 0)) := by
      rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [Amat_decomp, add_mul, ite_mul, zero_mul, Complex.ofReal_add,
        apply_ite Complex.ofReal, Complex.ofReal_zero, add_smul, ite_smul, zero_smul]
    -- the diagonal band lands on layer `k`
    have hband1 : (∑ j ∈ Finset.range (l + 1),
        (if j = k then (((k : ℝ) * dd r q * w k : ℝ) : ℂ) • L j else 0))
        = (((k : ℝ) * dd r q * w k : ℝ) : ℂ) • L k := by
      rw [Finset.sum_ite_eq' (Finset.range (l + 1)) k, if_pos hk]
    -- the lower band lands on layer `k + 1`, escaping `range (l+1)` iff `k = l`
    have hband2 : (∑ j ∈ Finset.range (l + 1),
        (if j = k + 1 then
          ((Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) * w k : ℝ) : ℂ) • L j else 0))
        = if k + 1 ∈ Finset.range (l + 1) then
            ((Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) * w k : ℝ) : ℂ) • L (k + 1)
          else 0 :=
      Finset.sum_ite_eq' (Finset.range (l + 1)) (k + 1) _
    -- the upper band lands on layer `k - 1` (with vanishing coefficient at `k = 0`)
    have hband3 : (∑ j ∈ Finset.range (l + 1),
        (if k = j + 1 then
          ((Real.sqrt (((j : ℝ) + 1) * ((m : ℝ) - j)) * w k : ℝ) : ℂ) • L j else 0))
        = ((w k : ℝ) : ℂ) •
            (((Real.sqrt ((k : ℝ) * ((m : ℝ) - k + 1)) : ℝ) : ℂ) • L (k - 1)) := by
      by_cases hk0 : k = 0
      · subst hk0
        rw [Finset.sum_eq_zero fun j _ => if_neg (by omega)]
        simp
      · have hswap : ∀ j ∈ Finset.range (l + 1),
            (if k = j + 1 then
              ((Real.sqrt (((j : ℝ) + 1) * ((m : ℝ) - j)) * w k : ℝ) : ℂ) • L j else 0)
              = (if j = k - 1 then
                  ((Real.sqrt (((j : ℝ) + 1) * ((m : ℝ) - j)) * w k : ℝ) : ℂ) • L j
                else 0) :=
          fun j _ => if_congr (by omega) rfl rfl
        rw [Finset.sum_congr rfl hswap, Finset.sum_ite_eq' (Finset.range (l + 1)) (k - 1),
          if_pos (Finset.mem_range.mpr (by omega))]
        have hs : Real.sqrt ((((k - 1 : ℕ) : ℝ) + 1) * ((m : ℝ) - ((k - 1 : ℕ) : ℝ)))
            = Real.sqrt ((k : ℝ) * ((m : ℝ) - (k : ℝ) + 1)) := by
          have hc : (((k - 1 : ℕ) : ℝ)) = (k : ℝ) - 1 := by
            rw [Nat.cast_sub (by omega), Nat.cast_one]
          rw [hc]
          congr 1
          ring
        rw [hs, smul_smul, ← Complex.ofReal_mul,
          mul_comm (w k) (Real.sqrt ((k : ℝ) * ((m : ℝ) - (k : ℝ) + 1)))]
    rw [hsplit, hband1, hband2, hband3]
    rcases eq_or_lt_of_le hkl with rfl | hkl'
    · rw [if_neg (by simp), if_pos rfl]
      push_cast
      module
    · rw [if_pos (Finset.mem_range.mpr (by omega)), if_neg (by omega)]
      push_cast
      module
  rw [Finset.sum_congr rfl fun k hk => (key k hk).symm, Finset.sum_add_distrib,
    Finset.sum_ite_eq' (Finset.range (l + 1)) l, if_pos (Finset.self_mem_range_succ l),
    Finset.sum_comm]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Complex.ofReal_sum, Finset.sum_smul]

end Regroup

/-! ### Linearity of the site and shell operators -/

section Residual

variable {m : ℕ} {K : Type*} [Field K] [Fintype K] [DecidableEq K]

omit [DecidableEq K] in
theorem siteOp_add (χ : AddChar K ℂ) (F : Finset K) (i : Fin m)
    (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    siteOp χ F i (ψ + φ) = siteOp χ F i ψ + siteOp χ F i φ := by
  apply PiLp.ext
  intro y
  simp only [PiLp.add_apply, siteOp_apply, mul_add, Finset.sum_add_distrib]

omit [DecidableEq K] in
theorem siteOp_smul (χ : AddChar K ℂ) (F : Finset K) (i : Fin m) (c : ℂ)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    siteOp χ F i (c • ψ) = c • siteOp χ F i ψ := by
  apply PiLp.ext
  intro y
  simp only [PiLp.smul_apply, siteOp_apply, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  ring

omit [DecidableEq K] in
theorem shellOp_add (χ : AddChar K ℂ) (Fs : Fin m → Finset K)
    (ψ φ : EuclideanSpace ℂ (Pattern m K)) :
    shellOp χ Fs (ψ + φ) = shellOp χ Fs ψ + shellOp χ Fs φ := by
  apply PiLp.ext
  intro y
  simp only [PiLp.add_apply, shellOp_apply, siteOp_add]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

omit [DecidableEq K] in
theorem shellOp_smul (χ : AddChar K ℂ) (Fs : Fin m → Finset K) (c : ℂ)
    (ψ : EuclideanSpace ℂ (Pattern m K)) :
    shellOp χ Fs (c • ψ) = c • shellOp χ Fs ψ := by
  apply PiLp.ext
  intro y
  simp only [PiLp.smul_apply, shellOp_apply, siteOp_smul, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

omit [DecidableEq K] in
theorem shellOp_zero (χ : AddChar K ℂ) (Fs : Fin m → Finset K) :
    shellOp χ Fs (0 : EuclideanSpace ℂ (Pattern m K)) = 0 := by
  apply PiLp.ext
  intro y
  simp [shellOp_apply, siteOp_apply]

omit [DecidableEq K] in
theorem shellOp_sum (χ : AddChar K ℂ) (Fs : Fin m → Finset K) (s : Finset ℕ)
    (f : ℕ → EuclideanSpace ℂ (Pattern m K)) :
    shellOp χ Fs (∑ k ∈ s, f k) = ∑ k ∈ s, shellOp χ Fs (f k) := by
  classical
  induction s using Finset.cons_induction with
  | empty => simpa using shellOp_zero χ Fs
  | cons a s ha ih => rw [Finset.sum_cons, Finset.sum_cons, shellOp_add, ih]

/-! ### The DQI state -/

/-- The DQI state of a real weight profile `w` truncated at layer `l`: the
`w`-weighted combination of the weight-layer states `0, …, l`. -/
noncomputable def dqiState (g : Fin m → K → ℂ) (w : ℕ → ℝ) (l : ℕ) :
    EuclideanSpace ℂ (Pattern m K) :=
  ∑ k ∈ Finset.range (l + 1), ((w k : ℝ) : ℂ) • layerState g k

omit [Fintype K] in
theorem dqiState_def (g : Fin m → K → ℂ) (w : ℕ → ℝ) (l : ℕ) :
    dqiState g w l = ∑ k ∈ Finset.range (l + 1), ((w k : ℝ) : ℂ) • layerState g k :=
  rfl

/-! ### The action of the objective on the DQI state -/

/-- **The objective on the DQI state: tridiagonal action plus one escaping
term.**  Applying the shifted objective to the profile-weighted DQI state
gives the mean drift `m f̄` times the state itself, plus `2 c₁` times the
truncated tridiagonal action `Amat · w` on the layers `0, …, l` — plus
**exactly one** term escaping the decodable range: the raising term of the
top layer, `w l √((l+1)(m-l)) • layerState (l+1)`. -/
theorem shellOp_dqiState (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (r : ℕ) (hcard : ∀ i, (Fs i).card = r) (hr : 0 < r) (hrq : r < Fintype.card K)
    (w : ℕ → ℝ) (l : ℕ) (hl : l < m) :
    shellOp χ Fs (dqiState (amp χ Fs) w l) =
      (((m : ℝ) * fbar r (Fintype.card K) : ℝ) : ℂ) • dqiState (amp χ Fs) w l
      + ((2 * cc1rq r (Fintype.card K) : ℝ) : ℂ) •
          (∑ j ∈ Finset.range (l + 1),
              ((∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k : ℝ) : ℂ)
                • layerState (amp χ Fs) j
            + ((w l * Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
                • layerState (amp χ Fs) (l + 1)) := by
  have hstep : ∀ k ∈ Finset.range (l + 1),
      shellOp χ Fs (((w k : ℝ) : ℂ) • layerState (amp χ Fs) k)
        = (((m : ℝ) * fbar r (Fintype.card K) : ℝ) : ℂ) •
            (((w k : ℝ) : ℂ) • layerState (amp χ Fs) k)
          + ((2 * cc1rq r (Fintype.card K) : ℝ) : ℂ) •
              (((w k : ℝ) : ℂ) •
                (((Real.sqrt ((k : ℝ) * ((m : ℝ) - k + 1)) : ℝ) : ℂ) •
                    layerState (amp χ Fs) (k - 1)
                  + (((k : ℝ) * dd r (Fintype.card K) : ℝ) : ℂ) • layerState (amp χ Fs) k
                  + ((Real.sqrt (((k : ℝ) + 1) * ((m : ℝ) - k)) : ℝ) : ℂ) •
                      layerState (amp χ Fs) (k + 1))) := by
    intro k hk
    have hkm : k ≤ m :=
      le_of_lt (lt_of_le_of_lt (Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)) hl)
    rw [shellOp_smul, shellOp_layerState χ hχ Fs r hcard hr hrq k hkm, smul_add,
      smul_comm ((w k : ℝ) : ℂ) (((m : ℝ) * fbar r (Fintype.card K) : ℝ) : ℂ),
      smul_comm ((w k : ℝ) : ℂ) ((2 * cc1rq r (Fintype.card K) : ℝ) : ℂ)]
  conv_lhs => rw [dqiState_def, shellOp_sum χ Fs, Finset.sum_congr rfl hstep]
  rw [Finset.sum_add_distrib, ← Finset.smul_sum, ← Finset.smul_sum,
    tridiag_regroup m r (Fintype.card K) w l (layerState (amp χ Fs)), ← dqiState_def]

/-! ### The quadratic form: the exact finite-size objective value -/

/-- The overlap of the DQI state with a single layer state reads off the
profile: it is `w j` for `j` in the decodable range and `0` above it. -/
theorem inner_dqiState_layerState (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (hr : ∀ i, 0 < (Fs i).card) (hrq : ∀ i, (Fs i).card < Fintype.card K)
    (w : ℕ → ℝ) (l : ℕ) (hl : l ≤ m) {j : ℕ} (hj : j ≤ m) :
    inner ℂ (dqiState (amp χ Fs) w l) (layerState (amp χ Fs) j)
      = if j ≤ l then ((w j : ℝ) : ℂ) else 0 := by
  rw [dqiState_def, sum_inner]
  have hterm : ∀ k ∈ Finset.range (l + 1),
      inner ℂ (((w k : ℝ) : ℂ) • layerState (amp χ Fs) k) (layerState (amp χ Fs) j)
        = if k = j then ((w k : ℝ) : ℂ) else 0 := by
    intro k hk
    have hkm : k ≤ m := le_trans (Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)) hl
    rw [inner_smul_left, layerState_amp_inner χ hχ Fs hr hrq hkm hj, Complex.conj_ofReal,
      mul_ite, mul_one, mul_zero]
  rw [Finset.sum_congr rfl hterm, Finset.sum_ite_eq' (Finset.range (l + 1)) j]
  simp only [Finset.mem_range, Nat.lt_succ_iff]

/-- Each layer state of an actual DQI instance is a unit vector. -/
theorem norm_layerState_amp (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (hr : ∀ i, 0 < (Fs i).card) (hrq : ∀ i, (Fs i).card < Fintype.card K)
    {k : ℕ} (hk : k ≤ m) : ‖layerState (amp χ Fs) k‖ = 1 := by
  have hinner := layerState_amp_inner χ hχ Fs hr hrq hk hk
  rw [if_pos rfl, inner_self_eq_norm_sq_to_K] at hinner
  have hsq : ‖layerState (amp χ Fs) k‖ ^ 2 = 1 := by
    refine Complex.ofReal_injective ?_
    push_cast
    exact hinner
  rw [← Real.sqrt_one, ← hsq, Real.sqrt_sq (norm_nonneg _)]

/-- **The exact finite-size DQI objective value.**  For a unit weight
profile, the expectation of the shifted objective in the DQI state is the
quadratic form `lamF` of the truncated tridiagonal shell matrix: the single
escaping `layerState (l+1)` component is orthogonal to every decodable layer
and drops out of the expectation entirely. -/
theorem inner_dqiState_shellOp (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (r : ℕ) (hcard : ∀ i, (Fs i).card = r) (hr : 0 < r) (hrq : r < Fintype.card K)
    (w : ℕ → ℝ) (l : ℕ) (hl : l < m)
    (hw : ∑ k ∈ Finset.range (l + 1), w k ^ 2 = 1) :
    inner ℂ (dqiState (amp χ Fs) w l) (shellOp χ Fs (dqiState (amp χ Fs) w l))
      = ((lamF m r (Fintype.card K) w l : ℝ) : ℂ) := by
  have hr0 : ∀ i, 0 < (Fs i).card := fun i => by rw [hcard i]; exact hr
  have hrq0 : ∀ i, (Fs i).card < Fintype.card K := fun i => by rw [hcard i]; exact hrq
  have hlm : l ≤ m := le_of_lt hl
  have hPsiL : ∀ j : ℕ, j ≤ m →
      inner ℂ (dqiState (amp χ Fs) w l) (layerState (amp χ Fs) j)
        = if j ≤ l then ((w j : ℝ) : ℂ) else 0 :=
    fun _ hj => inner_dqiState_layerState χ hχ Fs hr0 hrq0 w l hlm hj
  have hPP : inner ℂ (dqiState (amp χ Fs) w l) (dqiState (amp χ Fs) w l) = 1 := by
    nth_rewrite 2 [dqiState_def]
    rw [inner_sum]
    have hterm : ∀ k ∈ Finset.range (l + 1),
        inner ℂ (dqiState (amp χ Fs) w l) (((w k : ℝ) : ℂ) • layerState (amp χ Fs) k)
          = ((w k ^ 2 : ℝ) : ℂ) := by
      intro k hk
      have hkl : k ≤ l := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
      rw [inner_smul_right, hPsiL k (le_trans hkl hlm), if_pos hkl, ← Complex.ofReal_mul, sq]
    rw [Finset.sum_congr rfl hterm]
    exact_mod_cast congrArg Complex.ofReal hw
  rw [shellOp_dqiState χ hχ Fs r hcard hr hrq w l hl, inner_add_right, inner_smul_right,
    inner_smul_right, inner_add_right, inner_sum, inner_smul_right, hPP, mul_one,
    hPsiL (l + 1) (by omega), if_neg (by omega), mul_zero, add_zero]
  have hPD : ∀ j ∈ Finset.range (l + 1),
      inner ℂ (dqiState (amp χ Fs) w l)
          (((∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k : ℝ) : ℂ)
            • layerState (amp χ Fs) j)
        = ((∑ k ∈ Finset.range (l + 1), w j * Amat m r (Fintype.card K) j k * w k
            : ℝ) : ℂ) := by
    intro j hj
    have hjl : j ≤ l := Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)
    rw [inner_smul_right, hPsiL j (le_trans hjl hlm), if_pos hjl, ← Complex.ofReal_mul]
    congr 1
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [Finset.sum_congr rfl hPD]
  unfold lamF
  push_cast
  ring

/-! ### The residual split: exactly one escaping direction -/

/-- **The residual decomposition.**  Subtracting the objective value `lamF`
times the DQI state from its image under the objective leaves an explicit
combination of the decodable layers `0, …, l` plus a **single** component
along `layerState (l+1)` — no layer above `l + 1` appears.  This is the
precise sense in which the DQI objective leaks out of the decodable subspace
through exactly one direction. -/
theorem shellOp_dqiState_residual (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (r : ℕ) (hcard : ∀ i, (Fs i).card = r) (hr : 0 < r) (hrq : r < Fintype.card K)
    (w : ℕ → ℝ) (l : ℕ) (hl : l < m) :
    shellOp χ Fs (dqiState (amp χ Fs) w l)
        - ((lamF m r (Fintype.card K) w l : ℝ) : ℂ) • dqiState (amp χ Fs) w l
      = (∑ j ∈ Finset.range (l + 1),
          ((2 * cc1rq r (Fintype.card K) *
                (∑ k ∈ Finset.range (l + 1), Amat m r (Fintype.card K) j k * w k)
              + ((m : ℝ) * fbar r (Fintype.card K) - lamF m r (Fintype.card K) w l) * w j
            : ℝ) : ℂ) • layerState (amp χ Fs) j)
        + ((2 * cc1rq r (Fintype.card K) * w l *
              Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
            • layerState (amp χ Fs) (l + 1) := by
  rw [shellOp_dqiState χ hχ Fs r hcard hr hrq w l hl, dqiState_def, smul_add]
  rw [show ∀ A B C D : EuclideanSpace ℂ (Pattern m K), A + (B + C) - D = B + (A - D) + C
    from fun _ _ _ _ => by abel]
  rw [← sub_smul, ← Complex.ofReal_sub, residual_regroup, smul_smul, ← Complex.ofReal_mul,
    ← mul_assoc]

/-- The escaping component of the residual has norm exactly `sigmaEdge`. -/
theorem norm_residual_edge (χ : AddChar K ℂ) (hχ : χ ≠ 1) (Fs : Fin m → Finset K)
    (r : ℕ) (hcard : ∀ i, (Fs i).card = r) (hr : 0 < r) (hrq : r < Fintype.card K)
    (w : ℕ → ℝ) (l : ℕ) (hl : l < m) :
    ‖((2 * cc1rq r (Fintype.card K) * w l *
          Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) : ℝ) : ℂ)
        • layerState (amp χ Fs) (l + 1)‖ = sigmaEdge m r (Fintype.card K) w l := by
  have hr0 : ∀ i, 0 < (Fs i).card := fun i => by rw [hcard i]; exact hr
  have hrq0 : ∀ i, (Fs i).card < Fintype.card K := fun i => by rw [hcard i]; exact hrq
  rw [norm_smul, Complex.norm_real, Real.norm_eq_abs,
    norm_layerState_amp χ hχ Fs hr0 hrq0 (show l + 1 ≤ m by omega), mul_one]
  unfold sigmaEdge
  have hc1 : (0 : ℝ) ≤ cc1rq r (Fintype.card K) :=
    div_nonneg (Real.sqrt_nonneg _) (Nat.cast_nonneg _)
  have hs : (0 : ℝ) ≤ Real.sqrt (((l : ℝ) + 1) * ((m : ℝ) - l)) := Real.sqrt_nonneg _
  rw [abs_mul, abs_mul, abs_mul, abs_two, abs_of_nonneg hc1, abs_of_nonneg hs]
  ring

end Residual

end Dqi
