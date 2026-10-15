import Dqi.Channel
import Dqi.Perturbation

/-!
# The degradation bound, assembled

This file combines the master perturbation bound with the reachable-residual
machinery into the statement the paper actually uses: a bound on how far the
measured expectation can move, in terms of the failed probability mass.

The two forms correspond to the two cases of the appendix's reachable-residual
proposition.

* `degradation_of_blind` — the perturbation is confined to coordinates where the
  escaping component vanishes.  This is the sign and leakage channels: they keep
  a decodable syndrome, and `Embed.disjoint_decodableSyn_edgeSyn` says the
  escaping weight layer lives entirely on other syndromes.  The bound sees only
  `sdec`, never the (much larger) escaping component.

* `degradation_of_split` — the perturbation *can* reach the escaping component,
  but only through its restriction to the coordinates the perturbation
  populates.  This is the relocation channel, and `erel` is what
  `Counting.norm_restr_le_of_forall` bounds.

Both are second order in the perturbation except for the cross term, which is
where eigenstate protection lives: the loss is `O(mu)` rather than the generic
`O(sqrt mu)` precisely because `sdec` is small, not because `mu` is.

## What `mu` is, exactly

`norm_sub_le_two_sqrt_mass` fixes the interpretation: the `mu` these theorems
consume is `mass Ψpre F`, the amplitude mass of the **pre-oracle** state on the
failure set.  That is what the triangle argument gives, since the perturbation is
`V Ψpre - U Ψpre` and both images have norm `‖restr F Ψpre‖`.

The validation protocol, and the appendix's definition `mu = ∑_{y ∈ F} |a_y|^2`,
instead use the **post-oracle** amplitudes.  The two agree because the ideal
oracle `U` only applies phases, so it preserves every branch magnitude — the
appendix states this outright when introducing the master bound.

That identification is supplied by `Bridge.norm_restr_imageBlock`, which proves

  `‖restr (imageBlock lbl F) (V Ψ)‖ = sqrt (mass Ψ F)`

— the amplitude mass on the physical registers a set of branches reaches equals
the mass those branches carried beforehand.  So the two readings of `mu` are the
same number, and the identification is a theorem rather than a remark.

One hypothesis of that theorem is worth noticing: it needs `lbl` to be
**injective on a set containing the failure set and the state's support**
(`Set.InjOn`).  This is not bookkeeping.  If two branches of the state were
sent to one syndrome-and-ancilla label their amplitudes would interfere there
and the masses would genuinely differ.  For the ideal oracle the label is the
syndrome together with a clean ancilla, the relevant set is the decodable set,
and the injectivity is literally `syndrome_injOn_decodable` (`Dqi/Basic.lean`)
— the same decoding-distance condition that makes the DQI state normalized in
the first place.  (Global injectivity would be false for `m > n` by pigeonhole;
the on-set form is both what holds and all the proof needs.)
-/

namespace Dqi

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

section MassBound

variable {ι' : Type*} [Fintype ι']

/-- A vector splits into its restriction to a set and to the complement. -/
theorem restr_add_restr_compl (F : Finset ι) (Ψ : EuclideanSpace ℂ ι) :
    restr F Ψ + restr Fᶜ Ψ = Ψ := by
  apply PiLp.ext
  intro i
  by_cases h : i ∈ F <;> simp [h]

/-- **The failure mass controls the perturbation norm.**

If the ideal oracle `U` and the implementation `V` are linear maps that are
norm-preserving on vectors supported inside `S`, and agree on every
`S`-supported vector vanishing on the failure set `F`, then for any
`S`-supported state `Ψ` the perturbation `V Ψ - U Ψ` has norm at most
`2 * sqrt (mass Ψ F)`.  (The support-relative form matters: the end-to-end
theorem can only assert isometry and agreement on the decodable subspace.)

This is the step that ties the theorem to the experiment: `mass Ψ F` is exactly
the quantity the fuzz test estimates, by sampling a branch from `‖Ψ y‖ ^ 2` and
asking whether it lies in `F`.  It is the *only* place the failure mass enters
the degradation bound. -/
theorem norm_sub_le_two_sqrt_mass
    (U V : EuclideanSpace ℂ ι →ₗ[ℂ] EuclideanSpace ℂ ι') {S : Set ι}
    (hU : ∀ x : EuclideanSpace ℂ ι, (∀ i, x i ≠ 0 → i ∈ S) → ‖U x‖ = ‖x‖)
    (hV : ∀ x : EuclideanSpace ℂ ι, (∀ i, x i ≠ 0 → i ∈ S) → ‖V x‖ = ‖x‖)
    (F : Finset ι)
    (hagree : ∀ x : EuclideanSpace ℂ ι, (∀ i, x i ≠ 0 → i ∈ S) →
      (∀ i ∈ F, x i = 0) → V x = U x)
    (Ψ : EuclideanSpace ℂ ι) (hΨS : ∀ i, Ψ i ≠ 0 → i ∈ S) :
    ‖V Ψ - U Ψ‖ ≤ 2 * Real.sqrt (mass Ψ F) := by
  -- Restriction only zeroes coordinates, so it preserves `S`-supportedness.
  have hres : ∀ (G : Finset ι) (i : ι), (restr G Ψ) i ≠ 0 → i ∈ S := by
    intro G i hi
    refine hΨS i fun h0 => hi ?_
    simp [restr_apply, h0]
  -- Off the failure set the two oracles agree, so only the restriction survives.
  have hcompl : ∀ i ∈ F, (restr Fᶜ Ψ) i = 0 := by
    intro i hi
    simp [restr_apply, Finset.mem_compl, hi]
  have hsplit : V Ψ - U Ψ = V (restr F Ψ) - U (restr F Ψ) := by
    conv_lhs => rw [← restr_add_restr_compl F Ψ]
    rw [map_add, map_add, hagree _ (hres Fᶜ) hcompl]
    abel
  rw [hsplit]
  have hnorm : ‖restr F Ψ‖ = Real.sqrt (mass Ψ F) := norm_restr_eq_sqrt_mass Ψ F
  refine norm_le_two_sqrt _ _ _ ?_ ?_
  · rw [hV _ (hres F), hnorm]
  · rw [hU _ (hres F), hnorm]

end MassBound

/-- The ideal expectation of a self-adjoint operator in a state. -/
noncomputable def expct (M : EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι)
    (Ψ : EuclideanSpace ℂ ι) : ℝ :=
  (inner ℂ Ψ (M Ψ)).re

/-- A perturbation has zero overlap with anything vanishing on its support. -/
theorem inner_eq_zero_of_avoids (D X : EuclideanSpace ℂ ι)
    (h : ∀ i ∈ supp D, X i = 0) : inner ℂ D X = (0 : ℂ) := by
  rw [inner_eq_inner_restr_supp D X, restr_eq_zero _ _ h, inner_zero_right]

omit [DecidableEq ι] in
/-- The master bound with the cross term supplied directly rather than as
`‖D‖ * rho`.  A channel decomposition produces a bound of this shape — a sum of
per-channel contributions with different mass factors — which does not factor
through a single `rho`. -/
theorem expectation_perturbation_bound_of_cross
    (M : EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι) (hM : IsSelfAdjoint M)
    (Ψ D : EuclideanSpace ℂ ι) (hΨ : ‖Ψ‖ = 1) (hΨD : ‖Ψ + D‖ = 1)
    (mBound C : ℝ) (hmB : ‖M‖ ≤ mBound)
    (hC : ‖inner ℂ D (M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ)‖ ≤ C) :
    |expct M (Ψ + D) - expct M Ψ| ≤ (|expct M Ψ| + mBound) * ‖D‖ ^ 2 + 2 * C := by
  have hC0 : 0 ≤ C := le_trans (norm_nonneg _) hC
  rcases eq_or_ne D 0 with hD | hD
  · subst hD
    simpa using by positivity
  · have hpos : 0 < ‖D‖ := norm_pos_iff.mpr hD
    have hcross : ‖inner ℂ D (M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ)‖ ≤ ‖D‖ * (C / ‖D‖) := by
      rwa [mul_div_cancel₀ _ hpos.ne']
    have h := expectation_perturbation_bound M hM Ψ D mBound (C / ‖D‖) hΨ hΨD hmB hcross
    rwa [show 2 * ‖D‖ * (C / ‖D‖) = 2 * C by field_simp] at h

/-- **Degradation when the perturbation is blind to the escaping component.**

The residual `M Ψ - λ Ψ` splits as `X₁ + X₂`; the perturbation `D` populates only
coordinates where `X₂` vanishes.  Then the expectation moves by at most
`(|λ| + mBound) * 4 * mu + 4 * sqrt mu * sdec`, where `sdec` bounds `X₁` alone.

This is the sign-and-leakage case: such errors keep a decodable syndrome, so they
cannot see the escaping weight layer at all. -/
theorem degradation_of_blind
    (M : EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι) (hM : IsSelfAdjoint M)
    (Ψ D : EuclideanSpace ℂ ι) (hΨ : ‖Ψ‖ = 1) (hΨD : ‖Ψ + D‖ = 1)
    (mBound : ℝ) (hmB : ‖M‖ ≤ mBound)
    (X₁ X₂ : EuclideanSpace ℂ ι)
    (hsplit : M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ = X₁ + X₂)
    (sdec : ℝ) (hX₁ : ‖X₁‖ ≤ sdec)
    (havoid : ∀ i ∈ supp D, X₂ i = 0)
    (mu : ℝ) (hmu : 0 ≤ mu) (hD : ‖D‖ ≤ 2 * Real.sqrt mu) :
    |expct M (Ψ + D) - expct M Ψ|
      ≤ (|expct M Ψ| + mBound) * (4 * mu) + 4 * Real.sqrt mu * sdec := by
  have hsdec : 0 ≤ sdec := le_trans (norm_nonneg _) hX₁
  have hDnn : 0 ≤ ‖D‖ := norm_nonneg _
  -- The cross term sees only `X₁`, because `D` is blind to `X₂`.
  have hcross : ‖inner ℂ D (M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ)‖ ≤ ‖D‖ * sdec := by
    rw [hsplit]
    exact le_trans (norm_inner_le_of_avoids D X₁ X₂ havoid)
      (mul_le_mul_of_nonneg_left hX₁ hDnn)
  have hmain := expectation_perturbation_bound M hM Ψ D mBound sdec hΨ hΨD hmB hcross
  -- Convert `‖D‖` into the failure mass.
  have hsq : ‖D‖ ^ 2 ≤ 4 * mu := by
    have hsqrt : Real.sqrt mu ^ 2 = mu := Real.sq_sqrt hmu
    calc ‖D‖ ^ 2 ≤ (2 * Real.sqrt mu) ^ 2 := by
          exact pow_le_pow_left₀ hDnn hD 2
      _ = 4 * mu := by rw [mul_pow]; rw [hsqrt]; ring
  have hlin : 2 * ‖D‖ * sdec ≤ 4 * Real.sqrt mu * sdec := by
    have : 2 * ‖D‖ ≤ 4 * Real.sqrt mu := by linarith
    exact mul_le_mul_of_nonneg_right this hsdec
  have hcoef : 0 ≤ |expct M Ψ| + mBound :=
    add_nonneg (abs_nonneg _) (le_trans (norm_nonneg _) hmB)
  calc |expct M (Ψ + D) - expct M Ψ|
      ≤ (|expct M Ψ| + mBound) * ‖D‖ ^ 2 + 2 * ‖D‖ * sdec := hmain
    _ ≤ (|expct M Ψ| + mBound) * (4 * mu) + 4 * Real.sqrt mu * sdec := by
        gcongr

/-- **Degradation when the perturbation can reach the escaping component.**

Same as `degradation_of_blind` except that `D` may populate coordinates where
`X₂` is nonzero; the cross term then picks up `erel`, which bounds only the
*restriction* of `X₂` to those coordinates — not `X₂` itself.

This is the relocation case.  The gap between `‖X₂‖` and `erel` is the whole
content of the relocation counting: an error reaching few labels sees only a
sliver of the escaping layer. -/
theorem degradation_of_split
    (M : EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι) (hM : IsSelfAdjoint M)
    (Ψ D : EuclideanSpace ℂ ι) (hΨ : ‖Ψ‖ = 1) (hΨD : ‖Ψ + D‖ = 1)
    (mBound : ℝ) (hmB : ‖M‖ ≤ mBound)
    (X₁ X₂ : EuclideanSpace ℂ ι)
    (hsplit : M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ = X₁ + X₂)
    (sdec erel : ℝ) (hX₁ : ‖X₁‖ ≤ sdec)
    (hreach : ‖restr (supp D) X₂‖ ≤ erel)
    (mu : ℝ) (hmu : 0 ≤ mu) (hD : ‖D‖ ≤ 2 * Real.sqrt mu) :
    |expct M (Ψ + D) - expct M Ψ|
      ≤ (|expct M Ψ| + mBound) * (4 * mu) + 4 * Real.sqrt mu * (sdec + erel) := by
  have hsdec : 0 ≤ sdec := le_trans (norm_nonneg _) hX₁
  have herel : 0 ≤ erel := le_trans (norm_nonneg _) hreach
  have hDnn : 0 ≤ ‖D‖ := norm_nonneg _
  have hcross : ‖inner ℂ D (M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ)‖ ≤ ‖D‖ * (sdec + erel) := by
    rw [hsplit]
    refine le_trans (norm_inner_le_of_split D X₁ X₂ erel hreach) ?_
    exact mul_le_mul_of_nonneg_left (by linarith) hDnn
  have hmain := expectation_perturbation_bound M hM Ψ D mBound (sdec + erel) hΨ hΨD hmB hcross
  have hsq : ‖D‖ ^ 2 ≤ 4 * mu := by
    have hsqrt : Real.sqrt mu ^ 2 = mu := Real.sq_sqrt hmu
    calc ‖D‖ ^ 2 ≤ (2 * Real.sqrt mu) ^ 2 := pow_le_pow_left₀ hDnn hD 2
      _ = 4 * mu := by rw [mul_pow]; rw [hsqrt]; ring
  have hlin : 2 * ‖D‖ * (sdec + erel) ≤ 4 * Real.sqrt mu * (sdec + erel) := by
    have h2 : 2 * ‖D‖ ≤ 4 * Real.sqrt mu := by linarith
    exact mul_le_mul_of_nonneg_right h2 (by linarith)
  have hcoef : 0 ≤ |expct M Ψ| + mBound :=
    add_nonneg (abs_nonneg _) (le_trans (norm_nonneg _) hmB)
  calc |expct M (Ψ + D) - expct M Ψ|
      ≤ (|expct M Ψ| + mBound) * ‖D‖ ^ 2 + 2 * ‖D‖ * (sdec + erel) := hmain
    _ ≤ (|expct M Ψ| + mBound) * (4 * mu) + 4 * Real.sqrt mu * (sdec + erel) := by
        gcongr

/-- **Channel-resolved degradation.**

The perturbation splits into three channels.  Sign errors (`Ds`) and leakage
(`Dl`) keep a decodable syndrome, so they are blind to the escaping component
`X₂` and contribute only through `X₁`.  Only relocation (`Dr`) can reach `X₂`,
and it reaches only the part sitting on the labels it populates.

The point of the split is the mass factors.  The decodable term carries the
total mass `mu`; the escaping term carries only `mucl`, the mass of the
relocations whose remaining registers are clean.  For the task instances
those differ by two orders of magnitude, so resolving the channels is what keeps
the relocation allowance small.

Note the relocation term is `2 * sqrt mucl * erel`, not `4 * sqrt mucl * erel`:
the ideal image is orthogonal to `X₂`, so only the implemented image survives and
the triangle inequality is never spent. -/
theorem degradation_channels
    (M : EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι) (hM : IsSelfAdjoint M)
    (Ψ D Ds Dl Dr : EuclideanSpace ℂ ι) (hΨ : ‖Ψ‖ = 1) (hΨD : ‖Ψ + D‖ = 1)
    (hchan : D = Ds + Dl + Dr)
    (mBound : ℝ) (hmB : ‖M‖ ≤ mBound)
    (X₁ X₂ : EuclideanSpace ℂ ι)
    (hsplit : M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ = X₁ + X₂)
    (sdec erel : ℝ) (hX₁ : ‖X₁‖ ≤ sdec)
    (hblindS : ∀ i ∈ supp Ds, X₂ i = 0)
    (hblindL : ∀ i ∈ supp Dl, X₂ i = 0)
    (hreach : ‖restr (supp Dr) X₂‖ ≤ erel)
    (mu mucl : ℝ) (hmu : 0 ≤ mu)
    (hDn : ‖D‖ ≤ 2 * Real.sqrt mu) (hDrn : ‖Dr‖ ≤ Real.sqrt mucl) :
    |expct M (Ψ + D) - expct M Ψ|
      ≤ (|expct M Ψ| + mBound) * (4 * mu)
        + 4 * Real.sqrt mu * sdec + 2 * Real.sqrt mucl * erel := by
  have hsdec : 0 ≤ sdec := le_trans (norm_nonneg _) hX₁
  have herel : 0 ≤ erel := le_trans (norm_nonneg _) hreach
  have hDnn : 0 ≤ ‖D‖ := norm_nonneg _
  have hmucl : 0 ≤ Real.sqrt mucl := Real.sqrt_nonneg _
  -- The decodable component is seen by every channel.
  have hX1 : ‖inner ℂ D X₁‖ ≤ ‖D‖ * sdec :=
    le_trans (norm_inner_le_norm (𝕜 := ℂ) D X₁) (mul_le_mul_of_nonneg_left hX₁ hDnn)
  -- The escaping component is seen only by relocation.
  have hX2 : ‖inner ℂ D X₂‖ ≤ Real.sqrt mucl * erel := by
    have hzS : inner ℂ Ds X₂ = (0 : ℂ) := inner_eq_zero_of_avoids Ds X₂ hblindS
    have hzL : inner ℂ Dl X₂ = (0 : ℂ) := inner_eq_zero_of_avoids Dl X₂ hblindL
    have hDr : ‖inner ℂ Dr X₂‖ ≤ ‖Dr‖ * erel := norm_inner_le_of_restr Dr X₂ erel hreach
    have : inner ℂ D X₂ = inner ℂ Dr X₂ := by
      rw [hchan, inner_add_left, inner_add_left, hzS, hzL, zero_add, zero_add]
    rw [this]
    exact le_trans hDr (mul_le_mul_of_nonneg_right hDrn herel)
  have hcross : ‖inner ℂ D (M Ψ - ((expct M Ψ : ℝ) : ℂ) • Ψ)‖
      ≤ ‖D‖ * sdec + Real.sqrt mucl * erel := by
    rw [hsplit, inner_add_right]
    exact le_trans (norm_add_le _ _) (add_le_add hX1 hX2)
  have hmain := expectation_perturbation_bound_of_cross M hM Ψ D hΨ hΨD mBound
    (‖D‖ * sdec + Real.sqrt mucl * erel) hmB hcross
  have hsq : ‖D‖ ^ 2 ≤ 4 * mu := by
    have hsqrt : Real.sqrt mu ^ 2 = mu := Real.sq_sqrt hmu
    calc ‖D‖ ^ 2 ≤ (2 * Real.sqrt mu) ^ 2 := pow_le_pow_left₀ hDnn hDn 2
      _ = 4 * mu := by rw [mul_pow]; rw [hsqrt]; ring
  have hcoef : 0 ≤ |expct M Ψ| + mBound :=
    add_nonneg (abs_nonneg _) (le_trans (norm_nonneg _) hmB)
  have hDs : ‖D‖ * sdec ≤ 2 * Real.sqrt mu * sdec :=
    mul_le_mul_of_nonneg_right hDn hsdec
  calc |expct M (Ψ + D) - expct M Ψ|
      ≤ (|expct M Ψ| + mBound) * ‖D‖ ^ 2
        + 2 * (‖D‖ * sdec + Real.sqrt mucl * erel) := hmain
    _ ≤ (|expct M Ψ| + mBound) * (4 * mu)
        + 4 * Real.sqrt mu * sdec + 2 * Real.sqrt mucl * erel := by
        have := mul_le_mul_of_nonneg_left hsq hcoef
        linarith

end Dqi
