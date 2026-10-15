/-
# Soundness of the chain hull

`chainHull_covers`: every point of a sorted vertex set lies in the
convex hull of the vertices that `chainHull` keeps. This is the only
place where the computation discards vertices, so it carries the
geometric burden of the certificate.

Coq counterpart: `convexHull_sound` via `under_addUpperPoint` and
`over_addLowerPoint` (hddivsteps_convexhull.v). Here one lemma set
serves both chains: the direction bit `dir` selects the kept
orientation (`keepOf`) and the sign (`sg`) that orients the
y-inequalities.
-/
import Mathlib.Analysis.Convex.Hull
import Mathlib.Tactic.FieldSimp
import Hddivsteps590.Semantics

namespace Hddivsteps

/-! ### The value-lexicographic order (what `ddCmp` decides) -/

/-- Strict value-lexicographic order on `DD`. -/
private def dlt (p q : DD) : Prop :=
  p.toPt.1 < q.toPt.1 ∨ (p.toPt.1 = q.toPt.1 ∧ p.toPt.2 < q.toPt.2)

private theorem dlt_fst {p q : DD} (h : dlt p q) : p.toPt.1 ≤ q.toPt.1 :=
  h.elim le_of_lt fun h => le_of_eq h.1

private theorem dlt_snd {p q : DD} (h : dlt p q) (he : p.toPt.1 = q.toPt.1) :
    p.toPt.2 < q.toPt.2 :=
  h.elim (fun h => absurd he h.ne) And.right

private theorem dlt_ne {p q : DD} (h : dlt p q) : p.toPt ≠ q.toPt := fun he =>
  h.elim (fun h => h.ne (congrArg Prod.fst he))
    fun h => h.2.ne (congrArg Prod.snd he)

/-- Below a strict value bound, the `Eq` dedup branch of `addPoint` cannot fire. -/
private theorem ddCmp_ne_eq {p q : DD} (h : dlt q p) :
    ¬(ddCmp p q == Ordering.eq) = true := fun hb =>
  dlt_ne h ((ddCmp_eq_iff p q).1 (beq_iff_eq.1 hb)).symm

/-- The chain accumulator invariant: strictly descending in value order. -/
private def Desc (l : List DD) : Prop := l.Pairwise fun u v => dlt v u

/-! ### Structural facts about `addPoint` -/

private theorem addPoint_one (keep : Ordering) (p q : DD) :
    addPoint keep p [q] = if ddCmp p q == .eq then [p] else [p, q] := rfl

private theorem addPoint_two (keep : Ordering) (p q r : DD) (l : List DD) :
    addPoint keep p (q :: r :: l) =
      if orientation p q r == keep then p :: q :: r :: l
      else addPoint keep p (r :: l) := rfl

/-- If everything in `m` is strictly below `p` and `m` is nonempty, the
insertion keeps at least one point besides the new head `p`. -/
private theorem addPoint_cons_cons (keep : Ordering) (p : DD) :
    ∀ m : List DD, m ≠ [] → (∀ z ∈ m, dlt z p) →
      ∃ z t, addPoint keep p m = p :: z :: t := by
  intro m
  induction m with
  | nil => exact fun h _ => absurd rfl h
  | cons q t ih =>
    intro _ hlt
    cases t with
    | nil =>
      exact ⟨q, [], by rw [addPoint_one, if_neg (ddCmp_ne_eq (hlt q (by simp)))]⟩
    | cons r l =>
      by_cases hk : (orientation p q r == keep) = true
      · exact ⟨q, r :: l, by rw [addPoint_two, if_pos hk]⟩
      · obtain ⟨z, t', hz⟩ := ih (by simp) fun z hz => hlt z (List.mem_cons_of_mem q hz)
        exact ⟨z, t', by rw [addPoint_two, if_neg hk]; exact hz⟩

/-! ### The polyline and the covering predicate -/

/-- `x` lies on the polyline through the value points of `l` (on the segment
between some adjacent pair of vertices). -/
private inductive OnPath (x : ℚ × ℚ) : List DD → Prop
  | tl (a : DD) {l : List DD} : OnPath x l → OnPath x (a :: l)
  | seg {a b : DD} (l : List DD) (cf : ℚ) (hc0 : 0 ≤ cf) (hc1 : cf ≤ 1)
      (hx1 : x.1 = cf * a.toPt.1 + (1 - cf) * b.toPt.1)
      (hx2 : x.2 = cf * a.toPt.2 + (1 - cf) * b.toPt.2) : OnPath x (a :: b :: l)

/-- Every polyline point is in the convex hull of the vertex values. -/
private theorem onPath_mem_hull {x : ℚ × ℚ} {l : List DD} (h : OnPath x l) :
    x ∈ convexHull ℚ (DD.toPt '' {q | q ∈ l}) := by
  induction h with
  | tl a hp ih =>
    exact convexHull_mono (Set.image_mono fun q hq => List.mem_cons_of_mem a hq) ih
  | @seg a b l' cf hc0 hc1 hx1 hx2 =>
    have hx : x = cf • DD.toPt a + (1 - cf) • DD.toPt b :=
      Prod.ext_iff.2 ⟨by simpa using hx1, by simpa using hx2⟩
    have ha : DD.toPt a ∈ DD.toPt '' {q | q ∈ a :: b :: l'} := ⟨a, by simp, rfl⟩
    have hb : DD.toPt b ∈ DD.toPt '' {q | q ∈ a :: b :: l'} := ⟨b, by simp, rfl⟩
    rw [hx]
    exact (convex_convexHull ℚ _) (subset_convexHull ℚ _ ha)
      (subset_convexHull ℚ _ hb) hc0 (by linarith) (by ring)

/-- Direction sign: `+1` for the upper chain, `-1` for the lower chain. -/
private def sg : Bool → ℚ
  | true => 1
  | false => -1

/-- Kept orientation: `.gt` builds the upper chain, `.lt` the lower. -/
private def keepOf : Bool → Ordering
  | true => .gt
  | false => .lt

/-- `x` is covered by the polyline through `l` on side `dir`:
some polyline point shares its first coordinate and dominates its second
in the direction `dir` (Coq's `under` for `true`, `over` for `false`). -/
private def Cov (dir : Bool) (x : ℚ × ℚ) (l : List DD) : Prop :=
  ∃ q : ℚ × ℚ, OnPath q l ∧ x.1 = q.1 ∧ sg dir * x.2 ≤ sg dir * q.2

/-- Coverage witnessed by the point of segment `a–c` at parameter `t`. -/
private theorem cov_seg {dir : Bool} {x : ℚ × ℚ} {a c : DD} (l : List DD) {t : ℚ}
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (hx1 : x.1 = t * a.toPt.1 + (1 - t) * c.toPt.1)
    (hx2 : sg dir * x.2 ≤ sg dir * (t * a.toPt.2 + (1 - t) * c.toPt.2)) :
    Cov dir x (a :: c :: l) :=
  ⟨(t * a.toPt.1 + (1 - t) * c.toPt.1, t * a.toPt.2 + (1 - t) * c.toPt.2),
    OnPath.seg l t ht0 ht1 rfl rfl, hx1, hx2⟩

/-- A head vertex with a successor is covered (segment coefficient 1). -/
private theorem cov_head (dir : Bool) (p z : DD) (t : List DD) :
    Cov dir p.toPt (p :: z :: t) :=
  ⟨p.toPt, OnPath.seg t 1 zero_le_one le_rfl (by ring) (by ring), rfl, le_rfl⟩

/-- A second vertex is covered (segment coefficient 0). -/
private theorem cov_second (dir : Bool) (a p : DD) (t : List DD) :
    Cov dir p.toPt (a :: p :: t) :=
  ⟨p.toPt, OnPath.seg t 0 le_rfl zero_le_one (by ring) (by ring), rfl, le_rfl⟩

/-! ### The failed orientation test as a signed cross-product inequality -/

private theorem cross_of_not_keep {dir : Bool} {a b c : DD}
    (h : orientation a b c ≠ keepOf dir) :
    sg dir * cross a.toPt b.toPt c.toPt ≤ 0 := by
  cases dir
  · -- lower chain: not `.lt`, so the cross product is nonnegative
    have h0 : 0 ≤ cross a.toPt b.toPt c.toPt := by
      cases ho : orientation a b c with
      | lt => exact absurd ho h
      | eq => exact ((orientation_eq_iff a b c).1 ho).ge
      | gt => exact ((orientation_gt_iff a b c).1 ho).le
    simpa [sg] using h0
  · -- upper chain: not `.gt`, so the cross product is nonpositive
    have h0 : cross a.toPt b.toPt c.toPt ≤ 0 := by
      cases ho : orientation a b c with
      | lt => exact ((orientation_lt_iff a b c).1 ho).le
      | eq => exact ((orientation_eq_iff a b c).1 ho).le
      | gt => exact absurd ho h
    simpa [sg] using h0

/-! ### The popped vertex lies on the good side of the bridging segment -/

/-- When the orientation test pops `b` from `a :: b :: c :: _`, the vertex `b`
is covered by the segment `a–c`: there is a coefficient `e` placing a point of
that segment at `b`'s first coordinate, on the correct side vertically.
This is Coq's `(x_b − x_c)/(x_a − x_c)` interpolation with its sign analysis,
including the degenerate vertical case `x_a = x_c`. -/
private theorem b_witness {dir : Bool} {a b c : DD}
    (hba : dlt b a) (hca : dlt c a) (hcb : dlt c b)
    (hk : orientation a b c ≠ keepOf dir) :
    ∃ e : ℚ, 0 ≤ e ∧ e ≤ 1 ∧
      b.toPt.1 = e * a.toPt.1 + (1 - e) * c.toPt.1 ∧
      sg dir * b.toPt.2 ≤ sg dir * (e * a.toPt.2 + (1 - e) * c.toPt.2) := by
  have hc_a : c.toPt.1 ≤ a.toPt.1 := dlt_fst hca
  have hc_b : c.toPt.1 ≤ b.toPt.1 := dlt_fst hcb
  have hb_a : b.toPt.1 ≤ a.toPt.1 := dlt_fst hba
  rcases eq_or_lt_of_le hc_a with heq | hlt
  · -- vertical stack: all three first coordinates coincide
    have hb1a : b.toPt.1 = a.toPt.1 := le_antisymm hb_a (heq ▸ hc_b)
    have hb1c : c.toPt.1 = b.toPt.1 := heq.trans hb1a.symm
    cases dir
    · -- lower chain: witness the endpoint `c`
      have h2 : c.toPt.2 < b.toPt.2 := dlt_snd hcb hb1c
      exact ⟨0, le_rfl, zero_le_one, by rw [← hb1c]; ring, by simp only [sg]; linarith⟩
    · -- upper chain: witness the endpoint `a`
      have h2 : b.toPt.2 < a.toPt.2 := dlt_snd hba hb1a
      exact ⟨1, zero_le_one, le_rfl, by rw [hb1a]; ring, by simp only [sg]; linarith⟩
  · -- proper interpolation
    have hcross := cross_of_not_keep hk
    have hA : (0 : ℚ) < a.toPt.1 - c.toPt.1 := by linarith
    refine ⟨(b.toPt.1 - c.toPt.1) / (a.toPt.1 - c.toPt.1),
      div_nonneg (by linarith) hA.le, by rw [div_le_one hA]; linarith,
      by field_simp; ring, ?_⟩
    have hrepr :
        (b.toPt.1 - c.toPt.1) / (a.toPt.1 - c.toPt.1) * a.toPt.2 +
          (1 - (b.toPt.1 - c.toPt.1) / (a.toPt.1 - c.toPt.1)) * c.toPt.2 =
        ((b.toPt.1 - c.toPt.1) * a.toPt.2 + (a.toPt.1 - b.toPt.1) * c.toPt.2) /
          (a.toPt.1 - c.toPt.1) := by
      field_simp
      ring
    rw [hrepr, ← mul_div_assoc, le_div_iff₀ hA]
    unfold cross at hcross
    linarith

/-! ### The core lemma: one insertion preserves coverage
(Coq `under_addUpperPoint` / `over_addLowerPoint`) -/

private theorem cov_addPoint {dir : Bool} {x : ℚ × ℚ} {a : DD} {m : List DD}
    (hs : Desc (a :: m)) (h : Cov dir x (a :: m)) :
    Cov dir x (addPoint (keepOf dir) a m) := by
  induction m with
  | nil =>
    obtain ⟨q, hq, -, -⟩ := h
    cases hq with | tl _ h' => cases h'
  | cons b t ih =>
    cases t with
    | nil =>
      have hba : dlt b a := (List.pairwise_cons.1 hs).1 b (by simp)
      rw [addPoint_one, if_neg (ddCmp_ne_eq hba)]
      exact h
    | cons c l =>
      obtain ⟨ha', hs1⟩ := List.pairwise_cons.1 hs
      obtain ⟨hb', hs2⟩ := List.pairwise_cons.1 hs1
      by_cases hk : (orientation a b c == keepOf dir) = true
      · rw [addPoint_two, if_pos hk]
        exact h
      · rw [addPoint_two, if_neg hk]
        obtain ⟨e, he0, he1, heb1, heb2⟩ :=
          b_witness (ha' b (by simp)) (ha' c (by simp)) (hb' c (by simp))
            fun hh => hk (beq_iff_eq.2 hh)
        refine ih (List.pairwise_cons.2
          ⟨fun z hz => ha' z (List.mem_cons_of_mem b hz), hs2⟩) ?_
        obtain ⟨q, hq, hx1, hx2⟩ := h
        cases hq with
        | tl _ hq' =>
          cases hq' with
          | tl _ hq'' => exact ⟨q, OnPath.tl a hq'', hx1, hx2⟩
          | seg _ y hy0 hy1 hq1 hq2 =>
            -- `q` was on segment `b–c`; substitute the bridge bound for `b`
            refine cov_seg (t := e * y) l (mul_nonneg he0 hy0)
              (by linarith [mul_le_mul he1 hy1 hy0 zero_le_one])
              (by rw [hx1, hq1, heb1]; ring) (hx2.trans ?_)
            rw [hq2]
            linarith [mul_le_mul_of_nonneg_left heb2 hy0]
        | seg _ y hy0 hy1 hq1 hq2 =>
          -- `q` was on segment `a–b`; substitute the bridge bound for `b`
          have h1y : (0 : ℚ) ≤ 1 - y := by linarith
          refine cov_seg (t := y + (1 - y) * e) l
            (by linarith [mul_nonneg h1y he0])
            (by linarith [mul_nonneg h1y (by linarith : (0 : ℚ) ≤ 1 - e)])
            (by rw [hx1, hq1, heb1]; ring) (hx2.trans ?_)
          rw [hq2]
          linarith [mul_le_mul_of_nonneg_left heb2 h1y]

/-! ### Folding the chain (Coq `make_upper` / `make_lower`) -/

private def build (keep : Ordering) (l : List DD) : List DD :=
  l.foldl (fun acc p => addPoint keep p acc) []

private theorem build_append (keep : Ordering) (t : List DD) (a : DD) :
    build keep (t ++ [a]) = addPoint keep a (build keep t) := by
  simp [build, List.foldl_append]

/-- The accumulator stays strictly descending and keeps only input points. -/
private theorem build_spec (keep : Ordering) (l : List DD) (hl : l.Pairwise dlt) :
    Desc (build keep l) ∧ ∀ x ∈ build keep l, x ∈ l := by
  induction l using List.reverseRecOn with
  | nil => exact ⟨List.Pairwise.nil, by simp [build]⟩
  | append_singleton t a ih =>
    obtain ⟨hlt, -, hxa⟩ := List.pairwise_append.1 hl
    obtain ⟨hdesc, hmem⟩ := ih hlt
    rw [build_append]
    have hsub := addPoint_sublist keep a (build keep t)
    refine ⟨List.Pairwise.sublist hsub (List.pairwise_cons.2
      ⟨fun z hz => hxa z (hmem z hz) a (by simp), hdesc⟩), fun x hx => ?_⟩
    rcases List.mem_cons.1 (hsub.subset hx) with rfl | hxt
    · exact List.mem_append_right t (by simp)
    · exact List.mem_append_left [a] (hmem x hxt)

/-- Coq `under_make_upper` / `over_make_lower`: every input point is covered
by the finished chain, unless the chain is just that single point. -/
private theorem cov_build (dir : Bool) (p : DD) (l : List DD)
    (hl : l.Pairwise dlt) (hp : p ∈ l) :
    Cov dir p.toPt (build (keepOf dir) l) ∨ build (keepOf dir) l = [p] := by
  induction l using List.reverseRecOn with
  | nil => cases hp
  | append_singleton t a ih =>
    obtain ⟨hlt, -, hxa⟩ := List.pairwise_append.1 hl
    have hup : ∀ z ∈ t, dlt z a := fun z hz => hxa z hz a (by simp)
    obtain ⟨hdesc, hmem⟩ := build_spec (keepOf dir) t hlt
    rw [build_append]
    rcases List.mem_append.1 hp with hpt | hpa
    · left
      rcases ih hlt hpt with ⟨q, hq, h1, h2⟩ | hsing
      · exact cov_addPoint (List.pairwise_cons.2
          ⟨fun z hz => hup z (hmem z hz), hdesc⟩) ⟨q, OnPath.tl a hq, h1, h2⟩
      · -- the old chain was the single point `p` itself; `a` lands on top of it
        rw [hsing, addPoint_one, if_neg (ddCmp_ne_eq (hup p hpt))]
        exact cov_second dir a p []
    · obtain rfl : p = a := by simpa using hpa
      rcases eq_or_ne (build (keepOf dir) t) [] with hbe | hbn
      · exact .inr (by rw [hbe]; rfl)
      · obtain ⟨z, t', hz⟩ :=
          addPoint_cons_cons (keepOf dir) p _ hbn fun z hz => hup z (hmem z hz)
        exact .inl (by rw [hz]; exact cov_head dir p z t')

/-! ### Under the upper chain + over the lower chain ⇒ in the hull
(Coq `over_under`) -/

private theorem cov_mem_hull (x : ℚ × ℚ) (l₁ l₂ : List DD)
    (h₁ : Cov true x l₁) (h₂ : Cov false x l₂) :
    x ∈ convexHull ℚ (DD.toPt '' {q | q ∈ l₁ ++ l₂}) := by
  obtain ⟨q₁, hp₁, he₁, hy₁⟩ := h₁
  obtain ⟨q₂, hp₂, he₂, hy₂⟩ := h₂
  have hy₁' : x.2 ≤ q₁.2 := by simpa [sg] using hy₁
  have hy₂' : q₂.2 ≤ x.2 := by simpa [sg] using hy₂
  have hm₁ : q₁ ∈ convexHull ℚ (DD.toPt '' {q | q ∈ l₁ ++ l₂}) :=
    convexHull_mono (Set.image_mono fun q hq => List.mem_append_left l₂ hq)
      (onPath_mem_hull hp₁)
  have hm₂ : q₂ ∈ convexHull ℚ (DD.toPt '' {q | q ∈ l₁ ++ l₂}) :=
    convexHull_mono (Set.image_mono fun q hq => List.mem_append_right l₁ hq)
      (onPath_mem_hull hp₂)
  rcases eq_or_lt_of_le (hy₂'.trans hy₁') with heq | hlt
  · -- degenerate vertical: `x` coincides with the lower polyline point
    have hx : x = q₂ := Prod.ext_iff.2 ⟨he₂, le_antisymm (hy₁'.trans heq.ge) hy₂'⟩
    rw [hx]
    exact hm₂
  · -- vertical interpolation between the two polyline points
    have hden : (0 : ℚ) < q₁.2 - q₂.2 := by linarith
    set cf := (x.2 - q₂.2) / (q₁.2 - q₂.2) with hcf
    have hc0 : 0 ≤ cf := div_nonneg (by linarith) hden.le
    have hc1 : cf ≤ 1 := by rw [hcf, div_le_one hden]; linarith
    refine (convex_convexHull ℚ _).segment_subset hm₂ hm₁
      ⟨1 - cf, cf, by linarith, hc0, by ring, Prod.ext_iff.2 ⟨?_, ?_⟩⟩
    · simp only [Prod.fst_add, Prod.smul_fst, smul_eq_mul]
      rw [← he₁, ← he₂]
      ring
    · simp only [Prod.snd_add, Prod.smul_snd, smul_eq_mul]
      rw [hcf]
      field_simp
      ring

/-! ### The deliverable -/

theorem chainHull_covers (s : DSet) (hs : s.Sorted) :
    s.pts ⊆ convexHull ℚ (chainHull s).pts := by
  intro x hx
  obtain ⟨p, hp, rfl⟩ := hx
  have hpair : s.Pairwise dlt :=
    (sorted_iff_pairwise.mp hs).imp fun h => (ddCmp_lt_iff _ _).1 h
  have hh : (chainHull s).pts = DD.toPt '' {q | q ∈ build .gt s ++ build .lt s} :=
    DSet.pts_ofList _
  rw [hh]
  rcases cov_build true p s hpair hp with hcu | hequ
  · rcases cov_build false p s hpair hp with hcl | heql
    · exact cov_mem_hull p.toPt (build .gt s) (build .lt s) hcu hcl
    · -- the lower chain is the single point `p` itself
      exact subset_convexHull ℚ _
        ⟨p, List.mem_append_right _ (heql.symm ▸ List.mem_singleton_self p), rfl⟩
  · -- the upper chain is the single point `p` itself
    exact subset_convexHull ℚ _
      ⟨p, List.mem_append_left _ (hequ.symm ▸ List.mem_singleton_self p), rfl⟩

end Hddivsteps
