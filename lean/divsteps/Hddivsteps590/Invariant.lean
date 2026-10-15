/-
# The coverage invariant

`inState M ds st`: the normalized trajectory point `(g/M, f/M)` lies in
the hull that `st` tracks at the trajectory's delta key, or `g` is
already zero. This module proves that the seed state covers every valid
input (`inState_init`), that one divstep preserves coverage
(`inState_step`), and that hull emptiness forces `g = 0` within the
step count (`g_reaches_zero`) — the reachability content of the bound.

Proof of `inState_step`: the branch that `hdStep` takes maps the
trajectory point by a linear map. The same map sends the tracked hull
into the joined entry at the new delta key (`mem_hull_map` and the
lookup/join lemmas below). If `narrow` drops that entry,
`narrow_int_zero` gives `g = 0`. Otherwise `chainHull_covers` re-covers
the point in the re-hulled entry.

Two self-contained preludes open the file: the *hull algebra* (the three
branch maps as linear maps on ℚ², hull transport along them; Coq: the
matrix lemmas and `in_convex_hull_map_ext`) and the *narrow test*
(`narrow_sound`/`narrow_int_zero`: a passing test confines the hull to
the open slab `|M·x₁| < 1`, forcing a covered integer `g` to 0; Coq:
`in_narrow`).

Coq counterparts: `in_State`, `in_State_step`, `processDivstep_correct`
(hddivsteps_theory.v).
-/
import Mathlib.Analysis.Convex.Hull
import Mathlib.LinearAlgebra.Prod
import Mathlib.Tactic.FieldSimp
import Hddivsteps590.Semantics
import Hddivsteps590.Chain
import Hddivsteps590.NumberTheory

namespace Hddivsteps

/-! ## Hull algebra: the branch maps as linear maps on ℚ² -/

/-- Package any map of the 2×2 matrix form `p ↦ (a·p₁ + b·p₂, c·p₁ + d·p₂)`
as a linear map, keeping `f` itself as the underlying function (so the
branch maps below retain their stated definitional behavior). -/
private def mk₂ (f : ℚ × ℚ → ℚ × ℚ) (a b c d : ℚ)
    (hf : ∀ p, f p = (a * p.1 + b * p.2, c * p.1 + d * p.2) := by
      intro p; simp only [Prod.ext_iff]; constructor <;> ring) :
    (ℚ × ℚ) →ₗ[ℚ] (ℚ × ℚ) where
  toFun := f
  map_add' p q := by simp [hf, Prod.ext_iff]; constructor <;> ring
  map_smul' r p := by simp [hf, Prod.ext_iff]; constructor <;> ring

/-- `(x, y) ↦ (x/2, y)` — the even branch. -/
def evenMap : (ℚ × ℚ) →ₗ[ℚ] (ℚ × ℚ) :=
  mk₂ (fun p => (p.1 / 2, p.2)) (1/2) 0 0 1

/-- `(x, y) ↦ ((x−y)/2, x)` — the odd branch, positive delta (swap). -/
def oddPosMap : (ℚ × ℚ) →ₗ[ℚ] (ℚ × ℚ) :=
  mk₂ (fun p => ((p.1 - p.2) / 2, p.1)) (1/2) (-(1/2)) 1 0

/-- `(x, y) ↦ ((x+y)/2, y)` — the odd branch, nonpositive delta. -/
def oddNonposMap : (ℚ × ℚ) →ₗ[ℚ] (ℚ × ℚ) :=
  mk₂ (fun p => ((p.1 + p.2) / 2, p.2)) (1/2) (1/2) 0 1

/-- Covered points map to covered points: if `x` is in the hull of
`s.pts ∪ {0}` and `trans` realizes the linear map `L` on values, then
`L x` is in the hull of the mapped set (with the origin adjoined). -/
theorem mem_hull_map (L : (ℚ × ℚ) →ₗ[ℚ] (ℚ × ℚ)) (trans : DD → DD)
    (htrans : ∀ p : DD, (trans p).toPt = L p.toPt) (s : DSet) (x : ℚ × ℚ)
    (hx : x ∈ convexHull ℚ (s.pts ∪ {0})) :
    L x ∈ convexHull ℚ ((DSet.ofList (s.map trans)).pts ∪ {0}) := by
  have hpts : (DSet.ofList (s.map trans)).pts = L '' s.pts := by
    rw [DSet.pts_ofList, DSet.pts,
      show {p | p ∈ s.map trans} = trans '' {p | p ∈ s} by ext p; simp [eq_comm],
      ← Set.image_comp, ← Set.image_comp]
    exact Set.image_congr fun p _ => htrans p
  have h1 : L x ∈ L '' convexHull ℚ (s.pts ∪ {0}) := ⟨x, hx, rfl⟩
  rwa [L.image_convexHull, Set.image_union, Set.image_singleton, map_zero,
    ← hpts] at h1

/-- Transport along a hull inclusion of generators: if every generator of
`S` is covered by `T`'s hull, then `S`'s hull is contained in `T`'s hull.
(With `chainHull_covers` this replaces a set by its chain hull.) -/
theorem hull_transport {S T : Set (ℚ × ℚ)} (h : S ⊆ convexHull ℚ T) :
    convexHull ℚ S ⊆ convexHull ℚ T :=
  convexHull_min h (convex_convexHull ℚ T)

/-- Adjoining the origin respects hull inclusion of generators. -/
theorem hull_union_zero_mono {S T : Set (ℚ × ℚ)}
    (h : S ⊆ convexHull ℚ (T ∪ {0})) :
    convexHull ℚ (S ∪ {0}) ⊆ convexHull ℚ (T ∪ {0}) :=
  hull_transport <| Set.union_subset h <|
    Set.Subset.trans Set.subset_union_right (subset_convexHull ℚ _)

/-! ## Soundness of the narrow test -/

/-- Every hull point of a `narrow`-certified set lies in the open slab. -/
theorem narrow_sound (M : Int) (hM : 0 < M) (s : DSet) (hs : s.Sorted)
    (hn : narrow M s = true) :
    ∀ x ∈ convexHull ℚ (s.pts ∪ {0}),
      -1 < (M : ℚ) * x.1 ∧ (M : ℚ) * x.1 < 1 := by
  have hMQ : (0 : ℚ) < (M : ℚ) := by exact_mod_cast hM
  -- the open slab is convex: intersection of two open halfspaces of x ↦ M·x₁
  have hlin : IsLinearMap ℚ (fun x : ℚ × ℚ => (M : ℚ) * x.1) :=
    ⟨fun x y => mul_add _ x.1 y.1, fun c x => mul_left_comm _ c x.1⟩
  have hconv : Convex ℚ {x : ℚ × ℚ | -1 < (M : ℚ) * x.1 ∧ (M : ℚ) * x.1 < 1} :=
    Set.setOf_and ▸ (convex_halfSpace_gt hlin (-1)).inter (convex_halfSpace_lt hlin 1)
  -- every generator lies in the slab
  refine fun x hx => convexHull_min ?_ hconv hx
  rintro p (⟨q, hq, rfl⟩ | rfl)
  · have hne : s ≠ [] := List.ne_nil_of_mem hq
    simp only [narrow, List.head?_eq_some_head hne, List.getLast?_eq_some_getLast hne,
      Bool.and_eq_true, beq_iff_eq, D.cmp_lt_iff, D.mk_toRat, D.mul_toRat,
      D.red_toRat, zpow_zero, mul_one, Int.cast_neg, Int.cast_one] at hn
    exact ⟨hn.1.trans_le (mul_le_mul_of_nonneg_left
        (DSet.head_fst_le s hs _ (List.head?_eq_some_head hne) q hq) hMQ.le),
      (mul_le_mul_of_nonneg_left
        (DSet.fst_le_getLast s hs _ (List.getLast?_eq_some_getLast hne) q hq) hMQ.le).trans_lt hn.2⟩
  · -- the origin
    norm_num

/-- The integer consequence: an integer `g` whose scaled
image `g/M` is covered by a narrow set must be zero. -/
theorem narrow_int_zero (M : Int) (hM : 0 < M) (s : DSet) (hs : s.Sorted)
    (hn : narrow M s = true) (g : Int) (y : ℚ)
    (hcov : ((g : ℚ) / (M : ℚ), y) ∈ convexHull ℚ (s.pts ∪ {0})) :
    g = 0 := by
  obtain ⟨h1, h2⟩ := narrow_sound M hM s hs hn _ hcov
  rw [show (M : ℚ) * ((g : ℚ) / (M : ℚ), y).1 = (g : ℚ) by
    field_simp] at h1 h2
  have : (-1 : Int) < g ∧ g < 1 := by exact_mod_cast And.intro h1 h2
  omega

/-! ## The coverage invariant -/

/-- The coverage invariant (Coq `in_State`): the normalized trajectory
point is covered by the hull tracked at its delta key (or `g` is already
zero). -/
def inState (M : Int) (ds : HDState) (st : State) : Prop :=
  ds.g = 0 ∨
  ((ds.g : ℚ) / (M : ℚ), (ds.f : ℚ) / (M : ℚ)) ∈
    convexHull ℚ ((stateLookup ds.delta st).pts ∪ {0})

/-! ### Private assoc-list toolkit (Coq :302-359) -/

private theorem stateLookup_nil (k : Int) : stateLookup k ([] : State) = [] := rfl

private theorem stateLookup_cons (j k₁ : Int) (v₁ : DSet) (t : State) :
    stateLookup j ((k₁, v₁) :: t) = if j = k₁ then v₁ else stateLookup j t := rfl

/-- A lookup either misses (returns `[]`) or returns an actual entry. -/
private theorem stateLookup_mem (k : Int) :
    ∀ st : State, stateLookup k st = [] ∨ (k, stateLookup k st) ∈ st
  | [] => Or.inl rfl
  | (k₁, v₁) :: t => by
    rw [stateLookup_cons]
    by_cases h : k = k₁
    · exact Or.inr (by simp [h])
    · rw [if_neg h]
      exact (stateLookup_mem k t).imp id (List.mem_cons_of_mem _)

/-- A key foreign to every entry looks up to `[]`. -/
private theorem stateLookup_eq_nil {j : Int} :
    ∀ {st : State}, (∀ p ∈ st, j ≠ p.1) → stateLookup j st = [] := by
  intro st
  induction st with
  | nil => exact fun _ => rfl
  | cons a t ih =>
    intro h
    rw [stateLookup_cons, if_neg (h a (by simp))]
    exact ih fun p hp => h p (List.mem_cons_of_mem _ hp)

/-- Joining at a different key preserves lookups exactly (Coq
`State_lookup_join2`). -/
private theorem lookup_joinState_ne {j k : Int} (hjk : j ≠ k) (v : DSet) :
    ∀ st : State, stateLookup j (joinState k v st) = stateLookup j st
  | [] => by rw [joinState_nil, stateLookup_cons, if_neg hjk, stateLookup_nil]
  | (k₁, v₁) :: t => by
    rw [joinState_cons]
    rcases lt_trichotomy k k₁ with h | rfl | h
    · rw [if_pos h, stateLookup_cons, if_neg hjk]
    · rw [if_neg (lt_irrefl k), if_pos rfl, stateLookup_cons, stateLookup_cons,
        if_neg hjk, if_neg hjk]
    · rw [if_neg h.asymm, if_neg h.ne', stateLookup_cons, stateLookup_cons,
        lookup_joinState_ne hjk v t]

/-- On sorted keys, a same-key join unions the new points into the lookup
(Coq `State_lookup_join1`/`State_lookup_fromList`). -/
private theorem lookup_joinState_self {k : Int} (v : DSet) :
    ∀ {st : State}, st.Pairwise (fun a b : Int × DSet => a.1 < b.1) →
      (stateLookup k (joinState k v st)).pts = v.pts ∪ (stateLookup k st).pts := by
  intro st
  induction st with
  | nil =>
    intro _
    rw [joinState_nil, stateLookup_cons, if_pos rfl, stateLookup_nil, pts_nil,
      Set.union_empty]
  | cons a t ih =>
    obtain ⟨k₁, v₁⟩ := a
    intro hp
    obtain ⟨hhead, htail⟩ := List.pairwise_cons.mp hp
    rw [joinState_cons]
    rcases lt_trichotomy k k₁ with h | rfl | h
    · rw [if_pos h, stateLookup_cons, if_pos rfl, stateLookup_cons, if_neg h.ne,
        stateLookup_eq_nil fun p hp' => (h.trans (hhead p hp')).ne, pts_nil,
        Set.union_empty]
    · rw [if_neg (lt_irrefl k), if_pos rfl, stateLookup_cons, if_pos rfl,
        stateLookup_cons, if_pos rfl, DSet.pts_union]
    · rw [if_neg h.asymm, if_neg h.ne', stateLookup_cons, if_neg h.ne',
        stateLookup_cons, if_neg h.ne', ih htail]

/-- The joined value's points land in the lookup at its key. -/
private theorem subset_lookup_joinState {k : Int} (v : DSet) {st : State}
    (h : st.Pairwise (fun a b : Int × DSet => a.1 < b.1)) :
    v.pts ⊆ (stateLookup k (joinState k v st)).pts :=
  Set.subset_union_left.trans (lookup_joinState_self v h).symm.subset

/-- Joining never shrinks any lookup. -/
private theorem lookup_join_mono {j k : Int} (v : DSet) {st : State}
    (h : st.Pairwise (fun a b : Int × DSet => a.1 < b.1)) :
    (stateLookup j st).pts ⊆ (stateLookup j (joinState k v st)).pts := by
  rcases eq_or_ne j k with rfl | hjk
  · exact Set.subset_union_right.trans (lookup_joinState_self v h).symm.subset
  · rw [lookup_joinState_ne hjk]

/-! ### The fold of `processDivstep` (defs `pdStep`/`dropStep` in Semantics) -/

private theorem pdStep_def (acc : State) (k : Int) (v : DSet) :
    pdStep acc (k, v) =
      if 0 < k then
        joinState (2 - k) (DSet.ofList (v.map oddPosTrans))
          (joinState (2 + k) (DSet.ofList (v.map evenTrans)) acc)
      else
        joinState (2 + k) (DSet.ofList (v.map oddNonposTrans))
          (joinState (2 + k) (DSet.ofList (v.map evenTrans)) acc) := rfl

private theorem pdStep_mono {acc : State}
    (hacc : acc.Pairwise (fun a b : Int × DSet => a.1 < b.1))
    (kv : Int × DSet) (j : Int) :
    (stateLookup j acc).pts ⊆ (stateLookup j (pdStep acc kv)).pts := by
  unfold pdStep
  split <;>
    exact (lookup_join_mono _ hacc).trans
      (lookup_join_mono _ (joinState_pairwise _ _ acc hacc))

private theorem foldl_mono {j : Int} {S : Set (ℚ × ℚ)} :
    ∀ (l : List (Int × DSet)) (acc : State), (∀ kv ∈ l, DSet.Sorted kv.2) →
      acc.WF → S ⊆ (stateLookup j acc).pts →
      S ⊆ (stateLookup j (List.foldl pdStep acc l)).pts
  | [], _, _, _, hS => hS
  | kv :: t, acc, hl, hacc, hS =>
    foldl_mono t (pdStep acc kv) (fun x hx => hl x (List.mem_cons_of_mem _ hx))
      (pdStep_WF hacc kv (hl kv (by simp)))
      (hS.trans (pdStep_mono (wf_iff.mp hacc).1 kv j))

/-- If processing `(dk, v)` puts `S` into the lookup at `j`, so does the
whole fold over any list containing `(dk, v)`. -/
private theorem foldl_hit {st : State} {dk : Int} {v : DSet} (hmem : (dk, v) ∈ st)
    (hst : ∀ kv ∈ st, DSet.Sorted kv.2) {j : Int} {S : Set (ℚ × ℚ)}
    (hhit : ∀ acc : State, acc.WF → S ⊆ (stateLookup j (pdStep acc (dk, v))).pts) :
    S ⊆ (stateLookup j (st.foldl pdStep [])).pts := by
  obtain ⟨l₁, l₂, rfl⟩ := List.append_of_mem hmem
  rw [List.foldl_append, List.foldl_cons]
  have hA : (l₁.foldl pdStep []).WF :=
    foldl_WF l₁ [] (fun x hx => hst x (List.mem_append_left _ hx)) wf_nil
  exact foldl_mono l₂ _
    (fun x hx => hst x (List.mem_append_right _ (List.mem_cons_of_mem _ hx)))
    (pdStep_WF hA _ (hst _ hmem)) (hhit _ hA)

private theorem pdStep_hit_even {acc : State} (hacc : acc.WF) (k : Int) (v : DSet) :
    (DSet.ofList (v.map evenTrans)).pts ⊆
      (stateLookup (2 + k) (pdStep acc (k, v))).pts := by
  have hp := (wf_iff.mp hacc).1
  rw [pdStep_def]
  split <;>
    exact (subset_lookup_joinState _ hp).trans
      (lookup_join_mono _ (joinState_pairwise _ _ acc hp))

private theorem pdStep_hit_oddPos {acc : State} (hacc : acc.WF) {k : Int}
    (hk : 0 < k) (v : DSet) :
    (DSet.ofList (v.map oddPosTrans)).pts ⊆
      (stateLookup (2 - k) (pdStep acc (k, v))).pts := by
  rw [pdStep_def, if_pos hk]
  exact subset_lookup_joinState _ (joinState_pairwise _ _ acc (wf_iff.mp hacc).1)

private theorem pdStep_hit_oddNonpos {acc : State} (hacc : acc.WF) {k : Int}
    (hk : ¬ 0 < k) (v : DSet) :
    (DSet.ofList (v.map oddNonposTrans)).pts ⊆
      (stateLookup (2 + k) (pdStep acc (k, v))).pts := by
  rw [pdStep_def, if_neg hk]
  exact subset_lookup_joinState _ (joinState_pairwise _ _ acc (wf_iff.mp hacc).1)

/-- After the narrow filter, a surviving entry's lookup is its chain hull. -/
private theorem lookup_filterMap (M : Int) :
    ∀ s1 : State, s1.Pairwise (fun a b : Int × DSet => a.1 < b.1) →
      ∀ (k : Int) (w : DSet), (k, w) ∈ s1 → narrow M w = false →
        stateLookup k (s1.filterMap (dropStep M)) = chainHull w := by
  intro s1
  induction s1 with
  | nil => rintro _ k w ⟨⟩
  | cons a t ih =>
    intro hp k w hmem hnar
    obtain ⟨hhead, htail⟩ := List.pairwise_cons.mp hp
    rcases List.mem_cons.mp hmem with rfl | hmem'
    · have hdrop : dropStep M (k, w) = some (k, chainHull w) := by
        simp [dropStep, hnar]
      simp [hdrop, stateLookup_cons]
    · have hne : k ≠ a.1 := (hhead _ hmem').ne'
      cases h₀ : narrow M a.2 with
      | true =>
        have hdrop : dropStep M a = none := by simp [dropStep, h₀]
        simp only [List.filterMap_cons, hdrop]
        exact ih htail k w hmem' hnar
      | false =>
        have hdrop : dropStep M a = some (a.1, chainHull a.2) := by
          simp [dropStep, h₀]
        simp only [List.filterMap_cons, hdrop, stateLookup_cons, if_neg hne]
        exact ih htail k w hmem' hnar

/-! ### The branch maps, applied -/

private theorem evenMap_apply (x y : ℚ) : evenMap (x, y) = (x / 2, y) := rfl
private theorem oddPosMap_apply (x y : ℚ) : oddPosMap (x, y) = ((x - y) / 2, x) := rfl
private theorem oddNonposMap_apply (x y : ℚ) :
    oddNonposMap (x, y) = ((x + y) / 2, y) := rfl

/-- Coverage by an empty lookup — the hull of `{0}` alone — forces the
scaled integer to be zero. -/
private theorem eq_zero_of_lookup_nil {M : Int} (hM : 0 < M) {g : Int} {y : ℚ}
    (hcov : ((g : ℚ) / (M : ℚ), y) ∈ convexHull ℚ (DSet.pts ([] : DSet) ∪ {0})) :
    g = 0 := by
  rw [pts_nil, Set.empty_union, convexHull_singleton, Set.mem_singleton_iff] at hcov
  have h0 : (g : ℚ) / (M : ℚ) = 0 := congrArg Prod.fst hcov
  have hMne : (M : ℚ) ≠ 0 := by exact_mod_cast hM.ne'
  exact_mod_cast (div_eq_zero_iff.mp h0).resolve_right hMne

/-- From pre-filter coverage to the post-filter disjunction: a dropped
entry forces `g' = 0`; a survivor still covers after re-hulling. -/
private theorem step_finish (M : Int) (hM : 0 < M) (st : State)
    (hst : ∀ kv ∈ st, DSet.Sorted kv.2) (g' f' dk' : Int)
    (hcov : ((g' : ℚ) / (M : ℚ), (f' : ℚ) / (M : ℚ)) ∈
      convexHull ℚ ((stateLookup dk' (st.foldl pdStep [])).pts ∪ {0})) :
    g' = 0 ∨ ((g' : ℚ) / (M : ℚ), (f' : ℚ) / (M : ℚ)) ∈
      convexHull ℚ ((stateLookup dk' (processDivstep M st)).pts ∪ {0}) := by
  have hwf : (st.foldl pdStep []).WF := foldl_WF st [] hst wf_nil
  rcases stateLookup_mem dk' (st.foldl pdStep []) with hnil | hmem
  · exact Or.inl (eq_zero_of_lookup_nil hM (hnil ▸ hcov))
  have hsort : (stateLookup dk' (st.foldl pdStep [])).Sorted := hwf.2 _ hmem
  cases hnar : narrow M (stateLookup dk' (st.foldl pdStep [])) with
  | true => exact Or.inl (narrow_int_zero M hM _ hsort hnar g' _ hcov)
  | false =>
    refine Or.inr ?_
    rw [processDivstep_eq, lookup_filterMap M _ (wf_iff.mp hwf).1 _ _ hmem hnar]
    exact hull_union_zero_mono
      ((chainHull_covers _ hsort).trans (convexHull_mono Set.subset_union_left)) hcov

/-- One `hdStep` branch: push the covered point through the branch's linear
map into the joined pre-filter state, then across the narrow filter. -/
private theorem step_branch {M : Int} (hM : 0 < M) {ds : HDState} {st : State}
    (hwf : st.WF) (hmem : (ds.delta, stateLookup ds.delta st) ∈ st)
    (L : (ℚ × ℚ) →ₗ[ℚ] (ℚ × ℚ)) (trans : DD → DD)
    (htrans : ∀ p : DD, (trans p).toPt = L p.toPt)
    (hhit : ∀ acc : State, acc.WF →
      (DSet.ofList ((stateLookup ds.delta st).map trans)).pts ⊆
        (stateLookup (hdStep ds).delta
          (pdStep acc (ds.delta, stateLookup ds.delta st))).pts)
    (hpt : L ((ds.g : ℚ) / (M : ℚ), (ds.f : ℚ) / (M : ℚ)) =
      (((hdStep ds).g : ℚ) / (M : ℚ), ((hdStep ds).f : ℚ) / (M : ℚ)))
    (hcov : ((ds.g : ℚ) / (M : ℚ), (ds.f : ℚ) / (M : ℚ)) ∈
      convexHull ℚ ((stateLookup ds.delta st).pts ∪ {0})) :
    inState M (hdStep ds) (processDivstep M st) :=
  step_finish M hM st hwf.2 _ _ _ <| hpt ▸
    convexHull_mono (Set.union_subset_union_left _ (foldl_hit hmem hwf.2 hhit))
      (mem_hull_map L trans htrans _ _ hcov)

/-! ### The three deliverables -/

theorem inState_init (M f g : Int) (hM : 0 < M) (hfM : f ≤ M)
    (hg0 : 0 ≤ g) (hgf : g ≤ f) :
    inState M (hdInit f g) state0 := by
  have hMQ : (0 : ℚ) < (M : ℚ) := by exact_mod_cast hM
  have hfQ : (f : ℚ) ≤ (M : ℚ) := by exact_mod_cast hfM
  have hgQ : (0 : ℚ) ≤ (g : ℚ) := by exact_mod_cast hg0
  have hgfQ : (g : ℚ) ≤ (f : ℚ) := by exact_mod_cast hgf
  refine Or.inr ?_
  show ((g : ℚ) / (M : ℚ), (f : ℚ) / (M : ℚ)) ∈
    convexHull ℚ ((stateLookup 1 state0).pts ∪ {0})
  set S := (stateLookup 1 state0).pts ∪ {0}
  have hB : ((1 : ℚ), (1 : ℚ)) ∈ S := Set.mem_union_left _
    ⟨(⟨1, 0⟩, ⟨1, 0⟩), by simp [state0, stateLookup], by norm_num [DD.toPt, D.toRat]⟩
  have hC : ((0 : ℚ), (1 : ℚ)) ∈ S := Set.mem_union_left _
    ⟨(⟨0, 0⟩, ⟨1, 0⟩), by simp [state0, stateLookup], by norm_num [DD.toPt, D.toRat]⟩
  have hA : ((0 : ℚ), (0 : ℚ)) ∈ S := Set.mem_union_right _ rfl
  rcases eq_or_lt_of_le (hgQ.trans hgfQ) with hf0 | hfpos
  · -- `f = 0` forces `g = 0`: the trajectory point IS the origin
    rw [← hf0, show (g : ℚ) = 0 from le_antisymm (hgfQ.trans hf0.ge) hgQ, zero_div]
    exact subset_convexHull ℚ S hA
  · have hfne : (f : ℚ) ≠ 0 := hfpos.ne'
    have hMne : (M : ℚ) ≠ 0 := hMQ.ne'
    -- segment 1: `(g/f, 1)` sits between `(1, 1)` and `(0, 1)` …
    have hy : ((g : ℚ) / (f : ℚ), (1 : ℚ)) ∈ convexHull ℚ S := by
      refine (convex_convexHull ℚ S).segment_subset (subset_convexHull ℚ S hB)
        (subset_convexHull ℚ S hC)
        ⟨(g : ℚ) / f, 1 - (g : ℚ) / f, div_nonneg hgQ hfpos.le,
          sub_nonneg.mpr ((div_le_one hfpos).mpr hgfQ), by ring, ?_⟩
      simp only [Prod.smul_mk, smul_eq_mul, Prod.mk_add_mk, Prod.mk.injEq]
      constructor <;> ring
    -- … segment 2: scale it toward the origin by `f/M`
    refine (convex_convexHull ℚ S).segment_subset hy (subset_convexHull ℚ S hA)
      ⟨(f : ℚ) / M, 1 - (f : ℚ) / M, div_nonneg hfpos.le hMQ.le,
        sub_nonneg.mpr ((div_le_one hMQ).mpr hfQ), by ring, ?_⟩
    simp only [Prod.smul_mk, smul_eq_mul, Prod.mk_add_mk, Prod.mk.injEq]
    constructor <;> field_simp <;> ring

theorem inState_step (M : Int) (hM : 0 < M) (ds : HDState) (st : State)
    (hwf : st.WF) (hf : ds.f % 2 = 1) (h : inState M ds st) :
    inState M (hdStep ds) (processDivstep M st) := by
  rcases h with hg | hcov
  · exact Or.inl (hdStep_g_zero ds hg)
  rcases stateLookup_mem ds.delta st with hnil | hmem
  · exact Or.inl (hdStep_g_zero ds (eq_zero_of_lookup_nil hM (hnil ▸ hcov)))
  by_cases hpar : ds.g % 2 = 0
  · -- even: `delta' = 2 + delta`, `g' = g / 2`, `f' = f`
    have hd : (hdStep ds).delta = 2 + ds.delta := by simp [hdStep, hpar]
    refine step_branch hM hwf hmem evenMap evenTrans (fun p => evenTrans_toPt p)
      (fun acc hacc => hd ▸ pdStep_hit_even hacc _ _) ?_ hcov
    rw [show (hdStep ds).g = ds.g / 2 by simp [hdStep, hpar],
      show (hdStep ds).f = ds.f by simp [hdStep, hpar],
      evenMap_apply, Int.cast_div_charZero (by omega : (2 : ℤ) ∣ ds.g)]
    simp only [Prod.mk.injEq]
    exact ⟨by ring, trivial⟩
  · have hpar1 : ds.g % 2 = 1 := by omega
    by_cases hdel : 0 < ds.delta
    · -- odd, `delta > 0`: `delta' = 2 - delta`, `g' = (g - f) / 2`, `f' = g`
      have hd : (hdStep ds).delta = 2 - ds.delta := by simp [hdStep, hpar, hdel]
      refine step_branch hM hwf hmem oddPosMap oddPosTrans
        (fun p => oddPosTrans_toPt p)
        (fun acc hacc => hd ▸ pdStep_hit_oddPos hacc hdel _) ?_ hcov
      rw [show (hdStep ds).g = (ds.g - ds.f) / 2 by simp [hdStep, hpar, hdel],
        show (hdStep ds).f = ds.g by simp [hdStep, hpar, hdel],
        oddPosMap_apply, Int.cast_div_charZero (by omega : (2 : ℤ) ∣ ds.g - ds.f)]
      simp only [Prod.mk.injEq]
      exact ⟨by push_cast; ring, trivial⟩
    · -- odd, `delta ≤ 0`: `delta' = 2 + delta`, `g' = (g + f) / 2`, `f' = f`
      have hd : (hdStep ds).delta = 2 + ds.delta := by simp [hdStep, hpar, hdel]
      refine step_branch hM hwf hmem oddNonposMap oddNonposTrans
        (fun p => oddNonposTrans_toPt p)
        (fun acc hacc => hd ▸ pdStep_hit_oddNonpos hacc hdel _) ?_ hcov
      rw [show (hdStep ds).g = (ds.g + ds.f) / 2 by simp [hdStep, hpar, hdel],
        show (hdStep ds).f = ds.f by simp [hdStep, hpar, hdel],
        oddNonposMap_apply,
        Int.cast_div_charZero (by omega : (2 : ℤ) ∣ ds.g + ds.f)]
      simp only [Prod.mk.injEq]
      exact ⟨by push_cast; ring, trivial⟩

/-- Hull-emptiness at `(N, M)` forces every valid trajectory to finish:
the reachability half of the Coq bridge (`processDivstep_correct`). -/
theorem g_reaches_zero (N : Nat) (M f g : Int) (hodd : f % 2 = 1)
    (hfM : f ≤ M) (hg0 : 0 ≤ g) (hgf : g ≤ f)
    (hcert : certEmpty M N = true) :
    (hdRun N f g).g = 0 := by
  have hM : 0 < M := by omega
  have hwf : ∀ n : Nat, (Nat.repeat (processDivstep M) n state0).WF := fun n =>
    n.rec state0_wf fun _ ih => processDivstep_wf M _ ih
  have key : ∀ n : Nat,
      inState M (hdRun n f g) (Nat.repeat (processDivstep M) n state0) ∧
      (hdRun n f g).f % 2 = 1 := fun n =>
    n.rec ⟨inState_init M f g hM hfM hg0 hgf, hodd⟩ fun n ih =>
      ⟨inState_step M hM _ _ (hwf n) ih.2 ih.1, hdStep_odd_f _ ih.2⟩
  have hemp : Nat.repeat (processDivstep M) N state0 = [] :=
    List.isEmpty_iff.mp hcert
  rcases (key N).1 with h | h
  · exact h
  · rw [hemp, stateLookup_nil] at h
    exact eq_zero_of_lookup_nil hM h

end Hddivsteps
