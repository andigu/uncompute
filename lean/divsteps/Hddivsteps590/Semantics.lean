/-
# ℚ-semantics of the computational definitions

This module gives values to the computation: `D.toRat` maps a dyadic to
ℚ, `DD.toPt` maps a point to ℚ × ℚ, and `DSet.pts` collects the value
set of a vertex list. It proves that the dyadic operations are exact,
that the comparison functions decide the value order, and that the set
and state operations preserve membership and sortedness.

Coq counterparts: `Dadd_Q` … `Dhalf_Q` (hddivsteps_convexhull.v),
`Dred_eq`, `Dcompare_Q` (hddivsteps_base.v). The dyadics here are
canonical (odd mantissa, or zero), so the setoid layer of the Coq
development has no counterpart.
-/
import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Data.List.Sort
import Mathlib.Data.List.Chain
import Mathlib.Data.Set.Image
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Push
import Mathlib.Tactic.Ring
import Hddivsteps590.Defs

namespace Hddivsteps

/-! ## Value maps -/

/-- The value of a dyadic: `m * 2^e` (ℤ-exponent). -/
def D.toRat (a : D) : ℚ := a.m * (2 : ℚ) ^ a.e

/-- The value of a point. -/
def DD.toPt (p : DD) : ℚ × ℚ := (p.1.toRat, p.2.toRat)

/-- The set of point values of a `DSet` (a `Set`, i.e. value-level). -/
def DSet.pts (s : DSet) : Set (ℚ × ℚ) := DD.toPt '' {p | p ∈ s}

/-- Strict ascending sortedness in the value-lexicographic order. -/
def DSet.Sorted (s : DSet) : Prop := s.IsChain (fun a b => ddCmp a b = .lt)

/-- Lookup a delta key in the state (empty set when absent). -/
def stateLookup (k : Int) : State → DSet
  | [] => []
  | (k', v) :: t => if k = k' then v else stateLookup k t

/-- State well-formedness: strictly ascending keys, every value sorted. -/
def State.WF (st : State) : Prop :=
  st.IsChain (fun a b => a.1 < b.1) ∧ ∀ kv ∈ st, DSet.Sorted kv.2

/-! ## Dyadic arithmetic is exact (homomorphism into ℚ) -/

/-- `oddPart` really is the odd part: the mantissa times the extracted power
of two recovers the input, and (on nonzero input) the mantissa is odd. -/
private theorem oddPart_spec (n : Nat) :
    (oddPart n).1 * 2 ^ (oddPart n).2 = n ∧ (n ≠ 0 → (oddPart n).1 % 2 = 1) := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rw [oddPart]
    split
    next h => exact ⟨by simp, fun hn => show n % 2 = 1 by omega⟩
    next h =>
      obtain ⟨ih1, ih2⟩ := ih (n / 2) (by omega)
      rcases hmk : oddPart (n / 2) with ⟨m, k⟩
      rw [hmk] at ih1 ih2
      exact ⟨show m * 2 ^ (k + 1) = n by rw [pow_succ, ← mul_assoc, ih1]; omega,
        fun _ => ih2 (by omega)⟩

/-- Exponent-alignment identity: rewrite a dyadic at any smaller exponent. -/
private theorem D.toRat_shift (a : D) {e : Int} (h : e ≤ a.e) :
    a.toRat = ((a.m * 2 ^ (a.e - e).toNat : Int) : ℚ) * (2 : ℚ) ^ e := by
  have hk : (2 : ℚ) ^ ((a.e - e).toNat : Int) * (2 : ℚ) ^ e = (2 : ℚ) ^ a.e := by
    rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]; congr 1; omega
  rw [D.toRat, ← hk, zpow_natCast]
  push_cast
  ring

@[simp] theorem D.red_toRat (m e : Int) :
    (D.red m e).toRat = (m : ℚ) * (2 : ℚ) ^ e := by
  rw [D.red]
  split
  next h => subst h; simp [D.toRat]
  next h =>
    rcases hmk : oddPart m.natAbs with ⟨om, k⟩
    obtain ⟨h1, -⟩ := oddPart_spec m.natAbs
    rw [hmk] at h1
    have hZ : (if m < 0 then -(om : ℤ) else (om : ℤ)) * 2 ^ k = m := by
      have h0 : (om : ℤ) * 2 ^ k = m.natAbs := by exact_mod_cast h1
      split
      · rw [neg_mul, h0]; omega
      · rw [h0]; omega
    simp only [D.toRat]
    rw [zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0), zpow_natCast]
    conv_rhs => rw [← hZ]
    push_cast
    ring

@[simp] theorem D.add_toRat (a b : D) :
    (a.add b).toRat = a.toRat + b.toRat := by
  rw [D.add]
  split
  · rw [D.red_toRat, D.toRat_shift a (by omega : b.e ≤ a.e), D.toRat]; push_cast; ring
  · rw [D.red_toRat, D.toRat_shift b (by omega : a.e ≤ b.e), D.toRat]; push_cast; ring

@[simp] theorem D.sub_toRat (a b : D) :
    (a.sub b).toRat = a.toRat - b.toRat := by
  rw [D.sub]
  split
  · rw [D.red_toRat, D.toRat_shift a (by omega : b.e ≤ a.e), D.toRat]; push_cast; ring
  · rw [D.red_toRat, D.toRat_shift b (by omega : a.e ≤ b.e), D.toRat]; push_cast; ring

@[simp] theorem D.mul_toRat (a b : D) :
    (a.mul b).toRat = a.toRat * b.toRat := by
  rw [D.mul, D.red_toRat, D.toRat, D.toRat, zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
  push_cast
  ring

@[simp] theorem D.half_toRat (a : D) :
    a.half.toRat = a.toRat / 2 := by
  rw [D.half]
  split
  next h => simp [D.toRat, h]
  next h =>
    simp only [D.toRat]
    rw [zpow_sub₀ (by norm_num : (2 : ℚ) ≠ 0), zpow_one]
    ring

/-- Integer constants embed as expected (e.g. the `⟨-1,0⟩`, `⟨1,0⟩` used in
`narrow`, and `D.red M 0`). -/
@[simp] theorem D.mk_toRat (m : Int) : D.toRat ⟨m, 0⟩ = (m : ℚ) := by
  simp [D.toRat]

/-! ## Comparisons reflect values -/

/-- `compare` on ℤ transfers along the strictly monotone `x ↦ (x : ℚ) * 2^e`. -/
private theorem compare_cast_mul_zpow (x y e : ℤ) :
    compare x y = compare ((x : ℚ) * 2 ^ e) ((y : ℚ) * 2 ^ e) := by
  have h2 : (0 : ℚ) < 2 ^ e := by positivity
  rcases lt_trichotomy x y with h | h | h
  · rw [Int.compare_eq_lt.mpr h, eq_comm, compare_lt_iff_lt]
    exact mul_lt_mul_of_pos_right (by exact_mod_cast h) h2
  · rw [h, Int.compare_eq_eq.mpr rfl, eq_comm, compare_eq_iff_eq]
  · rw [Int.compare_eq_gt.mpr h, eq_comm, compare_gt_iff_gt]
    exact mul_lt_mul_of_pos_right (by exact_mod_cast h) h2

/-- `D.cmp` (align exponents, compare mantissas) computes `compare` of the
values — the shared core of the three `D.cmp_*_iff` lemmas. -/
private theorem D.cmp_eq_compare (a b : D) : a.cmp b = compare a.toRat b.toRat := by
  rw [D.cmp]
  split
  · rw [D.toRat_shift a (by omega : b.e ≤ a.e), D.toRat, ← compare_cast_mul_zpow]
  · rw [D.toRat_shift b (by omega : a.e ≤ b.e), D.toRat, ← compare_cast_mul_zpow]

theorem D.cmp_lt_iff (a b : D) : a.cmp b = .lt ↔ a.toRat < b.toRat := by
  rw [D.cmp_eq_compare, compare_lt_iff_lt]

theorem D.cmp_eq_iff (a b : D) : a.cmp b = .eq ↔ a.toRat = b.toRat := by
  rw [D.cmp_eq_compare, compare_eq_iff_eq]

theorem D.cmp_gt_iff (a b : D) : a.cmp b = .gt ↔ b.toRat < a.toRat := by
  rw [D.cmp_eq_compare, compare_gt_iff_gt]

/-- Componentwise form of `ddCmp_eq_iff`, used where the pair equality would
just be split again. -/
private theorem ddCmp_eq_iff' (p q : DD) :
    ddCmp p q = .eq ↔ p.1.toRat = q.1.toRat ∧ p.2.toRat = q.2.toRat := by
  simp only [ddCmp, Ordering.then_eq_eq, D.cmp_eq_iff]

theorem ddCmp_eq_iff (p q : DD) : ddCmp p q = .eq ↔ p.toPt = q.toPt := by
  rw [ddCmp_eq_iff', DD.toPt, DD.toPt, Prod.mk.injEq]

theorem ddCmp_lt_iff (p q : DD) :
    ddCmp p q = .lt ↔
      p.1.toRat < q.1.toRat ∨ (p.1.toRat = q.1.toRat ∧ p.2.toRat < q.2.toRat) := by
  simp only [ddCmp, Ordering.then_eq_lt, D.cmp_lt_iff, D.cmp_eq_iff]

/-- The `.gt` companion of `ddCmp_lt_iff` (kept private; consumers use the
public `lt`/`eq` forms). -/
private theorem ddCmp_gt_iff (p q : DD) :
    ddCmp p q = .gt ↔
      q.1.toRat < p.1.toRat ∨ (p.1.toRat = q.1.toRat ∧ q.2.toRat < p.2.toRat) := by
  simp only [ddCmp, Ordering.then_eq_gt, D.cmp_gt_iff, D.cmp_eq_iff]

/-- Antisymmetry: `.gt` one way is `.lt` the other. -/
private theorem ddCmp_gt_iff_lt {p q : DD} : ddCmp p q = .gt ↔ ddCmp q p = .lt := by
  rw [ddCmp_gt_iff, ddCmp_lt_iff]
  exact or_congr .rfl (and_congr eq_comm .rfl)

private theorem ddCmp_lt_trans {a b c : DD}
    (h1 : ddCmp a b = .lt) (h2 : ddCmp b c = .lt) : ddCmp a c = .lt := by
  rw [ddCmp_lt_iff] at h1 h2 ⊢
  rcases h1 with h1 | ⟨h1, h1'⟩ <;> rcases h2 with h2 | ⟨h2, h2'⟩
  exacts [.inl (h1.trans h2), .inl (h2 ▸ h1), .inl (h1 ▸ h2),
    .inr ⟨h1.trans h2, h1'.trans h2'⟩]

private theorem ddCmp_eq_lt_trans {a b c : DD}
    (h1 : ddCmp a b = .eq) (h2 : ddCmp b c = .lt) : ddCmp a c = .lt := by
  rw [ddCmp_eq_iff'] at h1
  rw [ddCmp_lt_iff] at h2 ⊢
  rcases h2 with h2 | ⟨h2, h2'⟩
  · exact .inl (h1.1 ▸ h2)
  · exact .inr ⟨h1.1.trans h2, h1.2 ▸ h2'⟩

-- required by `List.isChain_iff_pairwise`
private instance : Trans (fun a b : DD => ddCmp a b = .lt) (fun a b : DD => ddCmp a b = .lt)
    (fun a b : DD => ddCmp a b = .lt) := ⟨ddCmp_lt_trans⟩

theorem sorted_iff_pairwise {s : DSet} :
    s.Sorted ↔ s.Pairwise (fun a b => ddCmp a b = .lt) :=
  List.isChain_iff_pairwise

/-- The non-strict order used by `mergeSort`, in value form. -/
private theorem ddCmp_ne_gt_iff {p q : DD} :
    ddCmp p q ≠ .gt ↔
      p.1.toRat < q.1.toRat ∨ (p.1.toRat = q.1.toRat ∧ p.2.toRat ≤ q.2.toRat) := by
  rw [ne_eq, ddCmp_gt_iff]
  push Not
  constructor
  · rintro ⟨h1, h2⟩
    rcases h1.lt_or_eq with h | h
    · exact .inl h
    · exact .inr ⟨h, h2 h⟩
  · rintro (h | ⟨h1, h2⟩)
    · exact ⟨h.le, fun he => absurd (he ▸ h) (lt_irrefl _)⟩
    · exact ⟨h1.le, fun _ => h2⟩

private theorem ddCmp_le_trans (a b c : DD) :
    (ddCmp a b != .gt) = true → (ddCmp b c != .gt) = true → (ddCmp a c != .gt) = true := by
  simp only [bne_iff_ne, ddCmp_ne_gt_iff]
  rintro (h1 | ⟨h1, h1'⟩) (h2 | ⟨h2, h2'⟩)
  exacts [.inl (by linarith), .inl (by linarith), .inl (by linarith),
    .inr ⟨by linarith, by linarith⟩]

private theorem ddCmp_le_total (a b : DD) :
    ((ddCmp a b != .gt) || (ddCmp b a != .gt)) = true := by
  rcases h : ddCmp a b with _ | _ | _ <;> simp_all [ddCmp_gt_iff_lt]

/-! ### Structural equality (`BEq`) is lawful on `D` and `DD` -/

private theorem D.beq_iff {a b : D} : (a == b) = true ↔ a = b := by
  obtain ⟨am, ae⟩ := a
  obtain ⟨bm, be⟩ := b
  show (am == bm && ae == be) = true ↔ _
  simp

private instance : LawfulBEq D where
  eq_of_beq h := D.beq_iff.mp h
  rfl := D.beq_iff.mpr rfl

private theorem DD.beq_iff {p q : DD} : (p == q) = true ↔ p = q := beq_iff_eq

/-! ## Orientation is the sign of the cross product -/

/-- `cross p q r = (p₁-r₁)(q₂-r₂) - (p₂-r₂)(q₁-r₁)`; positive = left turn. -/
def cross (p q r : ℚ × ℚ) : ℚ :=
  (p.1 - r.1) * (q.2 - r.2) - (p.2 - r.2) * (q.1 - r.1)

theorem orientation_lt_iff (p q r : DD) :
    orientation p q r = .lt ↔ cross p.toPt q.toPt r.toPt < 0 := by
  rw [orientation, D.cmp_lt_iff, cross]
  simp only [D.mul_toRat, D.sub_toRat, DD.toPt, sub_neg]

theorem orientation_eq_iff (p q r : DD) :
    orientation p q r = .eq ↔ cross p.toPt q.toPt r.toPt = 0 := by
  rw [orientation, D.cmp_eq_iff, cross]
  simp only [D.mul_toRat, D.sub_toRat, DD.toPt, sub_eq_zero]

theorem orientation_gt_iff (p q r : DD) :
    orientation p q r = .gt ↔ 0 < cross p.toPt q.toPt r.toPt := by
  rw [orientation, D.cmp_gt_iff, cross]
  simp only [D.mul_toRat, D.sub_toRat, DD.toPt, sub_pos]

/-! ## Set operations: membership and sortedness -/

private theorem mem_dedupAdj (x : DD) (l : List DD) : x ∈ dedupAdj l ↔ x ∈ l := by
  fun_induction dedupAdj l with
  | case1 a b t hab ih => obtain rfl := DD.beq_iff.mp hab; simp [ih, List.mem_cons]
  | case2 a b t hab ih => simp [List.mem_cons, ih]
  | case3 l h => exact .rfl

@[simp] theorem DSet.pts_ofList (l : List DD) :
    (DSet.ofList l).pts = DD.toPt '' {p | p ∈ l} :=
  congrArg (DD.toPt '' ·) (Set.ext fun p => by
    simp only [Set.mem_setOf_eq, DSet.ofList, mem_dedupAdj, List.mem_mergeSort])

/-- `dedupAdj` of a list that is pairwise `lt`-or-structurally-equal is
strictly sorted.  (The `lt`-or-equal hypothesis is where value injectivity
enters: `dedupAdj`'s `BEq` is structural, so a value tie must already be a
structural tie.) -/
private theorem dedupAdj_pairwise {l : List DD}
    (hl : l.Pairwise (fun a b => ddCmp a b = .lt ∨ a = b)) :
    (dedupAdj l).Pairwise (fun a b => ddCmp a b = .lt) := by
  fun_induction dedupAdj l with
  | case1 a b t hab ih => exact ih hl.of_cons
  | case2 a b t hab ih =>
    obtain ⟨ha, htail⟩ := List.pairwise_cons.mp hl
    refine (ih htail).cons fun x hx => ?_
    have hx' := (mem_dedupAdj x _).mp hx
    rcases ha x hx' with h | rfl
    · exact h
    -- `a` reappears in `b :: t`: contradicts `a ≠ b` resp. strict sortedness
    have hne : a ≠ b := fun h => hab (DD.beq_iff.mpr h)
    rcases List.mem_cons.mp hx' with rfl | hxt
    · exact absurd rfl hne
    · have h1 := (ha b List.mem_cons_self).resolve_right hne
      have h2 := (List.rel_of_pairwise_cons htail hxt).resolve_right fun h => hne h.symm
      exact absurd (ddCmp_gt_iff_lt.mpr h2) (h1 ▸ nofun)
  | case3 l h =>
    match l, h with
    | [], _ => exact .nil
    | [a], _ => exact List.pairwise_singleton _ a
    | a :: b :: t, h => exact absurd rfl (h a b t)

/-- Master sortedness lemma for `DSet.ofList`.  `dedupAdj`'s `BEq` is
structural, so value-equal structurally-distinct points can both survive —
hence the value-injectivity hypothesis (satisfied by canonical dyadics). -/
theorem sorted_ofList_uniq (l : List DD)
    (huniq : ∀ p ∈ l, ∀ q ∈ l, ddCmp p q = .eq → p = q) :
    (DSet.ofList l).Sorted := by
  refine (dedupAdj_pairwise
    ((List.pairwise_mergeSort ddCmp_le_trans ddCmp_le_total l).imp_of_mem
      fun {a b} ha hb hab => ?_)).isChain
  rcases h : ddCmp a b with _ | _ | _
  · exact .inl rfl
  · exact .inr (huniq a (List.mem_mergeSort.mp ha) b (List.mem_mergeSort.mp hb) h)
  · exact absurd h (bne_iff_ne.mp hab)

/-- Any two value-equal members of a strictly sorted set coincide. -/
theorem sorted_uniq {s : DSet} (hs : s.Sorted) :
    ∀ p ∈ s, ∀ q ∈ s, ddCmp p q = .eq → p = q := by
  intro p hp q hq heq
  by_contra hne
  haveI : Std.Symm (fun a b : DD => ddCmp a b = .lt ∨ ddCmp b a = .lt) :=
    ⟨fun _ _ h => h.symm⟩
  have hpw : s.Pairwise (fun a b => ddCmp a b = .lt ∨ ddCmp b a = .lt) :=
    (sorted_iff_pairwise.mp hs).imp fun {a b} h => Or.inl h
  rcases hpw.forall hp hq hne with h | h
  · rw [h] at heq; cases heq
  · rw [← ddCmp_gt_iff_lt, heq] at h; cases h

private theorem mem_union {x : DD} (s t : DSet) : x ∈ DSet.union s t → x ∈ s ∨ x ∈ t := by
  fun_induction DSet.union s t with
  | case1 t => exact .inr
  | case2 s _ => exact .inl
  | case3 a t₁ b t₂ hab ih =>
    intro h
    rcases List.mem_cons.mp h with rfl | h'
    · exact .inl List.mem_cons_self
    · exact (ih h').imp_left (List.mem_cons_of_mem a)
  | case4 a t₁ b t₂ hab ih =>
    intro h
    rcases List.mem_cons.mp h with rfl | h'
    · exact .inr List.mem_cons_self
    · exact (ih h').imp_right (List.mem_cons_of_mem b)
  | case5 a t₁ b t₂ hab ih =>
    intro h
    rcases List.mem_cons.mp h with rfl | h'
    · exact .inl List.mem_cons_self
    · exact (ih h').imp (List.mem_cons_of_mem a) (List.mem_cons_of_mem b)

@[simp] theorem pts_nil : DSet.pts [] = ∅ := by
  simp [DSet.pts]

@[simp] theorem pts_cons (a : DD) (l : List DD) :
    DSet.pts (a :: l) = insert a.toPt (DSet.pts l) := by
  rw [DSet.pts, show {p : DD | p ∈ a :: l} = insert a {p | p ∈ l} from
    Set.ext fun x => List.mem_cons, Set.image_insert_eq, DSet.pts]

@[simp] theorem DSet.pts_union (s t : DSet) :
    (s.union t).pts = s.pts ∪ t.pts := by
  fun_induction DSet.union s t with
  | case1 t => simp [pts_nil]
  | case2 s _ => simp [pts_nil]
  | case3 a t₁ b t₂ hab ih =>
    rw [pts_cons a (DSet.union t₁ (b :: t₂)), ih, pts_cons a t₁, Set.insert_union]
  | case4 a t₁ b t₂ hab ih =>
    rw [pts_cons b (DSet.union (a :: t₁) t₂), ih, pts_cons b t₂, Set.union_insert]
  | case5 a t₁ b t₂ hab ih =>
    rw [pts_cons a (DSet.union t₁ t₂), ih, pts_cons a t₁, pts_cons b t₂, Set.insert_union,
      Set.union_insert, ← (ddCmp_eq_iff a b).mp hab, Set.insert_idem]

theorem DSet.sorted_union (s t : DSet) (hs : s.Sorted) (ht : t.Sorted) :
    (s.union t).Sorted := by
  rw [sorted_iff_pairwise] at hs ht ⊢
  revert hs ht
  fun_induction DSet.union s t with
  | case1 t => exact fun _ ht => ht
  | case2 s _ => exact fun hs _ => hs
  | case3 a t₁ b t₂ hab ih =>
    intro hs ht
    refine (ih hs.of_cons ht).cons fun x hx => ?_
    rcases mem_union _ _ hx with h | h
    · exact List.rel_of_pairwise_cons hs h
    · rcases List.mem_cons.mp h with rfl | h'
      · exact hab
      · exact ddCmp_lt_trans hab (List.rel_of_pairwise_cons ht h')
  | case4 a t₁ b t₂ hab ih =>
    intro hs ht
    have hba := ddCmp_gt_iff_lt.mp hab
    refine (ih hs ht.of_cons).cons fun x hx => ?_
    rcases mem_union _ _ hx with h | h
    · rcases List.mem_cons.mp h with rfl | h'
      · exact hba
      · exact ddCmp_lt_trans hba (List.rel_of_pairwise_cons hs h')
    · exact List.rel_of_pairwise_cons ht h
  | case5 a t₁ b t₂ hab ih =>
    intro hs ht
    refine (ih hs.of_cons ht.of_cons).cons fun x hx => ?_
    rcases mem_union _ _ hx with h | h
    · exact List.rel_of_pairwise_cons hs h
    · exact ddCmp_eq_lt_trans hab (List.rel_of_pairwise_cons ht h)

/-- One chain insertion only ever adds the new point and drops old ones. -/
theorem addPoint_sublist (keep : Ordering) (p : DD) (m : List DD) :
    List.Sublist (addPoint keep p m) (p :: m) := by
  fun_induction addPoint keep p m with
  | case2 q hq => exact List.cons_sublist_cons.2 (List.nil_sublist [q])
  | case5 q r l hor ih =>
    exact ih.trans (List.cons_sublist_cons.2 (List.sublist_cons_self q (r :: l)))
  | case1 | case3 q hq | case4 q r l hor => exact List.Sublist.refl _

private theorem mem_chain_foldl {keep : Ordering} {s : List DD} {x : DD}
    (h : x ∈ s.foldl (fun acc p => addPoint keep p acc) []) : x ∈ s := by
  refine List.foldlRecOn s (fun acc p => addPoint keep p acc)
    (motive := fun acc => ∀ y ∈ acc, y ∈ s) (by simp) ?_ x h
  intro acc hacc a ha y hy
  rcases List.mem_cons.mp ((addPoint_sublist keep a acc).subset hy) with rfl | h'
  exacts [ha, hacc y h']

/-- `chainHull` of a strictly sorted (hence value-injective) set is sorted.
Unconditionally this is false: `dedupAdj`'s structural `BEq` can keep two
value-equal non-canonical points, so the sortedness hypothesis is needed. -/
theorem sorted_chainHull_of_sorted (s : DSet) (hs : s.Sorted) :
    (chainHull s).Sorted := by
  have hmem : ∀ y ∈ (s.foldl (fun acc p => addPoint .gt p acc) [] ++
      s.foldl (fun acc p => addPoint .lt p acc) []), y ∈ s :=
    fun y hy => (List.mem_append.mp hy).elim mem_chain_foldl mem_chain_foldl
  exact sorted_ofList_uniq _ fun p hp q hq => sorted_uniq hs p (hmem p hp) q (hmem q hq)

/-- A `.lt` comparison bounds the first coordinates (non-strictly). -/
private theorem ddCmp_lt_fst_le {a b : DD} (h : ddCmp a b = .lt) :
    a.1.toRat ≤ b.1.toRat := by
  rcases (ddCmp_lt_iff a b).mp h with h' | ⟨h', -⟩ <;> exact h'.le

/-- In a sorted set the head has the least first coordinate (this is what
`narrow` reads: the pair-lexicographic minimum's first coordinate bounds
every first coordinate from below). -/
theorem DSet.head_fst_le (s : DSet) (hs : s.Sorted) (lo : DD)
    (hlo : s.head? = some lo) (p : DD) (hp : p ∈ s) :
    lo.1.toRat ≤ p.1.toRat := by
  cases s with
  | nil => cases hlo
  | cons a t =>
    obtain rfl : a = lo := by simpa using hlo
    rcases List.mem_cons.mp hp with rfl | hp'
    · exact le_rfl
    · exact ddCmp_lt_fst_le (List.rel_of_pairwise_cons (sorted_iff_pairwise.mp hs) hp')

/-- ... and the last has the greatest first coordinate. -/
theorem DSet.fst_le_getLast (s : DSet) (hs : s.Sorted) (hi : DD)
    (hhi : s.getLast? = some hi) (p : DD) (hp : p ∈ s) :
    p.1.toRat ≤ hi.1.toRat := by
  rw [← List.head?_reverse] at hhi
  have hpw := List.pairwise_reverse.mpr (sorted_iff_pairwise.mp hs)
  have hp' := List.mem_reverse.mpr hp
  cases hrev : s.reverse with
  | nil => rw [hrev] at hhi; cases hhi
  | cons a t =>
    rw [hrev] at hhi hp' hpw
    obtain rfl : a = hi := by simpa using hhi
    rcases List.mem_cons.mp hp' with rfl | hp''
    · exact le_rfl
    · exact ddCmp_lt_fst_le (List.rel_of_pairwise_cons hpw hp'')

/-! ## State well-formedness -/

-- required by `List.isChain_iff_pairwise`
private instance : Trans (fun a b : Int × DSet => a.1 < b.1)
    (fun a b : Int × DSet => a.1 < b.1) (fun a b : Int × DSet => a.1 < b.1) :=
  ⟨fun h1 h2 => lt_trans h1 h2⟩

theorem wf_iff {st : State} :
    st.WF ↔ st.Pairwise (fun a b => a.1 < b.1) ∧ ∀ kv ∈ st, DSet.Sorted kv.2 :=
  and_congr List.isChain_iff_pairwise Iff.rfl

theorem state0_wf : state0.WF := by
  refine ⟨List.isChain_singleton _, fun kv hkv => ?_⟩
  obtain rfl := List.mem_singleton.mp hkv
  exact List.isChain_cons_cons.mpr ⟨by decide, List.isChain_singleton _⟩

theorem joinState_nil (k : Int) (v : DSet) :
    joinState k v ([] : State) = [(k, v)] := rfl

theorem joinState_cons (k : Int) (v : DSet) (k₁ : Int) (v₁ : DSet) (t : State) :
    joinState k v ((k₁, v₁) :: t) =
      if k < k₁ then (k, v) :: (k₁, v₁) :: t
      else if k = k₁ then (k₁, v.union v₁) :: t
      else (k₁, v₁) :: joinState k v t := rfl

/-- Everything in `joinState k v st` is the new entry, an old entry, or a
key-collision union with an old entry. -/
theorem mem_joinState {k : Int} {v : DSet} :
    ∀ {st : State} {p : Int × DSet}, p ∈ joinState k v st →
      p = (k, v) ∨ p ∈ st ∨ ∃ q ∈ st, p = (q.1, v.union q.2) := by
  intro st p
  fun_induction joinState k v st with
  | case1 => exact fun hp => .inl (List.mem_singleton.mp hp)
  | case2 k₁ v₁ t h => exact fun hp => (List.mem_cons.mp hp).imp id .inl
  | case3 v₁ t h =>
    intro hp
    rcases List.mem_cons.mp hp with rfl | hp'
    · exact .inr (.inr ⟨(k, v₁), List.mem_cons_self, rfl⟩)
    · exact .inr (.inl (List.mem_cons_of_mem _ hp'))
  | case4 k₁ v₁ t h1 h2 ih =>
    intro hp
    rcases List.mem_cons.mp hp with rfl | hp'
    · exact .inr (.inl List.mem_cons_self)
    · exact (ih hp').imp id (Or.imp (List.mem_cons_of_mem _)
        fun ⟨q, hq, hpq⟩ => ⟨q, List.mem_cons_of_mem _ hq, hpq⟩)

theorem joinState_pairwise (k : Int) (v : DSet) :
    ∀ st : State, st.Pairwise (fun a b : Int × DSet => a.1 < b.1) →
      (joinState k v st).Pairwise (fun a b : Int × DSet => a.1 < b.1) := by
  intro st
  fun_induction joinState k v st with
  | case1 => exact fun _ => List.pairwise_singleton _ _
  | case2 k₁ v₁ t h =>
    intro hst
    refine hst.cons fun b hb => ?_
    rcases List.mem_cons.mp hb with rfl | hb'
    · exact h
    · exact h.trans ((List.pairwise_cons.mp hst).1 b hb')
  | case3 v₁ t h =>
    exact fun hst => List.pairwise_cons.mpr (List.pairwise_cons.mp hst)
  | case4 k₁ v₁ t h1 h2 ih =>
    intro hst
    obtain ⟨hhead, htail⟩ := List.pairwise_cons.mp hst
    refine (ih htail).cons fun b hb => ?_
    rcases mem_joinState hb with rfl | hb' | ⟨q, hq, rfl⟩
    · exact show k₁ < k by omega
    · exact hhead b hb'
    · exact hhead q hq

theorem joinState_WF {k : Int} {v : DSet} (hv : v.Sorted) {st : State}
    (hst : st.WF) : (joinState k v st).WF := by
  refine ⟨(joinState_pairwise k v st (wf_iff.mp hst).1).isChain, fun p hp => ?_⟩
  rcases mem_joinState hp with rfl | hp' | ⟨q, hq, rfl⟩
  · exact hv
  · exact hst.2 p hp'
  · exact DSet.sorted_union v q.2 hv (hst.2 q hq)

/-! ### The three point transitions are the expected value maps -/

@[simp] theorem evenTrans_toPt (p : DD) :
    (evenTrans p).toPt = (p.toPt.1 / 2, p.toPt.2) := by
  simp [evenTrans, DD.toPt]

@[simp] theorem oddPosTrans_toPt (p : DD) :
    (oddPosTrans p).toPt = ((p.toPt.1 - p.toPt.2) / 2, p.toPt.1) := by
  simp [oddPosTrans, DD.toPt]

@[simp] theorem oddNonposTrans_toPt (p : DD) :
    (oddNonposTrans p).toPt = ((p.toPt.1 + p.toPt.2) / 2, p.toPt.2) := by
  simp [oddNonposTrans, DD.toPt]

theorem evenTrans_ne {p q : DD} (h : ddCmp p q ≠ .eq) :
    ddCmp (evenTrans p) (evenTrans q) ≠ .eq := by
  simp only [ne_eq, ddCmp_eq_iff', evenTrans, D.half_toRat] at h ⊢
  exact fun ⟨h1, h2⟩ => h ⟨by linarith, h2⟩

theorem oddPosTrans_ne {p q : DD} (h : ddCmp p q ≠ .eq) :
    ddCmp (oddPosTrans p) (oddPosTrans q) ≠ .eq := by
  simp only [ne_eq, ddCmp_eq_iff', oddPosTrans, D.half_toRat, D.sub_toRat] at h ⊢
  exact fun ⟨h1, h2⟩ => h ⟨h2, by linarith⟩

theorem oddNonposTrans_ne {p q : DD} (h : ddCmp p q ≠ .eq) :
    ddCmp (oddNonposTrans p) (oddNonposTrans q) ≠ .eq := by
  simp only [ne_eq, ddCmp_eq_iff', oddNonposTrans, D.half_toRat, D.add_toRat] at h ⊢
  exact fun ⟨h1, h2⟩ => h ⟨by linarith, h2⟩

/-- Mapping a value-injective transition over a sorted set keeps `ofList`
strictly sorted. -/
theorem sorted_ofList_map {f : DD → DD}
    (hf : ∀ p q : DD, ddCmp p q ≠ .eq → ddCmp (f p) (f q) ≠ .eq)
    {v : DSet} (hv : v.Sorted) : (DSet.ofList (v.map f)).Sorted := by
  refine sorted_ofList_uniq _ fun p hp q hq heq => ?_
  obtain ⟨p', hp', rfl⟩ := List.mem_map.mp hp
  obtain ⟨q', hq', rfl⟩ := List.mem_map.mp hq
  rcases eq_or_ne p' q' with rfl | hne
  · rfl
  · exact absurd heq (hf _ _ fun he => hne (sorted_uniq hv p' hp' q' hq' he))

/-! ### The two stages of `processDivstep`, named -/

theorem wf_nil : State.WF ([] : State) := ⟨List.isChain_nil, by simp⟩

/-- The fold body of `processDivstep` (definitionally equal to the
anonymous lambda in `Defs.lean`). -/
def pdStep (acc : State) (kv : Int × DSet) : State :=
  if 0 < kv.1 then
    joinState (2 - kv.1) (DSet.ofList (kv.2.map oddPosTrans))
      (joinState (2 + kv.1) (DSet.ofList (kv.2.map evenTrans)) acc)
  else
    joinState (2 + kv.1) (DSet.ofList (kv.2.map oddNonposTrans))
      (joinState (2 + kv.1) (DSet.ofList (kv.2.map evenTrans)) acc)

/-- The filter body of `processDivstep`, named. -/
def dropStep (M : Int) (kv : Int × DSet) : Option (Int × DSet) :=
  if narrow M kv.2 then none else some (kv.1, chainHull kv.2)

theorem processDivstep_eq (M : Int) (st : State) :
    processDivstep M st = (st.foldl pdStep []).filterMap (dropStep M) := rfl

theorem pdStep_WF {acc : State} (hacc : acc.WF) (kv : Int × DSet)
    (hv : kv.2.Sorted) : (pdStep acc kv).WF := by
  have h1 : (joinState (2 + kv.1) (DSet.ofList (kv.2.map evenTrans)) acc).WF :=
    joinState_WF (sorted_ofList_map (fun _ _ => evenTrans_ne) hv) hacc
  unfold pdStep
  split
  · exact joinState_WF (sorted_ofList_map (fun _ _ => oddPosTrans_ne) hv) h1
  · exact joinState_WF (sorted_ofList_map (fun _ _ => oddNonposTrans_ne) hv) h1

theorem foldl_WF :
    ∀ (l : List (Int × DSet)) (acc : State), (∀ kv ∈ l, DSet.Sorted kv.2) →
      acc.WF → (List.foldl pdStep acc l).WF
  | [], _, _, h => h
  | kv :: t, _, hl, h =>
    foldl_WF t _ (fun x hx => hl x (List.mem_cons_of_mem _ hx))
      (pdStep_WF h kv (hl kv List.mem_cons_self))

/-- `dropStep` only ever re-keys an entry to itself and re-hulls its set. -/
private theorem dropStep_eq_some {M : Int} {kv x : Int × DSet}
    (h : dropStep M kv = some x) : x = (kv.1, chainHull kv.2) := by
  rw [dropStep] at h
  split at h
  · cases h
  · exact (Option.some.inj h).symm

/-- The narrow-filter + re-hull step preserves both state invariants. -/
private theorem filterMap_narrow_wf (M : Int) (s1 : State)
    (hp : s1.Pairwise (fun a b => a.1 < b.1)) (hv : ∀ kv ∈ s1, DSet.Sorted kv.2) :
    ((s1.filterMap (dropStep M)).Pairwise (fun a b => a.1 < b.1)) ∧
    ∀ kv ∈ s1.filterMap (dropStep M), DSet.Sorted kv.2 := by
  refine ⟨List.pairwise_filterMap.mpr (hp.imp fun hab x hx y hy => ?_), fun kv hkv => ?_⟩
  · rw [dropStep_eq_some hx, dropStep_eq_some hy]; exact hab
  · obtain ⟨a, ha, hfa⟩ := List.mem_filterMap.mp hkv
    rw [dropStep_eq_some hfa]
    exact sorted_chainHull_of_sorted _ (hv a ha)

theorem processDivstep_wf (M : Int) (st : State) (h : st.WF) :
    (processDivstep M st).WF := by
  have hfold := wf_iff.mp (foldl_WF st [] h.2 wf_nil)
  rw [processDivstep_eq, wf_iff]
  exact filterMap_narrow_wf M _ hfold.1 hfold.2

end Hddivsteps
