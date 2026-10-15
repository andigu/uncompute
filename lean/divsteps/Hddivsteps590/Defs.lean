/-
# The computational specification

This module defines the two computations that the theorems are about:

* `certEmpty M n` — the convex-hull certificate: `n` reachability steps
  (`processDivstep`) over exact dyadic rationals, with each hull kept as
  a strictly ascending sorted vertex list.
* `hdRun n f g` — the half-delta divstep iteration (`hdStep`) whose step
  count the certificate bounds.

The definitions re-implement `hddivsteps_base.v` and `hddivsteps_def.v`
from sipa/safegcd-bounds (Coq, MIT). See README.md for the calibration
evidence that pins this re-implementation to the Coq semantics.

This module imports nothing. Edit it with care: any change makes the
next build re-run the 590-iteration certificate (about 4 minutes).
-/

namespace Hddivsteps

/-! ## Dyadic rationals, canonical form -/

/-- Dyadic rational `m * 2^e`, kept canonical: `m` odd, or `m = 0 ∧ e = 0`. -/
structure D where
  m : Int
  e : Int
deriving BEq

/-- Odd part and 2-adic valuation of a natural (`oddPart 0 = (0, 0)`). -/
def oddPart (n : Nat) : Nat × Nat :=
  if _h : n = 0 ∨ n % 2 = 1 then (n, 0)
  else
    let (m, k) := oddPart (n / 2)
    (m, k + 1)
termination_by n
decreasing_by omega

/-- Canonicalizing constructor: strip trailing zero bits of the mantissa. -/
def D.red (m e : Int) : D :=
  if m = 0 then ⟨0, 0⟩
  else
    let (om, k) := oddPart m.natAbs
    ⟨if m < 0 then -(om : Int) else (om : Int), e + (k : Int)⟩

/-- Value comparison: align exponents, compare mantissas. -/
def D.cmp (a b : D) : Ordering :=
  if a.e ≥ b.e then compare (a.m * 2 ^ (a.e - b.e).toNat) b.m
  else compare a.m (b.m * 2 ^ (b.e - a.e).toNat)

def D.add (a b : D) : D :=
  if a.e ≥ b.e then .red (a.m * 2 ^ (a.e - b.e).toNat + b.m) b.e
  else .red (a.m + b.m * 2 ^ (b.e - a.e).toNat) a.e

def D.sub (a b : D) : D :=
  if a.e ≥ b.e then .red (a.m * 2 ^ (a.e - b.e).toNat - b.m) b.e
  else .red (a.m - b.m * 2 ^ (b.e - a.e).toNat) a.e

def D.mul (a b : D) : D := .red (a.m * b.m) (a.e + b.e)

/-- Exact halving.  Preserves canonical form. -/
def D.half (a : D) : D := if a.m = 0 then ⟨0, 0⟩ else ⟨a.m, a.e - 1⟩

/-! ## Points and the chain hull (Andrew monotone chain, exact arithmetic) -/

/-- A point is a `(g, f)` pair of dyadics, ordered lexicographically by value. -/
abbrev DD := D × D

def ddCmp (p q : DD) : Ordering := (D.cmp p.1 q.1).then (D.cmp p.2 q.2)

/-- Sign of the cross product `(x1-x3)(y2-y3) - (y1-y3)(x2-x3)`, exact. -/
def orientation (p1 p2 p3 : DD) : Ordering :=
  D.cmp (D.mul (D.sub p1.1 p3.1) (D.sub p2.2 p3.2))
        (D.mul (D.sub p1.2 p3.2) (D.sub p2.1 p3.1))

/-- One monotone-chain insertion step (upper chain for `keep = .gt`, lower
chain for `keep = .lt`); points arrive in ascending value order, the
accumulator holds the chain newest-first (descending). -/
def addPoint (keep : Ordering) (p : DD) : List DD → List DD
  | [] => [p]
  | [q] => if ddCmp p q == .eq then [p] else [p, q]
  | q :: r :: l =>
    if orientation p q r == keep then p :: q :: r :: l
    else addPoint keep p (r :: l)

/-- A point set: strictly ascending sorted list (dedup by value). -/
abbrev DSet := List DD

/-- Remove adjacent duplicates of a sorted list. -/
def dedupAdj : List DD → List DD
  | a :: b :: t => if a == b then dedupAdj (b :: t) else a :: dedupAdj (b :: t)
  | l => l

/-- Sort by value and dedup: turns any list of points into a `DSet`. -/
def DSet.ofList (l : List DD) : DSet :=
  dedupAdj (l.mergeSort (fun a b => ddCmp a b != .gt))

/-- Union of two sorted sets, by merge. -/
def DSet.union : DSet → DSet → DSet
  | [], l => l
  | l, [] => l
  | a :: t₁, b :: t₂ =>
    match ddCmp a b with
    | .lt => a :: DSet.union t₁ (b :: t₂)
    | .gt => b :: DSet.union (a :: t₁) t₂
    | .eq => a :: DSet.union t₁ t₂
termination_by s₁ s₂ => s₁.length + s₂.length

/-- The computational convex hull: fold the upper then the lower monotone
chain over the points in ascending value order, and re-form the set.
(Named `chainHull` to leave `convexHull` to mathlib.) -/
def chainHull (s : DSet) : DSet :=
  DSet.ofList (s.foldl (fun acc p => addPoint .gt p acc) [] ++
               s.foldl (fun acc p => addPoint .lt p acc) [])

/-! ## The hull reachability step (`processDivstep`) -/

def oddPosTrans (p : DD) : DD := (D.half (D.sub p.1 p.2), p.1)
def oddNonposTrans (p : DD) : DD := (D.half (D.add p.1 p.2), p.2)
def evenTrans (p : DD) : DD := (D.half p.1, p.2)

/-- All vertex first coordinates strictly inside `(-1/M, 1/M)`?
`s` is sorted, so min/max under the pair order are head/last. -/
def narrow (M : Int) (s : DSet) : Bool :=
  match s.head?, s.getLast? with
  | some lo, some hi =>
    (D.cmp ⟨-1, 0⟩ (D.mul (.red M 0) lo.1) == .lt) &&
    (D.cmp (D.mul (.red M 0) hi.1) ⟨1, 0⟩ == .lt)
  | _, _ => true

/-- The state: a map from the (half-delta) delta key to a point set, as an
ascending sorted association list. -/
abbrev State := List (Int × DSet)

/-- Insert `(k, v)` joining by set-union on key collision. -/
def joinState (k : Int) (v : DSet) : State → State
  | [] => [(k, v)]
  | (k', v') :: t =>
    if k < k' then (k, v) :: (k', v') :: t
    else if k = k' then (k', v.union v') :: t
    else (k', v') :: joinState k v t

/-- One hull divstep: each key emits its even-branch and odd-branch images
(`delta' = 2 + delta`, or the half-delta swap `delta' = 2 - delta` when
`delta > 0`), entries are joined by key with set union, then keys whose
hull is certified narrow are dropped and survivors are re-hulled. -/
def processDivstep (M : Int) (s : State) : State :=
  let s1 : State := s.foldl (init := []) fun acc (k, v) =>
    let acc := joinState (2 + k) (DSet.ofList (v.map evenTrans)) acc
    if 0 < k then joinState (2 - k) (DSet.ofList (v.map oddPosTrans)) acc
    else joinState (2 + k) (DSet.ofList (v.map oddNonposTrans)) acc
  s1.filterMap fun (k, v) => if narrow M v then none else some (k, chainHull v)

/-- Seed: the `(g, f)` pair set `{(0,1), (1,1)}` at delta key 1. -/
def state0 : State := [(1, [(⟨0, 0⟩, ⟨1, 0⟩), (⟨1, 0⟩, ⟨1, 0⟩)])]

/-- The certificate computation: is the state empty after `n` hull divsteps? -/
def certEmpty (M : Int) (n : Nat) : Bool :=
  (Nat.repeat (processDivstep M) n state0).isEmpty

/-- `M*`: the threshold of the 590-step bound; exceeds `2^256 - 2^32 - 977`. -/
def Mstar : Int :=
  0x101c840faed222b9ef3bfe933011b277f84ee3d3f6dc8b496d8cd75b8315afec2

/-! ## The concrete integer iteration (transcribed from `hddivsteps.step`)

All divisions below are exact — every numerator is even by construction
(`g` even in the even branch; `g ± f` even for odd `g`, odd `f`; `x + M`
even for odd `x`, odd `M`) — so Lean's `Int./` agrees with Coq's `Z.div`
on every division performed.  Likewise `x % 2 = 1 ↔ Z.odd x`. -/

structure HDState where
  delta : Int
  f : Int
  g : Int
  d : Int
  e : Int
  modulus : Int

def hdInit (f g : Int) : HDState :=
  { delta := 1, f := f, g := g, d := 0, e := 1, modulus := f }

def div2M (M x : Int) : Int := (if x % 2 = 1 then x + M else x) / 2

/-- One half-delta divstep (`INC = 2`). -/
def hdStep (s : HDState) : HDState :=
  if s.g % 2 = 0 then
    { s with delta := 2 + s.delta
           , g := s.g / 2
           , e := div2M s.modulus s.e }
  else if 0 < s.delta then
    { delta := 2 - s.delta
    , f := s.g
    , g := (s.g - s.f) / 2
    , d := s.e
    , e := div2M s.modulus (s.e - s.d)
    , modulus := s.modulus }
  else
    { s with delta := 2 + s.delta
           , g := (s.g + s.f) / 2
           , e := div2M s.modulus (s.e + s.d) }

/-- `n` divsteps from `(f, g)`. -/
def hdRun (n : Nat) (f g : Int) : HDState := Nat.repeat hdStep n (hdInit f g)

end Hddivsteps
