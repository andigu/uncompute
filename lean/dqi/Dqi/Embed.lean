import Dqi.Basic

/-!
# From patterns to syndromes

The development is written in two coordinate systems.  The weight-shell material
(`Shell`, `Residual`) is indexed by error *patterns*, because "weight `k`" is a
statement about a pattern.  The failure model (`Failure`) is indexed by what the
circuit actually holds — a syndrome, plus ancilla and record.  The appendix does
the same thing, calling pattern space "only a proof device"; in Lean the two are
different types, so the slide between them has to be a function.

This file supplies the piece of that translation the reachable-residual argument
actually needs, and it turns out to be a single disjointness fact:

`disjoint_decodableSyn_edgeSyn` — at strict decoding distance, **no syndrome is
reachable both from a decodable pattern and from a pattern one Hamming step
outside the decodable set.**

That is the whole content of "a sign or leakage error keeps a decodable syndrome,
so it cannot overlap the escaping weight layer".  The error sits at decodable
syndromes; the escaping layer sits at edge syndromes; the two sets are disjoint,
so `Channel.norm_inner_le_of_avoids` applies and the error is blind to the edge.

Note the distance hypothesis is exactly `2 * l + 1 < dualDist B` — strong enough
to separate the decodable set from the next layer, but *not* strong enough to
rule out aliasing *within* the next layer, which needs `2 * l + 2 < dualDist B`
(`Pullback.eq_of_syndrome_eq_succ`).  That gap is not an artefact: the Hermitian
instance sits at `dualDist = 2 * l + 2` exactly, so its edge layer really does
contain collisions, which is why the relocation term carries a collision count.
-/

namespace Dqi

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]
variable {m n : ℕ}

/-- The syndromes reachable from patterns the decoder handles. -/
noncomputable def decodableSyn (B : Matrix (Fin m) (Fin n) K) (l : ℕ) :
    Finset (Fin n → K) :=
  (Finset.univ.filter fun y : Pattern m K => hammingNorm y ≤ l).image (syndrome B)

/-- The syndromes reachable from patterns exactly one Hamming step outside the
decodable set — the layer the objective can escape into. -/
noncomputable def edgeSyn (B : Matrix (Fin m) (Fin n) K) (l : ℕ) :
    Finset (Fin n → K) :=
  (Finset.univ.filter fun y : Pattern m K => hammingNorm y = l + 1).image (syndrome B)

theorem mem_decodableSyn {B : Matrix (Fin m) (Fin n) K} {l : ℕ} {u : Fin n → K} :
    u ∈ decodableSyn B l ↔ ∃ y : Pattern m K, hammingNorm y ≤ l ∧ syndrome B y = u := by
  simp [decodableSyn]

theorem mem_edgeSyn {B : Matrix (Fin m) (Fin n) K} {l : ℕ} {u : Fin n → K} :
    u ∈ edgeSyn B l ↔ ∃ y : Pattern m K, hammingNorm y = l + 1 ∧ syndrome B y = u := by
  simp [edgeSyn]

/-- **The decodable syndromes and the edge syndromes are disjoint.**

If one syndrome were reachable from both a decodable pattern `y` and a
weight-`(l+1)` pattern `y'`, their difference would be a nonzero dual codeword of
weight at most `2 * l + 1`, contradicting the distance hypothesis.

This is the formal content of "sign and leakage errors cannot overlap the
next-layer residual": those channels keep a decodable syndrome, and the escaping
component lives entirely on edge syndromes. -/
theorem disjoint_decodableSyn_edgeSyn {B : Matrix (Fin m) (Fin n) K} {l : ℕ}
    (hstrict : 2 * l + 1 < dualDist B) :
    Disjoint (decodableSyn B l) (edgeSyn B l) := by
  rw [Finset.disjoint_left]
  intro u hu hu'
  obtain ⟨y, hy, hsy⟩ := mem_decodableSyn.mp hu
  obtain ⟨y', hy', hsy'⟩ := mem_edgeSyn.mp hu'
  have hsyn : syndrome B y' = syndrome B y := by rw [hsy, hsy']
  have hdual : IsDualCodeword B (y' - y) := isDualCodeword_sub_of_syndrome_eq hsyn
  have htri : hammingNorm (y' - y) ≤ hammingNorm y' + hammingNorm y := hammingNorm_sub_le y' y
  rcases eq_or_ne (y' - y) 0 with hz | hz
  · -- equal patterns cannot have different weights
    have : y' = y := by rwa [sub_eq_zero] at hz
    rw [this] at hy'
    omega
  · have hlow : dualDist B ≤ hammingNorm (y' - y) := dualDist_le_hammingNorm hz hdual
    omega

end Dqi
