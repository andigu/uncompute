# Formal error bounds for DQI decoders

This directory contains a Lean 4 formalization of a decoder-error bound for
decoded quantum interferometry (DQI), the optimization framework introduced by
Jordan et al. (2025), arXiv:2408.08292.
It supports the DQI decoder-error bound of the accompanying paper (arXiv: TBD).
The central result here is that decoder failures with total probability mass
`μ` change the expected DQI objective by order `μ`, rather than by the generic
order `√μ` expected from a state perturbation of norm `√μ`.

The improvement comes from the structure of the DQI state and objective. For
the exact DQI weight profile, the state is an eigenvector of the objective
after restriction to the subspace of decodable error patterns. Consequently,
perturbations that remain in that subspace have no first-order effect on the
expectation value. The formalization tracks the small residual introduced by
the implemented profile explicitly. It also proves that the objective sends
the state outside the decodable subspace in only one direction: the next
Hamming-weight layer. Sign errors and record leakage cannot reach that
direction, while syndrome relocation can do so only through a separately
bounded term. The formal development assembles these facts into a bound for
the full physical-register model.

The main entry point is `Dqi.dqi_end_to_end` in
[`Dqi/Complete.lean`](Dqi/Complete.lean). Readers interested primarily in the
statement and its assumptions can use this README as a guide. Lean checks
the proofs themselves.

## Setting

DQI addresses finite-field constraint problems by preparing a superposition
over low-weight error patterns `y` and then interfering the corresponding
syndromes. The decoder erases, or phase-corrects, the information about `y`
while it retains the syndrome. Correct decoding is therefore needed
for the intended interference pattern and hence for the expected value of the
optimization objective. The space-efficient, measurement-based construction
modeled here follows Khattar et al. (2025), arXiv:2510.10967.

The formal model starts immediately before this decoder operation. Its basis
states are patterns `y` of Hamming weight at most the decoding radius `l`,
and their squared amplitudes define the distribution from which decoder
validation samples. The ideal oracle attaches the correct syndrome and a clean
record to each branch. The implemented oracle may instead change a phase,
leave a non-clean record, or move the branch to another syndrome. The proof
determines how the probability mass in these three failure channels can change
the final objective.

## Main bound

Let `λ` denote the ideal expectation of the DQI objective, whose operator norm
is at most the number of constraints `m`. The Lean objective is the shifted
operator `∑ᵢ (2Pᵢ − 1)`. It scores `+1` per satisfied constraint and `−1` per
violated one, so `λ = 2⟨f⟩ − m`, where `⟨f⟩` is the expected number of
satisfied constraints. Let:

- `μ` be an upper bound on the total probability mass of failing decoder
  branches;
- `μ_cl` be an upper bound on the mass of *clean-relocation* failures, which
  move a branch to a different clean syndrome label while retaining the ideal
  record state;
- `σ_dec` be the norm of the chosen profile's residual within the decodable
  subspace; and
- `E` be a computed upper bound on the reach of clean relocations into the
  first layer outside the decodable subspace.

Then `Dqi.dqi_end_to_end` proves

```text
|<f>_impl - λ| ≤ (|λ| + m) * 4μ + 4√μ * σ_dec + 2√μ_cl * E.
```

The first term is quadratic in the norm `√μ` of the erroneous part of the
state. The second accounts for the fact that the implemented weight profile
need not be an exact eigenvector. The third is the only surviving first-order
cross term, and only clean relocation contributes to it. The theorem does not
leave `E` as an abstract parameter: it derives

```text
E = √|F_cl| * σ_edge * n_coll / √N_(l+1),
```

where `F_cl` is the set of clean-relocation branches, `σ_edge` is the
amplitude in the escaping direction, `n_coll` bounds the number of
weight-`l + 1` patterns with a common syndrome, and `N_(l+1)` is the size
of that weight layer.

For the measurement-based decoder analyzed in the paper, an implementation
error changes a branch phase. When a reset clears a workspace qubit that still
holds branch-dependent data, the error also leaves a non-clean record. Neither
error changes the syndrome label, so neither creates a clean relocation.
Therefore `μ_cl = 0` and the third term vanishes. The remaining terms are the
quadratic term and the profile residual. For Reed–Solomon, the boundary
correction described below widens the profile residual.

## Why the bound is linear in failure probability

A generic argument sees only that changing branches of total mass `μ` changes
the state vector by order `√μ`. Applying an operator-norm bound directly would
therefore give an order-`√μ` change in the expectation value.

The DQI state has additional protection. Within the decodable subspace, the
exact-profile state is an eigenvector of the shell objective. If an ideal
normalized state `ψ` is perturbed by `δ` while remaining in that subspace,
the term linear in `δ` vanishes in the expectation of the objective; only a
term proportional to `||δ||²`, hence to `μ`, remains. This is the same
stationary-point mechanism that makes a variational energy second-order in
wavefunction error. The `σ_dec` term quantifies the departure of the chosen
profile from this exact situation.

The formal proof also treats the fact that the shell objective can map `ψ`
outside the decodable subspace. `Dqi.shellOp_dqiState_residual` proves that
the escaping component lies entirely in the weight-`l + 1` layer. The
failure-channel analysis then proves:

- a sign error changes a branch phase but not its label or record;
- a leakage error leaves a record state orthogonal to the clean record; and
- a relocation error changes the clean syndrome label.

The first two channels are exactly orthogonal to the escaping component. Only
clean relocation can overlap it, producing the `2√μ_cl E` term above.

## Assumptions of the end-to-end theorem

The theorem separates the algebra proved in Lean from code parameters,
circuit semantics, and measured quantities supplied by the paper. Its
hypotheses are as follows.

| Assumption | Role in the proof | Source for the task instances | Lean parameters |
|---|---|---|---|
| **Strict dual-distance condition:** `2l + 1 < d⊥` | Prevents a decodable pattern and a pattern in the adjacent weight layer from sharing a syndrome. Lean defines `Dqi.dualDist` fail-closed as the minimum weight of a nonzero dual codeword. | The Hermitian-code construction gives `d⊥ ≥ 96`. Gu and Jordan (2025), arXiv:2510.06603, describe the code family. The Reed–Solomon instance has `d⊥ = 71` by the MDS property and is treated by the boundary theorem below. | `hstrict : 2 * l + 1 < dualDist B` |
| **Edge collision bound:** at most `n_coll` patterns of weight `l + 1` have any given syndrome | Bounds how strongly a relocated branch can overlap the escaping layer. | This is an instance-specific combinatorial input. Lean accepts it as a hypothesis rather than deriving it from the code parameters. | `ncoll`, `hcoll` |
| **Bent constancy:** each constraint has equal nonzero Fourier magnitudes | Makes the amplitudes within a weight layer flat. The common magnitude `(q - 1)^(-1/2)` is derived in `Dqi.norm_ghat_of_const`, not assumed. | The target sets of the task instances use the Maiorana–McFarland construction of Khattar et al. (2025), arXiv:2510.10967. | `hconst` |
| **Normalized weight profile:** real weights on layers `0, ..., l` with unit squared norm | Specifies the implemented, possibly rounded, DQI layer profile. | Instance data used by the implementation. | `w`, `hw` |
| **Branch-diagonal oracle model** | Represents both ideal and implemented decoders as maps that send each input branch to one output branch with a phase and record state. The ideal map supplies the correct syndrome and a fixed clean record; the implemented record is either clean or orthogonal to clean; both maps are isometries and agree away from the failure set. | Reversible gates permute computational-basis states, while outcome-conditioned corrections are diagonal phases. The measurement-based construction originates with Khattar et al. (2025), arXiv:2510.10967. | `hU`, `hV`, `hUiso`, `hViso`, `hUlbl`, `hUrec`, `horth`, `hagree` |
| **Measured failure masses:** `mass Psi_pre F ≤ μ` and clean-relocation mass `≤ μ_cl` | Connects the deterministic theorem to the validation experiment. These are masses in the pre-oracle pattern distribution, exactly the distribution sampled by validation. | Validation bounds `μ` from observed failures with a one-sided exact binomial interval. For the decoder studied here, the circuit model rules out relocation, so `μ_cl = 0`; a different implementation could instead supply a separately measured bound on `μ_cl`. | `mu`, `mucl`, `hmu`, `hmucl` |

All oracle assumptions are restricted to the decodable subspace, meaning
states supported on patterns of weight at most `l`. This restriction is
mathematically necessary. A global ideal-labeling isometry cannot exist when a
nonzero dual codeword has the same syndrome as the zero pattern: it would have
to map two orthogonal inputs to the same labeled branch. The concrete witness
in [`Dqi/Witness.lean`](Dqi/Witness.lean) confirms that the restricted
assumptions are jointly satisfiable.

The tridiagonal action of the objective on Hamming-weight layers is not an
assumption. It is proved at operator level by `Dqi.shellOp_layerState`, and
`Dqi.inner_dqiState_shellOp` recovers the quadratic form used in the paper.

### Reed–Solomon boundary regime

The end-to-end theorem above uses the strict inequality `2l + 1 < d⊥`. The
Reed–Solomon instance lies at the boundary `2l + 1 = d⊥` and therefore uses
the variant `Dqi.dqi_degradation_boundary`. There the norm `R_C` of a
boundary defect enters as a hypothesis. This development proves the
combinatorial ingredients used to bound it—the compatible-codeword packing
bound and the conversion from counts to an operator-norm bound—but does not
construct the defect operator. A computation outside this project gives the
numerical value of `R_C` for the `dqi-reed-solomon` task instance.

## Principal formal results

- `Dqi.dqi_end_to_end` ([`Dqi/Complete.lean`](Dqi/Complete.lean)) is the
  terminal theorem. It derives normalization, the failure-channel split,
  blindness of the sign and leakage channels, the residual decomposition, the
  relocation reach, and the sharp cross term from the assumptions listed
  above.
- `DqiWitness.witness_end_to_end`
  ([`Dqi/Witness.lean`](Dqi/Witness.lean)) instantiates every hypothesis with
  `l = 1`, `K = 𝔽₂`, and a matrix whose dual is the `[7,3]` simplex code
  (equivalently, the dual of the `[7,4]` Hamming code). Lean checks that its
  dual distance is `4`. This result establishes non-vacuity; it is not one of
  the task instances.
- `Dqi.shellOp_layerState` proves that the objective connects only adjacent
  Hamming-weight layers and gives the tridiagonal coefficients.
- `Dqi.shellOp_dqiState_residual` decomposes the action of the objective into
  its component inside the decodable subspace and the single escaping
  direction.
- `Dqi.expectation_perturbation_bound` is the abstract perturbation theorem:
  the expectation changes at second order except for a cross term involving
  the reachable residual.
- `Dqi.degradation_channels` resolves the perturbation by failure channel and
  proves exact blindness for sign and leakage errors.
- `Dqi.dqi_degradation_strict` specializes the abstract theorem to the DQI
  shell objective, a self-adjoint operator of norm at most `m`.
- `Dqi.norm_restr_imageBlock` identifies physical-register measurement mass
  with the mass of the corresponding pre-oracle branches.

## Scope of the formalization

The Lean development proves deterministic mathematical implications from the
stated hypotheses. It deliberately does not formalize the following inputs:

- **Circuit-to-oracle correspondence.** The paper argues that the physical
  decoder satisfies the branch-diagonal model. Lean takes
  `IsBranchDiagonal` and the associated restricted-subspace properties as
  hypotheses.
- **Statistical inference.** Detection probabilities and the
  Clopper–Pearson conversion from finite validation data to `μ` and `μ_cl`
  lie outside Lean.
- **Instance arithmetic.** The numerical evaluation of `σ_dec`, relocation
  reach, `R_C`, and the final percentage losses is outside this Lean project.
  This README does not independently certify those instance-specific values.
- **The boundary defect operator.** As noted above, Lean proves the
  combinatorial core of the `R_C` estimate but takes the resulting operator
  bound as a hypothesis.

These boundaries matter when citing the artifact: Lean verifies the
degradation theorem and its internal algebra, while the paper supplies the
physical-model justification, experimental confidence bounds, code facts, and
instance-specific numerical evaluation.

## Checking the development

The project pins Lean to `leanprover/lean4:v4.33.1` in `lean-toolchain` and
Mathlib to the revision recorded in `lake-manifest.json`. From the repository
root, run:

```sh
cd lean/dqi
lake exe cache get
lake build
```

The build should finish without errors or warnings. `Dqi/Witness.lean` also
contains an axiom check:

```lean
#print axioms DqiWitness.witness_end_to_end
-- [propext, Classical.choice, Quot.sound]
```

Thus the terminal instantiated result uses only Lean's standard axioms: it has
no dependency on `sorryAx` or on an additional `axiom` declaration. The
source contains no `sorry` and no linter suppressions.

## Module guide

| Module | Purpose |
|---|---|
| `Basic` | Error patterns, syndromes, dual distance, and injectivity on the decodable set |
| `Failure` | Physical registers, branch labels, branch-diagonal maps, failure sets, and probability mass |
| `Characters` | Additive-character orthogonality over a finite field |
| `Fourier`, `Flat` | DQI Fourier amplitudes, normalization, and the derived flat-magnitude property |
| `LayerState`, `Amplitudes` | Orthonormal Hamming-weight layer states and their relation to the DQI amplitude family |
| `SiteOp`, `Shell` | Single-site action and the tridiagonal shell identity |
| `Residual` | The residual decomposition and the quantities `lamF` and `sigmaEdge` |
| `ShellOp`, `Objective` | The objective as a bounded self-adjoint operator and its character-shift structure |
| `Pullback`, `Embed` | Strict-distance support facts and separation of decodable and edge syndromes |
| `Reachable`, `Channel` | Coordinate restrictions, failure-channel blindness, and splitting bounds |
| `Counting`, `Packing`, `Boundary` | Shell sizes, collision estimates, packing, and boundary-regime combinatorics |
| `Perturbation`, `Degradation` | Abstract and channel-resolved expectation perturbation bounds |
| `Bridge` | Branch-to-register indexing and measurement-mass identification |
| `Main`, `MainBoundary` | The strict-distance master theorem and the boundary variant |
| `FourierN`, `Phys`, `Intertwine` | Multivariable Fourier transform, the physical objective, and its pullback to pattern space |
| `Taxonomy` | Exclusive partition into relocation, leakage, and sign failures |
| `PhysFull` | The objective on the full register space and the clean-register embedding |
| `Ewire`, `CrossPhys` | Computed relocation reach and the sharp physical-register cross term |
| `Complete` | The end-to-end theorem `Dqi.dqi_end_to_end` and the full-register operator `synOpFullL` |
| `Witness` | A concrete Lean-checked instantiation of every end-to-end hypothesis |
