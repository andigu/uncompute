# Formal bounds and branch recovery for half-delta divsteps

This directory contains a Lean 4 development for the half-delta divstep
algorithm used to compute modular inverses. It certifies two facts needed by
the reversible modular-inversion circuit for the prime
`p = 2^256 - 2^32 - 977` (see the accompanying paper, arXiv:2610.22367):

1. every canonical field input reaches a terminal state within 590
   iterations; and
2. each divstep branch can be reconstructed from the state after the step,
   without retaining a complete history of branch decisions.

These results serve different circuit-level purposes. The termination theorem
gives a 590-step worst-case bound. A circuit on an input in superposition
cannot stop at an input-dependent time, so it needs such a fixed bound. The
`modinv-point-add` circuit schedules 533 steps and charges the rare longer
walks to its failure budget. The branch-recovery theorems justify erasing
branch controls from information already present in the output state. The
depth-two identity that the circuit uses reduces this recovery to two
low-order bits.

The development therefore certifies the mathematical obligations behind the
step bound and the recovery logic. It does not formalize the complete
quantum circuit or its resource count.

## Setting

A divstep updates a state containing an integer pair `(f, g)`, a counter, and
coefficient rows that track the transformation applied to the original input.
When `g` reaches zero, `|f|` is the greatest common divisor; on coprime
inputs, the coefficient rows yield a modular inverse. Divsteps originate with
Bernstein and Yang (2019), doi:10.46586/tches.v2019.i3.340-398.

This project uses the *half-delta* variant, or `hddivsteps`, which replaces the
ordinary counter updates `1 ± delta` with `2 ± delta`. The recurrence and the
convex-hull method that bounds it follow Wuille (2021),
<https://github.com/sipa/safegcd-bounds>.

Two equivalent state conventions appear in the proof:

- `hdRun n f g` is the `delta`-form recurrence used by the termination
  certificate and modular-inverse proof. It starts from
  `(delta, f, g, d, e) = (1, f, g, 0, 1)`.
- `zWalk p x n` is the frozen `z`-form recurrence used for branch recovery.
  It stops changing once `g = 0` and names the coefficient rows `fv` and
  `gv`.

The bridge theorem `Hddivsteps.zWalk_hdRun` proves that the two walks have
identical `f` and `g` trajectories and that `delta = 2z + 1` before
termination.

## Principal formal results

The terminal results are in
[`Hddivsteps590/Main.lean`](Hddivsteps590/Main.lean).

- `zWalk590` proves that, for every integer
  `0 ≤ x < 2^256 - 2^32 - 977`, the frozen walk satisfies
  `(zWalk fieldPrime x 590).g = 0`.
- `zWalk590_bound` gives the generic form: if `f` is odd,
  `f ≤ Mstar`, and `0 ≤ g ≤ f`, then `(zWalk f g 590).g = 0`.
  The theorem `fieldPrime_lt_Mstar` proves that the 257-bit threshold
  `Mstar` exceeds `p`.
- `hddivsteps590_prime_inverse` combines termination with the number theory.
  For an odd prime `f ≤ Mstar` and `0 < g < f`, the final state has
  `|f_end| = 1` and

  ```text
  d_end * f_end * g ≡ 1 (mod f).
  ```

  If `g = 0`, it instead proves that the carried coefficient `d` remains
  zero.

The branch-recovery results are in
[`Hddivsteps590/Recovery.lean`](Hddivsteps590/Recovery.lean).

- `det_zWalk`, `parity_zWalk`, and `scale_zWalk` establish the determinant,
  parity, and scaling invariants of the coefficient rows.
- `endpoint_zWalk`, `inverse_zWalk`, and `inverse_zWalk_unscaled` show how
  the terminal rows encode the scaled and ordinary modular inverses.
- `sigma_class`, `bbit_of_swap`, `bbit_of_ramp`, and
  `bbit_of_freeze` recover the locally visible branch classes.
- `window_zWalk` proves that the remaining negative-counter nonswaps leave a
  clean low-bit window in the coefficient rows.
- `ladder` and `ladder_det` give two equivalent formulas for recovering the
  ambiguous branch bit from that window.
- `finite_dependence` proves that, at depth `j`, one fixed decoder using two
  `j`-bit residues works for every reachable state.
- `j2_law` specializes the decoder at depth two to

  ```text
  b = gv'[1] XOR fv'[3].
  ```

  This is the two-bit identity used by the circuit's two-Toffoli decoder.

## Assumptions and theorem boundaries

| Result family | Assumptions | What is concluded |
|---|---|---|
| Generic 590-step bound | `f` is odd, `f ≤ Mstar`, and `0 ≤ g ≤ f` | The `f, g` trajectory has reached `g = 0` by iteration 590. |
| Concrete step bound for `p` | `0 ≤ x < fieldPrime` | The concrete frozen walk reaches `g = 0` by iteration 590. |
| Generic inverse theorem | `f` is an odd prime with `f ≤ Mstar`, and `0 < g < f` | The final coefficient represents the inverse of `g` modulo `f`, up to the explicitly included sign `f_end`. |
| Recovery invariants | The modulus `p` is odd and the state is reachable from the prescribed seed | The post-step state determines the swap class and branch bit through the stated local or low-bit formulas. |
| Computational certificate | The closed Boolean computation `certEmptyPar Mstar 590` evaluates to `true` | By the proved soundness chain, every integer trajectory in the generic input range terminates within 590 iterations. |

The concrete theorem `zWalk590` is a termination theorem and does
not need primality. The project does not separately prove that
`2^256 - 2^32 - 977` is prime. Applying the generic inverse theorem to that
modulus therefore requires the usual primality fact as an additional input.

## Why branch recovery matters

A straightforward reversible implementation would record every conditional
choice made by the forward recurrence and retain those bits until the
computation is reversed. That history costs workspace. This development proves
that the relevant choices are already encoded in the coefficient rows after
each step.

Three cases are visible immediately: whether the step swapped, whether a
non-swap occurred with a nonnegative counter, and whether the step terminated.
The only ambiguous case is a non-swap entered with a negative counter. For
that case, the window invariant shows that the normalized low bits of `gv'`
act as a shift register of earlier decisions. If
`j = v₂(fv')` and `q = fv' / 2^j`, the newest branch bit is

```text
bit_(j-1)(gv' * q^(-1) mod 2^j).
```

The determinant invariant supplies an equivalent formula involving only the
fixed modulus, `f'`, and `q`. Consequently, the recovery logic depends on
bounded low-order residues rather than on the full-width coefficient rows.

## Proof architecture

The termination and recovery arguments are developed independently and joined
only at the end.

### Termination certificate

`Hddivsteps.example590` states

```text
certEmpty Mstar 590 = true.
```

The computation tracks, for each counter value, a convex hull of scaled
`(g, f)` points using exact dyadic arithmetic. Each divstep maps those hulls
through the possible linear branch transformations. The `narrow` test removes
a hull once every integer trajectory it covers must have `g = 0`. After 590
iterations no hull remains.

The Boolean computation alone is not the theorem. Its soundness is proved in
three layers:

1. `Chain.lean` proves that the monotone-chain implementation covers all
   points in the input hull.
2. `Invariant.lean` proves that every concrete integer trajectory remains
   covered by the hull at its counter key and that a discarded narrow hull can
   contain only trajectories with `g = 0`.
3. `NumberTheory.lean` proves that termination yields the gcd and, for
   coprime inputs, the modular inverse.

### Branch recovery

`Recovery.lean` develops the determinant, parity, scaling, and window
invariants directly over `zWalk`. It then derives the local branch tests,
the general low-bit recovery formula, finite dependence, and the depth-two
identity.

`zWalk_hdRun` connects the two proof threads. `Main.lean` transfers the
certificate-backed termination result to `zWalk` and packages the generic
and concrete-prime conclusions.

## Trust base

The algebraic, number-theoretic, bridge, and branch-recovery theorems use only
the standard axioms reported by Mathlib:

- `propext`;
- `Classical.choice`; and
- `Quot.sound`.

The 590-step theorems additionally depend on the `native_decide` axiom
generated for `Hddivsteps.example590_par`. This axiom records that compiled
evaluation of the closed certificate expression returned `true`. Accepting
the certificate-backed results therefore requires trusting Lean's compiler and
runtime for that computation, in addition to the ordinary Lean kernel and
Mathlib trust base.

[`Hddivsteps590/Main.lean`](Hddivsteps590/Main.lean) ends with
`#print axioms` commands for each public result. The project contains no
`sorry` and introduces no other axioms.

## Relationship to the original certificate

The 590-step certificate and its soundness proof follow the Coq development of
Wuille (2021), <https://github.com/sipa/safegcd-bounds>.
[`Hddivsteps590/Defs.lean`](Hddivsteps590/Defs.lean) reimplements the
certificate computation (`hddivsteps_base.v`, `hddivsteps_def.v`). The
soundness proof in `Semantics.lean`, `Chain.lean`, `Invariant.lean`, and
`NumberTheory.lean` follows the structure of Wuille's Coq proof; each module
header names its Coq counterparts. `Recovery.lean`, `Main.lean`, and the
parallel evaluator in `Certificate.lean` are new. The MIT license of the
original (Copyright (c) 2021 Blockstream) is in
[`LICENSE-safegcd-bounds`](LICENSE-safegcd-bounds). This Lean project does not
import the Coq proof or prove a formal equivalence to it.

Instead, [`calibration/replica.py`](calibration/replica.py) independently
replicates the certificate computation in Python. It matches the Lean
computation on the small positive and negative fixtures, the threshold ladder,
and the full 590-step certificate. These checks are evidence that the Lean
definitions implement the intended recurrence. They are not premises of the
Lean theorems. The formal results apply to the Lean definitions as written.

`Certificate.lean` evaluates a task-parallel implementation for speed.
`processDivstepPar_eq` proves that this implementation equals the sequential
specification. Lean's logical model makes `Task.spawn` and `Task.get`
transparent here, so parallelism changes evaluation time but not the
proposition being decided.

## Scope of the formalization

This project deliberately does not formalize:

- **The complete reversible or quantum circuit.** The correspondence between
  circuit registers and the integer state, the gate decomposition, and the
  Toffoli and qubit counts are outside Lean.
- **Circuit synthesis from the general decoder.** Lean proves the recovery
  formulas and the depth-two identity, but not the particular Boolean circuits
  used at every recovery depth.
- **The primality of `p`.** The concrete termination
  theorem does not require it; the generic inverse theorem accepts primality
  as a hypothesis.
- **Equivalence to the Coq implementation.** The Python calibration provides
  cross-implementation evidence, not a formal refinement theorem.
- **Optimality of 590.** The certificate proves that 590 iterations suffice.
  It does not prove that no smaller uniform bound is possible.
- **The compiler evaluation inside `native_decide`.** As described above,
  that closed computation is represented by an additional axiom rather than
  reduced by the kernel.

These boundaries make the artifact's role precise: it certifies the recurrence
properties used to justify the circuit design, not the circuit implementation
as a whole.

## Checking the development

The project pins Lean and Mathlib to version `v4.32.2`. From the repository
root, run:

```sh
cd lean/divsteps
lake exe cache get
lake build
```

The first build evaluates the 590-step certificate and may take several
minutes. Its result is cached under `.lake`; subsequent unchanged builds can
reuse it. The axiom audit is printed while `Hddivsteps590/Main.lean` is
compiled.

The independent Python calibration is optional. From the repository root, run:

```sh
uv run python lean/divsteps/calibration/replica.py
uv run python lean/divsteps/calibration/replica.py --ladder 200
uv run python lean/divsteps/calibration/replica.py --full
```

The first command runs the small fixtures. The `--ladder N` option also checks
each threshold of the ladder up to `N` iterations; `--ladder 200` takes about
a minute and a half. The `--full` option recomputes the 590-step certificate
and runs substantially longer.

## Module guide

| Module | Purpose |
|---|---|
| `Defs` | Computational definitions: exact dyadics, convex hulls, the certificate state, and `hdRun` |
| `Certificate` | Task-parallel certificate evaluation and the proof that it equals the sequential specification |
| `Semantics` | Rational semantics of the dyadic computation and correctness of comparisons and state operations |
| `Chain` | Soundness of the monotone-chain convex-hull implementation |
| `Invariant` | Hull transport, the narrow test, trajectory coverage, and termination from certificate emptiness |
| `NumberTheory` | Gcd preservation and extraction of the modular inverse from a terminated `hdRun` |
| `Recovery` | The frozen `z`-walk, algebraic invariants, branch recovery, and the bridge to `hdRun` |
| `Main` | Generic and concrete-prime theorems together with the axiom audit |
