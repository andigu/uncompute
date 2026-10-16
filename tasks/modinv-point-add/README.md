# modinv-point-add — prime-field modular inversion

This task benchmarks modular inversion within a complete arithmetic circuit:
point addition on an elliptic curve over a 256-bit prime field. The modular
inverse in the chord slope dominates the cost of point addition. The
accompanying paper (arXiv:2610.22367) describes the circuits in its supplementary
section "Prime-field modular inversion" and in the main-text section
"Reversible modular inversion".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. The evaluator counts executed CCX and CCZ gates at
their promoted arity, so a bit-conditioned CX counts as a Toffoli. Classical
bit wires do not count toward peak qubits.

[results.jsonl](results.jsonl) holds one certified row per circuit file:
verdict, Toffoli count, peak qubits, TQ-cost, and the sha256 of the op stream.
The `.ops.xz` file beside each circuit file holds its certified op stream.

| circuit file | description | Toffolis | qubits | TQ-cost |
|---|---|---:|---:|---:|
| [`divsteps/circuit.py`](circuits/divsteps/README.md) | discovered circuit, $j_{\mathrm{store}}=5$ | 4,006,673 | 935 | 3,746,239,255 |
| `divsteps/wide.py` | divstep family | 3,293,278 | 1,191 | 3,922,294,098 |
| `divsteps/fill.py` | divstep family | 3,637,655 | 1,031 | 3,750,422,305 |
| `divsteps/j5.py` | divstep family | 4,317,926 | 927 | 4,002,717,402 |
| `divsteps/j6.py` | divstep family | 4,427,338 | 913 | 4,042,159,594 |
| `divsteps/j7.py` | divstep family | 4,649,514 | 904 | 4,203,160,656 |
| `divsteps/j8.py` | divstep family | 5,331,729 | 897 | 4,782,560,913 |
| `divsteps/j9.py` | divstep family | 7,866,779 | 893 | 7,025,033,647 |
| [`euclidean/circuit.py`](circuits/euclidean/README.md) | discovered circuit | 20,521,949 | 843 | 17,300,003,007 |
| [`luo-eea/circuit.py`](circuits/luo-eea/README.md) | baseline: Luo et al. (2026), arXiv:2607.13816 | 69,529,757 | 838 | 58,265,936,366 |

The paper's resource comparison is below. All circuits use the same affine
input and output. The counts include reversible cleanup. They exclude
application-specific storage or lookup that supplies the classical addend.

| construction | inversion strategy | Toffolis | qubits | TQ-cost | in this repository |
|---|---|---:|---:|---:|---|
| Schrottenloher (2026), arXiv:2606.02235 | binary gcd with a dialog | 1.9·10⁶ | 1,446 | 2.7·10⁹ | no (cited result) |
| **divstep circuit** (discovered) | divsteps with decoding | 4.0·10⁶ | 935 | 3.7·10⁹ | `circuits/divsteps/` |
| Luo et al. (2026), arXiv:2607.13816 | packed Euclidean walk | 7.0·10⁷ | 838 | 5.8·10¹⁰ | `circuits/luo-eea/` (baseline) |
| **specialized Euclidean** (discovered) | specialized packed walk | 2.1·10⁷ | 843 | 1.7·10¹⁰ | `circuits/euclidean/` |

The circuits trade Toffolis against qubits in different ways, so the paper
reports no single baseline or gain for this task.

This command runs one circuit and checks it:

```sh
uv run python tools/eval.py modinv-point-add --circuit tasks/modinv-point-add/circuits/euclidean/circuit.py
```

## 1. Operation

Each point on the curve is a pair of integers $(x,y)$ that satisfies

$$
y^2\equiv x^3+7\pmod p,\qquad p=2^{256}-2^{32}-977.
$$

Point addition combines two points $P$ and $Q$ into a third point
$R=P+Q$ on the same curve. Geometrically, it draws the line through $P$ and
$Q$, finds the third intersection with the curve, and reflects that point
across the horizontal axis. For the inputs of this task, $Q_x\ne P_x$, and the
slope of the chord is

$$
\lambda\equiv(Q_y-P_y)(Q_x-P_x)^{-1}\pmod p.
$$

The coordinates of the sum are

$$
R_x=\lambda^2-P_x-Q_x\pmod p,
\qquad
R_y=\lambda(P_x-R_x)-P_y\pmod p.
$$

The circuit is a reversible point adder (Proos and Zalka (2003),
arXiv:quant-ph/0301141). It takes a quantum point $P$ and a classically
supplied point $Q$, and it performs

$$
\lvert P\rangle\longmapsto\lvert P+Q\rangle.
$$

## 2. Applications

Once the slope $\lambda$ is known, the remaining coordinate updates need only
a fixed sequence of modular additions and multiplications. Modular inversion
instead follows a long input-dependent sequence of arithmetic transitions.
The circuit must later recover that sequence to reverse the computation and
clear its ancillas.

Because $p$ is prime and $w=Q_x-P_x\ne0$, there are integers $\alpha$ and
$\beta$ with $\gcd(p,w)=1=\alpha p+\beta w$. Reduction modulo $p$ gives
$\beta\equiv w^{-1}\pmod p$. Extended gcd algorithms find $\beta$ by tracking
the current values as integer combinations of $p$ and $w$.

Most prior point-addition circuits use the binary gcd algorithm. Its updates are
cheap, but the update that applies changes from step to step, and the reduced
pair does not show which update produced it. These circuits therefore keep a
*dialog*, introduced by Khattar et al. (2025), arXiv:2510.10967. The first
pass reduces the pair and records a compact symbol for each choice. A second
pass replays the record to build the coefficient. The complete record needs
hundreds of additional qubits.

The windowed algorithm of Babbush et al. (2026), arXiv:2603.28846, calls the
point adder 28 times. By a union bound, the failure budget of section 6 adds
less than 0.6% retry overhead to that algorithm.

## 3. Specification

### 3.1 Data representation

The circuit declares its registers in this order:

| register | width | meaning | required final value |
|---|---:|---|---|
| `target_x` | 256 qubits | input coordinate $P_x$ | $(P+Q)_x$ |
| `target_y` | 256 qubits | input coordinate $P_y$ | $(P+Q)_y$ |
| `offset_x` | 256 bits | addend coordinate $Q_x$ | input only |
| `offset_y` | 256 bits | addend coordinate $Q_y$ | input only |

Each coordinate register holds a little-endian unsigned integer, the
canonical residue modulo $p$. The two `offset` registers are classical bit
inputs with fresh values on every test. They are not constants compiled into
the circuit, so a circuit must be correct for any addend. A circuit may
compute any classical function of $Q$ at no cost.

### 3.2 Task instance

| quantity | value |
|---|---|
| field prime $p$ | `0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f` |
| coefficient $a$ | 0 |
| coefficient $b$ | 7 |
| $G_x$ | `0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798` |
| $G_y$ | `0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8` |
| subgroup order $n$ | `0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141` |

The base point $G$ lies on the curve and generates a subgroup of prime order
$n$. `contract.json` holds the same values, and `grader.py` asserts them at
import.

Each test reads two raw 256-bit little-endian scalars $s$ and $t$ from one
64-byte block of the seeded test stream. The grader reduces them modulo $n$
(modulo bias about $2^{-127}$) and forms

$$
P=[s]G,\qquad Q=[t]G.
$$

$P$ and $Q$ are therefore independent and nearly uniform points of the curve.
The grader skips a draw if either operand is the identity, if $P=Q$, or if
$P=-Q$. These are the cases where the affine formulas are incomplete. Each
exclusion has measure about $2^{-255}$.

### 3.3 Correctness and examples

A test passes only if three conditions hold. Both output registers contain the
exact canonical coordinates of $P+Q$. The global sign is clean. Every ancilla
has returned to $\lvert0\rangle$.

For example, take $P=G$ and $Q=2G$. The required result is $3G$:

| register | input | required output |
|---|---|---|
| `target_x` | `0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798` | `0xf9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9` |
| `target_y` | `0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8` | `0x388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672` |
| `offset_x` | `0xc6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5` | unchanged |
| `offset_y` | `0x1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a` | unchanged |

## 4. Discovered circuit

This task has two discovered circuits, `circuits/divsteps/` (the default that
`tools/eval.py` runs) and `circuits/euclidean/`.

### 4.1 Divstep circuit (`circuits/divsteps/`)

The search found a new reversible inversion circuit for prime fields, built on
divsteps (Bernstein and Yang (2019), doi:10.46586/tches.v2019.i3.340-398). In
a divstep, a small signed counter $z$ replaces the magnitude comparisons of
binary gcd. The sign of $z$ and the parity of one value select the next
update. A divstep never computes a multi-bit quotient. Reversible divsteps
exist over binary fields and polynomial rings (Banegas et al. (2021),
doi:10.46586/tches.v2021.i1.451-472; Khattar et al. (2025)). To our knowledge,
this circuit is the first reversible divstep inversion over a prime field.

To run a step backward, the circuit must know which of the three updates the
step applied. Cheap tests on a few bits of the updated state recover the
choice at about two thirds of the steps. For the remaining steps, the paper
proves that the choice is a function of the updated registers. A Lean 4
version of the proof is in `lean/divsteps/`. The circuit decodes the choice
when the decoder is cheap. Otherwise it stores one bit on a fixed-size history
stack. A cutoff $j_{\mathrm{store}}$ sets the boundary. The discovered circuit
uses $j_{\mathrm{store}}=5$ and a 60-qubit stack.

The circuit also uses these techniques (paper SI, "Divsteps"):

- **Early stopping.** A worst-case analysis (Wuille (2021)) proves that 590
  divsteps suffice for every nonzero input. The circuit runs a fixed schedule
  of 533 steps and freezes a walk when $g=0$. The first 498 steps omit the
  termination test.
- **Scale removal.** Each active reverse step halves the product register
  once, so the walk removes the factor $2^t$ without counting $t$.
- **Step-dependent qubit reuse.** Two 256-qubit registers hold the four walk
  values. A growing coefficient takes the positions that its shrinking
  partner frees.
- **Fused coordinate arithmetic.** The circuit adds $\lambda^2$ as three
  half-width products and folds high bits with
  $2^{256}\equiv2^{32}+977\pmod p$.

The point adder runs four walks: forward and reverse for each of its two
inversions. These walks use 63% of the Toffolis. The paper's union bound on
the failure probability is $1.599\times10^{-4}$ per point addition. Different
cutoffs and workspace sizes give the family of circuit files in the
certified results above. `j5.py` to `j9.py` set the cutoff from 5 to 9 on the
staged decoder. `fill.py` and `wide.py` are higher-qubit configurations.
`circuits/divsteps/README.md` describes the circuit in detail.

### 4.2 Specialized Euclidean circuit (`circuits/euclidean/`)

This circuit is at the low-qubit end of the frontier. The search started from
the baseline reconstruction of Luo et al. (2026) and specialized its schedule
for the prime $p$. The circuit keeps the packed register layout. It reduces
the Toffoli count by a factor of 3.4, from 70 million to 21 million, with five
additional qubits. Its techniques are fitted windows, a schedule cut from
1,616 to 1,432 steps, test-free positions, merged scans, measured carries,
and sparse-prime and truncated multipliers (paper SI, "Specialized Euclidean
construction"). The paper estimates its failure probability at about
$8\times10^{-5}$ per point addition. `circuits/euclidean/README.md`
describes the circuit.

## 5. Baseline

The baseline (`circuits/luo-eea/`) is our reconstruction of the
space-efficient Euclidean point adder of Luo et al. (2026), arXiv:2607.13816
(v3). It packs a shrinking remainder and a growing coefficient into opposite
ends of one register. Short counters mark where each value begins and ends.
The reconstruction uses the exact step bound of 1,616 steps and is exact on
the tested domain. `circuits/luo-eea/README.md` describes the
reconstruction.

## 6. Testing

The evaluator runs 100,000 tests. The seed of the test stream is a hash of the
emitted op stream, so any change to the circuit draws a new test set. A test
fails if an output coordinate is wrong, if the global sign is wrong, or if an
ancilla is not $\lvert0\rangle$ at the end. Each test gets one verdict, so a
test with several faults uses one failure.

A run passes with at most 20 failed tests out of 100,000, a failure rate of
$2\times10^{-4}$. This budget makes approximate arithmetic and truncated
inversion schedules legal.

## 7. References

- Babbush et al. (2026), arXiv:2603.28846. The windowed algorithm that calls
  the point adder.
- Banegas et al. (2021), doi:10.46586/tches.v2021.i1.451-472. Reversible
  divsteps over binary fields.
- Bernstein and Yang (2019), doi:10.46586/tches.v2019.i3.340-398. Divsteps.
- Khattar et al. (2025), arXiv:2510.10967. The dialog of choices, and
  reversible divsteps over polynomial rings.
- Luo et al. (2026), arXiv:2607.13816. The space-efficient Euclidean point
  adder of the baseline.
- Proos and Zalka (2003), arXiv:quant-ph/0301141. The reversible point adder.
- Schrottenloher (2026), arXiv:2606.02235. A binary-gcd point adder with a
  dialog.
- Wuille (2021), <https://github.com/sipa/safegcd-bounds>. The 590-step
  worst-case bound for divsteps.
