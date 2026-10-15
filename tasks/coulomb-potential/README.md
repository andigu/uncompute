# coulomb-potential — pairwise Coulomb potential

This task computes a fixed-point approximation of
$1/\lVert r_i-r_j\rVert$ from two three-dimensional grid points and adds it
into an accumulator. This pairwise potential appears in real-space and
first-quantized quantum simulations of chemistry. The accompanying paper
(arXiv: TBD) describes the discovered circuit in its Supplementary
Information, section "Numerical function evaluation", subsection "Coulomb
potential".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. The evaluator measures both numbers from the
emitted op stream. [results.jsonl](results.jsonl) holds the certified rows.

| circuit | description | Toffoli | qubits | TQ-cost |
|---|---|---:|---:|---:|
| [discovered](circuits/discovered/README.md) | discovered circuit | 1,140 | 142 | 161,880 |
| [haner-piecewise](circuits/haner-piecewise/README.md) | baseline: Häner, Roetteler, and Svore (2018), arXiv:1805.12445 | 6,355 | 252 | 1,601,460 |

Relative to the baseline, the discovered circuit uses 5.6× fewer Toffolis
and 1.8× fewer qubits. Its TQ-cost is 9.9× lower.

## 1. Operation

For distinct points $r_i,r_j\in\{0,\ldots,255\}^3$, let $n_2$ denote their
squared distance and let $F(n_2)$ denote the scaled inverse distance:

$$
n_2=\sum_{c\in\{x,y,z\}}(r_{i,c}-r_{j,c})^2,
\qquad
F(n_2)=\operatorname{round}\!\left(\frac{2^{24}}{\sqrt{n_2}}\right).
$$

Rounding selects the nearest integer, with halves rounded away from zero.
The circuit must preserve both points and add $F(n_2)$ into the accumulator
modulo $2^{25}$:

$$
\lvert r_i\rangle\lvert r_j\rangle\lvert k\rangle
\longmapsto
\lvert r_i\rangle\lvert r_j\rangle
\lvert k+F(n_2)\bmod 2^{25}\rangle.
$$

| register | width | meaning | required final value |
|---|---:|---|---|
| `rix` | 8 qubits | $x$ coordinate of $r_i$ | Unchanged |
| `riy` | 8 qubits | $y$ coordinate of $r_i$ | Unchanged |
| `riz` | 8 qubits | $z$ coordinate of $r_i$ | Unchanged |
| `rjx` | 8 qubits | $x$ coordinate of $r_j$ | Unchanged |
| `rjy` | 8 qubits | $y$ coordinate of $r_j$ | Unchanged |
| `rjz` | 8 qubits | $z$ coordinate of $r_j$ | Unchanged |
| `k` | 25 qubits | Modular accumulator | $k+F(n_2)\bmod2^{25}$, within the allowed error |

## 2. Applications

Grid-based and first-quantized quantum simulations of chemistry evaluate the
Coulomb interaction for particle pairs in each simulated time step. The
resource estimate of Kassal et al. (2008), arXiv:0801.2986, computes the
squared distance, applies a Newton-iteration inverse square root, and uses
the result to control a phase. Jones et al. (2012), arXiv:1204.0567,
similarly construct a $1/r_{ij}$ register that drives a quantum variable
rotation in a first-quantized potential-energy circuit.

Kivlichan et al. (2017), arXiv:1608.05696,
express real-space simulation costs in calls to an oracle that writes the
pairwise Coulomb value into a register. Their softened potential is
$q_iq_j/\sqrt{\lVert x_i-x_j\rVert^2+\Delta^2}$. It differs from this
task's unsoftened map by the $\Delta^2$ offset in the radicand. Su et al.
(2021), arXiv:2105.12767, cost first-quantized chemistry simulation in
which the Coulomb interaction of every particle pair enters each step.

## 3. Specification

### 3.1 Data representation

All seven registers are little-endian unsigned integers. Coordinate
differences are mathematical signed differences, although each coordinate
register holds an unsigned value. Squaring an unextended two's-complement
bit pattern therefore does not produce the required $n_2$.

The integer $F(n_2)$ represents the inverse distance on a $2^{-24}$
fixed-point grid. The accumulator is an unsigned residue modulo $2^{25}$.
Its initial value is arbitrary, so the circuit must add the result rather
than write or XOR it.

The grader reads accumulator errors as centered modular residues in
$(-2^{24},2^{24}]$. This convention makes the correctness band continuous
across modular wraparound.

### 3.2 Task instance

Each coordinate has eight bits, so both points lie on a $256^3$ grid. Equal
points are outside the domain because the inverse distance is singular. For
distinct points, the squared distance satisfies

$$
1\leq n_2\leq3\cdot255^2=195{,}075.
$$

Exactly 139,637 integers in this interval are sums of three squared
coordinate differences with magnitudes of at most 255. The output scale
makes the largest value, $F(1)=2^{24}$, fit in 25 bits without saturation.
Across the achievable domain, $F$ ranges from 37,986 to 16,777,216.

[reference.py](reference.py) enumerates the achievable values, checks the
ideal table, and documents the fixed instance in executable form.

### 3.3 Correctness and examples

For each tested input, first compute the integer discrepancy

$$
\Delta=(k_{\mathrm{out}}-k_{\mathrm{in}})-F(n_2).
$$

Reduce $\Delta$ modulo $2^{25}$ to its unique representative in the centered
interval $(-2^{24},2^{24}]$. The output is correct when all six coordinate
registers round-trip exactly and the reduced discrepancy obeys

$$
|\Delta|\leq32.
$$

The bound is inclusive: the grader accepts $\Delta=32$ and $\Delta=-32$,
and it rejects $\Delta=33$ and $\Delta=-33$. Each unit of $\Delta$ is one
accumulator least-significant bit, that is, $2^{-24}$ in the inverse
distance. The 32-unit band therefore permits an absolute error of
$32\cdot2^{-24}\approx1.91\times10^{-6}$ in $1/\lVert r_i-r_j\rVert$, below
$2\times10^{-6}$.

The band is the only approximation allowance. The specification prescribes
no intermediate difference, square, sum, or inverse-square-root
representation.

Representative ideal values are:

| $n_2$ | $F(n_2)$ |
|---:|---:|
| 1 | 16,777,216 |
| 2 | 11,863,283 |
| 3 | 9,686,330 |
| 4 | 8,388,608 |
| 5 | 7,502,999 |
| 16,384 | 131,072 |
| 195,075 | 37,986 |

For example, $r_i=(0,0,0)$ and $r_j=(1,0,0)$ give $n_2=1$. With
`k = 20,000,000`, the exact accumulator result is

$$
20{,}000{,}000+16{,}777{,}216\bmod2^{25}=3{,}222{,}784.
$$

The grader accepts any output within 32 centered modular units of that
value, provided all coordinate registers stay unchanged.

## 4. Discovered circuit

The [discovered circuit](circuits/discovered/README.md) evaluates a table of
lines after binary normalization. Its main ideas are:

- **Direct squared-distance accumulation.** The circuit adds the partial
  products of all three coordinate squares directly into one register. It
  forms the exact squared distance without three product registers or a
  separate summation stage.
- **A finer table of lines.** The circuit writes $n_2=2^e u$ with
  $1\le u<2$. The parity of $e$ and six leading mantissa bits select one of
  128 lines, and the next six bits give a local coordinate. Each line
  therefore needs at most six controlled additions instead of a general
  multiplication. The HRS baseline instead uses quadratics on 16 intervals.
- **Cheap erasure of the larger table.** Measurement-based uncomputation
  (MBU) of the lookup clears the table words with a 17-Toffoli phase repair
  instead of the 126 Toffolis of a reverse lookup.

The coefficients fit the exact integer computation that the circuit
performs. Over all achievable squared distances, the worst error is 32
units, that is, $1.91\times10^{-6}$ in the inverse distance.

## 5. Baseline

### Häner, Roetteler, and Svore normalized piecewise polynomial

The [haner-piecewise](circuits/haner-piecewise/README.md) circuit is our
gate-level reconstruction of two procedures from Häner, Roetteler, and
Svore (2018), arXiv:1805.12445 (HRS). We wrote it from the HRS paper's
description. The first procedure is the leading-bit exponent normalization
of the HRS inverse-square-root circuit. The second is the HRS automatic
piecewise-polynomial method. The circuit writes the squared distance as
$n_2=2^e u$, where $1\le u<2$. It approximates $u^{-1/2}$ on this fixed
mantissa interval, with separate coefficient banks for even and odd $e$,
and restores the scale factor $2^{-e/2}$ afterward. The polynomial applies
directly to $u^{-1/2}$, so the circuit needs no Newton refinement.

The circuit uses degree 2 on 8 Remez-selected mantissa intervals per
exponent parity. Its 16 coefficient banks use signed fixed-point format
$(n,p)=(26,2)$. Exhaustive evaluation over all 139,637 achievable squared
distances finds a worst error of 32 output units at $n_2=3$, equal to the
band. The measured count includes the distance calculation, normalization,
polynomial evaluation, rescaling, and complete uncomputation. This exact
instance does not appear in HRS.

For the same degree, interval count, and fixed-point format, the HRS
closed-form accounting gives 9,528 Toffolis and 240 qubits. The
implemented circuit uses temporary-AND adders and native-width comparators,
which reduce its Toffoli count relative to those formulas. Its measured
peak counts the transient multiplier wires.

### Other related work (no circuit in this repository)

**Exact table lookup.** $F$ depends on the coordinates only through $n_2$,
so an exact construction can use a classical table. It computes the squared
distance, retrieves the rounded value from the table, adds it to `k`, and
uncomputes its scratch registers. The exact table spends none of the
approximation allowance. Unary-iteration lookup (Babbush et al. 2018,
arXiv:1805.03662) traverses the full address space, at about one Toffoli
per address, so about 195,076 Toffolis here. The SelectSwap network (Low,
Kliuchnikov, and Schaeffer 2018, arXiv:1812.00954) instead stores several
table words and routes the selected word into a fixed output register. For
word width $b$, table size $N$, and $\lambda$ stored words, its gate
complexity is $O(\lambda b+N/\lambda)$.

Berry et al. (2019), arXiv:1902.02134, give concrete clean-ancilla costs.
For a table with $d$ addresses, their compute and measurement-based
uncompute formulas are

$$
C_{\mathrm{compute}}=\left\lceil\frac{d}{k}\right\rceil+b(k-1),
\qquad
C_{\mathrm{uncompute}}=\left\lceil\frac{d}{k'}\right\rceil+k'.
$$

At $d=195{,}076$ and $b=25$, minimization over powers of two selects
$k=64$ with compute cost 4,624 and $k'=512$ with uncompute cost 894. These
two windows minimize the Toffoli count, not the TQ-cost. A smaller bank
trades additional Toffolis for fewer qubits.

**Kassal et al. (2008), arXiv:0801.2986.** They form the squared pairwise
distance with three subtractions and three squarings. The two additions are
free in their accounting because the squarings share an accumulator. They
then obtain $1/r$ directly from $r^2$ without a square root. If $x_n$
approximates $1/r$, their Newton-Raphson update is

$$
x_{n+1}=\frac{x_n}{2}\left(3-r^2x_n^2\right).
$$

Their numerical experiments find that four iterations give better than
0.03% accuracy over the studied range. The arithmetic uses
quantum-Fourier-transform addition. For $m$-bit registers, their resource
estimate reports

$$
\frac{75}{4}m^3+\frac{51}{2}m^2
$$

elementary gates per particle pair and four $m$-qubit ancilla registers for
the Coulomb potential. These controlled-rotation gate counts use a different
accounting from the executed Toffoli counts in the table above.

**Jones et al. (2012), arXiv:1204.0567.** They compute the squared-distance
register

$$
\lvert r_{ij}^2\rangle
=
\lvert(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2\rangle
$$

from the position registers. If $a_n$ approximates $1/r_{ij}$, they refine
it with the Newton-Raphson update

$$
a_{n+1}=\frac{1}{2}a_n\left(3-a_n^2r_{ij}^2\right).
$$

They report convergence within five iterations in 32-bit arithmetic. The
resulting $1/r_{ij}$ register drives a quantum variable rotation with
scaling factor

$$
\xi=\frac{q_iq_j\,\delta t}{8\pi^2\varepsilon_0\hbar},
$$

where $q_i$ and $q_j$ are the particle charges and $\delta t$ is the time
step. Every computation before the rotation is then uncomputed. The work
gives the rotation machinery in detail but no per-pair Toffoli count for
the arithmetic.

**Leading-bit piecewise-constant tables.** A constant table can index the
leading bits of $n_2$, but the steep part of $F$ near the singularity
prevents broad ranges. $F$ decreases monotonically. A single constant
therefore approximates a consecutive run of achievable inputs within 32
units exactly when the targets at the two endpoints differ by at most
$2\cdot32=64$. Greedy extension of every run is therefore optimal.
A slope within each range removes this limit, because the line
follows the curve instead of staying fixed between its endpoints.

## 6. Testing

Each run makes 9,024 pseudorandom draws. The grader skips draws with
$r_i=r_j$ rather than replacing them, which leaves about 9,014 tested cases
on average. The initial 25-bit accumulator is uniform. This checks that the
circuit adds rather than overwrites the target, and it exercises modular
wraparound.

Independent uniform points concentrate away from both extremes: they almost
never produce unit separation, and they rarely approach the largest squared
distance. Testing therefore mixes three sampling modes:

| mode | probability | sampled region |
|---|---:|---|
| Uniform | $1/2$ | Independent points across the full grid |
| Near | $7/16$ | Signed per-axis offsets at seven bit widths from 2 through 8, including unit separation |
| Far | $1/16$ | Points in opposite 32-wide corner bands, giving $111{,}747\leq n_2\leq195{,}075$ |

Near-mode offsets wrap modulo 256, so they probe small separations
throughout the grid rather than only near its boundary. Far mode forces
each coordinate difference into $[193,255]$, which reaches the high-$n_2$
region that uniform sampling visits rarely. $F$ depends on the coordinates
only through $n_2$, so these modes exercise both ends of the output range.

Every tested case must pass. One changed coordinate or one accumulator
discrepancy outside the inclusive $\pm32$ band fails the run. There is no
allowance for failed shots on top of the band. For scale, a simple Newton
construction in [reference.py](reference.py) reaches a worst error of 14
units over the whole achievable domain.

## 7. Algorithm notes

Once the circuit has formed $n_2$, the relevant identity is

$$
\frac{1}{\lVert r_i-r_j\rVert}=\frac{1}{\sqrt{n_2}}.
$$

An inverse-square-root iteration can therefore operate directly on the
integer squared distance. A Newton construction begins with an
exponent-dependent estimate and refines it through fixed-point
multiplications. Its precision can increase between iterations, because
early approximations need not carry the final output width.

A circuit may build the three coordinate squares independently or
accumulate them through a shared carry-save structure. Truncated
multipliers, column-limited squarers, and recomputation of iteration
intermediates exchange nonlinear work for workspace. Their combined
numerical error matters only at the final accumulator, where it must stay
inside the single $\pm32$ band.

Lookup and interpolation form another family. The target depends only on
$n_2$, so leading and selected lower bits can address constants or
polynomial coefficients. Exact lookup has no approximation error but needs
a large table. Iterative arithmetic needs less table storage but more
multiplication and retained intermediates. Interpolation lies between
them. A slope follows the inverse-square-root curve, so it needs fewer
regions than the piecewise-constant cover in section 5.

## 8. References

- Häner, Roetteler, and Svore (2018), arXiv:1805.12445. Leading-bit
  normalization, fixed-point inverse square root, and piecewise-polynomial
  function evaluation.
- Kivlichan et al. (2017), arXiv:1608.05696.
  Real-space simulation costed in calls to a pairwise Coulomb oracle.
- Su et al. (2021), arXiv:2105.12767. Fault-tolerant first-quantized
  simulation of chemistry.
- Kassal et al. (2008), arXiv:0801.2986. Grid-based chemical dynamics with
  a Newton-iteration inverse square root.
- Jones et al. (2012), arXiv:1204.0567. A Newton-iteration $1/r_{ij}$
  register that feeds a quantum variable rotation.
- Gidney (2018), arXiv:1709.06648. Temporary logical AND and
  measurement-based uncomputation for low-Toffoli arithmetic.
- Cuccaro et al. (2004), arXiv:quant-ph/0410184.
  In-place ripple-carry addition.
- Babbush et al. (2018), arXiv:1805.03662. Unary-iteration table lookup.
- Low, Kliuchnikov, and Schaeffer (2018), arXiv:1812.00954. The SelectSwap
  network.
- Berry et al. (2019), arXiv:1902.02134. Clean-ancilla Toffoli costs for
  SelectSwap lookup and measurement-based unlookup.
