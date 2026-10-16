# hhl-angle — HHL rotation angle

This task computes the fixed-point angle $\arcsin(C/x)$ for $C=1/2$. The
angle drives the eigenvalue-dependent ancilla rotation in the linear-system
algorithm of Harrow, Hassidim, and Lloyd (HHL). The accompanying paper
(arXiv:2610.22367) describes the discovered circuit in its Supplementary
Information, section "Numerical function evaluation", subsection "HHL
rotation angle".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. The evaluator measures both numbers from the
emitted op stream. [results.jsonl](results.jsonl) holds the certified rows.

| circuit | description | Toffoli | qubits | TQ-cost |
|---|---|---:|---:|---:|
| [discovered](circuits/discovered/README.md) | discovered circuit | 1,348 | 115 | 155,020 |
| [haner-piecewise](circuits/haner-piecewise/README.md) | baseline: Häner, Roetteler, and Svore (2018), arXiv:1805.12445 | 16,468 | 300 | 4,940,400 |

Relative to the baseline, the discovered circuit uses 12× fewer Toffolis
and 2.6× fewer qubits. Its TQ-cost is 32× lower.

## 1. Operation

The circuit preserves `X` and adds the encoded angle to an arbitrary
accumulator `k`. Define

$$
f(X)=\operatorname{round}\!\left(
2^{32}\arcsin\!\left(\frac{2^{15}}{X}\right)
\right),
$$

where rounding is to the nearest integer with half-integer cases rounded
away from zero. The required reversible map is

$$
\lvert X\rangle\lvert k\rangle
\longmapsto
\lvert X\rangle
\lvert k+f(X)\bmod 2^{33}\rangle.
$$

| register | width | meaning | required final value |
|---|---:|---|---|
| `X` | 21 qubits | Encoded positive input $x=X/2^{16}$ | Unchanged |
| `k` | 33 qubits | Arbitrary modular accumulator | $k+f(X)\pmod{2^{33}}$ |

## 2. Applications

The HHL algorithm (Harrow, Hassidim, and Lloyd 2009, arXiv:0811.3171) uses
this function as its rotation angle. After phase estimation attaches an
eigenvalue estimate $\tilde\lambda$, the algorithm rotates an ancilla
according to

$$
\lvert\tilde\lambda\rangle\lvert 0\rangle
\longmapsto
\lvert\tilde\lambda\rangle\!\left(
\sqrt{1-\frac{C^2}{\tilde\lambda^2}}\lvert 0\rangle
+\frac{C}{\tilde\lambda}\lvert 1\rangle
\right).
$$

With $\theta=\arcsin(C/\tilde\lambda)$, this operation is
$R_y(2\theta)$. A fault-tolerant implementation can compute $\theta$ in
fixed point and use its bits to control elementary rotations.

The Poisson-equation algorithm of Cao et al. (2012), arXiv:1207.2485, builds
this pipeline from reversible arithmetic. Newton iteration produces the
reciprocal, and bisection against a sine circuit converts it to an arcsine
angle. The angle bits then drive the controlled rotation. Häner,
Roetteler, and Svore (2018), arXiv:1805.12445, give reversible reciprocal
and arcsine circuits for the same composition. Only the composed angle
reaches the controlled rotation. This task therefore evaluates the angle
directly and requires no separate reciprocal register.

## 3. Specification

### 3.1 Data representation

Both registers are little-endian. `X` is unsigned and represents
$x=X/2^{16}$. The integer $f(X)$ encodes an angle with 32 fractional bits,
so one output unit is $2^{-32}$ radians.

The accumulator `k` is an unsigned residue modulo $2^{33}$. Its initial
value is arbitrary, and the addition wraps at that modulus.

### 3.2 Task instance

The input domain is

$$
2^{16}\le X\le 2^{21}-1,
$$

which contains 2,031,616 values and represents
$1\le x\le 32-2^{-16}$. With $C=1/2$, the arcsine argument obeys

$$
\frac{C}{x}=\frac{2^{15}}{X}
\in\left(\frac{1}{64},\frac{1}{2}\right].
$$

The largest ideal addend occurs at `X = 65536` and is

$$
\operatorname{round}(2^{32}\pi/6)=2{,}248{,}839{,}617.
$$

The smallest occurs at `X = 2097151` and is 67,111,627. Both extreme
addends fit in the positive range of the 33-bit accumulator.

### 3.3 Correctness and examples

Let $k^\star=(k+f(X))\bmod 2^{33}$ be the ideal accumulator result. Define
the accumulator error as the centered residue

$$
\delta=\operatorname{center}_{2^{33}}(k_{\mathrm{out}}-k^\star).
$$

Here $\operatorname{center}_{2^{33}}$ selects the unique representative in
$(-2^{32},2^{32}]$. Subtraction modulo $2^{33}$ followed by centering gives
the shortest signed distance between two residues. The bound therefore
applies in the same way near the wraparound and away from it.

The grader accepts a shot only when both conditions hold:

- The returned `X` is bit-for-bit equal to the input `X`.
- The returned accumulator satisfies $|\delta|\le 429$ output units.

The accumulator condition is an absolute bound of
$429\cdot 2^{-32}\approx 9.99\times 10^{-8}$ radians on the added angle,
that is, an error below $10^{-7}$ radians. It applies to every shot. There
is no relative-error rule and no allowance for failed shots.

These examples give the ideal addend before accumulation:

| `X` | represented $x$ | ideal addend $f(X)$ |
|---:|---:|---:|
| 65,536 | 1 | 2,248,839,617 |
| 131,072 | 2 | 1,085,253,432 |
| 524,288 | 8 | 268,610,527 |
| 1,048,576 | 16 | 134,239,583 |
| 2,097,151 | $32-2^{-16}$ | 67,111,627 |

For the wraparound case `X = 65536` with initial accumulator $2^{33}-1$, the
ideal result is $2{,}248{,}839{,}616$. The centered-residue rule compares a
returned residue with that wrapped result without a discontinuity at zero.

## 4. Discovered circuit

The [discovered circuit](circuits/discovered/README.md) is a
piecewise-quadratic evaluator with three main ideas:

- **Power-of-two-aligned intervals.** The circuit uses 345 intervals of
  widths $2^8$ through $2^{15}$. Each interval has the form $[L,L+2^b)$
  with $L$ a multiple of $2^b$, so the leading input bits select it. No
  comparison against an arbitrary boundary is necessary.
- **Shared local coordinate.** A cyclic shift of the low 15 input bits maps
  every interval onto one 15-bit coordinate. The offline fit absorbs the
  constant offset that the shift leaves, so one arithmetic kernel serves
  every interval.
- **Direct accumulation.** The circuit overwrites the loaded slope with the
  inner Horner value and accumulates the outer product directly into `k`.
  Measurement-based uncomputation (MBU) clears the coefficients without a
  reverse lookup.

An offline search fits the integer coefficients to the exact truncated
arithmetic that the circuit performs. An exhaustive check over all
2,031,616 inputs gives a worst error of 427.15 output units, that is,
$9.95\times10^{-8}$ radians.

## 5. Baseline

### Häner, Roetteler, and Svore piecewise polynomial

The [haner-piecewise](circuits/haner-piecewise/README.md) circuit is our
gate-level reconstruction of the function-evaluation method of Häner,
Roetteler, and Svore (2018), arXiv:1805.12445 (HRS). We wrote it from the HRS
paper's description. The method partitions the domain and uses Remez
exchange to select a polynomial on each interval. It evaluates the selected
polynomial with truncated fixed-point arithmetic. The HRS parallel Horner
construction shares one Horner ladder and loads coefficients under
interval-label controls. Here "parallel" refers to the handling of the
interval polynomials, not to depth-parallel evaluation.

The baseline applies the method directly to $x\mapsto\arcsin(1/(2x))$ on
$1\le x<32$. It uses degree 3, 35 Remez-selected intervals, and signed
fixed-point format $(n,p)=(44,6)$: each word has 44 bits, 6 of them integer
bits. The partition search accepts an interval only when the quantized
datapath stays within the 429-unit band on every input in it. An exhaustive
check over all 2,031,616 inputs finds a worst error of 429 output units at
`X = 66097`, equal to the bound.

For these parameters, the HRS resource formulas give 41.7k Toffolis and 238
qubits. The table above instead reports the executed circuit. Its Toffoli
count is lower because it uses measurement-erased temporary-AND ripple
chains, native 21-bit comparators, and a Toffoli-free label-counter
increment. Its peak qubit count is higher because the evaluator counts live
temporary ANDs, including 64 in the widest multiplier row.

### Other related work (no circuit in this repository)

Carrera Vazquez, Hiptmair, and Woerner (2020), arXiv:2009.04484, construct
exactly the function $f(\lambda)=\arcsin(C/\lambda)$ for the HHL rotation.
The construction is one component of their treatment of HHL with Richardson
extrapolation. Above a low-end cutoff, their scheme divides the domain into
geometrically growing intervals $[a_i,a_{i+1}]$ with $a_{i+1}=5a_i$. This
gives $M=\lceil\log_5((N_l-1)/a)\rceil$ intervals, with Chebyshev
interpolation on each. For $x\in[1,a]$, the circuit applies the identity.
The construction has a proven error bound for polynomial degree $d$. Its
parallel reversible Horner evaluation has estimates of
$O(n_l^2d+Md\log M)$ Toffoli gates and $(d+1)n_l+\lceil\log M\rceil+1$
qubits. These are asymptotic estimates, not executed counts.

Cao et al. (2012), arXiv:1207.2485, separate reciprocal evaluation from
inverse-trigonometric evaluation. Newton iteration first computes a
fixed-point approximation $\omega=C_d/\hat\lambda$. A bisection then
narrows an interval that contains the angle whose sine equals $\omega$. It
starts at $\theta=\pi/4$. Each step evaluates $\sin\theta$ to error
$\varepsilon_1^2/2$, compares the result with $\omega$, and keeps the left
or right half. The loop performs $\lceil\log_2\varepsilon_1^{-2}\rceil+1$
steps. The paper defines $\varepsilon_1=\min\{1/(4M^2),\varepsilon\}$. It
states costs proportional to $(\log_2\varepsilon_1^{-1})^4$ operations and
$(\log_2\varepsilon_1^{-1})^3$ qubits. It gives no concrete circuit at this
task's register widths and precision.

Bhaskar et al. (2016), arXiv:1511.08253, give
a fixed-point reciprocal for inputs $w\ge 1$. Their module applies
$s=\lceil\log_2 b\rceil$ Newton steps in $b$-bit truncated arithmetic. The
step count is fixed in advance. They bound the total error by

$$
2^{-b}+\lceil\log_2 b\rceil\,2^{-b}
\le \frac{2+\log_2 b}{2^b}.
$$

The module supplies the reciprocal stage of an HHL angle pipeline but no
arcsine circuit.

## 6. Testing

The grader makes 9,024 pseudorandom draws. Each draw samples `X` uniformly
from all 21-bit patterns and `k` uniformly from all $2^{33}$ residues. The
grader skips a draw with $X<2^{16}$. The acceptance probability is $31/32$,
so about 8,742 tests run. The kept inputs are uniform over all 2,031,616
in-domain values, because a skip does not remap the draw.

The random accumulator tests the circuit as modular addition rather than as
assignment to a zeroed register. It also places tests across the whole
residue space, including the wraparound. Every tested shot must preserve
`X` exactly and satisfy the 429-unit bound.

## 7. References

- Harrow, Hassidim, and Lloyd (2009), arXiv:0811.3171. The HHL algorithm
  and its eigenvalue-dependent rotation.
- Häner, Roetteler, and Svore (2018), arXiv:1805.12445. Reversible
  fixed-point arithmetic and piecewise-polynomial function evaluation.
- Carrera Vazquez, Hiptmair, and Woerner (2020), arXiv:2009.04484.
  Piecewise Chebyshev interpolation of the HHL rotation angle.
- Cao et al. (2012), arXiv:1207.2485. Newton reciprocal followed by
  reversible sine bisection.
- Bhaskar et al. (2016), arXiv:1511.08253.
  Fixed-step Newton reciprocal and its error bound.
