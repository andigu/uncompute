# gaussian — fixed-point Gaussian

This task computes $e^{-x^2}$ in fixed point and adds the encoded value to an
accumulator. The operation supports on-the-fly molecular-orbital evaluation
and Gaussian state preparation. The accompanying paper (arXiv:2610.22367)
describes the discovered circuit in its Supplementary Information, section
"Numerical function evaluation", subsection "Fixed-point Gaussian".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. The evaluator measures both numbers from the
emitted op stream. [results.jsonl](results.jsonl) holds the certified rows.

| circuit | description | Toffoli | qubits | TQ-cost |
|---|---|---:|---:|---:|
| [discovered](circuits/discovered/README.md) | discovered circuit | 1,583 | 139 | 220,037 |
| [haner-piecewise](circuits/haner-piecewise/README.md) | baseline: Häner, Roetteler, and Svore (2018), arXiv:1805.12445 | 23,784 | 362 | 8,609,808 |

Relative to the baseline, the discovered circuit uses 15× fewer Toffolis
and 2.6× fewer qubits. Its TQ-cost is 39× lower.

## 1. Operation

The circuit preserves `X` and adds the encoded Gaussian value to an arbitrary
accumulator `k`:

$$
\lvert X\rangle\lvert k\rangle
\longmapsto
\left\lvert X\right\rangle
\left\lvert k+\operatorname{round}\!\left(
2^{32}e^{-(X_s/2^{32})^2}
\right)\bmod 2^{39}\right\rangle,
$$

where $X_s$ is the signed integer that `X` represents.

| register | width | meaning | required final value |
|---|---:|---|---|
| `X` | 39 qubits | Encoded signed input $x$ | Unchanged |
| `k` | 39 qubits | Arbitrary modular accumulator | $k+f(X_s)\pmod{2^{39}}$ |

## 2. Applications

Häner, Roetteler, and Svore (2018), arXiv:1805.12445, consider quantum
chemistry with on-the-fly integral computation. The second-quantized
simulation algorithm of Babbush et al. (2016), arXiv:1506.01020, specifies
and costs a basis-function oracle in place of a precomputed classical
database of integrals. At a sampled grid point, this oracle evaluates and
sums the Gaussians of the selected molecular spin-orbital into a register.
The work does not fix a particular reversible Gaussian evaluator.

Gaussian state preparation is another application. McArdle, Gilyén, and
Berta (2022), arXiv:2210.14892, list Gaussian amplitudes among the targets of
state preparation. The algorithm of Kitaev
and Webb (2008), arXiv:0801.0342, builds a Gaussian-amplitude superposition
one qubit at a time. Each step computes a rotation angle from truncated
theta-function sums of Gaussian exponentials. It stores the angle in extra
qubits in superposition, drives one controlled rotation from each stored
bit, and uncomputes the angle. A modular transformation handles parameters
for which direct truncation converges too slowly. Jordan, Lee, and Preskill
(2012), arXiv:1111.3633, use this method to prepare the
multivariate-Gaussian free-field vacuum in a lattice field-theory
simulation.

## 3. Specification

### 3.1 Data representation

Both registers are little-endian. `X` is a 39-bit two's-complement number
with seven integer bits, including the sign, and 32 fractional bits. Its
represented value is $x=X_s/2^{32}$.

The nonnegative addend is

$$
f(X_s)=\operatorname{round}\!\left(2^{32}e^{-x^2}\right),
$$

where rounding is half away from zero. The addend uses the same $2^{32}$
scale as `X`, so one output unit represents $2^{-32}$. The accumulator `k`
is an unsigned residue modulo $2^{39}$, and addition wraps at that modulus.

### 3.2 Task instance

The domain is the half-open interval

$$
-8\le x<8,
$$

equivalently $-8\cdot2^{32}\le X_s<8\cdot2^{32}$. The domain includes
$x=-8$ but excludes $x=8$. It contains exactly $2^{36}$ of the possible
39-bit input patterns.

The maximum addend is $f(0)=2^{32}$. The function is even and
non-increasing in $|x|$. It rounds to zero at $x=-8$ and at the largest
in-domain value $x=8-2^{-32}$.

The format matches the $10^{-7}$ Gaussian accuracy level of Häner,
Roetteler, and Svore (2018). Section 5 derives the register width from
their published resource table.

### 3.3 Correctness and examples

The input register `X` must round-trip bit-exactly. Let $k^\star$ be the
ideal modular sum from section 1, and first form the residue

$$
r=(k_{\mathrm{out}}-k^\star)\bmod 2^{39}.
$$

The error $\delta$ is this residue recentered into $(-2^{38},2^{38}]$: use
$\delta=r$ when $r\le2^{38}$ and $\delta=r-2^{39}$ otherwise. The
recentering prevents modular wraparound from inflating the measured error.

The units are output ULP, where one ULP is $2^{-32}$, the scale of the
addend. The grader accepts a shot only if

$$
|\delta|\le429.
$$

The radius is inclusive. It comes from

$$
\left\lfloor10^{-7}2^{32}\right\rfloor
=\left\lfloor429.4967296\right\rfloor
=429,
$$

the largest integral radius no wider than $10^{-7}$ in the represented
function value. Its real-value size is
$429\cdot2^{-32}\approx9.9884\times10^{-8}$. The grader applies the
condition to every tested shot separately.

These examples give the ideal addend before accumulation:

| represented $x$ | signed integer $X_s$ | ideal addend $f(X_s)$ |
|---:|---:|---:|
| $-8$ | $-8\cdot2^{32}=-34{,}359{,}738{,}368$ | $0$ |
| $-4$ | $-4\cdot2^{32}=-17{,}179{,}869{,}184$ | $483$ |
| $-1$ | $-2^{32}=-4{,}294{,}967{,}296$ | $1{,}580{,}030{,}169$ |
| $0$ | $0$ | $4{,}294{,}967{,}296$ |
| $1$ | $2^{32}=4{,}294{,}967{,}296$ | $1{,}580{,}030{,}169$ |
| $4$ | $4\cdot2^{32}=17{,}179{,}869{,}184$ | $483$ |

## 4. Discovered circuit

The [discovered circuit](circuits/discovered/README.md) never forms $x^2$.
Its main ideas are:

- **Cheap sign fold.** The circuit folds the input onto the nonnegative axis
  and approximates $e^{-y^2}$ directly. For a negative input, it
  complements the data bits but omits the "+1" of an exact absolute value.
  The complement needs only Clifford gates and represents
  $y=|x|-2^{-32}$. The one-unit offset stays inside the error budget and
  removes a carry chain.
- **Pruned coefficient lookup.** Ten bits of $y$ select one of 1,024 equal
  intervals of width $2^{-7}$ on $[0,8)$. Of these, 455 need quadratics, the
  next 59 need only constants, and the remaining 510 return zero. A
  variable-spaced lookup (Sanders et al. 2020, arXiv:2007.07391) stops at
  every subtree with one shared value. It costs 495 Toffolis instead of
  1,022.
- **Direct quadratic accumulation.** A truncated product updates the
  looked-up slope in place. The circuit then adds the intercept and
  accumulates the final product directly into `k`. No register holds $y^2$,
  a completed product, or the Gaussian value.

After measurement-based uncomputation (MBU) of the lookup (Berry et al.
2019, arXiv:1902.02134), the freed coefficient qubits serve as carry
ancillas for later arithmetic. The coefficients fit the
exact integer computation. The worst error over the full domain is
$9.99\times10^{-8}$.

## 5. Baseline

### Häner, Roetteler, and Svore piecewise polynomial

The [haner-piecewise](circuits/haner-piecewise/README.md) circuit is our
gate-level reconstruction of the method of Häner, Roetteler, and Svore
(2018), arXiv:1805.12445 (HRS). We wrote it from the HRS paper's description.
The HRS construction squares the input, partitions $w=x^2$, and selects a
minimax polynomial with the LABEL and NEXT$_a$ circuits. It evaluates the
polynomial with parallel Horner arithmetic and a pre-truncated multiplier.
The published $\exp(-x^2)$ block at the $10^{-7}$ level lists four
degree-and-interval configurations: $(3,32)$, $(4,15)$, $(5,10)$, and
$(6,7)$. For these configurations, the HRS resource estimates give Toffoli
counts of 20,504, 19,090, 21,180, and 23,254. They give qubit counts of
161, 199, 238, and 276.

Those Toffoli figures cover computation only; the HRS table states that
uncomputation adds another factor of two. For an even or odd function, the
printed degree is the cost of evaluation, while the implemented polynomial
has degree $2d$ or $2d+1$. The HRS appendix gives the parallel
piecewise-evaluation qubit formula

$$
(d+1)n+\left\lceil\log_2 M\right\rceil+1.
$$

The published qubit counts match $n=39$ in the formula without its
trailing unit:
$4\cdot39+5=161$, $5\cdot39+4=199$, $6\cdot39+4=238$, and
$7\cdot39+3=276$. Matching every row determines the 39-bit width. The seven
integer bits, including the sign, follow because $w=x^2<64$ needs six
magnitude bits and a sign bit.

The baseline uses degree 3 on 32 intervals in $(n,p)=(40,7)$ fixed-point
arithmetic without pebbling. This is the cheapest configuration of the
published design that meets this task's error band. It contains the
squarer, interval selection, Horner evaluation, sign fold, final modular
addition into `k`, and complete uncomputation. The baseline evaluates each
polynomial in the interval-local coordinate $t=w-a_l$, which meets the
required precision at these word sizes. The baseline therefore includes the reversible subtraction of
the selected left endpoint. It is not the printed HRS $(3,32)$ row. That row
estimates a 39-bit construction under the HRS accounting; the baseline is a
built 40-bit circuit.

A rigorous per-interval upper bound combines the error of the rounded
polynomial, coefficient quantization, truncation, and rounding for every
accepted interval. The bound stays below 429.4967 units, which forces the
integer residue to have magnitude at most 429. The bound replaces an
exhaustive check, which the $2^{36}$ in-domain inputs rule out.

The HRS closed-form accounting extends to the local-coordinate offset, sign
fold, and accumulation. For this configuration it then estimates 43.2k
Toffolis and 245 qubits. The measured circuit uses fewer Toffolis than that
estimate. It uses temporary-AND adders, truncated addend windows,
AND-ladder coefficient selection, constant-folded comparators, and a
Toffoli-free label increment. Its peak qubit count exceeds the estimate
because the unpebbled evaluation keeps five live words and counts
multiplier transients.

## 6. Testing

The grader makes 72,192 pseudorandom draws. Each draw independently samples
uniform 39-bit patterns for `X` and the initial `k`. The grader then skips
out-of-domain inputs. Exactly one eighth of all `X` patterns lie in
$[-8,8)$, so about 9,024 tests run. Their inputs are uniform over the
$2^{36}$ in-domain patterns. The random `k` exercises modular wraparound and
distinguishes addition from overwriting the accumulator.

There is no allowance for failed shots. The precision band already applies
to each shot, so a circuit spends its approximation inside that band. Every
tested shot must preserve `X` exactly and return an accumulator residue
whose recentered error is at most 429 ULP in magnitude.

## 7. Algorithm notes

One family reduces the polynomial's initial argument through the identity

$$
e^{-w}=\left(e^{-w/2^K}\right)^{2^K}.
$$

A larger $K$ shrinks the approximation interval and can lower the required
polynomial degree, but it adds $K$ fixed-point squarings. The error budget
must cover the initial approximation and coefficient quantization, and also
rounding and truncation at every later squaring. Errors before the
squaring chain also propagate through all remaining stages. Stronger
argument reduction therefore simplifies the initial fit, but it lengthens
the arithmetic chain and tightens the intermediate precision.

Piecewise methods trade coefficient-selection logic against arithmetic
complexity. They can vary degree and precision where the Gaussian is flat,
steep, or below the output scale.

## 8. References

- Häner, Roetteler, and Svore (2018), arXiv:1805.12445. Reversible
  fixed-point polynomial evaluation and the published Gaussian resource
  rows.
- Babbush et al. (2016), arXiv:1506.01020. On-the-fly integral computation
  with a basis-function oracle.
- Kitaev and Webb (2008), arXiv:0801.0342. Gaussian-superposition
  preparation from theta-function sums.
- Jordan, Lee, and Preskill (2012), arXiv:1111.3633. A free-field vacuum
  prepared by the Kitaev–Webb method.
- Sanders et al. (2020), arXiv:2007.07391. Variable-spaced table lookup that
  stops at subtrees with one shared value.
- Berry et al. (2019), arXiv:1902.02134. Measurement-based uncomputation of
  table lookups.
- McArdle, Gilyén, and Berta (2022), arXiv:2210.14892. Quantum state
  preparation, including Gaussian amplitudes.
