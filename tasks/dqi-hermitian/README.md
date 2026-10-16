# dqi-hermitian — Hermitian-code syndrome decoding for DQI

This task applies the sign $(-1)^{\langle c,e(s)\rangle_2}$ to a register
holding a syndrome $s$. Here $e(s)$ is the unique error of symbol weight at
most 47 with $He=s$ over $\mathrm{GF}(64)$. The circuit receives 123
six-qubit syndrome symbols and must return them unchanged. This is the
decoder-dependent phase of decoded quantum interferometry (DQI) for the
512-position Hermitian optimal polynomial intersection (HOPI) instance of
Gu and Jordan (2025), arXiv:2510.06603. The accompanying paper (arXiv:2610.22367)
treats this task in the SI section "Hermitian-code decoding for HOPI".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. A gate controlled on a classical bit counts at
its promoted arity. Clifford gates are free. The evaluator measures both
numbers from the op stream. `results.jsonl` holds the certified values.

| circuit | role | Toffoli | peak qubits | TQ-cost |
|---|---|---:|---:|---:|
| [`discovered`](circuits/discovered/README.md) | discovered circuit | 926,912 | 2,395 | 2.22×10⁹ |
| [`sakata-fengrao`](circuits/sakata-fengrao/README.md) | baseline: Feng–Rao majority voting, Sakata, Jensen, and Høholdt (1995) | 104,263,342 | 194,053 | 2.02×10¹³ |

The discovered circuit uses 110× fewer Toffolis and 81× fewer qubits than
the baseline, a 9,100× lower TQ-cost. The accompanying paper reports 30
wrong signs in 100,000 tests for the discovered circuit, within the
allowance of 100.

## 1. Operation

Let $H$ be the fixed $123\times512$ parity-check matrix defined under the
task instance, and let $c$ be the fixed 3072-bit mask. For every error
$e\in\mathrm{GF}(64)^{512}$ of symbol weight at most 47, the circuit must act
on its syndrome $s=He$ as

$$
\lvert s\rangle\longmapsto
(-1)^{\langle c,e\rangle_2}\lvert s\rangle.
$$

Here $\langle c,e\rangle_2$ is the binary inner product of $c$ with the
3072-bit expansion of $e$. Symbol $e_i$ occupies bits $6i$ through $6i+5$.
The designed-distance guarantee ensures that at most one error of weight at
most 47 has a given syndrome. The required sign therefore depends only on
the input registers.

| register | width | meaning | required final value |
|---|---:|---|---|
| $s_0,\ldots,s_{122}$ | 6 qubits each | Syndrome symbol $s_r\in\mathrm{GF}(64)$ | Unchanged |

The registers follow parity-check-row order, for 738 input qubits in
total. There is no output register: the result is solely the relative
phase applied to each computational-basis branch. Any decoded-error data
must be temporary.

## 2. Applications

DQI (Jordan et al. (2025), arXiv:2408.08292) turns max-LINSAT optimization
into coherent syndrome decoding. Its branches represent errors drawn from a
prescribed weight distribution. The decoder supplies the objective-dependent
sign that makes those branches interfere. The objective coefficients enter
this step through the fixed mask $c$. The
[dqi-reed-solomon](../dqi-reed-solomon/README.md) task asks for the same
phase map for the Reed–Solomon instance of Khattar et al. (2025),
arXiv:2510.10967.

Gu and Jordan (2025) replace the Reed–Solomon code with a one-point
Hermitian code. Distinct field elements must label the Reed–Solomon
positions, so six-bit symbols limit a Reed–Solomon code over
$\mathrm{GF}(64)$ to 64 positions. Reaching 512 positions would require
nine-bit symbols. A Hermitian curve over $\mathrm{GF}(q^2)$ instead
provides $q^3$ affine rational points. With $q=8$, it supplies all 512
positions with six-bit $\mathrm{GF}(64)$ symbols.

## 3. Specification

### 3.1 Data representation

- The field is
  $\mathrm{GF}(64)=\mathrm{GF}(2)[x]/(x^6+x^4+x^3+x+1)$. Elements are
  integers from 0 through 63 in the polynomial basis, with bit $t$ equal to
  the coefficient of $x^t$. This modulus is the `galois` library default,
  and the residue class of $x$ is primitive.
- Each syndrome register is little-endian: qubit $t$ of register $s_r$
  carries the coefficient of $x^t$ in that field element.
- The error word contains 512 symbols in the same basis. Symbol $e_i$ maps
  to mask bits $6i$ through $6i+5$, and the phase exponent is their binary
  inner product with $c$.

### 3.2 Task instance

The code lives on the genus-$28$ Hermitian curve

$$
y^8+y=x^9
$$

over $\mathrm{GF}(64)$. Its 512 affine rational points follow lexicographic
order as integer pairs $(x,y)$, with $x$ as the major key and $(0,0)$
first.

At the curve's point at infinity, $x$ and $y$ have pole orders 8 and 9.
Consequently, the monomial $x^ay^b$ has pole order $8a+9b$. The rows of $H$
are the 123 monomials satisfying

$$
0\le b\le7,\qquad 8a+9b\le150,
$$

ordered first by $b$ and then by $a$, and evaluated at the 512 affine
points. Restricting $b$ to this range loses nothing because the curve
equation reduces every occurrence of $y^8$.

The kernel of $H$ is a $[512,389]$ code. Its designed minimum distance obeys

$$
d\ge150-2(28)+2=96,
$$

so syndromes uniquely identify errors through symbol weight

$$
\left\lfloor\frac{96-1}{2}\right\rfloor=47.
$$

These are precisely the in-domain inputs. The circuit may implement any
reversible behavior on syndromes that have no such error.

Counter-mode SHA-256 generates the fixed mask $c$, with the fixed
instance tag `b"kickmix/hopi-512/fixup-c/v1"` and a little-endian 32-bit
counter. The tag is a fixed byte string; its text carries no meaning. The
generator truncates the stream to 384 bytes and packs it little-endian into
3072 bits. Bit $k$
of $c$ is bit $k\bmod8$ of byte $\lfloor k/8\rfloor$. There are no unused
trailing bits. The stream's SHA-256 digest is:

`10fe6addc16595734e774145bfb571af2f11870903306a5f047aaebd8a7b3181`

The mask has 1,545 set bits and lies outside the binary row space of the
bit-expanded parity-check matrix. Thus no linear function of the supplied
syndrome reproduces its phase. The executable
[classical oracle](reference.py) reconstructs the curve, matrix, mask, and
required sign.

### 3.3 Correctness and examples

A tested input is correct exactly when all 123 syndrome registers finish
with their initial values and the branch acquires
$(-1)^{\langle c,e\rangle_2}$. The evaluator checks register restoration and the
sign exactly; section 6 describes the failure allowance.

Every evaluation includes the following four fixed examples, recomputed
against the classical oracle:

| test error $e$ | syndrome $s$ | sign |
|---|---|---:|
| Zero | All 123 symbols 0 | $+1$ |
| Value 1 at point $(0,0)$ | $s_0=1$, all other symbols 0 | $-1$ |
| Value 42 at point $(17,15)$ | All 123 symbols nonzero: $s_0=42$, $s_1=9$, $s_2=47,\ldots$ | $+1$ |
| Value 63 at point $(63,61)$ | All 123 symbols nonzero: $s_0=63$, $s_1=8$, $s_2=34,\ldots$ | $-1$ |

In general, an error of value $v$ at a single point $P$ gives

$$
s_r=v\,m_r(P),
$$

where $m_r$ is the monomial defining row $r$. At $(0,0)$ every listed
monomial except the constant vanishes, explaining why that example has only
one nonzero syndrome symbol. Its sign is negative because bit 0 of $c$
equals 1.

## 4. Discovered circuit

The discovered circuit turns locator finding into a null-space problem for
a symmetric $48\times48$ matrix $A$ built from the syndromes. It never
stores $A$. It applies $A$ by evaluation–interpolation: fixed linear maps
evaluate a vector at 123 curve points, 123 pointwise field products apply
the syndrome-derived weights, and fixed linear maps return the result. A
block Lanczos walk over $\mathrm{GF}(64)$ then finds a null vector, which
is the locator. Breakdowns of the walk freeze the affected inputs until
scheduled repair rounds at depth two or three. A striped phase scan then
computes the phase from local coefficients of the locator and the evaluator,
16 curve points at a time, without building the decoded word. The reverse
walk recomputes most decisions and stores one bit per round. The SI section
"Hermitian-code decoding for HOPI" of the accompanying paper gives the
derivation. [circuits/discovered/README.md](circuits/discovered/README.md)
maps it to the code.

## 5. Baseline

The baseline is a direct reversible implementation of the published
Feng–Rao decoding algorithm, because no prior reversible compilation of a
Hermitian-code decoder exists.

### Feng–Rao majority voting

For Reed–Solomon codes, polynomial degree supplies a consecutive sequence
of syndrome constraints. A curve of positive genus has gaps in its possible
pole orders. Those gaps limit how far a decoder can extend a locator
recurrence using only the supplied syndromes. For this genus-$28$ curve,
the basic known-syndrome guarantee stops at

$$
\left\lfloor\frac{d-1-g}{2}\right\rfloor
=
\left\lfloor\frac{96-1-28}{2}\right\rfloor
=33.
$$

The code still determines every error through weight 47 uniquely; the loss
is in the decoding argument, not in the code distance. Between weights 34
and 47, a decoder must recover syndrome values at pole orders beyond the
known range before the locator recurrence can continue.

Feng and Rao (1993), doi:10.1109/18.179340, recover those values in
increasing pole order. Several valid recurrence relations predict each
unknown syndrome. The order bound ensures that the correct prediction gets
the majority while the error weight stays within half the designed
distance. Each accepted value then becomes known data for the next vote.
This closes the genus-induced gap and reaches the full radius of 47.
Sakata, Jensen, and Høholdt (1995), doi:10.1109/18.476248, combine this voting with the
multidimensional Berlekamp–Massey algorithm of Sakata (1990),
doi:10.1016/0890-5401(90)90039-K. Høholdt, van Lint, and Pellikaan (1998)
survey the one-point-code and order-bound theory. The
`sakata-fengrao` circuit executes all candidate updates and votes on a
fixed reversible schedule and is exact at every error weight through 47.
Its large qubit count comes from keeping the intermediate states of the
recurrence updates until the circuit reverses the computation.

## 6. Testing

Testing runs 100,000 randomized errors. The seed derives from the op
stream, so any change to the circuit draws a new test set. The symbol
weights follow the DQI distribution for this instance: the squared
principal eigenvector of the weight matrix of Jordan et al. (2025),
quantized on a $2^{-16}$ probability grid. The distribution is supported on
weights 29 through 47 and has mode 43; weights at most 28 receive no
quantized mass.

Conditional on the weight, the support is uniform among the 512 curve
points. Each selected position receives an independent value drawn
uniformly from the 63 nonzero field elements. The evaluator computes $s=He$,
loads the syndrome registers, and checks both their exact restoration and
the phase. The four fixed examples of section 3.3 cover zero and
single-symbol errors that the randomized distribution never draws.

At most 100 of the 100,000 tests may fail. A wrong decode of a fixed input
gives the wrong sign with probability at least one half over the
measurement record. The accompanying paper uses this to bound the failure
probability $\varepsilon$ under the DQI input distribution: 100 failures in
100,000 tests give $\varepsilon\le2.434\times10^{-3}$ at 97.5% confidence.
At that value, the loss in the expected satisfied fraction of DQI is at
most $7.0\times10^{-3}$. Ideal decoding gives an expected fraction of about
$0.7145$, against about $0.5726$ for one Prange trial, so the loss is at
most 4.9% of the advantage. Roughly half the sampled branches need a
negative sign, so a circuit that omits the phase cannot approach the
allowance.

## 7. Algorithm notes

### Locator spaces beyond the genus-limited radius

The linear-algebra decoder of Skorobogatov and Vlăduţ (1990),
doi:10.1109/18.57204, searches for a nonzero function $\sigma$ that
vanishes at every error point. If the locator space is
$\mathcal L(aP_\infty)$, each unknown error position imposes one
homogeneous linear condition on its coefficients. The syndromes provide the
products needed to solve those conditions, but the standard guarantee is
again limited to the radius of 33.

For weight 47, enlarge the search to $\mathcal L(75P_\infty)$. The divisor
degree exceeds $2g-2$, so the one-point form of the Riemann–Roch theorem
gives

$$
75=47+28,\qquad
\dim\mathcal L(75P_\infty)=75-28+1=48.
$$

At most 47 error points impose at most 47 vanishing conditions on 48
coefficients, so a nonzero locator exists. Without the extra 28 pole
orders, the function space would not contain enough degrees of freedom.

Existence alone does not guarantee that the supplied syndromes identify the
locator. Let $\psi_0,\ldots,\psi_{47}$ be a basis for the enlarged space
and define

$$
A_{k\ell}=S(\psi_k\psi_\ell).
$$

Every product has pole order at most 150, so the 123 input syndromes
determine each entry. If $M_E$ evaluates the basis functions on the error
support and $\Delta_E$ is the diagonal matrix of nonzero error values, then

$$
A=M_E^{\mathsf T}\Delta_E M_E.
$$

A true locator lies in $\ker A$. The converse holds when the evaluation
conditions in $M_E$ are independent, but special supports can make them
dependent. Under the DQI input distribution this happens with probability
about $8\times10^{-8}$. The enlarged-space route therefore reaches weight
47 for typical supports, not for every support. A circuit that uses it must
stay within the failure allowance of section 6. Gu and Jordan (2025)
identify low-overhead reversible decoding beyond the genus-limited radius
as the central circuit problem for this instance.

## 8. References

- Jordan et al. (2025), arXiv:2408.08292. DQI, its decoder-dependent phase,
  and the weight distribution used for testing.
- Gu and Jordan (2025), arXiv:2510.06603. Hermitian-code DQI and the HOPI
  instance family.
- Khattar et al. (2025), arXiv:2510.10967. Reed–Solomon decoding circuits
  for DQI.
- Feng and Rao (1993), doi:10.1109/18.179340. Majority voting for unknown
  syndromes.
- Sakata (1990), doi:10.1016/0890-5401(90)90039-K. The multidimensional
  Berlekamp–Massey algorithm.
- Sakata, Jensen, and Høholdt (1995), doi:10.1109/18.476248. Generalized
  Berlekamp–Massey decoding up to half the Feng–Rao bound.
- Skorobogatov and Vlăduţ (1990), doi:10.1109/18.57204. The linear-algebra
  locator method and its genus-dependent guarantee.
- Høholdt, van Lint, and Pellikaan (1998),
  [survey](https://ruudp.win.tue.nl/paper/31.pdf). One-point codes,
  pole-order bounds, and decoding through half the designed distance.
- Prange (1962), doi:10.1109/TIT.1962.1057777. Information-set decoding.
