# lattice-gauge-plaquette — nonabelian lattice-gauge simulation

This task computes one Symanzik-improved magnetic layer on a periodic $4^3$
lattice. The circuit multiplies the $\Sigma(72\times3)\subset SU(3)$ link
variables around every plaquette and rectangle and takes their real traces.
It adds the weighted sum into a 51-bit register and leaves all 192 links
unchanged. The accompanying paper (arXiv: TBD) describes the task and the
discovered circuit in its supplementary section "Nonabelian lattice-gauge
simulation".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. The evaluator measures both numbers from the
emitted op stream. [results.jsonl](results.jsonl) holds the certified rows.

| circuit | Toffoli | peak qubits | TQ-cost |
|---|---:|---:|---:|
| [discovered](circuits/discovered/) | 172,128 | 1,859 | 319,985,952 |
| [osorio-composed](circuits/osorio-composed/) (baseline) | 1,271,424 | 1,927 | 2,450,034,048 |

The discovered circuit uses 7.4× fewer Toffolis and 3.5% fewer qubits than
the baseline. Its TQ-cost is 7.7× lower.

The evaluator counts every executed Toffoli (CCX or CCZ) and the peak number
of live qubits. A CX or CZ conditioned on a measured bit also counts as a
Toffoli. Clifford gates and classical bit operations are free. The
persistent registers already use $192\cdot9+51=1{,}779$ qubits, so only the
workspace above that floor can shrink.

## 1. Operation

A lattice-gauge configuration assigns a group element to each directed link.
It describes parallel transport along that link; traversal in the opposite
direction contributes its inverse. Because the group is nonabelian, factor
order matters in the product around a closed path, called a Wilson loop.

Changing the local frame conjugates a closed-loop product, leaving its trace
unchanged. Reversing the loop replaces the product by its inverse. Since
$SU(3)$ matrices are unitary, this complex-conjugates the trace, so its real
part is also orientation-independent. Define

$$
S_P=\sum_{192\ \mathrm{plaquettes}} f(P),\qquad
S_R=\sum_{384\ \mathrm{rectangles}} f(R),\qquad
f(g)=2\,\mathrm{Re}\,\mathrm{tr}(M_g).
$$

The circuit implements

$$
\lvert g_0\cdots g_{191}\rangle\lvert k\rangle
\longmapsto
\lvert g_0\cdots g_{191}\rangle
\lvert k+a_P S_P+a_R S_R\bmod 2^{51}\rangle.
$$

| register | width | meaning | required final value |
|---|---:|---|---|
| `g0` … `g191` | 9 qubits each | Link group elements | Unchanged |
| `k` | 51 qubits | Modular accumulator | $k+a_PS_P+a_RS_R\bmod 2^{51}$ exactly |

Sites use $s=x+4y+16z$. The link from site $s$ in direction
$\mu\in\{0,1,2\}$ has index $\ell=3s+\mu$. The registers come in
increasing $\ell$ order, then `k`.

## 2. Applications

The group-register formulation of lattice gauge theory (Lamm, Lawrence, and
Yamauchi (2019), arXiv:1903.08807) stores each link in a finite group basis.
A Trotter step separates the kinetic evolution in the group Fourier basis
from the magnetic evolution. The magnetic evolution is diagonal in the
group-element basis. The kinetic layer lies outside the classical reversible
gate set of this circuit model.

The improved Hamiltonian of Carena et al. (2022),
arXiv:2203.02823, adds planar rectangles to the plaquette magnetic term. At
the tree-level improvement point,

$$
\beta_{V0}=\frac{5}{3},\qquad
\beta_{V1}=-\frac{1}{12},\qquad
\frac{\beta_{V1}}{\beta_{V0}}=-\frac{1}{20}.
$$

This combination cancels the leading lattice-spacing error. All loop
observables are diagonal and commute, even when their loops share links.
Their weighted sum can therefore drive the complete magnetic phase through
addition into a phase-gradient state. This task checks that addition for
arbitrary computational-basis `k`. Preparation of the gradient state and
conversion from integer weights to a physical angle are outside the task.

Osorio Perez et al. (2025), arXiv:2511.17437,
construct inversion, multiplication, and trace primitives over
$\Sigma(72\times3)$. This task applies them to the complete magnetic layer:
every plaquette and rectangle of the lattice.

## 3. Specification

### 3.1 Data representation

The finite subgroup $G=\Sigma(72\times3)\subset SU(3)$ of order 216 replaces
the continuous link domain. This gives each link a finite
basis while retaining nonabelian multiplication, inverses, and the defining
matrix trace.

In the encoding of Osorio Perez et al. (2025), arXiv:2511.17437, every
element has the unique normal form

$$
g=\omega^p C^q E^r V^{2s+t}X^u,
$$

where $p$, $q$, and $r$ are three-valued exponents and $s$, $t$, and $u$
are binary. These coordinates specify a unique ordered generator product
despite the generators not commuting. The nine-bit encoding is

$$
\lvert u\,t\,s\,r_1r_0\,q_1q_0\,p_1p_0\rangle.
$$

There are 216 valid codes among 512 bit patterns, and code `0` is the
identity. Invalid patterns are outside the input promise.

Conjugating an element does not change its trace, so
$f(hgh^{-1})=f(g)$ and $f$ is a class function. It also satisfies
$f(g^{-1})=f(g)$. Its exact values and multiplicities are

| $f$ | +6 | −3 | −2 | +1 | 0 | +2 | −1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Elements | 1 | 2 | 9 | 18 | 24 | 54 | 108 |

The value $+6$ occurs only at the identity, while $-3$ occurs at the two
nonidentity central elements. For a uniform group element,

$$
\mathbb E[f]=0,\qquad
\mathbb E[f^2]=\frac{432}{216}=2,\qquad
\mathrm{Var}[f]=2.
$$

Any two distinct loops differ in at least one link. Condition on every link
except one that lies in the first loop and not the second. The second loop
product is fixed, while the first remains uniform because multiplying a fixed
element by a uniform one is uniform. Its conditional mean is therefore zero,
so the covariance vanishes and the variances add:

$$
\sigma(S_P)=\sqrt{192\cdot2}=\sqrt{384}=19.5959\ldots\approx19.6,\qquad
\sigma(S_R)=\sqrt{384\cdot2}=\sqrt{768}=27.7128\ldots\approx27.7.
$$

The full ranges are

$$
S_P\in[-576,1152],\qquad S_R\in[-1152,2304].
$$

The accumulator is an unsigned residue modulo $2^{51}$, with errors expressed
as centered modular residues.

### 3.2 Task instance

The lattice has 64 sites, three positive directions, and 192 links.
Coordinates wrap periodically after four sites in each direction. For
$i<j$, the plaquette product is

$$
P_{ij}(\mathbf x)=
g_i(\mathbf x)\,
g_j(\mathbf x+\hat\imath)\,
g_i(\mathbf x+\hat\jmath)^{-1}\,
g_j(\mathbf x)^{-1}.
$$

The pairs $(0,1)$, $(0,2)$, and $(1,2)$ give 192 plaquettes. For ordered
$i\ne j$, the rectangle product is

$$
R_{ij}(\mathbf x)=
g_i(\mathbf x)\,
g_i(\mathbf x+\hat\imath)\,
g_j(\mathbf x+2\hat\imath)\,
g_i(\mathbf x+\hat\imath+\hat\jmath)^{-1}\,
g_i(\mathbf x+\hat\jmath)^{-1}\,
g_j(\mathbf x)^{-1}.
$$

The ordered pairs allow either direction to be the long side, giving 384
rectangles. Plaquettes use four distinct links with two forward and two
inverse factors; rectangles use six distinct links with three of each.
Every loop closes under the periodic boundaries.
[reference.py](reference.py) defines the complete loop set.

The 51-bit accumulator width follows from the $10^{-8}$ precision target and
resource assumptions of Osorio Perez et al. (2025), with the phase-error
accounting of Kan and Nam (2021), arXiv:2107.12769. The weights encode the
exact physical ratio $-1/20$. The plaquette weight is

$$
a_P=\mathtt{0x8EBA84840FB}=9{,}808{,}233{,}644{,}283.
$$

It is the top 51 bits, most-significant first, of the SHA-256 of the
instance tag `kickmix-plaquette-trace-accumulate-v2`, forced odd. The tag is
a fixed string; its text carries no meaning. The rectangle weight is

$$
a_R=2^{51}-\mathrm{round}(a_P/20)
=\mathtt{0x7FF8DD12F965A}
=2{,}251{,}309{,}402{,}003{,}034.
$$

Rounding is unambiguous because $a_P\bmod20=3$. No rescaling of the Trotter
step puts both loop families on the accumulator's binary grid, because
$2^s/20$ is never an integer for any $s$. The task therefore fixes an odd $a_P$
and an even $a_R$, and the required map has zero numerical tolerance.

### 3.3 Correctness and examples

Define

$$
\Delta=(k_{\mathrm{out}}-k_{\mathrm{in}})
-a_PS_P-a_RS_R\pmod{2^{51}}.
$$

Correctness requires every link to read back exactly and

$$
\Delta=0.
$$

Both the grader and [reference.py](reference.py) reproduce these exact
anchors. Increments use $k_{\mathrm{in}}=0$.

| configuration | $S_P$ | $S_R$ | exact increment |
|---|---:|---:|---:|
| All links identity | 1152 | 2304 | 1,161,977,387,651,968 |
| One link code 9, rest identity | 1128 | 2232 | 961,889,421,308,584 |
| $g_\ell$ = the $(\ell\bmod216)$-th smallest valid code | −100 | 31 | 1,255,773,687,108,314 |

The one-link result is unchanged at links `1`, `47`, `95`, `100`, and `191`.
For the final anchor, $k_{\mathrm{in}}=5$ gives
$k_{\mathrm{out}}=1{,}255{,}773{,}687{,}108{,}319$.

## 4. Discovered circuit

[`circuits/discovered/`](circuits/discovered/) combines three ideas. The
accompanying paper describes them in its section "Nonabelian lattice-gauge
simulation".

- **Specialized running-product representation.** The circuit keeps one
  running product $W$. It stores $W$ in a factor order that differs from the
  order of the links, with the coset representative $X_o=CEX$ in place of
  $X$. The three-valued coordinates of $W$ are one-hot. One update
  $W\gets Wg$ costs 35 Toffolis, against 146 for a general multiplication.
  Multiplying by an inverse link costs the same.
- **Reuse of loop products.** The real trace is invariant under cyclic
  rotation and inversion of a loop. The circuit edits $W$ from one loop
  representative into the next and keeps the shared factors. A classical
  dynamic program picks the representatives and the order. The schedule uses
  3,136 link multiplications and 480 inversions of $W$ (21 Toffolis each)
  for the 576 loops.
- **Direct trace readout.** A small Boolean circuit reads the trace class
  from the coordinates of $W$ for 20 Toffolis. Each of six flags controls the
  addition of a known constant. Readout, addition, and cleanup cost 76
  Toffolis per plaquette and 75 per rectangle.

The Toffoli count splits as follows. Updates of $W$ use 109,760, trace
readout and addition 43,392, inversions of $W$ 10,080, link recoding 8,832,
and segment anchors 64.

## 5. Baseline

[`circuits/osorio-composed/`](circuits/osorio-composed/) treats all 576 loops
independently. It reimplements, from the published descriptions, the
inversion and multiplication circuits of Osorio Perez et al. (2025),
arXiv:2511.17437. Their inner $\Sigma(36\times3)$ blocks come from Gustafson
et al. (2024), arXiv:2405.05973. For each loop, the baseline builds the
product from the identity. It then loads the 51-bit weighted trace by
unary-iteration QROM (Babbush et al. (2018), arXiv:1805.03662). A ripple
adder (Gidney (2018), arXiv:1709.06648) adds it to `k`. The baseline then
reverses the lookup and the loop product.

A multiplication costs 146 Toffolis and a link inversion 87. A plaquette
costs 1,702 Toffolis and a rectangle 2,460. Rectangles dominate, because
there are twice as many and each has six links rather than four.

## 6. Testing

The evaluator runs exactly 100,000 pseudorandom tests and skips none. It
draws links independently and effectively uniformly over the 216 valid
codes; the wide-word reduction has bias at most $2^{-56}$. The initial
accumulator is exactly uniform over all $2^{51}$ residues, making modular
wraparound routine rather than an isolated edge case.

Every test evaluates all 576 loops, for 57,600,000 exercised loop products.
The random suite does not reach the extreme configurations of section 3.3
because $\sigma(S_P)\approx19.6$ and $\sigma(S_R)\approx27.7$,
whereas their range widths are 1,728 and 3,456. The anchors cover those
configurations separately and are not part of the random draw.

The failure allowance is zero out of 100,000. One wrong trace class changes
`k` by $a_Pd$ for a plaquette or
$-\mathrm{round}(a_P/20)d$ for a rectangle, with
$d\in\{1,\ldots,9\}$. The minimum centered magnitudes are exactly
$9{,}808{,}233{,}644{,}283$ and
$\mathrm{round}(a_P/20)=490{,}411{,}682{,}214$. Thus even the smallest
misclassification is sharply separated from the correct residue, and no
near-miss allowance is useful.

## 7. Algorithm notes

For ordered factors $g_1,\ldots,g_m$, cyclicity and inversion invariance give

$$
f(g_1g_2\cdots g_m)
=f(g_2\cdots g_mg_1)
=f((g_1g_2\cdots g_m)^{-1}).
$$

A circuit may therefore choose a different starting edge or inverse
orientation for a loop, exposing shared prefixes or suffixes without changing
its trace. Arbitrary factor reordering remains invalid because the group is
nonabelian.

Every link appears in four plaquettes and 12 rectangles, including eight
long-side and four short-side rectangle uses. Hence

$$
192\cdot4+384\cdot6=3{,}072=192\cdot16.
$$

The graded target is the single weighted layer sum, not 576 separately
exposed trace values, so no intermediate trace must survive in the output.
Retaining partial products can use this incidence overlap. The circuit
must still transform those products into later ordered loops and eventually
uncompute them. Sharing therefore trades repeated multiplication against
workspace and reversible scheduling constraints.

A lookup family addresses 216 group codes and materializes a 51-bit weighted
addend. A direct-classification family instead identifies one of seven trace
values and controls its constant addition, trading QROM storage and cleanup
for Boolean classification. In either family, persistent I/O already uses

$$
192\cdot9+51=1{,}779
$$

qubits. Only the workspace above the 1,779 mandatory qubits is available to
reduce. The qubit axis is therefore nearly saturated, while the
executed-Toffoli axis has no comparable floor.

## 8. References

- Lamm, Lawrence, and Yamauchi (2019), arXiv:1903.08807. The
  group-register formulation and its primitive operations.
- Carena et al. (2022), arXiv:2203.02823. The improved
  magnetic Hamiltonian, the rectangle terms, and the tree-level couplings.
- Osorio Perez et al. (2025), arXiv:2511.17437. The
  group encoding and the inversion, multiplication, and trace primitives.
- Gustafson et al. (2024), arXiv:2405.05973. The $\Sigma(36\times3)$
  blocks that the $\Sigma(72\times3)$ construction extends.
- Babbush et al. (2018), arXiv:1805.03662. The unary-iteration QROM of the
  baseline lookup.
- Gidney (2018), arXiv:1709.06648. Temporary logical ANDs and the ripple
  adder of the baseline.
- Kan and Nam (2021), arXiv:2107.12769. The phase-error accounting behind
  the 51-bit accumulator width.
