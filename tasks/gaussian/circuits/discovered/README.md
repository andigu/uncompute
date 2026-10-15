# discovered — content-pruned quadratic QROM evaluator

This folder holds the discovered circuit for the `gaussian` task. Its
certified cost is 1,583 Toffolis and 139 peak qubits, a TQ-cost of 220,037.
The accompanying paper (arXiv: TBD) describes it in the Supplementary
Information, section "Numerical function evaluation", subsection
"Fixed-point Gaussian".

The content-pruned quadratic QROM evaluator divides the folded magnitude into
$1{,}024$ equal segments. It reads a local quadratic from a reversible
read-only table lookup, or QROM. The table stores the quadratic as a constant
and a slope that changes linearly across the segment. Evaluating that form directly
avoids materializing either $x^2$ or a full product register.

Only 514 segments need nonzero words. The lookup stops as soon as an address
subtree has uniform contents. Narrow signed-digit products retain only columns
that can affect the result. The circuit measures exhausted coefficient fields
in the $X$ basis and frees them. Repairing the resulting address-dependent
phases later is measurement-based uncomputation. For negative inputs, a
CNOT-only fold deliberately omits the carry needed for exact absolute value.
The error ledger charges its one-unit displacement.

```text
X, k ← the input and accumulator registers                                      ▷ init
if X₃₈ = 1: complement X₀…₃₇                                                    ▷ fold
j ← X₃₄…₂₅; D ← X₂₄…₀; U ← top₁₁(D); (a₀, s, c) ← pruned tree at j              ▷ lookup
k ← k + a₀·2⁷; measure a₀ in the X basis                                        ▷ a0add
for each t ∈ {0,…,5}: s ← s + ⌊uₜ·c·2²ᵗ⁻¹⁵⌋                                     ▷ mid_mul
k ← k − ⌊s·D/2²¹⌋ using nine search-chosen truncated rows                       ▷ final_rows
for each t ∈ {0,…,5} in reverse order: s ← s − ⌊uₜ·c·2²ᵗ⁻¹⁵⌋                    ▷ mid_unmul
measure s and c in the X basis; repair all recorded phases in one pruned pass   ▷ unlook
if X₃₈ = 1: complement X₀…₃₇                                                    ▷ unfold
```

Here $X$ and $k$ are the preserved input and modular accumulator.
$j=X_{34\ldots25}$ is the ten-bit segment address, and $D=X_{24\ldots0}$ is
the 25-bit coordinate within that segment. $U$ is the top eleven bits of
$D$, the only bits that survive the slope-correction scaling. $a_0$, $s$,
and $c$ are the constant, slope, and signed slope correction, and $u_t$ is
radix-4 digit $t$ of $U$. The shifts by 7, 15, and 21 align the stored
fields. The six and nine retained rows are the rows capable of reaching
those aligned outputs.

The sections "Uniform-subtree coefficient lookup" and "Direct truncated
quadratic accumulation" cover most of the Toffoli work. The section
"Batched measurement phase repair" explains why clearing the coefficient
fields costs less than reversing the lookup.

The offline construction is deterministic end to end. A fixed-seed draw
builds the sampling set. The circuit simulation that ranks candidates also
has a fixed seed. All floating-point work uses ordinary double-precision
arithmetic. An
aligned-stretch scan exhaustively re-proves the error bound over every
25-bit coordinate in every relevant segment. The evaluator then
tests the finished circuit independently on random inputs.

## Searched local model and signed fold (fold)

Let the output-scaled Gaussian on the nonnegative coordinate be

$$
g(y)=2^{32}e^{-y^2}.
$$

The segment address and local coordinate decompose the folded input as

$$
y=j\,2^{-7}+D\,2^{-32},
\qquad
0\le j<1{,}024,\quad 0\le D<2^{25}.
$$

Thus the ten address bits select intervals of width $2^{-7}$ without
comparisons or reversible boundary subtraction. Inside one interval, a slope
that varies linearly with $D$ can represent a quadratic value. The constant
anchors the value, and the initial slope supplies the linear term.
Multiplying the changing slope by $D$ supplies the quadratic term.
This representation matches the two in-place products used by the circuit.

For each segment, the offline search fits a quadratic to $g$ on 129 uniform
nodes by minimizing squared residuals, a least-squares fit. Its design matrix
has columns $1$, $u$, and $u^2$ in the normalized within-segment coordinate
$u$. The negated linear coefficient, clamped at zero, becomes the initial
slope. The negated quadratic coefficient becomes the slope correction.
Quantization can move the optimum because the circuit evaluates integers
with truncated products rather than real coefficients. The search therefore
tries 20 nearby quantized variants per segment and retains the per-segment
minimum. The variants combine five correction offsets and four slope offsets.

This offline search produces the table values, truncation schedule, and
region boundaries against the inclusive 429-output-ULP per-input criterion.
Its candidate space is the full grid of 15,552 knob tuples in
`gen_constants.py` (`SPACE`). One knob, `CAPW`, caps the columns of the
final Booth rows. A full run takes about 100 to 200 CPU-hours.

A deterministic sample first rejects visibly poor candidates. The search then
builds each survivor as a complete circuit and ranks it by executed Toffolis
times peak qubits. The ranking prices arithmetic simplifications and
temporary workspace together rather than estimating them separately. The
search accepts the first ranked tuple that passes the exact scan below.

For the final proof, the truncated Booth sum is constant on aligned stretches
of $D$. The ideal Gaussian is monotone within each segment, so the largest
absolute discrepancy on such a stretch occurs at an endpoint. Checking both
endpoints of every stretch therefore covers the full $2^{25}$-point
coordinate range. The check does not assume that the earlier sample found
the worst case. The scan also recenters each segment constant within its exact error
interval before accepting a candidate.

Every cheaper candidate in the ranking fails the exact scan. For example,
the neighboring tuple with CAPW = 17 saves five Toffolis but scans to 492.28.
The selected tuple scans to 428.28 output units. The cheaper arithmetic lies
outside the required band; it does not merely have less comfortable margin.

The search assigns quadratics to segments 0 through 454. Their exact
unrounded supremum is 428.28 output units. Segments 455 through 513 are flat
enough for constants. These 59 segments collapse into 22 contiguous runs of
shared words. Their supremum is 427.89. At $y=4.015625$, the ideal scaled value is
426.44 and decreases thereafter. Segments 514 through 1023 may therefore
return zero. One segment earlier, at $y=4.0078125$, the value is 454.02, which
explains why the zero region cannot begin sooner.

For a negative two's-complement input, exact absolute value would complement
the low 38 bits and add one. The circuit performs only the complement, which
costs CNOTs but leaves the magnitude code one unit below $|x|$. The effect of
that displacement follows from the maximum Gaussian slope:

$$
\max_y 2y e^{-y^2}
=\sqrt{2}\,e^{-1/2}
\approx0.8578.
$$

One input code unit represents $2^{-32}$ in $y$, and the output scale is
$2^{32}$, so those scale factors cancel. The omitted increment can
therefore change the output by at most 0.858 output units. Combining this
with the exact scan and then rounding gives an integer residue of at most
428 units for $x\ge0$ and 429 units for $x<0$.

The 25-bit constant field has one deliberate saturation case. After its
shift by $2^7$, its largest representable contribution is

$$
(2^{25}-1)2^7=2^{32}-128,
$$

whereas the ideal value at the origin is $2^{32}$. The complete emitted
addend at $x=0$ is 160 units below the ideal and remains inside the band.
Adding another constant bit would enlarge every table word and its addition
to represent one exceptional value more closely.

## Uniform-subtree coefficient lookup (lookup)

The address selects a 55-bit word consisting of a 25-bit constant, an 11-bit
signed correction, and 19 stored slope bits. The two low slope bits are
always zero. Wiring them as implied zeros avoids storing information that
never varies.

A binary address tree forms the conjunction of the branch decisions along
the current path. Each descent needs one temporary AND, and at most nine are
live simultaneously. At a leaf, conditioned CNOTs write the selected word.
A subtree containing only zeros needs no descent or write. More generally,
every leaf below a node can contain the same word. One conditioned write at
that node then replaces all work below it. Stopping a decision tree at uniform
subtrees is content pruning.

An unpruned walk over the same $1{,}024$ addresses measures $1{,}022$
Toffolis. Pruning the zero suffix and repeated-word subtrees reduces the walk
to 495, a saving of 527 Toffolis. This is a property of the table contents,
not merely the address count. The quadratic region has many distinct words,
whereas the constant and zero regions create large uniform subtrees.

The 139-qubit peak ($78+52+9$) follows from the input domain. There are 78
persistent input and accumulator wires and nine branch ANDs, but the 55-bit
word needs only 52 fresh wires. After the sign fold, $X_{35}$ through
$X_{37}$ are zero for every accepted input, so those three input wires
temporarily hold table bits. Once the circuit measures the fields away,
the phase repair (see "Batched measurement phase repair") can use the same wires.

## Direct truncated quadratic accumulation (a0add, mid_mul, final_rows, mid_unmul)

Writing a quadratic as a linearly changing slope gives the circuit the
in-place evaluation

$$
s'
=
s+\sum_{t=0}^{5}
\left\lfloor u_t c\,2^{2t-15}\right\rfloor,
\qquad
k
\leftarrow
k+2^7a_0-
\left\lfloor\frac{s'D}{2^{21}}\right\rfloor.
$$

Because $U$ is the top eleven bits of $D$, it approximates
$D/2^{14}$. The first sum consequently supplies the scaled correction
$cD/2^{29}$ to the slope. Multiplication of that corrected slope by $D$
then produces both the linear contribution and the term quadratic in $D$.
The circuit adds the constant directly to $k$. The first product changes $s$
in place, and the second product changes $k$ in place. No register ever holds
$D^2$ or either full product.

Radix-4 Booth recoding examines overlapping pairs of multiplier bits. It
replaces them by signed digits that select zero, one, or two times the
multiplicand. The overlap carries the sign information needed to
make adjacent digits represent the original binary integer. CNOTs and one
AND reduce each digit to a sign control and two mutually exclusive magnitude
controls. Since at most one control is active, the one-times and two-times
partial products in a column share an ancilla. Conditional complements
around an addition implement a negative digit without constructing a
separate negated operand.

The eleven-bit $U$ requires six Booth rows for the slope correction. One
shared AND propagates the signed correction's extension through all high
columns. Reversing those same six rows after accumulation restores the
looked-up slope exactly, which is necessary before its measurement-based
uncomputation. The final slope product would have more rows at full
precision. Only nine of them can reach retained accumulator bits. Complementing
$k$ before and after those rows converts the required subtraction into the
same addition machinery used for positive rows.

The search-chosen truncation acts at the partial-row level. The correction
product has a separate low cutoff for each row. Each final row retains at
most 18 columns and never touches $k$ below bit 6. The search omits the two
lowest rows. A conditioned bit immediately below a cutoff rounds the retained
part according to the discarded tail. It does not always round in the same
direction. This centers the truncation error around zero. The fitted
constants can therefore absorb its residual without a systematic bias.

Each row temporarily forms column ANDs and erases them by measurement in
newest-first order after the addition. Carry handling is also selected per
position. A carry erased by measurement costs one Toffoli and one extra live
qubit, while an in-place carry costs two Toffolis and no extra qubit. Using
only the former would exceed the qubit ceiling that the lookup already sets.
Using only the latter would spend unnecessary Toffolis. Each addition
therefore uses measurement-erased carries where workspace is available and
in-place carries for the remaining positions.

## Batched measurement phase repair (unlook)

Suppose one looked-up coefficient bit has value $b(j)$ at address $j$.
Measuring that bit in the $X$ basis disentangles and frees its qubit, but an
outcome $m\in\{0,1\}$ leaves the address amplitude multiplied by

$$
(-1)^{m b(j)}.
$$

The outcome is classical and known, so the unwanted phase is a known Boolean
function of the address. The circuit need not repair it immediately. It
measures the constant field after its addition. The slope and correction
remain until the reverse Booth rows restore their looked-up values. Their
measurements then produce phase functions of the same address. One pass
can therefore combine all three sets of repairs.

The repair groups addresses into blocks of sixteen. It expands the four low
address bits into a rail with exactly one active position, a one-hot rail.
A pruned tree visits only relevant prefixes of the six high bits. The repair
can then apply a phase that any measured field requests at the
corresponding block and rail position. It does not reconstruct that field's
55-bit word.

The rail uses layers of 2, 4, 8, and 16 live wires. Its first three layers
require 14 ANDs, and the pruned walk over the six high bits requires 36 more.
The joint cleanup therefore costs $50=14+36$ Toffolis. The rail borrows the three
folded-input wires that held table bits during the lookup. They host part of
the rail after the circuit frees the coefficient fields.

The straightforward alternative would keep or recreate the coefficient
registers and run the 495-Toffoli lookup backward. The deferred joint repair
clears the same address-dependent phases in 50 Toffolis. It also lets the
circuit release each field's qubits as soon as the field's numerical value
is not needed.
