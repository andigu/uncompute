# discovered — window-fitted interpolator

This folder holds the discovered circuit for the `coulomb-potential` task. Its
certified cost is 1,140 Toffolis and 142 peak qubits, a TQ-cost of 161,880.
The accompanying paper (arXiv: TBD) describes it in the Supplementary
Information, section "Numerical function evaluation", subsection "Coulomb
potential".

The window-fitted interpolator normalizes every ordinary input onto one fixed
mantissa interval, then approximates the resulting inverse-square-root curve
with 128 quantized lines. The fitting procedure works backward from the final
output band through the circuit's exact rescaling and rounding. It accepts a
line only when every input assigned to it reaches an allowed output. This
avoids separate error budgets for lookup, truncation, multiplication, and
rounding.

The construction spends lookup branches to avoid another polynomial degree
and its full-width reversible multiplication. Its six-bit local coordinate
turns multiplication into six conditional additions. Measurement-based
unlookup makes the larger tree economical by replacing a second tree walk
with a 17-Toffoli phase repair. The resulting pipeline is:

```text
d ← |rᵢ − rⱼ|, coordinate by coordinate                         ▷ diffs
n₂ ← d[x]² + d[y]² + d[z]²                                     ▷ squarer
e ← ⌊log₂ n₂⌋; m ← n₂ · 2^min(17 − e, 15)                     ▷ normalize
s, g ← fitted_tree(parity(e), m[16:11])                        ▷ qrom
b ← m[10:5]; p ← Σ over set bᵢ: ⌊s · 2ⁱ / 4⌋                 ▷ multiply
v ← g − p                                                       ▷ subtract
v ← corrected value if n₂ ∈ {1, 2, 3}                          ▷ patch
w ← round(v · 2^(6 − ⌊e/2⌋))                                  ▷ shift
k ← k + w mod 2²⁵                                              ▷ accumulate
reverse the rescaling                                           ▷ un-shift
reverse the exceptional-input correction                        ▷ un-patch
restore g and p from v                                          ▷ un-subtract
clear p by reversing the product rows                           ▷ un-multiply
measure s and g; repair the address phase                       ▷ un-qrom
reverse normalization of m into n₂                              ▷ un-normalize
clear n₂                                                        ▷ un-squarer
restore rᵢ                                                      ▷ un-diffs
```

Here $r_i$ and $r_j$ are the input points, and $d$ holds their three
coordinate magnitudes. $n_2$ is their squared norm, and $e$ is its
leading-bit position. $m$ is the normalized 18-bit mantissa, and $s$ and
$g$ are the selected slope and intercept. $b_i$ is remainder bit $i$ of the
six-bit local coordinate $b$, and $p$ is the slope contribution. $v$ is the
line value, $w$ is the rescaled output, and $k$ is the accumulator. Among
the constants, 17 aligns the leading bit, 15 is the largest encoded
normalization shift, and 6 restores the output scale. Division by 4
accounts for the slope's two fractional bits.

Most of the remaining nonlinear work is in the squared-distance stage
(see "Packed distance computation").
The circuit stores no coefficient table. Every build recomputes the complete
table deterministically, then exhaustively proves the fit over every
achievable squared distance before emitting the circuit. Five inputs reach
the inclusive error boundary. The evaluator independently derives both
cost and correctness from the emitted op stream alone.

## Binary normalization and backward fitting (normalize, un-normalize)

Normalization is binary scientific notation. Writing the squared distance as
$n_2=2^e u$, where $1\leq u<2$, separates its scale from its shape. The
inverse square root then factors as

$$
\frac{1}{\sqrt{n_2}}=2^{-e/2}\frac{1}{\sqrt{u}}.
$$

Without this step, a table must follow the inverse-square-root curve across
the entire input range. After normalization, it needs to approximate only
the single curve $u^{-1/2}$ on the fixed interval $[1,2)$. The circuit restores
the power-of-two factor afterward. Even and odd values of $e$ give two scale
cases, so exponent parity joins the mantissa bits in the lookup address.

Four zero tests examine groups of 8, 4, 2, and 1 leading bits. Conditional
swaps move the leading 1 to bit 17 of the 18-bit mantissa. Their four results
encode both the shift and its parity, so the circuit needs no exponent
register. The circuit caps the shift at 15, which normalizes every
$n_2\geq4$. A later stage corrects the three smaller inputs.

Let $F(n_2)$ be the ideal scaled output. For a fixed exponent, define the
circuit's exact rescaling map by

$$
R_e(v)=\operatorname{round}\!\left(v\,2^{6-\lfloor e/2\rfloor}\right).
$$

The fitter starts from the required final band and pulls it backward through
this map. Each achievable input receives the set of admissible intermediate
integers

$$
A(n_2)=\{v\in\mathbb Z:|R_e(v)-F(n_2)|\leq32\}.
$$

Inputs that reach the same address and remainder must produce the same
quantized line value, so the fitter intersects their admissible sets. For a
candidate slope, the slope contribution at every remainder turns its
intersection into an interval of legal intercepts. The fitter searches
slopes outward from the curve's analytic derivative and accepts the first
slope whose intercept intervals have a common integer:

```text
fit(block):
    for each slope s, starting near the curve's local derivative:
        derive every legal intercept interval from A(n₂)
        intersect the intervals over the block
        if the intersection contains an integer:
            return s and a legal intercept g
    return failure

build(block):
    if fit(block) succeeds:
        return one line
    split block into equal halves
    return build(left half) and build(right half)
```

This backward fit evaluates truncation, line arithmetic, rescaling, and
rounding as one composed integer map. It therefore uses error wherever the
actual circuit permits it. It does not reserve independent margins that the
circuit may never need together. The fitted slopes cover the domain with 128
lines. A coarser line table would require another polynomial degree and another
full-width reversible multiplication.

## Full-depth line evaluation (qrom, multiply, subtract, un-subtract, un-multiply)

The lookup address contains the exponent-parity bit and mantissa bits 16
through 11, giving seven address bits and 128 slots. Mantissa bits 10 through
5 form the six-bit remainder. The address and remainder omit the lowest five
mantissa bits, but the backward fit already includes their effect.

The remainder is disjoint from the address. It therefore resets independently
inside every address slot. Suppose one tree leaf covered adjacent slots. The
same line would then repeat the same 64 local values in each slot. It
could not use an address bit to continue across their boundary. The
admissible-interval intersections reject every such merge, so the tree
deliberately reaches full depth with one leaf per slot.

Each leaf supplies a seven-bit slope with two fractional bits and an 18-bit
intercept whose leading 1 is implicit. Only 17 intercept bits therefore need
wires. For every set remainder bit $b_i$, the circuit adds

$$
\left\lfloor\frac{s2^i}{4}\right\rfloor
$$

to an 11-bit product register. Subtracting this product from the widened
intercept evaluates the descending line. Rather than use one 17-bit chain,
the circuit splits the subtraction into an eight-bit low chunk and a
nine-bit high chunk. It writes the borrow between them to its own wire and
keeps that wire live through the accumulator update. The exact reverse pass
erases that wire. Clearing it earlier would force the circuit to recompute
the borrow on the way back. The product is narrower than the stored
intercept, so the subtraction needs no sign case.

The circuit chooses this partition together with the unlookup strategy.
A seven-bit remainder that shared its top bit with the address would make the
lookup cheaper. The disjoint six-bit remainder instead removes one product row
on both passes. Full depth is the
deliberate price of making the arithmetic following each lookup smaller.

## Measurement-based unlookup (un-qrom)

After the circuit reverses the rescaling, subtraction, and product rows, the
slope and intercept again contain pure functions of the still-live address.
Reversing the lookup tree would clear them, but it would repeat all 126 ANDs
of the forward walk.

Instead, the circuit measures each looked-up bit in the X basis and resets
its wire to zero, which costs no Toffolis (measurement-based unlookup; Berry
et al. 2019, arXiv:1902.02134). If the measured bit stored the Boolean
function $f(a)$ of address $a$ and the measurement outcome is $y$, the
address state acquires the sign $(-1)^{y f(a)}$. Combining all measured word
bits leaves one known sign for each address, obtained from the parity of
their outcomes. The measurement clears the data wires; only this Boolean
phase function needs repair.

The repair splits the seven address bits into three high bits and four low
bits. A short walk distinguishes the eight high-bit prefixes. Beneath each
prefix, the sixteen low-bit signs form a Boolean truth table. A Boolean
change of basis, the Möbius transform into the Reed-Muller basis, rewrites
that table as an XOR of monomials in the four low bits. Each nonzero
coefficient then selects a phase gate, which is free in Toffoli accounting.

The individual low bits need no workspace. A shared bank supplies the higher
monomials using six pair products, four triple products, and one quadruple
product, for eleven temporary ANDs. Distinguishing the eight prefixes takes
six more. The complete repair therefore costs 17 Toffolis instead of the
126 ANDs required by a second full tree walk. Once the four-bit monomial bank
exists, adding leaf corrections changes free phase gates rather than growing
another lookup tree.

## Packed distance computation (diffs, squarer, un-squarer, un-diffs)

Each coordinate of $r_i$ is temporarily replaced by the magnitude
$|r_{i,c}-r_{j,c}|$. A complement subtraction records the sign on one
scratch qubit, and conditional negation produces the magnitude. Three sign
qubits retain enough information to restore both input points during the
reverse pass.

For one eight-bit magnitude, write its bits as $d_i$. Expanding the square
gives diagonal and cross terms:

$$
d^2=\sum_i d_i4^i+\sum_{i<j}d_id_j2^{i+j+1}.
$$

Every term has a known output column. The terms are partitioned into four
groups, normally with at most one term in each column, so each group is a
sparse integer that can be ripple-added directly. This avoids both a
compressor tree and a separate carry register.

There is one collision. The cross term $d_0d_7$ and the diagonal term
$d_4^2=d_4$ both occupy column 8. Their XOR remains in column 8, while their
conjunction carries into column 9. Exhaustive verification over all 256
magnitudes proves that the four groups sum to the exact square.

Three coordinates produce twelve groups. The first group seeds the empty
squared-distance register without an adder, and eleven ripple additions
accumulate the rest. Each ripple stops at the highest bit that the current
partial sum can reach because every higher bit is still known to be zero.

An exhaustive dynamic program considers the possible emission orders and
minimizes total adder ANDs under this carry-ceiling rule. It finds an order
using 135 adder ANDs. The straightforward axis-at-a-time order uses 141
under the same rule, so scheduling saves six Toffolis without changing the
groups or their arithmetic.

## Rescaling and exceptional inputs (patch, shift, accumulate, un-shift, un-patch)

The normalization cap sends $n_2=1,2,3$ to three recognizable mantissa
patterns instead of their ordinary table positions. Three normalized bits
already distinguish the cases. Two temporary conjunctions select XOR
corrections that replace the looked-up intercept with the required
pre-rescaling value, avoiding another normalization stage.

Three rows of conditional swaps apply the variable rescaling. The first bit
discarded by the shift becomes the carry-in to the final addition, which
implements binary round-to-nearest without a separate rounding register.

An ancilla-free ripple-carry adder (Cuccaro et al.
2004, arXiv:quant-ph/0410184) updates the accumulator. A temporary-AND
carry chain would reduce its nonlinear gate count. However, it would keep
roughly 23 carry ANDs live when no workspace headroom remains. The ancilla-free adder pays more Toffolis to avoid that
qubit increase. After the update, every arithmetic stage runs in reverse;
only the lookup words leave through the phase-corrected measurement described
in Measurement-based unlookup.
