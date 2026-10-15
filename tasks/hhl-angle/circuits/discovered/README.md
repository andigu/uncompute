# discovered — adaptive piecewise-quadratic interpolator

This folder holds the discovered circuit for the `hhl-angle` task. Its
certified cost is 1,348 Toffolis and 115 peak qubits, a TQ-cost of 155,020.
The accompanying paper (arXiv: TBD) describes it in the Supplementary
Information, section "Numerical function evaluation", subsection "HHL
rotation angle".

The adaptive piecewise-quadratic interpolator fits the rapidly changing end
of the domain with narrow power-of-two-aligned subranges. It fits the flatter
end with wide ones. Alignment turns interval membership into a prefix question,
so the fit can use 345 pieces without paying for general boundary
comparisons. Quantized quadratics then replace higher-degree arithmetic.

The circuit recodes overlapping pairs of multiplier bits as signed radix-4
digits. Each digit selects zero, one, or two shifted copies of the
multiplicand with either sign. The product needs seven rows instead of thirteen, while the
five-bit curvature word becomes three rows. Measurement-based uncomputation
removes the coefficient words, and one shared tree walk repairs the
resulting phases.

```text
(g₂, sh, W, g₀) ← tree_lookup(X)                         ▷ lookup
dn ← rotate_left(X[0…14], sh)                            ▷ rotate
k ← k + g₀; measure g₀ in the X basis                    ▷ c0add
v ← W − Q(dn, g₂)                                        ▷ vcompute
k ← k − P(dn, v)                                         ▷ product
remove the accumulated slope-bias contribution           ▷ comp
reverse the Q rows so that v ← W                         ▷ vuncompute
restore X; measure W, sh, and g₂ in the X basis           ▷ unrotate
correct phases using the recorded measurement outcomes    ▷ fixup
```

Here `tree_lookup` is the dyadic tree descent. $X$ and $k$ are the input and
accumulator. $g_0$, $g_2$, $W$, and $sh$ are the intercept, curvature,
biased slope, and rotation words. $dn$ is the shared coordinate, and $v$ is
the inner result. $Q$ and $P$ are the certified truncated curvature and
product sums.

The digit rows described in “Booth-recoded evaluation” account for 643
Toffolis. The two tree walks described in “Adaptive fitting and tree
lookup” and “Measurement cleanup and shared phase correction” account for
555. The circuit is deterministic. Exhaustive classical certification over
all 2,031,616 inputs proves exact partition coverage and every signed-prefix
window. It also proves a worst centered error of 427.15 output units against
the hard 429-unit band, which admits no failed shot. The evaluator
independently certifies the emitted op stream.

## Adaptive fitting and tree lookup (lookup)

The target bends most strongly near the low end of the domain and becomes
progressively flatter as the input grows. Equal-width pieces would therefore
waste coefficients in flat regions or leave curved regions too inaccurate.
An offline search begins with power-of-two-aligned subranges in each of the
five octave intervals. It fits a quantized quadratic using the same
truncated integer operations that the circuit executes.

The search retains a candidate subrange only when its quadratic stays within
428.5 output units at every integer input in that subrange. Otherwise the
search halves it and tries again. The fit-time bound sits half an output unit
below the hard 429-unit band. The reason is that the conversion to the graded
integer itself rounds the ideal real-valued angle. Exhaustive evaluation fixes the final
table only after checking every input in every retained subrange.

The resulting dyadic partition has 345 leaves with widths from $2^8$
through $2^{15}$. Of these, 27 leaves use the full width $2^{15}$. Each leaf stores a
26-bit intercept divisible by $2^6$, a 16-bit biased slope, a five-bit
curvature coefficient, and a three-bit rotation amount. A uniform partition
at the smallest retained width would require 7,936 subranges. Adapting the
widths therefore removes thousands of unnecessary table entries without
introducing arbitrary boundaries.

A subrange of width $2^b$ begins at a multiple of $2^b$. Membership
therefore depends only on the leading $21-b$ bits of `X`. The circuit
follows those prefixes through a binary tree, holding one temporary path
flag at each active split. CNOTs controlled by a leaf flag write the set
bits of its four stored words. Table output width therefore adds no Toffolis
to the leaf action.

A full binary tree with 345 leaves has 344 internal nodes. No control acts
on the root, and its first split can use an input bit directly. The other
343 internal splits each require one temporary AND (Gidney 2018,
arXiv:1709.06648). Repointing a parent flag between its children costs only
a CNOT. This descent is the unary iteration of Babbush et al. (2018),
arXiv:1805.03662. The resulting lookup costs 343
Toffolis, so adding a fitted piece costs almost exactly one Toffoli of
traversal. That cheap marginal cost lets the fit place narrow pieces where
the coarser digit arithmetic needs them.

## Coordinate alignment (rotate, unrotate)

Every leaf must feed the same arithmetic datapath even though leaf widths
vary. For a leaf $[L,L+2^b)$, write the input as $X=L+\delta$, where
$0\le\delta<2^b$. A natural common coordinate would move the $b$ varying
bits to the top of a 15-bit word:

$$
d_0=2^{15-b}\delta.
$$

Then $d_0/2^{15}$ represents the same relative position inside every leaf.
A plain in-place left shift cannot form $d_0$, however, because it discards
the fixed high bits of the low 15-bit input slice. Preserving those bits
separately would require more ancillas.

The table instead stores $sh=15-b$ and cyclically rotates the low 15 bits of
`X` left by $sh$. The varying bits move to the desired high positions, while
the fixed bits wrap into the low positions and survive. If
their wrapped value is $c_L$, the actual coordinate is

$$
dn=d_0+c_L.
$$

The value $c_L$ is constant throughout its leaf. Replacing $d_0$ by
$dn-c_L$ merely changes a quadratic's coefficients, so the offline fit
absorbs this translation without any runtime subtraction.

The three bits of $sh$ control rotations by 1, 2, and 4 positions. Each
amount is coprime to 15, so each permutation is one 15-cycle and requires
14 controlled swaps. The three forward rotations cost 42 Toffolis, and the
three inverse rotations cost another 42 while restoring `X` exactly.

## Booth-recoded evaluation (c0add, vcompute, product, comp, vuncompute)

The circuit evaluates the fitted quadratic in nested form. Let $g_1$ be the stored
slope and define its biased working word by

$$
W=(g_1\mathbin{\ll}4)+2^{16}.
$$

The circuit first adds the intercept $g_0$ to `k` and measures the intercept
word away. It then forms

$$
v=W-Q(dn,g_2)
$$

over the slope word. Finally, it subtracts $P(dn,v)$ directly in `k`.
Neither a full outer product nor a completed angle occupies a separate
register.

For the outer product, the multiplier is
$Y=dn\mathbin{\gg}2$, a 13-bit integer. Ordinary shift-and-add
multiplication would use one row for each multiplier bit. Instead,
overlapping groups of three bits define signed digits

$$
d_t=y_{2t-1}+y_{2t}-2y_{2t+1}
\in\{-2,-1,0,1,2\},
$$

with the boundary bits set to zero. Multiplying this expression by $4^t$ and
summing makes the overlapping terms telescope:

$$
\sum_t d_t4^t=Y.
$$

Each digit therefore accounts exactly for two binary positions while
choosing only zero, one, or two times a shifted multiplicand, possibly
negated. This signed two-bit grouping is radix-4 Booth recoding (Booth
1951, doi:10.1093/qjmam/4.2.236). It reduces
the 13-bit product from thirteen rows to seven without approximating the
multiplier.

Linear CNOT preprocessing of each overlapping bit triple exposes the digit
sign and the selector for $\lvert d_t\rvert=1$. The remaining selector, for
$\lvert d_t\rvert=2$, needs one non-column AND. Each output column then
chooses between the corresponding bit of $v$ and the bit one position
deeper, because doubling the multiplicand is a shift. The sign wire converts
the selected bits to one's-complement form, and the row adder supplies the
conditional carry that completes negation.

The configuration search in `gen_constants.py` chooses the truncation schedule

$$
B=[4,5,5,5,6,6,6]
$$

and corresponding right shifts

$$
sig=[13,12,10,8,7,5,3].
$$

Thus row $t$ contributes

$$
d_t\,(v\mathbin{\gg}sig_t)\,2^{B_t}.
$$

The deeper source bit for $\lvert d_t\rvert=2$ folds doubling into the shift
before truncation. Once the circuit adds a row, measurement-based
uncomputation removes its column ANDs. Two classically conditioned CZs repair each
phase kick while the selected operand pair is still available.

The curvature product uses the same signed representation. Its five-bit
coefficient becomes three digit rows over
$dq=dn\mathbin{\gg}3$. The configuration search keeps the emission order
highest digit first. The order with the highest digit first and the two
lower digits swapped ties it at equal cost.
Signed digits make this order relevant because successive partial sums need
not move monotonically toward the final value.

To see the required range proof, let $S_j$ be the algebraic curvature sum
after the first $j$ emitted rows. The working slope after that prefix has
the representative

$$
v_j=(W\bmod 2^{17})-S_j.
$$

With unsigned rows, monotonicity can sometimes reduce range checking to one
edge or to the final result. Signed rows can increase one prefix and decrease
the next. A safe final value therefore says nothing by itself about whether
an intermediate prefix crossed either boundary. The search accepts the table
only after exhaustive certification of the two-sided invariant, for every
input and after every curvature row:

$$
0\le (W\bmod 2^{17})-S_j<2^{17}.
$$

This invariant licenses the 17-bit conjugated carry chains. Every prefix has
a valid representative in precisely the range those chains preserve. A
$2^{16}$ modulus would be insufficient. Bit 12 of $g_1$ moves into bit 16
after the four-place shift, so that bit can wrap the working window.
The $2^{17}$ modulus retains it and makes intermediate modular wraps exact.

The prefix-by-prefix proof also certifies the reverse computation without a
second range argument. Uncomputation emits the curvature rows in reverse, so
it visits the same certified partial sums in the opposite order. The fixed
table separately certifies that the completed $v$ satisfies
$0\le v<2^{20}$, preventing a complement-subtract wrap.

The bias inside $W$ contributes through every outer-product digit. Because
Booth recoding preserves $\sum_t d_t4^t=Y$ exactly, the circuit combines
these digitwise bias terms rather than correcting them row by row. One carry-only
addition handles the 13 addend bits against a 26-bit target; its zero tail
only propagates the carry. This costs 25 Toffolis, and measurement-based
uncomputation removes every carry AND at zero Toffoli cost.

The intercept uses a hybrid ripple-carry addition costing 41 Toffolis. Its
lowest 11 positions keep temporary carry ANDs and erase them by measurement;
the higher positions use the ancilla-free majority/unmajority ripple of
Cuccaro et al. (2004), arXiv:quant-ph/0410184. Product
rows use the same dial. A longer measured-carry prefix saves Toffolis but
leaves more temporary qubits live. A shorter prefix saves qubits but
repeats nonlinear work in the majority steps.

A full-circuit sweep covered the fixed cap grid and every measured-carry
prefix length. It selected a live-qubit cap of 117 and a prefix of 11 by
minimizing the executed-Toffoli-times-peak-qubit product. Relative to the
uncapped schedule, this choice adds 64 Toffolis and removes 12 peak qubits.
Tightening the cap further does not lower the peak because other circuit
moments bind it, while the digit rows continue paying for additional
majority conversions.

## Measurement cleanup and shared phase correction (c0add, unrotate, fixup)

Suppose a looked-up coefficient wire contains a classical leaf-dependent bit
$b$. Measuring that wire in the X basis removes it, but measurement outcome
$m=1$ multiplies every branch with $b=1$ by $-1$. The unwanted sign is
therefore $(-1)^{bm}$ and depends on the leaf selected by `X`.

For a recorded outcome of 1, the circuit reconstructs the corresponding leaf
flag and applies a CZ between that flag and a wire held at $|1\rangle$. This
contributes exactly the required sign and costs no Toffolis. Outcomes of zero
need no correction. The circuit measures the intercept after its addition.
The slope, rotation, and curvature words remain until the circuit reverses
their arithmetic and restores `X`. Delaying the repair allows one final traversal to service
all four words together.

Reconstructing every full-depth leaf flag would cost 343 Toffolis. A
classical cost search over candidate stopping depths instead selected level
12 for the fixed tree. The retained upper traversal costs 182 Toffolis.
Below that level, a shared one-hot decode of input bits 7 through 11
distinguishes 211 of the 345 leaves.

Growing this five-bit decode one input bit at a time produces 32 one-hot
wires with $2^5-2=30$ Toffolis. The retained traversal and decoder therefore
cost 212 Toffolis together. This floor works because all narrower leaves
share the one-hot bank; the circuit does not rebuild it for each measured
word.
