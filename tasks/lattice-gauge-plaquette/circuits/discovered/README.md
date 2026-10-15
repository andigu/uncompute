# discovered — inversion-walk chain

This folder holds the discovered circuit for the `lattice-gauge-plaquette`
task. Its certified cost is 172,128 Toffolis and 1,859 peak qubits, a TQ-cost
of 319,985,952. The accompanying paper (arXiv: TBD) describes it in its
section "Nonabelian lattice-gauge simulation"; this README maps the
construction to the code.

The inversion-walk chain keeps one running product, an ordered group word.
The circuit edits that word from one loop representative into the next.
Cyclic rotation and inversion preserve the real trace. The schedule may
therefore select whichever rotated or reversed representative shares the
most useful ordered segment with its neighbors. The circuit restarts from
the identity when editing the current word would cost more.

The circuit combines that schedule with coordinates specialized for repeated
right multiplication and with direct classification of the seven trace
values. Only the running product takes a one-hot expansion; the 192 links
remain compact. A smaller update replaces general group multiplication, and
six trace flags replace a full-width lookup.

```text
layer:
    declare g₀…g₁₉₁ and k                                  ▷ init
    for each link ℓ: gℓ ← recode(gℓ)                        ▷ encode
    W ← identity in mirror coordinates                     ▷ loops
    walk_row(opening program)                              ▷ loops
    repeat 3:                                              ▷ loops
        shift links by one row                             ▷ loops
        walk_row(row program)                              ▷ loops
    shift links by one row                                 ▷ loops
    repeat 3:                                              ▷ loops
        shift links by one plane                           ▷ loops
        walk_row(plane program)                            ▷ loops
        repeat 3:                                          ▷ loops
            shift links by one row                         ▷ loops
            walk_row(row program)                          ▷ loops
        shift links by one row                             ▷ loops
    shift links by one plane                               ▷ loops
    close W at the identity                                ▷ loops
    for each link ℓ in reverse: gℓ ← recode⁻¹(gℓ)          ▷ decode
    return                                                 ▷ decode

walk_row(program):
    for each fired loop L in program: fire_loop(L)          ▷ loops
    return                                                  ▷ loops

fire_loop(L):
    W ← morph W to the next representative                  ▷ loops
    flags ← trace class of W                                ▷ loops
    k ← k + a(L)f(W), controlled by flags                   ▷ loops
    erase flags by measurement                              ▷ loops
    return                                                  ▷ loops
```

Here $g_\ell$ is link $\ell$; $k$ is the accumulator; $W$ is the running
product; $L$ is a loop. `recode` changes group coordinates, and `morph` edits
the ordered word. A `program` is a translated sequence of fires; `flags`
encode the trace class; and $a(L)f(W)$ is the weighted constant for $L$.

The sections below explain the remaining nonlinear work. The circuit is
deterministic. The evaluator checks the complete circuit on 100,000 random
inputs. The map is exact, so the grader permits no failed test.

## Mirror coordinates and specialized updates (encode, decode, loops)

A link enters in the compact order

$$
g=\omega^p C^q E^r V^{2s+t}X^u.
$$

Using the same order for a frequently updated running product is expensive.
In a right multiplication $Wg$, the incoming $C^qE^r$ block must cross the
accumulated $V$ and $X$ factors, producing coupled coordinate changes and
phase corrections. A general normal-form multiplier must handle those
changes for two arbitrary operands.

The circuit first replaces $X$ by an equivalent coset representative $X_o$.
The choice of $X_o$ makes the $V$ and $X_o$ factors close among themselves.
Their update then uses modular arithmetic and fixed transformations instead
of spilling corrections into the other coordinates. Links retain the factor
order

$$
g=\omega^p C^q E^r V^{2s+t}X_o^u,
$$

while the running product uses the opposite placement of the $C,E$ block:

$$
W=\omega^P V^M X_o^\upsilon C^Q E^R,
$$

with $P,Q,R\in\mathbb Z_3$, $M\in\mathbb Z_4$, and $\upsilon\in\mathbb Z_2$.
The accumulated $C^QE^R$ factors now meet the incoming $C^qE^r$ factors at
the multiplication boundary. The circuit can combine them before fixed
transformations account for $V$, $X_o$, and the binary coset bit.

Reordering still creates recurring corrections to the phase trit $P$. Rather
than reconstructing that correction during every append, the running state
stores the sheared coordinate

$$
\widetilde P=P+QR\pmod 3.
$$

The shear folds the conjugation correction into the coordinate itself.
Updating $\widetilde P$ then requires the remaining bilinear trit terms only.
Those terms use a three-product decomposition, the Karatsuba method, with
shared controlled rotations.

Recoding a link into this compatible form costs 23 Toffolis, and the inverse
recode has the same cost. These are one-time transformations at opposite
ends of the circuit, paid once per link and amortized over all 576 loop
evaluations. Changing coordinates is worthwhile because every intervening
append uses the specialized form. Such an append costs 35 Toffolis, compared
with 146 for a general normal-form multiplication.

## One-hot running state with compact links (loops)

The running product contains three trits,
$\widetilde P$, $Q$, and $R$. Each trit occupies three wires with exactly
one wire set. Incrementing a trit modulo three becomes a cyclic permutation of
those wires, and negation becomes a swap. Fixed transformations of the
trits can therefore act by controlled wire permutations instead of repeatedly
decoding and re-encoding compact two-qubit values.

Together with compact coordinates for $M$ and $\upsilon$, this gives a 12-wire
running product rather than a nine-wire compact group register. That
expansion is economical because only one running product exists. Applying
the same one-hot expansion to all 192 links would add three wires per link,
or 576 extra qubits. The links therefore remain in their compact nine-qubit
encoding after recoding, while only the shared running state pays for the
permutation-friendly representation.

The same one-hot rails later expose useful parities for trace classification.
Thus the representation pays once for cheap modular updates and supplies the
Boolean coordinates used when the circuit fires a loop.

## Inversion-walk scheduling and translated rows (loops)

The running product is an ordered word in a nonabelian group, so the circuit
cannot rearrange its factors freely. Only its right end is directly editable. If

$$
W=abcde,
$$

then right-multiplication by $e^{-1}$ and $d^{-1}$ removes the old suffix.
The circuit can then append new factors. It can therefore retain a common
prefix, but right multiplication alone cannot reach a useful segment at the
left end.

Inverting the entire product reverses the word:

$$
W^{-1}=e^{-1}d^{-1}c^{-1}b^{-1}a^{-1}.
$$

The old left end is now at the editable right end. This move is legal because
the real trace is invariant under inversion. Cyclicity of trace also permits
any cyclic rotation of a loop word. The circuit may consequently fire each
loop in any rotated or reversed form. This enlarges the set of ordered
segments that loops can share, without changing any loop's value.

The evaluation order is not hand-written. An exact dynamic program searches
over loop orders, cyclic and inverted representatives, and restart points.
Its moves append or cancel a link, invert the running word, or begin a new
anchored segment. The dynamic program is `gen_constants.py` in this folder.
Rerunning it reproduces `constants.json` exactly. Every one of its 577
inter-fire transitions matches the optimum in the corresponding pairwise
table, and their total matches the chain optimum. This proves optimality
within that move set, not among all possible circuit constructions.

The selected walk uses 3,136 link multiplications and 480 running-product
inversions. It has 32 anchored chain segments, each opened and closed by a
one-Toffoli anchor, for 64 anchors in all. The first opening begins the walk,
so 31 openings are genuine restarts. To open a segment, the circuit writes
the running state directly from one inverted link. In mirror coordinates
this load is affine in the stored link bits except for a single AND. It
therefore costs one Toffoli, and reversing the operation clears the
endpoint at the same cost. A restart can therefore cost less than editing
through several link multiplications.

Measured in Toffolis, the multiplication layer accounts for

$$
3{,}136(35)+480(21)+64(1)=119{,}904.
$$

The schedule lets $W$ end a transition inverted, because the trace readout
does not depend on the orientation of $W$.

Translation symmetry compresses the walk into an opening-row program, a
program translated through $y$, a program translated through $z$, and a
closing tail. Their lengths are 260, 266, 266, and 6 tokens. The flat walk
consists of the opening program, 12 translated copies of the $y$ program,
three translated copies of the $z$ program, and then the tail. Its token
count is therefore

$$
260+12(266)+3(266)+6=4{,}256.
$$

The walk uses 20 register shifts to carry these 16 row programs across the
periodic lattice. Each shift uses 1,296 logical Swaps, giving 25,920 Swaps
in all. A shift allocates no qubits and costs zero Toffolis. Swaps are
Clifford gates: the evaluator counts them, but they do not enter the
TQ-cost. The combined shifts return the link registers to their original
permutation before decoding.

## Direct trace classification and accumulation (loops)

The trace has seven possible values, but only six are nonzero:

$$
6,\ -3,\ 2,\ -2,\ 1,\ -1.
$$

The circuit extracts six flags that form an XOR basis for these values. The
values $-3$, $-2$, and $1$ each set one flag. The values $6$, $2$, and $-1$
set that same flag and one more. The zero class sets no flag because that
case requires no addition. Each bit of the addend is a fixed XOR parity of
the flags, so the circuit never materializes a 51-bit value selected from a
216-entry table.

The classifier derives its flags from parities already visible in the
one-hot coordinates. It forms a $3\times3$ grid for $(Q,R)$ and one-hot rows
for $(M,\upsilon)$. A rank decomposition over $\mathbb F_2$ expresses each class
indicator through row, rail, and cell parities together with the necessary
products. This turns trace evaluation into a small Boolean classification
problem rather than an address-based lookup.

A 51-bit ripple needs 50 carry bits. At the moment the addition runs, eight
general scratch wires and 12 wires from the classification scaffolding hold
zero. The adder borrows those 20 wires instead of allocating them. A borrowed
wire that was not actually zero would corrupt the sum. Only the remaining 30
carries need wires of their own.

The circuit computes the classifier first, for 20 Toffolis. After the
addition, measurement-based uncomputation erases the flags at a cost of six
Toffolis. The plaquette ripple between those steps uses 50 Toffolis. The
rectangle weight is even, so its lowest carry step vanishes and its ripple
uses 49. The two sums are therefore

$$
T_P=20+50+6=76,\qquad T_R=20+49+6=75.
$$

A lookup-based route would instead spend 478 Toffolis fetching the weighted
addend, adding it, and removing the lookup.
