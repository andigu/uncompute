# haner-piecewise — baseline: HRS piecewise-polynomial inverse square root

This folder holds the baseline circuit for the `coulomb-potential` task. Its
certified cost is 6,355 Toffolis and 252 peak qubits, a TQ-cost of
1,601,460. The circuit is our gate-level reconstruction of the
inverse-square-root construction of Häner, Roetteler, and Svore (2018),
arXiv:1805.12445 (HRS). We wrote it from the HRS paper's description. The
accompanying paper (arXiv:2610.22367) uses it as the HRS baseline in the
Supplementary Information, section "Numerical function evaluation".

The construction locates the leading bit and normalizes to a mantissa on a
unit interval. It evaluates a piecewise polynomial there and shifts the
result back. A first-one scan writes the squared distance as $n_2 = 2^e u$
with $1 \le u < 2$, and the polynomial approximates $u^{-1/2}$ on
$[1, 2)$. A reversible barrel shift restores the remaining factor
$2^{-\lfloor e/2\rfloor}$. The coefficient banks for odd exponents include
the factor $\sqrt{2}$. Normalization replaces a lookup or iteration over
the whole 18-bit input range with one smooth function on one interval. The
configuration is one point of the HRS parametric design. It uses degree 2
on 8 minimax (Remez) intervals per exponent parity. The 16 coefficient
banks are in signed fixed point with 26 bits, 2 of them integer bits. This
sizing keeps the whole quantized datapath inside the allowed output error.

An exact schoolbook front end computes the squared distance. Per axis, it
computes a 9-bit signed difference, a sign-conditioned negation, and eight
partial-product rows accumulated with temporary ANDs. Interval selection
follows HRS. Seven comparators against fixed thresholds drive the label
counter through the HRS Toffoli-free increment. The fired comparators
always form a prefix, so CX-loading $i \oplus (i+1)$ under each flag
telescopes the counter. Each coefficient bank loads under one AND-ladder
match flag. The Horner ladder runs with no intermediate uncomputation, and
all three polynomial words are live at once. Each step's multiplier is the
HRS pre-truncated row form. Row $b$ adds a shifted mantissa window under
one control bit, and the top row subtracts for the two's-complement sign. A
Bennett mirror of every stage but the final addition returns all scratch
to zero.

## Cost

$$6{,}355\ \text{Toffoli} \times 252\ \text{qubits} = 1{,}601{,}460$$

| Phase | Role | Toffoli |
|---|---|---:|
| dist | squared signed differences accumulate the 18-bit distance word | 582 |
| scan | first-one scan writes exponent flags, parity, and the shift word | 17 |
| normalize | conditioned swaps move the distance word into the mantissa | 153 |
| label | 7 threshold comparators drive the interval counter | 151 |
| lead | bank-match ladders load the leading coefficient | 48 |
| h1 | first Horner step: bank loads plus 26 multiplier rows | 1,047 |
| h0 | second Horner step: bank loads plus 26 multiplier rows | 1,047 |
| rescale | barrel-rotate the padded word right by the half-exponent | 129 |
| accumulate | add the rescaled window into the accumulator | 24 |
| un-* | Bennett mirror of every stage but the accumulate | 3,157 |

The peak of 252 qubits occurs at the widest multiplier row of h0. It holds
202 persistent wires plus 50 transient AND wires: 25 carry and 25 addend.
The persistent wires are 73 input and output wires, 18 one-hot exponent
flags, and 5 parity and shift bits. The rest are a 25-bit mantissa, a 3-bit
label, and three 26-bit polynomial words. Four devices reduce the Toffoli
count: temporary-AND ripple adders, classically folded comparator
constants, AND-ladder bank matches, and the HRS Toffoli-free label
increment. The width includes the multiplier transients (50 live wires
here) and the 18 one-hot exponent flags. Freeing the emptied distance word
partly offsets that width.

`occupancy.json`, `occupancy.pdf`, and `occupancy.png` chart the live
qubits along the executed-Toffoli axis. The chart stacks each wire set that
the circuit claims, such as the label and the polynomial words, and a pool
of unclaimed wires that holds the transient ANDs.
`tools/occupancy_profile.py` writes these files from `circuit.py`.

## Tables

gen_constants.py searches the partition and writes the banks and thresholds
to constants.json, which the circuit loads. The search tiles the mantissa
words $[2^{24}, 2^{25})$ from left to right. Bisection extends each interval
to the largest right threshold that passes one test. The `mpmath` library
fits the degree-2 minimax polynomial of $u^{-1/2}$ at a fixed precision.
The coefficients round to the $2^{-24}$ grid, one bank per exponent parity.
The exact integer datapath then runs on every achievable $n_2$ whose
mantissa falls in the interval. The interval passes when every output stays
within the allowed 32 units. The fit uses software arithmetic and the
datapath uses integers, so every machine writes the same tables.

## Accuracy

The only error source is the deterministic fixed-point error of the
quantized datapath. It combines the minimax residual, coefficient
quantization, row truncation, and rescale truncation. The search admits an
interval only when its error meets the band, so the tables meet it by
construction. checks.py evaluates the tables independently over all
139,637 achievable squared distances. The worst error is 32 output units of
$2^{-24}$ at $n_2 = 3$, equal to the allowed 32. The circuit therefore never
produces an out-of-tolerance output. checks.py also matches its ideal table
against the task's fixed digest. The build pins the total cost and stops if
the cost changes. The evaluator certifies the finished circuit
independently.

## Attribution

The construction is the piecewise-polynomial function evaluation of Häner,
Roetteler, and Svore (2018), arXiv:1805.12445. We built the circuit from the
description in that paper. It contains no code from the authors. We chose
the configuration: degree-2 minimax fits on 8 intervals per exponent parity,
26-bit words with 2 integer bits. We chose these gate-level devices:
temporary-AND adders (Gidney 2018, arXiv:1709.06648) and classically folded
comparator constants.
