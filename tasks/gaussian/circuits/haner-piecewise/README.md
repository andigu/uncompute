# haner-piecewise — baseline: HRS piecewise-polynomial Gaussian

This folder holds the baseline circuit for the `gaussian` task. Its certified
cost is 23,784 Toffolis and 362 peak qubits, a TQ-cost of 8,609,808. The
circuit is our gate-level reconstruction of the piecewise-polynomial
function evaluation of Häner, Roetteler, and Svore (2018), arXiv:1805.12445
(HRS), applied to $e^{-x^2}$. We wrote it from the HRS paper's description.
The accompanying paper (arXiv: TBD) uses it as the HRS baseline in the
Supplementary Information, section "Numerical function evaluation".

The circuit uses the HRS building blocks throughout: an even-function sign
fold, the pre-truncated fixed-point square $w = x^2$, comparator-driven
interval labeling, coefficient selection under the label, pre-truncated
Horner evaluation, and a full uncompute mirror. The configuration is degree
3 on 32 minimax intervals of $w$, in signed fixed point with 40-bit words
and 7 integer bits.

The circuit evaluates each real-valued minimax fit in the local coordinate
$t = w - a_l$ of its interval. In that coordinate, the integer coefficients
meet the $10^{-7}$ error bound at these word sizes. The circuit includes
the reversible subtraction of the selected left endpoint, and the cost
table counts it.

The arithmetic uses temporary ANDs (Gidney 2018, arXiv:1709.06648), which
measurement erases at zero Toffoli cost. Each interval comparator is a bare
carry chain with its constant bound folded in classically. The circuit
uncomputes nothing early. The square word and all four polynomial words
stay live to the end. The closing mirror replays every stage but the final
addition in reverse.

## Cost

$$23{,}784\ \text{Toffoli} \times 362\ \text{qubits} = 8{,}609{,}808$$

| Phase | Role | Toffoli |
|---|---|---:|
| fold | sign fold to the exact magnitude | 35 |
| square | 36 pre-truncated square rows | 2,194 |
| label | 31 comparators step the interval counter | 1,186 |
| offset | load, subtract, and clear the selected endpoint | 287 |
| lead | load the leading coefficient | 124 |
| h2 / h1 / h0 | Horner steps: a load plus 40 multiplier rows | 2,681–2,685 |
| round | add the rounded polynomial word into the accumulator | 38 |
| un-* | the mirror of every stage but round | 11,873 |

The circuit reaches its peak of 362 qubits at the widest h0 multiplier row.
The peak is 283 persistent wires plus 79 transient AND wires. The
persistent wires are input 39, accumulator 39, label 5, and five 40-bit
words: the square/local word plus four polynomial words. The transient
wires are 39 carries, 39 addend ANDs, and the top addend AND. The endpoint
word lives only inside the offset stage, so its 40 wires never widen the
Horner peak.

The HRS closed-form accounting prices this configuration at 43,212
Toffolis and 245 qubits. The built circuit uses fewer Toffolis than that
formula. It uses temporary-AND adders, truncated addend windows, AND-ladder
coefficient selection, and constant-folded comparators. It also uses the
HRS Toffoli-free label increment: the counter's change bits are CX-loaded
under each comparator flag. The circuit uses more qubits than the formula.
The built circuit keeps five live words in the unpebbled ladder, and its
qubit count includes the multiplier transients.

gen_constants.py selects the configuration. It sweeps the HRS printed
degrees, word sizes, and pebbling schedules, and it prices each candidate
with the HRS closed-form accounting. The chosen configuration has the
lowest Toffoli-times-qubit product under that accounting.

`occupancy.json`, `occupancy.pdf`, and `occupancy.png` chart the live
qubits along the executed-Toffoli axis. The chart stacks each wire set that
the circuit claims, such as the label and the polynomial words, and a pool
of unclaimed wires that holds the transient ANDs.
`tools/occupancy_profile.py` writes these files from `circuit.py`.

## Accuracy

All error is deterministic: the minimax fit, its coefficient quantization,
and the fixed truncations of the square and the Horner rows. The task has
$2^{36}$ in-domain inputs, so an exhaustive check is not practical.
Instead, gen_constants.py bounds each interval's error from above. It
accepts an interval only if the bound stays below 429.4967 units (one unit
is $2^{-32}$). The bound evaluates the rounded polynomial, exactly as the
circuit loads it, against $e^{-w}$ on a 4,000-point grid. It adds the
interpolation remainder between grid points, derived from a bound on the
error's second derivative. It also adds the truncation residues of the
square and the Horner rows, and the two rounding steps. The bound uses no
safety factor. The addend and the ideal value are both integers, so the
bound forces the integer residue under $\lfloor 429.4967 \rfloor = 429$.

The fit uses `mpmath` at a fixed precision and exact rational arithmetic,
so every machine writes the same tables. circuit.py reads the tables from
constants.json and checks their shape before it builds. The evaluator
certifies the finished circuit independently.

## Attribution

The construction is the piecewise-polynomial function evaluation of Häner,
Roetteler, and Svore (2018), arXiv:1805.12445. We built the circuit from the
description in that paper. It contains no code from the authors. We chose
the configuration: degree-3 minimax fits on 32 intervals of $w = x^2$,
40-bit words with 7 integer bits. We quantize each fit in the local
coordinate $t = w - a_l$. The adders use temporary ANDs (Gidney 2018, arXiv:1709.06648).
