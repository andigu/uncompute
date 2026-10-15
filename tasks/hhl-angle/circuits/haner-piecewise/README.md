# haner-piecewise — baseline: HRS piecewise-polynomial evaluation

This folder holds the baseline circuit for the `hhl-angle` task. Its
certified cost is 16,468 Toffolis and 300 peak qubits, a TQ-cost of
4,940,400. The circuit is our gate-level reconstruction of the
piecewise-polynomial method of Häner, Roetteler, and Svore (2018),
arXiv:1805.12445 (HRS). We wrote it from the HRS paper's description.
The accompanying paper (arXiv: TBD) uses it as the HRS baseline in the Supplementary Information, section
"Numerical function evaluation".

The circuit uses 35 intervals, each with a degree-3 minimax fit. It
evaluates the fit in signed fixed point on 44-bit words with 6 integer
bits. Comparators against the interval boundaries write a 6-bit label, and
the label selects the interval's coefficients. A Horner ladder of
controlled multiply-adds evaluates the polynomial. One closing addition
rounds the value into the output accumulator. The interval selection,
coefficient loading, and Horner evaluation follow the HRS constructions.
The circuit does not use table-lookup constructions from outside HRS, such
as the unary iteration of Babbush et al. (2018), arXiv:1805.03662. Its
adders use the temporary logical AND of Gidney (2018), arXiv:1709.06648.

The label stage costs little. Each comparator is one carry chain of
temporary ANDs that computes the carry-out of
$X + (2^{21} - \text{bound})$. The constant folds in classically, and
measurement erases the chain at zero Toffoli cost. The counter behind the
comparators costs no Toffolis. The boundaries ascend, so the fired flags
always form a prefix. CX-loading the bits of $i \oplus (i+1)$ under flag
$i$ then telescopes the counter to the interval index. This replaces a
five-Toffoli controlled increment per boundary. Coefficient selection pays
for its address match once per interval, not once per bit. An AND ladder
over the six label bits computes one match flag (five Toffolis), and that
flag CX-loads the whole 44-bit word.

The multiplier is the HRS pre-truncated form. An exact product would need a
register wider than the word. Each Horner row instead adds a shifted copy of
the input directly into the next 44-bit word, truncated to the same width.
No carry chain grows past the word. Rounding costs no gate of its own. The
round-to-nearest offset, half of the dropped low window, rides the degree-0
coefficient load, so the closing addition is a plain ripple. Cleanup also
follows HRS. After that addition, the whole computation runs backward (a
Bennett mirror). The mirror returns the label and all four coefficient
words to zero at the full recompute price, half of the Toffoli count.

## Cost

$$16{,}468\ \text{Toffoli} \times 300\ \text{qubits} = 4{,}940{,}400$$

| Phase | Role | Toffoli |
|---|---|---:|
| label | 34 comparators drive the interval counter | 642 |
| lead | select and load the leading coefficient | 175 |
| h2 | Horner step: coefficient load + 44 multiplier rows | 2,467 |
| h1 | the same on the next coefficient | 2,467 |
| h0 | the same on the last; its word carries the offset | 2,467 |
| round | ripple-add the rounded window into the accumulator | 32 |
| un-h0 .. un-label | the mirror of every stage but round | 8,218 |

Each Horner step is 175 Toffolis of coefficient selection (35 intervals at
5 each) plus 2,292 across its 44 controlled rows. The comparators average
18.9 ANDs because each chain starts at its constant's lowest set bit. The
circuit reaches its peak of 300 qubits at the widest multiplier row of the
last Horner step. That peak is 236 resident wires plus that row's 64
transients. The resident wires are 21 input + 33 accumulator + 6 label +
four 44-bit words. The transients are 43 carry ANDs + 21 addend ANDs. The
ladder is unpebbled: every coefficient word stays live from its load to its
mirror. The four words alone therefore account for 176 of the 300 qubits.

The HRS closed-form accounting at this configuration gives 41,658 Toffolis
and 238 qubits. The measured circuit uses fewer Toffolis because of
temporary-AND chains, comparators at the input's native 21 bits, and the
Toffoli-free counter increment. It uses more qubits, because the circuit
counts the widest row's 64 live ANDs.

gen_constants.py fits the tables and writes them to constants.json, which
circuit.py loads. The search grows the partition left to right over the raw
input grid. For each interval, it finds the widest stop whose fit passes an
exact test. The test fits a degree-3 minimax (Remez) polynomial and rounds
its coefficients to the nearest multiple of $2^{-38}$. It then runs the
circuit's fixed-point datapath on every integer input of the interval. The
test passes only if every output lies within the task band of 429 units of
$2^{-32}$ radians. The search therefore prices coefficient rounding and
product truncation directly, and it needs no safety margin. The fit runs in
`mpmath` at a fixed precision and the datapath on exact integers, never on
hardware floats. Every machine therefore derives the same tables and the
same circuit. The search takes about one minute. Most of that time computes
the exact ideal outputs.

`occupancy.json`, `occupancy.pdf`, and `occupancy.png` chart the live
qubits along the executed-Toffoli axis. The chart stacks each wire set that
the circuit claims, such as the label and the polynomial words, and a pool
of unclaimed wires that holds the transient ANDs.
`tools/occupancy_profile.py` writes these files from `circuit.py`.

## Accuracy

The only error source is deterministic: the minimax fits and the truncated
fixed-point datapath fixed in the tables. checks.py evaluates all 2,031,616
in-domain inputs. The worst deviation is 429 units of $2^{-32}$ radians at
X = 66,097, exactly the allowed deviation, so no headroom remains. The
search fills each interval up to the band by design. Coefficient rounding
uses much of the error budget. Near x = 32, rounding the cubic coefficient
alone can cost up to 256 units.

checks.py computes the ideal with the grader's rounding rule, so it needs
no rounding slack. The search's exact `mpmath` ideal matches that
floating-point ideal on every in-domain input. checks.py reads the
coefficients and interval boundaries from constants.json, so it checks the
same tables that the circuit loads. The build pins the total cost and stops
if the cost changes. The evaluator certifies the finished circuit
independently.

## Attribution

The construction is the piecewise-polynomial function evaluation of Häner,
Roetteler, and Svore (2018), arXiv:1805.12445. We built the circuit from the
description in that paper. It contains no code from the authors. We chose
the configuration: degree-3 minimax fits on 35 intervals, 44-bit words with
6 integer bits. We chose these gate-level devices:
temporary-AND adders (Gidney 2018, arXiv:1709.06648) and comparators at the
input's native 21 bits.
