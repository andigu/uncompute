# khattar-eea — baseline: the published Reed–Solomon decoder as one circuit

This folder holds the baseline circuit for the `dqi-reed-solomon` task. Its
certified cost is 5,125,070 Toffolis and 1,938 peak qubits, a TQ-cost of
9,932,385,660. The circuit is our transcription of the Reed–Solomon
syndrome decoder in the code release of Khattar et al. (2025),
arXiv:2510.10967, at their instance (m = 4095, n = 70, b = 12). It composes
the decoder into one executable circuit. A forward extended-Euclidean pass solves the key
equation. A 4,095-point Chien/Forney sweep pays the phase. A tape-driven
adjoint restores the syndromes and all workspace. The Euclidean pass runs
in the synchronized form of Proos and Zalka (2003),
arXiv:quant-ph/0301141. In place of
data-dependent division steps, 222 identical seven-stage cycles advance
the machine on a fixed schedule. The op sequence, and with it the
Toffoli count, is therefore the same on every input. Inside a cycle, every
comparison against a classical slot index is a unary-iteration walk over
the counter's bit-trie. The walk costs about one Toffoli per visited leaf,
the released code's own lookup idiom. Every width and count follows the
released code's rule for this instance, read from the task
contract. The rule gives 3(n+1)+9 = 222 cycles, m' = n+3 = 73 array
slots, and 7-bit degree counters (the bit length of m'). It also gives an
8-bit halting counter (the bit length of the cycle count) and 12-bit
field words. The multiplier is the release's own GF(2^12) multiplier:
three Maslov-optimized 6-bit polynomial products inside one Karatsuba
level, 51 Toffoli.

Three further choices shape the circuit. First, the machine initializes
from the actual syndrome degree, in place. A suffix-zero scan, a degree
counter, and seven controlled block-shifts realize the data-dependent
placement that the released code performs classically. A fixed placement
would mis-decode every leading-zero syndrome. Second, divisions keep their
junk. The Itoh-Tsujii inversion intermediates stay live between compute
and erase. Each per-point division therefore costs 306 Toffoli forward and
nothing to undo: the erase measures the intermediates away instead of
rerunning the ladder. The inversion chain is the released bloq's for
b = 12. It computes x, x^3, x^15, x^255, x^1023, x^2047 with five
multiplies, then a final free squaring. Third, the circuit never builds an
error register. The per-point Forney constant is classical given the
machine's final degree difference $n_b - n_B$. It therefore folds into the
fixed phase mask at build time ($c' = M^\top c$). CZ gates on
mask-selected quotient bits pay the whole phase, exact by XOR-linearity,
at zero Toffoli.
The circuit makes the sign exact for $n_b - n_B \in \{1, 2\}$ with one
flag qubit $d_2 = [n_b - n_B = 2]$.
The circuit computes the flag from the final counters for 10 Toffoli, and
the adjoint clears it for 10 more. At every root, the flag selects the
fold at the exponent one higher. The circuit pays that correction on the
XOR difference of the two folds under root AND $d_2$. This takes one
temporary AND per sweep point. The quotient register holds garbage at
non-root points, so the correction must stay gated on the root flag. The
difference bits vanish at the first point and wherever the mask slice is
zero. Only 4,093 of the 4,095 points therefore pay it.

The adjoint comes from a structural tape. The forward pass records
its plain ops, allocations, AND pairs, and multiplier calls. The
adjoint replays the tape in reverse with compute and release roles
swapped. The restore therefore costs exactly the forward Toffoli and
needs no workspace of its own.

The op stream is loop-compressed. The 222 cycles are one op-stream
loop, and so are the 4,095 sweep points and the 222 reversed cycles.
The tape records the cycle body once, and the adjoint emits its
reverse under its own loop. The per-point differences of the sweep
are build-time constant bits, so data-gated tables carry them. The
tables hold the CZ payments of the folded mask and of the correction
slice, and the root-AND-$d_2$ pair. The stream expands to the same 176,506,382 ops
as the unrolled form.

## Cost

The certified cost is 5,125,070 Toffolis and 1,938 peak qubits:

$$5{,}125{,}070\ \text{Toffoli} \times 1{,}938\ \text{qubits} = 9{,}932{,}385{,}660\ \text{(TQ-cost)}$$

| Phase | Role | Toffoli |
|---|---|---:|
| `init` | place the syndrome by its actual degree | 6,661 |
| `eea` | 222 identical cycles solve the key equation, then the Forney flag | 1,904,770 |
| `sweep` | evaluate all 4,095 points and pay the phase | 1,302,208 |
| `adjoint` | the reversed tape restores syndromes and workspace | 1,911,431 |

Init 6,661 = scan 2,032 + count 21 + shift 4,608 + constants 0; the
shift is 384 word-controlled swaps of 12-qubit registers. Each
Euclidean cycle costs 8,580 (the released call graph budgets 8,754);
the Forney flag costs 10. Each sweep point costs 317 = 11 (the
root-flag AND chain) + 255 (the keep-junk inversion, five multiplies)
+ 51 (the omega x inverse product). The 4,093 corrected points add
1 (root AND $d_2$). The 51-Toffoli multiplier replays a gate list
read from the release's own optimized polynomial-multiply circuits.
The list is data in constants.json; gen_constants.py extracts it from
the release. The
adjoint equals `init` plus `eea` exactly — the tape replays every recorded
Toffoli once.

Peak 1,938 lands in the Forney step of the first corrected sweep point.
That is the second point, since the first pays no correction. The peak
holds 1,806 register-file qubits plus 132 sweep transients. The 1,806
are the 1,818 persistent qubits minus the 12-qubit quotient. The quotient
is zero at every cycle boundary, and the circuit frees it before the
sweep. The 132 = 36 evaluation qubits (three words) + 11 root chain +
12 inverse + 60 kept inversion junk + 12 quotient + 1 correction gate.
The forward pass peaks lower, at 1,930. The register file stays
allocated through the sweep because the adjoint still needs it. Khattar
et al. report 1,885 logical qubits for the reversible decoder (their
Table 6). That count is a per-stage maximum with each stage alone (EEA
1,883, phase fixup 902). The Toffoli total is within 2.1% of the
decoder-stage estimate of 5,021,091. That estimate comes from the
released code, recomputed with the Qualtran estimator. Three items that
this circuit includes dominate the difference: the degree-dependent init,
the omega x inverse product, and the 4,113 Toffoli of the exact Forney
sign.

## Accuracy

The task allows at most 100 failed tests in 100,000, one shared
allowance for every failure kind. One failure class remains; everything
else is exact on every basis state. That class is a quotient residue. The
quotient scratch must return to zero at every cycle boundary. A
residue would leave workspace unrestored and fail the shot. The residue
occurred zero times in 56,000 seeded draws under the task's weight
law (rate below 5.4e-5 at 95% confidence). The state is reachable only
from syndromes that no error of symbol weight at most 35 produces.
The Forney exponent is exact. The common case has final counters
$n_b - n_B = 1$. The machine ends one symbol short ($n_b - n_B = 2$,
exponent one higher) exactly when the error values XOR to zero. Omega's
top coefficient $(\prod_k x_k)\sum_j e_j$ then vanishes. This case
covers 2.3e-4 of the weight-law draws (13 of the same 56,000), and the
flag $d_2$ pays it correctly. The circuit does not correct a second
vanishing coefficient ($n_b - n_B = 3$, probability about $2^{-24}$ per
draw). It is the only residual sign error, far below one expected shot
in 100,000.

Three deliberate choices keep the rest exact. The stop constant is
n/2 + 1 = 36. It decodes symbol weight exactly 35, the heaviest graded weight, in the same 222
cycles. Below that weight it is cycle-identical. The cycle's flag
update uses equality tests only, matching the released design. Stage 1
always starts with $n_A \le n_B$, so the cycle needs no less-than
comparator. The
unary walks cover counters up to m' = 73, and the flag covers
$n_b - n_B \in \{1, 2\}$ at the end of a genuine run. Division follows the released quantum semantics — the
inverse of zero is zero — so a vanishing $\sigma'$ is harmless. The build
refuses to emit if its cost drifts, and the evaluator checks the finished
circuit independently.

## Attribution

This circuit transliterates the reversible Reed–Solomon decoder in the code
that Khattar et al. (2025), arXiv:2510.10967, released on Zenodo
(doi:10.5281/zenodo.17301475). The authors of that release are Tanuj
Khattar, Noah Shutty, Craig Gidney, Adam Zalcman, Noureldin Yosri, Dmitri
Maslov, Ryan Babbush, and Stephen Jordan. The release is licensed under the
Creative Commons Attribution 4.0 International license (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). `circuit.py` follows the
released bloqs stage by stage, and its comments cite the released files by
name and line. `constants.json` holds the gate list of the released
51-Toffoli $\mathrm{GF}(2^{12})$ multiplier; `gen_constants.py` extracts it
from the release.

This folder adapts the released code into one executable circuit;
comments in `circuit.py` mark each adaptation.
