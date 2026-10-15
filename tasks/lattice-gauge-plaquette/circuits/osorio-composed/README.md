# osorio-composed — baseline: the published circuits, loop by loop

This folder holds the baseline circuit for the `lattice-gauge-plaquette`
task. Its certified cost is 1,271,424 Toffolis and 1,927 peak qubits, a
TQ-cost of 2,450,034,048. The circuit is our reconstruction of the
loop-by-loop layer application that Osorio Perez et al. (2025),
arXiv:2511.17437, price. The accompanying paper (arXiv: TBD) uses it as the
baseline in its section "Nonabelian lattice-gauge simulation".

The published per-link
gate tables are linear in the loop count, with no discount for links shared
between loops. Each of the 576 Wilson loops recomputes its holonomy from
scratch with the published multiplication and inversion circuits. It then
accumulates its addend and uncomputes everything by the mirror schedule.
Loops share nothing: no inverse reuse, no partial products, no batching.

A loop runs in three substages. The circuit assembles the holonomy out of
place, because the published gates act in place and the link registers must
return unchanged. Each factor is CNOT-copied into a fresh register. On
return links the circuit inverts it there (Osorio Perez et al. (2025),
Sec. V) over the $\Sigma(36\times3)$ blocks of Gustafson et al. (2024),
arXiv:2405.05973. The circuit then multiplies the
running product into that register from the left (Sec. VI), so the register
holds the product so far times the new factor. The addend comes from a unary-iteration
lookup (Babbush et al. (2018), arXiv:1805.03662) pruned to the 216 valid
codes. The build folds the weighted traces
$a_P f$ and $a_R f \bmod 2^{51}$ into the two 216-entry tables classically.
A ripple adder (Gidney (2018), arXiv:1709.06648) adds the fetched word into
the accumulator, and the
self-inverse lookup un-fetches it. The uncompute substage replays the exact
inverse sequence, and measurement erases every temporary AND. Each block
reproduces the group multiplication tables.

## Cost

$$1{,}271{,}424\ \text{Toffoli} \times 1{,}927\ \text{qubits} = 2{,}450{,}034{,}048$$

| Phase | Role | Toffoli |
|---|---|---:|
| `plaq` | the 192 plaquette loops: assemble, accumulate, mirror (1,702 each) | 326,784 |
| `rect` | the 384 rectangle loops, six factors (2,460 each) | 944,640 |

Every count closes from four block costs. A multiplication is 146
Toffoli, an inversion 87, each addend lookup walk 214, and the 51-bit
ripple adder 50. A plaquette assembles three products and two
inversions (612). It fetches, adds, and un-fetches (214 + 50 + 214 =
478), then mirrors the assembly back down (612): 1,702 in all. A
rectangle assembles five products and three inversions: 991 up, 478
add, 991 down, 2,460 in all. The walk's 214 depends only on the
216-key address set, never on the stored values. Both lookup tables
therefore cost the same. The 24 codes with zero trace keep empty table
entries, and the walk still branches to them.

Peak 1,927 = 1,779 I/O + 45 product wires + 2 loop scratch + 51
addend + 50 live adder ANDs. Every rectangle add substage reaches it;
plaquette loops top out at 1,909. The widest single loop sets the
peak. The circuit frees every working register when its loop closes,
and the next loop reuses the same wires. The layer therefore never
grows past one rectangle's working set.

## Accuracy

Exact: every holonomy, table fetch, and modular addition is exact,
and the mirror restores every ancilla. No measurement outcome
conditions any Toffoli, so the executed count equals the emitted
count on every shot. The build asserts the Toffoli total and the peak
and refuses to emit if either differs. The evaluator checks the
finished circuit.

## Attribution

The circuit reconstructs published constructions from their papers. It
uses no released code.

- Osorio Perez et al. (2025), arXiv:2511.17437. The group encoding, the
  inversion circuit (Sec. V), the multiplication circuit (Sec. VI), and the
  loop-by-loop application of the layer.
- Gustafson et al. (2024), arXiv:2405.05973. The $\Sigma(36\times3)$
  blocks inside those circuits.
- Babbush et al. (2018), arXiv:1805.03662. The unary-iteration lookup of
  the addend.
- Gidney (2018), arXiv:1709.06648. Temporary logical ANDs and the ripple
  adder.

Our choices: the out-of-place assembly of each loop product (the published
gates act in place) and the pruning of the lookup to the 216 valid codes.
