# luo-eea — baseline: space-efficient Euclidean point addition of Luo et al. (2026)

This folder holds the baseline circuit for the `modinv-point-add` task. Its
certified cost is 69,529,757 Toffolis and 838 peak qubits, a TQ-cost of
58,265,936,366. The circuit is our reconstruction of the space-efficient
extended-Euclidean (EEA) affine point adder of Luo et al. (2026),
arXiv:2607.13816 (v3). The accompanying paper (arXiv:2610.22367) uses it in the
supplementary section "Specialized Euclidean construction" and in its
resource comparison for point addition. "The paper" in this README and in
`circuit.py` means Luo et al.

The reconstruction covers Algorithms 1-3, the Section-4 block constructions,
the Fig. 12 division and multiplication flows, the Fig. 13 schedule, and the
Appendix A Horner multiplier. An out-of-place slope computation would hold x,
its inverse, and their product live at once. Here the division y := y/x runs
in place. A forward EEA pass turns x into its inverse, and one multiplication
forms the quotient. Measurement then erases the old y, and the mirrored
inverse pass restores x. Phase gates conditioned on those measurement
results, wrapped around one recomputation, repay the phase debt that the
erasure leaves. The closing multiplication y := x*y is the same sandwich
inside out. The working set never grows past the two point registers, one
accumulator, and 70 bookkeeping qubits. The circuit drops the control wire of
Fig. 13, because the task map has no control. The per-shot classical addend
enters through bit-conditioned coordinate arithmetic. These bit conditions
replace the generation-time constants of the paper. Free classical bit
networks build derived values such as $3Q_x$.

## Implementation choices

The reconstruction makes six implementation choices. Each choice holds on
every input.

- **Step budget.** The circuit derives the step budget exactly, by dynamic
  programming over quotient-sequence continuants. All-2 quotient chains grow
  slowest, at 0.6358 bits per step unit, so they set the budget. The budget
  is 1,616 steps for the prime p. The phase repair of the sandwich needs a
  budget that covers every input. An under-budgeted run leaves a sign error
  on the whole shot.
- **Window lower bounds.** The same dynamic program gives every window lower
  bound exactly.
- **Two widened windows.** Two window right endpoints cover the full reach
  of their operations. One window covers the region that the phase-4 comparison
  spans. The other window covers the register edge that the new-length scan
  can reach.
- **Clean ancilla.** The location-controlled adder cells use one clean
  ancilla for their triple-Toffoli. This ancilla accounts for one of the
  three extra qubits.
- **Swap address.** The merged-swap address includes its phase-2 offset.
- **Termination.** On the steps after termination, gates leave only the
  shift count advancing.

The code labels the bounds `D-bounds`, the widened windows `D-K3` and
`D-K5`, the swap address `D-swap`, and the termination gating `D-freeze`.

The circuit expands to about 260 million operations, but the recorded
program stays small. Each EEA pass records one step body inside a repeat
block. Per-step class tables gate every step-dependent structure: the
active-window iteration trees, the end-of-iteration block, and the
telescoping write patterns. The Horner multiplier records one iteration the
same way. The expansion at step T is exactly the gate sequence of a direct
per-step generator, and the op stream stays under 45 MB.

## Cost

$$69{,}529{,}757\ \text{Toffoli} \times 838\ \text{qubits} = 58{,}265{,}936{,}366$$

| Phase | Role | Toffoli |
|---|---|---:|
| s1-const-subs | subtract the addend's coordinates from x and y | 16,320 |
| s2-idiv | divide y by x in place: the chord slope, the first sandwich | 33,630,627 |
| s3-squ | square the slope into the accumulator | 1,112,063 |
| s4-sub-squ | subtract the square from x; add back three times the addend's x-coordinate | 10,973 |
| s5-squ-dag | uncompute the square | 1,112,063 |
| s6-imul | multiply y by x in place: the second sandwich | 33,630,627 |
| s7-final | negate x and apply the closing addend adds | 17,084 |

The two sandwiches carry the total. Each runs the EEA forward and inverse.
The step loops of the four passes cost 15,142,882 Toffolis each, 60,571,528
in all. The rest costs 8,958,229. The eight Horner multiplications take
8,896,504, below the paper's 8 × 17n² = 8,912,896. The addend arithmetic and
the EEA set-up take 61,725. The circuit uses measurement-uncomputed ANDs where
the paper prices them (tree nodes, range scans, length-update bits). The
location-controlled adder cells carry their control through a clean-ancilla
triple-Toffoli. That realization reproduces the paper's 11(K−k) and 9(K−k)
block prices.

The paper prices the per-step endpoint arithmetic as O(log n). Here it
uses the same AND toolbox: incrementers and ripple adders whose carry ANDs
release by measurement (w−1 Toffolis for a w-bit add). The circuit computes
the endpoints in place in the length registers. One frame, `lq` = L and
`ls` = R, serves both r blocks. The swap address comes from it as J = L − P1.
The t endpoint comes from it as R − ℓ_r' (phase 4) or ℓ_t + 1 (phase 3),
swapped into ℓ_t under P2. The shift-count and ℓ_q updates of a step each fuse
into one controlled increment. The circuit builds the arithmetic outside the
step from the Appendix A toolbox of the paper: the constant adds and negation
of the wrapper, and the square-subtract of Fig. 13.

The peak of 838 qubits is static; every register lives through the EEA.
Field and work: 774 = 3n + 6 (the two point registers, the accumulator, and
six window extras). Lengths: 37 = 9 + 9 + 10 + 9 across the four length
registers. The shift count carries one extra bit for its post-termination
residual. Controls: 4 (the two phase bits, `Sign`, `Iter`). Scratch: 23, made
of two 9-qubit tree-path registers and 5 single qubits. The length updates run
while a tree is still open, so one tree-path register is not enough. The
single qubits are the accumulator and carry of the cells, the clean ancilla of
the triple-Toffoli of the cells, the block-control qubit, and the alive flag of
the termination gating. Every other temporary uses an idle path qubit. The
phase-selected t endpoint and the swap control P1 xor P2 take no register of
their own. While their blocks run, the endpoint lives in ℓ_t and the swap
control in P2.

## Comparison with the published count

Version 3 of the paper reports 70,290,000 Toffolis and 835 qubits at n = 256.
This circuit executes 1.1% fewer Toffolis (−760,243) and holds 3 more qubits.
The gap decomposes exactly:

- Counting adds 890,667. The paper multiplies per-block prices by
  multiplicities. This circuit counts every executed gate, including the
  per-step endpoint preparations, phase ladders, and interval-scan edges
  around the blocks. Where the paper prices a block, the cells here match:
  all five priced blocks land within 0.5% of the figure of the paper.
- The termination gating adds 103,424. The swap-address offset costs no
  gates.
- The per-shot addend adds 25,490. Coordinate-arithmetic gates that carry a
  bit condition bill at their promoted kind. The generation-time constants of
  the paper compile to Cliffords or fold away instead.
- This circuit executes 1,779,824 fewer Toffolis than the paper prices, in
  two places. The paper's bound is 1,620 steps; the exact DP bound is 1,616
  (−95,496). The windows save 1,684,328: the old-length scan here stops at
  T/4 + 4 instead of the register edge (−1,554,480), and the exact lower
  bounds trim the windows on net (−129,848). The upper caps are those of
  version 3.
- The counting line inherits the paper's rounding to 0.01M.

The 3 extra qubits sit in the scratch line. The paper's 20 cover two 9-qubit
tree paths plus the accumulator and carry of the cells. This circuit adds
three: the clean ancilla of the triple-Toffoli, the alive flag of the
termination gating, and the block-control qubit.

## Accuracy

The circuit is exact: it uses none of the failure budget of the task. Every
arithmetic block is exact on its domain. The exact step budget covers the
worst quotient chain for p. The only excluded inputs are the exceptional
cases that the paper also excludes: an identity operand, and operand pairs
that share an x-coordinate (doubling and cancellation). Together they have measure about $2^{-255}$,
and the test generator of the task skips them. The build refuses to emit if
its pinned cost or the exact step budget changes.

Shared code: `../_shared/luo_core.py` (the arithmetic and template layer) and
`../_shared/clsbits.py` (the classical-bit input layer).

## Attribution

The construction is that of Luo et al. (2026), arXiv:2607.13816, version 3.
We built the circuit from the text of the paper at its n = 256 parameters; it
uses no released code. The section "Implementation choices" describes the
choices this reconstruction makes. The section "Comparison with the
published count" accounts for every difference from the published cost.
