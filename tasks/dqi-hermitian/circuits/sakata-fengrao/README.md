# sakata-fengrao — baseline: oblivious lazy pivot-slot Feng–Rao decoder

This folder holds the baseline circuit for the `dqi-hermitian` task. Its
certified cost is 104,263,342 Toffolis and 194,053 peak qubits, a TQ-cost
of 20,232,614,305,126. The circuit is a direct reversible implementation of
the published Feng–Rao decoding algorithm: the majority-voting decoder of
Feng and Rao (1993), doi:10.1109/18.179340, in the form of Sakata, Jensen,
and Høholdt (1995), doi:10.1109/18.476248. No prior reversible compilation
of this decoder exists. `circuit.py` builds a parametric stage pipeline and
runs it at the task instance: field parameter $q_0 = 8$, $K = 48$ pivot
slots, and the rerun schedule described below.

The circuit runs the decoder as a fixed gate schedule. The classical algorithm branches on data at every step:
which cell claims a pivot, which candidate wins a vote, when
elimination ends. A circuit run in superposition cannot branch.
Every elimination cell executes all three branch bodies under mutually
exclusive mask qubits: store below a pivot, extend a pivot row, claim
a new pivot. Pivot bookkeeping lives in 48
provisioned slots addressed by walking one-hots. Each vote is a
strict-majority threshold scan over 64 value counters instead of a
data-dependent argmax. The same gates execute on every input.

The circuit never materializes the syndrome staircase matrix. Holding the 149 x
149 array of GF(64) words and rewriting it in place is the expensive
default; here every entry stays virtual. The raw part of an entry is an
XOR of syndrome words — linear over GF(2), so it costs only CX. The
elimination correction is a sum of at most 48 stored products, one per
claimed slot: sub-pivot entry times normalized pivot row. Only those
products cost Toffoli. The circuit stores the sub-pivot entry itself,
rather than the classical elimination factor. Every transient value word
can then erase by measurement, and CZ against still-live registers
repairs its phase debt. The inner loop thus uncomputes at zero Toffoli.
The same linearity pays off downstream. The kernel test matrix fills by
CX from the syndrome words, and membership evaluation multiplies by
classical constants (CX only). The final solve gathers the parity-check
columns flagged as error locations under a walking one-hot (classical
bits under a quantum mask, CX only). Its elimination therefore runs on a
compacted 123 x 76 system instead of one column per code position.

What stays live across the solve stage, which owns the peak, sets
the peak qubits. A last-in-first-out unwind would hold the kernel
elimination block there. Instead the rerun schedule reverses the kernel
before solve and rebuilds it gate for gate afterward; solve does not
touch the kernel's inputs. The vote counters and membership scratch
are provably zero across the same span. The circuit frees them and
later revives them at their exact wire ids.

## Cost

The certified cost is 104,263,342 Toffolis and 194,053 peak qubits:

$$104{,}263{,}342\ \text{Toffoli} \times 194{,}053\ \text{qubits} = 20{,}232{,}614{,}305{,}126\ \text{(TQ-cost)}$$

| Phase | Role | Toffoli |
|---|---|---:|
| alloc / teardown | allocate, then free, the 109,853-wire persistent store | 0 |
| votes/cells | the 7,801 masked elimination cells | 20,452,429 |
| votes/round | the 27 strict-majority votes, one voted word each | 10,071,312 |
| kernel | masked Gauss-Jordan elimination of the 75 x 48 test matrix | 3,976,992 |
| member | flag the points where every kernel basis vector vanishes | 318,976 |
| un-kernel | the kernel reversed twice: before solve, and at the end | 7,946,784 |
| solve | compact the flagged columns; eliminate the 123 x 76 system | 12,130,099 |
| pay | the sign: CZ of the solved symbols against the fixed vector | 194,048 |
| un-solve | the solve stage reversed | 12,120,874 |
| re-kernel | the kernel rebuilt forward, gate for gate | 3,976,992 |
| un-member | the membership sweep replayed (self-inverse) | 318,976 |
| un-votes | vote rounds and elimination cells reversed | 32,755,860 |

Solve reaches the peak of 194,053 qubits. It sums 738 syndrome I/O + 108,726
voting bank + 512 membership flags + 83,588 solve block + 489
elimination transients. The voting bank holds the sub-pivot stores,
normalized pivot rows, pivot inverses, slot one-hots, used flags, claim
bits, and voted words. The solve block is 56,088 for the compacted
system and 17,550 for stored pivot-row tails. Pivot selection and the
compaction one-hot take the other 9,950. The other 615 wires of the persistent store are
the vote counters (520) and the per-column membership scratch (95).
They are provably zero across the span that holds the peak. The
circuit frees them there and revives them at their exact wire ids.

Measurements back each schedule trade. Reversing the kernel before
solve and rebuilding it afterward costs 7,950,384 Toffoli — one extra
forward and one extra reverse. It keeps the 39,435-wire kernel block
off the peak. The solve elimination stores only the tail of each
normalized pivot row, from its own column on, which roughly halves
pivot storage. The kernel elimination stores full rows because
membership reads entries left of the pivot. The circuit cleans the
vote counters every round (count up, threshold scan, count down)
rather than carrying argmax-chain state. That choice costs about 6
percent more Toffoli and saves about 25 percent of the qubits.

## Accuracy

The evaluation tolerates up to 100 failed shots in 100,000; the design
spends none of that allowance. Every gadget
is exact GF(64) arithmetic. The decode is exact at every error
weight up to the unique-decoding radius 47 — the largest weight the
tested input distribution produces. The circuit provisions each way the
fixed schedule could fall short for its worst decodable case:

- The 48 pivot slots cover every claim sequence. The error weight
  bounds the claim count, and the largest count observed over draws
  from the input weight law is 47.
- The 75 compaction slots cover every flagged-point set (at most
  radius plus genus points).
- The vote counters are wide enough for the largest candidate count
  of any round.
- The true value always reaches the strict-majority threshold,
  because the Feng-Rao vote bound guarantees it a strict majority.

A full-rank test matrix (an empty kernel basis) cannot occur for a
decodable syndrome, so the design predicts zero failed shots. The
whole allowance is headroom. The build asserts the Toffoli and qubit totals and
refuses to emit if either drifts. The evaluator checks the finished
circuit independently.

## Attribution

The decoding algorithm is the majority-voting decoder of Feng and Rao
(1993), doi:10.1109/18.179340, in the form of Sakata, Jensen, and Høholdt
(1995), doi:10.1109/18.476248. It extends the multidimensional
Berlekamp–Massey algorithm of Sakata (1990),
doi:10.1016/0890-5401(90)90039-K. We built the circuit from these papers;
no released code exists, and no code is transcribed. The reversible
schedule is our own: masked execution of every branch body, the pivot
slots, the threshold votes, the virtual staircase matrix, and the rerun
schedule.
