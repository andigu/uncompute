# euclidean — specialized Euclidean point addition

This folder holds the discovered circuit for the `modinv-point-add` task. Its
certified cost is 20,521,949 Toffolis and 843 peak qubits, a TQ-cost of
17,300,003,007. The accompanying paper (arXiv: TBD) describes it in the
supplementary section "Specialized Euclidean construction"; this README maps
the construction to the code.

The task has two discovered circuits; this one is the second, beside
`../divsteps/`. It sits at the low-qubit end of the Toffoli–qubit frontier:
843 qubits against 935 for the divstep circuit, at about five times the
Toffoli count.

The search did not find a new inversion algorithm here. It started from the
space-efficient Euclidean point adder of Luo et al. (2026), arXiv:2607.13816.
Our reconstruction of that adder is the baseline in `../luo-eea/` (70 million
Toffolis, 838 qubits). The search specialized the schedule of the baseline
for the prime $p=2^{256}-2^{32}-977$. The schedule is the fixed list of steps,
and of bit positions within each step, that the circuit runs. The
specialization reduces the Toffoli count to 21 million, a factor of 3.4, with
five additional qubits.

## Construction

The chord slope and the output coordinates come from two in-place field
operations: one division and one multiplication. Each operation runs an
extended Euclidean walk forward, uses its result, and runs the walk backward.
The reverse pass is the inverse circuit, so the walk leaves every walk register
clean without a stored history. The walk packs a shrinking remainder and a
growing coefficient into opposite ends of one register. Two 259-bit registers
(a 256-bit register plus three extra wires each) hold the four values. Four
short counters mark the boundaries of the values.

Like the baseline, this circuit derives the worst-case step bound (1,616
steps) and every worst-case window bound exactly, by dynamic programming over
quotient-sequence continuants. All-2 quotient chains grow slowest, so they set
the bound.

The circuit keeps the packed layout of the baseline. It reduces the cost of
the schedule in the following ways (paper terms; the code names follow in
parentheses):

- **Fitted windows** (`wl_windows`). Each scan covers a window fitted to the
  positions that sampled walks use, not the worst-case range.
- **Schedule cut** (`NMAX_SCHED`). The step loop stops at 1,432 steps instead
  of the worst-case 1,616.
- **Test-free positions** (`tcore`, `rcore`, the certainty cores in the
  code). Where every sampled walk uses a position, the circuit runs the
  arithmetic without the counter test. A scan is then a few tested positions
  near the moving boundaries plus a run of test-free positions.
- **Merged scans.** One merged compare-and-subtract pass replaces the two
  passes of the baseline. The circuit erases the intermediate bits by
  measurement. The conditional rotations collapse into one unconditional
  relabeling of positions per step (Cliffords only) plus a counter-controlled
  barrel correction at the ends of the walk.
- **Measured carries.** A carry that only one position needs is erased by
  measurement.
- **Sparse-prime multipliers.** The eight Horner multipliers fold high bits
  with $2^{256}\equiv2^{32}+977\pmod p$.
- **Truncated multipliers.** The fold chains and comparisons of the
  multipliers stop at fixed bit windows.

The fitted windows, the schedule cut, the test-free positions, and the
truncated multipliers fail on rare inputs. Those failures count against the
failure budget. The other changes are exact.

## Cost

$$20{,}521{,}949\ \text{Toffoli} \times 843\ \text{qubits} = 17{,}300{,}003{,}007$$

| Phase | Role | Toffoli |
|---|---|---:|
| s1-const-subs | shift both coordinates by the addend's coordinates | 16,320 |
| s2-idiv | in-place division writes the chord slope: the first walk pair | 10,011,835 |
| s3-squ | square the slope into the workspace | 226,951 |
| s4-sub-squ | subtract the squared slope; add three times the addend's x coordinate | 10,973 |
| s5-squ-dag | the mirrored square erases the workspace | 226,951 |
| s6-imul | in-place multiplication writes the output coordinate: the second walk pair | 10,011,835 |
| s7-final | negate, then apply the closing addend shifts | 17,084 |

The two walk pairs are 98 percent of the total. They are equal by
construction, as are the square and its mirror. The peak of 843 qubits is
512 I/O + 256 workspace + 75 machine wires (6 walk-frame extras, 36 length
counters, 4 flags, 29 scratch). The circuit allocates all of them at
initialization, so the peak is flat until teardown. Of the machine wires, 55
are zero between the walks and serve as the multiplier pool in the middle
phases. This pool splits the fold chains of the multiplier into two blocks.
A 64-wire pool would save 48,960 Toffolis and cost 9 more qubits.

The t-side endpoint takes no register of its own. The step computes it in
place in the length counters and swaps it into `lt` under `Phase2`. The
small-register arithmetic of a step releases its carry ANDs by measurement.
A counter costs w−1 Toffolis (w−2 uncontrolled), and a ripple adder costs w−1
for w bits. The per-step update of the shift count is one increment in
complement fans controlled on Phase2 = 0. The two ℓ_q updates of a step and
the frame teardown net to one controlled decrement. The op stream records one
fixed step body under a symbolic loop. Per-step gating tables turn each tree
node and scan cell on or off, so a narrower window deletes its Toffolis.

## Fitting the schedule

`gen_constants.py` fits the window schedules and both tables of test-free
positions, and it writes `constants.json`. A fast classical model of the walk
(`fasteea.c`, with a Python fallback) matches the bit-level model of the
circuit step for step. A base pass over 60,000 sampled walks takes a per-step
quantile of the positions each scan needs. A second pass over 300,000 walks
sets each margin to the larger of an empirical quantile and an exponential
tail fit. The pass splits the per-walk failure share across scans in
proportion to their Toffoli sensitivity.

## Accuracy

The task allows 20 failed tests per 100,000 ($2\times10^{-4}$), one verdict per
test. A wrong value, a leftover phase, and a dirty ancilla use the same
budget. Five failure channels apply, per test (a test runs two walks):

| Channel | Priced worst case | Observed |
|---|---:|---:|
| fitted-window escapes | ~1.6e-4 | ~1.4e-5 |
| test-free-position escapes | ~3e-4 | ~5.2e-5 |
| schedule cut | 1.3e-5 (direct measurement) | 1.3e-5 |
| multiplier fold and comparison escapes | ~2e-6 (analytic) | — |
| multiplier windows and carry slack | < 1e-10 | — |

The priced column sums to about 4.8e-4, 2.4 times the budget. The accuracy
case rests on the observed column instead. A run over 10,000,000 sampled
walks observes 72 window escapes and 258 test-free-position escapes (60 on
the t side, 198 on the r side), about 6.6e-5 per test. With the schedule cut
and the multipliers, the total is about 8e-5 per test: about 8 expected
failures per 100,000 against the allowed 20. `../../results.jsonl` holds the
certified verdict. The build refuses to emit if its pinned cost or tables
change.

Shared code: `../_shared/luo_core.py` (the arithmetic and template layer) and
`../_shared/clsbits.py` (the classical-bit input layer).
