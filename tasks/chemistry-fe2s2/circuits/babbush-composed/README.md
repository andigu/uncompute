# babbush-composed — baseline: alias sampling and SELECT composed end to end

This folder holds the baseline circuit for the `chemistry-fe2s2` task. Its
certified cost is 55,473 Toffolis and 139 peak qubits, a TQ-cost of
7,710,747. The circuit is our reconstruction of the alias sampling and
SELECT circuits of Babbush et al. (2018), arXiv:1805.03662. The
accompanying paper (arXiv: TBD) uses it as the baseline in the SI section
"Quantum chemistry simulation".

The circuit implements the coherent alias-sampling step of Babbush et
al. (2018), arXiv:1805.03662 (Sec. III.D, Fig. 11) composed
with the same paper's SELECT (Secs. III.B and IV.A), end to end. The
alias decision and the Pauli application share nothing. Both table
traversals are the paper's sawtooth unary iteration (Sec. III.A,
Figs. 5-7). The alias lookup carries the standard variable-spaced
refinement of Sanders et al. (2020), arXiv:2007.07391
(Fig. 3). An aligned subtree whose in-domain addresses all store the
same word loads once at the subtree root. The walk never enters
out-of-domain subtrees (addresses at or above 41,052). The table's
runs of symmetry-degenerate coefficients alias identical words, so
the merging is unusually effective here. The lookup executes 13,061
Toffolis against the dense walk's 41,050. The frame-blocked order
gathers runs of identical words, and the melt alias schedule extends
them. The SELECT walk stays dense — the Pauli masks vary row to row,
so merging is inert there.

The pipeline runs in five steps:

1. The lookup walk CXes the addressed 34-bit row word into a scratch
   register.
2. A strict 16-bit comparator built from temporary ANDs computes
   $f=[\sigma<\mathrm{keep}_i]$.
3. A short multiplexor writes $\mathrm{sel}$ onto a 16-wire selection
   scratch and folds the selected sign into the phase through a
   scratch Z.
4. A second, fully independent walk over the scratch applies each
   row's Pauli masks as CZ, CX, and Z at its leaves.
5. The walk runs while the alias word is still live, so X-basis
   measurement then erases the scratch. The kickback resolves through
   outcome-conditioned Z and CZ, with no Toffolis. These gates act on
   the word's alt bits, the index register, and a held copy of $f$. A
   second comparator pass clears the $f$ copy against the keep bits.

Erasing the row word needs no second lookup. The measurement-based
unlookup of Berry et al. (2019), arXiv:1902.02134 (App. C)
X-measures all 34 word wires. It folds the outcomes into one classical
fixup bit per table row. It repairs the phase kickback with a one-hot
routing window plus a coarser walk of outcome-conditioned CZs. The
window size is $k'=32$. The Toffoli-optimal $k'=256$ of Berry et al.
would park a 256-wire one-hot register and lift the peak far above the
comparator stage's. The baseline therefore takes the member of the
published family with the lowest TQ-cost. It pays
1,312 unlookup Toffolis against 417, for 197 fewer peak qubits.

## Cost

| stage | Toffoli | peak |
|---|---:|---:|
| alias lookup (merged) | 13,061 | 121 |
| comparator + mux + sign | 34 | 139 |
| SELECT walk | 41,050 | 138 |
| selection erasure | 16 | 123 |
| unlookup ($k'=32$) | 1,312 | 114 |
| **total** | **55,473** | **139** |

Peak 139 = 72 I/O (16+16+40) + 34 word + 16 comparator carries + 16
selection scratch + the mux sign scratch, reached inside the mux.

## Accuracy

The task allows no failures. The circuit computes the task map exactly
on the whole domain and makes no approximation.

## Attribution

The circuit reconstructs published constructions from their papers. It
uses no released code.

- Babbush et al. (2018), arXiv:1805.03662. Coherent alias sampling
  (Sec. III.D), SELECT (Secs. III.B and IV.A), and sawtooth unary
  iteration (Sec. III.A).
- Sanders et al. (2020), arXiv:2007.07391. The variable-spaced lookup
  that loads a uniform subtree once.
- Berry et al. (2019), arXiv:1902.02134. The measurement-based unlookup
  (App. C).

Our choices: the window size $k'=32$ of the unlookup (see above), and
the task's row order and alias schedule, which the lookup merging
uses.
