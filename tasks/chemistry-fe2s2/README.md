# chemistry-fe2s2 — alias selection and Pauli SELECT for an iron–sulfur cluster

This task is the data-dependent block of one qubitization walk step for
a fixed 40-qubit Hamiltonian of the [2Fe-2S] iron–sulfur cluster. The
circuit compares a 16-bit value against a per-row threshold to select a
Hamiltonian term. It applies the selected term's Pauli operator to the
system register and puts the term's coefficient sign into the global
phase. It then erases the selection, so no table data remains at the
boundary. The accompanying paper (arXiv:2610.22367) describes the task and the
discovered circuit in its SI section "Quantum chemistry simulation".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. The evaluator measures both numbers from the
emitted op stream. [results.jsonl](results.jsonl) holds the certified rows.

| circuit | Toffoli | peak qubits | TQ-cost |
|---|---:|---:|---:|
| [`circuits/discovered`](circuits/discovered/README.md) | 14,978 | 126 | 1,887,228 |
| [`circuits/babbush-composed`](circuits/babbush-composed/README.md) (baseline) | 55,473 | 139 | 7,710,747 |

Relative to the baseline, the discovered circuit uses 3.7× fewer
Toffolis and 9.4% fewer qubits, for a 4.1× lower TQ-cost.

## 1. Operation

Qubitization encodes a Hamiltonian $H=\sum_\ell w_\ell P_\ell$ as a block
of a unitary walk operator. Each $P_\ell$ is a Pauli operator. PREPARE
loads the weights $\lvert w_\ell\rvert/\lambda$ into an index register,
with $\lambda=\sum_\ell\lvert w_\ell\rvert$. SELECT applies the signed
Pauli operator that the index names.

A standard PREPARE uses the alias method (Walker (1977),
doi:10.1145/355744.355749; Vose (1991), doi:10.1109/32.92917;
Babbush et al. (2018), arXiv:1805.03662; Berry et al. (2019),
arXiv:1902.02134). Classical preprocessing spreads the probabilities
over $L$ equal buckets, one per term. Bucket $i$ holds term $i$ and one
alternate term $\mathrm{alt}_i$, and a 16-bit threshold
$\mathrm{keep}_i$ sets the split. The circuit draws a bucket $i$ and a
16-bit value $\tau$ uniformly. It selects term $i$ if
$\tau<\mathrm{keep}_i$, and term $\mathrm{alt}_i$ otherwise.

This task is the map $U_{\mathrm{alias}}^\dagger\,\mathrm{SELECT}\,U_{\mathrm{alias}}$.
$U_{\mathrm{alias}}$ computes the selection from $i$ and $\tau$.
SELECT applies the selected term. $U_{\mathrm{alias}}^\dagger$ erases
the selection. The uniform superposition over $i$ and $\tau$, its
reversal, and the walk reflection are not part of the task.

Each table row $\ell$ stores three alias fields:

- $\mathrm{keep}_\ell$, a 16-bit threshold;
- $\mathrm{alt}_\ell$, a 16-bit alternate row index;
- $\theta_\ell$, the coefficient sign bit ($0$ for $w_\ell>0$, $1$ for $w_\ell<0$).

For a row index $i$ and a comparand $\tau$, define

$$
h=[\tau<\mathrm{keep}_i],
\qquad
\mathrm{sel}=\begin{cases}i & h=1,\\ \mathrm{alt}_i & h=0.\end{cases}
$$

The comparison is strict. On a basis state, $X$ flips a bit, $Z$
contributes a sign that the bit sets, and $Y$ does both. For row
$\ell$, let $a_\ell$ mark the qubits where $P_\ell$ has $X$ or $Y$.
Let $z_\ell$ mark the qubits where $P_\ell$ has $Z$ or $Y$. Every
operator in the table has an even number of $Y$ factors, so its phase
is real:

$$
P_\ell\lvert x\rangle
=(-1)^{\eta_\ell\oplus\mathrm{parity}(x\wedge z_\ell)}
\lvert x\oplus a_\ell\rangle,
\qquad
\eta_\ell=\frac{n_Y(P_\ell)\bmod 4}{2}.
$$

The circuit must implement

$$
\lvert i\rangle_{16}\lvert\tau\rangle_{16}\lvert x\rangle_{40}
\longmapsto
(-1)^{\theta_{\mathrm{sel}}\oplus\eta_{\mathrm{sel}}\oplus
\mathrm{parity}(x\wedge z_{\mathrm{sel}})}
\lvert i\rangle\lvert\tau\rangle\lvert x\oplus a_{\mathrm{sel}}\rangle .
$$

$h$ and $\mathrm{sel}$ do not appear in the output. The circuit must
return every register that holds them to zero before the boundary.

| register | width | value | final value |
|---|---:|---|---|
| `i` | 16 qubits | table row index $i$ | unchanged |
| `sigma` | 16 qubits | comparand $\tau$ | unchanged |
| `x` | 40 qubits | system register, one qubit per spin-orbital | $x\oplus a_{\mathrm{sel}}$ |

The code uses other names for some of these quantities. The grader,
the reference, and the circuits call the comparison bit `f` and the
comparand `sigma`. They call the masks $a_\ell$, $z_\ell$, and
$\eta_\ell$ `flip`, `zy`, and `ny2`.

## 2. Applications

Phase estimation applies the qubitization walk operator many times, so
the cost of one walk step sets the cost of the simulation.

Iron–sulfur clusters are standard strongly correlated targets for
quantum chemistry: many electronic configurations contribute
significantly to their states (Sharma et al. (2014), arXiv:1408.5080).
Low et al. (2025), arXiv:2502.15882, estimate resources for this
[2Fe-2S] active space.

## 3. Specification

### 3.1 Data representation

All registers are little-endian unsigned integers. Each table row is a
34-bit alias word. From the most significant bit down, it packs
$\theta_\ell$ (1 bit), $\theta_{\mathrm{alt}_\ell}$ (1 bit),
$\mathrm{alt}_\ell$ (16 bits), and $\mathrm{keep}_\ell$ (16 bits).
Row $i$ also stores the sign of row $\mathrm{alt}_i$, so one lookup of
row $i$ returns both possible values of $\theta_{\mathrm{sel}}$. Bit
$t$ of a mask refers to system qubit $t$.

### 3.2 Task instance

The Hamiltonian is the (30e,20o) active-space model of the [2Fe-2S]
complex of Li and Chan (2017), doi:10.1021/acs.jctc.7b00270. The
reference reads it from the authors' published FCIDUMP file, stored
here as `data/fcidump.xz` (see [Data files](#data-files)). The 20 spatial
orbitals give 40 interleaved spin-orbitals. The Jordan–Wigner mapping
turns them into 40 qubits. This gives 232,644 non-identity Pauli terms
with coefficient one-norm 137.2599.

The table keeps the terms with $\lvert w_\ell\rvert\ge10^{-4}$ Ha,
following the truncation practice of sparse qubitization (Berry et al.
(2019), arXiv:1902.02134). It keeps $L=41{,}052$ terms with one-norm
$\lambda=133.4650$ (97.2% of the total). The coefficients round to 16
bits of alias precision.

The map is required for every $0\le i<41{,}052$, every $\tau$, and every
40-bit $x$. The index register can hold 24,484 further values. The
grader never tests them, so the circuit may act on them with any
reversible behavior.

Each system qubit is a spin-orbital, and the Hamming weight of $x$ is
the electron count. The active space holds 30 electrons, and the
Hartree–Fock state is `0x3fffffff`. The grader draws $x$ from all
$2^{40}$ basis states, so a correct circuit cannot assume a fixed
electron count or parity.

#### Row order and frames

The row order and the alias schedule suit the circuit. Two freedoms make
this possible. A joint permutation of terms, coefficients, and Pauli
operators leaves $H$ unchanged. Many alias tables also give the same
rounded distribution.

A *frame* is a fixed Clifford transformation of the 40 system qubits
that maps one Pauli operator of the table to another. The row order uses
seven commuting frames:

| paper | code | origin | preserves coefficient magnitudes |
|---|---|---|---|
| $\sigma_{\alpha\beta}$ | `F` | exchange of spin-up and spin-down modes | yes |
| $\sigma_{XY}$ | `C` | $X\leftrightarrow Y$ exchange from spin-rotation symmetry | yes |
| $\sigma_{\mathrm{Fe}}$ | `D` | reflection that exchanges the two iron centers | no |
| $\sigma'_{\mathrm{Fe}}$ | `D2` | `D` composed with four orbital-pair swaps | no |
| $\sigma_{\mathrm{thio}}$ | `s_thio` | swap of terminal thiolate orbitals | no |
| $\sigma_{3d}$ | `s_d45` | swap of iron $3d$ orbitals | no |
| $\sigma_{\mu\mathrm{S}}$ | `x_muS` | swap of bridging sulfur orbitals | no |

The spin flip pairs every kept term with a distinct partner of equal
magnitude (20,400 pairs) or maps it to itself (252 terms). The
$X\leftrightarrow Y$ exchange matches 9,942 pair couples into quartets.
The other five frames map Pauli operators exactly where the table uses
them, but their coefficients differ by a few percent.

Rows form 3,221 blocks. A block of rank $k$ spans $2^k$ aligned
addresses, and address bit $p$ of the block switches its $p$-th frame.
Blocks with the same frame set form 14 contiguous regions, and
`contract.json` lists them. Blocks appear in decreasing size, up to
rank 5.

#### Alias schedule

The alias schedule makes nearby rows store identical alias words.
Heavy terms fill the free space of light rows in whole runs of equal
coefficients, so alias words form long constant runs. In the address
range that `F` keys, 18,463 of the 20,374 adjacent row pairs store
identical words. [`reference.py`](reference.py) and
[`alias_sched.py`](alias_sched.py) derive the table from the FCIDUMP
and check every entry.

The table also has these exact properties:

- 820 rows flip no qubit, 23,304 rows flip two, and 16,928 rows flip four.
- Every phase is $\pm1$, because every row has an even number of $Y$ factors.
- Row 40,608 is the only row with $\mathrm{keep}=0$. Its comparison
  always fails, and its alternate is itself.

#### Data files

- `data/fcidump.xz` is the FCIDUMP file `Fe2S2_and_Fe4S4/Fe2S2/fe2s2`
  of github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters,
  compressed with xz. `contract.json` pins its SHA-256.
- `data/table.json.xz` holds the 41,052 terms with their coefficients
  and alias words. `reference.py --emit` writes it.
- `data/masks.json.xz` holds the Pauli masks $a_\ell$, $z_\ell$, and
  $\eta_\ell$ of every row.
- `contract.json` holds the derived values that the grader asserts and
  the circuits read, including the 14 frame regions.

### 3.3 Correctness and examples

The map is exact. Section 6 states the conditions that every test shot
must meet. These examples apply the map to three inputs.

Row 0 is $X_{26}Z_{27}Z_{28}X_{29}X_{35}X_{36}$ with
$\mathrm{keep}=8{,}250$, $\mathrm{alt}=36{,}653$, and $\theta=0$. Row
40,608 is $Z_{12}$ with $\mathrm{keep}=0$.

- $i=0$, $\tau=0$, $x=0$: $\tau<8{,}250$, so $h=1$ and
  $\mathrm{sel}=0$. Row 0 flips qubits 26, 29, 35, and 36, so `x`
  ends as `0x1824000000`. The sign is $+1$.
- $i=0$, $\tau=8{,}250$, $x$ all ones: the strict comparison fails,
  so $h=0$ and $\mathrm{sel}=36{,}653$. Row 36,653 is $Z_{35}$. It
  flips nothing, so `x` stays all ones. Bit 35 of $x$ is 1, so the
  sign is $-1$.
- $i=40{,}608$, $\tau=0$, $x=$ `0x1000`: the comparison with
  $\mathrm{keep}=0$ fails, so $h=0$ and $\mathrm{sel}=40{,}608$, the
  row's own alternate. $Z_{12}$ flips nothing, and bit 12 of $x$ is 1,
  so the sign is $-1$.

## 4. Discovered circuit

[`circuits/discovered`](circuits/discovered/README.md) shortens both
table traversals. Frame blocks shorten SELECT: the walk visits the
3,221 block representatives for 3,327 Toffolis. The seven frames and
their region flags cost 534 more. The alias schedule makes lookup
subtrees repeat, and the lookup visits each repeated subtree once. This
lowers the alias lookup from 41,050 Toffolis for a row-by-row traversal
to 10,271. Two comparator passes and two controlled swaps cost 64
Toffolis. Measurement-based uncomputation and one joint phase repair
erase the alias fields for 782 Toffolis. The SI section "Quantum
chemistry simulation" of the paper gives the details.

## 5. Baseline

[`circuits/babbush-composed`](circuits/babbush-composed/README.md)
composes the published circuits independently. A QROM built on
unary iteration (Babbush et al. (2018), arXiv:1805.03662) loads the
34-bit alias word. The lookup merges repeated words (Sanders et al.
(2020), arXiv:2007.07391) and costs 13,061 Toffolis. A comparator and a
multiplexor select the term and its sign. A second, independent unary
iteration over $\mathrm{sel}$ applies the Pauli operator for 41,050
Toffolis. Measurement-based uncomputation (Berry et al. (2019),
arXiv:1902.02134) erases the selection and the alias word. The two
traversals contribute more than 97% of the Toffolis.

## 6. Testing

The grader runs 18,419,296 test shots: 1,642,080 structured probes
and $2^{24}=16{,}777{,}216$ pseudorandom draws.

- **Structured probes.** The grader tests every row at five comparand
  values: $0$, $2^{16}-1$, $\mathrm{keep}-1$, $\mathrm{keep}$, and
  $\mathrm{keep}+1$, clamped at the domain edges. It combines each with
  eight system values: $0$, all ones, the Hartree–Fock state
  `0x3fffffff`, $a_i$, $a_{\mathrm{alt}_i}$, and three hash-derived
  values. The first hash-derived value has odd parity. These probes
  cover both comparator branches of every row except row 40,608. They
  also cover both boundary cells and both parity sectors of every row.
- **Pseudorandom draws.** The grader draws $i$ by unbiased rejection
  sampling. It skips a draw with probability $24{,}952/2^{32}$.
  $\tau$ is uniform over 16 bits, and $x$ is uniform over 40 bits.

Every shot must end with exact register values, a clean workspace, and
the exact global sign. No failure budget applies. A leftover value of
the selection fails the workspace check. A defect in a measurement
correction flips the sign on at least half of the measurement outcomes.
Distinct rows have distinct Pauli operators, so the output and the sign
fix $\mathrm{sel}$ on every $(i,\tau)$. The contract is exact because a
sign error on one term acts coherently across the many walk steps. Such
an error shifts the estimated spectrum.

Run the grader self-test and the reference check from the repository
root:

```sh
uv run python tasks/chemistry-fe2s2/grader.py
uv run python tasks/chemistry-fe2s2/reference.py --skip-derive
```

With `--skip-derive`, `reference.py` rebuilds the table from the stored
coefficients and checks the pinned digests. Without the flag, it also
derives the coefficients again from the FCIDUMP file. That step needs
the `pyscf` and `openfermion` packages. The project does not install
them, so add them for one run:

```sh
uv run --with pyscf --with openfermion python tasks/chemistry-fe2s2/reference.py
```

## 7. References

- Babbush et al. (2018), arXiv:1805.03662. Unary iteration, QROM,
  alias sampling, and SELECT.
- Berry et al. (2019), arXiv:1902.02134. Sparse qubitization, its
  truncation practice, and measurement-based uncomputation of lookups.
- Sanders et al. (2020), arXiv:2007.07391. Merged lookup of repeated words.
- Li and Chan (2017), doi:10.1021/acs.jctc.7b00270. The (30e,20o)
  [2Fe-2S] active-space model and its FCIDUMP file.
- Sharma et al. (2014), arXiv:1408.5080. Iron–sulfur clusters as
  strongly correlated targets.
- Low et al. (2025), arXiv:2502.15882. A resource estimate for this
  active space.
- Walker (1977), doi:10.1145/355744.355749, and Vose (1991),
  doi:10.1109/32.92917. The alias method.
