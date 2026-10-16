# dqi-reed-solomon — Reed–Solomon syndrome decoding for DQI

This task applies the phase that a Reed–Solomon decoder supplies inside
decoded quantum interferometry (DQI). The instance is the optimal polynomial
intersection (OPI) instance of Khattar et al. (2025), arXiv:2510.10967. The
circuit acts on the 840-qubit syndrome register: 70 field symbols of 12 bits
each. On a superposition of syndromes, the basis-state signs become the
relative phases that drive the DQI interference. The accompanying paper
(arXiv:2610.22367) treats this task in the main-text section "Reversible syndrome
decoding for DQI" and in the SI section "Reed–Solomon decoding for OPI".

## Certified results

The cost of a circuit is its TQ-cost: the executed Toffoli count times the
peak number of live qubits. A gate controlled on a classical bit counts at
its promoted arity. Clifford gates are free. The evaluator measures both
numbers from the op stream. `results.jsonl` holds the certified values.

| circuit | role | Toffoli | peak qubits | TQ-cost |
|---|---|---:|---:|---:|
| [`discovered`](circuits/discovered/README.md) | discovered circuit | 442,397 | 1,837 | 8.13×10⁸ |
| [`khattar-eea`](circuits/khattar-eea/README.md) | baseline: Khattar et al. (2025) | 5,125,070 | 1,938 | 9.93×10⁹ |

The discovered circuit uses 12× fewer Toffolis and 5.2% fewer qubits than
the baseline, a 12× lower TQ-cost. The accompanying paper reports 7
wrong signs in 100,000 tests for the discovered circuit, within the
allowance of 100.

## 1. Operation

Regard an error word $e\in\mathrm{GF}(2^{12})^{4095}$ as the polynomial
$e(z)=\sum_{j=0}^{4094}e_jz^j$. Its 70-symbol syndrome consists of the
evaluations

$$
S_i=e(\gamma^i)=\sum_{j=0}^{4094}e_j\gamma^{ij},
\qquad i=1,\ldots,70.
$$

There is no evaluation at the zeroth power. Tested errors have symbol
weight at most $35=n/2$. The syndrome map defines a
$[4095,4025,71]_{2^{12}}$ Reed–Solomon code, so these 70 symbols determine
every such error uniquely.

For $s=(S_1,\ldots,S_{70})$, the circuit implements

$$
|s\rangle\longmapsto(-1)^{\langle c,e\rangle}|s\rangle,
$$

where $c$ is a fixed 49140-bit vector, the mask. The inner product is
binary: symbol $e_j$ occupies bits $12j$ through $12j+11$ of the packed
error word. The sign is a global phase for one basis state. It varies
with $s$, so it is a relative phase on a superposition.

| register | width | meaning | required final value |
|---|---:|---|---|
| `s[0]` … `s[69]` | 12 qubits each (840 total) | Syndrome symbol $S_{r+1}$ in register $r$ | Unchanged, with phase $(-1)^{\langle c,e\rangle}$ |

The schema declares the registers in order from `s[0]` through `s[69]`.
There is no output register.

## 2. Applications

DQI (Jordan et al. (2025), arXiv:2408.08292) turns certain optimization
problems into decoding problems. For OPI, the Reed–Solomon syndrome indexes
the amplitudes, and the interference depends on phases of the corresponding
low-weight errors. Syndrome decoding supplies the map from each syndrome to
its error-dependent sign.

The space-efficient construction of Khattar et al. (2025) generates the
error symbols one at a time and removes them with $X$-basis measurements.
The measurement outcomes leave the phase $(-1)^{\langle c,e\rangle}$, and a
reversible decoder removes it using only the syndrome. The decoder never
exposes the reconstructed error. It recovers enough information to apply
$\langle c,e\rangle$ and then erases its work. This decoder dominates the
cost of the DQI circuit, so the task isolates it. Khattar et al. argue that
the DQI computation at this instance is a candidate for verifiable quantum
advantage.

## 3. Specification

### 3.1 Data representation

The field is

$$
\mathrm{GF}(2^{12})
=
\mathrm{GF}(2)[x]/(x^{12}+x^7+x^6+x^5+x^3+x+1).
$$

The modulus has integer representation 4331 and is the primitive default
polynomial used by the `galois` package. Integers from 0 through 4095
encode the field elements: bit $t$ is the coefficient of $x^t$. The
primitive element is $\gamma=x$, represented by integer 2.

Register `s[r]` holds $S_{r+1}$. Within that register, qubit $t$ holds the
coefficient of $x^t$. The binary word paired with $c$ packs each error symbol in the
same little-endian coefficient order.

### 3.2 Task instance

The 4095 error positions correspond in order to
$\gamma^0,\gamma^1,\ldots,\gamma^{4094}$, which enumerate the nonzero field
elements. The syndrome uses the next 70 powers, from $\gamma^1$ through
$\gamma^{70}$.

Counter-mode SHA-256 generates the phase mask $c$. The tag is a fixed
byte string that defines the instance; its text carries no meaning:

```text
c = SHA-256(tag ‖ le32(0)) ‖ SHA-256(tag ‖ le32(1)) ‖ …
tag = b"kickmix/rs-decode-4095/fixup-c/v1"
```

The generator truncates the stream to 6143 bytes and packs it little-endian
into 49140 bits. Bit $k$ of $c$ is bit $k\bmod 8$ of byte $\lfloor k/8\rfloor$; the
top four bits of the final byte are unused. The stream digest is
`8fcd44e5a419ea29d84bb443d75f10dc9ac85dde2085b7851d92cd198eb3785a`,
and the mask has popcount 24492.

The 840-by-49140 binary syndrome matrix has full row rank 840. The fixed
mask lies outside its row space. Consequently, no linear parity of the supplied syndrome bits can replace
$\langle c,e\rangle$; the phase requires decoding or an equivalent nonlinear computation.

### 3.3 Correctness and examples

The circuit must be correct on the syndromes of errors of symbol weight
at most 35. Testing does not exercise other 840-bit inputs. For an
in-domain input, the circuit must preserve every syndrome qubit and apply
exactly the sign $(-1)^{\langle c,e\rangle}$. It must also return every
workspace qubit to zero before freeing it. Section 6 describes the failure
allowance.

Three fixed test cases make the encodings concrete:

- For $e=0$, all syndrome registers hold 0 and the required sign is $+1$.
- For $e_0=1$ with every other symbol zero, $S_i=1$ for all $i$. Bit 0 of
  $c$ is 0, so the required sign is $+1$.
- For $e_{2047}=2049$, the symbol has bits 0 and 11 set. Its contribution
  is therefore the parity of mask bits 24564 and 24575. Those bits are 0
  and 1, so the required sign is $-1$.

## 4. Discovered circuit

The discovered circuit replaces the synchronized Euclidean solve with the
Berlekamp–Massey (BM) recurrence (Massey (1969), doi:10.1109/TIT.1969.1054260).
It puts the retain and replace updates of BM on one shared arithmetic path.
The replace update produces a locator that differs from the textbook one by
a nonzero scale factor, which leaves its roots unchanged. The circuit stores
no accepted discrepancy: the invariant $\sigma_0\delta=1$ recovers it from
the locator's constant coefficient. Only retain rounds store a discrepancy,
and the circuit allocates one stack entry for them. Per-round coefficient
cutoffs follow the measured growth of the locator under the DQI input
distribution.

After BM, the circuit computes the phase bit directly instead of the
decoded error word. It overwrites the syndrome with the error evaluator in
place. A half-degree Euclidean pass then forms one polynomial $\Psi$ that
equals $1/\sigma'$ at every root of the locator. The scan over the 4,024
retained positions therefore costs 23 Toffolis per position and performs no
field division. The SI section "Reed–Solomon decoding for OPI" of the
accompanying paper gives the derivation.
[circuits/discovered/README.md](circuits/discovered/README.md) maps it to
the code.

## 5. Baseline

### Synchronized extended-Euclidean decoder

Khattar et al. (2025) turn Reed–Solomon decoding into a forward reversible
Euclidean pass, a sweep over field points, and a reversed pass. The forward
computation solves the key equation

$$
\sigma(z)S(z)\equiv\omega(z)\pmod{z^{70}},
$$

where $\sigma$ is the error-locator polynomial and $\omega$ is the error
evaluator. It follows the extended-Euclidean method of Sugiyama et al.
(1975), doi:10.1016/S0019-9958(75)90090-X, applied to $z^{70}$ and the
syndrome polynomial.

Classically, the Euclidean iteration count, quotient degrees, and stopping
point depend on the syndrome. The reversible construction replaces that
control flow with 222 fixed cycles, selected coherently by counters and
flags. Each cycle merges polynomial division, normalization, the Bézout
update, and the conditional swap.

The storage scheme follows the synchronized register sharing of Proos and
Zalka (2003), arXiv:quant-ph/0301141, adapted to binary-field polynomial
arithmetic by Kaye and Zalka (2004), arXiv:quant-ph/0407095. Two shared
registers each pack a remainder and a Bézout cofactor from opposite ends.
Degree invariants guarantee that both members of each pair fit at every
cycle. The quotient occupies the zero padding between them instead of
requiring another polynomial register.

Multiplying two quantum $\mathrm{GF}(2^{12})$ elements costs 51 Toffolis in
the published construction. Multiplication by a fixed field constant,
squaring, and every other $\mathbb{F}_2$-linear map use only CNOTs. Once
the circuit holds $\sigma$ and $\omega$, the sweep visits the 4095 nonzero
field points. Fixed-constant multiplications advance the evaluations from
one point to the next. A root of $\sigma$ identifies an error location, and
Forney's formula recovers the symbol value. The sweep applies that value's
contribution to $\langle c,e\rangle$ immediately. The field inversion in
Forney's formula uses an Itoh–Tsujii chain that maps zero to zero. The
reverse Euclidean pass then restores the syndrome and clears the shared
polynomial registers.

Khattar et al. estimate 5,717,729 Toffolis for the end-to-end DQI circuit
at this instance and 1,885 logical qubits for its reversible decoder
(their Table 6). Recomputing the decoder stage from their released code
(Zenodo, doi:10.5281/zenodo.17301475) with the Qualtran resource
estimator gives 5,021,091 Toffolis. Resource formulas
stand in for parts of the released decoder, so these are estimates. The
`khattar-eea` circuit realizes the complete decoder as an executable
circuit. Its Toffoli count is within about 2% of the 5,021,091-Toffoli
estimate. The qubit counts differ because the estimate takes stage-wise
maxima, while the composed circuit keeps the Euclidean state during the
sweep[^decoder-qubits]. The `khattar-eea` circuit computes the per-point
Forney exponent exactly with one flag qubit and one root-gated Toffoli per
sweep point. [circuits/khattar-eea/README.md](circuits/khattar-eea/README.md)
describes the circuit and its cost.

[^decoder-qubits]: The estimate counts each decoder stage at its own
    interface. The sweep stage counts 902 qubits because its interface
    includes only one shared polynomial register. The other 983 qubits stay
    allocated in the composed circuit for the reverse pass.

## 6. Testing

Testing runs 100,000 pseudorandom shots. The seed derives from the op
stream, so any change to the circuit draws a new test set. The first four
shots are fixed anchors: the zero error and the single-symbol errors
$(j,e_j)=(0,1)$, $(4094,4095)$, and $(2047,2049)$.

For each remaining shot, the test generator draws the error weight $k$
from the DQI principal-eigenvector distribution $|w_k|^2$ at
$p=2016/4096$, quantized on a $2^{-16}$ grid. Its support is
$19\le k\le35$. The mode is $k=32$ with probability 18.4%; weight 35
occurs with probability 2.8%; and the probability of $k\le24$ is about
0.51%. Weights below 19 have zero quantized mass, and the anchors cover
weights 0 and 1.

At the selected weight, the generator samples positions without
replacement from the 4095 field points. It samples each nonzero symbol
value uniformly from 1 through 4095.

At most 100 of the 100,000 shots may fail. A shot receives one verdict. It
fails if any syndrome register ends incorrectly, the sign is wrong, or any
workspace qubit remains nonzero.

A wrong decode of a fixed input gives the wrong sign with probability at
least one half over the measurement record. The accompanying paper uses
this to bound the failure probability $\varepsilon$ under the DQI input
distribution: 100 failures in 100,000 tests give
$\varepsilon\le2.434\times10^{-3}$ at 97.5% confidence. At that value, the
loss in the expected satisfied fraction of DQI is at most
$5.6\times10^{-3}$. That loss is at most 7.4% of the instance's advantage
over a single trial of Prange's algorithm.

## 7. Algorithm notes

- **Full-radius decoding.** The stopping condition $n_B<n/2+1$ decodes
  weight 35 correctly within
  the same iteration budget $3(n+1)+9=222$ and with the same Forney
  exponent. It gives identical outputs for weights at most 34. Weight 35
  occurs on about 2.8% of test shots, far above the failure allowance.

- **The zero syndrome occurs.** Anchor shot 0 supplies $e=0$ on every run.
  A circuit must preserve the all-zero syndrome and apply sign $+1$. Under the released
  quantum division semantics, inversion maps $0$ to $0$ through the Fermat
  power circuit. The reversible machine is therefore total on the zero
  syndrome without a special branch.

- **The sweep reverses the position index.** Write the swept point as
  $a=\gamma^i$ for $i=0,\ldots,4094$. In the released register convention,
  a root of $\sigma$ at sweep index $i$ corresponds to error position
  $j=(m-i)\bmod m$. Forney's formula gives the symbol value up to a
  per-point power of $a$ determined by that register layout.

A circuit may use any algorithm that implements the required
transformation. It need not build the decoded word as a register, and it
need not follow the extended-Euclidean design.

## 8. References

- Jordan et al. (2025), arXiv:2408.08292. DQI and its reduction of OPI to
  Reed–Solomon decoding.
- Khattar et al. (2025), arXiv:2510.10967. The reversible decoder and the
  instance of this task. Code release: Zenodo,
  doi:10.5281/zenodo.17301475 (CC BY 4.0).
- Massey (1969), doi:10.1109/TIT.1969.1054260. The Berlekamp–Massey
  algorithm.
- Proos and Zalka (2003), arXiv:quant-ph/0301141. Synchronized register
  sharing for the extended Euclidean algorithm.
- Kaye and Zalka (2004), arXiv:quant-ph/0407095. Register sharing for
  binary-field polynomial arithmetic.
- Sugiyama et al. (1975), doi:10.1016/S0019-9958(75)90090-X. The
  extended-Euclidean key-equation method.
- Prange (1962), doi:10.1109/TIT.1962.1057777. Information-set decoding.
