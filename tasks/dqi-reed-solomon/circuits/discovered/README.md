# discovered — truncated Berlekamp–Massey decoder

This folder holds the discovered circuit for the `dqi-reed-solomon` task.
Its certified cost is 442,397 Toffolis and 1,837 peak qubits, a TQ-cost of
812,683,289. The accompanying paper (arXiv:2610.22367) describes it in the SI
section "Reed–Solomon decoding for OPI"; this README maps the construction
to the code.

The truncated Berlekamp–Massey decoder replaces a synchronized Euclidean
decoder with 70 local updates of an error-locator polynomial, whose roots
identify the erroneous positions. Berlekamp–Massey is a recurrence-finding
algorithm, so each round touches only the coefficient range that can already
be active. Conditional swaps put the retain and replace cases on one arithmetic
path, allowing them to share an inversion, multiplication, and correction.

The circuit also avoids producing the 4,095-symbol decoded word when the task
needs only one phase bit. It contracts the error-value recovery identity known
as Forney's formula directly with the phase mask. Interpolation moves the
syndrome-linear contribution onto Z gates and removes 71 field points from
the nonlinear sweep, leaving 4,024. Forney's formula divides by the locator
derivative at every point; the circuit removes that division. A
half-degree Euclidean pass computes, once per shot, a polynomial $\Psi$
equal to the reciprocal derivative at every root. The sweep then pays only
a root test and one gated pairing per point. The syndrome words supply the
workspace. The first 35 temporarily store the error evaluator. The same
in-place transform zeroes the last 35, which then store the Euclidean
cofactors.
These choices give the following reversible pipeline.

```text
decode_phase:
    declare S and locator state (σ, τ, Δ, δ, d)                          ▷ init
    apply the syndrome-linear contribution with Z gates                 ▷ zlin
    repeat 70 shared-path locator updates                               ▷ bm
    scale σ so that its constant coefficient is one                     ▷ monicize
    overwrite S with σS: ω in words 1–35, zeros in words 36–70          ▷ omega
    split σ into halves and build Ψ = 1/σ′ mod σ in words 36–70         ▷ euclid
    for each of 4,024 retained points, test σ and apply the phase       ▷ sweep
    unwind Ψ and restore the halves of σ                                ▷ euclid_inv
    reverse the product that formed σS and restore S                    ▷ omega_restore
    restore the original scale of σ                                    ▷ unmonicize
    replay 69 reversible locator updates backward                       ▷ reverse
    release the zeroed workspace                                       ▷ cleanup
```

`decode_phase` is the complete reversible routine. $S$ is the syndrome
polynomial, $\sigma$ is the error locator, and $\tau$ is an earlier locator
used as a correction template. $\Delta$ and $\delta$ are the current and last
accepted discrepancies, and $d$ controls template replacement. $\omega$ is
the error evaluator, and $\Psi$ is the reciprocal locator derivative
modulo $\sigma$.

Most remaining Toffoli work lies in the locator updates, their reversal,
and the root sweep described below. The gate sequence is deterministic and
independent of the syndrome. Its approximations are the measured coefficient
cutoffs and the one-entry reversal stack of the locator updates. Across 403
million draws under the weight law, draws cross a cutoff at
$3.8\times10^{-5}$ and overflow the stack at $1.0\times10^{-4}$, an
aggregate $1.4\times10^{-4}$. One Euclidean class adds to these: a second
zero pivot in one run, which the fixed schedule does not repair. It occurs
at $4.5\times10^{-6}$ (100 of 22.4 million draws under the weight law). The same 22.4 million draws found no cofactor
support outside the measured windows and no failed erase claim. The total
sits seven times under the allowance of $10^{-3}$. On the approximate locator
paths the syndrome is always restored. Such a shot carries a wrong phase,
or, on some stack overflows, a bank qubit that is still nonzero when the
circuit frees it.
Both count against the allowance, and neither moves the syndrome. The
evaluator checks the phase and the workspace of the finished circuit.

The peak of 1,837 qubits occurs in the Euclidean passes (`euclid/seed`,
`euclid/R`, `euclid_inv/R`). The evaluator also counts 124,823,679
Clifford gates. The accompanying paper reports 7 wrong signs in 100,000
tests, within the allowance of 100. The Toffolis split by phase as follows.

| Phase             | Toffolis | Share |
|-------------------|---------:|------:|
| `bm`              |  141,071 | 31.9% |
| `monicize`        |    1,583 |  0.4% |
| `omega`           |   15,246 |  3.4% |
| `euclid`          |   44,470 | 10.1% |
| `sweep`           |   92,552 | 20.9% |
| `euclid_inv`      |   44,470 | 10.1% |
| `omega_restore`   |   15,246 |  3.4% |
| `unmonicize`      |    1,583 |  0.4% |
| `reverse`         |   86,176 | 19.5% |

The sweep share is 4,024 retained points at 23 Toffolis each, against 80
with a per-point division. The two Euclidean passes and the extension of the in-place transform
to 70 words add 107,084 Toffolis in total. The reverse pass costs less than
the forward locator updates. Measurement erases every value that the forward
pass computed into a fresh register, at zero Toffoli cost. The reverse pass
recomputes only the values that the forward pass erased by measurement. The
table omits the three phases that use no Toffoli (`init`, `zlin`, `cleanup`).

## From the key equation to shared locator updates (bm, reverse)

Let $E$ be the set of error positions. Package those positions into the
error-locator polynomial

$$
\sigma(z)=\prod_{j\in E}(1-\gamma^jz).
$$

If position $j$ contains an error, then $\sigma(\gamma^{-j})=0$. The roots of
$\sigma$ therefore identify the positions without recording them one by one.
Its degree $L=|E|$ is the error weight, so the task promise gives
$L\le35$. The constant coefficient is initially fixed as $\sigma_0=1$.

Collect the 70 supplied syndrome symbols into

$$
S(z)=S_1+S_2z+\cdots+S_{70}z^{69}.
$$

Multiplying the full syndrome generating series by $\sigma$ cancels the
denominator associated with each error position. What remains is a polynomial
$\omega$, called the error evaluator, whose degree is less than $L$. With
only the first 70 syndromes available, this relation becomes the key equation

$$
\sigma(z)S(z)=\omega(z)\pmod {z^{70}}.
$$

This equation contains the reason Berlekamp–Massey applies. Because
$\deg\omega<L$, every coefficient of $\sigma(z)S(z)$ from degree $L$
through degree 69 must vanish. Writing
$\sigma(z)=\sum_{k=0}^{L}\sigma_kz^k$, the coefficient of $z^{r-1}$ is

$$
0=\sum_{k=0}^{L}\sigma_kS_{r-k},
\qquad r=L+1,\ldots,70.
$$

Since $\sigma_0=1$ and subtraction equals addition in characteristic two,
this rearranges to

$$
S_r=\sigma_1S_{r-1}+\cdots+\sigma_LS_{r-L}.
$$

The locator coefficients are therefore a rule for predicting every syndrome
after the first $L$ from the preceding $L$ syndromes. Finding the error
positions has become the problem of finding the shortest recurrence obeyed
by the syndrome sequence. Berlekamp–Massey builds that recurrence one
syndrome at a time.

Suppose rounds before $r$ have produced a candidate $\sigma$ that correctly
predicts all processed syndromes. The next round evaluates

$$
\Delta=\sum_i\sigma_iS_{r-i}.
$$

The discrepancy is the
amount by which the current prediction rule misses $S_r$. If $\Delta=0$, the
candidate already explains the new syndrome and needs no polynomial
correction.

When $\Delta\ne0$, the algorithm needs a correction that changes the equation
for $S_r$ without disturbing equations already satisfied. It keeps an
earlier locator $\tau$ as a correction template, together with the nonzero
discrepancy $\delta$ that this template once produced. At each later round,
the algorithm multiplies the template by $z$.

Multiplying by $z$ keeps earlier equations satisfied. At the round where the algorithm
saved $\tau$, its product with the syndrome had a known nonzero coefficient
$\delta$, while the earlier relevant coefficients were zero. Multiplying
$\tau$ by $z$ moves every product coefficient one place later. After $m$
rounds, the accumulated factor $z^m$ moves that known discrepancy into
alignment with the current syndrome equation, while its zeros remain aligned
with all earlier equations. The shifted template can therefore alter the new
prediction without reopening old ones.

The desired cancellation fixes the factor that scales the aligned template:

$$
\alpha=\Delta/\delta
$$

Its contribution to the current
equation becomes $\alpha\delta=\Delta$, so

$$
\sigma\leftarrow\sigma-\alpha\tau
$$

removes the discrepancy exactly.

BM retains the earlier correction template until the current locator has
accumulated enough information from subsequent syndromes to take over its
role. A recurrence of order $L$ for $n$ processed syndromes satisfies $n-L$
constraints after its initial values. On a nonzero-discrepancy round, the
current candidate replaces the template if it has satisfied more constraints
than the stored candidate had when selected. The algorithm retains the
template if the current candidate has satisfied fewer. Either choice is valid
when the counts tie; Massey (1969), doi:10.1109/TIT.1969.1054260, p. 125,
describes this freedom.
Here recurrence order need not equal the degree of an intermediate polynomial.

The signed counter $d$ implements one such template-selection policy. Before
one-based round $r$, $r-2L=d+h$, where $h$ starts at 1 and toggles between 0
and 1 on replacement. This $h$ is proof bookkeeping, not a circuit bit.
When $\Delta\ne0$, the algorithm must replace for $r>2L$, may replace for
$r=2L$, and does not replace for $r<2L$. The circuit's test $d\ge0$ selects every
required replacement and some optional ties. On replacement, $L\gets r-L$
and $d\gets-d$; otherwise $L$ is unchanged and $d\gets d+1$. Together
with the toggle of $h$ and advance of $r$, these updates preserve the identity.
Thus $d$ is not literally the gap $r-2L$, and replacement does not always
increase order. The old locator and its nonzero discrepancy become the new
template pair whenever the circuit selects replacement.

A direct circuit would compile the retain and replace cases as two
separately controlled arithmetic paths. Only one acts on any basis state, but
both paths remain present and both contribute gates. The two cases perform
the same expensive mathematical work: divide one discrepancy by another,
multiply a locator by that quotient, and add the correction. They differ
only in which locator and discrepancy remain persistent.

The circuit moves that difference into controlled routing. On the replace
branch, the unrescaled update would form

$$
\alpha=\frac{\Delta}{\delta_{\mathrm{old}}},
\qquad
\sigma_{\mathrm{unrescaled}}
=
\sigma_{\mathrm{old}}-\alpha\tau_{\mathrm{old}},
$$

then retain $\sigma_{\mathrm{old}}$ as the new template and
$\delta_{\mathrm{new}}=\Delta$ as the accepted discrepancy. Instead, the
circuit first exchanges the current and template locators and exchanges the
current and persistent discrepancies. It then forms

$$
w=\frac{\delta_{\mathrm{old}}}{\delta_{\mathrm{new}}}
$$

and sends the shared arithmetic path through

$$
\sigma_{\mathrm{circuit}}
=
\tau_{\mathrm{old}}-w\sigma_{\mathrm{old}}
=
-w\,\sigma_{\mathrm{unrescaled}}.
$$

The circuit locator differs from the unrescaled one only by the nonzero factor
$-w$. Scaling a polynomial does not move its roots. It also scales $\omega$
and the formal derivative $\sigma'$ together, so the quotient
$\omega/\sigma'$ used during value recovery is unchanged. This rescaling lets
both nonzero-discrepancy cases share one 66-Toffoli inversion, one 42-Toffoli
field multiplication, and one locator multiply-add.

The other saving comes from not processing all 36 possible locator
coefficients in every round. After $r$ syndromes, the deterministic
worst-case degree is $\min(r,35)$, but the target distribution usually grows
much more slowly. Once the algorithm fixes $\sigma_0=1$, a degree-$L$ locator has $L$
unknown coefficients. Its evaluator has at most another $L$ unknown
coefficients because $\deg\omega<L$. Each supplied syndrome contributes one
coefficient constraint to the key equation. Thus $r$ syndromes typically pin
down a locator of degree around $r/2$, well below the worst-case bound early
in the run.

The circuit measures its truncation schedule rather than guessing it.
Classical Berlekamp–Massey simulations on the target syndrome
distribution record how far the active locator and template coefficients
usually extend in each round. Those observations determine a fixed cutoff
for every round: the smallest window such that every clipped
coefficient slot carries less than $4\times10^{-5}$ of the mass. The
measurement uses 403 million draws, and `gen_constants.py` writes the
result to `constants.json` (`bm_windows`). The resulting circuit remains
deterministic; only the choice of its static coefficient ranges depends
on the distribution.

The prefixes expand through the run and reach the full degree-35 range by
the final locator update. The discrepancy range loses one
coefficient on alternating rounds. The correction range stops one
coefficient before the swap range, except at the final update, which
covers the full range. This gradual expansion explains why the per-round
arithmetic grows instead of paying the maximum-width cost from the first
syndrome onward.

A rare input can activate a coefficient just beyond its cutoff. The
circuit stays reversible and clears its workspace, but the truncated
locator can give the wrong phase. Among 403 million independent draws
under the chosen cutoffs, 15,283 crossed one of these shortened
boundaries, a rate of $3.8\times10^{-5}$.

## In-place error evaluator (monicize, omega, omega_restore, unmonicize)

The locator says where errors occur. The evaluator $\omega$ from the
same key equation encodes their values,

$$
\sigma(z)S(z)=\omega(z)\pmod {z^{70}}.
$$

The root sweep needs the first 35 coefficients of $\omega$. Giving them a
separate twelve-bit field register each would require 420 additional qubits.
The first 35 syndrome words are otherwise idle during the sweep, so the
circuit transforms those words into $\omega$ in place and reverses the
transformation afterward. The same transform continues through all 70
words: coefficients 35 to 69 of $\sigma S$ vanish on every correct decode,
because $\deg\omega<\deg\sigma\le 35$. The last 35 syndrome words
therefore also turn into zeros, 420 qubits of workspace for the Euclidean
cofactors of the next stages.

The `monicize` phase first divides every live locator coefficient by
$\sigma_0$. Ordinarily $\sigma_0$ is nonzero because a locator is a product
of factors with constant coefficient one. A shortened coefficient history
can produce $\sigma_0=0$ on an approximate path, so the circuit substitutes
scale one in that case. This makes the operation defined and reversible on
every input.

With $\sigma_0=1$, coefficient $i$ of $\sigma S$ (the evaluator for $i<35$) is

$$
\omega_i
=
S_{i+1}
+
\sum_{j=1}^{i}\sigma_jS_{i-j+1},
\qquad 0\le i<70.
$$

Each output is the syndrome word it replaces plus a function of earlier
syndrome words, for $0\le i<70$ with $\sigma_j=0$ beyond $j=35$. The
transformation is triangular: before overwriting $S_{i+1}$, the circuit has
already consumed every occurrence of that word needed by later calculations
in the chosen order. Reversing the same sequence recovers the original
syndrome exactly, on every input, including approximate-decoder paths where
the last 35 words are not zero.

The circuit organizes the product into middle-product blocks. A middle product
computes a consecutive band of polynomial-product coefficients rather than
forming the full convolution. Each block evaluates its two input polynomials
at fixed field points, multiplies the corresponding quantum values, and
interpolates only the requested output band.

Evaluation and interpolation are fixed $\mathbb{F}_2$-linear maps. They use
CNOTs but no Toffolis. Only products of two data-dependent field values carry
nonlinear cost. A block covering $a+b-1$ product diagonals therefore needs
exactly $a+b-1$ variable field multiplications, with no work spent on
coefficients that the sweep never reads. The 70-word transform splits into
two 35-word triangles of 147 products each and one 69-product block. That
block feeds the last 35 words from the first 35. The total is 363 products,
216 more than the evaluator alone.

Once the circuit has applied the phase, `omega_restore` reverses the middle products
and reconstructs all 70 syndrome words. The `unmonicize` phase then
restores the locator scale expected by the backward Berlekamp–Massey rounds.
This order makes the full syndrome available before the backward rounds
reconstruct any recurrence history.

## Chien search and phase-only Forney recovery (zlin, sweep)

The phase mask assigns twelve binary weights to every error position. For
position $j$, those weights determine a field element $\mu_j$ such that

$$
\langle c_j,e_j\rangle_2=\operatorname{Tr}(\mu_j e_j).
$$

Here $\operatorname{Tr}$ is the field trace from $\mathrm{GF}(2^{12})$ to
$\mathrm{GF}(2)$, a binary-valued linear function. Its pairing with field
multiplication is nondegenerate, so every parity of the twelve coordinate bits
has a unique representation of this form.

This trace expression exposes a part of the phase that the circuit can apply
before decoding. Choose a degree-70 interpolation polynomial

$$
U(X)=\sum_{i=1}^{70}u_iX^i
$$

whose value agrees with $\mu_j$ at 70 selected code positions. Define the
residual mask value by

$$
\widetilde{\mu}_j=\mu_j-U(\gamma^j).
$$

Using the syndrome definition, the interpolated part satisfies

$$
\sum_j\operatorname{Tr}\!\left(U(\gamma^j)e_j\right)
=
\sum_{i=1}^{70}\operatorname{Tr}(u_iS_i).
$$

The right side is a fixed parity of the supplied syndrome bits. The `zlin`
phase applies it with Z gates, using no Toffolis and no workspace. The
residual $\widetilde{\mu}_j$ vanishes at the 70 interpolation positions and,
for this mask, at one additional position where $\mu_j$ happens to equal
$U(\gamma^j)$. The interpolation set includes the one position where the
mask itself assigns $\mu_j=0$. Those 71 positions need no nonlinear value
recovery, leaving 4,024 retained positions for the sweep. The sweep
visits all 4,095 points in order and skips the 23-Toffoli point body at a
removed position.

The sweep searches for roots of the locator at every possible error
position. Position $j$ corresponds to

$$
\beta_j=\gamma^{-j}.
$$

Evaluating $\sigma$ independently at all 4,095 points would repeat nearly
the same polynomial calculation. Instead, advancing from $\beta_j$ to
$\beta_{j+1}$ multiplies each degree-$r$ contribution by the fixed constant
$\gamma^{-r}$. Fixed field multiplication is binary-linear, so these updates
use CNOTs but no Toffolis. This incremental exhaustive root scan is a Chien
search.

The formula for an error value follows directly from the two polynomials
already in hand. Recall that the locator factors over the error positions:

$$
\sigma(z)=\prod_{i\in E}(1-\gamma^iz).
$$

Differentiate the product by omitting one factor at a time:

$$
\sigma'(z)
=
\sum_{i\in E}
(-\gamma^i)
\prod_{\ell\in E\setminus\{i\}}
(1-\gamma^\ell z).
$$

Now set $z=\beta_j=\gamma^{-j}$ for an actual error position $j$. If the
summation index is $i\ne j$, the remaining product still includes the factor

$$
1-\gamma^j\beta_j=1-\gamma^j\gamma^{-j}=0.
$$

Every derivative term with $i\ne j$ therefore vanishes. Only the term that
differentiated away the $j$th locator factor survives:

$$
\sigma'(\beta_j)
=
-\gamma^j
\prod_{\ell\in E\setminus\{j\}}
(1-\gamma^\ell\beta_j).
$$

The evaluator reaches the same surviving product by a different route. Grouping
the infinite syndrome series by error position gives

$$
S_\infty(z)
=
\sum_{i\in E}
\frac{e_i\gamma^i}{1-\gamma^iz}.
$$

Multiplying by $\sigma(z)$ cancels one denominator in each summand, giving

$$
\omega(z)
=
\sum_{i\in E}
e_i\gamma^i
\prod_{\ell\in E\setminus\{i\}}
(1-\gamma^\ell z).
$$

Evaluate this expression at $\beta_j$. For every term indexed by $i\ne j$,
the product still contains the vanishing $j$th locator factor, just as it did
in the derivative. The sole survivor is

$$
\omega(\beta_j)
=
e_j\gamma^j
\prod_{\ell\in E\setminus\{j\}}
(1-\gamma^\ell\beta_j).
$$

The derivative and evaluator values now display their common factor
explicitly. It is nonzero because the locator roots are distinct. Dividing
cancels both that product and the accompanying power of $\gamma$, leaving
the error symbol up to the fixed sign convention. In characteristic two the
minus sign itself disappears.

The circuit stores the evaluator with a shifted coefficient layout and
enumerates positions through the inverse points $\beta_j=\gamma^{-j}$.
Translating between that layout and the algebraic convention above introduces
a known position-dependent power of $\gamma$. Denote its inverse by
$\kappa_j$. The implemented recovery identity is therefore

$$
e_j
=
\kappa_j
\frac{\omega(\beta_j)}{\sigma'(\beta_j)}.
$$

This is Forney's formula in the circuit's indexing convention. The constant
$\kappa_j$ depends only on $j$, not on the syndrome or decoded error, so the
build computes it classically. The sweep folds it
into the residual mask instead of constructing or multiplying by it at run
time.

An ordinary decoder would construct all twelve bits of $e_j$, pair them with
the corresponding mask bits, and then erase them. The phase circuit absorbs
$\kappa_j$ and the residual mask into the fixed constant

$$
\nu_j=\widetilde{\mu}_j\kappa_j
$$

and computes only

$$
\operatorname{Tr}\!\left(
  \nu_j\frac{\omega(\beta_j)}{\sigma'(\beta_j)}
\right).
$$

Normalization gives
$\sigma(z)=1+\sum_{i=1}^{35}\sigma_i z^i$. The circuit tests whether
$\beta_j$ is a root, applies the contracted trace on the selected branch,
and uncomputes all point-local data before advancing. The twelve-bit field
quotient never exists as a register.

Evaluating this quotient at every retained point with a tower-field
division would cost 80 Toffolis per point. The next section removes the
division.

## Division-free Forney recovery (euclid, sweep, euclid_inv)

A per-point division spends 69 of those 80 Toffolis on
$\omega(\beta_j)/\sigma'(\beta_j)$. It spends them at every one of the 4,024
retained points, although at most 35 of them are roots. The circuit divides
once per shot instead. It computes a polynomial $\Psi$ of degree below 36 whose value at
every root is the reciprocal derivative,

$$
\Psi(\beta)\,\sigma'(\beta)=1\qquad\text{whenever }\sigma(\beta)=0 .
$$

The sweep then evaluates $\omega$ and $\Psi$ at each point by CNOTs and pays
only for a flag-gated pairing of the two values.

**Half-degree split.** In characteristic two, squaring is a linear map, so
the locator splits into even and odd halves,

$$
\sigma(z)=E(z)^2+z\,O(z)^2,\qquad
E_i=\sqrt{\sigma_{2i}},\quad O_i=\sqrt{\sigma_{2i+1}},
$$

with $E_0=1$ and $\deg E,\deg O\le 17$. The derivative is $\sigma'=O^2$,
and $\gcd(E,O)=1$ exactly when $\sigma$ has distinct roots. The square roots
are free, so the circuit rewrites the locator words in place.

**Bezout pair.** If $AE+BO=1$, then $\Psi=B^2+zA^2$ satisfies
$\Psi\sigma'\equiv 1\pmod\sigma$. The circuit runs the Euclidean algorithm on
$(E,O)$, not on $(\sigma,\sigma')$: the halves have degree 17, so every
polynomial involved fits in 18 words. A data-independent circuit needs a
fixed shape, so the circuit shifts both halves up by $m=\lfloor(35-L)/2\rfloor$
words, where $L$ is the locator degree. The shift makes $\deg E'=17$ for
every $L$. The Berlekamp–Massey counter already encodes $L$: after adding
two, its bits 2 to 6 are $m$. The shifted pair has
$\gcd(E',O')=z^m$, and the pair with $AE'+BO'=z^m$ gives
$\Psi(\beta)=1/\sigma'(\beta)$ at every root exactly, because
$(z^{2m}\sigma)'=z^{2m}\sigma'$ in characteristic two.

**Fixed Euclid schedule.** The forward pass eliminates top coefficients in
17 blocks of two. Each block tests whether the running remainder has a
nonzero top word and, if so, swaps the two halves under that flag. It then
inverts the pivot once, forms two quotient coefficients, and applies both
eliminations as one evaluation–interpolation product. The two top words that the
eliminations clear equal quotient times pivot. Measurement therefore erases
them at zero Toffoli cost, and the quotients move into the cleared words.
Sometimes a remainder degree drops by two instead of one (a zero pivot,
about 0.36% of shots). Then a controlled rotation realigns the remainder once per run. A six-bit
record notes where. One more zero-pivot test sits right after the first
elimination, where a degree defect of the odd half would otherwise go
unrepaired. A rare leading-zero case (odd $L$ with $\sigma_{L-1}=0$) swaps
the halves before the pass starts.

**Cofactors in the freed syndrome words.** The in-place evaluator transform
runs over all 70 syndrome words. The first 35 become $\omega$.
The last 35 become the coefficients 35 to 69 of $\sigma S$, which vanish on
every correct decode by the key equation. These 420 freed qubits, plus one
extra word, hold the Bezout pair as two 18-word cyclic registers. The
backward pass replays the forward journal in reverse, restoring $(E',O')$.
Between blocks it accumulates $a\leftarrow a+(q_{\mathrm{lo}}z+q_{\mathrm{hi}}z^2)b$
and swaps $a$ and $b$ under the block flag. It also repeats the zero-pivot
rotation on $b$. The pair starts from $1/h$ placed at cyclic position $m+1$, where $h$
is the last pivot; the placement cancels the shift so the finished pair sits
unrotated. Each of these steps touches only the register words that can be
nonzero at that block. The windows come from 400,000 draws under the weight
law plus all anchor inputs (`psi_windows` in `constants.json`), and
`gen_constants.py` regenerates them.

**Per-point sweep.** The sweep never rotates a word. At point
$x=\gamma^t$ it reads each operand as a parity of stored bits: the value
$E'_k$ and $b_k$ would reach after $t$ multiplies by $\gamma^k$, and $O'_k$
and $a_k$ after $t$ multiplies by $\gamma^{k+2048}=\sqrt{\gamma}^{\,2k+1}$.
Each bit of such a product is a fixed parity of the stored word. Every
operand is therefore a precomputed parity set over the unrotated words. The
evaluator load into `som` works the same way. The alternative rotates
106 words between points with constant-multiply CNOT networks. A
lattice-surgery compiler must run those networks as real CNOTs. Here every
sweep CNOT wraps one gate and is undone after it, so it compiles to one
wide parity measurement. The two forms have the same Toffoli count and
peak; only the Clifford count differs. Under a lattice-surgery time model with $1.5d$ rounds
per Toffoli and $10d$ per constant-multiply network, for code distance $d$,
one run takes about 3.5 times less time. Because the words stay fixed, the
circuit records the whole sweep as one data-gated `repeat` body. This shrinks
the stream about threefold and leaves the executed gates unchanged. The XOR of the 36 locator words
is then $E'(x)+\sqrt{x}\,O'(x)=x^m\sqrt{\sigma(x)}$, zero exactly at a
root, and the XOR of the 36 pair words is $V=B(x)+\sqrt{x}\,A(x)$ with
$\Psi(x)=V^2$. With $U=\omega(x)$ and the residual mask value $\nu$,

$$
\operatorname{Tr}(\nu U\Psi)=\operatorname{Tr}\!\left(\sqrt{\nu U}\,V\right)
=\sum_{a=0}^{11}\langle r_a(\nu),U\rangle\,V_a ,
$$

a bilinear form of rank 12 with rows $r_a$ fixed per point. Each term is
one Toffoli gated by the root flag.

| Step                                          | Toffolis |
|-----------------------------------------------|---------:|
| locator-root test (AND of twelve bits)        |       11 |
| gated pairing, rank 12                        |       12 |
| **total per point**                           |   **23** |

`euclid_inv` is the literal reverse of `euclid`. The one-way costs are as
follows. The 70-word transform costs 9,072 extra Toffolis, the setup 5, the
shift 1,416, and the leading-zero swap 227. The forward pass costs 15,201, the seed 390,
and the backward pass with the cofactor steps 27,231. The sweep costs 92,552
Toffolis, against 321,920 with a per-point division. The circuit XOR-accumulates the seed and
recomputes it on the reverse instead of erasing it by measurement. The
syndrome words it lands in are zero only on a correct decode.

## Reconstructing compressed reversal history (bm, reverse)

A reversible locator update must eventually recover every value overwritten
during the forward pass. The current discrepancy is the difficult one.
Keeping a twelve-bit history word for every round would avoid reconstruction,
but it would add 840 qubits to the circuit width. The circuit instead
derives a displaced discrepancy from the updated locator on replace rounds,
where the algebra makes that possible. On negative-$d$ rounds it saves
discrepancies.

For the first 69 rounds, there are three cases. If $\Delta=0$, the locator
does not change, so there is no nonzero discrepancy to recover. If
$\Delta\ne0$ and $d\ge0$, the circuit replaces the template: it exchanges the current locator with
the shifted correction template. The new discrepancy
$\delta_{\mathrm{new}}=\Delta$ remains live, while the
exchange displaces $\delta_{\mathrm{old}}$. In this case the circuit can
reconstruct the old value.

To see why, examine the constant coefficients after the exchange. The
circuit multiplied the template by $z$ at the start of the round, so it has constant
coefficient zero. The exchange puts that shifted template into the register
that then holds the updated $\sigma$, while the old locator occupies the
$\tau$ register. Let $\tau_0$ be the old locator's constant coefficient and
let

$$
w=\frac{\delta_{\mathrm{old}}}{\delta_{\mathrm{new}}}.
$$

The shared update subtracts $w\tau$ from the shifted template. Since the
shifted template contributes zero to the constant term, the updated constant
coefficient comes entirely from the correction:

$$
\sigma_0=w\tau_0.
$$

The old discrepancy is therefore determined by values that remain live after
the round:

$$
\delta_{\mathrm{old}}
=
\sigma_0\tau_0^{-1}\delta_{\mathrm{new}}.
$$

The same argument applies to every earlier replace round, so the
constant coefficient carries the whole scale history: on every round of a
correct decode,

$$
\sigma_0\,\delta = 1,
$$

where $\delta$ is the accepted discrepancy. The circuit therefore stores no
accepted discrepancy at all. It forms $p=\Delta\sigma_0$ with one field
multiplication and inverts $p$ with the 66-Toffoli chain. The keep branch
needs $w=\Delta/\delta=p$, and the replace branch needs
$w=\delta_{\mathrm{old}}/\Delta=p^{-1}$, so a 12-Toffoli controlled
selection gives $w$. After the update, three identities erase the
temporaries. The selection erases against $p$, $p^{-1}$ and the branch
flag. The product erases through the claim
$p=\Delta(\tau_0+\sigma_0)+[\text{replace}]$, which holds because the
replace branch leaves $\sigma_0=\Delta^{-1}$ and the keep branch with
$\Delta\ne0$ leaves $\tau_0=0$. The temporary discrepancy register erases
through $\Delta=[\text{replace}]\,\sigma_0^{-1}$ after a second inversion
chain on $\sigma_0$. Every claim is a measurement erase, free on the
forward pass and recomputed by `reverse`.

Thus the replace branch does not carry a displaced discrepancy through the
evaluator and root sweep, and no discrepancy register is persistent. The
locator itself contains enough information to reconstruct it during
`reverse`. The claims fail only where the scale identity fails. The model
check found 2 such shots in 200,000 law draws, both on approximate locator
paths where the phase is already wrong.

The third kind of round has $\Delta\ne0$ and $d<0$. It retains the
template, so no locator exchange occurs. The accepted discrepancy
stays $\delta_{\mathrm{old}}=\sigma_0^{-1}$, while the current $\Delta$
exists only in the temporary discrepancy register. The post-update state does determine a
nonzero $\Delta$. Write the retained template as $z^mQ$ with $Q_0\ne0$. Then
$\Delta=[z^m]{(\sigma'Q^{-1})}/\sigma'_0$, where $\sigma'$ is the updated locator. A classical
model of this circuit's update recovered all 7,527 nonzero pushed values this way, all at
$m=1$. The circuit does not use this rule. Reconstruction would save the 19 stack qubits
but cost about 7.7k Toffolis, a net increase in TQ-cost. The circuit therefore saves $\Delta$
before it erases the temporary register.

The implementation pushes on every negative-$d$ round among the first 69,
including a zero word when $\Delta=0$. These entries naturally form a
last-in, first-out stack because the reverse pass visits the rounds from newest to oldest.
The balance dynamics permit at most 34 such pushes. The circuit saves the final
discrepancy unconditionally and omits the final template exchange, because
no later round needs it. Thus a worst-case-safe stack needs 35 entries. Each entry
contains a twelve-bit discrepancy and a seven-bit round tag:

$$
35\times(12+7)=665\ \text{qubits}.
$$

The round tag tells the reverse pass when to pop the top entry.
Without it, the circuit would have to retain all 69 per-round push flags
through the evaluator and sweep.

The target distribution rarely produces negative-$d$ rounds. Starting at
$d=0$, consecutive nonzero discrepancies select replace and leave $d=0$.
A zero discrepancy raises $d$; a subsequent replacement can make it negative,
causing retain rounds until the counter returns to zero. Accidental zero
discrepancies before completion of the locator are uncommon over
$\mathrm{GF}(2^{12})$. After completion, discrepancies stay zero and do not
start new negative-$d$ excursions. The target distribution therefore almost
never needs the full stack. This circuit
allocates one entry, or 19 qubits, rather than 665. Among 403 million
independent draws under the chosen cutoffs, 40,996 required a second
entry and could therefore fail. That is a rate of
$1.0\times10^{-4}$; 167 required a third. A second entry would cut the
aggregate failure rate to $3.8\times10^{-5}$ (the cutoff crossings
alone). But it stays live from round 4 through the sweep and
so sits inside the peak. Every push and pop also moves it with 19
Fredkin gates. It costs 19 qubits and 2,470 Toffolis, a 1.6% higher
TQ-cost, for a margin the allowance does not need. The stack size is the
second distribution-dependent approximation, alongside the measured
coefficient cutoffs.

During `reverse`, the circuit visits the locator rounds from newest to oldest.
It pops a saved discrepancy for each recorded negative-$d$ round and uses
the constant-coefficient identity for each replace round. By then
`omega_restore` has recovered the first 35 syndrome words, so all 70 symbols
needed by the recurrence are again available. Even when a coefficient crosses a
cutoff or the stack overflows, the chosen reversible path still restores the
syndrome. On an overflow, the second push writes its round tag over the
stale one. The corrupted tag can misroute one reversed round and leave a
bank qubit nonzero when the circuit frees it. On 258 sampled
overflow draws, the circuit restored the syndrome exactly on every one.

## Bilinear field arithmetic (bm, monicize, euclid, unmonicize, reverse)

The construction of Khattar et al. (2025) uses 51 Toffolis for a
general multiplication in $\mathrm{GF}(2^{12})$ (one level of Karatsuba
over a 17-product six-term formula). It uses 255 Toffolis for an inversion
(five multiplications). This circuit instead decomposes the same
bilinear map into 42 rank-one terms, found by a Chinese-remainder
evaluation over $\mathrm{GF}(4)$. The twelve-bit product is a six-term
polynomial product over $\mathrm{GF}(4)$. The evaluation takes it at the
four $\mathrm{GF}(4)$ points and infinity and reduces it modulo three
irreducible quadratics. This gives 14 $\mathrm{GF}(4)$ products of 3
Toffolis each. A rank-one term forms one binary parity from each operand
and multiplies those two parity bits with one Toffoli. CNOTs then
distribute the result to selected output bits.

The CNOT networks before and after the Toffoli layer are fixed linear
maps. The Toffoli count is therefore exactly the number of bilinear
terms, giving 42 for a full field multiplication. The same synthesis
gives 15 terms for the six-bit subfield $\mathrm{GF}(64)$ and 6 for the
three-bit subfield $\mathrm{GF}(8)$.

Inversion uses a two-level tower of subfields. With $N=x\bar{x}\in\mathrm{GF}(64)$ and
$M=N\bar{N}\in\mathrm{GF}(8)$,

$$
x^{-1}=\bar{x}N^{-1},
\qquad N^{-1}=\bar{N}M^{-1},
\qquad M^{-1}=M^{6}=(M^{3})^{2}.
$$

The norm $N$ costs 15 Toffolis and $M$ costs 6. A three-product
quadratic form builds $M^3$, and squaring then produces $M^{-1}$ through
a fixed linear map, so that step uses CNOTs but no Toffolis. Two
$\mathrm{GF}(8)$ multiplications form $N^{-1}$ (12 Toffolis) and two
$\mathrm{GF}(64)$ multiplications form $x^{-1}$ (30). The full
inversion costs 66 Toffolis, against 255 for the construction of Khattar
et al. A one-level tower at this instance would need a generic
$\mathrm{GF}(64)$ product inside its power chain. The Euclidean pass inverts its pivot
once per block with this chain.

The exponentiation also defines the zero input without a branch. Every
positive power of zero is zero, so the circuit maps zero to zero.
Discrepancy division and the Euclidean pivots therefore remain
reversible on exceptional paths, including the all-zero syndrome.
