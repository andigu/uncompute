# discovered — matrix-free block-Lanczos locator walk

This folder holds the discovered circuit for the `dqi-hermitian` task. Its
certified cost is 926,912 Toffolis and 2,395 peak qubits, a TQ-cost of
2,219,954,240. The accompanying paper (arXiv: TBD) describes it in the SI
section "Hermitian-code decoding for HOPI"; this README maps the
construction to the code.

The matrix-free block-Lanczos locator saves cost by storing neither the symmetric
$48\times48$ syndrome matrix nor the 512-symbol decoded error. It applies the
matrix through fixed linear evaluation maps and 123 pointwise field
multiplications, then keeps only neighboring Krylov vectors while accumulating
a kernel vector. When a nonzero vector is orthogonal to itself, a grouped
Lanczos update repairs the otherwise undefined projection.

Repair arithmetic appears in only 17 of 52 scheduled slots. Ordinary slots
cost $5{,}400$ Toffolis, while depth-two and depth-three repair slots cost
$11{,}964$ and $16{,}557$. Once the walk yields the locator, a striped local sweep
extracts only the error information needed for the phase and clears each batch
before proceeding.

```text
PhaseFixup:
    η ← linear image of s; w ← y₀; x ← 0                  ▷ init
    x ← scheduled_block_lanczos(A, w, η)                  ▷ P1
    mode ← classify_terminal_state(A, w, x)               ▷ mux
    for each curve-point stripe, apply local_phase(s, c, x, mode) ▷ tail
    erase mode and any retained terminal data             ▷ unmux
    measure x and record its phase-correction bits         ▷ xseam
    reverse the scheduled walk; apply corrections; return s ▷ P2
```

Here $s$ is the syndrome, $c$ the fixed phase mask, and $\eta$ the
syndrome-derived weights defining the implicit operator $A$. $y_0$ is the
fixed Krylov seed, $w$ its live Lanczos vector, $x$ the accumulated locator,
and `mode` the supported terminal case selected for the local sweep.

The schedule and its reverse are deterministic. A register-level model
checks them across ordinary, depth-two, depth-three, multiple-breakdown,
terminal, and abstaining branches. An abstain withholds
the sign when the supported schedule cannot finish, but still restores the
input and clears the workspace. The accompanying paper reports 30 wrong
signs among 100,000 tests, within the allowance of 100.

## Locator equations from the syndrome (init)

Let $E$ be the unknown error support, with error value $e_i$ at curve point
$P_i$. For any function $f$ on the curve, the supplied syndrome information
defines the linear functional

$$
S(f)=\sum_{P_i\in E} e_i f(P_i).
$$

The locator is a bivariate function

$$
\sigma=\sum_{k=0}^{47}\sigma_k\psi_k,
$$

where $\psi_0,\ldots,\psi_{47}$ are the first 48 monomials in pole order.
Each error point requires $\sigma(P_i)=0$. There are at most 47 such
homogeneous conditions on 48 coefficients, so some nonzero locator always
exists.

The circuit replaces those unknown point conditions with a matrix equation
whose entries are input syndromes. Define the symmetric matrix

$$
A_{k\ell}=S(\psi_k\psi_\ell).
$$

Each product $\psi_k\psi_\ell$ has pole order at most 150. Reducing it with
the curve relation $y^8+y=x^9$ leaves one or two of the 123 monomials whose
syndromes the input supplies. Every entry of $A$ is therefore either one input
symbol or the sum of two; the enlarged locator space requires no additional
syndrome information.

To see why a kernel vector is a locator, let $M_E$ evaluate the basis on the
error support,

$$
(M_E)_{ik}=\psi_k(P_i),
\qquad
\Delta_E=\operatorname{diag}(e_i:P_i\in E).
$$

Expanding a matrix entry gives the factorization

$$
A=M_E^{\mathsf T}\Delta_E M_E.
$$

For the coefficient vector $\boldsymbol{\sigma}$ of a genuine locator,

$$
(A\boldsymbol{\sigma})_k
 =S(\psi_k\sigma)
 =\sum_{P_i\in E}e_i\psi_k(P_i)\sigma(P_i)
 =0.
$$

Thus every locator lies in $\ker A$. Conversely, if the vanishing conditions
are independent, then $M_E$ has full row rank. Its transpose is injective and
$\Delta_E$ is invertible because error values are nonzero, so

$$
A\boldsymbol{\sigma}=0
\quad\Longrightarrow\quad
M_E\boldsymbol{\sigma}=0.
$$

Under this condition, every kernel vector is a genuine locator. The measured
probability of a dependent support under the specified DQI distribution is
about $8\times10^{-8}$; such supports are the exceptional case of this
reduction.

## Matrix-free Krylov and Lanczos steps (P1)

Materializing $A$ would preserve many field words that the circuit needs only
while multiplying a vector. Instead, the circuit factors its action through
evaluation at 123 fixed curve points:

$$
A\boldsymbol{u}
=
\Phi^{\mathsf T}
\operatorname{diag}(\boldsymbol{\eta})
\Phi\boldsymbol{u}.
$$

The map $\Phi$ evaluates the function represented by $\boldsymbol{u}$, and
the syndrome-derived weights $\eta_r$ represent the functional $S$ in that
evaluation basis. The transpose map returns the weighted evaluations to the
48 locator coordinates.

Both surrounding maps are fixed and $\mathrm{GF}(2)$-linear, so they require
only CNOTs. Nonlinearity appears solely in the 123 pointwise products
$\eta_r(\Phi\boldsymbol{u})_r$. A bilinear decomposition over
$\mathrm{GF}(4)$ implements one $\mathrm{GF}(64)$ multiplication with 15
Toffolis, making one application of $A$ cost

$$
123\cdot15=1{,}845
$$

Toffolis without storing any matrix entry.

Starting from $y_0$, repeated applications generate the Krylov sequence

$$
y_0,\ Ay_0,\ A^2y_0,\ldots.
$$

All vectors lie in a 48-dimensional space, so some nontrivial relation must
occur:

$$
\lambda_0y_0+\lambda_1Ay_0+\cdots+\lambda_dA^dy_0=0.
$$

A relation is not automatically a kernel vector; its constant coefficient
must vanish. Let $v\in\ker A$ satisfy $v^{\mathsf T}y_0\ne0$. Symmetry gives

$$
v^{\mathsf T}A^ry_0
=
(A^rv)^{\mathsf T}y_0
=
0
\qquad (r\ge1).
$$

Taking the inner product of the dependence with $v$ therefore leaves only
$\lambda_0v^{\mathsf T}y_0=0$, forcing $\lambda_0=0$. The shifted
combination

$$
x=\sum_{r=1}^{d}\lambda_rA^{r-1}y_0
$$

then satisfies $Ax=0$ and supplies the locator coefficients.

Keeping every Krylov vector until this dependence appears would defeat the
space saving. The circuit instead orthogonalizes the same growing spaces
with a short recurrence, the symmetric Lanczos method. If
$w_j$ is orthogonal to the earlier Krylov space $\mathcal K_j$, then for
$i\le j-2$,

$$
Aw_i\in\mathcal K_{i+2}\subseteq\mathcal K_j.
$$

Symmetry now gives

$$
w_i^{\mathsf T}Aw_j
=
(Aw_i)^{\mathsf T}w_j
=
0.
$$

Thus $Aw_j$ already has no component along any basis vector earlier than
$w_{j-1}$. Removing only its components along $w_{j-1}$ and $w_j$ produces
$w_{j+1}$, so the next step needs the preceding and current vectors rather
than the complete history. As soon as it computes a dependence coefficient,
the circuit multiply-accumulates its live basis vector into $x$.

## Scheduled breakdown repairs (P1)

An ordinary one-vector projection removes the component of a candidate $v$
along $w$ using

$$
\alpha=
\frac{w^{\mathsf T}v}{w^{\mathsf T}w}.
$$

Over $\mathrm{GF}(64)$, a nonzero vector can satisfy
$w^{\mathsf T}w=0$. The vector is then self-orthogonal, and the division that
defines $\alpha$ is impossible. This is a Lanczos breakdown rather than a
zero vector.

The standard repair groups consecutive Krylov vectors until their joint Gram
matrix is invertible. For

$$
V_d=[\,w\mid Aw\mid\cdots\mid A^{d-1}w\,],
\qquad
G_d=V_d^{\mathsf T}V_d,
$$

the circuit can project the block when $G_d$ is nonsingular. This grouped
update is block Lanczos. Here a **window** is an
interval of the walk. A self-orthogonal vector opens it, and it closes when a
supported depth-two or depth-three Gram block becomes invertible.

A lane means one computational-basis branch of the reversible computation.
The circuit conditions the ordinary update on

$$
m_0=w^{\mathsf T}w\ne0.
$$

When $m_0=0$ on a lane, the update gates do not act. Its walk state therefore
freezes without a separate liveness flag. The state remains unchanged until a
scheduled repair slot groups the required vectors and advances them
together.

The schedule offers depth-two repairs every third walk round and depth-three
repairs every sixth. Four extension rounds follow the 48 ordinary walk steps, giving
52 slots in which late windows may close. A window deeper than three, or a
lane that still has not reached its terminal state after the extension
rounds, abstains.

Concentrating the full Gram solve and grouped projection in 17 of the 52
slots keeps the other slots at the $5{,}400$-Toffoli ordinary rate. A
depth-two service slot costs $11{,}964$ Toffolis, and a depth-three slot costs
$16{,}557$. Slot 0 costs $5{,}364$ because the fixed seed cannot open a
window there, so breakdown-trigger machinery is unnecessary.

Windows deeper than three cause the scheduling abstains; they are rare
enough to stay inside the failure allowance that the evaluator checks.

## Terminal-case tests in released storage (mux, unmux)

The walk can finish in several supported terminal forms, depending on which
Krylov relation closed and which derived vectors are zero. Before the phase
sweep, the circuit classifies that form and routes the corresponding local
recovery formula. If no check certifies a supported form, the lane abstains
instead of risking a wrong phase.

A locator-side vector contains 48 field symbols, or 288 bits. Exact zero
tests use flat 288-leaf AND trees, costing 287 Toffolis each. They run inside
the temporary repair workspace after that workspace has become free, so the
classification does not require a resident test register.

One terminal condition would otherwise require an additional exact test of a
288-bit derived vector. The circuit keeps only a fixed one-sided 48-bit
linear sketch. A zero vector always produces a zero sketch, so the test
cannot reject a genuine zero. A nonzero vector can alias to zero with
probability $2^{-48}$ per lane.

One terminal form needs an already-derived operator image during the sweep.
That image remains in the released repair workspace; the circuit does not
copy it into a separate bank. After the tail applies the phase, the
unmux phase reverses the classification, clears the retained image, and
returns the same workspace to zero.

## Stripe-local phase recovery (tail)

The locator identifies possible error positions but does not by itself give
the error values needed by the mask. Constructing the entire decoded word
would require 3,072 live qubits before its eventual uncomputation. The tail
instead derives only the local locator and evaluator data needed at the
current curve points.

For each stripe, fixed linear maps evaluate the locator and the
syndrome-derived evaluator input at 206 interpolation points. The circuit
performs the 206 pointwise field products and applies another fixed linear
map that extracts only the local coefficients required by that stripe. The
surrounding maps cost only CNOTs. The circuit fills, uses, and releases the
product workspace stripe by stripe inside storage that the walk vacated.

At each sweep point, the local coefficients determine whether the locator
vanishes there and, if so, the associated error contribution to the phase.
The circuit implements the field norm, subfield inverse, and bilinear mask
pairing as fixed quadratic and bilinear forms over the subfield tower. Together they
cost 33 fixed-count ANDs per sweep point.

After it applies the phase contribution, the circuit clears the local values
before it processes the next point. No global evaluator and no decoded error word
ever become live.

## Measurement seam and reverse walk (xseam, P2)

After the tail, the circuit does not read the locator accumulator $x$ as
data, but $x$ remains entangled with the walk that produced it. The circuit
measures $x$ at the seam and turns the outcomes into phase corrections while retracing the
52 slots. This removes $x$ without retaining the forward Krylov history.

Some slots exchange the roles of live basis vectors. Each forward slot writes
its swap decision into one append-only journal bit. The reverse slot uses
that bit to undo the swap, then clears it once the reverse walk rebuilds the data that originally
derived the decision. Measuring the journal would not
work: the reverse needs the decision as a coherent control before those
source values are available again.

Other temporary decisions are cheaper to recreate than to store. The circuit
reruns exact zero tests while their source vectors are live, and it repeats
fixed linear gathers using CNOTs only. When a repair discards data by
measurement, the matching reverse slot recomputes and repairs that data.
Reversing a depth-three service requires one extra application of $A$ to
recreate its third Krylov vector before the circuit can erase the grouped
update.

Forward multiply-accumulates into $x$ become controlled phase corrections
after the seam measurement. Those corrections use CZ gates and therefore add
no Toffolis. As a result, the reverse walk uses $51{,}741$ fewer Toffolis than
the forward walk while still restoring the syndrome and clearing every
temporary register.