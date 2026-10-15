# divsteps — algebraic branch-decoding divstep inversion

This folder holds the discovered circuit for the `modinv-point-add` task.
The accompanying paper (arXiv: TBD) describes it in the supplementary section
"Prime-field modular inversion", subsection "Divsteps"; this README maps the
construction to the code. `../../results.jsonl` holds the certified cost.

Algebraic branch-decoding divstep inversion replaces quotient-driven Euclid
with regular binary updates controlled by parity and a small signed counter.
The circuit deliberately runs 533 steps against a proved 590-step worst-case
bound and charges the omitted tail to the failure budget. The resulting point
adder uses about one fifth as many Toffolis as the specialized
register-sharing Euclidean construction.

Reversal is the central space problem. Parity and sign reveal about two
thirds of the branch decisions with negligible additional Toffolis. Storing
the remaining decisions would require 198 qubits per walk. Algebraic
decoding recovers most of those decisions from the surviving arithmetic
state. Fresh random walks use 21.2 stored bits on average. The circuit reserves a
60-bit stack across four walk legs: forward and reverse for each of the two
inversions.

```text
point_add:
    declare (x₁, y₁) and allocate the divstep workspace              ▷ init
    x₁ ← x₁ − x₂; y₁ ← y₁ − y₂                                       ▷ s1-const-subs
    run the forward division walk and store deep branches            ▷ s2-walkA
    multiply y₁ by the scaled inverse left by the walk               ▷ s2-seam
    reverse the walk while removing the inverse's scale              ▷ s2-walkB
    repair the measurement phase and move the quotient to y₁         ▷ s2-recompute
    x₁ ← x₁ + 3x₂                                                    ▷ s4-sub-squ
    x₁ ← x₁ − (1 − B)λ₀²                                             ▷ squ-a-lo
    x₁ ← x₁ − (F − B)λ₁²                                             ▷ squ-b-hi
    x₁ ← x₁ − B(λ₀ + λ₁)²                                            ▷ squ-c-sum
    A ← x₁ · y₁                                                      ▷ s6-mul1
    run the forward multiplication walk                              ▷ s6-walkC
    recover y₁ and repair the measurement phase                      ▷ s6-seam
    reverse the walk and restore A                                   ▷ s6-walkD
    x₁ ← −x₁ + x₂; y₁ ← y₁ − y₂                                      ▷ s7-final
    release the zeroed workspace                                     ▷ close
```

Here $(x_1,y_1)$ is the quantum point, $(x_2,y_2)$ is the classical addend,
and $A$ is a temporary product accumulator. The slope $\lambda=y_1$ splits
into 128-bit halves $\lambda=\lambda_0+B\lambda_1$ with $B=2^{128}$, and
$F=2^{32}+977$.

Most remaining work lies in the four walks and their branch reconstruction;
the gate schedule is deterministic. Conservative arithmetic bounds combined
with rates measured on fresh random walks give a union bound of
$1.599\times10^{-4}$, below the allowed $2\times10^{-4}$. The evaluator
checks the complete circuit independently, and `../../results.jsonl` holds
its certified verdict.

## Circuit files and cost

Each circuit file builds on `engine.py` at different settings. The table lists
the executed Toffoli count and the peak qubit count of each.

| file | Toffoli | qubits | setting |
|---|---|---|---|
| `circuit.py` | 4,006,673 | 935 | measurement erase, qubit ceiling 935 |
| `fill.py` | 3,637,655 | 1,031 | measurement erase, ceiling 1,031 |
| `wide.py` | 3,293,278 | 1,191 | subtraction erase, ceiling 1,191 |
| `j5.py` | 4,317,926 | 927 | staged decode, history threshold $j=5$ |
| `j6.py` | 4,427,338 | 913 | staged decode, $j=6$ |
| `j7.py` | 4,649,514 | 904 | staged decode, $j=7$ |
| `j8.py` | 5,331,729 | 897 | staged decode, $j=8$ |
| `j9.py` | 7,866,779 | 893 | staged decode, $j=9$ |

`circuit.py` and `j5.py` both store branch bits from depth $j=5$. They differ
in the engine. `circuit.py` runs the measurement-erase engine (`HmrWalk`): its
inverse-scale rider uses a windowed fold class, and its qubit ceiling is 935.
`j5.py` runs the staged-decode engine (`StagedWalk`): its rider uses an exact
comparator stage, and its qubit ceiling is 927.

`gen_constants.py` writes `constants.json`. It derives these dials from
censuses of random walks and from trial builds:

- the schedule length $S=533$ and the untested prefix $T_{\min}=498$, from
  the stopping-time census and per-walk failure allotments;
- the width $W_z=6$ of the counter $z$, from the counter-width census;
- the qubit ceiling of `circuit.py`, `wide.py`, and `j5.py`, `j6.py`,
  `j8.py`, `j9.py`: the lowest ceiling each build attains.

The engine derives the ceiling of `fill.py` in code: the multiplier base plus
one 256-wire block.

These dials are hand-set: the history-stack cap $C_r=60$, the start of the
stack-cap grid for `j6.py` to `j9.py`, the ceiling 904 of `j7.py`, and the
fold and comparison windows of the multipliers. A full run of
`gen_constants.py` takes about 30 minutes on 8 cores for the censuses, plus
about 60 trial builds for the ceilings.

## Why point addition becomes two GCD walks (s1-const-subs, s2-walkA, s6-walkC)

For distinct affine points, the chord slope is

$$
\lambda=(y_2-y_1)(x_2-x_1)^{-1}\pmod p.
$$

Additions, subtractions, and multiplications have regular bit-level circuits.
Inversion is harder because the required arithmetic path depends on the
input. It dominates the affine construction even though the group law
contains only one division.

Let $w=x_2-x_1$. Since $w\ne0$ and $p$ is prime, $\gcd(p,w)=1$. Bézout's
identity therefore supplies integers $\alpha$ and $\beta$ such that

$$
1=\alpha p+\beta w.
$$

Reducing this equation modulo $p$ removes the first term:

$$
\beta w\equiv1\pmod p.
$$

Thus $\beta$ is the desired inverse. An extended-GCD walk finds it by
reducing $(p,w)$ while updating coefficients that express the current values
as combinations of the original pair.

The mathematical formula needs one inverse. The reversible circuit invokes
the inversion machinery twice. The first invocation forms and consumes the
slope. The second removes the slope-dependent state after the circuit has
overwritten the coordinates. Each invocation has a forward and a reverse leg, giving the
four walks named in the overview.

## Fixed-form divsteps and truncation (s2-walkA, s2-walkB, s6-walkC, s6-walkD)

Quotient Euclid repeatedly divides one remainder by the other. It converges
quickly, but each quotient may have a different width and value. A reversible
implementation must select and apply those variable updates coherently across
the active register widths.

A binary-GCD step avoids multi-bit division. It makes one operand even by
adding or subtracting the other when necessary, then halves the even result.
The regular form used here, called a divstep, tracks two signed integers
$(f,g)$ and a small signed counter called the half delta. The parity of $g$
and the sign of the counter determine whether the step swaps the integers,
adds or subtracts $f$ from $g$, and halves the result.

This control rule matters because the shape of a divstep does not depend on a
multi-bit quotient. Every iteration can use the same fixed sequence of
controlled arithmetic layers. Only the control values change with the data.
Companion coefficient rows undergo matching linear updates so that the final
state still reveals the Bézout coefficient needed for inversion.

Following Wuille (2021), the Lean 4 development in `lean/divsteps/` proves
that 590 divsteps suffice for every nonzero field element. This construction schedules only 533. That
choice is a deliberate approximation: inputs needing the omitted tail may
fail. The construction charges their contribution explicitly to the common
failure budget rather than claiming an exact worst-case guarantee.

## Reconstructing branch bits (s2-walkA, s2-walkB, s6-walkC, s6-walkD)

Reversing an arithmetic update requires knowing which forward branch produced
the current state. A divstep may swap the integer rows, take an odd nonswap
branch, or merely halve an even value. The reduced integers alone do not
always identify that choice.

The walk carries coefficient rows $(f_v,g_v)$ alongside $(f,g)$ and preserves
the determinant relation

$$
f\,g_v-g\,f_v=p.
$$

This state contains more history than is immediately apparent. Row parity
reveals swap steps. The half delta reveals branches in the corresponding sign
region, and the step that first reaches $g=0$ has a forced predecessor.
Together, these inexpensive tests recover about two thirds of all decisions.

The ambiguous cases form runs of nonswap steps after a swap. During such a
run, $f$ remains fixed, $f_v$ repeatedly doubles, and an odd branch adds the
previous $f_v$ into $g_v$. Each decision is therefore deposited at a
successively higher binary position in the surviving coefficient state. If
$j$ is the number of trailing zeros in $f_v$ and $f_v=2^j q$ with $q$ odd,
removing the common odd factor exposes the current decision:

$$
b=\operatorname{bit}_{j-1}
  \left(g_vq^{-1}\bmod 2^j\right).
$$

The decoder implements the shallow instances of this identity. Deeper cases
are cheaper to remember than to reconstruct. Without decoding, the remaining
history would require 198 qubits per walk. Decoding reduces the measured
average to 21.2 stored bits, while a 60-bit stack covers the selected
operating range.

## Sharing shrinking and growing rows (s2-walkA, s2-walkB, s6-walkC, s6-walkD)

The integer rows $f$ and $g$ begin wide and lose significant bits as the GCD
walk progresses. Their coefficient rows $f_v$ and $g_v$ behave oppositely:
they begin small and grow as more linear updates accumulate. Giving every row
a permanent full-width register would reserve space for values that cannot
all be large at the same time.

The circuit fixes each row's width and position separately at every divstep.
A shrinking integer and a growing coefficient occupy opposite portions of a
shared word. When the integer releases a position, its paired coefficient may
claim that position in a later step. The circuit fixes this schedule before execution,
so the quantum data controls arithmetic within a predetermined layout rather
than dynamically resizing registers.

Temporary pressure can still occur when a row outgrows its assigned region.
The construction borrows workspace only for that step and moves the excess
bits into positions that have become vacant before continuing. This turns
short-lived deficits from different stages into one reusable reserve instead
of separate permanent allocations.

## Removing the inverse scale (s2-seam, s2-recompute, s6-mul1, s6-seam)

Divsteps introduce a power-of-two scale. If the walk performs $t$ active
steps before reaching $g=0$, its surviving coefficient satisfies

$$
f_vw\equiv2^t f\pmod p,
$$

where the remaining integer is $f=\pm1$. Multiplying by $f$ removes its sign,
because $f^2=1$, and gives

$$
f f_v\equiv2^t w^{-1}\pmod p.
$$

The circuit multiplies the numerator by this scaled inverse. During reversal,
the circuit pairs each active divstep with one modular halving of the result. After
all active steps have been undone, those halvings have removed the complete
factor $2^t$. The circuit therefore obtains the quotient without storing $t$
or applying a variable shift.

The first forward-and-reverse walk pair divides the coordinate difference
$y_1-y_2$ by $x_1-x_2$ to obtain the chord slope. Once that slope has updated
the coordinates, a second walk pair reconstructs the inverse-dependent
relation needed to erase it while forming the final $y$ coordinate.

X-basis measurement removes an obsolete numerator. Its outcome may leave a
data-dependent phase. The circuit therefore recomputes the measured Boolean
value from the surviving arithmetic state, applies the conditional phase
correction, and uncomputes the correction workspace. This avoids retaining a
second copy of the numerator through the surrounding coordinate arithmetic.

## Fusing the three half-squares (s4-sub-squ, squ-a-lo, squ-b-hi, squ-c-sum)

The $x$ update needs $\lambda^2$ only once, to subtract it from $x_1$. The
circuit never forms the complete square. With $s=\lambda_0+\lambda_1$, the
identity $2\lambda_0\lambda_1=s^2-\lambda_0^2-\lambda_1^2$ and
$B^2=2^{256}\equiv F\pmod p$ give

$$
\lambda^2\equiv(1-B)\lambda_0^2+(F-B)\lambda_1^2+B\,s^2\pmod p.
$$

Each of the three phases squares one value $v$ of $n=128$ or $129$ bits into
a temporary register of $2n$ qubits. The phase subtracts the product from
$x_1$ with its coefficient and clears the register before the next phase
starts. The sum $s$ forms in place on the $\lambda_1$ qubits plus one extra
qubit, and the circuit removes $\lambda_0$ from it afterwards.

The square uses $n-1$ row additions. Row $i$ adds

$$
2^{m}+(2v_i-1)R_i+v_i,\qquad R_i=v\gg(i+1),\quad m=n-1-i,
$$

at bit offset $2i+1$. The circuit complements the $R_i$ qubits when
$v_i=0$, so the sign costs no controlled adder: a row costs $m+1$
Toffolis. The register starts at the bitwise complement of $v$. After the
last row, one bit flip completes the register to $v^2-1$, and every
subtraction from $x_1$ adds the missing 1 through its carry input. Each
row's partial sum stays inside its window, so the row uses the register's
zero high qubits as its carry qubits.

Subtracting a product slices it at bit 256. Bits above bit 255 return at
the signed digits of $F=2^{32}+2^{10}-2^{6}+2^{4}+1$. A slice that ends
well below bit 256 is added without modular reduction. A slice that
reaches bit 255 is added with a carry bit. When the carry bit is set, the
circuit adds $F$ at the low end, because $2^{256}\equiv F$. A measurement
then erases the carry bit, and a short comparison of the top bits corrects
the measurement phase.

At the 935-qubit point the three phases use 56,380 Toffolis, of which the
rows use 49,788. The adders take carry qubits only up to the point's qubit
ceiling, so the square never raises the peak.

## Sparse-prime arithmetic (s1-const-subs, s2-seam, s2-recompute, s4-sub-squ, s6-mul1, s6-seam, s7-final)

The prime $p$ has the sparse form

$$
p=2^{256}-2^{32}-977.
$$

Consequently,

$$
2^{256}\equiv2^{32}+977\pmod p.
$$

A bit or product limb above the field boundary need not pass through a
general modular division. A shifted copy at the low end together with a short
constant multiple can replace its weight. Repeating this fold
brings the contribution back into the field-width destination, after which a
canonical correction completes the reduction.

The construction applies this identity while it generates product terms. It
folds each high contribution directly into the coordinate or quotient
register that needs it. The alternative materializes a double-width product,
reduces it, copies the residue, and reverses the product. Compared with that,
this streaming form removes the long-lived product register and its
duplicate uncomputation. The sparse-prime rewrite itself is exact; only the separately
bounded carry and comparison limits contribute failure probability.

## Bounding deliberate approximations (s2-walkA, s2-walkB, squ-a-lo, squ-b-hi, squ-c-sum, s6-walkC, s6-walkD)

The algebraic identities used for branch decoding, inverse scaling,
half-square fusion, and sparse-prime reduction are exact. Approximation enters
only through finite operating limits. These are the 533-step walk schedule,
bounded row widths, the 60-bit history stack, and bounded carry and comparison
work in the modular arithmetic.

In the fused square, each of the 44 product slices stops its carry
propagation, its $F$ correction, or its comparison $t$ bits past the
operands, and fails with probability at most $2\cdot2^{-t}$. The circuit
derives $t=28$ from the $6\times10^{-7}$ that the failure budget allots to
the square. A zero half of $\lambda$ also fails, with probability about
$2^{-127}$.

Conservative arithmetic arguments bound the carry and comparison events.
Rates measured on fresh random walks bound the truncation, row-width, and
history events. Adding these contributions gives the
$1.599\times10^{-4}$ union bound. A union bound does not assume that the
events are independent; it remains valid even when one input violates several
limits at once.

The full evaluator checks exact coordinates, phase, and workspace cleanup
under the task distribution.