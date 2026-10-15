# Parameter derivation: Nonabelian lattice-gauge simulation

This note fixes the parameters of this task: a complete
Symanzik-improved magnetic accumulator over a periodic $4^3$ lattice. The
loop structure and relative rectangle coefficient come from the improved
Hamiltonian used for $\Sigma(72\times3)\subset SU(3)$. The lattice size and
absolute integer scale are choices of this task.

## 1. Exact target map

The task implements

$$
|g_0\cdots g_{191}\rangle|k\rangle\mapsto
|g_0\cdots g_{191}\rangle
|k+a_PS_P+a_RS_R\bmod 2^{51}\rangle,
$$

where

$$
S_P=\sum_{192\ \mathrm{plaquettes}}f(P),\qquad
S_R=\sum_{384\ \mathrm{rectangles}}f(R),\qquad
f(g)=2\operatorname{Re}\operatorname{tr}(M_g).
$$

The circuit must preserve every link register and keep the modular
accumulator exact. The task allows no arithmetic tolerance and no failed
test.

## 2. Physical structure

At the tree-level improvement point,

$$
V_I=\beta_{V0}V_{KS}+\beta_{V1}V_{\mathrm{rect}},\qquad
\beta_{V0}=\frac53,\qquad \beta_{V1}=-\frac1{12},
$$

and the bent-loop coefficient vanishes. Hence the relative rectangle weight
is

$$
\frac{\beta_{V1}}{\beta_{V0}}=-\frac1{20}.
$$

All Wilson-loop phases are diagonal in the group basis, so the complete
magnetic layer can be represented by one accumulation of the weighted sum.
The kinetic layer acts in the Fourier basis and is outside this task.

## 3. Lattice and loops

The site index is $s=x+4y+16z$, with all coordinates periodic modulo four.
The link leaving site $s$ in direction $\mu\in\{0,1,2\}$ has index
$\ell=3s+\mu$. Thus the interface has 192 nine-qubit link registers followed
by the 51-qubit accumulator.

For every site and $i<j$, the plaquette is

$$
P_{ij}(\mathbf x)=U_i(\mathbf x)U_j(\mathbf x+\hat i)
U_i(\mathbf x+\hat j)^{-1}U_j(\mathbf x)^{-1}.
$$

For every ordered pair $i\ne j$, the rectangle is

$$
R_{ij}(\mathbf x)=U_i(\mathbf x)U_i(\mathbf x+\hat i)
U_j(\mathbf x+2\hat i)U_i(\mathbf x+\hat i+\hat j)^{-1}
U_i(\mathbf x+\hat j)^{-1}U_j(\mathbf x)^{-1}.
$$

This gives 192 plaquettes and 384 rectangles. Every link belongs to four
plaquettes and twelve rectangles.

## 4. Integer coefficients

The plaquette coefficient is the first 51 bits, MSB-first, of
SHA-256(`kickmix-plaquette-trace-accumulate-v2`), forced odd:

$$
a_P=\mathtt{0x8EBA84840FB}=9{,}808{,}233{,}644{,}283.
$$

The tag is a fixed string; its text carries no meaning.

The rectangle coefficient preserves the physical ratio at the chosen
fixed-point precision:

$$
a_R=-\operatorname{round}(a_P/20)\bmod2^{51}
=\mathtt{0x7FF8DD12F965A}.
$$

Because $a_P\bmod20=3$, nearest-integer rounding is unambiguous. These
integers define an exact modular map, and the circuit may make no further
approximation.

## 5. Group representation

Each link uses the nine-bit normal-form encoding of Osorio Perez et al.
(2025), arXiv:2511.17437. Of the 512 bit
patterns, 216 encode elements of
$\Sigma(72\times3)$; invalid patterns are outside the promise. Exact
$\mathbb Z[\omega][1/3]$ arithmetic determines the multiplication, inverse,
and trace tables. The trace function takes the seven values
$\{-3,-2,-1,0,1,2,6\}$.

## 6. Testing contract

Each of the $10^5$ tests draws all links independently and uniformly from the
216 valid codes and draws the initial accumulator uniformly modulo $2^{51}$.
A test passes exactly when the circuit restores every link and

$$
(k_{\mathrm{out}}-k_{\mathrm{in}}-a_PS_P-a_RS_R)\bmod2^{51}=0.
$$

The randomized initial accumulator exercises modular wraparound. Every
test also requires clean ancillas and the exact phase.

## 7. Pinned conventions

- The circuit declares the registers in link-index order, then the
  accumulator.
- The reference enumerates plaquettes by ascending site, then by
  $(i,j)\in\{(0,1),(0,2),(1,2)\}$.
- The reference enumerates rectangles by ascending site, then by ordered
  $i\ne j$ in lexicographic order.
- All loop products use inverses on their return links.
- The accumulator width is $m=51$, and `eps = 0`. The width follows from
  the $10^{-8}$ precision target and resource assumptions of Osorio Perez
  et al. (2025), with the phase-error accounting of Kan and Nam (2021).

## 8. Sources

- Carena et al. (2022), arXiv:2203.02823. The improved Hamiltonian, the
  rectangle term, and the tree-level couplings.
- Osorio Perez et al. (2025), arXiv:2511.17437. The $\Sigma(72\times3)$
  physical setting and the primitive group circuits.
- Lamm, Lawrence, and Yamauchi (2019), arXiv:1903.08807. The
  group-register formulation of lattice gauge theory.
- Kan and Nam (2021), arXiv:2107.12769. The phase-error accounting behind
  the accumulator width.
