# uncompute.gadgets

This package holds the shared circuit gadgets: small reversible
subroutines from the literature that several circuits use, with one shared
copy each. It also holds the wire plumbing (`wires.py`) that prepares a
circuit's wires for the gadgets. Classical math lives in `uncompute.classical`,
and Bennett choreography lives in `uncompute.bennett`. This package only emits
ops.

## Conventions

- A gadget takes the live `Builder` as its first argument. It emits ops
  directly. It does not build op lists.
- Each gadget names its source paper and states its executed-Toffoli cost.
- Gadgets are library code like `builder.py`. A change to a gadget changes
  the stream of every circuit that uses it.
- Algorithm variants have their own names instead of mode flags.

## Gadgets

### qrom.py — table lookup

| Name | What it does | Cost | Source |
|---|---|---|---|
| `walk` | Unary iteration over an address window. Calls `body(ctrl, addr)` once per in-window address. | one Toffoli per two-sided tree node; `2^n - 2` for a full window | Babbush et al. (2018), arXiv:1805.03662, Figs. 5-7 |
| `selectswap_lookup` | Full SelectSwap (QROAM) lookup: select, swap tree, copy, measurement unlookup. `out=` sets the copy target. | `ceil(d/lam) - 2 + bw*(lam-1)` for `d > lam`, plus unlookup | Low et al. (2018), arXiv:1812.00954, Fig. 1(c); Berry et al. (2019), arXiv:1902.02134, App. B |
| `erase_register` | Measurement unlookup of a register that holds `word(addr)`. `free=` returns its qubits to the free-list. | `ceil(d/ku) - 2 + ku - 1` for `d > ku` | Berry et al. (2019), arXiv:1902.02134, App. C |
| `phase_fixup` | The App. C phase fixup: one-hot window, unary iteration, Clifford-only reverse routing. | included in the counts above | Berry et al. (2019), arXiv:1902.02134, App. C |
| `swap_perm` | Classical model of the swap tree: which word sits in each slot. | classical | — |
| `xor_outcomes` | Classical parity of a list of measurement outcomes. | classical | — |
| `berry_ku` | The App. C argmin window size `ku`. | classical | Berry et al. (2019), arXiv:1902.02134, App. C |

### arith.py — integer arithmetic

| Name | What it does | Cost | Source |
|---|---|---|---|
| `gidney_add` | `k += a mod 2^len(k)`. Accepts a shorter operand. Restores `a`. | `len(k) - 1` Toffoli | Gidney (2018), arXiv:1709.06648 |
| `gidney_sub` | `k -= a mod 2^len(k)` by complementation. | `len(k) - 1` Toffoli | Gidney (2018), arXiv:1709.06648 |
| `cdkm_add_co` | `tv += av` with the carry XORed onto `cout`. In place. No fresh ancillas. | 2 Toffoli per addend bit | Cuccaro et al. (2004), arXiv:quant-ph/0410184 |
| `mcx` | Multi-controlled X through a temporary-AND ladder. Controls carry polarities. | `n - 1` Toffoli for `n` controls | Gidney (2018), arXiv:1709.06648 |
| `ge_const_flag`, `ge_const_release` | Flag `[reg >= bound]` for a classical bound: the carry-out of `reg + (2^n - bound)` with the constant folded in. Returns `(recs, flag, pol)`; the release is measurement-based. | `n - 1 - ctz(2^n - bound)` Toffoli; release 0 | Gidney (2018), arXiv:1709.06648; Häner, Roetteler, and Svore (2018), arXiv:1805.12445 |
| `label_increment` | HRS label step `i`: `lbl ^= i ^ (i+1)` when `reg >= bound`. Over ascending bounds the label counts the bounds `<= reg`. | the comparator only | Häner, Roetteler, and Svore (2018), arXiv:1805.12445 |
| `hrs_mul_rows` | HRS pre-truncated signed multiplier `W += V * X * 2^-shift`: one `add(ctrl=)` row per `V` bit, the top row subtracts. Sets phase `{stage}/bNN` per row. | per row `len(T) - 1 + len(A)` | Häner, Roetteler, and Svore (2018), arXiv:1805.12445 |
| `and_chain`, `release_chain`, `eq_const` | AND chain over `(qubit, pol)` literals (the `pol=` form) that keeps its top wire live; released later from the trail. `eq_const` is the chain for `[wires == k]`. | `n - 1` Toffoli; release 0 | Gidney (2018), arXiv:1709.06648 |
| `and_ladder`, `and_ladder_release` | X-conjugates the open controls and reduces them to one flag; the body runs at the call site between the two calls. | `n - 1` Toffoli; release 0 | Gidney (2018), arXiv:1709.06648 |
| `booth_columns`, `booth_columns_release` | Booth radix-4 pair-select columns `(s1 AND va) XOR (s2 AND vb) XOR neg` and their measurement-based release. `pool=` as in `add`. | 1 per column + 1 per two-term column; release 0 | radix-4 recoding, Booth (1951), doi:10.1093/qjmam/4.2.236; temporary AND, Gidney (2018), arXiv:1709.06648 |

### linear.py — GF(2) linear layers

| Name | What it does | Cost | Source |
|---|---|---|---|
| `linear_in_place` | Applies an invertible GF(2) matrix to a register: `qs <- rows * qs`. Rows are packed ints. | 0 Toffoli; CX and SWAP only | textbook Gauss-Jordan |
| `linear_xor` | `dst ^= rows * src`, one CX per set entry, row-major order. Rows from `uncompute.classical.f2_rows_of`. | 0 Toffoli | textbook |

### gf_tower.py — GF(q0^2) tower fields

| Name | What it does | Cost | Source |
|---|---|---|---|
| `GF` | Field-parametric word gadgets for GF(q0^2), q0 in {4, 8}: `mul_xor`, `mbu_release_product`, `inv_xor`, `nonzero_flag`, `mw_xor`, `ctr_inc`, `ctr_dec`, and the linear helpers. | per method; see the class docstring | Karatsuba multiplication; temporary AND and measurement-based uncomputation, Gidney (2018), arXiv:1709.06648; inversion, Itoh and Tsujii (1988), doi:10.1016/0890-5401(88)90024-7 |

### wires.py — wire plumbing

| Name | What it does | Cost | Source |
|---|---|---|---|
| `declare_registers` | Allocates and declares the task's I/O registers in schema order; returns `dict name -> WireList`. | 0 ops (declarations only) | — |
| `WirePool` | Fronts a bank of caller-certified \|0> wires and duck-types `qubit`/`recycle`/`and_compute`/`and_release`, so any gadget's `pool=` trades certified spares against peak width. Falls through to the Builder when empty. | a pooled AND is 1 CCX + measurement release | — |

Neither is a circuit subroutine — they are how a circuit gets its wires into
shape for the gadgets that are.

Not here by design: classical table-building and reference models
(GF(2^n) and GF(2) linear maps) live in `uncompute.classical`; the Bennett-reversal
choreography classes (`Journal`, `Stages`) live in `uncompute.bennett`.
