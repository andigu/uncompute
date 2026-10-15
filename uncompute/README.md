# uncompute (Python circuit library)

The `uncompute` package builds a reversible circuit in Python and serializes it as an op
stream. The `Builder` writes the stream to the path in `UNCOMPUTE_OPS_PATH` when that variable
is set, else to `./ops.bin`. The evaluator (`tools/eval.py`) reads the stream, runs the
randomized test, and reports the cost. The evaluator has a pure-Python reference engine and
an optional C kernel that gives identical results.

**The op stream is the interface between a circuit program and the evaluator.** The
evaluator runs the circuit program as a subprocess and reads only the stream. Any program
that writes a valid stream therefore gets the same result.

## Install

From the repository root:

```bash
uv sync
```

Circuit programs also run without the install. The evaluator (`tools/eval.py`) puts the
repository root on the `PYTHONPATH` of the circuit subprocess. The package sits at the
root, so `import uncompute` resolves in both cases.

## Quickstart

```python
from uncompute import Builder

b = Builder()
x = b.qubits(8)
b.qubit_register(x)  # register 0

t = b.qubit()  # scratch ancilla
b.ccx(x[1], x[2], t)  # t = x1 & x2
b.ccx(x[3], t, x[0])  # x0 ^= x3 & t
b.ccx(x[1], x[2], t)  # uncompute t -> |0>
b.recycle(t)  # return t to the LIFO free-list

print("peak qubits:", b.peak_qubits)
b.emit()
```

## API

The `Builder` counts peak qubits with the same rule as the evaluator.

- **Allocation** — `qubit()`, `qubits(n)`, `recycle(q)`, `recycle_all(qs)`. `recycle`
  resets `q` and returns its id to a **LIFO free-list**. The next `qubit()` reuses that id.
  `peak_qubits` is the maximum number of qubits live at the same time. This number is the
  qubit factor of the TQ-cost and equals the `qubits` value that the evaluator reports.
  `reclaim(ids)` takes named ids back from the free-list.
  `contiguous_qubits(n, mint=True)` returns a contiguous ascending run of ids. It takes the
  best fit from the free runs in the pool, else it mints new ids past the frontier
  (`mint=False` raises instead). Both calls change only the allocator state and emit no op.
  `release_recycle(name, ws)` releases the named wire-set `ws` and then recycles its wires in
  reverse order.
- **Registers** — `qubit_register(qs)` and `bit_register(bits)`. Declare the registers in
  the order of the task schema (register 0, 1, …). A Bit-kind register is a classical
  input. The evaluator writes the drawn value of each shot into it before the first gate.
  The evaluator never reads a Bit-kind register back.
- **Gates** — `x, z, r, cx, cz, swap, cswap, ccx, ccz, hmr, neg`. `ccx(c1, c2, tgt)` takes
  its two controls first. `cswap(c, a, b)` emits the Fredkin gate as the standard
  one-Toffoli expansion (`CX(b,a); CCX(c,a,b); CX(b,a)`).
- **Outcomes** — `m = b.hmr(q)` measures `q` in the X basis, resets it, and returns an opaque
  `Outcome` that holds the measured bit. `z`, `cz`, and `neg` take `cond=<Outcome>`: the
  sign applies only on shots where the outcome is 1. Only these three phase gates accept an
  outcome condition. A condition narrows the action of the gate, not its cost. `m1 ^= m2`
  accumulates in place (one OXOR op). `m1 ^ m2` creates a new accumulator (only outside
  `repeat` bodies).
- **Bits** — `c = b.bit()` and `b.bits(n)` return opaque `Bit` handles. A bit holds
  classical per-shot input data in its own address space and does not count as a qubit.
  The ops `bit_invert`, `bit_store0`, and `bit_store1` act on bits only and cost nothing.
  Each takes `when=<Bit>`, and `c1 ^= c2` is short for `bit_invert(c1, when=c2)`. These ops
  derive any classical value at zero cost. No quantum op and no outcome writes a bit.
  A Bit conditions `x`, `z`, `neg`, `cx`, and `cz` through `when=`. The gate then acts only
  on shots where the bit is 1. The evaluator **counts the gate at its promoted kind**:
  X→CX, CX→CCX (a Toffoli), Z→CZ, CZ→CCZ (a Toffoli), NEG→Z (free). `ccx`, `ccz`, and
  `swap` do not accept `when=`. The Toffoli count is therefore a constant of the stream.
- **Temporary AND** — `t = b.and_compute(c1, c2, pol=(True, True))` computes the AND with
  one Toffoli (Gidney 2018, arXiv:1709.06648). `b.and_release(t, c1, c2, pol=(True, True))`
  releases it by measurement with 0 Toffoli and recycles `t`. A False `pol` entry ANDs the
  negated control. The `pol` argument expands to an X sandwich of ordinary X gates. The
  stream therefore equals the hand-written spelling, and `mirror` needs no special case.
  For an ancilla that must stay allocated after its release, write the same bytes
  explicitly: `m = b.hmr(t); b.cz(c1, c2, cond=m); b.r(t)`. The circuit then releases the
  wire-set of the ancilla and calls `b.recycle(t, reset_recorded=True)`, which emits no
  further op. `mirror` cannot invert the explicit spelling (a bare `hmr` raises).
- **Uncompute (Bennett mirror)** — `m = b.checkpoint()` marks a position.
  `b.mirror(m, until=None, phase_prefix="un-")` emits the **exact Bennett inverse** of the
  region. Self-inverse gates (X/Z/CX/CZ/CCX/CCZ/SWAP/NEG) appear again in reverse order.
  `and_compute` becomes `and_release`, and the mirrored release costs 0 Toffoli. `qubit`
  becomes `recycle`. The LIFO free-list returns the same ids, and `mirror` asserts this.
  Any op in the region without an inverse (bare `hmr`, `r`, OXOR, registers, conditioned
  phases) raises; `mirror` skips nothing. `until=` builds the conjugation F·A·F⁻¹: it
  mirrors F after a stage A that stays in place. Mirrored spans get `un-*` phase labels.
- **Stages** (`uncompute.bennett`) — `st = Stages(b)`; `with st.stage(name, set_name=None,
  unwind=None): …` records a stage head (the allocations of the stage and its ownership
  `claim`s). Everything after the block is the stage body.
  `st.unwind(prefix="un-", until=None)` then unwinds the stages in reverse order. For each
  stage, it calls `set_phase(prefix + name)` and mirrors the body, which returns the wires
  of the stage to |0⟩. It then calls `release` on the named set and mirrors the head,
  which recycles the head allocations. `until=` stops the last body before a tail that
  stays in place (the F·A·F⁻¹ shape). `unwind=` supplies a hand-written inverse for an
  irregular stage that allocates and recycles inside itself. `Stages` only orders calls to
  the checkpoint and mirror primitives.
- **Phase labels** — `b.set_phase("reduction")` and `with b.phase("reduction"): …` tag the
  gates and allocations in a region. Labels are metadata and emit **no** op, so the stream
  is the same with or without labels. `emit()` also prints one `UNCOMPUTE-META {...}` line on
  stdout with `phase_summary()`. The line holds the emitted Toffoli count and the local
  qubit peak of each phase. It also holds `peak_phase`, the phase in which the qubit count
  reached its global peak. The line shows where the Toffoli count and the qubits go. The
  evaluator copies it into `report["circuit_meta"]`; the line does not affect the result.
- **Output** — `emit()` streams the ops to `UNCOMPUTE_OPS_PATH` or `./ops.bin`. Pass
  `meta=False` to suppress the `UNCOMPUTE-META` line.

Gate methods reject repeated qubit operands at once (the evaluator applies the same rule),
so an aliasing error appears when the circuit is built.

## Local simulation (`Simulator`)

`uncompute.Simulator` is a bit-parallel classical simulator with the gate semantics of the
evaluator. It packs **any number of lanes into one Python integer per wire**. A sweep over
a full input domain (or a large random sample) therefore costs a few integer operations per
gate:

```python
from uncompute import Builder, Simulator

b = Builder()
x = b.qubits(8)
b.qubit_register(x)
# ... build circuit ...

xs = list(range(256))  # chosen inputs: exhaustive here,
sim = Simulator(b, batch=len(xs), seed=0)  # sampled or worst-case for big domains
sim.write(x, xs)  # little-endian qubit list, one value per lane
sim.run()
outs = sim.read(x)  # -> list[int], one per lane
sim.assert_clean()  # the evaluator's phase and ancilla checks
print("avg executed Toffoli:", sim.avg_toffoli)
```

The simulator supports three checks before a full evaluation:

- **Correctness sweeps** — compare `read()` with a reference model on chosen inputs. For a
  task with an error budget, this measures the true worst case.
- **Measured cost** — `stats.toffoli` and `avg_toffoli` count gates per lane with the
  rules of the evaluator. Every op counts on every live lane at its promoted kind, because
  a condition narrows the action, not the cost. The average therefore equals the emitted
  count at promoted arity (X and Z are free). `floor(avg_toffoli) * b.peak_qubits` is a
  local estimate of the TQ-cost.
- **Validity check** — `assert_clean()` checks for zero global phase and for every
  non-register qubit at |0⟩ on every lane. Declared-register qubits are exempt, as in the
  evaluator. `outputs=...` adds exemptions for subcircuits without registers. `phase=mask`
  accepts an intended per-lane sign pattern. In **strict mode** (the default), a per-draw
  fixup ledger also proves that each `hmr` kick cancels for *both* outcome values. The
  proof is algebraic and takes one pass. Strict mode also raises at any `r` on a nonzero
  wire. A correct circuit gives the same RNG stream, state, and statistics with strict mode
  on or off. Strict mode finds a dirty circuit with certainty instead of with probability
  ~1 − 2⁻ᵇᵃᵗᶜʰ. `strict=False` keeps only the sampled checks.

Two parts of the evaluation stay with the evaluator: the seeded input distribution and the
task oracle. The simulator does not know the correct output; the caller supplies it.
`apply(ops)` is public for incremental use, for example to replay one subcircuit inside a
search loop. `read_outs` unpacks measurement outcomes per lane in the same way.

## What makes a circuit valid

The evaluator accepts a circuit only if **all** of these conditions hold on every tested
shot:

- **Output** — the output registers match the reference oracle of the task.
- **Sign** — the global sign equals the expected sign.
- **Ancilla** — every qubit outside a declared register is |0⟩ at the end of the circuit.
  A scratch qubit must also be |0⟩ when `recycle()` resets it, because a reset of a
  nonzero qubit randomizes the sign. The pattern is **compute → use → uncompute →
  `recycle`**. In the quickstart, the circuit computes the scratch `t`, uses it, applies
  the same Toffoli again to uncompute it, and then recycles it.
- **Distinct operands** — the qubit operands of a gate are distinct (the `Builder`
  checks this at once).

`recycle(q)` returns `q` to a LIFO free-list that the next `qubit()` reuses. Early recycling
therefore lowers `peak_qubits`, which equals the `qubits` value that the evaluator reports.
Some tasks allow a small fraction of wrong outputs. In those tasks, deliberate truncation
is a valid optimization, and the randomized test checks the error budget.

## Workflow

```bash
# Run the discovered circuit of a task and evaluate it
# (--out DIR keeps report.json and tq_cost.json):
uv run python tools/eval.py <task> --out <dir>
```

For use from Python, put `tools/` on `sys.path` and call `from eval import evaluate`.
`evaluate()` returns one flat result dict with keys such as `valid`, `tq_cost`, `toffoli`,
`qubits`, `clifford`, `fail_reason`, `phases`, `peak_phase`, and `report`.

For large shot counts, the `tools/eval.py` flags `--batch-shots` and `--jobs` split the
evaluation into shards.

## Stream format

The `uncompute/ops.py` module docstring specifies the op stream: its segments and its op records.
`uncompute/evalcore/compact_reader.py` is the reference parser and lists the validation rules.
