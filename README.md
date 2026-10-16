# uncompute

This repository holds the circuits, task definitions, and evaluator for the paper
[*Discovering state-of-the-art quantum circuits with large language models*](https://arxiv.org/abs/2610.22367)
(Gu, Kemeny, Yelin, and Lukin, 2026, arXiv:2610.22367).

Each task fixes one subroutine of a fault-tolerant quantum algorithm. For each task, the
repository holds the circuit that an LLM-guided search discovered and, where one exists, a
baseline circuit that reconstructs a published construction. The evaluator counts the cost
of every circuit exactly and tests the circuit on random inputs.

## Results

A circuit costs its Toffoli count *T* and its peak number of live qubits *Q*.
The TQ-cost is the product *T × Q*. The gain *G*<sub>TQ</sub> is the baseline TQ-cost
divided by the discovered TQ-cost. The table repeats Table I of the paper, with counts
rounded as in the paper. Each task's `results.jsonl` holds the exact certified counts.

| Task | Paper section | Baseline (*T*, *Q*) | Discovered (*T*, *Q*) | *G*<sub>TQ</sub> |
|---|---|---|---|---|
| [`dqi-reed-solomon`](tasks/dqi-reed-solomon/) | SI I.1 | 5.1×10⁶, 1,938 | 4.4×10⁵, 1,837 | 12 |
| [`dqi-hermitian`](tasks/dqi-hermitian/) | SI I.2 | 1.0×10⁸, 1.9×10⁵ | 9.3×10⁵, 2,395 | — |
| [`hhl-angle`](tasks/hhl-angle/) | SI III.1 | 1.6×10⁴, 300 | 1,348, 115 | 32 |
| [`coulomb-potential`](tasks/coulomb-potential/) | SI III.2 | 6,355, 252 | 1,140, 142 | 9.9 |
| [`gaussian`](tasks/gaussian/) | SI III.3 | 2.4×10⁴, 362 | 1,583, 139 | 39 |
| [`chemistry-fe2s2`](tasks/chemistry-fe2s2/) | SI IV | 5.5×10⁴, 139 | 1.5×10⁴, 126 | 4.1 |
| [`lattice-gauge-plaquette`](tasks/lattice-gauge-plaquette/) | SI V | 1.3×10⁶, 1,927 | 1.7×10⁵, 1,859 | 7.7 |
| [`modinv-point-add`](tasks/modinv-point-add/) | SI II | — | 4.0×10⁶, 935 | — |

## Circuit model

The circuits use the kickmix circuit model of Babbush et al. (2026), arXiv:2603.28846.
The model allows classical reversible gates, X-basis measurement-based uncomputation, and
diagonal phase gates. Each shot of such a circuit stays one classical basis state with one
sign, so the evaluator simulates every shot exactly.

The evaluator accepts a circuit only if every test shot meets four conditions:

- The output registers match the task's reference function.
- The global sign equals the expected sign.
- Every qubit outside a declared register is zero at the end of the circuit.
- The qubit operands of each gate are distinct.

A task can allow a small failure budget. The circuit may then approximate the function, and
the grader states the budget. The test inputs come from a hash of the emitted circuit, so
any change to the circuit draws new tests.

## Layout

```text
tasks/<task>/          one task
  README.md            the task, its baseline, and its discovered circuit
  grader.py            the test distribution, the output check, and the failure budget
  reference.py         the reference function in plain Python
  contract.json        the constants that define the task instance
  results.jsonl        the certified Toffoli and qubit counts of each circuit
  circuits/<name>/     one circuit (a discovered circuit or a baseline)
  circuits/_shared/    helper code that several circuits of the task import
uncompute/             the Builder, the op-stream serializer, and the simulator
uncompute/evalcore/    the evaluator
tools/                 evaluation, certification, and qubit-occupancy tools
lean/                  Lean 4 proofs of two results that the circuits rely on
```

A task folder can also hold helper modules of its reference (for example
`alias_sched.py` or `instance.py`), input data under `data/`, or a note on
how the task parameters follow from the physics (`PARAMS-DERIVATION.md`).
The `uncompute/ops.py` module docstring specifies the op-stream format.

Every circuit folder holds these files:

- `circuit.py` builds the circuit with `uncompute.Builder` and writes its op stream.
- `circuit.ops.xz` is the certified op stream.
- `README.md` describes the construction.
- `meta.json` holds the expected cost and the labels that the profiling tools
  use.

A circuit folder can also hold these files:

- `gen_constants.py` is the search or derivation that produces the circuit's
  tuned constants. It takes no arguments. It reruns the search and writes
  `constants.json`. A circuit without this file computes its constants in
  `circuit.py`.
- `constants.json` holds only the values that `circuit.py` reads.
- `checks.py` checks a worst-case property of the circuit, such as the error
  bound of its tables, on every input. `circuit.py` does not import it.
- C sources (for example `search.c`) run the inner loops of `gen_constants.py`.
- More circuit files, each with its own `.ops.xz`, give other points of a
  cost trade-off. `meta.json` lists them.
- `liveness.json` (or `liveness.json.xz`) and `profile.pdf` record which
  qubits are live at each step of the circuit.
- `occupancy.json`, `occupancy.pdf`, and `occupancy.png` show how named
  registers share the workspace over the circuit.

The `modinv-point-add` task has two discovered circuits: `circuits/divsteps/`
(the default that `tools/eval.py` runs) and `circuits/euclidean/`.

Baselines that transcribe released code say so in their README, with the
attribution that the source license requires. Every baseline README ends with
an Attribution section.

## Checking a result

Install the environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
make kernel          # optional C kernel; the Python evaluator gives identical results
```

Run a circuit, test it, and print its cost:

```bash
uv run python tools/eval.py coulomb-potential
```

Check a certified op stream without running the circuit:

```bash
uv run python tools/eval.py coulomb-potential \
    --ops tasks/coulomb-potential/circuits/discovered/circuit.ops.xz
```

The `--circuit` option selects another circuit of the task. Large circuits take minutes to
hours on one machine. The `--jobs` option splits the test shots across worker processes.

## Liveness and occupancy profiles

Regenerate the liveness profile of a discovered circuit, or of another circuit
folder, and render it:

```bash
uv run python tools/liveness_profile.py tasks/coulomb-potential
uv run python tools/liveness_profile.py tasks/coulomb-potential --profile-dir circuits/haner-piecewise
uv run python tools/render_liveness.py tasks/coulomb-potential/circuits/discovered/liveness.json
```

The renderer writes `profile.pdf` beside the JSON file. The styled render needs
a LaTeX installation. `tools/occupancy_profile.py` draws the occupancy charts;
its docstring lists its options.

## Lean proofs

- [`lean/divsteps`](lean/divsteps/) proves the step bound and the branch-recovery rule
  that the `modinv-point-add` divstep circuit relies on.
- [`lean/dqi`](lean/dqi/) proves the decoder-error bound that the DQI tasks rely on.

## Citation

```bibtex
@misc{gu2026discovering,
  title         = {Discovering state-of-the-art quantum circuits with large language models},
  author        = {Gu, Andi and Kemeny, Alek H. and Yelin, Susanne F. and Lukin, Mikhail D.},
  year          = {2026},
  eprint        = {2610.22367},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## License

The code is under the [Apache License 2.0](LICENSE).
