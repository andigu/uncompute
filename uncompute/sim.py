"""Bit-parallel classical simulator for uncompute circuits.

The evaluator runs on this wide-lane simulation core, and circuit code can
use it while building a circuit. An uncompute circuit holds one classical basis
string and one sign per shot, so bitwise ops simulate it exactly. Each wire
is an arbitrary-width Python int that packs ``batch`` shots ("lanes"). A
full input-domain sweep or a large random sample therefore costs a few
bigint ops per gate, whatever the batch size.

What this gives a circuit at build time:

* **Correctness sweeps** against your own reference model. You choose the lane
  population: exhaustive when the input domain is small, sampled or
  hand-picked otherwise.
* **Measured billed gate counts** (``stats``), matching the evaluator's cost
  model: every op bills every live lane at its promoted kind (a condition —
  outcome on Z/CZ/NEG, bit via ``when=`` — modulates the per-lane action,
  never the billing), and X/Z are free (classically tracked) exactly as in
  the evaluator. So ``stats.toffoli / batch`` is the evaluator's
  avg-executed-Toffoli measured locally — a constant of the emitted
  circuit — and combined with ``Builder.peak_qubits`` gives the TQ-cost
  locally.
* **The evaluator's validity gates, locally**: :meth:`Simulator.assert_clean`
  checks zero phase and every non-output qubit restored to |0> on every lane.
  Measurement (``hmr``) and reset (``r``) phase kicks are drawn from a seeded
  XOF just like the evaluator's, so a dirty uncomputation poisons the
  phase on ~half the lanes and is caught with probability ~1 - 2^-batch.
  **Strict mode** (default) upgrades that to a deterministic certificate: a
  per-draw fixup ledger proves each measurement kick is cancelled for both
  outcome values algebraically, and ``r`` on a nonzero wire raises at the
  offending op. For a clean circuit strict mode is invisible (identical RNG
  stream, state, and stats). ``strict=False`` keeps only the sampled
  checks — the semantics the evaluator's verdicts are defined over, so
  its executors turn strict mode off.

What it deliberately does NOT replicate: the seeded input distribution and
the task oracle (compare :meth:`Simulator.read` against your own reference
model — the simulator does not know what "correct" means), and the
task's error budget. Those remain the evaluator's job.

Typical use::

    from uncompute import Builder, Simulator

    b = Builder()
    x = b.qubits(14); b.qubit_register(x)
    # ... build circuit, keeping qubit lists for inputs/outputs ...

    xs = [...]                              # the inputs YOU chose to test
    sim = Simulator(b, batch=len(xs), seed=0)
    sim.write(x, xs)                        # little-endian, one value per lane
    sim.run()
    outs = sim.read(y)                      # -> list[int], one per lane
    sim.assert_clean()                      # phase + ancilla-cleanliness gates
    avg_tof = sim.stats.toffoli / sim.batch
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .compact import WKIND_QUBIT, Regs

# Billing follows the evaluator's cost model: every op bills every live lane
# at its billed kind (ops.gate_class prices a bit-conditioned op at its
# promoted kind), so the counts are constants of the circuit.
from .ops import CC_ALLOWED_KINDS as _CC_ALLOWED
from .ops import NO_BIT, NO_OUT, NO_QUBIT, OpKind, _popcount, op_violation
from .ops import gate_class as _gate_class

_CLASS_PLAIN = {int(k): _gate_class(k, False) for k in OpKind}
_CLASS_COND = {k: _gate_class(k, True) for k in _CC_ALLOWED}

# Gate families of the reference apply loop. A "toggle" kind XORs its
# target with the action mask ANDed with its control qubits; a "sign" kind
# XORs the global sign with the action mask ANDed with its factor qubits.
# Tuple positions: qa = 1, qb = 2, qt = 3.
_TOGGLE_CONTROLS = {int(OpKind.X): (), int(OpKind.CX): (2,), int(OpKind.CCX): (1, 2)}
_SIGN_FACTORS = {
    int(OpKind.NEG): (),
    int(OpKind.Z): (3,),
    int(OpKind.CZ): (2, 3),
    int(OpKind.CCZ): (1, 2, 3),
}
_K_SWAP = int(OpKind.SWAP)
_K_HMR = int(OpKind.HMR)
_K_R = int(OpKind.R)
_K_OXOR = int(OpKind.OXOR)
_K_BIT_INVERT = int(OpKind.BIT_INVERT)
_K_BIT_STORE0 = int(OpKind.BIT_STORE0)
_K_BIT_STORE1 = int(OpKind.BIT_STORE1)


_FAST = False  # cached _simfast module; False = untried, None = unavailable


def _fast_engine():
    """Return the C fast-path module, or None when it cannot be imported.
    Without it, the Simulator runs the pure-Python loop.
    """
    global _FAST
    if _FAST is False:
        try:
            from . import _simfast

            _FAST = _simfast
        except (ImportError, OSError):
            _FAST = None
    return _FAST


@dataclass
class SimStats:
    """Executed gate totals summed over lanes (divide by ``batch`` for the avg)."""

    toffoli: int = 0
    clifford: int = 0


class CleanlinessError(AssertionError):
    """Raised by assert_clean: leftover phase or a non-|0> non-output qubit."""


class Simulator:
    """Simulate a circuit over ``batch`` parallel input lanes.

    Billed-gate accounting matches the evaluator's cost model: every op
    bills every live lane at its promoted kind — a condition (outcome on
    the free phase gates Z/CZ/NEG, bit via ``when=``) modulates the
    per-lane action, never the billing — so ``stats.toffoli / batch``
    equals the evaluator's average executed Toffoli, a constant of the
    emitted circuit. X and Z are free and are not counted.

    Args:
        source: a :class:`~uncompute.Builder` (its ``.ops`` are run by
            :meth:`run`) or a list of op 9-tuples. May be ``None`` if you only
            use :meth:`apply` with explicit op lists.
        batch: number of parallel lanes (shots). Each qubit/outcome/phase is
            one Python int with lane j at bit j.
        seed: seeds the XOF driving ``hmr``/``r`` measurement outcomes.
            Deterministic: same seed + same circuit = same outcomes, so
            cleanliness failures reproduce exactly.
        strict: (default True) additionally certify measurement cleanliness
            *algebraically*, not just on the sampled outcomes: a per-draw
            fixup **ledger** proves each ``hmr`` kick is cancelled for BOTH
            outcome values (checked in :meth:`assert_clean`, or at the op
            when a still-dirty outcome id is overwritten), and ``r`` on a
            wire that is nonzero on any lane raises at the op (a reset kick
            has no conditioning channel, so it is never fixable). For a
            clean circuit strict mode changes nothing — the RNG stream,
            state, and stats are bit-identical — it only converts
            "caught with probability 1 - 2^-batch at the end" into "caught
            deterministically, at the offending op where possible".
            ``strict=False`` keeps only the sampled checks.
            Caveat: the ledger tracks outcomes written by ops; direct pokes
            to ``sim.outs`` bypass it.
    """

    def __init__(self, source=None, batch=64, seed=0, strict=True):
        if batch < 1:
            raise ValueError("batch must be >= 1")
        self._source = source
        self.batch = batch
        self.live_mask = (1 << batch) - 1
        self.phase = 0
        self.qubits = []  # one int per qubit id; grown on demand
        self.outs = []  # one int per outcome id (HMR writes, OXOR accumulates)
        self.bits = []  # one int per classical bit id (inputs + BIT_* writes)
        self.stats = SimStats()
        # The qubits of the source Builder's Qubit registers (run() adds
        # them). The evaluator exempts exactly these from its ancilla-garbage
        # scan (register qubits hold outputs and restored inputs);
        # assert_clean mirrors that.
        self.register_qubits = set()
        self._seed = b"uncompute-sim" + int(seed).to_bytes(8, "little")
        self._ctr = 0
        self._nbytes = (batch + 7) // 8
        # ---- strict-mode fixup ledger (per DRAW, not per outcome slot) --------
        # Each hmr draw owes the phase kick (-1)^(draw & value); conditioned
        # Z/CZ/NEG discharge the debt through the outcome slots that carry the
        # draw's bit (OXOR moves bits between slots, so slots are GF(2) sums
        # of draws — the slot->draw-set map keeps the algebra exact). A draw
        # with residual 0 is cancelled for either outcome value, algebraically.
        # No branch explosion: outcomes condition only Z/CZ/NEG, so
        # wire values are outcome-INDEPENDENT and the total phase is affine in
        # the draws — certifying all 2^N outcome branches is exactly checking
        # the N per-draw linear coefficients (this ledger) plus the constant
        # term (the sampled phase that assert_clean already checks). One pass,
        # one extra lanes-int per live draw.
        # Invariant: d in any slot set  <=>  _draw_refs[d] >= 1  <=>  d in ledger.
        self.strict = strict
        self.ledger = {}  # draw id -> unresolved phase-residual lanes mask
        self._slot_draws = {}  # outcome id -> set of draw ids its bits carry
        self._draw_refs = {}  # draw id -> number of slots carrying it
        self._draw_src = {}  # draw id -> (op index, outcome id), for messages
        self._op_n = 0  # ops applied so far (cumulative across apply calls)

    # ---- lane RNG (seeded XOF, like the evaluator's measurement RNG) -----------
    def _draw(self):
        """One fresh random bit per lane, from the seeded SHAKE-256 stream."""
        self._ctr += 1
        h = hashlib.shake_256(self._seed + self._ctr.to_bytes(8, "little"))
        return int.from_bytes(h.digest(self._nbytes), "little") & self.live_mask

    def _retire_slot(self, ot):
        """Strict mode: drop outcome slot ``ot``'s draw set (its bits are about
        to be overwritten by a fresh HMR). A draw whose last carrying slot
        vanishes can never be fixed up again, so its residual must be 0.
        """
        for d in self._slot_draws.pop(ot, ()):
            self._draw_refs[d] -= 1
            if not self._draw_refs[d]:
                del self._draw_refs[d]
                r = self.ledger.pop(d)
                opn, o_id = self._draw_src.pop(d)
                if r:
                    lane = (r & -r).bit_length() - 1
                    raise CleanlinessError(
                        f"op #{self._op_n} HMR o{ot}: overwrites the last carrier "
                        f"of a draw (HMR o{o_id} at op #{opn}) whose fixup residual "
                        f"is unresolved on {_popcount(r)}/{self.batch} lanes "
                        f"(first: lane {lane})"
                    )

    # ---- state access ----------------------------------------------------------
    def _grow(self, arr, i):
        if i >= NO_QUBIT:  # NO_QUBIT == NO_OUT sentinel
            raise ValueError("op references a sentinel (unset) qubit/outcome field")
        if i >= len(arr):
            arr.extend([0] * (i + 1 - len(arr)))

    def _q(self, i):
        self._grow(self.qubits, i)
        return self.qubits[i]

    def _o(self, i):
        self._grow(self.outs, i)
        return self.outs[i]

    def _b(self, i):
        self._grow(self.bits, i)
        return self.bits[i]

    # ---- I/O: pack/unpack lane values (little-endian wire lists) ---------------
    def write(self, qubits, values):
        """Write ``values[j]`` into lanes j of the little-endian qubit list."""
        self._write(self.qubits, qubits, values)

    def _write(self, arr, wires, values):
        values = list(values)
        if len(values) != self.batch:
            raise ValueError(f"expected {self.batch} values (one per lane), got {len(values)}")
        nw = len(wires)
        for j, v in enumerate(values):
            if v < 0 or v >> nw:
                raise ValueError(f"value {v} (lane {j}) does not fit in {nw} wires")
        for i, w in enumerate(wires):
            self._grow(arr, w)
            col = 0
            for j, v in enumerate(values):
                col |= ((v >> i) & 1) << j
            arr[w] = col

    def read(self, qubits):
        """Read the little-endian qubit list as one int per lane."""
        return self._read(self.qubits, qubits)

    def read_outs(self, outs):
        """Read a little-endian list of outcome ids as one int per lane
        (inspection only — outcomes are written solely by HMR/OXOR).
        """
        return self._read(self.outs, outs)

    def write_bits(self, bits, values):
        """Write ``values[j]`` into lanes j of the little-endian bit-id list
        (``Bit`` handles or raw ids) — the test-harness spelling of the
        evaluator's per-shot Bit-register injection.
        """
        self._write(self.bits, [getattr(c, "id", c) for c in bits], values)

    def _read(self, arr, wires):
        out = [0] * self.batch
        for i, w in enumerate(wires):
            self._grow(arr, w)
            col = arr[w]
            for j in range(self.batch):
                out[j] |= ((col >> j) & 1) << i
        return out

    # ---- execution (the evaluator's exact gate semantics) ----------------------
    def run(self):
        """Apply every op of the construction-time ``source``. Returns ``self``.

        Loop-form Builders (``repeat``/``gated``/tables) take the native
        recorded-form fast path when the C kernel is available: the recorded
        loops are packed AS loops and expanded in C, so Python cost scales
        with the description, not the expansion (``iter_expanded`` stays the
        normative expansion; any native halt replays through it).
        """
        if self._source is None:
            raise ValueError("no source circuit; construct with one or use apply()")
        regs = getattr(self._source, "_registers", None)
        if regs:
            for kind, wires in Regs(regs).registers:
                if kind == WKIND_QUBIT:
                    self.register_qubits.update(wires)
        if type(self) is Simulator:
            fast = _fast_engine()
            if fast is not None and fast.enabled():
                done = fast.run_source(self, self._source)
                if done is not None:
                    return done
        expand = getattr(self._source, "iter_expanded", None)
        ops = expand() if callable(expand) else getattr(self._source, "ops", self._source)
        return self.apply(ops)

    def apply(self, ops, validate=True):
        """Apply an iterable of op 9-tuples. Returns ``self``.

        ``validate=False`` skips the per-op operand check
        (:func:`uncompute.ops.op_violation`). Only a caller that already
        checked every op should pass it, such as the evaluator's stream
        interpreter.

        When the C kernel is built (``make kernel``), the chunk runs on the
        native engine (:mod:`uncompute._simfast`), which then syncs the state
        back. Its results are identical to :meth:`_apply_py`, the normative
        reference. The reference runs instead for Simulator subclasses and
        under ``UNCOMPUTE_KERNEL=0``.
        """
        if type(self) is Simulator:
            fast = _fast_engine()
            if fast is not None and fast.enabled():
                return fast.apply(self, ops, validate)
        return self._apply_py(ops, validate)

    def _apply_py(self, ops, validate=True):
        """Run the reference apply loop, which defines the gate semantics."""
        live = self.live_mask
        npop = _popcount(live)
        for op in ops:
            kind, _qa, _qb, qt, ot, oc, _rt, ct, cc = op
            if validate:
                reason = op_violation(op)
                if reason is not None:
                    raise ValueError(reason)

            # Bill every live lane at the billed kind. Conditions narrow the
            # action mask below, never the billing.
            cls = _CLASS_COND[kind] if cc != NO_BIT else _CLASS_PLAIN.get(kind)
            if cls == "toffoli":
                self.stats.toffoli += npop
            elif cls == "clifford":
                self.stats.clifford += npop

            # The action mask: live lanes, narrowed by a bit condition.
            act = live if cc == NO_BIT else live & self._b(cc)

            controls = _TOGGLE_CONTROLS.get(kind)
            factors = _SIGN_FACTORS.get(kind)
            if controls is not None:
                m = act
                for pos in controls:
                    m &= self._q(op[pos])
                self.qubits[qt] = self._q(qt) ^ m
            elif factors is not None:
                m = act
                for pos in factors:
                    m &= self._q(op[pos])
                if oc == NO_OUT:
                    self.phase ^= m
                else:
                    # Outcome condition: flip only where o[oc] is 1. The
                    # strict ledger records m as the fix-up coefficient of
                    # every draw that o[oc] carries.
                    self.phase ^= m & self._o(oc)
                    if self.strict:
                        for d in self._slot_draws.get(oc, ()):
                            self.ledger[d] ^= m
            elif kind == _K_SWAP:
                a, t = self._q(op[2]), self._q(qt)
                diff = (a ^ t) & act
                self.qubits[op[2]] = a ^ diff
                self.qubits[qt] = t ^ diff
            elif kind == _K_HMR:
                r = self._draw()
                val = self._q(qt) & act
                if self.strict:
                    self._retire_slot(ot)
                    d = self._ctr  # unique per draw
                    self._slot_draws[ot] = {d}
                    self._draw_refs[d] = 1
                    self.ledger[d] = val
                    self._draw_src[d] = (self._op_n, ot)
                self.outs[ot] = (self._o(ot) & ~act) | (r & act)
                self.phase ^= val & r
                self.qubits[qt] &= ~act
            elif kind == _K_R:
                r = self._draw()
                val = self._q(qt) & act
                if self.strict and val:
                    lane = (val & -val).bit_length() - 1
                    raise CleanlinessError(
                        f"op #{self._op_n} R on qubit {qt}: wire is nonzero "
                        f"on {_popcount(val)}/{self.batch} lanes (first: lane "
                        f"{lane}); a reset of a nonzero wire kicks a random "
                        "sign that no later op can undo"
                    )
                self.phase ^= val & r
                self.qubits[qt] &= ~act
            elif kind == _K_OXOR:
                self.outs[ot] = self._o(ot) ^ (self._o(oc) & act)
                if self.strict:
                    self._merge_slot(ot, oc)
            elif kind == _K_BIT_INVERT:
                self.bits[ct] = self._b(ct) ^ act
            elif kind == _K_BIT_STORE0:
                self.bits[ct] = self._b(ct) & ~act
            elif kind == _K_BIT_STORE1:
                self.bits[ct] = self._b(ct) | act
            else:
                raise ValueError(f"unknown op kind {kind}")
            self._op_n += 1
        return self

    def _merge_slot(self, ot, oc):
        """Strict mode: after ``OXOR ot ^= oc``, slot ``ot`` carries the
        GF(2) sum of both draw sets. A draw in both sets cancels out of
        ``ot``; its reference count cannot reach zero here, because ``oc``
        still carries it.
        """
        src = self._slot_draws.get(oc)
        if not src:
            return
        dst = self._slot_draws.setdefault(ot, set())
        for d in src:
            if d in dst:
                dst.discard(d)
                self._draw_refs[d] -= 1
            else:
                dst.add(d)
                self._draw_refs[d] += 1
        if not dst:
            del self._slot_draws[ot]

    # ---- validity gates ---------------------------------------------------------
    def assert_clean(self, outputs=(), phase=0):
        """Check the evaluator's phase/reversibility gates on every lane.

        Mirrors the evaluator's verdicts exactly: every qubit belonging to a
        **declared register** (a Qubit register of the source Builder) is
        exempt — registers hold outputs and restored inputs — and every OTHER
        qubit must be |0> on every lane, with zero global phase. ``outputs``
        adds extra allowed qubit ids, for subcircuit tests that declare no
        registers; note the real evaluator exempts register qubits only, so an
        ``outputs`` entry that is not a register qubit has no evaluator
        counterpart. Raises :class:`CleanlinessError` naming the first
        offender and lane; passes silently otherwise.

        In strict mode (the default) the fixup **ledger** is checked first:
        every ``hmr`` draw's phase kick must be cancelled *algebraically*
        (for both outcome values) by the conditioned Z/CZ/NEG fixups — the
        deterministic form of the sampled phase check that follows.

        ``phase`` is the *expected* per-lane global-phase mask (an int, lane
        j at bit j; default 0). Batteries that verify intentional per-lane
        signs pass their expected mask here instead of pre-cancelling
        ``sim.phase`` by hand.
        """
        if self.strict:
            for d, r in self.ledger.items():
                if r:
                    opn, o_id = self._draw_src[d]
                    lane = (r & -r).bit_length() - 1
                    raise CleanlinessError(
                        f"measurement fixup residual for outcome o{o_id} (HMR "
                        f"at op #{opn}) unresolved on {_popcount(r)}/"
                        f"{self.batch} lanes (first: lane {lane}) — the "
                        "conditioned fixups do not reproduce the demolished "
                        "value"
                    )
        expected = phase & self.live_mask
        if self.phase != expected:
            diff = self.phase ^ expected
            lane = (diff & -diff).bit_length() - 1
            n = _popcount(diff)
            if expected:
                raise CleanlinessError(
                    f"global phase differs from the expected mask on "
                    f"{n}/{self.batch} lanes (first: lane {lane})"
                )
            raise CleanlinessError(
                f"leftover global phase on {n}/{self.batch} lanes (first: lane {lane})"
                " — dirty uncomputation/reset kickback"
            )
        out = self.register_qubits | set(outputs)
        for q, v in enumerate(self.qubits):
            if q in out or not v:
                continue
            lane = (v & -v).bit_length() - 1
            raise CleanlinessError(
                f"qubit {q} not |0> on {_popcount(v)}/{self.batch} lanes"
                f" (first: lane {lane}) — ancilla not uncomputed"
            )

    @property
    def avg_toffoli(self):
        """Executed Toffoli count averaged over lanes (the Toffoli factor of the TQ-cost)."""
        return self.stats.toffoli / self.batch
