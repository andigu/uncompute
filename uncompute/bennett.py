"""uncompute.bennett — Bennett-reversal choreography over the Builder primitives.

``Builder.mirror`` inverts a contiguous region whose events it can prove
exact conjugates for (alloc/free, the Gidney AND pair, involutions). The
two classes here are the shared engines for the reversal patterns mirror
cannot express on its own; both are pure choreography over public Builder
calls and emit nothing themselves.

* :class:`Journal` — composite conjugate pairs: circuits whose uncompute
  strategy pairs *composite* gadgets (a bilinear MAC released by a
  measurement mesh, a schedule that recomputes some rounds and releases
  others) record each forward step's conjugate as it is emitted, then
  replay the conjugates in reverse.
* :class:`Stages` — the staged-unwind choreography: record stage heads
  (allocations + ownership claims), then unwind every stage in reverse
  through the standard ``set_phase`` / mirror-body / release /
  mirror-head loop, byte-identical to the hand-rolled spelling.

Neither class verifies a recorded conjugate; the author asserts it, and
the evaluator's reversibility and phase checks hold the proof obligation
at eval time.
"""

from contextlib import contextmanager

from .builder import Mark


class Journal:
    """A list of ``(fwd, rev)`` thunk pairs with forward/reverse replay.

    Three recording styles, matching the patterns found in circuits:

    * ``step(fwd, rev=None)`` — run ``fwd()`` now and journal the pair
      (``rev=None`` marks a self-inverse step). Recording honors
      ``recording``, so replays through the same emitters do not re-journal.
    * ``push(rev, fwd=None)`` — journal a step that the caller has already
      emitted, when its conjugate is only known after emission (e.g. a MAC
      whose release-vs-recompute choice depends on state observed while
      emitting). Honors ``recording``.
    * ``defer(fwd, rev=None)`` — journal without running: build a program
      to play later with ``forward()``. Ignores ``recording`` (construction
      is always explicit).

    ``mark()`` returns an index for slicing; ``forward(since, until)`` and
    ``reverse(since, until)`` replay a slice. ``reverse`` plays the ``rev``
    thunks in reverse order; ``forward`` plays the ``fwd`` thunks in order
    and raises on entries recorded with ``push(rev)`` alone.
    """

    def __init__(self):
        self._steps = []
        self.recording = True

    def __len__(self):
        return len(self._steps)

    def mark(self):
        return len(self._steps)

    def step(self, fwd, rev=None):
        fwd()
        if self.recording:
            self._steps.append((fwd, rev if rev is not None else fwd))

    def push(self, rev, fwd=None):
        if self.recording:
            self._steps.append((fwd, rev))

    def defer(self, fwd, rev=None):
        self._steps.append((fwd, rev if rev is not None else fwd))

    def forward(self, since=0, until=None):
        stop = len(self._steps) if until is None else until
        for fwd, _ in self._steps[since:stop]:
            if fwd is None:
                raise ValueError(
                    "forward(): entry has no forward replayer (recorded with push(rev) only)"
                )
            fwd()

    def reverse(self, since=0, until=None):
        stop = len(self._steps) if until is None else until
        for _, rev in reversed(self._steps[since:stop]):
            rev()


class Stages:
    """The staged-unwind recorder for one Builder.

    The ``with st.stage(...)`` block is the stage HEAD — the stage's
    allocations and ownership ``claim`` s; everything recorded after the
    block, up to the next stage (or the ``until=`` boundary at unwind), is
    the stage BODY. :meth:`unwind` then replays the stack in reverse
    through the standard choreography: ``set_phase(prefix + name)``; mirror
    the body region (the stage's wires return to |0>); ``release`` the
    named set, if any; mirror the head (replaying the frees). Pure
    choreography over public Builder calls — ``checkpoint`` /
    ``mirror(until=, phase_prefix=)`` / ``set_phase`` / ``release`` — so
    the emitted stream is byte-identical to the hand-rolled per-stage
    mirror loop.
    """

    def __init__(self, b):
        self.b = b
        self._stack = []  # (name, set_name, head, body, wires, unwind)
        self._open = False  # True inside an open stage() head block

    @contextmanager
    def stage(self, name, set_name=None, unwind=None):
        """Record one unwindable stage for a later :meth:`unwind`.

        The with-block is the stage HEAD — the stage's allocations and
        ownership :meth:`~uncompute.Builder.claim` s. Everything recorded
        after the block, up to the next stage (or the ``until=`` boundary
        at unwind), is the stage BODY. ``set_name`` names the ownership
        set claimed in the head; its claimed wires are collected (from the
        Builder's wire log, so claims made inside head-block helpers
        count) for the unwind's release. ``unwind`` swaps the standard
        unwind choreography for a callback — the escape hatch for a stage
        whose inverse is hand-rolled (say one that allocates and frees its
        word inside the stage): at unwind time the callback is invoked in
        place of the mirrors, and the stage's head mark still bounds the
        previous stage's body.

        Stages open at the top level only: their marks must stay valid
        until :meth:`unwind`, and an in-loop mark dies when its ``repeat``
        frame closes — use ``checkpoint``/``mirror`` directly inside loop
        bodies.
        """
        b = self.b
        if b._loop_stack:
            raise RuntimeError(
                "stage() inside a repeat() block: the stage's marks would "
                "be invalidated when the frame closes (in-loop marks die "
                "with their frame) — use checkpoint()/mirror() inside loop "
                "bodies"
            )
        if b._gate_ctx is not None:
            raise RuntimeError(
                "stage() inside a gated() block: mirrored ops must not silently acquire a data gate"
            )
        if self._open:
            raise RuntimeError("stage() blocks do not nest: close the open stage head first")
        head = b.checkpoint()
        wl_pos = len(b.wire_log)
        self._open = True
        try:
            yield
        finally:
            self._open = False
        body = b.checkpoint()
        wires = []
        if set_name is not None:
            for kind, x, _tok, _live in b.wire_log[wl_pos:]:
                if kind == "claim" and x[0] == set_name:
                    wires.extend(x[1])
            if not wires:
                raise ValueError(
                    f"stage({name!r}): set_name={set_name!r} but the head "
                    "block claimed no wires into that set"
                )
        self._stack.append((str(name), set_name, head, body, tuple(wires), unwind))

    def unwind(self, prefix="un-", until=None):
        """Unwind every recorded :meth:`stage` in reverse order.

        Per stage, the standard choreography: ``set_phase(prefix + name)``;
        mirror the body region (the stage's wires return to |0>);
        release the named set, if any; mirror the head (replaying
        the frees); then the stage's head mark becomes the next (earlier)
        stage's body boundary. A stage carrying an ``unwind=`` callback
        contributes the callback instead — boundary maintenance unchanged.
        ``until=`` caps the LAST stage's body: checkpoint before an
        un-unwound tail (the conjugation shape F·A·F⁻¹) and pass that
        Mark; by default the body runs to the current position.
        """
        b = self.b
        if not isinstance(prefix, str):
            raise TypeError(f"unwind() prefix= must be a str, got {type(prefix).__name__}")
        if self._open:
            raise RuntimeError("unwind() inside an open stage() head block")
        if not self._stack:
            raise RuntimeError("unwind(): no stages recorded")
        if b._loop_stack:
            raise RuntimeError("unwind() inside a repeat() block: stages are top-level regions")
        if until is not None and not isinstance(until, Mark):
            raise TypeError(
                f"unwind() until= takes a Mark from checkpoint(), got {type(until).__name__}"
            )
        after = b.checkpoint() if until is None else until
        while self._stack:
            name, set_name, head, body, wires, cb = self._stack.pop()
            if cb is not None:
                cb()
            else:
                b.set_phase(prefix + name)
                b.mirror(body, until=after, phase_prefix=prefix)
                if set_name is not None:
                    b.release(set_name, wires)
                b.mirror(head, body, phase_prefix=prefix)
            after = head
