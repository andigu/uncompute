"""Wire plumbing: schema register declaration and spare-wire pooling.

Neither function is a circuit subroutine — they are how a circuit gets its
wires into shape for the gadgets that are: ``declare_registers`` allocates
and declares the task's I/O schema in order, and :class:`WirePool`
fronts a bank of caller-certified |0> wires so any gadget's ``pool=``
parameter can trade certified spare wires against peak width.
"""


def declare_registers(b, spec):
    """Allocate and declare the task's I/O registers in schema order.

    ``spec`` is an iterable of ``(name, width)`` pairs. Each register is
    allocated with ``qubits`` and declared with
    ``qubit_register`` immediately — the pairing every circuit
    hand-rolls and occasionally forgets. Returns ``dict name -> WireList``
    (the Builder's own return convention, so schema registers stay
    affine-indexable); the schema width for the leak guard is
    ``sum(len(v) for v in regs.values())``.
    """
    regs = {}
    for name, width in spec:
        qs = b.qubits(width)
        b.qubit_register(qs)
        assert name not in regs, f"duplicate register name {name!r}"
        regs[name] = qs
    return regs


class WirePool:
    """Spare-wire pooling for temporary ANDs.

    A ``WirePool`` fronts a bank of caller-certified |0> wires (typically
    persistent register bits that are provably zero over a window of the
    circuit) and duck-types the four Builder methods gadgets use to place
    temporary ANDs: ``qubit`` / ``recycle`` / ``and_compute`` /
    ``and_release``.  A pooled AND is a plain CCX onto a spare — the wire
    already exists, so no fresh allocation raises the peak — released
    measurement-based (HMR + conditional CZ) back into the bank.  When the
    bank is empty every call falls through to the Builder, i.e. the
    default behavior.  Pass a pool to any gadget's ``pool=`` to trade
    certified spare wires against peak width.
    """

    def __init__(self, b, spares=()):
        self.b = b
        self._free = list(spares)
        self._spares = set(spares)

    def qubit(self):
        return self._free.pop() if self._free else self.b.qubit()

    def recycle(self, q):
        if q in self._spares:
            self._free.append(q)
        else:
            self.b.recycle(q)

    def and_compute(self, a, c):
        if self._free:
            t = self._free.pop()
            self.b.ccx(a, c, t)
            return t
        return self.b.and_compute(a, c)

    def and_release(self, t, a, c):
        if t in self._spares:
            self.b.cz(a, c, cond=self.b.hmr(t))
            self._free.append(t)
        else:
            self.b.and_release(t, a, c)
