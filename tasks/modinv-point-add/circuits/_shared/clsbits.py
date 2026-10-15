"""Classical-bit input layer for the modinv-point-add circuits
(circuits/divsteps, circuits/euclidean, circuits/luo-eea): the task's
point I/O registers and the classical-bit networks (free ops) over the
Bit-kind addend registers.

A circuit loads this file by path (uncompute.testing.load_module); it has
no circuit-local state.  ../_shared/luo_core.py loads it the same way
and re-exports its names.
"""

from uncompute.gadgets import declare_registers


def declare_point_io(b, n):
    """Declare the four I/O registers in schema order: the quantum target
    point (registers 0/1, n qubits each) and the per-shot classical addend
    Q (registers 2/3, n Bit-kind wires each).  Returns (x, y, ox, oy).
    """
    regs = declare_registers(b, (("target_x", n), ("target_y", n)))
    ox = b.bits(n)
    b.bit_register(ox)  # register 2: offset_x (classical)
    oy = b.bits(n)
    b.bit_register(oy)  # register 3: offset_y (classical)
    return regs["target_x"], regs["target_y"], ox, oy


# ===========================================================================
# classical-bit networks (free ops): per-shot cbspec derivation from the
# Bit-kind input registers
# ===========================================================================


class ClsBits:
    """Classical logic network builder over uncompute bit ops (zero qubits,
    zero Toffoli; BIT_INVERT / BIT_STORE0 / BIT_STORE1 only).
    """

    def __init__(self, b):
        self.b = b
        self._zero = None

    def zero(self):
        if self._zero is None:
            self._zero = self.b.bit()
        return self._zero

    def copy(self, src):
        t = self.b.bit()
        self.b.bit_invert(t, when=src)
        return t

    def not_(self, src):
        t = self.copy(src)
        self.b.bit_invert(t)
        return t

    def and_(self, x, y):
        t = self.copy(x)
        ny = self.not_(y)
        self.b.bit_store0(t, when=ny)
        return t

    def or_(self, x, y):
        t = self.copy(x)
        self.b.bit_store1(t, when=y)
        return t

    def mux(self, sel, a1, a0):
        t = self.and_(sel, a1)
        ns = self.not_(sel)
        u = self.and_(ns, a0)
        self.b.bit_store1(t, when=u)
        return t

    def add_bits(self, A, B, w):
        """Return (A + B) as w+1 bits."""
        z = self.zero()
        A = list(A) + [z] * (w - len(A))
        B = list(B) + [z] * (w - len(B))
        out, c = [], None
        for i in range(w):
            a, b2 = A[i], B[i]
            axb = self.copy(a)
            self.b.bit_invert(axb, when=b2)
            if c is None:
                out.append(self.copy(axb))
                c = self.and_(a, b2)
            else:
                s = self.copy(axb)
                self.b.bit_invert(s, when=c)
                out.append(s)
                t1 = self.and_(a, b2)
                t2 = self.and_(c, axb)
                c = self.or_(t1, t2)
        out.append(c)
        return out

    def add_const(self, A, const, w):
        """Return (A + const) as w+1 bits."""
        z = self.zero()
        A = list(A) + [z] * (w - len(A))
        out, c = [], None
        for i in range(w):
            a = A[i]
            kb = (const >> i) & 1
            if c is None:
                out.append(self.copy(a) if not kb else self.not_(a))
                c = self.copy(a) if kb else self.copy(z)
            else:
                s = self.copy(a)
                self.b.bit_invert(s, when=c)
                if kb:
                    self.b.bit_invert(s)
                out.append(s)
                c = self.or_(a, c) if kb else self.and_(a, c)
        out.append(c)
        return out

    def mod_reduce_once(self, W, p, n):
        """W (n+1 bits, < 2p) -> W mod p (n bits)."""
        w1 = n + 1
        D = self.add_const(W, (1 << w1) - p, w1)
        geq = D[w1]
        return [self.mux(geq, D[i], W[i]) for i in range(n)]

    def mod_add(self, A, B, p, n):
        return self.mod_reduce_once(self.add_bits(A, B, n), p, n)

    def triple_mod_p(self, A, p, n):
        return self.mod_add(A, self.mod_add(A, A, p, n), p, n)


def const_modp_bit_specs(cb, C, p, n):
    """Per-shot const_modp cbspecs from a little-endian Bit list C on
    the ClsBits network cb -- the same four vectors as the compile-time
    spec builder: k1 = (C+2^n-p) mod 2^n, k2 = (p-C) mod 2^n =
    (~C+p+1) mod 2^n, k3 = C, k4 = (2^n-C) mod 2^n = (~C+1) mod 2^n.
    0 < C < p is the input-domain contract (the compile-time builder's
    assert), carried by the caller's drawn values.
    """
    notC = [cb.not_(c) for c in C]
    return {
        "k1": cb.add_const(C, ((1 << n) - p) % (1 << n), n)[:n],
        "k2": cb.add_const(notC, (p + 1) % (1 << n), n)[:n],
        "k3": list(C),
        "k4": cb.add_const(notC, 1, n)[:n],
    }


def point_cbspecs(b, ox, oy, p, n):
    """Per-shot const_modp cbspecs of the addend Q = (ox, oy): returns
    (S_x, S_y, S_3x) for Q.x, Q.y and 3*Q.x mod p, built on one fresh
    ClsBits network in that order.
    """
    cb = ClsBits(b)
    s_x = const_modp_bit_specs(cb, ox, p, n)
    s_y = const_modp_bit_specs(cb, oy, p, n)
    s_3x = const_modp_bit_specs(cb, cb.triple_mod_p(ox, p, n), p, n)
    return s_x, s_y, s_3x
