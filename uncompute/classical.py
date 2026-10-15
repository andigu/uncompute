"""uncompute.classical — classical math shared by circuits and their checks.

Nothing here touches a Builder: these are the pure-Python models and
table-builders whose OUTPUTS enter emitted circuits (lookup tables, linear
layers, test vectors, reference values): GF(2^n) scalar arithmetic and
GF(2) linear-map helpers. One exp/log-table field (``GF2n``) serves every
finite-field circuit. ``poly`` is always the full reduction polynomial
including the x^n term (e.g. 0x13 for GF(16)).
"""


# ---- GF(2^n) scalar arithmetic ---------------------------------------------


def _clmul_mod(a, b, n, poly):
    """Return a * b in GF(2^n) mod ``poly`` (LSB-first schoolbook + reduce)."""
    prod = 0
    for i in range(n):
        if (a >> i) & 1:
            prod ^= b << i
    for bit in range(2 * n - 2, n - 1, -1):
        if (prod >> bit) & 1:
            prod ^= poly << (bit - n)
    return prod & ((1 << n) - 1)


class GF2n:
    """GF(2^n) mod ``poly``, by exp/log tables over a primitive element.

    ``gen`` is the smallest primitive element, so ``gen == 2`` (the element
    x) whenever x is primitive. ``exp[k] = gen^k`` for k in [0, 2*order):
    the table repeats once, so ``exp[log[a] + log[b]]`` needs no modulo.
    ``log[a]`` is the exponent k with gen^k = a, for nonzero a. ``log[0]`` is 0, a
    placeholder: callers test for zero first. Hot loops may index ``exp``
    and ``log`` directly. The constructor raises ValueError when ``poly``
    is not irreducible of degree n. Tables cost O(2^n), so n stays small.
    """

    __slots__ = ("exp", "gen", "log", "n", "order")

    def __init__(self, n, poly):
        self.n = n
        self.order = order = (1 << n) - 1  # multiplicative group order
        for gen in range(1, 1 << n) if poly >> n == 1 else ():
            exp, x = [1], gen
            while x != 1 and len(exp) < order:
                exp.append(x)
                x = _clmul_mod(x, gen, n, poly)
            if x == 1 and len(exp) == order:
                break
        else:
            raise ValueError(f"poly {poly:#x} is not irreducible of degree {n}")
        self.gen = gen
        self.exp = exp + exp
        self.log = [0] * (order + 1)
        for k, v in enumerate(exp):
            self.log[v] = k

    def mul(self, a, b):
        """Return a * b."""
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def pow(self, a, e):
        """Return a^e for any integer e. 0^0 is 1; 0^e is 0 otherwise."""
        if a == 0:
            return int(e == 0)
        return self.exp[self.log[a] * e % self.order]

    def inv(self, a):
        """Return a^-1. The inverse is totalized: 0 maps to 0."""
        return self.exp[self.order - self.log[a]] if a else 0

    def sqrt(self, a):
        """Return the unique square root, a^(2^(n-1))."""
        return self.exp[(self.log[a] << (self.n - 1)) % self.order] if a else 0

    def trace(self, a, k=None):
        """Return a + a^2 + a^4 + ... + a^(2^(k-1)).

        ``k=None`` means k = n: the absolute trace, 0 or 1 for every a.
        For a in the subfield GF(2^k), k dividing n, the same sum is the
        subfield trace to GF(2).
        """
        t = 0
        for _ in range(self.n if k is None else k):
            t ^= a
            a = self.mul(a, a)
        return t


# ---- GF(2) linear-map helpers ----------------------------------------------


def parity(x):
    """GF(2) parity of an int's bits."""
    return x.bit_count() & 1


def f2_rows_of(fn, n):
    """Return the rows of the GF(2)-linear map ``fn`` on n-bit ints.

    Row i is a packed int: bit j is set when input bit j feeds output
    bit i (LSB-first, the layout ``gadgets.linear_xor`` takes). The call
    checks linearity over all 2^n inputs and raises AssertionError on a
    nonlinear ``fn``, which would otherwise yield a silently wrong
    matrix.
    """
    cols = [fn(1 << j) for j in range(n)]
    for v in range(1 << n):
        image = 0
        for j in range(n):
            if (v >> j) & 1:
                image ^= cols[j]
        assert fn(v) == image, "map is not GF(2)-linear"
    return [sum((((cols[j] >> i) & 1) << j) for j in range(n)) for i in range(n)]


def f2_matinv(rows, n):
    """Inverse of the n x n GF(2) matrix ``rows`` (LSB-first packed ints)
    by partial-pivot Gauss-Jordan; asserts invertibility.
    """
    A = list(rows)
    I = [1 << i for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if (A[r] >> col) & 1:
                piv = r
                break
        assert piv is not None, "singular matrix"
        A[col], A[piv] = A[piv], A[col]
        I[col], I[piv] = I[piv], I[col]
        for r in range(n):
            if r != col and ((A[r] >> col) & 1):
                A[r] ^= A[col]
                I[r] ^= I[col]
    return I


def f2_transpose(rows, n):
    """Transpose of the n x n GF(2) matrix ``rows`` (LSB-first packed
    ints).
    """
    return [sum(((rows[r] >> c) & 1) << r for r in range(n)) for c in range(n)]
