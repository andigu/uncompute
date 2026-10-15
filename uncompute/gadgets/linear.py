"""GF(2) linear-layer synthesis.

Source: conventional Gauss-Jordan row elimination, emitted as the reversed
elimination schedule — the textbook in-place CNOT/SWAP realization of an
invertible F2 matrix, as used across the finite-field literature. ``linear_xor`` is the textbook
out-of-place form: one CNOT per set matrix entry.
"""


def linear_in_place(b, qs, rows):
    """Apply the invertible GF(2) matrix ``rows`` to register ``qs`` in
    place: qs <- rows * qs. ``rows[i]`` is the packed-int row i (bit j =
    column j, little-endian to match register order).

    Synthesis is leftmost-pivot Gauss-Jordan elimination recorded and
    emitted in reverse: 0 Toffoli, only CX and SWAP; no ancillas. Raises
    (via the identity assert) if the matrix is singular.
    """
    n = len(qs)
    a = list(rows)
    assert len(a) == n
    ops = []
    for col in range(n):
        pivot = next(r for r in range(col, n) if (a[r] >> col) & 1)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            ops.append(("swap", col, pivot))
        for r in range(n):
            if r != col and (a[r] >> col) & 1:
                a[r] ^= a[col]
                ops.append(("cx", col, r))
    assert a == [1 << i for i in range(n)], "singular matrix"
    for kind, i, j in reversed(ops):
        (b.swap if kind == "swap" else b.cx)(qs[i], qs[j])


def linear_xor(b, src, dst, rows):
    """XOR the GF(2) matrix-vector product onto a second register:
    dst ^= rows * src. ``rows[i]`` is the packed-int row i (bit j =
    column j), for example from :func:`uncompute.classical.f2_rows_of`.

    Emits one CX per set matrix entry in row-major order: for each dst
    bit i, for each src bit j. 0 Toffoli; no ancillas. ``src`` and
    ``dst`` must be disjoint.
    """
    for i, row in enumerate(rows):
        for j in range(len(src)):
            if (row >> j) & 1:
                b.cx(src[j], dst[i])
