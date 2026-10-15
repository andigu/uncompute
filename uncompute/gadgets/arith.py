"""Reversible integer arithmetic gadgets.

Sources
-------
[GID]   Gidney (2018), arXiv:1709.06648. Fig. 4 (temporary AND: compute 1 Toffoli,
        measurement-based erase 0); the n-bit adder built from n-1
        temporary ANDs.
[CDKM]  Cuccaro et al. (2004), arXiv:quant-ph/0410184. The
        MAJ/UMA ripple chain (2 Toffoli per position, carries live in the
        operand bits); subtraction by complementing the target (the
        paper's subtraction remark).
[HRS]   Häner et al. (2018), arXiv:1805.12445. The piecewise-polynomial
        evaluator: the label register (one comparator per interval bound
        drives a counter) and the pre-truncated signed multiplier.

``add`` is the one general in-place adder; ``gidney_add`` / ``gidney_sub``
/ ``cdkm_add_co`` are its named corners, kept as working vocabulary.
"""


def maj(b, c, bq, a2):
    """[CDKM] MAJ step: ``a2 <- MAJ(c, bq, a2)`` (the ripple carry), with
    ``c <- c^a2`` and ``bq <- bq^a2`` left in one's-complement form for the
    matching :func:`uma`. 1 executed Toffoli, no ancilla.
    """
    b.cx(a2, bq)
    b.cx(a2, c)
    b.ccx(c, bq, a2)


def maj_inv(b, c, bq, a2):
    """Exact adjoint of :func:`maj` (each gate is self-inverse; reversed
    order). 1 executed Toffoli.
    """
    b.ccx(c, bq, a2)
    b.cx(a2, c)
    b.cx(a2, bq)


def uma(b, c, bq, a2):
    """[CDKM] UMA step: undoes :func:`maj`'s carry and leaves the sum in
    ``bq`` (``bq <- c^bq^a2_orig``), restoring ``a2``. 1 executed Toffoli,
    no ancilla.
    """
    b.ccx(c, bq, a2)
    b.cx(a2, c)
    b.cx(c, bq)


def uma_inv(b, c, bq, a2):
    """Exact adjoint of :func:`uma`. 1 executed Toffoli."""
    b.cx(c, bq)
    b.cx(a2, c)
    b.ccx(c, bq, a2)


def add(b, T, A, *, neg=None, gl=None, shared=None, cout=None, cin=None, pool=None, ctrl=None):  # noqa: C901, PLR0912  (one adder, many option corners)
    """``T += A (+ neg)`` mod ``2^len(T)`` — the general in-place adder.

    ``T`` is the little-endian target; ``A`` the operand, ``len(A) <=
    len(T)``. Operand entries may be qubit ids or ``None`` (a zero bit
    that still propagates the carry), and one wire may repeat across
    positions when named via ``shared=`` (it is restored inside every
    step — the Booth one-wire sign extension). Offsets need no parameter:
    slice the target, ``add(b, T[off:], A)``.

    Carry discipline per position — the (qubits, Toffoli) dial:

    * positions ``< gl`` (plus every ``None`` position, every ``shared``
      position, and the zero/sign-extension tail) ride [GID] measured
      carries: 1 Toffoli and one transient AND wire each, erased
      measurement-based;
    * operand positions ``>= gl`` ride the [CDKM] MAJ/UMA chain:
      2 Toffoli, zero transient wires (the carry lives in the operand
      bit, restored by UMA).

    ``gl=None`` means all-[GID] (the ``gidney_add`` corner); ``gl=0``
    all-[CDKM] (with ``cout``/``cin`` the ``cdkm_add_co`` corner).

    ``neg`` (a read-only wire) is a conditional-negate carry-in: with
    ``A`` holding the operand XOR ``neg`` (one's-complement form), the
    effect is ``T += A_orig`` when neg=0 and ``T -= A_orig`` when neg=1,
    sign-extended across the high tail when ``A`` is short (the tail is
    complemented against ``neg`` around its carry ripple).

    ``cout``: XOR the true full-width carry-out onto this wire — the top
    position then runs a real chain step instead of the carry-dropping CX
    finish. ``cin``: caller-provided |0> wire seeding the first MAJ step
    (allocated internally when omitted; [CDKM]-corner callers pass their
    own to stay at zero fresh ancillas).

    ``pool``: alloc/AND provider duck-typing the Builder's ``qubit
    / recycle / and_compute / and_release`` (see
    :class:`~uncompute.gadgets.wires.WirePool`); default the Builder.

    ``ctrl`` (a read-only wire): the fused controlled [GID] corner —
    ``T += (A AND ctrl)``. Each addend position's AND with ``ctrl`` is
    computed lazily inside the carry chain and released measurement-based
    on the return sweep, so no pre-computed addend register is ever live.
    Takes a gap-free ``A`` and the other options at their defaults
    (``neg``/``gl``/``shared``/``cout``/``cin`` are the uncontrolled
    dial). Executed Toffoli = len(T) - 1 + len(A); the chain's ANDs stay
    live until the unwind (the row's transient width, traded against the
    recompute Toffoli a live-capped schedule would pay). ``len(T) == 1``
    degenerates to one CCX.

    Executed Toffoli = #[GID] steps + 2 * #MAJ steps. Transient width =
    the live [GID] ancillas — every pre-tail [GID] carry stays live until
    the unwind, which is exactly what ``gl`` caps — plus the MAJ seed if
    allocated. ``A``, ``neg``, ``cin`` are restored exactly.
    """
    w = pool if pool is not None else b
    n = len(T)
    A = list(A)[:n]
    while A and A[-1] is None:
        A.pop()
    m = len(A)
    if m == 0:
        return
    if ctrl is not None:
        assert neg is None and gl is None and shared is None and cout is None and cin is None, (
            "ctrl= is the fused all-[GID] corner: no neg/gl/shared/cout/cin"
        )
        assert all(ai is not None for ai in A), "ctrl= takes a gap-free operand"
        _add_ctrl(b, w, T, A, ctrl)
        return
    if gl is None:
        gl = n
    recs: list[tuple] = []  # forward step log, unwound in exact reverse
    c = neg  # current carry wire (read-only while it is neg)
    c0 = None  # internally allocated MAJ seed
    for p in range(n):
        ai = A[p] if p < m else None
        if p == m and neg is not None:
            # sign extension: complement the tail against neg and ripple
            # the folded carry; the unwind replays the CXs in reverse.
            cxs = [(neg, T[t]) for t in range(m, n)]
            cxs.append((neg, c))
            for u, v in cxs:
                b.cx(u, v)
            recs.append(("fold", cxs))
        if p == n - 1 and cout is None:
            # top sum bit, carry-out dropped (mod 2^n)
            if ai is not None:
                b.cx(ai, T[p])
            if c is not None:
                b.cx(c, T[p])
            break
        if ai is None:
            if c is None:
                continue
            anc = w.and_compute(c, T[p])  # carry-only [GID] step
            recs.append(("g2", p, None, c, anc))
            c = anc
        elif p >= gl and (shared is None or ai != shared):
            if c is None or c == neg:  # MAJ needs a writable seed
                c0 = cin if cin is not None else w.qubit()
                if neg is not None:
                    b.cx(neg, c0)
                c = c0
            maj(b, c, T[p], ai)  # [CDKM] in-place step
            recs.append(("maj", p, c, ai))
            c = ai
        elif c is None:
            anc = w.and_compute(ai, T[p])  # first [GID] step
            recs.append(("g0", p, ai, None, anc))
            c = anc
        else:
            b.cx(c, ai)  # [GID] chain step
            b.cx(c, T[p])
            anc = w.and_compute(ai, T[p])
            b.cx(c, anc)
            if shared is not None and ai == shared:
                b.cx(c, ai)  # restore the shared wire inside the step
                recs.append(("g3", p, ai, c, anc))
            else:
                recs.append(("g1", p, ai, c, anc))
            c = anc
    if cout is not None:
        b.cx(c, cout)
    for rec in reversed(recs):
        k = rec[0]
        if k == "maj":
            _, p, prev, ai = rec
            uma(b, prev, T[p], ai)
        elif k == "fold":
            for u, v in reversed(rec[1]):
                b.cx(u, v)
        elif k == "g1":
            _, p, ai, cw, anc = rec
            b.cx(cw, anc)
            w.and_release(anc, ai, T[p])
            b.cx(cw, ai)
            b.cx(ai, T[p])
        elif k == "g3":
            _, p, ai, cw, anc = rec
            b.cx(cw, anc)
            b.cx(cw, ai)
            w.and_release(anc, ai, T[p])
            b.cx(cw, ai)
            b.cx(ai, T[p])
        elif k == "g0":
            _, p, ai, _cw, anc = rec
            w.and_release(anc, ai, T[p])
            b.cx(ai, T[p])
        else:  # g2
            _, p, _ai, cw, anc = rec
            w.and_release(anc, cw, T[p])
            b.cx(cw, T[p])
    if c0 is not None:
        if neg is not None:
            b.cx(neg, c0)
        if cin is None:
            w.recycle(c0)


def _add_ctrl(b, w, T, A, ctrl):
    """Emit the fused controlled [GID] chain behind ``add(..., ctrl=)`` — the
    addend ANDs are computed inside the carry chain and released
    measurement-based on the return sweep.
    """
    n, m = len(T), len(A)
    assert 1 <= m <= n
    if n == 1:
        b.ccx(A[0], ctrl, T[0])
        return
    addend = {}
    chain = []
    a0 = w.and_compute(A[0], ctrl)
    addend[0] = a0
    chain.append(w.and_compute(a0, T[0]))
    for i in range(1, n - 1):
        ci = chain[-1]
        if i < m:
            ai = w.and_compute(A[i], ctrl)
            addend[i] = ai
            b.cx(ci, ai)
            b.cx(ci, T[i])
            cn = w.and_compute(ai, T[i])
            b.cx(ci, cn)
        else:
            cn = w.and_compute(ci, T[i])
        chain.append(cn)
    if m == n:
        an = w.and_compute(A[n - 1], ctrl)
        b.cx(an, T[n - 1])
        w.and_release(an, A[n - 1], ctrl)
    b.cx(chain[-1], T[n - 1])
    for i in range(n - 2, 0, -1):
        ci, cn = chain[i - 1], chain[i]
        if i < m:
            ai = addend[i]
            b.cx(ci, cn)
            w.and_release(cn, ai, T[i])
            b.cx(ci, ai)
            b.cx(ai, T[i])
            w.and_release(ai, A[i], ctrl)
        else:
            w.and_release(cn, ci, T[i])
            b.cx(ci, T[i])
    w.and_release(chain[0], a0, T[0])
    b.cx(a0, T[0])
    w.and_release(a0, A[0], ctrl)


def gidney_add(b, a, k):
    """k += a mod 2^len(k), len(a) <= len(k): the all-[GID] corner of
    :func:`add` — len(k)-1 Toffoli, len(k)-1 transient AND wires.
    """
    assert 1 <= len(a) <= len(k)
    add(b, k, a)


def gidney_sub(b, a, k):
    """k -= a mod 2^len(k) by complementation ([CDKM] subtraction remark:
    k - a = NOT(NOT k + a)). Same Toffoli cost as gidney_add.
    """
    for q in k:
        b.x(q)
    add(b, k, a)
    for q in k:
        b.x(q)


def cdkm_add_co(b, av, tv, cout, cin):
    """tv += av with carry-out XORed onto ``cout``: the all-[CDKM] corner
    of :func:`add` — 2 Toffoli per addend bit, zero fresh ancillas
    (``cin`` is a caller-provided |0> carry-in wire, restored). Chosen
    over the [GID] chain where qubits are scarcer than Toffolis at the
    call site's liveness peak.
    """
    add(b, list(tv), list(av), gl=0, cout=cout, cin=cin)


def mcx(b, ctrls, tgt):
    """Multi-controlled X via a [GID] temporary-AND ladder.

    ``ctrls`` is a sequence of ``(qubit, polarity)`` pairs; polarity-0 (open)
    controls are X-conjugated. Degenerate cases (0/1/2 controls) emit
    X/CX/CCX directly. n controls cost n-1 Toffoli (n-2 ladder ANDs released
    measurement-based + one CCX).
    """
    opens = [q for q, pol in ctrls if pol == 0]
    for q in opens:
        b.x(q)
    qs = [q for q, _ in ctrls]
    n = len(qs)
    if n == 0:
        b.x(tgt)
    elif n == 1:
        b.cx(qs[0], tgt)
    elif n == 2:
        b.ccx(qs[0], qs[1], tgt)
    else:
        ladder = [b.and_compute(qs[0], qs[1])]
        for q in qs[2:-1]:
            ladder.append(b.and_compute(ladder[-1], q))
        b.ccx(ladder[-1], qs[-1], tgt)
        for i in range(len(ladder) - 1, -1, -1):
            b.and_release(ladder[i], qs[0] if i == 0 else ladder[i - 1], qs[i + 1])
    for q in opens:
        b.x(q)


# ---- constant comparator and the [HRS] interval label ----------------------


def ge_const_flag(b, reg, bound):
    """Compute the flag [reg >= bound] for a classical ``bound``.

    The flag is the carry-out of reg + (2^n - bound), n = len(reg), with
    the constant folded in classically. The chain starts at the lowest
    set bit of the constant, whose carry is that ``reg`` wire itself.
    Each higher position is one [GID] temporary AND: a 0 constant bit
    ANDs the carry with the reg bit; a 1 bit ORs them, held as the
    complement (NOT carry AND NOT reg bit). Open controls are X
    conjugated around each AND.

    Returns ``(recs, flag, pol)``. ``pol`` False means ``flag`` holds the
    complement of [reg >= bound]. The flag can be a ``reg`` wire (a
    constant with one set bit). Release with :func:`ge_const_release`
    while ``reg`` still holds its value.

    Executed Toffoli = n - 1 - (index of the lowest set bit of
    2^n - bound). Requires 0 < bound < 2^n.
    """
    n = len(reg)
    assert 0 < bound < (1 << n), "bound outside (0, 2^n)"
    addend = (1 << n) - bound
    recs = []
    carry, pol = None, True
    for j in range(n):
        kj = (addend >> j) & 1
        xq = reg[j]
        if carry is None:
            if kj:
                carry = xq  # first set bit: carry = reg_j itself
            continue
        # kj = 0: carry' = carry AND reg_j.  kj = 1: carry' = carry OR
        # reg_j, held as its complement (NOT carry AND NOT reg_j).
        if kj == 0:
            conj = () if pol else (carry,)
        else:
            conj = (xq, carry) if pol else (xq,)
        for w in conj:
            b.x(w)
        t = b.and_compute(carry, xq)
        for w in conj:
            b.x(w)
        recs.append((t, carry, xq, conj))
        carry, pol = t, (kj == 0)
    return recs, carry, pol


def ge_const_release(b, recs):
    """Release a :func:`ge_const_flag` chain in reverse, measurement-based.

    Each release re-applies its AND's compute-time X conjugation, so the
    phase fixup acts on the computed values. 0 Toffoli.
    """
    for t, c1, c2, conj in reversed(recs):
        for w in conj:
            b.x(w)
        b.and_release(t, c1, c2)
        for w in conj:
            b.x(w)


def label_increment(b, reg, bound, i, lbl):
    """[HRS] label step ``i``: lbl ^= i ^ (i + 1) when reg >= bound.

    The step computes the :func:`ge_const_flag` flag, X-corrects a
    complemented flag, CX-loads the set bits of i ^ (i + 1) into ``lbl``
    under it, undoes the X, and releases the chain. When the bounds
    ascend strictly and step i uses bound i, the fired steps form a
    prefix 0..m-1. Their XORs telescope, so ``lbl`` ends at m, the count
    of bounds <= reg. The steps add 0 Toffoli beyond the comparator.
    """
    recs, flag, pol = ge_const_flag(b, reg, bound)
    if not pol:
        b.x(flag)
    step = i ^ (i + 1)
    for j in range(len(lbl)):
        if (step >> j) & 1:
            b.cx(flag, lbl[j])
    if not pol:
        b.x(flag)
    ge_const_release(b, recs)


def hrs_mul_rows(b, stage, V, W, X, shift):
    """[HRS] pre-truncated signed multiplier: W += V * X * 2^-shift.

    ``V`` is a two's-complement word of len(W) bits; ``X`` is unsigned.
    Row ``bit`` adds X shifted by bit - shift under control V[bit]: the
    low shift - bit bits of X drop, and the high bits wrap mod
    2^len(W). The top row subtracts, by an X sandwich on its target
    slice. Each row runs :func:`add` with ``ctrl=`` under the phase
    ``f"{stage}/b{bit:02d}"``.

    Executed Toffoli per row = len(T) - 1 + len(A) for the target slice
    T = W[max(0, bit - shift):] and the operand slice
    A = X[max(0, shift - bit):][:len(T)]; a row with an empty A emits
    nothing.
    """
    top = len(W) - 1
    for bit in range(len(W)):
        b.set_phase(f"{stage}/b{bit:02d}")
        T = W[max(0, bit - shift) :]
        A = X[max(0, shift - bit) :][: len(T)]
        if bit == top:
            for q in T:
                b.x(q)
            add(b, T, A, ctrl=V[bit])
            for q in T:
                b.x(q)
        else:
            add(b, T, A, ctrl=V[bit])


# ---- AND chains and ladders ------------------------------------------------


def and_chain(b, lits):
    """[GID] AND chain over ``lits``, a sequence of ``(qubit, pol)``
    pairs with at least two entries. A false ``pol`` ANDs the negated
    qubit (the Builder's ``pol=`` form, one X sandwich per AND).

    Returns ``(top, trail)``: ``top`` holds the AND of all literals and
    stays live until :func:`release_chain` (trail). n literals cost
    n - 1 Toffoli. Unlike :func:`mcx`, the call writes no target and
    keeps the chain live. Calls go through ``b.and_compute``, so a
    Builder subclass or pool that overrides it keeps working.
    """
    (q0, p0), (q1, p1) = lits[0], lits[1]
    t = b.and_compute(q0, q1, pol=(p0, p1))
    trail = [(t, q0, q1, (p0, p1))]
    for q, p in lits[2:]:
        t2 = b.and_compute(t, q, pol=(True, p))
        trail.append((t2, t, q, (True, p)))
        t = t2
    return t, trail


def release_chain(b, trail):
    """Release an :func:`and_chain` trail in reverse, measurement-based.
    0 Toffoli. The literals must still hold their compute-time values.
    """
    for t, c1, c2, pol in reversed(trail):
        b.and_release(t, c1, c2, pol=pol)


def eq_const(b, wires, k):
    """[wires == k] for a classical ``k`` (little-endian ``wires``): the
    :func:`and_chain` of the literals wire_i == bit i of k. Returns
    ``(top, trail)``. len(wires) - 1 Toffoli; needs two or more wires.
    """
    return and_chain(b, [(q, bool((k >> i) & 1)) for i, q in enumerate(wires)])


def and_ladder(b, ctrls):
    """[GID] AND ladder that reduces ``ctrls`` to one flag wire.

    ``ctrls`` is a sequence of ``(qubit, polarity)`` pairs. The call
    X-conjugates every open (polarity-0) control first, and the X stays
    on for the ladder's lifetime. It then ANDs the first two controls
    and each further control into the running AND. The caller emits the
    body under ``flag`` and then calls :func:`and_ladder_release`
    (rec), which releases the ANDs in reverse and removes the X.

    Returns ``(flag, rec)``. One control: ``flag`` is that control wire
    (after its X), no AND. No controls: ``flag`` is None. n controls
    cost n - 1 Toffoli; the release costs 0.
    """
    opens = [q for q, pol in ctrls if pol == 0]
    for q in opens:
        b.x(q)
    qs = [q for q, _ in ctrls]
    ladder = []
    if len(qs) >= 2:
        ladder.append(b.and_compute(qs[0], qs[1]))
        for q in qs[2:]:
            ladder.append(b.and_compute(ladder[-1], q))
    flag = ladder[-1] if ladder else (qs[0] if qs else None)
    return flag, (qs, ladder, opens)


def and_ladder_release(b, rec):
    """Release an :func:`and_ladder` in reverse, measurement-based, then
    remove the X on the open controls (in the compute order). 0 Toffoli.
    """
    qs, ladder, opens = rec
    for i in range(len(ladder) - 1, -1, -1):
        b.and_release(ladder[i], qs[0] if i == 0 else ladder[i - 1], qs[i + 1])
    for q in opens:
        b.x(q)


# ---- Booth pair-select columns ---------------------------------------------


def booth_columns(b, s1, s2, pairs, neg=None, pool=None):
    """Build the addend columns of a signed Booth radix-4 digit row.

    Column i = (s1 AND va) XOR (s2 AND vb) XOR neg for ``pairs[i] =
    (va, vb)``: ``s1`` selects the |d| = 1 operand bit and ``s2`` the
    |d| = 2 bit. Either entry of a pair may be None (that term drops;
    not both). ``neg`` (optional) is the digit sign, XORed on for the
    adder's one's-complement ``neg=`` input. Each column is one fresh
    [GID] AND plus one CCX when both terms are present.

    Returns ``(cols, reps)``: ``reps[i]`` lists the control pairs of
    column i, for :func:`booth_columns_release`. ``pool`` duck-types the
    Builder's ``and_compute``/``recycle`` (see
    :class:`~uncompute.gadgets.wires.WirePool`); default the Builder.
    """
    w = pool if pool is not None else b
    cols, reps = [], []
    for va, vb in pairs:
        if va is not None:
            tq = w.and_compute(s1, va)
            rep = [(s1, va)]
            if vb is not None:
                b.ccx(s2, vb, tq)
                rep.append((s2, vb))
        else:
            tq = w.and_compute(s2, vb)
            rep = [(s2, vb)]
        if neg is not None:
            b.cx(neg, tq)
        cols.append(tq)
        reps.append(rep)
    return cols, reps


def booth_columns_release(b, cols, reps, neg=None, pool=None):
    """[GID] measurement-based release of :func:`booth_columns`, newest
    column first: undo the ``neg`` XOR, measure the column in the X
    basis, and repair the phase kick with one conditioned CZ per control
    pair. 0 Toffoli. The controls must still hold their build-time
    values; pass the same ``neg`` and ``pool`` as the build.
    """
    w = pool if pool is not None else b
    for tq, rep in zip(reversed(cols), reversed(reps)):
        if neg is not None:
            b.cx(neg, tq)
        m = b.hmr(tq)
        for u, v in rep:
            b.cz(u, v, cond=m)
        w.recycle(tq)
