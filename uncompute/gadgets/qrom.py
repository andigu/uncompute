"""QROM address iteration gadgets.

Source
------
Babbush et al. (2018), arXiv:1805.03662 — Sec. III.A, Figs. 5-7 (merged
sawtooth unary iteration), Fig. 4; Gidney (2018), arXiv:1709.06648
(measurement-based temporary-AND erasure).
"""


def walk(b, bits, ctrl, bit, seg_lo, seg_hi, lo, hi, body):
    """Sawtooth unary iteration (arXiv:1805.03662 Figs. 5-7) over the addresses
    of [lo, hi) that lie in the tree segment [seg_lo, seg_hi) spanned by
    bits[0..bit].

    ``bits`` is the little-endian address register; call with
    ``walk(b, bits, None, len(bits) - 1, 0, 1 << len(bits), lo, hi, body)``
    to iterate the window [lo, hi), and ``body(ctrl, addr)`` fires once per
    in-window address with ``ctrl`` the qubit that is 1 exactly for ``addr``.

    ``ctrl`` is the live iteration ancilla ("is the address in this segment"),
    or None above the first two-sided split (uncontrolled iteration).
    One AND per two-sided internal node; the 0->1 branch step is the Fig. 6
    CNOT merge; AND erasure is the measurement-based Fig. 4 uncompute.
    One-sided nodes consume nothing (the paper's control-removal rule).

    Executed Toffoli: one per two-sided internal node — for the full window
    [0, 2**n) that is 2**n - 2; peak ancillas: one per live tree level.
    """
    if bit < 0:
        body(ctrl, seg_lo)
        return
    mid = (seg_lo + seg_hi) >> 1
    q = bits[bit]
    go_left = lo < mid  # some in-window address has this bit 0
    go_right = hi > mid  # some in-window address has this bit 1
    if go_left and go_right:
        if ctrl is None:
            # Uncontrolled top split: the address bit itself is the subtree control.
            b.x(q)
            walk(b, bits, q, bit - 1, seg_lo, mid, lo, hi, body)
            b.x(q)
            walk(b, bits, q, bit - 1, mid, seg_hi, lo, hi, body)
        else:
            b.x(q)
            t = b.and_compute(ctrl, q)  # t = ctrl AND (bit == 0); 1 Toffoli
            b.x(q)
            walk(b, bits, t, bit - 1, seg_lo, mid, lo, hi, body)
            b.cx(ctrl, t)  # Fig. 6 merge: t -> ctrl AND (bit == 1)
            walk(b, bits, t, bit - 1, mid, seg_hi, lo, hi, body)
            b.and_release(t, ctrl, q)  # Fig. 4 measurement-based erasure; 0 Toffoli
    elif go_left:
        walk(b, bits, ctrl, bit - 1, seg_lo, mid, lo, hi, body)
    else:
        walk(b, bits, ctrl, bit - 1, mid, seg_hi, lo, hi, body)


# ---------------------------------------------------------------------------
# SelectSwap (QROAM) lookup — [LKS] Low et al. (2018), arXiv:1812.00954
# Fig. 1(c); [BGMMB] Berry et al. (2019), arXiv:1902.02134 App. B (clean-ancilla QROAM,
# Toffoli ceil(d/lam) + bw*(lam-1)) and App. C (measurement-based unlookup,
# Toffoli ceil(d/ku) + ku).
# ---------------------------------------------------------------------------


def berry_ku(d):
    """[BGMMB] App. C: argmin over power-of-two k' of ceil(d/k') + k'."""
    best = min((-(-d // k) + k, k) for k in (1 << i for i in range(d.bit_length())))
    return best[1]


def swap_perm(lam):
    """Word index sitting in each slot after the Swap network, per low-address
    value (mirrors the emitted controlled-swap tree exactly; [LKS] Fig. 1(c)).
    """
    log_lam = lam.bit_length() - 1
    perm = []
    for lv in range(lam):
        slots = list(range(lam))
        for i in range(log_lam):
            if (lv >> i) & 1:
                for s in range(0, lam, 1 << (i + 1)):
                    slots[s], slots[s + (1 << i)] = slots[s + (1 << i)], slots[s]
        assert slots[0] == lv
        perm.append(slots)
    return perm


def xor_outcomes(terms):
    """Outcome-algebra parity of a list of measurement outcomes ([BGMMB]
    App. C free classical fixup processing, as uncompute outcome algebra).

    Returns None when the parity is identically zero (no fixup gate needed),
    the outcome itself for a single term (conditioning never consumes a
    handle), else a fresh accumulator m0 ^ m1 extended in place (acc ^= mk).
    """
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    acc = terms[0] ^ terms[1]
    for mk in terms[2:]:
        acc ^= mk
    return acc


def xor_word(b, ctrl, word, target):
    """XOR the classical ``word`` onto the little-endian register
    ``target`` under ``ctrl``: one CX per set bit (arXiv:1805.03662
    Fig. 10 — data as presence/absence of CNOTs), plain X writes when
    ``ctrl`` is None. Clifford-only, not in the TQ-cost; target bits above
    ``word``'s length are untouched.
    """
    t = 0
    while word:
        if word & 1:
            if ctrl is None:
                b.x(target[t])
            else:
                b.cx(ctrl, target[t])
        word >>= 1
        t += 1


def _word_terms(word, outcomes):
    """Run the :func:`xor_word` scan on the measurement record instead of the
    register: the outcome handles at ``word``'s set bits, in bit order.
    """
    terms = []
    t = 0
    while word:
        if word & 1:
            terms.append(outcomes[t])
        word >>= 1
        t += 1
    return terms


def phase_fixup(b, addr, d, ku, g, *, claim_set=None, phase=None):
    """[BGMMB] App. C Fig. "unlookupc": phase-fix the measurement kickback via
    a k'-window one-hot network + unary iteration; Clifford-only reverse
    routing (Figs. qrominv/qromclif). `g` maps drawable address -> fixup bit;
    non-drawable addresses get no fixup.

    ``claim_set``/``phase`` are pure-bookkeeping hooks (no stream ops
    either way): the window joins the named ownership set for its whole
    life (per-wire releases before each measurement-based erasure), and
    the stages get ``{phase}:prep`` / ``:fixup`` / ``:unprep`` marks.
    """
    log_ku = ku.bit_length() - 1
    lu, hu = addr[:log_ku], addr[log_ku:]
    n_fix = -(-d // ku)
    assert n_fix <= 1 << len(hu)
    if phase is not None:
        b.set_phase(f"{phase}:prep")
    u = b.qubits(ku)
    if claim_set is not None:
        b.claim(claim_set, u)
    b.x(u[0])
    for i in reversed(range(log_ku)):  # route the 1 from slot 0 to slot l'
        for s in range(0, ku, 1 << (i + 1)):
            b.cswap(lu[i], u[s], u[s + (1 << i)])

    def fixup_leaf(ctrl, hv):
        base = hv << log_ku
        for lv in range(ku):
            gb = g.get(base | lv)
            if gb is not None:
                b.cz(ctrl, u[lv], cond=gb)  # H CX H = CZ, emitted native

    if phase is not None:
        b.set_phase(f"{phase}:fixup")
    walk(b, hu, None, len(hu) - 1, 0, 1 << len(hu), 0, n_fix, fixup_leaf)

    if phase is not None:
        b.set_phase(f"{phase}:unprep")
    for i in range(log_ku):  # Clifford-only reverse routing
        for s in range(0, ku, 1 << (i + 1)):
            v = u[s + (1 << i)]
            b.cx(v, u[s])
            if claim_set is not None:
                b.release(claim_set, [v])
            b.and_release(v, lu[i], u[s])
    b.x(u[0])
    if claim_set is not None:
        b.release(claim_set, [u[0]])
    b.recycle(u[0])


def selectswap_lookup(
    b, addr, d, word, bw, lam, ku, valid_addrs, name, out=None, *, claim_set=None, phase=None
):
    """[BGMMB] App. B clean-ancilla QROAM ([LKS] Fig. 1(c)): select + swap +
    copy into ``out`` (a fresh bw-bit register when None — the task's
    own accumulator may be passed instead), then App. C swap-register erase.
    Returns the register holding h XOR word(addr).

    ``claim_set``/``phase`` are pure-bookkeeping hooks (no stream ops
    either way): the swap registers join the named ownership set from
    allocation to their measurement-based retirement, and both hooks
    forward to the :func:`phase_fixup` window. ``out`` is never claimed —
    it is (or becomes) the caller's register.
    """
    log_lam = lam.bit_length() - 1
    lbits, hbits = addr[:log_lam], addr[log_lam:]
    n_chunks = -(-d // lam)
    assert n_chunks <= 1 << len(hbits)

    r = [b.qubits(bw) for _ in range(lam)]
    if claim_set is not None:
        for j in range(lam):
            b.claim(claim_set, r[j])

    def select_leaf(ctrl, hv):
        base = hv << log_lam
        for j in range(lam):
            xor_word(b, ctrl, word[base | j], r[j])

    with b.phase(f"{name}:select"):
        walk(b, hbits, None, len(hbits) - 1, 0, 1 << len(hbits), 0, n_chunks, select_leaf)

    with b.phase(f"{name}:swap"):
        for i in range(log_lam):
            for s in range(0, lam, 1 << (i + 1)):
                hi_slot = s + (1 << i)
                for t in range(bw):
                    b.cswap(lbits[i], r[s][t], r[hi_slot][t])

    with b.phase(f"{name}:copy"):
        if out is None:
            out = b.qubits(bw)
        for t in range(bw):
            b.cx(r[0][t], out[t])

    with b.phase(f"{name}:unswap"):
        # X-basis-measure the whole swap register, record the mask.
        m = [[b.hmr(r[j][t]) for t in range(bw)] for j in range(lam)]
        if claim_set is not None:
            for j in range(lam):
                b.release(claim_set, r[j])
        for j in reversed(range(lam)):
            for q in reversed(r[j]):
                b.recycle(q)
        # Classical fixup table over drawable addresses: slot j held
        # word(base | perm[lv][j]) at measurement time.
        perm = swap_perm(lam)
        g = {}
        for av in valid_addrs:
            hv, lv = av >> log_lam, av & (lam - 1)
            base = hv << log_lam
            prow = perm[lv]
            terms = []
            for j in range(lam):
                terms.extend(_word_terms(word[base | prow[j]], m[j]))
            gb = xor_outcomes(terms)
            if gb is not None:
                g[av] = gb
        phase_fixup(b, addr, d, ku, g, claim_set=claim_set, phase=phase)

    return out


def pruned_walk(
    b, bits_msb, lo, span, leaf, nonzero, uniform=None, ctrl=None, mark=None, pool=None
):
    """Data-pruned recursive QROM descent (the gaussian lookup walk).

    Complements :func:`walk`: ``walk`` iterates a dense address WINDOW and
    drops controls at one-sided nodes because the window implies the
    address bit; here pruning is by table CONTENT — ``nonzero(lo, span)``
    False skips a subtree outright, ``uniform(lo, span)`` True fires one
    ``leaf(lo, ctrl)`` for the whole span (its contract: every address in
    the span carries the same nonzero payload) — so the address bit is
    never implied, and every visited internal node with a live control
    costs one temporary AND, one-sided nodes included (an X sandwich
    selects a left-only child).

    ``bits_msb`` is the address register msb-first. ``leaf(lo, ctrl)``
    receives ``ctrl=None`` only above the first split of an uncontrolled
    walk (write unconditionally). ``mark(lo, span)`` hooks subtree entries
    (phase marks / profiling). ``pool`` as in :func:`~uncompute.gadgets.arith.add`.

    Executed Toffoli = #visited internal nodes with a live control.
    """
    w = pool if pool is not None else b
    if not nonzero(lo, span):
        return
    if mark is not None:
        mark(lo, span)
    if span == 1:
        leaf(lo, ctrl)
        return
    if uniform is not None and ctrl is not None and uniform(lo, span):
        leaf(lo, ctrl)
        return
    hb = bits_msb[0]
    half = span >> 1
    ln = nonzero(lo, half)
    rn = nonzero(lo + half, half)
    if ctrl is None:
        if ln:
            b.x(hb)
            pruned_walk(b, bits_msb[1:], lo, half, leaf, nonzero, uniform, hb, mark, pool)
            b.x(hb)
        if rn:
            pruned_walk(b, bits_msb[1:], lo + half, half, leaf, nonzero, uniform, hb, mark, pool)
        return
    if ln and rn:
        anc = w.and_compute(ctrl, hb)
        pruned_walk(b, bits_msb[1:], lo + half, half, leaf, nonzero, uniform, anc, mark, pool)
        b.cx(ctrl, anc)
        pruned_walk(b, bits_msb[1:], lo, half, leaf, nonzero, uniform, anc, mark, pool)
        b.cx(ctrl, anc)
        w.and_release(anc, ctrl, hb)
    elif rn:
        anc = w.and_compute(ctrl, hb)
        pruned_walk(b, bits_msb[1:], lo + half, half, leaf, nonzero, uniform, anc, mark, pool)
        w.and_release(anc, ctrl, hb)
    else:
        b.x(hb)
        anc = w.and_compute(ctrl, hb)
        pruned_walk(b, bits_msb[1:], lo, half, leaf, nonzero, uniform, anc, mark, pool)
        w.and_release(anc, ctrl, hb)
        b.x(hb)


def erase_register(
    b, reg, addr, d, word, ku, valid_addrs, name, free=False, *, claim_set=None, phase=None
):
    """[BGMMB] App. C measurement unlookup of a register whose contents equal
    word(addr) on every drawable state: X-basis measure it, then phase-fix
    over the (still live) address register. With ``free=False`` the
    register's qubits are NOT freed (they may belong to the task
    schema; `hmr` demolishes each to |0>); ``free=True`` returns them to the
    free-list before the fixup allocates its one-hot window.

    ``claim_set``/``phase`` are pure-bookkeeping hooks (no stream ops
    either way), forwarded to the :func:`phase_fixup` window — the only
    wires this call allocates. ``reg``'s own set membership stays the
    caller's business.
    """
    with b.phase(name):
        mu = [b.hmr(q) for q in reg]
        if free:
            for q in reversed(reg):
                b.recycle(q)
        g = {}
        for av in valid_addrs:
            gb = xor_outcomes(_word_terms(word[av], mu))
            if gb is not None:
                g[av] = gb
        phase_fixup(b, addr, d, ku, g, claim_set=claim_set, phase=phase)
