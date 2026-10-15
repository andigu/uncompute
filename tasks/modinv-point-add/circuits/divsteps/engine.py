"""modinv-point-add, divstep family: the engine behind the family's
circuit files — in-place point addition |P> -> |P+Q> with the field
inversions computed by divstep walks over windowed rows.

Two engines share one walk and arithmetic stack:

    hmr_main       the measurement-erase engine (fill.py, wide.py)
    staged_main    the staged-decode engine (j5.py .. j9.py)

circuit.py spells the discovered circuit point's phase flow flat on the Hmr
engine's pieces (HmrDivstepInv, MulMachine, emit_tail).  Each other
circuit file sets its dials through <engine>_main(cfg, entry_file).
README.md carries the frontier table; gen_constants.py regenerates
constants.json.

Sections, in load order: decode payload grouping and the base gadget
vocabulary; the windowed walk (rows, gadget kit, schedule, home
pools); the level-j >= 5 decode layers; the two walk variants
(HmrWalk, StagedWalk); the k-block lazy multiplier family; the fused
square (FusedSquare, subtract_square); the shared engine layer
(DivstepInvBase, MulMachine, the emit tail); the Hmr engine; the
staged engine.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys
from collections.abc import Callable

from uncompute import Builder
from uncompute.gadgets import (
    WirePool,
    add,
    and_chain,
    eq_const,
    gidney_add,
    gidney_sub,
    maj,
    maj_inv,
    release_chain,
    uma_inv,
)
from uncompute.testing import assert_metrics, emit_and_report, load_contract, load_module

HERE = pathlib.Path(__file__).resolve().parent

N = 256
P_MOD = int(load_contract(__file__)["p"], 16)  # the grader asserts contract.json

# Shared arithmetic/template layer (../_shared/luo_core.py): the
# Gidney-carry helper vocabulary, the classical-bit network layer
# (ClsBits + const_modp_bit_specs), the modular
# add/double/const-add/negate flows, and the k-block lazy
# modular-multiplication family.  LC.init(LUO_CORE_DEPS) (after this
# circuit's mul dispatch) binds luo_core's circuit-local callables back.
LC = load_module(HERE.parent / "_shared" / "luo_core.py", "divsteps_luo_core")

# The searched and derived tables (gen_constants.py writes them).
_CONSTANTS = json.loads((HERE / "constants.json").read_bytes())

# Decode forms for the production prime, parsed to (const, terms) with
# terms [("ag"|"af", i)]: b = const XOR window bits. Derived
# algebraically from the anchored-ladder theorem by gen_constants.py.
# The j = 2 stratum is the proven XOR law b = gv'[1] ^ fv'[3], emitted
# directly.
FORMS_P = {}
for _key, _rec in _CONSTANTS["j34_forms"]["forms"].items():
    _j = int(_key[1])
    _f3 = int(_key.split("=")[1])
    _terms = tuple((_t[:2], int(_t[4])) for _t in _rec["terms"])
    FORMS_P[(_j, _f3)] = (_rec["const"], list(_terms))


# ===========================================================================
# decode payload shaping: grouped ESOP over the form strata
# ===========================================================================


def group_payload(forms, j):
    """Group the (j, f3) stratum forms into payload items
    [(cubes, term)]: each cube is a literal dict {f3 bit: value} over
    the three f-window bits, each term ("ag", i) | ("af", i) |
    ("const",). The emitted circuit computes b as an XOR of cubes AND
    window bits; strata absent from `forms` are don't-cares in the
    cover. Payload shape at the production forms: 8 chain ANDP + 10
    payload CCX per evaluation.
    """
    universe = sorted(f3 for (jj, f3) in forms if jj == j)
    items = []
    allterms = sorted({t for (jj, _), (_, ts) in forms.items() if jj == j for t in ts})
    for term in allterms:
        onset = [f3 for f3 in universe if term in forms[(j, f3)][1]]
        items.append((_cube_cover(onset, universe), term))
    conset = [f3 for f3 in universe if forms[(j, f3)][0] == 1]
    if conset:
        items.append((_cube_cover(conset, universe), ("const",)))
    return items


def _cube_cover(onset, universe):
    """Minimal ESOP (XOR of cubes) matching `onset` on the care set;
    outside it, don't-care. Three variables; a cube is a literal dict
    ({} = always-1). Cost order: (total literals, cube count). The
    cover must be an exact XOR cover: two cubes overlapping on a care
    point cancel there, so containment is checked by XOR, never by OR.
    Exhaustive over cube subsets to size 5.
    """
    onset = set(onset)
    care = set(universe)
    if not onset:
        return []
    cubes_all = []
    for nlit in range(4):
        for varset in itertools.combinations(range(3), nlit):
            for vals in range(1 << nlit):
                cube = {v: (vals >> k) & 1 for k, v in enumerate(varset)}
                cov = frozenset(
                    m for m in range(8) if all((m >> v) & 1 == bb for v, bb in cube.items())
                )
                cubes_all.append((cube, cov))
    for k in range(1, 6):
        best = None
        for combo in itertools.combinations(range(len(cubes_all)), k):
            acc = set()
            for ci in combo:
                acc ^= cubes_all[ci][1]
            if (acc & care) == onset:
                cost = (sum(len(cubes_all[ci][0]) for ci in combo), k)
                if best is None or cost < best[0]:
                    best = (cost, [cubes_all[ci][0] for ci in combo])
        if best is not None:
            return best[1]
    raise AssertionError("no ESOP cover")


# ===========================================================================
# gadget layer
# ===========================================================================


class GadgetBase:
    """Gadget vocabulary over a Builder: Gidney AND pairs, polarity
    chains, controlled ripple adds, controlled shifts. Emission only;
    costs are counted by the Builder.

    local: the shared gadget adders keep carries live across the sum
    fan; this walk's per-step adds release every interior carry inside
    the ripple (2W-1 Toffoli, zero retained transients), which is what
    holds the walk-phase transient count at the frame above.
    """

    def __init__(self, b):
        self.b = b

    def andp(self, c1, c2, pol=(True, True)):
        return self.b.and_compute(c1, c2, pol=pol)

    def andm(self, t, c1, c2, pol=(True, True)):
        self.b.and_release(t, c1, c2, pol=pol)

    # Thin wrappers over the shared chain gadgets (the ladder call
    # sites use these method names too).
    def and_chain(self, wires, pols):
        return and_chain(self.b, list(zip(wires, pols)))

    def release_chain(self, trail):
        release_chain(self.b, trail)

    def zero_test(self, wires):
        return eq_const(self.b, wires, 0)

    def eq_const(self, wires, const):
        return eq_const(self.b, wires, const)

    def ctrl_add(self, ctl, src, tgt, cin=None):
        """tgt += src + cin under ctl. 2W-1 Toffoli; carries die inside."""
        b = self.b
        W = len(tgt)
        assert len(src) == W
        carries = []
        trail: list[tuple] = []  # ("plain", c, a, t) | ("maj", c, a, t, prev)
        prev = cin
        for i in range(W - 1):
            a, t = src[i], tgt[i]
            if prev is None:
                c = self.andp(a, t)
                trail.append(("plain", c, a, t))
            else:
                b.cx(prev, a)
                b.cx(prev, t)
                c = self.andp(a, t)
                b.cx(prev, c)
                trail.append(("maj", c, a, t, prev))
            carries.append(c)
            prev = c
        for i in range(W - 1, -1, -1):
            cin_i = cin if i == 0 else carries[i - 1]
            if cin_i is not None:
                b.cx(cin_i, src[i])
            b.ccx(ctl, src[i], tgt[i])
            if cin_i is not None:
                b.cx(cin_i, src[i])
            if i >= 1:
                ent = trail[i - 1]
                if ent[0] == "plain":
                    _, c, a, t = ent
                    self.andm(c, a, t)
                else:
                    _, c, a, t, pv = ent
                    b.cx(pv, c)
                    self.andm(c, a, t)
                    b.cx(pv, a)
                    b.cx(pv, t)

    def ctrl_sub(self, ctl, src, tgt, cin=None):
        for w in tgt:
            self.b.x(w)
        self.ctrl_add(ctl, src, tgt, cin=cin)
        for w in tgt:
            self.b.x(w)

    def ctrl_inc(self, ctl, tgt, dec=False):
        """tgt += ctl (dec: tgt -= ctl) on the shared all-[GID] adder:
        the control is the one-bit operand.
        """
        if dec:
            gidney_sub(self.b, [ctl], tgt)
        else:
            gidney_add(self.b, [ctl], tgt)

    # controlled in-place rotations (arithmetic shifts on signed rows)
    def shift_down(self, ctl, reg):
        b = self.b
        for i in range(len(reg) - 1):
            b.cswap(ctl, reg[i], reg[i + 1])
        b.ccx(ctl, reg[-2], reg[-1])

    def shift_down_inv(self, ctl, reg):
        b = self.b
        b.ccx(ctl, reg[-2], reg[-1])
        for i in range(len(reg) - 2, -1, -1):
            b.cswap(ctl, reg[i], reg[i + 1])

    def shift_up(self, ctl, reg):
        b = self.b
        b.ccx(ctl, reg[-2], reg[-1])
        for i in range(len(reg) - 2, -1, -1):
            b.cswap(ctl, reg[i], reg[i + 1])

    def shift_up_inv(self, ctl, reg):
        b = self.b
        for i in range(len(reg) - 1):
            b.cswap(ctl, reg[i], reg[i + 1])
        b.ccx(ctl, reg[-2], reg[-1])


# ===========================================================================
# windowed divstep walk: rows, gadget kit, schedule, home pools
#
# The divstep walk with its branch record (the R stack) and branch
# decode, run over SCHEDULED row widths.
#
# ROW FRAME. Each signed row is a list of low positions (ids) plus one
# dedicated sign wire; the value is two's complement of width W+1 with
# the top bit on the sign wire (v = low - sign*2^W). The window tables
# bound abs(v).bit_length() <= W at every in-schedule step (the
# quantile fit). Window transitions are Clifford and exact: grow appends a
# position initialized to a sign copy (CX), shrink removes the top
# position by the inverse CX + free — legal exactly when the value
# fits the smaller width, which is the window table's guarantee; the
# strict simulator fails closed on any violation (free of a nonzero
# wire).
#
# PAIR WINDOWS. The sigma-step row swap exchanges CONTENT between f
# and g (per-lane data; a relabel is not addressable), so the pair
# pads to one shared width max(WF, WG)(t) for the swap; likewise fv/gv
# pad to max(WFV, WGV)(t). This is the compile-time premium of a
# could-host stream over any per-lane fill statistic: the stream must
# allocate window homes for every lane at once.
#
# HOMES. The g row's low ids ride the x register (the walk consumes x
# in place and the reverse walk regrows it); the f row's low ids ride
# the free data word (A at walk-A, the HMR-freed y word at the other
# legs). As the fg pair shrinks, its freed word homes are claimed by
# the growing v pair (fv behind g on the x word, gv behind f on the
# free word); v growth beyond the freed frontier draws a fresh
# straddle id, and a per-step repack (swaps, Clifford) migrates
# straddle residents into freed band slots so the straddle census
# tracks the pointwise band deficit instead of its running maximum.
# The packing is compile-time (window tables only) and per-step
# disjoint by construction; at the walk's end fv sits mostly on the x
# word — the parked row the plateau mul consumes in place.
#
# ADDS. Row adds are in-place-carry controlled CDKM/Cuccaro ripples
# (MAJ up, ctrl-UMA down; one carry wire, no residual) instead of
# allocated Gidney carry chains, so walk-phase transients
# stay in the tens instead of the row-width class. The controlled add
# masks the addend through the control (CX fan), adds, and unmasks —
# executed == emitted, no data-dependent existence. A hybrid low block
# converts carries to clean ANDs exactly up to the census ceiling
# (ctrl_add_rows below).
#
# POSITION READS ABOVE THE WINDOW. The decode and j-flag chains read
# fixed value-bit positions (fv bits 2..4+, gv bits 1..7, f bits
# 1..4). Bit k of a two's complement value equals the sign bit for
# every k >= W, so pos() substitutes the sign wire there. The
# substituted chains stay legal (distinct wires; a flag that is
# identically zero on such steps is the correct value of that flag)
# and are billed as armed — could-host discipline is unchanged.
#
# STEP LABELS. The step comments below tag the parts of one divstep.
# The forward leg runs F1-F17 at step t:
#   F1   sigma root s = (z >= 0) AND g odd: the swap condition.
#   F2   under s, swap f with g and fv with gv, and set z to -z-1.
#   F3   branch bit b, a copy of g[0].
#   F4   under b, add f into g and fv into gv (under s AND b, set
#        g = f - g and gv = fv - gv instead).
#   F5   z += 1 on live lanes.
#   F6   shift g down one bit and fv up one bit, then shrink the rows
#        to the step's exit widths.
#   F7   freeze detection: a live lane with g = 0 sets frz and writes
#        its step offset t+1-Tmin into tst.
#   F8   class flag c3 = live AND NOT s AND z <= 0. The row state
#        does not reveal the branch of these steps.
#   F9   level flags on v2(fv) for c3 steps; the deepest flag selects
#        the steps whose b goes to the record.
#   F10-F12 are unused: the forward leg holds b, so it decodes nothing.
#   F13  push b onto the record R on selected steps.
#   F14  release the flags and zero tests of F8 and F9.
#   F15  release s against fv[1].
#   F16  release the freeze witness of F7 (armed steps only).
#   F17  release b by an X-basis measurement (HMR); the reverse leg
#        applies the phase fixup at the same step.
# The reverse leg runs R1-R17 at step t and undoes the forward step:
#   R1   unfreeze: a frozen lane whose tst equals t+1-Tmin clears frz
#        and tst; fireR marks that lane.
#   R2   sigma witness s', recomputed from the post-step state.
#   R3   class flag c3 as in F8, with an extra NOT fireR literal.
#   R4   level flags as in F9.
#   R5   pop the record into b' on selected steps.
#   R6   decode the branch of the other c3 steps.
#   R7   assemble b' from the decode, s' and the unfreeze term.
#   R8-R10 are unused: without F10-F12 there is no decoded bit to
#        check against the forward b.
#   R11  phase fixup for the F17 measurement of b.
#   R12  release the decode and the flags.
#   R13  release fireR.
#   R14  undo F6, F5 and F4: un-shift, z -= 1, un-add.
#   R15  release b' against g[0].
#   R16  undo the F2 swap.
#   R17  release s' against the restored sigma root.
#
# ORDERING RULES. The record, the decode and the freeze logic obey
# four rules. The comments at each site state the rule that applies.
#   An AND trail releases before any of its literals changes.
#   A decode releases before the row bits it reads change.
#   A lane that freezes at step t pushes no record at t, and the
#   reverse leg pops none at t.
#   The sigma witness s' gates the unfreeze term in b'.
# ===========================================================================

# ===========================================================================
# windowed signed rows
# ===========================================================================


class WRow:
    """A signed row at scheduled width: ids[0..W-1] low positions plus
    the dedicated sign wire (two's complement of width W+1). Ids come
    from pre-allocated home bands (HomePool) and stay allocated for
    the whole walk — grow seeds a |0> band id with a sign copy,
    shrink returns the id to |0> and gives it back to the pool. The
    census therefore counts band residency (the window-envelope
    realization), never per-lane fill.
    """

    def __init__(self, b, pool, prefs, ids, sign, tape=None, replay=None):
        self.b = b
        self.pool = pool
        self.prefs = prefs
        self.ids = list(ids)
        self.sign = sign
        # Forward-leg band journal: every id return pushes its band
        # name; a take pops the journal when non-empty and draws from
        # THAT band, falling back to the straddle when a repack
        # consumed the named band's slots (the pad needs the overflow
        # then anyway). Journal empty => net-growth take: preference
        # order. The journal is PER-ROW: a band-global journal lets
        # one row's take pair with another row's give and park rows
        # onto foreign words (fv ids onto the mul target word).
        #
        # Mirror discipline: every fwd take/give (and repack move)
        # appends to the shared tape; the reverse leg does not re-run
        # the journal — it pops the tape and inverts each entry
        # id-exactly (a band draw pops the same free-list top its
        # mirror give pushed; a straddle rebirth is a fresh alloc, the
        # id number immaterial since row ops address the ids list).
        # State equality at every mirror point is by construction, and
        # the walk ends with g on the x word in its original order; a
        # pairing fault surfaces at build time (a named band with no
        # free id, a tape mismatch) or as a nonzero-wire free.
        self.journal = []
        self.tape = tape if tape is not None else []
        self.replay = replay if replay is not None else [False]

    def _take(self):
        if self.replay[0]:
            op, row, nm = self.tape.pop()
            assert op == "g" and row is self, ("tape take mismatch", op)
            return self.pool.draw(nm)
        if self.journal:
            nm = self.journal.pop()
            if nm != HomePool.DYN and not self.pool.free.get(nm):
                nm = HomePool.DYN
            w = self.pool.draw(nm)
        else:
            w = self.pool.take(*self.prefs)
        self.tape.append(("t", self, self.pool.owner[w]))
        return w

    def _give(self, w):
        if self.replay[0]:
            op, row, nm = self.tape.pop()
            assert op == "t" and row is self, ("tape give mismatch", op)
            got = self.pool.give(w)
            assert got == nm, ("tape band mismatch", got, nm)
            return
        nm = self.pool.give(w)
        self.journal.append(nm)
        self.tape.append(("g", self, nm))

    @property
    def W(self):
        return len(self.ids)

    def pos(self, k):
        """Value bit k: a low position, or the sign wire for k >= W."""
        return self.ids[k] if k < len(self.ids) else self.sign

    def wires(self):
        return self.ids + [self.sign]

    def grow(self, seed_sign=True):
        """Append one position (sign extension: new id = sign copy),
        or a raw |0> id (seed_sign=False, the shift-up headroom).
        """
        w = self._take()
        if seed_sign:
            self.b.cx(self.sign, w)
        self.ids.append(w)
        return w

    def shrink(self):
        """Remove the top position (value provably fits: window law;
        the strict simulator fails closed on a nonzero residue).
        """
        w = self.ids.pop()
        self.b.cx(self.sign, w)
        self._give(w)

    def grow_to(self, W):
        while len(self.ids) < W:
            self.grow()

    def shrink_to(self, W):
        while len(self.ids) > W:
            self.shrink()

    def drop_raw_all(self):
        """Return every id without the sign-extension CX (closer path:
        the ids are already |0> but the sign wire may be live).
        """
        while self.ids:
            self._give(self.ids.pop())

    def grow_raw_to(self, W):
        """Take ids without sign seeding (closer_inv path)."""
        while len(self.ids) < W:
            self.ids.append(self._take())


# ===========================================================================
# gadget kit (windowed, in-place carry: shared CDKM cells)
# ===========================================================================


class G(GadgetBase):
    """The base gadget vocabulary (GadgetBase: andp/andm AND pairs,
    polarity chains, zero_test/eq_const, the controlled counter
    inc/dec), extended with the windowed additions below: hybrid
    controlled row adds, unconditional inc/dec, and the blocked zero
    test (wide rows use ztest_blocked instead of zero_test for the
    transient cap). ANDP/ANDM discipline is GadgetBase's (every chain's
    trail releases before its literals mutate).
    """

    # ---- controlled in-place row add, hybrid spelling: tgt += ctl *
    # src over equal wire lists (ids + sign, two's complement mod
    # 2^len — exact signed add when the result fits, which the window
    # schedule guarantees). Carries are computed unconditionally and
    # fully uncomputed; only the sum writes are controlled. The low k
    # positions carry via clean ANDs on fresh transient wires (1
    # Toffoli each, measurement-released on the way down: 2/position
    # total), the rest MAJ-ripple on the shared carry wire with
    # ctrl-UMA down (3/position), and the TOP position writes its sum
    # directly against the running carry (1 Toffoli — its carry-out
    # is dead mod 2^len). k adapts to the census headroom at emission
    # (qcap − live, qcap = the census ceiling), so the transients
    # never move the peak — by construction; k = 0 degrades to the
    # ripple gadget. cin=wire seeds the carry (the sigma-subtract
    # +1); the seed is returned by the down pass.
    def _hybrid_k(self, w, qcap):
        if qcap is None:
            return 0
        return max(0, min(qcap - self.b.active_qubits, w - 1))

    def ctrl_add_rows(self, ctl, src_w, tgt_w, carry, cin=None, qcap=None):
        b = self.b
        w = len(src_w)
        assert len(tgt_w) == w
        k = self._hybrid_k(w, qcap)
        if cin is not None:
            b.cx(cin, carry)
        cw = [carry]
        for i in range(k):  # low: AND carries up
            t_, s_ = tgt_w[i], src_w[i]
            b.cx(cw[-1], t_)
            b.cx(cw[-1], s_)
            nxt = b.and_compute(t_, s_)
            b.cx(cw[-1], nxt)
            cw.append(nxt)
        hi = cw[-1]
        for i in range(k, w - 1):  # high: MAJ ripple up
            maj(b, tgt_w[i], src_w[i], hi)
        b.cx(hi, src_w[w - 1])  # top: sum only
        b.ccx(ctl, src_w[w - 1], tgt_w[w - 1])
        b.cx(hi, src_w[w - 1])
        for i in range(w - 2, k - 1, -1):  # high: ctrl-UMA down
            t_, s_ = tgt_w[i], src_w[i]
            b.ccx(t_, s_, hi)
            b.cx(hi, t_)
            b.ccx(ctl, s_, t_)
            b.cx(hi, s_)
        for i in range(k - 1, -1, -1):  # low: release + write
            t_, s_ = tgt_w[i], src_w[i]
            b.cx(cw[i], cw[i + 1])
            b.and_release(cw[i + 1], t_, s_)
            b.cx(cw[i], t_)
            b.ccx(ctl, s_, t_)
            b.cx(cw[i], s_)
        if cin is not None:
            b.cx(cin, carry)

    def ctrl_sub_rows(self, ctl, src_w, tgt_w, carry, cin=None, qcap=None):
        """Exact adjoint of ctrl_add_rows (controlled subtract): the
        add's emission reversed, AND mints and releases swapped.
        """
        b = self.b
        w = len(src_w)
        assert len(tgt_w) == w
        k = self._hybrid_k(w, qcap)
        if cin is not None:
            b.cx(cin, carry)
        cw = [carry] + [None] * k
        for i in range(k):  # low: unwrite + mint
            t_, s_ = tgt_w[i], src_w[i]
            b.cx(cw[i], s_)
            b.ccx(ctl, s_, t_)
            b.cx(cw[i], t_)
            cw[i + 1] = b.and_compute(t_, s_)
            b.cx(cw[i], cw[i + 1])
        hi = cw[k]
        for i in range(k, w - 1):  # high: UMA-inv up
            t_, s_ = tgt_w[i], src_w[i]
            b.cx(hi, s_)
            b.ccx(ctl, s_, t_)
            b.cx(hi, t_)
            b.ccx(t_, s_, hi)
        b.cx(hi, src_w[w - 1])  # top: unsum
        b.ccx(ctl, src_w[w - 1], tgt_w[w - 1])
        b.cx(hi, src_w[w - 1])
        for i in range(w - 2, k - 1, -1):  # high: MAJ-inv down
            maj_inv(b, tgt_w[i], src_w[i], hi)
        for i in range(k - 1, -1, -1):  # low: release down
            t_, s_ = tgt_w[i], src_w[i]
            b.cx(cw[i], cw[i + 1])
            b.and_release(cw[i + 1], t_, s_)
            b.cx(cw[i], t_)
            b.cx(cw[i], s_)
        if cin is not None:
            b.cx(cin, carry)

    # Unconditional inc/dec (pre-freeze steps: the live control is a
    # constant-1, so the carry chain roots at the register's low bit:
    # reg[1:] += reg[0] on the shared all-[GID] adder, then X on reg[0]).
    def inc(self, reg):
        add(self.b, reg[1:], [reg[0]])
        self.b.x(reg[0])

    def dec(self, reg):
        b = self.b
        for w in reg:
            b.x(w)
        self.inc(reg)
        for w in reg:
            b.x(w)

    # ---- blocked zero test: flag <- [all wires 0], live transients
    # bounded by `block` + ceil(W/block) keepers (the k-block keeper
    # pattern on clean ANDs; erases are and_release, free). Returns
    # (flag, state) -- release with zero_test_free(state) which
    # recomputes the keeper anchors (ANDP) and releases everything
    # (the recompute is the uncompute price; ~2W Toffoli per full
    # test cycle).
    def ztest_blocked(self, wires, block=24):
        b = self.b
        assert wires
        keepers = []  # (keeper, block_wires, prev_keeper)
        prev = None
        i = 0
        while i < len(wires):
            blk = wires[i : i + block]
            lits = ([prev] if prev is not None else []) + list(blk)
            pols = ([True] if prev is not None else []) + [False] * len(blk)
            if len(lits) == 1:
                cur, trail = lits[0], []
                # single literal, no chain: keeper = fresh AND of the
                # literal with itself is illegal; copy via CX pair
                cur = b.qubit()
                b.x(wires[i])
                b.cx(wires[i], cur)
                b.x(wires[i])
                keepers.append(("copy", cur, blk, prev))
                prev = cur
                i += len(blk)
                continue
            cur, trail = self.and_chain(lits, pols)
            # keep the top, release the chain internals (free)
            keeper = b.qubit()
            b.cx(cur, keeper)
            self.release_chain(trail)
            keepers.append(("chain", keeper, blk, prev))
            prev = keeper
            i += len(blk)
        return prev, keepers

    def ztest_free(self, keepers):
        """Uncompute the keeper ladder top-down: recompute each
        block's chain only to the penultimate literal and retire the
        keeper FREE against (chain top, last literal) — its state is
        exactly that AND, so the measurement release saves one
        Toffoli per block over the full-recompute + CX-off form.
        """
        b = self.b
        for kind, keeper, blk, prev in reversed(keepers):
            if kind == "copy":
                b.x(blk[0])
                b.cx(blk[0], keeper)
                b.x(blk[0])
                b.recycle(keeper)
                continue
            lits = ([prev] if prev is not None else []) + list(blk)
            pols = ([True] if prev is not None else []) + [False] * len(blk)
            if len(lits) == 2:
                b.and_release(keeper, lits[0], lits[1], pol=(pols[0], pols[1]))
                continue
            cur, trail = self.and_chain(lits[:-1], pols[:-1])
            b.and_release(keeper, cur, lits[-1], pol=(True, pols[-1]))
            self.release_chain(trail)


# ===========================================================================
# window schedule + home plan (compile-time)
# ===========================================================================


class Schedule:
    """Per-step width tables (value bits) + derived pair/headroom pads
    + the home plan. Tables are the deployed quantile fit
    (constants.json section windows_q).
    """

    def __init__(self, S, WF, WG, WFV, WGV, n):
        self.S = S
        self.WF, self.WG = list(WF), list(WG)
        self.WFV, self.WGV = list(WFV), list(WGV)
        self.n = n
        for t in range(S):
            # emission sanity: pair envelopes are usable as schedules
            assert self.WFV[t] >= 1 and self.WGV[t] >= 1

    def pre(self, tbl, t):
        if t == 0:
            return {id(self.WF): self.n, id(self.WG): self.n, id(self.WFV): 1, id(self.WGV): 1}[
                id(tbl)
            ]
        return tbl[t - 1]

    def fg_pair(self, t):
        """Width both f and g pad to for the sigma swap at step t."""
        return max(self.pre(self.WF, t), self.pre(self.WG, t))

    def v_pair(self, t):
        return max(self.pre(self.WFV, t), self.pre(self.WGV, t))


class HomePool:
    """Home bands for row ids. Fixed bands ("xw", "fw") are existing
    word wire lists (the io/data words the rows ride; always
    allocated). The "dyn" band allocates fresh ids on demand and
    frees on return, so the measured census tracks max-concurrency.
    LIFO discipline everywhere; the walk's mirror-exact width
    sequencing makes reverse-leg id assignment reproduce the forward
    one (pops == reversed pushes).
    """

    DYN = "dyn"

    def __init__(self, b, bands, claim_dyn="walk/straddle"):
        self.b = b
        self.free = {nm: list(ids) for nm, ids in bands.items()}
        self.owner = {}
        self.claim_dyn = claim_dyn
        self.dyn_live = 0
        self.dyn_peak = 0
        for nm, ids in bands.items():
            for w in ids:
                self.owner[w] = nm

    def draw(self, nm):
        if nm == self.DYN:
            w = self.b.qubit()
            self.b.claim(self.claim_dyn, [w])
            self.owner[w] = self.DYN
            self.dyn_live += 1
            self.dyn_peak = max(self.dyn_peak, self.dyn_live)
            return w
        return self.free[nm].pop()

    def take(self, *prefs):
        for nm in prefs:
            if nm == self.DYN or self.free.get(nm):
                return self.draw(nm)
        raise RuntimeError(f"home pools exhausted: {prefs}")

    def give(self, w):
        nm = self.owner[w]
        if nm == self.DYN:
            self.b.release_recycle(self.claim_dyn, [w])
            del self.owner[w]
            self.dyn_live -= 1
            return nm
        self.free[nm].append(w)
        return nm

    def rebind_band(self, nm, ids):
        """Point a fixed band at another word (walk-B/C/D: the free
        word changes). Every previous id must have been returned.
        """
        old = [w for w, o in self.owner.items() if o == nm]
        assert len(self.free[nm]) == len(old), (
            f"rebind {nm}: {len(old) - len(self.free[nm])} ids still out"
        )
        for w in old:
            del self.owner[w]
        self.free[nm] = list(ids)
        for w in ids:
            self.owner[w] = nm


# ===========================================================================
# the windowed walk
# ===========================================================================


class WalkWBase:
    """The divstep walk over scheduled row widths.

    Homes: g rides xw (the x register's word), f rides fw (the free
    data word), fv fills behind g on xw, gv behind f on fw, straddle
    band for the overflow. The forward walk consumes x in place; the
    reverse walk regrows it. The record R, the decode and the freeze
    logic read row bits through WRow.pos() and follow the step labels
    and ordering rules in the section header above.

    Parameters: prm the walk parameters (Wz/Cr/S/p + Tmin/Qcap; see
    hmr_composed_params), sched a
    Schedule, xw/fw the two word wire-lists (len n), extra the
    straddle band. rider: optional dict(legs, stage) — legs maps
    'fwd'/'rev' to 'halve'|'double', and stage(mode, alive) emits one
    alive-conditioned mod-p halve (or double) of the caller's rider
    register, riding each armed step adjacent to the
    alive-conditioned shifts.
    """

    # ---- state __init__ sets (the walk variants add their decode tables)
    b: Builder
    p: int
    S: int
    Cr: int
    Wt: int
    sched: Schedule
    spans: list[int]
    pool: HomePool
    signs: list[int]
    z: list[int]
    R: list[int]
    rider: dict | None
    _tape: list[tuple]
    _replay: list[bool]
    tst: list[int]  # armed window only: _arm_alloc/_arm_free
    # the rows: live from init_rows to uninit_rows
    f: WRow
    g: WRow
    fv: WRow
    gv: WRow

    def __init__(
        self,
        b,
        prm,
        forms,
        spans,
        sched,
        xw,
        fw,
        extra,
        phase_fwd="walk/step",
        phase_rev="walk/un-step",
    ):
        self.b = b
        self.prm = prm
        self.forms = forms
        self.grouped = {j: group_payload(forms, j) for j in (3, 4)}
        self.spans = spans
        self.sched = sched
        self.phase_fwd = phase_fwd
        self.phase_rev = phase_rev
        self.Wz = prm["Wz"]
        self.Cr = prm["Cr"]
        self.S = prm["S"]
        self.p = prm["p"]
        # Tmin: the pre-freeze pin (a quantile of the tstop census;
        # P(tstop <= Tmin) is a booked spend). Steps t < Tmin carry no freeze
        # machinery (F7/F16/R1/R13) and run the pre-freeze spellings:
        # frz is provably 0 on every lane there, so alive controls are
        # constant-1 and drop. A lane that does freeze early (the
        # priced Tmin escape) walks on undetected — its g stays 0
        # (self-gating), its fv/z overrun the window and z caps — and
        # fails its shot through the existing contained escape
        # channels. Tmin = 0 (the default) arms every step.
        self.Tmin = prm.get("Tmin", 0)
        # tst stores the freeze index OFFSET (t+1-Tmin, values in
        # [1, S-Tmin]; 0 = never fired), so its width is set by the
        # armed window, not the schedule.
        self.Wt = max(1, (self.S - self.Tmin).bit_length())
        # Qcap: the peak-census ceiling the hybrid adders fill to
        # (None = plain ripple adds).
        self.qcap = prm.get("Qcap")
        n = sched.n
        assert len(xw) == n and len(fw) == n
        A = b.qubits
        self.pool = HomePool(b, {"xw": list(xw), "fw": list(fw), "extra": list(extra)})
        # the mirror tape: fwd take/give/repack entries, popped and
        # inverted by the reverse leg (see WRow)
        self._tape = []
        self._replay = [False]
        self.signs = A(4)  # f, g, fv, gv sign wires
        self.z = A(self.Wz)
        # tst/frz are armed-window machinery: allocated at the first
        # armed forward step (t = Tmin) and freed once the reverse leg
        # un-runs it — below Tmin they carry no data (the Tmin
        # theorem), so the pre-freeze arena census does not hold them.
        self.tst = None  # ty: ignore[invalid-assignment]  # disarmed sentinel
        self.frz = None
        # R grows with the span reach (fill before step t's push is at
        # most min(spans[t], t), a theorem), so early steps carry no
        # idle record cells; the reverse leg trims it back down.
        self.R = []
        self.o_fwd = [None] * self.S
        self.rider = None
        # rows are constructed at seed time (init_rows)
        self.f = self.g = self.fv = self.gv = None  # ty: ignore[invalid-assignment]  # pre-seed sentinel
        b.claim("walk/signs", list(self.signs))
        b.claim("walk/counters", list(self.z))

    def step_fwd(self, t):
        raise NotImplementedError

    def step_rev(self, t):
        raise NotImplementedError

    # ---- machine wires owned by the walk (band ids excluded: xw/fw
    # are register/data words, extra is claimed by the caller; tst/
    # frz self-manage inside the armed window)
    def machine_wires(self):
        return list(self.signs) + list(self.z) + list(self.R)

    def release_claims(self):
        b = self.b
        b.release("walk/signs", list(self.signs))
        b.release("walk/counters", list(self.z))
        b.release("record/R", list(self.R))

    # ---- armed-window machinery lifetime (tst/frz) and the record
    # brigade's span-curve sizing
    def _arm_alloc(self):
        b = self.b
        self.tst = [b.qubit() for _ in range(self.Wt)]
        self.frz = b.qubit()
        b.claim("walk/tstfrz", list(self.tst) + [self.frz])

    def _arm_free(self):
        b = self.b
        b.release_recycle("walk/tstfrz", list(self.tst) + [self.frz])
        self.tst = None  # ty: ignore[invalid-assignment]  # disarmed sentinel
        self.frz = None

    def _r_fit(self, t):
        """Grow (fwd) / trim (rev) the record brigade to the step's
        reach + 1 cells. Cells beyond the reach are |0> on every
        in-schedule lane; an R-overflow lane leaves dirt there and
        fails through the existing priced record channel.
        """
        b = self.b
        need = min(min(self.spans[t], t) + 1, self.Cr)
        while len(self.R) < need:
            w = b.qubit()
            b.claim("record/R", [w])
            self.R.append(w)
        while len(self.R) > need:
            w = self.R.pop()
            b.release_recycle("record/R", [w])

    # ---- straddle repack. After a step's exit shrinks (and after the
    # closer), every straddle-resident row id whose own band has a
    # free slot migrates there: one swap (Clifford, the freed band
    # slot is |0>), and the vacated straddle wire returns to the
    # allocator. Without it the straddle census ratchets at the
    # running MAXIMUM of the band deficit (an id drawn when the band
    # was full keeps its wire for its whole life — the Hall bound on
    # no-movement home schemes, met exactly by the journal), while the
    # machinery moments that bind the walk census run at parked
    # widths where the pointwise deficit is 8-22 wires lower. The
    # scan order is fixed (xw: fv then g; fw: gv then f; ascending
    # position), so the move list is schedule-determined; the reverse
    # leg pops it from the tape and inverts each move.
    def _repack(self):
        b = self.b
        for nm, rows in (("xw", (self.fv, self.g)), ("fw", (self.gv, self.f))):
            for row in rows:
                for i in range(len(row.ids)):
                    if self.pool.owner.get(row.ids[i]) != HomePool.DYN:
                        continue
                    if not self.pool.free[nm]:
                        break
                    old = row.ids[i]
                    w = self.pool.draw(nm)
                    b.swap(old, w)
                    row.ids[i] = w
                    self.pool.give(old)
                    self._tape.append(("m", row, i, nm))

    def _repack_inv(self):
        b = self.b
        while self._tape and self._tape[-1][0] == "m":
            _, row, i, nm = self._tape.pop()
            w = self.pool.draw(HomePool.DYN)
            b.swap(row.ids[i], w)
            got = self.pool.give(row.ids[i])
            assert got == nm, ("repack band mismatch", got, nm)
            row.ids[i] = w

    # ------------------------------------------------------------- seed
    def init_rows(self):
        """Construct the row objects on the home bands. g's initial
        window is the full word (the x value already ON xw); f seeds
        the modulus onto fw; fv/gv seed at width 1. Preference order:
        the paired word band, then the fixed extra band, then the
        dynamic straddle (fresh ids, census = max concurrency).
        """
        b, n = self.b, self.sched.n
        P_ = self.pool
        DYN = HomePool.DYN
        # g: the x word wires, in place, exactly the word order
        gids = [P_.take("xw") for _ in range(n)]
        gids.reverse()  # take() pops from the tail; restore 0..n-1
        self.g = WRow(
            b, P_, ("xw", "extra", DYN), gids, self.signs[1], tape=self._tape, replay=self._replay
        )
        self.f = WRow(
            b,
            P_,
            ("fw", "extra", DYN),
            [P_.take("fw") for _ in range(n)][::-1],
            self.signs[0],
            tape=self._tape,
            replay=self._replay,
        )
        for i in range(n):
            if (self.p >> i) & 1:
                b.x(self.f.ids[i])
        self.fv = WRow(
            b,
            P_,
            ("xw", "extra", DYN),
            [P_.take("extra", DYN)],
            self.signs[2],
            tape=self._tape,
            replay=self._replay,
        )
        self.gv = WRow(
            b,
            P_,
            ("fw", "extra", DYN),
            [P_.take("extra", DYN)],
            self.signs[3],
            tape=self._tape,
            replay=self._replay,
        )
        b.x(self.gv.ids[0])

    def uninit_rows(self):
        """Inverse of init_rows at the walk's end state (reverse leg
        complete: g = x on xw, f = p on fw, fv = 0/W1, gv = 1/W1).
        """
        b, n = self.b, self.sched.n
        assert not self._tape, f"mirror tape not consumed: {len(self._tape)}"
        b.x(self.gv.ids[0])
        for i in range(n):
            if (self.p >> i) & 1:
                b.x(self.f.ids[i])
        for row in (self.fv, self.gv):
            self.pool.give(row.ids.pop())
        for row in (self.f, self.g):
            while row.ids:
                self.pool.give(row.ids.pop())
        self.f = self.g = self.fv = self.gv = None  # ty: ignore[invalid-assignment]  # torn-down sentinel

    # ---------------------------------------------------- alive shifts
    # The g down-shift is UNCONDITIONED: it is self-gating. On a live
    # lane bit 0 is provably 0 (the F4 add made g even) and the shift
    # is the step's halve; on a frozen lane g = 0 exactly — every id
    # and the sign wire (the freeze condition the ztest verifies) — so
    # plain swaps and the sign-copy refill act as the identity. The
    # alive control the fv shift needs is redundant here, and dropping
    # it turns the whole shift Clifford (swaps + CX, unscored).
    def _shift_down_g(self, row):
        b = self.b
        ids = row.ids
        for i in range(len(ids) - 1):
            b.swap(ids[i], ids[i + 1])
        b.cx(row.sign, ids[-1])

    def _shift_down_g_inv(self, row):
        b = self.b
        ids = row.ids
        b.cx(row.sign, ids[-1])
        for i in range(len(ids) - 2, -1, -1):
            b.swap(ids[i], ids[i + 1])

    def _shift_up(self, alive, row):
        """row <- row << 1 on alive lanes (pre-shift value fits one id
        below the top: the window law); top id = sign copy there, so
        the clear + rotate is exact.
        """
        b = self.b
        ids = row.ids
        b.ccx(alive, row.sign, ids[-1])
        for i in range(len(ids) - 2, -1, -1):
            b.cswap(alive, ids[i], ids[i + 1])

    def _shift_up_inv(self, alive, row):
        b = self.b
        ids = row.ids
        for i in range(len(ids) - 1):
            b.cswap(alive, ids[i], ids[i + 1])
        b.ccx(alive, row.sign, ids[-1])

    # Pre-freeze spellings (t < Tmin): the walk treats every lane as
    # alive below the pre-freeze pin, so the alive control is a
    # constant-1 and the whole shift is Clifford (swaps + CX).
    def _shift_up_free(self, row):
        b = self.b
        ids = row.ids
        b.cx(row.sign, ids[-1])
        for i in range(len(ids) - 2, -1, -1):
            b.swap(ids[i], ids[i + 1])

    def _shift_up_free_inv(self, row):
        b = self.b
        ids = row.ids
        for i in range(len(ids) - 1):
            b.swap(ids[i], ids[i + 1])
        b.cx(row.sign, ids[-1])

    # ------------------------------------------------------ rider stage
    def _rider_stage(self, mode):
        """One alive-conditioned mod-p halve (mode='halve') or double
        (mode='double') of the rider register — the scale-fix stage
        riding this step, armed unconditionally on the rider span
        (could-host). The spelling is the caller's stage closure (the
        windowed fold class in the Hmr engine, the exact
        comparator-killed form in the staged engine); the alive
        control (~frz, X-conjugated here) is passed in. Stages are
        armed only on the data window [Tmin, S): the Tmin fixed
        halvings live in the seam (the halve-Horner absorption plus
        appended fixed halves — DivstepInvBase.row_mul), so a walk leg
        carries exactly the tstop-Tmin data-bearing stages.
        """
        b = self.b
        assert self.rider is not None
        b.x(self.frz)
        self.rider["stage"](mode, self.frz)
        b.x(self.frz)

    def emit_fwd(self):
        for t in range(self.S):
            self.step_fwd(t)

    def emit_rev(self):
        self._replay[0] = True
        for t in range(self.S - 1, -1, -1):
            self.step_rev(t)

    # -------------------------------------------------------- closers
    # Endpoint state (walk exit windows): f = +-1 (two's complement:
    # sign s_f, every id = s_f... ids [1,1..] at f=-1, [1,0..] at
    # f=+1; id0 = 1 always since |f| = 1 odd), g = 0 (0 ids), and
    # gv = f*p exactly (the determinant pin at g = 0: f*gv = +p).
    #
    # closer: (1) f collapse -- CX(s_f) fan over the ids, X(id0),
    #   CX(s_f, id0): f=+1 (s=0) ids [1,0..] -> fan idle -> X ->
    #   [0..]; f=-1 (s=1) ids [1,1..] -> fan -> [0,0..] -> X ->
    #   [1,0..] -> CX -> [0..]. All f ids |0>; s_f stays LIVE (the
    #   endpoint sign, consumed by the caller's sign fix).
    # (2) gv cancel -- gv = f*p two's complement (W ids + sign): the
    #   branch pair (p, 2^n - p) is affine in s_gv, so an X/CX fan
    #   zeroes it (see _gv_fan; zero Toffoli, zero doors). s_gv then
    #   equals s_f exactly (gv = f*p); one CX(s_f, s_gv) kills it.
    def _gv_fan(self, sgv):
        """Emit the gv cancel/rebuild, which is pure Clifford: gv = f*p exactly
        (the determinant pin), so at width n its ids are p_i on the
        +p branch and (2^n - p)_i on the -p branch — affine in the
        one live wire s_gv: ids_i = p_i XOR s_gv*d_i with d = p XOR
        (2^n - p). An X fan on p's bits plus a CX(s_gv) fan on d's
        bits zeroes both branches (self-inverse).
        """
        b = self.b
        n = self.sched.n
        pv = self.p % (1 << n)
        dv = pv ^ (((1 << n) - pv) % (1 << n))
        for i in range(n):
            if (dv >> i) & 1:
                b.cx(sgv, self.gv.ids[i])
            if (pv >> i) & 1:
                b.x(self.gv.ids[i])

    def closer(self):
        b = self.b
        n = self.sched.n
        sf, sgv = self.f.sign, self.gv.sign
        # width normalization (all values provably fit: g = 0, |f| =
        # 1, |gv| = p at every lane — the endpoint theorem)
        self.g.shrink_to(0)
        self.f.shrink_to(1)
        # (1) f collapse: id0 = 1 on BOTH branches (|f| = 1 odd)
        b.x(self.f.ids[0])
        self.f.drop_raw_all()
        # (2) gv cancel at width n (|±p| fits n ids + sign)
        self.gv.shrink_to(n)
        self._gv_fan(sgv)
        b.cx(sf, sgv)
        self.gv.drop_raw_all()
        self._repack()

    def closer_inv(self):
        b = self.b
        self._replay[0] = True
        self._repack_inv()
        sf, sgv = self.f.sign, self.gv.sign
        self.gv.grow_raw_to(self.sched.n)
        b.cx(sf, sgv)
        self._gv_fan(sgv)
        self.gv.grow_to(self.sched.WGV[self.S - 1])
        self.f.grow_raw_to(1)
        b.x(self.f.ids[0])
        self.f.grow_to(self.sched.WF[self.S - 1])
        self.g.grow_to(self.sched.WG[self.S - 1])

    # ---------------------------------------------- dirty row borrow
    def dirty_ids(self, k):
        """k row ids as a borrowed-dirty bank (live data, restored by
        the borrowing gadget). Compile-time: the row id lists are
        schedule-known at every step.
        """
        ids = self.fv.ids + self.gv.ids + self.f.ids + self.g.ids
        assert len(ids) >= k, (len(ids), k)
        return ids[:k]

    # ---- step blocks both walk variants run verbatim (they differ
    # in where the F5 z step sits and in the flag/decode layer)
    def _fwd_add(self, g, t):
        """Run the forward step entry through F4: arm at t = Tmin, pad
        the pairs, mint the sigma root, swap the rows, mint the branch
        bit and run the controlled row adds. Returns (s, bw).
        """
        b = self.b
        sc = self.sched
        if t == self.Tmin:
            self._arm_alloc()
        # width entry: pair pads for the sigma swap
        Wm = sc.fg_pair(t)
        Vm = sc.v_pair(t)
        self.f.grow_to(Wm)
        self.g.grow_to(Wm)
        self.fv.grow_to(Vm)
        self.gv.grow_to(Vm)
        # F1 sigma root (frozen lanes have g[0] = 0, so s = 0)
        s = g.andp(self.z[self.Wz - 1], self.g.pos(0), pol=(False, True))
        # F2 conditional row swap + z negate-conjugation
        for fw_, gw_ in zip(self.f.wires(), self.g.wires()):
            b.cswap(s, fw_, gw_)
        for fw_, gw_ in zip(self.fv.wires(), self.gv.wires()):
            b.cswap(s, fw_, gw_)
        for i in range(self.Wz):
            b.cx(s, self.z[i])
        # F3 branch-bit mint
        bw = b.qubit()
        b.cx(self.g.pos(0), bw)
        # F4 branch arithmetic: add headroom, negate-conjugated
        # controlled adds ([CDKM] cells, shared gadget)
        self.g.grow_to(Wm + 1)
        self.f.grow_to(Wm + 1)
        self.gv.grow_to(Vm + 1)
        self.fv.grow_to(Vm + 1)
        u = g.andp(s, bw)
        for w_ in self.g.wires():
            b.cx(u, w_)
        for w_ in self.gv.wires():
            b.cx(u, w_)
        # the add carry is a step transient (|0> outside the two
        # calls), so it is not carried in the arena census
        car = b.qubit()
        g.ctrl_add_rows(bw, self.f.wires(), self.g.wires(), car, cin=u, qcap=self.qcap)
        g.ctrl_add_rows(bw, self.fv.wires(), self.gv.wires(), car, cin=u, qcap=self.qcap)
        b.recycle(car)
        g.andm(u, s, bw)
        return s, bw

    def _z_step(self, g, early, dec=False):
        """F5 z += 1 (dec: the reverse leg's z -= 1) on live lanes.
        Pre-freeze every lane is live, so the step is unconditional.
        """
        b = self.b
        if early:
            if dec:
                g.dec(self.z)
            else:
                g.inc(self.z)
            return
        b.x(self.frz)
        g.ctrl_inc(self.frz, self.z, dec=dec)
        b.x(self.frz)

    def _fwd_shift_exit(self, t, early):
        """F6 shifts (g down self-gating, fv up on live lanes), then the
        step's exit windows and the straddle repack.
        """
        b = self.b
        sc = self.sched
        self._shift_down_g(self.g)
        if early:
            self._shift_up_free(self.fv)
        else:
            b.x(self.frz)
            self._shift_up(self.frz, self.fv)
            b.x(self.frz)
        self.g.shrink_to(sc.WG[t])
        self.f.shrink_to(sc.WF[t])
        self.fv.shrink_to(sc.WFV[t])
        self.gv.shrink_to(sc.WGV[t])
        self._repack()

    def _s_death(self, g, s, fire):
        """F15 s death: s ^= fv[1] & ~(frz & ~fire). Pre-freeze
        (fire=None) the guard is identically 1 (frz = 0), leaving a
        plain CX.
        """
        b = self.b
        if fire is None:
            b.cx(self.fv.pos(1), s)
            b.recycle(s)
            return
        a1 = g.andp(self.frz, fire, pol=(True, False))
        # s holds fv[1] & ~a1 here — retire it free against
        # the live pair instead of minting a second witness
        g.andm(s, self.fv.pos(1), a1, pol=(True, False))
        g.andm(a1, self.frz, fire, pol=(True, False))

    def _rev_unshift(self, t, early):
        """R14 un-run, first part: regrow the fwd step's add widths and
        undo the F6 shifts. Width calls are the exact mirror of the
        forward step's sequence (here: reverse of the fwd exit shrinks
        g,f,fv,gv), and the id assignment is mirror-by-construction:
        every take/give pops the tape the forward leg wrote and inverts
        it id-exactly (WRow), so the reverse leg reproduces the forward
        wire assignment state-by-state and the walk ends with g on the
        x word in its original order.
        """
        b = self.b
        sc = self.sched
        Wm = sc.fg_pair(t)
        Vm = sc.v_pair(t)
        self.gv.grow_to(Vm + 1)
        self.fv.grow_to(Vm + 1)
        self.f.grow_to(Wm + 1)
        self.g.grow_to(Wm + 1)
        if early:
            self._shift_up_free_inv(self.fv)
        else:
            b.x(self.frz)
            self._shift_up_inv(self.frz, self.fv)
            b.x(self.frz)
        self._shift_down_g_inv(self.g)

    def _rev_unadd(self, g, t, sp_, bpw, early):
        """R14 un-adds, R15 b' death, R16 un-swap, R17 s' death and the
        exit windows; disarm at t = Tmin.
        """
        b = self.b
        sc = self.sched
        Wm = sc.fg_pair(t)
        Vm = sc.v_pair(t)
        u = g.andp(sp_, bpw)
        car = b.qubit()
        g.ctrl_sub_rows(bpw, self.fv.wires(), self.gv.wires(), car, cin=u, qcap=self.qcap)
        g.ctrl_sub_rows(bpw, self.f.wires(), self.g.wires(), car, cin=u, qcap=self.qcap)
        b.recycle(car)
        for w_ in self.gv.wires():
            b.cx(u, w_)
        for w_ in self.g.wires():
            b.cx(u, w_)
        g.andm(u, sp_, bpw)
        # R15 b' death by the pre-add witness (g[0] = b here)
        b.cx(self.g.pos(0), bpw)
        b.recycle(bpw)
        # R16 un-swap (inverse of F2) at the F2 widths. The add
        # headroom drops first, in the mirror order of the F4
        # grows (fwd: g,f,gv,fv -> mirror: fv,gv,f,g)
        self.fv.shrink_to(Vm)
        self.gv.shrink_to(Vm)
        self.f.shrink_to(Wm)
        self.g.shrink_to(Wm)
        for i in range(self.Wz):
            b.cx(sp_, self.z[i])
        for fw_, gw_ in zip(self.fv.wires(), self.gv.wires()):
            b.cswap(sp_, fw_, gw_)
        for fw_, gw_ in zip(self.f.wires(), self.g.wires()):
            b.cswap(sp_, fw_, gw_)
        # R17 s' death vs the pre-state sigma root (pre-freeze the
        # ~frz literal is constant-1 and drops): sp' holds the
        # root's value on every lane, so it retires free against
        # the live literals (armed: one witness AND, not two)
        if early:
            g.andm(sp_, self.z[self.Wz - 1], self.g.pos(0), pol=(False, True))
        else:
            m1 = g.andp(self.z[self.Wz - 1], self.g.pos(0), pol=(False, True))
            g.andm(sp_, m1, self.frz, pol=(True, False))
            g.andm(m1, self.z[self.Wz - 1], self.g.pos(0), pol=(False, True))
        # exit: the (t-1)-step windows, mirror of the fwd entry
        # grows (fwd: f,g,fv,gv -> mirror: gv,fv,g,f)
        self.gv.shrink_to(sc.pre(sc.WGV, t))
        self.fv.shrink_to(sc.pre(sc.WFV, t))
        self.g.shrink_to(sc.pre(sc.WG, t))
        self.f.shrink_to(sc.pre(sc.WF, t))
        if t == self.Tmin:
            self._arm_free()


# ===========================================================================
# level-j decode layers, j >= 5 (PPRM payloads)
#
# Level-j decode layers, j >= 5: the PPRM payload of the
# anchored-ladder branch function at level j, derived from the
# theorem's second form at import cost only.
#
# At a class-3 post-state with j = v2(fv'), writing q = fv' >> j,
#
#     b = bit_{j-1}( p * (f' * q)^{-1} mod 2^j ),
#
# a function of (f' mod 2^j, q mod 2^j) — the decode identity of README.md,
# unconditional over every odd p and every start. Beyond j = 4 the
# f' strata are no longer affine in the q window (the affine-form
# structure of the j34 layers stops there), so these layers ship the
# function's full positive-polarity Reed-Muller form instead:
# monomials over the 2(j-1) window bits
#
#     fbit i  <->  f'.pos(i),    i in 1..j-1
#     qbit i  <->  fv'.pos(j+i), i in 1..j-1  (q bit i).
#
# Emission (StagedWalk: _decode_apply / pprm_level) roots
# every monomial's AND chain at the level rung and shares chain
# prefixes across monomials along the precomputed traversal program.
# At j = 5 and the prime p: 26 monomials of degree <= 3 with
# 12 interior trie nodes, so that layer costs 12 ANDP + 26 payload CCX
# per evaluation.
#
# decode5_derive(p, j) recomputes the monomial list from scratch for
# any odd p.
# ===========================================================================


def decode5_derive(p, j=5):
    """PPRM monomials [(fset, qset), ...] of the level-j branch
    function for odd modulus p; fset/qset are ascending tuples of
    window-bit indices (1-based). A () , () entry is the constant-1
    monomial (emitted as a plain CX off the j flag).
    """
    assert p % 2 == 1
    M = 1 << j
    nw = j - 1  # free window bits per operand
    nv = 2 * nw
    N = 1 << nv
    tt = [0] * N
    for fh in range(1 << nw):
        fp = (1 + 2 * fh) % M
        for qh in range(1 << nw):
            q = (1 + 2 * qh) % M
            tt[(fh << nw) | qh] = ((p * pow((fp * q) % M, -1, M)) % M) >> (j - 1) & 1
    anf = tt[:]
    for i in range(nv):
        for m in range(N):
            if m & (1 << i):
                anf[m] ^= anf[m ^ (1 << i)]
    monos = []
    for m in range(N):
        if anf[m]:
            fset = tuple(i + 1 for i in range(nw) if (m >> (nw + i)) & 1)
            qset = tuple(i + 1 for i in range(nw) if (m >> i) & 1)
            monos.append((fset, qset))
    # canonical order: by literal tuple, so shared prefixes are adjacent
    monos.sort()
    return monos


# ===========================================================================
# the Hmr engine's walk (j-flag variant)
#
# Windowed divstep walk, j-flag variant: the forward/reverse step
# ladders over the shared machinery above (rows, gadget kit, schedule,
# home pools, and the common WalkWBase methods).
# This module's own layer is the resident j-flag ladder (_jflags), the
# anchor-cached decode (_decode/_pred_death), and the step spellings
# that drive them.
# ===========================================================================


class HmrWalk(WalkWBase):
    # ---------------------------------------------------- decode layer
    # (the level flags, the decode and the record port; row bits
    # read through WRow.pos)
    def _jflags(self, g, c3):
        """Compute the j anchor flags. a1 = c3 AND ~fv[2] is the XOR c3 ^ j2
        (j2 and a1 partition c3), and a2 = a1 ^ j3 likewise — both
        mint by CX pairs (Clifford), not ANDs.
        """
        b = self.b
        j2 = g.andp(c3, self.fv.pos(2))
        a1 = b.qubit()
        b.cx(c3, a1)
        b.cx(j2, a1)
        j3 = g.andp(a1, self.fv.pos(3))
        a2 = b.qubit()
        b.cx(a1, a2)
        b.cx(j3, a2)
        j4 = g.andp(a2, self.fv.pos(4))
        j5p = b.qubit()
        b.cx(a2, j5p)
        b.cx(j4, j5p)
        return {"j2": j2, "a1": a1, "j3": j3, "a2": a2, "j4": j4, "j5p": j5p}

    def _jflags_release(self, g, jf, c3):
        b = self.b
        b.cx(jf["j4"], jf["j5p"])
        b.cx(jf["a2"], jf["j5p"])
        b.recycle(jf["j5p"])
        g.andm(jf["j4"], jf["a2"], self.fv.pos(4))
        b.cx(jf["j3"], jf["a2"])
        b.cx(jf["a1"], jf["a2"])
        b.recycle(jf["a2"])
        g.andm(jf["j3"], jf["a1"], self.fv.pos(3))
        b.cx(jf["j2"], jf["a1"])
        b.cx(c3, jf["a1"])
        b.recycle(jf["a1"])
        g.andm(jf["j2"], c3, self.fv.pos(2))

    def _decode(self, g, jf, pred):
        """Decode payload. Identical cubes across strata mint their
        AND chain ONCE (keyed by base wire + cube); every payload
        hangs off the shared leaf. Anchors stay per-payload.
        """
        b = self.b
        anchors = []
        undo = []
        cache = {}
        b.ccx(jf["j2"], self.gv.pos(1), pred)
        b.ccx(jf["j2"], self.fv.pos(3), pred)
        anchors += [(jf["j2"], self.gv.pos(1)), (jf["j2"], self.fv.pos(3))]
        for j in (3, 4):
            base = jf["j3"] if j == 3 else jf["j4"]
            for cubes, term in self.grouped[j]:
                for cube in cubes:
                    key = (base, tuple(sorted(cube.items())))
                    if key in cache:
                        leaf = cache[key]
                    else:
                        ks = sorted(cube)
                        wires = [base] + [self.f.pos(1 + v) for v in ks]
                        pols = [True] + [bool(cube[v]) for v in ks]
                        if len(wires) == 1:
                            leaf, trail = base, None
                        else:
                            leaf, trail = g.and_chain(wires, pols)
                        cache[key] = leaf
                        undo.append(trail)
                    if term[0] == "const":
                        b.cx(leaf, pred)
                        anchors.append((leaf,))
                    else:
                        kind, i = term
                        w = self.gv.pos(j - 1 + i) if kind == "ag" else self.fv.pos(j + i)
                        b.ccx(leaf, w, pred)
                        anchors.append((leaf, w))
        return anchors, undo

    def _decode_release(self, g, undo):
        for trail in reversed(undo):
            if trail is not None:
                g.release_chain(trail)

    def _pred_death(self, pred, anchors):
        b = self.b
        m = b.hmr(pred)
        for a in anchors:
            if len(a) == 1:
                b.z(a[0], cond=m)
            else:
                b.cz(a[0], a[1], cond=m)
        b.recycle(pred)

    # ------------------------------------------------------ forward step
    # Steps F1-F17 (labels in the walk section header) over windowed
    # rows. Two choices set the cost: (i) the g zero tests use
    # the blocked-keeper form (transient cap ~block+keepers vs W; the
    # uncompute recomputes each block, +~WG(t) Toffoli per test); (ii)
    # row adds are controlled [CDKM] ripples at scheduled widths.
    def step_fwd(self, t):
        b = self.b
        g = G(b)
        with b.phase(self.phase_fwd):
            s, bw = self._fwd_add(g, t)
            # F5 z += 1 on live lanes (pre-freeze: every lane is live)
            early = t < self.Tmin
            self._z_step(g, early)
            self._fwd_shift_exit(t, early)
            # rider stage: armed on every step of the data window
            # [Tmin, S). A lane's alive steps are [0, tstop), so
            # armed AND alive covers exactly the tstop - Tmin
            # data-bearing stages — the walk itself unary-decodes the
            # count; a lane still live at the schedule cut fails
            # through the cut's own priced channel. Mirror slot:
            # between R13 and R14 on the rev leg.
            if self.rider is not None and self.rider["legs"].get("fwd") and self.Tmin <= t:
                self._rider_stage(self.rider["legs"]["fwd"])
            # F7 freeze detect + flips (blocked zero test: the zf
            # witness lives through F16, coexisting with the decode
            # transients — a plain chain's trail there pushes the
            # armed-window census past the walk peak). tst stores the
            # freeze index OFFSET t+1-Tmin (values 1..S-Tmin; 0 =
            # never fired).
            if not early:
                # block=16: R13's small-block argument, bounded below
                # by the keeper count — unlike R13's witness, zf's
                # keeper ladder lives through the decode (F16 releases
                # it), so extra keepers land on the decode apex, the
                # arena's binding moment. 16 halves the chain apex at
                # the widest armed rows (WG <= ~24) without adding a
                # keeper; the ztest Toffoli count is block-invariant.
                zf, zfk = g.ztest_blocked(self.g.ids + [self.g.sign], block=16)
                fire = g.andp(zf, self.frz, pol=(True, False))
                b.cx(fire, self.frz)
                for i in range(self.Wt):
                    if ((t + 1 - self.Tmin) >> i) & 1:
                        b.cx(fire, self.tst[i])
            # F8 class flags; frz already flipped, so c3 reads the NEW
            # frozen state: a lane that freezes at this step pushes no
            # record. Pre-freeze, frz is a constant 0 and drops from the
            # chain. Armed steps keeperize the z ztest (one block >=
            # Wz): the chain trail would otherwise sit through the
            # decode apex — the arena's binding moment — at Wz-1 live
            # wires vs the keeper's 1; the F14 recompute is the
            # uncompute price (Wz-2 Toffoli per test).
            if early:
                znz, znzt = g.zero_test(list(self.z))
            else:
                znz, znzk = g.ztest_blocked(list(self.z), block=self.Wz)
            zpos = g.andp(self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                c3, c3t = g.and_chain([s, zpos], [False, False])
            else:
                c3, c3t = g.and_chain([self.frz, s, zpos], [False, False, False])
            # F9 j-flags
            jf = self._jflags(g, c3)
            # (F11 absent: the forward leg never READS the decoded
            # bit — bw carries the branch and the record port stores
            # it. The j flags remain: F13's store port reads the
            # ladder's last rung.)
            # F13 R push: brigade rotation up, then port. The span
            # clips at t: fill before step t's push is at most the
            # number of prior class-3 steps <= t (a theorem).
            self._r_fit(t)
            sp = min(self.spans[t], t)
            for i in range(sp - 1, -1, -1):
                b.cswap(jf["j5p"], self.R[i], self.R[i + 1])
            b.cswap(jf["j5p"], bw, self.R[0])
            # F14 flag/test deaths (no forward decode to kill)
            self._jflags_release(g, jf, c3)
            g.release_chain(c3t)
            g.andm(zpos, self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                g.release_chain(znzt)
            else:
                g.ztest_free(znzk)
            # F15 s death; armed steps then run F16
            self._s_death(g, s, None if early else fire)
            if not early:
                # F16 fire death: witness [tst == t+1-Tmin] & (g == 0)
                # & frz
                eqt, eqtt = g.eq_const(self.tst, t + 1 - self.Tmin)
                w1 = g.andp(eqt, zf)
                # fire holds w1 & frz — free retirement
                g.andm(fire, w1, self.frz)
                g.andm(w1, eqt, zf)
                g.release_chain(eqtt)
                g.ztest_free(zfk)
            # F17 b death: HMR; the outcome-conditioned fixup lands on
            # the reverse leg's b' at the mirrored step
            self.o_fwd[t] = b.hmr(bw)
            b.recycle(bw)

    # ------------------------------------------------------ reverse step
    def step_rev(self, t):
        b = self.b
        g = G(b)
        with b.phase(self.phase_rev):
            # entry widths = fwd step t's exit windows
            self._r_fit(t)
            # R1 unfreeze comparator; its AND trail releases BEFORE the
            # controlled tst fan changes the trail's literals. Pre-freeze
            # steps carry no unfreeze machinery (tst < Tmin+1 is
            # unreachable).
            early = t < self.Tmin
            if not early:
                eqt, eqtt = g.eq_const(self.tst, t + 1 - self.Tmin)
                fireR = g.andp(eqt, self.frz)
                g.release_chain(eqtt)
                b.cx(fireR, self.frz)
                for i in range(self.Wt):
                    if ((t + 1 - self.Tmin) >> i) & 1:
                        b.cx(fireR, self.tst[i])
            # R2 sigma witness from the post-state carrier (pre-freeze
            # the ~frz literal is constant-1: a plain copy)
            if early:
                sp_ = b.qubit()
                b.cx(self.fv.pos(1), sp_)
            else:
                sp_ = g.andp(self.frz, self.fv.pos(1), pol=(False, True))
            # R3 class flags; the extra ~fireR literal mirrors F8: the
            # step at which a lane froze pops no record. Armed steps
            # keeperize the z ztest exactly as F8 (the trail would sit
            # through the R6 decode apex); R12 recomputes at release.
            if early:
                znz, znzt = g.zero_test(list(self.z))
            else:
                znz, znzk = g.ztest_blocked(list(self.z), block=self.Wz)
            zpos = g.andp(self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                c3, c3t = g.and_chain([sp_, zpos], [False, False])
            else:
                c3, c3t = g.and_chain([self.frz, sp_, zpos, fireR], [False, False, False, False])
            # R4 j-flags
            jf = self._jflags(g, c3)
            # R5 R pop: port out first, then brigade rotation down
            bpw = b.qubit()
            b.cswap(jf["j5p"], self.R[0], bpw)
            sp = min(self.spans[t], t)
            for i in range(sp):
                b.cswap(jf["j5p"], self.R[i], self.R[i + 1])
            # R6 decode
            pred = b.qubit()
            anchors, undo = self._decode(g, jf, pred)
            # R7 b' assembly: pred ^ s' ^ (fireR & ~s'); the unfreeze
            # term is gated by the sigma witness. Pre-freeze there
            # is no unfreeze term.
            b.cx(pred, bpw)
            b.cx(sp_, bpw)
            if not early:
                fnos = g.andp(fireR, sp_, pol=(True, False))
                b.cx(fnos, bpw)
                g.andm(fnos, fireR, sp_, pol=(True, False))
            # (R8-R10 mismatch-safety machinery absent — see F11.)
            # R11 deferred forward-b fixup: demolished value = b' & ~j5p
            b.x(jf["j5p"])
            b.cz(bpw, jf["j5p"], cond=self.o_fwd[t])
            b.x(jf["j5p"])
            # R12 decode deaths BEFORE un-run changes the anchors the
            # decode read
            self._pred_death(pred, anchors)
            self._decode_release(g, undo)
            self._jflags_release(g, jf, c3)
            g.release_chain(c3t)
            g.andm(zpos, self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                g.release_chain(znzt)
            else:
                g.ztest_free(znzk)
            # R13 fireR death (g still post-state; g == 0 is absorbing,
            # so (g == 0) & ~frz witnesses exactly a just-fired lane)
            if not early:
                # block=8: the armed-window row is narrow (WG <= ~26),
                # so small blocks cut the ladder apex (~25 -> ~12 live
                # transients) at the arena's binding moment; the ztest
                # Toffoli count is block-invariant.
                zfr, zfrk = g.ztest_blocked(self.g.ids + [self.g.sign], block=8)
                # fireR holds zfr & ~frz — free retirement (before
                # ztest_free so zfr is live)
                g.andm(fireR, zfr, self.frz, pol=(True, False))
                g.ztest_free(zfrk)
            # rider stage (mirror slot of the fwd leg's; same
            # [Tmin, S) arm; runs before the repack inverse, so it
            # sees the same post-repack row state the fwd rider saw)
            if self.rider is not None and self.rider["legs"].get("rev") and self.Tmin <= t:
                self._rider_stage(self.rider["legs"]["rev"])
            self._repack_inv()
            # R14 un-run: un-shifts, z decrement, un-adds, at the fwd
            # step's grown widths (the mirror law: _rev_unshift)
            self._rev_unshift(t, early)
            self._z_step(g, early, dec=True)
            self._rev_unadd(g, t, sp_, bpw, early)


# ===========================================================================
# the staged engine's walk (staged-decode variant)
#
# Windowed divstep walk, staged-decode variant: the forward/reverse
# step ladders over the shared machinery above (rows, gadget kit,
# schedule, home pools, and the common WalkWBase methods). This
# variant's own layer is the bridged store rung (_chain_flip/_store_*),
# the staged decode over the levels j = 2..JSTORE-1 with PPRM payloads
# from decode5_derive (_decode_apply), and the step spellings that
# drive them.
# ===========================================================================


class StagedWalk(WalkWBase):
    def __init__(
        self,
        b,
        prm,
        forms,
        spans,
        sched,
        xw,
        fw,
        extra,
        phase_fwd="walk/step",
        phase_rev="walk/un-step",
    ):
        super().__init__(b, prm, forms, spans, sched, xw, fw, extra, phase_fwd, phase_rev)
        # decode levels: (k, PPRM payload) for k in 2..JSTORE-1, the
        # payloads derived from the prime (decode5_derive). JSTORE =
        # the raw-record store threshold on v2(fv') — prm["Jstore"],
        # default 7.
        self.JSTORE = prm.get("Jstore", 7)
        self.levels = [
            (k, decode5_derive(prm["p"], k) if k >= 5 else None) for k in range(2, self.JSTORE)
        ]
        # per-level traversal programs for the PPRM DFS
        # (_decode_apply pprm_level): the prefix keep/pop/push
        # derivation depends only on the monomial list, so it
        # runs once here; the emission-time DFS resolves pos()
        # and emits.
        self.levels_prog = {
            k: self._pprm_prog(monos) for k, monos in self.levels if monos is not None
        }

    # ---------------------------------------------------- decode layer
    # (the level flags, the decode and the record port; row bits
    # read through WRow.pos)
    def _chain_flip(self, g, lits, pols, w, block=3):
        """XOR AND(lits under pols) onto w through bridged blocked
        chains: each block's chain is copied to a bridge keeper and
        released, so the mint-time apex is ~block + bridges,
        independent of len(lits). Bridges unwind by re-mint (their
        literals are unchanged at every call site).
        """
        b = self.b
        cur, curp = lits[0], pols[0]
        keepers = []
        i = 1
        while i < len(lits):
            blk, bp = lits[i : i + block], pols[i : i + block]
            leaf, trail = g.and_chain([cur] + blk, [curp] + bp)
            if i + block < len(lits):
                k = b.qubit()
                b.cx(leaf, k)
                g.release_chain(trail)
                keepers.append((k, cur, curp, blk, bp))
                cur, curp = k, True
            else:
                b.cx(leaf, w)
                g.release_chain(trail)
            i += block
        for k, cu, cp, blk, bp in reversed(keepers):
            leaf, trail = g.and_chain([cu] + blk, [cp] + bp)
            b.cx(leaf, k)
            g.release_chain(trail)
            b.recycle(k)

    def _store_flip(self, g, c3, w):
        """XOR the store predicate c3 AND [v2(fv') >= JSTORE] (fv
        bits 2..JSTORE-1 all clear on a class-3 post-state) onto w
        via the bridged blocked chain (mint apex ~4).
        """
        lits = [c3] + [self.fv.pos(k) for k in range(2, self.JSTORE)]
        pols = [True] + [False] * (self.JSTORE - 2)
        self._chain_flip(g, lits, pols, w)

    def _store_mint(self, g, c3):
        w = self.b.qubit()
        self._store_flip(g, c3, w)
        return w

    def _store_kill(self, g, c3, w):
        self._store_flip(g, c3, w)
        self.b.recycle(w)

    def _pprm_prog(self, monos):
        """Return the PPRM DFS traversal program for one level: per
        monomial, the prefix depth to keep, the literal specs to
        push, and the last literal spec (None = the constant-1
        monomial, a plain flag payload). Derived by running the DFS
        prefix logic once — the program depends only on the monomial
        list, never on emission-time state; monomials arrive
        prefix-sorted (decode5 canonical order).
        """
        prog = []
        stack = []
        for fs, qs in monos:
            lits = [("f", i) for i in fs] + [("q", i) for i in qs]
            if not lits:
                prog.append((0, (), None))
                continue
            pref = lits[:-1]
            keep = 0
            while keep < len(stack) and keep < len(pref) and stack[keep] == pref[keep]:
                keep += 1
            del stack[keep:]
            pushes = tuple(pref[len(stack) :])
            stack.extend(pushes)
            prog.append((keep, pushes, lits[-1]))
        return prog

    def _decode_apply(self, g, c3, pred, cond=None):
        """Emit the branch payload over decode levels j = 2..6: the j2
        terms, the j3/j4 grouped ESOP, and the j5/j6 PPRM layers
        (decode5_derive). Everything is STAGED off the class root c3:
        the level rung j_k = neg_{k-1} AND fv[k] (neg = the running
        c3 AND ~fv[2..k-1] prefix chain) mints, carries its level's
        payload, and dies before the next level; cube chains and trie
        paths release depth-first inside each level. Live decode
        transients: the neg prefix (grows to 4) + one rung + the
        payload path — no resident flag ladder.

        Two modes on one traversal. cond=None (apply): XOR each term
        into pred (payload CCX/CX). cond=m (death): pred has been
        HMR'd with outcome m and the same traversal lands the
        outcome-conditioned phase fixup (cz/z, unscored) — the death
        pass pays only the chain re-mints, never the payload
        Toffolis. The per-term phases compose to the XOR's phase
        exactly (XOR = sum mod 2).
        """
        b = self.b

        def pay2(leaf, w):
            if cond is None:
                b.ccx(leaf, w, pred)
            else:
                b.cz(leaf, w, cond=cond)

        def pay1(leaf):
            if cond is None:
                b.cx(leaf, pred)
            else:
                b.z(leaf, cond=cond)

        def esop_level(j, base):
            last = None  # (key, leaf, trail): adjacent reuse
            for cubes, term in self.grouped[j]:
                for cube in cubes:
                    key = (j, tuple(sorted(cube.items())))
                    if last is not None and last[0] == key:
                        leaf, trail = last[1], last[2]
                    else:
                        if last is not None and last[2] is not None:
                            g.release_chain(last[2])
                        ks = sorted(cube)
                        wires = [base] + [self.f.pos(1 + v) for v in ks]
                        pols = [True] + [bool(cube[v]) for v in ks]
                        if len(wires) == 1:
                            leaf, trail = base, None
                        else:
                            leaf, trail = g.and_chain(wires, pols)
                        last = (key, leaf, trail)
                    if term[0] == "const":
                        pay1(leaf)
                    else:
                        kind, i = term
                        w = self.gv.pos(j - 1 + i) if kind == "ag" else self.fv.pos(j + i)
                        pay2(leaf, w)
            if last is not None and last[2] is not None:
                g.release_chain(last[2])

        def pprm_level(flag, prog, qoff):
            # DFS over the level's precomputed traversal program
            # (__init__: levels_prog / _pprm_prog); prefix reuse is
            # baked into the program, so emission only resolves
            # pos() lookups and emits
            stack = []

            def wire_of(spec):
                kind, i = spec
                return self.f.pos(i) if kind == "f" else self.fv.pos(qoff + i)

            def pop_to(depth):
                while len(stack) > depth:
                    spec, node = stack.pop()
                    prev = stack[-1][1] if stack else flag
                    g.andm(node, prev, wire_of(spec))

            for keep, pushes, last in prog:
                if last is None:
                    pay1(flag)
                    continue
                pop_to(keep)
                for spec in pushes:
                    prev = stack[-1][1] if stack else flag
                    stack.append((spec, g.andp(prev, wire_of(spec))))
                leaf = stack[-1][1] if stack else flag
                pay2(leaf, wire_of(last))
            pop_to(0)

        # rung copy-out: the level rung mints onto one wire via the
        # bridged blocked chain (_chain_flip: apex ~4 at every k),
        # carries its level's payload alone, and a second flip kills
        # it. No prefix chain persists into the payload apex.
        def rung_flip(k, w):
            lits = [c3] + [self.fv.pos(i) for i in range(2, k)] + [self.fv.pos(k)]
            pols = [True] + [False] * (k - 2) + [True]
            self._chain_flip(g, lits, pols, w)

        for k, _monos in self.levels:
            if k == 2:
                rung = g.andp(c3, self.fv.pos(2))
                pay2(rung, self.gv.pos(1))
                pay2(rung, self.fv.pos(3))
                g.andm(rung, c3, self.fv.pos(2))
                continue
            rung = b.qubit()
            rung_flip(k, rung)
            if k in (3, 4):
                esop_level(k, rung)
            else:
                pprm_level(rung, self.levels_prog[k], k)
            rung_flip(k, rung)
            b.recycle(rung)

    # ------------------------------------------------------ forward step
    # Steps F1-F17 (labels in the walk section header) over windowed
    # rows. Two choices set the cost: (i) the g zero tests use
    # the blocked-keeper form (transient cap ~block+keepers vs W; the
    # uncompute recomputes each block, +~WG(t) Toffoli per test); (ii)
    # row adds are controlled [CDKM] ripples at scheduled widths.
    def step_fwd(self, t):
        b = self.b
        g = G(b)
        with b.phase(self.phase_fwd):
            s, bw = self._fwd_add(g, t)
            early = t < self.Tmin
            self._fwd_shift_exit(t, early)
            # F5 z += 1 on live lanes, RESEQUENCED after the exit
            # shrinks: z is first read at F7/F8, so the inc's carry
            # chain runs at exit row widths, off the add-window bulge
            # (run inside the add window, this chain is the leg's
            # peak moment). Mirror slot on the reverse leg: before
            # the repack inverse.
            self._z_step(g, early)
            # rider stage: armed on every step of the data window
            # [Tmin, S). A lane's alive steps are [0, tstop), so
            # armed AND alive covers exactly the tstop - Tmin
            # data-bearing stages — the walk itself unary-decodes the
            # count; a lane still live at the schedule cut fails
            # through the cut's own priced channel. Mirror slot:
            # between R13 and R14 on the rev leg.
            if self.rider is not None and self.rider["legs"].get("fwd") and self.Tmin <= t:
                self._rider_stage(self.rider["legs"]["fwd"])
            # F7 freeze detect + flips (blocked zero test: the zf
            # witness lives through F16, coexisting with the decode
            # transients — a plain chain's trail there pushes the
            # armed-window census past the walk peak). tst stores the
            # freeze index OFFSET t+1-Tmin (values 1..S-Tmin; 0 =
            # never fired).
            if not early:
                # blocked mint (block=8), witness COPIED OUT and the
                # keeper ladder released at once: one wire persists
                # to F16, where a re-minted test kills it (g is
                # untouched between F7 and F16, so the recompute
                # reads the same value). What binds the census is a
                # ladder persisting under the store/decode spikes,
                # not the mint apex.
                zf, zfk = g.ztest_blocked(self.g.ids + [self.g.sign], block=6)
                zfw = b.qubit()
                b.cx(zf, zfw)
                g.ztest_free(zfk)
                fire = g.andp(zfw, self.frz, pol=(True, False))
                b.cx(fire, self.frz)
                for i in range(self.Wt):
                    if ((t + 1 - self.Tmin) >> i) & 1:
                        b.cx(fire, self.tst[i])
            # F8 class flags; frz already flipped, so c3 reads the NEW
            # frozen state: a lane that freezes at this step pushes no
            # record. Pre-freeze, frz is a constant 0 and drops from the
            # chain. Armed steps keeperize the z ztest (one block >=
            # Wz): the chain trail would otherwise sit through the
            # decode apex — the arena's binding moment — at Wz-1 live
            # wires vs the keeper's 1; the F14 recompute is the
            # uncompute price (Wz-2 Toffoli per test).
            if early:
                znz, znzt = g.zero_test(list(self.z))
            else:
                znz, znzk = g.ztest_blocked(list(self.z), block=self.Wz)
            zpos = g.andp(self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                c3, c3t = g.and_chain([s, zpos], [False, False])
            else:
                # armed: the class root copies out, its trail
                # releases at mint time (persist 1, not 3); the F14
                # kill re-mints the chain (literals unchanged there)
                top, tr = g.and_chain([self.frz, s, zpos], [False, False, False])
                c3 = b.qubit()
                b.cx(top, c3)
                g.release_chain(tr)
            # F9 store rung (the forward leg's ONLY flag — F10-F12
            # absent: the forward leg never READS the decoded bit;
            # bw carries the branch and the record port stores it)
            j7p = self._store_mint(g, c3)
            # F13 R push: brigade rotation up, then port. The span
            # clips at t: fill before step t's push is at most the
            # number of prior class-3 steps <= t (a theorem).
            self._r_fit(t)
            sp = min(self.spans[t], t)
            for i in range(sp - 1, -1, -1):
                b.cswap(j7p, self.R[i], self.R[i + 1])
            b.cswap(j7p, bw, self.R[0])
            # F14 flag/test deaths (no forward decode to kill)
            self._store_kill(g, c3, j7p)
            if early:
                g.release_chain(c3t)
            else:
                top, tr = g.and_chain([self.frz, s, zpos], [False, False, False])
                b.cx(top, c3)
                g.release_chain(tr)
                b.recycle(c3)
            g.andm(zpos, self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                g.release_chain(znzt)
            else:
                g.ztest_free(znzk)
            # F15 s death; armed steps then run F16
            self._s_death(g, s, None if early else fire)
            if not early:
                # F16 fire death: witness [tst == t+1-Tmin] & (g == 0)
                # & frz; then the zfw kill by re-minted test
                eqt, eqtt = g.eq_const(self.tst, t + 1 - self.Tmin)
                w1 = g.andp(eqt, zfw)
                # fire holds w1 & frz — free retirement
                g.andm(fire, w1, self.frz)
                g.andm(w1, eqt, zfw)
                g.release_chain(eqtt)
                zf2, zfk2 = g.ztest_blocked(self.g.ids + [self.g.sign], block=6)
                b.cx(zf2, zfw)
                g.ztest_free(zfk2)
                b.recycle(zfw)
            # F17 b death: HMR; the outcome-conditioned fixup lands on
            # the reverse leg's b' at the mirrored step
            self.o_fwd[t] = b.hmr(bw)
            b.recycle(bw)

    # ------------------------------------------------------ reverse step
    def step_rev(self, t):
        b = self.b
        g = G(b)
        with b.phase(self.phase_rev):
            # entry widths = fwd step t's exit windows
            self._r_fit(t)
            # R1 unfreeze comparator; its AND trail releases BEFORE the
            # controlled tst fan changes the trail's literals. Pre-freeze
            # steps carry no unfreeze machinery (tst < Tmin+1 is
            # unreachable).
            early = t < self.Tmin
            if not early:
                eqt, eqtt = g.eq_const(self.tst, t + 1 - self.Tmin)
                fireR = g.andp(eqt, self.frz)
                g.release_chain(eqtt)
                b.cx(fireR, self.frz)
                for i in range(self.Wt):
                    if ((t + 1 - self.Tmin) >> i) & 1:
                        b.cx(fireR, self.tst[i])
            # R2 sigma witness from the post-state carrier (pre-freeze
            # the ~frz literal is constant-1: a plain copy)
            if early:
                sp_ = b.qubit()
                b.cx(self.fv.pos(1), sp_)
            else:
                sp_ = g.andp(self.frz, self.fv.pos(1), pol=(False, True))
            # R3 class flags; the extra ~fireR literal mirrors F8: the
            # step at which a lane froze pops no record. Armed steps
            # keeperize the z ztest exactly as F8 (the trail would sit
            # through the R6 decode apex); R12 recomputes at release.
            if early:
                znz, znzt = g.zero_test(list(self.z))
            else:
                znz, znzk = g.ztest_blocked(list(self.z), block=self.Wz)
            zpos = g.andp(self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                c3, c3t = g.and_chain([sp_, zpos], [False, False])
            else:
                top, tr = g.and_chain([self.frz, sp_, zpos, fireR], [False, False, False, False])
                c3 = b.qubit()
                b.cx(top, c3)
                g.release_chain(tr)
            # R4 store rung
            j7p = self._store_mint(g, c3)
            # R5 R pop: port out first, then brigade rotation down
            bpw = b.qubit()
            b.cswap(j7p, self.R[0], bpw)
            sp = min(self.spans[t], t)
            for i in range(sp):
                b.cswap(j7p, self.R[i], self.R[i + 1])
            # R6 decode (staged rungs off c3; see _decode_apply)
            pred = b.qubit()
            self._decode_apply(g, c3, pred)
            # R7 b' assembly: pred ^ s' ^ (fireR & ~s'); the unfreeze
            # term is gated by the sigma witness. Pre-freeze there
            # is no unfreeze term.
            b.cx(pred, bpw)
            b.cx(sp_, bpw)
            if not early:
                fnos = g.andp(fireR, sp_, pol=(True, False))
                b.cx(fnos, bpw)
                g.andm(fnos, fireR, sp_, pol=(True, False))
            # (R8-R10 mismatch-safety machinery absent — see F9.)
            # R11 deferred forward-b fixup: demolished value = b' & ~j7p
            # (a stored step parked its b in R, so the F17 HMR saw 0)
            b.x(j7p)
            b.cz(bpw, j7p, cond=self.o_fwd[t])
            b.x(j7p)
            # R12 decode death BEFORE the un-run changes the rows the
            # decode read: HMR + the staged outcome-conditioned fixup pass
            m_pred = b.hmr(pred)
            self._decode_apply(g, c3, None, cond=m_pred)
            b.recycle(pred)
            self._store_kill(g, c3, j7p)
            if early:
                g.release_chain(c3t)
            else:
                top, tr = g.and_chain([self.frz, sp_, zpos, fireR], [False, False, False, False])
                b.cx(top, c3)
                g.release_chain(tr)
                b.recycle(c3)
            g.andm(zpos, self.z[self.Wz - 1], znz, pol=(False, False))
            if early:
                g.release_chain(znzt)
            else:
                g.ztest_free(znzk)
            # R13 fireR death (g still post-state; g == 0 is absorbing,
            # so (g == 0) & ~frz witnesses exactly a just-fired lane)
            if not early:
                # block=6: the armed-window row is narrow (WG <= ~26)
                # and at block=8 this mint's chain apex is the leg's
                # peak moment; the ztest Toffoli count is
                # block-invariant.
                zfr, zfrk = g.ztest_blocked(self.g.ids + [self.g.sign], block=6)
                # fireR holds zfr & ~frz — free retirement (before
                # ztest_free so zfr is live)
                g.andm(fireR, zfr, self.frz, pol=(True, False))
                g.ztest_free(zfrk)
            # rider stage (mirror slot of the fwd leg's; same
            # [Tmin, S) arm; runs before the repack inverse, so it
            # sees the same post-repack row state the fwd rider saw)
            if self.rider is not None and self.rider["legs"].get("rev") and self.Tmin <= t:
                self._rider_stage(self.rider["legs"]["rev"])
            # z -= 1 (mirror of the resequenced F5: at exit widths,
            # before the repack inverse regrows the add window)
            self._z_step(g, early, dec=True)
            self._repack_inv()
            # R14 un-run: un-shifts, un-adds, at the fwd
            # step's grown widths (the mirror law: _rev_unshift)
            self._rev_unshift(t, early)
            self._rev_unadd(g, t, sp_, bpw, early)


# ===========================================================================
# the k-block lazy modular-multiplication family
#
# The k-block lazy modular-multiplication family lives in
# ../_shared/luo_core.py (shared with circuits/euclidean): the
# borrowed-dirty Gidney constant add/compare ladder (gidney_add_const,
# gidney_cmp_ge), the controlled ripple add and comparator cells
# (ctrl_qq_add_flag, cuccaro_lt_flag), the k-block keeper gadgets
# (kb_ctrl_add_flag, kb_lt_flag, kb_fold_const, kb_dbl_fold), the lazy
# controlled mod-add (gc_lazy_cma), and the windowed Horner multiplier
# (horner_mul). This circuit keeps its priced windows and the dispatch
# (_mul_cma / _mul_dbl) here and passes them to horner_mul (mul_modp).
#
# Window widths: circuits/euclidean runs these windows at
# FOLD_WIN = 63 / CMP_TOPG = 32; this circuit runs 58 / 27. Each bit
# removed from a window roughly doubles that window's escape rate; the
# rates are booked in the composed failure budget (README.md, Bounding
# deliberate approximations). The constants are module-local: every
# fold/compare site of this circuit reads them from here.
#
# Coupling: LUO_CORE_DEPS is the inverse binding: luo_core's
# circuit-local callables, satisfied here where its flows are used and
# by a fail-loud placeholder where they are not.
# ===========================================================================

kb_fold_const = LC.kb_fold_const
kb_dbl_fold = LC.kb_dbl_fold
gc_lazy_cma = LC.gc_lazy_cma


# ---------------------------------------------------------------------------
# Priced mul windows: fold chains cap at FOLD_WIN — the pseudo-Mersenne
# fold constants' ones end at bit 33 (asserted below), beyond which
# carries only propagate (fwd) or saturate against the flag (inv, the
# truncated kb-sums cancel) — and the lazy-flag compare reads only
# the top CMP_TOPG bits. The two doors are priced escapes: a fold
# carry past the cap costs ~ c/2^FOLD_WIN per call, a compare tie
# across the truncated window ~ 2^-CMP_TOPG per call, booked per
# site count in the composed failure budget (README.md, Bounding
# deliberate approximations). circuits/euclidean runs 63 / 32; this
# circuit narrows both by 5 bits (~2^5 on each escape rate) and books
# the larger rates.
# ---------------------------------------------------------------------------
FOLD_WIN = 58
CMP_TOPG = 27

# priced-window precondition: the fold constant's ones end at bit 33,
# strictly inside the FOLD_WIN cap — beyond it carries only propagate.
assert ((1 << N) - P_MOD).bit_length() == 33 <= FOLD_WIN


# Dispatch: the k-block lazy gadgets carry the production stream. The
# door mechanism is classical and analytic (the fold/tie channels
# above); the composed 100K evaluation exercises the full-width
# composition.


def _mul_cma(b, ctrl, addend, target, p, n, ma, inv=False):
    anc = ma["anc"]
    assert anc, "mul pool empty at a lazy-cma site"
    gc_lazy_cma(b, ctrl, addend, target, p, n, ma, anc, inv=inv, win=FOLD_WIN, topg=CMP_TOPG)


def _mul_dbl(b, reg, p, n, ma, inv=False):
    anc = ma["anc"]
    assert anc, "mul pool empty at a lazy-double site"
    kb_dbl_fold(b, reg, p, n, anc, inv=inv, win=FOLD_WIN)


def mul_modp(b, xc, y, out, p, n, ma, inv=False):
    """out <- xc*y mod p (out starts |0>): the shared windowed Horner
    multiplier on this circuit's dispatch (_mul_cma / _mul_dbl above).
    """
    LC.horner_mul(b, xc, y, out, p, n, ma, cma=_mul_cma, dbl=_mul_dbl, inv=inv)


# ---------------------------------------------------------------------------
# luo_core circuit-local callables (_DEPS): the multiplier its
# modular flows resolve, plus fail-loud placeholders for the names
# outside this circuit's flows (the EEA step-template machinery).
# ---------------------------------------------------------------------------


def _unused_dependency(*_a, **_k):
    raise RuntimeError("luo_core dependency outside this circuit's flows")


LUO_CORE_DEPS = {
    "StepTables": _unused_dependency,
    "c3x": _unused_dependency,
    "emit_eea": _unused_dependency,
    "mul_modp": mul_modp,
}


# luo_core's circuit-local callables, satisfied by the dispatch above.
LC.init(LUO_CORE_DEPS)


# ===========================================================================
# the fused square: out -= lambda^2 mod p without a full product register
#
# With B = 2^128, lambda = l0 + B*l1 and s = l0 + l1, the identity
# 2*l0*l1 = s^2 - l0^2 - l1^2 and B^2 == F (mod p) give
#
#     lambda^2 == (1 - B)*l0^2 + (F - B)*l1^2 + B*s^2       (mod p).
#
# Each half-width square goes into a temporary product register, is folded
# into the coordinate register with its coefficient, and is cleared before
# the next square starts.  The product register holds v^2 - 1 (see
# _square_rows); every fold adds the missing 1 through its carry-in.
#
# A fold splits the shifted product at bit 256 and reduces the part above
# with 2^256 == F (mod p), using the signed digits of F.  A slice that
# ends at least `margin` bits below bit 256 is added without reduction
# (short accumulation); a slice that reaches the top is added with one
# conditional F correction (modular accumulation).  Both truncate their
# carry and comparison work to `margin` bits; _square_margin derives the
# margin from the failure share the circuit books for this component.
#
# Every adder takes temporary-AND carry wires only while the live count
# stays at or below the point's arena ceiling (`cap`), and otherwise
# keeps the carry in the addend wire (in-place majority, 2 Toffoli).
# ===========================================================================

SQ_HALF = N // 2
SQ_FOLD = (1 << N) - P_MOD  # 2^256 == SQ_FOLD (mod p)
SQ_FAIL_SHARE = 6e-7  # the failure budget's share for the fused square


def _signed_digits(k):
    """Return the non-adjacent form of k as [(position, +1 or -1)]."""
    out, pos = [], 0
    while k:
        if k & 1:
            d = 2 - (k & 3)
            out.append((pos, d))
            k -= d
        k >>= 1
        pos += 1
    return out


SQ_FOLD_DIGITS = _signed_digits(SQ_FOLD)
assert sum(d << j for j, d in SQ_FOLD_DIGITS) == SQ_FOLD
assert max(j for j, _ in SQ_FOLD_DIGITS) < SQ_HALF


def _ripple_add(b, tgt, addend, cin=None, cout=None, room=0, spares=()):
    """tgt += addend + cin (mod 2^len(tgt)); the carry out of the top
    position XORs onto `cout` when given.

    `addend` entries are wires or None (zero bits); `cin` is a wire read
    as the carry-in bit.  Carry wires come from `spares` and then from up
    to `room` fresh wires; each such position costs 1 Toffoli
    (temporary AND, measurement release).  The remaining addend positions
    keep their carry in the addend wire (majority chain, 2 Toffoli).
    Zero positions always need a carry wire.  `spares` are zero wires the
    caller lends as carry wires (used first, returned zero).  The addend
    and `cin` return to their input values.
    """
    w = len(tgt)
    a = list(addend) + [None] * (w - len(addend))
    assert len(a) == w
    pool = WirePool(b, list(spares))
    last = w if cout is not None else w - 1  # positions that make a carry
    zeros = sum(1 for i in range(last) if a[i] is None)
    wires = room + len(spares) - zeros
    assert wires >= 0, "ripple add: no carry wires left for the zero positions"
    log = []
    c = cin
    for i in range(last):
        ai, ti = a[i], tgt[i]
        if ai is None:
            if c is None:
                continue
            nxt = pool.and_compute(c, ti)
            log.append(("z", i, c, nxt))
        elif c is None:
            nxt = pool.and_compute(ai, ti)
            wires -= 1
            log.append(("g", i, None, nxt))
        elif wires > 0:
            b.cx(c, ai)
            b.cx(c, ti)
            nxt = pool.and_compute(ai, ti)
            b.cx(c, nxt)
            wires -= 1
            log.append(("g", i, c, nxt))
        else:
            b.cx(ai, ti)
            b.cx(ai, c)
            b.ccx(c, ti, ai)
            nxt = ai
            log.append(("m", i, c, None))
        c = nxt
    if cout is not None:
        if c is not None:
            b.cx(c, cout)
    else:
        if a[w - 1] is not None:
            b.cx(a[w - 1], tgt[w - 1])
        if c is not None:
            b.cx(c, tgt[w - 1])
    for kind, i, ci, anc in reversed(log):
        ai, ti = a[i], tgt[i]
        if kind == "z":
            pool.and_release(anc, ci, ti)
            b.cx(ci, ti)
        elif kind == "m":
            b.ccx(ci, ti, ai)
            b.cx(ai, ci)
            b.cx(ci, ti)
        elif ci is None:
            pool.and_release(anc, ai, ti)
            b.cx(ai, ti)
        else:
            b.cx(ci, anc)
            pool.and_release(anc, ai, ti)
            b.cx(ci, ai)
            b.cx(ai, ti)


def _add_scaled_bit(b, tgt, ctl, k):
    """tgt += ctl * k (mod 2^len(tgt)) for a classical constant k:
    one temporary AND per position from the lowest set bit of k up to
    the top position.
    """
    w = len(tgt)
    log = []
    c = None
    for i in range(w - 1):
        ti, on = tgt[i], (k >> i) & 1
        if on and c is None:
            nxt = b.and_compute(ctl, ti)
        elif on:
            b.cx(c, ti)
            b.cx(ctl, c)
            nxt = b.and_compute(ti, c)
            b.cx(ctl, c)
            b.cx(c, nxt)
        elif c is not None:
            nxt = b.and_compute(c, ti)
        else:
            continue
        log.append((i, on, c, nxt))
        c = nxt
    if (k >> (w - 1)) & 1:
        b.cx(ctl, tgt[w - 1])
    if c is not None:
        b.cx(c, tgt[w - 1])
    for i, on, ci, anc in reversed(log):
        ti = tgt[i]
        if on and ci is None:
            b.and_release(anc, ctl, ti)
            b.cx(ctl, ti)
        elif on:
            b.cx(ci, anc)
            b.cx(ctl, ci)
            b.and_release(anc, ti, ci)
            b.cx(ctl, ci)
            b.cx(ctl, ti)
        else:
            b.and_release(anc, ci, ti)
            b.cx(ci, ti)


def _phase_if_below(b, xs, ys, m):
    """Apply (-1)^(m * [X < Y]) for equal-width wire lists X = xs, Y = ys.
    [X < Y] is the carry out of (~X + Y): one temporary AND per position,
    all released by measurement.
    """
    for q in xs:
        b.x(q)
    log = []
    c = None
    for xq, yq in zip(xs, ys):
        if c is not None:
            b.cx(c, xq)
            b.cx(c, yq)
        nxt = b.and_compute(xq, yq)
        if c is not None:
            b.cx(c, nxt)
        log.append((xq, yq, c, nxt))
        c = nxt
    b.z(c, cond=m)
    for xq, yq, ci, anc in reversed(log):
        if ci is not None:
            b.cx(ci, anc)
        b.and_release(anc, xq, yq)
        if ci is not None:
            b.cx(ci, xq)
            b.cx(ci, yq)
    for q in xs:
        b.x(q)


class FusedSquare:
    """out -= lam^2 mod p in three square terms (low, high, cross).

    Call low_term(), high_term() and cross_term() in any order, then
    close().  `cap` is the arena ceiling the component must not exceed.
    """

    def __init__(self, b, lam, out, cap):
        assert len(lam) == N and len(out) == N
        self.b, self.out, self.cap = b, list(out), int(cap)
        self.l0, self.l1 = list(lam[:SQ_HALF]), list(lam[SQ_HALF:])
        self.margin = _square_margin()
        self.one = b.qubit()
        b.x(self.one)

    def room(self):
        return self.cap - self.b.active_qubits

    # -- square of v into a fresh 2n-wire register ---------------------------

    def _rows(self, v, prod, clear):
        """Build prod = v^2 - 1 from |0> (clear=False), or return it to |0>
        (clear=True).  Row i adds 2^m + (2 v_i - 1) R + v_i at offset 2i+1,
        with R = v >> (i+1) and m = len(R): the R wires are complemented
        when v_i = 0, the row's addend ends with v_i itself, and the carry-in
        is 1.  Row i's carries borrow the zero wires above its window.
        """
        b, n = self.b, len(v)
        top = prod[2 * n - 1]
        order = range(n - 1)
        if clear:
            b.cx(v[n - 1], top)
            b.x(top)
            order = reversed(order)
        else:
            for j in range(n):
                b.cx(v[j], prod[j])
                b.x(prod[j])
        for i in order:
            ctl, rest = v[i], v[i + 1 :]
            window = prod[2 * i + 1 : n + i + 2]
            b.x(ctl)
            for q in rest:
                b.cx(ctl, q)
            b.x(ctl)
            if clear:
                for q in window:
                    b.x(q)
            _ripple_add(
                b, window, rest + [ctl], cin=self.one, room=self.room(), spares=prod[n + i + 2 :]
            )
            if clear:
                for q in window:
                    b.x(q)
            b.x(ctl)
            for q in rest:
                b.cx(ctl, q)
            b.x(ctl)
        if clear:
            for j in range(n):
                b.x(prod[j])
                b.cx(v[j], prod[j])
        else:
            b.cx(v[n - 1], top)
            b.x(top)

    # -- folding a shifted slice into out ------------------------------------

    def _short(self, val, shift, sign, cin):
        """out += sign * 2^shift * (val + cin), no reduction: the slice ends
        `margin` bits below bit 256 and the carry stops there.
        """
        b = self.b
        window = self.out[shift : shift + len(val) + self.margin]
        if sign < 0:
            for q in window:
                b.x(q)
        _ripple_add(b, window, val, cin=cin, room=self.room())
        if sign < 0:
            for q in window:
                b.x(q)

    def _modular(self, val, shift, sign, cin):
        """out += sign * 2^shift * (val + cin) mod p for a slice that ends at
        bit 255: add with carry-out c, add F when c is set, then erase c
        by measurement and recompute it from the top `margin` bits as a
        phase ([out < slice] exactly when the add wrapped).
        """
        b, out, t = self.b, self.out, self.margin
        assert shift + len(val) == N
        if sign < 0:
            for q in out:
                b.x(q)
        carry = b.qubit()
        _ripple_add(b, out[shift:], val, cin=cin, cout=carry, room=self.room())
        fold_window = out[: SQ_FOLD.bit_length() + t]
        assert self.room() >= len(fold_window) - 1, "fused square: ceiling too low"
        _add_scaled_bit(b, fold_window, carry, SQ_FOLD)
        m = b.hmr(carry)
        b.recycle(carry)
        assert self.room() >= t, "fused square: ceiling too low"
        _phase_if_below(b, out[N - t :], val[len(val) - t :], m)
        if sign < 0:
            for q in out:
                b.x(q)

    def _fold(self, val, shift, sign, cin):
        """out += sign * 2^shift * (val + cin) mod p for any slice: the part
        at or above bit 256 re-enters at the signed digits of F.
        """
        low, high = val[: N - shift], val[N - shift :]
        if _is_short(shift, len(low), self.margin):
            self._short(low, shift, sign, cin)
        else:
            self._modular(low, shift, sign, cin)
        for j, d in SQ_FOLD_DIGITS if high else ():
            self._fold(high, j, sign * d, None)

    # -- the three terms ------------------------------------------------------

    def _term(self, v, folds):
        b = self.b
        prod = b.qubits(2 * len(v))
        b.claim("square/product", prod)
        self._rows(v, prod, clear=False)
        for shift, sign in folds:
            self._fold(prod, shift, sign, self.one)
        self._rows(v, prod, clear=True)
        b.release("square/product", prod)
        for q in reversed(prod):
            b.recycle(q)

    def low_term(self):
        """out -= (1 - B) * l0^2."""
        self._term(self.l0, _folds_low())

    def high_term(self):
        """out -= (F - B) * l1^2."""
        self._term(self.l1, _folds_high())

    def cross_term(self):
        """out -= B * (l0 + l1)^2; the sum forms in place on l1 plus one
        fresh top wire and is undone afterwards.
        """
        b = self.b
        spill = b.qubit()
        s = self.l1 + [spill]
        _ripple_add(b, s, self.l0, room=self.room())
        self._term(s, _folds_cross())
        for q in s:
            b.x(q)
        _ripple_add(b, s, self.l0, room=self.room())
        for q in s:
            b.x(q)
        b.recycle(spill)

    def close(self):
        self.b.x(self.one)
        self.b.recycle(self.one)


def _folds_low():
    return [(0, -1), (SQ_HALF, +1)]


def _folds_high():
    return [(SQ_HALF, +1)] + [(j, -d) for j, d in SQ_FOLD_DIGITS]


def _folds_cross():
    return [(SQ_HALF, -1)]


def _is_short(shift, width, margin):
    return shift + width + margin <= N


def _fold_events(width, shift):
    """Count the accumulations _fold emits for a width-bit slice."""
    low = min(width, N - shift)
    high = width - low
    n = 1
    if high:
        n += sum(_fold_events(high, j) for j, _ in SQ_FOLD_DIGITS)
    return n


def _square_margin():
    """Return the truncation margin t in bits: each accumulation fails with
    probability at most 2 * 2^-t (short: carry escape, wrap past p;
    modular: F-window escape, top-bit tie), and the sum over all
    accumulations of the three terms stays within SQ_FAIL_SHARE.
    """
    t = 1
    while True:
        ops = 0
        for width, folds in (
            (2 * SQ_HALF, _folds_low()),
            (2 * SQ_HALF, _folds_high()),
            (2 * SQ_HALF + 2, _folds_cross()),
        ):
            ops += sum(_fold_events(width, shift) for shift, _ in folds)
        if 2 * ops * 2.0**-t <= SQ_FAIL_SHARE:
            return t
        t += 1


def subtract_square(b, lam, out, cap):
    """out -= lam^2 mod p (phases squ-a-lo, squ-b-hi, squ-c-sum)."""
    sq = FusedSquare(b, lam, out, cap)
    b.set_phase("squ-a-lo")
    sq.low_term()
    b.set_phase("squ-b-hi")
    sq.high_term()
    b.set_phase("squ-c-sum")
    sq.cross_term()
    sq.close()


def _cdkm_sub(b, a, acc, c_in):
    """acc -= a + c_in (mod 2^len(acc)): the exact adjoint of
    uncompute.gadgets.add(b, acc, a, gl=0, cin=c_in), gate for gate in reverse
    (MAJ steps become maj_inv, UMA steps uma_inv).  2(n-1) Toffoli, no
    ancilla; a and c_in return to their input values.
    """
    n = len(a)
    assert n == len(acc)
    if n == 0:
        return
    if n == 1:
        b.cx(a[0], acc[0])
        b.cx(c_in, acc[0])
        return
    uma_inv(b, c_in, acc[0], a[0])
    for i in range(1, n - 1):
        uma_inv(b, a[i - 1], acc[i], a[i])
    b.cx(a[n - 1], acc[n - 1])
    b.cx(a[n - 2], acc[n - 1])
    for i in range(n - 2, 0, -1):
        maj_inv(b, a[i - 1], acc[i], a[i])
    maj_inv(b, c_in, acc[0], a[0])


# ===========================================================================
# the shared engine layer: the units both engines execute verbatim
#
# The row fold/scan arithmetic, the DivstepInvBase walk sandwich (each
# engine's subclass binds its walk class and parameter callables), and
# the build/emit tail.
# ===========================================================================


def _tstop_tmin():
    """Return the pre-freeze pin Tmin (gen_constants.py derives it from
    the stopping-time census: P(tstop <= Tmin) within its budget).
    """
    return int(_CONSTANTS["tstop"]["Tmin"])


def _sched_s():
    """Return the schedule length S (gen_constants.py derives it from the
    stopping-time census: P(tstop > S) within its budget).
    """
    return int(_CONSTANTS["tstop"]["S"])


def _z_width():
    """Return the z register width Wz (gen_constants.py derives it from
    the z-width census: P(z needs more bits) within its budget).
    """
    return int(_CONSTANTS["z"]["Wz"])


def qcap_floor(name):
    """Return the arena ceiling of floor point `name` (circuit, wide,
    j5, j6, j8, j9): the lowest ceiling its build attains, which
    gen_constants.py finds by bisection.
    """
    return int(_CONSTANTS["qcap"][name])


def full_windows():
    """Return the deployed width tables: the quantile section (1e8-walk C
    census, greedy at the 2.85e-5/walk any-step budget), one entry per
    scheduled step.
    """
    w = _CONSTANTS["windows_q"]
    return w["WF"], w["WG"], w["WFV"], w["WGV"]


def odd_fold_const(n, p):
    """Return c = 2^n mod p, made odd (+p when even): the fold-class
    constant of the halve, double and negate gadgets. An odd c keeps
    c - 1 even, so a fold of c - 1 leaves bit 0 fixed.
    """
    c = (1 << n) % p
    if c % 2 == 0:
        c += p
    return c


# ---------------------------------------------------------------------------
# mul layer: the k-block multiplier on this circuit's machine wires
# ---------------------------------------------------------------------------


def mk_ma(singles, pool):
    """Return the mul-aux role dict the k-block gadgets address: 7 machine
    singles + the gadget pool.
    """
    return {
        "carry": singles["carry"],
        "flag": singles["flag"],
        "top": singles["top"],
        "tq": singles["tq"],
        "clean3": singles["clean3"],
        "anc": pool,
    }


# Live wires outside the mul pool at an off-plateau mul site: the
# x1, y1 and A words plus the 7 machine singles.  A ceiling of
# MUL_BASE + k leaves k wires for the pool.
MUL_BASE = 3 * N + 7


class MulMachine:
    """The mul machinery of one point-add build: 7 machine singles
    (carry, flag, top, tq, clean3) and the lazy k-block pool, with the
    role dict ma the gadgets address. The singles allocate at
    construction; the pool starts empty and pool_width grows it at mul
    sites. Every shed frees LIFO, so the walk legs and the square
    window run on a census that carries no idle mul wires.
    """

    def __init__(self, b):
        self.b = b
        self.singles = {
            "carry": b.qubit(),
            "flag": b.qubit(),
            "top": b.qubit(),
            "tq": b.qubit(),
            "clean3": [b.qubit() for _ in range(3)],
        }
        self.pool = []  # lazy: pool_width grows it at mul sites
        self.ma = mk_ma(self.singles, self.pool)

    def claim(self):
        """Claim the construction-time singles and the (empty) pool."""
        self.b.claim("mul/singles", self._singles_list())
        self.b.claim("mul/pool", self.pool)

    def pool_width(self, k):
        """Shrink/regrow the mul pool to k wires (LIFO tail; the
        plateau narrowing — the freed wires lower the walk-phase
        census).
        """
        b, pool = self.b, self.pool
        cur = len(pool)
        if k < cur:
            b.release_recycle("mul/pool", pool[k:])
            del pool[k:]
        elif k > cur:
            got = [b.qubit() for _ in range(k - cur)]
            b.claim("mul/pool", got)
            pool.extend(got)
        self.ma["anc"] = pool

    def _singles_list(self):
        singles = self.singles
        return [
            singles[k] for k in ("carry", "flag", "top", "tq") if singles[k] is not None
        ] + list(singles["clean3"])

    def singles_off(self):
        """Free the mul singles (idle across the walk legs; the freed
        wires lower the walk arena census).
        """
        b = self.b
        live = self._singles_list()
        b.release_recycle("mul/singles", live)
        self.singles.update(carry=None, flag=None, top=None, tq=None, clean3=[])

    def singles_on(self, seam=False):
        """seam=True: mint just the lazy-cma flag — the plateau
        row-mul path (gc_lazy_cma / kb_fold_const lifts / sign_fix)
        reads no other single, and the freed 6 lower the seam
        census.
        """
        b, singles = self.b, self.singles
        if seam:
            singles.update(carry=None, top=None, tq=None, clean3=[], flag=b.qubit())
        else:
            singles.update(
                carry=b.qubit(),
                flag=b.qubit(),
                top=b.qubit(),
                tq=b.qubit(),
                clean3=[b.qubit() for _ in range(3)],
            )
        b.claim("mul/singles", self._singles_list())
        self.ma.update(
            carry=singles["carry"],
            flag=singles["flag"],
            top=singles["top"],
            tq=singles["tq"],
            clean3=singles["clean3"],
        )

    def on(self, k, seam=False):
        """Arm the mul machinery: singles (seam variant: the lazy-cma
        flag alone), then the pool at k wires.
        """
        self.singles_on(seam=seam)
        self.pool_width(k)

    def off(self):
        """Shed the mul machinery: pool parked at width 0, singles
        freed.
        """
        self.pool_width(0)
        self.singles_off()


def row_reduce(b, ids, p, n, word, cin, inv=False):
    """Fold a parked magnitude row's bits >= n into its low word:
    lo += hi*c (mod 2^n), c = 2^n mod p — after which a Horner scan
    of ids[:n] alone is value-identical mod p.  hi stays untouched
    (the scan just ignores it) and the inverse subtracts the same
    sum, so the row round-trips EXACTLY on every lane, including the
    low-word wrap lanes (lo + hi*c >= 2^n, P ~ 2^-(n-59) under the
    fuzz law): a wrap only shifts the product by -c mod p — a priced
    escape, contained to its shot.

    word: n scratch wires, all |0>, returned |0> (the free out word
    at a fwd seam entry / post-inv-scan, or the HMR-freed word at the
    s2 unpark).  cin: one |0> carry root (a pool wire), restored.
    The hi*c accumulator builds tap-by-tap at the running value
    bound (Cuccaro cells on word slices, operand padding from the
    |0> tail), folds into lo with one word-width Cuccaro, and
    unbuilds in exact reverse.
    """
    hi = list(ids[n:])
    if not hi:
        return
    lo = list(ids[:n])
    c = (1 << n) % p
    taps = [j for j in range(c.bit_length()) if (c >> j) & 1]
    m = len(hi)

    def taps_run(sub):
        order = list(reversed(taps)) if sub else taps
        for j in order:
            part = sum(1 << jj for jj in taps if jj <= j)
            if j == taps[0]:
                for i, w in enumerate(hi):  # tap 0 slot: plain
                    b.cx(w, word[j + i])  # copy, self-inverse
                continue
            top = (part * ((1 << m) - 1)).bit_length()
            acc = word[j:top]
            pad = word[top : top + (len(acc) - m)]
            a = hi + pad
            if sub:
                _cdkm_sub(b, a, acc, cin)
            else:
                add(b, acc, a, gl=0, cin=cin)

    taps_run(False)
    if inv:
        _cdkm_sub(b, word[:n], lo, cin)
    else:
        add(b, lo, word[:n], gl=0, cin=cin)
    taps_run(True)


def halve_horner_scan(b, fvrow, y, out, p, n, ma, halve, inv=False, scan=None):
    """out += Fv * y * 2^-W mod p (W = the row's scheduled width), Fv
    the parked SIGNED row (fvrow: WRow; the MAGNITUDE bits drive the
    rounds, the two's-complement encoding is pre-lifted by the
    caller; the residue-class sign is the caller's sign fix).

    HALVE-Horner, bottom to top: out <- (out + b_k * y) * 2^-1 per
    round.  The mod-p halve (the caller's `halve` closure) replaces
    the dbl-Horner doubling at the same gadget class and absorbs W
    of the walk's fixed pre-freeze scale 2^-Tmin — the seam appends
    the remaining Tmin-W halves and the rider legs arm only the
    data-bearing window [Tmin, S).  cma = the k-block per-round
    gadget (exact below n=128, lazy at n=256); ma['anc'] picks the
    pool plan.

    Zero-target edge: out is |0> at the fwd entry (and returns to
    |0> at the inv exit), so the first fwd / last inv round's cma is
    a CCX copy fan (the mul_modp zt rewrite, map-identical).
    """
    ids = list(fvrow.ids)
    if scan is not None:
        ids = ids[:scan]
    order = range(len(ids)) if not inv else range(len(ids) - 1, -1, -1)
    ma = dict(ma)
    ma["dirty"] = list(y) + ([ma["tq"]] if ma.get("tq") is not None else [])
    for k in order:
        if not inv:
            if k == 0:
                for yi, oi in zip(y, out):
                    b.ccx(ids[0], yi, oi)
            else:
                _mul_cma(b, ids[k], y, out, p, n, ma)
            halve(False)
        else:
            halve(True)
            if k == 0:
                for yi, oi in zip(y, out):
                    b.ccx(ids[0], yi, oi)
            else:
                _mul_cma(b, ids[k], y, out, p, n, ma, inv=True)


# ---------------------------------------------------------------------------
# the composed inversion sandwich (one per idiv/imul walk pair)
# ---------------------------------------------------------------------------


class DivstepInvBase:
    """One windowed walk instance + its seam closures, bound to the
    build's wire plan.  xw hosts g/fv; the start() host word hosts
    f/gv (the A word at walk-A, the HMR-freed y word at walks B/C/D);
    arm_rider names the word the scale fix rides.
    """

    WALK: type[HmrWalk | StagedWalk]  # the engine's walk class
    composed_params: Callable[[], dict]  # the engine's parameter callables
    composed_spans: Callable[[], list[int]]
    walk: WalkWBase  # live from start() to finish()
    rider_reg: list[int]  # set by arm_rider

    def __init__(self, b, xw, prm=None):
        prm = prm or self.composed_params()
        self.b = b
        self.prm = prm
        self.n = prm["n"]
        self.p = prm["p"]
        WF, WG, WFV, WGV = full_windows()
        self.sched = Schedule(prm["S"], WF, WG, WFV, WGV, self.n)
        self.spans = self.composed_spans()
        self.forms = FORMS_P
        self.xw = list(xw)
        self.walk = None  # ty: ignore[invalid-assignment]  # pre-start sentinel
        self.Tmin = prm.get("Tmin", 0)

    # -- gadget closures on borrowed row/word dirt --------------------
    def _clean3(self):
        b = self.b
        return [b.qubit() for _ in range(3)]

    def _free3(self, w):
        for q in reversed(w):
            self.b.recycle(q)

    def _rider_stage_exact(self, mode, alive=None, reg=None):
        """Exact mod-p halve/double: exact on every lane, zero
        doors.  halve: mint pw = alive AND
        A[0] (CCX), rotate parity to top, add ((p+1)/2 - 2^(n-1)) mod
        2^n under pw, then kill pw against the range witness
        [A >= (p+1)/2] (the halve is a bijection: odd inputs land
        exactly in the upper range) — one comparator each way.
        double = exact inverse.  alive=None is the unconditional form
        (the seam's fixed-scale blocks, `reg` explicit): parity mint
        and rotation drop to CX/swap, comparator structure
        unchanged.
        """
        b = self.b
        reg = list(reg if reg is not None else self.rider_reg)
        n = len(reg)
        thr = (self.p + 1) >> 1
        K = (thr - (1 << (n - 1))) % (1 << n)
        # gidney_cmp_ge flips out on carry(reg + kb): [reg >= thr]
        # needs kb = 2^n - thr
        kb_ge = LC.int_bits(((1 << n) - thr) % (1 << n), n)
        dirty = self.walk.dirty_ids(n)
        c3 = self._clean3()

        def pw_xor(t, pw):
            if alive is None:
                b.cx(t, pw)
            else:
                b.ccx(alive, t, pw)

        def rot(i, j):
            if alive is None:
                b.swap(reg[i], reg[j])
            else:
                b.cswap(alive, reg[i], reg[j])

        def cmp_kill(pw):
            t = b.qubit()
            LC.gidney_cmp_ge(b, reg, kb_ge, t, dirty[:n], c3)
            pw_xor(t, pw)
            LC.gidney_cmp_ge(b, reg, kb_ge, t, dirty[:n], c3)
            b.recycle(t)
            b.recycle(pw)

        def cmp_mint():
            pw = b.qubit()
            t = b.qubit()
            LC.gidney_cmp_ge(b, reg, kb_ge, t, dirty[:n], c3)
            pw_xor(t, pw)
            LC.gidney_cmp_ge(b, reg, kb_ge, t, dirty[:n], c3)
            b.recycle(t)
            return pw

        if mode == "halve":
            pw = b.qubit()
            pw_xor(reg[0], pw)
            for i in range(n - 1):
                rot(i, i + 1)
            LC.gidney_add_const(b, reg, LC.int_bits(K, n), dirty[: n - 1], c3, ctrl=pw)
            cmp_kill(pw)
        else:
            pw = cmp_mint()
            LC.gidney_add_const(
                b, reg, LC.int_bits(((1 << n) - K) % (1 << n), n), dirty[: n - 1], c3, ctrl=pw
            )
            for i in range(n - 2, -1, -1):
                rot(i, i + 1)
            pw_xor(reg[0], pw)
            b.recycle(pw)
        self._free3(c3)

    # -- lifecycle ----------------------------------------------------
    def start(self, host_w, phase_fwd, phase_rev):
        """Construct the walk on (xw, host_w) and seed the rows.  The
        host word carries f/gv (A at walk-A, the HMR-freed y1 word at
        walks B/C/D); row growth beyond the two words draws fresh
        straddle ids (max-concurrency census).
        """
        self.walk = self.WALK(
            self.b,
            self.prm,
            self.forms,
            self.spans,
            self.sched,
            self.xw,
            list(host_w),
            [],
            phase_fwd=phase_fwd,
            phase_rev=phase_rev,
        )
        self.walk.init_rows()
        return self.walk

    def sign_fix(self, target, ma, inv=False):
        """target <- p - target under nc = s_f XOR s_fv (out =
        |fv|*y*2^-t*f*sign(fv); f*sign(fv) = -1 exactly when s_f !=
        s_fv; the closer killed s_gv).  Controlled mod-p negate: CX
        fan + fold-class subtract of c-1 — the fan gives ~v =
        2^n-1-v and 2^n - c = p exactly (c = 2^n mod p made odd), so
        ~v - (c-1) = p - v as an integer.  The fold's carry-past-
        window door is the FOLD_WIN class (~(c-1)/2^58 per call).
        """
        b = self.b
        sf, sv = self.walk.f.sign, self.walk.fv.sign
        b.cx(sv, sf)  # sf holds the negate control
        nc = sf
        n = len(target)
        c = odd_fold_const(n, self.p)
        assert (1 << n) - c == self.p
        if not inv:
            for q in target:
                b.cx(nc, q)
            kb_fold_const(b, nc, c - 1, list(target), ma["anc"], inv=True, win=FOLD_WIN)
        else:
            kb_fold_const(b, nc, c - 1, list(target), ma["anc"], inv=False, win=FOLD_WIN)
            for q in target:
                b.cx(nc, q)
        b.cx(sv, sf)

    def _fold_stage(self, mode, reg, anc, alive=None):
        """One mod-p halve (mode='halve') or double (mode='double') of
        reg by the windowed fold class: w = reg[0] (under alive), fold
        of -(c-1) under w, rotate (under alive). The even constant c-1
        leaves reg[0] fixed, so w releases against it. alive=None is
        the unconditional form (the seam's fixed-scale halves): w is a
        CX copy and the rotation is plain swaps. Exact for reg >= c-1;
        reg < c-1 (the tiny-A door) and the fold window carry (the
        FOLD_WIN door class) are priced escapes — README.
        """
        b, g = self.b, G(self.b)
        n = len(reg)
        cm1 = odd_fold_const(n, self.p) - 1

        def mint():
            if alive is not None:
                return g.andp(alive, reg[0])
            w = b.qubit()
            b.cx(reg[0], w)
            return w

        def kill(w):
            if alive is not None:
                g.andm(w, alive, reg[0])
                return
            b.cx(reg[0], w)
            b.recycle(w)

        def rot(i):
            if alive is None:
                b.swap(reg[i], reg[i + 1])
            else:
                b.cswap(alive, reg[i], reg[i + 1])

        if mode == "halve":
            w = mint()
            kb_fold_const(b, w, cm1, reg, anc, inv=True, win=FOLD_WIN)
            kill(w)
            for i in range(n - 1):
                rot(i)
        else:
            for i in range(n - 2, -1, -1):
                rot(i)
            w = mint()
            kb_fold_const(b, w, cm1, reg, anc, inv=False, win=FOLD_WIN)
            kill(w)

    def _halve_out(self, out, anc, inv=False):
        """One unconditional mod-p halve (inv: double) of `out`: the
        seam runs on every lane, so the fold stage drops its alive
        control.
        """
        self._fold_stage("double" if inv else "halve", list(out), anc)

    def row_mul(self, y, out, ma, inv=False):
        """out += |fv| * y * 2^-Tmin mod p on the parked signed row.
        fwd: magnitude lift, mod-p pre-reduction of the row at full
        width (row_reduce: lo += hi*c mod 2^n, borrowing the |0> out
        word; the top bits' rounds vanish, the halve count is Tmin
        regardless), halve-Horner rounds (absorbing W of the fixed
        pre-freeze scale), Tmin-W appended halves (the rest of the
        fixed scale; the rider legs carry only the data window
        [Tmin, S)).  The un-reduce + un-lift are NOT emitted here:
        the s2 call defers them to row_unpark (after HMR frees the
        multiplicand word for scratch), and the s6 apply/erase pair
        carries them on its inv call — the row parks in magnitude
        form through the sign fix + phase fixup, which never touch
        the row ids.  inv: the exact mirror (fixed halves, inverse
        scan, un-reduce borrowing the again-|0> out word, lift
        undone).  The lift's +1 is a fold-class const-add (carry
        past FOLD_WIN ~ 2^-58 door).
        """
        b = self.b
        row = self.walk.fv
        scan = min(len(row.ids), self.n)
        extra = self.Tmin - scan

        def lift(inv_):
            self._row_lift(row, ma, inv_)

        def halve(inv_):
            self._halve_out(out, ma["anc"], inv=inv_)

        if not inv:
            lift(False)
            self._row_reduce(row, out, ma)  # out |0>: borrow it
            halve_horner_scan(b, row, y, out, self.p, self.n, ma, halve, scan=scan)
            for _ in range(max(0, extra)):
                halve(False)
            for _ in range(max(0, -extra)):
                halve(True)
        else:
            for _ in range(max(0, -extra)):
                halve(False)
            for _ in range(max(0, extra)):
                halve(True)
            halve_horner_scan(b, row, y, out, self.p, self.n, ma, halve, inv=True, scan=scan)
            # post-inv-scan out is |0> again: borrow it
            self._row_reduce(row, out, ma, inv=True)
            lift(True)

    def _row_lift(self, row, ma, inv_):
        b = self.b
        if not inv_:
            for w in row.ids:
                b.cx(row.sign, w)
            kb_fold_const(b, row.sign, 1, list(row.ids), ma["anc"], win=FOLD_WIN)
        else:
            kb_fold_const(b, row.sign, 1, list(row.ids), ma["anc"], inv=True, win=FOLD_WIN)
            for w in reversed(row.ids):
                b.cx(row.sign, w)

    def _row_reduce(self, row, zeros, ma, inv=False):
        """row_reduce plumbing.  zeros: a |0> word to borrow as the
        accumulator/pad scratch (the out word at the fwd entry and
        after the inv scan; the HMR-demolished multiplicand word at
        the s2 unpark — demolished ids stay allocated, so borrowing
        them is the census-neutral spelling).  Carry root: a pool
        wire, |0> between cma calls.
        """
        if len(row.ids) <= self.n:
            return
        row_reduce(
            self.b, list(row.ids), self.p, self.n, list(zeros)[: self.n], ma["anc"][0], inv=inv
        )

    def row_unpark(self, zeros, ma):
        """Deferred un-reduce + lift_back for the s2 seam: runs after
        HMR demolished the multiplicand word (zeros) to |0>, which
        supplies the scratch.
        """
        row = self.walk.fv
        self._row_reduce(row, zeros, ma, inv=True)
        self._row_lift(row, ma, True)

    def finish(self):
        self.walk.uninit_rows()
        self.walk.release_claims()
        for w in self.walk.machine_wires():
            self.b.recycle(w)
        assert self.walk.pool.dyn_live == 0, "straddle leak"
        self.walk = None  # ty: ignore[invalid-assignment]  # torn-down sentinel


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------


def build_point(build_circuit, p, n):
    """Full-width build on a fresh Builder.  Returns (b, regs)."""
    b = Builder()
    b.set_phase("init")
    regs = build_circuit(b, p, n)
    return b, regs


def emit_tail(b, label, entry_file):
    """Run the shared emit tail on a built circuit: per-phase census
    table, the cost pin and ancilla-leak guard (assert_metrics reads
    the entry_file pin from meta.json), emit_and_report.
    """
    summ = b.phase_summary()
    tof = summ["total_emitted_toffoli"]
    for name in b._phase_order:
        st = b._phase_stats[name]
        print(f"# {name:>16}  toffoli={st['toffoli']:>9,}  peak={st['peak']:>4}", file=sys.stderr)
    print(f"divstep {label}: {tof:,} Toffoli x {b.peak_qubits} qubits", file=sys.stderr)
    assert_metrics(b, entry_file)
    emit_and_report(b)


def run_main(label, build_circuit, p, n, entry_file):
    """Build on a fresh Builder and run the shared emit tail."""
    b, _ = build_point(build_circuit, p, n)
    emit_tail(b, label, entry_file)


# ===========================================================================
# the Hmr engine (measurement erase; fill.py, wide.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# dials: configure() installs a point's cfg as module state (no
# environment reads; the point files are the configuration authority)
# ---------------------------------------------------------------------------
QCAP = None  # the arena ceiling (composed_params Qcap)
WIDE = None  # subtract-erase variant: fresh host words, no
# measurement fixup pairs


def hmr_configure(cfg):
    """Install a point's dials."""
    global QCAP, WIDE
    QCAP = int(cfg["qcap"])
    WIDE = bool(cfg["wide"])


def hmr_composed_params(p=P_MOD, n=N):
    # S = 533: the schedule cut (1e8-walk tstop census:
    # P(tstop > 533) = 9.3e-6/walk booked; the proof bound is 590;
    # gen_constants.py derives S).
    # A lane still live at the cutoff fails its shot through the
    # unified budget, and the rider — armed on the same window —
    # mis-scales exactly the same lanes, so the schedule and rider
    # channels are one correlated escape.  The walk indexes [0, S).
    #
    # Tmin = 498 (_tstop_tmin, the quantile-priced pre-freeze pin —
    # the booked rate is P(tstop <= 498) = 3e-6/walk).  Steps t < Tmin
    # carry no freeze machinery and run the
    # pre-freeze spellings (HmrWalk); an early-frozen
    # lane walks on undetected and fails its shot through the
    # contained z-cap / window channels.
    # Qcap: the peak census ceiling (the point's qcap dial) the
    # walk's hybrid adders fill their transient carries to (the walk
    # kit's ctrl_add_rows), so the conversion is census-neutral by
    # construction.
    # Wz = 6: the z counter's per-walk max width census (j = 5 fill
    # census, 1e8 walks) shows P(z needs > 6 bits) =
    # 9e-8/walk — the z-cap escape at this pin, booked to the
    # unified ledger (contained and partially self-healing).
    # Cr = 60: the R-stack cap, hand-set; spans clip at the brigade
    # reach 59 (cap exceed 2.4e-6/walk booked).
    assert QCAP is not None, "configure(cfg) not called"
    return {
        "p": p,
        "n": n,
        "S": _sched_s(),
        "Tmin": _tstop_tmin(),
        "Wz": _z_width(),
        "Cr": 60,
        "Qcap": QCAP,
    }


def hmr_composed_spans():
    """Return the deployed brigade reach: the quantile fill section (1e8-walk
    C census, greedy at the 1e-6/walk any-step budget, 2e8 holdout);
    the absmax+8 grade stays committed as the reference.
    """
    return list(_CONSTANTS["spans_q"]["reach"])


# ---------------------------------------------------------------------------
# the composed inversion sandwich (one per idiv/imul walk pair)
# ---------------------------------------------------------------------------


class HmrDivstepInv(DivstepInvBase):
    """The engine-specific rider layer over the shared sandwich:
    the windowed fold-class stage.
    """

    WALK = HmrWalk
    composed_params = staticmethod(hmr_composed_params)
    composed_spans = staticmethod(hmr_composed_spans)

    def _rider_stage_full(self, mode, alive):
        """Full-width rider stage: the windowed fold class (_fold_stage)
        under the alive control, on 16 fresh fold ancillas. The tiny-A
        door has P ~ 2^-224 per stage under the fuzz law.
        """
        b = self.b
        anc = [b.qubit() for _ in range(16)]
        self._fold_stage(mode, self.rider_reg, anc, alive)
        for q in reversed(anc):
            b.recycle(q)

    def arm_rider(self, reg, legs):
        """Arm the fold-class stage.  Stages arm on the data window
        [Tmin, S) — the schedule itself bounds the span, and a lane
        still live at the cut fails through the cut's own priced
        channel (the two channels fail the same lanes).
        """
        self.rider_reg = list(reg)
        self.walk.rider = {"legs": legs, "stage": self._rider_stage_full}


# ---------------------------------------------------------------------------
# the point-add flow
# ---------------------------------------------------------------------------

# Pool sizing: the k-block mul family costs ~1,275 Toffoli LESS per
# pool wire (fewer keeper blocks, less erase recompute), so each pool
# rides its phase's width ceiling exactly.  The pool allocates lazily
# (pool_width), so no phase pays for a pool it does not use.
# Operating point: the point's arena ceiling (the qcap dial) — width
# governs.  The armed-step machinery moments of the walk legs bind
# the peak there, and every pool/vent/adder ceiling ties to that
# arena exactly.
HMR_POOL_PLATEAU_SEAM = 24  # s6 seam: the host word stays live, so the
# pool holds to the plan floor there
HMR_POOL_PLATEAU = 24  # seam plateau row-muls: 24 = the k-block plan
# floor for n=256 (_plan needs sum(caps) >= nc)


def _hmr_pool_off():
    # off-plateau muls fill MUL_BASE -> the dial, saturating at the
    # single-block plan (256)
    assert QCAP is not None, "configure(cfg) not called"
    return min(256, QCAP - MUL_BASE)


def hmr_build_circuit(b, p, n, prm=None):  # noqa: PLR0915 - straight-line circuit emitter
    """Emit the point addition |P> -> |P+Q>; Q per shot on the
    Bit-kind inputs.  Returns (x1, y1, ox, oy).
    """
    x1, y1, ox, oy = LC.declare_point_io(b, n)
    # per-shot cbspecs (free classical networks of the addend bits)
    S_x2, S_y2, S_3x2 = LC.point_cbspecs(b, ox, oy, p, n)

    A = b.qubits(n)
    mm = MulMachine(b)
    singles, ma = mm.singles, mm.ma
    b.claim("io/x1", x1)
    b.claim("io/y1", y1)
    b.claim("A", A)
    mm.claim()

    inv = HmrDivstepInv(b, x1, prm=prm)

    with b.phase("s1-const-subs"):
        LC.const_modp(b, x1, S_x2, singles["flag"], y1, singles["clean3"], p, n, inv=True)
        LC.const_modp(b, y1, S_y2, singles["flag"], x1, singles["clean3"], p, n, inv=True)

    # ================= s2-idiv: y1 <- y1 / x1 ==========================
    # walk-A fwd (g rides x1; f/gv host = the |0> A word) -> closer
    # (fv parked signed; A word back |0>) -> plateau row-mul A =
    # |fv|*y1 + sign fix (A = fv*y1*sgn = y1*x1^{-1}*2^{+tstop}) ->
    # HMR(y1) -> closer_inv (f/gv rehomed to the freed y1 word) ->
    # walk-B rev with rider halves (A *= 2^{-tstop}) -> recompute/
    # phase-fix/erase y1 -> swap(y1, A).
    # The mul pool and its plan exist only at mul sites: the walk legs
    # allocate their own scratch (decode chains, zero-test keepers,
    # rider fold anc), so the pool parks at width 0 around them and
    # the walk census does not carry idle pool wires.
    with b.phase("s2-walkA"):
        mm.off()
        inv.start(A, "s2-walkA", "s2-walkB")
        inv.walk.emit_fwd()
    if WIDE:
        with b.phase("s2-seam"):
            inv.walk.closer()
            mm.on(256, seam=True)
            inv.row_mul(y1, A, ma)
            inv.sign_fix(A, ma)
            # wide variant: y1 stays LIVE through the reverse walk (no
            # measurement erase). The un-reduce borrows a fresh |0> word
            # and the walk's f/gv rehome to a fresh host word.
            mm.pool_width(13)  # unpark: carry root + the
            # row-lift fold at its k-block plan floor
            zw = b.qubits(n)
            b.claim("wide/unpark", zw)
            inv.row_unpark(zw, ma)
            b.release_recycle("wide/unpark", zw)
            wideB = b.qubits(n)
            b.claim("wide/host", wideB)
            mm.off()
            inv.walk.pool.rebind_band("fw", list(wideB))
            inv.walk.closer_inv()
            inv.arm_rider(A, {"rev": "halve"})
        with b.phase("s2-walkB"):
            inv.walk.emit_rev()
            inv.walk.rider = None
            inv.finish()
            b.release_recycle("wide/host", wideB)
        with b.phase("s2-erase"):
            # u = lambda * d exactly, so one inverse product erases the
            # numerator in place of the measure + recompute-fixup pair
            mm.on(_hmr_pool_off())
            mul_modp(b, x1, A, y1, p, n, ma, inv=True)  # y1 -= x1*A = 0
            for yq, aq in zip(y1, A):
                b.swap(yq, aq)

    else:
        with b.phase("s2-seam"):
            mm.on(HMR_POOL_PLATEAU, seam=True)
            inv.walk.closer()
            inv.row_mul(y1, A, ma)
            inv.sign_fix(A, ma)
            mbits = [b.hmr(q) for q in y1]
            inv.row_unpark(y1, ma)
            inv.walk.pool.rebind_band("fw", list(y1))
            inv.walk.closer_inv()
            inv.arm_rider(A, {"rev": "halve"})
            mm.off()
        with b.phase("s2-walkB"):
            inv.walk.emit_rev()
            inv.walk.rider = None
            inv.finish()
        with b.phase("s2-recompute"):
            mm.on(_hmr_pool_off())
            mul_modp(b, x1, A, y1, p, n, ma)  # y1 = x1*A = y exact
            for q, m in zip(y1, mbits):
                b.z(q, cond=m)  # Z^b phase fixup
            mul_modp(b, x1, A, y1, p, n, ma, inv=True)
            for yq, aq in zip(y1, A):
                b.swap(yq, aq)

    # ================= s4: x1 <- 3*Q.x - lambda^2 ======================
    # The fused square subtracts lambda^2 from x1 term by term, so no
    # square register survives the phase.  The A word is |0> and idle
    # here; it is freed so the square's carries can use its width, and
    # allocated again afterwards.
    with b.phase("s4-sub-squ"):
        mm.pool_width(0)
        b.release_recycle("A", A)
        subtract_square(b, y1, x1, QCAP)
        b.set_phase("s4-sub-squ")
        LC.const_modp(b, x1, S_3x2, singles["flag"], y1, singles["clean3"], p, n)
        A[:] = b.qubits(n)
        b.claim("A", A)
        mm.pool_width(_hmr_pool_off())

    # ================= s6-imul: y1 <- x1 * y1 ==========================
    # A = x1*y1 (exact, tape empty) -> HMR(y1) -> walk-C fwd on x1
    # (f/gv host = the freed y1 word; rider halves: A *= 2^{-tstop})
    # -> closer -> plateau row-mul y1 = |fv|*A + sign fix (= y exact)
    # -> Z^b -> erase pair -> closer_inv -> walk-D rev with rider
    # doubles (A restored to x1*y1) -> swap(y1, A).
    with b.phase("s6-mul1"):
        mul_modp(b, x1, y1, A, p, n, ma)
    if WIDE:
        with b.phase("s6-walkC"):
            # wide variant: lambda stays LIVE (no measurement erase); the
            # walk's f/gv host is a fresh word
            mm.off()
            wideC = b.qubits(n)
            b.claim("wide/host", wideC)
            inv.start(wideC, "s6-walkC", "s6-walkD")
            inv.arm_rider(A, {"fwd": "halve", "rev": "double"})
            inv.walk.emit_fwd()
        with b.phase("s6-seam"):
            inv.walk.closer()
            mm.on(HMR_POOL_PLATEAU_SEAM, seam=True)
            # lambda = |fv|*A*2^-Tmin up to the endpoint sign, so the
            # inverse sign fix + ONE inverse row-mul erase it — no
            # recompute pair, no phase fixup. row_mul(inv) is the exact
            # adjoint of the forward call, whose first moves unwind the
            # scan against a PREPPED row (lifted magnitude, mod-p
            # reduced); no forward call ran here, so the prep is applied
            # explicitly first. The closer-emptied host word supplies the
            # reduce's |0> borrow.
            inv.sign_fix(y1, ma, inv=True)
            inv._row_lift(inv.walk.fv, ma, False)
            inv._row_reduce(inv.walk.fv, list(wideC), ma)
            inv.row_mul(A, y1, ma, inv=True)  # y1 -> |0>, row unprepped
            mm.off()
            inv.walk.closer_inv()
        with b.phase("s6-walkD"):
            inv.walk.emit_rev()
            inv.walk.rider = None
            inv.finish()
            b.release_recycle("wide/host", wideC)
            mm.on(_hmr_pool_off())
            for yq, aq in zip(y1, A):
                b.swap(yq, aq)

    else:
        with b.phase("s6-walkC"):
            mbits2 = [b.hmr(q) for q in y1]
            mm.off()
            inv.start(y1, "s6-walkC", "s6-walkD")
            inv.arm_rider(A, {"fwd": "halve", "rev": "double"})
            inv.walk.emit_fwd()
        with b.phase("s6-seam"):
            mm.on(HMR_POOL_PLATEAU, seam=True)
            inv.walk.closer()
            inv.row_mul(A, y1, ma)  # y1 = |fv|*A*2^-Tmin
            inv.sign_fix(y1, ma)  # y1 = y exact
            for q, m in zip(y1, mbits2):
                b.z(q, cond=m)
            inv.sign_fix(y1, ma, inv=True)
            # erase pair: the facing lifts cancel (the row parks in
            # magnitude form through the sign fix + fixup, which never
            # touch the row ids; the inv call carries the un-reduce +
            # un-lift)
            inv.row_mul(A, y1, ma, inv=True)
            inv.walk.closer_inv()
            mm.off()
        with b.phase("s6-walkD"):
            inv.walk.emit_rev()
            inv.walk.rider = None
            inv.finish()
            mm.on(_hmr_pool_off())
            for yq, aq in zip(y1, A):
                b.swap(yq, aq)

    # ================= s7-final ========================================
    with b.phase("s7-final"):
        LC.neg_modp(b, x1, y1, singles["clean3"], p, n)
        LC.const_modp(b, x1, S_x2, singles["flag"], y1, singles["clean3"], p, n)
        LC.const_modp(b, y1, S_y2, singles["flag"], x1, singles["clean3"], p, n, inv=True)

    # ================= teardown ========================================
    b.set_phase("close")
    b.release("A", A)
    mm.off()
    for w in reversed(list(A)):
        b.recycle(w)
    return x1, y1, ox, oy


def hmr_main(cfg, entry_file):
    """Configure from a point's cfg and emit through the shared tail
    (build, census table, pin assert, emit_and_report).
    """
    hmr_configure(cfg)
    run_main("hmr", hmr_build_circuit, P_MOD, N, entry_file)


# ===========================================================================
# the staged-decode engine (j5.py .. j9.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# dials: configure() installs a point's cfg as module state (no
# environment reads; the point files are the configuration authority)
# ---------------------------------------------------------------------------

# The arena ceiling (the peak census the walk's hybrid adders fill
# their transient carries to; composed_params Qcap).
OPT_QCAP = None

# The raw-record store threshold on v2(fv'): decode levels
# 2..JSTORE-1 emit, deeper branches store.  Each threshold carries
# its census section (_spans; regenerate: python gen_constants.py).
OPT_JSTORE = None


def staged_configure(cfg):
    """Install a point's dials."""
    global OPT_JSTORE, OPT_QCAP
    OPT_JSTORE = int(cfg["jstore"])
    OPT_QCAP = int(cfg["qcap"])


def _staged_spans(jstore):
    """Census section + brigade cap for a store threshold: j = 5 is
    the reference section "spans_q" (cap 60, hand-set); deeper
    thresholds carry their own sections spans_q_j<j>.
    """
    if jstore == 5:
        return "spans_q", 60
    key = f"spans_q_j{jstore}"
    return key, int(_CONSTANTS[key]["cap"])


def staged_composed_params(p=P_MOD, n=N):
    # S = 533: the schedule cut (1e8-walk tstop census:
    # P(tstop > 533) = 9.3e-6/walk booked; the proof bound is 590;
    # gen_constants.py derives S).
    # A lane still live at the cutoff fails its shot through the
    # unified budget, and the rider — armed on the same window —
    # mis-scales exactly the same lanes, so the schedule and rider
    # channels are one correlated escape.  The walk indexes [0, S).
    #
    # Tmin = 498 (_tstop_tmin, the quantile-priced pre-freeze pin —
    # the booked rate is P(tstop <= 498) = 3e-6/walk).  Steps t < Tmin
    # carry no freeze machinery and run the
    # pre-freeze spellings (StagedWalk); an early-frozen
    # lane walks on undetected and fails its shot through the
    # contained z-cap / window channels.
    # Qcap: the peak census ceiling (the point's qcap dial) the
    # walk's hybrid adders fill their transient carries to (the walk
    # kit's ctrl_add_rows).
    # Wz = 6: the z counter's per-walk max width census (j = 5 fill
    # census, 1e8 walks) shows P(z needs > 6 bits) =
    # 9e-8/walk — the z-cap escape at this pin, booked to the
    # unified ledger (contained and partially self-healing).
    # Cr: the R-stack cap at the deployed store threshold, from the
    # matching census section (1e8-walk census + cap sweep each).
    assert OPT_QCAP is not None, "configure(cfg) not called"
    cr = _staged_spans(OPT_JSTORE)[1]
    return {
        "p": p,
        "n": n,
        "S": _sched_s(),
        "Tmin": _tstop_tmin(),
        "Wz": _z_width(),
        "Cr": cr,
        "Qcap": OPT_QCAP,
        "Jstore": OPT_JSTORE,
    }


def staged_composed_spans():
    """Return the brigade reach at the deployed store threshold (section
    spans_q_j<OPT_JSTORE>: 1e8-walk C census, the deployed
    Laplace-smoothed greedy at 1e-6/walk, fresh 2e8 holdout).
    """
    return list(_CONSTANTS[_staged_spans(OPT_JSTORE)[0]]["reach"])


# ---------------------------------------------------------------------------
# the composed inversion sandwich (one per idiv/imul walk pair)
# ---------------------------------------------------------------------------


class StagedDivstepInv(DivstepInvBase):
    """The engine-specific rider binding over the shared sandwich:
    the exact comparator stage.
    """

    WALK = StagedWalk
    composed_params = staticmethod(staged_composed_params)
    composed_spans = staticmethod(staged_composed_spans)

    def arm_rider(self, reg, legs):
        """Arm the exact comparator stage (zero doors;
        borrowed-dirty row wires + 3 clean transients).  Stages arm
        on the data window [Tmin, S).
        """
        self.rider_reg = list(reg)
        self.walk.rider = {"legs": legs, "stage": self._rider_stage_exact}


# ---------------------------------------------------------------------------
# the point-add flow
# ---------------------------------------------------------------------------

# Pool sizing: the k-block mul family costs ~1,275 Toffoli LESS per
# pool wire (fewer keeper blocks, less erase recompute), so each pool
# rides its phase's width ceiling exactly.  The pool allocates lazily
# (pool_width), so no phase pays for a pool it does not use.
# Operating point: the point's arena ceiling (the qcap dial) — width
# governs.  The armed-step machinery moments of the walk legs bind
# the peak there, and every pool/vent/adder ceiling ties to that
# arena exactly.
STAGED_POOL_PLATEAU = 24  # seam plateau row-muls: 24 = the k-block plan


# floor for n=256 (_plan needs sum(caps) >= nc)
def _staged_pool_off():
    # off-plateau muls fill MUL_BASE -> the arena ceiling
    assert OPT_QCAP is not None, "configure(cfg) not called"
    return OPT_QCAP - MUL_BASE


def staged_build_circuit(b, p, n, prm=None):
    """Emit the point addition |P> -> |P+Q>; Q per shot on the
    Bit-kind inputs.  Returns (x1, y1, ox, oy).
    """
    x1, y1, ox, oy = LC.declare_point_io(b, n)
    # per-shot cbspecs (free classical networks of the addend bits)
    S_x2, S_y2, S_3x2 = LC.point_cbspecs(b, ox, oy, p, n)

    A = b.qubits(n)
    mm = MulMachine(b)
    singles, ma = mm.singles, mm.ma
    b.claim("io/x1", x1)
    b.claim("io/y1", y1)
    b.claim("A", A)
    mm.claim()

    inv = StagedDivstepInv(b, x1, prm=prm)

    with b.phase("s1-const-subs"):
        LC.const_modp(b, x1, S_x2, singles["flag"], y1, singles["clean3"], p, n, inv=True)
        LC.const_modp(b, y1, S_y2, singles["flag"], x1, singles["clean3"], p, n, inv=True)

    # ================= s2-idiv: y1 <- y1 / x1 ==========================
    # walk-A fwd (g rides x1; f/gv host = the |0> A word) -> closer
    # (fv parked signed; A word back |0>) -> plateau row-mul A =
    # |fv|*y1 + sign fix (A = fv*y1*sgn = y1*x1^{-1}*2^{+tstop}) ->
    # HMR(y1) -> closer_inv (f/gv rehomed to the freed y1 word) ->
    # walk-B rev with rider halves (A *= 2^{-tstop}) -> recompute/
    # phase-fix/erase y1 -> swap(y1, A).
    # The mul pool and its plan exist only at mul sites: the walk legs
    # allocate their own scratch (decode chains, zero-test keepers,
    # rider fold anc), so the pool parks at width 0 around them and
    # the walk census does not carry idle pool wires.
    with b.phase("s2-walkA"):
        mm.off()
        inv.start(A, "s2-walkA", "s2-walkB")
        inv.walk.emit_fwd()
    with b.phase("s2-seam"):
        # the closer pair is Clifford and needs no mul machinery:
        # the plateau pool exists only across the row-mul window, so
        # pool 24 and the closer's wide rows never coexist
        inv.walk.closer()
        mm.on(STAGED_POOL_PLATEAU, seam=True)
        inv.row_mul(y1, A, ma)
        inv.sign_fix(A, ma)
        mbits = [b.hmr(q) for q in y1]
        inv.row_unpark(y1, ma)
        mm.off()
        inv.walk.pool.rebind_band("fw", list(y1))
        inv.walk.closer_inv()
        inv.arm_rider(A, {"rev": "halve"})
    with b.phase("s2-walkB"):
        inv.walk.emit_rev()
        inv.walk.rider = None
        inv.finish()
    with b.phase("s2-recompute"):
        mm.on(_staged_pool_off())
        mul_modp(b, x1, A, y1, p, n, ma)  # y1 = x1*A = y exact
        for q, m in zip(y1, mbits):
            b.z(q, cond=m)  # Z^b phase fixup
        mul_modp(b, x1, A, y1, p, n, ma, inv=True)
        for yq, aq in zip(y1, A):
            b.swap(yq, aq)

    # ================= s4: x1 <- 3*Q.x - lambda^2 ======================
    # The fused square subtracts lambda^2 from x1 term by term
    # (subtract_square); the A word is freed for its width and
    # allocated again afterwards.
    with b.phase("s4-sub-squ"):
        mm.off()
        b.release_recycle("A", A)
        # the mul singles stay off through the square window
        subtract_square(b, y1, x1, OPT_QCAP)
        b.set_phase("s4-sub-squ")
        mm.singles_on()
        LC.const_modp(b, x1, S_3x2, singles["flag"], y1, singles["clean3"], p, n)
        A[:] = b.qubits(n)
        b.claim("A", A)
        mm.pool_width(_staged_pool_off())

    # ================= s6-imul: y1 <- x1 * y1 ==========================
    # A = x1*y1 (exact, tape empty) -> HMR(y1) -> walk-C fwd on x1
    # (f/gv host = the freed y1 word; rider halves: A *= 2^{-tstop})
    # -> closer -> plateau row-mul y1 = |fv|*A + sign fix (= y exact)
    # -> Z^b -> erase pair -> closer_inv -> walk-D rev with rider
    # doubles (A restored to x1*y1) -> swap(y1, A).
    with b.phase("s6-mul1"):
        mul_modp(b, x1, y1, A, p, n, ma)
    with b.phase("s6-walkC"):
        mbits2 = [b.hmr(q) for q in y1]
        mm.off()
        inv.start(y1, "s6-walkC", "s6-walkD")
        inv.arm_rider(A, {"fwd": "halve", "rev": "double"})
        inv.walk.emit_fwd()
    with b.phase("s6-seam"):
        inv.walk.closer()
        mm.on(STAGED_POOL_PLATEAU, seam=True)
        inv.row_mul(A, y1, ma)  # y1 = |fv|*A*2^-Tmin
        inv.sign_fix(y1, ma)  # y1 = y exact
        for q, m in zip(y1, mbits2):
            b.z(q, cond=m)
        inv.sign_fix(y1, ma, inv=True)
        # erase pair: the facing lifts cancel (the row parks in
        # magnitude form through the sign fix + fixup, which never
        # touch the row ids; the inv call carries the un-reduce +
        # un-lift)
        inv.row_mul(A, y1, ma, inv=True)
        mm.off()
        inv.walk.closer_inv()
    with b.phase("s6-walkD"):
        inv.walk.emit_rev()
        inv.walk.rider = None
        inv.finish()
        mm.on(_staged_pool_off())
        for yq, aq in zip(y1, A):
            b.swap(yq, aq)

    # ================= s7-final ========================================
    with b.phase("s7-final"):
        LC.neg_modp(b, x1, y1, singles["clean3"], p, n)
        LC.const_modp(b, x1, S_x2, singles["flag"], y1, singles["clean3"], p, n)
        LC.const_modp(b, y1, S_y2, singles["flag"], x1, singles["clean3"], p, n, inv=True)

    # ================= teardown ========================================
    b.set_phase("close")
    b.release("A", A)
    mm.off()
    for w in reversed(list(A)):
        b.recycle(w)
    return x1, y1, ox, oy


def staged_main(cfg, entry_file):
    """Configure from a point's cfg and emit through the shared tail
    (build, census table, pin assert, emit_and_report).
    """
    staged_configure(cfg)
    run_main("staged", staged_build_circuit, P_MOD, N, entry_file)
