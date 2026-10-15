#!/usr/bin/env python3
"""Regenerate constants.json for lattice-gauge-plaquette circuits/discovered
by running the exact-DP chain-walk search.

The search.  The register walks the lattice holding one running group
product.  It edits the product only at its right end (f/i push g or
g^-1, 35 T each).  The 21-T in-place inversion v moves edits to the
word's other end.  The 1-T anchors l<y>/u<y> connect the identity to
the single-letter word g_y^-1.  The trace is invariant under cyclic
rotation and inversion of a loop's holonomy word.  So each of the 576
Wilson loops can fire at any of its rotation/inversion forms: 8 per
plaquette, 12 per rectangle, 96 forms per site.

  * Morph oracle: the exact cheapest token path between two words, by
    Dijkstra over the graph of contiguous subwords of the endpoint
    words (both orientations) plus the identity.  Edges are push/pop
    (35), invert (21) and anchor (1).  Teardown through the identity is
    a path in this graph, so it needs no special case.
  * Site tour DP: all 64 sites are lattice translates, so one table
    serves them all.  TOUR[c][b] is the cheapest way to fire all 9 loops
    of a site, starting at form c and ending at form b.
  * Junction tables: the raster (x-fastest) site sequence has three
    boundary classes: +x within a row, x-wrap+y at a row end, and
    xy-wrap+z at a plane end.  J[b][c'] is the morph cost across one.
  * Chain DP over the 64-site raster chain.  The walk opens from the
    identity and closes to it.

The morph model is net-flip: a morph may use any number of
inversions, so the register may hold a fired form inverted.  That is
valid because the trace-add is inversion-blind.  The optimum is 3,136
mults + 480 inversions + 64 anchors = 119,904 T.

Determinism.  Per-loop forms enumerate inverted-first (loop_forms).
The backtrack picks end states first.  At every
argmin tie it prefers rectangle forms over plaquette forms, then the
lowest local form index.  Dijkstra paths follow heap order over the
sorted node interning.  Nothing is random and nothing reads the clock.

The encoding.  The DP optimum is row-periodic.  So constants.json
stores the schedule as three row programs (open_row, y_row, z_row) plus
the closing tail.  The flat 4,256-token schedule is open_row, then
twelve lattice translates of y_row and three of z_row (one per plane
crossing), rastered x-fastest over the 4^3 lattice, then the tail.
y_row/z_row link ids are frame-local: link 3*(x + 4*dy + 16*dz) + mu
addresses the link dy rows / dz planes above the row being processed,
mod 4.  open_row and tail are absolute.  reconstruct() is the one
authority for row form -> flat schedule; extract() asserts that the
stored forms reconstruct the searched schedule exactly.

RUN: uv run python gen_constants.py   (about 2 s; writes
     constants.json and reports whether it changed)
"""

import heapq
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "constants.json"

# ---------------------------------------------------------------------------
# Lattice geometry and loop words.  Conventions pinned by the task
# (reference.py) and circuit.py section F: 4^3 periodic lattice, site
# s = x+4y+16z, link l = 3s+mu, plaquette words [(B,+1),(C,-1),(D,-1),(A,+1)]
# and rectangle words [(l3,+1),(l4,-1),(l5,-1),(l6,-1),(l1,+1),(l2,+1)].
# ---------------------------------------------------------------------------
COST_MULT = 35  # f<l> / i<l>: right-multiply by g_l / g_l^-1
COST_INV = 21  # v: invert the register in place
COST_ANCHOR = 1  # l<l> / u<l>: identity <-> g_l^-1


def sidx(x, y, z):
    return (x % 4) + 4 * (y % 4) + 16 * (z % 4)


def shift(s, mu, d=1):
    x, y, z = s % 4, (s // 4) % 4, s // 16
    if mu == 0:
        x += d
    elif mu == 1:
        y += d
    else:
        z += d
    return sidx(x, y, z)


def link(s, mu):
    return 3 * s + mu


def build_words():
    plaq_words = []
    for s in range(64):
        for i in range(3):
            for j in range(i + 1, 3):
                A = link(s, i)
                B = link(shift(s, i), j)
                Cc = link(shift(s, j), i)
                D = link(s, j)
                plaq_words.append([(B, 1), (Cc, -1), (D, -1), (A, 1)])
    rect_words = []
    for s in range(64):
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                l1 = link(s, i)
                l2 = link(shift(s, i), i)
                l3 = link(shift(s, i, 2), j)
                l4 = link(shift(shift(s, i), j), i)
                l5 = link(shift(s, j), i)
                l6 = link(s, j)
                rect_words.append([(l3, 1), (l4, -1), (l5, -1), (l6, -1), (l1, 1), (l2, 1)])
    assert len(plaq_words) == 192 and len(rect_words) == 384
    return plaq_words, rect_words


PLAQ_WORDS, RECT_WORDS = build_words()

# Global loop ids: 0..191 plaquettes, 192..575 rectangles.
LOOPS = [tuple(w) for w in PLAQ_WORDS] + [tuple(w) for w in RECT_WORDS]
LOOP_KIND = ["P"] * 192 + ["R"] * 384


def inv_word(w):
    return tuple((l, -s) for (l, s) in reversed(w))


def rotations(w):
    return [tuple(w[k:]) + tuple(w[:k]) for k in range(len(w))]


def loop_forms(lid):
    """All fireable literal words for loop lid: cyclic rotations of the word
    and of its inverse.  Returns list of (word, rot, invflag).  INVERTED
    forms come first: the tie-break rule (end-state-first backtrack;
    prefer rectangle-loop forms, then the lowest form index) is defined
    over this order.
    """
    w = LOOPS[lid]
    out = []
    seen = set()
    for inv in (1, 0):
        base = inv_word(w) if inv else w
        for k, r in enumerate(rotations(base)):
            if r not in seen:
                seen.add(r)
                out.append((r, k, inv))
    return out


def census(prog):
    """Token census of a flat schedule and its Toffoli price."""
    c = {}
    for op in prog:
        c[op[0]] = c.get(op[0], 0) + 1
    get = c.get
    return {
        "mults": get("f", 0) + get("i", 0),
        "inversions": get("inv", 0),
        "anchors": get("li", 0) + get("ul", 0),
        "P": get("XP", 0),
        "R": get("XR", 0),
        "toffoli": (get("f", 0) + get("i", 0)) * COST_MULT
        + get("inv", 0) * COST_INV
        + (get("li", 0) + get("ul", 0)) * COST_ANCHOR,
    }


KIND = {"f": "f", "i": "i", "l": "li", "u": "ul", "v": "inv", "P": "XP", "R": "XR"}
REV = {v: k for k, v in KIND.items()}

# Structural pin of the schedule's row forms.
FORM_LENS = {"open_row": 260, "y_row": 266, "z_row": 266, "tail": 6}
FORMS = ("open_row", "y_row", "z_row", "tail")


def encode(prog):
    """Op tuples -> token list (the inverse of decode)."""
    return [REV[op[0]] + (str(op[1]) if len(op) > 1 else "") for op in prog]


def translate(op, y_r, z_r):
    """Frame-local op -> absolute op for the row at raster (y_r, z_r)."""
    if len(op) < 2:
        return op
    s, mu = divmod(op[1], 3)
    x, dy, dz = s % 4, (s // 4) % 4, s // 16
    s2 = x + 4 * ((dy + y_r) % 4) + 16 * ((dz + z_r) % 4)
    return (op[0], 3 * s2 + mu)


def reconstruct(forms):
    """Row forms -> the flat absolute schedule (op tuples)."""
    prog = list(forms["open_row"])
    for r in range(1, 16):
        y_r, z_r = r % 4, r // 4
        form = forms["z_row"] if y_r == 0 else forms["y_row"]
        prog += [translate(op, y_r, z_r) for op in form]
    prog += forms["tail"]
    return prog


def render(forms):
    """Canonical file text: 16 tokens per line, indent=1."""
    out = {}
    for name in FORMS:
        toks = encode(forms[name])
        out[name] = [" ".join(toks[i : i + 16]) for i in range(0, len(toks), 16)]
    return json.dumps(out, indent=1) + "\n"


INF = 1 << 30


# ---------------------------------------------------------------------------
# Morph oracle
# ---------------------------------------------------------------------------
def _subwords(w):
    """All contiguous subwords of w, both orientations, incl. ()."""
    out = {()}
    n = len(w)
    for a in range(n):
        for b in range(a + 1, n + 1):
            sw = w[a:b]
            out.add(sw)
            out.add(inv_word(sw))
    return out


class MorphGraph:
    """Shared word graph over the contiguous subwords (both orientations,
    plus the identity) of a set of endpoint words.

    Edges: pop/push at the right end (COST_MULT), in-place inversion
    (COST_INV), and the 1-Toffoli anchors linking the identity to every
    single-inverse-letter node.  Any path is a legal token sequence, and
    an optimal morph between two endpoint words never pushes a letter
    absent from the target nor re-pushes popped material, so its
    intermediate states are contiguous subwords of the endpoints -- the
    graph contains an optimal path for every endpoint pair it covers.
    """

    def __init__(self, words):
        nodes = set()
        for w in words:
            nodes |= _subwords(w)
        self.node_list = sorted(nodes, key=lambda t: (len(t), t))
        self.idx = {w: i for i, w in enumerate(self.node_list)}
        self.adj = [[] for _ in self.node_list]
        for w, i in self.idx.items():
            if not w:
                continue
            # pop / push pair between w and w[:-1]
            j = self.idx[w[:-1]]
            l, s = w[-1]
            pop_tok = ("i", l) if s > 0 else ("f", l)
            push_tok = ("f", l) if s > 0 else ("i", l)
            self.adj[i].append((j, COST_MULT, pop_tok))
            self.adj[j].append((i, COST_MULT, push_tok))
            # inversion (no reduced word equals its inverse)
            wi = inv_word(w)
            k = self.idx.get(wi)
            if k is not None and wi > w:
                self.adj[i].append((k, COST_INV, ("inv",)))
                self.adj[k].append((i, COST_INV, ("inv",)))
            # anchors
            if len(w) == 1 and w[0][1] == -1:
                e = self.idx[()]
                self.adj[i].append((e, COST_ANCHOR, ("ul", w[0][0])))
                self.adj[e].append((i, COST_ANCHOR, ("li", w[0][0])))

    def node(self, w):
        return self.idx[w]

    def dijkstra(self, src):
        """Distances from src to every node.  Returns (dist, prev) with
        prev[j] = (i, token) for path recovery.
        """
        n = len(self.node_list)
        dist = [INF] * n
        prev = [None] * n
        s = self.node(src)
        dist[s] = 0
        pq = [(0, s)]
        while pq:
            d, i = heapq.heappop(pq)
            if d > dist[i]:
                continue
            for j, c, tok in self.adj[i]:
                nd = d + c
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = (i, tok)
                    heapq.heappush(pq, (nd, j))
        return dist, prev

    def path(self, prev, dst):
        toks = []
        j = self.node(dst)
        while prev[j] is not None:
            i, tok = prev[j]
            toks.append(tok)
            j = i
        toks.reverse()
        return toks


def morph(U, V):
    """Exact cheapest token sequence taking the register word U -> V."""
    g = MorphGraph([U, V])
    dist, prev = g.dijkstra(U)
    d = dist[g.node(V)]
    if d >= INF:
        raise RuntimeError("morph unreachable")
    return d, g.path(prev, V)


# ---------------------------------------------------------------------------
# Form tables
# ---------------------------------------------------------------------------
def site_loops(s):
    """Return the 9 loop ids of site s in local order: 3 plaquettes, 6 rectangles."""
    return [3 * s + k for k in range(3)] + [192 + 6 * s + k for k in range(6)]


def site_forms(s):
    """Return the 96 (word, loop, rot, inv) forms of site s, in a translation-
    consistent local order.
    """
    out = []
    for lid in site_loops(s):
        for w, rot, inv in loop_forms(lid):
            out.append((w, lid, rot, inv))
    assert len(out) == 96
    return out


def cost_table(forms_a, forms_b):
    """Morph-cost matrix between two form lists (shared graph, one Dijkstra
    per source form).
    """
    words_a = [f[0] for f in forms_a]
    words_b = [f[0] for f in forms_b]
    g = MorphGraph(words_a + words_b + [()])
    tgt = [g.node(w) for w in words_b]
    T = np.full((len(forms_a), len(forms_b)), INF, np.int64)
    for i, wa in enumerate(words_a):
        dist, _prev = g.dijkstra(wa)
        T[i] = [dist[j] for j in tgt]
    return T


# ---------------------------------------------------------------------------
# Site tour DP (pinned endpoints)
# ---------------------------------------------------------------------------
def tour_dp(forms, M):
    """DP over (mask of the 9 site loops, last form), batched over all 96
    start forms with numpy.  Returns dist of shape (96, 512, 96):
    dist[c, mask, b] = cheapest partial tour that fired exactly the loops
    in mask, started with a fire at form c, and last fired form b.
    """
    nf = len(forms)
    loops9 = sorted({f[1] for f in forms})
    lidx = {L: k for k, L in enumerate(loops9)}
    fl = np.array([lidx[f[1]] for f in forms])  # form -> local loop index
    dist = np.full((nf, 1 << 9, nf), INF, np.int64)
    for c in range(nf):
        dist[c, 1 << fl[c], c] = 0
    masks = sorted(range(1, 1 << 9), key=lambda m: bin(m).count("1"))
    forms_of_loop = [np.where(fl == k)[0] for k in range(9)]
    for mask in masks:
        D = dist[:, mask, :]
        if not (D < INF).any():
            continue
        # min-plus relax: C[c, f'] = min_b D[c, b] + M[b, f']
        C = (D[:, :, None] + M[None, :, :]).min(axis=1)
        for k in range(9):
            if mask & (1 << k):
                continue
            m2 = mask | (1 << k)
            cols = forms_of_loop[k]
            # NB: advanced indexing returns a copy; assign explicitly.
            dist[:, m2, cols] = np.minimum(dist[:, m2, cols], C[:, cols])
    return dist, fl


def backtrack_tour(dist, M, fl, c, b, pick):
    """Recover the fire order (form indices) of an optimal tour c ->...-> b."""
    full = (1 << 9) - 1
    seq = [b]
    mask = full
    cur = b
    while mask != (1 << fl[c]):
        m_prev = mask & ~(1 << fl[cur])
        target = dist[c, mask, cur]
        prevs = np.where(dist[c, m_prev, :] + M[:, cur] == target)[0]
        assert len(prevs) > 0, "tour backtrack failed"
        cur = pick(prevs)
        mask = m_prev
        seq.append(cur)
    assert cur == c
    seq.reverse()
    return seq


# ---------------------------------------------------------------------------
# Chain DP
# ---------------------------------------------------------------------------
def boundary_class(k):
    """Offset class of the raster boundary between site k and k+1."""
    x = k % 4
    y = (k // 4) % 4
    if x < 3:
        return "x"
    return "z" if y == 3 else "y"


def search(verbose=True):
    """Run the full DP and return the emitted op-tuple schedule."""
    t_all = time.time()
    say = print if verbose else (lambda *_a, **_k: None)

    forms0 = site_forms(0)
    words0 = [f[0] for f in forms0]

    M0 = cost_table(forms0, forms0)
    g0 = MorphGraph(words0 + [()])
    d_id, _p = g0.dijkstra(())
    open0 = np.array([d_id[g0.node(w)] for w in words0], np.int64)
    # Reversal symmetry: a token path reversed (f<->i, li<->ul swapped) is
    # a legal equal-cost morph in the other direction, so close == open.
    close0 = open0.copy()

    JX = cost_table(forms0, site_forms(1))  # +x within a row
    JY = cost_table(site_forms(3), site_forms(4))  # x-wrap +y
    JZ = cost_table(site_forms(15), site_forms(16))  # xy-wrap +z

    dist, fl = tour_dp(forms0, M0)
    TOUR = dist[:, (1 << 9) - 1, :]  # (96, 96)

    # Deterministic argmin tie-break: prefer rectangle-loop forms over
    # plaquette forms, then the lowest local form index.
    def pick(idxs):
        return int(min(idxs, key=lambda i: (0 if forms0[i][1] >= 192 else 1, i)))

    # Chain DP over the 64-site raster chain.
    J = {"x": JX, "y": JY, "z": JZ}
    B = (open0[:, None] + TOUR).min(axis=0)  # site 0: best cost ending at b
    B_hist = [B.copy()]
    for k in range(63):
        Jk = J[boundary_class(k)]
        E = (B[:, None] + Jk).min(axis=0)  # entering site k+1 at form c'
        B = (E[:, None] + TOUR).min(axis=0)
        B_hist.append(B.copy())
    total = int((B + close0).min())
    say(f"chain DP total (mult layer): {total:d} T")

    # Backtrack: per-site (start form, end form) pins, end states first.
    pins = [None] * 64
    cur_b = pick(np.where(B_hist[63] + close0 == total)[0])
    for k in range(63, 0, -1):
        Jk = J[boundary_class(k - 1)]
        target = B_hist[k][cur_b]
        Ecand = B_hist[k - 1][:, None] + Jk  # (b_prev, c_k)
        cks = np.where(Ecand.min(axis=0) + TOUR[:, cur_b] == target)[0]
        assert len(cks) > 0, f"chain backtrack failed at site {k}"
        c_k = pick(cks)
        b_prev = pick(np.where(Ecand[:, c_k] == Ecand[:, c_k].min())[0])
        pins[k] = (c_k, cur_b)
        cur_b = b_prev
    c0 = pick(np.where(open0 + TOUR[:, cur_b] == B_hist[0][cur_b])[0])
    pins[0] = (c0, cur_b)

    # Emit tokens site by site as Dijkstra paths (deterministic: heap
    # order over the sorted node interning).
    prog = []
    prev_word = ()
    for k in range(64):
        fs = site_forms(k)  # translation-consistent local form order
        c, b = pins[k]  # ty: ignore[not-iterable]  # the backtrack fills every pin
        for fidx in backtrack_tour(dist, M0, fl, c, b, pick):
            w_target = fs[fidx][0]
            _c, toks = morph(prev_word, w_target)
            prog.extend(toks)
            lid = fs[fidx][1]
            prog.append(("XP",) if LOOP_KIND[lid] == "P" else ("XR",))
            prev_word = w_target
    _c, toks = morph(prev_word, ())
    prog.extend(toks)

    cs = census(prog)
    say(
        f"emitted schedule census: mults={cs['mults']:d} "
        f"inversions={cs['inversions']:d} anchors={cs['anchors']:d} "
        f"P={cs['P']:d} R={cs['R']:d}  Toffoli={cs['toffoli']:d}"
    )
    assert cs["toffoli"] == total, "emission does not reproduce the DP total"
    say("search + emission: %.1fs" % (time.time() - t_all))
    return prog


# ---------------------------------------------------------------------------
# Row-form extraction and the file
# ---------------------------------------------------------------------------
FIRES_PER_ROW = 36  # 4 sites x 9 loops


def row_split(prog):
    """Cut the flat schedule after every 36th fire: 16 rows + the tail."""
    rows, cur, fires = [], [], 0
    for op in prog:
        cur.append(op)
        if op[0] in ("XP", "XR"):
            fires += 1
            if fires % FIRES_PER_ROW == 0:
                rows.append(cur)
                cur = []
    assert len(rows) == 16, len(rows)
    return rows, cur


def extract(prog):
    """Flat schedule -> row forms.  y_row is row 1 and z_row is row 4,
    both moved back into their own frame; reconstruct() is then asserted
    to rebuild the full schedule, so every other row must be a translate.
    """
    rows, tail = row_split(prog)
    forms = {
        "open_row": rows[0],
        "y_row": [translate(op, 3, 0) for op in rows[1]],
        "z_row": [translate(op, 0, 3) for op in rows[4]],
        "tail": tail,
    }
    assert reconstruct(forms) == prog, "schedule is not row-periodic"
    return forms


def generate():
    """Run the search and return the canonical constants.json text."""
    forms = extract(search(verbose=False))
    for name, n in FORM_LENS.items():
        assert len(forms[name]) == n, (name, len(forms[name]))
    return render(forms)


def main():
    new = generate()
    old = OUT.read_text() if OUT.exists() else None
    OUT.write_text(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
