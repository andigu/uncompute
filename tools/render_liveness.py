#!/usr/bin/env python3
"""Render a register-liveness profile (liveness.json) as a stacked lifetime chart.

One panel.  x = executed Toffoli; y = live qubits, stacked by allocation
age: persistent registers at the bottom, then every temporary wire packed
contiguously in birth order (earliest-allocated lowest).  A wire sinks as
older wires beneath it are freed, so the stack's top edge is *exactly* the
live-count silhouette — no holes, no separate skyline needed.  Each wire is
colored by the phase that allocated it (mirror phases inherit their forward
partner's color), giving the fine-grained decomposition of the headroom.

UNIFIED ACTIVITY SHADING: when liveness.json carries an ``activity``
section (tools/liveness_profile.py always writes one), each lifetime bar — and each
persistent register row — is additionally shaded by state over time using
its interval's ``state_segments``: *active* (operand of a gate; full role
color), *idle* (untouched, may hold data; pale role color), or *|0>*
(op-semantics-forced; near-white).  Flat
journal-style profiles degenerate to a full-width state raster; churn
profiles keep the familiar lifetime chart and show parked temps as pale
segments.  Bars without state data (pooled intervals) render in one
uniform color.

Zero-Toffoli phases get a 1-Toffoli sliver on the x axis (disclosed in the
axis label) so measurement-based teardowns stay visible.  This view
hides slot fragmentation (holes are packed away); the
`intervals` records in liveness.json retain physical slot ids (`q`) if a
fragmentation view is ever wanted.

Usage:
    uv run python tools/render_liveness.py <liveness.json[.xz]> [-o out.pdf]

Reads the optional naming sidecar ``meta.json`` next to the input for
register labels and phase display names.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import click
import matplotlib
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
from _figstyle import apply_paper_style
from matplotlib.patches import Patch

# Named palettes.  Slot order matters: the first len(registers) slots color
# the persistent registers (drawn at alpha 0.5, so they read as pale bands);
# the rest form the role hue pool, assigned by descending spacetime area.
# "default" is the validated categorical order (adjacent pairs pass CVD +
# normal-vision checks); the alternatives are established CVD-safe schemes
# reordered to the same register-pale/role-pop structure.
PALETTES = {
    # default: the validated hue order of "classic" below, gently
    # desaturated + luminance-evened; same slot semantics, easier on the eye
    "default": [
        "#4d87cd",
        "#3c9950",
        "#dd8fae",
        "#e0a437",
        "#35ab85",
        "#dd7350",
        "#5d50a8",
        "#cf5c5b",
    ],
    # saturated primaries (the hue order that "default" softens)
    "classic": [
        "#2a78d6",
        "#008300",
        "#e87ba4",
        "#eda100",
        "#1baf7a",
        "#eb6834",
        "#4a3aa7",
        "#e34948",
    ],
    # Paul Tol "muted" (reordered): low-chroma, print-friendly
    "tol-muted": [
        "#88ccee",
        "#44aa99",
        "#cc6677",
        "#ddcc77",
        "#117733",
        "#882255",
        "#332288",
        "#aa4499",
    ],
    # Okabe-Ito (reordered): the classic CVD-safe eight
    "okabe-ito": [
        "#56b4e9",
        "#009e73",
        "#cc79a7",
        "#e69f00",
        "#0072b2",
        "#d55e00",
        "#000000",
        "#f0e442",
    ],
    # Paul Tol "bright" + wine to fill slot 8
    "tol-bright": [
        "#66ccee",
        "#228833",
        "#ee6677",
        "#ccbb44",
        "#4477aa",
        "#aa3377",
        "#882255",
        "#bbbbbb",
    ],
    # Petroff 8 (CMS/HEP categorical, reordered to register/role structure)
    "petroff": [
        "#7a21dd",
        "#5790fc",
        "#e42536",
        "#f89c20",
        "#964a8b",
        "#9c9ca1",
        "#86c8dd",
        "#656364",
    ],
}
P8 = PALETTES["default"]
OTHER = "#9a9a9a"  # muted roles (scratch): light neutral, recedes
FOLDED = "#4f4f4f"  # real roles beyond the hue pool: dark neutral, distinct
INK = "#333333"

LABEL_MIN_FRAC = 0.025  # phases narrower than this get no x label (hover on web)
MIN_BAR = 0.3  # minimum drawn lifetime, in padded-Toffoli units


def phase_label(name: str, labels: dict) -> str:
    if name.startswith("un-"):
        return phase_label(name[3:], labels) + r"$^{-1}$"
    if name in labels:
        return labels[name]
    if "/" in name:
        st, rest = name.split("/", 1)
        return f"{labels.get(st, st)}/{rest}"
    return name


def canon(ph: str) -> str:
    """Mirror phases allocate the structural twins of their forward
    partner's wires — color them identically.
    """
    return ph.removeprefix("un-")


def coarse(name: str, depth: int) -> str:
    """Coarsen a hierarchical phase name (``stage/sub/...``) to ``depth``
    components, preserving the mirror prefix.  depth <= 0 keeps the full
    name.
    """
    if depth <= 0:
        return name
    mir = name.startswith("un-")
    core = name[3:] if mir else name
    core = "/".join(core.split("/")[:depth])
    return ("un-" + core) if mir else core


def stage(ph: str) -> str:
    """Top-level stage of a phase name, mirror prefix preserved.  Role
    resolution and the transient test run at stage level so meta.json
    rules are independent of how finely the circuit marks sub-phases.
    """
    return coarse(ph, 1)


def coarsen_phases(phases: list[dict], depth: int) -> list[dict]:
    """Merge consecutive phases whose names collapse to the same coarse
    name within the same group; spans concatenate, toffoli/counts sum,
    peaks max.
    """
    out: list[dict] = []
    for p in phases:
        cn = coarse(p["name"], depth)
        if out and out[-1]["name"] == cn and out[-1]["group"] == p["group"]:
            m = out[-1]
            m["span"][1] = p["span"][1]
            m["toffoli"] += p["toffoli"]
            if "span_toffoli" in p:
                m["span_toffoli"] = m.get("span_toffoli", 0) + p["span_toffoli"]
            m["peak"] = max(m["peak"], p["peak"])
            for k, v in p["counts"].items():
                m["counts"][k] = m["counts"].get(k, 0) + v
        else:
            out.append({**p, "name": cn, "span": list(p["span"]), "counts": dict(p["counts"])})
    return out


ZERO_FC = "#f0f0ef"  # provably-|0> shading (near-white, reads as empty)
HOLD_ALPHA = 0.30  # holding-data shading (pale role color)
ACT_ALPHA = 0.92  # active shading (full role color)


class PaddedAxis:
    """The x geometry: Toffoli scale with 1-Toffoli slivers for zero phases.

    Loop-form profiles carry span_toffoli (loop-compressed geometry: one
    representative iteration per repeat) beside toffoli (true mass); the
    axis plots geometry, labels/annotations report mass.
    """

    def __init__(self, phases: list[dict], fine_phases: list[dict]):
        tofs = [p.get("span_toffoli", p["toffoli"]) for p in phases]
        self.T = np.r_[0.0, np.cumsum(tofs)]
        self.E = np.r_[0.0, np.cumsum([max(t, 1.0) for t in tofs])]
        self.x_total = self.E[-1]
        self.n_padded = sum(1 for t in tofs if t < 1)
        # state segments live in FINE phase indices; map them onto the padded
        # axis through the fine cumulative-Toffoli boundaries (precomputed:
        # plaquette-scale profiles have ~200k segments)
        FT = np.r_[0.0, np.cumsum([p.get("span_toffoli", p["toffoli"]) for p in fine_phases])]
        self.seg_pad = [self.enter(float(t)) for t in FT]

    def enter(self, t: float) -> float:
        """Return the padded x where Toffoli position t is entered."""
        T, E = self.T, self.E
        i = int(np.searchsorted(T, t, side="left"))
        if i < len(T) and T[i] == t:
            return E[i]
        return E[i - 1] + (t - T[i - 1]) * (E[i] - E[i - 1]) / (T[i] - T[i - 1])

    def leave(self, t: float) -> float:
        """Return the padded x where Toffoli position t is left (past any zero-phase slivers)."""
        i = int(np.searchsorted(self.T, t, side="right")) - 1
        return self.E[i] if self.T[i] == t else self.enter(t)

    def state_runs(self, iv, x0, xe):
        """Padded-x state runs for one bar; [(xa, xb, state)] covering
        [x0, xe].  State 3 = uniform (no activity data for this bar).
        """
        ss = iv.get("state_segments")
        if not ss:
            return [(x0, xe, 3)]
        runs: list[list] = []
        for a, b, st in ss:
            xa, xb = self.seg_pad[a], self.seg_pad[b]
            if runs and runs[-1][2] == st:
                runs[-1][1] = xb
            elif xb > xa:
                runs.append([xa, xb, st])
        if not runs:
            return [(x0, xe, 3)]
        runs[0][0] = min(runs[0][0], x0)
        runs[-1][1] = max(runs[-1][1], xe)
        return [tuple(r) for r in runs]


def draw_registers(ax, axis: PaddedAxis, ivals: list[dict], regs: list[dict], has_act: bool):
    """Draw the persistent register bands (always the oldest cohort: rows fixed).

    One path for both worlds: rows without state_segments fall through
    state_runs to a single uniform run (state 3 = the classic 0.5 band).
    Returns the register row edges.
    """
    x_total = axis.x_total
    reg_colors = P8[: len(regs)]
    reg_edges = np.r_[0.0, np.cumsum([r["size"] for r in regs])]
    # contiguous rows with identical runs merge into one rect: a classic
    # profile collapses to one band per register (no alpha tiling seams),
    # an activity profile keeps row boundaries only where states differ
    prows = sorted((iv for iv in ivals if iv["died"] is None), key=lambda iv: iv["q"])
    groups: list[list] = []  # [y0, y1, runs, color]
    for iv in prows:
        q = iv["q"]
        if q >= reg_edges[-1]:
            # persistent wire outside the declared register id range: a
            # payload carrier left by freed-id routing (the register's
            # value ends on a reused non-register id). Neutral band — id
            # alone cannot say which register it carries.
            c = "#6f6f6f"
        else:
            c = reg_colors[int(np.searchsorted(reg_edges, q, side="right")) - 1]
        runs = axis.state_runs(iv, 0.0, x_total)
        if groups and groups[-1][1] == q and groups[-1][3] == c and groups[-1][2] == runs:
            groups[-1][1] = q + 1
        else:
            groups.append([q, q + 1, runs, c])
    ppolys: dict = {0: [], 1: [], 2: [], 3: []}
    pcols: dict = {1: [], 2: [], 3: []}
    for y0, y1, runs, c in groups:
        for xa, xb, st in runs:
            if xb <= xa:
                continue
            ppolys[st].append([(xa, y0), (xb, y0), (xb, y1), (xa, y1)])
            if st:
                pcols[st].append(c)
    for st, alpha in ((0, 1.0), (1, HOLD_ALPHA), (2, ACT_ALPHA), (3, 0.5)):
        if ppolys[st]:
            ax.add_collection(
                mc.PolyCollection(
                    ppolys[st],
                    facecolors=ZERO_FC if st == 0 else pcols[st],
                    edgecolors="none",
                    alpha=alpha,
                    zorder=1,
                )
            )
    for j, r in enumerate(regs):
        ax.text(
            x_total * 0.985,
            (reg_edges[j] + reg_edges[j + 1]) / 2,
            r.get("label", r["name"]),
            ha="right",
            va="center",
            color=INK if has_act else "white",
            fontsize=6.2,
            zorder=6,
        )
    return reg_edges


def _head_leaf(sname: str) -> tuple[str, str]:
    core = sname.removeprefix("~")
    head, _, leaf = core.partition("/")
    return ("~" + head) if sname.startswith("~") else head, leaf


def _leaf_key(leaf: str):
    return tuple((0, int(c)) if c.isdigit() else (1, c) for c in leaf.split("/"))


class RoleColors:
    """The ROLE of each temporary wire and the color that codes it.

    Roles resolve from meta.json rules over the measured (born_phase ->
    died_phase, transient?) signature. Wires freed within their birth phase
    default to a muted "scratch" role so the structural words carry the
    color. Hues go to roles by descending spacetime area; roles beyond the
    hue pool fold to one dark neutral.
    """

    def __init__(self, temps: list[dict], meta: dict, labels: dict, pool: list[str]):
        role_rules = meta.get("roles", [])
        # meta roles are display entries keyed by set head ({"set": "~syn02",
        # "name": ..., "muted": ...}); entries WITHOUT a set key are the
        # fallback matcher for set-less intervals (keys phase/died/transient).
        # Roles are read from wire-set names, ramps from their leaf paths.
        self.labels = labels
        self.rules_by_set = {r["set"]: r for r in role_rules if "set" in r}
        self.fallback_rules = [r for r in role_rules if "set" not in r]

        weight: dict[str, float] = {}
        self.muted_roles: set[str] = set()
        self.rule_of_name: dict[str, dict | None] = {}
        shade_vals: dict[str, set] = {}
        for iv in temps:
            name, muted, head, leaf = self.role_of(iv)
            self.rule_of_name.setdefault(
                name, self.rules_by_set.get(head) if head is not None else None
            )
            if muted:
                self.muted_roles.add(name)
                continue
            weight[name] = weight.get(name, 0.0) + max(iv["died"] - iv["born"], 0.25)
            if leaf:
                shade_vals.setdefault(name, set()).add(leaf)
        self.shade_order = {k: sorted(v, key=_leaf_key) for k, v in shade_vals.items()}
        ranked = sorted(weight, key=lambda k: weight[k], reverse=True)
        self.colored = ranked[: len(pool)]
        self.color_of = {ph: pool[i] for i, ph in enumerate(self.colored)}
        self.folded = ranked[len(pool) :]

    def role_of(self, iv) -> tuple[str, bool, str | None, str]:
        """(display name, muted, matched set head | None, leaf path).
        Set memberships win — ownership names before decorative, in
        claim order; fallback rules cover set-less intervals.
        """
        sets_ = iv.get("sets") or []
        for want_deco in (False, True):
            for sname in sets_:
                if sname.startswith("~") != want_deco:
                    continue
                head, leaf = _head_leaf(sname)
                rule = self.rules_by_set.get(head)
                if rule is not None:
                    return (rule["name"], bool(rule.get("muted")), head, leaf)
                return head.lstrip("~"), False, head, leaf
        b = canon(stage(iv["born_phase"]))
        dd = canon(stage(iv["died_phase"]))
        transient = stage(iv["born_phase"]) == stage(iv["died_phase"])
        for r in self.fallback_rules:
            ph = r.get("phase")
            if ph is not None and b not in (ph if isinstance(ph, list) else [ph]):
                continue
            if "died" in r and dd != r["died"]:
                continue
            if "transient" in r and r["transient"] != transient:
                continue
            return r["name"], bool(r.get("muted")), None, ""
        if transient:
            return "scratch", True, None, ""
        return f"alloc in {phase_label(b, self.labels)}", False, None, ""

    def color_for(self, iv):
        name, muted, _head, leaf = self.role_of(iv)
        if muted:
            return OTHER
        if name not in self.color_of:
            return FOLDED
        base = self.color_of[name]
        vs = self.shade_order.get(name)
        if leaf and vs and len(vs) > 1:
            # lightness ramp within the role hue: lowest leaf lightest
            p = vs.index(leaf) / (len(vs) - 1)
            f = 0.55 + 0.45 * p
            r0, g0, b0 = matplotlib.colors.to_rgb(base)
            return (1 - (1 - r0) * f, 1 - (1 - g0) * f, 1 - (1 - b0) * f)
        return base

    def legend_handles(self) -> list[Patch]:
        """Legend patches: hued roles by area, then folded, then muted roles."""

        def legend_label(name: str) -> str:
            rule = self.rule_of_name.get(name)
            if rule and rule.get("shade_note"):
                return f"{name} ({rule['shade_note']})"
            return name

        handles = [
            Patch(fc=self.color_of[ph], alpha=0.88, label=legend_label(ph)) for ph in self.colored
        ]
        for name in self.folded:
            handles.append(Patch(fc=FOLDED, alpha=0.88, label=name))
        for name in sorted(self.muted_roles):
            handles.append(Patch(fc=OTHER, alpha=0.88, label=name))
        return handles


def draw_temporaries(ax, axis: PaddedAxis, temps: list[dict], persistent: int, color_for):
    """Stack the temporaries in cohorts by birth order and draw their bars."""
    evs = []
    for i, iv in enumerate(temps):
        evs.append((iv["born_seq"], 1, i))
        evs.append((iv["died_seq"], 0, i))
    evs.sort()
    live: list[int] = []  # temp indices, oldest first
    segs = {i: [] for i in range(len(temps))}  # (x, y_bottom) breakpoints
    xend: dict[int, float] = {}
    for _seq, kind, i in evs:
        if kind == 1:
            live.append(i)
            segs[i].append((axis.enter(temps[i]["born"]), persistent + len(live) - 1))
        else:
            x = axis.leave(temps[i]["died"])
            p = live.index(i)
            live.pop(p)
            xend[i] = x
            for j in live[p:]:
                segs[j].append((x, segs[j][-1][1] - 1))
    if live:
        raise SystemExit(f"stacking error: {len(live)} temps never freed")

    # each bar drawn piecewise: the stacking breakpoints (a wire sinks as
    # older wires beneath it are freed) crossed with its activity state
    # runs; without activity data a bar is a single uniform run (state 3)
    rects: dict = {st: ([], []) for st in (0, 1, 2, 3)}  # polys, colors
    for i, iv in enumerate(temps):
        bp = segs[i]
        xe = max(xend[i], bp[-1][0], bp[0][0] + MIN_BAR)
        runs = axis.state_runs(iv, bp[0][0], xe)
        c = color_for(iv)
        for k, (x, y) in enumerate(bp):
            x2 = bp[k + 1][0] if k + 1 < len(bp) else xe
            if x2 <= x:
                continue
            for ra, rb, st in runs:
                a, b2 = max(x, ra), min(x2, rb)
                if b2 <= a:
                    continue
                polys, cols = rects[st]
                polys.append([(a, y), (b2, y), (b2, y + 1), (a, y + 1)])
                cols.append(ZERO_FC if st == 0 else c)
    for st, alpha in ((0, 1.0), (1, HOLD_ALPHA), (2, ACT_ALPHA), (3, 0.88)):
        polys, cols = rects[st]
        if polys:
            ax.add_collection(
                mc.PolyCollection(polys, facecolors=cols, edgecolors="none", alpha=alpha, zorder=2)
            )


def draw_phase_marks(ax, E, phases: list[dict], peak: int, persistent: int, boxed: list[str]):
    """Draw phase boundaries, the peak ceiling and optional phase-envelope boxes."""
    for t in E[1:-1]:
        ax.axvline(t, color="white", lw=0.7, zorder=3)
    ax.axhline(peak, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=5)
    # optional phase-envelope boxes (meta "boxed_phases"): outline a phase's
    # temporary envelope from the register top to its measured peak
    from matplotlib.patches import Rectangle

    for i, p in enumerate(phases):
        if p["name"] in boxed:
            ax.add_patch(
                Rectangle(
                    (E[i], persistent),
                    E[i + 1] - E[i],
                    p["peak"] - persistent,
                    fill=False,
                    ec=INK,
                    lw=0.7,
                    zorder=5,
                )
            )


def annotate_loops(ax, axis: PaddedAxis, loops: list[dict]) -> tuple[list, list]:
    """Mark the wide top-level loop regions with their trip counts.

    One representative iteration is shown; trip counts are annotated only
    where a region is wide enough to read (>= 2% of the axis) — micro-loops
    compress silently and are counted in the axis note (loop-form profiles
    only). Returns all top-level loops and the annotated ones.
    """
    all_loops = [lp for lp in loops if lp.get("depth", 0) == 0]
    wide = []
    for lp in all_loops:
        xa = axis.enter(float(lp["span"][0]))
        xb = axis.leave(float(lp["span"][1]))
        if xb - xa >= 0.02 * axis.x_total:
            wide.append((xa, xb, lp["count"]))
    import matplotlib.transforms as mtransforms

    _lb = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for xa, xb, cnt in wide:
        for xv in (xa, xb):
            ax.axvline(xv, color="0.35", lw=0.5, ls=(0, (2, 2)), zorder=6)
        ax.text(
            (xa + xb) / 2,
            0.975,
            f"$\\times{cnt}$",
            transform=_lb,
            ha="center",
            va="top",
            fontsize=6.5,
            color="0.25",
            zorder=6,
        )
    return all_loops, wide


def label_axes(ax, d: dict, axis: PaddedAxis, phases, labels, reg_edges, has_act: bool) -> None:
    """Set the phase ticks, limits, loop marks, axis labels and title."""
    E, x_total = axis.E, axis.x_total
    peak = d["totals"]["peak_qubits"]
    ticks, ticklabels, skipped = [], [], []
    for i, p in enumerate(phases):
        if E[i + 1] - E[i] >= LABEL_MIN_FRAC * x_total:
            ticks.append((E[i] + E[i + 1]) / 2)
            ticklabels.append(phase_label(p["name"], labels))
        else:
            skipped.append(p["name"])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=30, ha="right", fontsize=6)
    ax.tick_params(axis="x", pad=1)
    if skipped:
        print(f"unlabeled narrow phases: {skipped}")
    if not has_act:
        ax.text(
            0.995,
            0.985,
            f"utilization {100 * d['derived']['utilization']:.0f}\\%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=INK,
            zorder=6,
        )
    ax.set_xlim(0, x_total)
    ax.set_ylim(0, peak * 1.09)
    ax.set_yticks(sorted({int(e) for e in reg_edges[1:]} | {peak}))
    all_loops, wide = annotate_loops(ax, axis, d.get("loops", []))
    # Disclose the sliver padding in the axis label (docstring promise): every
    # zero-Toffoli phase occupies a 1-Toffoli sliver on this axis; loop-form
    # profiles disclose the compressed regions too.
    n_padded = axis.n_padded
    _sliver = (
        f"{n_padded} zero-Toffoli phase{'s' if n_padded != 1 else ''} as 1-Toffoli slivers"
        if n_padded
        else ""
    )
    _loopnote = (
        (
            f"{len(all_loops)} repeat region"
            f"{'s' if len(all_loops) != 1 else ''} shown as one "
            r"iteration" + (f", {len(wide)} annotated $\\times n$" if wide else "")
        )
        if all_loops
        else ""
    )
    _notes = "; ".join(s for s in (_loopnote, _sliver) if s)
    ax.set_xlabel("Executed Toffoli" + (f" ({_notes})" if _notes else ""))
    ax.set_ylabel("Live qubits")
    ax.set_title(
        f"\\textbf{{{d['task']}}}: Toffoli {d['totals']['toffoli']} $\\times$ peak {peak} qubits",
        pad=5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")


def draw_legend(fig, ax, handles: list[Patch], d: dict, has_act: bool) -> None:
    """Draw the role legend, plus the activity-state key on shaded profiles."""
    if not has_act:
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            handlelength=1.2,
            borderpad=0.2,
            bbox_to_anchor=(0.005, 0.99),
            ncols=2,
            columnspacing=0.8,
        )
        return
    # full-bleed shaded profiles have no in-axes whitespace: roles go
    # above (registers are labeled in-band), the state key (with the
    # qubit-time fractions and the allocation utilization) below
    fig.legend(
        handles=handles,
        loc="outside upper center",
        frameon=False,
        handlelength=1.2,
        borderpad=0.2,
        ncols=min(len(handles), 4),
        columnspacing=0.9,
        fontsize=6.2,
    )
    af = d["activity"]["derived"]
    state_handles = [
        Patch(fc="#777777", alpha=ACT_ALPHA, label="active (gates this span)"),
        Patch(fc="#777777", alpha=HOLD_ALPHA, label="idle (may hold data)"),
        Patch(fc=ZERO_FC, ec="#bbbbbb", lw=0.4, label="$|0\\rangle$ (empty)"),
    ]
    leg2 = fig.legend(
        handles=state_handles,
        loc="outside lower center",
        frameon=False,
        handlelength=1.2,
        borderpad=0.2,
        ncols=3,
        columnspacing=1.2,
        fontsize=6.2,
        title_fontsize=6.5,
        title=(
            f"qubit-time: active {100 * af['active_fraction']:.0f}\\%"
            f" / idle {100 * af['idle_fraction']:.0f}\\% / "
            f"$|0\\rangle$ {100 * af['zero_fraction']:.0f}\\%"
            f"  (allocation utilization "
            f"{100 * d['derived']['utilization']:.0f}\\%)"
        ),
    )
    leg2.get_title().set_color(INK)


@click.command()
@click.argument("json_path", type=Path)
@click.option("-o", "--out", type=Path, default=None)
@click.option(
    "--depth",
    type=int,
    default=1,
    help="phase-name depth to display: 1 = top-level stages "
    "(default), 0 = full sub-phase granularity, n = "
    "first n name components",
)
@click.option(
    "--palette",
    type=click.Choice(sorted(PALETTES)),
    default="default",
    help="categorical color scheme (see PALETTES)",
)
def main(json_path: Path, out: Path | None, depth: int, palette: str) -> None:
    """Render a register-liveness profile (liveness.json) as a stacked lifetime chart."""
    global P8
    P8 = PALETTES[palette]
    raw = json_path.read_bytes()
    if json_path.suffix == ".xz":
        raw = lzma.decompress(raw)
    d = json.loads(raw)
    meta_p = json_path.parent / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    labels = meta.get("phase_labels", {})
    out = out or json_path.parent / "profile.pdf"

    # one unified view; see the module docstring (UNIFIED ACTIVITY SHADING)
    has_act = "activity" in d
    if has_act and d["activity"].get("states") != ["zero", "idle", "active"]:
        raise SystemExit(f"unknown activity state taxonomy: {d['activity'].get('states')}")
    peak = d["totals"]["peak_qubits"]
    persistent = d["persistent_qubits"]
    regs = d["registers"]
    phases = coarsen_phases(d["phases"], depth)
    ivals = d["intervals"]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.05, 4.15 if has_act else 3.4), constrained_layout=True)
    axis = PaddedAxis(phases, d["phases"])
    reg_edges = draw_registers(ax, axis, ivals, regs, has_act)
    temps = [iv for iv in ivals if iv["died"] is not None]
    roles = RoleColors(temps, meta, labels, pool=P8[len(regs) :])
    draw_temporaries(ax, axis, temps, persistent, roles.color_for)
    draw_phase_marks(ax, axis.E, phases, peak, persistent, meta.get("boxed_phases", []))
    label_axes(ax, d, axis, phases, labels, reg_edges, has_act)
    draw_legend(fig, ax, roles.legend_handles(), d, has_act)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
