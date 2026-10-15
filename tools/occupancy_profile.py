#!/usr/bin/env python3
r"""Region-occupancy chart for circuits that declare ownership wire-sets.

Renders the declared sharing schedule directly: at every position of
the LOOP-COMPRESSED executed-Toffoli axis (each ``repeat()`` region
occupies the width of one representative iteration — the same axis as
tools/liveness_profile.py; flat programs degenerate to plain executed
Toffoli), the stacked occupancy of each top-level OWNERSHIP set
(``b.claim``/``b.release`` names without the decorative ``~`` prefix),
plus POOL = live − Σ owned — exact by subtraction and asserted
non-negative at every event. Event positions resolve from Builder
position tokens in closed form, never by materializing the expansion
(loop-form circuits expand to 10^9+ ops). The stack's top edge is the
live count by construction: no per-wire interval reconstruction, no
pooling, no stacking heuristics (contrast liveness_profile.py, the
per-wire lifetime chart). The exact skyline is overlaid as the
envelope and the global-peak sample is spliced in exactly.

Ownership discipline is VALIDATED while building (fail-closed):
claims require live-and-unowned wires, releases require ownership,
and freeing an owned wire is an error. The tool reports the wires still
owned at the end; owned wires outside the declared registers count as
payload carriers. Decorative sets are exempt (overlap
anything; membership ends at free).

The tool imports the circuit module and calls its ``build()`` (raw
build — post passes never touch the allocator; the Toffoli axis is the
raw count, disclosed). ``meta.json`` beside the circuit may provide
``roles`` display entries ({"set": name, "name": label}) reused for
band labels.

Rendering matches tools/render_liveness.py (the register-liveness
look): paper typography from tools/_figstyle, the same figure
geometry, stage names as rotated x ticks, peak ceiling dashes, in-axes
legend.  The styled render needs a LaTeX installation (text.usetex).
``--draft`` renders an unstyled preview without LaTeX, and
``--render-json`` restyles a stored occupancy.json without a rebuild:

    uv run python tools/occupancy_profile.py <circuit.py> --draft \
        [--window LO HI NAME]... [--samples N] [-o OUTDIR]
    uv run python tools/occupancy_profile.py \
        --render-json <folder>/occupancy.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, NoReturn

import click

from uncompute.compact import WKIND_QUBIT, Regs

PALETTE = [
    "#4d87cd",
    "#3c9950",
    "#dd8fae",
    "#e0a437",
    "#35ab85",
    "#dd7350",
    "#5d50a8",
    "#cf5c5b",
    "#88ccee",
    "#999933",
    "#aa4499",
    "#8a6f4e",
]
POOL_C = "#9a9a9a"  # anonymous pool: light neutral, recedes
MUTED_C = "#6f6f6f"  # declared-muted scaffolding roles (meta.json
# "muted": true) — one dark neutral, never a hue;
# hues are reserved for the non-muted roles and
# assigned in first-claim order, no cycling
INK = "#333333"


def fail(msg: str) -> NoReturn:
    raise SystemExit(f"FAIL: {msg}")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("occupancy_target", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validated_build(mod):
    """Run mod.build() under the VANILLA Builder and validate the
    ownership discipline by replaying its native ``wire_log`` — the
    tool is a pure consumer of a finished Builder (no subclass, no
    patching, no import-order sensitivity).
    """
    circ, *_ = mod.build()
    owner = {}
    live = set()
    violations = []

    def viol(m):
        violations.append(m)
        if len(violations) <= 10:
            print(f"VIOLATION: {m}")

    for kind, x, _token, _live_n in circ.wire_log:
        if kind == "alloc":
            live.add(x)
        elif kind == "free":
            r = owner.get(x)
            if r is not None:
                viol(f"free of wire {x} still owned by {r}")
            live.discard(x)
        elif kind == "claim":
            name, wires = x
            if name.startswith("~"):
                continue
            for q in wires:
                if q not in live:
                    viol(f"claim {name}: wire {q} not live")
                prev = owner.get(q)
                if prev is not None:
                    viol(f"claim {name}: wire {q} owned by {prev}")
                owner[q] = name
        elif kind == "release":
            name, wires = x
            if name.startswith("~"):
                continue
            for q in wires:
                if owner.get(q) != name:
                    viol(f"release {name}: wire {q} owned by {owner.get(q)}")
                else:
                    del owner[q]
    from collections import Counter

    final = Counter(owner.values())
    reg_ids = (
        {w for kind, wires in Regs(circ._registers).registers if kind == WKIND_QUBIT for w in wires}
        if circ._registers
        else set(owner)
    )
    stray = [q for q in owner if q not in reg_ids]
    # every owned survivor must be a declared-register wire OR the build
    # ends before registers are declared on the final carriers (the
    # payload moves to them last): accept survivors that are live; the
    # hard rule is the violation list
    if violations:
        fail(f"{len(violations)} ownership violations")
    print(
        f"ownership: consistent; final owned {dict(final)}"
        + (f" ({len(stray)} on non-register ids: payload carriers)" if stray else "")
    )
    return circ


def resolve_events(circ) -> tuple[list[tuple[int, str, Any]], int]:
    """Place every wire_log event on the LOOP-COMPRESSED executed-Toffoli axis.

    One chronological merge of the log against a document-order walk of the
    recording — O(recorded items), never the expansion. Returns the events
    as chronological (t, kind, payload) rows and the axis length.
    """
    from liveness_profile import _event_before, _leaf7, _walk_recorded

    from uncompute.compact import OpKind

    tof_kinds = {int(OpKind.CCX), int(OpKind.CCZ)}
    tables = {t.table_id: t.bits for t in circ._tables}
    resolved: list[tuple[int, str, Any]] = []  # (t, kind, payload) — chronological
    log = circ.wire_log
    li = 0
    T = 0
    stream = _walk_recorded(circ._items, tables)
    pending = next(stream, None)
    while li < len(log) or pending is not None:
        if li < len(log) and (pending is None or _event_before(log[li][2], pending[1])):
            kind, x, _token, live_n = log[li]
            if kind == "mark":
                resolved.append((T, "p", x))
            elif kind in ("alloc", "free"):
                resolved.append((T, "l", live_n))
            elif kind in ("claim", "release"):
                resolved.append((T, "s", (kind, x[0], len(x[1]))))
            li += 1
            continue
        assert pending is not None  # log exhausted, so the walk is not
        if pending[0] == "leaf" and _leaf7(pending[2])[0] in tof_kinds:
            T += 1
        pending = next(stream, None)
    return resolved, T


def ownership_heads(resolved) -> list[str]:
    """Return the top-level ownership set names in first-claim order."""
    heads = []
    for _t, kind, payload in resolved:
        if kind == "s" and payload[0] == "claim" and not payload[1].startswith("~"):
            h = payload[1].split("/", 1)[0]
            if h not in heads:
                heads.append(h)
    return heads


def sample_occupancy(resolved, heads, total: int, W: int, peak_qubits: int) -> dict:
    """Sample the stacked occupancy into W buckets and splice in the exact peak.

    Each bucket keeps its first (edge) sample and its skyline maximum; an
    empty bucket carries the previous sample forward.
    """
    sizes = {h: 0 for h in heads}
    live_n = 0
    stage_marks, cur_stage = [], None
    edge: list[tuple[dict[str, int], int] | None] = [None] * W
    sky = [0] * W
    peak_sample = (0, dict(sizes), 0)

    def bucket(t):
        return min(W - 1, t * W // max(total, 1))

    for t, kind, payload in resolved:
        if kind == "p":
            st = payload.split("/", 1)[0]
            if st != cur_stage:
                stage_marks.append((t, st))
                cur_stage = st
            continue
        if kind == "s":
            op, name, k = payload
            if name.startswith("~"):
                continue
            h = name.split("/", 1)[0]
            sizes[h] += k if op == "claim" else -k
        else:
            live_n = payload
        pool = live_n - sum(sizes.values())
        assert pool >= 0, (t, live_n, sizes)
        if live_n > peak_sample[0]:
            peak_sample = (live_n, dict(sizes), t)
        b = bucket(t)
        sky[b] = max(sky[b], live_n)
        if edge[b] is None:
            edge[b] = (dict(sizes), live_n)
    cur = ({h: 0 for h in heads}, 0)
    filled = []  # an empty bucket carries the previous sample forward
    for b in range(W):
        cur = edge[b] or cur
        filled.append(cur)
        if sky[b] == 0:
            sky[b] = cur[1]
    assert peak_sample[0] == peak_qubits

    xs = [total * b // W for b in range(W)]
    series = {h: [filled[b][0][h] for b in range(W)] for h in heads}
    lives = [filled[b][1] for b in range(W)]
    pools = [lives[b] - sum(series[h][b] for h in heads) for b in range(W)]
    pt = peak_sample[2]
    pi = next((i for i in range(W) if xs[i] >= pt), W)
    xs.insert(pi, pt)
    for h in heads:
        series[h].insert(pi, peak_sample[1][h])
    pools.insert(pi, peak_sample[0] - sum(peak_sample[1].values()))
    lives.insert(pi, peak_sample[0])
    return {
        "x": xs,
        "series": series,
        "pools": pools,
        "lives": lives,
        "sky": sky,
        "stage_marks": stage_marks,
    }


def window_series(resolved, heads, lo: int, hi: int):
    """Return every event in [lo, hi] unbucketed: x, stacked bands + POOL, live."""
    sizes = {h: 0 for h in heads}
    live_n = 0
    wx, ws, wp, wl = [], {h: [] for h in heads}, [], []
    for t, kind, payload in resolved:
        if kind == "p":
            continue
        if kind == "s":
            op, nm, k = payload
            if nm.startswith("~"):
                continue
            h = nm.split("/", 1)[0]
            sizes[h] += k if op == "claim" else -k
        else:
            live_n = payload
        if lo <= t <= hi:
            wx.append(t)
            for h in heads:
                ws[h].append(sizes[h])
            wp.append(live_n - sum(sizes.values()))
            wl.append(live_n)
    return wx, [ws[h] for h in heads] + [wp], wl


@click.command()
@click.argument("circuit", type=Path, required=False)
@click.option(
    "--window", nargs=3, multiple=True, metavar="LO HI NAME", type=click.Tuple([str, str, str])
)
@click.option("--samples", type=int, default=3000)
@click.option("-o", "--outdir", type=Path, default=None)
@click.option(
    "--draft",
    is_flag=True,
    help="skip the paper figure style (preview without LaTeX)",
)
@click.option(
    "--render-json",
    type=Path,
    default=None,
    help="restyle an existing occupancy.json without building",
)
def main(
    circuit: Path | None,
    window: tuple,
    samples: int,
    outdir: Path | None,
    draft: bool,
    render_json: Path | None,
):
    """Region-occupancy chart for circuits that declare ownership wire-sets."""
    if render_json:
        if circuit or window:
            fail("--render-json takes no circuit and no --window (windows need a build)")
        render_json_main(render_json, outdir, draft)
        return
    if circuit is None:
        fail("circuit path required (or --render-json)")
    circuit = circuit.resolve()
    outdir = outdir or circuit.parent
    outdir.mkdir(parents=True, exist_ok=True)
    mod = load_module(circuit)
    if not hasattr(mod, "build"):
        fail(
            "circuit module has no build() — the occupancy tool runs "
            "raw builds (post passes never touch the allocator)"
        )
    meta_p = circuit.parent / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    disp = {r["set"]: r.get("name", r["set"]) for r in meta.get("roles", []) if "set" in r}
    muted_sets = {r["set"] for r in meta.get("roles", []) if "set" in r and r.get("muted")}

    print(f"building {circuit.name} (vanilla Builder; wire_log consumer) ...")
    circ = validated_build(mod)
    resolved, total = resolve_events(circ)
    print(f"build done: peak {circ.peak_qubits}, compressed-axis toffoli {total:,}")

    heads = ownership_heads(resolved)
    if not heads:
        fail("circuit declares no ownership sets — nothing to chart")
    occ = sample_occupancy(resolved, heads, total, samples, circ.peak_qubits)

    stem = outdir / "occupancy"
    (stem.with_suffix(".json")).write_text(
        json.dumps(
            {
                "circuit": circuit.name,
                "toffoli_compressed": total,
                "axis": "loop-compressed: each repeat region occupies one representative iteration",
                "peak": circ.peak_qubits,
                "x": occ["x"],
                "heads": heads,
                "series": {**occ["series"], "POOL": occ["pools"]},
                "live": occ["lives"],
                "sky": occ["sky"],
                "stages": [{"toffoli": t, "stage": st} for t, st in occ["stage_marks"]],
            },
            indent=1,
        )
        + "\n"
    )

    windows_data = []
    for lo_s, hi_s, name in window:
        lo, hi = int(lo_s), int(hi_s)
        windows_data.append((name, lo, hi, *window_series(resolved, heads, lo, hi)))

    render_all(
        draft=draft,
        variant=circuit.parent.name,
        total=total,
        peak=circ.peak_qubits,
        heads=heads,
        disp=disp,
        muted_sets=muted_sets,
        plabels=meta.get("phase_labels", {}),
        stage_marks=occ["stage_marks"],
        xs=occ["x"],
        data=[occ["series"][h] for h in heads] + [occ["pools"]],
        lives=occ["lives"],
        sky_xy=([total * b // samples for b in range(samples)], occ["sky"]),
        stem=stem,
        windows=windows_data,
    )


def render_json_main(render_json: Path, outdir: Path | None, draft: bool):
    """--render-json: restyle an existing occupancy.json (no build)."""
    jp = render_json.resolve()
    d = json.loads(jp.read_text())
    outdir = outdir or jp.parent
    outdir.mkdir(parents=True, exist_ok=True)
    meta_p = jp.parent / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    disp = {r["set"]: r.get("name", r["set"]) for r in meta.get("roles", []) if "set" in r}
    muted_sets = {r["set"] for r in meta.get("roles", []) if "set" in r and r.get("muted")}
    heads = d.get("heads") or [h for h in d["series"] if h != "POOL"]
    total, peak = d["toffoli_compressed"], d["peak"]
    sky = d.get("sky")
    sky_xy = ([total * b // len(sky) for b in range(len(sky))], sky) if sky else None
    render_all(
        draft=draft,
        variant=jp.parent.name,
        total=total,
        peak=peak,
        heads=heads,
        disp=disp,
        muted_sets=muted_sets,
        plabels=meta.get("phase_labels", {}),
        stage_marks=[(s["toffoli"], s["stage"]) for s in d["stages"]],
        xs=d["x"],
        data=[d["series"][h] for h in heads] + [d["series"]["POOL"]],
        lives=d["live"],
        sky_xy=sky_xy,
        stem=outdir / "occupancy",
        windows=[],
    )
    return 0


def render_all(
    *,
    draft,
    variant,
    total,
    peak,
    heads,
    disp,
    muted_sets,
    plabels,
    stage_marks,
    xs,
    data,
    lives,
    sky_xy,
    stem,
    windows,
):
    """Render the chart(s).  Styling matches tools/render_liveness.py
    (the register-liveness look): paper typography via
    tools/_figstyle, the same figure geometry, stage names as
    rotated x ticks, peak ceiling dashes, in-axes legend.  The styled
    render needs LaTeX (usetex); --draft previews without it.
    """
    if draft:
        import matplotlib

        matplotlib.use("Agg")
    else:
        try:
            from _figstyle import apply_paper_style
        except ImportError as e:
            fail(f"paper figure style unavailable ({e}); pass --draft for an unstyled preview")
        apply_paper_style()
    import matplotlib.pyplot as plt

    # Hue = non-muted role (first-claim order, never cycled); declared-
    # muted scaffolding folds to one dark neutral; the pool recedes.
    head_colors = []
    hue = iter(PALETTE)
    for h in heads:
        head_colors.append(MUTED_C if h in muted_sets else next(hue, MUTED_C))

    def esc(s):
        return s if draft else s.replace("_", r"\_")

    def bold(s):
        return esc(s) if draft else f"\\textbf{{{esc(s)}}}"

    times = "x" if draft else "$\\times$"
    label_min_frac = 0.025  # stages narrower than this go unlabeled

    def stage_name(st):
        if st.startswith("un-"):
            return stage_name(st[3:]) + "$^{-1}$"
        return esc(plabels.get(st, st))

    def render(xs_, data_, lives_, title, path, sky_line=None):
        fig, ax = plt.subplots(figsize=(7.05, 3.4), constrained_layout=True)
        ax.stackplot(
            xs_,
            data_,
            colors=head_colors + [POOL_C],
            linewidth=0.4,
            edgecolor="white",
            labels=[disp.get(h, h) for h in heads] + ["pool"],
        )
        if sky_line is not None:
            ax.plot(*sky_line, color=INK, lw=0.7, alpha=0.8, label="exact skyline")
        else:
            ax.plot(xs_, lives_, color=INK, lw=0.6, alpha=0.9, label="live count (exact)")
        ax.axhline(peak, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=5)
        # stage regions as in the liveness render: thin white boundary
        # lines, names as rotated x ticks at region centers
        ticks, ticklabels, skipped = [], [], []
        bounds = [t for t, _st in stage_marks] + [xs_[-1]]
        for i, (t, st) in enumerate(stage_marks):
            if xs_[0] < t < xs_[-1]:
                ax.axvline(t, color="white", lw=0.7, zorder=3)
            lo = max(t, xs_[0])
            hi = min(bounds[i + 1], xs_[-1])
            if hi <= lo:
                continue
            if hi - lo >= label_min_frac * (xs_[-1] - xs_[0]):
                ticks.append((lo + hi) / 2)
                ticklabels.append(stage_name(st))
            else:
                skipped.append(st)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=30, ha="right", fontsize=6)
        ax.tick_params(axis="x", pad=1)
        if skipped:
            print(f"unlabeled narrow stages: {skipped}")
        ax.set_xlim(xs_[0], xs_[-1])
        ax.set_ylim(0, peak * 1.09)
        yt = [int(v) for v in ax.get_yticks() if 0 <= v < peak * 0.97]
        ax.set_yticks(sorted(set(yt) | {peak}))
        ax.set_xlabel("Executed Toffoli (loop-compressed: each repeat region once)")
        ax.set_ylabel("Live qubits")
        ax.set_title(title, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False, which="both")
        ax.legend(
            loc="upper left",
            frameon=False,
            handlelength=1.2,
            borderpad=0.2,
            bbox_to_anchor=(0.005, 0.99),
            ncols=2,
            columnspacing=0.8,
        )
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

    render(
        xs,
        data,
        lives,
        f"{bold(variant)}: ownership occupancy, "
        f"Toffoli {total:,} (compressed) {times} "
        f"peak {peak} qubits",
        stem,
        sky_line=sky_xy,
    )
    print(f"wrote {stem}.png/.pdf")

    for name, lo, hi, wx, wdata, wl in windows:
        wstem = stem.parent / f"occupancy-{name}"
        render(
            wx,
            wdata,
            wl,
            f"{bold(name)} window [{lo:,}..{hi:,}]: event-exact, {len(wx):,} samples",
            wstem,
        )
        print(f"wrote {wstem}.png/.pdf ({len(wx):,} event-exact samples)")


if __name__ == "__main__":
    main()
