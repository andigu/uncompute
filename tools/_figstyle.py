"""Figure style for the liveness and occupancy renders.

The style is the scienceplots 'science' style plus LaTeX text in the
typography of the accompanying paper (serif, 8pt base). ``text.usetex``
needs a LaTeX installation with the packages in ``LATEX_PREAMBLE``.
Import this module FIRST: it selects the Agg backend before pyplot loads.
Then call ``apply_paper_style()`` (tools/render_liveness.py,
tools/occupancy_profile.py).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

# Shared axes/text ink: a near-black slate, softer than pure black.
INK = "#25313D"

# Without T1 fontenc a figure falls back to OT1 Computer Modern, and
# \textbf{\textsc{...}} silently degrades to the medium weight. The
# scienceplots 'science' style sets this key to amsmath+amssymb alone; those
# are carried forward here because PAPER_RC replaces the value wholesale.
LATEX_PREAMBLE = " ".join(  # noqa: FLY002
    (
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{bold-extra}",
        r"\renewcommand{\ttdefault}{lmtt}",
    )
)

# The paper's typography (LaTeX text, serif, 8pt base).
PAPER_RC = {
    "text.usetex": True,
    "text.latex.preamble": LATEX_PREAMBLE,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.titlesize": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
}


def apply_paper_style(extra_rc: dict | None = None) -> None:
    """Apply the scienceplots 'science' style + PAPER_RC to the process."""
    plt.style.use(["science"])
    rc = dict(PAPER_RC)
    if extra_rc:
        rc.update(extra_rc)
    plt.rcParams.update(rc)
