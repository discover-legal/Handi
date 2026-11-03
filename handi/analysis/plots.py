"""matplotlib style setup only. Charts are designed per hypothesis later.

Palette: the dataviz skill's validated categorical palette (light mode, 8 slots,
fixed order, colorblind-safe per its validator). Assign hues in order; never cycle.
"""
from __future__ import annotations

PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SEQUENTIAL_HUE = "Blues"  # single hue, light -> dark


def style() -> list[str]:
    """Set a clean rcParams set and return the categorical palette (list of hex)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=PALETTE),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "patch.linewidth": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "font.family": "sans-serif",
        "text.color": INK,
        "image.cmap": SEQUENTIAL_HUE,
    })
    return list(PALETTE)
