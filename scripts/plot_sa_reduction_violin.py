"""
Distribution view of the reduction, all conditions on one axis.

    python plot_sa_reduction_violin.py [--mode sa|fg]

Companion to `plot_sa_reduction.py`, which shows the same numbers seed by seed.
Here the seed axis is collapsed into a distribution: one condition group per
column, each holding one violin per layout. Colour means layout throughout, the
same palette as the seed figure, so the figures read as a set.

    --mode sa   what annealing adds on top of each router (the default)
                (cs -> cs+sa) and (fg -> fg+sa), for j=8 and j=20
    --mode fg   what fine-grained routing alone buys over the standard method
                (cs -> fg), for j=8 and j=20

Each cell holds only n=10 seeds, which is thin for a kernel density estimate, so
the violin is never shown alone: the 10 raw points are drawn over it and the
violin is clipped to the observed range rather than allowed to grow tails into
values no seed produced. Read the points; the violin is only there to shape them.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_sa_reduction import (  # noqa: E402
    DEFAULT_CSV,
    GRID,
    INK,
    INK_MUTED,
    LAYOUTS,
    LAYOUT_STYLE,
    SURFACE,
    _style_axes,
    analysis_root,
    load_reductions,
    print_summary,
    summarize,
)

IMAGE_DIR = analysis_root / "images" / "sa_reduction"

ROUTER_LABEL = {"cs": "coarse-grained", "fg": "fine-grained"}
JS = (8, 20)

# Which reduction it is comes from the title, so the axis just names the quantity.
YLABEL = "reduction"

MODES = {
    # conditions are (base, tuned, j), left to right.
    "sa": {
        "conditions": tuple((base, base + "+sa", j) for j in JS for base in ("cs", "fg")),
        "tick": lambda base, tuned, j: f"{ROUTER_LABEL[base]}\nj={j}",
        "split_every": 2,  # rule between the two j halves
        "title": "SA reduction for different conditions",
        "stem": "sa_reduction_violin",
    },
    "fg": {
        "conditions": tuple(("cs", "fg", j) for j in JS),
        "tick": lambda base, tuned, j: f"j={j}",
        "split_every": None,
        "title": "Fine-grained reduction for different conditions",
        "stem": "fg_reduction_violin",
    },
}

SLOT_OFFSET = (-0.27, -0.09, 0.09, 0.27)  # one slot per layout inside a group
VIOLIN_WIDTH = 0.17
JITTER = 0.035


def clip_to_data(body, values):
    """Trim a violin's KDE tails to the range the seeds actually cover.

    With n=10 the estimate runs well past min/max, which would draw reductions
    nobody measured. The body is a single filled path whose y-coordinates are
    the data axis, so clamping those vertices squares the violin off.
    """
    verts = body.get_paths()[0].vertices
    verts[:, 1] = np.clip(verts[:, 1], values.min(), values.max())


def plot_violins(red, spec, out_path):
    conditions = spec["conditions"]
    # Keep the slot pitch fixed so groups look the same whatever the mode.
    fig, ax = plt.subplots(figsize=(2.0 + 1.7 * len(conditions), 4.8))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    rng = np.random.default_rng(0)  # deterministic jitter -- figure is stable
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)

    for group, (base, tuned, j) in enumerate(conditions):
        for slot, layout in enumerate(LAYOUTS):
            rows = red[
                (red["base"] == base)
                & (red["tuned"] == tuned)
                & (red["j"] == j)
                & (red["layout"] == layout)
            ]
            values = rows["reduction"].to_numpy()
            if values.size == 0:
                continue
            color, marker = LAYOUT_STYLE[layout]
            centre = group + SLOT_OFFSET[slot]

            parts = ax.violinplot(
                values,
                positions=[centre],
                widths=VIOLIN_WIDTH,
                showextrema=False,
                showmedians=False,
            )
            for body in parts["bodies"]:
                clip_to_data(body, values)
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.28)
                body.set_linewidth(0.8)
                body.set_zorder(2)

            # Mean, as a solid rule across the slot.
            ax.hlines(
                values.mean(),
                centre - VIOLIN_WIDTH / 2,
                centre + VIOLIN_WIDTH / 2,
                color=color,
                linewidth=1.6,
                zorder=4,
            )
            # The 10 seeds themselves -- the actual evidence.
            ax.plot(
                centre + rng.uniform(-JITTER, JITTER, values.size),
                values,
                linestyle="none",
                marker=marker,
                markersize=4.2,
                color=color,
                markeredgecolor=SURFACE,
                markeredgewidth=0.7,
                alpha=0.95,
                zorder=3,
            )

    if spec["split_every"]:
        for boundary in range(spec["split_every"], len(conditions), spec["split_every"]):
            ax.axvline(boundary - 0.5, color=GRID, linewidth=1.0, zorder=0)

    # Legend carries layout identity; proxies keep it independent of draw order.
    handles = [
        plt.Line2D(
            [], [], linestyle="none", marker=LAYOUT_STYLE[name][1],
            markersize=6.5, color=LAYOUT_STYLE[name][0],
            markeredgecolor=SURFACE, markeredgewidth=1.0, label=name,
        )
        for name in LAYOUTS
    ]

    ax.set_xlim(-0.55, len(conditions) - 0.45)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([spec["tick"](*cond) for cond in conditions])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel(YLABEL, color=INK, fontsize=9)
    ax.set_title(spec["title"], loc="left", fontsize=10, color=INK, pad=24)
    ax.text(
        0.0, 1.012,
        "each point corresponds to one seed",
        transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom",
    )

    legend = ax.legend(
        handles=handles, frameon=False, fontsize=8, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
    )
    for text in legend.get_texts():
        text.set_color(INK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=sorted(MODES), default="sa")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found -- run extract_schedule_lengths.py first")

    spec = MODES[args.mode]
    comparisons = tuple(dict.fromkeys((base, tuned) for base, tuned, _ in spec["conditions"]))

    red = load_reductions(args.csv, comparisons)
    print_summary(summarize(red))

    out_path = args.out or IMAGE_DIR / f"{spec['stem']}.png"
    print(f"\nwrote {plot_violins(red, spec, out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
