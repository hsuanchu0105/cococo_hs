"""
Absolute schedule length across the four methods, as small multiples.

    python plot_schedule_lengths.py [--csv ../analysis/schedule_lengths.csv]

The reduction figures normalise the decrease away; this one shows the lengths
themselves. 320 numbers on one axis would be unreadable, so they are split into
a 2x4 grid -- rows are j, columns are layout -- leaving 40 numbers per panel.

Within a panel the x-axis walks the methods in increasing sophistication,

    cs -> fg -> cs+sa -> fg+sa

and each seed is drawn as one thin line joining *its own* four lengths, with the
mean over the 10 seeds laid on top in bold. The methods are paired -- the same
seed is the same circuit routed four ways -- so a line across them tracks one
circuit and its downward slope is the decrease. (Contrast the seed axis in
plot_sa_reduction.py, where neighbouring points are unrelated runs and joining
them would invent a trend.)

All panels share one y-axis: the layouts are sized to hold the same number of
data qubits, so their lengths are directly comparable and the grid should be
read as one picture, not eight.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_sa_reduction import (  # noqa: E402
    DEFAULT_CSV,
    INK,
    INK_MUTED,
    LAYOUTS,
    LAYOUT_STYLE,
    SURFACE,
    _style_axes,
    analysis_root,
)

DEFAULT_OUT = analysis_root / "images" / "sa_reduction" / "schedule_lengths.png"

METHOD_ORDER = ("cs", "fg", "cs+sa", "fg+sa")
JS = (8, 20)


def load_lengths(csv_path):
    raw = pd.read_csv(csv_path)
    missing = set(METHOD_ORDER) - set(raw["method"])
    if missing:
        raise SystemExit(f"{csv_path}: missing methods {sorted(missing)}")
    return raw


def print_summary(lengths):
    print(f"mean schedule length over {lengths['seed'].nunique()} seeds\n")
    header = f"{'layout':<7} {'j':>3} " + " ".join(f"{m:>7}" for m in METHOD_ORDER) + f" {'cs-fg+sa':>9}"
    print(header)
    for j in JS:
        for layout in LAYOUTS:
            cell = lengths[(lengths["layout"] == layout) & (lengths["j"] == j)]
            means = [cell.loc[cell["method"] == m, "length"].mean() for m in METHOD_ORDER]
            drop = (means[0] - means[-1]) / means[0]
            print(
                f"{layout:<7} {j:>3} " + " ".join(f"{v:>7.1f}" for v in means) + f" {drop:>8.1%}"
            )
        print()


def plot_grid(lengths, out_path):
    fig, axes = plt.subplots(
        len(JS), len(LAYOUTS), figsize=(11.0, 6.0), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(METHOD_ORDER))

    for row, j in enumerate(JS):
        for col, layout in enumerate(LAYOUTS):
            ax = axes[row][col]
            _style_axes(ax)
            color, marker = LAYOUT_STYLE[layout]

            cell = lengths[(lengths["layout"] == layout) & (lengths["j"] == j)]
            wide = cell.pivot(index="seed", columns="method", values="length")[
                list(METHOD_ORDER)
            ]

            # One line per seed: the same circuit routed four ways.
            for _, seed_row in wide.iterrows():
                ax.plot(
                    x, seed_row.to_numpy(), color=color, linewidth=0.8,
                    alpha=0.35, zorder=2,
                )
            means = wide.mean().to_numpy()
            ax.plot(
                x, means, color=color, linewidth=2.2, marker=marker, markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3,
            )
            # Anchor the eye: where the mean starts and where it ends up.
            for idx in (0, len(METHOD_ORDER) - 1):
                ax.annotate(
                    f"{means[idx]:.0f}",
                    (x[idx], means[idx]),
                    textcoords="offset points",
                    xytext=(0, 9 if idx == 0 else -14),
                    ha="center", fontsize=7.5, color=INK, zorder=4,
                )

            if row == 0:
                ax.set_title(layout, loc="center", fontsize=10, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel("schedule length", color=INK, fontsize=9)
            if col == len(LAYOUTS) - 1:
                ax.text(
                    1.06, 0.5, f"j = {j}", transform=ax.transAxes, rotation=270,
                    va="center", ha="left", fontsize=9, color=INK,
                )

    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, fontsize=8)
    axes[0][0].set_xlim(-0.45, len(METHOD_ORDER) - 0.55)

    fig.suptitle(
        "Schedule length by method, layout and j",
        x=0.005, y=0.995, ha="left", va="top", fontsize=11, color=INK,
    )
    fig.text(
        0.005, 0.952,
        "each thin line is one seed carried across the four methods; bold line = mean",
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found -- run extract_schedule_lengths.py first")

    lengths = load_lengths(args.csv)
    print_summary(lengths)
    print(f"wrote {plot_grid(lengths, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
