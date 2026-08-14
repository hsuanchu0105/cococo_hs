"""
How much does simulated annealing shorten the schedule, on top of each router?

    python plot_sa_reduction.py [--csv ../analysis/schedule_lengths.csv]

Reads the tidy table written by `extract_schedule_lengths.py` and plots, for
each base router and each j,

    reduction = (len(base) - len(base+sa)) / len(base)

with `base` in {cs, fg}. One figure per (base, j): seed on x, reduction on y,
one colour per layout, plus a dashed line at each layout's mean.

Reductions go NEGATIVE -- annealing sometimes lengthens the schedule (the whole
hex / j=20 / cs column averages below zero) -- so the y-axis crosses zero and
every panel carries a zero reference line. Nothing is clipped.

Seeds are labels 1..10, not an ordered quantity, so the markers are never
joined by a line: a connecting line would suggest a trend across seeds that
does not exist.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

project_root = Path(__file__).resolve().parent.parent
analysis_root = project_root.parent / "analysis"

DEFAULT_CSV = analysis_root / "schedule_lengths.csv"
DEFAULT_OUTDIR = analysis_root / "images" / "sa_reduction"
DEFAULT_SUMMARY = analysis_root / "summary" / "sa_reduction.csv"

LAYOUTS = ("single", "pair", "triple", "hex")
COMPARISONS = (("cs", "cs+sa"), ("fg", "fg+sa"))
BASE_LABEL = {"cs": "coarse-grained routing", "fg": "fine-grained routing"}

# --------------------------------------------------------------------------- #
# style -- same design system as analyze_overlap.py
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Categorical, validated against #fcfcfb (scripts/validate_palette.js):
# lightness band, chroma floor and contrast all pass, worst adjacent CVD pair
# dE 14.3 (tritan). Keeps the blue/orange opening of the older Office palette
# in scripts/plot_reduction_std.py; its grey (zero chroma) and yellow (too
# light, 1.6:1) could not stay. Marker shape repeats the identity so the
# series survive greyscale printing.
LAYOUT_STYLE = {
    "single": ("#4472C4", "o"),
    "pair": ("#C55A11", "s"),
    "triple": ("#9932CC", "^"),
    "hex": ("#4C8C1E", "D"),
}
# Nudge each layout off the seed tick so same-seed markers do not collide.
LAYOUT_OFFSET = dict(zip(LAYOUTS, (-0.18, -0.06, 0.06, 0.18)))


def _style_axes(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def load_reductions(csv_path, comparisons=COMPARISONS):
    """Long table -> one row per (base, tuned, layout, j, seed) with its reduction.

    `comparisons` is any sequence of (base, tuned) method pairs; both are kept on
    the row, since a base method can be compared against more than one other
    (cs -> cs+sa and cs -> fg both exist).
    """
    raw = pd.read_csv(csv_path)
    wide = raw.pivot_table(
        index=["layout", "j", "max_ov", "seed"], columns="method", values="length"
    ).reset_index()

    missing = [m for pair in comparisons for m in pair if m not in wide.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing method columns {missing}")
    if wide[list(sum(comparisons, ()))].isna().any().any():
        raise SystemExit(f"{csv_path}: some (layout, j, seed) cells have no length")

    frames = []
    for base, tuned in comparisons:
        frame = wide[["layout", "j", "max_ov", "seed"]].copy()
        frame["base"] = base
        frame["tuned"] = tuned
        frame["base_len"] = wide[base]
        frame["tuned_len"] = wide[tuned]
        frame["reduction"] = (wide[base] - wide[tuned]) / wide[base]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(red):
    rows = []
    for (base, tuned, j, layout), group in red.groupby(
        ["base", "tuned", "j", "layout"], sort=False
    ):
        values = group["reduction"].to_numpy()
        rows.append(
            {
                "base": base,
                "tuned": tuned,
                "j": j,
                "max_ov": int(group["max_ov"].iloc[0]),
                "layout": layout,
                "n_seeds": len(values),
                "mean": values.mean(),
                "std": np.std(values, ddof=1),
                "min": values.min(),
                "max": values.max(),
                "n_negative": int((values < 0).sum()),
            }
        )
    order = {name: i for i, name in enumerate(LAYOUTS)}
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["base", "tuned", "j", "layout"],
        key=lambda s: s.map(order) if s.name == "layout" else s,
    ).reset_index(drop=True)


def print_summary(summary):
    header = f"{'layout':<7} {'mean':>9} {'std':>8} {'min':>9} {'max':>9} {'neg':>4}"
    pairs = dict.fromkeys(zip(summary["base"], summary["tuned"]))
    for base, tuned in pairs:
        rows = summary[(summary["base"] == base) & (summary["tuned"] == tuned)]
        for j in sorted(rows["j"].unique()):
            block = rows[rows["j"] == j]
            print(f"\nreduction ({base} -> {tuned}), j={j}, max_ov={block['max_ov'].iloc[0]}")
            print(header)
            for _, row in block.iterrows():
                print(
                    f"{row['layout']:<7} {row['mean']:>8.2%} {row['std']:>7.2%} "
                    f"{row['min']:>8.2%} {row['max']:>8.2%} {row['n_negative']:>4d}"
                )


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #


def plot_panel(red, base, j, ylim, out_path):
    block = red[(red["base"] == base) & (red["j"] == j)]
    max_ov = int(block["max_ov"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)

    for layout in LAYOUTS:
        rows = block[block["layout"] == layout].sort_values("seed")
        if rows.empty:
            continue
        color, marker = LAYOUT_STYLE[layout]
        x = rows["seed"].to_numpy() + LAYOUT_OFFSET[layout]
        y = rows["reduction"].to_numpy()

        ax.axhline(
            y.mean(), color=color, linewidth=1.0, linestyle=(0, (5, 3)), alpha=0.75, zorder=2
        )
        # No line joining the seeds -- seed order carries no meaning.
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=marker,
            markersize=6.5,
            color=color,
            markeredgecolor=SURFACE,  # surface ring keeps overlaps readable
            markeredgewidth=1.0,
            label=layout,
            zorder=3,
        )

    ax.set_xlim(0.4, 10.6)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("seed", color=INK, fontsize=9)
    ax.set_ylabel("schedule-length reduction from SA", color=INK, fontsize=9)
    ax.set_title(
        f"SA reduction over {BASE_LABEL[base]}  (j={j}, max_ov={max_ov})",
        loc="left",
        fontsize=10,
        color=INK,
        pad=8,
    )

    legend = ax.legend(
        frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13)
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
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found -- run extract_schedule_lengths.py first")

    red = load_reductions(args.csv)
    summary = summarize(red)
    print_summary(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    # One y-scale for all four panels, so they can be compared side by side.
    lo, hi = red["reduction"].min(), red["reduction"].max()
    pad = 0.08 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    print()
    for base, _ in COMPARISONS:
        for j in sorted(red.loc[red["base"] == base, "j"].unique()):
            path = plot_panel(red, base, j, ylim, args.outdir / f"sa_reduction_{base}_j{j}.png")
            print(f"wrote {path}")
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
