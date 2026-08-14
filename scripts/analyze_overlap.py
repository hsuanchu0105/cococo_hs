"""
Analyze the overlap-component structure of saved fine-grained schedules.

    python analyze_overlap.py SCHEDULE.pkl [MORE.pkl ...] --max-overlap 10

One file  -> both figures, the evolution figure covering that run's layers.
Several   -> the distribution figure pools every run; the evolution figure is
             built for the first file only, so the layer axis stays meaningful.

Outputs are named after the first pickle stem (the convention used by
scripts/evaluations_fg/.../redo_plots_table_output.py):

    images/overlap_evolution_<stem>.png     per-layer size distribution + saturation
    images/overlap_distribution_<stem>.png  pooled size histogram + overlap-node spread
    summary/overlap_summary_<stem>.json     the same numbers as data

By default `images/` and `summary/` are siblings of the schedules, i.e. an
analysis directory laid out as

    analysis/schedules/*.pkl
    analysis/images/
    analysis/summary/

works with no flags at all (a `schedules/` parent is looked through). Override
with --out-dir / --summary-dir.

`max_overlap` is NOT stored in the schedule pickle, so it must be supplied; the
summary reports the observed maximum too, so a mismatch is easy to spot.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from cococo import overlap_analysis as oa  # noqa: E402

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Categorical pair, validated (worst adjacent dE 21.4 deutan / 33.9 normal).
# Same roles as elsewhere in the repo: lightseagreen = the fine-grained series.
C_PATHS = "#20B2AA"  # lightseagreen -- share of paths
C_COMPS = "#9932CC"  # darkorchid    -- share of components

# Sequential ramp for magnitude: one hue, light -> dark.
CMAP = LinearSegmentedColormap.from_list("teal_seq", [SURFACE, C_PATHS, "#0b4f4c"])


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


def _title(prefix, text):
    """Panel title, prefixed with the layout in parentheses when known."""
    return f"({prefix}) {text}" if prefix else text


# --------------------------------------------------------------------------- #
# figure 1 -- per-layer distribution and saturation
# --------------------------------------------------------------------------- #


def plot_evolution(components, layers, max_overlap, run, out_path, title_prefix=""):
    """
    Top: heatmap, cell (layer, s) = the share of that layer's paths sitting in a
    component of size s. Bottom: max and mean component size, as a fraction of
    the cap.

    The heatmap shows every component; the strip tracks the largest explicitly.
    Cells are weighted by paths (size x count), not by component count -- a
    size-6 component is one component but six paths -- and then normalized per
    column, so each column is the layer's conditional distribution "given a
    routed path, how big is its component" and layers with different gate counts
    stay comparable.
    """
    components = [record for record in components if record.run == run]
    layers = [record for record in layers if record.run == run]
    if not components:
        return

    n_layers = max(record.layer for record in components) + 1
    y_max = max(max_overlap or 0, max(record.size for record in components))

    counts = np.zeros((y_max, n_layers))
    for record in components:
        counts[record.size - 1, record.layer] += record.size  # paths, not components

    totals = counts.sum(axis=0)
    counts = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    fig, (ax_map, ax_sat) = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)

    mesh = ax_map.pcolormesh(
        np.arange(n_layers + 1),
        np.arange(y_max + 1) + 0.5,
        np.ma.masked_less_equal(counts, 0),
        cmap=CMAP,
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=True,
    )
    _style_axes(ax_map, grid_axis=None)
    ax_map.set_ylabel("cluster size", fontsize=9, color=INK)
    ax_map.set_ylim(0.5, y_max + 0.5)
    ax_map.set_yticks(range(1, y_max + 1))

    if max_overlap:
        ax_map.axhline(
            max_overlap,
            color=C_COMPS,
            linewidth=1.2,
            linestyle="--",
            zorder=5,
        )
        # Outside the axes: on a saturated run the top row is solid dark and
        # the label would be unreadable sitting on top of it.
        ax_map.annotate(
            f"max_overlap = {max_overlap}",
            xy=(0.998, 1.01),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color=C_COMPS,
            annotation_clip=False,
        )

    cbar = fig.colorbar(
        mesh,
        ax=[ax_map, ax_sat],
        pad=0.015,
        fraction=0.035,
        ticks=[0, 0.25, 0.5, 0.75, 1.0],
    )
    # Each column sums to 100%: it is the layer's conditional distribution.
    cbar.set_label("% of the layer's paths", fontsize=8, color=INK)
    cbar.ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=7, length=3, width=0.8)
    cbar.outline.set_visible(False)

    # + 0.5 so the curves sit on the centres of the heatmap columns above
    layer_idx = [record.layer + 0.5 for record in layers]
    cap = max_overlap or max(record.max_size for record in layers)
    # Mark the individual layers while they are still countable; past that the
    # markers merge into a band and the line alone is more legible.
    marker = "o" if len(layers) <= 60 else None
    for values, color, label in (
        ([record.max_size / cap for record in layers], C_COMPS, "largest cluster"),
        ([record.mean_size / cap for record in layers], C_PATHS, "mean cluster"),
    ):
        ax_sat.plot(
            layer_idx,
            values,
            color=color,
            linewidth=1.2,
            label=label,
            marker=marker,
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
        )
    _style_axes(ax_sat)
    ax_sat.set_ylim(0, 1.02)
    ax_sat.set_yticks([0, 0.5, 1.0])
    ax_sat.set_yticklabels(["0", "50%", "100%"])
    ax_sat.set_ylabel("of capacity", fontsize=9, color=INK)
    ax_sat.set_xlabel("routing layer", fontsize=9, color=INK)
    ax_sat.set_xlim(0, n_layers)
    # Below the axis: the curves fill the whole strip on a saturated run, so an
    # in-plot legend would sit on top of them.
    legend = ax_sat.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        ncols=2,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    ax_map.set_title(
        _title(title_prefix, "Overlap clusters per layer"),
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 2 -- pooled distribution
# --------------------------------------------------------------------------- #


def plot_distribution(components, max_overlap, out_path, title_prefix=""):
    """
    Left: pooled component-size histogram, components vs. share of paths.
    Right: how many lattice nodes a component actually shares.

    Pooled over layers (and over runs when several pickles are given), so each
    component of the run is one observation.
    """
    histogram = oa.size_histogram(components)
    if not histogram:
        return

    sizes = list(range(1, max(max_overlap or 0, max(histogram)) + 1))
    share_comps = [100 * histogram.get(s, {}).get("share_components", 0) for s in sizes]
    share_paths = [100 * histogram.get(s, {}).get("share_paths", 0) for s in sizes]

    fig, (ax_hist, ax_ov) = plt.subplots(
        1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.28}
    )
    fig.patch.set_facecolor(SURFACE)

    x = np.arange(len(sizes))
    width = 0.40
    gap = 0.012  # 2px-ish surface gap between adjacent fills
    ax_hist.bar(
        x - width / 2 - gap,
        share_comps,
        width,
        color=C_COMPS,
        label="% of all clusters",
    )
    ax_hist.bar(
        x + width / 2 + gap,
        share_paths,
        width,
        color=C_PATHS,
        label="% of all routed paths",
    )

    # Direct-label only where the two normalizations diverge most: that gap is
    # the point of the panel.
    divergence = np.argmax(np.abs(np.array(share_comps) - np.array(share_paths)))
    for offset, values in (
        (-width / 2 - gap, share_comps),
        (width / 2 + gap, share_paths),
    ):
        # Ink, not the series colour: the bar underneath already carries
        # identity, and the light teal is unreadable as text.
        ax_hist.annotate(
            f"{values[divergence]:.0f}%",
            xy=(x[divergence] + offset, values[divergence]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )

    _style_axes(ax_hist)
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(sizes)
    # room for the direct labels, which sit half a bar-width off centre
    ax_hist.set_xlim(-0.8, len(sizes) - 0.2)
    ax_hist.set_xlabel("cluster size", fontsize=9, color=INK)
    ax_hist.set_ylabel("percentage in a total run (%)", fontsize=9, color=INK)
    ax_hist.set_title(
        _title(title_prefix, "Contribution of the cluster through layers"),
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )
    legend = ax_hist.legend(fontsize=8, frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)

    if max_overlap:
        ax_hist.axvline(
            len(sizes) - 0.5 if max_overlap >= len(sizes) else max_overlap - 0.5,
            color=INK_MUTED,
            linewidth=1.0,
            linestyle="--",
        )
        # Sits low and right, clear of the legend and of the (empty) tail bars.
        ax_hist.annotate(
            f"capacity = {max_overlap}",
            xy=(len(sizes) - 0.7, ax_hist.get_ylim()[1] * 0.42),
            ha="right",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )

    overlaps = [record.n_overlap_nodes for record in components if record.size >= 2]
    if overlaps:
        # Bins widen with the range so the heavy tail does not dissolve into
        # single-count spikes.
        step = max(1, round(max(overlaps) / 35))
        heights, _, _ = ax_ov.hist(
            overlaps,
            bins=np.arange(0.5, max(overlaps) + step + 0.5, step),
            color=C_PATHS,
        )
        # Log only once the tail spans enough orders of magnitude to need it;
        # on a short run it just turns counts of 2 and 5 into 2x10^0 and 5x10^0.
        if heights.max() >= 20:
            ax_ov.set_yscale("log")
        else:
            ax_ov.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    _style_axes(ax_ov)
    ax_ov.set_xlabel("shared graph nodes per cluster", fontsize=9, color=INK)
    ax_ov.set_ylabel("cluster count", fontsize=9, color=INK)
    ax_ov.set_title(
        _title(title_prefix, "Histogram of number of shared nodes"),
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def expand(patterns):
    paths = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            matches = sorted(glob.glob(pattern))
        else:
            matches = [pattern]
        if not matches:
            raise SystemExit(f"no schedule matched {pattern!r}")
        paths.extend(Path(p) for p in matches)
    return paths


def default_dirs(first_pkl):
    """
    `images/` and `summary/` next to the schedules.

    A `schedules/` ancestor is looked through and any folders below it are
    mirrored, so `analysis/schedules/run42/x.pkl` lands in
    `analysis/images/run42` and `analysis/summary/run42`, while a bare
    `analysis/x.pkl` lands in `analysis/images` and `analysis/summary`.
    """
    parent = Path(first_pkl).resolve().parent
    for ancestor in [parent, *parent.parents]:
        if ancestor.name == "schedules":
            root = ancestor.parent
            sub = parent.relative_to(ancestor)
            return root / "images" / sub, root / "summary" / sub
    return parent / "images", parent / "summary"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl", nargs="+", help="schedule .pkl file(s) or glob(s)")
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=10,
        help="the strict_k cap the run used; not recoverable from the pickle "
        "(default: 10, as in scripts/optimize_fg_cluster.py)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where the figures go (default: images/ beside the schedules)",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="where the JSON goes (default: summary/ beside the schedules)",
    )
    parser.add_argument("--tag", default=None, help="override the output name stem")
    parser.add_argument(
        "--title-prefix",
        default=None,
        help="text shown in parentheses before each panel title; defaults to the "
        "layout parsed out of the schedule filename",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=("png", "pdf", "svg"),
        help="figure format (default: png)",
    )
    parser.add_argument(
        "--no-endpoints",
        action="store_true",
        help="ignore path endpoints when computing overlaps",
    )
    args = parser.parse_args()

    paths = expand(args.pkl)
    components, layers = oa.records_from_files(
        paths, include_endpoints=not args.no_endpoints
    )

    summary = oa.summarize(components, layers, args.max_overlap)
    print(oa.format_summary(summary))

    if not components:
        return

    if summary["max_size_observed"] > args.max_overlap:
        print(
            f"\nWARNING: observed a component of {summary['max_size_observed']} paths "
            f"but --max-overlap is {args.max_overlap}; the cap is wrong for this run."
        )

    image_dir, summary_dir = default_dirs(paths[0])
    image_dir = args.out_dir or image_dir
    summary_dir = args.summary_dir or summary_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    stem = args.tag or paths[0].stem
    if len(paths) > 1 and not args.tag:
        stem = f"{stem}_pooled{len(paths)}"

    # e.g. schedule_fine-grained_layout-hex_j-8_seed1047329 -> "hex"
    prefix = args.title_prefix
    if prefix is None:
        match = re.search(r"layout-([A-Za-z0-9]+)", paths[0].stem)
        prefix = match.group(1) if match else ""

    evolution_path = image_dir / f"overlap_evolution_{stem}.{args.format}"
    distribution_path = image_dir / f"overlap_distribution_{stem}.{args.format}"
    summary_path = summary_dir / f"overlap_summary_{stem}.json"

    plot_evolution(
        components, layers, args.max_overlap, paths[0].stem, evolution_path, prefix
    )
    plot_distribution(components, args.max_overlap, distribution_path, prefix)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if len(paths) > 1:
        print(f"\nevolution figure covers the first run only: {paths[0].stem}")

    print(f"\nwrote {evolution_path}\n      {distribution_path}\n      {summary_path}")


if __name__ == "__main__":
    main()
