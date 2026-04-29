from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx


Pos = tuple[int, int]


@dataclass
class MoveEvent:
    kind: str                  # "steiner" or "idle"
    logical_id: int | None
    src: Pos
    dst: Pos
    path: list[Pos]
    label: str


@dataclass
class LayerFrame:
    layer_idx: int
    layout_before: dict[int, Pos]
    layout_after: dict[int, Pos]
    vdp_dict: dict[Any, list[Pos]]
    steiner: dict[Any, Any] | None
    move_events: list[MoveEvent]


@dataclass
class DisplayFrame:
    layer_idx: int
    substep_idx: int
    substep_kind: str          # "base", "idle_progress", "final"
    layout_current: dict[int, Pos]
    vdp_dict: dict[Any, list[Pos]]
    steiner: dict[Any, Any] | None
    active_idle_move: MoveEvent | None
    idle_progress_idx: int | None


def _is_pos(x: Any) -> bool:
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and all(isinstance(v, int) for v in x)
    )


def _layout_rev(layout: dict[int, Pos]) -> dict[Pos, int]:
    return {pos: qid for qid, pos in layout.items()}


def _parse_idle_label(label: str) -> tuple[Pos, Pos]:
    src_txt, dst_txt = label[len("idle_") :].split("_to_", maxsplit=1)
    src = ast.literal_eval(src_txt)
    dst = ast.literal_eval(dst_txt)

    if not (_is_pos(src) and _is_pos(dst)):
        raise ValueError(f"Could not parse idle label: {label}")

    return src, dst


def _extract_steiner_moves(
    entry: dict[str, Any],
    layout_before: dict[int, Pos],
) -> list[MoveEvent]:
    rev = _layout_rev(layout_before)
    steiner = entry.get("steiner") or {}
    move_type = entry.get("move_type") or {}

    moves: list[MoveEvent] = []

    for key, tree_paths in steiner.items():
        mt = move_type.get(key)
        if mt is None:
            continue

        p1, p2 = tree_paths
        support = list(dict.fromkeys(list(p1) + list(p2)))

        if len(key) == 3:
            a, b, terminal = key

            if mt == "control":
                src = a
            elif mt == "target":
                src = b
            else:
                continue

            dst = terminal
            qid = rev.get(src)

            moves.append(
                MoveEvent(
                    kind="steiner",
                    logical_id=qid,
                    src=src,
                    dst=dst,
                    path=support,
                    label=f"q{qid}: {mt} {src} → {dst}",
                )
            )

        elif len(key) == 2:
            a, terminal = key
            src = a
            dst = terminal
            qid = rev.get(src)

            moves.append(
                MoveEvent(
                    kind="steiner",
                    logical_id=qid,
                    src=src,
                    dst=dst,
                    path=support,
                    label=f"q{qid}: single {src} → {dst}",
                )
            )

    return moves


def _extract_idle_moves(
    entry: dict[str, Any],
    layout_before: dict[int, Pos],
) -> list[MoveEvent]:
    rev = _layout_rev(layout_before)
    vdp_dict = entry.get("vdp_dict") or {}

    moves: list[MoveEvent] = []

    for key, path in vdp_dict.items():
        if not (isinstance(key, str) and key.startswith("idle_")):
            continue

        src, dst = _parse_idle_label(key)
        qid = rev.get(src)

        # This path is the Dijkstra path stored in vdp_dict.
        path = list(path)

        # In your code it may be stored as gap -> danger_qubit, but for animation
        # we want physical motion src -> dst.
        if path and path[0] == dst and path[-1] == src:
            path = list(reversed(path))
        elif not path or path[0] != src or path[-1] != dst:
            path = [src, dst]

        moves.append(
            MoveEvent(
                kind="idle",
                logical_id=qid,
                src=src,
                dst=dst,
                path=path,
                label=f"q{qid}: idle {src} → {dst}",
            )
        )

    return moves


def _apply_move(
    layout_before: dict[int, Pos],
    mv: MoveEvent,
) -> dict[int, Pos]:
    layout_after = dict(layout_before)
    if mv.logical_id is not None:
        layout_after[mv.logical_id] = mv.dst
    return layout_after


def _apply_moves(
    layout_before: dict[int, Pos],
    moves: list[MoveEvent],
) -> dict[int, Pos]:
    layout_after = dict(layout_before)
    for mv in moves:
        layout_after = _apply_move(layout_after, mv)
    return layout_after


def build_layer_frames(
    schedule: list[dict[str, Any]],
    initial_layout: dict[int, Pos] | None = None,
) -> list[LayerFrame]:
    frames: list[LayerFrame] = []
    current_layout = dict(initial_layout) if initial_layout is not None else None

    for i, entry in enumerate(schedule):
        if entry.get("layout") is not None:
            layout_before = dict(entry["layout"])
        elif entry.get("layout_before") is not None:
            layout_before = dict(entry["layout_before"])
        elif current_layout is not None:
            layout_before = dict(current_layout)
        else:
            raise ValueError(
                "Cannot infer layout_before. Please pass initial_layout=layout."
            )

        moves = (
            _extract_steiner_moves(entry, layout_before)
            + _extract_idle_moves(entry, layout_before)
        )

        if entry.get("layout_after") is not None:
            layout_after = dict(entry["layout_after"])
        else:
            layout_after = _apply_moves(layout_before, moves)

        frames.append(
            LayerFrame(
                layer_idx=i,
                layout_before=layout_before,
                layout_after=layout_after,
                vdp_dict=entry.get("vdp_dict") or {},
                steiner=entry.get("steiner"),
                move_events=moves,
            )
        )

        current_layout = layout_after

    return frames


def build_display_frames(
    layer_frames: list[LayerFrame],
) -> list[DisplayFrame]:
    """
    Build drawable frames.

    We keep the clean one-layer view, but if there are idle moves we insert
    additional subframes showing the idle qubit explicitly moving along the
    Dijkstra path.
    """
    out: list[DisplayFrame] = []

    for lf in layer_frames:
        # Base frame: show the layer before any explicit idle progression.
        out.append(
            DisplayFrame(
                layer_idx=lf.layer_idx,
                substep_idx=0,
                substep_kind="base",
                layout_current=dict(lf.layout_before),
                vdp_dict=lf.vdp_dict,
                steiner=lf.steiner,
                active_idle_move=None,
                idle_progress_idx=None,
            )
        )

        layout_running = dict(lf.layout_before)
        substep = 1

        # Only animate idle moves progressively.
        idle_moves = [mv for mv in lf.move_events if mv.kind == "idle"]
        non_idle_moves = [mv for mv in lf.move_events if mv.kind != "idle"]

        # First apply non-idle moves logically to the final layout state if you want
        # the later idle move to be based on all earlier routing effects.
        # If you prefer them to stay invisible in the clean viewer, do not animate them.
        for mv in non_idle_moves:
            layout_running = _apply_move(layout_running, mv)

        for mv in idle_moves:
            path = mv.path if mv.path else [mv.src, mv.dst]

            # Show progression node by node.
            for k in range(len(path)):
                temp_layout = dict(layout_running)

                # place the moving idle qubit at the current path node
                if mv.logical_id is not None:
                    temp_layout[mv.logical_id] = path[k]

                out.append(
                    DisplayFrame(
                        layer_idx=lf.layer_idx,
                        substep_idx=substep,
                        substep_kind="idle_progress",
                        layout_current=temp_layout,
                        vdp_dict=lf.vdp_dict,
                        steiner=lf.steiner,
                        active_idle_move=mv,
                        idle_progress_idx=k,
                    )
                )
                substep += 1

            layout_running = _apply_move(layout_running, mv)

        # Final frame.
        out.append(
            DisplayFrame(
                layer_idx=lf.layer_idx,
                substep_idx=substep,
                substep_kind="final",
                layout_current=dict(lf.layout_after),
                vdp_dict=lf.vdp_dict,
                steiner=lf.steiner,
                active_idle_move=None,
                idle_progress_idx=None,
            )
        )

    return out


def _draw_path(
    ax: plt.Axes,
    pos: dict[Pos, tuple[float, float]],
    path: list[Pos],
    *,
    color: str = "red",
    linewidth: float = 3.0,
    alpha: float = 0.8,
    linestyle: str = "-",
):
    for u, v in zip(path[:-1], path[1:]):
        if u not in pos or v not in pos:
            continue

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
        )


def _draw_clean_layer(
    ax: plt.Axes,
    g: nx.Graph,
    frame: DisplayFrame,
    *,
    factories: list[Pos] | set[Pos] | None = None,
    title_prefix: str = "Movable-qubit lattice-surgery routing",
    show_active_labels: bool = True,
    show_move_text: bool = True,
):
    factories = list(factories or [])
    pos = {n: n for n in g.nodes()}

    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    subtitle = f"Layer {frame.layer_idx + 1}"
    if frame.substep_kind == "idle_progress":
        subtitle += " | idle move"
    ax.set_title(f"{title_prefix}\n{subtitle}", fontsize=18)

    # Background lattice.
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        width=0.7,
        alpha=0.25,
        edge_color="lightblue",
    )

    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=105,
        node_color="lightgray",
        edgecolors="none",
        alpha=0.95,
    )

    # Factory patches.
    if factories:
        factory_nodes = [f for f in factories if f in g.nodes()]
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            nodelist=factory_nodes,
            node_size=150,
            node_shape="s",
            node_color="gray",
            edgecolors="black",
            linewidths=1.2,
        )

    # Base VDP paths: red only.
    for key, path in frame.vdp_dict.items():
        if isinstance(key, str) and key.startswith("idle_"):
            # Keep idle Dijkstra path visible, but not too dominant.
            _draw_path(
                ax,
                pos,
                list(path),
                color="red",
                linewidth=2.0,
                alpha=0.35,
                linestyle=":",
            )
        else:
            _draw_path(
                ax,
                pos,
                list(path),
                color="red",
                linewidth=2.8,
                alpha=0.8,
                linestyle="-",
            )

    # Steiner teleportation support: faint red.
    if frame.steiner:
        for _, tree_paths in frame.steiner.items():
            p1, p2 = tree_paths
            _draw_path(
                ax,
                pos,
                list(p1),
                color="red",
                linewidth=4.0,
                alpha=0.24,
                linestyle="-",
            )
            _draw_path(
                ax,
                pos,
                list(p2),
                color="red",
                linewidth=4.0,
                alpha=0.24,
                linestyle="--",
            )

    # Draw all logical/data qubits as green rings.
    data_nodes = [p for p in frame.layout_current.values() if p in g.nodes()]

    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        nodelist=data_nodes,
        node_size=210,
        node_color="white",
        edgecolors="limegreen",
        linewidths=2.2,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        nodelist=data_nodes,
        node_size=85,
        node_color="gray",
        edgecolors="none",
        alpha=0.95,
    )

    active_qids: set[int] = set()
    text = "No logical-qubit relocation in this layer"

    # If an idle move is currently being animated, highlight it in orange.
    if frame.active_idle_move is not None:
        mv = frame.active_idle_move
        active_qids.add(mv.logical_id) if mv.logical_id is not None else None

        path = mv.path if mv.path else [mv.src, mv.dst]
        prog = frame.idle_progress_idx if frame.idle_progress_idx is not None else 0

        # traversed prefix of the idle path in orange
        traversed = path[: prog + 1]
        _draw_path(
            ax,
            pos,
            traversed,
            color="orange",
            linewidth=4.2,
            alpha=0.95,
            linestyle="-",
        )

        # current moving idle qubit position
        current_node = path[prog]
        if current_node in g.nodes():
            # orange ring
            nx.draw_networkx_nodes(
                g,
                pos,
                ax=ax,
                nodelist=[current_node],
                node_size=320,
                node_color="white",
                edgecolors="orange",
                linewidths=3.0,
            )
            # small orange fill
            nx.draw_networkx_nodes(
                g,
                pos,
                ax=ax,
                nodelist=[current_node],
                node_size=120,
                node_color="orange",
                edgecolors="black",
                linewidths=1.0,
            )

        # emphasize source and destination
        for special_node in [mv.src, mv.dst]:
            if special_node in g.nodes():
                nx.draw_networkx_nodes(
                    g,
                    pos,
                    ax=ax,
                    nodelist=[special_node],
                    node_size=285,
                    node_color="none",
                    edgecolors="orange",
                    linewidths=2.0,
                )

        text = mv.label

    # Label only active qubits.
    if show_active_labels and active_qids:
        labels = {
            qpos: f"q{qid}"
            for qid, qpos in frame.layout_current.items()
            if qid in active_qids and qpos in g.nodes()
        }
        nx.draw_networkx_labels(
            g,
            pos,
            labels=labels,
            ax=ax,
            font_size=9,
            font_weight="bold",
        )

    if show_move_text:
        ax.text(
            0.01,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.88,
            ),
        )


def make_clean_routing_html_animation(
    g: nx.Graph,
    schedule: list[dict[str, Any]],
    *,
    initial_layout: dict[int, Pos] | None = None,
    factories: list[Pos] | set[Pos] | None = None,
    figsize: tuple[float, float] = (18, 8),
    interval: int = 900,
    title_prefix: str = "Movable-qubit lattice-surgery routing",
    show_active_labels: bool = True,
    show_move_text: bool = True,
    save_path: str | Path | None = None,
):
    """
    Clean HTML animation:
      - normal paths in red
      - logical/data qubits in green rings
      - idle-moving qubit highlighted in orange
      - idle_move_back shown explicitly along the Dijkstra path

    Usage:
        from IPython.display import HTML
        anim = make_clean_routing_html_animation(...)
        HTML(anim.to_jshtml())
    """
    layer_frames = build_layer_frames(schedule, initial_layout=initial_layout)
    display_frames = build_display_frames(layer_frames)

    fig, ax = plt.subplots(figsize=figsize)

    def update(k: int):
        _draw_clean_layer(
            ax,
            g,
            display_frames[k],
            factories=factories,
            title_prefix=title_prefix,
            show_active_labels=show_active_labels,
            show_move_text=show_move_text,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(display_frames),
        interval=interval,
        repeat=True,
    )

    plt.close(fig)

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix != ".html":
            raise ValueError("save_path should end with .html")
        save_path.write_text(anim.to_jshtml(), encoding="utf-8")

    return anim