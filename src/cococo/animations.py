
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx


Pos = tuple[int, int]


@dataclass
class MoveEvent:
    kind: str                  # "steiner", "idle_teleport", "idle_back"
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
    idle_teleport: dict[Any, Any] | None
    move_events: list[MoveEvent]


@dataclass
class DisplayFrame:
    layer_idx: int
    layout_current: dict[int, Pos]
    layout_after: dict[int, Pos]
    vdp_dict: dict[Any, list[Pos]]
    steiner: dict[Any, Any] | None
    idle_teleport: dict[Any, Any] | None
    move_events: list[MoveEvent]


def _is_pos(x: Any) -> bool:
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and all(isinstance(v, int) for v in x)
    )


def _is_idle_back_key(key: Any) -> bool:
    return (
        isinstance(key, tuple)
        and len(key) == 3
        and key[0] == "idle_back"
        and _is_pos(key[1])
        and _is_pos(key[2])
    )


def _is_idle_teleport_key(key: Any) -> bool:
    return (
        isinstance(key, tuple)
        and len(key) == 3
        and key[0] == "idle"
        and _is_pos(key[1])
        and _is_pos(key[2])
    )


def _is_special_idle_key(key: Any) -> bool:
    return _is_idle_back_key(key) or _is_idle_teleport_key(key)


def _layout_rev(layout: dict[int, Pos]) -> dict[Pos, int]:
    return {pos: qid for qid, pos in layout.items()}


def _orient_path(path: list[Pos], src: Pos, dst: Pos) -> list[Pos]:
    """
    Ensure the path direction is src -> dst.

    For your new idle_back code, path is already danger_qubit -> gap.
    This helper is still useful for robustness.
    """
    path = list(path)

    if path and path[0] == src and path[-1] == dst:
        return path

    if path and path[0] == dst and path[-1] == src:
        return list(reversed(path))

    return [src, dst]


def _extract_steiner_moves(
    entry: dict[str, Any],
    layout_before: dict[int, Pos],
) -> list[MoveEvent]:
    """
    Extract CNOT/T teleportation moves from entry["steiner"] and entry["move_type"].

    This does not include idle teleportation.
    Idle teleportation now lives in entry["idle_teleport"].
    """
    rev = _layout_rev(layout_before)
    steiner = entry.get("steiner") or {}
    move_type = entry.get("move_type") or {}

    moves: list[MoveEvent] = []

    for key, tree_paths in steiner.items():
        if _is_special_idle_key(key):
            continue

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


def _extract_idle_teleport_moves(
    entry: dict[str, Any],
    layout_before: dict[int, Pos],
) -> list[MoveEvent]:
    """
    Extract idle teleportation moves from:

        entry["idle_teleport"] = {
            ("idle", q, terminal): (path_idle, None)
        }

    This is the new idle teleportation coming from annealing.
    """
    rev = _layout_rev(layout_before)
    idle_teleport = entry.get("idle_teleport") or {}

    moves: list[MoveEvent] = []

    for key, value in idle_teleport.items():
        if not _is_idle_teleport_key(key):
            continue

        _, q, terminal = key
        path, _ = value

        src = q
        dst = terminal
        path = _orient_path(list(path), src, dst)

        qid = rev.get(src)

        moves.append(
            MoveEvent(
                kind="idle_teleport",
                logical_id=qid,
                src=src,
                dst=dst,
                path=path,
                label=f"q{qid}: idle teleport {src} → {dst}",
            )
        )

    return moves


def _extract_idle_back_moves(
    entry: dict[str, Any],
    layout_before: dict[int, Pos],
) -> list[MoveEvent]:
    """
    Extract idle move-back from the new vdp_dict tuple keys:

        ("idle_back", danger_qubit, available_gap): path_idle

    This is not in entry["idle_teleport"], but in entry["vdp_dict"].
    """
    rev = _layout_rev(layout_before)
    vdp_dict = entry.get("vdp_dict") or {}

    moves: list[MoveEvent] = []

    for key, path in vdp_dict.items():
        if not _is_idle_back_key(key):
            continue

        _, danger_qubit, available_gap = key

        src = danger_qubit
        dst = available_gap
        path = _orient_path(list(path), src, dst)

        qid = rev.get(src)

        moves.append(
            MoveEvent(
                kind="idle_back",
                logical_id=qid,
                src=src,
                dst=dst,
                path=path,
                label=f"q{qid}: idle back {src} → {dst}",
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
    """
    Convert raw schedule entries into clean LayerFrame objects.

    Supports your new format:

        entry["steiner"]
        entry["idle_teleport"]
        entry["vdp_dict"] with ("idle_back", danger_qubit, gap)
    """
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
            + _extract_idle_teleport_moves(entry, layout_before)
            + _extract_idle_back_moves(entry, layout_before)
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
                idle_teleport=entry.get("idle_teleport"),
                move_events=moves,
            )
        )

        current_layout = layout_after

    return frames


def build_display_frames(
    layer_frames: list[LayerFrame],
) -> list[DisplayFrame]:
    """
    One display frame per schedule layer.

    Unlike the previous version, we do NOT animate idle paths node-by-node.
    Idle teleportation and idle move-back are drawn as whole yellow dashed routes
    in the same layer frame.
    """
    return [
        DisplayFrame(
            layer_idx=lf.layer_idx,
            layout_current=dict(lf.layout_before),
            layout_after=dict(lf.layout_after),
            vdp_dict=lf.vdp_dict,
            steiner=lf.steiner,
            idle_teleport=lf.idle_teleport,
            move_events=lf.move_events,
        )
        for lf in layer_frames
    ]


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

    ax.set_title(
        f"{title_prefix}\nLayer {frame.layer_idx + 1}",
        fontsize=18,
    )

    # ------------------------------------------------------------------
    # 1. Background lattice
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Factories
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Normal VDP paths in red.
    #    Skip idle_back here because idle_back will be drawn in yellow.
    # ------------------------------------------------------------------
    for key, path in frame.vdp_dict.items():
        if _is_idle_back_key(key):
            continue

        _draw_path(
            ax,
            pos,
            list(path),
            color="red",
            linewidth=2.8,
            alpha=0.8,
            linestyle="-",
        )

    # ------------------------------------------------------------------
    # 4. CNOT/T Steiner support in faint red.
    # ------------------------------------------------------------------
    if frame.steiner:
        for key, tree_paths in frame.steiner.items():
            if _is_special_idle_key(key):
                continue

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

            if p2 is not None:
                _draw_path(
                    ax,
                    pos,
                    list(p2),
                    color="red",
                    linewidth=4.0,
                    alpha=0.24,
                    linestyle="--",
                )

    # ------------------------------------------------------------------
    # 5. Idle paths in yellow.
    #
    #    This includes:
    #      - idle_back from vdp_dict
    #      - idle teleport from idle_teleport
    #
    #    Both are drawn as whole routes in one frame.
    # ------------------------------------------------------------------
    idle_moves = [
        mv for mv in frame.move_events
        if mv.kind in {"idle_back", "idle_teleport"}
    ]

    for mv in idle_moves:
        _draw_path(
            ax,
            pos,
            mv.path,
            color="gold",
            linewidth=4.0,
            alpha=0.95,
            linestyle="--",
        )

    # ------------------------------------------------------------------
    # 6. Logical/data qubits as green rings.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 7. Highlight idle source/destination in yellow.
    # ------------------------------------------------------------------
    active_qids: set[int] = set()

    for mv in idle_moves:
        if mv.logical_id is not None:
            active_qids.add(mv.logical_id)

        for special_node in [mv.src, mv.dst]:
            if special_node in g.nodes():
                nx.draw_networkx_nodes(
                    g,
                    pos,
                    ax=ax,
                    nodelist=[special_node],
                    node_size=310,
                    node_color="none",
                    edgecolors="gold",
                    linewidths=3.0,
                )

    # ------------------------------------------------------------------
    # 8. Highlight non-idle teleport source/destination in black.
    # ------------------------------------------------------------------
    steiner_moves = [
        mv for mv in frame.move_events
        if mv.kind == "steiner"
    ]

    for mv in steiner_moves:
        if mv.logical_id is not None:
            active_qids.add(mv.logical_id)

        for special_node in [mv.src, mv.dst]:
            if special_node in g.nodes():
                nx.draw_networkx_nodes(
                    g,
                    pos,
                    ax=ax,
                    nodelist=[special_node],
                    node_size=285,
                    node_color="none",
                    edgecolors="black",
                    linewidths=2.0,
                )

    # ------------------------------------------------------------------
    # 9. Active labels only.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 10. Small text box.
    # ------------------------------------------------------------------
    #if show_move_text:
    #    if frame.move_events:
    #        text = "\n".join(mv.label for mv in frame.move_events)
    ##    else:
     #       text = "No logical-qubit relocation in this layer"

    #    ax.text(
    #        0.01,
    #        0.02,
    #        text,
    #        transform=ax.transAxes,
    #        fontsize=10,
    #        ha="left",
    #        va="bottom",
    #        bbox=dict(
    #            boxstyle="round",
    #            facecolor="white",
    #            edgecolor="lightgray",
    #            alpha=0.88,
    #        ),
    #    )
    if show_move_text:
        if frame.move_events:
            text = "\n".join(mv.label for mv in frame.move_events)
        else:
            text = "No logical-qubit relocation in this layer"

        # Get current graph limits.
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Add extra vertical space below the lattice.
        y_range = y_max - y_min
        extra_bottom_space = 0.22 * y_range
        ax.set_ylim(y_min - extra_bottom_space, y_max)

        # Put the text box in data coordinates, below the graph.
        ax.text(
            x_min,
            y_min - 0.08 * y_range,
            text,
            fontsize=10,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.92,
            ),
            zorder=100,
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
    embed_limit_mb: int = 100,
    save_path: str | Path | None = None,
):
    """
    Clean HTML animation.

    Current color convention:
      - red solid       : normal VDP routing paths
      - faint red       : CNOT/T teleportation support
      - yellow dashed   : idle teleportation and idle move-back
      - green rings     : logical/data qubit positions
      - yellow rings    : idle source/destination
      - black rings     : CNOT/T teleport source/destination

    This version does not animate idle motion node-by-node.
    Each layer is shown in one clean frame.
    """
    mpl.rcParams["animation.embed_limit"] = embed_limit_mb

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

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(anim.to_jshtml(), encoding="utf-8")

    return anim


pos = tuple[int, int]
Gate = tuple[pos, pos]
@dataclass(frozen=True)
class RouteInfo:
    gate: Gate
    path: tuple[pos, ...]
    ancilla_idx: int
    first_part: FirstPart = "control"

    @property
    def ancilla(self) -> pos:
        return self.path[self.ancilla_idx]

    @property
    def control(self) -> tuple[pos, ...]:
        return self.path[: self.ancilla_idx + 1]

    @property
    def target(self) -> tuple[pos, ...]:
        return self.path[self.ancilla_idx :]

    @property
    def subpath1(self) -> tuple[pos, ...]:
        """The part routed first."""
        if self.first_part == "control":
            return self.control
        return self.target

    @property
    def subpath2(self) -> tuple[pos, ...]:
        """The part routed second."""
        if self.first_part == "control":
            return self.target
        return self.control
    

def plot_fine_routes(
    graph: nx.Graph,
    routes_by_layer: dict[int, dict[Gate, RouteInfo]],
    layer_idx: int | None = None,
    *,
    ax=None,
    show: bool = True,
    save_path: str | None = None,
    show_gate_label: bool = False,
    show_route_label: bool = False,
):
    """
    Plot fine-grained routes stored as:

        routes_by_layer[layer_idx][gate] = RouteInfo(...)

    Green solid line:
        route_info.subpath1, the part routed first

    Blue dashed line:
        route_info.subpath2, the part routed second

    Orange dot:
        route_info.ancilla
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # --------------------------------------------------
    # 1. Node positions
    # --------------------------------------------------
    graph_pos = nx.get_node_attributes(graph, "pos")

    if not graph_pos:
        graph_pos = {node: node for node in graph.nodes}

    def xy(node: pos) -> tuple[float, float]:
        p = graph_pos.get(node, node)
        return p[0], p[1]

    # --------------------------------------------------
    # 2. Drawing helpers
    # --------------------------------------------------
    used_labels: set[str] = set()

    def label_once(label: str) -> str | None:
        if label in used_labels:
            return None
        used_labels.add(label)
        return label

    def draw_path(
        path: tuple[pos, ...],
        *,
        color: str,
        label: str,
        linewidth: float = 3.0,
        linestyle: str = "-",
        zorder: int = 3,
    ) -> None:
        if path is None or len(path) == 0:
            return

        for u, v in zip(path[:-1], path[1:]):
            x1, y1 = xy(u)
            x2, y2 = xy(v)

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                solid_capstyle="round",
                zorder=zorder,
                label=label_once(label),
            )

        xs, ys = zip(*(xy(node) for node in path))
        ax.scatter(
            xs,
            ys,
            color=color,
            s=20,
            zorder=zorder + 1,
        )

    def draw_gate(gate: Gate, *, zorder: int = 10) -> None:
        """
        Draw CNOT endpoints.

        gate = (control, target)
        """

        control, target = gate

        cx, cy = xy(control)
        tx, ty = xy(target)

        ax.scatter(
            [cx],
            [cy],
            color="red",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            zorder=zorder,
            label=label_once("control"),
        )
        ax.text(
            cx + 0.12,
            cy + 0.12,
            "C",
            fontsize=10,
            zorder=zorder + 1,
        )

        ax.scatter(
            [tx],
            [ty],
            color="purple",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            zorder=zorder,
            label=label_once("target"),
        )
        ax.text(
            tx + 0.12,
            ty + 0.12,
            "T",
            fontsize=10,
            zorder=zorder + 1,
        )

    # --------------------------------------------------
    # 3. Draw background graph
    # --------------------------------------------------
    nx.draw_networkx_edges(
        graph,
        pos=graph_pos,
        ax=ax,
        edge_color="lightgray",
        width=1.0,
        alpha=0.7,
    )

    nx.draw_networkx_nodes(
        graph,
        pos=graph_pos,
        ax=ax,
        node_size=20,
        node_color="lightgray",
        alpha=0.8,
    )

    # --------------------------------------------------
    # 4. Select routes
    # --------------------------------------------------
    if layer_idx is None:
        selected_routes: list[tuple[int, Gate, RouteInfo]] = [
            (lyr, gate, route_info)
            for lyr, layer_routes in routes_by_layer.items()
            for gate, route_info in layer_routes.items()
        ]
        title = "All fine-grained routes"
    else:
        selected_routes = [
            (layer_idx, gate, route_info)
            for gate, route_info in routes_by_layer.get(layer_idx, {}).items()
        ]
        title = f"Fine-grained routes in layer {layer_idx}"

    # --------------------------------------------------
    # 5. Draw routes
    # --------------------------------------------------
    for lyr, gate, route_info in selected_routes:
        path = route_info.path
        ancilla_node = route_info.ancilla

        # First-routed part
        draw_path(
            route_info.subpath1,
            color="green",
            linestyle="-",
            linewidth=3.0,
            label="subpath1 / routed first",
            zorder=4,
        )

        # Second-routed part
        draw_path(
            route_info.subpath2,
            color="blue",
            linestyle="--",
            linewidth=2.2,
            label="subpath2 / routed second",
            zorder=5,
        )

        # Full path faintly underneath, optional but useful for debugging.
        if show_route_label:
            mid_idx = len(path) // 2
            mx, my = xy(path[mid_idx])
            ax.text(
                mx + 0.08,
                my + 0.08,
                f"L{lyr}",
                fontsize=8,
                zorder=9,
            )

        # Ancilla
        ax.scatter(
            [xy(ancilla_node)[0]],
            [xy(ancilla_node)[1]],
            color="orange",
            s=90,
            edgecolors="black",
            linewidths=0.8,
            zorder=8,
            label=label_once("ancilla"),
        )

        # Gate endpoints
        draw_gate(gate)

        # Optional full gate label
        if show_gate_label:
            x, y = xy(ancilla_node)
            ax.text(
                x + 0.12,
                y + 0.12,
                f"L{lyr}: {gate}\nfirst={route_info.first_part}",
                fontsize=8,
                zorder=9,
            )

    # --------------------------------------------------
    # 6. Final formatting
    # --------------------------------------------------
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax