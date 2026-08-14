"""
Post-processing of the overlap structure of fine-grained routing schedules.

A fine-grained schedule lets CNOT paths share lattice nodes across the two
routing phases. The router caps how many mutually overlapping paths may be
committed together via `max_overlap` (`overlap_type="strict_k"`), and it keeps
that structure in `self.overlap_graphs[layer_idx]`: one node per gate, an edge
between two gates whose paths share at least one lattice node.

That graph is a pure function of the layer's paths, so it can be rebuilt offline
from a saved schedule `.pkl` with no router, lattice or layout state -- which is
what this module does. `build_overlap_graph` mirrors the incremental edge logic
of `TeleportationRouter.commit_route` in `utils_routing.py`, and its connected
components are exactly the components that `get_local_overlap_graph` reports
while routing.

Caveat when interpreting the numbers: a schedule records only the routes that
were *accepted*. A route rejected because its component would have exceeded
`max_overlap` is deferred to a later layer and leaves no trace here, so these
statistics measure realized saturation, never the rejection rate.

`RouteInfo` is duck-typed via its `.path` attribute rather than imported, to
avoid a circular import with `utils_routing` (same reasoning as in
`internal_testing.test_duplicate_nodes_fg`).
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

pos = tuple[int, int]
Gate = tuple[pos, pos]

# A layer's routes, either as {gate: RouteInfo} ("fine_routes") or as
# {gate: [node, ...]} ("vdp_dict").
Routes = Mapping[Gate, Any]


# --------------------------------------------------------------------------- #
# rebuilding the overlap graph
# --------------------------------------------------------------------------- #


def _paths_of(routes: Routes) -> dict[Gate, tuple[pos, ...]]:
    """Normalize `fine_routes` or `vdp_dict` to {gate: path tuple}."""
    paths: dict[Gate, tuple[pos, ...]] = {}
    for gate, value in routes.items():
        path = getattr(value, "path", value)  # RouteInfo.path, else the list
        paths[gate] = tuple(path)
    return paths


def build_overlap_graph(
    routes: Routes,
    *,
    include_endpoints: bool = True,
) -> nx.Graph:
    """
    Rebuild a layer's overlap graph from its routes alone.

    Node per gate (attribute `path`), edge between two gates whose paths share a
    lattice node, carrying `overlap_nodes: set[pos]` and `num_overlap: int`.
    Equivalent to `self.overlap_graphs[layer_idx]` after every route of the
    layer has been committed.
    """
    paths = _paths_of(routes)

    G = nx.Graph()
    node_to_gates: dict[pos, set[Gate]] = defaultdict(set)

    for gate, path in paths.items():
        G.add_node(gate, path=path)

        nodes = set(path) if include_endpoints else set(path[1:-1])

        for node in nodes:
            existing_gates = node_to_gates[node]

            for other_gate in existing_gates:
                if other_gate == gate:
                    continue

                # two paths may overlap on several (consecutive) nodes
                if G.has_edge(gate, other_gate):
                    G[gate][other_gate]["overlap_nodes"].add(node)
                    G[gate][other_gate]["num_overlap"] = len(
                        G[gate][other_gate]["overlap_nodes"]
                    )
                else:
                    G.add_edge(
                        gate,
                        other_gate,
                        overlap_nodes={node},
                        num_overlap=1,
                    )

            existing_gates.add(gate)

    return G


def layer_routes(entry: Any) -> Routes | None:
    """
    The routes of one schedule entry, or None if the layer routed nothing.

    Prefers `fine_routes` (full RouteInfo) and falls back to `vdp_dict`, so
    coarse schedules -- where `fine_routes` is None -- still work and, being
    vertex-disjoint, must yield only size-1 components.
    """
    if not isinstance(entry, Mapping):
        return None

    fine_routes = entry.get("fine_routes")
    if fine_routes:
        return fine_routes

    vdp_dict = entry.get("vdp_dict")
    if vdp_dict:
        return vdp_dict

    return None


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComponentRecord:
    """One connected component of one layer's overlap graph."""

    run: str
    layer: int
    size: int  # paths in the component == nodes of the local graph
    n_edges: int
    n_overlap_nodes: int  # distinct lattice nodes shared by >= 2 of its paths
    total_path_nodes: int  # summed path lengths, for a density feel
    max_edge_overlap: int  # largest pairwise overlap inside the component


@dataclass(frozen=True)
class LayerRecord:
    """Per-layer aggregate of the overlap structure."""

    run: str
    layer: int
    n_paths: int
    n_components: int
    max_size: int
    mean_size: float
    share_paths_coupled: float  # fraction of the layer's paths in components >= 2


def component_records(
    schedule: Sequence[Any],
    *,
    run: str = "",
    include_endpoints: bool = True,
) -> list[ComponentRecord]:
    """Every overlap component of every layer of one schedule."""
    records: list[ComponentRecord] = []

    for layer_idx, entry in enumerate(schedule):
        routes = layer_routes(entry)
        if not routes:
            continue

        G = build_overlap_graph(routes, include_endpoints=include_endpoints)

        for component in nx.connected_components(G):
            H = G.subgraph(component)

            overlap_nodes: set[pos] = set()
            max_edge_overlap = 0
            for _, _, data in H.edges(data=True):
                overlap_nodes |= data["overlap_nodes"]
                max_edge_overlap = max(max_edge_overlap, data["num_overlap"])

            records.append(
                ComponentRecord(
                    run=run,
                    layer=layer_idx,
                    size=H.number_of_nodes(),
                    n_edges=H.number_of_edges(),
                    n_overlap_nodes=len(overlap_nodes),
                    total_path_nodes=sum(
                        len(H.nodes[gate]["path"]) for gate in H.nodes
                    ),
                    max_edge_overlap=max_edge_overlap,
                )
            )

    return records


def layer_records(components: Iterable[ComponentRecord]) -> list[LayerRecord]:
    """Collapse component records to one record per (run, layer)."""
    by_layer: dict[tuple[str, int], list[ComponentRecord]] = defaultdict(list)
    for record in components:
        by_layer[(record.run, record.layer)].append(record)

    records: list[LayerRecord] = []
    for (run, layer), group in sorted(by_layer.items()):
        sizes = [record.size for record in group]
        n_paths = sum(sizes)
        records.append(
            LayerRecord(
                run=run,
                layer=layer,
                n_paths=n_paths,
                n_components=len(sizes),
                max_size=max(sizes),
                mean_size=n_paths / len(sizes),
                share_paths_coupled=sum(s for s in sizes if s >= 2) / n_paths,
            )
        )

    return records


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_schedule(path: str | Path) -> list[Any]:
    """Load a schedule `.pkl` (needs `cococo` importable for RouteInfo)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def records_from_files(
    paths: Sequence[str | Path],
    *,
    include_endpoints: bool = True,
) -> tuple[list[ComponentRecord], list[LayerRecord]]:
    """
    Component and layer records for several schedules, tagged by file stem.

    The `run` tag lets the aggregate distribution pool across seeds while
    keeping per-run curves separable.
    """
    components: list[ComponentRecord] = []

    for path in paths:
        path = Path(path)
        schedule = load_schedule(path)
        components.extend(
            component_records(
                schedule,
                run=path.stem,
                include_endpoints=include_endpoints,
            )
        )

    return components, layer_records(components)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #


def size_histogram(
    components: Iterable[ComponentRecord],
) -> dict[int, dict[str, float]]:
    """
    Component-size histogram under both normalizations.

    Per size: how many *components* had it, and how many *paths* lived in such a
    component (size x count). The two diverge sharply -- a size-8 component is
    one component but eight paths -- and reading only the first badly understates
    how much overlap the router exploits.
    """
    counts = Counter(record.size for record in components)
    n_components = sum(counts.values())
    n_paths = sum(size * count for size, count in counts.items())

    if not n_components:
        return {}

    return {
        size: {
            "n_components": counts[size],
            "share_components": counts[size] / n_components,
            "n_paths": size * counts[size],
            "share_paths": size * counts[size] / n_paths,
        }
        for size in sorted(counts)
    }


def summarize(
    components: Sequence[ComponentRecord],
    layers: Sequence[LayerRecord],
    max_overlap: int | None,
) -> dict[str, Any]:
    """JSON-serializable summary of one or more runs."""
    if not components:
        return {"n_components": 0, "n_paths": 0, "n_layers": 0}

    histogram = size_histogram(components)
    sizes = [record.size for record in components]
    n_paths = sum(sizes)
    max_size_observed = max(sizes)

    summary: dict[str, Any] = {
        "runs": sorted({record.run for record in components}),
        "n_layers": len(layers),
        "n_components": len(components),
        "n_paths": n_paths,
        "max_overlap": max_overlap,
        "max_size_observed": max_size_observed,
        "mean_size": n_paths / len(components),
        "share_paths_coupled": sum(s for s in sizes if s >= 2) / n_paths,
        "size_histogram": {str(k): v for k, v in histogram.items()},
        "overlap_nodes_per_component": {
            "mean": sum(r.n_overlap_nodes for r in components) / len(components),
            "max": max(r.n_overlap_nodes for r in components),
            "histogram": {
                str(k): v
                for k, v in sorted(
                    Counter(
                        r.n_overlap_nodes for r in components if r.size >= 2
                    ).items()
                )
            },
        },
    }

    if max_overlap:
        n_at_cap = sum(1 for record in layers if record.max_size >= max_overlap)
        summary.update(
            {
                "mean_saturation": summary["mean_size"] / max_overlap,
                "max_saturation": max_size_observed / max_overlap,
                "layers_touching_cap": n_at_cap,
                "share_layers_touching_cap": n_at_cap / len(layers) if layers else 0.0,
                "cap_binding": max_size_observed >= max_overlap,
            }
        )

    return summary


def format_summary(summary: Mapping[str, Any]) -> str:
    """
    Human-readable summary table.

    This is also the "table view" that relieves the low contrast of the light
    series color in the figures -- the numbers are always available as text.
    """
    if not summary.get("n_components"):
        return "No overlap components found (no routed layers in the schedule)."

    max_overlap = summary.get("max_overlap")
    lines = [
        f"runs                 : {len(summary['runs'])}",
        f"layers               : {summary['n_layers']}",
        f"components / paths   : {summary['n_components']} / {summary['n_paths']}",
        f"mean component size  : {summary['mean_size']:.2f} paths",
        f"max component size   : {summary['max_size_observed']} paths"
        + (f"  (cap {max_overlap})" if max_overlap else ""),
        f"paths in comps >= 2  : {100 * summary['share_paths_coupled']:.1f}%",
    ]

    if max_overlap:
        lines += [
            f"mean saturation      : {100 * summary['mean_saturation']:.1f}% of cap",
            f"max saturation       : {100 * summary['max_saturation']:.1f}% of cap",
            f"layers touching cap  : {summary['layers_touching_cap']}"
            f" ({100 * summary['share_layers_touching_cap']:.1f}%)"
            + ("" if summary["cap_binding"] else "   -> cap never reached"),
        ]

    lines += ["", f"{'size':>5} {'#comps':>8} {'comp%':>7} {'#paths':>8} {'path%':>7}"]
    for size, row in summary["size_histogram"].items():
        lines.append(
            f"{size:>5} {row['n_components']:>8} {100 * row['share_components']:>6.1f}%"
            f" {row['n_paths']:>8} {100 * row['share_paths']:>6.1f}%"
        )

    return "\n".join(lines)


def records_as_dicts(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Dataclass records -> plain dicts, for JSON dumping."""
    return [asdict(record) for record in records]
