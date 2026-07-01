from dataclasses import dataclass
from typing import Literal
from collections import defaultdict
import itertools
import networkx as nx
from dataclasses import replace
import random
import numpy as np



pos = tuple[int, int]
Gate = tuple[pos, pos]

FirstPart = Literal["control", "target"]


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
        
class BasicRouter:
    

    def __init__(
        self,
        g: nx.Graph,
        logical_pos: list[pos],
        factory_pos: list[pos],
        valid_path: str,
        t: int,
        metric: str,
        use_dag: bool,
    ):
        """Class for shortest-first routing based compilation.

        Args:
            g (nx.Graph): Macroscopic Routing Graph. Created via mqt.cococo.layouts
            logical_pos (list[pos]): Logical positions on the graph. Also from mqt.cococo.layouts
            factory_pos (list[pos]): Positions of the factories. Also from mqt.cococo.layouts
            valid_path (str): Either "cc" or "sc" for color code and surface code. However, revisit usefulness of "sc".
            t (int): Reset time for the factories
            metric (str): Either "exact" or "crossing", but it is recommended to use "exact"
            use_dag (bool, optional): Determins whether DAG structure from qiskit is used or naive sequential layering. It is recommended to use `True`.
        """
        self.g = g
        self.logical_pos = logical_pos
        self.factory_pos = factory_pos
        if valid_path not in {"cc", "sc"}:
            raise NotImplementedError(
                "Other valid path setups are not implemented yet."
            )
        self.valid_path = valid_path
        self.use_dag = use_dag
        self.t = t

        self.factory_times = {}
        for factory in factory_pos:
            self.factory_times.update({factory: t})

        self.logical_pos_temp = None

        self.metric = metric
        if metric not in {"crossing", "exact"}:
            raise NotImplementedError(
                "Other metrics than crossing and exact not implemented yet."
            )
        self.routes_by_layer: dict[int, dict[Gate, RouteInfo]]= defaultdict(dict)
        self.overlap_graphs: dict[int, nx.Graph] = defaultdict(nx.Graph)
        self.node_to_gates_by_layer: dict[int, dict[pos, set[Gate]]] = defaultdict(
            lambda: defaultdict(set)
        )

    def make_route_info(
        self,
        gate: Gate,
        path: list[pos],
        ancilla_idx: int = 0 ,
        first_part: FirstPart = "control" ,
    ) -> RouteInfo:
        return RouteInfo(
            gate=gate,
            path=tuple(path),
            ancilla_idx=ancilla_idx,
            first_part=first_part,
        )
    
    # helper function
    def route_nodes(
        self,
        route_info: RouteInfo,
        *,
        include_endpoints: bool = True,
    ) -> set[pos]:
        if include_endpoints:
            return set(route_info.path)

        return set(route_info.path[1:-1])

    def candidate_direct_overlaps(
        self,
        layer_idx: int,
        route_info: RouteInfo,
        *,
        include_endpoints: bool = True,
    ) -> dict[Gate, set[pos]]:
        """
        Check which existing routes in this layer directly overlap
        with the candidate route.

        Does not modify self.overlap_graphs.
        """

        node_to_gates = self.node_to_gates_by_layer[layer_idx]
        candidate_nodes = self.route_nodes(
            route_info,
            include_endpoints=include_endpoints,
        )

        overlaps: dict[Gate, set[pos]] = defaultdict(set)

        for node in candidate_nodes:
            for other_gate in node_to_gates.get(node, set()):
                if other_gate == route_info.gate:
                    continue

                overlaps[other_gate].add(node)

        return dict(overlaps)
        
    def get_local_overlap_graph(
        self,
        layer_idx: int,
        route_info: RouteInfo,
        *,
        include_endpoints: bool = True,
    ) -> nx.Graph:
        """
        Build the local overlap graph containing the candidate route.

        This is a virtual graph. It does not modify self.overlap_graphs.
        """

        G = self.overlap_graphs[layer_idx]

        direct_overlaps = self.candidate_direct_overlaps(
            layer_idx,
            route_info,
            include_endpoints=include_endpoints,
        )

        candidate_gate = route_info.gate

        # If no direct overlaps, it is valid anyway
        if not direct_overlaps:
            H = nx.Graph()
            H.add_node(candidate_gate, route_info=route_info, is_candidate=True)
            return H

        # Collect existing connected components touched by this candidate.
        affected_gates: set[Gate] = set()

        for other_gate in direct_overlaps:
            if G.has_node(other_gate):
                affected_gates.update(nx.node_connected_component(G, other_gate))
            else:
                affected_gates.add(other_gate)

        # Copy only affected part.
        H = G.subgraph(affected_gates).copy()

        # Add candidate.
        H.add_node(candidate_gate, route_info=route_info, is_candidate=True)

        # Add candidate overlap edges.
        for other_gate, overlap_nodes in direct_overlaps.items():
            H.add_edge(
                candidate_gate,
                other_gate,
                overlap_nodes=set(overlap_nodes),
                num_overlap=len(overlap_nodes),
                is_candidate_edge=True,
            )

        return H


    def check_validity(
        self,
        new_info: RouteInfo,
        path_ov_info: RouteInfo,
        allowed_overlap: str = "strict2",
    ) -> tuple[bool, RouteInfo | None, RouteInfo | None, str]:
        """
        Check whether a new route candidate is valid against one existing overlapping route.

        Returns:
            valid:
                Whether the pair is valid.

            updated_new_info:
                Updated RouteInfo for the new route, with chosen ancilla_idx and first_part.

            updated_path_ov_info:
                Updated RouteInfo for the existing overlapping route.

            reason:
                Explanation for debugging.
        """
        if allowed_overlap == "strict2":
            new_path = list(new_info.path)
            path_ov = list(path_ov_info.path)

            # --------------------------------------------------
            # 1. Record overlap positions along new_path
            # --------------------------------------------------
            overlap_rec = np.zeros(len(new_path), dtype=int)

            for i, node in enumerate(new_path):
                if node in path_ov:
                    overlap_rec[i] = 1

            overlap_indices = np.nonzero(overlap_rec)[0]

            if len(overlap_indices) == 0:
                # No overlap, so this pair is automatically valid.
                return True, new_info, path_ov_info, "No overlap."

            # --------------------------------------------------
            # 2. Check whether overlap is in one consecutive block
            # --------------------------------------------------
            starts = np.where((overlap_rec == 1) & np.r_[True, overlap_rec[:-1] == 0])[0]

            if len(starts) >= 2:
                return (
                    False,
                    None,
                    None,
                    "Two paths overlap in multiple disconnected places.",
                )

            ov_start = overlap_indices[0] # start of overlap on new found path
            ov_end = overlap_indices[-1] # end of overlap 

            # --------------------------------------------------
            # 3. Find corresponding overlap interval in path_ov
            # --------------------------------------------------
            try:
                ov_st2 = path_ov.index(new_path[ov_start])
                ov_end2 = path_ov.index(new_path[ov_end])
            except ValueError:
                return (
                    False,
                    None,
                    None,
                    "Overlap node was not found in path_ov. This should not happen.",
                )

            if ov_st2 > ov_end2:
                ov_st2, ov_end2 = ov_end2, ov_st2

            # --------------------------------------------------
            # 4. Count available space on both sides
            # --------------------------------------------------
            # left side: before overlap
            d11 = ov_start
            d21 = ov_st2

            # right side: after overlap
            # Important: this excludes the overlap endpoint itself.
            d12 = len(new_path) - 1 - ov_end
            d22 = len(path_ov) - 1 - ov_end2

            # A side is usable if there is at least one internal node
            # where we can place the ancilla.
            new_left_ok = d11 >= 2
            new_right_ok = d12 >= 2
            old_left_ok = d21 >= 2
            old_right_ok = d22 >= 2

            # --------------------------------------------------
            # 5. Choose which side to place the two ancillas
            # --------------------------------------------------
            # same-side case:
            #   new left + old left
            #   new right + old right
            #
            # opposite-side case:
            #   new left + old right
            #   new right + old left

            chosen_case = None

            if new_left_ok and old_left_ok:
                chosen_case = ("left", "left")

            elif new_right_ok and old_right_ok:
                chosen_case = ("right", "right")

            elif new_left_ok and old_right_ok:
                chosen_case = ("left", "right")

            elif new_right_ok and old_left_ok:
                chosen_case = ("right", "left")

            else:
                return (
                    False,
                    new_info,
                    path_ov_info,
                    (
                        "No valid ancilla placement. "
                        f"d11={d11}, d12={d12}, d21={d21}, d22={d22}"
                    ),
                )

            new_side, old_side = chosen_case

            # --------------------------------------------------
            # 6. Randomly choose valid ancilla positions
            # --------------------------------------------------
            if new_side == "left":
                a1 = random.randint(1, ov_start - 1)
                new_first_part: FirstPart = "control"
            else:
                a1 = random.randint(ov_end + 1, len(new_path) - 2)
                new_first_part = "target"

            if old_side == "left":
                a2 = random.randint(1, ov_st2 - 1)
                old_first_part: FirstPart = "control"
            else:
                a2 = random.randint(ov_end2 + 1, len(path_ov) - 2)
                old_first_part = "target"

            # --------------------------------------------------
            # 7. Return updated RouteInfo objects
            # --------------------------------------------------
            updated_new_info = replace(
                new_info,
                ancilla_idx=a1,
                first_part=new_first_part,
            )

            updated_path_ov_info = replace(
                path_ov_info,
                ancilla_idx=a2,
                first_part=old_first_part,
            )

            reason = (
                f"Valid strict2 pair. "
                f"overlap new_path[{ov_start}:{ov_end}], "
                f"path_ov[{ov_st2}:{ov_end2}], "
                f"chosen sides={chosen_case}, "
                f"a1={a1}, a2={a2}."
            )

            return True, updated_new_info, updated_path_ov_info, reason


    def find_fine_grained_vdp(self,
            layer_idx: int,
            layer: list[tuple[pos, pos] | pos],
            logical_pos: None | list[pos],
            factory_times: dict[pos, int],
            overlap_type:str,
        ):
            
        paths_current_layer = [] 
        gates_current_layer = layer.copy()
        for gate in gates_current_layer:
            g_temp = self.g.copy()
            if logical_pos is None:
                nodes_to_remove = [
                    x for x in self.logical_pos if x != gate[0] and x != gate[1]
                ]
            else:
                nodes_to_remove = [
                    x for x in logical_pos if x != gate[0] and x != gate[1]
                ]
            g_temp.remove_nodes_from(nodes_to_remove)
            path = self.valid_path_method()(g_temp, gate[0], gate[1])
            # check overlap with all routed paths in this layer 
            if paths_current_layer:
                route_info = self.make_route_info(gate, path)
                local_og = self.get_local_overlap_graph(layer_idx, route_info)
                
                if overlap_type == "strict2":
                    if local_og.number_of_nodes() >= 3:
                        valid = False 
                    
                    # no overlap with existing paths
                    elif local_og.number_of_nodes() == 1:
                        paths_current_layer.append(path)
                        self.update_overlap_graph(layer_idx, route_info)
                        self.routes_by_layer[layer_idx][gate] = route_info
                    else:
                        # exactly 2 nodes: candidate + one old route
                        old_gates = [gt for gt in local_og.nodes if gt != route_info.gate]
                        old_gate = old_gates[0]
                    
                        path_ov_info = local_og.nodes[old_gate]["route_info"]
                    
                        valid, updated_new, updated_old, reason = self.check_validity(
                            route_info,
                            path_ov_info,
                            allowed_overlap="strict2",
                        )
                        

                    if valid:
                        self.update_overlap_graph(layer_idx, updated_new)
                        paths_current_layer.append(path)
                        # update route info 
                        self.routes_by_layer[layer_idx][gate] = updated_new  
                        self.routes_by_layer[layer_idx][old_gate] = updated_old 
                    else:
                        # find alternative route 
                        nodes_occupied = []
                        for path in paths_current_layer:
                            for node in path:
                                nodes_occupied.append(node)
                        g_temp.remove_nodes_from(nodes_occupied)
                        path = self.valid_path_method()(g_temp, gate[0], gate[1])
                        paths_current_layer.append(path)
                        # update route info 
                        route_info = self.make_route_info(gate, path)
                        self.update_overlap_graph(layer_idx, route_info)
                        self.routes_by_layer[layer_idx][gate] = route_info

            else:
                # first path in this layer 
                paths_current_layer.append(path)
                route_info = self.make_route_info(gate, path)
                self.routes_by_layer[layer_idx][gate] = route_info
                self.update_overlap_graph(layer_idx, route_info)