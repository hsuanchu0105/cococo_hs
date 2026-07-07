"""Routing Routines for Lattice Surgery Compilation"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import collections
import itertools
import random
import warnings
import pickle
from datetime import datetime
import cococo.internal_testing as tst
import cococo.dag_helper as dag_helper
import copy 

import sys
import logging
from typing import TypedDict

from dataclasses import dataclass
from typing import Literal
from collections import defaultdict
from dataclasses import replace





logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.handlers = [handler]


lock_penalty = 200

pos = tuple[int, int]
Gate = tuple[pos, pos]

FirstPart = Literal["control", "target"]


@dataclass
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
    """
    Basic Routing for CNOT + T gates based on shortest-first VDP solving.
    """


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

    
    @staticmethod
    def path_sc(g: nx.Graph, control: pos, target: pos):
        """
        Find the shortest path with Dijkstra but with constraints for standard sc.

        #TODO Extract this method out of the LookaheadQuilting Class

        This means, that control qubits are only entered horizontally, targets vertically.
        thus horizontal bdrys are Z_L and vertical X_L.

        This method is actually useless because the Lookahead optimization in this class is only valid for color code connectivity.
        """
        g_temp = g.copy()
        # for control, remove vertical edges
        vertical_neighbors = [
            (control[0], control[1] + 1),
            (control[0], control[1] - 1),
        ]
        for neigh in vertical_neighbors:
            if (neigh, control) in g_temp.edges() or (control, neigh) in g_temp.edges():
                g_temp.remove_edge(neigh, control)

        # for target, remove horizontal edges
        horizontal_neighbors = [(target[0] + 1, target[1]), (target[0] - 1, target[1])]
        for neigh in horizontal_neighbors:
            if (neigh, target) in g_temp.edges() or (target, neigh) in g_temp.edges():
                g_temp.remove_edge(neigh, target)
        # run dijkstra on the adapted graph
        path = nx.dijkstra_path(g_temp, control, target)
        return path

    @staticmethod
    def path_cc(g: nx.Graph, control: pos, target: pos):
        """
        Find the shortest path with Dijkstra for a color code architecture.

        There is only constraint: directly neighboring control/target cannot be directly connected.
        Hence, length of path must at least be 3.
        """
        g_temp = g.copy()
        if (control, target) in g_temp.edges():
            g_temp.remove_edge(control, target)
        path = nx.dijkstra_path(g_temp, control, target)
        return path

    def valid_path_method(self):
        """
        Calls the correct version of Dijkstra depending on `self.valid_path`.
        """
        if self.valid_path == "cc":
            return self.path_cc
        elif self.valid_path == "sc":
            for n in self.g.nodes():
                if "pos" not in self.g.nodes[n]:
                    msg = "Node does not have pos attribute."
                    raise RuntimeError(msg)
                if self.g.nodes[n]["pos"] != n:
                    msg = """       
                            Node pos attribute does not match node label. 
                            Make sure you construct the pos of your initial graph like this: 

                            g = nx.grid_2d_graph(m, n)
                            pos = {node: node for node in g.nodes()}
                            for node in g.nodes():
                                g.nodes[node]["pos"] = node
                        """
                    raise RuntimeError(msg)
            return self.path_sc

        else:
            raise NotImplementedError("Other valid paths not implemented yet.")

    @staticmethod
    def split_layer_terminal_pairs(terminal_pairs):
        """
        Split terminal_pairs into layers of disjoint qubit support.

        Only really needed if self.use_dag=False. If true, it is often computed uselessly.
        """
        layers = []
        current_layer: list[
            tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
        ] = []
        used_qubits = set()

        for pair in terminal_pairs:
            if isinstance(pair[0], tuple) and isinstance(pair[1], tuple):
                if pair[0] in used_qubits or pair[1] in used_qubits:
                    layers.append(current_layer)
                    current_layer = [pair]
                    used_qubits = set(pair)
                else:
                    current_layer.append(pair)
                    used_qubits.update(pair)
            elif isinstance(pair[1], int):
                if pair in used_qubits:
                    layers.append(current_layer)
                    current_layer = [pair]
                    used_qubits = {pair}
                else:
                    current_layer.append(pair)
                    used_qubits.update([pair])

        if current_layer:
            layers.append(current_layer)

        return layers
    
    def split_teleport_dct(self, teleport_dct: dict):
        """
        Split a mixed teleport dictionary into:
            - steiner_dct: CNOT/Steiner entries
            - idle_move_dct: idle teleport entries

        Expected key types:
            ("idle", q, terminal) -> (path, None)
            (a, b, terminal)     -> (path1, path2)
        """
        if teleport_dct is None:
            return None, None
        
        steiner_dct = {}
        idle_move_dct = {}

        for key, value in teleport_dct.items():
            is_idle = (
                isinstance(key, tuple)
                and len(key) == 3
                and key[0] == "idle"
            )

            if is_idle:
                idle_move_dct[key] = value
            else:
                steiner_dct[key] = value

        return steiner_dct, idle_move_dct
    
    def find_max_vdp_set(
        self,
        layer: list[tuple[pos, pos] | pos],
        logical_pos: None | list[pos],
        factory_times: dict[pos, int],
    ):
        """
        Finds the approximation for a vdp set for a given layer and returns the routing as well as a potential remainder.

        If logical_pos is given as input, you use the input instead of the self.logical_pos
        """
        factory_times_temp = factory_times.copy()
        vdp_dict: dict[
            tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
            list[tuple[int, int]],
        ] = {}
        terminal_pairs_remainder = []
        successful_terminals = []  # gather successful terminal pairs
        flag_problem = False
        g_temp = self.g.copy()
        dct_qubits = (
            {}
        )  # a dct which checks whether a qubit was already used in the layer
        terminal_pairs_current = layer.copy()
        flattened_terminals = [
            pair
            for item in layer.copy()
            for pair in (item if isinstance(item[0], tuple) else [item])
        ]
        for t in flattened_terminals:
            dct_qubits.update({t: False})
        dct_qubits_copy = dct_qubits.copy()
        flattened_terminals_and_factories = (
            flattened_terminals.copy() + self.factory_pos.copy()
        )  # was using self.flattened before, which was problem because not updated after logical repositioning

        while (
            len(terminal_pairs_current) > 0 and flag_problem is False
        ):  # noqa: PLR1702
            paths_temp_lst = (
                []
            )  # gather all possible paths here, between all terminal pairs (cnots) and between all qubits for a tgate with all factories
            tp_list: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = (
                []
            )  # same order, actually redundant but error otherwise
            for t_p in terminal_pairs_current:

                # cnot
                if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):

                    g_temp_temp = g_temp.copy()
                    # ADAPT THE GRAPH AND REMOVE ALL LOGICAL DATA PATCHES DESPITE THE t_p
                    if logical_pos is None:
                        nodes_to_remove = [
                            x for x in self.logical_pos if x != t_p[0] and x != t_p[1]
                        ]
                    else:
                        nodes_to_remove = [
                            x for x in logical_pos if x != t_p[0] and x != t_p[1]
                        ]
                    g_temp_temp.remove_nodes_from(nodes_to_remove)

                    if dct_qubits[t_p[0]] or dct_qubits[t_p[1]]:
                        flag_problem = True
                        break
                    terminals_temp = [
                        pair
                        for pair in flattened_terminals_and_factories.copy()
                        if pair != t_p[0] and pair != t_p[1]
                    ]
                    terminals_temp = list(set(terminals_temp))
                    g_temp_temp.remove_nodes_from(terminals_temp)
                    # find shortest path of t_p
                    try:
                        path = self.valid_path_method()(g_temp_temp, t_p[0], t_p[1]) #Dijkstra 
                        paths_temp_lst.append(path)
                        tp_list.append(t_p)
                    except nx.NetworkXNoPath:
                        # skip the t_p if no path exists
                        pass  # therefore just pass

                # t gate
                elif isinstance(t_p[1], int):
                    g_temp_temp = g_temp.copy()
                    # ADAPT THE GRAPH AND REMOVE ALL LOGICAL DATA PATCHES DESPITE THE t_p position where the t gate is applied
                    if logical_pos is None:
                        nodes_to_remove = [x for x in self.logical_pos if x != t_p]
                    else:
                        nodes_to_remove = [x for x in logical_pos if x != t_p]
                    g_temp_temp.remove_nodes_from(nodes_to_remove)

                    if dct_qubits[t_p]:
                        flag_problem = True
                        break
                    dist_factories = {}
                    for factory in self.factory_pos:
                        g_temp_temp2 = g_temp_temp.copy()
                        if (
                            factory_times_temp[factory] == 0
                        ):  # only include available factories
                            # remove other terminals
                            terminals_temp = [
                                pair
                                for pair in flattened_terminals_and_factories.copy()
                                if pair not in {t_p, factory}
                            ]
                            terminals_temp = list(set(terminals_temp))
                            g_temp_temp2.remove_nodes_from(terminals_temp)
                            try:
                                path = self.valid_path_method()(
                                    g_temp_temp2, t_p, factory
                                )
                            except nx.NetworkXNoPath:
                                continue
                            dist_factories.update({factory: path})
                    # choose shortest available path or if no elements in dist_factories, flag_problem = True
                    if len(dist_factories) == 0:
                        pass
                    else:
                        nearest_factory = min(
                            dist_factories, key=lambda k: len(dist_factories[k])
                        ) # min returns the key s.t. dist_factories has smallest value
                        path = dist_factories[nearest_factory]
                        paths_temp_lst.append(path)
                        tp_list.append(t_p)

                if flag_problem:
                    break  # type: ignore[unreachable]

            # add case for flag problem to avoid infinite loop
            # if only t gates in terminal_pairs_current and empty paths_temp_lst, because then we are stuck because of reset time of factories
            all_t = []
            for _ in terminal_pairs_current:
                if isinstance(t_p[0], int) and isinstance(t_p[1], int):
                    all_t.append(True)
                else:
                    all_t.append(False)
            if all(all_t) and len(paths_temp_lst) == 0:
                flag_problem = True

            if len(paths_temp_lst) != 0 and not flag_problem:
                shortest_path = min(paths_temp_lst, key=len)
                shortest_idx = paths_temp_lst.index(
                    shortest_path
                )  # index in current terminal_pairs_current
                t_p = tp_list[shortest_idx]  # terminal_pairs_current[shortest_idx]
                # update already used qubits based on chosen t_p path
                if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                    dct_qubits[t_p[0]] = True
                    dct_qubits[t_p[1]] = True
                elif isinstance(t_p[1], int):
                    dct_qubits[t_p] = True
                    # update the times of the factory patch (which is the position at one end of the path)
                    if shortest_path[0] == t_p:
                        factory_times_temp[shortest_path[-1]] = self.t + 1
                    elif shortest_path[-1] == t_p:
                        factory_times_temp[shortest_path[0]] = self.t + 1
                    else:
                        msg = "Factory not in path."
                        raise RuntimeError(msg)

                # remove nodes from g_temp from path
                for node in shortest_path[1:-1]:
                    g_temp.remove_node(node)
                successful_terminals.append(t_p)
                vdp_dict.update({t_p: shortest_path})

                # remove t_p from terminal_pairs_current
                terminal_pairs_current = [x for x in terminal_pairs_current if x != t_p]

            if len(paths_temp_lst) == 0 or flag_problem:
                terminal_pairs_remainder = [
                    s for s in terminal_pairs_current if s not in successful_terminals
                ]
                dct_qubits = dct_qubits_copy.copy()
                flag_problem = True  # in case the paths_temp_list is empty, you need to set the flag_problem to handle this case as well. otherwise endless loop.

        # check whether the keys in vdp_dict fit the start and end point of the path
        for pair, path in vdp_dict.items():
            start, end = path[0], path[-1]
            if isinstance(pair[1], tuple) and set(pair) != {start, end}:
                msg = f"The path does not coincide with the terminal pair. There is a bug. terminal_pair = {pair} but path = {path}"
                raise RuntimeError(msg)
            if isinstance(pair[1], int) and pair not in {start, end}:
                msg = f"The path does not coincide with the T gate location. There is a bug. terminal_pair = {pair} but path = {path}"
                raise RuntimeError(msg)

        return vdp_dict, terminal_pairs_remainder, factory_times_temp


    def make_route_info(
        self,
        gate: Gate,
        path: list[pos],
        ancilla_idx: int ,
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

    def default_ancilla_idx(self, path: list[pos]) -> int:
        if len(path) < 3:
            raise ValueError(f"Path too short for internal ancilla: {path}")
        return len(path) // 2

    def get_overlaps(
        self,
        layer_idx: int,
        route_info: RouteInfo,
        *,
        include_endpoints: bool = True,
    ) -> dict[Gate, set[pos]]:
        """
        Check which existing routes in this layer overlap with the candidate route.

        return dict{gate: set of overlapped nodes}
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

        It does not modify self.overlap_graphs.
        """

        #print("local overlap function check: ")
        #print("Gate: ", route_info.gate)
        G = self.overlap_graphs[layer_idx]

        #print("Overlap graph before adding this path", G)

        direct_overlaps = self.get_overlaps(
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
        local_og: nx.Graph,
        allowed_overlap: str = "strict2",
    ) -> tuple[bool, dict[Gate, RouteInfo] | None, str]:
        """
        Dispatch overlap-validity checking on the chosen method.

        Given the candidate route `new_info` and the local overlap graph `local_og`
        (the connected component the candidate joins, carrying every gate's RouteInfo
        on its nodes and pairwise `overlap_nodes` on its edges), decide whether the
        whole component can be routed in the two-phase model.

        Returns (valid, updated, reason) where `updated` maps every gate in the
        component (the candidate plus any re-assigned existing routes) to its final
        RouteInfo, or None when the candidate cannot be placed.
        """
        if allowed_overlap == "strict2":
            others = [g for g in local_og.nodes if g != new_info.gate]
            if len(others) != 1:
                return False, None, (
                    f"strict2 expects exactly one overlapping path, got {len(others)}."
                )
            path_ov_info = local_og.nodes[others[0]]["route_info"]
            valid, upd_new, upd_old, reason = self._check_validity_strict2(
                new_info, path_ov_info
            )
            if not valid:
                return False, None, reason
            return True, {new_info.gate: upd_new, path_ov_info.gate: upd_old}, reason
        elif allowed_overlap == "strict_k":
            return self._check_validity_strict_k(local_og)
        else:
            raise NotImplementedError(
                f"Unknown allowed_overlap={allowed_overlap!r}"
            )

    def _check_validity_strict2(
        self,
        new_info: RouteInfo,
        #local_ov_gf: nx.Graph,
        path_ov_info: RouteInfo,
        allowed_overlap: str = "strict2",
    ) -> tuple[bool, RouteInfo | None, RouteInfo | None, str]:
        """
        Check whether a new route candidate is valid given overlapped path(s) and the allowed overlap criterion
        If valid, update ancilla positions and routing order for overlapped paths if needed 

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

            #print("new path: Control to overlap: ", d11)
            #print("new path: Target to overlap: ", d12)
            #print("old path: Control to overlap: ", d21)
            #print("old path: Target to overlap: ", d22)
            # A side is usable if there is at least one internal node where we can place the ancilla.
            new_left_ok = d11 >= 2
            new_right_ok = d12 >= 2
            old_left_ok = d21 >= 2
            old_right_ok = d22 >= 2

            # --------------------------------------------------
            # 5. Choose which side to place the two ancillas
            # --------------------------------------------------

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
                # not valid for strict2 method 
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
            #print("chosen case:", chosen_case)
            # --------------------------------------------------
            # 6. Randomly choose valid ancilla positions
            # --------------------------------------------------
            if new_side == "left":
                a1 = random.randint(1, ov_start - 1)
            else:
                a1 = random.randint(ov_end + 1, len(new_path) - 2)

            if old_side == "left":
                a2 = random.randint(1, ov_st2 - 1)
            else:
                a2 = random.randint(ov_end2 + 1, len(path_ov) - 2)

            if chosen_case == ("left", "left") or chosen_case == ("right", "right"):
                new_first_part = "control"
                old_first_part = "target"
            elif chosen_case == ("left", "right") or chosen_case == ("right", "left"):
                new_first_part = "control"
                old_first_part = "control"


            #print("new first part:", new_first_part)
            #print("old first part:", old_first_part)
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
    
    @staticmethod
    def _is_consecutive(sorted_idxs: list[int]) -> bool:
        """True if the sorted indices form one contiguous run (a single block)."""
        return all(
            sorted_idxs[k + 1] - sorted_idxs[k] == 1
            for k in range(len(sorted_idxs) - 1)
        )

    @staticmethod
    def _block_phase(side: str, first_part: FirstPart) -> int:
        """
        Phase (1 = routed first, 2 = routed second) in which a path occupies a block
        that lies on the given side of its ancilla, given which end is routed first.

        control side = path[:ancilla]; target side = path[ancilla:].
        """
        if side == "control":
            return 1 if first_part == "control" else 2
        return 2 if first_part == "control" else 1

    @staticmethod
    def _ancilla_region_bounds(
        path_len: int,
        blocks: list[dict],
        region: int,
    ) -> tuple[int, int]:
        """
        Inclusive [lo, hi] range of ancilla indices for `region` of a path.

        `region` == j means the ancilla sits between block j-1 and block j, so blocks
        0..j-1 fall on the control side and blocks j..end on the target side. The
        ancilla must be an internal node (1..path_len-2) strictly outside every block.
        Returns lo > hi when the gap has no room for an ancilla.
        """
        m = len(blocks)
        lo = 1 if region == 0 else blocks[region - 1]["end"] + 1
        hi = (path_len - 2) if region == m else blocks[region]["start"] - 1
        return lo, hi

    def _check_validity_strict_k(
        self,
        local_og: nx.Graph,
    ) -> tuple[bool, dict[Gate, RouteInfo] | None, str]:
        """
        General overlap criterion for a component of paths (caller caps the size).

        Two-phase model: each path is split at an ancilla into a first part (phase 1)
        and a second part (phase 2). All first parts must be pairwise node-disjoint and
        all second parts pairwise node-disjoint. Every grid node has capacity 2, so a
        node shared by >= 3 paths is infeasible.

        The routine re-solves the whole component: it may re-assign the ancilla_idx and
        first_part of already-committed paths as well as the new candidate. On success
        it returns (True, {gate: updated RouteInfo for every gate}, reason).
        """
        gates = list(local_og.nodes)
        route_infos = {g: local_og.nodes[g]["route_info"] for g in gates}

        # 1. Reject any grid node shared by >= 3 paths (infeasible in two phases).
        node_count: dict[pos, int] = defaultdict(int)
        for gt in gates:
            for node in route_infos[gt].path:
                node_count[node] += 1
        if any(c >= 3 for c in node_count.values()):
            return False, None, "A grid node is shared by >= 3 paths."

        # 2. Build each gate's overlap blocks from the component's edges.
        # lookup table 
        idx_of = {
            g: {node: i for i, node in enumerate(route_infos[g].path)}
            for g in gates
        }
        blocks_by_gate: dict[Gate, list[dict]] = {g: [] for g in gates}

        for g, h in local_og.edges:
            overlap_nodes = local_og[g][h].get("overlap_nodes")
            if not overlap_nodes:
                return False, None, f"Edge {g}-{h} carries no overlap_nodes."

            ig = sorted(idx_of[g][n] for n in overlap_nodes)
            ih = sorted(idx_of[h][n] for n in overlap_nodes)
            if not self._is_consecutive(ig) or not self._is_consecutive(ih):
                return (
                    False,
                    None,
                    f"Overlap between {g} and {h} is not a single consecutive block.",
                )
            # this route overlaps neighbor h from index start to index end
            blocks_by_gate[g].append({"start": ig[0], "end": ig[-1], "neighbor": h})
            blocks_by_gate[h].append({"start": ih[0], "end": ih[-1], "neighbor": g})

        # orders the overlap blocks from left to right along each path.
        for g in gates:
            blocks_by_gate[g].sort(key=lambda b: b["start"])

        # 3. Per gate, enumerate placement options (ancilla region x first_part).
        #    Each option maps every neighbour's block to a phase.
        options_by_gate: dict[Gate, list[dict]] = {}
        for g in gates:
            path_len = len(route_infos[g].path)
            blocks = blocks_by_gate[g]
            gate_options: list[dict] = []
            for region in range(len(blocks) + 1):
                lo, hi = self._ancilla_region_bounds(path_len, blocks, region)
                if lo > hi:
                    continue  # no room for an ancilla in this gap
                for first_part in ("control", "target"):
                    # decide whether each overlap block lies on the control side or target side of the ancilla 
                    phase_of_block = {
                        b["neighbor"]: self._block_phase(
                            "control" if bi < region else "target",
                            first_part,
                        )
                        for bi, b in enumerate(blocks)
                    }
                    gate_options.append(
                        {
                            "lo": lo,
                            "hi": hi,
                            "first_part": first_part,
                            "phase_of_block": phase_of_block,
                        }
                    )
            if not gate_options:
                return False, None, f"No room to place an ancilla on gate {g}."
            options_by_gate[g] = gate_options

        # 4. Backtracking search for a component-wide consistent assignment:
        #    every shared block must be routed in opposite phases by its two paths.
        chosen: dict[Gate, dict] = {}

        def consistent(g: Gate, opt: dict) -> bool:
            for h, h_opt in chosen.items():
                if local_og.has_edge(g, h):
                    if opt["phase_of_block"][h] == h_opt["phase_of_block"][g]:
                        return False
            return True

        def backtrack(i: int) -> bool:
            # finish checking all gates -> done 
            if i == len(gates):
                return True
            g = gates[i]
            for opt in options_by_gate[g]:
                if consistent(g, opt):
                    chosen[g] = opt
                    if backtrack(i + 1):
                        return True
                    del chosen[g]
            return False

        if not backtrack(0):
            return (
                False,
                None,
                "No phase assignment satisfies the two-phase disjointness constraints.",
            )

        # 5. Materialise concrete ancilla indices and first_part for every gate.
        updated: dict[Gate, RouteInfo] = {}
        for g in gates:
            opt = chosen[g]
            ancilla_idx = random.randint(opt["lo"], opt["hi"])
            updated[g] = replace(
                route_infos[g],
                ancilla_idx=ancilla_idx,
                first_part=opt["first_part"],
            )

        return True, updated, f"Valid strict_k assignment for {len(gates)} paths."

    def _commit_component(
        self,
        layer_idx: int,
        candidate_gate: Gate,
        updated: dict[Gate, RouteInfo],
    ) -> None:
        """
        Commit a validated overlap component.

        `updated` maps every gate in the component to its final RouteInfo. Existing
        routes only change ancilla_idx / first_part (their paths are unchanged), so they
        are updated in place; the new candidate is committed via `commit_route`, which
        also creates its overlap edges and node_to_gates entries.
        """
        for gate, info in updated.items():
            if gate == candidate_gate:
                continue
            self.routes_by_layer[layer_idx][gate] = info
            if self.overlap_graphs[layer_idx].has_node(gate):
                self.overlap_graphs[layer_idx].nodes[gate]["route_info"] = info

        self.commit_route(layer_idx, updated[candidate_gate])

    def commit_route(
    self,
    layer_idx: int,
    route_info: RouteInfo,
    *,
    include_endpoints: bool = True,
    ) -> None:
        """
        For a valid new route, commit the route by storing the route in self.routes_by_layer, 
        update self.overlap_graphs[layer_idx], and self.node_to_gates_by_layer[layer_idx]
        """
        gate = route_info.gate
        
        node_to_gates = self.node_to_gates_by_layer[layer_idx]
        G = self.overlap_graphs[layer_idx]

        # 1. Store route
        self.routes_by_layer[layer_idx][gate] = route_info

        # 2. Add graph node, even if there is no overlap
        G.add_node(gate, route_info=route_info)

        # 3. Update node_to_gates and overlap edges
        for node in self.route_nodes(route_info, include_endpoints=include_endpoints):
            existing_gates = node_to_gates[node]

            for other_gate in existing_gates:
                if other_gate == gate:
                    continue
                
                # could happen that two paths overlap with several nodes (consecutively)
                if self.overlap_graphs[layer_idx].has_edge(gate, other_gate):
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
        
        #print("updated overlap graph: ", G)

    def find_fine_grained_vdp(self,
            layer_idx: int,
            layer: list[tuple[pos, pos] | pos],
            logical_pos: None | list[pos],
            factory_times: dict[pos, int],
            overlap_type:str,
        ) -> list[Gate]:
            
        paths_current_layer = [] 
        gates_current_layer = layer.copy()
        remainder_terminal_pairs = []
        for gate in gates_current_layer:

            g_temp = self.g.copy()

            nodes_to_remove = [
                x for x in (self.logical_pos if logical_pos is None else logical_pos)
                if x != gate[0] and x != gate[1]
            ]
            g_temp.remove_nodes_from(nodes_to_remove)

            path = self.valid_path_method()(g_temp, gate[0], gate[1])
            #print("Gate", gate, " : ", path )
            route_info = self.make_route_info(
                gate,
                path,
                ancilla_idx=self.default_ancilla_idx(path),
                first_part="control",
            )
            if paths_current_layer:
                local_og = self.get_local_overlap_graph(layer_idx, route_info)
                n = local_og.number_of_nodes()

                if n == 1:
                    # No overlap with any committed route in this layer.
                    self.commit_route(layer_idx, route_info)
                    paths_current_layer.append(path)
                    continue

                # Component-size cap per method (strict2 = 2 paths, strict_k = up to 4).
                if overlap_type == "strict2":
                    within_cap = n == 2
                elif overlap_type == "strict_k":
                    within_cap = n <= 5
                else:
                    raise NotImplementedError(
                        f"Unknown overlap_type={overlap_type!r}"
                    )

                if within_cap:
                    valid, updated, reason = self.check_validity(
                        route_info,
                        local_og,
                        allowed_overlap=overlap_type,
                    )
                else:
                    valid, updated, reason = (
                        False,
                        None,
                        f"Component size {n} exceeds cap for {overlap_type}.",
                    )

                if valid:
                    self._commit_component(layer_idx, route_info.gate, updated)
                    paths_current_layer.append(path)
                    continue

                # Alternative route: re-route avoiding every node already occupied.
                nodes_occupied = [
                    node
                    for old_path in paths_current_layer
                    for node in old_path
                ]

                g_temp.remove_nodes_from(nodes_occupied)
                try:
                    alt_path = self.valid_path_method()(g_temp, gate[0], gate[1])
                    alt_info = self.make_route_info(
                        gate,
                        alt_path,
                        ancilla_idx=self.default_ancilla_idx(alt_path),
                        first_part="control",
                    )
                    paths_current_layer.append(alt_path)
                    self.commit_route(layer_idx, alt_info)
                except nx.NetworkXNoPath:
                    logger.info(
                        f"No path found for gate {gate}"
                    )
                    remainder_terminal_pairs.append(gate)
                        
            else:   
                paths_current_layer.append(path)
                self.commit_route(layer_idx, route_info)

        return remainder_terminal_pairs
    

    def find_total_fine_grained_vdp_dyn(
    self,
    layers,
    logical_pos,
    factory_times,
    layout = None,
    overlap_type: str = "strict2",
    testing: bool = False
    ):
        if self.use_dag and layout is None:
            raise ValueError(
                "If self.use_dag=True, layout must be provided."
            )

        # Optional but recommended: clear previous fine-grained routing state
        self.routes_by_layer.clear()
        self.overlap_graphs.clear()
        self.node_to_gates_by_layer.clear()

        remaining_layers = [layer.copy() for layer in layers]

        # Build DAG if needed
        if self.use_dag:
            terminal_pairs_temp = []
            for layer in remaining_layers:
                terminal_pairs_temp += layer

            dag = dag_helper.terminal_pairs_into_dag(
                terminal_pairs_temp,
                layout,
            )
        else:
            dag = None

        fine_layer_idx = 0

        while remaining_layers:
            if self.use_dag:
                current_layer = dag_helper.extract_layer_from_dag(
                    dag,
                    layout,
                    0,
                )
            else:
                current_layer = remaining_layers[0]

            terminal_pairs_remainder = self.find_fine_grained_vdp(
                fine_layer_idx,
                current_layer,
                logical_pos,
                factory_times,
                overlap_type,
            )

            if self.use_dag:
                remaining_layers, dag = dag_helper.push_remainder_into_layers_dag(
                    dag,
                    terminal_pairs_remainder,
                    layout,
                    current_layer,
                )
            else:
                remaining_layers = self.push_remainder_into_layers(
                    remaining_layers,
                    terminal_pairs_remainder,
                    delete_layer_zero=True,
                )

            fine_layer_idx += 1

        if testing:
            if tst.test_duplicate_nodes_fg(self.routes_by_layer):
                logger.info("Successful fine grained routing")
            else:
                logger.info("Problematic! Nodes are used by two paths")



    def push_remainder_into_layers(
        self,
        layers,
        remainder: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]],
        delete_layer_zero: bool = True,
    ) -> list[list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Updates a copy of layers_cnot_t (removed used stuff and takes remainder of previous layer, pushes through).

        Only really needed if self.use_dag=False. If true, it is often computed uselessly.

        Args:
            terminal_pairs (list[tuple[int,int]]): terminal pairs to be layered.
            remainder (list[tuple[int,int]]): remaining gates which could not be routed so far in current layer.

        Returns:
            list[list[tuple[int,int]]]: layered gates with remainder being pushed into next layer.
        """
        initial_layers = layers.copy()

        #if delete_layer_zero:
        #    if len(initial_layers) > 1:
        #        del initial_layers[
        #            0
        #        ]  # delete already processed layer (remainder was part of this layer)
        #    elif len(initial_layers) == 1 and len(remainder) != 0:
        #        del initial_layers[0]

        if delete_layer_zero and len(initial_layers) > 0:
            del initial_layers[0]

        if not remainder:
            return initial_layers
        
        i = 0
        flag = True
        while flag is True:
            try:
                initial_layers[i] = (
                    remainder + initial_layers[i]
                )  # push remainder in front of the new zeroth entry (previously entry 1)
            except (
                IndexError
            ):  # if no further initial_layer[i] available but still the previous layer was split
                initial_layers.append(remainder)
            layers = self.split_layer_terminal_pairs(initial_layers[i]) # Split terminal_pairs into layers of disjoint qubit support
            if len(layers) == 1:
                # adding remainder to initial_layers[0] caused no conflict, so we are finished
                break
            if len(layers) >= 2:  # push further through
                initial_layers[i] = layers[0]
                remainder = [item for sublayer in layers[1:] for item in sublayer]
                i += 1
            else:
                msg = f"Something weird happened during pushing remainders. len(layers)={len(layers)}, layers = {layers}"
                raise RuntimeError(msg)

        return initial_layers
    
    def find_total_vdp_layers_dyn(
        self,
        next_layers,
        logical_pos,
        factory_times: dict[pos, int],
        layout,
        testing: bool = False,
    ) -> list[
        dict[
            tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
            list[tuple[int, int]],
        ]
    ]:
        """
        Finds all routes for given logical layers called `next_layers`.

        This is only used if self.use_dag = False

        Dynamically pushes remaining gates into the next layers.
        """

        if self.use_dag:
            if layout is None:
                raise ValueError(
                    "If self.use_dag=True you need to enter a layout in find_total_vdp_layers_dyn"
                )

        factory_times_temp = factory_times.copy()
        terminal_pairs = []
        for layer in next_layers:
            terminal_pairs += layer
        layers_cnot_t_orig = next_layers.copy()
        vdp_layers: list[
            dict[
                tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
                list[tuple[int, int]],
            ]
        ] = [] # each layer is a dict, key: gate, value: path

        layers_cnot_t_prev = None

        stuck_counter = 0

        terminal_pairs_temp = []
        for layer_temp in layers_cnot_t_orig:
            terminal_pairs_temp += layer_temp
        dag = dag_helper.terminal_pairs_into_dag(terminal_pairs_temp, layout)

        while len(layers_cnot_t_orig) > 0:
            layer = 0  # since we adapt the layers_cnot_t_orig in place, always layer=0 needed

            if self.use_dag:
                current_layer = dag_helper.extract_layer_from_dag(
                    dag, layout, layer
                )  # layer=0
            else:
                current_layer = layers_cnot_t_orig[layer]

            vdp_dict, terminal_pairs_remainder, factory_times_temp = (
                self.find_max_vdp_set(
                    current_layer,
                    logical_pos=logical_pos,
                    factory_times=factory_times_temp,  #!IMPORTANT: logical_pos is not taken from self.logical_pos here, because it would be overwritten too often in the annealing procedure. but we need to compute the metric correctly with this _dyn method
                )
            )  # layer is successively reordered within find_max_vdp_set

            keys: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = []
            for lst in vdp_layers:
                keys += list(lst.keys())
            if layers_cnot_t_prev == layers_cnot_t_orig and len(keys) == len(
                terminal_pairs
            ):
                break
            layers_cnot_t_prev = layers_cnot_t_orig.copy()

            if len(vdp_dict) == 0:
                stuck_counter += 1
            if stuck_counter > 10 * self.t:
                return None, factory_times_temp  # need a stuck counter!!!

            for key in factory_times_temp:
                if factory_times_temp[key] != 0:
                    factory_times_temp[key] -= 1

            vdp_layers.append(vdp_dict)
            if self.use_dag:
                initial_layers_update, dag = dag_helper.push_remainder_into_layers_dag(
                    dag, terminal_pairs_remainder, layout, current_layer
                )
            else:
                initial_layers_update = self.push_remainder_into_layers(
                    layers_cnot_t_orig, terminal_pairs_remainder
                )
            layers_cnot_t_orig = initial_layers_update
            if len(layers_cnot_t_orig) == 0:
                break

        # it might be possible that there are bugs. hence, check whether vdp layers really contains as main paths as there are gates.
        keys = []
        for lst in vdp_layers:
            keys += lst.keys()
        assert len(keys) == len(
            terminal_pairs
        ), f"The dynamic routing has a bug. There are {len(terminal_pairs)} to be routed, but the final vdp_layers only has {len(keys)} paths."

        if testing:
            if tst.check_order_dyn_gates_st(terminal_pairs, vdp_layers, layout=layout):
                logger.info("stim test succeeded for standard routing (:")
            else:
                logger.info("stim test failed - THERE IS A PROBLEM!")
            if tst.check_duplicate_nodes_per_layer_st(vdp_layers):
                logger.info("no duplicates found in standard routing (:")

            if logical_pos is None:
                if tst.check_path_on_logical_st(vdp_layers, self.logical_pos):
                    logger.info("paths do not occupy logical pos (:")
            else:
                if tst.check_path_on_logical_st(vdp_layers, logical_pos):
                    logger.info("paths do not occupy logical pos (:")
            if tst.test_times_t_gates_st(vdp_layers, self.t, self.factory_pos):
                logger.info("Reset times make sense, all good(:")

        return vdp_layers, factory_times_temp
    
    def count_crossings(self, layers, logical_pos_temp: list[pos] | None = None):
        """
        Heuristic energy function to minimize.

        Computes number of crossings of Dijkstra paths without constraints.
        If there is a shortest path which is not possible at all (due to locking), then add high, hard coded penalty.
        """
        # lock_penalty = 50
        total_penalty = 0
        lst_crossings = []
        for layer in layers:
            paths = []
            for t_p in layer:
                g_temp = (
                    self.g.copy()
                )  # this is duplicate code from `order_terminal_pairs`
                terminal_pairs_flattened = [
                    pair for sublist in layer for pair in sublist
                ]
                terminals_temp = [
                    pair
                    for pair in terminal_pairs_flattened
                    if pair != t_p[0] and pair != t_p[1]
                ]
                terminals_temp = list(set(terminals_temp))
                g_temp.remove_nodes_from(terminals_temp)
                nodes_to_remove = [
                    x for x in logical_pos_temp if x != t_p[0] and x != t_p[1]
                ]  # these are also necessary since logical vertices which are not used in the layer are important nevertheless.
                g_temp.remove_nodes_from(nodes_to_remove)
                try:
                    path = self.valid_path_method()(g_temp, t_p[0], t_p[1])
                    paths.append(path)
                except nx.NetworkXNoPath as exc:
                    total_penalty += lock_penalty
            # check the paths for overlaps
            # Create a mapping of elements to the sublists they appear in
            element_to_sublists = collections.defaultdict(set)
            for i, sublist in enumerate(paths):
                for element in sublist:
                    element_to_sublists[element].add(i)
            # Count crossings (pairwise sublist overlaps for each element)
            crossing_count = 0
            for sublists in element_to_sublists.values():
                if len(sublists) > 1:
                    crossing_count += len(list(itertools.combinations(sublists, 2)))
            lst_crossings.append(crossing_count)

        return sum(lst_crossings) + total_penalty

    def count_crossings_per_layer(self, layers, t_crossings: bool = False) -> list[int]:
        """Counts the crossings of the simple paths between cnots and between shortest factory to qubit path (respecting terminals and factory positions) per layer.

        Args:
            t_crossings (bool): decides whether the crossings to the factory are included (true) or not (false).

        Returns:
            list[int]: Number of crossings per initial layer. len is len(self.layers_cnot_t_orig)
        """
        # ! TODO: remove redundancies (order_terminal_pairs very similar)
        lst_crossings = []
        flattened_terminals_and_factories = self.factory_pos.copy()
        for layer in layers:
            temp = [
                pair
                for item in layer.copy()
                for pair in (item if isinstance(item[0], tuple) else [item])
            ]
            flattened_terminals_and_factories += temp

        for layer in layers:
            paths = []
            for t_p in layer:
                g_temp = self.g.copy()
                if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                    terminals_temp = [
                        pair
                        for pair in flattened_terminals_and_factories.copy()
                        if pair != t_p[0] and pair != t_p[1]
                    ]
                    terminals_temp = list(set(terminals_temp))
                    g_temp.remove_nodes_from(terminals_temp)
                    try:
                        path = nx.dijkstra_path(g_temp, t_p[0], t_p[1])
                        paths.append(path)
                    except nx.NetworkXNoPath as exc:
                        msg = (
                            "Your choice of terminal pairs locks in at least one terminal. "
                            "Reconsider your choice of terminal pairs."
                        )
                        raise ValueError(msg) from exc

                elif t_crossings:
                    dist_factories = (
                        {}
                    )  # gather distances to each factory to greedily choose the shortest path
                    for factory in self.factory_pos:
                        g_temp = self.g.copy()
                        terminals_temp = [
                            pair
                            for pair in flattened_terminals_and_factories.copy()
                            if pair not in {t_p, factory}
                        ]
                        terminals_temp = list(set(terminals_temp))
                        g_temp.remove_nodes_from(terminals_temp)
                        try:
                            path = nx.dijkstra_path(g_temp, t_p, factory)
                        except nx.NetworkXNoPath as exc:
                            msg = (
                                "Your choice of terminal pairs locks in at least one terminal. "
                                "Reconsider your choice of terminal pairs."
                            )
                            raise ValueError(msg) from exc
                        dist_factories.update({factory: path})
                    # choose shortest factory path
                    nearest_factory = min(
                        dist_factories, key=lambda k: len(dist_factories[k])
                    )
                    paths.append(dist_factories[nearest_factory])
            # check the paths for overlaps
            # Create a mapping of elements to the sublists they appear in
            element_to_sublists = collections.defaultdict(set)
            for i, sublist in enumerate(paths):
                for element in sublist:
                    element_to_sublists[element].add(i)
            # Count crossings (pairwise sublist overlaps for each element)
            crossing_count = 0
            for sublists in element_to_sublists.values():
                if len(sublists) > 1:
                    crossing_count += len(list(itertools.combinations(sublists, 2)))
            lst_crossings.append(crossing_count)
        return lst_crossings


class TeleportationRouter(BasicRouter):
    """Compilation routine that exploits CNOT + teleportation steps."""

    def __init__(
        self,
        g: nx.Graph,
        logical_pos: list[pos],
        factory_pos: list[pos],
        valid_path: str,
        t: int,
        metric: str,
        use_dag: bool,
        seed: int,
    ):
        """Compilation routine that exploits CNOT + teleportation steps based on BasicRouter.

        Args:
            g (nx.Graph): Macroscopic Routing Graph. Created via mqt.cococo.layouts
            logical_pos (list[pos]): Logical positions on the graph. Also from mqt.cococo.layouts
            factory_pos (list[pos]): Positions of the factories. Also from mqt.cococo.layouts
            valid_path (str): Either "cc" or "sc" for color code and surface code. However, revisit usefulness of "sc".
            t (int): Reset time for the factories
            metric (str): Either "exact" or "crossing", but it is recommended to use "exact"
            use_dag (bool, optional): Determins whether DAG structure from qiskit is used or naive sequential layering. It is recommended to use `True`.
        """
        super().__init__(g, logical_pos, factory_pos, valid_path, t, metric, use_dag)
        self.seed = seed
        random.seed(seed)

    
    def initialize_steiner(
        self, vdp_dict, steiner_init_type: str, layers=None, k_lookahead=None
    ):
        """
        Initialize a random steiner tree per path which are non-overlapping.

        This initialization depends on the vdp_dict you already have for your current layer. This is fixed, it only tries to find additional terminal of a 3-terminal steiner tree.

        If layers and k_lookahead are not None, only a limited number of trees is initialized; namely only for the qubits which are actually used in layers[:k_lookahead]. Other qubits are not moved.
        However, this turned out not to be really useful, because movements can be relevant even if this constraint does not hold.
        Therefore, layers and k_lookahead default to None.
        """

        vdp_dict_reduced = vdp_dict.copy()

        # find out which qubits are actually moved in layers[:k_lookahead]
        if layers is not None and k_lookahead is not None:
            layers_temp = layers[:k_lookahead]
            terminals = []
            for layer in layers_temp:
                terminals += layer
            qubits_k_lookahead = [t for outer in terminals for t in outer]
            # remove those from vdp_dict which are not used, such that no tree is created for them
            vdp_dict_reduced = {}
            for key, val in vdp_dict.items():
                if key[0] in qubits_k_lookahead or key[1] in qubits_k_lookahead:
                    vdp_dict_reduced.update({key: val})

        
        # remove the nodes from the graph which are already occupied by the magic/logical patches
        # we allow the 3-terminal to be placed on the path, thus the graph must be adapted per terminal choice
        # because for path a, you can place the terminal along path a, but not along path b
        g_temp = self.g.copy()
        g_temp.remove_nodes_from(self.factory_pos)
        g_temp.remove_nodes_from(self.logical_pos)

        if steiner_init_type not in {"full_random", "on_path_random"}:
            raise NotImplementedError(
                "Make sure that `steiner_init_type` is either full_random or on_path_random. Other possibilities not implemented yet."
            )

        steiner_dct = {}

        # for each already present path, choose one random ancilla terminal which is allowed to be on the same path, but not on another path
        for key, path in vdp_dict_reduced.items():
            if isinstance(key, tuple) and key[0] == "idle_back":
                continue  # we do not want to create a tree at an idling path!
            other_paths = [
                pos for keyy, path in vdp_dict.items() if keyy != key for pos in path
            ]  # collect all terminals occupied by other paths which is not the present one, this also captures potential idling paths.
            g_temp_temp = g_temp.copy()
            g_temp_temp.remove_nodes_from(other_paths)
            # choose some node on the path randomly
            flag = False
            pathcopy = path.copy()
            path = path[1:-1]  # remove last and first node from the list because those are logical data patches
            random.shuffle(path)
            if steiner_init_type == "full_random":
                for node_on_path in path:  # loop in case a random node has no other reachable nodes
                    # determine all reachable nodes from that chosen node
                    reachable_nodes = list(
                        nx.single_source_shortest_path_length(
                            g_temp_temp, node_on_path
                        ).keys()
                    )
                    if reachable_nodes:
                        flag = True
                        break  # if there is a reachable node found, take it, otherwise try another random node
                if not flag:
                    continue  # skip this path if no reachable node found
                # select a random reachable node
                terminal_node = random.choice(reachable_nodes)
                #! TODO are these three lines redundant?
                # determine the path between the node which is ensured on the path and the terminal
                path_steiner = nx.dijkstra_path(
                    g_temp_temp, node_on_path, terminal_node
                )
                paths_lst_temp = (
                    []
                )  # collect all paths from path1[1:-1] to new_terminal
                for node_on_path in path:
                    try:
                        path_temp = nx.dijkstra_path(
                            g_temp_temp, node_on_path, terminal_node
                        )
                        paths_lst_temp.append(path_temp)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                if paths_lst_temp:
                    path_steiner = min(paths_lst_temp, key=len)
            elif steiner_init_type == "on_path_random":
                terminal_node = random.choice(path)  # choose a random terminal ON the path
                path_steiner = [
                    terminal_node
                ]  # terminal on the path does not need an extended path, but list should not be empty, otherwise error.
            else:
                raise ValueError(
                    "steiner_init_type must be on_path_random or full_random."
                )
            # add to list
            if isinstance(key[0], tuple):  # if CNOT key: tuple[pos, pos], (terminal_node, ) makes it a tuple
                tup = key + (terminal_node,)
            elif isinstance(key[0], int):  # if T gate, key: pos
                tup = (key, terminal_node)
            steiner_dct.update({tup: [pathcopy, path_steiner]})
            # also remove the nodes from the graph such that no overlapping steiner trees can be generated
            g_temp.remove_nodes_from(path_steiner)

        #print("steiner dict: ", steiner_dct)
        
        return steiner_dct

    

    def initialize_idle_moves(
        self,
        vdp_dict: dict,
        steiner_dct: dict,
        max_idle_moves: int | None = None,
        layers = None, 
        k_lookahead = None, 
    ):
        """
        Initialize random idle-qubit teleportation moves.

        Returns:
            idle_move_dct:
                {
                    ("idle", q, terminal): path_idle
                }
        """
        def logical_qbts_in_vdp(key):
            if isinstance(key, tuple) and key[0] == "idle_back":
                # logical qubits is in gap position after idle_move_back
                return [key[2]]

            elif isinstance(key[0], tuple):
                # CNOT key: ((x1, y1), (x2, y2))
                return [key[0], key[1]]

            elif isinstance(key[0], int):
                # T-gate key: (x, y)
                return [key]

            raise RuntimeError(f"Unexpected vdp key: {key}")


        def occupied_ancillas_from_vdp(vdp_dict):
            occupied = set()
            for key, path in vdp_dict.items():
                #occupied.update(path)
                if isinstance(key, tuple) and key[0] == "idle_back": # idle 
                    occupied.update(path[1:])
                elif isinstance(key[0], tuple): # CNOT
                    occupied.update(path[1:-1])
                elif isinstance(key[0], int): # T 
                    occupied.update(path[1:])
                else:
                    raise RuntimeError(f"Unexpected VDP key: {key}")
            return occupied

        def occupied_nodes_from_steiner(steiner_dct):
            occupied = set()

            if steiner_dct is None:
                return occupied

            for key, (path1, path2) in steiner_dct.items():
                occupied.update(path1)
                occupied.update(path2)

            return occupied
        
        #logical_positions = list(layout.values())

        #print("vdp dict: ", vdp_dict)

        active_qubits = set()
        for key in vdp_dict:
            active_qubits.update(logical_qbts_in_vdp(key))

        #print("vdp dict: ", vdp_dict.keys())

    

        # only choose idle qubits which are active in future k layers
        if layers is not None and k_lookahead is not None:
            layers_temp = layers[:k_lookahead]
            terminals = []
            for layer in layers_temp:
                terminals += layer
            qubits_k_lookahead = set([t for outer in terminals for t in outer])
            #print("qubits used in next k layers: ", qubits_k_lookahead)
            idle_qubits = [
                q for q in self.logical_pos
                if q not in active_qubits and q in qubits_k_lookahead
            ]
        else:
            idle_qubits = [
                q for q in self.logical_pos
                if q not in active_qubits
            ]

        #print("idle qubits:", idle_qubits)
        random.shuffle(idle_qubits)

        if max_idle_moves is not None:
            idle_qubits = idle_qubits[:max_idle_moves]


        #print("idle qubits (max):", idle_qubits)
        occupied = set()
        occupied.update(occupied_ancillas_from_vdp(vdp_dict))
        occupied.update(occupied_nodes_from_steiner(steiner_dct)) 

        
        idle_move_dct = {}

        used_idle_paths = set()

        for q in idle_qubits:
            g_temp = self.g.copy()

            # Remove factories.
            g_temp.remove_nodes_from(self.factory_pos)

            # Remove occupied current-layer routing/Steiner nodes.
            g_temp.remove_nodes_from(occupied)
            g_temp.remove_nodes_from(used_idle_paths)

            # Remove all other logical qubits.
            for pos in self.logical_pos:
                if pos != q and pos in g_temp.nodes:
                    g_temp.remove_node(pos)

            if q not in g_temp.nodes:
                #print("skip idle qubit ", q)
                continue

            reachable = list(nx.single_source_shortest_path_length(g_temp, q).keys())
            reachable = [node for node in reachable if node != q] # exclude q 

            if not reachable:
                #print("Initializing idle step - No reachable qubits from ", q)
                continue
            

            terminal = random.choice(reachable)
            path_idle = nx.dijkstra_path(g_temp, q, terminal)

            idle_move_dct[("idle", q, terminal)] = (path_idle, None)
            
            # Prevent other idle moves from overlapping this one.
            used_idle_paths.update(path_idle)

        return idle_move_dct

    
    def perturbation(self, teleport_dct: dict, radius: int, vdp_dict: dict):
        """
        Computes a perturbation of a given collection of paths within a given radius of edges around the current terminal.

        For each path a new location of the 2nd (3rd terminal) is updated randomly.
        """
        def occupied_nodes_for_others(key, path_pair):
            """
            collect the occupied nodes except logical data qubits
            """
            path1, path2 = path_pair
            nodes = []

            if key[0] == "idle":
                if path1 is not None:
                    nodes += path1[1:]      # exclude idle source q
                return nodes

            # CNOT or T teleport
            if path1 is not None:
                nodes += path1[1:-1]        # exclude logical endpoints
            if path2 is not None:
                nodes += path2              # branch path

            return nodes
        
        if self.logical_pos_temp is None:
            raise RuntimeError(
                "Need to initialize logical pos temp properly in a summarizing method."
            )   

        teleport_dct_update = teleport_dct.copy()

        if not teleport_dct_update:
            g_tt = self.g.copy()
        # used_nodes = set()
        new_terminal = None

        # find perturbation of each item in teleport_dct 
        for key_tree, (path1, path2) in teleport_dct.items():
            
            # each key got their own graph 
            g_temp = self.g.copy()
            g_temp.remove_nodes_from(self.factory_pos)
            # determine whether tree corresponds to an idle, a CNOT or T gate
            if key_tree[0] == "idle":
                _, q, terminal = key_tree
                g_temp.remove_nodes_from([x for x in self.logical_pos_temp if x != q])
                # exclude nodes from other steiner or idle teleportation paths
                other_paths = [
                    pos
                    for keyy, path_pair in teleport_dct_update.items()
                    if keyy != key_tree
                    for pos in occupied_nodes_for_others(keyy,path_pair)
                ]
                
            elif len(key_tree) == 3:
                (a, b, terminal) = key_tree
                g_temp.remove_nodes_from([x for x in self.logical_pos_temp])
                other_paths = [
                    pos
                    for keyy, path_pair in teleport_dct_update.items()
                    if keyy != key_tree
                    for pos in occupied_nodes_for_others(keyy, path_pair)
                ]
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                g_temp.remove_nodes_from([x for x in self.logical_pos_temp])
                other_paths = [
                    pos
                    for keyy, path_pair in teleport_dct_update.items()
                    if keyy != key_tree
                    for pos in occupied_nodes_for_others(keyy, path_pair)
                ]
            else:
                raise RuntimeError(
                    "Something is wrong with the allocation of keys in the steiner_dict"
                )

            # a terminal can be placed on the path. in this case you are NOT allowed to remove it! the above somehow sometimes add terminal, hence remove it again
            #if terminal in other_paths:
            #    other_paths.remove(terminal)
            #if path2 and path2[0] in other_paths:
            #    other_paths.remove(path2[0])

            protected = {terminal}
            if path2:
                protected.add(path2[0])
            other_paths = [node for node in other_paths if node not in protected]

            #g_temp_temp = g_temp.copy()
            g_temp.remove_nodes_from(other_paths)

            
            # remove nodes from vdp dict (the tree is not allowed to be on or cross another path)
            for path_label, path in vdp_dict.items():
                if isinstance(path_label, tuple) and path_label[0] == "idle_back":
                    nodes_to_delete = path[1:]  # for idle move you need to delete more
                elif isinstance(path_label[0], tuple):  # cnot
                    nodes_to_delete = path[1:-1]
                elif isinstance(path_label[0], int):  # t
                    nodes_to_delete = path[1:]
                for node in nodes_to_delete:
                    if path2:
                        path_con = path1 + path2
                    else:
                        path_con = path1
                    if (
                        node in g_temp.nodes() and node not in path_con
                    ):  # {terminal, path2[0]}
                        g_temp.remove_node(node)

            # find "neighborhood" of the terminal
            neighborhood = set(
                nx.single_source_shortest_path_length(
                    g_temp, terminal, cutoff=radius
                ).keys()
            )
            neighborhood = sorted(neighborhood)
            #print("neighborhood", neighborhood)
            
            #print("g_temp", g_temp)
            
            # the single source shortest path,... ensures that only reachable nodes are included
            # choose one of them
            if len(neighborhood) == 1:  # if only one neighbor, i.e. the terminal itself
                # new_terminal = None #to skip the updating of the root node below.
                continue
            if key_tree[0] == "idle":
                path_terminal = None
                while True:
                    new_terminal = random.choice(list(neighborhood))
                    if new_terminal == terminal:  # do not want same terminal again
                        continue
                    try:
                        path_terminal = nx.dijkstra_path(
                            g_temp, q, new_terminal
                        )  
                    except nx.NetworkXNoPath:
                        warnings.warn(
                            "If this is called you need to check why this is happening."
                        )
                    if path_terminal:
                        break
            else:
                path_terminal = None
                while True:
                    new_terminal = random.choice(list(neighborhood))
                    if new_terminal == terminal:  # do not want same terminal again
                        continue
                    try:
                        #print("new terminal: ", new_terminal)
                        path_terminal = nx.dijkstra_path(g_temp, path2[0], new_terminal)  # path2[0] is the connecting node on the path
                    except nx.NetworkXNoPath:
                        warnings.warn(
                            "If this is called you need to check why this is happening."
                        )
                    if path_terminal:
                        break

                #!TODO should i skip this since we do it globally afterwards again?
                # (A) loop to possibly find shorter path_terminal
                paths_lst_temp = []  # collect all paths from path1[1:-1] to new_terminal
                for node_on_path in path1[1:-1]:
                    try:
                        path_temp = nx.dijkstra_path(
                            g_temp, node_on_path, new_terminal
                        )
                        paths_lst_temp.append(path_temp)
                    except nx.NetworkXNoPath:
                        pass
                if paths_lst_temp:
                    path_terminal = min(paths_lst_temp, key=len)

            # delete old entry and add new with updated key
            teleport_dct_update.pop(key_tree, None)
            if key_tree[0] == "idle":
                _, q, terminal = key_tree
                new_key_tree = ("idle", q, new_terminal)
                teleport_dct_update[new_key_tree] = (path_terminal, None)
            elif len(key_tree) == 3:
                (a, b, terminal) = key_tree
                new_key_tree = (a, b, new_terminal)
                teleport_dct_update[new_key_tree] = (path1, path_terminal)
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                new_key_tree = (a, new_terminal)
                teleport_dct_update[new_key_tree] = (path1, path_terminal)
            # remove_branch_nodes += path_terminal

        # it is possible that (A) does not capture everything, as the terminal path may change in a later iteration and thus make even shorter paths possible.
        if (
            new_terminal is not None
        ):  # if the neighborhood has 1 item only, the above breaks. then we do not want to do this reduction.
            teleport_dct_update_second = teleport_dct_update.copy()
            for key_tree, (path1, path2) in teleport_dct_update.items():
                g_tt = self.g.copy()
                g_tt.remove_nodes_from(self.factory_pos)
                if key_tree[0] =="idle":
                    _, q, terminal = key_tree
                    g_tt.remove_nodes_from([x for x in self.logical_pos_temp if x != q])
                elif len(key_tree) == 3:
                    (a, b, terminal) = key_tree
                    g_tt.remove_nodes_from([x for x in self.logical_pos_temp])
                elif len(key_tree) == 2:
                    (a, terminal) = key_tree
                    g_tt.remove_nodes_from([x for x in self.logical_pos_temp])
                else:
                    raise ValueError("steiner dct keys are wrong.")
                other_paths = [
                    pos
                    for keyy, path_pair in teleport_dct_update_second.items()
                    if keyy != key_tree
                    for pos in occupied_nodes_for_others(keyy, path_pair)
                ]
                #if terminal in other_paths:
                #    other_paths.remove(terminal)
                #if path2 and path2[0] in other_paths:
                #    other_paths.remove(path2[0])
                #if terminal in other_paths:
                #    other_paths.remove(terminal)
                protected = {terminal}
                if path2:
                    protected.add(path2[0])
                other_paths = [node for node in other_paths if node not in protected]

                #g_temp_temp = g_temp.copy()
                g_tt.remove_nodes_from(other_paths)
                for path_label, path in vdp_dict.items():
                    if isinstance(path_label, tuple) and path_label[0] == "idle_back":
                        nodes_to_delete = path[1:]  
                    else:
                        nodes_to_delete = path[1:-1]
                    for node in nodes_to_delete:
                        if path2:
                            path_con = path1 + path2
                        else:
                            path_con = path1
                        if (
                            node in g_tt.nodes() and node not in path_con
                        ):  # {terminal, path2[0]}:
                            g_tt.remove_node(node)
                paths_lst_temp = (
                    []
                )  # collect all paths from path1[1:-1] to new_terminal
                if key_tree[0] == "idle":                             
                    try:
                        path_temp = nx.dijkstra_path(
                            g_tt, q, terminal
                        )
                        paths_lst_temp.append(path_temp)
                    except nx.NetworkXNoPath:
                        pass
                else:
                    for node_on_path in path1[1:-1]:
                        try:
                            path_temp = nx.dijkstra_path(
                                g_tt, node_on_path, terminal
                            )
                            paths_lst_temp.append(path_temp)
                        except nx.NetworkXNoPath:
                            pass
                if paths_lst_temp:
                    path_terminal = min(paths_lst_temp, key=len)
                    teleport_dct_update_second.pop(key_tree, None)
                    if key_tree[0] == "idle":
                        _, q, terminal = key_tree
                        new_key_tree = ("idle", q, terminal)
                        teleport_dct_update_second[new_key_tree] = (path_terminal, None)
                    elif len(key_tree) == 3:
                        (a, b, terminal) = key_tree
                        new_key_tree = (a, b, terminal)
                        teleport_dct_update_second[new_key_tree] = (path1, path_terminal)
                    elif len(key_tree) == 2:
                        (a, terminal) = key_tree
                        new_key_tree = (a, terminal)
                        teleport_dct_update_second[new_key_tree] = (path1, path_terminal)
        else:
            teleport_dct_update_second = teleport_dct_update
            g_tt = self.g.copy()
            g_tt.remove_nodes_from(self.factory_pos)

        #print("teleport dct after perturbation: ", teleport_dct_update_second)
        return teleport_dct_update_second, g_tt
    @staticmethod
    def replace_pos(lst: list[tuple[pos, pos] | pos], old: pos, new: pos):
        """
        Helper function to replace pos values in lists.

        This is needed to update logical_pos etc. during the optimization.
        """
        result = []
        for item in lst:
            if isinstance(item[0], int):
                result.append(new if item == old else item)
            else:
                result.append(tuple(new if sub == old else sub for sub in item))
        return result
    def update_layers(self, layers, old_pos, new_pos):
        """
        update old_pos to new_pos in layers 
        """
        result = []
        for layer in layers:  
            result.append( self.replace_pos(layer, old_pos, new_pos) )
        return result

    def calculate_cost(self, metric, layers, logical_pos, factory_times, layout):
        cost = 0
        if metric == "crossing":
                cost = self.count_crossings(
                    layers, logical_pos
                )  # overwrite this in the upcoming loop
        elif metric == "exact":
            schedule, _ = self.find_total_vdp_layers_dyn(
                layers,
                logical_pos,
                factory_times,
                layout,
            )  # initially the self.logical pos can be used. later you need a logical_pos outside of self
            if schedule is not None:
                cost = len(schedule)
            else:
                cost = lock_penalty
        else:
            raise NotImplementedError(
                "Other metrics than crossing and exact not implemented yet."
            )
        return cost, schedule
    
    
    
    def run_annealing(
        self,
        next_layers,
        init_steiner_dct: dict,
        init_idle_dct: dict,
        max_iters: int,
        T_start: float,
        T_end: float,
        alpha: int,
        k_lookahead: int,
        radius: int,
        vdp_dict,
        layout,
        include_steiner_teleport: bool = True,
        include_idle_teleport: bool = False, 
    ):  # , danger_qubits: dict, available_gaps: list):
        """
        Plug together all previous methods to run annealing for k future layers.

        Args:
            next_layers:
            init_steiner_dct:
            init_idle_dct:
            max_iters:
            T_start:
            T_end:
            alpha:
            k_lookahead:
            radius:
            vdp_dict:
            layout:
            include_steiner_teleport: 
            include_idle_teleport:

        Returns:
            best_steiner
            best_idle
            best_cost
            best_schedule
            cost_history
            best_move_type_lst
            teleport_history
            graph_history

        """
        
    
        if include_steiner_teleport:
            steiner_dct = copy.deepcopy(init_steiner_dct)
        else:
            steiner_dct = {}

        if include_idle_teleport:
            idle_move_dct = copy.deepcopy(init_idle_dct)
        else:
            idle_move_dct = {}

        teleport_dct = {**steiner_dct, **idle_move_dct}
        #!TODO INCLUDE IDLE MOVING GAPS AS PART OF THE ANNEALING TO AVOID SEQUENTIALIZATION
        
        if T_start < T_end:
            raise ValueError("T_start must be larger than T_end")
        if alpha >= 1.0 or alpha <= 0:
            raise ValueError("alpha must be between 0 and 1")

        self.logical_pos_temp = self.logical_pos.copy()
        factory_times_copy = self.factory_times.copy()
        cost, schedule = self.calculate_cost(self.metric, next_layers[:k_lookahead], self.logical_pos_temp, factory_times_copy, layout)
        
        best_teleport = teleport_dct
        best_cost = cost
        best_move_type_lst = {}
        best_layout = layout.copy()
        if self.metric == "exact" and schedule is not None:
            best_schedule = schedule.copy()
        else:
            best_schedule = None
        #! adapt the logical positions etc
        cost_history = [cost]  # add initial cost to cost history
        teleport_history = []
        graph_history = []

        

        T = T_start
        for step in range(max_iters):
            candidate, g_temp_temp = self.perturbation(teleport_dct, radius, vdp_dict)
            graph_history.append(g_temp_temp)

            #!NOT NEEDED ANYMORE
            if candidate is None:  # i.e. if no other element could be found
                warnings.warn(
                    "No new neighborhood could be explored. Either you are stuck in a local minimum or simply used to manny iters"
                )
                break  # early break

            # after the perturbation you have to update the logical pos, otherwise you will run into issues
            next_layers_copy = copy.deepcopy(next_layers)
            move_type_lst_temp = {}
            # compute the cost of the candidate
            # 1. change the position of the target/control to the ancilla spot for all paths (adapt next_layer)
            logical_pos_temp = self.logical_pos_temp.copy()
            layout_rev = {j: i for i, j in layout.items()}
            layout_mod = layout.copy()
            for key_candidate, (path1, path2) in candidate.items():

                is_idle = (isinstance(key_candidate, tuple) and len(key_candidate) == 3 and key_candidate[0] == "idle")
                # ----------------------------------------------------
                # Idle teleport:
                # key = ("idle", q, terminal)
                # path = p1 = q -> terminal
                # ----------------------------------------------------
                if is_idle:
                    move_type = "idle"
                    _, q, terminal = key_candidate
                    next_layers_copy = self.update_layers(next_layers_copy, q, terminal)
                    logical_pos_temp = self.replace_pos(logical_pos_temp, q, terminal)
                    label = layout_rev[q]
                    layout_mod[label] = terminal
                    move_type_lst_temp.update({("idle", q, terminal): move_type})

                elif len(key_candidate) == 3:
                    (a, b, terminal) = key_candidate
                    # randomly choose whether we shift control to ancilla or target to ancilla
                    move_type = random.choice(["target", "control"])
                    move_type_lst_temp.update({(a, b, terminal): move_type})
                    old_pos = None
                    new_pos = None
                    if move_type == "target":
                        old_pos = b
                        new_pos = terminal
                    else:
                        old_pos = a
                        new_pos = terminal

                    next_layers_copy = self.update_layers(next_layers_copy, old_pos, new_pos)
                    logical_pos_temp = self.replace_pos(logical_pos_temp, old_pos, new_pos)

                    label = layout_rev[old_pos]
                    layout_mod[label] = new_pos
                elif len(key_candidate) == 2:
                    (a, terminal) = key_candidate
                    # move type is fix, one can only move the "a"
                    move_type = "singlequbit"
                    move_type_lst_temp.update({(a, terminal): move_type})
                    # update layers
                    next_layers_copy = self.update_layers(next_layers_copy, a, terminal)
                    logical_pos_temp = self.replace_pos(logical_pos_temp, a, terminal)
                    label = layout_rev[a]
                    layout_mod[label] = terminal
                else:
                    raise RuntimeError("Something wrong with keys of candidate tree")
            # 2. compute the crossing metric for next_layer
            # try:
            layers_for_metric = next_layers_copy[:k_lookahead]
            candidate_cost, schedule = self.calculate_cost(self.metric, layers_for_metric, logical_pos_temp, factory_times_copy, layout_mod)
            #print("move type list temp", move_type_lst_temp)

            # except ValueError:
            #    continue #skip if some config locks the qubits such that you cannot even evaluate count crossings
            delta = candidate_cost - cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                teleport_dct, cost = candidate, candidate_cost
                if cost < best_cost:  # update the best cost
                    best_teleport, best_cost = teleport_dct.copy(), cost
                    best_move_type_lst = move_type_lst_temp.copy()
                    best_schedule = schedule.copy()
                    best_layout = layout_mod.copy()
                cost_history.append((cost, layers_for_metric))
            teleport_history.append(candidate)
            # cool
            T = max(T_end, T * alpha)
            
            

        # if there is no improvement possible at all, make sure you return a none best teleport
        if len(best_move_type_lst) == 0:
            best_teleport = None
            logger.info("No Teleportation improvement possible in this layer.")
        else:
            logger.info("Teleportation found for this layer.")

        logger.info("Final Temperature T = %.6e", T)

        best_steiner, best_idle = self.split_teleport_dct(best_teleport)

        return (
            best_steiner,
            best_idle, 
            best_cost,
            best_schedule,
            cost_history,
            best_move_type_lst,
            teleport_history,
            graph_history,
        )

    def reduce_teleport_moves(
        self, steiner_dct, idle_dct, move_type_lst, next_layers, best_cost, k_lookahead, layout
    ):
        """
        Given the current dictionaries, make sure that you effectively use as least movements as possible.

        This means, we go through small subsets of the tree solution and check at what point we reach the same optimized cost.
        This can scale horribly if you have too many dicts in your solution.
        """
        factory_times_copy = self.factory_times.copy()
        flag = False
        best_dct_temp = None
        best_schedule = None

        teleport_dct = {**steiner_dct, **idle_dct}
        

        for r in range(1, len(teleport_dct) + 1):
            for subset in itertools.combinations(teleport_dct.items(), r):
                # translate everything into the setup of the steiner_dct movement
                logical_pos_temp = self.logical_pos_temp.copy()
                # dct_temp = {(a,b,terminal): (path1, path2) for (a,b,terminal), (path1, path2) in subset}
                dct_temp = {
                    key: (path1, path2) for key, (path1, path2) in subset
                }  # allows different types of keys
                next_layers_copy = next_layers.copy()
                layout_mod = layout.copy()
                layout_rev = {j: i for i, j in layout.items()}
                for key_subset, (path1, path2) in dct_temp.items():
                    if key_subset[0] == "idle":
                        _, q, terminal = key_subset
                    elif len(key_subset) == 3:
                        (a, b, terminal) = key_subset
                    elif len(key_subset) == 2:
                        (a, terminal) = key_subset
                    else:
                        raise RuntimeError("something wrong with subset steiner keys")

                    # randomly choose whether we shift control to ancilla or target to ancilla
                    move_type = move_type_lst[key_subset]
                    if move_type == "idle":
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(
                                next_layer, q, terminal
                            )
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(
                            logical_pos_temp, q, terminal
                        )
                        label = layout_rev[q]
                        layout_mod[label] = terminal
                    elif move_type == "target":
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(
                                next_layer, b, terminal
                            )
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(
                            logical_pos_temp, b, terminal
                        )
                        label = layout_rev[b]
                        layout_mod[label] = terminal
                    elif (
                        move_type == "control" or move_type == "singlequbit"
                    ):  # we denote the control a and the singlequibt a, so this can be summarized
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(
                                next_layer, a, terminal
                            )
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(
                            logical_pos_temp, a, terminal
                        )
                        label = layout_rev[a]
                        layout_mod[label] = terminal
                    else:
                        raise RuntimeError(
                            f"other move type than expected: {move_type}"
                        )
                # 2. compute the crossing metric for next_layer
                try:
                    if self.metric == "crossing":
                        candidate_cost = self.count_crossings(
                            next_layers_copy[:k_lookahead], logical_pos_temp
                        )
                    elif self.metric == "exact":
                        schedule, _ = self.find_total_vdp_layers_dyn(
                            next_layers_copy[:k_lookahead],
                            logical_pos_temp,
                            factory_times_copy,
                            layout_mod,
                        )
                        if schedule is not None:
                            candidate_cost = len(schedule)
                        else:
                            candidate_cost = lock_penalty
                    else:
                        raise NotImplementedError(
                            "Other metrics than crossing and exact not implemented yet."
                        )
                except ValueError:
                    continue

                if candidate_cost == best_cost:
                    flag = True
                    best_dct_temp = dct_temp
                    best_schedule = schedule.copy() if self.metric == "exact" else None
                    break
            if flag:
                break

        if flag:
            steiner_reduced, idle_reduced = self.split_teleport_dct(best_dct_temp) # {(a,b,terminal): (path1, path2) for (a,b,terminal), (path1, path2) in subset}
            # move_type_lst_red = {(a,b,terminal): move_type_lst[(a,b,terminal)] for (a,b,terminal), (_, _) in best_dct_temp.items()}
            move_type_lst_red = {
                key: move_type_lst[key] for key, (_, _) in best_dct_temp.items()
            }
            final_schedule = best_schedule.copy()
        else:  # return the inputs if nothing is found
            steiner_reduced = steiner_dct
            idle_reduced = idle_dct
            move_type_lst_red = move_type_lst
            final_schedule = None
        return steiner_reduced, idle_reduced, move_type_lst_red, final_schedule

    def idle_move_back(
        self,
        schedule,
        danger_qubits,
        available_gaps,
        danger_qubits_temp,
        available_gaps_temp,
        layout,
        layers,
        reduce_time_stamp: bool,
        jump_harvesting: bool,
        best_schedule: None | list,
    ):
        """
        Subroutine of `optimize_layers` to move back qubits in dangerous positions asap.
        """
        # instead of adding another schedule_temp, take schedule[-1], adapt it and replace
        schedule_temp = schedule[-1].copy()
        flag_idle_move = False

        # distinguish between the cases with jump_harvest = True and False. If true, you need to check more than schedule[-1] but also the future layers from "best_schedule" which was retrieved during SA.
        # we want to avoid moves if the qubits are included in the k_lookahead layers since we want to guarantee that the routing computed for the metric in SA is the same as effectively used in k_lookahead layers to exploit the SA optimization fully without destroying stuff
        if jump_harvesting and best_schedule is None:
            raise ValueError(
                "If `jump_harvest = True` you need to give a best_schedule as input for idle_move_back!"
            )

        #!Try to move re-allocated qubits back into the left gaps (does not need to be the original position, in case some other gap is closer)
        #!TODO priority ordering of the danger_qubits (those which appear earlier in upcoming layers must be attempted to be moved back first)
        danger_qubits_copy = danger_qubits.copy()
        logical_pos_temp = list(layout.values())
        next_layers_copy = layers.copy()
        layout_rev = {j: i for i, j in layout.items()}
        layout_mod = layout.copy()
        # filter the danger_qubits to those which are idling right now? to avoid useless runs
        flattened_vdp_dict_current = [
            item for pair in schedule_temp["vdp_dict"].keys() for item in pair
        ]
        if (
            jump_harvesting
        ):  # include the (remaining) k_lookahead layers for the current jump because we do not want to alter the stuff from SA for multiple k_lookahead
            for layer in best_schedule:
                for key in layer.keys():
                    flattened_vdp_dict_current.append(key[0])
                    flattened_vdp_dict_current.append(key[1])

        danger_qubits_idling = {
            qubit: time
            for (qubit, time) in danger_qubits_copy.items()
            if qubit not in flattened_vdp_dict_current
        }

        danger_qubits_idling = {
            qubit: time for (qubit, time) in danger_qubits_idling.items() if time <= 0
        }
        idle_move_labels = []
        vdp_dict = schedule_temp["vdp_dict"]
        for danger_qubit in danger_qubits_idling.keys():
            # TODO order the available gaps regarding how close they are to the current danger_qubit
            # go through gaps and take the one to which a path is available
            path_idle = None
            for (
                gap
            ) in available_gaps:  #!TODO order available_gaps according to distance

                # skip the gap if it is currently occupied by some path
                flag_skip = False

                # create a graph copy where all already occupied ancillas from vdp_dict are removed
                g_copy = self.g.copy()
                initial_nodes = set(g_copy.nodes())
                if schedule_temp["steiner"] is not None:
                    for steiner in schedule_temp["steiner"].values():
                        for node in steiner[0]:
                            if node == gap:
                                flag_skip = True
                            if node != danger_qubit:  # and node != gap:
                                g_copy.remove_node(node)
                        for node in steiner[1]:
                            if node == gap:
                                flag_skip = True
                            if (
                                node in g_copy.nodes() and node != danger_qubit
                            ):  # and node != gap: #at least one node in steiner[1] is already in steiner[0]
                                g_copy.remove_node(node)
                for path in schedule_temp["vdp_dict"].values():
                    for node in path:
                        if node == gap:
                            flag_skip = True
                        if (
                            node in g_copy.nodes() and node != danger_qubit
                        ):  # and node != gap:
                            g_copy.remove_node(node)

                if schedule_temp["idle_teleport"]:
                    for path, _ in schedule_temp["idle_teleport"].values():
                        for node in path:
                            if node == gap:
                                flag_skip = True
                            if (
                                node in g_copy.nodes() and node != danger_qubit
                            ):  # and node != gap:
                                g_copy.remove_node(node)

                for pos in layout_mod.values():
                    if (
                        pos in g_copy.nodes() and pos != danger_qubit and pos != gap
                    ):  #!just in case you find a bug, this was node != gap before, i dont know why this worked before i moved this into an own method
                        g_copy.remove_node(pos)
                final_nodes = set(g_copy.nodes())
                removed_nodes = initial_nodes - final_nodes

                if flag_skip:
                    continue

                path_idle = None
                try:
                    path_idle = nx.dijkstra_path(g_copy, gap, danger_qubit)
                    #path_idle = nx.dijkstra_path(g_copy, danger_qubit, gap)
                    # danger_qubits_copy.remove(danger_qubit)
                    del danger_qubits_copy[danger_qubit]
                    available_gaps.remove(gap)
                    label_idle = ("idle_back", danger_qubit, gap)
                    #label_idle = f"idle_{danger_qubit}_to_{gap}"
                    vdp_dict.update({label_idle: path_idle}) 
                    idle_move_labels.append(label_idle)
                    logical_pos_temp = self.replace_pos(
                        logical_pos_temp, danger_qubit, gap
                    )
                    for j, next_layer in enumerate(
                        next_layers_copy
                    ):  # update all future layers
                        next_layers_copy[j] = self.replace_pos(
                            next_layer, danger_qubit, gap
                        )
                    label = layout_rev[danger_qubit]
                    layout_mod[label] = gap
                    layout_rev = {
                        j: i for i, j in layout_mod.items()
                    }  #!update layout_rev
                    flag_idle_move = True
                    break  # because if path found you do not want to find another path to the same danger qubit
                except nx.NetworkXNoPath:
                    continue
            if path_idle is None:
                logger.info(
                    f"No idling path back could be found for danger qubit {danger_qubit}"
                )

        # reduce the time stamp of danger_qubits_copy by 1
        if reduce_time_stamp:
            danger_qubits_copy = {
                qubit: time - 1 for (qubit, time) in danger_qubits_copy.items()
            }

        danger_qubits = danger_qubits_copy.copy()
        # add those danger qubits and empty spots which where added in this layer
        # danger_qubits += danger_qubits_temp #those temp danger qubits have k_lookahead as timelabel
        danger_qubits |= danger_qubits_temp
        available_gaps += available_gaps_temp
        
        # if an element appears both in danger_qubits and available_gaps, this hints to the case that a qubit in a danger position was moved again. hence delete the double elements
        shared_elements = set(danger_qubits.keys()) & set(available_gaps)
        danger_qubits = {
            qubit: time
            for (qubit, time) in danger_qubits.items()
            if qubit not in shared_elements
        }
        available_gaps = [x for x in available_gaps if x not in shared_elements]

        #!update terminal_pairs, logical_pos etc
        schedule_temp["vdp_dict"] = vdp_dict
        self.logical_pos = logical_pos_temp.copy()
        self.logical_pos_temp = logical_pos_temp.copy()
        schedule_temp["logical_pos"] = self.logical_pos.copy()
        schedule_temp["idle_move_label"] = idle_move_labels.copy()
        # should I update schedule_temp["layout"] here? 
        layers = next_layers_copy.copy()
        layout = layout_mod.copy()

        if flag_idle_move:
            schedule[-1] = schedule_temp.copy()

        return schedule, danger_qubits, available_gaps, layout, layers

    def optimize_layers(
        self,
        terminal_pairs,
        layout,
        max_iters,
        T_start,
        T_end,
        alpha,
        radius,
        k_lookahead,
        max_idle_teleport, 
        steiner_init_type,
        jump_harvesting: str,
        reduce_teleport: bool,
        idle_move_type: str,
        filename: str,
        include_steiner_teleport: bool = True,
        include_idle_teleport: bool= False,
        reduce_init_steiner: bool = False,
        reduce_init_idle: bool = False, 
        stimtest: bool = False,
    ):
        """
        Optimize the positions in batches of size k_lookahead.
        This means we do the following:
        1. find an initial layer structure #!TODO copy from old code split layer
        2. route the first layer, and push remainder into next layers
        3. run SA with k_lookahead layers (i.e. k layers are counted in the metric)
        4. move idling qubits back (in different points in time depending on `idle_move_type.`)
        Repeat this layer by layer.

        `jump_harvesting`: bool
            False: Default sliding window method.
            True: 1 layer is used for steiner search of k future layers. but then you do not just iterate through EACH layer but you skip the k layers, because you do not want to destroy the optimization by optimizing too much!
        `idle_move_type`: str
            asap: moving back is done as frequent as possible. this may destroy however structure of the predicted routing from the steiner search.
            later: means that moving back is only done when the steiner search is done. if a locking occurs, extra layers with moving back are necessary, but no moving back during the routing of the k_lookahead layers with jump_harvesting = True.
        """

        if idle_move_type not in {"asap", "later"}:
            raise ValueError("`move_idle_type` must be `asap` or `later`")

        schedule = []
        

        available_gaps = []  # a list of gap positions which are free due to moves
        danger_qubits = (
            {}
        )  # a dict of current dangerous qubits and time label. after k layers they should be moved back again.
        improvement_history = []

        # initialize a layering, but this will by dynamically adapted via pushing
        if self.use_dag:
            dag = dag_helper.terminal_pairs_into_dag(terminal_pairs, layout)
            layers = []
            for layer in range(len(list(dag.layers()))):
                layers.append(dag_helper.extract_layer_from_dag(dag, layout, layer))
        else:
            layers = self.split_layer_terminal_pairs(terminal_pairs)
        if len(layers) == 1:
            raise RuntimeError(
                "Your choice of terminal pairs does not lead to multiple layers. This method is not suitable for your input."
            )
        # if some qubit is locked due to replacement, you will end up in an infinite loop here. make sure this does not happen
        no_progress_counter = 0
        it = -1
        while len(layers) != 0:

            flag_skip_teleport = False
            it += 1
            schedule_temp = {
                "steiner": None,
                "idle_teleport": None,
                "vdp_dict": None,
                "move_type": None,
                "logical_pos": None,
                "layout": None,
                "cost_history": None,
                "idle_move_label": None,
            }
            schedule_temp_for_later = schedule_temp.copy()
            # find vdp solution for the front layer (we adapt layers dynamically, meaning we delete stuff which is already routed)
            vdp_dict, terminal_pairs_remainder, self.factory_times = (
                self.find_max_vdp_set(layers[0], None, self.factory_times)
            )
            logger.info(
                "Iteration %d: |vdp_dict|=%d, pushing |terminal_pairs_remainder|=%d, remaining |layers|=%d",
                it,
                len(vdp_dict),
                len(terminal_pairs_remainder),
                len(layers),
            )
            if len(vdp_dict) == 0:
                no_progress_counter += 1
            if no_progress_counter > self.t:
                warnings.warn(
                    "For more than t (=reset time) layers you could not route anything. Most likely a qubit you need got locked in."
                )
                return schedule
            if len(layers) == 1 and len(terminal_pairs_remainder) == 0:
                # no more steiner tree needed
                schedule_temp["vdp_dict"] = vdp_dict
                schedule_temp["logical_pos"] = self.logical_pos.copy()
                schedule_temp["layout"] = layout.copy()
                schedule.append(schedule_temp)
                with open(filename, "wb") as f:
                    pickle.dump(schedule, f)
                break
            # remove what is already routed from `layers` + push the remainder into the next layers
            if self.use_dag:
                layers, dag = dag_helper.push_remainder_into_layers_dag(
                    dag, terminal_pairs_remainder, layout, layers[0]
                )
            else:
                layers = self.push_remainder_into_layers(
                    layers, terminal_pairs_remainder
                )

            # update factory times
            for key in self.factory_times:
                if self.factory_times[key] != 0:
                    self.factory_times[key] -= 1

            if idle_move_type == "later":
                # this does idle moves BEFORE we do the steiner search, i.e. later the metric and real routing will coincide.
                # if vdp_dict is empty, this also adds layers of pure idle moves to avoid getting stuck.
                reduce_time_stamp = True
                schedule_temp["vdp_dict"] = vdp_dict

                # if an element appears both in danger_qubits and available_gaps, this hints to the case that a qubit in a danger position was moved again. hence delete the double elemts
                # this is done in idle_move_back, but needs to be done beforehand too
                shared_elements = set(danger_qubits.keys()) & set(available_gaps)
                danger_qubits = {
                    qubit: time
                    for (qubit, time) in danger_qubits.items()
                    if qubit not in shared_elements
                }
                available_gaps = [x for x in available_gaps if x not in shared_elements]

                # do not overwrite schedule, because this idle move back is too early to easily directly overwrite schedule
                # extract important info from schedule_temp_temp into schedule_temp
                schedule_temp_temp, danger_qubits, available_gaps, layout, layers = (
                    self.idle_move_back(
                        [schedule_temp],
                        danger_qubits,
                        available_gaps,
                        {},
                        [],
                        layout,
                        layers,
                        reduce_time_stamp,
                        False,
                        None,
                    )
                )
                vdp_dict = schedule_temp_temp[-1]["vdp_dict"]
                assert (
                    len(schedule_temp_temp) == 1
                ), "internal error if this is not right"
                schedule_temp = schedule_temp_temp[-1]

            # find optimal steiner tree(s) for current layer
            
            steiner_dct = {}
            idle_move_dct = {}


            layers_steiner = None
            k_steiner = None 
            layers_idle = None
            k_idle = None

            if reduce_init_steiner:
                layers_steiner = layers
                k_steiner = k_lookahead             
            if reduce_init_idle:
                layers_idle = layers
                k_idle = k_lookahead 
            

            if include_steiner_teleport:
                steiner_dct = self.initialize_steiner(
                    vdp_dict, steiner_init_type, layers=layers_steiner, k_lookahead=k_steiner
                )
                if include_idle_teleport:
                    idle_move_dct = self.initialize_idle_moves(
                        vdp_dict, steiner_dct, max_idle_teleport, layers = layers_idle, k_lookahead = k_idle
                    )

            else:
                if include_idle_teleport:
                    idle_move_dct = self.initialize_idle_moves(
                        vdp_dict, steiner_dct, max_idle_teleport, layers = layers_idle, k_lookahead = k_idle
                    )
                    #print("idle_move_dct: ", idle_move_dct)


            if len(steiner_dct) == 0 and len(idle_move_dct) == 0:
                logger.info("No Steiner and idle teleportation initialized")
                best_steiner_init = None
                best_idle_init = None

            else:
                (
                    best_steiner_init,
                    best_idle_init,
                    best_cost,
                    best_schedule,
                    cost_history,
                    move_type_lst,
                    steiner_history,
                    graph_history,
                ) = self.run_annealing(
                    layers,
                    steiner_dct,
                    idle_move_dct,
                    max_iters,
                    T_start,
                    T_end,
                    alpha,
                    k_lookahead,
                    radius=radius,
                    vdp_dict=schedule_temp["vdp_dict"],
                    layout=layout,
                    include_steiner_teleport=include_steiner_teleport,
                    include_idle_teleport=include_idle_teleport, 
                )
                improvement_history.append((best_cost, cost_history[0]))

            print("best_steiner_init: ", best_steiner_init)
            print("best_idle_init: ", best_idle_init)
            if best_idle_init:
                print("len of best idle init", len(best_idle_init))

            # do not use a steiner if the SA could not find a good best_steiner. then it is set to none
            if not best_steiner_init and not best_idle_init:  # break earlier, similar to above
                schedule_temp["vdp_dict"] = vdp_dict
                schedule_temp["logical_pos"] = self.logical_pos.copy()
                schedule_temp["layout"] = layout.copy()
                schedule.append(schedule_temp)
                with open(filename, "wb") as f:
                    pickle.dump(schedule, f)
                flag_skip_teleport = True

            if flag_skip_teleport is False:
                if reduce_teleport:
                    best_steiner, best_idle, move_type_lst, best_schedule_temp = (
                        self.reduce_teleport_moves(
                            best_steiner_init,
                            best_idle_init,
                            move_type_lst,
                            layers,
                            best_cost,
                            k_lookahead,
                            layout,
                        )
                    )
                    if len(best_steiner) + len(best_idle) < len(best_steiner_init) + len(best_idle_init):
                        logger.info("Complexity of teleportation could be reduced.")
                    best_schedule = best_schedule_temp.copy()
                    print("best steiner after reduce: ", best_steiner)
                    print("best idle after reduce: ", best_idle)
                    print("len of best idle after reduce: ", len(best_idle))

                else:
                    best_steiner = best_steiner_init  # only rename
                    best_idle = best_idle_init

                # update the logical pos etc for the next iteration
                logical_pos_temp = self.logical_pos_temp.copy()
                layout_rev = {j: i for i, j in layout.items()}
                layout_mod = layout.copy()
                schedule_temp["layout"] = (
                    layout.copy()
                )  # add current layout, the adapted layout is for the next iteration.
                next_layers_copy = layers.copy()
                available_gaps_temp = (
                    []
                )  # need temporary list, because you do not want to add it to the log already since you cannot move those newly added danger qubits in this current layer, you need to do it in the next
                danger_qubits_temp = {}
                for key_tree, (path1, path2) in best_steiner.items():
                    if len(key_tree) == 3:
                        (a, b, terminal) = key_tree
                    elif len(key_tree) == 2:
                        (a, terminal) = key_tree
                    else:
                        raise RuntimeError("something wrong with keys of best_steiner")

                    # update the available_gaps and danger_qubits (add and remove accordingly later)
                    move_type = move_type_lst[key_tree]
                    if (
                        terminal in available_gaps
                    ):  # if terminal is in available_gaps you need to remove it
                        available_gaps.remove(terminal)
                    else:  # if terminal is not in available gaps, then it becoms a danger qubit
                        if (
                            not jump_harvesting
                        ):  # case distinction to make sure that not k-1=1-1=0 if lookahead=1
                            danger_qubits_temp.update({terminal: k_lookahead})
                        else:
                            danger_qubits_temp.update({terminal: k_lookahead - 1})

                    if move_type == "target":
                        for j, next_layer in enumerate(
                            next_layers_copy
                        ):  # update all future layers
                            next_layers_copy[j] = self.replace_pos(
                                next_layer, b, terminal
                            )
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(
                            logical_pos_temp, b, terminal
                        )
                        label = layout_rev[b]
                        layout_mod[label] = terminal
                        available_gaps_temp.append(b)
                    elif move_type == "control" or move_type == "singlequbit":
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(
                                next_layer, a, terminal
                            )
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(
                            logical_pos_temp, a, terminal
                        )
                        label = layout_rev[a]
                        layout_mod[label] = terminal
                        available_gaps_temp.append(a)
                    else:
                        raise RuntimeError(
                            f"other move type than expected: {move_type}"
                        )
                
                for key, path in best_idle.items():
                    _, q, terminal = key

                    if terminal in available_gaps: 
                        available_gaps.remove(terminal)
                    else:  
                        if not jump_harvesting:  
                            danger_qubits_temp.update({terminal: k_lookahead})
                        else:
                            danger_qubits_temp.update({terminal: k_lookahead - 1})

                    for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(
                                next_layer, q, terminal
                            )
                    logical_pos_temp = self.replace_pos(
                            logical_pos_temp, q, terminal
                        )
                    label = layout_rev[q]
                    layout_mod[label] = terminal
                    available_gaps_temp.append(q)

                self.logical_pos = logical_pos_temp.copy()
                self.logical_pos_temp = logical_pos_temp.copy()
                layers = next_layers_copy.copy()
                layout = layout_mod.copy()
                # add everything to solution
                schedule_temp["steiner"] = best_steiner
                schedule_temp["idle_teleport"] = best_idle
                schedule_temp["vdp_dict"] = vdp_dict
                schedule_temp["move_type"] = move_type_lst
                schedule_temp["logical_pos"] = self.logical_pos.copy()
                schedule_temp["cost_history"] = cost_history

                schedule.append(schedule_temp)
                #print("best idle added: ", schedule_temp["idle_teleport"])
            else:
                danger_qubits_temp = {}  # trivial lists to avoid error below
                available_gaps_temp = []

            # attempt idle moving back if asap
            if idle_move_type == "asap":
                if not jump_harvesting:
                    reduce_time_stamp = True
                    schedule, danger_qubits, available_gaps, layout, layers = (
                        self.idle_move_back(
                            schedule,
                            danger_qubits,
                            available_gaps,
                            danger_qubits_temp,
                            available_gaps_temp,
                            layout,
                            layers,
                            reduce_time_stamp,
                            jump_harvesting,
                            None,
                        )
                    )
                elif jump_harvesting and flag_skip_teleport:
                    reduce_time_stamp = True
                    schedule, danger_qubits, available_gaps, layout, layers = (
                        self.idle_move_back(
                            schedule,
                            danger_qubits,
                            available_gaps,
                            danger_qubits_temp,
                            available_gaps_temp,
                            layout,
                            layers,
                            reduce_time_stamp,
                            False,
                            None,
                        )
                    )  # because otherwise error
                elif jump_harvesting and not flag_skip_teleport:
                    reduce_time_stamp = False
                    schedule, danger_qubits, available_gaps, layout, layers = (
                        self.idle_move_back(
                            schedule,
                            danger_qubits,
                            available_gaps,
                            danger_qubits_temp,
                            available_gaps_temp,
                            layout,
                            layers,
                            reduce_time_stamp,
                            jump_harvesting,
                            best_schedule,
                        )
                    )
                else:
                    raise RuntimeError(
                        "Other combinations of jump_harvesting and flag_skip_steiner relevant???"
                    )
            elif idle_move_type == "later":
                # since the idle move is before the steiner search we do not need danger_qubits_temp actually, but needs to be added nevertheless.
                danger_qubits |= danger_qubits_temp.copy()
                available_gaps += available_gaps_temp.copy()

            with open(filename, "wb") as f:
                pickle.dump(schedule, f)

            flag_finished = False
            layers_k = layers[:k_lookahead].copy()
            layers_after_k = layers[k_lookahead:].copy()

            has_teleport = bool(best_steiner_init) or bool(best_idle_init)
            if (
                jump_harvesting and has_teleport and k_lookahead > 1
            ):  # if no best steiner found, just standard further iterations
                # route the next k_lookahead-1 layers without steiner optimization to "harvest" the full potential of previous optimization and without disturbing the previous optimization by new steiner moves
                # make the temp files empty because otherwise repetitive action which is superfluous
                available_gaps_temp = []
                danger_qubits_temp = {}
                if self.metric != "exact":
                    raise ValueError(
                        "if `jump_harvesting=True` you also need to use the exact metric"
                    )
                # this appears like a redundant routing step, but the routes from best_schedule will not necessarily fully coincide with the routing here, since idle_move_back can make it happen that routings can be shorter than in best_schedule
                flag_identical_schedules = True  # the schedules of best_schedule and the routing here can be the same. however, if there is some idle move it can happen that the routing here becomes better than in the computation of the metric
                vdp_dict_present_temp = []  #!DELETE THIS AGAIN ONLY FOR DEBUGGING

                # this routing is effectively redundant, but if idle_type = asap it is important to route again because the schedule can alter due to moves.
                # while len(best_schedule) > 1 : #1 not 0 because the very last layer should be used for a steiner opt again in next it
                if self.use_dag:
                    terminal_pairs_temp = []
                    for layer_temp in layers_k:
                        terminal_pairs_temp += layer_temp
                    dag_k = dag_helper.terminal_pairs_into_dag(
                        terminal_pairs_temp, layout
                    )
                while (
                    len(layers_k) > 1
                ):  # loop based on layers_k because otherwise "asap" may run into skipped gates.
                    # initialize another schedule temp
                    schedule_temp = schedule_temp_for_later.copy()
                    # route
                    vdp_dict, terminal_pairs_remainder, self.factory_times = (
                        self.find_max_vdp_set(
                            layers_k[0], self.logical_pos, self.factory_times
                        )
                    )
                    if len(layers_k) == 1 and len(terminal_pairs_remainder) == 0:
                        # no further steps needed
                        schedule_temp["vdp_dict"] = vdp_dict
                        schedule_temp["logical_pos"] = self.logical_pos.copy()
                        schedule_temp["layout"] = layout.copy()
                        schedule.append(schedule_temp)
                        with open(filename, "wb") as f:
                            pickle.dump(schedule, f)
                        flag_finished = True
                        break

                    # update factory times
                    for key in self.factory_times:
                        if self.factory_times[key] != 0:
                            self.factory_times[key] -= 1

                    vdp_dict_present_temp.append(vdp_dict)
                    # check whether vdp_dict is equivalent to current best_schedule and remove it from best schedule
                    if (
                        flag_identical_schedules
                    ):  # only test as long as we expect identical schedules. this is not alll the time the case.
                        matching = True
                        for vdp_key in best_schedule[0].keys():
                            if vdp_key not in vdp_dict.keys():
                                matching = False
                        if (
                            idle_move_type == "asap"
                        ):  # if asap idle move type then there's no problem if matching wrong
                            del best_schedule[0]

                        elif idle_move_type == "later":
                            if matching:
                                del best_schedule[0]
                            else:
                                raise RuntimeError(
                                    "Mismatch between exact metric routing and real routing in jump harvest. If you do not care you should turn on `idle_move_type == asap`"
                                )

                    # push
                    if self.use_dag:
                        layers_k, dag_k = dag_helper.push_remainder_into_layers_dag(
                            dag_k, terminal_pairs_remainder, layout, layers_k[0]
                        )
                    else:
                        layers_k = self.push_remainder_into_layers(
                            layers_k, terminal_pairs_remainder
                        )

                    # update and add another schedule_temp
                    schedule_temp["vdp_dict"] = vdp_dict
                    schedule_temp["logical_pos"] = (
                        self.logical_pos.copy()
                    )  # redundant info
                    schedule_temp["layout"] = layout.copy()  # redundant info
                    schedule.append(
                        schedule_temp
                    )  # this schedule is adapted in case "asap"

                    # also try idle moves back
                    reduce_time_stamp = False
                    if idle_move_type == "asap":
                        schedule, danger_qubits, available_gaps, layout, layers_k = (
                            self.idle_move_back(
                                schedule,
                                danger_qubits,
                                available_gaps,
                                danger_qubits_temp,
                                available_gaps_temp,
                                layout,
                                layers_k,
                                reduce_time_stamp,
                                jump_harvesting,
                                best_schedule,
                            )
                        )
                        # what qubits where moved?
                        new_idle_moves = [
                            x
                            for x in schedule[-1]["vdp_dict"].keys()
                            if isinstance(x, tuple) and x[0] == "idle_back"
                        ]
                        danger_gap_list = []
                        for idle_key in new_idle_moves:
                            #parts = label_idle.split("_") # label_idle = f"idle_{danger_qubit}_to_{gap}"
                            #danger_qubit = parts[1]
                            #gap = parts[3]
                            #danger_qubit = tuple(
                            #    map(int, danger_qubit.strip("()").split(","))
                            #)  # into tuple again
                            #gap = tuple(map(int, gap.strip("()").split(",")))
                            _, danger_qubit, gap = idle_key
                            danger_gap_list.append((danger_qubit, gap))
                        # also layers_after_k need to be updated if there was something moved back
                        for danger_qubit, gap in danger_gap_list:
                            for j, next_layer in enumerate(
                                layers_after_k
                            ):  # update all future layers
                                layers_after_k[j] = self.replace_pos(
                                    next_layer, danger_qubit, gap
                                )

                # update global layers here. with the terminal pairs remainder which was the last in the above loop
                layers = layers_k.copy() + layers_after_k.copy()

                # after adding, reinitialize the layers structure again from the dag
                # reinitialize dag here, because structure changed!!!
                if self.use_dag:
                    terminals_temp_for_reinit = []
                    for layer in layers:
                        terminals_temp_for_reinit += layer

                    dag = dag_helper.terminal_pairs_into_dag(
                        terminals_temp_for_reinit, layout
                    )
                    # also update layers
                    layers = []
                    for layer in range(len(list(dag.layers()))):
                        layers.append(
                            dag_helper.extract_layer_from_dag(dag, layout, layer)
                        )

                # reduce the time stamps in danger_qubits by k_lookahead (because initialized this way), but in the previous loop we avoided the stepwise reduction of the time labels
                danger_qubits = {
                    qubit: time - k_lookahead + 1
                    for (qubit, time) in danger_qubits.items()
                }

            if flag_finished:
                break  # otherwise last layer may appear twice

            with open(filename, "wb") as f:
                pickle.dump(schedule, f)

        if not tst.check_num_gates(terminal_pairs, schedule):
            warnings.warn(
                "The number of input gates and the routed gates in the schedule do NOT coincide!!"
            )
        else:
            logger.info(
                "Number of gates in schedule and initial terminal_pairs coincides (:"
            )

        if stimtest:
            if tst.check_order_dyn_gates(terminal_pairs, schedule):
                logger.info(
                    "Stim test succeeded: Pushing gates does not cause trouble(:"
                )
            else:
                warnings.warn("Stim test failed: Pushing gates causes trouble):")

        # test whether something overlapping
        if tst.check_duplicate_nodes_per_layer(schedule):
            logger.info(
                "No duplicates found in any layer of the schedule - hence all good(:"
            )
        if tst.check_path_on_logical(schedule):
            logger.info("No path/tree is placed on a logical pos. All good(:")
        if tst.test_times_t_gates_opt(schedule, self.t, self.factory_pos):
            logger.info("All good with the reset times (:")

        return schedule, improvement_history
