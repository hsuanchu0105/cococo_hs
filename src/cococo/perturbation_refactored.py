"""Clarified rewrite of ``TeleportationRouter.perturbation``.

This module holds a standalone, behaviour-identical version of the SA
neighbour-generator that currently lives inline in
``cococo/utils_routing.py`` (``TeleportationRouter.perturbation``).  The
original stays in place and active; this one is meant to be readable and can be
bound onto the class for A/B comparison::

    import cococo.perturbation_refactored as pr
    utils.TeleportationRouter.perturbation = pr.perturbation

Structure of the algorithm (unchanged, just de-duplicated):

* **Pass A** -- for every teleport entry, build a per-entry constraint graph,
  take the ``radius`` neighbourhood of its current terminal, draw a *random new*
  terminal, route the branch to it, then try to shorten that branch over the
  allowed junction start nodes.  The entry is re-keyed on the new terminal.
* **Pass B** -- only if pass A actually moved something.  Rebuild the same kind
  of constraint graph and try once more to shorten each branch, now against the
  *fixed* terminals chosen in pass A.

Both passes share the constraint-graph construction, the idle / CNOT / T key
branching, the ``fine_allowed_junctions`` restriction and the pop-then-reinsert
update; those are the helpers below.
"""

import networkx as nx
import random
import warnings

import cococo.internal_testing as tst


def perturbation(self, teleport_dct: dict, radius: int, vdp_dict: dict):
    """
    Computes a perturbation of a given collection of paths within a given radius of edges around the current terminal.

    For each path a new location of the 2nd (3rd terminal) is updated randomly.
    """

    # ------------------------------------------------------------------
    # helpers (close over `self` / `vdp_dict`)
    # ------------------------------------------------------------------

    def occupied_nodes_for_others(key, path_pair):
        """
        collect the occupied nodes except logical data qubits
        """
        path1, path2 = path_pair
        nodes = set()

        if key[0] == "idle":
            if path1 is not None:
                nodes.update(path1[1:])     # exclude idle source q
            return nodes

        # CNOT or T teleport
        if path1 is not None:
            nodes.update(path1[1:-1])        # exclude logical endpoints
        if path2 is not None:
            nodes.update(path2)            # branch path

        return list(nodes)

    def classify(key_tree, bad_key_error):
        """
        Split a teleport key into ``(kind, source, terminal)``.

        ``("idle", q, t) -> ("idle", q, t)``, ``(a, b, t) -> ("cnot", None, t)``,
        ``(a, t) -> ("t", None, t)``.  `bad_key_error` is raised for anything
        else (the two passes historically raise different exception types).
        """
        if key_tree[0] == "idle":
            _, q, terminal = key_tree
            return "idle", q, terminal
        if len(key_tree) == 3:
            _, _, terminal = key_tree
            return "cnot", None, terminal
        if len(key_tree) == 2:
            _, terminal = key_tree
            return "t", None, terminal
        raise bad_key_error

    def build_constraint_graph(
        key_tree, kind, source, terminal, path1, path2, siblings
    ):
        """
        The graph this entry's branch may be routed through: everything is
        removed that it is not allowed to touch.

        `siblings` is the dict whose *other* entries block nodes; it is the dict
        being built up by the current pass, so later entries see earlier updates.
        """
        g = self.g.copy()
        g.remove_nodes_from(self.factory_pos)

        # occupied ancillas from other paths 
        other_paths = [
            pos
            for keyy, path_pair in siblings.items()
            if keyy != key_tree
            for pos in occupied_nodes_for_others(keyy, path_pair)
        ]

        # data qubits are obstacles -- except the idle qubit that is moving
        if kind == "idle":
            g.remove_nodes_from([x for x in self.logical_pos_temp if x != source])
        else:
            g.remove_nodes_from([x for x in self.logical_pos_temp])
            if kind == "cnot":
                # this gate's own vdp path is blocked too, apart from the
                # T-junction the branch hangs off
                for node in path1:
                    if node != path2[0]:
                        other_paths.append(node)

        protected = {terminal}
        if path2:
            protected.add(path2[0])
        other_paths = [node for node in other_paths if node not in protected]
        g.remove_nodes_from(other_paths)

        # also need to delete paths in vdp_dict
        path_con = path1 + path2 if path2 else path1
        for path_label, path in vdp_dict.items():
            if isinstance(path_label, tuple) and path_label[0] == "idle_back":
                nodes_to_delete = path[1:]  # for idle move you need to delete more
            elif isinstance(path_label[0], tuple):  # cnot
                nodes_to_delete = path[1:-1]
            elif isinstance(path_label[0], int):  # t
                nodes_to_delete = path[1:]
            for node in nodes_to_delete:
                if node in g.nodes() and node not in path_con:
                    g.remove_node(node)

        return g

    def allowed_starts(key_tree, path1):
        """
        Candidate T-junction nodes on the gate's own path.  In fine-grained mode
        the junction may only sit in the committed ancilla's region.
        """
        allowed = None
        if len(key_tree) == 3 and self.fine_allowed_junctions:
            allowed = self.fine_allowed_junctions.get((key_tree[0], key_tree[1]))
        if allowed is None:
            return path1[1:-1]
        return [n for n in path1[1:-1] if n in allowed]

    def shortest_branch(g, starts, target):
        """Shortest branch from any allowed start to `target`, or None."""
        candidates = []
        for node_on_path in starts:
            try:
                candidates.append(nx.dijkstra_path(g, node_on_path, target))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # NodeNotFound: a start node removed from the graph (e.g. an
                # overlap-block node of path1 occupied by another route)
                pass
        return min(candidates, key=len) if candidates else None

    def reinsert(dct, key_tree, kind, source, terminal_new, path1, branch):
        """Drop the old key and store the entry under its (possibly new) terminal."""
        dct.pop(key_tree, None)
        if kind == "idle":
            dct[("idle", source, terminal_new)] = (branch, None)
        elif kind == "cnot":
            a, b, _ = key_tree
            dct[(a, b, terminal_new)] = (path1, branch)
        else:  # single-qubit / T teleport
            a, _ = key_tree
            dct[(a, terminal_new)] = (path1, branch)

    # ------------------------------------------------------------------

    if self.logical_pos_temp is None:
        raise RuntimeError(
            "Need to initialize logical pos temp properly in a summarizing method."
        )

    teleport_dct_update = teleport_dct.copy()

    if not teleport_dct_update:
        g_tt = self.g.copy()

    new_terminal = None

    # ------------------------------------------------------------------
    # (A) move every terminal to a random node in its radius-neighbourhood
    # ------------------------------------------------------------------
    for key_tree, (path1, path2) in teleport_dct.items():
        kind, source, terminal = classify(
            key_tree,
            RuntimeError(
                "Something is wrong with the allocation of keys in the steiner_dict"
            ),
        )

        g_temp = build_constraint_graph(
            key_tree, kind, source, terminal, path1, path2,
            siblings=teleport_dct_update, 
        )

        # single_source_shortest_path_length keeps only reachable nodes
        neighborhood = sorted(
            set(
                nx.single_source_shortest_path_length(
                    g_temp, terminal, cutoff=radius
                ).keys()
            )
        )
        if len(neighborhood) == 1:  # only the terminal itself -> nothing to move
            continue

        # draw a new terminal that is actually reachable from the branch root
        branch_root = source if kind == "idle" else path2[0]
        path_terminal = None
        while True:
            new_terminal = random.choice(list(neighborhood))
            if new_terminal == terminal:  # do not want same terminal again
                continue
            try:
                path_terminal = nx.dijkstra_path(g_temp, branch_root, new_terminal)
            except nx.NetworkXNoPath:
                warnings.warn(
                    "If this is called you need to check why this is happening."
                )
            if path_terminal:
                break

        if kind != "idle":
            # try to hang the branch off a better junction on the gate's path
            shorter = shortest_branch(
                g_temp, allowed_starts(key_tree, path1), new_terminal
            )
            if shorter is not None:
                path_terminal = shorter

        reinsert(
            teleport_dct_update, key_tree, kind, source, new_terminal, path1, path_terminal
        )

    # ------------------------------------------------------------------
    # (B) (A) may miss shortenings, because a terminal moved in a later
    #     iteration can open up a shorter branch for an earlier entry.
    # ------------------------------------------------------------------
    if new_terminal is not None:
        # if the neighborhood had 1 item only, (A) never moved anything and we
        # do not want this reduction.
        teleport_dct_update_second = teleport_dct_update.copy()
        for key_tree, (path1, path2) in teleport_dct_update.items():
            kind, source, terminal = classify(
                key_tree, ValueError("steiner dct keys are wrong.")
            )

            g_tt = build_constraint_graph(
                key_tree, kind, source, terminal, path1, path2,
                siblings=teleport_dct_update_second, 
            )

            if kind == "idle":
                try:
                    path_terminal = nx.dijkstra_path(g_tt, source, terminal)
                except nx.NetworkXNoPath:
                    path_terminal = None
            else:
                path_terminal = shortest_branch(
                    g_tt, allowed_starts(key_tree, path1), terminal
                )

            if path_terminal:
                # same terminal as before -- only the branch is replaced
                reinsert(
                    teleport_dct_update_second, key_tree, kind, source,
                    terminal, path1, path_terminal,
                )
    else:
        teleport_dct_update_second = teleport_dct_update
        g_tt = self.g.copy()
        g_tt.remove_nodes_from(self.factory_pos)

    tst.check_perturbation(teleport_dct_update_second, vdp_dict)

    return teleport_dct_update_second, g_tt
