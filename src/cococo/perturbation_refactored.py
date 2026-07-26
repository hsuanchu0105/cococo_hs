"""Clarified rewrite of ``TeleportationRouter.perturbation``, with a mobile T-junction.

This module holds a standalone version of the SA neighbour-generator that
currently lives inline in ``cococo/utils_routing.py``
(``TeleportationRouter.perturbation``).  The original stays in place and active;
this one is meant to be readable and can be bound onto the class::

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

**This is NOT behaviour-identical to the inline original.**  The original adds
the gate's own ``path1`` to ``other_paths`` and deletes it before routing, which
also deletes every candidate T-junction -- so the "try all junctions" loop only
ever saw the junction it started with (all others raised ``NodeNotFound``, which
the ``except`` clause swallowed) and the junction could never move.  Here
``path1`` stays in the constraint graph and is stripped per candidate inside
``shortest_branch`` instead, so:

* every allowed junction is usable as a start node -- the junction moves too;
* the branch still cannot run along ``path1``: it meets it only at its own start
  node, so gluing branch onto ``path1`` can never close a cycle.

Invariant worth asserting on the result, for every CNOT/T entry ``(p1, p2)``:
``p2[0] in p1`` and ``set(p2[1:]) & set(p1) == set()``.
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
            # NOTE: the gate's own path1 is deliberately left in the graph here.
            # Removing it up front would also remove every candidate T-junction,
            # so no junction other than the current one could ever be used as a
            # dijkstra source.  `shortest_branch` strips path1 per candidate
            # instead -- that keeps every junction usable while still forbidding
            # the branch from running along path1.  The `node not in path_con`
            # guard in the vdp sweep below is what keeps path1 alive here.

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

    def shortest_branch(g_full, starts, target, path1):
        """
        Shortest branch to `target` over every allowed junction, such that the
        branch meets `path1` only at its own start node.

        `g_full` still contains `path1`.  We strip `path1` once, then re-attach
        one candidate junction at a time: dijkstra started at `j` then cannot
        step onto any other path1 node, because none of them are in the graph.
        So `branch & set(path1) == {j}` holds by construction -- no cycle can
        form when the branch is glued onto path1 -- while every candidate is
        still reachable as a source.
        """
        if target in set(path1):
            # The terminal already lies on the gate's own path, so it is part of
            # the tree already and needs no branch -- same convention as
            # `on_path_random` in initialize_steiner, which stores [terminal].
            # Routing a real branch here would leave path1 and rejoin it at the
            # terminal, i.e. close a cycle.  Only legal if the terminal is itself
            # an allowed junction.
            return [target] if target in set(starts) else None

        best = None
        g = g_full.copy()  # one copy total, not one per candidate
        g.remove_nodes_from(path1)  # target is not on path1, so it survives

        for j in starts:
            if j not in g_full:
                # candidate removed for some other reason (e.g. occupied by
                # another route), so it cannot serve as a junction
                continue

            neighbours = [n for n in g_full.neighbors(j) if n in g]
            g.add_node(j)
            g.add_edges_from((j, n) for n in neighbours)
            try:
                branch = nx.dijkstra_path(g, j, target)
                if best is None or len(branch) < len(best):
                    best = branch
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            g.remove_node(j)  # drops j and its edges -> g is restored

        return best

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

        # reachable nodes from terminal
        neighborhood = sorted(
            set(
                nx.single_source_shortest_path_length(
                    g_temp, terminal, cutoff=radius
                ).keys()
            )
        )
        if len(neighborhood) == 1:  # only the terminal itself -> nothing to move
            continue

        # Draw a new terminal and route to it.  For a steiner tree every allowed
        # junction is tried, so the T-junction moves along with the terminal.
        starts = None if kind == "idle" else allowed_starts(key_tree, path1)
        path_terminal = None
        tried = set()
        while True:
            new_terminal = random.choice(list(neighborhood))
            tried.add(new_terminal)
            if new_terminal != terminal:  # do not want same terminal again
                if kind == "idle":
                    # the idle corridor is replaced wholesale, so it may reuse
                    # its own old nodes -- no path1 restriction here
                    try:
                        path_terminal = nx.dijkstra_path(g_temp, source, new_terminal)
                    except nx.NetworkXNoPath:
                        warnings.warn(
                            "If this is called you need to check why this is happening."
                        )
                else:
                    path_terminal = shortest_branch(
                        g_temp, starts, new_terminal, path1
                    )
                if path_terminal:
                    break
            if len(tried) >= len(neighborhood):
                # every reachable terminal tried and none routable -> give up on
                # this entry rather than spinning forever
                break

        if not path_terminal:
            continue

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
                    g_tt, allowed_starts(key_tree, path1), terminal, path1
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
