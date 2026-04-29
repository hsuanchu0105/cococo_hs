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

        path_num = 5
        

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
            paths_cand_dict = {}  # gather all possible paths here, between all terminal pairs (cnots) and between all qubits for a tgate with all factories
            tp_list: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = (
                []
            )  # same order, actually redundant but error otherwise
            node_penalty = collections.defaultdict(int)

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
                        if self.valid_path == "cc":
                            if (t_p[0], t_p[1]) in g_temp_temp.edges():
                                g_temp_temp.remove_edge(t_p[0], t_p[1])
                        
                        path_generator = nx.shortest_simple_paths(
                        g_temp_temp, t_p[0], t_p[1]
                        )

                        path_candidates = []
                        for path in path_generator:
                            path_candidates.append(path)
                            if len(path_candidates) >= path_num:
                                break

                        # update penalty
                        for path in path_candidates:
                            for node in path[1:-1]:
                                node_penalty[node] += 1

                        
                        paths_cand_dict[t_p] = path_candidates
                        tp_list.append(t_p)
                        
                    except nx.NetworkXNoPath:
                        # skip the t_p if no path exists
                        pass  # therefore just pass

                # t gate
                #! TODO modification for T gate is not done 
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
                        paths_cand_dict.append(path)
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
            if all(all_t) and len(paths_cand_dict) == 0:
                flag_problem = True

            for key_gate, path_candidates in paths_cand_dict.items():
                if paths_cand_dict[key_gate] and not flag_problem:

                    opt_path = min(
                        paths_cand_dict[key_gate],
                        key=lambda path: (
                            sum(node_penalty[node] for node in path[1:-1]),
                            len(path),
                        ),
                    )

                    t_p = key_gate

                    # update already used qubits based on chosen t_p path
                    if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                        dct_qubits[t_p[0]] = True
                        dct_qubits[t_p[1]] = True
                    elif isinstance(t_p[1], int):
                        dct_qubits[t_p] = True
                        # update the times of the factory patch (which is the position at one end of the path)
                        if opt_path[0] == t_p:
                            factory_times_temp[opt_path[-1]] = self.t + 1
                        elif opt_path[-1] == t_p:
                            factory_times_temp[opt_path[0]] = self.t + 1
                        else:
                            msg = "Factory not in path."
                            raise RuntimeError(msg)

                    # remove nodes from g_temp from path
                    for node in opt_path[1:-1]:
                        g_temp.remove_node(node)
                    successful_terminals.append(t_p)
                    vdp_dict.update({t_p: opt_path})

                    # remove t_p from terminal_pairs_current
                    terminal_pairs_current = [x for x in terminal_pairs_current if x != t_p]
                    break
                else:
                    pass

            if not paths_cand_dict or flag_problem:
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