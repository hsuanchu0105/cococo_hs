
# penalized version

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
    path_srch = 3
    

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


    node_penalty = collections.defaultdict(int)

    for t_p in layer.copy():

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

                paths_temp = []
                for path in path_generator:
                    paths_temp.append(path)
                    if len(paths_temp) >= path_num:
                        break

                # update penalty
                for path in paths_temp:
                    for node in path[1:-1]:
                        node_penalty[node] += 1

            except nx.NetworkXNoPath:
                # skip the t_p if no path exists
                pass  # therefore just pass

    while (
        len(terminal_pairs_current) > 0 and flag_problem is False
    ):  # noqa: PLR1702
        paths_cand_dict = {}  # gather all possible paths here, between all terminal pairs (cnots) and between all qubits for a tgate with all factories
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
                    if self.valid_path == "cc":
                        if (t_p[0], t_p[1]) in g_temp_temp.edges():
                            g_temp_temp.remove_edge(t_p[0], t_p[1])
                    
                    path_generator = nx.shortest_simple_paths(
                    g_temp_temp, t_p[0], t_p[1]
                    )

                    path_candidates = []
                    for path in path_generator:
                        path_candidates.append(path)
                        if len(path_candidates) >= path_srch:
                            break

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
                    #! TODO 
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

        for key_gate, _ in paths_cand_dict.items():
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
        def gap_distance(g, danger_qubit, gap):
            try:
                return nx.shortest_path_length(g, danger_qubit, gap)
            except nx.NetworkXNoPath:
                return float("inf")
            
        def check_gap_flag(schedule, gap):
            if schedule["steiner"] is not None:
                    for steiner in schedule["steiner"].values():
                        for node in steiner[0]:
                            if node == gap:
                                return True
                        for node in steiner[1]:
                            if node == gap:
                                return True
            for path in schedule["vdp_dict"].values():
                for node in path:
                    if node == gap:
                        return True
            return False
        
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
            #! TODO the gap is currently deleted, is the gap part of layout?
            # go through gaps and take the one to which a path is available
            path_idle = None
            g_copy = self.g.copy()
            #initial_nodes = set(g_copy.nodes())
                
            if schedule_temp["steiner"] is not None:
                for steiner in schedule_temp["steiner"].values():
                    for node in steiner[0]: 
                        if node != danger_qubit:
                            g_copy.remove_node(node)
                    for node in steiner[1]:
                        if node in g_copy.nodes() and node != danger_qubit:   # and node != gap: #at least one node in steiner[1] is already in steiner[0]
                            g_copy.remove_node(node)
            for path in schedule_temp["vdp_dict"].values():
                for node in path :
                    if node in g_copy.nodes() and node != danger_qubit:  # and node != gap:
                        g_copy.remove_node(node)

            for pos in layout_mod.values():
                if pos in g_copy.nodes() and pos != danger_qubit:  #!just in case you find a bug, this was node != gap before, i dont know why this worked before i moved this into an own method
                    g_copy.remove_node(pos)


            candidate_gaps = [
                gap for gap in available_gaps
                if not check_gap_flag(schedule_temp, gap)
            ]

            path_idle = None
            ordered_gaps = sorted(
            candidate_gaps,
            key = lambda gap:gap_distance(g_copy, danger_qubit, gap)
            )
            if ordered_gaps:
                opt_gap = ordered_gaps[0]
                dist = gap_distance(g_copy, danger_qubit, opt_gap)
                if dist < float("inf"):
                    path_idle = nx.dijkstra_path(g_copy, opt_gap, danger_qubit)
                    # danger_qubits_copy.remove(danger_qubit)
                    del danger_qubits_copy[danger_qubit]
                    available_gaps.remove(opt_gap)
                    label_idle = f"idle_{danger_qubit}_to_{opt_gap}"
                    vdp_dict.update({label_idle: path_idle}) #! TODO now we have string keys, will be troubled if we call this function again because of line 1378
                    idle_move_labels.append(label_idle)
                    logical_pos_temp = self.replace_pos(
                        logical_pos_temp, danger_qubit, opt_gap
                    )
                    for j, next_layer in enumerate(
                        next_layers_copy
                    ):  # update all future layers
                        next_layers_copy[j] = self.replace_pos(
                            next_layer, danger_qubit, opt_gap
                        )
                    label = layout_rev[danger_qubit]
                    layout_mod[label] = opt_gap
                    layout_rev = {
                        j: i for i, j in layout_mod.items()
                    }  #!update layout_rev
                    flag_idle_move = True
                # because if path found you do not want to find another path to the same danger qubit
            elif path_idle == None:
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