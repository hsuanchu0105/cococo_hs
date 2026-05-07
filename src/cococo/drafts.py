import copy 
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
        next_layers_copy = copy.deepcopy(layers)
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

def flattened_paths(path_pair):
    path1, path2 = path_pair
    nodes = []
    if path1 is not None:
        nodes += path1 #!
    if path2 is not None:
        nodes += path2
    return nodes

def perturbation(self, teleport_dct: dict, radius: int, vdp_dict: dict):
        """
        Computes a perturbation of a given collection of trees within a given radius of edges around the current terminal.

        For each tree in steiner_dct a new location of the 3rd terminal is updated randomly.
        """
        if self.logical_pos_temp is None:
            raise RuntimeError(
                "Need to initialize logical pos temp properly in a summarizing method."
            )
        g_temp = self.g.copy()
        g_temp.remove_nodes_from(self.factory_pos)
        #g_temp.remove_nodes_from(
        #    self.logical_pos_temp
        #)  # logical pos temp must be successively updated in a loop where you apply multiple perturbations

        teleport_dct_update = teleport_dct.copy()
        # used_nodes = set()
        new_terminal = None

        for key_tree, (path1, path2) in teleport_dct.items():

            is_idle = isinstance(key_tree, tuple) and len(key_tree) == 3 and key_tree[0] == "idle"

            g_temp_temp = g_temp.copy()
            # determine whether tree corresponds to an idle, a CNOT or T gate 
            if is_idle:
                ("idle", q, terminal) = key_tree
                g_temp_temp.remove_nodes_from(
                    [x for x in self.logical_pos_temp if x != q]
                )
            elif len(key_tree) == 3:
                (a, b, terminal) = key_tree 
                g_temp_temp.remove_nodes_from(
                    [x for x in self.logical_pos_temp]
                )
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                g_temp_temp.remove_nodes_from(
                    [x for x in self.logical_pos_temp]
                )
            else:
                raise RuntimeError(
                    "Something is wrong with the allocation of keys in the steiner_dict"
                )
            other_paths = [
                pos
                for keyy, path_pair in teleport_dct_update.items()
                if keyy != key_tree
                for pos in flattened_paths(path_pair)
            ]

            # a terminal can be placed on the path. in this case you are NOT allowed to remove it! the above somehow sometimes add terminal, hence remove it again
            if terminal in other_paths:
                other_paths.remove(terminal)
            if path2 is not None and path2[0] in other_paths:
                other_paths.remove(path2[0])

            
            g_temp_temp.remove_nodes_from(other_paths)

            # remove nodes from vdp dict (the tree is not allowed to be on or cross another path)
            for path_label, path in vdp_dict.items():
                if isinstance(path_label, str):
                    nodes_to_delete = path[1:]  # for idle move you need to delete more
                elif isinstance(path_label[0], tuple):  # cnot
                    nodes_to_delete = path[1:-1]
                elif isinstance(path_label[0], int):  # t
                    nodes_to_delete = path[1:]
                for node in nodes_to_delete:
                    current_tree_nodes = set(path1 or [])
                    if path2 is not None:
                        current_tree_nodes.update(path2)
                    if (
                        node in g_temp_temp.nodes() and node not in current_tree_nodes
                    ):  # {terminal, path2[0]}
                        g_temp_temp.remove_node(node)
            # find "neighborhood" of the terminal
            neighborhood = set(
                nx.single_source_shortest_path_length(
                    g_temp_temp, terminal, cutoff=radius
                ).keys()
            )
            # the single source shortest path,... ensures that only reachable nodes are included
            # choose one of them
            candidates = list(neighborhood - {terminal})
            random.shuffle(candidates)

            path_terminal = None

            for new_terminal in candidates:
                try:
                    if is_idle:
                        path_terminal = nx.dijkstra_path(g_temp_temp, q, new_terminal)
                    elif len(key_tree) == 3 and path2 is not None:
                        path_terminal = nx.dijkstra_path(g_temp_temp, path2[0], new_terminal)
                    else: # T gate 
                        continue
                    break
                except nx.NetworkXNoPath:
                    continue

            if path_terminal is None:
                continue

            # delete old entry and add new with updated key
            teleport_dct_update.pop(key_tree, None)
            if key_tree[0] == "idle":
                ("idle", q, terminal) = key_tree
                new_key_tree = ("idle", q, new_terminal)
                teleport_dct_update[new_key_tree] = (path_terminal, None)
            elif len(key_tree) == 3:
                (a, b, terminal) = key_tree
                new_key_tree = (a, b, new_terminal)
                teleport_dct_update[new_key_tree] = (path1, path_terminal)
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                new_key_tree = (a, new_terminal)
                teleport_dct_update[new_key_tree] = (path1, path_terminal) #! TODO
            
            
            # remove_branch_nodes += path_terminal

        # it is possible that (A) does not capture everything, as the terminal path may change in a later iteration and thus make even shorter paths possible.
        if (
            new_terminal is not None
        ):  # if the neighborhood has 1 item only, the above breaks. then we do not want to do this reduction.
            teleport_dct_update_sec = teleport_dct_update.copy()
            for key_tree, (path1, path2) in teleport_dct_update.items():
                if key_tree[0] == "idle":
                    ("idle", q, terminal) = key_tree
                elif len(key_tree) == 3:
                    (a, b, terminal) = key_tree
                elif len(key_tree) == 2:
                    (a, terminal) = key_tree
                else:
                    raise ValueError("steiner dct keys are wrong.")
                other_paths = [
                    pos
                    for keyy, path_pair in teleport_dct_update.items()
                    if keyy != key_tree
                    for pos in flattened_paths(path_pair)
                ]
                if terminal in other_paths:
                    other_paths.remove(terminal)
                if path2 is not None and path2[0] in other_paths:
                    other_paths.remove(path2[0])
                if terminal in other_paths:
                    other_paths.remove(terminal)
                g_temp_temp = g_temp.copy()
                g_temp_temp.remove_nodes_from(other_paths)
                for path_label, path in vdp_dict.items():
                    if isinstance(path_label, str):
                        nodes_to_delete = path[
                            1:
                        ]  # for idle move you need to delete more
                    else:
                        nodes_to_delete = path[1:-1]
                    for node in nodes_to_delete:
                        current_tree_nodes = set(path1 or [])
                        if path2 is not None:
                            current_tree_nodes.update(path2)
                        if (
                            node in g_temp_temp.nodes() and node not in current_tree_nodes
                        ):  # {terminal, path2[0]}:
                            g_temp_temp.remove_node(node)
                paths_lst_temp = (
                    []
                )  # collect all paths from path1[1:-1] to new_terminal
                if key_tree[0] == "idle":               
                    try:
                        path_temp = nx.dijkstra_path(
                            g_temp_temp, q, terminal
                        )
                        paths_lst_temp.append(path_temp)
                    except nx.NetworkXNoPath:
                        pass
                elif len(key_tree) == 3:
                    for node_on_path in path1[1:-1]:
                        try:
                            path_temp = nx.dijkstra_path(
                                g_temp_temp, node_on_path, terminal
                            )
                            paths_lst_temp.append(path_temp)
                        except nx.NetworkXNoPath:
                            pass
                else: # T gate 
                    pass 
                if paths_lst_temp:
                    path_terminal = min(paths_lst_temp, key=len)
                    if key_tree[0] == "idle":
                        ("idle", q, terminal) = key_tree
                        new_key_tree = ("idle", q, terminal)
                    elif len(key_tree) == 3:
                        (a, b, terminal) = key_tree
                        new_key_tree = (a, b, terminal)
                    elif len(key_tree) == 2:
                        (a, terminal) = key_tree
                        new_key_tree = (a, terminal)
                    teleport_dct_update_sec.pop(key_tree, None)
                    if len(key_tree) == 3 and key_tree[0] != "idle":
                        teleport_dct_update_sec[new_key_tree] = (path1, path_terminal)
                    else:
                        teleport_dct_update_sec[new_key_tree] = (path_terminal, None)
        if new_terminal is None:
            teleport_dct_update_sec = teleport_dct_update
            g_temp_temp = g_temp.copy()
        return teleport_dct_update_sec, g_temp_temp


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

def split_teleport_dct(teleport_dct: dict):
    """
    Split a mixed teleport dictionary into:
        - steiner_dct: CNOT/Steiner entries
        - idle_move_dct: idle teleport entries

    Expected key types:
        ("idle", q, terminal) -> (path, None)
        (a, b, terminal)     -> (path1, path2)
    """
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
            logger.info("No Steiner improvement possible in this layer.")
        else:
            logger.info("Steiner found for this layer.")

        logger.info("Final Temperature T = %.6e", T)

        best_steiner, best_idle = split_teleport_dct(best_teleport)

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


def qubits_in_vdp_key(key):
    if isinstance(key, str):
        return []

    if isinstance(key[0], tuple):
        # CNOT key: ((x1, y1), (x2, y2))
        return [key[0], key[1]]

    if isinstance(key[0], int):
        # T-gate key: (x, y)
        return [key]

    raise RuntimeError(f"Unexpected vdp key: {key}")


def occupied_nodes_from_vdp(vdp_dict):
    occupied = set()
    for key, path in vdp_dict.items():
        if isinstance(key, str): # idle 
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

def initialize_idle_moves(
    self,
    vdp_dict: dict,
    steiner_dct: dict,
    layout: dict,
    max_idle_moves: int | None = None,
):
    """
    Initialize random idle-qubit teleportation moves.

    Returns:
        idle_move_dct:
            {
                ("idle", q, terminal): path_idle
            }
    """
    logical_positions = list(layout.values())

    active_qubits = set()
    
    
    for key in vdp_dict:
        active_qubits.update(qubits_in_vdp_key(key))

    idle_qubits = [
        q for q in logical_positions
        if q not in active_qubits
    ]

    random.shuffle(idle_qubits)

    if max_idle_moves is not None:
        idle_qubits = idle_qubits[:max_idle_moves]

    occupied = set()
    occupied.update(occupied_nodes_from_vdp(vdp_dict))
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
        for pos in logical_positions:
            if pos != q and pos in g_temp.nodes:
                g_temp.remove_node(pos)

        if q not in g_temp.nodes:
            continue

        reachable = list(nx.single_source_shortest_path_length(g_temp, q).keys())

        if not reachable:
            continue

        terminal = random.choice(reachable)
        path_idle = nx.dijkstra_path(g_temp, q, terminal)

        idle_move_dct[("idle", q, terminal)] = path_idle

        # Prevent other idle moves from overlapping this one.
        used_idle_paths.update(path_idle)

    return idle_move_dct

# perturbation second tryout
def perturbation(self, steiner_dct: dict, radius: int, vdp_dict: dict):
        """
        Computes a perturbation of a given collection of trees within a given radius of edges around the current terminal.

        For each tree in steiner_dct a new location of the 3rd terminal is updated randomly.
        """
        if self.logical_pos_temp is None:
            raise RuntimeError(
                "Need to initialize logical pos temp properly in a summarizing method."
            )
        g_temp = self.g.copy()
        g_temp.remove_nodes_from(self.factory_pos)
        g_temp.remove_nodes_from(
            self.logical_pos_temp
        )  # logical pos temp must be successively updated in a loop where you apply multiple perturbations

        steiner_dct_update = steiner_dct.copy()
        # used_nodes = set()
        new_terminal = None

        for key_tree, (path1, path2) in steiner_dct.items():

            # determine whether tree corresponds to a CNOT or T gate
            if len(key_tree) == 3:
                (a, b, terminal) = key_tree
                other_paths = [
                    pos
                    for keyy, path in steiner_dct_update.items()
                    if keyy != (a, b, terminal)
                    for pos in path[0][1:-1]
                ]
                other_paths += [
                    pos
                    for keyy, path in steiner_dct_update.items()
                    if keyy != (a, b, terminal)
                    for pos in path[1]
                ]  # not 1:-1 because 3rd terminal is not allowed to be part of other terminal path
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                other_paths = [
                    pos
                    for keyy, path in steiner_dct_update.items()
                    if keyy != (a, terminal)
                    for pos in path[0][1:-1]
                ]
                other_paths += [
                    pos
                    for keyy, path in steiner_dct_update.items()
                    if keyy != (a, terminal)
                    for pos in path[1]
                ]
            else:
                raise RuntimeError(
                    "Something is wrong with the allocation of keys in the steiner_dict"
                )

            # a terminal can be placed on the path. in this case you are NOT allowed to remove it! the above somehow sometimes add terminal, hence remove it again
            if terminal in other_paths:
                other_paths.remove(terminal)
            if path2[0] in other_paths:
                other_paths.remove(path2[0])

            g_temp_temp = g_temp.copy()
            g_temp_temp.remove_nodes_from(other_paths)

            # remove nodes from vdp dict (the tree is not allowed to be on or cross another path)
            for path_label, path in vdp_dict.items():
                if isinstance(path_label, str):
                    nodes_to_delete = path[1:]  # for idle move you need to delete more
                elif isinstance(path_label[0], tuple):  # cnot
                    nodes_to_delete = path[1:-1]
                elif isinstance(path_label[0], int):  # t
                    nodes_to_delete = path[1:]
                for node in nodes_to_delete:
                    if (
                        node in g_temp_temp.nodes() and node not in path1 + path2
                    ):  # {terminal, path2[0]}
                        g_temp_temp.remove_node(node)
            # find "neighborhood" of teh terminal
            neighborhood = set(
                nx.single_source_shortest_path_length(
                    g_temp_temp, terminal, cutoff=radius
                ).keys()
            )
            # the single source shortest path,... ensures that only reachable nodes are included
            # choose one of them
            if len(neighborhood) == 1:  # if only one neighbor, i.e. the terminal itself
                # new_terminal = None #to skip the updating of the root node below.
                break
            while True:
                new_terminal = random.choice(list(neighborhood))
                if new_terminal == terminal:  # do not want same terminal again
                    continue
                try:
                    path_terminal = nx.dijkstra_path(
                        g_temp_temp, path2[0], new_terminal
                    )  # path2[0] is the connecting node on the path
                except nx.NetworkXNoPath:
                    warnings.warning(
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
                        g_temp_temp, node_on_path, new_terminal
                    )
                    paths_lst_temp.append(path_temp)
                except nx.NetworkXNoPath:
                    pass
            if paths_lst_temp:
                path_terminal = min(paths_lst_temp, key=len)

            # delete old entry and add new iwth updated key
            steiner_dct_update.pop(key_tree, None)
            if len(key_tree) == 3:
                (a, b, terminal) = key_tree
                new_key_tree = (a, b, new_terminal)
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                new_key_tree = (a, new_terminal)
            steiner_dct_update[new_key_tree] = (path1, path_terminal)
            # remove_branch_nodes += path_terminal

        # it is possible that (A) does not capture everything, as the terminal path may change in a later iteration and thus make even shorter paths possible.
        if (
            new_terminal is not None
        ):  # if the neighborhood has 1 item only, the above breaks. then we do not want to do this reduction.
            steiner_dct_update_second = steiner_dct_update.copy()
            for key_tree, (path1, path2) in steiner_dct_update.items():
                if len(key_tree) == 3:
                    (a, b, terminal) = key_tree
                elif len(key_tree) == 2:
                    (a, terminal) = key_tree
                else:
                    raise ValueError("steiner dct keys are wrong.")
                other_paths = [
                    pos
                    for keyy, path in steiner_dct_update_second.items()
                    if keyy != key_tree
                    for pos in path[0][1:-1]
                ]
                other_paths += [
                    pos
                    for keyy, path in steiner_dct_update_second.items()
                    if keyy != key_tree
                    for pos in path[1]
                ]
                if terminal in other_paths:
                    other_paths.remove(terminal)
                if path2[0] in other_paths:
                    other_paths.remove(path2[0])
                if terminal in other_paths:
                    other_paths.remove(terminal)
                g_temp_temp = g_temp.copy()
                g_temp_temp.remove_nodes_from(other_paths)
                for path_label, path in vdp_dict.items():
                    if isinstance(path_label, str):
                        nodes_to_delete = path[
                            1:
                        ]  # for idle move you need to delete more
                    else:
                        nodes_to_delete = path[1:-1]
                    for node in nodes_to_delete:
                        if (
                            node in g_temp_temp.nodes() and node not in path1 + path2
                        ):  # {terminal, path2[0]}:
                            g_temp_temp.remove_node(node)
                paths_lst_temp = (
                    []
                )  # collect all paths from path1[1:-1] to new_terminal
                for node_on_path in path1[1:-1]:
                    try:
                        path_temp = nx.dijkstra_path(
                            g_temp_temp, node_on_path, terminal
                        )
                        paths_lst_temp.append(path_temp)
                    except nx.NetworkXNoPath:
                        pass
                if paths_lst_temp:
                    path_terminal = min(paths_lst_temp, key=len)
                    if len(key_tree) == 3:
                        (a, b, terminal) = key_tree
                        new_key_tree = (a, b, terminal)
                    elif len(key_tree) == 2:
                        (a, terminal) = key_tree
                        new_key_tree = (a, terminal)
                    steiner_dct_update_second.pop(key_tree, None)
                    steiner_dct_update_second[new_key_tree] = (path1, path_terminal)
        if new_terminal is None:
            steiner_dct_update_second = steiner_dct_update
            g_temp_temp = g_temp.copy()
        return steiner_dct_update_second, g_temp_temp