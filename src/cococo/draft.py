
steiner_dct = {}
idle_move_dct = {}

if include_steiner_teleport:
    if reduce_init_steiner:
        steiner_dct = self.initialize_steiner(
            vdp_dict, steiner_init_type, layers=layers, k_lookahead=k_lookahead
        )
        if include_idle_teleport:
            idle_move_dct = self.initialize_idle_moves(
                vdp_dict, steiner_dct, layout, max_idle_teleport
            )
    else:
        steiner_dct = self.initialize_steiner(
        vdp_dict, steiner_init_type, layers=None, k_lookahead=None
        )
        if include_idle_teleport:
            idle_move_dct = self.initialize_idle_moves(
                vdp_dict, steiner_dct, layout, max_idle_teleport
            )

else:
    if include_idle_teleport:
        idle_move_dct = self.initialize_idle_moves(
            vdp_dict, steiner_dct, layout, max_idle_teleport
        )


if len(steiner_dct) == 0 and len(idle_move_dct) == 0:
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



# original version

if reduce_init_steiner:
    steiner_dct = self.initialize_steiner(
        vdp_dict, steiner_init_type, layers=layers, k_lookahead=k_lookahead
    )
    if include_idle_teleport:
        idle_move_dct = self.initialize_idle_moves(
            vdp_dict, steiner_dct, layout, max_idle_teleport
        )
    else:
        idle_move_dct = {}
    if len(steiner_dct) == 0 and len(idle_move_dct) == 0:
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
        # store improvement history
        improvement_history.append((best_cost, cost_history[0]))
else:
    steiner_dct = self.initialize_steiner(
        vdp_dict, steiner_init_type, layers=None, k_lookahead=None
    )
    if include_idle_teleport:
        idle_move_dct = self.initialize_idle_moves(
            vdp_dict, steiner_dct, layout, max_idle_teleport
        )
    else:
        idle_move_dct = {}
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
    # store improvement history
    improvement_history.append((best_cost, cost_history[0]))
