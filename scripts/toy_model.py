import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

import networkx as nx
import cococo.utils_routing as utils
import plotting
import cococo.layouts as layouts


def toy_model(data, ancilla, example):
    g = nx.Graph()

    # 5 data qubits
    data_qubit_locs = data

    # ancilla / routing nodes between them and around them
    ancilla_locs = ancilla

    g.add_nodes_from(data_qubit_locs)
    g.add_nodes_from(ancilla_locs)

    if example in {1, 2, 3}:
        # horizontal corridor
        g.add_edges_from([
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((2, 0), (3, 0)),
            ((3, 0), (4, 0)),

            ((0, 1), (1, 1)),
            ((1, 1), (2, 1)),
            ((2, 1), (3, 1)),
            ((3, 1), (4, 1)),
            ((4, 1), (5, 1)),
            
            ((1, 2), (2, 2)),
            ((2, 2), (3, 2)),
            ((3, 2), (4, 2)),
            ((4, 2), (5, 2)),

            ((2, 3), (3, 3)),
            ((3, 3), (4, 3)),
        ])

        # vertical escape/corridor edges
        g.add_edges_from([
            ((0, 0), (0, 1)),
            ((2, 0), (2, 1)),
            ((4, 0), (4, 1)),

            ((1, 1), (1, 2)),
            ((3, 1), (3, 2)),
            ((5, 1), (5, 2)),

            ((2, 2), (2, 3)),
            ((4, 2), (4, 3)),
        ])
    elif example == 4:
        node_set = set(g.nodes)

        for x, y in g.nodes:
            for neighbor in [(x + 1, y)]:
                if neighbor in node_set:
                    g.add_edge((x, y), neighbor)

        for x, y in g.nodes:
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                for neighbor in [(x, y + 1)]:
                    if neighbor in node_set:
                        g.add_edge((x, y), neighbor)
            

    factory_ring = []

    nx.set_node_attributes(g, {node: node for node in g.nodes()}, "pos")

    return g, data_qubit_locs, factory_ring


def gen_toy_model(example: int):
    """
    example = 1: 5 data qubits
    example = 2: 4 data qubits
    example = 3: 6 data qubits
    """
    if example == 1:
        data_qubit_loc = [
        (0, 0),
        (5, 2),
        (4, 0),
        (2, 2),
        (2, 1),
        ]
        ancilla_loc = [
        (1, 0),
        (2, 0),
        (3, 0),

        (0, 1),
        (1, 1),
        (3, 1),
        (4, 1),
        (5, 1),

        (1, 2),
        (3, 2), 
        (4, 2),

        (2, 3),
        (3, 3),
        (4, 3),
        ]
        

    elif example == 2:
        data_qubit_loc = [
        (0, 0),
        (5, 2),
        (4, 0),
        (2, 2),
        
        ]
        ancilla_loc = [
        (1, 0),
        (2, 0),
        (3, 0),

        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),

        (1, 2),
        (3, 2), 
        (4, 2),

        (2, 3),
        (3, 3),
        (4, 3),
        ]
    elif example == 3:
        data_qubit_loc = [
        (0, 0),
        (5, 2),
        (4, 0),
        (2, 2),
        (2, 1),
        (4, 3),     
        ]
        ancilla_loc = [
        (1, 0),
        (2, 0),
        (3, 0),

        (0, 1),
        (1, 1),
        (3, 1),
        (4, 1),
        (5, 1),

        (1, 2),
        (3, 2), 
        (4, 2),

        (2, 3),
        (3, 3),
        ]
    elif example == 4:
        data_qubit_loc = [
        (0, 0),
        (1, 0),
        (4, 0),
        (5, 0),
        (8, 0), 
        (9, 0),
        (2, 2),
        (3, 2),
        (6, 2), 
        (7, 2),
        (10, 2),
        (11, 2),
             
        ]
        ancilla_loc = [
        (2, 0),
        (3, 0),
        (6, 0), 
        (7, 0),
        (10, 0),


        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (10, 1),
        (11, 1),

        (1, 2),
        (4, 2), 
        (5, 2),
        (8, 2),
        (9, 2),
        (12, 2),

        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
        (8, 3),
        (9, 3),
        (10, 3),
        (11, 3),
        (12, 3),
        ]
    g, data_qubit_locs, factory_ring = toy_model(data_qubit_loc, ancilla_loc, example)
    return g, data_qubit_locs, factory_ring

def gen_gates(example: int):
    pairs = []
    if example == 1:
        pairs = [(1, 3), (0, 2), (4, 1), (2, 3)]
    if example == 2:
        pairs = [(0, 3), (2, 1), (3, 2), (0, 1)]
    if example == 3:
        pairs = [(1, 3), (0, 2), (4, 1), (2, 3), (3, 4), (1, 2)]   
    if example == 4:
        pairs = [(1, 8), (3, 4), (5, 10), (7, 2), (1, 8), (2, 11)]
        #pairs = [(1, 8), (3, 4), (5, 10), (7, 2), (1, 8), (2, 11), (4, 7), (6, 3), (1, 3), (4, 11), (1, 6), (10, 11), (1, 7), (4, 9)]
    return pairs 

example = 4
g, data_qubit_locs, factory_ring = gen_toy_model(example)
layout = {i: pos for i, pos in enumerate(data_qubit_locs)}
factories = []
t = 2

plotting.plot_lattice_paths(g, {}, {}, layout, factories, size = (18,8))

pairs = gen_gates(example)
terminal_pairs = layouts.translate_layout_circuit(pairs, layout) #let's stick to the simple layout
#print("terminal pairs: ", terminal_pairs)


router = utils.BasicRouter(g, data_qubit_locs, factories, valid_path = "cc", t=t, metric = "exact", use_dag = False)
layers = router.split_layer_terminal_pairs(terminal_pairs)
vdp_layers, _ = router.find_total_vdp_layers_dyn(layers, data_qubit_locs, router.factory_times, layout, testing = True)
print(vdp_layers)

print("Len of schedule without teleportation: ", len(vdp_layers))


router = utils.TeleportationRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag = True, seed = 49218)
layers = router.split_layer_terminal_pairs(terminal_pairs)

max_iters = 100
T_start = 100.0
T_end = 0.1
alpha = 0.95
t=4 #mock value for cnot circuit
radius = 3
k_lookahead = 1
metric = "exact"

steiner_init_type = "full_random"
jump_harvesting = True
stimtest = True

reduce_steiner = True
idle_move_type = "later"

schedule, _ = router.optimize_layers(        
        terminal_pairs,
        layout,
        max_iters,
        T_start,
        T_end,
        alpha,
        radius = radius,
        k_lookahead = k_lookahead,
        steiner_init_type = steiner_init_type,
        jump_harvesting = jump_harvesting,
        reduce_teleport = reduce_steiner,
        idle_move_type = idle_move_type,
        include_steiner_teleport = True,
        include_idle_teleport = True,
        reduce_init_steiner = True,
        reduce_init_idle = True,
        stimtest = True, 
)

print("Len of schedule with teleport router: ", len(schedule))
print("Reduction Delta: ", len(vdp_layers) - len(schedule))




from IPython.display import HTML
from cococo.animation_routing_html import make_clean_routing_html_animation
from datetime import datetime
import matplotlib as mpl 

mpl.rcParams["animation.embed_limit"] = 100  # MB
Path("animation/toy_model").mkdir(exist_ok=True)
filename = f"animation/toy_model/animation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"

anim = make_clean_routing_html_animation(
    g,
    schedule,
    initial_layout=layout,
    factories=factories,
    figsize=(18, 8),
    interval=900,
    save_path = filename, 
)

HTML(anim.to_jshtml())
