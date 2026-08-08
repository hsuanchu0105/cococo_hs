import sys
from pathlib import Path

import matplotlib as mpl
#mpl.use("Agg")  # headless: this script only saves animations, never shows a GUI window

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

import cococo.layouts as layouts
import cococo.utils_routing as utils
import cococo.circuit_construction as circuit_construction
import cococo.internal_testing as internal_testing
from cococo.animations import make_fine_sa_routing_animation

from datetime import datetime

from IPython.display import HTML
from cococo.animations import make_clean_routing_html_animation

layout_type = "single"
m = 4
n = 4
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
layout = {i: j for i,j in enumerate(data_qubit_locs)}
t=2


layout_rev = {j:i for i,j in enumerate(data_qubit_locs)}
def qubit(pos, schedule = None):
    if schedule is not None:
        ly_rev = {j: i for i,j in schedule["layout"].items()}
        return ly_rev[pos]
    return layout_rev[pos]



q = len(data_qubit_locs)
print("number of data qubits: ", q)
j = 8
num_gates = 4 * q
max_overlap = 5

seed = 1
dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates, seed) # at least num_gates gates
#print("pairs: ", pairs)
print("number of gates: ", len(pairs))


terminal_pairs = layouts.translate_layout_circuit(pairs, layout)
#print("terminal pairs: ", terminal_pairs)

router = utils.BasicRouter(g, data_qubit_locs, factories, valid_path = "cc", t=t, metric = "exact", use_dag = True)
# each layer has disjoint logical support, however it doesn't guarantee that all those gates can be physically routed at the same time on the lattice
layers = router.split_layer_terminal_pairs(terminal_pairs)
vdp_layers, _ = router.find_total_vdp_layers_dyn(layers, data_qubit_locs, router.factory_times, layout, testing = True)



for i, layer in enumerate(layers):
    #print(i, layer, "\n")
    layer_trans =[]
    for gate in layer:
        layer_trans.append((qubit(gate[0]), qubit(gate[1])))
    #print(i, layer_trans , "\n")



print("Len of schedule (standard): ", len(vdp_layers))

router2 = utils.TeleportationRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag = True, seed =  49218  )
router2.find_total_fine_grained_vdp_dyn(layers, data_qubit_locs, None, layout = layout, max_overlap = max_overlap, overlap_type = "strict_k", testing = True)
print("Len of schedule (fine grained): ", len(router2.routes_by_layer))

max_iters = 100
T_start = 100.0
T_end = 0.1
alpha = 0.95
t=4 #mock value for cnot circuit
radius = 10
k_lookahead = 5
max_idle_teleport = 10
metric = "exact"

steiner_init_type = "full_random"
jump_harvesting = True
stimtest = True

reduce_teleport = False
idle_move_type = "later"

filename = f'../../Output_Files/schedule/schedule_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'


router4 = utils.TeleportationRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag = True, seed =  49218  )
schedule, _ = router4.optimize_layers(
        terminal_pairs,
        layout,
        max_iters,
        T_start,
        T_end,
        alpha,
        radius = radius,
        k_lookahead = k_lookahead,
        max_idle_teleport = max_idle_teleport,
        steiner_init_type = steiner_init_type,
        jump_harvesting = jump_harvesting,
        reduce_teleport = reduce_teleport,
        idle_move_type = idle_move_type,
        vdp_type = "fine_grained",
        overlap_type =  "strict_k",
        max_overlap = max_overlap,
        filename = filename,
        include_steiner_teleport = True,
        include_idle_teleport = False,
        reduce_init_steiner = True,
        reduce_init_idle = False,
        stimtest = True,
)


for i, sch in enumerate(schedule):
    print("Layer ", i+1, " : ")
    print("Steiner: ", sch['steiner'])
    print("Idle telep: ", sch['idle_teleport'])
    print("Fine routes: ")
    for gate, route_info in sch['fine_routes'].items():
        print(qubit(gate[0], sch), qubit(gate[1], sch))
        #print(gate)



print("Len of schedule (fg+sa): ", len(schedule))
print("Reduction Delta: ", len(vdp_layers) - len(schedule))


Path("animation").mkdir(exist_ok=True)
filename = f"../../Output_Files/animation/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"

#combined animation: two-phase routing (frame A) + teleportation (frame B) per layer
make_fine_sa_routing_animation(
    router.g,
    schedule,
    initial_layout=layout,
    factories=factories,
    interval=900,
    save_path=filename,
    figsize=(18, 8),
)
