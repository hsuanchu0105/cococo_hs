import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))


import cococo.layouts as layouts
import cococo.utils_routing as utils
import cococo.circuit_construction as circuit_construction
import cococo.internal_testing as internal_testing
import plotting



layout_type = "hex"
m = 4
n = 4
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
layout = {i: j for i,j in enumerate(data_qubit_locs)}
t=2

#print("layout: ", layout)

#print(g)
#print("data qubit location", data_qubit_locs)

#print("factory ring: ", factory_ring)

plotting.plot_lattice_paths(g, {}, {}, layout, factories, size = (18,8))


q = len(data_qubit_locs)
print("number of data qubits: ", q)
j = 8
num_gates = q*2


# j gates per layer on q qubits 
# pairs indicate the qubit index (0, ..., q)
dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates, ) # at least num_gates gates
#print("pairs: ", pairs)
print("number of gates: ", len(pairs))

# terminal pairs indicate the 2d coordinates 
terminal_pairs = layouts.translate_layout_circuit(pairs, layout) #let's stick to the simple layout
#print("terminal pairs: ", terminal_pairs)

router = utils.BasicRouter(g, data_qubit_locs, factories, valid_path = "cc", t=t, metric = "exact", use_dag = True)
# each layer has disjoint logical support, however it doesn't guarantee that all those gates can be physically routed at the same time on the lattice
layers = router.split_layer_terminal_pairs(terminal_pairs)
vdp_layers, _ = router.find_total_vdp_layers_dyn(layers, data_qubit_locs, router.factory_times, layout, testing = True)
print("Len of schedule without teleportation: ", len(vdp_layers))

#print_vdp_layers_with_qubit_labels(vdp_layers, layout)

router = utils.TeleportationRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag = True, seed =  49218  )
layers = router.split_layer_terminal_pairs(terminal_pairs)

max_iters = 100
T_start = 100.0
T_end = 0.1
alpha = 0.95
t=4 #mock value for cnot circuit
radius = 10
k_lookahead = 5
metric = "exact"

steiner_init_type = "full_random"
jump_harvesting = True
stimtest = True

reduce_teleport = True
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
        reduce_teleport = reduce_teleport,
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
Path("animation").mkdir(exist_ok=True)
filename = f"animation/single_both_j8_flip_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"

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
