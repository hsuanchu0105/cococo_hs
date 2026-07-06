import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))


import cococo.layouts as layouts
import cococo.utils_routing as utils
#import cococo.SA_mod as utils
import cococo.circuit_construction as circuit_construction
import cococo.internal_testing as internal_testing
import plotting
from cococo.animations import plot_fine_routes
from cococo.animations import animate_fine_routes
import cococo.dag_helper as dag_helper

from datetime import datetime

seed = 45

layout_type = "single"
m = 4
n = 8
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
layout = {i: j for i,j in enumerate(data_qubit_locs)}
t=2

#plotting.plot_lattice_paths(g, {}, {}, layout, factories, size = (18,8))


q = len(data_qubit_locs)
print("number of data qubits: ", q)
j = 8
num_gates = 4 * q


# j gates per layer on q qubits 
# pairs indicate the qubit index (0, ..., q)
dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates, seed) # at least num_gates gates
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

i = 0
"""
while i < len(layers):
    terminal_pairs_remainder = router.find_fine_grained_vdp(
        i,
        layers[i],
        None,
        None,
        "strict2",
    )

    if router.use_dag:
        next_layer_update, dag = dag_helper.push_remainder_into_layers_dag(
            dag,
            terminal_pairs_remainder,
            layout,
            layers[i],
        )

        # You need to decide how next_layer_update corresponds to layers.
        # If it represents the remaining future layers, assign it carefully.
        layers = layers[: i+1] + next_layer_update

    else:
        if terminal_pairs_remainder:
            future_layers_update = router.push_remainder_into_layers(
                layers[i+1 :],
                terminal_pairs_remainder,
                delete_layer_zero=False,
            )

            layers = layers[: i+1] + future_layers_update

    i += 1

"""
router.find_total_fine_grained_vdp_dyn(layers, None, None, layout = layout, overlap_type = "strict_k", testing = True)
#print(router.overlap_graphs.values())
#print(router.routes_by_layer)


    

print("Len of schedule (fine grained): ", len(router.routes_by_layer))

#plot_fine_routes(g, router.routes_by_layer)

Path("animation").mkdir(exist_ok=True)
filename = f"../../Output_Files/animation/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"

animate_fine_routes(
    graph=router.g,
    routes_by_layer=router.routes_by_layer,
    interval=800,
    pause_between_layers=0,
    save_path=filename, 
    figsize=(18, 8),
)

