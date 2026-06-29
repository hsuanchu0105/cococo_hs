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

from datetime import datetime

seed = 45

layout_type = "single"
m = 4
n = 4
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
layout = {i: j for i,j in enumerate(data_qubit_locs)}
t=2

#plotting.plot_lattice_paths(g, {}, {}, layout, factories, size = (18,8))


q = len(data_qubit_locs)
print("number of data qubits: ", q)
j = 8
num_gates = 2 * q


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
paths = router.find_fine_grained_vdp(layers[0], 0, None, None)

for key, route in router.routes.items():
    print("gate: ", key[1])
    print("path : ", route["path"], "\n")
    print("ancilla : ", route["ancilla"], "\n")
    print("subpath1 : ", route["subpath1"], "\n")
    print("subpath2 : ", route["subpath2"], "\n")
#print(paths)

plot_fine_routes(g, router.routes, layer_idx=0)