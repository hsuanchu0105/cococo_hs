import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))


import cococo.layouts as layouts
import cococo.utils_routing as utils
import cococo.circuit_construction as circuit_construction
import cococo.internal_testing as internal_testing
import plotting



layout_type = "triple"
m = 4
n = 4
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
layout = {i: j for i,j in enumerate(data_qubit_locs)}
t=2 



print(g)
#print(data_qubit_locs)
print("factory ring: ", factory_ring)

#plotting.plot_lattice_paths(g, {}, {}, layout, factories, size = (18,8))


q = len(data_qubit_locs)
j = 8
num_gates = q*2
# j gates per layer on q qubits 
# pairs indicate the qubit index (0, ..., q)
dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates, )
print("pairs: ", pairs)

# terminal pairs indicate the 2d coordinates 
terminal_pairs = layouts.translate_layout_circuit(pairs, layout) #let's stick to the simple layout
print("terminal pairs: ", terminal_pairs)

router = utils.BasicRouter(g, data_qubit_locs, factories, valid_path = "cc", t=t, metric = "exact", use_dag = True)
layers = router.split_layer_terminal_pairs(terminal_pairs)
vdp_layers, _ = router.find_total_vdp_layers_dyn(layers, data_qubit_locs, router.factory_times, layout, testing = True)
print("Len of schedule without teleportation: ", len(vdp_layers))