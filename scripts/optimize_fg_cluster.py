import sys
import os
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))



from datetime import datetime
import cococo.layouts as layouts
import cococo.utils_routing as utils
import cococo.circuit_construction as circuit_construction
import cococo.internal_testing as internal_testing
import cococo.dag_helper as dag_helper
import argparse

layout_type = "triple"
m = 8
n = 8
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
layout = {i: j for i,j in enumerate(data_qubit_locs)}
t=2


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

seed = args.seed

q = len(data_qubit_locs)
print("number of data qubits: ", q)
j = 20 
num_gates = 8 * q


# j gates per layer on q qubits 
# pairs indicate the qubit index (0, ..., q)
dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates, seed) # at least num_gates gates
#print("pairs: ", pairs)
print("number of gates: ", len(pairs))

# terminal pairs indicate the 2d coordinates 
terminal_pairs = layouts.translate_layout_circuit(pairs, layout) #let's stick to the simple layout
#print("terminal pairs: ", terminal_pairs)

router1 = utils.BasicRouter(g, data_qubit_locs, factories, valid_path = "cc", t=t, metric = "exact", use_dag = True)
# each layer has disjoint logical support, however it doesn't guarantee that all those gates can be physically routed at the same time on the lattice
layers = router1.split_layer_terminal_pairs(terminal_pairs)
vdp_layers, _ = router1.find_total_vdp_layers_dyn(layers, data_qubit_locs, router1.factory_times, layout, testing = True)
print("Len of schedule (standard): ", len(vdp_layers))

#print_vdp_layers_with_qubit_labels(vdp_layers, layout)

max_overlap = 10

router2 = utils.TeleportationRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag = True, seed =  49218  )
router2.find_total_fine_grained_vdp_dyn(layers, data_qubit_locs, None, layout = layout, max_overlap = max_overlap, overlap_type = "strict_k", testing = True)
print("Len of schedule (fine grained): ", len(router2.routes_by_layer))



router3 = utils.TeleportationRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag = True, seed =  49218  )

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

output_dir = Path("/dss/dsshome1/0B/go49xav2/MA/Results")
schedule_dir = output_dir / "schedule"
schedule_dir.mkdir(parents=True, exist_ok=True)
#filename = schedule_dir / f"schedule_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"

task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "noarray")
job_id = os.environ.get("SLURM_JOB_ID", "nojid")

filename = (
    schedule_dir
    / f"schedule_fine-grained_layout-{layout_type}_j-{j}_seed{args.seed}_job-{job_id}_task-{task_id}_"
      f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
)

schedule_fine, _ = router3.optimize_layers(        
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

router4 = utils.TeleportationRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag = True, seed =  49218  )

filename = (
    schedule_dir
    / f"schedule_coarse_grain_layout-{layout_type}_j-{j}_seed{args.seed}_job-{job_id}_task-{task_id}_"
      f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
)

schedule_coarse, _ = router4.optimize_layers(
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
        vdp_type = "coarse_grained",
        overlap_type = None,
        max_overlap = max_overlap,
        filename = filename,
        include_steiner_teleport = True,
        include_idle_teleport = False,
        reduce_init_steiner = True,
        reduce_init_idle = False,
        stimtest = True,
)

# reduction of schedule length vs the non-fine-grained (coarse VDP) schedule
baseline_len = len(vdp_layers)
fg_len = len(router2.routes_by_layer)
fg_sa = len(schedule_fine)
cs_sa = len(schedule_coarse)
reduction_fg = (baseline_len - fg_len) / baseline_len * 100 if baseline_len else 0.0
reduct_cssa = (baseline_len - cs_sa) / baseline_len * 100 if baseline_len else 0.0
reduct_fgsa = (baseline_len - fg_sa) / baseline_len * 100 if baseline_len else 0.0


# machine-parseable line for the benchmark harness to collect
print(f"BENCH_RESULT seed={seed} gates={len(terminal_pairs)} "
      f"standard vdp={baseline_len} fine-grained={fg_len} coarse + SA = {cs_sa} fine-grained + SA = {fg_sa} "
      f"red_fg={reduction_fg:.2f} red_fg+sa={reduct_fgsa:.2f} "
      )
