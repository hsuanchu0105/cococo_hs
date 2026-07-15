import sys
from pathlib import Path

# must come before the cococo imports below, or they cannot be resolved
# scripts/evaluations_fg/fix_q_vary_depth/ -> repo root is 3 levels up
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

import cococo.utils_routing as utils
import cococo.layouts as layouts
import cococo.internal_testing as tst
import cococo.circuit_construction as circuit_construction
import cococo.dag_helper as dag_helper
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pickle

from datetime import datetime
import time

import json

# -----params-------

j = 8
q = 120
d = 320
num_gates = 2560
# The JSON holds reps = 20 independent random circuits of 2560 gates each
reps = 20


# ------load circuits------
circ_dir = Path(__file__).parent  # json sits next to this script
circ_path = circ_dir / f"true_seq_circs_j{j}_q{q}_numgates{num_gates}d{d}_x{reps}.json"
with open(circ_path, "r") as f:
    pairs_lst = json.load(f)
# turn into tuples again
pairs_lst = [[(el[0], el[1]) for el in pairs] for pairs in pairs_lst]

# ------geometry params---------
factories = []
m, n = 4, 5
layout_type = "hex"
g, data_qubit_locs, _ = layouts.gen_layout_scalable(
    layout_type, m, n, factories, remove_edges=False
)  #!important, no edges removed here!!!
layout = {i: j for i, j in enumerate(data_qubit_locs)}

assert q == len(data_qubit_locs), "given q does not coincide with your chosen layout"

# how many of the 20 circuits in the JSON you actually use
n_circ = 10

gates_list = [320, 640, 1280, 2560]
depths_list = [40, 80, 160, 320]
for num_gates, depth in zip(gates_list, depths_list):
    assert num_gates == depth * j, "depths list and gates_list does not coincide"
gates_str = "_".join(map(str, gates_list))

use_dag = True

# -----params fine-grained-------

valid_path = "cc"

t = 4  # mock value for cnot circuit
metric = "exact"
overlap_type = "strict_k"

testing = True

date_str = datetime.now().strftime("%Y-%m-%d")
# outputs land next to this script, independent of the working directory
out_dir = Path(__file__).parent
filename = f"circuit_depths_fg_m{m}_n{n}_layout{layout_type}_ncirc{n_circ}_num_gates{gates_str}_overlap{overlap_type}_{date_str}_usedag{use_dag}_p.pkl"
path = out_dir / filename
pdf_name = filename.replace(".pkl", ".pdf")

terminal_pairs_list = []
for pairs in pairs_lst[:n_circ]:
    print("Number of gates", len(pairs))
    terminal_pairs = layouts.translate_layout_circuit(pairs, layout)
    terminal_pairs_list.append(terminal_pairs)

#print("terminal pairs lst")
#for el in terminal_pairs_list:
#    print(el)


# -----------run--------------
results_list_st = (
    []
)  # list of sublists, in each sublist results of each of the num_circs
results_list_fg = []
layers_tot = []
# run layer for layer
start = time.time()
for num_gates in gates_list:
    print("num_gates: ", num_gates)
    results_temp_st = []
    results_temp_fg = []
    layers_temp = []
    for circ in terminal_pairs_list:
        temp_circuit = circ[:num_gates]  # cut off the gates

        # standard
        quilt = utils.BasicRouter(
            g,
            data_qubit_locs,
            factories,
            valid_path,
            t,
            metric,
            use_dag=use_dag,
        )  # reinitialize because logical pos changes
        layers = quilt.split_layer_terminal_pairs(temp_circuit)
        vdp_layers, _ = quilt.find_total_vdp_layers_dyn(
            layers, data_qubit_locs, {}, layout
        )
        results_temp_st.append(vdp_layers.copy())

        # log
        dag = dag_helper.terminal_pairs_into_dag(temp_circuit, layout)
        log_layers = []
        for layer in range(len(list(dag.layers()))):
            log_layers.append(dag_helper.extract_layer_from_dag(dag, layout, layer))
        layers_temp.append(log_layers.copy())

        # fine grained
        quilt = utils.BasicRouter(
            g,
            data_qubit_locs,
            factories,
            valid_path,
            t,
            metric,
            use_dag=use_dag,
        )
        # writes its schedule into quilt.routes_by_layer, returns nothing
        quilt.find_total_fine_grained_vdp_dyn(
            quilt.split_layer_terminal_pairs(temp_circuit),
            None,
            None,
            layout=layout,
            overlap_type=overlap_type,
            testing=testing,
        )
        fine_routes = {
            idx: dict(routes) for idx, routes in quilt.routes_by_layer.items()
        }
        print("fine grained depth: ", len(fine_routes))
        results_temp_fg.append(fine_routes)

    results_list_st.append(results_temp_st)
    results_list_fg.append(results_temp_fg)
    layers_tot.append(layers_temp)

    save = [results_list_st, results_list_fg, layers_tot]
    with open(path, "wb") as f:
        pickle.dump(save, f)
end = time.time()
print("total runtime: ", end - start)


# reload
with open(path, "rb") as f:
    saved = pickle.load(f)
[results_list_st, results_list_fg, layers_tot] = saved


# ---------extract data-----------

# compute mean and std of standard approach
depths_st = [[len(el) for el in sublist] for sublist in results_list_st]
depths_mean_st = [np.mean(lst) for lst in depths_st]
depths_std_st = [np.std(lst) for lst in depths_st]
print("depths_mean_st", depths_mean_st)

# logical
depths_log = [[len(el) for el in sublist] for sublist in layers_tot]
depths_mean_log = [np.mean(lst) for lst in depths_log]
depths_std_log = [np.std(lst) for lst in depths_log]

print("depths_mean_log", depths_mean_log)


labels = [f"({g},\n {d})" for g, d in zip(gates_list, depths_list)]


# mean and std of the fine-grained approach
# the routine is deterministic, so there is one schedule per circuit (no sigma runs)
depths_fg = [[len(routes) for routes in sublist] for sublist in results_list_fg]
depths_mean_fg = [np.mean(lst) for lst in depths_fg]
depths_std_fg = [np.std(lst) for lst in depths_fg]
print("depths_mean_fg", depths_mean_fg)

# figsize
size = (4, 3)


# plot total abs

plt.figure(figsize=size)
plt.errorbar(
    gates_list,
    depths_mean_fg,
    yerr=depths_std_fg,
    fmt="*-",
    color="lightseagreen",
    capsize=3,
    label="Depth Fine-grained",
)
plt.errorbar(
    gates_list,
    depths_mean_st,
    yerr=depths_std_st,
    fmt=".-",
    color="darkorchid",
    capsize=3,
    label="Depth St.",
)
plt.errorbar(
    gates_list,
    depths_mean_log,
    yerr=depths_std_log,
    fmt=".--",
    color="yellowgreen",
    capsize=3,
    label="Depth Log.",
)
plt.xlabel("Num. Gates / Log. Depth")
plt.ylabel("Depth of Schedule")
plt.xscale("log")
plt.grid(True, which="both", ls="--", alpha=0.7)
# plt.xticks(gates_list, gates_list)
plt.xticks(gates_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.legend()
plt.tight_layout()

plt.savefig(out_dir / ("plot_total_" + pdf_name))
plt.clf()


# plot improvement d fg - c / d st -c
improvements_mean = []
improvements_std = []
for lst_fg, lst_st, c_list in zip(depths_fg, depths_st, depths_log):
    temp = []
    for el_fg, el_st, c in zip(lst_fg, lst_st, c_list):
        if el_st == c:
            # standard routing already achieves the logical depth, ratio undefined
            continue
        temp.append((el_fg - c) / (el_st - c))
    print("improvements temp", temp)
    improvements_mean.append(np.mean(temp) if temp else np.nan)
    improvements_std.append(np.std(temp) if temp else np.nan)

plt.figure(figsize=size)
plt.errorbar(gates_list, improvements_mean, yerr=improvements_std, fmt=".-", capsize=3)
plt.ylabel(r"$\frac{d_{fg}-c}{d_{st}-c}$")
plt.xlabel("Num. Gates / Log. Depth")
plt.xscale("log")
plt.grid(True, which="both", ls="--", alpha=0.7)
# plt.xticks(gates_list, gates_list)
plt.xticks(gates_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.tight_layout()
plt.savefig(out_dir / ("plot_improvements_total_" + pdf_name))
plt.clf()

# plot absolute differences
diff_fg_mean = []
diff_fg_std = []
for lst_fg, lst_st in zip(depths_fg, depths_st):
    differences = [el_st - el_fg for el_st, el_fg in zip(lst_st, lst_fg)]
    print("abs differences", differences)
    diff_fg_mean.append(np.mean(differences))
    diff_fg_std.append(np.std(differences))
plt.figure(figsize=size)
plt.errorbar(gates_list, diff_fg_mean, yerr=diff_fg_std, fmt=".-", capsize=3)
plt.ylabel(r"$\Delta = d_{st} - d_{fg}$")
plt.xlabel("Num. Gates / Log. Depth")
plt.xscale("log")
# plt.xticks(gates_list, gates_list)
plt.xticks(gates_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig(out_dir / ("plot_abs_reductions_total_" + pdf_name))
