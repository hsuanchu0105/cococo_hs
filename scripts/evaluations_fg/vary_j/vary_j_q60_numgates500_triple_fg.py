
from pathlib import Path 
import sys 

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


# ----params-----

factories = []
m, n = 2, 5
layout_type = "triple"
g, data_qubit_locs, _ = layouts.gen_layout_scalable(
    layout_type, m, n, factories, remove_edges=False
)  #!important, no edges removed here!!!
layout = {i: j for i, j in enumerate(data_qubit_locs)}

q = len(data_qubit_locs)
print("q = ", len(data_qubit_locs))

n_circ = 1   # go through 5 circuits

j_lst = [5, 15]
#j_lst = [2, 3, 4, 5, 10, 15, 20, 25, 30]
j_lst_str = [str(el) for el in j_lst]
j_str = "_".join(map(str, j_lst))

num_gates = 500

use_dag = True

reps = 20

date_str = datetime.now().strftime("%Y-%m-%d")
# outputs land next to this script, independent of the job's working directory
out_dir = Path(__file__).parent
filename = f"varyj_layouttype{layout_type}_q{q}_m{m}_n{n}_j{j_str}_numgates{num_gates}_dvaries_ncirc{n_circ}_{date_str}_usedag{use_dag}.pkl"
path = out_dir / filename
pdf_name = filename.replace(".pkl", ".pdf")

# ------------params fg-----------------
valid_path = "cc"

t = 4  # mock value for cnot circuit
metric = "exact"
overlap_type = "strict_k"

testing = True

# -------------runs------------

results_fg = []
results_st = []

start = time.time()
for j in j_lst:
    print(f"=======j={j}, num_gates={num_gates}======")
    # load example circuits and cut off ncirc
    d = np.ceil(
        num_gates / j
    )  # round up because last layer will not be full of j gates but be a layer nevertheless.
    path_circuits = out_dir / f"true_seq_circs_j{j}_q{q}_numgates{num_gates}d{d}_x{reps}.json"
    try:
        with open(path_circuits, "r") as f:
            pairs_lst = json.load(f)
    except FileNotFoundError:
        print("new circs sampled")
        pairs_lst = []
        for r in range(reps):
            dag, pairs = circuit_construction.create_random_sequential_circuit_dag(
                j, q, num_gates, seed=j * 1000 + r
            )
            pairs_lst.append(pairs)
        with open(path_circuits, "w") as f:
            json.dump(pairs_lst, f)
    # turn into tuples again
    pairs_lst = [[(el[0], el[1]) for el in pairs] for pairs in pairs_lst]
    pairs_lst = pairs_lst[:n_circ]

    results_fg_temp = []
    results_st_temp = []

    for i, pairs in enumerate(pairs_lst):
        print(f"-------i={i}--------")
        # into terminal pairs
        terminal_pairs = layouts.translate_layout_circuit(pairs, layout)

        # check logical depth
        dag = dag_helper.terminal_pairs_into_dag(terminal_pairs, layout)
        layers = []
        for layer in range(len(list(dag.layers()))):
            layers.append(dag_helper.extract_layer_from_dag(dag, layout, layer))
        if len(layers) != d:
            raise ValueError(
                f"The number of logical layers {len(layers)} does not coincide with desired depth {d}"
            )

        # run standard
        quilt = utils.BasicRouter(
            g,
            data_qubit_locs,
            factories,
            valid_path,
            t,
            metric,
            use_dag=use_dag,
        )  # reinitialize because logical pos changes
        layers = quilt.split_layer_terminal_pairs(terminal_pairs)
        vdp_layers, _ = quilt.find_total_vdp_layers_dyn(
            layers, data_qubit_locs, {}, layout
        )
        results_st_temp.append(vdp_layers)

        # run fg
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
            quilt.split_layer_terminal_pairs(terminal_pairs),
            None,
            None,
            layout=layout,
            overlap_type=overlap_type,
            testing=testing,
        )
        fine_routes = {
            idx: dict(routes) for idx, routes in quilt.routes_by_layer.items()
        }
        results_fg_temp.append(fine_routes)

    results_fg.append(results_fg_temp)
    results_st.append(results_st_temp)
    save = [results_fg, results_st]
    with open(path, "wb") as f:
        pickle.dump(save, f)

end = time.time()
print(f"Simulation took {(end-start)/60} minutes")

print(path)
# reload
with open(path, "rb") as f:
    saved = pickle.load(f)
[results_fg, results_st] = saved

# ---------extract data-----------

# compute mean and std of standard approach
depths_st = [[len(el) for el in sublist] for sublist in results_st]
depths_mean_st = [np.mean(lst) for lst in depths_st]
depths_std_st = [np.std(lst) for lst in depths_st]
print("depths_mean_st", depths_mean_st)

# fg
depths_fg = [[len(el) for el in sublist] for sublist in results_fg]
depths_mean_fg = [np.mean(lst) for lst in depths_fg]
depths_std_fg = [np.std(lst) for lst in depths_fg]

print("depths_mean_fg", depths_mean_fg)

deltas_lst = [
    [st - fg for st, fg in zip(sublist_st, sublist_fg)]
    for sublist_st, sublist_fg in zip(depths_st, depths_fg)
]
deltas_mean = [np.mean(lst) for lst in deltas_lst]
deltas_std = [np.std(lst) for lst in deltas_lst]

print("deltas mean", deltas_mean)

# ---------plot------------
size = (3.5, 2.5)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 10

plt.figure(figsize=size)
plt.errorbar(
    j_lst,
    depths_mean_fg,
    yerr=depths_std_fg,
    fmt="*-",
    color="lightseagreen",
    capsize=3,
    label=r"$\tilde{d}_{\mathrm{fg}}$",
)
plt.errorbar(
    j_lst,
    depths_mean_st,
    yerr=depths_std_st,
    fmt=".-",
    color="darkorchid",
    capsize=3,
    label=r"$\tilde{d}_{\mathrm{st}}$",
)

plt.xlabel("$g$")
plt.ylabel(r"$\tilde{d}$")

# plt.xscale("log", base=2)

plt.grid(True, which="both", ls="--", alpha=0.7)
plt.xticks(j_lst, j_lst_str)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.legend()
plt.tight_layout()

plt.savefig(
    out_dir / ("plot_total_" + pdf_name),
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
)
plt.clf()

# -------------

plt.figure(figsize=size)
plt.errorbar(j_lst, deltas_mean, yerr=deltas_std, fmt=".-", capsize=3)

# plt.xlabel("Num. gates per logical layer $j$")
plt.xlabel("$g$")
plt.ylabel(r"$\Delta = \tilde{d}_{st}-\tilde{d}_{fg}$")

# plt.xscale("log", base=2)

plt.grid(True, which="both", ls="--", alpha=0.7)
plt.xticks(j_lst, j_lst_str)
# plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
# plt.legend()
plt.tight_layout()

plt.savefig(
    out_dir / ("plot_absdelta_" + pdf_name),
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
)
plt.clf()
