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


# ----------determine parameters to load correct file-------------
q = 120
j = 8

factories = []
m, n = 4, 5  # 4, 5  # 10, 6#10, 12 #4,5
layout_type = "hex"
g, data_qubit_locs, _ = layouts.gen_layout_scalable(
    layout_type, m, n, factories, remove_edges=False
)  #!important, no edges removed here!!!
layout = {i: j for i, j in enumerate(data_qubit_locs)}

assert q == len(data_qubit_locs), "given q does not coincide with your chosen layout"

sigma = 1
n_circ = 10

gates_list = [320, 640, 1280, 2560]  # [300, 600, 1200, 2400]#[320, 640, 1280,2560]
depths_list = [40, 80, 160, 320]  # [20, 40, 80, 160]#[40, 80, 160, 320]

for num_gates, depth in zip(gates_list, depths_list):
    assert num_gates == depth * j, "depths list and gates_list does not coincide"
gates_str = "_".join(map(str, gates_list))

use_dag = True

date_str = datetime.now().strftime("%Y-%m-%d")
date_str = "2025-07-15"
# path = f"circuit_depths_m{m}_n{n}_layout{layout_type}_sigma{sigma}_ncirc{n_circ}_num_gates{gates_str}_j{j}_{date_str}_usedag{use_dag}_p.pkl"
#path = f"circuit_depths_m{m}_n{n}_layout{layout_type}_sigma{sigma}_ncirc{n_circ}_num_gates{gates_str}_{date_str}_usedag{use_dag}_p.pkl"
#path = f"circuit_depths_fg_m4_n5_layouthex_ncirc10_num_gates320_640_1280_2560_overlapstrict_k_2026-07-14_usedagTrue_p.pkl"
#path = f"circuit_depths_fg_m10_n6_layoutsingle_ncirc10_num_gates320_640_1280_2560_overlapstrict_k_2026-07-15_usedagTrue_p.pkl"
#path = f"circuit_depths_fg_m4_n4_layouthex_ncirc10_num_gates320_640_1280_2560_overlapstrict_k_2026-07-15_usedagTrue_p.pkl"
#path = f"circuit_depths_fg_m4_n4_layouttriple_ncirc10_num_gates320_640_1280_2560_overlapstrict_k_2026-07-15_usedagTrue_p.pkl"

#path = f"circuit_depths_fg_m6_n8_layoutpair_ncirc10_num_gates320_640_1280_2560_overlapstrict_k_2026-07-15_usedagTrue_p.pkl"
path = f"circuit_depths_fg_m8_n12_layoutsingle_ncirc10_num_gates320_640_1280_2560_overlapstrict_k_2026-07-15_usedagTrue_p.pkl"
print("path: ", path)


# ---------plots-------------


# reload
with open(path, "rb") as f:
    saved = pickle.load(f)
[results_list_st, results_list_opt,  layers_tot] = saved


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


labels = [f"({g}, {d})" for g, d in zip(gates_list, depths_list)]


# fine-grained: one deterministic schedule per circuit (no sigma runs).
# each entry is a routes_by_layer dict, so its len is the schedule depth.
depths_fg = [[len(routes) for routes in sublist] for sublist in results_list_opt]
depths_mean_opt = [np.mean(lst) for lst in depths_fg]
depths_std_opt = [np.std(lst) for lst in depths_fg]
print("depths_mean_opt", depths_mean_opt)

# figsize
size = (4.5, 3.2)


# plot total abs

plt.figure(figsize=size)
plt.errorbar(
    depths_list,
    depths_mean_opt,
    yerr=depths_std_opt,
    fmt="*",
    color="lightseagreen",
    capsize=3,
    label="Depth Fg.",
)
plt.errorbar(
    depths_list,
    depths_mean_st,
    yerr=depths_std_st,
    fmt=".",
    color="darkorchid",
    capsize=3,
    label="Depth St.",
)
plt.errorbar(
    depths_list,
    depths_mean_log,
    yerr=depths_std_log,
    fmt="v",
    color="yellowgreen",
    capsize=3,
    label="Depth Log.",
)

# --- Linear fits ----------------------------------------------------------

# Convert to numpy arrays for fitting
x = np.array(depths_list)  #!Plot v.s. depths or gates?
y_opt = np.array(depths_mean_opt)
y_st = np.array(depths_mean_st)
y_log = np.array(depths_mean_log)

# Fit: y = a*x + b
fit_opt = np.polyfit(x, y_opt, 1)
fit_st = np.polyfit(x, y_st, 1)
fit_log = np.polyfit(x, y_log, 1)

# Generate smooth x range for fit lines
x_fit = np.linspace(min(x), max(x), 200)

# Evaluate fits
y_fit_opt = np.polyval(fit_opt, x_fit)
y_fit_st = np.polyval(fit_st, x_fit)
y_fit_log = np.polyval(fit_log, x_fit)

# --- Plot fits (in shades of grey) ----------------------------------------

plt.plot(
    x_fit,
    y_fit_opt,
    color="lightseagreen",
    linestyle="-",
    alpha=0.7,
    label=rf"Fit Fg: $y={fit_opt[0]:.1f}x {fit_opt[1]:+.1f}$",
)

plt.plot(
    x_fit,
    y_fit_st,
    color="darkorchid",
    linestyle="-",
    alpha=0.7,
    label=rf"Fit St: $y={fit_st[0]:.1f}x {fit_st[1]:+.1f}$",
)

plt.plot(
    x_fit,
    y_fit_log,
    color="yellowgreen",
    linestyle="-",
    alpha=0.7,
    label = rf"Fit Log: $y={fit_log[0]:.1f}x {fit_log[1]:+.1f}$",
)


plt.xlabel("Num. Gates / Log. Depth")
plt.ylabel("Depth of Schedule")
plt.xscale("log")
plt.yscale("log")

# yticks_temp = [100,200,300,400,500,600,700] #!ADAPT MANUALLY
# plt.yticks(yticks_temp, [str(y) for y in yticks_temp])  # custom text labels
# plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())

plt.grid(True, which="both", ls="--", alpha=0.7)
# plt.xticks(gates_list, gates_list)
plt.xticks(depths_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.legend(fontsize=7)
plt.tight_layout()

plt.rcParams['savefig.dpi'] = 300
plt.savefig("plot_total_" + path.replace(".pkl", "_with_fit.jpg"))
plt.clf()


# plot improvement d opt - c / d st -c
improvements_mean = []
improvements_std = []
for lst_opt, lst_st, c_list in zip(depths_fg, depths_st, depths_log):
    temp = []
    for el_opt, el_st, c in zip(lst_opt, lst_st, c_list):
        if el_st == c:
            # standard routing already achieves the logical depth, ratio undefined
            continue
        temp.append((el_opt - c) / (el_st - c))
        print("el opt", el_opt)
        print("el st", el_st)
        print("c", c)
    print("improvements temp", temp)
    improvements_mean.append(np.mean(temp))
    improvements_std.append(np.std(temp))
    print("improvement mean", np.mean(temp))
    print("improvements std", np.std(temp))

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
plt.savefig("plot_improvements_total_" + path.replace(".pkl", ".pdf"))
plt.clf()

# plot absolute differences
diff_opt_mean = []
diff_opt_std = []
for lst_opt, lst_st in zip(depths_fg, depths_st):
    differences = [el_st - el_opt for el_st, el_opt in zip(lst_st, lst_opt)]
    print("abs differences", differences)
    diff_opt_mean.append(np.mean(differences))
    diff_opt_std.append(np.std(differences))
plt.figure(figsize=size)
plt.errorbar(gates_list, diff_opt_mean, yerr=diff_opt_std, fmt=".-", capsize=3)
plt.ylabel(r"$\Delta = d_{st} - d_{fg}$")
plt.xlabel("Num. Gates / Log. Depth")
plt.xscale("log")
# plt.xticks(gates_list, gates_list)
plt.xticks(gates_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig("plot_abs_reductions_total_" + path.replace(".pkl", ".pdf"))


# ---------output for table-------------
def fmt(mean, std, precision=2):
    """Return mean ± std formatted for LaTeX."""
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def latex_table_rows(gate_counts, *cols):
    """
    gate_counts: list of gate values
    cols: list of tuples [(mean_list, std_list), ...]
    """
    for i, gates in enumerate(gate_counts):
        # row = [f" & {gates}"]
        row = []
        for j, (means, stds) in enumerate(cols):
            # Use 3 decimals for 'r', others 2
            prec = 1
            if j == 0:
                row.append(f"& ${int(means[i])}$")  # integer for logical depth
            elif j == 3 or j == len(cols) - 1:
                # row.append(f" & {fmt(means[i], stds[i], 3)}")
                row.append(f" & {fmt(means[i]*100, stds[i]*100, 1)}\\%")  # percentage
            # elif j == len(cols)-1:
            #    row.append(f" & {fmt(means[i], stds[i], 2)}")

            else:
                row.append(f" & {fmt(means[i], stds[i], prec)}")
        row.append(" \\\\")
        print("".join(row))


diff_opt_mean_rel = [diff / dlog for diff, dlog in zip(diff_opt_mean, depths_mean_log)]
diff_opt_std_rel = [diff / dlog for diff, dlog in zip(diff_opt_std, depths_mean_log)]

diff_mean = []
diff_std = []
for dopt_lst, dst_lst in zip(depths_fg, depths_st):
    temp = []
    for dopt, dst in zip(dopt_lst, dst_lst):
        temp.append((dst - dopt) / dst)
    diff_mean.append(np.mean(temp))
    diff_std.append(np.std(temp))


latex_table_rows(
    gates_list,
    (depths_mean_log, depths_std_log),
    (depths_mean_st, depths_std_st),
    (depths_mean_opt, depths_std_opt),
    (improvements_mean, improvements_std),
    (diff_opt_mean, diff_opt_std),
    # (diff_opt_mean_rel, diff_opt_std_rel)
    (diff_mean, diff_std),
)


# print quality of fit
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Predictions from the linear model
y_pred_opt = np.polyval(fit_opt, x)
y_pred_st = np.polyval(fit_st, x)
y_pred_log = np.polyval(fit_log, x)

r2_opt = r_squared(y_opt, y_pred_opt)
r2_st = r_squared(y_st, y_pred_st)
r2_log = r_squared(y_log, y_pred_log)

print(f"R² (Opt) = {r2_opt}")
print(f"R² (St)  = {r2_st}")
print(f"R² (Log) = {r2_log}")
