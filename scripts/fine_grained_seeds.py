"""
Fine-grained routing benchmark: one max_overlap value, all seeds in a single job.

The SLURM array now varies only max_overlap; this script loops over every seed
internally and reports mean/std across them, so each array task produces one
complete data point instead of a fragment that has to be re-aggregated later.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# resolve from the script's own location, not the cwd, so the job works
# regardless of the directory sbatch launches it from
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

import numpy as np

import cococo.layouts as layouts
import cococo.utils_routing as utils
import cococo.circuit_construction as circuit_construction
import cococo.internal_testing as internal_testing
import cococo.dag_helper as dag_helper

from datetime import datetime

# ----------------------------- args -----------------------------

# the 10 seeds that used to be spread across the array's seed dimension
DEFAULT_SEEDS = [
    1047329, 20119759, 3214159, 82421337, 52512011,
    6210003, 27350431, 3846233, 99950341, 14061987,
]

parser = argparse.ArgumentParser()
parser.add_argument("--max-ovl", type=int, required=True)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=DEFAULT_SEEDS,
    help="seeds to average over (default: the 10 standard benchmark seeds)",
)
args = parser.parse_args()

max_overlap = args.max_ovl
seeds = args.seeds

# ----------------------------- geometry -----------------------------

layout_type = "single"
m = 12
n = 18
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
    layout_type, m, n, factories, remove_edges
)
layout = {i: j for i, j in enumerate(data_qubit_locs)}
t = 2

q = len(data_qubit_locs)
print("number of data qubits: ", q)
j = 20
num_gates = 4 * q
overlap_type = "strict_k"

# ----------------------------- run all seeds -----------------------------

rows = []
run_start = time.perf_counter()

for seed in seeds:
    print(f"===== seed {seed} (max_overlap={max_overlap}) =====")

    # j gates per layer on q qubits
    # pairs indicate the qubit index (0, ..., q)
    dag, pairs = circuit_construction.create_random_sequential_circuit_dag(
        j, q, num_gates, seed
    )
    print("number of gates: ", len(pairs))

    # terminal pairs indicate the 2d coordinates
    terminal_pairs = layouts.translate_layout_circuit(pairs, layout)

    # fresh router per seed: each seed is an independent circuit, and the
    # router accumulates per-run state across a routing call
    router = utils.BasicRouter(
        g, data_qubit_locs, factories,
        valid_path="cc", t=t, metric="exact", use_dag=True,
    )
    layers = router.split_layer_terminal_pairs(terminal_pairs)
    vdp_layers, _ = router.find_total_vdp_layers_dyn(
        layers, data_qubit_locs, router.factory_times, layout, testing=True
    )
    baseline_len = len(vdp_layers)
    print("Len of schedule without teleportation: ", baseline_len)

    t_start = time.perf_counter()
    router.find_total_fine_grained_vdp_dyn(
        layers, None, None,
        layout=layout,
        max_overlap=max_overlap,
        overlap_type=overlap_type,
        testing=True,
    )
    elapsed = time.perf_counter() - t_start

    fg_len = len(router.routes_by_layer)
    print("Len of schedule (fine grained): ", fg_len)

    # reduction of schedule length vs the non-fine-grained (coarse VDP) schedule
    reduction = (baseline_len - fg_len) / baseline_len if baseline_len else 0.0

    # machine-parseable per-seed line, unchanged format so existing log
    # scrapers keep working
    print(
        f"BENCH_RESULT seed={seed} max_overlap={max_overlap} "
        f"gates={len(terminal_pairs)} baseline={baseline_len} layers={fg_len} "
        f"red={reduction:.4f} seconds={elapsed:.4f}"
    )

    rows.append({
        "seed": seed,
        "gates": len(terminal_pairs),
        "baseline": baseline_len,
        "layers": fg_len,
        "reduction": reduction,
        "seconds": elapsed,
    })

total_elapsed = time.perf_counter() - run_start

# ----------------------------- aggregate -----------------------------


def mean_std(key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.mean(vals)), float(np.std(vals))


baseline_mean, baseline_std = mean_std("baseline")
layers_mean, layers_std = mean_std("layers")
red_mean, red_std = mean_std("reduction")
sec_mean, sec_std = mean_std("seconds")

print("=" * 60)
print(f"max_overlap = {max_overlap}   n_seeds = {len(rows)}")
print(f"baseline   mean={baseline_mean:.3f}  std={baseline_std:.3f}")
print(f"fg layers  mean={layers_mean:.3f}  std={layers_std:.3f}")
print(f"reduction  mean={red_mean:.4f}  std={red_std:.4f}")
print(f"seconds    mean={sec_mean:.4f}  std={sec_std:.4f}")
print(f"total wall time: {total_elapsed:.2f}s")
print("=" * 60)

# single machine-parseable summary line per array task
print(
    f"BENCH_SUMMARY max_overlap={max_overlap} n_seeds={len(rows)} "
    f"baseline_mean={baseline_mean:.4f} baseline_std={baseline_std:.4f} "
    f"layers_mean={layers_mean:.4f} layers_std={layers_std:.4f} "
    f"red_mean={red_mean:.4f} red_std={red_std:.4f} "
    f"seconds_mean={sec_mean:.4f} seconds_std={sec_std:.4f}"
)

# ----------------------------- dump -----------------------------

# max_overlap is in the filename so concurrent array tasks cannot clobber
# each other; the per-seed rows are kept so the aggregate can be recomputed
out_dir = Path(__file__).parent / "benchmark_results"
out_dir.mkdir(exist_ok=True)
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = out_dir / (
    f"fg_seeds_layout{layout_type}_m{m}_n{n}_q{q}_j{j}_numgates{num_gates}"
    f"_overlap{overlap_type}{max_overlap}_nseeds{len(rows)}_{date_str}.json"
)
with open(out_path, "w") as f:
    json.dump({
        "layout_type": layout_type,
        "m": m,
        "n": n,
        "q": q,
        "j": j,
        "num_gates": num_gates,
        "t": t,
        "overlap_type": overlap_type,
        "max_overlap": max_overlap,
        "seeds": seeds,
        "rows": rows,
        "summary": {
            "baseline_mean": baseline_mean, "baseline_std": baseline_std,
            "layers_mean": layers_mean, "layers_std": layers_std,
            "reduction_mean": red_mean, "reduction_std": red_std,
            "seconds_mean": sec_mean, "seconds_std": sec_std,
            "total_seconds": total_elapsed,
        },
    }, f, indent=2)
print("wrote", out_path)
