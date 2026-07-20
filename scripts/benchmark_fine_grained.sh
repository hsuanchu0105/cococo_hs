#!/usr/bin/env bash
#
# Benchmark find_fine_grained_vdp by running fine_grained_test.py across several
# circuit-initialization seeds and collecting per-run timing.
#
# Usage:
#   ./benchmark_fine_grained.sh                 # seeds 1..10 (default)
#   ./benchmark_fine_grained.sh 1 2 3 42 100    # explicit seed list
#   PY=python3.11 ./benchmark_fine_grained.sh   # override interpreter
#
# Must be run from the scripts/ directory (fine_grained_test.py imports the local
# plotting.py and resolves src via ../src).

set -euo pipefail

# Run from the directory this script lives in (== scripts/).
cd "$(dirname "$0")"

PY="${PY:-python3.11}"

# Seeds: use CLI args if given, else 1..10.
if [ "$#" -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(21787058 125483928 973316763 88455461 580817535 991114146 964501795 889112014 527718978 235259617)
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTDIR="benchmark_results"
mkdir -p "$OUTDIR"
CSV="${OUTDIR}/fine_grained_${STAMP}.csv"
LOG="${OUTDIR}/fine_grained_${STAMP}.log"

echo "seed,gates,baseline,layers,red_pct,seconds" > "$CSV"
echo "Benchmarking find_fine_grained_vdp over seeds: ${SEEDS[*]}"
echo "  interpreter : $PY"
echo "  csv         : $CSV"
echo "  full log    : $LOG"
echo

for seed in "${SEEDS[@]}"; do
    printf 'seed %-4s ... ' "$seed"

    # BENCH=1 disables the stim correctness check and the animation so the timing
    # reflects the routing only. Full stdout goes to the log; we scrape the
    # single BENCH_RESULT line for the CSV.
    if ! BENCH=1 "$PY" fine_grained_bench.py "$seed" >>"$LOG" 2>&1; then
        echo "FAILED (see $LOG)"
        continue
    fi

    line="$(grep "^BENCH_RESULT" "$LOG" | tail -n 1)"
    # line: BENCH_RESULT seed=7 gates=1024 baseline=180 layers=154 red=14.44 seconds=64.5779
    gates="$(sed -n 's/.*gates=\([0-9]*\).*/\1/p' <<<"$line")"
    baseline="$(sed -n 's/.*baseline=\([0-9]*\).*/\1/p' <<<"$line")"
    layers="$(sed -n 's/.*layers=\([0-9]*\).*/\1/p' <<<"$line")"
    red="$(sed -n 's/.*red=\(-\?[0-9.]*\).*/\1/p' <<<"$line")"
    seconds="$(sed -n 's/.*seconds=\([0-9.]*\).*/\1/p' <<<"$line")"

    echo "${seed},${gates},${baseline},${layers},${red},${seconds}" >> "$CSV"
    printf 'gates=%s baseline=%s layers=%s red=%s%% %ss\n' \
        "$gates" "$baseline" "$layers" "$red" "$seconds"
done

echo
echo "Done. Summary:"
# Mean / min / max over the seconds ($6) and reduction ($5) columns.
awk -F, 'NR>1 && $6!="" {
           n++; s+=$6; r+=$5;
           if (min=="" || $6<min) min=$6;
           if ($6>max) max=$6;
         }
         END {
           if (n>0)
             printf "  runs=%d  time mean=%.3fs min=%.3fs max=%.3fs  red mean=%.2f%%\n", \
                    n, s/n, min, max, r/n;
           else
             print "  no successful runs";
         }' "$CSV"
echo "  csv: $CSV"
