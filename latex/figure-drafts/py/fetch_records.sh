#!/usr/bin/env bash
# Pull the small per-particle subsets the A2 draft (cumulative completions) plots
# from the cluster scratch mount. Only the columns the plot needs are kept.
set -euo pipefail
R=${ASYNC_ABC_SCRATCH:-/home/juhe/remotes/scratch/herold2/async-abc}
D="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$D"
COLS='method,replicate,worker_id,wall_time,sim_start_time,sim_end_time,generation'
extract() { # <file> <method regex> <max wall_time> <replicate or "">
  awk -F, -v want="$COLS" -v pat="$2" -v tmax="$3" -v rep="$4" '
    NR==1 { n=split(want,w,","); for(i=1;i<=NF;i++) idx[$i]=i;
            out=""; for(j=1;j<=n;j++) out=out (j>1?",":"") w[j]; print out; next }
    $1 ~ pat && (rep=="" || $2==rep) && ($(idx["wall_time"])+0) <= tmax {
            out=""; for(j=1;j<=n;j++) out=out (j>1?",":"") $(idx[w[j]]); print out }' "$1"
}
echo "[$(date +%T)] cpm twin w48"
extract "$R/cpmtwin_20260729/scaling_cpm_twin/data/raw_results_w48_k100.csv" '^async_propulate_abc__k100__w48$' 1e9 "" > "$D/a2_cpm_twin_w48.csv"
echo "[$(date +%T)] straggler twin f20"
extract "$R/twin2_20260729/straggler_twin_f20/data/raw_results.csv" 'slowdown20x' 1e9 "" > "$D/a2_straggler_twin_f20.csv"
echo "[$(date +%T)] cpm async w48 (885 MB scan)"
extract "$R/rerun_20260707/scaling_cpm/data/raw_results.csv" '^async_propulate_abc__k100__w48$' 1e9 "" > "$D/a2_cpm_async_w48.csv"
echo "[$(date +%T)] straggler async f20, replicate 0, first 60 s (6.2 GB scan)"
extract "$R/twin2_20260729/straggler_async_wall/data/raw_results.csv" 'slowdown20x' 60 0 > "$D/a2_straggler_async_f20.csv"
echo "[$(date +%T)] done"
touch "$D/.fetch_done"
