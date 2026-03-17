#!/bin/bash
# Submit one sbatch job per worker count for the scaling experiment.
#
# Usage:
#   ./submit_scaling.sh <output_dir>
#
# Each job uses exactly N MPI ranks, matching the worker count being measured.
# --nodes is computed automatically as ceil(N / CORES_PER_NODE).
# Time limits are rough estimates; tune them after a test run.

set -euo pipefail

OUTPUT_DIR="${1:?Usage: submit_scaling.sh <output_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORES_PER_NODE=48

WORKER_COUNTS=(  1    2    4    8   16   32   48   96  192  256)
TIMELIMITS=(
    "00:30:00"   # N=1
    "00:30:00"   # N=2
    "00:30:00"   # N=4
    "00:30:00"   # N=8
    "01:00:00"   # N=16
    "01:00:00"   # N=32
    "01:00:00"   # N=48
    "02:00:00"   # N=96
    "04:00:00"   # N=192
    "08:00:00"   # N=256
)

mkdir -p "$OUTPUT_DIR"

for i in "${!WORKER_COUNTS[@]}"; do
    n="${WORKER_COUNTS[$i]}"
    t="${TIMELIMITS[$i]}"
    nodes=$(( (n + CORES_PER_NODE - 1) / CORES_PER_NODE ))
    echo "Submitting N=${n} (${nodes} node(s), time limit ${t}) ..."
    sbatch \
        --ntasks="$n" \
        --nodes="$nodes" \
        --time="$t" \
        --job-name="abc_scaling_${n}" \
        --output="${OUTPUT_DIR}/abc_scaling_${n}-%j.out" \
        "$SCRIPT_DIR/scaling_single.sh"
done

echo "All scaling jobs submitted. Monitor with: squeue -u \$USER"
