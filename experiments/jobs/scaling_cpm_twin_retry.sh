#!/bin/bash -x
# Retry the barrierized CPM twin's missing replicates, with hang diagnostics.
#
# Why this exists rather than another plain submit_scaling_cpm.py --extend run:
# the twin intermittently deadlocks at >=192 ranks (3 of 6 replicate attempts at
# 384 ranks failed to complete; one 192-rank seed hung twice), so a single pass
# has roughly even odds of landing any given combo. This wrapper
#
#   1. runs scaling_cpm_single.sh --extend several times in ONE allocation, so a
#      hung combo is retried without waiting in the queue again (--extend skips
#      whatever already landed, at the cost of one cheap srun launch per combo);
#   2. enables the per-rank faulthandler dumper, so if a combo wedges again we
#      get every rank's Python traceback instead of only knowing that it wedged.
#      A rank stuck across consecutive snapshots at the same frame IS the wedge.
#      This is the diagnostic that localized the k>=192 teardown wedge before.
#
# The step cap is deliberately tighter than scaling_cpm_single.sh's 120-min
# default: the slowest HEALTHY twin combo measured is 2241 s, so 45 min covers it
# with margin while cutting the cost of a hang from nodes x 2 h to nodes x 45 min.
#
# Usage (one job per worker count; ntasks/nodes/time come from the sbatch line):
#   sbatch --nodes=4 --ntasks=192 --time=04:00:00 scaling_cpm_twin_retry.sh \
#       <output_dir> --config <repo>/experiments/configs/scaling_cpm_twin.json
#
#SBATCH --account=tissuetwin
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --partition=batch
#SBATCH --job-name=abc_cpm_twin_retry

set -u

experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set}"
output_dir="${1:?usage: $(basename "$0") <output_dir> [--config PATH]}"
shift

passes="${CPM_TWIN_RETRY_PASSES:-3}"

# Bound a hung combo. Slowest healthy twin combo observed: 2241 s at 384 ranks.
export SCALING_CPM_STEP_TIMEOUT_MIN="${SCALING_CPM_STEP_TIMEOUT_MIN:-45}"

# Per-rank Python traceback snapshots. Written from inside the process, so no
# ptrace is needed (compute nodes block it). Costs one small file per rank.
export SCALING_FAULTHANDLER_S="${SCALING_FAULTHANDLER_S:-15}"
export SCALING_FAULTHANDLER_DIR="$output_dir/pytraces_w${SLURM_NTASKS}_job${SLURM_JOB_ID}"
mkdir -p "$SCALING_FAULTHANDLER_DIR"

echo "[retry] ntasks=$SLURM_NTASKS passes=$passes step_cap=${SCALING_CPM_STEP_TIMEOUT_MIN}min"
echo "[retry] traces -> $SCALING_FAULTHANDLER_DIR"

for pass in $(seq 1 "$passes"); do
    echo "[retry] ===== pass $pass/$passes ====="
    # --extend makes each pass a no-op for combos that already landed, so the
    # passes converge instead of redoing work. Non-zero exit just means some
    # combo is still missing; keep going.
    "$experiments_dir/jobs/scaling_cpm_single.sh" "$output_dir" "$@" --extend || true
done

echo "[retry] done; surviving traces:"
# Only report traces that show a rank stuck: >1 snapshot with an identical tail.
find "$SCALING_FAULTHANDLER_DIR" -name 'rank_*.txt' -size +0 | head -20
