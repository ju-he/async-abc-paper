#!/bin/bash -x
# One Cellular Potts inference experiment from one config
# (experiments/scripts/cellular_potts_runner.py).
#
# run_experiments.sh runs the whole paper campaign against the shipped
# cellular_potts.json; this runs a single named config, which is what the
# two-parameter setup of .plans/cpm_setup_proposal_2026-09-19.md needs -- both
# for the one-method validation run and for the production one.
#
# Budget it in SIMULATIONS, not evaluations: with
# "n_replicates_per_evaluation": 4 each evaluation is four NAStJA runs, so a
# 2000-evaluation method replicate costs 8000 simulations.
#
# Usage (via submit.sh, which injects EXPERIMENTS_DIR / NASTJAPY_PATH):
#   experiments/jobs/submit.sh --time=02:00:00 \
#       experiments/jobs/cpm_inference.sh <output_dir> <config> [runner args...]
#
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --job-name=cpm_infer
#SBATCH --output=/tmp/cpm_infer-%j.out
set -uo pipefail

nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $(basename "$0") <output_dir> <config> [runner args...]" >&2
    exit 2
fi
output_dir="$1"
config="$2"
shift 2

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

# Avoids the intermittent pscom MPI-teardown hang on non-scaling multi-rank jobs.
export PROPULATE_SKIP_DISCONNECT=1

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true
cp "$config" "$output_dir/" 2>/dev/null || true

srun --ntasks="${SLURM_NTASKS}" --cpus-per-task=1 \
    python "$experiments_dir/scripts/cellular_potts_runner.py" \
        --config "$config" \
        --output-dir "$output_dir" \
        "$@" < /dev/null
status=$?

# The benchmark removes its own per-evaluation directories; this is the sweep
# for a job that died mid-evaluation. Scratch inodes are the binding constraint
# on this campaign (see .plans/scratch_inode_recovery_plan.md).
rm -rf "$output_dir/cpm_sims"
echo "[cpm_inference] exit status $status"
exit $status
