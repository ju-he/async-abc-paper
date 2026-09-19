#!/bin/bash -x
# Prior corpus for a CPM setup (diag_cpm_prior_corpus.py): N prior draws scored
# through the shipped benchmark, from which the rejection-ABC posterior at any
# acceptance level is read afterwards with --mode analyse.
#
# Budget it in SIMULATIONS: with "n_replicates_per_evaluation": 4 a 2000-draw
# corpus is 8000 NAStJA runs, about 15 minutes on one 48-rank node.
#
# Usage (via submit.sh, which injects EXPERIMENTS_DIR / NASTJAPY_PATH):
#   experiments/jobs/submit.sh experiments/jobs/cpm_prior_corpus.sh \
#       <output_dir> <config> [--n-draws 2000]
#
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --job-name=cpm_prior
#SBATCH --output=/tmp/cpm_prior-%j.out
set -uo pipefail

nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $(basename "$0") <output_dir> <config> [driver args...]" >&2
    exit 2
fi
output_dir="$1"
config="$2"
shift 2

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

export PROPULATE_SKIP_DISCONNECT=1

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

srun --ntasks="${SLURM_NTASKS}" --cpus-per-task=1 \
    python "$experiments_dir/scripts/diag_cpm_prior_corpus.py" \
        --config "$config" \
        --out "$output_dir" \
        "$@" < /dev/null
status=$?

# The benchmark removes its own per-evaluation directories; this is the sweep
# for a job that died mid-evaluation. Scratch inodes are the binding constraint.
rm -rf "$output_dir/sims"
echo "[cpm_prior] exit status $status"
exit $status
