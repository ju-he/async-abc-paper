#!/bin/bash -x
# Cellular Potts protocol + parameter screening (diag_cpm_screening.py).
#
# One node is plenty: a 50^3 evaluation is a few seconds, and the whole design
# is ~1500 evaluations.  Override --ntasks/--nodes/--time/--partition on the
# sbatch command line for the larger-domain variants.
#
# Usage (via submit.sh, which injects EXPERIMENTS_DIR / NASTJAPY_PATH):
#   experiments/jobs/submit.sh --ntasks=48 --nodes=1 --time=01:00:00 \
#       experiments/jobs/cpm_screening.sh <output_dir> [--blocksize 50] [...]
#
# Every extra argument is forwarded verbatim to diag_cpm_screening.py.
#
#SBATCH --account=tissuetwin
#SBATCH --ntasks=48
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --job-name=cpm_screen
#SBATCH --output=/tmp/cpm_screen-%j.out
set -uo pipefail

nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $(basename "$0") <output_dir> [diag_cpm_screening.py args...]" >&2
    exit 2
fi
output_dir="$1"
shift

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Per-evaluation simulation directories are removed by the driver itself; this
# keeps them off the (inode-constrained) shared tree while the job runs.
srun --ntasks="${SLURM_NTASKS}" --cpus-per-task=1 \
    python "$experiments_dir/scripts/diag_cpm_screening.py" \
        --mode simulate \
        --out "$output_dir" \
        "$@" < /dev/null
status=$?

# The driver removes its own eval dirs; this is the belt-and-braces sweep for a
# job that died mid-evaluation.  The scratch inode budget is the binding
# constraint on this campaign (see .plans/scratch_inode_recovery_plan.md).
rm -rf "$output_dir/sims"
echo "[cpm_screening] exit status $status"
exit $status
