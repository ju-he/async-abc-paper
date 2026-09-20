#!/bin/bash -x
# Generate a CPM reference dataset (generate_cpm_reference.py) on a compute node.
#
# Reference generation runs NAStJA, so it cannot go on a login node. Small: four
# seeds at 80^3/t=1001 is about four simulations.
#
# Usage (via submit.sh):
#   experiments/jobs/submit.sh --partition=devel --ntasks=1 --time=00:30:00 \
#       experiments/jobs/cpm_reference.sh [generator args...]
#
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=devel
#SBATCH --job-name=cpm_ref
#SBATCH --output=/tmp/cpm_ref-%j.out
set -uo pipefail

nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

srun --ntasks=1 python "$experiments_dir/scripts/generate_cpm_reference.py" "$@" < /dev/null
status=$?
echo "[cpm_reference] exit status $status"
exit $status
