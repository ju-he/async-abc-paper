#!/bin/bash -x
# Run the scaling experiment for a single worker count.
# Called by submit_scaling.py, which overrides --ntasks, --nodes, --time,
# --job-name, and --output on the sbatch command line.
#
# Usage (via submit_scaling.py, recommended):
#   python submit_scaling.py <output_dir>
#
# Manual standalone use:
#   sbatch --ntasks=48 --nodes=1 --time=01:00:00 scaling_single.sh <output_dir>
#
#SBATCH --account=tissuetwin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_scaling
#SBATCH --output=/tmp/abc_scaling-%j.out

nastjapy_path=NASTJAPY_PATH
output_dir="${1:?Usage: scaling_single.sh <output_dir>}"
experiments_dir="$(cd "$(dirname "$0")/.." && pwd)"

n_workers="${SLURM_NTASKS}"

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

srun -n "$n_workers" python "$experiments_dir/scripts/scaling_runner.py" \
    --config "$experiments_dir/configs/scaling.json" \
    --output-dir "$output_dir" \
    --n-workers "$n_workers"
