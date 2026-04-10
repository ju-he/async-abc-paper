#!/bin/bash -x
# CommWorldMap 48-rank verification (Phase 2 MPI-01, CONTEXT.md D-01/D-02).
#
# Usage:
#   sbatch experiments/jobs/verify_commworldmap_48.sh <output_dir>
#
# Writes a verification JSON to <output_dir>/verify_commworldmap_48.json.
# Exit code 0 = verification PASS, non-zero = FAIL (see SLURM stdout for details).
#
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --job-name=commworldmap_verify_48
#SBATCH --output=/tmp/commworldmap_verify_48-%j.out

set -u

nastjapy_path=/p/project1/tissuetwin/herold2/nastjapy
experiments_dir=/p/project1/tissuetwin/herold2/async-abc-paper/experiments

output_dir="${1:?Usage: $(basename "$0") <output_dir>}"
mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

srun python "$experiments_dir/scripts/verify_commworldmap_48.py" \
    "$output_dir/verify_commworldmap_48.json"
