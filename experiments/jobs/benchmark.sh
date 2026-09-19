#!/bin/bash -x
# One benchmark experiment from one runner and one config.
#
# cpm_inference.sh is the Cellular Potts special case; this is the same thing
# for any benchmark runner, so a single arm can be re-run without going through
# run_all_paper_experiments.py.
#
# Usage (via submit.sh, which injects EXPERIMENTS_DIR / NASTJAPY_PATH):
#   experiments/jobs/submit.sh --time=02:00:00 experiments/jobs/benchmark.sh \
#       gaussian_mean_runner <output_dir> <config> [runner args...]
#
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_bench
#SBATCH --output=/tmp/abc_bench-%j.out
set -uo pipefail

nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"

if [ "$#" -lt 3 ]; then
    echo "Usage: $(basename "$0") <runner> <output_dir> <config> [runner args...]" >&2
    exit 2
fi
runner="$1"
output_dir="$2"
config="$3"
shift 3

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

export PROPULATE_SKIP_DISCONNECT=1

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true
cp "$config" "$output_dir/" 2>/dev/null || true

srun --ntasks="${SLURM_NTASKS}" --cpus-per-task=1 \
    python "$experiments_dir/scripts/${runner}.py" \
        --config "$config" \
        --output-dir "$output_dir" \
        "$@" < /dev/null
status=$?
echo "[benchmark] $runner exit status $status"
exit $status
