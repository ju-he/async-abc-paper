#!/bin/bash -x
#SBATCH --account=tissuetwin
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_test_all
#SBATCH --output=/tmp/abc_test_all-%j.out
# Override SLURM log path at submission time: sbatch --output=<dir>/abc_test_all-%j.out ...

# Paths are injected by experiments/jobs/submit.sh via `sbatch --export`.
backend_path="${SIM_BACKEND_PATH:?SIM_BACKEND_PATH not set — submit via experiments/jobs/submit.sh}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via experiments/jobs/submit.sh}"
default_output_root="${SITE_SCRATCH_ROOT:?SITE_SCRATCH_ROOT not set — submit via experiments/jobs/submit.sh}/test"
output_dir=""
extend_flag=""

usage() {
    echo "Usage: $(basename "$0") [output_dir] [--extend]" >&2
}

for arg in "$@"; do
    case "$arg" in
        --extend)
            extend_flag="--extend"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            usage
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
        *)
            if [ -n "$output_dir" ]; then
                usage
                echo "Unexpected extra argument: $arg" >&2
                exit 2
            fi
            output_dir="$arg"
            ;;
    esac
done

output_dir="${output_dir:-$default_output_root/test_all_${SLURM_JOB_ID:-local}}"

module restore sim_backend
module load ParaStationMPI
source "$backend_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true
echo "Using output_dir=$output_dir"

srun python "$experiments_dir/run_all_paper_experiments.py" \
    --test \
    --experiments gaussian_mean gandk lotka_volterra realistic_workload sbc \
                  straggler runtime_heterogeneity scaling sensitivity ablation \
    --output-dir "$output_dir" \
    ${extend_flag:+"$extend_flag"}
