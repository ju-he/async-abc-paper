#!/bin/bash -x
# Run the scaling experiment for a single worker count.
# Called by submit_scaling.py, which overrides --ntasks, --nodes, --time,
# --job-name, and --output on the sbatch command line.
#
# Usage (via submit_scaling.py, recommended):
#   python submit_scaling.py <output_dir> [--test] [--small]
#
# Manual standalone use:
#   sbatch --ntasks=48 --nodes=1 --time=01:00:00 scaling_single.sh <output_dir> \
#       [--config /path/to/scaling.json] [--test] [--small] [--extend]
#
#SBATCH --account=tissuetwin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_scaling
#SBATCH --output=/tmp/abc_scaling-%j.out

nastjapy_path=/p/project1/tissuetwin/herold2/nastjapy
experiments_dir=/p/project1/tissuetwin/herold2/async-abc-paper/experiments
config_path="$experiments_dir/configs/scaling.json"
output_dir=""
test_flag=""
small_flag=""
extend_flag=""

usage() {
    echo "Usage: $(basename "$0") <output_dir> [--config PATH] [--test] [--small] [--extend]" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            shift
            if [ "$#" -eq 0 ]; then
                usage
                echo "Missing value for --config" >&2
                exit 2
            fi
            config_path="$1"
            ;;
        --test)
            test_flag="--test"
            ;;
        --small)
            small_flag="--small"
            ;;
        --extend)
            extend_flag="--extend"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            usage
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [ -n "$output_dir" ]; then
                usage
                echo "Unexpected extra argument: $1" >&2
                exit 2
            fi
            output_dir="$1"
            ;;
    esac
    shift
done

if [ -z "$output_dir" ]; then
    usage
    exit 2
fi


n_workers="${SLURM_NTASKS}"

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

srun -n "$n_workers" python "$experiments_dir/scripts/scaling_runner.py" \
    --config "$config_path" \
    --output-dir "$output_dir" \
    --n-workers "$n_workers" \
    ${test_flag:+"$test_flag"} \
    ${small_flag:+"$small_flag"} \
    ${extend_flag:+"$extend_flag"}
