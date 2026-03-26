#!/bin/bash -x
# Run multiple low-rank scaling worker-count jobs inside one full-node allocation.
#
# Usage:
#   sbatch --ntasks=48 --nodes=1 scaling_packed.sh <output_dir> \
#       --workers 1,4,8,16 [--config /path/to/scaling.json] [--test] [--small] [--extend]
#
# Called by submit_scaling.py for worker counts smaller than the node size.
#
#SBATCH --account=tissuetwin
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --job-name=abc_scaling_bundle
#SBATCH --output=/tmp/abc_scaling_bundle-%j.out

set -u

nastjapy_path=/p/project1/tissuetwin/herold2/nastjapy
experiments_dir=/p/project1/tissuetwin/herold2/async-abc-paper/experiments
config_path="$experiments_dir/configs/scaling.json"
output_dir=""
workers_csv=""
test_flag=""
small_flag=""
extend_flag=""

usage() {
    echo "Usage: $(basename "$0") <output_dir> --workers 1,4,8 [--config PATH] [--test] [--small] [--extend]" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workers)
            shift
            if [ "$#" -eq 0 ]; then
                usage
                echo "Missing value for --workers" >&2
                exit 2
            fi
            workers_csv="$1"
            ;;
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

if [ -z "$output_dir" ] || [ -z "$workers_csv" ]; then
    usage
    exit 2
fi

IFS=',' read -r -a workers <<< "$workers_csv"
if [ "${#workers[@]}" -eq 0 ]; then
    echo "No workers parsed from --workers=$workers_csv" >&2
    exit 2
fi

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

pids=()
failed=0
had_any_run=0
for n_workers in "${workers[@]}"; do
    srun --exclusive -N 1 -n "$n_workers" python "$experiments_dir/scripts/scaling_runner.py" \
        --config "$config_path" \
        --output-dir "$output_dir" \
        --n-workers "$n_workers" \
        --skip-finalize \
        ${test_flag:+"$test_flag"} \
        ${small_flag:+"$small_flag"} \
        ${extend_flag:+"$extend_flag"} &
    pids+=("$!")
    had_any_run=1
done

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [ "$had_any_run" -ne 0 ]; then
    python "$experiments_dir/scripts/scaling_runner.py" \
        --config "$config_path" \
        --output-dir "$output_dir" \
        --finalize-only \
        ${test_flag:+"$test_flag"} \
        ${small_flag:+"$small_flag"}
fi

exit "$failed"
