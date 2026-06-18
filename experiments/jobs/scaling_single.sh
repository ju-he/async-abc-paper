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

# Paths are injected by submit_scaling.py (or submit.sh) via `sbatch --export`.
nastjapy_path="${NASTJAPY_PATH:?NASTJAPY_PATH not set — submit via submit_scaling.py}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via submit_scaling.py}"
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

# One srun (a fresh MPI world, hence a single Propulate MPI_Comm_free) per
# (k, replicate) combo. At >=48 ranks, sweeping many combos inside one
# long-lived process makes the per-combo MPI_Comm_free hang in ParaStation
# pscom_close (see .plans/bug-fixes); isolating each combo in its own process
# sidesteps the repeated teardown entirely. Aggregates are rebuilt at the end.
runner="$experiments_dir/scripts/scaling_runner.py"
status=0
while read -r k rep; do
    [ -z "$k" ] && continue
    srun -n "$n_workers" python "$runner" \
        --config "$config_path" \
        --output-dir "$output_dir" \
        --n-workers "$n_workers" \
        --k "$k" \
        --replicate "$rep" \
        --skip-finalize \
        ${test_flag:+"$test_flag"} \
        ${small_flag:+"$small_flag"} \
        ${extend_flag:+"$extend_flag"} || status=1
done < <(python "$runner" \
    --config "$config_path" \
    --output-dir "$output_dir" \
    --n-workers "$n_workers" \
    --print-combos \
    ${test_flag:+"$test_flag"} \
    ${small_flag:+"$small_flag"})

# Rebuild aggregate CSVs/plots/metadata from the per-combo shards (single rank,
# no MPI teardown).
python "$runner" \
    --config "$config_path" \
    --output-dir "$output_dir" \
    --n-workers "$n_workers" \
    --finalize-only \
    ${test_flag:+"$test_flag"} \
    ${small_flag:+"$small_flag"}

exit "$status"
