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
#SBATCH --output=/tmp/abc_scaling_cpm-%j.out

# Paths are injected by submit_scaling.py (or submit.sh) via `sbatch --export`.
backend_path="${SIM_BACKEND_PATH:?SIM_BACKEND_PATH not set — submit via submit_scaling.py}"
experiments_dir="${EXPERIMENTS_DIR:?EXPERIMENTS_DIR not set — submit via submit_scaling.py}"
config_path="$experiments_dir/configs/scaling_cpm.json"
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

module restore sim_backend
module load ParaStationMPI
source "$backend_path/.venv/bin/activate"

mkdir -p "$output_dir"
cp "$0" "$output_dir/" 2>/dev/null || true

# Skip the ParaStation MPI_Comm_free that hangs in pscom_close at high message
# volume (see scaling_single.sh). With one combo per process, skipping Free
# leaks one communicator per process, reclaimed at process exit — safe. realistic workload's
# slow sims keep volume low so it has not hit the hang, but enabling this keeps
# it robust. Set PROPULATE_SKIP_DISCONNECT=0 to reproduce.
export PROPULATE_SKIP_DISCONNECT="${PROPULATE_SKIP_DISCONNECT:-1}"

# One srun (a fresh MPI world, hence a single Propulate MPI_Comm_free) per
# (k, replicate) combo — see scaling_single.sh / .plans/bug-fixes. realistic workload's slow
# sims keep message volume low so it has not hit the pscom teardown hang, but
# isolating combos keeps it robust as the grid grows. Aggregates rebuilt at end.
runner="$experiments_dir/scripts/scaling_cpm_runner.py"

# Read the (k, replicate) grid into an array FIRST, then loop. Do NOT feed the
# grid into a `while read ... done < <(...)` loop: srun reads stdin and swallows
# the rest of the list, so only the first combo would run. `< /dev/null` on srun
# is belt-and-suspenders against the same footgun.
mapfile -t combos < <(python "$runner" \
    --config "$config_path" \
    --output-dir "$output_dir" \
    --n-workers "$n_workers" \
    --print-combos \
    ${test_flag:+"$test_flag"} \
    ${small_flag:+"$small_flag"})

status=0
for combo in "${combos[@]}"; do
    [ -z "$combo" ] && continue
    read -r k rep <<< "$combo"
    srun -n "$n_workers" python "$runner" \
        --config "$config_path" \
        --output-dir "$output_dir" \
        --n-workers "$n_workers" \
        --k "$k" \
        --replicate "$rep" \
        --skip-finalize \
        ${test_flag:+"$test_flag"} \
        ${small_flag:+"$small_flag"} \
        ${extend_flag:+"$extend_flag"} < /dev/null || status=1
done

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
