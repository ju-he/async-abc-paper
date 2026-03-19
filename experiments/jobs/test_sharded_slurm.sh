#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: test_sharded_slurm.sh [output_dir]

Submit a small sharded SLURM smoke test that verifies:
  1. two shard jobs can run and the last one merges final results
  2. estimate-only sharded --test mode writes both timing estimates

The script submits tiny Gaussian benchmark jobs via sbatch, waits for them to
leave the queue, then validates the resulting artifacts on disk.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 127
    fi
}

submit_sbatch() {
    local script_path="$1"
    sbatch --parsable "$script_path"
}

wait_for_jobs() {
    local job_ids_csv="$1"
    while squeue -h -j "$job_ids_csv" | grep -q .; do
        sleep 5
    done
}

show_logs() {
    local log_dir="$1"
    if [[ -d "$log_dir" ]]; then
        for log in "$log_dir"/*.out; do
            [[ -f "$log" ]] || continue
            echo "----- $log -----" >&2
            tail -n 100 "$log" >&2 || true
        done
    fi
}

require_cmd sbatch
require_cmd squeue

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiments_dir="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${1:-/p/scratch/tissuetwin/herold2/async-abc/test/sharded_smoke_$(date +%Y%m%d_%H%M%S)}"
nastjapy_path="${NASTJAPY_PATH:-/p/project1/tissuetwin/herold2/nastjapy}"
account="${SLURM_ACCOUNT_OVERRIDE:-tissuetwin}"
partition="${SLURM_PARTITION_OVERRIDE:-batch}"
time_limit="${SLURM_TIME_OVERRIDE:-00:10:00}"

mkdir -p "$output_root"
echo "Using output_root=$output_root"

merge_cfg="$output_root/gaussian_merge.json"
timing_cfg="$output_root/gaussian_timing.json"
merge_out="$output_root/merge_smoke"
timing_out="$output_root/timing_smoke"
merge_job_dir="$output_root/jobs_merge"
timing_job_dir="$output_root/jobs_timing"
merge_log_dir="$merge_job_dir/logs"
timing_log_dir="$timing_job_dir/logs"
mkdir -p "$merge_job_dir" "$timing_job_dir" "$merge_log_dir" "$timing_log_dir"

trap 'status=$?; if [[ $status -ne 0 ]]; then echo "SLURM sharded smoke test failed." >&2; show_logs "$merge_log_dir"; show_logs "$timing_log_dir"; fi; exit $status' EXIT

cat >"$merge_cfg" <<'EOF'
{
  "experiment_name": "gaussian_mean",
  "benchmark": {
    "name": "gaussian_mean",
    "observed_data_seed": 42,
    "n_obs": 40,
    "true_mu": 0.0,
    "sigma_obs": 1.0,
    "prior_low": -5.0,
    "prior_high": 5.0
  },
  "methods": ["rejection_abc"],
  "inference": {
    "max_simulations": 80,
    "n_workers": 1,
    "k": 10,
    "tol_init": 50.0,
    "n_generations": 2,
    "scheduler_type": "acceptance_rate",
    "perturbation_scale": 0.8
  },
  "execution": {
    "n_replicates": 2,
    "base_seed": 1
  },
  "plots": {}
}
EOF

cp "$merge_cfg" "$timing_cfg"

write_job_script() {
    local script_path="$1"
    local log_path="$2"
    local command="$3"
    cat >"$script_path" <<EOF
#!/bin/bash -x
#SBATCH --account=$account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time=$time_limit
#SBATCH --partition=$partition
#SBATCH --job-name=$(basename "$script_path" .sbatch)
#SBATCH --output=$log_path

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "$output_root"
srun $command
EOF
    chmod +x "$script_path"
}

echo "Submitting merge smoke shards..."
merge_job_ids=()
for shard_index in 0 1; do
    script_path="$merge_job_dir/gaussian_merge_shard_${shard_index}.sbatch"
    log_path="$merge_log_dir/gaussian_merge_shard_${shard_index}-%j.out"
    command="$python_bin $experiments_dir/scripts/gaussian_mean_runner.py --config $merge_cfg --output-dir $merge_out --shard-index $shard_index --num-shards 2"
    write_job_script "$script_path" "$log_path" "$command"
    job_id="$(submit_sbatch "$script_path")"
    merge_job_ids+=("$job_id")
    echo "submitted merge shard $shard_index as job $job_id"
done

merge_job_ids_csv="$(IFS=,; echo "${merge_job_ids[*]}")"
wait_for_jobs "$merge_job_ids_csv"

test -f "$merge_out/gaussian_mean/data/raw_results.csv"
test -f "$merge_out/_shards/gaussian_mean/merge.done.json"

"$python_bin" - <<'PY' "$merge_out/gaussian_mean/data/raw_results.csv"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with open(path) as f:
    rows = list(csv.DictReader(f))
replicates = {int(row["replicate"]) for row in rows}
if replicates != {0, 1}:
    raise SystemExit(f"expected merged replicates {{0, 1}}, got {replicates}")
print(f"merge ok: merged replicates={sorted(replicates)}")
PY

echo "Preparing estimate-only sharded test plan..."
"$python_bin" - <<'PY' "$timing_out" "$timing_cfg" "$experiments_dir"
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from async_abc.io.config import load_config
from async_abc.utils.sharding import ShardLayout, build_plan_payload, update_plan

output_dir = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
full_cfg = load_config(config_path, test_mode=False)
test_cfg = load_config(config_path, test_mode=True)
layout = ShardLayout(output_dir, "gaussian_mean", 0)
plan = build_plan_payload(
    experiment_name="gaussian_mean",
    config_path=str(config_path),
    runner_script=str((Path(sys.argv[3]) / "scripts" / "gaussian_mean_runner.py").resolve()),
    unit_kind="replicate",
    full_total_units=int(full_cfg["execution"]["n_replicates"]),
    actual_total_units=int(test_cfg["execution"]["n_replicates"]),
    requested_num_shards=2,
    actual_num_shards=1,
    test_mode=True,
    extend=False,
    shard_assignments=[[0]],
)
update_plan(layout, plan)
PY

echo "Submitting estimate-only test shard..."
timing_script="$timing_job_dir/gaussian_timing_shard_0.sbatch"
timing_log="$timing_log_dir/gaussian_timing_shard_0-%j.out"
timing_command="$python_bin $experiments_dir/scripts/gaussian_mean_runner.py --config $timing_cfg --output-dir $timing_out --test --shard-index 0 --num-shards 1"
write_job_script "$timing_script" "$timing_log" "$timing_command"
timing_job_id="$(submit_sbatch "$timing_script")"
echo "submitted timing shard as job $timing_job_id"
wait_for_jobs "$timing_job_id"

test -f "$timing_out/gaussian_mean/data/timing.csv"
test -f "$timing_out/_shards/gaussian_mean/merge.done.json"

echo "Validating timing estimate fields..."
"$python_bin" - <<'PY' "$timing_out/gaussian_mean/data/timing.csv"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with open(path) as f:
    rows = list(csv.DictReader(f))
row = rows[-1]
unsharded = float(row["estimated_full_unsharded_s"])
sharded = float(row["estimated_full_sharded_wall_s"])
if not (unsharded > 0 and sharded > 0):
    raise SystemExit("timing estimates were not populated")
if sharded > unsharded:
    raise SystemExit("sharded estimate should not exceed unsharded estimate for this smoke test")
print(f"timing ok: unsharded={unsharded:.3f}s sharded={sharded:.3f}s")
PY

trap - EXIT
echo
echo "SLURM sharded smoke test completed successfully."
echo "Merge smoke output:  $merge_out"
echo "Timing smoke output: $timing_out"
