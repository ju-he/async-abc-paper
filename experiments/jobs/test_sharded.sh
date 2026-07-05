#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: test_sharded.sh [output_dir]

Local smoke test for sharded execution. It verifies:
  1. replicate sharding runs in two shards and the second shard triggers final merge
  2. estimate-only test mode writes both sharded and unsharded timing estimates

If output_dir is omitted, a temporary directory under /tmp is used.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiments_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$experiments_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${1:-/tmp/async_abc_sharded_smoke_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$output_root"
echo "Using output_root=$output_root"

merge_cfg="$output_root/gaussian_merge.json"
timing_cfg="$output_root/gaussian_timing.json"
merge_out="$output_root/merge_smoke"
timing_out="$output_root/timing_smoke"

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

echo "Running merge smoke test shards..."
"$python_bin" "$experiments_dir/scripts/gaussian_mean_runner.py" \
    --config "$merge_cfg" \
    --output-dir "$merge_out" \
    --shard-index 0 \
    --num-shards 2

"$python_bin" "$experiments_dir/scripts/gaussian_mean_runner.py" \
    --config "$merge_cfg" \
    --output-dir "$merge_out" \
    --shard-index 1 \
    --num-shards 2

test -f "$merge_out/gaussian_mean/data/raw_results.csv"
test -f "$merge_out/_shards/gaussian_mean/runs/default/merge.done.json"

echo "Preparing estimate-only sharded test plan..."
"$python_bin" - <<'PY' "$timing_out" "$timing_cfg" "$experiments_dir"
import json
import sys
from pathlib import Path

experiments_dir = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(experiments_dir))
from async_abc.io.config import load_config
from async_abc.utils.sharding import ShardLayout, build_plan_payload, update_plan

output_dir = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
full_cfg = load_config(config_path, test_mode=False)
test_cfg = load_config(config_path, test_mode=True)
layout = ShardLayout(output_dir, "gaussian_mean", "default", 0)
plan = build_plan_payload(
    experiment_name="gaussian_mean",
    config_path=str(config_path),
    runner_script=str((experiments_dir / "scripts" / "gaussian_mean_runner.py").resolve()),
    unit_kind="replicate",
    full_total_units=int(full_cfg["execution"]["n_replicates"]),
    actual_total_units=int(test_cfg["execution"]["n_replicates"]),
    target_total_units=int(full_cfg["execution"]["n_replicates"]),
    requested_num_shards=2,
    actual_num_shards=1,
    test_mode=True,
    extend=False,
    run_id="default",
    completed_unit_indices=[],
    pending_unit_indices=[0],
    shard_assignments=[[0]],
)
update_plan(layout, plan)
PY

echo "Running estimate-only sharded test shard..."
"$python_bin" "$experiments_dir/scripts/gaussian_mean_runner.py" \
    --config "$timing_cfg" \
    --output-dir "$timing_out" \
    --test \
    --shard-index 0 \
    --num-shards 1

test -f "$timing_out/gaussian_mean/data/timing.csv"
test -f "$timing_out/_shards/gaussian_mean/runs/default/merge.done.json"

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

echo
echo "Sharded smoke test completed successfully."
echo "Merge smoke output:  $merge_out"
echo "Timing smoke output: $timing_out"
