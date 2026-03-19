#!/usr/bin/env python3
"""Submit sharded SLURM jobs for replicate-based experiments and SBC."""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
CORES_PER_NODE = 48
DEFAULT_ACCOUNT = "tissuetwin"
DEFAULT_PARTITION = "batch"
DEFAULT_TIME = "04:00:00"
DEFAULT_NASTJAPY = "/p/project1/tissuetwin/herold2/nastjapy"
DEFAULT_EXPERIMENTS_DIR = "/p/project1/tissuetwin/herold2/async-abc-paper/experiments"

sys.path.insert(0, str(EXPERIMENTS_DIR))

from async_abc.io.config import load_config  # noqa: E402
from async_abc.utils.sharding import ShardLayout, build_plan_payload, split_indices, update_plan  # noqa: E402
import run_all_paper_experiments as run_all  # noqa: E402


def _parse_job_id(output: str) -> str:
    tokens = output.strip().split()
    return tokens[-1] if tokens else ""


def _render_script(
    *,
    runner_path: Path,
    config_path: Path,
    output_dir: Path,
    shard_index: int,
    actual_num_shards: int,
    test_mode: bool,
    extend: bool,
    account: str,
    partition: str,
    time_limit: str,
    ntasks: int,
    nodes: int,
    job_name: str,
    log_path: Path,
    nastjapy_path: str,
    experiments_dir_override: str,
) -> str:
    args = [
        "python",
        str(runner_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(actual_num_shards),
    ]
    if test_mode:
        args.append("--test")
    if extend:
        args.append("--extend")
    command = " ".join(args)
    return f"""#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=2
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}

nastjapy_path={nastjapy_path}
experiments_dir={experiments_dir_override}

module restore nastjapy
module load ParaStationMPI
source "$nastjapy_path/.venv/bin/activate"

mkdir -p "{output_dir}"
srun {command}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Root directory for sharded experiment outputs.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[name for name in run_all.EXPERIMENT_REGISTRY if name != "scaling"],
        help="Experiments to shard-submit (default: all except scaling).",
    )
    parser.add_argument(
        "--jobs-per-experiment",
        type=int,
        default=4,
        help="Requested shard count per experiment before test-mode reduction. Default: 4.",
    )
    parser.add_argument("--test", action="store_true", help="Use test-mode configs and estimate-only sharding.")
    parser.add_argument("--extend", action="store_true", help="Pass --extend to the shard jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Write scripts and print sbatch commands without submitting.")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT, help=f"SLURM account (default: {DEFAULT_ACCOUNT}).")
    parser.add_argument("--partition", default=DEFAULT_PARTITION, help=f"SLURM partition (default: {DEFAULT_PARTITION}).")
    parser.add_argument("--time", dest="time_limit", default=DEFAULT_TIME, help=f"SLURM wall time (default: {DEFAULT_TIME}).")
    parser.add_argument("--nastjapy-path", default=DEFAULT_NASTJAPY, help="Path containing the cluster virtualenv.")
    parser.add_argument(
        "--experiments-dir",
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Cluster path to the experiments directory used inside batch scripts.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    jobs_root = output_dir / "_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    for experiment_name in args.experiments:
        if experiment_name not in run_all.EXPERIMENT_REGISTRY:
            raise SystemExit(f"Unknown experiment: {experiment_name}")
        if experiment_name == "scaling":
            raise SystemExit("Scaling remains on submit_scaling.py and is not supported here.")

        runner_name, config_name = run_all.EXPERIMENT_REGISTRY[experiment_name]
        runner_path = EXPERIMENTS_DIR / "scripts" / runner_name
        config_path = EXPERIMENTS_DIR / "configs" / config_name

        full_cfg = load_config(config_path, test_mode=False)
        actual_cfg = load_config(config_path, test_mode=args.test)
        unit_kind = "trial" if experiment_name == "sbc" else "replicate"
        full_units = int(full_cfg["sbc"]["n_trials"]) if unit_kind == "trial" else int(full_cfg["execution"]["n_replicates"])
        actual_units = int(actual_cfg["sbc"]["n_trials"]) if unit_kind == "trial" else int(actual_cfg["execution"]["n_replicates"])
        requested_num_shards = max(1, min(full_units, int(args.jobs_per_experiment)))
        actual_num_shards = 1 if args.test else max(1, min(actual_units, int(args.jobs_per_experiment)))
        shard_assignments = split_indices(actual_units, actual_num_shards)

        plan_layout = ShardLayout(output_dir, experiment_name, 0)
        plan_path = plan_layout.plan_path
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan = build_plan_payload(
            experiment_name=experiment_name,
            config_path=str(config_path.resolve()),
            runner_script=str(runner_path.resolve()),
            unit_kind=unit_kind,
            full_total_units=full_units,
            actual_total_units=actual_units,
            requested_num_shards=requested_num_shards,
            actual_num_shards=actual_num_shards,
            test_mode=args.test,
            extend=args.extend,
            shard_assignments=shard_assignments,
        )

        n_tasks = int(actual_cfg["inference"]["n_workers"])
        nodes = max(1, math.ceil(n_tasks / CORES_PER_NODE))
        script_dir = jobs_root / experiment_name
        script_dir.mkdir(parents=True, exist_ok=True)
        submitted_job_ids: list[str] = []

        for shard_index in range(actual_num_shards):
            script_path = script_dir / f"{experiment_name}_shard_{shard_index:03d}.sbatch"
            log_path = script_dir / f"{experiment_name}_shard_{shard_index:03d}-%j.out"
            job_name = f"abc_{experiment_name[:32]}_{shard_index:03d}"
            script_path.write_text(
                _render_script(
                    runner_path=runner_path,
                    config_path=config_path,
                    output_dir=output_dir,
                    shard_index=shard_index,
                    actual_num_shards=actual_num_shards,
                    test_mode=args.test,
                    extend=args.extend,
                    account=args.account,
                    partition=args.partition,
                    time_limit=args.time_limit,
                    ntasks=n_tasks,
                    nodes=nodes,
                    job_name=job_name,
                    log_path=log_path,
                    nastjapy_path=args.nastjapy_path,
                    experiments_dir_override=args.experiments_dir,
                )
            )
            submit_cmd = ["sbatch", str(script_path)]
            if args.dry_run:
                print(f"[dry-run] {' '.join(submit_cmd)}")
            else:
                result = subprocess.run(submit_cmd, check=True, capture_output=True, text=True)
                job_id = _parse_job_id(result.stdout)
                submitted_job_ids.append(job_id)
                print(f"{experiment_name} shard {shard_index}: submitted job {job_id}")

        plan["submitted_job_ids"] = submitted_job_ids
        update_plan(plan_layout, plan)


if __name__ == "__main__":
    main()
