#!/usr/bin/env python3
"""Submit sharded SLURM jobs for replicate-based experiments and SBC."""
from __future__ import annotations

import argparse
import csv
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

sys.path.insert(0, str(EXPERIMENTS_DIR))

from async_abc.io.config import load_config  # noqa: E402
from async_abc.utils.sharding import (  # noqa: E402
    ShardLayout,
    build_plan_payload,
    detect_completed_replicates,
    make_run_id,
    split_indices,
    split_items,
    update_plan,
    validate_extension_compatibility,
)
import run_all_paper_experiments as run_all  # noqa: E402


def _default_sharded_experiments() -> list[str]:
    """Return all shard-supported experiments in registry order."""
    return [name for name in run_all.EXPERIMENT_REGISTRY if name != "scaling"]


def _resolve_experiments(requested: list[str]) -> list[str]:
    """Expand the special 'all' token to every shard-supported experiment."""
    if "all" in requested:
        return _default_sharded_experiments()
    return requested


def _read_sharded_estimate(output_dir: Path, experiment_name: str) -> float | None:
    """Return the most recent estimated_full_sharded_wall_s from a prior test run, or None."""
    timing_csv = output_dir / experiment_name / "data" / "timing.csv"
    if not timing_csv.exists() or timing_csv.stat().st_size == 0:
        return None
    try:
        with open(timing_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in reversed(rows):
            if row.get("test_mode", "").lower() not in ("true", "1"):
                continue
            val = row.get("estimated_full_sharded_wall_s", "").strip()
            if val:
                return float(val)
    except Exception:
        pass
    return None


def _format_slurm_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS for SLURM --time."""
    s = max(1, int(math.ceil(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


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
    shard_run_id: str,
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
        "--shard-run-id",
        shard_run_id,
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
        default=_default_sharded_experiments(),
        help="Experiments to shard-submit, or 'all' for all except scaling (default).",
    )
    parser.add_argument(
        "--jobs-per-experiment",
        type=int,
        default=4,
        help="Requested shard count per experiment before test-mode reduction. Default: 4.",
    )
    parser.add_argument("--test", action="store_true", help="Use test-mode configs and estimate-only sharding.")
    parser.add_argument("--extend", action="store_true", help="Pass --extend to the shard jobs.")
    parser.add_argument(
        "--add-replicates",
        action="store_true",
        help="Treat config execution.n_replicates as an additional replicate count and submit only missing replicates.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write scripts and print sbatch commands without submitting.")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT, help=f"SLURM account (default: {DEFAULT_ACCOUNT}).")
    parser.add_argument("--partition", default=DEFAULT_PARTITION, help=f"SLURM partition (default: {DEFAULT_PARTITION}).")
    parser.add_argument("--time", dest="time_limit", default=DEFAULT_TIME, help=f"SLURM wall time (default: {DEFAULT_TIME}).")
    parser.add_argument("--nastjapy-path", default=DEFAULT_NASTJAPY, help="Path containing the cluster virtualenv.")
    args = parser.parse_args()
    experiment_names = _resolve_experiments(args.experiments)

    if args.add_replicates and args.test:
        raise SystemExit("--add-replicates is not supported with --test")
    extend_mode = bool(args.extend or args.add_replicates)

    output_dir = Path(args.output_dir).resolve()
    jobs_root = output_dir / "_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    for experiment_name in experiment_names:
        if experiment_name not in run_all.EXPERIMENT_REGISTRY:
            raise SystemExit(f"Unknown experiment: {experiment_name}")
        if experiment_name == "scaling":
            raise SystemExit("Scaling remains on submit_scaling.py and is not supported here.")
        if args.add_replicates and experiment_name == "sbc":
            raise SystemExit("--add-replicates is only supported for replicate-based experiments, not sbc")

        runner_name, config_name = run_all.EXPERIMENT_REGISTRY[experiment_name]
        runner_path = EXPERIMENTS_DIR / "scripts" / runner_name
        config_path = EXPERIMENTS_DIR / "configs" / config_name

        full_cfg = load_config(config_path, test_mode=False)
        actual_cfg = load_config(config_path, test_mode=args.test)
        unit_kind = "trial" if experiment_name == "sbc" else "replicate"
        full_units = int(full_cfg["sbc"]["n_trials"]) if unit_kind == "trial" else int(full_cfg["execution"]["n_replicates"])
        actual_units = int(actual_cfg["sbc"]["n_trials"]) if unit_kind == "trial" else int(actual_cfg["execution"]["n_replicates"])
        if extend_mode:
            validate_extension_compatibility(output_dir, full_cfg)
        completed_units = detect_completed_replicates(output_dir, full_cfg) if unit_kind == "replicate" else []
        if args.add_replicates:
            target_total_units = len(completed_units) + full_units
        else:
            target_total_units = full_units
        if args.test:
            pending_units = list(range(actual_units))
            target_total_units = full_units
            completed_units = []
        else:
            pending_units = [idx for idx in range(target_total_units) if idx not in set(completed_units)]
        if extend_mode and not pending_units:
            print(f"{experiment_name}: nothing to do; all {target_total_units} replicates are already complete")
            continue

        if args.test:
            requested_num_shards = max(1, min(target_total_units, int(args.jobs_per_experiment)))
        else:
            requested_num_shards = max(1, min(len(pending_units), int(args.jobs_per_experiment)))
        actual_num_shards = 1 if args.test else max(1, min(len(pending_units), int(args.jobs_per_experiment)))
        if args.test and pending_units:
            pending_units = pending_units[:actual_num_shards]
        shard_assignments = split_items(pending_units, actual_num_shards)
        run_id = make_run_id()

        plan_layout = ShardLayout(output_dir, experiment_name, run_id, 0)
        plan_layout.plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan = build_plan_payload(
            experiment_name=experiment_name,
            config_path=str(config_path.resolve()),
            runner_script=str(runner_path.resolve()),
            unit_kind=unit_kind,
            full_total_units=full_units,
            actual_total_units=actual_units,
            target_total_units=target_total_units,
            requested_num_shards=requested_num_shards,
            actual_num_shards=actual_num_shards,
            test_mode=args.test,
            extend=extend_mode,
            run_id=run_id,
            completed_unit_indices=completed_units,
            pending_unit_indices=pending_units,
            shard_assignments=shard_assignments,
        )

        n_tasks = int(actual_cfg["inference"]["n_workers"])
        nodes = max(1, math.ceil(n_tasks / CORES_PER_NODE))
        script_dir = jobs_root / experiment_name / run_id
        script_dir.mkdir(parents=True, exist_ok=True)
        submitted_job_ids: list[str] = []

        # Use timing estimate from a prior test run if available; add 50 % margin.
        _TIMING_MARGIN = 1.5
        _MIN_WALL_S = 30 * 60  # 30 minutes floor
        estimated_s = _read_sharded_estimate(output_dir, experiment_name)
        if estimated_s is not None:
            time_limit = _format_slurm_time(max(_MIN_WALL_S, estimated_s * _TIMING_MARGIN))
            print(f"{experiment_name}: using estimated wall time {time_limit} (estimate={estimated_s:.0f}s × {_TIMING_MARGIN})")
        else:
            time_limit = args.time_limit

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
                    shard_run_id=run_id,
                    test_mode=args.test,
                    extend=extend_mode,
                    account=args.account,
                    partition=args.partition,
                    time_limit=time_limit,
                    ntasks=n_tasks,
                    nodes=nodes,
                    job_name=job_name,
                    log_path=log_path,
                    nastjapy_path=args.nastjapy_path,
                )
            )
            submit_cmd = ["sbatch", str(script_path)]
            if args.dry_run:
                print(f"[dry-run] {' '.join(submit_cmd)}")
            else:
                result = subprocess.run(submit_cmd, check=True, capture_output=True, text=True)
                job_id = _parse_job_id(result.stdout)
                submitted_job_ids.append(job_id)
                print(f"{experiment_name} run {run_id} shard {shard_index}: submitted job {job_id}")

        plan["submitted_job_ids"] = submitted_job_ids
        update_plan(plan_layout, plan)


if __name__ == "__main__":
    main()
