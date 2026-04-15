#!/usr/bin/env python3
"""Submit scaling_cpm jobs defined in scaling_cpm.json.

Analogous to submit_scaling.py but for the Cellular Potts Model benchmark.
Key difference: CPM runs are wall-time dominated (SMC always hits wall_time_limit),
so elapsed-time extrapolation scales by (full_reps × full_wall) / (active_reps × active_wall)
rather than by simulation count.

Usage
-----
Small-tier dry run to preview jobs::

    python submit_scaling_cpm.py /path/to/output --small --dry-run

Submit small-tier jobs::

    python submit_scaling_cpm.py /path/to/output --small

Submit production jobs (uses timing CSV from a prior small run)::

    python submit_scaling_cpm.py /path/to/output \\
        --timing-csv /path/to/small_output/scaling_cpm/data/timing.csv

Manual fallback (no prior run)::

    python submit_scaling_cpm.py /path/to/output --base-time 26.0
"""
import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

CORES_PER_NODE = 48
SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(EXPERIMENTS_DIR))
from async_abc.io.config import compose_run_mode, load_config  # noqa: E402
from async_abc.utils.sharding import make_run_id  # noqa: E402

DEFAULT_ACCOUNT = "tissuetwin"
DEFAULT_PARTITION = "batch"


def _format_time(hours: float) -> str:
    total_minutes = math.ceil(hours * 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def _read_timing_elapsed(
    timing_csv: Path,
    *,
    experiment_name: str = "scaling_cpm",
    run_mode: str,
) -> float | None:
    if not timing_csv.exists():
        return None
    latest_elapsed = None
    with open(timing_csv) as f:
        for row in csv.DictReader(f):
            if row.get("experiment_name") != experiment_name:
                continue
            if row.get("run_mode") != run_mode:
                continue
            elapsed_s = row.get("elapsed_s", "").strip()
            if not elapsed_s:
                continue
            latest_elapsed = float(elapsed_s)
    return latest_elapsed


# Per-workload overhead for MPI executor lifecycle (measured ~50 s, padded to 60 s).
MPI_EXECUTOR_OVERHEAD_S = 60.0


def _job_time_hours(
    workload_count: int,
    *,
    wall_time_limit_s: float,
    safety: float,
    min_time: float,
    max_time: float,
    finalize_slack_s: float,
    mpi_overhead_s: float = MPI_EXECUTOR_OVERHEAD_S,
) -> float:
    per_workload = float(wall_time_limit_s) + float(mpi_overhead_s)
    time_hours = (
        float(safety) * int(workload_count) * per_workload + float(finalize_slack_s)
    ) / 3600.0
    return max(float(min_time), min(float(max_time), time_hours))


def _effective_wall_time_limit_s(
    scaling_cfg: dict,
    inference_cfg: dict,
    *,
    test_mode: bool,
) -> float:
    wall_time_limit_s = scaling_cfg.get("wall_time_limit_s")
    wall_time_budgets_s = [
        float(value) for value in scaling_cfg.get("wall_time_budgets_s", [])
        if float(value) > 0
    ]
    if wall_time_limit_s in (None, "") and wall_time_budgets_s:
        wall_time_limit_s = max(wall_time_budgets_s)
    wall_time_limit_s = float(wall_time_limit_s or 0.0)
    if test_mode and wall_time_limit_s > 0.0:
        test_wall_limit = inference_cfg.get("max_wall_time_s")
        if test_wall_limit not in (None, ""):
            wall_time_limit_s = min(wall_time_limit_s, float(test_wall_limit))
    return wall_time_limit_s


def _pack_small_worker_counts(worker_counts: list[int], *, capacity: int) -> tuple[list[list[int]], list[int]]:
    small = sorted([int(v) for v in worker_counts if int(v) < int(capacity)], reverse=True)
    large = sorted([int(v) for v in worker_counts if int(v) >= int(capacity)])
    bundles: list[list[int]] = []
    remaining: list[int] = []
    for n_workers in small:
        placed = False
        for idx, free in enumerate(remaining):
            if n_workers <= free:
                bundles[idx].append(n_workers)
                remaining[idx] -= n_workers
                placed = True
                break
        if not placed:
            bundles.append([n_workers])
            remaining.append(int(capacity) - n_workers)
    return [sorted(b) for b in bundles], large


def _render_packed_script(
    *,
    packed_script: Path,
    config_path: Path,
    output_dir: Path,
    workers_csv: str,
    test_mode: bool,
    small_mode: bool,
    extend: bool,
    account: str,
    partition: str,
    time_limit: str,
    ntasks: int,
    job_name: str,
    log_path: Path,
) -> str:
    flags = []
    if test_mode:
        flags.append("--test")
    if small_mode:
        flags.append("--small")
    if extend:
        flags.append("--extend")
    flag_str = " ".join(flags)
    return (
        f"#!/bin/bash -x\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks={ntasks}\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --threads-per-core=2\n"
        f"#SBATCH --time={time_limit}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --output={log_path}\n"
        f"\n"
        f"exec {packed_script} {output_dir} --workers {workers_csv}"
        f" --config {config_path}"
        f"{' ' + flag_str if flag_str else ''}\n"
    )


def _render_standalone_script(
    *,
    scaling_script: Path,
    config_path: Path,
    output_dir: Path,
    test_mode: bool,
    small_mode: bool,
    extend: bool,
    account: str,
    partition: str,
    time_limit: str,
    ntasks: int,
    nodes: int,
    job_name: str,
    log_path: Path,
) -> str:
    flags = []
    if test_mode:
        flags.append("--test")
    if small_mode:
        flags.append("--small")
    if extend:
        flags.append("--extend")
    flag_str = " ".join(flags)
    return (
        f"#!/bin/bash -x\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --nodes={nodes}\n"
        f"#SBATCH --ntasks={ntasks}\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --threads-per-core=2\n"
        f"#SBATCH --time={time_limit}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --output={log_path}\n"
        f"\n"
        f"exec {scaling_script} {output_dir}"
        f" --config {config_path}"
        f"{' ' + flag_str if flag_str else ''}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for experiment results and SLURM logs.",
    )
    parser.add_argument(
        "--config",
        default=str(EXPERIMENTS_DIR / "configs" / "scaling_cpm.json"),
        help="Path to scaling_cpm.json (default: experiments/configs/scaling_cpm.json).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use test-mode worker counts and wall times.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use the small-tier config.",
    )
    parser.add_argument(
        "--timing-csv",
        dest="timing_csv",
        metavar="PATH",
        help=(
            "Path to a timing.csv from a previous matching-mode run "
            "(e.g. <output>/scaling_cpm/data/timing.csv). "
            "Used to derive per-N wall-time estimates."
        ),
    )
    parser.add_argument(
        "--base-time",
        type=float,
        default=None,
        dest="base_time",
        metavar="HOURS",
        help=(
            "Estimated per-worker-count job wall time in hours. "
            "Fallback when --timing-csv is not given. Default: 26.0 h (production), 3.5 h (small)."
        ),
    )
    parser.add_argument(
        "--safety",
        type=float,
        default=1.5,
        metavar="FACTOR",
        help="Safety multiplier on top of estimated wall time. Default: 1.5",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.5,
        dest="min_time",
        metavar="HOURS",
        help="Minimum time limit per job in hours. Default: 0.5",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=24.0,
        dest="max_time",
        metavar="HOURS",
        help="Maximum time limit per job in hours. Default: 24.0",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="Pass --extend to skip already-completed worker counts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    parser.add_argument(
        "--account",
        default=DEFAULT_ACCOUNT,
        help=f"SLURM account (default: {DEFAULT_ACCOUNT}).",
    )
    parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
        help=f"SLURM partition (default: {DEFAULT_PARTITION}).",
    )
    args = parser.parse_args()

    run_mode = compose_run_mode("small" if args.small else "full", args.test)
    config_path = Path(args.config).resolve()
    full_cfg = load_config(config_path, test_mode=False, small_mode=False)
    active_cfg = load_config(config_path, test_mode=args.test, small_mode=args.small)

    scaling_cfg = active_cfg["scaling"]
    full_scaling_cfg = full_cfg["scaling"]
    if args.test:
        worker_counts = [int(w) for w in scaling_cfg.get("test_worker_counts", [1])]
    else:
        worker_counts = [int(w) for w in scaling_cfg["worker_counts"]]
    full_reps = int(full_cfg["execution"]["n_replicates"])
    active_reps = int(active_cfg["execution"]["n_replicates"])
    k_values = list(scaling_cfg.get("k_values", [active_cfg["inference"].get("k", 100)]))
    if args.test:
        k_values = list(scaling_cfg.get("test_k_values", k_values))
    methods = list(active_cfg.get("methods", []))
    n_replicates = int(active_cfg["execution"]["n_replicates"])
    wall_time_limit_s = _effective_wall_time_limit_s(
        scaling_cfg,
        active_cfg["inference"],
        test_mode=args.test,
    )
    finalize_slack_s = max(300.0, 0.1 * wall_time_limit_s * max(1, len(k_values)))
    workload_count = max(1, len(k_values) * max(1, len(methods)) * n_replicates)

    # --- Determine base wall time for the job (in seconds) ---
    # CPM is wall-time dominated: extrapolate by replicate × wall-time ratio,
    # not by simulation count.
    full_wall = _effective_wall_time_limit_s(
        full_scaling_cfg,
        full_cfg["inference"],
        test_mode=False,
    )
    active_wall = wall_time_limit_s

    base_time_s: float
    timing_source: str

    if args.timing_csv:
        active_elapsed = _read_timing_elapsed(
            Path(args.timing_csv),
            experiment_name=active_cfg.get("experiment_name", "scaling_cpm"),
            run_mode=run_mode,
        )
        if active_elapsed is not None:
            # Scale by (full_reps × full_wall) / (active_reps × active_wall)
            wall_ratio = (full_reps * full_wall) / max(1.0, active_reps * active_wall)
            base_time_s = active_elapsed * wall_ratio
            timing_source = (
                f"timing CSV ({run_mode}): {active_elapsed:.1f} s "
                f"× {wall_ratio:.2f} (wall-time ratio {active_reps}rep×{active_wall:.0f}s "
                f"→ {full_reps}rep×{full_wall:.0f}s) = {base_time_s / 3600:.2f} h"
            )
        else:
            print(
                f"Warning: no scaling_cpm row with run_mode={run_mode!r} found in {args.timing_csv}. "
                "Falling back to --base-time."
            )
            default_h = 3.5 if (args.small or args.test) else 26.0
            base_time_s = (args.base_time or default_h) * 3600
            timing_source = f"--base-time fallback: {base_time_s / 3600:.1f} h"
    else:
        default_h = 3.5 if (args.small or args.test) else 26.0
        base_time_s = (args.base_time or default_h) * 3600
        timing_source = f"--base-time: {base_time_s / 3600:.1f} h"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    jobs_dir = output_dir / "_jobs" / "scaling_cpm" / run_id
    jobs_dir.mkdir(parents=True, exist_ok=True)
    scaling_script = SCRIPT_DIR / "scaling_cpm_single.sh"
    packed_script = SCRIPT_DIR / "scaling_cpm_packed.sh"
    packed_bundles, standalone_counts = _pack_small_worker_counts(
        worker_counts,
        capacity=CORES_PER_NODE,
    )

    print(
        f"Config:    {config_path}\n"
        f"Mode:      {run_mode}\n"
        f"Workers:   {worker_counts}\n"
        f"Bundles:   {packed_bundles}\n"
        f"Standalone:{standalone_counts}\n"
        f"Timing:    {timing_source}\n"
        f"k_values:  {k_values}\n"
        f"Methods:   {methods}\n"
        f"Reps:      {n_replicates}\n"
        f"Wall cap:  {wall_time_limit_s:.1f} s\n"
        f"Workload:  {workload_count} combos per worker-count job\n"
        f"Finalize:  {finalize_slack_s:.1f} s slack\n"
        f"Safety:    {args.safety}×  (min={args.min_time} h, max={args.max_time} h)\n"
        f"Output:    {output_dir}\n"
        f"Jobs:      {jobs_dir}\n"
    )

    for bundle in packed_bundles:
        time_hours = _job_time_hours(
            workload_count=workload_count,
            wall_time_limit_s=wall_time_limit_s,
            safety=args.safety,
            min_time=args.min_time,
            max_time=args.max_time,
            finalize_slack_s=finalize_slack_s,
        )
        time_str = _format_time(time_hours)
        bundle_label = "_".join(str(n) for n in bundle)
        workers_csv = ",".join(str(n) for n in bundle)
        job_name = f"abc_scaling_cpm_bundle_{bundle_label}"
        script_path = jobs_dir / f"scaling_cpm_bundle_{bundle_label}.sbatch"
        log_path = jobs_dir / f"scaling_cpm_bundle_{bundle_label}-%j.out"

        script_path.write_text(
            _render_packed_script(
                packed_script=packed_script,
                config_path=config_path,
                output_dir=output_dir,
                workers_csv=workers_csv,
                test_mode=args.test,
                small_mode=args.small,
                extend=args.extend,
                account=args.account,
                partition=args.partition,
                time_limit=time_str,
                ntasks=CORES_PER_NODE,
                job_name=job_name,
                log_path=log_path,
            )
        )

        submit_cmd = ["sbatch", str(script_path)]
        tag = "[dry-run]" if args.dry_run else ""
        print(
            f"  bundle={bundle!r}  ntasks={CORES_PER_NODE}  nodes=1  "
            f"time={time_str}  workload={workload_count}x{wall_time_limit_s:.0f}s+{finalize_slack_s:.0f}s  {tag}"
        )
        if args.dry_run:
            print(f"    {' '.join(submit_cmd)}")
        else:
            subprocess.run(submit_cmd, check=True)

    for n in standalone_counts:
        nodes = math.ceil(n / CORES_PER_NODE)
        time_hours = _job_time_hours(
            workload_count=workload_count,
            wall_time_limit_s=wall_time_limit_s,
            safety=args.safety,
            min_time=args.min_time,
            max_time=args.max_time,
            finalize_slack_s=finalize_slack_s,
        )
        time_str = _format_time(time_hours)
        job_name = f"abc_scaling_cpm_{n}"
        script_path = jobs_dir / f"scaling_cpm_{n}.sbatch"
        log_path = jobs_dir / f"scaling_cpm_{n}-%j.out"

        script_path.write_text(
            _render_standalone_script(
                scaling_script=scaling_script,
                config_path=config_path,
                output_dir=output_dir,
                test_mode=args.test,
                small_mode=args.small,
                extend=args.extend,
                account=args.account,
                partition=args.partition,
                time_limit=time_str,
                ntasks=n,
                nodes=nodes,
                job_name=job_name,
                log_path=log_path,
            )
        )

        submit_cmd = ["sbatch", str(script_path)]
        tag = "[dry-run]" if args.dry_run else ""
        print(
            f"  N={n:3d}  nodes={nodes}  time={time_str}  "
            f"workload={workload_count}x{wall_time_limit_s:.0f}s+{finalize_slack_s:.0f}s  {tag}"
        )
        if args.dry_run:
            print(f"    {' '.join(submit_cmd)}")
        else:
            subprocess.run(submit_cmd, check=True)

    if not args.dry_run:
        print(f"\nAll {len(packed_bundles) + len(standalone_counts)} scaling_cpm jobs submitted.")
        print("Monitor with:  squeue -u $USER")


if __name__ == "__main__":
    main()
