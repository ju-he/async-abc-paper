#!/usr/bin/env python3
"""Submit scaling jobs defined in scaling.json.

Worker counts are read directly from scaling.json so the script stays in sync
with any config changes.

Worker counts smaller than ``CORES_PER_NODE`` are packed into shared single-node
jobs so low-rank runs do not waste full-node allocations on systems without
partial-node scheduling. Each packed job launches one ``srun --exclusive`` per
worker count and lets them run concurrently on the same node.

Wall-time limits are derived from a previous matching-mode timing output:

    base_time(N=1) = active_elapsed × (full_sims × full_reps) / (active_sims × active_reps)
    time_limit(N)  = clamp(base_time(N=1) / N × safety, min=15 min, max=24 h)

This means larger N (more workers) automatically get shorter time limits.

Usage
-----
After running a mode-matched timing run, use its timing CSV to set wall times automatically::

    python submit_scaling.py /path/to/output \\
        --timing-csv /path/to/test_output/timing_summary_test.csv

Small-tier dry run based on small-test timings::

    python submit_scaling.py /path/to/output --small --test --dry-run \\
        --timing-csv /path/to/test_output/timing_summary_small_test.csv

Dry run to preview without submitting::

    python submit_scaling.py /path/to/output --dry-run \\
        --timing-csv /path/to/test_output/timing_summary_test.csv

Manual fallback (no test run yet)::

    python submit_scaling.py /path/to/output --base-time 2.0
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


def _format_time(hours: float) -> str:
    total_minutes = math.ceil(hours * 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def _read_timing_elapsed(
    timing_csv: Path,
    *,
    experiment_name: str = "scaling",
    run_mode: str,
) -> float | None:
    """Return elapsed_s from the most recent timing row matching experiment_name and run_mode."""
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


def _job_time_hours(
    workload_count: int,
    *,
    wall_time_limit_s: float,
    safety: float,
    min_time: float,
    max_time: float,
    finalize_slack_s: float,
) -> float:
    time_hours = (
        float(safety) * int(workload_count) * float(wall_time_limit_s) + float(finalize_slack_s)
    ) / 3600.0
    return max(float(min_time), min(float(max_time), time_hours))


def _pack_small_worker_counts(worker_counts: list[int], *, capacity: int) -> tuple[list[list[int]], list[int]]:
    """Pack worker counts smaller than *capacity* into shared-node bundles."""
    small = sorted([int(value) for value in worker_counts if int(value) < int(capacity)], reverse=True)
    large = sorted([int(value) for value in worker_counts if int(value) >= int(capacity)])
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

    return [sorted(bundle) for bundle in bundles], large


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
        default=str(EXPERIMENTS_DIR / "configs" / "scaling.json"),
        help="Path to scaling.json (default: experiments/configs/scaling.json).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use test-mode worker counts and timing rows.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use the small-tier config while still estimating full production wall times.",
    )
    parser.add_argument(
        "--timing-csv",
        dest="timing_csv",
        metavar="PATH",
        help=(
            "Path to a timing_summary_<run_mode>.csv from a previous matching-mode run. "
            "Used to derive per-N wall-time limits from the measured elapsed time."
        ),
    )
    parser.add_argument(
        "--base-time",
        type=float,
        default=None,
        dest="base_time",
        metavar="HOURS",
        help=(
            "Estimated N=1 production wall time in hours. "
            "Fallback when --timing-csv is not given. Default: 4.0 h."
        ),
    )
    parser.add_argument(
        "--safety",
        type=float,
        default=2.0,
        metavar="FACTOR",
        help="Safety multiplier applied on top of the estimated wall time. Default: 2.0",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.25,
        dest="min_time",
        metavar="HOURS",
        help="Minimum time limit per job in hours. Default: 0.25 (15 min).",
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
        help="Pass --extend to scaling_runner.py to skip already-completed worker counts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    args = parser.parse_args()

    run_mode = compose_run_mode("small" if args.small else "full", args.test)
    config_path = Path(args.config).resolve()
    full_cfg = load_config(config_path, test_mode=False, small_mode=False)
    active_cfg = load_config(config_path, test_mode=args.test, small_mode=args.small)

    scaling_cfg = active_cfg["scaling"]
    worker_counts: list[int]
    if args.test:
        worker_counts = scaling_cfg.get("test_worker_counts", [1])
    else:
        worker_counts = scaling_cfg["worker_counts"]
    full_sims = full_cfg["inference"]["max_simulations"]
    full_reps = full_cfg["execution"]["n_replicates"]
    active_sims = active_cfg["inference"]["max_simulations"]
    active_reps = active_cfg["execution"]["n_replicates"]
    k_values = list(scaling_cfg.get("k_values", [active_cfg["inference"].get("k", 100)]))
    methods = list(active_cfg.get("methods", []))
    n_replicates = int(active_cfg["execution"]["n_replicates"])
    wall_time_limit_s = scaling_cfg.get("wall_time_limit_s")
    wall_time_budgets_s = [float(value) for value in scaling_cfg.get("wall_time_budgets_s", []) if float(value) > 0]
    if wall_time_limit_s in (None, "") and wall_time_budgets_s:
        wall_time_limit_s = max(wall_time_budgets_s)
    wall_time_limit_s = float(wall_time_limit_s or 0.0)
    finalize_slack_s = max(300.0, 0.1 * wall_time_limit_s * max(1, len(k_values)))
    workload_count = max(1, len(k_values) * max(1, len(methods)) * n_replicates)

    # --- Determine base wall time for N=1 production (in seconds) ---
    base_time_s: float
    timing_source: str

    if args.timing_csv:
        active_elapsed = _read_timing_elapsed(Path(args.timing_csv), run_mode=run_mode)
        if active_elapsed is not None:
            ratio = (full_sims * full_reps) / (active_sims * active_reps)
            base_time_s = active_elapsed * ratio
            timing_source = (
                f"timing CSV ({run_mode}): {active_elapsed:.1f} s × {ratio:.1f} "
                f"= {base_time_s / 3600:.2f} h estimated for N=1"
            )
        else:
            print(
                f"Warning: no scaling row with run_mode={run_mode!r} found in {args.timing_csv}. "
                "Falling back to --base-time."
            )
            base_time_s = (args.base_time or 4.0) * 3600
            timing_source = f"--base-time fallback: {base_time_s / 3600:.1f} h"
    else:
        base_time_s = (args.base_time or 4.0) * 3600
        timing_source = f"--base-time: {base_time_s / 3600:.1f} h"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaling_script = SCRIPT_DIR / "scaling_single.sh"
    packed_script = SCRIPT_DIR / "scaling_packed.sh"
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
    )

    for bundle in packed_bundles:
        bundle_job_hours = [
            _job_time_hours(
                workload_count=workload_count,
                wall_time_limit_s=wall_time_limit_s,
                safety=args.safety,
                min_time=args.min_time,
                max_time=args.max_time,
                finalize_slack_s=finalize_slack_s,
            )
            for _n in bundle
        ]
        time_hours = max(bundle_job_hours)
        time_str = _format_time(time_hours)
        bundle_label = "_".join(str(n) for n in bundle)
        workers_csv = ",".join(str(n) for n in bundle)

        cmd = [
            "sbatch",
            f"--ntasks={CORES_PER_NODE}",
            "--nodes=1",
            f"--time={time_str}",
            f"--job-name=abc_scaling_bundle_{bundle_label}",
            f"--output={output_dir}/abc_scaling_bundle_{bundle_label}-%j.out",
            str(packed_script),
            str(output_dir),
            "--workers",
            workers_csv,
            "--config",
            str(config_path),
            *(["--test"] if args.test else []),
            *(["--small"] if args.small else []),
            *(["--extend"] if args.extend else []),
        ]

        tag = "(dry run)" if args.dry_run else ""
        print(
            f"  bundle={bundle!r}  ntasks={CORES_PER_NODE}  nodes=1  "
            f"time={time_str}  workload={workload_count}x{wall_time_limit_s:.0f}s+{finalize_slack_s:.0f}s  {tag}"
        )
        if args.dry_run:
            print(f"    {' '.join(cmd)}")
        else:
            subprocess.run(cmd, check=True)

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
        cmd = [
            "sbatch",
            f"--ntasks={n}",
            f"--nodes={nodes}",
            f"--time={time_str}",
            f"--job-name=abc_scaling_{n}",
            f"--output={output_dir}/abc_scaling_{n}-%j.out",
            str(scaling_script),
            str(output_dir),
            "--config",
            str(config_path),
            *(["--test"] if args.test else []),
            *(["--small"] if args.small else []),
            *(["--extend"] if args.extend else []),
        ]

        tag = "(dry run)" if args.dry_run else ""
        print(
            f"  N={n:3d}  nodes={nodes}  time={time_str}  "
            f"workload={workload_count}x{wall_time_limit_s:.0f}s+{finalize_slack_s:.0f}s  {tag}"
        )
        if args.dry_run:
            print(f"    {' '.join(cmd)}")
        else:
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        print(f"\nAll {len(packed_bundles) + len(standalone_counts)} scaling jobs submitted.")
        print("Monitor with:  squeue -u $USER")


if __name__ == "__main__":
    main()
