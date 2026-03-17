#!/usr/bin/env python3
"""Submit one sbatch scaling job per worker count defined in scaling.json.

Worker counts are read directly from scaling.json so the script stays in sync
with any config changes.

Wall-time limits are derived from a previous test run's timing output:

    base_time(N=1) = test_elapsed × (full_sims × full_reps) / (test_sims × test_reps)
    time_limit(N)  = clamp(base_time(N=1) / N × safety, min=15 min, max=24 h)

This means larger N (more workers) automatically get shorter time limits.

Usage
-----
After running test_all.sh, use its timing CSV to set wall times automatically::

    python submit_scaling.py /path/to/output \\
        --timing-csv /path/to/test_output/timing_summary.csv

Dry run to preview without submitting::

    python submit_scaling.py /path/to/output --dry-run \\
        --timing-csv /path/to/test_output/timing_summary.csv

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
from async_abc.io.config import load_config  # noqa: E402


def _format_time(hours: float) -> str:
    total_minutes = math.ceil(hours * 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:00"


def _read_test_elapsed(timing_csv: Path, experiment_name: str = "scaling") -> float | None:
    """Return elapsed_s from the most recent test-mode row for experiment_name."""
    if not timing_csv.exists():
        return None
    rows = []
    with open(timing_csv) as f:
        for row in csv.DictReader(f):
            if row["experiment_name"] == experiment_name and row["test_mode"] == "True":
                rows.append(row)
    return float(rows[-1]["elapsed_s"]) if rows else None  # most recent


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
        "--timing-csv",
        dest="timing_csv",
        metavar="PATH",
        help=(
            "Path to timing_summary.csv from a previous test run. "
            "Used to derive per-N wall-time limits from the measured test elapsed time."
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
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    full_cfg = load_config(config_path, test_mode=False)
    test_cfg = load_config(config_path, test_mode=True)

    worker_counts: list[int] = full_cfg["scaling"]["worker_counts"]
    full_sims = full_cfg["inference"]["max_simulations"]
    full_reps = full_cfg["execution"]["n_replicates"]
    test_sims = test_cfg["inference"]["max_simulations"]
    test_reps = test_cfg["execution"]["n_replicates"]

    # --- Determine base wall time for N=1 production (in seconds) ---
    base_time_s: float
    timing_source: str

    if args.timing_csv:
        test_elapsed = _read_test_elapsed(Path(args.timing_csv))
        if test_elapsed is not None:
            ratio = (full_sims * full_reps) / (test_sims * test_reps)
            base_time_s = test_elapsed * ratio
            timing_source = (
                f"timing CSV: {test_elapsed:.1f} s (test) × {ratio:.1f} "
                f"= {base_time_s / 3600:.2f} h estimated for N=1"
            )
        else:
            print(
                f"Warning: no scaling test-mode row found in {args.timing_csv}. "
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

    print(
        f"Config:    {config_path}\n"
        f"Workers:   {worker_counts}\n"
        f"Base time: {timing_source}\n"
        f"Safety:    {args.safety}×  (min={args.min_time} h, max={args.max_time} h)\n"
        f"Output:    {output_dir}\n"
    )

    for n in worker_counts:
        nodes = math.ceil(n / CORES_PER_NODE)
        time_hours = base_time_s / n * args.safety / 3600
        time_hours = max(args.min_time, min(args.max_time, time_hours))
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
        ]

        tag = "(dry run)" if args.dry_run else ""
        print(f"  N={n:3d}  nodes={nodes}  time={time_str}  {tag}")
        if args.dry_run:
            print(f"    {' '.join(cmd)}")
        else:
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        print(f"\nAll {len(worker_counts)} scaling jobs submitted.")
        print("Monitor with:  squeue -u $USER")


if __name__ == "__main__":
    main()
