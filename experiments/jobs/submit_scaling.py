#!/usr/bin/env python3
"""Submit one sbatch scaling job per worker count defined in scaling.json.

Time limits are computed as:
    time(N) = max(--min-time, --base-time / N * --safety)

so jobs with more workers get shorter limits (they finish faster).

Usage
-----
Dry run to preview submissions::

    python submit_scaling.py /path/to/output --dry-run

Submit with default time estimates (base 4 h for N=1, 3x safety factor)::

    python submit_scaling.py /path/to/output

Override base wall-time estimate (e.g. if test run showed N=1 takes ~1 h)::

    python submit_scaling.py /path/to/output --base-time 1.0
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

CORES_PER_NODE = 48
SCRIPT_DIR = Path(__file__).parent


def _format_time(hours: float) -> str:
    total_minutes = math.ceil(hours * 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:00"


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
        default=str(SCRIPT_DIR.parent / "configs" / "scaling.json"),
        help="Path to scaling.json (default: experiments/configs/scaling.json).",
    )
    parser.add_argument(
        "--base-time",
        type=float,
        default=4.0,
        dest="base_time",
        metavar="HOURS",
        help=(
            "Estimated wall time (hours) for N=1. "
            "Larger N get shorter limits: max(min_time, base_time / N * safety). "
            "Run test_all.sh first and read the printed estimate to calibrate this. "
            "Default: 4.0"
        ),
    )
    parser.add_argument(
        "--safety",
        type=float,
        default=3.0,
        metavar="FACTOR",
        help="Safety multiplier applied on top of the inverse-scaled estimate. Default: 3.0",
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
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    worker_counts: list[int] = config["scaling"]["worker_counts"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaling_script = SCRIPT_DIR / "scaling_single.sh"

    print(
        f"Config:    {config_path}\n"
        f"Workers:   {worker_counts}\n"
        f"Base time: {args.base_time} h (N=1), safety={args.safety}x, "
        f"min={args.min_time} h\n"
        f"Output:    {output_dir}\n"
    )

    for n in worker_counts:
        nodes = math.ceil(n / CORES_PER_NODE)
        time_hours = max(args.min_time, args.base_time / n * args.safety)
        time_str = _format_time(time_hours)

        cmd = [
            "sbatch",
            f"--ntasks={n}",
            f"--nodes={nodes}",
            f"--time={time_str}",
            f"--job-name=abc_scaling_{n}",
            f"--output={output_dir}/abc_scaling_{n}-%j.out",
            str(scaling_script),
            str(output_dir),   # passed as $1 to scaling_single.sh
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
