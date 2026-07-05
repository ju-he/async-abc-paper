#!/usr/bin/env python3
"""Submit realistic-workload scaling jobs across a power-of-2 worker/node sweep.

Thin wrapper over :mod:`submit_scaling_realistic`: it generates a power-of-2 sweep
(configurable maximum), writes a derived ``scaling_realistic`` config carrying those
worker counts, prints a total **CPU-hour / node-hour** estimate, and then
delegates the actual submission to ``submit_scaling_realistic.py``.  All the heavy
lifting --- sbatch rendering, node math (``nodes = ceil(N / 48)``), single-node
bin-packing of sub-node worker counts, the per-combo MPI-isolated run wrappers
(``PROPULATE_SKIP_DISCONNECT`` etc.), and SLURM account/partition detection ---
is reused unchanged, so this stays a thin, low-risk layer on top.

Sweep selection (mutually exclusive, one required):

* ``--max-workers N``  worker counts = ``1, 2, 4, ..., N``       (N a power of 2)
* ``--max-nodes M``    full-node steps = ``48 * (1, 2, 4, ..., M)`` (M a power of 2)

Examples
--------
Estimate CPU-hours for a worker sweep up to 256 (no submission)::

    python submit_realistic_node_scaling.py /scratch/.../run_realistic_nodes \\
        --max-workers 256 --dry-run

Submit a full-node sweep up to 8 nodes (48, 96, 192, 384 workers)::

    python submit_realistic_node_scaling.py /scratch/.../run_realistic_nodes --max-nodes 8

Shorter runs (override per-run wall-time / replicates)::

    python submit_realistic_node_scaling.py /scratch/.../run_realistic_nodes \\
        --max-workers 128 --wall-time-s 900 --reps 3 --dry-run
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

# Reuse the production submitter's helpers + constants verbatim.
import submit_scaling_realistic as scpm  # noqa: E402

CORES_PER_NODE = scpm.CORES_PER_NODE


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _powers_of_two_up_to(n: int) -> list[int]:
    out, p = [], 1
    while p <= n:
        out.append(p)
        p *= 2
    return out


def _worker_sweep(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[int]:
    if args.max_workers is not None:
        if not _is_power_of_two(args.max_workers):
            parser.error("--max-workers must be a power of 2")
        return _powers_of_two_up_to(args.max_workers)
    if not _is_power_of_two(args.max_nodes):
        parser.error("--max-nodes must be a power of 2")
    return [CORES_PER_NODE * m for m in _powers_of_two_up_to(args.max_nodes)]


def _estimate(cfg: dict, worker_counts: list[int], args: argparse.Namespace) -> None:
    """Print a per-job and total node-hour / CPU-hour estimate.

    Mirrors the job layout that ``submit_scaling_realistic.py`` will produce: sub-node
    worker counts are bin-packed onto single nodes, larger counts run standalone
    on ``ceil(N / 48)`` nodes; every job's wall time comes from the same
    ``_job_time_hours`` model the submitter uses.
    """
    methods = list(cfg.get("methods", []))
    reps = int(cfg["execution"]["n_replicates"])
    k_values = list(cfg["scaling"]["k_values"])
    wall_s = scpm._effective_wall_time_limit_s(cfg["scaling"], cfg["inference"], test_mode=False)
    workload = max(1, len(k_values) * max(1, len(methods)) * reps)
    finalize_slack = max(300.0, 0.1 * wall_s * max(1, len(k_values)))

    def job_hours() -> float:
        return scpm._job_time_hours(
            workload_count=workload,
            wall_time_limit_s=wall_s,
            safety=args.safety,
            min_time=args.min_time,
            max_time=args.max_time,
            finalize_slack_s=finalize_slack,
        )

    bundles, standalone = scpm._pack_small_worker_counts(worker_counts, capacity=CORES_PER_NODE)

    print(f"Realistic-workload node-scaling sweep")
    print(f"  worker counts : {worker_counts}")
    print(f"  k_values      : {k_values}")
    print(f"  methods       : {methods}")
    print(f"  replicates    : {reps}")
    print(f"  wall/run      : {wall_s:.0f} s   (workload {workload} combos/job, +{finalize_slack:.0f}s finalize)")
    print(f"  packing       : bundles={bundles}  standalone={standalone}\n")

    header = f"  {'job':<22}{'nodes':>6}{'walltime':>11}{'node-h':>9}{'CPU-h':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    total_node_h = total_cpu_h = 0.0
    for bundle in bundles:
        th = job_hours()
        node_h, cpu_h = 1 * th, 1 * th * CORES_PER_NODE
        total_node_h += node_h
        total_cpu_h += cpu_h
        label = "bundle " + "+".join(str(n) for n in bundle)
        print(f"  {label:<22}{1:>6}{scpm._format_time(th):>11}{node_h:>9.1f}{cpu_h:>9.0f}")
    for n in standalone:
        nodes = math.ceil(n / CORES_PER_NODE)
        th = job_hours()
        node_h, cpu_h = nodes * th, nodes * th * CORES_PER_NODE
        total_node_h += node_h
        total_cpu_h += cpu_h
        print(f"  {'workers ' + str(n):<22}{nodes:>6}{scpm._format_time(th):>11}{node_h:>9.1f}{cpu_h:>9.0f}")
    print("  " + "-" * (len(header) - 2))
    n_jobs = len(bundles) + len(standalone)
    print(f"  {'TOTAL (' + str(n_jobs) + ' jobs)':<22}{'':>6}{'':>11}{total_node_h:>9.1f}{total_cpu_h:>9.0f}")
    print(f"\n  => {total_node_h:.1f} node-hours  =  {total_cpu_h:.0f} CPU-hours  "
          f"(worst-case wall, {CORES_PER_NODE} cores/node, {args.safety}x safety)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dir", help="Output dir for results, derived config, and SLURM logs.")
    sweep = parser.add_mutually_exclusive_group(required=True)
    sweep.add_argument("--max-workers", type=int, default=None,
                       help="Largest worker count (power of 2); sweep = 1,2,4,...,N.")
    sweep.add_argument("--max-nodes", type=int, default=None,
                       help="Largest node count (power of 2); sweep = 48*[1,2,4,...,M] (full nodes).")
    parser.add_argument("--config", default=str(EXPERIMENTS_DIR / "configs" / "scaling_realistic.json"),
                        help="Base scaling_realistic config (default: experiments/configs/scaling_realistic.json).")
    parser.add_argument("--k", type=int, default=100, help="Single archive size k (default: 100).")
    parser.add_argument("--wall-time-s", type=float, default=None, dest="wall_time_s",
                        help="Override per-run wall_time_limit_s (default: from config).")
    parser.add_argument("--reps", type=int, default=None, help="Override n_replicates (default: from config).")
    parser.add_argument("--safety", type=float, default=1.5, help="Wall-time safety multiplier (default: 1.5).")
    parser.add_argument("--min-time", type=float, default=0.5, dest="min_time", help="Min job hours (default: 0.5).")
    parser.add_argument("--max-time", type=float, default=24.0, dest="max_time", help="Max job hours (default: 24).")
    parser.add_argument("--extend", action="store_true", help="Skip already-completed worker counts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the estimate and the sbatch commands; do not submit.")
    parser.add_argument("--account", default=None, help="SLURM account (default: auto-detect).")
    parser.add_argument("--partition", default=None, help="SLURM partition (default: auto-detect).")
    args = parser.parse_args()

    worker_counts = _worker_sweep(args, parser)

    # Derive a config from the raw base JSON (avoid re-processing a loaded config).
    base = json.loads(Path(args.config).read_text())
    base["scaling"]["worker_counts"] = worker_counts
    base["scaling"]["k_values"] = [args.k]
    if args.wall_time_s is not None:
        base["scaling"]["wall_time_limit_s"] = args.wall_time_s
        base["scaling"]["wall_time_budgets_s"] = [args.wall_time_s]
    if args.reps is not None:
        base["execution"]["n_replicates"] = args.reps

    _estimate(base, worker_counts, args)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    derived = out / "scaling_realistic_nodesweep.json"
    derived.write_text(json.dumps(base, indent=2))
    print(f"  derived config written: {derived}")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "submit_scaling_realistic.py"), str(out),
        "--config", str(derived),
        "--safety", str(args.safety),
        "--min-time", str(args.min_time),
        "--max-time", str(args.max_time),
    ]
    if args.extend:
        cmd.append("--extend")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.account:
        cmd += ["--account", args.account]
    if args.partition:
        cmd += ["--partition", args.partition]

    print(f"  delegating to: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if args.dry_run:
            # The CPU-hour estimate above is independent of site detection; only the
            # sbatch-command preview (and real submission) needs the cluster's SLURM
            # site config (SYSTEMNAME) or SIM_BACKEND_PATH. Don't fail the estimate, and
            # don't surface the off-cluster traceback.
            print("\n  NOTE: the sbatch preview/submission step needs the target cluster's\n"
                  "  site detection (run on the JUWELS login node) or SIM_BACKEND_PATH set.\n"
                  "  The CPU-hour estimate above does not depend on it.")
        else:
            sys.stderr.write(result.stderr)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
