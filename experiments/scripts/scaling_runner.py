#!/usr/bin/env python3
"""Scaling experiment: vary n_workers and record throughput.

In test mode only runs with n_workers=1.
On HPC, call this script via ``mpirun -n N`` for each N in worker_counts.
"""
import csv
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.runner import compute_scaling_factor, find_completed_combinations, format_duration, make_arg_parser, run_method_distributed, write_timing_csv
from async_abc.utils.seeding import make_seeds

logger = logging.getLogger(__name__)


def _interp_time(w: int, measured: dict) -> float:
    """Log-linearly interpolate/extrapolate wall time for worker count *w*.

    *measured* maps worker count → average measured wall time (in seconds).
    - Below the smallest tested count: scale linearly (throughput ∝ workers).
    - Between tested counts: log-linear interpolation on wall_time vs log(workers).
    - Above the largest tested count: use the largest tested time (throughput saturates).
    """
    keys = sorted(measured)
    if w <= keys[0]:
        # Linear throughput extrapolation: time ∝ 1/workers
        return measured[keys[0]] * keys[0] / w
    if w >= keys[-1]:
        return measured[keys[-1]]
    # Find bracketing pair
    lo = max(k for k in keys if k <= w)
    hi = min(k for k in keys if k >= w)
    if lo == hi:
        return measured[lo]
    t_lo, t_hi = measured[lo], measured[hi]
    frac = (math.log(w) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(t_lo) + frac * (math.log(t_hi) - math.log(t_lo)))


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Scaling experiment.")
    parser.add_argument(
        "--n-workers", type=int, default=None, dest="n_workers",
        help="Run only this specific worker count (for HPC: match --ntasks=N).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    scaling_cfg = cfg.get("scaling", {})
    worker_counts = scaling_cfg.get("worker_counts", [1])

    if test_mode:
        worker_counts = scaling_cfg.get("test_worker_counts", [1])
    elif args.n_workers is not None:
        worker_counts = [args.n_workers]

    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    throughput_path = output_dir.data / "throughput_summary.csv"
    done = find_completed_combinations(throughput_path, ["n_workers", "method", "replicate"]) if args.extend else set()

    throughput_rows = []
    experiment_start = time.time()
    for n_workers in worker_counts:
        inference_cfg = {**cfg["inference"], "n_workers": n_workers}
        for method in cfg["methods"]:
            for replicate, seed in enumerate(seeds):
                if (str(n_workers), method, str(replicate)) in done:
                    logger.info(
                        "[scaling] --extend: skipping n_workers=%s %s replicate=%s",
                        n_workers,
                        method,
                        replicate,
                    )
                    continue
                t0 = time.time()
                records = run_method_distributed(
                    method, bm.simulate, bm.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
                elapsed = time.time() - t0
                if is_root_rank():
                    n_sims = len(records)
                    throughput = n_sims / elapsed if elapsed > 0 else float("inf")
                    throughput_rows.append({
                        "n_workers": n_workers,
                        "method": method,
                        "replicate": replicate,
                        "seed": seed,
                        "n_simulations": n_sims,
                        "wall_time_s": elapsed,
                        "throughput_sims_per_s": throughput,
                        "test_mode": test_mode,
                    })

    experiment_elapsed = time.time() - experiment_start
    name = cfg["experiment_name"]
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(experiment_elapsed))
    if estimate_mode and is_root_rank():
        cfg_full = load_config(args.config, test_mode=False, small_mode=False)
        full_sims = cfg_full["inference"]["max_simulations"]
        full_reps = cfg_full["execution"]["n_replicates"]
        test_sims = cfg["inference"]["max_simulations"]
        test_reps = cfg["execution"]["n_replicates"]
        sim_ratio = (full_sims * full_reps) / (test_sims * test_reps)
        full_worker_counts = cfg_full.get("scaling", {}).get("worker_counts", worker_counts)

        # Average measured wall_time per worker count (across methods and replicates)
        from collections import defaultdict
        times_by_w: dict = defaultdict(list)
        for row in throughput_rows:
            times_by_w[row["n_workers"]].append(row["wall_time_s"])
        measured_avg = {w: sum(ts) / len(ts) for w, ts in times_by_w.items()}

        # Sum per-config full-run estimates; interpolate for untested configs
        estimated = sum(
            _interp_time(w, measured_avg) * sim_ratio for w in full_worker_counts
        )
        _, _, note = compute_scaling_factor(
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        logger.info(
            "[%s] Estimated full run: ~%s  (%s)",
            name,
            format_duration(estimated),
            note,
        )
    if not is_root_rank():
        return

    write_timing_csv(output_dir.data / "timing.csv", name, experiment_elapsed, estimated, test_mode, run_mode)

    # Write throughput summary — append when extending, overwrite otherwise
    if throughput_rows:
        write_header = not args.extend or not throughput_path.exists() or throughput_path.stat().st_size == 0
        mode = "a" if args.extend and not write_header else "w"
        with open(throughput_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(throughput_rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(throughput_rows)

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("scaling_curve") or plots_cfg.get("efficiency"):
        from async_abc.plotting.reporters import plot_scaling_summary

        plot_scaling_summary(throughput_rows, output_dir)

    write_metadata(output_dir, cfg, extra={"worker_counts": worker_counts})


if __name__ == "__main__":
    main()
