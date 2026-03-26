#!/usr/bin/env python3
"""Scaling experiment: vary n_workers and record throughput.

In test mode only runs with n_workers=1.
On HPC, call this script via ``mpirun -n N`` for each N in worker_counts.
"""
import csv
import logging
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.runner import compute_scaling_factor, format_duration, make_arg_parser, run_method_distributed, write_timing_comparison_csv, write_timing_csv
from async_abc.utils.seeding import make_seeds

logger = logging.getLogger(__name__)

_THROUGHPUT_FIELDNAMES = [
    "n_workers",
    "method",
    "replicate",
    "seed",
    "n_simulations",
    "wall_time_s",
    "throughput_sims_per_s",
    "test_mode",
]
_THROUGHPUT_SHARD_RE = re.compile(r"throughput_summary_w(?P<n_workers>\d+)\.csv$")


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


def _simulation_count(records) -> int:
    """Return a method-agnostic simulation-attempt count for throughput summaries."""
    attempt_records = [record for record in records if record.record_kind == "simulation_attempt"]
    if attempt_records:
        return len(attempt_records)

    attempt_counts = [
        int(record.attempt_count)
        for record in records
        if record.attempt_count is not None
    ]
    if attempt_counts:
        return max(attempt_counts)
    return len(records)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    try:
        path.chmod(0o644)
    except OSError:
        pass


def _scaling_shard_path(data_dir: Path, n_workers: int) -> Path:
    return data_dir / f"throughput_summary_w{n_workers}.csv"


def _load_scaling_rows(output_dir: OutputDir) -> list[dict]:
    shard_paths = []
    for path in sorted(output_dir.data.glob("throughput_summary_w*.csv")):
        if _THROUGHPUT_SHARD_RE.fullmatch(path.name):
            shard_paths.append(path)
    if shard_paths:
        rows = []
        for path in shard_paths:
            rows.extend(_read_rows(path))
        return rows
    return _read_rows(output_dir.data / "throughput_summary.csv")


def _find_completed_scaling(output_dir: OutputDir, key_cols: list[str]) -> set[tuple[str, ...]]:
    rows = _load_scaling_rows(output_dir)
    return {tuple(str(row.get(col, "")) for col in key_cols) for row in rows}


def _sort_throughput_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["n_workers"]),
            str(row["method"]),
            int(row["replicate"]),
            int(row["seed"]),
        ),
    )


def _write_scaling_shards(output_dir: OutputDir, throughput_rows: list[dict]) -> None:
    by_workers: dict[int, list[dict]] = {}
    for row in throughput_rows:
        n_workers = int(row["n_workers"])
        by_workers.setdefault(n_workers, []).append(row)
    for n_workers, rows in by_workers.items():
        _write_rows_atomic(
            _scaling_shard_path(output_dir.data, n_workers),
            _sort_throughput_rows(rows),
            _THROUGHPUT_FIELDNAMES,
        )


def rebuild_scaling_outputs(output_dir: OutputDir, cfg: dict, fallback_rows: list[dict] | None = None) -> list[dict]:
    """Rebuild aggregate scaling CSV/plots/metadata from per-worker shard CSVs."""
    aggregate_rows = _load_scaling_rows(output_dir)
    if not aggregate_rows and fallback_rows:
        aggregate_rows = list(fallback_rows)
    aggregate_rows = _sort_throughput_rows(aggregate_rows)
    if not aggregate_rows:
        return []

    throughput_path = output_dir.data / "throughput_summary.csv"
    _write_rows_atomic(throughput_path, aggregate_rows, _THROUGHPUT_FIELDNAMES)

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("scaling_curve") or plots_cfg.get("efficiency"):
        from async_abc.plotting.reporters import plot_scaling_summary

        plot_scaling_summary(aggregate_rows, output_dir)

    worker_counts = sorted({int(row["n_workers"]) for row in aggregate_rows})
    write_metadata(output_dir, cfg, extra={"worker_counts": worker_counts})
    return aggregate_rows


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

    done = _find_completed_scaling(output_dir, ["n_workers", "method", "replicate"]) if args.extend else set()

    # Clean up stale checkpoint dirs so we start fresh (skip when extending).
    if not args.extend and is_root_rank():
        for n_workers in worker_counts:
            tag = f"w{n_workers}"
            for replicate, seed in enumerate(seeds):
                ckpt_dir = output_dir.logs / f"propulate_rep{replicate}_seed{seed}__{tag}"
                if ckpt_dir.exists():
                    logger.info("[scaling] Removing stale checkpoint dir: %s", ckpt_dir)
                    shutil.rmtree(ckpt_dir)

    throughput_rows = []
    experiment_start = time.time()
    for n_workers in worker_counts:
        inference_cfg = {**cfg["inference"], "n_workers": n_workers, "_checkpoint_tag": f"w{n_workers}"}
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
                    n_sims = _simulation_count(records)
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
    write_timing_comparison_csv(Path(args.output_dir))

    if throughput_rows:
        _write_scaling_shards(output_dir, throughput_rows)

    rebuild_scaling_outputs(output_dir, cfg, fallback_rows=throughput_rows)


if __name__ == "__main__":
    main()
