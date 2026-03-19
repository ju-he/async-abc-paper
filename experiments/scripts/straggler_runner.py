#!/usr/bin/env python3
"""Persistent straggler experiment."""
import csv
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import RecordWriter
from async_abc.plotting.export import save_figure
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.shard_finalizers import finalize_experiment_by_name
from async_abc.utils.sharding import (
    ShardLayout,
    build_plan_payload,
    ensure_plan,
    estimate_sharded_wall_time,
    is_shard_mode,
    maybe_finalize_sharded_run,
    prepare_shard_workspace_distributed,
    read_json,
    split_indices,
    validate_shard_args,
    write_shard_failure_status,
    write_shard_status,
)
from async_abc.utils.runner import compute_scaling_factor, find_completed_combinations, format_duration, make_arg_parser, run_method_distributed, write_timing_comparison_csv, write_timing_csv
from async_abc.utils.seeding import make_seeds

logger = logging.getLogger(__name__)


def _current_worker_rank() -> int:
    """Best-effort worker rank for MPI or multiprocessing workers."""
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except Exception:
        pass

    proc = multiprocessing.current_process()
    if proc._identity:
        return int(proc._identity[0] - 1)
    return 0


def _make_straggler_simulate(
    simulate_fn,
    straggler_rank: int,
    slowdown_factor: float,
    base_sleep_s: float,
    test_mode: bool = False,
):
    """Wrap simulate_fn with a persistent slowdown on one worker."""

    def wrapped(params, seed):
        result = simulate_fn(params, seed=seed)
        if not test_mode and _current_worker_rank() == straggler_rank:
            time.sleep(float(slowdown_factor) * float(base_sleep_s))
        return result

    return wrapped


def _plot_throughput_vs_slowdown(throughput_rows, output_dir: OutputDir) -> None:
    if not throughput_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = sorted({row["base_method"] for row in throughput_rows})
    for method in methods:
        rows = [row for row in throughput_rows if row["base_method"] == method]
        rows = sorted(rows, key=lambda row: row["slowdown_factor"])
        ax.plot(
            [row["slowdown_factor"] for row in rows],
            [row["throughput_sims_per_s"] for row in rows],
            marker="o",
            label=method,
        )

    ax.set_xlabel("slowdown factor")
    ax.set_ylabel("throughput (sim/s)")
    ax.set_title("Throughput vs. straggler slowdown")
    ax.legend(frameon=False)
    fig.tight_layout()

    data = {
        "slowdown_factor": [row["slowdown_factor"] for row in throughput_rows],
        "base_method": [row["base_method"] for row in throughput_rows],
        "replicate": [row["replicate"] for row in throughput_rows],
        "throughput_sims_per_s": [row["throughput_sims_per_s"] for row in throughput_rows],
        "wall_time_s": [row["wall_time_s"] for row in throughput_rows],
    }
    save_figure(fig, output_dir.plots / "throughput_vs_slowdown", data=data)


def _finalize_sharded(cfg: dict, layout: ShardLayout, actual_num_shards: int) -> None:
    owner_id = f"{os.getenv('SLURM_JOB_ID', 'manual')}:{layout.shard_index}"
    maybe_finalize_sharded_run(
        layout=layout,
        actual_num_shards=actual_num_shards,
        owner_id=owner_id,
        finalize_fn=lambda shard_dirs, statuses: finalize_experiment_by_name(cfg, layout, shard_dirs, statuses),
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Persistent straggler experiment.")
    args = parser.parse_args(argv)
    validate_shard_args(args)

    cfg = load_config(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    experiment_name = cfg["experiment_name"]

    bm = make_benchmark(cfg["benchmark"])
    straggler_cfg = cfg["straggler"]
    slowdown_factors = [float(x) for x in straggler_cfg.get("slowdown_factor", [1.0])]
    straggler_rank = int(straggler_cfg.get("straggler_rank", 0))
    base_sleep_s = float(straggler_cfg.get("base_sleep_s", 0.1))

    selected_replicates = list(range(cfg["execution"]["n_replicates"]))

    if is_shard_mode(args):
        output_root = Path(args.output_dir)
        full_cfg = load_config(args.config, test_mode=False, small_mode=False)
        shard_index = args.shard_index if args.shard_index is not None else 0
        layout = ShardLayout(output_root, experiment_name, args.shard_run_id, shard_index)
        plan = read_json(layout.plan_path)
        if not plan:
            actual_num_shards = args.num_shards or 1
            plan = ensure_plan(
                layout,
                build_plan_payload(
                    experiment_name=experiment_name,
                    config_path=str(Path(args.config).resolve()),
                    unit_kind="replicate",
                    full_total_units=full_cfg["execution"]["n_replicates"],
                    actual_total_units=cfg["execution"]["n_replicates"],
                    target_total_units=cfg["execution"]["n_replicates"],
                    requested_num_shards=actual_num_shards,
                    actual_num_shards=actual_num_shards,
                    test_mode=test_mode,
                    small_mode=small_mode,
                    run_mode=run_mode,
                    extend=args.extend,
                    run_id=args.shard_run_id,
                    completed_unit_indices=[],
                    pending_unit_indices=list(range(cfg["execution"]["n_replicates"])),
                    shard_assignments=split_indices(cfg["execution"]["n_replicates"], actual_num_shards),
                    runner_script=str(Path(__file__).resolve()),
                ),
            )
        actual_num_shards = int(plan["actual_num_shards"])
        if args.finalize_only:
            if is_root_rank():
                _finalize_sharded(cfg, layout, actual_num_shards)
            return
        selected_replicates = [int(idx) for idx in plan["shard_assignments"][shard_index]]
        mode = prepare_shard_workspace_distributed(layout)
        if mode == "skip":
            if is_root_rank():
                _finalize_sharded(cfg, layout, actual_num_shards)
            return
        if is_root_rank():
            write_shard_status(
                layout,
                state="running",
                unit_indices=selected_replicates,
                extra={"started_at_s": time.time(), "run_mode": run_mode},
            )
        output_dir = layout.shard_output_dir
    else:
        output_dir = OutputDir(args.output_dir, experiment_name).ensure()

    seed_count = cfg["execution"]["n_replicates"]
    if selected_replicates:
        seed_count = max(seed_count, max(selected_replicates) + 1)
    seeds = make_seeds(seed_count, cfg["execution"]["base_seed"])

    csv_path = output_dir.data / "raw_results.csv"
    writer = RecordWriter(csv_path)

    all_records = []
    throughput_rows = []
    worst_records = []
    original_simulate = bm.simulate

    experiment_start = time.time()
    try:
        for slowdown_factor in slowdown_factors:
            bm.simulate = _make_straggler_simulate(
                original_simulate,
                straggler_rank=straggler_rank,
                slowdown_factor=slowdown_factor,
                base_sleep_s=base_sleep_s,
                test_mode=test_mode,
            )
            factor_records = []
            use_extend = args.extend and not is_shard_mode(args)
            done = find_completed_combinations(csv_path, ["method", "replicate"]) if use_extend else set()
            for method in cfg["methods"]:
                tagged_method = f"{method}__straggler_slowdown{slowdown_factor:.4g}x"
                for replicate in selected_replicates:
                    seed = seeds[replicate]
                    if (tagged_method, str(replicate)) in done:
                        logger.info(
                            "[straggler] --extend: skipping %s replicate=%s",
                            tagged_method,
                            replicate,
                        )
                        continue
                    t0 = time.time()
                    records = run_method_distributed(
                        method,
                        bm.simulate,
                        bm.limits,
                        {**cfg["inference"], "_checkpoint_tag": f"slowdown{slowdown_factor:.4g}x"},
                        output_dir,
                        replicate,
                        seed,
                    )
                    elapsed = time.time() - t0
                    for record in records:
                        record.method = tagged_method
                    if is_root_rank():
                        writer.write(records)
                        all_records.extend(records)
                        factor_records.extend(records)
                        throughput_rows.append(
                            {
                                "slowdown_factor": slowdown_factor,
                                "base_method": method,
                                "method": tagged_method,
                                "replicate": replicate,
                                "seed": seed,
                                "n_simulations": len(records),
                                "wall_time_s": elapsed,
                                "throughput_sims_per_s": len(records) / elapsed if elapsed > 0 else float("nan"),
                                "test_mode": test_mode,
                            }
                        )

            if slowdown_factor == max(slowdown_factors):
                worst_records = factor_records

        throughput_path = output_dir.data / "throughput_vs_slowdown_summary.csv"
        if throughput_rows:
            with open(throughput_path, "w", newline="") as f:
                writer_csv = csv.DictWriter(f, fieldnames=list(throughput_rows[0].keys()))
                writer_csv.writeheader()
                writer_csv.writerows(throughput_rows)

        elapsed = time.time() - experiment_start
    except Exception as exc:
        bm.simulate = original_simulate
        if is_shard_mode(args) and is_root_rank():
            write_shard_failure_status(
                layout,
                unit_indices=selected_replicates,
                started_at_s=experiment_start,
                exc=exc,
            )
        raise

    bm.simulate = original_simulate
    name = experiment_name
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(elapsed))
    if estimate_mode and is_root_rank():
        factor, extra, note = compute_scaling_factor(
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        estimated = elapsed * factor + extra
        logger.info(
            "[%s] Estimated full run: ~%s  (%s)",
            name,
            format_duration(estimated),
            note,
        )
    if not is_root_rank():
        return

    if is_shard_mode(args):
        estimated_sharded = None
        if estimated is not None:
            estimated_sharded = estimate_sharded_wall_time(
                estimated,
                int(plan.get("full_total_units", full_cfg["execution"]["n_replicates"])),
                int(plan.get("requested_num_shards", actual_num_shards)),
            )
        write_timing_csv(
            output_dir.data / "timing.csv",
            name,
            elapsed,
            estimated,
            test_mode,
            run_mode,
            estimated_full_unsharded_s=estimated,
            estimated_full_sharded_wall_s=estimated_sharded,
            aggregate_compute_s=elapsed,
        )
        write_shard_status(
            layout,
            state="completed",
            unit_indices=selected_replicates,
            elapsed_s=elapsed,
            estimated_full_s=estimated,
            estimated_full_unsharded_s=estimated,
            estimated_full_sharded_wall_s=estimated_sharded,
            aggregate_compute_s=elapsed,
            extra={"started_at_s": experiment_start, "finished_at_s": time.time(), "run_mode": run_mode},
        )
        _finalize_sharded(cfg, layout, actual_num_shards)
        return

    write_timing_csv(output_dir.data / "timing.csv", name, elapsed, estimated, test_mode, run_mode)
    write_timing_comparison_csv(Path(args.output_dir))

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("throughput_vs_slowdown"):
        _plot_throughput_vs_slowdown(throughput_rows, output_dir)
    if plots_cfg.get("gantt") and worst_records:
        from async_abc.plotting.reporters import plot_worker_gantt

        plot_worker_gantt(worst_records, output_dir)

    write_metadata(
        output_dir,
        cfg,
        extra={
            "slowdown_factors": slowdown_factors,
            "straggler_rank": straggler_rank,
            "base_sleep_s": base_sleep_s,
        },
    )


if __name__ == "__main__":
    main()
