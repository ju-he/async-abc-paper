#!/usr/bin/env python3
"""Runtime heterogeneity experiment.

Wraps the benchmark simulator with a LogNormal sleep to mimic heterogeneous
HPC workloads.  Measures idle-worker fraction and throughput over time.

The ``heterogeneity`` config block accepts either a scalar ``sigma`` (single
variance level) or a list ``sigma_levels`` (sweep over multiple levels).
In test mode the sleep is skipped so the pipeline completes quickly.
"""
import logging
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord
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
from async_abc.utils.runner import compute_corrected_estimate, format_duration, make_arg_parser, run_experiment, write_timing_comparison_csv, write_timing_csv
from async_abc.benchmarks import make_benchmark

logger = logging.getLogger(__name__)


def _make_heterogeneous_simulate(simulate_fn, mu: float, sigma: float, seed: int,
                                  test_mode: bool = False):
    """Wrap simulate_fn with a LogNormal wall-clock sleep."""
    worker_attempts = defaultdict(int)

    def _worker_id() -> str:
        try:
            from mpi4py import MPI

            return str(MPI.COMM_WORLD.Get_rank())
        except Exception:
            pass
        proc = multiprocessing.current_process()
        if getattr(proc, "_identity", None):
            return str(int(proc._identity[0] - 1))
        return "0"

    def _stable_delay_seed(eval_seed: int, worker_id: str, worker_attempt: int) -> int:
        payload = f"{seed}|{sigma}|{eval_seed}|{worker_id}|{worker_attempt}".encode("ascii")
        return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big") % (2**31)

    def wrapped(params, seed):
        worker_id = _worker_id()
        worker_attempt = worker_attempts[worker_id]
        worker_attempts[worker_id] += 1
        delay_rng = np.random.default_rng(_stable_delay_seed(int(seed), worker_id, worker_attempt))
        delay = float(delay_rng.lognormal(mean=mu, sigma=sigma))
        result = simulate_fn(params, seed=seed)
        if not test_mode:
            time.sleep(delay)
        return result

    return wrapped


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
    parser = make_arg_parser("Runtime heterogeneity experiment.")
    args = parser.parse_args(argv)
    validate_shard_args(args)

    cfg = load_config(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    experiment_name = cfg["experiment_name"]

    bm = make_benchmark(cfg["benchmark"])
    het = cfg.get("heterogeneity", {})
    mu = float(het.get("mu", 0.0))

    # Support both single sigma and a list of sigma levels for a sweep
    if "sigma_levels" in het:
        sigma_levels = list(het["sigma_levels"])
    else:
        sigma_levels = [float(het.get("sigma", 1.0))]

    original_simulate = bm.simulate
    all_records = []

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

        unit_indices = [int(idx) for idx in plan["shard_assignments"][shard_index]]
        mode = prepare_shard_workspace_distributed(layout)
        if mode == "skip":
            if is_root_rank():
                _finalize_sharded(cfg, layout, actual_num_shards)
            return
        if is_root_rank():
            write_shard_status(
                layout,
                state="running",
                unit_indices=unit_indices,
                extra={"started_at_s": time.time(), "run_mode": run_mode},
            )

        output_dir = layout.shard_output_dir
        experiment_start = time.time()
        try:
            for sigma in sigma_levels:
                wrapped_simulate = _make_heterogeneous_simulate(
                    original_simulate, mu, sigma, seed=42, test_mode=test_mode
                )
                bm.simulate = wrapped_simulate
                sigma_cfg = {**cfg, "inference": {**cfg["inference"], "_checkpoint_tag": f"sigma{sigma}"}}
                all_records.extend(
                    run_experiment(
                        sigma_cfg,
                        output_dir,
                        benchmark=bm,
                        extend=False,
                        replicate_indices=unit_indices,
                        record_transform=lambda record, sigma=sigma: ParticleRecord(
                            method=f"{record.method}__sigma{sigma}",
                            replicate=record.replicate,
                            seed=record.seed,
                            step=record.step,
                            params=record.params,
                            loss=record.loss,
                            weight=record.weight,
                            tolerance=record.tolerance,
                            wall_time=record.wall_time,
                            worker_id=record.worker_id,
                            sim_start_time=record.sim_start_time,
                            sim_end_time=record.sim_end_time,
                            generation=record.generation,
                            record_kind=record.record_kind,
                            time_semantics=record.time_semantics,
                            attempt_count=record.attempt_count,
                        ),
                    )
                )
            bm.simulate = original_simulate

            experiment_elapsed = time.time() - experiment_start
            estimated_unsharded = None
            estimated_sharded = None
            if estimate_mode and is_root_rank():
                estimated_unsharded = compute_corrected_estimate(
                    experiment_elapsed,
                    output_dir.data / "raw_results.csv",
                    args.config,
                    small_mode=small_mode,
                    test_mode=test_mode,
                )
                estimated_sharded = estimate_sharded_wall_time(
                    estimated_unsharded,
                    int(plan.get("full_total_units", full_cfg["execution"]["n_replicates"])),
                    int(plan.get("requested_num_shards", actual_num_shards)),
                )
            if is_root_rank():
                write_timing_csv(
                    output_dir.data / "timing.csv",
                    experiment_name,
                    experiment_elapsed,
                    estimated_unsharded,
                    test_mode,
                    run_mode,
                    estimated_full_unsharded_s=estimated_unsharded,
                    estimated_full_sharded_wall_s=estimated_sharded,
                    aggregate_compute_s=experiment_elapsed,
                )
                write_shard_status(
                    layout,
                    state="completed",
                    unit_indices=unit_indices,
                    elapsed_s=experiment_elapsed,
                    estimated_full_s=estimated_unsharded,
                    estimated_full_unsharded_s=estimated_unsharded,
                    estimated_full_sharded_wall_s=estimated_sharded,
                    aggregate_compute_s=experiment_elapsed,
                    extra={"started_at_s": experiment_start, "finished_at_s": time.time(), "run_mode": run_mode},
                )
                _finalize_sharded(cfg, layout, actual_num_shards)
        except Exception as exc:
            bm.simulate = original_simulate
            if is_root_rank():
                write_shard_failure_status(
                    layout,
                    unit_indices=unit_indices,
                    started_at_s=experiment_start,
                    exc=exc,
                )
            raise
        return

    output_dir = OutputDir(args.output_dir, experiment_name).ensure()
    experiment_start = time.time()
    for sigma in sigma_levels:
        wrapped_simulate = _make_heterogeneous_simulate(
            original_simulate, mu, sigma, seed=42, test_mode=test_mode
        )
        bm.simulate = wrapped_simulate
        sigma_cfg = {**cfg, "inference": {**cfg["inference"], "_checkpoint_tag": f"sigma{sigma}"}}
        records = run_experiment(
            sigma_cfg,
            output_dir,
            benchmark=bm,
            extend=args.extend,
            record_transform=lambda record, sigma=sigma: ParticleRecord(
                method=f"{record.method}__sigma{sigma}",
                replicate=record.replicate,
                seed=record.seed,
                step=record.step,
                params=record.params,
                loss=record.loss,
                weight=record.weight,
                tolerance=record.tolerance,
                wall_time=record.wall_time,
            worker_id=record.worker_id,
            sim_start_time=record.sim_start_time,
            sim_end_time=record.sim_end_time,
            generation=record.generation,
            record_kind=record.record_kind,
            time_semantics=record.time_semantics,
            attempt_count=record.attempt_count,
        ),
    )
        all_records.extend(records)

    bm.simulate = original_simulate

    experiment_elapsed = time.time() - experiment_start
    name = experiment_name
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(experiment_elapsed))
    if estimate_mode and is_root_rank():
        estimated = compute_corrected_estimate(
            experiment_elapsed,
            output_dir.data / "raw_results.csv",
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        logger.info("[%s] Estimated full run: ~%s", name, format_duration(estimated))
    if is_root_rank():
        write_timing_csv(output_dir.data / "timing.csv", name, experiment_elapsed, estimated, test_mode, run_mode)
        write_timing_comparison_csv(Path(args.output_dir))

    if is_root_rank() and any(cfg.get("plots", {}).values()):
        from async_abc.plotting.reporters import plot_benchmark_diagnostics

        plot_benchmark_diagnostics(all_records, cfg, output_dir)
    if is_root_rank() and cfg.get("plots", {}).get("gantt"):
        from async_abc.plotting.reporters import plot_worker_gantt

        plot_worker_gantt(all_records, output_dir)
    plots_cfg = cfg.get("plots", {})
    if is_root_rank() and plots_cfg.get("idle_fraction"):
        from async_abc.plotting.reporters import plot_idle_fraction

        plot_idle_fraction(all_records, output_dir)
    if is_root_rank() and plots_cfg.get("throughput_over_time"):
        from async_abc.plotting.reporters import plot_throughput_over_time

        plot_throughput_over_time(all_records, output_dir)
    if is_root_rank() and plots_cfg.get("idle_fraction_comparison"):
        from async_abc.plotting.reporters import plot_idle_fraction_comparison

        plot_idle_fraction_comparison(all_records, output_dir)
    if is_root_rank():
        from async_abc.plotting.reporters import write_runtime_debug_summary

        write_runtime_debug_summary(all_records, output_dir)
        write_metadata(output_dir, cfg, extra={"heterogeneity": het, "sigma_levels": sigma_levels})


if __name__ == "__main__":
    main()
