"""Shared runner entrypoint for plain benchmark experiments."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

from ..io.config import get_run_mode, is_small_mode, is_test_mode
from ..io.paths import OutputDir
from .mpi import is_root_rank as default_is_root_rank
from .runner import (
    format_duration,
    make_arg_parser,
)
from .sharding import (
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

logger = logging.getLogger(__name__)

RuntimeCfgHook = Callable[[dict, OutputDir], dict]


def _identity_runtime_cfg(cfg: dict, output_dir: OutputDir) -> dict:
    del output_dir
    return cfg


def _finalize_sharded(
    cfg: dict,
    layout: ShardLayout,
    actual_num_shards: int,
    *,
    finalize_experiment_by_name_fn: Callable,
) -> None:
    owner_id = f"{os.getenv('SLURM_JOB_ID', 'manual')}:{layout.shard_index}"
    maybe_finalize_sharded_run(
        layout=layout,
        actual_num_shards=actual_num_shards,
        owner_id=owner_id,
        finalize_fn=lambda shard_dirs, statuses: finalize_experiment_by_name_fn(
            cfg, layout, shard_dirs, statuses
        ),
    )


def run_benchmark_runner(
    argv: list[str] | None = None,
    *,
    description: str,
    runner_script_path: str,
    configure_logging_fn: Callable[[], None],
    load_config_fn: Callable,
    run_experiment_fn: Callable,
    compute_corrected_estimate_fn: Callable,
    write_timing_csv_fn: Callable,
    write_timing_comparison_csv_fn: Callable,
    write_metadata_fn: Callable,
    finalize_experiment_by_name_fn: Callable,
    is_root_rank_fn: Callable[[], bool] = default_is_root_rank,
    prepare_runtime_cfg: RuntimeCfgHook | None = None,
) -> None:
    """Run one plain benchmark experiment with optional sharding support."""
    configure_logging_fn()
    parser = make_arg_parser(description)
    args = parser.parse_args(argv)
    validate_shard_args(args)

    prepare_runtime_cfg = prepare_runtime_cfg or _identity_runtime_cfg

    cfg = load_config_fn(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    experiment_name = cfg["experiment_name"]

    if is_shard_mode(args):
        output_root = Path(args.output_dir)
        full_cfg = load_config_fn(args.config, test_mode=False, small_mode=False)
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
                    shard_assignments=split_indices(
                        cfg["execution"]["n_replicates"],
                        actual_num_shards,
                    ),
                    runner_script=runner_script_path,
                ),
            )
        actual_num_shards = int(plan["actual_num_shards"])

        if args.finalize_only:
            if is_root_rank_fn():
                _finalize_sharded(
                    cfg,
                    layout,
                    actual_num_shards,
                    finalize_experiment_by_name_fn=finalize_experiment_by_name_fn,
                )
            return

        unit_indices = [int(idx) for idx in plan["shard_assignments"][shard_index]]
        mode = prepare_shard_workspace_distributed(layout)
        if mode == "skip":
            if is_root_rank_fn():
                _finalize_sharded(
                    cfg,
                    layout,
                    actual_num_shards,
                    finalize_experiment_by_name_fn=finalize_experiment_by_name_fn,
                )
            return
        if is_root_rank_fn():
            write_shard_status(
                layout,
                state="running",
                unit_indices=unit_indices,
                extra={"started_at_s": time.time(), "run_mode": run_mode},
            )

        output_dir = layout.shard_output_dir
        runtime_cfg = prepare_runtime_cfg(cfg, output_dir)
        t0 = time.time()
        try:
            run_kwargs = {"replicate_indices": unit_indices}
            if args.extend:
                run_kwargs["extend"] = True
            run_experiment_fn(runtime_cfg, output_dir, **run_kwargs)
            elapsed = time.time() - t0

            estimated_unsharded = None
            estimated_sharded = None
            if estimate_mode and is_root_rank_fn():
                estimated_unsharded = compute_corrected_estimate_fn(
                    elapsed,
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
            if is_root_rank_fn():
                write_timing_csv_fn(
                    output_dir.data / "timing.csv",
                    experiment_name,
                    elapsed,
                    estimated_unsharded,
                    test_mode,
                    run_mode,
                    estimated_full_unsharded_s=estimated_unsharded,
                    estimated_full_sharded_wall_s=estimated_sharded,
                    aggregate_compute_s=elapsed,
                )
                write_shard_status(
                    layout,
                    state="completed",
                    unit_indices=unit_indices,
                    elapsed_s=elapsed,
                    estimated_full_s=estimated_unsharded,
                    estimated_full_unsharded_s=estimated_unsharded,
                    estimated_full_sharded_wall_s=estimated_sharded,
                    aggregate_compute_s=elapsed,
                    extra={
                        "started_at_s": t0,
                        "finished_at_s": time.time(),
                        "run_mode": run_mode,
                    },
                )
                _finalize_sharded(
                    runtime_cfg,
                    layout,
                    actual_num_shards,
                    finalize_experiment_by_name_fn=finalize_experiment_by_name_fn,
                )
        except Exception as exc:
            if is_root_rank_fn():
                write_shard_failure_status(
                    layout,
                    unit_indices=unit_indices,
                    started_at_s=t0,
                    exc=exc,
                )
            raise
        return

    output_dir = OutputDir(args.output_dir, experiment_name).ensure()
    runtime_cfg = prepare_runtime_cfg(cfg, output_dir)

    t0 = time.time()
    run_kwargs = {}
    if args.extend:
        run_kwargs["extend"] = True
    records = run_experiment_fn(runtime_cfg, output_dir, **run_kwargs)
    elapsed = time.time() - t0

    estimated = None
    if is_root_rank_fn():
        logger.info("[%s] Done in %s", experiment_name, format_duration(elapsed))
    if estimate_mode and is_root_rank_fn():
        estimated = compute_corrected_estimate_fn(
            elapsed,
            output_dir.data / "raw_results.csv",
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        logger.info(
            "[%s] Estimated full run: ~%s",
            experiment_name,
            format_duration(estimated),
        )
    if is_root_rank_fn():
        write_timing_csv_fn(
            output_dir.data / "timing.csv",
            experiment_name,
            elapsed,
            estimated,
            test_mode,
            run_mode,
        )
        write_timing_comparison_csv_fn(Path(args.output_dir))

    if is_root_rank_fn() and any(runtime_cfg.get("plots", {}).values()):
        from ..plotting.reporters import plot_benchmark_diagnostics

        plot_benchmark_diagnostics(records, runtime_cfg, output_dir)
    if is_root_rank_fn():
        write_metadata_fn(output_dir, runtime_cfg)
