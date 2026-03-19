#!/usr/bin/env python3
"""Runner for the g-and-k distribution benchmark experiment."""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import is_test_mode, load_config
from async_abc.io.paths import OutputDir
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
    write_shard_status,
)
from async_abc.utils.runner import (
    compute_corrected_estimate,
    format_duration,
    make_arg_parser,
    run_experiment,
    write_timing_csv,
)

logger = logging.getLogger(__name__)


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
    parser = make_arg_parser("G-and-k distribution benchmark experiment.")
    args = parser.parse_args(argv)
    validate_shard_args(args)

    cfg = load_config(args.config, test_mode=args.test)
    test_mode = is_test_mode(cfg)
    experiment_name = cfg["experiment_name"]

    if is_shard_mode(args):
        output_root = Path(args.output_dir)
        full_cfg = load_config(args.config, test_mode=False)
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
                extra={"started_at_s": time.time()},
            )

        output_dir = layout.shard_output_dir
        t0 = time.time()
        run_experiment(cfg, output_dir, extend=False, replicate_indices=unit_indices)
        elapsed = time.time() - t0

        estimated_unsharded = None
        estimated_sharded = None
        if test_mode and is_root_rank():
            estimated_unsharded = compute_corrected_estimate(
                elapsed, output_dir.data / "raw_results.csv", args.config
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
                elapsed,
                estimated_unsharded,
                test_mode,
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
                extra={"started_at_s": t0, "finished_at_s": time.time()},
            )
            _finalize_sharded(cfg, layout, actual_num_shards)
        return

    output_dir = OutputDir(args.output_dir, experiment_name).ensure()

    t0 = time.time()
    records = run_experiment(cfg, output_dir, extend=args.extend)
    elapsed = time.time() - t0

    name = experiment_name
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(elapsed))
    if test_mode and is_root_rank():
        estimated = compute_corrected_estimate(
            elapsed, output_dir.data / "raw_results.csv", args.config
        )
        logger.info("[%s] Estimated full run: ~%s", name, format_duration(estimated))
    if is_root_rank():
        write_timing_csv(output_dir.data / "timing.csv", name, elapsed, estimated, test_mode)

    if is_root_rank() and any(cfg.get("plots", {}).values()):
        from async_abc.plotting.reporters import plot_benchmark_diagnostics

        plot_benchmark_diagnostics(records, cfg, output_dir)
    if is_root_rank():
        write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
