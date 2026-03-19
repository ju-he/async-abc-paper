#!/usr/bin/env python3
"""Runner for the Cellular Potts model benchmark experiment.

Reference Data
--------------
The default CPM config already points at bundled reference assets in
``experiments/assets/cellular_potts``. Regenerate them only if you want to
replace the default reference data::

    python experiments/scripts/generate_cpm_reference.py \\
        --config-template experiments/assets/cellular_potts/sim_config.json \\
        --config-builder-params experiments/assets/cellular_potts/config_builder_params.json \\
        --parameter-space experiments/assets/cellular_potts/parameter_space_division_motility.json \\
        --true-params '{"division_rate": 0.03, "motility": 2000}' \\
        --seed 0

Then update ``reference_data_path`` in ``experiments/configs/cellular_potts.json``
if you want this runner to use the newly generated directory instead.
"""
import sys
import time
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
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
    write_shard_failure_status,
    write_shard_status,
)
from async_abc.utils.runner import (
    compute_corrected_estimate,
    format_duration,
    make_arg_parser,
    run_experiment,
    write_timing_comparison_csv,
    write_timing_csv,
)

logger = logging.getLogger(__name__)


def _prepare_runtime_cfg(cfg: dict, output_dir: OutputDir) -> dict:
    """Return cfg with CPM scratch output redirected into this experiment run."""
    cfg = dict(cfg)
    benchmark_cfg = dict(cfg["benchmark"])
    benchmark_cfg["output_dir"] = str(output_dir.root / "cpm_sims")
    cfg["benchmark"] = benchmark_cfg
    return cfg


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
    parser = make_arg_parser("Cellular Potts model benchmark experiment.")
    args = parser.parse_args(argv)
    validate_shard_args(args)

    cfg = load_config(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    experiment_name = cfg["experiment_name"]

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
        cfg = _prepare_runtime_cfg(cfg, output_dir)
        t0 = time.time()
        try:
            run_experiment(cfg, output_dir, replicate_indices=unit_indices)
            elapsed = time.time() - t0
            estimated_unsharded = None
            estimated_sharded = None
            if estimate_mode and is_root_rank():
                estimated_unsharded = compute_corrected_estimate(
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
            if is_root_rank():
                write_timing_csv(
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
                    extra={"started_at_s": t0, "finished_at_s": time.time(), "run_mode": run_mode},
                )
                _finalize_sharded(cfg, layout, actual_num_shards)
        except Exception as exc:
            if is_root_rank():
                write_shard_failure_status(
                    layout,
                    unit_indices=unit_indices,
                    started_at_s=t0,
                    exc=exc,
                )
            raise
        return

    output_dir = OutputDir(args.output_dir, experiment_name).ensure()
    cfg = _prepare_runtime_cfg(cfg, output_dir)

    t0 = time.time()
    records = run_experiment(cfg, output_dir)
    elapsed = time.time() - t0

    name = experiment_name
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(elapsed))
    if estimate_mode and is_root_rank():
        estimated = compute_corrected_estimate(
            elapsed,
            output_dir.data / "raw_results.csv",
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        logger.info("[%s] Estimated full run: ~%s", name, format_duration(estimated))
    if is_root_rank():
        write_timing_csv(output_dir.data / "timing.csv", name, elapsed, estimated, test_mode, run_mode)
        write_timing_comparison_csv(Path(args.output_dir))

    if is_root_rank() and any(cfg.get("plots", {}).values()):
        from async_abc.plotting.reporters import plot_benchmark_diagnostics

        plot_benchmark_diagnostics(records, cfg, output_dir)
    if is_root_rank():
        write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
