#!/usr/bin/env python3
"""Ablation study: run named configuration variants.

Each entry in ``ablation_variants`` overrides inference parameters and
produces its own CSV file, enabling component-by-component analysis.
"""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord, RecordWriter
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
    parser = make_arg_parser("Ablation study experiment.")
    args = parser.parse_args(argv)
    validate_shard_args(args)

    cfg = load_config(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    experiment_name = cfg["experiment_name"]

    bm = make_benchmark(cfg["benchmark"])
    variants = cfg.get("ablation_variants", [])
    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    selected_replicates = list(range(n_replicates))

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

    seed_count = n_replicates
    if selected_replicates:
        seed_count = max(seed_count, max(selected_replicates) + 1)
    seeds = make_seeds(seed_count, base_seed)

    experiment_start = time.time()
    try:
        for variant in variants:
            name = variant.get("name", "unnamed")
            # Merge base inference config with variant overrides (exclude "name" key)
            overrides = {k: v for k, v in variant.items() if k != "name"}
            inference_cfg = {**cfg["inference"], **overrides}
            csv_name = f"ablation_{name}.csv"
            csv_path = output_dir.data / csv_name
            use_extend = args.extend and not is_shard_mode(args)
            done = find_completed_combinations(csv_path, ["method", "replicate"]) if use_extend else set()
            writer = RecordWriter(csv_path)

            for method in cfg["methods"]:
                tagged_method = f"{method}__{name}"
                for replicate in selected_replicates:
                    seed = seeds[replicate]
                    if (tagged_method, str(replicate)) in done:
                        logger.info(
                            "[ablation] --extend: skipping %s replicate=%s",
                            tagged_method,
                            replicate,
                        )
                        continue
                    records = run_method_distributed(
                        method, bm.simulate, bm.limits,
                        {**inference_cfg, "_checkpoint_tag": name}, output_dir, replicate, seed,
                    )
                    for r in records:
                        r.method = tagged_method
                    if is_root_rank():
                        writer.write(records)

        experiment_elapsed = time.time() - experiment_start
    except Exception as exc:
        if is_shard_mode(args) and is_root_rank():
            write_shard_failure_status(
                layout,
                unit_indices=selected_replicates,
                started_at_s=experiment_start,
                exc=exc,
            )
        raise
    exp_name = experiment_name
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", exp_name, format_duration(experiment_elapsed))
    if estimate_mode and is_root_rank():
        factor, extra, note = compute_scaling_factor(
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        estimated = experiment_elapsed * factor + extra
        logger.info(
            "[%s] Estimated full run: ~%s  (%s)",
            exp_name,
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
            exp_name,
            experiment_elapsed,
            estimated,
            test_mode,
            run_mode,
            estimated_full_unsharded_s=estimated,
            estimated_full_sharded_wall_s=estimated_sharded,
            aggregate_compute_s=experiment_elapsed,
        )
        write_shard_status(
            layout,
            state="completed",
            unit_indices=selected_replicates,
            elapsed_s=experiment_elapsed,
            estimated_full_s=estimated,
            estimated_full_unsharded_s=estimated,
            estimated_full_sharded_wall_s=estimated_sharded,
            aggregate_compute_s=experiment_elapsed,
            extra={"started_at_s": experiment_start, "finished_at_s": time.time(), "run_mode": run_mode},
        )
        _finalize_sharded(cfg, layout, actual_num_shards)
        return

    write_timing_csv(output_dir.data / "timing.csv", exp_name, experiment_elapsed, estimated, test_mode, run_mode)
    write_timing_comparison_csv(Path(args.output_dir))

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("ablation_comparison"):
        from async_abc.plotting.reporters import plot_ablation_summary

        plot_ablation_summary(output_dir.data, variants, output_dir, benchmark_cfg=cfg.get("benchmark", {}))

    write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
