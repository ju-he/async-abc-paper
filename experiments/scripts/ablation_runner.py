#!/usr/bin/env python3
"""Ablation study: run named configuration variants.

Each entry in ``ablation_variants`` overrides inference parameters and
produces its own CSV file, enabling component-by-component analysis.
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.io.config import is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord, RecordWriter
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.runner import compute_scaling_factor, find_completed_combinations, format_duration, make_arg_parser, run_method_distributed, write_timing_csv
from async_abc.utils.seeding import make_seeds

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Ablation study experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test)
    test_mode = is_test_mode(cfg)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    variants = cfg.get("ablation_variants", [])
    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    experiment_start = time.time()
    for variant in variants:
        name = variant.get("name", "unnamed")
        # Merge base inference config with variant overrides (exclude "name" key)
        overrides = {k: v for k, v in variant.items() if k != "name"}
        inference_cfg = {**cfg["inference"], **overrides}
        csv_name = f"ablation_{name}.csv"
        csv_path = output_dir.data / csv_name
        done = find_completed_combinations(csv_path, ["method", "replicate"]) if args.extend else set()
        writer = RecordWriter(csv_path)

        for method in cfg["methods"]:
            tagged_method = f"{method}__{name}"
            for replicate, seed in enumerate(seeds):
                if (tagged_method, str(replicate)) in done:
                    logger.info(
                        "[ablation] --extend: skipping %s replicate=%s",
                        tagged_method,
                        replicate,
                    )
                    continue
                records = run_method_distributed(
                    method, bm.simulate, bm.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
                for r in records:
                    r.method = tagged_method
                if is_root_rank():
                    writer.write(records)

    experiment_elapsed = time.time() - experiment_start
    exp_name = cfg["experiment_name"]
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", exp_name, format_duration(experiment_elapsed))
    if test_mode and is_root_rank():
        factor, extra, note = compute_scaling_factor(args.config)
        estimated = experiment_elapsed * factor + extra
        logger.info(
            "[%s] Estimated full run: ~%s  (%s)",
            exp_name,
            format_duration(estimated),
            note,
        )
    if not is_root_rank():
        return

    write_timing_csv(output_dir.data / "timing.csv", exp_name, experiment_elapsed, estimated, test_mode)

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("ablation_comparison"):
        from async_abc.plotting.reporters import plot_ablation_summary

        plot_ablation_summary(output_dir.data, variants, output_dir)

    write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
