#!/usr/bin/env python3
"""Sensitivity analysis: grid sweep over ABCPMC hyperparameters.

For each combination in ``sensitivity_grid``, runs a full experiment and
writes results to a separate CSV named after the variant.
"""
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.inference.method_registry import run_method
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import RecordWriter
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import compute_scaling_factor, find_completed_combinations, format_duration, make_arg_parser, write_timing_csv
from async_abc.utils.seeding import make_seeds


def _grid_variants(sensitivity_grid: dict) -> list:
    """Expand a dict of {param: [values]} into a list of variant dicts."""
    keys = list(sensitivity_grid.keys())
    value_lists = [sensitivity_grid[k] for k in keys]
    variants = []
    for combo in itertools.product(*value_lists):
        variants.append(dict(zip(keys, combo)))
    return variants


def _format_variant_value(value) -> str:
    return str(value)


def _variant_name(variant: dict) -> str:
    parts = [
        f"{key}={_format_variant_value(value)}"
        for key, value in sorted(variant.items())
    ]
    return "__".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = make_arg_parser("Sensitivity analysis experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    grid = cfg.get("sensitivity_grid", {})

    if args.test:
        # Shrink grid to first value of each parameter for speed
        grid = {k: v[:1] for k, v in grid.items()}
        print(f"[{cfg['experiment_name']}] test mode: grid shrunk to 1 variant", flush=True)

    variants = _grid_variants(grid)
    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    experiment_start = time.time()
    for variant in variants:
        # Build a unique name from the variant parameters
        variant_name = _variant_name(variant)
        inference_cfg = {**cfg["inference"], **variant}
        if "tol_init_multiplier" in variant:
            inference_cfg["tol_init"] = (
                float(cfg["inference"]["tol_init"]) * float(variant["tol_init_multiplier"])
            )
            del inference_cfg["tol_init_multiplier"]
        csv_name = f"sensitivity_{variant_name}.csv"
        csv_path = output_dir.data / csv_name
        done = find_completed_combinations(csv_path, ["method", "replicate"]) if args.extend else set()
        writer = RecordWriter(csv_path)

        for method in cfg["methods"]:
            tagged_method = f"{method}__{variant_name}"
            for replicate, seed in enumerate(seeds):
                if (tagged_method, str(replicate)) in done:
                    print(f"[sensitivity] --extend: skipping {tagged_method} replicate={replicate}", flush=True)
                    continue
                records = run_method(
                    method, bm.simulate, bm.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
                # Tag records with variant info
                for r in records:
                    r.method = tagged_method
                writer.write(records)

    experiment_elapsed = time.time() - experiment_start
    name = cfg["experiment_name"]
    estimated = None
    print(f"[{name}] Done in {format_duration(experiment_elapsed)}", flush=True)
    if args.test:
        factor, extra, note = compute_scaling_factor(args.config)
        estimated = experiment_elapsed * factor + extra
        print(
            f"[{name}] Estimated full run: ~{format_duration(estimated)}  ({note})",
            flush=True,
        )
    write_timing_csv(output_dir.data / "timing.csv", name, experiment_elapsed, estimated, args.test)

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("sensitivity_heatmap"):
        from async_abc.plotting.reporters import plot_sensitivity_summary

        plot_sensitivity_summary(output_dir.data, grid, output_dir)

    write_metadata(output_dir, cfg, extra={"sensitivity_grid": grid})


if __name__ == "__main__":
    main()
