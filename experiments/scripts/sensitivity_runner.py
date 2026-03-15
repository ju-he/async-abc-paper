#!/usr/bin/env python3
"""Sensitivity analysis: grid sweep over ABCPMC hyperparameters.

For each combination in ``sensitivity_grid``, runs a full experiment and
writes results to a separate CSV named after the variant.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.inference.method_registry import run_method
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import RecordWriter
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import make_arg_parser
from async_abc.utils.seeding import make_seeds


def _grid_variants(sensitivity_grid: dict) -> list:
    """Expand a dict of {param: [values]} into a list of variant dicts."""
    keys = list(sensitivity_grid.keys())
    value_lists = [sensitivity_grid[k] for k in keys]
    variants = []
    for combo in itertools.product(*value_lists):
        variants.append(dict(zip(keys, combo)))
    return variants


def main() -> None:
    parser = make_arg_parser("Sensitivity analysis experiment.")
    args = parser.parse_args()

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    grid = cfg.get("sensitivity_grid", {})

    if args.test:
        # Shrink grid to first value of each parameter for speed
        grid = {k: v[:1] for k, v in grid.items()}

    variants = _grid_variants(grid)
    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    for variant in variants:
        # Build a unique name from the variant parameters
        variant_name = "_".join(f"{k}{v}" for k, v in sorted(variant.items()))
        inference_cfg = {**cfg["inference"], **variant}
        csv_name = f"sensitivity_{variant_name}.csv"
        writer = RecordWriter(output_dir.data / csv_name)

        for method in cfg["methods"]:
            for replicate, seed in enumerate(seeds):
                records = run_method(
                    method, bm.simulate, bm.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
                # Tag records with variant info
                for r in records:
                    r.method = f"{method}__{'__'.join(f'{k}_{v}' for k, v in sorted(variant.items()))}"
                writer.write(records)

    write_metadata(output_dir, cfg, extra={"sensitivity_grid": grid})


if __name__ == "__main__":
    main()
