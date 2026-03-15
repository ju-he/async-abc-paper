#!/usr/bin/env python3
"""Ablation study: run named configuration variants.

Each entry in ``ablation_variants`` overrides inference parameters and
produces its own CSV file, enabling component-by-component analysis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.inference.method_registry import run_method
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord, RecordWriter
from async_abc.plotting.reporters import plot_ablation_summary
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import make_arg_parser
from async_abc.utils.seeding import make_seeds


def main() -> None:
    parser = make_arg_parser("Ablation study experiment.")
    args = parser.parse_args()

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    variants = cfg.get("ablation_variants", [])
    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    for variant in variants:
        name = variant.get("name", "unnamed")
        # Merge base inference config with variant overrides (exclude "name" key)
        overrides = {k: v for k, v in variant.items() if k != "name"}
        inference_cfg = {**cfg["inference"], **overrides}
        csv_name = f"ablation_{name}.csv"
        writer = RecordWriter(output_dir.data / csv_name)

        for method in cfg["methods"]:
            for replicate, seed in enumerate(seeds):
                records = run_method(
                    method, bm.simulate, bm.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
                for r in records:
                    r.method = f"{method}__{name}"
                writer.write(records)

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("ablation_comparison"):
        plot_ablation_summary(output_dir.data, variants, output_dir)

    write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
