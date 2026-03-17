#!/usr/bin/env python3
"""Ablation study: run named configuration variants.

Each entry in ``ablation_variants`` overrides inference parameters and
produces its own CSV file, enabling component-by-component analysis.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.inference.method_registry import run_method
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.io.records import ParticleRecord, RecordWriter
from async_abc.plotting.reporters import plot_ablation_summary
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import compute_scaling_factor, find_completed_combinations, format_duration, make_arg_parser, write_timing_csv
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
                    print(f"[ablation] --extend: skipping {tagged_method} replicate={replicate}", flush=True)
                    continue
                records = run_method(
                    method, bm.simulate, bm.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
                for r in records:
                    r.method = tagged_method
                writer.write(records)

    experiment_elapsed = time.time() - experiment_start
    exp_name = cfg["experiment_name"]
    estimated = None
    print(f"[{exp_name}] Done in {format_duration(experiment_elapsed)}", flush=True)
    if args.test:
        factor, extra, note = compute_scaling_factor(args.config)
        estimated = experiment_elapsed * factor + extra
        print(
            f"[{exp_name}] Estimated full run: ~{format_duration(estimated)}  ({note})",
            flush=True,
        )
    write_timing_csv(output_dir.data / "timing.csv", exp_name, experiment_elapsed, estimated, args.test)

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("ablation_comparison"):
        plot_ablation_summary(output_dir.data, variants, output_dir)

    write_metadata(output_dir, cfg)


if __name__ == "__main__":
    main()
