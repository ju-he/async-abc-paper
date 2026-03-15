#!/usr/bin/env python3
"""Scaling experiment: vary n_workers and record throughput.

In test mode only runs with n_workers=1.
On HPC, call this script via ``mpirun -n N`` for each N in worker_counts.
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks import make_benchmark
from async_abc.inference.method_registry import run_method
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import make_arg_parser
from async_abc.utils.seeding import make_seeds


def main() -> None:
    parser = make_arg_parser("Scaling experiment.")
    args = parser.parse_args()

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    scaling_cfg = cfg.get("scaling", {})
    worker_counts = scaling_cfg.get("worker_counts", [1])

    if args.test:
        worker_counts = [1]

    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    throughput_rows = []
    for n_workers in worker_counts:
        inference_cfg = {**cfg["inference"], "n_workers": n_workers}
        for method in cfg["methods"]:
            for replicate, seed in enumerate(seeds):
                t0 = time.time()
                records = run_method(
                    method, bm.simulate, bm.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
                elapsed = time.time() - t0
                n_sims = len(records)
                throughput = n_sims / elapsed if elapsed > 0 else float("inf")
                throughput_rows.append({
                    "n_workers": n_workers,
                    "method": method,
                    "replicate": replicate,
                    "seed": seed,
                    "n_simulations": n_sims,
                    "wall_time_s": elapsed,
                    "throughput_sims_per_s": throughput,
                })

    # Write throughput summary
    throughput_path = output_dir.data / "throughput_summary.csv"
    if throughput_rows:
        with open(throughput_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(throughput_rows[0].keys()))
            writer.writeheader()
            writer.writerows(throughput_rows)

    write_metadata(output_dir, cfg, extra={"worker_counts": worker_counts})


if __name__ == "__main__":
    main()
