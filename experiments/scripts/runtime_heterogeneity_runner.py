#!/usr/bin/env python3
"""Runtime heterogeneity experiment.

Wraps the benchmark simulator with a LogNormal sleep to mimic heterogeneous
HPC workloads.  Measures idle-worker fraction and throughput over time.

The ``heterogeneity`` config block accepts either a scalar ``sigma`` (single
variance level) or a list ``sigma_levels`` (sweep over multiple levels).
In test mode the sleep is skipped so the pipeline completes quickly.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.plotting.reporters import plot_benchmark_diagnostics, plot_worker_gantt
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import compute_scaling_factor, format_duration, make_arg_parser, run_experiment, write_timing_csv
from async_abc.benchmarks import make_benchmark


def _make_heterogeneous_simulate(simulate_fn, mu: float, sigma: float, seed: int,
                                  test_mode: bool = False):
    """Wrap simulate_fn with a LogNormal wall-clock sleep."""
    rng = np.random.default_rng(seed)

    def wrapped(params, seed):
        delay = float(rng.lognormal(mean=mu, sigma=sigma))
        result = simulate_fn(params, seed=seed)
        if not test_mode:
            time.sleep(delay)
        return result

    return wrapped


def main() -> None:
    parser = make_arg_parser("Runtime heterogeneity experiment.")
    args = parser.parse_args()

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    bm = make_benchmark(cfg["benchmark"])
    het = cfg.get("heterogeneity", {})
    mu = float(het.get("mu", 0.0))

    # Support both single sigma and a list of sigma levels for a sweep
    if "sigma_levels" in het:
        sigma_levels = list(het["sigma_levels"])
    else:
        sigma_levels = [float(het.get("sigma", 1.0))]

    original_simulate = bm.simulate
    all_records = []

    experiment_start = time.time()
    for sigma in sigma_levels:
        wrapped_simulate = _make_heterogeneous_simulate(
            original_simulate, mu, sigma, seed=42, test_mode=args.test
        )
        bm.simulate = wrapped_simulate
        records = run_experiment(cfg, output_dir, benchmark=bm)
        # Tag each record with the sigma level used
        for r in records:
            r.method = f"{r.method}__sigma{sigma}"
        all_records.extend(records)

    bm.simulate = original_simulate

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

    plot_benchmark_diagnostics(all_records, cfg, output_dir)
    if cfg.get("plots", {}).get("gantt"):
        plot_worker_gantt(all_records, output_dir)
    write_metadata(output_dir, cfg, extra={"heterogeneity": het, "sigma_levels": sigma_levels})


if __name__ == "__main__":
    main()
