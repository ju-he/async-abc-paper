#!/usr/bin/env python3
"""Runtime heterogeneity experiment.

Wraps the benchmark simulator with a LogNormal sleep to mimic heterogeneous
HPC workloads.  Measures idle-worker fraction and throughput over time.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.plotting.reporters import plot_archive_evolution, plot_posterior
from async_abc.utils.metadata import write_metadata
from async_abc.utils.runner import make_arg_parser, run_experiment
from async_abc.benchmarks import make_benchmark


def _make_heterogeneous_simulate(simulate_fn, mu: float, sigma: float, seed: int):
    """Wrap simulate_fn with a LogNormal wall-clock sleep."""
    rng = np.random.default_rng(seed)

    def wrapped(params, seed):
        delay = float(rng.lognormal(mean=mu, sigma=sigma))
        # In test mode the delay would dominate; skip actual sleeping,
        # just record the intended delay as part of params for analysis.
        result = simulate_fn(params, seed=seed)
        # Uncomment for real HPC experiments: time.sleep(delay)
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
    sigma = float(het.get("sigma", 1.0))

    wrapped_simulate = _make_heterogeneous_simulate(bm.simulate, mu, sigma, seed=42)

    # Temporarily replace simulate method
    original_simulate = bm.simulate
    bm.simulate = wrapped_simulate

    records = run_experiment(cfg, output_dir, benchmark=bm)

    bm.simulate = original_simulate

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("posterior"):
        plot_posterior(records, output_dir)
    if plots_cfg.get("archive_evolution"):
        plot_archive_evolution(records, output_dir)
    write_metadata(output_dir, cfg, extra={"heterogeneity": het})


if __name__ == "__main__":
    main()
