"""Shared experiment execution logic used by all runner scripts."""
import argparse
import csv
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..benchmarks import make_benchmark
from ..inference.method_registry import run_method
from ..io.config import load_config
from ..io.paths import OutputDir
from ..io.records import ParticleRecord, RecordWriter
from ..utils.seeding import make_seeds


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def compute_scaling_factor(
    config_path: Union[str, Path],
) -> Tuple[float, float, str]:
    """Estimate how much longer the full run takes compared to a test run.

    Returns
    -------
    factor : float
        Linear multiplier: ``test_elapsed * factor`` estimates the compute portion.
    extra_seconds : float
        Additional fixed time beyond the linear scaling (expected sleep for
        runtime_heterogeneity experiments; 0 for all others).
    note : str
        Parenthetical describing what the full run entails.
    """
    cfg = load_config(config_path, test_mode=False)
    test_cfg = load_config(config_path, test_mode=True)

    full_sims = cfg["inference"]["max_simulations"]
    full_workers = cfg["inference"]["n_workers"]
    full_reps = cfg["execution"]["n_replicates"]
    test_sims = test_cfg["inference"]["max_simulations"]
    test_workers = test_cfg["inference"]["n_workers"]
    test_reps = test_cfg["execution"]["n_replicates"]

    factor = (full_sims * full_reps / full_workers) / (
        test_sims * test_reps / test_workers
    )
    extra_seconds = 0.0

    if "sensitivity_grid" in cfg:
        grid = cfg["sensitivity_grid"]
        full_grid_size = 1
        for v in grid.values():
            full_grid_size *= len(v)
        test_grid = {k: v[:1] for k, v in grid.items()}
        test_grid_size = 1
        for v in test_grid.values():
            test_grid_size *= len(v)
        factor *= full_grid_size / test_grid_size
        note = f"{full_sims} sims × {full_reps} reps, {full_grid_size} grid variants"

    elif "scaling" in cfg:
        full_worker_counts = cfg["scaling"].get("worker_counts", [1])
        test_worker_counts = [1]
        factor *= len(full_worker_counts) / len(test_worker_counts)
        note = f"{full_sims} sims × {full_reps} reps, {len(full_worker_counts)} worker configs"

    elif "heterogeneity" in cfg:
        het = cfg["heterogeneity"]
        mu = float(het.get("mu", 0.0))
        sigmas = het.get("sigma_levels", [het.get("sigma", 1.0)])
        # Expected sleep per sim: exp(mu + sigma^2/2); multiply by sequential sims per worker
        sims_per_sigma = full_sims * full_reps / full_workers
        extra_seconds = sum(
            math.exp(mu + s ** 2 / 2) * sims_per_sigma for s in sigmas
        )
        note = (
            f"{full_sims} sims × {full_reps} reps, "
            f"+{format_duration(extra_seconds)} sleep"
        )

    elif "sbc" in cfg:
        full_trials = int(cfg["sbc"].get("n_trials", 1))
        test_trials = int(test_cfg.get("sbc", {}).get("n_trials", 1))
        factor *= full_trials / test_trials
        note = f"{full_sims} sims × {full_trials} SBC trials, {full_workers} workers"

    elif "straggler" in cfg:
        straggler = cfg["straggler"]
        slowdown_factors = straggler.get("slowdown_factor", [1])
        base_sleep_s = float(straggler.get("base_sleep_s", 0.0))
        sims_per_sweep = full_sims * full_reps / full_workers
        extra_seconds = sum(float(s) * base_sleep_s * sims_per_sweep for s in slowdown_factors)
        note = (
            f"{full_sims} sims × {full_reps} reps, {len(slowdown_factors)} slowdown levels, "
            f"+{format_duration(extra_seconds)} straggler sleep"
        )

    else:
        note = f"{full_sims} sims × {full_reps} reps, {full_workers} workers"

    return factor, extra_seconds, note


_TIMING_FIELDNAMES = [
    "experiment_name",
    "elapsed_s",
    "estimated_full_s",
    "test_mode",
    "timestamp",
]


def write_timing_csv(
    path: Union[str, Path],
    experiment_name: str,
    elapsed_s: float,
    estimated_full_s: Optional[float],
    test_mode: bool,
) -> None:
    """Append one timing row to a CSV file, writing the header if needed."""
    path = Path(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TIMING_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "experiment_name": experiment_name,
            "elapsed_s": round(elapsed_s, 3),
            "estimated_full_s": (
                round(estimated_full_s, 3) if estimated_full_s is not None else ""
            ),
            "test_mode": test_mode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })


def make_arg_parser(description: str = "") -> argparse.ArgumentParser:
    """Return a pre-configured ArgumentParser for experiment runners."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="Path to JSON experiment config.")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Root directory for results.")
    p.add_argument("--test", action="store_true",
                   help="Test mode: small budget, ≤8 workers.")
    return p


def run_experiment(
    cfg: Dict[str, Any],
    output_dir: OutputDir,
    benchmark=None,
    methods: Optional[List[str]] = None,
    csv_name: str = "raw_results.csv",
) -> List[ParticleRecord]:
    """Run all methods × replicates and write results to CSV.

    Parameters
    ----------
    cfg:
        Full validated config dict.
    output_dir:
        Already-ensured :class:`~async_abc.io.paths.OutputDir`.
    benchmark:
        Benchmark instance.  If ``None``, instantiated from ``cfg["benchmark"]``.
    methods:
        Method names to run.  Defaults to ``cfg["methods"]``.
    csv_name:
        Filename for the output CSV inside ``output_dir.data``.

    Returns
    -------
    List[ParticleRecord]
        All records produced (across all methods and replicates).
    """
    if benchmark is None:
        benchmark = make_benchmark(cfg["benchmark"])
    if methods is None:
        methods = cfg["methods"]

    inference_cfg = cfg["inference"]
    n_replicates = cfg["execution"]["n_replicates"]
    base_seed = cfg["execution"]["base_seed"]
    seeds = make_seeds(n_replicates, base_seed)

    writer = RecordWriter(output_dir.data / csv_name)
    all_records: List[ParticleRecord] = []

    for method in methods:
        for replicate, seed in enumerate(seeds):
            try:
                records = run_method(
                    method, benchmark.simulate, benchmark.limits,
                    inference_cfg, output_dir, replicate, seed,
                )
            except ImportError as exc:
                warnings.warn(
                    f"Skipping method '{method}' (missing dependency): {exc}",
                    stacklevel=2,
                )
                print(f"[runner] WARNING: skipping '{method}': {exc}", file=sys.stderr)
                break  # skip all replicates for this method
            writer.write(records)
            all_records.extend(records)

    return all_records
