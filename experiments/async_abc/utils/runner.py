"""Shared experiment execution logic used by all runner scripts."""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmarks import make_benchmark
from ..inference.method_registry import run_method
from ..io.config import load_config
from ..io.paths import OutputDir
from ..io.records import ParticleRecord, RecordWriter
from ..utils.seeding import make_seeds


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
