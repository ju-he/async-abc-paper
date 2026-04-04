#!/usr/bin/env python3
"""Regenerate plots from existing experiment outputs without re-running inference.

Usage::

    python scripts/replot.py /path/to/run_root gaussian_mean gandk
    python scripts/replot.py /path/to/run_root all
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from async_abc.analysis import base_method_name
from async_abc.io.paths import OutputDir
from async_abc.io.records import load_records, ParticleRecord
from async_abc.plotting.reporters import (
    plot_ablation_summary,
    plot_benchmark_diagnostics,
    plot_idle_fraction,
    plot_idle_fraction_comparison,
    plot_scaling_grid,
    plot_sensitivity_summary,
    plot_throughput_over_time,
    plot_worker_gantt,
    write_runtime_debug_summary,
)
from async_abc.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)

# Experiments that use raw_results.csv as the standard record store.
_BENCHMARK_EXPERIMENTS = {
    "gaussian_mean", "gandk", "lotka_volterra", "cellular_potts",
}


def _load_metadata(output_dir: OutputDir) -> dict:
    meta_path = output_dir.data / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json at {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def _replot_benchmark(name: str, output_dir: OutputDir, cfg: dict) -> None:
    """Standard benchmark: load raw_results.csv → plot_benchmark_diagnostics + gantt."""
    csv_path = output_dir.data / "raw_results.csv"
    if not csv_path.exists():
        logger.warning("[%s] No raw_results.csv — skipping", name)
        return
    records = load_records(csv_path)
    logger.info("[%s] Loaded %d records from %s", name, len(records), csv_path)

    if any(cfg.get("plots", {}).values()):
        plot_benchmark_diagnostics(records, cfg, output_dir)
    if cfg.get("plots", {}).get("gantt"):
        plot_worker_gantt(records, output_dir)


def _replot_runtime_heterogeneity(name: str, output_dir: OutputDir, cfg: dict) -> None:
    """Runtime heterogeneity: benchmark diagnostics + gantt + idle/throughput plots."""
    csv_path = output_dir.data / "raw_results.csv"
    if not csv_path.exists():
        logger.warning("[%s] No raw_results.csv — skipping", name)
        return
    records = load_records(csv_path)
    logger.info("[%s] Loaded %d records from %s", name, len(records), csv_path)

    plots_cfg = cfg.get("plots", {})
    if any(plots_cfg.values()):
        plot_benchmark_diagnostics(records, cfg, output_dir)
    if plots_cfg.get("gantt"):
        plot_worker_gantt(records, output_dir)
    if plots_cfg.get("idle_fraction"):
        plot_idle_fraction(records, output_dir)
    if plots_cfg.get("throughput_over_time"):
        plot_throughput_over_time(records, output_dir)
    if plots_cfg.get("idle_fraction_comparison"):
        plot_idle_fraction_comparison(records, output_dir)
    write_runtime_debug_summary(records, output_dir)


def _replot_straggler(name: str, output_dir: OutputDir, cfg: dict) -> None:
    """Straggler: throughput vs slowdown plot + gantt for worst slowdown."""
    from async_abc.utils.shard_finalizers import _plot_throughput_vs_slowdown

    plots_cfg = cfg.get("plots", {})
    throughput_csv = output_dir.data / "throughput_vs_slowdown_summary.csv"
    if plots_cfg.get("throughput_vs_slowdown") and throughput_csv.exists():
        with open(throughput_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        logger.info("[%s] Loaded %d throughput rows", name, len(rows))
        _plot_throughput_vs_slowdown(rows, output_dir)

    raw_csv = output_dir.data / "raw_results.csv"
    if plots_cfg.get("gantt") and raw_csv.exists():
        records = load_records(raw_csv)
        write_runtime_debug_summary(records, output_dir)
        slowdown_pattern = re.compile(r"__straggler_slowdown([0-9.eE+-]+)x$")
        tagged = [(r, float(m.group(1))) for r in records if (m := slowdown_pattern.search(r.method))]
        if tagged:
            worst = max(f for _, f in tagged)
            worst_records = [r for r, f in tagged if f == worst]
            async_worst_records = [
                record for record in worst_records
                if base_method_name(record.method) == "async_propulate_abc"
                and record.record_kind == "simulation_attempt"
                and record.worker_id is not None
                and record.sim_start_time is not None
                and record.sim_end_time is not None
            ]
            if async_worst_records:
                plot_worker_gantt(async_worst_records, output_dir)


def _replot_sensitivity(name: str, output_dir: OutputDir, cfg: dict) -> None:
    """Sensitivity: heatmap from per-variant CSVs."""
    if cfg.get("plots", {}).get("sensitivity_heatmap"):
        plot_sensitivity_summary(output_dir.data, cfg.get("sensitivity_grid", {}), output_dir)


def _replot_ablation(name: str, output_dir: OutputDir, cfg: dict) -> None:
    """Ablation: bar chart from per-variant CSVs."""
    if cfg.get("plots", {}).get("ablation_comparison"):
        plot_ablation_summary(
            output_dir.data,
            cfg.get("ablation_variants", []),
            output_dir,
            benchmark_cfg=cfg.get("benchmark", {}),
        )


def _replot_scaling(name: str, output_dir: OutputDir, cfg: dict) -> None:
    """Scaling: regenerate all scaling plots from aggregate throughput/budget CSVs."""
    def _load(path: Path) -> list:
        if not path.exists():
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    throughput_rows = _load(output_dir.data / "throughput_summary.csv")
    budget_rows = _load(output_dir.data / "budget_summary.csv")
    if not throughput_rows:
        logger.warning("[%s] No throughput_summary.csv in %s — skipping", name, output_dir.data)
        return
    logger.info("[%s] Loaded %d throughput rows, %d budget rows", name, len(throughput_rows), len(budget_rows))
    plot_scaling_grid(throughput_rows=throughput_rows, budget_rows=budget_rows, output_dir=output_dir)


def _replot_sbc(name: str, output_dir: OutputDir, cfg: dict) -> None:
    """SBC: regenerate rank histogram and coverage table from saved CSVs."""
    from async_abc.utils.shard_finalizers import _plot_coverage_table, _plot_rank_histogram

    plots_cfg = cfg.get("plots", {})
    ranks_csv = output_dir.data / "sbc_ranks.csv"
    coverage_csv = output_dir.data / "coverage.csv"
    if plots_cfg.get("rank_histogram") and ranks_csv.exists():
        _plot_rank_histogram(pd.read_csv(ranks_csv), output_dir)
    if plots_cfg.get("coverage_table") and coverage_csv.exists():
        _plot_coverage_table(pd.read_csv(coverage_csv), output_dir)


_REPLOT_DISPATCH = {
    "gaussian_mean": _replot_benchmark,
    "gandk": _replot_benchmark,
    "lotka_volterra": _replot_benchmark,
    "cellular_potts": _replot_benchmark,
    "runtime_heterogeneity": _replot_runtime_heterogeneity,
    "straggler": _replot_straggler,
    "sensitivity": _replot_sensitivity,
    "sensitivity_gandk": _replot_sensitivity,
    "ablation": _replot_ablation,
    "sbc": _replot_sbc,
    "scaling": _replot_scaling,
}

ALL_EXPERIMENTS = list(_REPLOT_DISPATCH.keys())


def replot_experiment(run_root: Path, name: str) -> None:
    """Regenerate all configured plots for a single experiment."""
    output_dir = OutputDir(run_root, name)
    if not output_dir.data.exists():
        logger.warning("[%s] No data directory at %s — skipping", name, output_dir.data)
        return

    meta = _load_metadata(output_dir)
    cfg = meta.get("config", {})
    output_dir.plots.mkdir(parents=True, exist_ok=True)

    handler = _REPLOT_DISPATCH.get(name)
    if handler is None:
        logger.warning("[%s] No replot handler registered — skipping", name)
        return

    logger.info("[%s] Regenerating plots…", name)
    handler(name, output_dir, cfg)
    logger.info("[%s] Done.", name)


def main(argv: list[str] | None = None) -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Regenerate plots from existing experiment outputs.",
    )
    parser.add_argument("run_root", help="Root directory of the experiment run.")
    parser.add_argument(
        "experiments", nargs="+", metavar="NAME",
        help=f"Experiment names, or 'all'. Choices: {', '.join(ALL_EXPERIMENTS)}",
    )
    args = parser.parse_args(argv)

    experiments = ALL_EXPERIMENTS if "all" in args.experiments else args.experiments
    run_root = Path(args.run_root).resolve()

    for name in experiments:
        try:
            replot_experiment(run_root, name)
        except Exception:
            logger.exception("[%s] Failed to replot", name)


if __name__ == "__main__":
    main()
