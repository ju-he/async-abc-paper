#!/usr/bin/env python3
"""Simulation-based calibration runner."""
import copy
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.analysis.sbc import empirical_coverage, sbc_ranks
from async_abc.benchmarks import make_benchmark
from async_abc.io.config import load_config
from async_abc.io.paths import OutputDir
from async_abc.plotting.export import save_figure
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.utils.mpi import is_root_rank
from async_abc.utils.runner import (
    compute_scaling_factor,
    format_duration,
    make_arg_parser,
    run_method_distributed,
    write_timing_csv,
)
from async_abc.utils.seeding import make_seeds

logger = logging.getLogger(__name__)


def _write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _posterior_samples(records, param_name: str) -> np.ndarray:
    accepted = [r for r in records if r.tolerance is not None]
    final = accepted if not accepted else [
        r for r in accepted if r.tolerance == min(x.tolerance for x in accepted)
    ]
    return np.asarray([r.params[param_name] for r in final if param_name in r.params], dtype=float)


def _true_param_config(base_cfg: dict, true_params: dict[str, float], observed_seed: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["observed_data_seed"] = int(observed_seed)
    for param, value in true_params.items():
        cfg[f"true_{param}"] = float(value)
    return cfg


def _plot_rank_histogram(ranks_df: pd.DataFrame, output_dir: OutputDir) -> None:
    if ranks_df.empty:
        return

    methods = list(ranks_df["method"].dropna().unique())
    fig, axes = plt.subplots(len(methods), 1, figsize=(6, max(3.5, 2.8 * len(methods))), squeeze=False)
    for idx, method in enumerate(methods):
        ax = axes[idx, 0]
        group = ranks_df[ranks_df["method"] == method]
        bins = int(group["n_samples"].iloc[0]) + 1 if not group.empty else 10
        ax.hist(group["rank"], bins=min(max(bins, 5), 30), color="steelblue", alpha=0.8)
        ax.set_title(f"Rank histogram: {method}")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
    fig.tight_layout()
    save_figure(fig, output_dir.plots / "rank_histogram", data={col: ranks_df[col].tolist() for col in ranks_df.columns})


def _plot_coverage_table(coverage_df: pd.DataFrame, output_dir: OutputDir) -> None:
    if coverage_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in coverage_df.groupby("method", dropna=False, sort=True):
        ax.plot(group["coverage_level"], group["empirical_coverage"], marker="o", label=method or "method")
    line = np.linspace(0.0, 1.0, 50)
    ax.plot(line, line, linestyle="--", color="grey", label="ideal")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("SBC empirical coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir.plots / "coverage_table", data={col: coverage_df[col].tolist() for col in coverage_df.columns})


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Simulation-based calibration experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, test_mode=args.test)
    output_dir = OutputDir(args.output_dir, cfg["experiment_name"]).ensure()

    sbc_cfg = cfg["sbc"]
    benchmark_cfg = cfg["benchmark"]
    n_trials = int(sbc_cfg["n_trials"])
    coverage_levels = [float(x) for x in sbc_cfg["coverage_levels"]]

    base_benchmark = make_benchmark(benchmark_cfg)
    param_names = list(base_benchmark.limits.keys())
    if not param_names:
        raise ValueError("SBC requires at least one inferable parameter.")

    seeds = make_seeds(n_trials, int(cfg["execution"]["base_seed"]))
    trial_records = []

    t0 = time.time()
    for trial_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        true_params = {
            name: float(rng.uniform(base_benchmark.limits[name][0], base_benchmark.limits[name][1]))
            for name in param_names
        }
        trial_benchmark = make_benchmark(
            _true_param_config(
                benchmark_cfg,
                true_params=true_params,
                observed_seed=seed,
            )
        )

        for method_idx, method in enumerate(cfg["methods"]):
            method_seed = int(seed + 1000 * (method_idx + 1))
            records = run_method_distributed(
                method,
                trial_benchmark.simulate,
                trial_benchmark.limits,
                cfg["inference"],
                output_dir,
                replicate=trial_idx,
                seed=method_seed,
            )
            for param in param_names:
                samples = _posterior_samples(records, param)
                if samples.size == 0:
                    continue
                trial_records.append(
                    {
                        "trial": trial_idx,
                        "method": method,
                        "param": param,
                        "true_value": true_params[param],
                        "posterior_samples": samples,
                    }
                )

    elapsed = time.time() - t0
    name = cfg["experiment_name"]
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(elapsed))
    if args.test and is_root_rank():
        factor, extra, note = compute_scaling_factor(args.config)
        estimated = elapsed * factor + extra
        logger.info(
            "[%s] Estimated full run: ~%s  (%s)",
            name,
            format_duration(estimated),
            note,
        )
    if not is_root_rank():
        return

    write_timing_csv(output_dir.data / "timing.csv", name, elapsed, estimated, args.test)

    ranks_df = sbc_ranks(trial_records)
    coverage_df = empirical_coverage(trial_records, coverage_levels)
    _write_dataframe_csv(ranks_df, output_dir.data / "sbc_ranks.csv")
    _write_dataframe_csv(coverage_df, output_dir.data / "coverage.csv")

    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("rank_histogram"):
        _plot_rank_histogram(ranks_df, output_dir)
    if plots_cfg.get("coverage_table"):
        _plot_coverage_table(coverage_df, output_dir)

    write_metadata(
        output_dir,
        cfg,
        extra={
            "n_trials": n_trials,
            "coverage_levels": coverage_levels,
        },
    )


if __name__ == "__main__":
    main()
