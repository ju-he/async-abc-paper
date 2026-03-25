#!/usr/bin/env python3
"""Simulation-based calibration runner."""
import copy
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.analysis import final_state_results
from async_abc.analysis.sbc import empirical_coverage, sbc_ranks
from async_abc.benchmarks import make_benchmark
from async_abc.io.config import get_run_mode, is_small_mode, is_test_mode, load_config
from async_abc.io.paths import OutputDir
from async_abc.plotting.export import save_figure
from async_abc.utils.logging_utils import configure_logging
from async_abc.utils.metadata import write_metadata
from async_abc.inference.method_registry import method_execution_mode_for_cfg
from async_abc.utils.mpi import allgather, get_rank, get_world_size, is_root_rank
from async_abc.utils.shard_finalizers import finalize_experiment_by_name
from async_abc.utils.sharding import (
    ShardLayout,
    build_plan_payload,
    ensure_plan,
    estimate_sharded_wall_time,
    is_shard_mode,
    maybe_finalize_sharded_run,
    prepare_shard_workspace_distributed,
    read_json,
    split_indices,
    validate_shard_args,
    write_shard_failure_status,
    write_shard_status,
)
from async_abc.utils.runner import (
    compute_scaling_factor,
    format_duration,
    make_arg_parser,
    run_method_distributed,
    write_timing_comparison_csv,
    write_timing_csv,
)
from async_abc.utils.seeding import make_seeds

logger = logging.getLogger(__name__)


def _write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _write_trial_records_jsonl(trial_records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in trial_records:
            payload = dict(row)
            payload["posterior_samples"] = [float(x) for x in np.asarray(payload["posterior_samples"], dtype=float)]
            f.write(json.dumps(payload) + "\n")


def _posterior_samples(records, param_name: str, *, archive_size: int | None = None) -> np.ndarray:
    final = []
    for result in final_state_results(records, archive_size=archive_size):
        final.extend(result.records)
    return np.asarray([r.params[param_name] for r in final if param_name in r.params], dtype=float)


def _true_param_config(base_cfg: dict, true_params: dict[str, float], observed_seed: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["observed_data_seed"] = int(observed_seed)
    for param, value in true_params.items():
        cfg[f"true_{param}"] = float(value)
    return cfg


def _build_trial_context(
    *,
    base_benchmark,
    benchmark_cfg: dict,
    param_names: list[str],
    seed: int,
) -> tuple[dict[str, float], object]:
    rng = np.random.default_rng(seed)
    true_params = {
        name: float(rng.uniform(base_benchmark.limits[name][0], base_benchmark.limits[name][1]))
        for name in param_names
    }
    trial_benchmark = make_benchmark(
        _true_param_config(benchmark_cfg, true_params=true_params, observed_seed=seed)
    )
    return true_params, trial_benchmark


def _extend_trial_records(
    *,
    target: list[dict],
    trial_idx: int,
    method: str,
    true_params: dict[str, float],
    param_names: list[str],
    records,
    archive_size: int | None,
) -> None:
    for param in param_names:
        samples = _posterior_samples(records, param, archive_size=archive_size)
        if samples.size == 0:
            continue
        target.append({
            "trial": trial_idx,
            "method": method,
            "param": param,
            "true_value": true_params[param],
            "posterior_samples": samples,
        })


def _log_method_phase(method: str) -> None:
    if not is_root_rank():
        return
    if method == "abc_smc_baseline":
        logger.info("[sbc] entering abc_smc_baseline phase")
    else:
        logger.info("[sbc] entering %s phase", method)


def _plot_rank_histogram(ranks_df: pd.DataFrame, output_dir: OutputDir) -> None:
    if ranks_df.empty:
        return

    methods = list(ranks_df["method"].dropna().unique())
    fig, axes = plt.subplots(len(methods), 1, figsize=(6, max(3.5, 2.8 * len(methods))), squeeze=False)
    for idx, method in enumerate(methods):
        ax = axes[idx, 0]
        group = ranks_df[ranks_df["method"] == method]
        bins = int(group["n_samples"].max()) + 1 if not group.empty else 10
        ax.hist(group["rank"], bins=min(max(bins, 5), 30), color="steelblue", alpha=0.8)
        ax.set_title(f"Rank histogram: {method}")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "rank_histogram",
        data={col: ranks_df[col].tolist() for col in ranks_df.columns},
        metadata={
            "plot_name": "rank_histogram",
            "title": "Rank histogram",
            "summary_plot": True,
            "diagnostic_plot": False,
            "experiment_name": output_dir.root.name,
            "benchmark": False,
            "methods": methods,
        },
    )


def _plot_coverage_table(coverage_df: pd.DataFrame, output_dir: OutputDir) -> None:
    if coverage_df.empty:
        return

    plot_df = coverage_df.copy()
    grouped = plot_df.groupby(["method", "param"], dropna=False, sort=False)
    if "n_trials" not in plot_df.columns:
        plot_df["n_trials"] = grouped["empirical_coverage"].transform("count")
    if "empirical_coverage_ci_low" not in plot_df.columns or "empirical_coverage_ci_high" not in plot_df.columns:
        z = 1.959963984540054
        ci_low: list[float] = []
        ci_high: list[float] = []
        for row in plot_df.itertuples(index=False):
            n = max(int(getattr(row, "n_trials", 0)), 1)
            p = float(row.empirical_coverage)
            denom = 1.0 + (z * z) / n
            center = (p + (z * z) / (2.0 * n)) / denom
            margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
            ci_low.append(max(0.0, center - margin))
            ci_high.append(min(1.0, center + margin))
        plot_df["empirical_coverage_ci_low"] = ci_low
        plot_df["empirical_coverage_ci_high"] = ci_high

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in plot_df.groupby("method", dropna=False, sort=True):
        group = group.sort_values("coverage_level")
        ax.plot(group["coverage_level"], group["empirical_coverage"], marker="o", label=method or "method")
        if {"empirical_coverage_ci_low", "empirical_coverage_ci_high"} <= set(group.columns):
            ax.fill_between(
                group["coverage_level"],
                group["empirical_coverage_ci_low"],
                group["empirical_coverage_ci_high"],
                alpha=0.15,
            )
    line = np.linspace(0.0, 1.0, 50)
    ax.plot(line, line, linestyle="--", color="grey", label="ideal")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("SBC empirical coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "coverage_table",
        data={col: plot_df[col].tolist() for col in plot_df.columns},
        metadata={
            "plot_name": "coverage_table",
            "title": "SBC empirical coverage",
            "summary_plot": True,
            "diagnostic_plot": False,
            "experiment_name": output_dir.root.name,
            "benchmark": False,
            "methods": sorted(plot_df["method"].dropna().unique().tolist()),
        },
    )


def _finalize_sharded(cfg: dict, layout: ShardLayout, actual_num_shards: int) -> None:
    owner_id = f"{os.getenv('SLURM_JOB_ID', 'manual')}:{layout.shard_index}"
    maybe_finalize_sharded_run(
        layout=layout,
        actual_num_shards=actual_num_shards,
        owner_id=owner_id,
        finalize_fn=lambda shard_dirs, statuses: finalize_experiment_by_name(cfg, layout, shard_dirs, statuses),
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = make_arg_parser("Simulation-based calibration experiment.")
    args = parser.parse_args(argv)
    validate_shard_args(args)

    cfg = load_config(args.config, test_mode=args.test, small_mode=args.small)
    test_mode = is_test_mode(cfg)
    small_mode = is_small_mode(cfg)
    run_mode = get_run_mode(cfg)
    estimate_mode = run_mode != "full"
    experiment_name = cfg["experiment_name"]

    sbc_cfg = cfg["sbc"]
    benchmark_cfg = cfg["benchmark"]
    n_trials = int(sbc_cfg["n_trials"])
    coverage_levels = [float(x) for x in sbc_cfg["coverage_levels"]]
    archive_size = cfg.get("inference", {}).get("k")

    base_benchmark = make_benchmark(benchmark_cfg)
    param_names = list(base_benchmark.limits.keys())
    if not param_names:
        raise ValueError("SBC requires at least one inferable parameter.")

    seeds = make_seeds(n_trials, int(cfg["execution"]["base_seed"]))
    trial_records = []
    selected_trials = list(range(n_trials))

    if is_shard_mode(args):
        output_root = Path(args.output_dir)
        full_cfg = load_config(args.config, test_mode=False, small_mode=False)
        shard_index = args.shard_index if args.shard_index is not None else 0
        layout = ShardLayout(output_root, experiment_name, args.shard_run_id, shard_index)
        plan = read_json(layout.plan_path)
        if not plan:
            actual_num_shards = args.num_shards or 1
            plan = ensure_plan(
                layout,
                build_plan_payload(
                    experiment_name=experiment_name,
                    config_path=str(Path(args.config).resolve()),
                    unit_kind="trial",
                    full_total_units=full_cfg["sbc"]["n_trials"],
                    actual_total_units=n_trials,
                    target_total_units=n_trials,
                    requested_num_shards=actual_num_shards,
                    actual_num_shards=actual_num_shards,
                    test_mode=test_mode,
                    small_mode=small_mode,
                    run_mode=run_mode,
                    extend=args.extend,
                    run_id=args.shard_run_id,
                    completed_unit_indices=[],
                    pending_unit_indices=list(range(n_trials)),
                    shard_assignments=split_indices(n_trials, actual_num_shards),
                    runner_script=str(Path(__file__).resolve()),
                ),
            )
        actual_num_shards = int(plan["actual_num_shards"])
        if args.finalize_only:
            if is_root_rank():
                _finalize_sharded(cfg, layout, actual_num_shards)
            return
        selected_trials = [int(idx) for idx in plan["shard_assignments"][shard_index]]
        mode = prepare_shard_workspace_distributed(layout)
        if mode == "skip":
            if is_root_rank():
                _finalize_sharded(cfg, layout, actual_num_shards)
            return
        if is_root_rank():
            write_shard_status(
                layout,
                state="running",
                unit_indices=selected_trials,
                extra={"started_at_s": time.time(), "run_mode": run_mode},
            )
        output_dir = layout.shard_output_dir
    else:
        output_dir = OutputDir(args.output_dir, experiment_name).ensure()

    # Classify methods into execution phases once (use base_benchmark for classification).
    all_methods = cfg["methods"]
    all_ranks_methods = [
        m for m in all_methods
        if method_execution_mode_for_cfg(m, cfg["inference"], base_benchmark.simulate) == "all_ranks"
    ]
    rank_parallel_methods = [
        m for m in all_methods
        if method_execution_mode_for_cfg(m, cfg["inference"], base_benchmark.simulate) == "rank_parallel"
    ]
    other_methods = [m for m in all_methods if m not in set(all_ranks_methods) and m not in set(rank_parallel_methods)]
    phase1_methods = all_ranks_methods + other_methods
    method_idx_map = {m: idx for idx, m in enumerate(all_methods)}

    t0 = time.time()
    try:
        # ── Phase 1: all_ranks (and other) methods, method-major ────────────────
        for method in phase1_methods:
            _log_method_phase(method)
            for trial_idx in selected_trials:
                seed = seeds[trial_idx]
                true_params, trial_benchmark = _build_trial_context(
                    base_benchmark=base_benchmark,
                    benchmark_cfg=benchmark_cfg,
                    param_names=param_names,
                    seed=seed,
                )
                method_seed = int(seed + 1000 * (method_idx_map[method] + 1))
                if is_root_rank():
                    logger.info("[runner] starting method %s trial=%s", method, trial_idx)
                records = run_method_distributed(
                    method,
                    trial_benchmark.simulate,
                    trial_benchmark.limits,
                    cfg["inference"],
                    output_dir,
                    replicate=trial_idx,
                    seed=method_seed,
                )
                if is_root_rank():
                    logger.info("[runner] finished method %s trial=%s", method, trial_idx)
                _extend_trial_records(
                    target=trial_records,
                    trial_idx=trial_idx,
                    method=method,
                    true_params=true_params,
                    param_names=param_names,
                    records=records,
                    archive_size=archive_size,
                )

        # ── Barrier: drain MPI state before rank_parallel phase ──────────────────
        if rank_parallel_methods:
            allgather(None)

        # ── Phase 2: rank_parallel methods, method-major on each rank ────────────
        if rank_parallel_methods:
            my_rank = get_rank()
            world_size = get_world_size()
            my_trials = selected_trials[my_rank::world_size]

            phase2_error = None
            my_trial_records = []
            for method in rank_parallel_methods:
                _log_method_phase(method)
                if phase2_error:
                    break
                for trial_idx in my_trials:
                    if phase2_error:
                        break
                    seed = seeds[trial_idx]
                    true_params, trial_benchmark = _build_trial_context(
                        base_benchmark=base_benchmark,
                        benchmark_cfg=benchmark_cfg,
                        param_names=param_names,
                        seed=seed,
                    )
                    method_seed = int(seed + 1000 * (method_idx_map[method] + 1))
                    try:
                        if is_root_rank():
                            logger.info("[runner] starting method %s trial=%s", method, trial_idx)
                        records = run_method_distributed(
                            method,
                            trial_benchmark.simulate,
                            trial_benchmark.limits,
                            cfg["inference"],
                            output_dir,
                            replicate=trial_idx,
                            seed=method_seed,
                        )
                        if is_root_rank():
                            logger.info("[runner] finished method %s trial=%s", method, trial_idx)
                        _extend_trial_records(
                            target=my_trial_records,
                            trial_idx=trial_idx,
                            method=method,
                            true_params=true_params,
                            param_names=param_names,
                            records=records,
                            archive_size=archive_size,
                        )
                    except ImportError as exc:
                        phase2_error = ("ImportError", str(exc))
                        break
                    except Exception:
                        phase2_error = ("Exception", traceback.format_exc())
                        break

            all_errors = allgather(phase2_error)
            first_error = next((e for e in all_errors if e is not None), None)
            if first_error is not None:
                kind, message = first_error
                if kind == "ImportError":
                    raise ImportError(message)
                raise RuntimeError(message)

            all_trial_records_by_rank = allgather(my_trial_records)
            if is_root_rank():
                for rank_records in all_trial_records_by_rank:
                    trial_records.extend(rank_records)

        elapsed = time.time() - t0
    except Exception as exc:
        if is_shard_mode(args) and is_root_rank():
            write_shard_failure_status(
                layout,
                unit_indices=selected_trials,
                started_at_s=t0,
                exc=exc,
            )
        raise
    name = cfg["experiment_name"]
    estimated = None
    if is_root_rank():
        logger.info("[%s] Done in %s", name, format_duration(elapsed))
    if estimate_mode and is_root_rank():
        factor, extra, _ = compute_scaling_factor(
            args.config,
            small_mode=small_mode,
            test_mode=test_mode,
        )
        estimated = elapsed * factor + extra
        logger.info("[%s] Estimated full run: ~%s", name, format_duration(estimated))
    if not is_root_rank():
        return

    estimated_sharded = None
    if estimate_mode and is_shard_mode(args) and estimated is not None:
        estimated_sharded = estimate_sharded_wall_time(
            estimated,
            int(plan.get("full_total_units", full_cfg["sbc"]["n_trials"])),
            int(plan.get("requested_num_shards", actual_num_shards)),
        )
    write_timing_csv(
        output_dir.data / "timing.csv",
        name,
        elapsed,
        estimated,
        test_mode,
        run_mode,
        estimated_full_unsharded_s=estimated,
        estimated_full_sharded_wall_s=estimated_sharded,
        aggregate_compute_s=elapsed,
    )
    _write_trial_records_jsonl(trial_records, output_dir.data / "sbc_trials.jsonl")

    if is_shard_mode(args):
        write_shard_status(
            layout,
            state="completed",
            unit_indices=selected_trials,
            elapsed_s=elapsed,
            estimated_full_s=estimated,
            estimated_full_unsharded_s=estimated,
            estimated_full_sharded_wall_s=estimated_sharded,
            aggregate_compute_s=elapsed,
            extra={"started_at_s": t0, "finished_at_s": time.time(), "run_mode": run_mode},
        )
        _finalize_sharded(cfg, layout, actual_num_shards)
        return

    write_timing_comparison_csv(Path(args.output_dir))
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
