"""Experiment-specific shard merge/finalization helpers."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.sbc import empirical_coverage, sbc_ranks
from ..io.paths import OutputDir
from ..io.records import ParticleRecord, load_records
from ..plotting.reporters import (
    plot_ablation_summary,
    plot_benchmark_diagnostics,
    plot_sensitivity_summary,
    plot_worker_gantt,
)
from ..plotting.export import save_figure
from ..utils.metadata import write_metadata
from ..utils.runner import write_timing_csv
from .sharding import (
    ShardLayout,
    merge_csv_group,
    publish_directory_atomically,
    read_json,
    shard_timing_summary,
)


def _timing_estimates(statuses: List[Dict[str, Any]]) -> Dict[str, float | None]:
    unsharded = [
        float(s["estimated_full_unsharded_s"])
        for s in statuses
        if s.get("estimated_full_unsharded_s") not in (None, "")
    ]
    sharded = [
        float(s["estimated_full_sharded_wall_s"])
        for s in statuses
        if s.get("estimated_full_sharded_wall_s") not in (None, "")
    ]
    return {
        "estimated_full_unsharded_s": max(unsharded) if unsharded else None,
        "estimated_full_sharded_wall_s": max(sharded) if sharded else None,
    }


def _sort_raw_result_row(row: Dict[str, str]) -> tuple:
    return (
        row.get("method", ""),
        int(row.get("replicate", 0) or 0),
        int(row.get("step", 0) or 0),
    )


def _merge_raw_records(shard_dirs: List[OutputDir], destination: Path) -> List[ParticleRecord]:
    merge_csv_group(
        [shard_dir.data / "raw_results.csv" for shard_dir in shard_dirs],
        destination,
        sort_key=_sort_raw_result_row,
    )
    return load_records(destination)


def _timing_payload(cfg: dict, statuses: List[Dict[str, Any]]) -> Dict[str, float | None]:
    summary = shard_timing_summary(statuses)
    estimates = _timing_estimates(statuses)
    return {
        "elapsed_s": summary["elapsed_s"],
        "aggregate_compute_s": summary["aggregate_compute_s"],
        "estimated_full_s": estimates["estimated_full_unsharded_s"],
        "estimated_full_unsharded_s": estimates["estimated_full_unsharded_s"],
        "estimated_full_sharded_wall_s": estimates["estimated_full_sharded_wall_s"],
    }


def _publish_temp_output(layout: ShardLayout, tmp_output_dir: OutputDir) -> None:
    publish_directory_atomically(tmp_output_dir.root, layout.final_output_dir.root)


def _temp_output_dir(layout: ShardLayout) -> OutputDir:
    temp_root = layout.output_root / "_shards" / layout.experiment_name / "_merge_tmp"
    if temp_root.exists():
        import shutil

        shutil.rmtree(temp_root)
    return OutputDir(temp_root, layout.experiment_name).ensure()


def _plot_throughput_vs_slowdown(throughput_rows, output_dir: OutputDir) -> None:
    if not throughput_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = sorted({row["base_method"] for row in throughput_rows})
    for method in methods:
        rows = [row for row in throughput_rows if row["base_method"] == method]
        rows = sorted(rows, key=lambda row: float(row["slowdown_factor"]))
        ax.plot(
            [float(row["slowdown_factor"]) for row in rows],
            [float(row["throughput_sims_per_s"]) for row in rows],
            marker="o",
            label=method,
        )

    ax.set_xlabel("slowdown factor")
    ax.set_ylabel("throughput (sim/s)")
    ax.set_title("Throughput vs. straggler slowdown")
    ax.legend(frameon=False)
    fig.tight_layout()

    data = {
        "slowdown_factor": [row["slowdown_factor"] for row in throughput_rows],
        "base_method": [row["base_method"] for row in throughput_rows],
        "replicate": [row["replicate"] for row in throughput_rows],
        "throughput_sims_per_s": [row["throughput_sims_per_s"] for row in throughput_rows],
        "wall_time_s": [row["wall_time_s"] for row in throughput_rows],
    }
    save_figure(fig, output_dir.plots / "throughput_vs_slowdown", data=data)


def _plot_rank_histogram(ranks_df, output_dir: OutputDir) -> None:
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


def _plot_coverage_table(coverage_df, output_dir: OutputDir) -> None:
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


def finalize_benchmark_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    records = _merge_raw_records(shard_dirs, tmp_output.data / "raw_results.csv")
    timing = _timing_payload(cfg, statuses)
    write_timing_csv(
        tmp_output.data / "timing.csv",
        cfg["experiment_name"],
        timing["elapsed_s"] or 0.0,
        timing["estimated_full_s"],
        bool(cfg.get("inference", {}).get("test_mode", False)),
        estimated_full_unsharded_s=timing["estimated_full_unsharded_s"],
        estimated_full_sharded_wall_s=timing["estimated_full_sharded_wall_s"],
        aggregate_compute_s=timing["aggregate_compute_s"],
    )
    if any(cfg.get("plots", {}).values()):
        plot_benchmark_diagnostics(records, cfg, tmp_output)
    write_metadata(
        tmp_output,
        cfg,
        extra={
            "sharding": {
                "actual_num_shards": len(shard_dirs),
                "statuses": statuses,
            }
        },
    )
    _publish_temp_output(layout, tmp_output)
    return {
        "record_count": len(records),
        "timing": timing,
    }


def finalize_sensitivity_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    variant_names = sorted({path.name for shard_dir in shard_dirs for path in shard_dir.data.glob("sensitivity_*.csv")})
    for filename in variant_names:
        merge_csv_group(
            [shard_dir.data / filename for shard_dir in shard_dirs],
            tmp_output.data / filename,
            sort_key=_sort_raw_result_row,
        )
    timing = _timing_payload(cfg, statuses)
    write_timing_csv(
        tmp_output.data / "timing.csv",
        cfg["experiment_name"],
        timing["elapsed_s"] or 0.0,
        timing["estimated_full_s"],
        bool(cfg.get("inference", {}).get("test_mode", False)),
        estimated_full_unsharded_s=timing["estimated_full_unsharded_s"],
        estimated_full_sharded_wall_s=timing["estimated_full_sharded_wall_s"],
        aggregate_compute_s=timing["aggregate_compute_s"],
    )
    if cfg.get("plots", {}).get("sensitivity_heatmap"):
        plot_sensitivity_summary(tmp_output.data, cfg.get("sensitivity_grid", {}), tmp_output)
    write_metadata(tmp_output, cfg, extra={"sharding": {"actual_num_shards": len(shard_dirs), "statuses": statuses}})
    _publish_temp_output(layout, tmp_output)
    return {"timing": timing}


def finalize_ablation_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    variant_names = sorted({path.name for shard_dir in shard_dirs for path in shard_dir.data.glob("ablation_*.csv")})
    for filename in variant_names:
        merge_csv_group(
            [shard_dir.data / filename for shard_dir in shard_dirs],
            tmp_output.data / filename,
            sort_key=_sort_raw_result_row,
        )
    timing = _timing_payload(cfg, statuses)
    write_timing_csv(
        tmp_output.data / "timing.csv",
        cfg["experiment_name"],
        timing["elapsed_s"] or 0.0,
        timing["estimated_full_s"],
        bool(cfg.get("inference", {}).get("test_mode", False)),
        estimated_full_unsharded_s=timing["estimated_full_unsharded_s"],
        estimated_full_sharded_wall_s=timing["estimated_full_sharded_wall_s"],
        aggregate_compute_s=timing["aggregate_compute_s"],
    )
    if cfg.get("plots", {}).get("ablation_comparison"):
        plot_ablation_summary(tmp_output.data, cfg.get("ablation_variants", []), tmp_output)
    write_metadata(tmp_output, cfg, extra={"sharding": {"actual_num_shards": len(shard_dirs), "statuses": statuses}})
    _publish_temp_output(layout, tmp_output)
    return {"timing": timing}


def finalize_straggler_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    records = _merge_raw_records(shard_dirs, tmp_output.data / "raw_results.csv")
    merge_csv_group(
        [shard_dir.data / "throughput_vs_slowdown_summary.csv" for shard_dir in shard_dirs],
        tmp_output.data / "throughput_vs_slowdown_summary.csv",
        sort_key=lambda row: (
            float(row.get("slowdown_factor", 0.0) or 0.0),
            row.get("base_method", ""),
            int(row.get("replicate", 0) or 0),
        ),
    )
    timing = _timing_payload(cfg, statuses)
    write_timing_csv(
        tmp_output.data / "timing.csv",
        cfg["experiment_name"],
        timing["elapsed_s"] or 0.0,
        timing["estimated_full_s"],
        bool(cfg.get("inference", {}).get("test_mode", False)),
        estimated_full_unsharded_s=timing["estimated_full_unsharded_s"],
        estimated_full_sharded_wall_s=timing["estimated_full_sharded_wall_s"],
        aggregate_compute_s=timing["aggregate_compute_s"],
    )
    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("throughput_vs_slowdown"):
        with open(tmp_output.data / "throughput_vs_slowdown_summary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        _plot_throughput_vs_slowdown(rows, tmp_output)
    if plots_cfg.get("gantt"):
        slowdown_pattern = re.compile(r"__straggler_slowdown([0-9.eE+-]+)x$")
        factors = []
        for record in records:
            match = slowdown_pattern.search(record.method)
            if match:
                factors.append(float(match.group(1)))
        if factors:
            worst = max(factors)
            worst_records = [r for r in records if slowdown_pattern.search(r.method) and float(slowdown_pattern.search(r.method).group(1)) == worst]
            plot_worker_gantt(worst_records, tmp_output)
    write_metadata(tmp_output, cfg, extra={"sharding": {"actual_num_shards": len(shard_dirs), "statuses": statuses}})
    _publish_temp_output(layout, tmp_output)
    return {"record_count": len(records), "timing": timing}


def finalize_runtime_heterogeneity_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    records = _merge_raw_records(shard_dirs, tmp_output.data / "raw_results.csv")
    timing = _timing_payload(cfg, statuses)
    write_timing_csv(
        tmp_output.data / "timing.csv",
        cfg["experiment_name"],
        timing["elapsed_s"] or 0.0,
        timing["estimated_full_s"],
        bool(cfg.get("inference", {}).get("test_mode", False)),
        estimated_full_unsharded_s=timing["estimated_full_unsharded_s"],
        estimated_full_sharded_wall_s=timing["estimated_full_sharded_wall_s"],
        aggregate_compute_s=timing["aggregate_compute_s"],
    )
    if any(cfg.get("plots", {}).values()):
        plot_benchmark_diagnostics(records, cfg, tmp_output)
    if cfg.get("plots", {}).get("gantt"):
        plot_worker_gantt(records, tmp_output)
    write_metadata(tmp_output, cfg, extra={"sharding": {"actual_num_shards": len(shard_dirs), "statuses": statuses}})
    _publish_temp_output(layout, tmp_output)
    return {"record_count": len(records), "timing": timing}


def finalize_sbc_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    trial_records: List[Dict[str, Any]] = []
    for shard_dir in shard_dirs:
        path = shard_dir.data / "sbc_trials.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                payload["posterior_samples"] = np.asarray(payload["posterior_samples"], dtype=float)
                trial_records.append(payload)
    ranks_df = sbc_ranks(trial_records)
    coverage_df = empirical_coverage(trial_records, cfg["sbc"]["coverage_levels"])
    ranks_df.to_csv(tmp_output.data / "sbc_ranks.csv", index=False)
    coverage_df.to_csv(tmp_output.data / "coverage.csv", index=False)
    timing = _timing_payload(cfg, statuses)
    write_timing_csv(
        tmp_output.data / "timing.csv",
        cfg["experiment_name"],
        timing["elapsed_s"] or 0.0,
        timing["estimated_full_s"],
        bool(cfg.get("inference", {}).get("test_mode", False)),
        estimated_full_unsharded_s=timing["estimated_full_unsharded_s"],
        estimated_full_sharded_wall_s=timing["estimated_full_sharded_wall_s"],
        aggregate_compute_s=timing["aggregate_compute_s"],
    )
    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("rank_histogram"):
        _plot_rank_histogram(ranks_df, tmp_output)
    if plots_cfg.get("coverage_table"):
        _plot_coverage_table(coverage_df, tmp_output)
    write_metadata(tmp_output, cfg, extra={"sharding": {"actual_num_shards": len(shard_dirs), "statuses": statuses}})
    _publish_temp_output(layout, tmp_output)
    return {
        "trial_records": len(trial_records),
        "timing": timing,
    }


def finalize_experiment_by_name(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dispatch to the appropriate shard finalizer for *cfg*."""
    name = cfg["experiment_name"]
    if name in {"gaussian_mean", "gandk", "lotka_volterra", "cellular_potts"}:
        return finalize_benchmark_experiment(cfg, layout, shard_dirs, statuses)
    if name == "sensitivity":
        return finalize_sensitivity_experiment(cfg, layout, shard_dirs, statuses)
    if name == "ablation":
        return finalize_ablation_experiment(cfg, layout, shard_dirs, statuses)
    if name == "straggler":
        return finalize_straggler_experiment(cfg, layout, shard_dirs, statuses)
    if name == "runtime_heterogeneity":
        return finalize_runtime_heterogeneity_experiment(cfg, layout, shard_dirs, statuses)
    if name == "sbc":
        return finalize_sbc_experiment(cfg, layout, shard_dirs, statuses)
    raise ValueError(f"No shard finalizer registered for experiment_name={name!r}")
