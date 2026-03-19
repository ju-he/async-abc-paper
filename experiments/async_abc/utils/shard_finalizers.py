"""Experiment-specific shard merge/finalization helpers."""
from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
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
    detect_completed_replicates_in_output,
    existing_extension_history,
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


def _is_extension_batch(layout: ShardLayout) -> bool:
    return bool(read_json(layout.plan_path).get("extend"))


def _merge_sources(layout: ShardLayout, shard_dirs: List[OutputDir], filename: str) -> List[Path]:
    """Return source paths for a merge: existing output first (if extending), then all shard outputs."""
    sources = [shard_dir.data / filename for shard_dir in shard_dirs]
    if _is_extension_batch(layout):
        return [layout.final_output_dir.data / filename] + sources
    return sources


def _copy_existing_timing_history(layout: ShardLayout, tmp_output: OutputDir) -> None:
    if not _is_extension_batch(layout):
        return
    existing_path = layout.final_output_dir.data / "timing.csv"
    if existing_path.exists():
        shutil.copy2(existing_path, tmp_output.data / "timing.csv")


def _write_batch_timing(cfg: dict, layout: ShardLayout, tmp_output: OutputDir, timing: Dict[str, float | None]) -> None:
    _copy_existing_timing_history(layout, tmp_output)
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


def _metadata_extra(cfg: dict, layout: ShardLayout, statuses: List[Dict[str, Any]], tmp_output: OutputDir) -> Dict[str, Any]:
    plan = read_json(layout.plan_path)
    completed_replicates = (
        detect_completed_replicates_in_output(tmp_output, cfg)
        if plan.get("unit_kind") == "replicate"
        else []
    )
    history = existing_extension_history(layout.output_root, cfg["experiment_name"])
    history.append(
        {
            "run_id": layout.run_id,
            "completed_unit_indices": plan.get("completed_unit_indices", []),
            "pending_unit_indices": plan.get("pending_unit_indices", []),
            "submitted_job_ids": plan.get("submitted_job_ids", []),
            "finalized_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return {
        "completed_replicates": completed_replicates,
        "completed_replicate_count": len(completed_replicates),
        "last_shard_run_id": layout.run_id,
        "extension_runs": history,
        "sharding": {
            "run_id": layout.run_id,
            "actual_num_shards": len(statuses),
            "statuses": statuses,
            "plan": plan,
        },
    }


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
    temp_root = layout.run_root / "_merge_tmp"
    if temp_root.exists():
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
    merge_csv_group(_merge_sources(layout, shard_dirs, "raw_results.csv"), tmp_output.data / "raw_results.csv", sort_key=_sort_raw_result_row)
    records = load_records(tmp_output.data / "raw_results.csv")
    timing = _timing_payload(cfg, statuses)
    _write_batch_timing(cfg, layout, tmp_output, timing)
    if any(cfg.get("plots", {}).values()):
        plot_benchmark_diagnostics(records, cfg, tmp_output)
    write_metadata(tmp_output, cfg, extra=_metadata_extra(cfg, layout, statuses, tmp_output))
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
    if _is_extension_batch(layout):
        variant_names = sorted(set(variant_names) | {path.name for path in layout.final_output_dir.data.glob("sensitivity_*.csv")})
    for filename in variant_names:
        merge_csv_group(_merge_sources(layout, shard_dirs, filename), tmp_output.data / filename, sort_key=_sort_raw_result_row)
    timing = _timing_payload(cfg, statuses)
    _write_batch_timing(cfg, layout, tmp_output, timing)
    if cfg.get("plots", {}).get("sensitivity_heatmap"):
        plot_sensitivity_summary(tmp_output.data, cfg.get("sensitivity_grid", {}), tmp_output)
    write_metadata(tmp_output, cfg, extra=_metadata_extra(cfg, layout, statuses, tmp_output))
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
    if _is_extension_batch(layout):
        variant_names = sorted(set(variant_names) | {path.name for path in layout.final_output_dir.data.glob("ablation_*.csv")})
    for filename in variant_names:
        merge_csv_group(_merge_sources(layout, shard_dirs, filename), tmp_output.data / filename, sort_key=_sort_raw_result_row)
    timing = _timing_payload(cfg, statuses)
    _write_batch_timing(cfg, layout, tmp_output, timing)
    if cfg.get("plots", {}).get("ablation_comparison"):
        plot_ablation_summary(tmp_output.data, cfg.get("ablation_variants", []), tmp_output)
    write_metadata(tmp_output, cfg, extra=_metadata_extra(cfg, layout, statuses, tmp_output))
    _publish_temp_output(layout, tmp_output)
    return {"timing": timing}


def finalize_straggler_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    merge_csv_group(_merge_sources(layout, shard_dirs, "raw_results.csv"), tmp_output.data / "raw_results.csv", sort_key=_sort_raw_result_row)
    records = load_records(tmp_output.data / "raw_results.csv")
    merge_csv_group(
        _merge_sources(layout, shard_dirs, "throughput_vs_slowdown_summary.csv"),
        tmp_output.data / "throughput_vs_slowdown_summary.csv",
        sort_key=lambda row: (
            float(row.get("slowdown_factor", 0.0) or 0.0),
            row.get("base_method", ""),
            int(row.get("replicate", 0) or 0),
        ),
    )
    timing = _timing_payload(cfg, statuses)
    _write_batch_timing(cfg, layout, tmp_output, timing)
    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("throughput_vs_slowdown"):
        with open(tmp_output.data / "throughput_vs_slowdown_summary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        _plot_throughput_vs_slowdown(rows, tmp_output)
    if plots_cfg.get("gantt"):
        slowdown_pattern = re.compile(r"__straggler_slowdown([0-9.eE+-]+)x$")
        tagged_records: List[tuple[ParticleRecord, float]] = []
        for record in records:
            match = slowdown_pattern.search(record.method)
            if match:
                tagged_records.append((record, float(match.group(1))))
        if tagged_records:
            worst = max(factor for _, factor in tagged_records)
            worst_records = [record for record, factor in tagged_records if factor == worst]
            plot_worker_gantt(worst_records, tmp_output)
    write_metadata(tmp_output, cfg, extra=_metadata_extra(cfg, layout, statuses, tmp_output))
    _publish_temp_output(layout, tmp_output)
    return {"record_count": len(records), "timing": timing}


def finalize_runtime_heterogeneity_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)
    merge_csv_group(_merge_sources(layout, shard_dirs, "raw_results.csv"), tmp_output.data / "raw_results.csv", sort_key=_sort_raw_result_row)
    records = load_records(tmp_output.data / "raw_results.csv")
    timing = _timing_payload(cfg, statuses)
    _write_batch_timing(cfg, layout, tmp_output, timing)
    if any(cfg.get("plots", {}).values()):
        plot_benchmark_diagnostics(records, cfg, tmp_output)
    if cfg.get("plots", {}).get("gantt"):
        plot_worker_gantt(records, tmp_output)
    write_metadata(tmp_output, cfg, extra=_metadata_extra(cfg, layout, statuses, tmp_output))
    _publish_temp_output(layout, tmp_output)
    return {"record_count": len(records), "timing": timing}


def finalize_sbc_experiment(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tmp_output = _temp_output_dir(layout)

    def _load_jsonl_trials(path: Path) -> List[Dict[str, Any]]:
        result = []
        if not path.exists():
            return result
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                payload["posterior_samples"] = np.asarray(payload["posterior_samples"], dtype=float)
                result.append(payload)
        return result

    trial_sources = _merge_sources(layout, shard_dirs, "sbc_trials.jsonl")
    trial_records: List[Dict[str, Any]] = []
    for path in trial_sources:
        trial_records.extend(_load_jsonl_trials(path))
    ranks_df = sbc_ranks(trial_records)
    coverage_df = empirical_coverage(trial_records, cfg["sbc"]["coverage_levels"])
    ranks_df.to_csv(tmp_output.data / "sbc_ranks.csv", index=False)
    coverage_df.to_csv(tmp_output.data / "coverage.csv", index=False)
    timing = _timing_payload(cfg, statuses)
    _write_batch_timing(cfg, layout, tmp_output, timing)
    plots_cfg = cfg.get("plots", {})
    if plots_cfg.get("rank_histogram"):
        _plot_rank_histogram(ranks_df, tmp_output)
    if plots_cfg.get("coverage_table"):
        _plot_coverage_table(coverage_df, tmp_output)
    write_metadata(tmp_output, cfg, extra=_metadata_extra(cfg, layout, statuses, tmp_output))
    _publish_temp_output(layout, tmp_output)
    return {
        "trial_records": len(trial_records),
        "timing": timing,
    }


_FINALIZER_REGISTRY: Dict[str, Any] = {
    "gaussian_mean": finalize_benchmark_experiment,
    "gandk": finalize_benchmark_experiment,
    "lotka_volterra": finalize_benchmark_experiment,
    "cellular_potts": finalize_benchmark_experiment,
    "sensitivity": finalize_sensitivity_experiment,
    "ablation": finalize_ablation_experiment,
    "straggler": finalize_straggler_experiment,
    "runtime_heterogeneity": finalize_runtime_heterogeneity_experiment,
    "sbc": finalize_sbc_experiment,
}


def finalize_experiment_by_name(
    cfg: dict,
    layout: ShardLayout,
    shard_dirs: List[OutputDir],
    statuses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dispatch to the appropriate shard finalizer for *cfg*."""
    name = cfg["experiment_name"]
    fn = _FINALIZER_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"No shard finalizer registered for experiment_name={name!r}")
    return fn(cfg, layout, shard_dirs, statuses)
