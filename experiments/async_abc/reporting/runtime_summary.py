"""Shared runtime summary builders for HPC-focused experiments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from ..analysis import barrier_overhead_fraction, base_method_name, final_state_results, posterior_quality_curve
from ..io.records import ParticleRecord


def compute_idle_fraction(records: List[ParticleRecord]) -> Dict[str, Dict[int, float]]:
    """Compute per-method, per-replicate idle fractions from timing data."""
    timed = [r for r in _attempt_timing_records(records) if r.worker_id is not None]
    if not timed:
        return {}

    by_method_rep: Dict[str, Dict[int, List[ParticleRecord]]] = {}
    for record in timed:
        by_method_rep.setdefault(record.method, {}).setdefault(int(record.replicate), []).append(record)

    result: Dict[str, Dict[int, float]] = {}
    for method, by_rep in by_method_rep.items():
        result[method] = {}
        for replicate, recs in by_rep.items():
            workers = {record.worker_id for record in recs}
            n_workers = len(workers)
            span = max(float(record.sim_end_time) for record in recs) - min(float(record.sim_start_time) for record in recs)
            if span <= 0 or n_workers == 0:
                result[method][replicate] = float("nan")
                continue
            total_busy = sum(float(record.sim_end_time) - float(record.sim_start_time) for record in recs)
            result[method][replicate] = 1.0 - total_busy / (n_workers * span)
    return result


def runtime_utilization_rows(records: List[ParticleRecord]) -> pd.DataFrame:
    """Return per-replicate utilization-loss rows for runtime heterogeneity sweeps."""
    rows: list[dict[str, object]] = []
    idle_data = compute_idle_fraction(records)
    for method, by_replicate in sorted(idle_data.items()):
        sigma = _sigma_from_method(method)
        if sigma is None:
            continue
        for replicate, value in sorted(by_replicate.items()):
            rows.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "base_method": base_method_name(method),
                    "replicate": int(replicate),
                    "measurement_method": "worker_idle",
                    "utilization_loss_fraction": float(value),
                }
            )

    barrier_df = barrier_overhead_fraction(records)
    if not barrier_df.empty:
        for row in barrier_df.itertuples(index=False):
            method = str(row.method)
            sigma = _sigma_from_method(method)
            if sigma is None:
                continue
            base_method = base_method_name(method)
            if base_method not in {"abc_smc_baseline", "pyabc_smc"}:
                continue
            rows.append(
                {
                    "sigma": sigma,
                    "method": method,
                    "base_method": base_method,
                    "replicate": int(row.replicate),
                    "measurement_method": "barrier_overhead",
                    "utilization_loss_fraction": float(row.barrier_overhead_fraction),
                }
            )

    return (
        pd.DataFrame(rows).sort_values(["sigma", "base_method", "measurement_method", "replicate"]).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def normalize_runtime_utilization_summary(
    records: List[ParticleRecord],
    *,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize a utilization summary for the heterogeneity plotting contract."""
    if summary_df is None:
        return runtime_utilization_rows(records)
    normalized = summary_df.copy()
    if normalized.empty:
        return normalized
    if "measurement_method" not in normalized.columns:
        normalized["measurement_method"] = "worker_idle"
    required = {"sigma", "base_method", "replicate", "measurement_method", "utilization_loss_fraction"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"Runtime utilization summary is missing required columns: {sorted(missing)}")
    return normalized


def runtime_performance_summary(records: List[ParticleRecord], cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return one aggregate row per runtime-heterogeneity method and replicate."""
    benchmark_cfg = cfg.get("benchmark", {})
    true_params = _true_params_from_benchmark_cfg(benchmark_cfg)
    archive_size = cfg.get("inference", {}).get("k")
    idle_map = compute_idle_fraction(records)
    rows = []
    for tagged_method in sorted({record.method for record in records if "__sigma" in str(record.method)}):
        sigma = _sigma_from_method(tagged_method)
        if sigma is None:
            continue
        method_records = [record for record in records if record.method == tagged_method]
        for replicate in sorted({int(record.replicate) for record in method_records}):
            subset = [record for record in method_records if int(record.replicate) == replicate]
            if not subset:
                continue
            attempts = _attempt_records(subset)
            elapsed = _elapsed_wall_time(attempts)
            rows.append(
                {
                    "sigma": sigma,
                    "base_method": base_method_name(tagged_method),
                    "method": tagged_method,
                    "replicate": int(replicate),
                    "measurement_method": "worker_idle",
                    "elapsed_wall_time_s": float(elapsed),
                    "total_attempts": int(len(attempts)),
                    "final_posterior_size": int(_final_posterior_size(subset, archive_size=archive_size)),
                    "final_quality_wasserstein": float(
                        _final_quality_wasserstein(subset, true_params=true_params, archive_size=archive_size)
                    ),
                    "throughput_sims_per_s": float(len(attempts) / elapsed) if elapsed > 0 else float("nan"),
                    "utilization_loss_fraction": float(
                        idle_map.get(tagged_method, {}).get(replicate, float("nan"))
                    ),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["sigma", "base_method", "replicate"]).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def straggler_performance_summary_row(
    records: List[ParticleRecord],
    *,
    cfg: Dict[str, Any],
    tagged_method: str,
) -> Dict[str, float | int]:
    """Return summary metrics for one tagged straggler run."""
    archive_size = cfg.get("inference", {}).get("k")
    attempts = _attempt_records(records)
    replicate = int(records[0].replicate) if records else 0
    return {
        "total_attempts": int(len(attempts)),
        "final_posterior_size": int(_final_posterior_size(records, archive_size=archive_size)),
        "final_quality_wasserstein": float(
            _final_quality_wasserstein(
                records,
                true_params=_true_params_from_benchmark_cfg(cfg.get("benchmark", {})),
                archive_size=archive_size,
            )
        ),
        "utilization_loss_fraction": float(
            compute_idle_fraction(records).get(tagged_method, {}).get(replicate, float("nan"))
        ),
    }


def _attempt_timing_records(records: Iterable[ParticleRecord]) -> List[ParticleRecord]:
    timed = [
        record for record in records
        if record.sim_start_time is not None and record.sim_end_time is not None
    ]
    if not timed:
        return []

    selected: List[ParticleRecord] = []
    by_method_replicate: Dict[tuple[str, int], List[ParticleRecord]] = {}
    for record in timed:
        by_method_replicate.setdefault((record.method, int(record.replicate)), []).append(record)

    for group in by_method_replicate.values():
        attempt_rows = [record for record in group if record.record_kind == "simulation_attempt"]
        selected.extend(attempt_rows if attempt_rows else [record for record in group if record.worker_id is not None])
    return selected


def _attempt_records(records: List[ParticleRecord]) -> List[ParticleRecord]:
    attempt_rows = [record for record in records if record.record_kind == "simulation_attempt"]
    return attempt_rows if attempt_rows else list(records)


def _elapsed_wall_time(records: List[ParticleRecord]) -> float:
    start_times = [float(record.sim_start_time) for record in records if record.sim_start_time is not None]
    end_times = [float(record.sim_end_time) for record in records if record.sim_end_time is not None]
    if start_times and end_times and max(end_times) > min(start_times):
        return float(max(end_times) - min(start_times))
    return max((float(record.wall_time) for record in records), default=0.0)


def _final_posterior_size(records: List[ParticleRecord], *, archive_size: int | None) -> int:
    for result in final_state_results(records, archive_size=archive_size):
        return int(result.n_particles_used)
    return 0


def _final_quality_wasserstein(
    records: List[ParticleRecord],
    *,
    true_params: Dict[str, float],
    archive_size: int | None,
) -> float:
    if not true_params:
        return float("nan")
    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="wall_time",
        checkpoint_strategy="quantile",
        checkpoint_count=8,
        archive_size=archive_size,
    )
    if quality_df.empty:
        return float("nan")
    ordered = quality_df.sort_values("axis_value")
    return float(ordered.iloc[-1]["wasserstein"])


def _true_params_from_benchmark_cfg(benchmark_cfg: Dict[str, Any]) -> Dict[str, float]:
    true_params = {}
    for key, value in benchmark_cfg.items():
        if key.startswith("true_") and isinstance(value, (int, float)):
            true_params[key.removeprefix("true_")] = float(value)
    return true_params


def _sigma_from_method(method: str) -> float | None:
    if "__sigma" not in method:
        return None
    try:
        return float(method.split("__sigma", 1)[1])
    except ValueError:
        return None
