"""High-level plot reporters called by runner scripts.

Each function reads already-collected records (or CSVs) and produces the
standard set of figures for a given experiment type.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from ..analysis import (
    benchmark_plot_audit,
    barrier_overhead_fraction,
    base_method_name,
    final_state_records,
    final_state_results,
    generation_spans,
    lotka_tol_init_diagnostic,
)
from .common import (
    archive_evolution_plot,
    corner_plot,
    gantt_plot,
    idle_fraction_comparison_plot,
    idle_fraction_plot,
    posterior_comparison_plot,
    posterior_quality_plot,
    posterior_plot,
    scaling_plot,
    sensitivity_heatmap,
    threshold_summary_plot,
    throughput_over_time_plot,
    tolerance_trajectory_plot,
)
from .export import save_figure, write_plot_metadata


def _param_names(records: List[ParticleRecord]) -> List[str]:
    for r in records:
        if r.params:
            return list(r.params.keys())
    return []


def _final_population(records: List[ParticleRecord]) -> List[ParticleRecord]:
    """Return pooled final posterior records across all method/replicate states."""
    return final_state_records(records)


def _merge_metadata(meta_path: Path, extra: Dict[str, Any]) -> None:
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2))


def _nonbenchmark_plot_metadata(
    output_dir: OutputDir,
    *,
    plot_name: str,
    title: str,
    methods: List[str] | None = None,
    summary_plot: bool = False,
    diagnostic_plot: bool = False,
    skip_reason: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "plot_name": plot_name,
        "title": title,
        "summary_plot": bool(summary_plot),
        "diagnostic_plot": bool(diagnostic_plot),
        "experiment_name": output_dir.root.name,
        "benchmark": False,
        "methods": sorted(methods or []),
    }
    if skip_reason:
        metadata["skip_reason"] = skip_reason
    if extra:
        metadata.update(extra)
    return metadata


def _attempt_timing_records(records: List[ParticleRecord]) -> List[ParticleRecord]:
    timed = [
        record for record in records
        if record.sim_start_time is not None and record.sim_end_time is not None
    ]
    if not timed:
        return []

    selected: List[ParticleRecord] = []
    by_method_replicate: Dict[tuple[str, int], List[ParticleRecord]] = defaultdict(list)
    for record in timed:
        by_method_replicate[(record.method, int(record.replicate))].append(record)

    for _, group in sorted(by_method_replicate.items()):
        attempt_rows = [record for record in group if record.record_kind == "simulation_attempt"]
        if attempt_rows:
            selected.extend(attempt_rows)
        else:
            selected.extend([record for record in group if record.worker_id is not None])
    return selected


def _runtime_debug_frame(records: List[ParticleRecord]) -> pd.DataFrame:
    timed = _attempt_timing_records(records)
    if not timed:
        return pd.DataFrame(
            columns=[
                "method",
                "base_method",
                "replicate",
                "worker_id",
                "n_attempts",
                "total_busy_s",
                "min_start",
                "max_end",
                "elapsed_wall_s",
                "active_span_s",
                "record_kind",
                "time_semantics",
            ]
        )

    run_bounds: Dict[tuple[str, int], tuple[float, float]] = {}
    for (method, replicate), group in defaultdict(list, {
        key: [r for r in timed if r.method == key[0] and int(r.replicate) == key[1]]
        for key in {(record.method, int(record.replicate)) for record in timed}
    }).items():
        starts = [float(record.sim_start_time) for record in group]
        ends = [float(record.sim_end_time) for record in group]
        run_bounds[(method, int(replicate))] = (min(starts), max(ends))

    rows: list[dict[str, object]] = []
    by_worker: Dict[tuple[str, int, str], List[ParticleRecord]] = defaultdict(list)
    for record in timed:
        worker_id = str(record.worker_id) if record.worker_id is not None else "unknown"
        by_worker[(record.method, int(record.replicate), worker_id)].append(record)

    for (method, replicate, worker_id), group in sorted(by_worker.items()):
        starts = np.asarray([float(record.sim_start_time) for record in group], dtype=float)
        ends = np.asarray([float(record.sim_end_time) for record in group], dtype=float)
        run_start, run_end = run_bounds[(method, replicate)]
        rows.append(
            {
                "method": method,
                "base_method": base_method_name(method),
                "replicate": int(replicate),
                "worker_id": worker_id,
                "n_attempts": int(len(group)),
                "total_busy_s": float(np.sum(ends - starts)),
                "min_start": float(starts.min()),
                "max_end": float(ends.max()),
                "elapsed_wall_s": float(run_end - run_start),
                "active_span_s": float(ends.max() - starts.min()),
                "record_kind": group[0].record_kind or "",
                "time_semantics": group[0].time_semantics or "",
            }
        )
    return pd.DataFrame(rows)


def write_runtime_debug_summary(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    stem_name: str = "runtime_debug_summary",
) -> None:
    debug_df = _runtime_debug_frame(records)
    csv_path = output_dir.data / f"{stem_name}.csv"
    debug_df.to_csv(csv_path, index=False)
    methods = sorted(debug_df["base_method"].dropna().unique().tolist()) if not debug_df.empty else []
    metadata = _nonbenchmark_plot_metadata(
        output_dir,
        plot_name=stem_name,
        title="Runtime debug summary",
        methods=methods,
        diagnostic_plot=True,
        skip_reason="empty_summary" if debug_df.empty else None,
        extra={"csv": str(csv_path)},
    )
    write_plot_metadata(output_dir.data / stem_name, metadata=metadata)


def _ci_half_width(values: np.ndarray, ci_level: float) -> float:
    """Return a t-based confidence half-width for finite *values*."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    from scipy.stats import t

    sem = float(np.std(finite, ddof=1) / np.sqrt(finite.size))
    if not np.isfinite(sem):
        return float("nan")
    return float(t.ppf(0.5 + 0.5 * float(ci_level), finite.size - 1) * sem)


def _summarize_scalar(values: np.ndarray, ci_level: float) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    half_width = _ci_half_width(finite, ci_level=ci_level)
    if not np.isfinite(half_width):
        return mean, float("nan"), float("nan")
    return mean, mean - half_width, mean + half_width


def _step_curve_summary(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    ci_level: float,
    log_y: bool = False,
    lower_bound: float | None = None,
) -> pd.DataFrame:
    """Aggregate per-replicate step curves to a mean + CI summary."""
    rows: list[dict[str, object]] = []
    for method, method_group in frame.groupby("method", sort=True):
        x_grid = np.array(sorted(method_group[x_col].dropna().unique().tolist()), dtype=float)
        if x_grid.size == 0:
            continue
        replicate_values: list[np.ndarray] = []
        replicate_ids: list[int] = []
        for replicate, rep_group in method_group.groupby("replicate", sort=True):
            rep_group = rep_group.sort_values(x_col)
            xs = rep_group[x_col].to_numpy(dtype=float)
            ys = rep_group[y_col].to_numpy(dtype=float)
            if xs.size == 0:
                continue
            aligned = np.full(x_grid.shape, np.nan, dtype=float)
            idx = np.searchsorted(xs, x_grid, side="right") - 1
            valid = idx >= 0
            aligned[valid] = ys[idx[valid]]
            if log_y:
                positive = aligned > 0
                aligned[~positive] = np.nan
                aligned[positive] = np.log10(aligned[positive])
            replicate_values.append(aligned)
            replicate_ids.append(int(replicate))
        if not replicate_values:
            continue
        stack = np.vstack(replicate_values)
        for point_idx, x_val in enumerate(x_grid):
            point = stack[:, point_idx]
            finite = point[np.isfinite(point)]
            if finite.size == 0:
                continue
            mean, ci_low, ci_high = _summarize_scalar(finite, ci_level=ci_level)
            if log_y:
                mean = 10 ** mean
                ci_low = 10 ** ci_low if np.isfinite(ci_low) else float("nan")
                ci_high = 10 ** ci_high if np.isfinite(ci_high) else float("nan")
            if lower_bound is not None:
                mean = max(float(lower_bound), float(mean))
                if np.isfinite(ci_low):
                    ci_low = max(float(lower_bound), float(ci_low))
                if np.isfinite(ci_high):
                    ci_high = max(float(lower_bound), float(ci_high))
            rows.append(
                {
                    "method": method,
                    x_col: float(x_val),
                    y_col: float(mean),
                    f"{y_col}_ci_low": float(ci_low),
                    f"{y_col}_ci_high": float(ci_high),
                    "n_replicates": int(finite.size),
                }
            )
    return pd.DataFrame(rows)


def _write_benchmark_audit(
    records: List[ParticleRecord],
    *,
    true_params: Dict[str, float],
    output_dir: OutputDir,
    archive_size: int | None,
    min_particles_for_threshold: int,
) -> pd.DataFrame:
    """Write benchmark audit CSV/JSON and return the dataframe."""
    import json

    audit_df = benchmark_plot_audit(
        records,
        true_params=true_params,
        archive_size=archive_size,
        min_particles_for_threshold=min_particles_for_threshold,
    )
    csv_path = output_dir.data / "plot_audit.csv"
    audit_df.to_csv(csv_path, index=False)
    summary = {
        "rows": int(len(audit_df)),
        "invalid_quality_rows": int((~audit_df["paper_quality_plots_allowed"]).sum()) if not audit_df.empty else 0,
        "invalid_threshold_rows": int((~audit_df["paper_threshold_plots_allowed"]).sum()) if not audit_df.empty else 0,
        "source_raw_files": [str(output_dir.data / "raw_results.csv")],
    }
    with open(output_dir.data / "plot_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return audit_df


def _write_lotka_tol_init_diagnostic(
    records: List[ParticleRecord],
    *,
    cfg: Dict[str, Any],
    output_dir: OutputDir,
) -> None:
    """Write Lotka-specific fallback diagnostics and tol_init recommendation."""
    import json

    diagnostic_df, summary = lotka_tol_init_diagnostic(records)
    diagnostic_df.to_csv(output_dir.data / "lotka_tol_init_diagnostic.csv", index=False)
    summary_payload = {
        **summary,
        "current_tol_init": float(cfg.get("inference", {}).get("tol_init", float("nan"))),
    }
    with open(output_dir.data / "lotka_tol_init_diagnostic.json", "w") as f:
        json.dump(summary_payload, f, indent=2)


def _paper_plot_allowed(audit_df: pd.DataFrame, column: str) -> bool:
    if audit_df.empty or column not in audit_df.columns:
        return True
    return bool(audit_df[column].all())


def _audit_skip_metadata(
    *,
    plot_name: str,
    audit_df: pd.DataFrame,
    gate_column: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    invalid = audit_df.loc[~audit_df[gate_column]].copy() if not audit_df.empty and gate_column in audit_df.columns else pd.DataFrame()
    metadata: Dict[str, Any] = {
        "plot_name": plot_name,
        "skip_reason": "plot_audit_failed",
        "audit_gate": gate_column,
    }
    if not invalid.empty:
        metadata["invalid_rows"] = invalid.to_dict(orient="records")
    if extra:
        metadata.update(extra)
    return metadata


def plot_posterior(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
) -> None:
    """Posterior histogram for each parameter from the final population."""
    final = _final_population(records) if archive_size is None else final_state_records(records, archive_size=archive_size)
    if not final:
        return
    state_results = final_state_results(records, archive_size=archive_size)
    methods = sorted({r.method for r in final})
    sample_counts = {
        result.method: result.n_particles_used
        for result in state_results
    }
    for param in _param_names(final):
        if len(methods) <= 1:
            samples = np.array([r.params[param] for r in final], dtype=float)
            posterior_plot(samples, param_name=param, path_stem=output_dir.plots / f"posterior_{param}")
        else:
            method_samples = {
                m: np.array([r.params[param] for r in final if r.method == m], dtype=float)
                for m in methods
            }
            save_paths = posterior_comparison_plot(
                method_samples, param_name=param, path_stem=output_dir.plots / f"posterior_{param}",
            )
            meta_path = save_paths.get("meta")
            if meta_path is not None and meta_path.exists():
                import json

                meta = json.loads(meta_path.read_text())
                meta["sample_counts"] = sample_counts
                meta_path.write_text(json.dumps(meta, indent=2))


def _archive_evolution_frame(records: List[ParticleRecord]) -> pd.DataFrame:
    """Tolerance versus simulation attempts, one row per observable state."""
    rows = []
    seen = set()
    for record in records:
        if record.tolerance is None:
            continue
        attempt_count = int(record.attempt_count) if record.attempt_count is not None else int(record.step)
        key = (record.method, int(record.replicate), attempt_count, float(record.tolerance))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "method": record.method,
                "replicate": int(record.replicate),
                "attempt_count": attempt_count,
                "tolerance": float(record.tolerance),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["method", "replicate", "attempt_count", "tolerance"])
    return pd.DataFrame(rows).sort_values(["method", "replicate", "attempt_count"]).reset_index(drop=True)


def plot_archive_evolution(records: List[ParticleRecord], output_dir: OutputDir, *, ci_level: float = 0.95) -> None:
    """Paper summary: tolerance versus simulation attempts with mean + CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectory_df = _archive_evolution_frame(records)
    if trajectory_df.empty:
        return

    summary_df = _step_curve_summary(
        trajectory_df,
        x_col="attempt_count",
        y_col="tolerance",
        ci_level=ci_level,
        log_y=True,
    )
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in summary_df.groupby("method", sort=True):
        group = group.sort_values("attempt_count")
        ax.plot(group["attempt_count"], group["tolerance"], linewidth=1.8, label=method)
        valid_ci = group["tolerance_ci_low"].notna() & group["tolerance_ci_high"].notna() & (group["n_replicates"] >= 2)
        if valid_ci.any():
            ax.fill_between(
                group.loc[valid_ci, "attempt_count"],
                group.loc[valid_ci, "tolerance_ci_low"],
                group.loc[valid_ci, "tolerance_ci_high"],
                alpha=0.2,
            )
    ax.set_xlabel("simulation attempts")
    ax.set_ylabel("tolerance ε")
    ax.set_title("Tolerance vs. simulation attempts")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "archive_evolution",
        data={col: summary_df[col].tolist() for col in summary_df.columns},
        metadata={"plot_name": "tolerance_vs_simulations", "summary_plot": True, "ci_level": float(ci_level)},
    )


def plot_archive_evolution_diagnostic(records: List[ParticleRecord], output_dir: OutputDir) -> None:
    """Diagnostic replicate-level tolerance versus attempts plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectory_df = _archive_evolution_frame(records)
    if trajectory_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for (method, replicate), group in trajectory_df.groupby(["method", "replicate"], sort=True):
        label = method if trajectory_df["replicate"].nunique() == 1 else f"{method} (rep {replicate})"
        ax.plot(group["attempt_count"], group["tolerance"], marker="o", linewidth=1.2, label=label)
    ax.set_xlabel("simulation attempts")
    ax.set_ylabel("tolerance ε")
    ax.set_title("Tolerance vs. simulation attempts")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "archive_evolution_diagnostic",
        data={col: trajectory_df[col].tolist() for col in trajectory_df.columns},
        metadata={"plot_name": "tolerance_vs_simulations_diagnostic", "diagnostic_plot": True},
    )


def plot_worker_gantt(records: List[ParticleRecord], output_dir: OutputDir) -> None:
    """Worker timeline from simulation start/end metadata."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timed = [record for record in _attempt_timing_records(records) if record.worker_id is not None]
    if not timed:
        return

    def _condition_name(method: str) -> str:
        return method.split("__", 1)[1] if "__" in method else "all"

    included_methods = sorted({r.method for r in timed})
    base_methods = sorted({base_method_name(r.method) for r in timed})
    conditions = sorted({_condition_name(r.method) for r in timed})
    show_base_methods = len(base_methods) > 1
    show_conditions = len(conditions) > 1
    if show_base_methods and show_conditions:
        title = "Worker timeline by method and condition"
    elif show_base_methods:
        title = "Worker timeline by method"
    elif show_conditions:
        title = "Worker timeline by condition"
    else:
        title = "Worker timeline"

    if len(base_methods) <= 1 and len(conditions) <= 1:
        fig = gantt_plot(timed)
        if fig.axes:
            fig.axes[0].set_title(title)
    else:
        fig, axes = plt.subplots(
            len(base_methods),
            len(conditions),
            figsize=(max(7.5, 5.2 * len(conditions)), max(3.5, 2.8 * len(base_methods))),
            squeeze=False,
        )
        for row_idx, base_method in enumerate(base_methods):
            for col_idx, condition in enumerate(conditions):
                ax = axes[row_idx, col_idx]
                panel_records = [
                    r for r in timed
                    if base_method_name(r.method) == base_method and _condition_name(r.method) == condition
                ]
                if not panel_records:
                    ax.set_axis_off()
                    continue
                gantt_plot(panel_records, ax=ax)
                legend = ax.get_legend()
                if legend is not None and len({r.method for r in panel_records}) <= 1:
                    legend.remove()
                title_parts = []
                if show_base_methods:
                    title_parts.append(base_method)
                if show_conditions:
                    title_parts.append("all conditions" if condition == "all" else condition.replace("_", " "))
                ax.set_title(" | ".join(title_parts) if title_parts else title)
        fig.tight_layout()
    omitted_methods = sorted({r.method for r in records if r.method not in {t.method for t in timed}})
    show_replicate = len({int(r.replicate) for r in timed}) > 1
    data = {
        "method": [r.method for r in timed],
        "base_method": [base_method_name(r.method) for r in timed],
        "condition": [_condition_name(r.method) for r in timed],
        "replicate": [r.replicate for r in timed],
        "worker_id": [r.worker_id for r in timed],
        "lane_label": [
            f"rep {int(r.replicate)} | worker {r.worker_id}" if show_replicate else f"worker {r.worker_id}"
            for r in timed
        ],
        "sim_start_time": [r.sim_start_time for r in timed],
        "sim_end_time": [r.sim_end_time for r in timed],
        "generation": [r.generation for r in timed],
    }
    save_paths = save_figure(
        fig,
        output_dir.plots / "worker_gantt",
        data=data,
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name="worker_gantt",
            title=title,
            methods=sorted({base_method_name(method) for method in included_methods}),
            diagnostic_plot=True,
            extra={
                "included_methods": included_methods,
                "omitted_methods": omitted_methods,
                "base_methods": base_methods,
                "conditions": conditions,
            },
        ),
    )


def _compute_idle_fraction(records: List[ParticleRecord]) -> Dict[str, Dict]:
    """Compute per-method, per-replicate idle fractions from timing data.

    Returns ``{method: {replicate: idle_fraction}}``."""
    timed = [r for r in _attempt_timing_records(records) if r.worker_id is not None]
    if not timed:
        return {}

    by_method_rep: Dict[str, Dict[int, List[ParticleRecord]]] = {}
    for r in timed:
        by_method_rep.setdefault(r.method, {}).setdefault(r.replicate, []).append(r)

    result: Dict[str, Dict] = {}
    for method, by_rep in by_method_rep.items():
        result[method] = {}
        for rep, recs in by_rep.items():
            workers = {r.worker_id for r in recs}
            n_workers = len(workers)
            span = max(r.sim_end_time for r in recs) - min(r.sim_start_time for r in recs)
            if span <= 0 or n_workers == 0:
                result[method][rep] = float("nan")
                continue
            total_busy = sum(
                float(r.sim_end_time) - float(r.sim_start_time) for r in recs
            )
            result[method][rep] = 1.0 - total_busy / (n_workers * span)
    return result


def plot_idle_fraction(records: List[ParticleRecord], output_dir: OutputDir) -> None:
    """Bar chart of utilization-loss fraction per method/sigma level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_df = _runtime_utilization_rows(records)
    if summary_df.empty:
        return
    mean_rows: list[dict[str, object]] = []
    for (sigma, base_method, measurement_method), group in summary_df.groupby(
        ["sigma", "base_method", "measurement_method"], sort=True
    ):
        mean, ci_low, ci_high = _summarize_scalar(
            group["utilization_loss_fraction"].to_numpy(dtype=float),
            ci_level=0.95,
        )
        mean_rows.append(
            {
                "sigma": float(sigma),
                "base_method": base_method,
                "measurement_method": measurement_method,
                "utilization_loss_fraction": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_replicates": int(group["replicate"].nunique()),
            }
        )
    mean_df = pd.DataFrame(mean_rows)
    if mean_df.empty:
        return
    for col in ("utilization_loss_fraction", "ci_low", "ci_high"):
        mean_df[col] = mean_df[col].clip(lower=0.0, upper=1.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    series = list(
        dict.fromkeys(
            (row["base_method"], row["measurement_method"])
            for _, row in mean_df.iterrows()
        )
    )
    sigmas = sorted(mean_df["sigma"].unique().tolist())
    x = np.arange(len(sigmas), dtype=float)
    width = 0.8 / max(1, len(series))
    for idx, (method, measurement_method) in enumerate(series):
        subset = mean_df[
            (mean_df["base_method"] == method)
            & (mean_df["measurement_method"] == measurement_method)
        ]
        subset = subset.sort_values("sigma")
        sigma_to_row = {float(row["sigma"]): row for _, row in subset.iterrows()}
        y = np.array(
            [
                float(sigma_to_row[sigma]["utilization_loss_fraction"]) if sigma in sigma_to_row else np.nan
                for sigma in sigmas
            ],
            dtype=float,
        )
        yerr = np.array(
            [
                float(sigma_to_row[sigma]["ci_high"] - sigma_to_row[sigma]["utilization_loss_fraction"])
                if sigma in sigma_to_row and np.isfinite(float(sigma_to_row[sigma]["ci_high"]))
                else np.nan
                for sigma in sigmas
            ],
            dtype=float,
        )
        offset = (idx - (len(series) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            y,
            width=width,
            alpha=0.8,
            label=f"{method} ({measurement_method})",
        )
        finite_err = np.isfinite(yerr)
        if finite_err.any():
            ax.errorbar(
                (x + offset)[finite_err],
                y[finite_err],
                yerr=yerr[finite_err],
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.0,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ={sigma}" for sigma in sigmas])
    ax.set_xlabel("heterogeneity (σ)")
    ax.set_ylabel("utilization loss fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Utilization loss by heterogeneity and method")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "idle_fraction",
        data={col: mean_df[col].tolist() for col in mean_df.columns},
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name="idle_fraction",
            title="Utilization loss by heterogeneity and method",
            methods=sorted(mean_df["base_method"].dropna().unique().tolist()),
            summary_plot=True,
        ),
    )


def plot_throughput_over_time(
    records: List[ParticleRecord], output_dir: OutputDir, n_bins: int = 20,
) -> None:
    """Throughput over wall-clock time, faceted by heterogeneity condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timed = _attempt_timing_records(records)
    if not timed:
        return

    rows: list[dict[str, object]] = []
    for record in timed:
        method = str(record.method)
        sigma = 0.0
        if "__sigma" in method:
            try:
                sigma = float(method.split("__sigma", 1)[1])
            except ValueError:
                sigma = 0.0
        rows.append(
            {
                "sigma": sigma,
                "base_method": base_method_name(method),
                "replicate": int(record.replicate),
                "sim_start_time": float(record.sim_start_time),
                "sim_end_time": float(record.sim_end_time),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return

    frame["relative_end_time"] = 0.0
    for _, group in frame.groupby(["sigma", "base_method", "replicate"], sort=False):
        frame.loc[group.index, "relative_end_time"] = (
            group["sim_end_time"] - float(group["sim_start_time"].min())
        )

    sigmas = sorted(frame["sigma"].dropna().unique().tolist())
    fig, axes = plt.subplots(
        len(sigmas),
        1,
        figsize=(7.0, max(3.5, 2.8 * len(sigmas))),
        squeeze=False,
    )
    summary_rows: list[dict[str, object]] = []
    cmap = plt.get_cmap("tab10")
    plotted = False
    for sigma_idx, sigma in enumerate(sigmas):
        ax = axes[sigma_idx, 0]
        sigma_frame = frame.loc[frame["sigma"] == sigma].copy()
        max_time = float(sigma_frame["relative_end_time"].max())
        if not np.isfinite(max_time) or max_time <= 0:
            continue
        edges = np.linspace(0.0, max_time, n_bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
        for method_idx, base_method in enumerate(sorted(sigma_frame["base_method"].unique().tolist())):
            method_frame = sigma_frame.loc[sigma_frame["base_method"] == base_method]
            replicate_curves: list[np.ndarray] = []
            for _, rep_group in method_frame.groupby("replicate", sort=True):
                counts, _ = np.histogram(rep_group["relative_end_time"].to_numpy(dtype=float), bins=edges)
                replicate_curves.append(counts.astype(float) / max(bin_width, 1e-12))
            if not replicate_curves:
                continue
            stack = np.vstack(replicate_curves)
            mean_curve = np.mean(stack, axis=0)
            ax.plot(mids, mean_curve, linewidth=1.8, label=base_method, color=cmap(method_idx % 10))
            plotted = True
            for x_val, y_val in zip(mids, mean_curve):
                summary_rows.append(
                    {
                        "sigma": float(sigma),
                        "base_method": base_method,
                        "bin_mid": float(x_val),
                        "throughput_sims_per_s": float(y_val),
                        "n_replicates": int(stack.shape[0]),
                    }
                )
        ax.set_xlabel("wall-clock time since run start")
        ax.set_ylabel("throughput (sim/s)")
        ax.set_title(f"Throughput over time: σ={sigma:g}")
        ax.legend(frameon=False, fontsize="small")

    if not plotted:
        return

    fig.tight_layout()
    summary_df = pd.DataFrame(summary_rows)
    save_figure(
        fig,
        output_dir.plots / "throughput_over_time",
        data={col: summary_df[col].tolist() for col in summary_df.columns},
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name="throughput_over_time",
            title="Throughput over time by heterogeneity and method",
            methods=sorted(summary_df["base_method"].dropna().unique().tolist()),
            summary_plot=True,
            extra={"facet_by": "sigma"},
        ),
    )


def plot_idle_fraction_comparison(records: List[ParticleRecord], output_dir: OutputDir) -> None:
    """Utilization-loss fraction vs. sigma with per-replicate scatter and mean line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_df = _runtime_utilization_rows(records)
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_rows: list[dict[str, object]] = []
    for (sigma, base_method, measurement_method), group in summary_df.groupby(
        ["sigma", "base_method", "measurement_method"], sort=True
    ):
        mean, ci_low, ci_high = _summarize_scalar(
            group["utilization_loss_fraction"].to_numpy(dtype=float),
            ci_level=0.95,
        )
        mean_rows.append(
            {
                "sigma": float(sigma),
                "base_method": base_method,
                "measurement_method": measurement_method,
                "utilization_loss_fraction": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    mean_df = pd.DataFrame(mean_rows)
    if mean_df.empty:
        return
    for col in ("utilization_loss_fraction", "ci_low", "ci_high"):
        mean_df[col] = mean_df[col].clip(lower=0.0, upper=1.0)
    for (method, measurement_method), group in summary_df.groupby(["base_method", "measurement_method"], sort=True):
        ax.scatter(
            group["sigma"].to_numpy(dtype=float),
            group["utilization_loss_fraction"].to_numpy(dtype=float),
            s=22,
            alpha=0.35,
            label=f"{method} {measurement_method} samples",
        )
        mean_group = mean_df[
            (mean_df["base_method"] == method)
            & (mean_df["measurement_method"] == measurement_method)
        ].sort_values("sigma")
        ax.plot(
            mean_group["sigma"].to_numpy(dtype=float),
            mean_group["utilization_loss_fraction"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            label=f"{method} ({measurement_method})",
        )
        valid_ci = mean_group["ci_low"].notna() & mean_group["ci_high"].notna()
        if valid_ci.any():
            ax.fill_between(
                mean_group.loc[valid_ci, "sigma"].to_numpy(dtype=float),
                mean_group.loc[valid_ci, "ci_low"].to_numpy(dtype=float),
                mean_group.loc[valid_ci, "ci_high"].to_numpy(dtype=float),
                alpha=0.15,
            )
    ax.set_xlabel("heterogeneity (σ)")
    ax.set_ylabel("utilization loss fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Utilization loss vs. heterogeneity and method")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "idle_fraction_comparison",
        data={col: summary_df[col].tolist() for col in summary_df.columns},
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name="idle_fraction_comparison",
            title="Utilization loss vs. heterogeneity and method",
            methods=sorted(summary_df["base_method"].dropna().unique().tolist()),
            summary_plot=True,
        ),
    )


def plot_quality_vs_wall_time_diagnostic(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
    emit_legacy_alias: bool = False,
    audit_df: pd.DataFrame | None = None,
) -> None:
    """Appendix-style posterior quality diagnostic over wall-clock time."""
    from ..analysis import posterior_quality_curve

    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="wall_time",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if quality_df.empty:
        write_plot_metadata(
            output_dir.plots / "quality_vs_wall_time_diagnostic",
            metadata={
                "plot_name": "quality_vs_wall_time_diagnostic",
                "skip_reason": "missing_true_params_or_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    metadata = {
        "plot_name": "quality_vs_wall_time_diagnostic",
        "axis_kind": "wall_time",
        "state_kind": "observable_posterior_state",
        "checkpoint_strategy": "quantile",
        "source_raw_files": [str(output_dir.data / "raw_results.csv")],
    }
    _save_quality_curve_artifact(
        output_dir.plots / "quality_vs_wall_time_diagnostic",
        quality_df,
        axis_kind="wall_time",
        metadata=metadata,
    )
    if emit_legacy_alias:
        _save_quality_curve_artifact(
            output_dir.plots / "quality_vs_time",
            quality_df,
            axis_kind="wall_time",
            metadata={**metadata, "plot_name": "quality_vs_time", "deprecated_alias_for": "quality_vs_wall_time_diagnostic"},
        )


def plot_quality_vs_wall_time(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
    ci_level: float = 0.95,
    audit_df: pd.DataFrame | None = None,
) -> None:
    """Paper summary: posterior quality over wall-clock time."""
    from ..analysis import posterior_quality_curve

    audit_df = audit_df if audit_df is not None else pd.DataFrame()
    if true_params and not _paper_plot_allowed(audit_df, "paper_quality_plots_allowed"):
        write_plot_metadata(
            output_dir.plots / "quality_vs_wall_time",
            metadata=_audit_skip_metadata(
                plot_name="quality_vs_wall_time",
                audit_df=audit_df,
                gate_column="paper_quality_plots_allowed",
                extra={"source_raw_files": [str(output_dir.data / "raw_results.csv")]},
            ),
        )
        return

    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="wall_time",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if quality_df.empty:
        write_plot_metadata(
            output_dir.plots / "quality_vs_wall_time",
            metadata={
                "plot_name": "quality_vs_wall_time",
                "skip_reason": "missing_true_params_or_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return
    _save_quality_summary_artifact(
        output_dir.plots / "quality_vs_wall_time",
        quality_df,
        axis_kind="wall_time",
        ci_level=ci_level,
        metadata={
            "plot_name": "quality_vs_wall_time",
            "axis_kind": "wall_time",
            "summary_plot": True,
            "ci_level": float(ci_level),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_quality_vs_posterior_samples(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
    ci_level: float = 0.95,
    audit_df: pd.DataFrame | None = None,
) -> None:
    """Paper summary: convergence over posterior sample count."""
    from ..analysis import posterior_quality_curve

    audit_df = audit_df if audit_df is not None else pd.DataFrame()
    if true_params and not _paper_plot_allowed(audit_df, "paper_quality_plots_allowed"):
        write_plot_metadata(
            output_dir.plots / "quality_vs_posterior_samples",
            metadata=_audit_skip_metadata(
                plot_name="quality_vs_posterior_samples",
                audit_df=audit_df,
                gate_column="paper_quality_plots_allowed",
                extra={"source_raw_files": [str(output_dir.data / "raw_results.csv")]},
            ),
        )
        return

    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="posterior_samples",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if quality_df.empty:
        write_plot_metadata(
            output_dir.plots / "quality_vs_posterior_samples",
            metadata={
                "plot_name": "quality_vs_posterior_samples",
                "skip_reason": "missing_true_params_or_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    _save_quality_summary_artifact(
        output_dir.plots / "quality_vs_posterior_samples",
        quality_df,
        axis_kind="posterior_samples",
        ci_level=ci_level,
        metadata={
            "plot_name": "quality_vs_posterior_samples",
            "axis_kind": "posterior_samples",
            "summary_plot": True,
            "ci_level": float(ci_level),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_quality_vs_posterior_samples_diagnostic(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
) -> None:
    """Diagnostic replicate-level convergence over posterior sample count."""
    from ..analysis import posterior_quality_curve

    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="posterior_samples",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if quality_df.empty:
        write_plot_metadata(
            output_dir.plots / "quality_vs_posterior_samples_diagnostic",
            metadata={
                "plot_name": "quality_vs_posterior_samples_diagnostic",
                "skip_reason": "missing_true_params_or_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return
    _save_quality_curve_artifact(
        output_dir.plots / "quality_vs_posterior_samples_diagnostic",
        quality_df,
        axis_kind="posterior_samples",
        metadata={
            "plot_name": "quality_vs_posterior_samples_diagnostic",
            "axis_kind": "posterior_samples",
            "diagnostic_plot": True,
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_quality_vs_attempt_budget(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
    ci_level: float = 0.95,
    audit_df: pd.DataFrame | None = None,
) -> None:
    """Paper summary: convergence over cumulative attempt budget."""
    from ..analysis import posterior_quality_curve

    audit_df = audit_df if audit_df is not None else pd.DataFrame()
    if true_params and not _paper_plot_allowed(audit_df, "paper_quality_plots_allowed"):
        write_plot_metadata(
            output_dir.plots / "quality_vs_attempt_budget",
            metadata=_audit_skip_metadata(
                plot_name="quality_vs_attempt_budget",
                audit_df=audit_df,
                gate_column="paper_quality_plots_allowed",
                extra={"source_raw_files": [str(output_dir.data / "raw_results.csv")]},
            ),
        )
        return

    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="attempt_budget",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if quality_df.empty:
        write_plot_metadata(
            output_dir.plots / "quality_vs_attempt_budget",
            metadata={
                "plot_name": "quality_vs_attempt_budget",
                "skip_reason": "missing_true_params_or_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    _save_quality_summary_artifact(
        output_dir.plots / "quality_vs_attempt_budget",
        quality_df,
        axis_kind="attempt_budget",
        ci_level=ci_level,
        metadata={
            "plot_name": "quality_vs_attempt_budget",
            "axis_kind": "attempt_budget",
            "summary_plot": True,
            "ci_level": float(ci_level),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_quality_vs_attempt_budget_diagnostic(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
) -> None:
    """Diagnostic replicate-level convergence over cumulative attempt budget."""
    from ..analysis import posterior_quality_curve

    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="attempt_budget",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if quality_df.empty:
        write_plot_metadata(
            output_dir.plots / "quality_vs_attempt_budget_diagnostic",
            metadata={
                "plot_name": "quality_vs_attempt_budget_diagnostic",
                "skip_reason": "missing_true_params_or_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return
    _save_quality_curve_artifact(
        output_dir.plots / "quality_vs_attempt_budget_diagnostic",
        quality_df,
        axis_kind="attempt_budget",
        metadata={
            "plot_name": "quality_vs_attempt_budget_diagnostic",
            "axis_kind": "attempt_budget",
            "diagnostic_plot": True,
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_time_to_target_summary(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    target_wasserstein: float,
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    ci_level: float = 0.95,
    min_particles_for_threshold: int = 1,
    audit_df: pd.DataFrame | None = None,
) -> None:
    """Main paper summary: wall-clock time required to reach a target quality."""
    from ..analysis import time_to_threshold

    audit_df = audit_df if audit_df is not None else pd.DataFrame()
    if true_params and not _paper_plot_allowed(audit_df, "paper_threshold_plots_allowed"):
        write_plot_metadata(
            output_dir.plots / "time_to_target_summary",
            metadata=_audit_skip_metadata(
                plot_name="time_to_target_summary",
                audit_df=audit_df,
                gate_column="paper_threshold_plots_allowed",
                extra={
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                    "target_wasserstein": float(target_wasserstein),
                },
            ),
        )
        return

    summary_df = time_to_threshold(
        records,
        true_params=true_params,
        target_wasserstein=target_wasserstein,
        axis_kind="wall_time",
        archive_size=archive_size,
        min_particles=int(min_particles_for_threshold),
    )
    if summary_df.empty:
        write_plot_metadata(
            output_dir.plots / "time_to_target_summary",
            metadata={
                "plot_name": "time_to_target_summary",
                "skip_reason": "missing_true_params_or_threshold_rows",
                "target_wasserstein": float(target_wasserstein),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    _save_threshold_summary_artifact(
        output_dir.plots / "time_to_target_summary",
        summary_df,
        axis_kind="wall_time",
        include_replicates=False,
        metadata={
            "plot_name": "time_to_target_summary",
            "axis_kind": "wall_time",
            "target_wasserstein": float(target_wasserstein),
            "summary_plot": True,
            "ci_level": float(ci_level),
            "min_particles_for_threshold": int(min_particles_for_threshold),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_time_to_target_diagnostic(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    target_wasserstein: float,
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    min_particles_for_threshold: int = 1,
) -> None:
    """Diagnostic replicate-level threshold summary over wall-clock time."""
    from ..analysis import time_to_threshold

    summary_df = time_to_threshold(
        records,
        true_params=true_params,
        target_wasserstein=target_wasserstein,
        axis_kind="wall_time",
        archive_size=archive_size,
        min_particles=int(min_particles_for_threshold),
    )
    if summary_df.empty:
        write_plot_metadata(
            output_dir.plots / "time_to_target_diagnostic",
            metadata={
                "plot_name": "time_to_target_diagnostic",
                "skip_reason": "missing_true_params_or_threshold_rows",
                "target_wasserstein": float(target_wasserstein),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return
    _save_threshold_summary_artifact(
        output_dir.plots / "time_to_target_diagnostic",
        summary_df,
        axis_kind="wall_time",
        include_replicates=True,
        metadata={
            "plot_name": "time_to_target_diagnostic",
            "axis_kind": "wall_time",
            "target_wasserstein": float(target_wasserstein),
            "diagnostic_plot": True,
            "min_particles_for_threshold": int(min_particles_for_threshold),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_attempts_to_target_summary(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    target_wasserstein: float,
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    ci_level: float = 0.95,
    min_particles_for_threshold: int = 1,
    audit_df: pd.DataFrame | None = None,
) -> None:
    """Alternative summary: simulation attempts required to reach target quality."""
    from ..analysis import time_to_threshold

    audit_df = audit_df if audit_df is not None else pd.DataFrame()
    if true_params and not _paper_plot_allowed(audit_df, "paper_threshold_plots_allowed"):
        write_plot_metadata(
            output_dir.plots / "attempts_to_target_summary",
            metadata=_audit_skip_metadata(
                plot_name="attempts_to_target_summary",
                audit_df=audit_df,
                gate_column="paper_threshold_plots_allowed",
                extra={
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                    "target_wasserstein": float(target_wasserstein),
                },
            ),
        )
        return

    summary_df = time_to_threshold(
        records,
        true_params=true_params,
        target_wasserstein=target_wasserstein,
        axis_kind="attempt_budget",
        archive_size=archive_size,
        min_particles=int(min_particles_for_threshold),
    )
    if summary_df.empty:
        write_plot_metadata(
            output_dir.plots / "attempts_to_target_summary",
            metadata={
                "plot_name": "attempts_to_target_summary",
                "skip_reason": "missing_true_params_or_threshold_rows",
                "target_wasserstein": float(target_wasserstein),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    _save_threshold_summary_artifact(
        output_dir.plots / "attempts_to_target_summary",
        summary_df,
        axis_kind="attempt_budget",
        include_replicates=False,
        metadata={
            "plot_name": "attempts_to_target_summary",
            "axis_kind": "attempt_budget",
            "target_wasserstein": float(target_wasserstein),
            "summary_plot": True,
            "ci_level": float(ci_level),
            "min_particles_for_threshold": int(min_particles_for_threshold),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_attempts_to_target_diagnostic(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    target_wasserstein: float,
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
    min_particles_for_threshold: int = 1,
) -> None:
    """Diagnostic replicate-level attempt-threshold summary."""
    from ..analysis import time_to_threshold

    summary_df = time_to_threshold(
        records,
        true_params=true_params,
        target_wasserstein=target_wasserstein,
        axis_kind="attempt_budget",
        archive_size=archive_size,
        min_particles=int(min_particles_for_threshold),
    )
    if summary_df.empty:
        write_plot_metadata(
            output_dir.plots / "attempts_to_target_diagnostic",
            metadata={
                "plot_name": "attempts_to_target_diagnostic",
                "skip_reason": "missing_true_params_or_threshold_rows",
                "target_wasserstein": float(target_wasserstein),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return
    _save_threshold_summary_artifact(
        output_dir.plots / "attempts_to_target_diagnostic",
        summary_df,
        axis_kind="attempt_budget",
        include_replicates=True,
        metadata={
            "plot_name": "attempts_to_target_diagnostic",
            "axis_kind": "attempt_budget",
            "target_wasserstein": float(target_wasserstein),
            "diagnostic_plot": True,
            "min_particles_for_threshold": int(min_particles_for_threshold),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_quality_vs_time(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    checkpoint_steps: List[int],
    output_dir: OutputDir,
    *,
    archive_size: int | None = None,
) -> None:
    """Deprecated wrapper that now emits the appendix wall-time diagnostic alias."""
    if not true_params:
        return
    checkpoint_count = len(checkpoint_steps) if checkpoint_steps else 8
    plot_quality_vs_wall_time_diagnostic(
        records,
        true_params=true_params,
        output_dir=output_dir,
        archive_size=archive_size,
        checkpoint_count=checkpoint_count,
        emit_legacy_alias=True,
    )


def plot_corner(
    records: List[ParticleRecord],
    param_names: List[str],
    output_dir: OutputDir,
    true_params: Optional[Dict[str, float]] = None,
    *,
    archive_size: int | None = None,
) -> None:
    """Corner plot for the final population."""
    final = final_state_records(records, archive_size=archive_size)
    if not final or not param_names:
        return

    methods = sorted({r.method for r in final})
    fig = corner_plot(
        final, param_names=param_names, true_params=true_params,
        method_labels=methods if len(methods) > 1 else None,
    )
    data: Dict[str, list] = {"method": [r.method for r in final]}
    for name in param_names:
        data[name] = [r.params.get(name) for r in final]
    save_figure(
        fig,
        output_dir.plots / "corner",
        data=data,
        metadata={
            "sample_counts": {
                result.method: result.n_particles_used
                for result in final_state_results(records, archive_size=archive_size)
            },
        },
    )


def plot_tolerance_trajectory(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    ci_level: float = 0.95,
) -> None:
    """Paper summary: tolerance schedule over wall-clock time."""
    from ..analysis import tolerance_over_wall_time

    trajectory_df = tolerance_over_wall_time(records)
    if trajectory_df.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_df = _step_curve_summary(
        trajectory_df,
        x_col="wall_time",
        y_col="tolerance",
        ci_level=ci_level,
        log_y=True,
    )
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in summary_df.groupby("method", sort=True):
        group = group.sort_values("wall_time")
        ax.plot(group["wall_time"], group["tolerance"], linewidth=1.8, label=method)
        valid_ci = group["tolerance_ci_low"].notna() & group["tolerance_ci_high"].notna() & (group["n_replicates"] >= 2)
        if valid_ci.any():
            ax.fill_between(
                group.loc[valid_ci, "wall_time"],
                group.loc[valid_ci, "tolerance_ci_low"],
                group.loc[valid_ci, "tolerance_ci_high"],
                alpha=0.2,
            )
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("tolerance")
    ax.set_yscale("log")
    ax.set_title("Tolerance trajectory")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    missing_methods = sorted({record.method for record in records if record.tolerance is None})
    save_figure(
        fig,
        output_dir.plots / "tolerance_trajectory",
        data={col: summary_df[col].tolist() for col in summary_df.columns},
        metadata={
            "missing_tolerance_methods": missing_methods,
            "summary_plot": True,
            "ci_level": float(ci_level),
        },
    )


def plot_tolerance_trajectory_diagnostic(records: List[ParticleRecord], output_dir: OutputDir) -> None:
    """Diagnostic replicate-level tolerance schedule over wall-clock time."""
    from ..analysis import tolerance_over_wall_time

    trajectory_df = tolerance_over_wall_time(records)
    if trajectory_df.empty:
        return

    fig = tolerance_trajectory_plot(trajectory_df)
    missing_methods = sorted({record.method for record in records if record.tolerance is None})
    save_figure(
        fig,
        output_dir.plots / "tolerance_trajectory_diagnostic",
        data={col: trajectory_df[col].tolist() for col in trajectory_df.columns},
        metadata={"missing_tolerance_methods": missing_methods, "diagnostic_plot": True},
    )


def plot_scaling_summary(throughput_rows: List[Dict[str, Any]], output_dir: OutputDir) -> None:
    """Scaling + efficiency plot from throughput rows."""
    if not throughput_rows:
        return

    # Average throughput per n_workers
    by_n: Dict[int, List[float]] = defaultdict(list)
    for row in throughput_rows:
        n = int(row["n_workers"])
        t = float(row["throughput_sims_per_s"])
        by_n[n].append(t)

    if 1 not in by_n:
        return  # need baseline

    throughput_by_workers = {n: float(np.mean(ts)) for n, ts in sorted(by_n.items())}
    stem = output_dir.plots / "scaling"
    scaling_plot(throughput_by_workers, path_stem=stem)


def plot_sensitivity_summary(
    data_dir: Path,
    grid: Dict[str, List],
    output_dir: OutputDir,
) -> None:
    """Heatmap of mean final tolerance across the sensitivity grid."""
    axes = list(grid.keys())
    if len(axes) < 2:
        return

    row_key = "k" if "k" in grid else axes[0]
    col_key = "perturbation_scale" if "perturbation_scale" in grid else axes[1]
    facet_key = "tol_init_multiplier" if "tol_init_multiplier" in grid else None
    scheduler_key = "scheduler_type" if "scheduler_type" in grid else None
    row_vals = grid[row_key]
    col_vals = grid[col_key]
    facet_vals = grid[facet_key] if facet_key else None
    scheduler_vals = grid[scheduler_key] if scheduler_key else None

    # Build lookup: (row_val, col_val, facet_val, scheduler_val) → mean final tolerance
    tol_map: Dict = defaultdict(list)
    for csv_path in sorted(data_dir.glob("sensitivity_*.csv")):
        rows = _read_csv(csv_path)
        if not rows:
            continue
        tolerances = [
            float(r["tolerance"]) for r in rows
            if r.get("tolerance") not in ("", None)
        ]
        if not tolerances:
            continue
        mean_tol = float(np.mean(tolerances[-max(1, len(tolerances) // 10):]))

        # Parse variant name from filename stem
        stem_str = csv_path.stem[len("sensitivity_"):]  # strip prefix
        variant = _parse_variant_stem(stem_str, list(grid))
        rv = variant.get(row_key)
        cv = variant.get(col_key)
        fv = variant.get(facet_key) if facet_key else None
        sv = variant.get(scheduler_key) if scheduler_key else None
        if rv is not None and cv is not None:
            tol_map[(rv, cv, fv, sv)].append(mean_tol)

    if not tol_map:
        return

    row_labels = [str(v) for v in row_vals]
    col_labels = [str(v) for v in col_vals]
    stem = output_dir.plots / "sensitivity_heatmap"
    if facet_key and facet_vals is not None and scheduler_key and scheduler_vals is not None:
        matrix = np.full((len(facet_vals), len(scheduler_vals), len(row_vals), len(col_vals)), np.nan)
        for f_idx, fv in enumerate(facet_vals):
            for s_idx, sv in enumerate(scheduler_vals):
                for i, rv in enumerate(row_vals):
                    for j, cv in enumerate(col_vals):
                        key = (rv, cv, fv, sv)
                        if key in tol_map:
                            matrix[f_idx, s_idx, i, j] = float(np.mean(tol_map[key]))
        if not np.isfinite(matrix).any():
            return
        paths = sensitivity_heatmap(
            matrix,
            row_labels=row_labels,
            col_labels=col_labels,
            path_stem=stem,
            facet_row_labels=[str(v) for v in facet_vals],
            facet_col_labels=[str(v) for v in scheduler_vals],
        )
        _merge_metadata(
            paths["meta"],
            _nonbenchmark_plot_metadata(
                output_dir,
                plot_name="sensitivity_heatmap",
                title="Sensitivity heatmap",
                summary_plot=True,
            ),
        )
        return

    if facet_key and facet_vals is not None:
        matrix = np.full((len(facet_vals), len(row_vals), len(col_vals)), np.nan)
        for f_idx, fv in enumerate(facet_vals):
            for i, rv in enumerate(row_vals):
                for j, cv in enumerate(col_vals):
                    key = (rv, cv, fv, None)
                    if key in tol_map:
                        matrix[f_idx, i, j] = float(np.mean(tol_map[key]))
        if not np.isfinite(matrix).any():
            return
        paths = sensitivity_heatmap(
            matrix,
            row_labels=row_labels,
            col_labels=col_labels,
            path_stem=stem,
            facet_labels=[str(v) for v in facet_vals],
        )
        _merge_metadata(
            paths["meta"],
            _nonbenchmark_plot_metadata(
                output_dir,
                plot_name="sensitivity_heatmap",
                title="Sensitivity heatmap",
                summary_plot=True,
            ),
        )
        return

    matrix = np.full((len(row_vals), len(col_vals)), np.nan)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            vals = []
            for key, key_vals in tol_map.items():
                if key[0] == rv and key[1] == cv:
                    vals.extend(key_vals)
            if vals:
                matrix[i, j] = float(np.mean(vals))
    if not np.isfinite(matrix).any():
        return
    paths = sensitivity_heatmap(matrix, row_labels=row_labels, col_labels=col_labels, path_stem=stem)
    _merge_metadata(
        paths["meta"],
        _nonbenchmark_plot_metadata(
            output_dir,
            plot_name="sensitivity_heatmap",
            title="Sensitivity heatmap",
            summary_plot=True,
        ),
    )


def plot_ablation_summary(
    data_dir: Path,
    variants: List[Dict[str, Any]],
    output_dir: OutputDir,
) -> None:
    """Bar chart comparing mean final tolerance across ablation variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_names = [v.get("name", f"v{i}") for i, v in enumerate(variants)]
    mean_tols = []
    ci_lows = []
    ci_highs = []
    n_values = []

    for name in variant_names:
        csv_path = data_dir / f"ablation_{name}.csv"
        if not csv_path.exists():
            mean_tols.append(float("nan"))
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))
            n_values.append(0)
            continue
        rows = _read_csv(csv_path)
        tolerances = [
            float(r["tolerance"]) for r in rows
            if r.get("tolerance") not in ("", None)
        ]
        if tolerances:
            tail = np.asarray(tolerances[-max(1, len(tolerances) // 10):], dtype=float)
            mean, ci_low, ci_high = _summarize_scalar(tail, ci_level=0.95)
            mean_tols.append(mean)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
            n_values.append(int(np.isfinite(tail).sum()))
        else:
            mean_tols.append(float("nan"))
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))
            n_values.append(0)

    fig, ax = plt.subplots(figsize=(max(5, len(variant_names) * 1.2), 4))
    x = np.arange(len(variant_names))
    ax.bar(x, mean_tols, color="steelblue", alpha=0.8)
    for idx, (mean, ci_low, ci_high, n_obs) in enumerate(zip(mean_tols, ci_lows, ci_highs, n_values)):
        if np.isfinite(mean) and np.isfinite(ci_low) and np.isfinite(ci_high) and n_obs >= 2:
            ax.errorbar(
                idx,
                mean,
                yerr=[[mean - ci_low], [ci_high - mean]],
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=30, ha="right")
    ax.set_ylabel("mean final tolerance ε")
    ax.set_title("Ablation comparison")
    fig.tight_layout()

    data = {
        "variant": variant_names,
        "mean_final_tolerance": mean_tols,
        "mean_final_tolerance_ci_low": ci_lows,
        "mean_final_tolerance_ci_high": ci_highs,
        "n_observations": n_values,
    }
    stem = output_dir.plots / "ablation_comparison"
    save_figure(
        fig,
        stem,
        data=data,
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name="ablation_comparison",
            title="Ablation comparison",
            summary_plot=True,
        ),
    )


def plot_benchmark_diagnostics(
    records: List[ParticleRecord],
    cfg: Dict[str, Any],
    output_dir: OutputDir,
) -> None:
    """Emit the configured benchmark plots for a standard benchmark runner."""
    plots_cfg = cfg.get("plots", {})
    benchmark_cfg = cfg.get("benchmark", {})
    inference_cfg = cfg.get("inference", {})
    analysis_cfg = cfg.get("analysis", {})
    true_params = _true_params_from_cfg(records, benchmark_cfg)
    archive_size = inference_cfg.get("k")
    emit_paper = bool(plots_cfg.get("emit_paper_summaries", True))
    emit_diagnostics = bool(plots_cfg.get("emit_diagnostics", True))
    ci_level = float(analysis_cfg.get("ci_level", 0.95))
    min_particles_for_threshold = int(analysis_cfg.get("min_particles_for_threshold", archive_size or 100))
    if benchmark_cfg.get("name") == "lotka_volterra":
        _write_lotka_tol_init_diagnostic(records, cfg=cfg, output_dir=output_dir)
    audit_df = _write_benchmark_audit(
        records,
        true_params=true_params,
        output_dir=output_dir,
        archive_size=archive_size,
        min_particles_for_threshold=min_particles_for_threshold,
    )

    if plots_cfg.get("posterior"):
        plot_posterior(records, output_dir, archive_size=archive_size)
    if plots_cfg.get("archive_evolution") and emit_paper:
        plot_archive_evolution(records, output_dir)
    if plots_cfg.get("archive_evolution") and emit_diagnostics:
        plot_archive_evolution_diagnostic(records, output_dir)
    if plots_cfg.get("corner"):
        plot_corner(
            records,
            param_names=_param_names(records),
            output_dir=output_dir,
            true_params=true_params,
            archive_size=archive_size,
        )
    if plots_cfg.get("tolerance_trajectory") and emit_paper:
        plot_tolerance_trajectory(records, output_dir, ci_level=ci_level)
    if plots_cfg.get("tolerance_trajectory") and emit_diagnostics:
        plot_tolerance_trajectory_diagnostic(records, output_dir)
    if plots_cfg.get("quality_vs_time"):
        target = float(analysis_cfg.get("target_wasserstein", 1.0))
        checkpoint_count = len(_default_checkpoint_steps(records)) if _default_checkpoint_steps(records) else 8
        if emit_paper:
            plot_quality_vs_wall_time(
                records,
                true_params=true_params,
                output_dir=output_dir,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
                ci_level=ci_level,
                audit_df=audit_df,
            )
            plot_quality_vs_posterior_samples(
                records,
                true_params=true_params,
                output_dir=output_dir,
                archive_size=archive_size,
                ci_level=ci_level,
                audit_df=audit_df,
            )
            plot_quality_vs_attempt_budget(
                records,
                true_params=true_params,
                output_dir=output_dir,
                archive_size=archive_size,
                ci_level=ci_level,
                audit_df=audit_df,
            )
            plot_time_to_target_summary(
                records,
                true_params=true_params,
                target_wasserstein=target,
                output_dir=output_dir,
                archive_size=archive_size,
                ci_level=ci_level,
                min_particles_for_threshold=min_particles_for_threshold,
                audit_df=audit_df,
            )
            plot_attempts_to_target_summary(
                records,
                true_params=true_params,
                target_wasserstein=target,
                output_dir=output_dir,
                archive_size=archive_size,
                ci_level=ci_level,
                min_particles_for_threshold=min_particles_for_threshold,
                audit_df=audit_df,
            )
        if emit_diagnostics:
            plot_quality_vs_wall_time_diagnostic(
                records,
                true_params=true_params,
                output_dir=output_dir,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
                audit_df=audit_df,
            )
            plot_quality_vs_posterior_samples_diagnostic(
                records,
                true_params=true_params,
                output_dir=output_dir,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
            )
            plot_quality_vs_attempt_budget_diagnostic(
                records,
                true_params=true_params,
                output_dir=output_dir,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
            )
            plot_time_to_target_diagnostic(
                records,
                true_params=true_params,
                target_wasserstein=target,
                output_dir=output_dir,
                archive_size=archive_size,
                min_particles_for_threshold=min_particles_for_threshold,
            )
            plot_attempts_to_target_diagnostic(
                records,
                true_params=true_params,
                target_wasserstein=target,
                output_dir=output_dir,
                archive_size=archive_size,
                min_particles_for_threshold=min_particles_for_threshold,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _runtime_utilization_rows(records: List[ParticleRecord]) -> pd.DataFrame:
    """Return per-replicate utilization-loss rows for runtime heterogeneity sweeps."""
    rows: list[dict[str, object]] = []
    idle_data = _compute_idle_fraction(records)
    for method, by_replicate in sorted(idle_data.items()):
        if "__sigma" not in method:
            continue
        try:
            sigma = float(method.split("__sigma", 1)[1])
        except ValueError:
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
            base_method = base_method_name(method)
            if base_method not in {"abc_smc_baseline", "pyabc_smc"}:
                continue
            if "__sigma" not in method:
                continue
            try:
                sigma = float(method.split("__sigma", 1)[1])
            except ValueError:
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

    return pd.DataFrame(rows).sort_values(["sigma", "base_method", "replicate"]) if rows else pd.DataFrame()


def plot_generation_timeline(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    stem_name: str = "generation_timeline",
    title: str = "Sync generation timeline",
) -> None:
    """Timeline plot for synchronous generation spans."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spans_df = generation_spans(records)
    if spans_df.empty:
        return
    spans_df = spans_df.sort_values(["method", "replicate", "generation"]).reset_index(drop=True)
    labels = [
        f"{row.method} rep {int(row.replicate)} g{int(row.generation)}"
        for row in spans_df.itertuples(index=False)
    ]
    fig_height = max(3.5, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(8, min(10.0, fig_height)))
    for idx, row in enumerate(spans_df.itertuples(index=False)):
        ax.barh(
            idx,
            float(row.gen_duration),
            left=float(row.gen_start),
            height=0.7,
            alpha=0.85,
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("generation")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / stem_name,
        data={col: spans_df[col].tolist() for col in spans_df.columns},
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name=stem_name,
            title=title,
            methods=sorted({base_method_name(str(method)) for method in spans_df["method"].dropna().unique().tolist()}),
            diagnostic_plot=True,
        ),
    )


def _save_quality_curve_artifact(
    stem: Path,
    quality_df,
    *,
    axis_kind: str,
    metadata: Dict[str, Any],
) -> None:
    fig = posterior_quality_plot(quality_df, axis_kind=axis_kind)
    save_figure(
        fig,
        stem,
        data={col: quality_df[col].tolist() for col in quality_df.columns},
        metadata=metadata,
    )


def _save_quality_summary_artifact(
    stem: Path,
    quality_df: pd.DataFrame,
    *,
    axis_kind: str,
    ci_level: float,
    metadata: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_df = _step_curve_summary(
        quality_df,
        x_col="axis_value",
        y_col="wasserstein",
        ci_level=ci_level,
        log_y=False,
        lower_bound=0.0,
    )
    if summary_df.empty:
        write_plot_metadata(stem, metadata={**metadata, "skip_reason": "empty_summary"})
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    title = {
        "wall_time": "Posterior quality vs. wall-clock time",
        "posterior_samples": "Posterior quality vs. posterior samples",
        "attempt_budget": "Posterior quality vs. simulation attempts",
    }[axis_kind]
    xlabel = {
        "wall_time": "wall-clock time",
        "posterior_samples": "posterior samples",
        "attempt_budget": "simulation attempts",
    }[axis_kind]
    for method, group in summary_df.groupby("method", sort=True):
        group = group.sort_values("axis_value")
        ax.plot(group["axis_value"], group["wasserstein"], linewidth=1.8, label=method)
        valid_ci = group["wasserstein_ci_low"].notna() & group["wasserstein_ci_high"].notna() & (group["n_replicates"] >= 2)
        if valid_ci.any():
            ax.fill_between(
                group.loc[valid_ci, "axis_value"],
                group.loc[valid_ci, "wasserstein_ci_low"],
                group.loc[valid_ci, "wasserstein_ci_high"],
                alpha=0.2,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("wasserstein")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(
        fig,
        stem,
        data={col: summary_df[col].tolist() for col in summary_df.columns},
        metadata=metadata,
    )


def _save_threshold_summary_artifact(
    stem: Path,
    summary_df,
    *,
    axis_kind: str,
    include_replicates: bool,
    metadata: Dict[str, Any],
) -> None:
    value_col = {
        "wall_time": "wall_time_to_threshold",
        "posterior_samples": "posterior_samples_to_threshold",
        "attempt_budget": "attempts_to_threshold",
    }[axis_kind]
    if summary_df.empty or value_col not in summary_df.columns or not np.isfinite(summary_df[value_col].to_numpy(dtype=float)).any():
        write_plot_metadata(stem, metadata={**metadata, "skip_reason": "threshold_not_reached"})
        return
    fig = threshold_summary_plot(summary_df, axis_kind=axis_kind, include_replicates=include_replicates)
    save_figure(
        fig,
        stem,
        data={col: summary_df[col].tolist() for col in summary_df.columns},
        metadata=metadata,
    )


def _parse_scalar(raw: str):
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_variant_stem(stem_str: str, keys: List[str]) -> Dict[str, Any]:
    """Decode a sensitivity variant filename stem into a dict."""
    if "=" in stem_str:
        variant = {}
        for part in stem_str.split("__"):
            if "=" not in part:
                return {}
            key, raw = part.split("=", 1)
            variant[key] = _parse_scalar(raw)
        return variant

    # Backward compatibility for legacy stems like:
    # "k10_perturbation_scale0.4_scheduler_typeacceptance_rate"
    ordered_keys = sorted(keys)
    remainder = stem_str
    variant = {}
    for index, key in enumerate(ordered_keys):
        prefix = key if index == 0 else f"_{key}"
        if not remainder.startswith(prefix):
            return {}
        start = len(prefix)
        if index + 1 < len(ordered_keys):
            next_prefix = f"_{ordered_keys[index + 1]}"
            end = remainder.find(next_prefix, start)
            if end == -1:
                return {}
            raw = remainder[start:end]
            remainder = remainder[end:]
        else:
            raw = remainder[start:]
            remainder = ""
        variant[key] = _parse_scalar(raw)
    return variant


def _true_params_from_cfg(records: List[ParticleRecord], benchmark_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Extract true parameter values from benchmark config using param names."""
    true_params: Dict[str, float] = {}
    for param in _param_names(records):
        key = f"true_{param}"
        if key in benchmark_cfg:
            true_params[param] = float(benchmark_cfg[key])
    return true_params


def _default_checkpoint_steps(records: List[ParticleRecord], count: int = 8) -> List[int]:
    """Choose a small set of evenly spaced simulation checkpoints."""
    max_step = max((int(r.step) for r in records), default=0)
    if max_step <= 0:
        return []
    n = min(count, max_step)
    return sorted({int(round(x)) for x in np.linspace(1, max_step, num=n)})
