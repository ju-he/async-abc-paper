"""High-level plot reporters called by runner scripts.

Each function reads already-collected records (or CSVs) and produces the
standard set of figures for a given experiment type.
"""
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from ..io.paths import OutputDir
from ..io.records import ParticleRecord, load_records
from ..reporting import (
    benchmark_plot_metadata,
    nonbenchmark_plot_metadata as _nonbenchmark_plot_metadata,
    runtime_utilization_rows as _runtime_utilization_rows,
    compute_idle_fraction as _compute_idle_fraction,
    normalize_runtime_utilization_summary,
    write_gaussian_analytic_summary,
)
from ..analysis import (
    benchmark_plot_audit,
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
    _build_state_kind_map,
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


def _configured_max_wall_time_s(output_dir: OutputDir) -> float | None:
    """Return the configured wall-time budget from metadata, if available."""
    meta_path = output_dir.data / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        value = meta.get("config", {}).get("inference", {}).get("max_wall_time_s")
    except Exception:
        return None
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0.0 else None


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
    cfg: Dict[str, Any] | None = None,
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
            save_paths = posterior_plot(samples, param_name=param, path_stem=output_dir.plots / f"posterior_{param}")
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
        if cfg is not None:
            meta_path = save_paths.get("meta")
            if meta_path is not None:
                _merge_metadata(
                    meta_path,
                    benchmark_plot_metadata(
                        cfg,
                        plot_name=f"posterior_{param}",
                        output_dir=output_dir,
                        extra={"sample_counts": sample_counts},
                    ),
                )


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


def plot_archive_evolution(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
    ci_level: float = 0.95,
) -> None:
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="archive_evolution",
            output_dir=output_dir,
            extra={"ci_level": float(ci_level)},
        ) if cfg is not None else {"plot_name": "tolerance_vs_simulations", "summary_plot": True, "ci_level": float(ci_level)},
    )


def plot_archive_evolution_diagnostic(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
) -> None:
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="archive_evolution_diagnostic",
            output_dir=output_dir,
        ) if cfg is not None else {"plot_name": "tolerance_vs_simulations_diagnostic", "diagnostic_plot": True},
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

def plot_idle_fraction(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    summary_df: pd.DataFrame | None = None,
) -> None:
    """Bar chart of utilization-loss fraction per method/sigma level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df is None:
        summary_df = normalize_runtime_utilization_summary(records)
    else:
        summary_df = normalize_runtime_utilization_summary(records, summary_df=summary_df)
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

    max_wall_time_s = _configured_max_wall_time_s(output_dir)

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
        if max_wall_time_s is not None:
            max_time = min(max_time, float(max_wall_time_s))
        if not np.isfinite(max_time) or max_time <= 0:
            continue
        edges = np.linspace(0.0, max_time, n_bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
        for method_idx, base_method in enumerate(sorted(sigma_frame["base_method"].unique().tolist())):
            method_frame = sigma_frame.loc[sigma_frame["base_method"] == base_method]
            replicate_curves: list[np.ndarray] = []
            for _, rep_group in method_frame.groupby("replicate", sort=True):
                end_times = rep_group["relative_end_time"].to_numpy(dtype=float)
                if max_wall_time_s is not None:
                    end_times = end_times[end_times <= max_time]
                counts, _ = np.histogram(end_times, bins=edges)
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
            extra={
                "facet_by": "sigma",
                "max_wall_time_s": None if max_wall_time_s is None else float(max_wall_time_s),
            },
        ),
    )


def plot_idle_fraction_comparison(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    summary_df: pd.DataFrame | None = None,
) -> None:
    """Utilization-loss fraction vs. sigma with per-replicate scatter and mean line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df is None:
        summary_df = normalize_runtime_utilization_summary(records)
    else:
        summary_df = normalize_runtime_utilization_summary(records, summary_df=summary_df)
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
            extra={
                "paper_primary": True,
                "evidence_role": "hpc_primary",
                "performance_metric": "utilization_loss_vs_heterogeneity",
                "summary_source": "runtime_performance_summary.csv" if summary_df is not None else None,
            },
        ),
    )


def plot_quality_by_sigma(
    records: List[ParticleRecord],
    cfg: Dict[str, Any],
    output_dir: OutputDir,
    *,
    summary_df: pd.DataFrame | None = None,
) -> None:
    """Headline heterogeneity figure.

    When ``summary_df`` is provided, plot final quality at the fixed wall-clock
    budget versus ``sigma`` using aggregate per-replicate summaries. Otherwise
    fall back to the richer raw-record wall-time traces.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..analysis import posterior_quality_curve

    if summary_df is not None and not summary_df.empty:
        df = summary_df.copy()
        df = df.dropna(subset=["sigma", "base_method", "final_quality_wasserstein"])
        if not df.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            rows: list[dict[str, object]] = []
            for method, group in df.groupby("base_method", sort=True):
                sigma_rows = []
                for sigma, sigma_group in group.groupby("sigma", sort=True):
                    mean, ci_low, ci_high = _summarize_scalar(
                        sigma_group["final_quality_wasserstein"].to_numpy(dtype=float),
                        ci_level=0.95,
                    )
                    n_replicates = int(sigma_group["replicate"].nunique())
                    sigma_rows.append((float(sigma), mean, ci_low, ci_high, n_replicates))
                    rows.append(
                        {
                            "sigma": float(sigma),
                            "base_method": str(method),
                            "final_quality_wasserstein": mean,
                            "final_quality_wasserstein_ci_low": ci_low,
                            "final_quality_wasserstein_ci_high": ci_high,
                            "n_replicates": n_replicates,
                        }
                    )
                sigma_rows.sort(key=lambda item: item[0])
                ax.plot(
                    [row[0] for row in sigma_rows],
                    [row[1] for row in sigma_rows],
                    marker="o",
                    label=str(method),
                )
                if any(np.isfinite(row[2]) and np.isfinite(row[3]) for row in sigma_rows):
                    ax.fill_between(
                        [row[0] for row in sigma_rows],
                        [row[2] for row in sigma_rows],
                        [row[3] for row in sigma_rows],
                        alpha=0.15,
                    )
            ax.set_xlabel("heterogeneity (σ)")
            ax.set_ylabel("final quality at fixed wall-clock budget (Wasserstein)")
            ax.set_title("Posterior quality at fixed wall-clock budget by heterogeneity")
            ax.legend(frameon=False, fontsize="small")
            fig.tight_layout()
            save_figure(
                fig,
                output_dir.plots / "quality_by_sigma",
                data={col: [row[col] for row in rows] for col in rows[0].keys()} if rows else None,
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name="quality_by_sigma",
                    title="Posterior quality at fixed wall-clock budget by σ",
                    methods=sorted(df["base_method"].dropna().unique().tolist()),
                    summary_plot=True,
                    extra={
                        "paper_primary": True,
                        "evidence_role": "hpc_primary",
                        "performance_metric": "quality_at_fixed_wallclock_budget",
                        "summary_source": "runtime_performance_summary.csv",
                        "n_sigma_levels": int(df["sigma"].nunique()),
                    },
                ),
            )
            return

    benchmark_cfg = cfg.get("benchmark", {})
    inference_cfg = cfg.get("inference", {})
    true_params = _true_params_from_cfg(records, benchmark_cfg)
    if not true_params:
        write_plot_metadata(
            output_dir.plots / "quality_by_sigma",
            metadata={
                "plot_name": "quality_by_sigma",
                "skip_reason": "no_true_params",
                "summary_plot": True,
            },
        )
        return

    archive_size = inference_cfg.get("k")

    sigma_set: set[float] = set()
    for r in records:
        if "__sigma" in r.method:
            try:
                sigma_set.add(float(r.method.split("__sigma", 1)[1]))
            except ValueError:
                pass
    sigmas = sorted(sigma_set)
    if not sigmas:
        write_plot_metadata(
            output_dir.plots / "quality_by_sigma",
            metadata={
                "plot_name": "quality_by_sigma",
                "skip_reason": "no_sigma_tagged_records",
                "summary_plot": True,
            },
        )
        return

    fig, axes = plt.subplots(
        len(sigmas), 1,
        figsize=(7.0, max(3.5, 2.8 * len(sigmas))),
        squeeze=False,
    )
    summary_rows: list[dict[str, object]] = []
    cmap = plt.get_cmap("tab10")

    for sigma_idx, sigma in enumerate(sigmas):
        ax = axes[sigma_idx, 0]
        sigma_records = [r for r in records if f"__sigma{sigma}" in r.method]
        if not sigma_records:
            ax.set_title(f"σ={sigma:g} — no data")
            ax.set_visible(False)
            continue

        # Strip __sigma{X} suffix so quality_curve groups by base method name
        stripped = [
            ParticleRecord(
                method=base_method_name(r.method),
                replicate=r.replicate,
                seed=r.seed,
                step=r.step,
                params=r.params,
                loss=r.loss,
                weight=r.weight,
                tolerance=r.tolerance,
                wall_time=r.wall_time,
                worker_id=r.worker_id,
                sim_start_time=r.sim_start_time,
                sim_end_time=r.sim_end_time,
                generation=r.generation,
                record_kind=r.record_kind,
                time_semantics=r.time_semantics,
                attempt_count=r.attempt_count,
            )
            for r in sigma_records
        ]

        quality_df = posterior_quality_curve(
            stripped,
            true_params=true_params,
            axis_kind="wall_time",
            checkpoint_strategy="quantile",
            checkpoint_count=8,
            archive_size=archive_size,
        )
        if quality_df.empty:
            ax.set_title(f"σ={sigma:g} — no quality data")
            ax.text(0.5, 0.5, "no quality data", ha="center", va="center", transform=ax.transAxes)
            continue

        panel_summary = _step_curve_summary(
            quality_df,
            x_col="axis_value",
            y_col="wasserstein",
            ci_level=0.95,
            log_y=False,
            lower_bound=0.0,
        )
        sk_map = _build_state_kind_map(quality_df)
        for method_idx, (method, group) in enumerate(panel_summary.groupby("method", sort=True)):
            group = group.sort_values("axis_value")
            color = cmap(method_idx % 10)
            sk_suffix = f" [{sk_map[method]}]" if method in sk_map else ""
            ax.plot(
                group["axis_value"], group["wasserstein"],
                linewidth=1.8, label=f"{method}{sk_suffix}", color=color,
            )
            valid_ci = (
                group["wasserstein_ci_low"].notna()
                & group["wasserstein_ci_high"].notna()
                & (group["n_replicates"] >= 2)
            )
            if valid_ci.any():
                ax.fill_between(
                    group.loc[valid_ci, "axis_value"],
                    group.loc[valid_ci, "wasserstein_ci_low"],
                    group.loc[valid_ci, "wasserstein_ci_high"],
                    alpha=0.15, color=color,
                )
            for _, row in group.iterrows():
                summary_rows.append(
                    {
                        "sigma": float(sigma),
                        "base_method": str(method),
                        "wall_time": float(row["axis_value"]),
                        "wasserstein": float(row["wasserstein"]),
                        "n_replicates": int(row["n_replicates"]),
                    }
                )
        ax.set_xlabel("wall-clock time")
        ax.set_ylabel("wasserstein")
        ax.set_title(f"Quality vs. wall-clock time: σ={sigma:g}")
        ax.legend(frameon=False, fontsize="small")

    fig.tight_layout()
    data = (
        {col: [row[col] for row in summary_rows]
         for col in ["sigma", "base_method", "wall_time", "wasserstein", "n_replicates"]}
        if summary_rows
        else None
    )
    save_figure(
        fig,
        output_dir.plots / "quality_by_sigma",
        data=data,
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name="quality_by_sigma",
            title="Posterior quality vs. wall-clock time by σ",
            methods=sorted({base_method_name(r.method) for r in records}),
            summary_plot=True,
            extra={"n_sigma_levels": len(sigmas), "n_data_rows": len(summary_rows)},
        ),
    )


def plot_quality_vs_wall_time_diagnostic(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_wall_time_diagnostic",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
                "plot_name": "quality_vs_wall_time_diagnostic",
                "skip_reason": "missing_true_params_or_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    metadata = benchmark_plot_metadata(
        cfg or {},
        plot_name="quality_vs_wall_time_diagnostic",
        output_dir=output_dir,
        extra={
            "axis_kind": "wall_time",
            "state_kind": "observable_posterior_state",
            "checkpoint_strategy": "quantile",
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_wall_time",
                output_dir=output_dir,
                extra=_audit_skip_metadata(
                    plot_name="quality_vs_wall_time",
                    audit_df=audit_df,
                    gate_column="paper_quality_plots_allowed",
                    extra={"source_raw_files": [str(output_dir.data / "raw_results.csv")]},
                ),
            ) if cfg is not None else _audit_skip_metadata(
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_wall_time",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="quality_vs_wall_time",
            output_dir=output_dir,
            extra={
                "axis_kind": "wall_time",
                "ci_level": float(ci_level),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
            "plot_name": "quality_vs_wall_time",
            "axis_kind": "wall_time",
            "summary_plot": True,
            "ci_level": float(ci_level),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def _progress_export_frame(
    *,
    tolerance_df: pd.DataFrame | None = None,
    quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if tolerance_df is not None and not tolerance_df.empty:
        tol_frame = tolerance_df.copy()
        tol_frame["metric"] = "tolerance"
        tol_frame["axis_kind"] = "wall_time"
        tol_frame["axis_value"] = tol_frame["wall_time"]
        frames.append(tol_frame)
    if quality_df is not None and not quality_df.empty:
        qual_frame = quality_df.copy()
        qual_frame["metric"] = "wasserstein"
        frames.append(qual_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def plot_progress_summary(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
    ci_level: float = 0.95,
    audit_df: pd.DataFrame | None = None,
) -> None:
    """Paper summary: tolerance and Wasserstein progress over wall-clock time."""
    from ..analysis import posterior_quality_curve, tolerance_over_wall_time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit_df = audit_df if audit_df is not None else pd.DataFrame()
    tolerance_df = tolerance_over_wall_time(records)
    tolerance_summary = pd.DataFrame()
    if not tolerance_df.empty:
        tolerance_summary = _step_curve_summary(
            tolerance_df,
            x_col="wall_time",
            y_col="tolerance",
            ci_level=ci_level,
            log_y=True,
        )

    quality_summary = pd.DataFrame()
    quality_skip_reason = None
    if true_params and not _paper_plot_allowed(audit_df, "paper_quality_plots_allowed"):
        quality_skip_reason = "paper_quality_plots_disallowed_by_audit"
    else:
        quality_df = posterior_quality_curve(
            records,
            true_params=true_params,
            axis_kind="wall_time",
            checkpoint_strategy="quantile",
            checkpoint_count=checkpoint_count,
            archive_size=archive_size,
        )
        if quality_df.empty:
            quality_skip_reason = "missing_true_params_or_quality_rows"
        else:
            quality_summary = _step_curve_summary(
                quality_df,
                x_col="axis_value",
                y_col="wasserstein",
                ci_level=ci_level,
                log_y=False,
                lower_bound=0.0,
            )
            if quality_summary.empty:
                quality_skip_reason = "empty_quality_summary"

    if tolerance_summary.empty and quality_summary.empty:
        write_plot_metadata(
            output_dir.plots / "progress_summary",
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="progress_summary",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_tolerance_and_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
                "plot_name": "progress_summary",
                "skip_reason": "missing_tolerance_and_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    n_panels = int(not tolerance_summary.empty) + int(not quality_summary.empty)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4), squeeze=False)
    axes_list = list(axes.ravel())
    panel_idx = 0
    if not tolerance_summary.empty:
        ax = axes_list[panel_idx]
        panel_idx += 1
        for method, group in tolerance_summary.groupby("method", sort=True):
            group = group.sort_values("wall_time")
            ax.plot(group["wall_time"], group["tolerance"], linewidth=1.8, label=method)
            valid_ci = (
                group["tolerance_ci_low"].notna()
                & group["tolerance_ci_high"].notna()
                & (group["n_replicates"] >= 2)
            )
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
        ax.set_title("Tolerance progress")
        ax.legend(frameon=False, fontsize=8)
    if not quality_summary.empty:
        ax = axes_list[panel_idx]
        sk_map = _build_state_kind_map(quality_df) if not quality_df.empty else {}
        for method, group in quality_summary.groupby("method", sort=True):
            group = group.sort_values("axis_value")
            sk_suffix = f" [{sk_map[method]}]" if method in sk_map else ""
            ax.plot(group["axis_value"], group["wasserstein"], linewidth=1.8, label=f"{method}{sk_suffix}")
            valid_ci = (
                group["wasserstein_ci_low"].notna()
                & group["wasserstein_ci_high"].notna()
                & (group["n_replicates"] >= 2)
            )
            if valid_ci.any():
                ax.fill_between(
                    group.loc[valid_ci, "axis_value"],
                    group.loc[valid_ci, "wasserstein_ci_low"],
                    group.loc[valid_ci, "wasserstein_ci_high"],
                    alpha=0.2,
                )
        ax.set_xlabel("wall-clock time")
        ax.set_ylabel("wasserstein")
        ax.set_title("Posterior quality progress")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    export_df = _progress_export_frame(
        tolerance_df=tolerance_summary,
        quality_df=quality_summary,
    )
    save_figure(
        fig,
        output_dir.plots / "progress_summary",
        data={col: export_df[col].tolist() for col in export_df.columns},
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="progress_summary",
            output_dir=output_dir,
            extra={
                "ci_level": float(ci_level),
                "has_tolerance_panel": bool(not tolerance_summary.empty),
                "has_wasserstein_panel": bool(not quality_summary.empty),
                "quality_skip_reason": quality_skip_reason,
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
            "plot_name": "progress_summary",
            "summary_plot": True,
            "ci_level": float(ci_level),
            "has_tolerance_panel": bool(not tolerance_summary.empty),
            "has_wasserstein_panel": bool(not quality_summary.empty),
            "quality_skip_reason": quality_skip_reason,
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_quality_vs_posterior_samples(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_posterior_samples",
                output_dir=output_dir,
                extra=_audit_skip_metadata(
                    plot_name="quality_vs_posterior_samples",
                    audit_df=audit_df,
                    gate_column="paper_quality_plots_allowed",
                    extra={"source_raw_files": [str(output_dir.data / "raw_results.csv")]},
                ),
            ) if cfg is not None else _audit_skip_metadata(
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_posterior_samples",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="quality_vs_posterior_samples",
            output_dir=output_dir,
            extra={
                "axis_kind": "posterior_samples",
                "ci_level": float(ci_level),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_posterior_samples_diagnostic",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="quality_vs_posterior_samples_diagnostic",
            output_dir=output_dir,
            extra={
                "axis_kind": "posterior_samples",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_attempt_budget",
                output_dir=output_dir,
                extra=_audit_skip_metadata(
                    plot_name="quality_vs_attempt_budget",
                    audit_df=audit_df,
                    gate_column="paper_quality_plots_allowed",
                    extra={"source_raw_files": [str(output_dir.data / "raw_results.csv")]},
                ),
            ) if cfg is not None else _audit_skip_metadata(
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_attempt_budget",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="quality_vs_attempt_budget",
            output_dir=output_dir,
            extra={
                "axis_kind": "attempt_budget",
                "ci_level": float(ci_level),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="quality_vs_attempt_budget_diagnostic",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="quality_vs_attempt_budget_diagnostic",
            output_dir=output_dir,
            extra={
                "axis_kind": "attempt_budget",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
            "plot_name": "quality_vs_attempt_budget_diagnostic",
            "axis_kind": "attempt_budget",
            "diagnostic_plot": True,
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_progress_diagnostic(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
) -> None:
    """Diagnostic replicate-level tolerance and Wasserstein progress over wall-clock time."""
    from ..analysis import posterior_quality_curve, tolerance_over_wall_time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tolerance_df = tolerance_over_wall_time(records)
    quality_df = posterior_quality_curve(
        records,
        true_params=true_params,
        axis_kind="wall_time",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if tolerance_df.empty and quality_df.empty:
        write_plot_metadata(
            output_dir.plots / "progress_diagnostic",
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="progress_diagnostic",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_tolerance_and_quality_rows",
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
                "plot_name": "progress_diagnostic",
                "skip_reason": "missing_tolerance_and_quality_rows",
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        )
        return

    n_panels = int(not tolerance_df.empty) + int(not quality_df.empty)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4), squeeze=False)
    axes_list = list(axes.ravel())
    panel_idx = 0
    if not tolerance_df.empty:
        tolerance_trajectory_plot(tolerance_df, ax=axes_list[panel_idx])
        axes_list[panel_idx].set_title("Tolerance progress")
        panel_idx += 1
    if not quality_df.empty:
        posterior_quality_plot(quality_df, axis_kind="wall_time", ax=axes_list[panel_idx])
        axes_list[panel_idx].set_title("Posterior quality progress")
    fig.tight_layout()

    export_df = _progress_export_frame(
        tolerance_df=tolerance_df,
        quality_df=quality_df,
    )
    save_figure(
        fig,
        output_dir.plots / "progress_diagnostic",
        data={col: export_df[col].tolist() for col in export_df.columns},
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="progress_diagnostic",
            output_dir=output_dir,
            extra={
                "has_tolerance_panel": bool(not tolerance_df.empty),
                "has_wasserstein_panel": bool(not quality_df.empty),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
            "plot_name": "progress_diagnostic",
            "diagnostic_plot": True,
            "has_tolerance_panel": bool(not tolerance_df.empty),
            "has_wasserstein_panel": bool(not quality_df.empty),
            "source_raw_files": [str(output_dir.data / "raw_results.csv")],
        },
    )


def plot_time_to_target_summary(
    records: List[ParticleRecord],
    true_params: Dict[str, float],
    target_wasserstein: float,
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="time_to_target_summary",
                output_dir=output_dir,
                extra=_audit_skip_metadata(
                    plot_name="time_to_target_summary",
                    audit_df=audit_df,
                    gate_column="paper_threshold_plots_allowed",
                    extra={
                        "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                        "target_wasserstein": float(target_wasserstein),
                    },
                ),
            ) if cfg is not None else _audit_skip_metadata(
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="time_to_target_summary",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_threshold_rows",
                    "target_wasserstein": float(target_wasserstein),
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="time_to_target_summary",
            output_dir=output_dir,
            extra={
                "axis_kind": "wall_time",
                "target_wasserstein": float(target_wasserstein),
                "ci_level": float(ci_level),
                "min_particles_for_threshold": int(min_particles_for_threshold),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="time_to_target_diagnostic",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_threshold_rows",
                    "target_wasserstein": float(target_wasserstein),
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="time_to_target_diagnostic",
            output_dir=output_dir,
            extra={
                "axis_kind": "wall_time",
                "target_wasserstein": float(target_wasserstein),
                "min_particles_for_threshold": int(min_particles_for_threshold),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="attempts_to_target_summary",
                output_dir=output_dir,
                extra=_audit_skip_metadata(
                    plot_name="attempts_to_target_summary",
                    audit_df=audit_df,
                    gate_column="paper_threshold_plots_allowed",
                    extra={
                        "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                        "target_wasserstein": float(target_wasserstein),
                    },
                ),
            ) if cfg is not None else _audit_skip_metadata(
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="attempts_to_target_summary",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_threshold_rows",
                    "target_wasserstein": float(target_wasserstein),
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="attempts_to_target_summary",
            output_dir=output_dir,
            extra={
                "axis_kind": "attempt_budget",
                "target_wasserstein": float(target_wasserstein),
                "ci_level": float(ci_level),
                "min_particles_for_threshold": int(min_particles_for_threshold),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
            metadata=benchmark_plot_metadata(
                cfg or {},
                plot_name="attempts_to_target_diagnostic",
                output_dir=output_dir,
                extra={
                    "skip_reason": "missing_true_params_or_threshold_rows",
                    "target_wasserstein": float(target_wasserstein),
                    "source_raw_files": [str(output_dir.data / "raw_results.csv")],
                },
            ) if cfg is not None else {
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="attempts_to_target_diagnostic",
            output_dir=output_dir,
            extra={
                "axis_kind": "attempt_budget",
                "target_wasserstein": float(target_wasserstein),
                "min_particles_for_threshold": int(min_particles_for_threshold),
                "source_raw_files": [str(output_dir.data / "raw_results.csv")],
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
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
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="corner",
            output_dir=output_dir,
            extra={
                "sample_counts": {
                    result.method: result.n_particles_used
                    for result in final_state_results(records, archive_size=archive_size)
                },
            },
        ) if cfg is not None else {
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
    cfg: Dict[str, Any] | None = None,
    true_params: Dict[str, float] | None = None,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
    ci_level: float = 0.95,
) -> None:
    """Paper summary: tolerance and Wasserstein progress over wall-clock time."""
    from ..analysis import posterior_quality_curve, tolerance_over_wall_time

    trajectory_df = tolerance_over_wall_time(records)
    quality_df = posterior_quality_curve(
        records,
        true_params=true_params or {},
        axis_kind="wall_time",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if trajectory_df.empty and quality_df.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tolerance_summary = pd.DataFrame()
    if not trajectory_df.empty:
        tolerance_summary = _step_curve_summary(
            trajectory_df,
            x_col="wall_time",
            y_col="tolerance",
            ci_level=ci_level,
            log_y=True,
        )
    quality_summary = pd.DataFrame()
    if not quality_df.empty:
        quality_summary = _step_curve_summary(
            quality_df,
            x_col="axis_value",
            y_col="wasserstein",
            ci_level=ci_level,
            log_y=False,
            lower_bound=0.0,
        )
    if tolerance_summary.empty and quality_summary.empty:
        return

    n_panels = int(not tolerance_summary.empty) + int(not quality_summary.empty)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4), squeeze=False)
    axes_list = list(axes.ravel())
    panel_idx = 0
    if not tolerance_summary.empty:
        ax = axes_list[panel_idx]
        panel_idx += 1
        for method, group in tolerance_summary.groupby("method", sort=True):
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
    if not quality_summary.empty:
        ax = axes_list[panel_idx]
        sk_map = _build_state_kind_map(quality_df) if not quality_df.empty else {}
        for method, group in quality_summary.groupby("method", sort=True):
            group = group.sort_values("axis_value")
            sk_suffix = f" [{sk_map[method]}]" if method in sk_map else ""
            ax.plot(group["axis_value"], group["wasserstein"], linewidth=1.8, label=f"{method}{sk_suffix}")
            valid_ci = group["wasserstein_ci_low"].notna() & group["wasserstein_ci_high"].notna() & (group["n_replicates"] >= 2)
            if valid_ci.any():
                ax.fill_between(
                    group.loc[valid_ci, "axis_value"],
                    group.loc[valid_ci, "wasserstein_ci_low"],
                    group.loc[valid_ci, "wasserstein_ci_high"],
                    alpha=0.2,
                )
        ax.set_xlabel("wall-clock time")
        ax.set_ylabel("wasserstein")
        ax.set_title("Wasserstein trajectory")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    missing_methods = sorted({record.method for record in records if record.tolerance is None})
    export_df = _progress_export_frame(
        tolerance_df=tolerance_summary,
        quality_df=quality_summary,
    )
    save_figure(
        fig,
        output_dir.plots / "tolerance_trajectory",
        data={col: export_df[col].tolist() for col in export_df.columns},
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="tolerance_trajectory",
            output_dir=output_dir,
            extra={
                "missing_tolerance_methods": missing_methods,
                "ci_level": float(ci_level),
                "has_tolerance_panel": bool(not tolerance_summary.empty),
                "has_wasserstein_panel": bool(not quality_summary.empty),
            },
        ) if cfg is not None else {
            "missing_tolerance_methods": missing_methods,
            "summary_plot": True,
            "ci_level": float(ci_level),
            "has_tolerance_panel": bool(not tolerance_summary.empty),
            "has_wasserstein_panel": bool(not quality_summary.empty),
        },
    )


def plot_tolerance_trajectory_diagnostic(
    records: List[ParticleRecord],
    output_dir: OutputDir,
    *,
    cfg: Dict[str, Any] | None = None,
    true_params: Dict[str, float] | None = None,
    archive_size: int | None = None,
    checkpoint_count: int = 8,
) -> None:
    """Diagnostic replicate-level tolerance and Wasserstein progress over wall-clock time."""
    from ..analysis import posterior_quality_curve, tolerance_over_wall_time

    trajectory_df = tolerance_over_wall_time(records)
    quality_df = posterior_quality_curve(
        records,
        true_params=true_params or {},
        axis_kind="wall_time",
        checkpoint_strategy="quantile",
        checkpoint_count=checkpoint_count,
        archive_size=archive_size,
    )
    if trajectory_df.empty and quality_df.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = int(not trajectory_df.empty) + int(not quality_df.empty)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4), squeeze=False)
    axes_list = list(axes.ravel())
    panel_idx = 0
    if not trajectory_df.empty:
        tolerance_trajectory_plot(trajectory_df, ax=axes_list[panel_idx])
        axes_list[panel_idx].set_title("Tolerance trajectory")
        panel_idx += 1
    if not quality_df.empty:
        posterior_quality_plot(quality_df, axis_kind="wall_time", ax=axes_list[panel_idx])
        axes_list[panel_idx].set_title("Wasserstein trajectory")
    fig.tight_layout()

    missing_methods = sorted({record.method for record in records if record.tolerance is None})
    export_df = _progress_export_frame(
        tolerance_df=trajectory_df,
        quality_df=quality_df,
    )
    save_figure(
        fig,
        output_dir.plots / "tolerance_trajectory_diagnostic",
        data={col: export_df[col].tolist() for col in export_df.columns},
        metadata=benchmark_plot_metadata(
            cfg or {},
            plot_name="tolerance_trajectory_diagnostic",
            output_dir=output_dir,
            extra={
                "missing_tolerance_methods": missing_methods,
                "has_tolerance_panel": bool(not trajectory_df.empty),
                "has_wasserstein_panel": bool(not quality_df.empty),
            },
        ) if cfg is not None else {
            "missing_tolerance_methods": missing_methods,
            "diagnostic_plot": True,
            "has_tolerance_panel": bool(not trajectory_df.empty),
            "has_wasserstein_panel": bool(not quality_df.empty),
        },
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


def _scaling_token(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _budget_token(budget_s: float) -> str:
    if float(budget_s).is_integer():
        return str(int(budget_s))
    return str(budget_s).replace(".", "p")


def _scaling_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in ("k", "n_workers", "replicate", "seed", "requested_max_simulations", "n_simulations", "final_n_particles"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in (
        "max_wall_time_s",
        "elapsed_wall_time_s",
        "throughput_sims_per_s",
        "worker_utilization",
        "final_quality_wasserstein",
        "final_tolerance",
        "budget_s",
        "attempts_by_budget",
        "posterior_samples_by_budget",
        "quality_wasserstein_by_budget",
        "best_tolerance_by_budget",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _save_scaling_dataframe(fig, stem: Path, frame: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    save_figure(
        fig,
        stem,
        data={col: frame[col].tolist() for col in frame.columns},
        metadata=metadata,
    )


def _plot_workers_lines(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    series_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    stem: Path,
    metadata: Dict[str, Any],
    yerr_col: Optional[str] = None,
    log_x: bool = False,
    hline: Optional[float] = None,
    ideal_y_col: Optional[str] = None,
) -> None:
    if frame.empty:
        return
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ideal_drawn = False
    for series_value, group in frame.groupby(series_col, sort=True):
        group = group.sort_values(x_col)
        x_vals = group[x_col].to_numpy(dtype=float)
        y_vals = group[y_col].to_numpy(dtype=float)
        (line,) = ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, label=str(series_value))
        if yerr_col is not None and yerr_col in group.columns:
            y_err = group[yerr_col].to_numpy(dtype=float)
            ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, alpha=0.15, color=line.get_color())
        if ideal_y_col is not None and not ideal_drawn and ideal_y_col in group.columns:
            ax.plot(x_vals, group[ideal_y_col].to_numpy(dtype=float), "--", color="grey", linewidth=1.2, label="ideal", zorder=0)
            ideal_drawn = True
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=1.0, zorder=0)
    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: str(int(x)) if x == int(x) else str(x))
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize="small", title=series_col.replace("_", " "))
    fig.tight_layout()
    _save_scaling_dataframe(fig, stem, frame, metadata)


def _plot_heatmap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    stem: Path,
    metadata: Dict[str, Any],
) -> None:
    if frame.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = frame.pivot(index="k", columns="n_workers", values=value_col).sort_index().sort_index(axis=1)
    if pivot.empty:
        return
    n_rows = len(pivot.index)
    n_cols = len(pivot.columns)
    # Use integer indices so each cell has equal visual size regardless of the
    # log-spaced k / n_workers values; true labels are set via tick formatters.
    col_edges = np.arange(n_cols + 1) - 0.5
    row_edges = np.arange(n_rows + 1) - 0.5
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    mesh = ax.pcolormesh(col_edges, row_edges, pivot.to_numpy(dtype=float), shading="flat")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(int(value)) for value in pivot.columns])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(int(value)) for value in pivot.index])
    ax.set_xlabel("n_workers")
    ax.set_ylabel("k")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, shrink=0.9, label=value_col.replace("_", " "))
    fig.tight_layout()
    _save_scaling_dataframe(fig, stem, frame, metadata)


def _plot_quality_vs_attempts_scatter(frame: pd.DataFrame, *, stem: Path, metadata: Dict[str, Any], title: str) -> None:
    if frame.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    methods = sorted(frame["base_method"].dropna().unique().tolist())
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, method in enumerate(methods):
        subset = frame.loc[frame["base_method"] == method]
        scatter = ax.scatter(
            subset["attempts_by_budget"].to_numpy(dtype=float),
            subset["quality_wasserstein_by_budget"].to_numpy(dtype=float),
            c=subset["n_workers"].to_numpy(dtype=float),
            cmap="viridis",
            marker=markers[idx % len(markers)],
            s=56,
            alpha=0.9,
            label=str(method),
        )
    ax.set_xlabel("attempts by budget")
    ax.set_ylabel("quality (Wasserstein)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize="small")
    fig.colorbar(scatter, ax=ax, shrink=0.9, label="n_workers")
    fig.tight_layout()
    _save_scaling_dataframe(fig, stem, frame, metadata)


def plot_scaling_grid(
    *,
    throughput_rows: List[Dict[str, Any]],
    budget_rows: List[Dict[str, Any]],
    output_dir: OutputDir,
) -> None:
    """Generate grid-aware scaling plots from run-level and budget summaries."""
    throughput_df = _scaling_frame(throughput_rows)
    budget_df = _scaling_frame(budget_rows)
    if throughput_df.empty or budget_df.empty:
        return

    methods = sorted(throughput_df["base_method"].dropna().unique().tolist())
    budget_df = budget_df.copy()
    budget_df["throughput_by_budget"] = np.where(
        budget_df["elapsed_wall_time_s"] > 0,
        budget_df["attempts_by_budget"] / budget_df["elapsed_wall_time_s"],
        np.nan,
    )

    _throughput_grp = throughput_df.groupby(["base_method", "k", "n_workers"])
    throughput_mean = _throughput_grp[["throughput_sims_per_s"]].mean().reset_index()
    throughput_mean = throughput_mean.merge(
        _throughput_grp[["throughput_sims_per_s"]].std().reset_index().rename(
            columns={"throughput_sims_per_s": "throughput_sims_per_s_std"}
        ),
        on=["base_method", "k", "n_workers"],
        how="left",
    )
    if "worker_utilization" in throughput_df.columns:
        throughput_mean = throughput_mean.merge(
            _throughput_grp[["worker_utilization"]].mean().reset_index(),
            on=["base_method", "k", "n_workers"],
            how="left",
        ).merge(
            _throughput_grp[["worker_utilization"]].std().reset_index().rename(
                columns={"worker_utilization": "worker_utilization_std"}
            ),
            on=["base_method", "k", "n_workers"],
            how="left",
        )
    throughput_mean["efficiency"] = np.nan
    throughput_mean["efficiency_std"] = np.nan
    throughput_mean["ideal_throughput"] = np.nan
    for (method, k), group in throughput_mean.groupby(["base_method", "k"], sort=False):
        baseline = group.loc[group["n_workers"] == 1, "throughput_sims_per_s"]
        if baseline.empty or float(baseline.iloc[0]) <= 0:
            continue
        t1 = float(baseline.iloc[0])
        throughput_mean.loc[group.index, "efficiency"] = (
            group["throughput_sims_per_s"] / (t1 * group["n_workers"])
        )
        throughput_mean.loc[group.index, "efficiency_std"] = (
            group["throughput_sims_per_s_std"] / (t1 * group["n_workers"])
        )
        throughput_mean.loc[group.index, "ideal_throughput"] = t1 * group["n_workers"]

    _budget_stats_cols = [
        "attempts_by_budget",
        "posterior_samples_by_budget",
        "quality_wasserstein_by_budget",
        "best_tolerance_by_budget",
        "throughput_by_budget",
    ]
    _budget_grp = budget_df.groupby(["base_method", "k", "n_workers", "budget_s"])
    budget_mean = _budget_grp[_budget_stats_cols].mean().reset_index()
    budget_mean = budget_mean.merge(
        _budget_grp[_budget_stats_cols].std().reset_index().rename(
            columns={col: f"{col}_std" for col in _budget_stats_cols}
        ),
        on=["base_method", "k", "n_workers", "budget_s"],
        how="left",
    )

    for method in methods:
        safe_method = _scaling_token(method)
        method_throughput = throughput_mean.loc[throughput_mean["base_method"] == method].copy()
        if not method_throughput.empty:
            _plot_workers_lines(
                method_throughput,
                x_col="n_workers",
                y_col="throughput_sims_per_s",
                series_col="k",
                xlabel="workers",
                ylabel="throughput (sim/s)",
                title=f"Throughput vs workers: {method}",
                stem=output_dir.plots / f"throughput_vs_workers__{safe_method}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"throughput_vs_workers__{safe_method}",
                    title=f"Throughput vs workers: {method}",
                    methods=[method],
                    summary_plot=True,
                    extra={"group_by": "k", "paper_primary": True, "evidence_role": "hpc_primary"},
                ),
                yerr_col="throughput_sims_per_s_std",
                ideal_y_col="ideal_throughput",
                log_x=True,
            )
            _plot_workers_lines(
                method_throughput.dropna(subset=["efficiency"]),
                x_col="n_workers",
                y_col="efficiency",
                series_col="k",
                xlabel="workers",
                ylabel="parallel efficiency",
                title=f"Efficiency vs workers: {method}",
                stem=output_dir.plots / f"efficiency_vs_workers__{safe_method}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"efficiency_vs_workers__{safe_method}",
                    title=f"Efficiency vs workers: {method}",
                    methods=[method],
                    summary_plot=True,
                    extra={"group_by": "k", "paper_primary": True, "evidence_role": "hpc_primary"},
                ),
                yerr_col="efficiency_std",
                hline=1.0,
                log_x=True,
            )
            if "worker_utilization" in method_throughput.columns and method_throughput["worker_utilization"].notna().any():
                _plot_workers_lines(
                    method_throughput,
                    x_col="n_workers",
                    y_col="worker_utilization",
                    series_col="k",
                    xlabel="workers",
                    ylabel="worker utilization",
                    title=f"Worker utilization vs workers: {method}",
                    stem=output_dir.plots / f"worker_utilization_vs_workers__{safe_method}",
                    metadata=_nonbenchmark_plot_metadata(
                        output_dir,
                        plot_name=f"worker_utilization_vs_workers__{safe_method}",
                        title=f"Worker utilization vs workers: {method}",
                        methods=[method],
                        summary_plot=True,
                        extra={"group_by": "k", "paper_primary": True, "evidence_role": "hpc_primary"},
                    ),
                    yerr_col="worker_utilization_std",
                    hline=1.0,
                    log_x=True,
                )

        method_budget = budget_mean.loc[budget_mean["base_method"] == method].copy()
        for budget_s in sorted(method_budget["budget_s"].dropna().unique().tolist()):
            budget_subset = method_budget.loc[method_budget["budget_s"] == budget_s].copy()
            if budget_subset.empty:
                continue
            budget_label = _budget_token(float(budget_s))
            _plot_workers_lines(
                budget_subset,
                x_col="n_workers",
                y_col="quality_wasserstein_by_budget",
                series_col="k",
                xlabel="workers",
                ylabel="quality (Wasserstein)",
                title=f"Quality at fixed wall-clock budget T={budget_s:g}s: {method}",
                stem=output_dir.plots / f"quality_at_budget__{safe_method}__T{budget_label}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"quality_at_budget__{safe_method}__T{budget_label}",
                    title=f"Quality at fixed wall-clock budget T={budget_s:g}s: {method}",
                    methods=[method],
                    summary_plot=True,
                    extra={
                        "budget_s": float(budget_s),
                        "group_by": "k",
                        "paper_primary": True,
                        "evidence_role": "hpc_primary",
                        "performance_metric": "quality_at_fixed_wallclock_budget",
                    },
                ),
                yerr_col="quality_wasserstein_by_budget_std",
                log_x=True,
            )
            _plot_workers_lines(
                budget_subset,
                x_col="n_workers",
                y_col="attempts_by_budget",
                series_col="k",
                xlabel="workers",
                ylabel="attempts by budget",
                title=f"Attempts at T={budget_s:g}s: {method}",
                stem=output_dir.plots / f"attempts_at_budget__{safe_method}__T{budget_label}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"attempts_at_budget__{safe_method}__T{budget_label}",
                    title=f"Attempts at T={budget_s:g}s: {method}",
                    methods=[method],
                    summary_plot=True,
                    extra={
                        "budget_s": float(budget_s),
                        "group_by": "k",
                        "paper_primary": True,
                        "evidence_role": "hpc_primary",
                        "performance_metric": "attempts_at_fixed_wallclock_budget",
                    },
                ),
                yerr_col="attempts_by_budget_std",
                log_x=True,
            )
            _plot_heatmap(
                budget_subset,
                value_col="quality_wasserstein_by_budget",
                title=f"Quality heatmap at T={budget_s:g}s: {method}",
                stem=output_dir.plots / f"quality_heatmap__{safe_method}__T{budget_label}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"quality_heatmap__{safe_method}__T{budget_label}",
                    title=f"Quality heatmap at T={budget_s:g}s: {method}",
                    methods=[method],
                    summary_plot=True,
                    extra={"budget_s": float(budget_s), "axes": ["n_workers", "k"]},
                ),
            )
            _plot_heatmap(
                budget_subset,
                value_col="throughput_by_budget",
                title=f"Throughput heatmap at T={budget_s:g}s: {method}",
                stem=output_dir.plots / f"throughput_heatmap__{safe_method}__T{budget_label}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"throughput_heatmap__{safe_method}__T{budget_label}",
                    title=f"Throughput heatmap at T={budget_s:g}s: {method}",
                    methods=[method],
                    summary_plot=True,
                    extra={"budget_s": float(budget_s), "axes": ["n_workers", "k"]},
                ),
            )

        for n_workers in sorted(method_budget["n_workers"].dropna().unique().tolist()):
            worker_subset = method_budget.loc[method_budget["n_workers"] == n_workers].copy()
            if worker_subset.empty:
                continue
            _plot_workers_lines(
                worker_subset,
                x_col="budget_s",
                y_col="quality_wasserstein_by_budget",
                series_col="k",
                xlabel="wall-clock budget (s)",
                ylabel="quality (Wasserstein)",
                title=f"Quality vs time: {method}, w={int(n_workers)}",
                stem=output_dir.plots / f"quality_vs_time__{safe_method}__w{int(n_workers)}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"quality_vs_time__{safe_method}__w{int(n_workers)}",
                    title=f"Quality vs time: {method}, w={int(n_workers)}",
                    methods=[method],
                    summary_plot=True,
                    extra={"n_workers": int(n_workers), "group_by": "k"},
                ),
                yerr_col="quality_wasserstein_by_budget_std",
            )
            _plot_workers_lines(
                worker_subset,
                x_col="budget_s",
                y_col="attempts_by_budget",
                series_col="k",
                xlabel="wall-clock budget (s)",
                ylabel="attempts by budget",
                title=f"Attempts vs time: {method}, w={int(n_workers)}",
                stem=output_dir.plots / f"attempts_vs_time__{safe_method}__w{int(n_workers)}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"attempts_vs_time__{safe_method}__w{int(n_workers)}",
                    title=f"Attempts vs time: {method}, w={int(n_workers)}",
                    methods=[method],
                    summary_plot=True,
                    extra={"n_workers": int(n_workers), "group_by": "k"},
                ),
                yerr_col="attempts_by_budget_std",
            )

        # Key plot: quality vs wall-clock budget, one curve per worker count at fixed k.
        # Reading horizontally: time needed to reach quality Q with N workers.
        # Reading vertically: quality achieved given budget T with N workers.
        for k_val in sorted(method_budget["k"].dropna().unique().tolist()):
            k_subset = method_budget.loc[method_budget["k"] == k_val].copy()
            if k_subset.empty:
                continue
            k_label = _scaling_token(int(k_val))
            _plot_workers_lines(
                k_subset,
                x_col="budget_s",
                y_col="quality_wasserstein_by_budget",
                series_col="n_workers",
                xlabel="wall-clock budget (s)",
                ylabel="quality (Wasserstein)",
                title=f"Quality vs fixed wall-clock budget: {method}, k={int(k_val)}",
                stem=output_dir.plots / f"quality_vs_budget_by_workers__{safe_method}__k{k_label}",
                metadata=_nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=f"quality_vs_budget_by_workers__{safe_method}__k{k_label}",
                    title=f"Quality vs fixed wall-clock budget: {method}, k={int(k_val)}",
                    methods=[method],
                    summary_plot=True,
                    extra={
                        "k": int(k_val),
                        "group_by": "n_workers",
                        "paper_primary": True,
                        "evidence_role": "hpc_primary",
                        "performance_metric": "quality_at_fixed_wallclock_budget",
                    },
                ),
                yerr_col="quality_wasserstein_by_budget_std",
            )

    throughput_all = throughput_mean.copy()
    throughput_all["series"] = throughput_all.apply(
        lambda row: f"{row['base_method']} | k={int(row['k'])}",
        axis=1,
    )
    _plot_workers_lines(
        throughput_all,
        x_col="n_workers",
        y_col="throughput_sims_per_s",
        series_col="series",
        xlabel="workers",
        ylabel="throughput (sim/s)",
        title="Throughput vs workers: all methods, all k",
        stem=output_dir.plots / "throughput_vs_workers__all_methods_all_k",
        metadata=_nonbenchmark_plot_metadata(
            output_dir,
            plot_name="throughput_vs_workers__all_methods_all_k",
            title="Throughput vs workers: all methods, all k",
            methods=methods,
            summary_plot=True,
            extra={"group_by": "method,k"},
        ),
        log_x=True,
    )

    for budget_s in sorted(budget_mean["budget_s"].dropna().unique().tolist()):
        budget_subset = budget_mean.loc[budget_mean["budget_s"] == budget_s].copy()
        if budget_subset.empty:
            continue
        budget_label = _budget_token(float(budget_s))
        budget_subset["series"] = budget_subset.apply(
            lambda row: f"{row['base_method']} | k={int(row['k'])}",
            axis=1,
        )
        _plot_workers_lines(
            budget_subset,
            x_col="n_workers",
            y_col="quality_wasserstein_by_budget",
            series_col="series",
            xlabel="workers",
            ylabel="quality (Wasserstein)",
            title=f"Quality at fixed wall-clock budget T={budget_s:g}s: all methods, all k",
            stem=output_dir.plots / f"quality_at_budget__all_methods_all_k__T{budget_label}",
            metadata=_nonbenchmark_plot_metadata(
                output_dir,
                plot_name=f"quality_at_budget__all_methods_all_k__T{budget_label}",
                title=f"Quality at fixed wall-clock budget T={budget_s:g}s: all methods, all k",
                methods=methods,
                summary_plot=True,
                extra={"budget_s": float(budget_s), "group_by": "method,k"},
            ),
            log_x=True,
        )
        _plot_workers_lines(
            budget_subset,
            x_col="n_workers",
            y_col="attempts_by_budget",
            series_col="series",
            xlabel="workers",
            ylabel="attempts by budget",
            title=f"Attempts at T={budget_s:g}s: all methods, all k",
            stem=output_dir.plots / f"attempts_at_budget__all_methods_all_k__T{budget_label}",
            metadata=_nonbenchmark_plot_metadata(
                output_dir,
                plot_name=f"attempts_at_budget__all_methods_all_k__T{budget_label}",
                title=f"Attempts at T={budget_s:g}s: all methods, all k",
                methods=methods,
                summary_plot=True,
                extra={"budget_s": float(budget_s), "group_by": "method,k"},
            ),
            log_x=True,
        )
        _plot_quality_vs_attempts_scatter(
            budget_subset,
            stem=output_dir.plots / f"quality_vs_attempts_scatter__T{budget_label}",
            metadata=_nonbenchmark_plot_metadata(
                output_dir,
                plot_name=f"quality_vs_attempts_scatter__T{budget_label}",
                title=f"Quality vs attempts scatter at T={budget_s:g}s",
                methods=methods,
                summary_plot=True,
                extra={"budget_s": float(budget_s)},
            ),
            title=f"Quality vs attempts scatter at T={budget_s:g}s",
        )


def plot_sensitivity_summary(
    data_dir: Path,
    grid: Dict[str, List],
    output_dir: OutputDir,
    quality_df: Optional[pd.DataFrame] = None,
) -> None:
    """Heatmap of posterior quality (Wasserstein) across the sensitivity grid.

    Parameters
    ----------
    data_dir:
        Directory containing ``sensitivity_*.csv`` particle files.
    grid:
        The sensitivity grid dict ``{param: [values]}``.
    output_dir:
        Output directory for plots and data.
    quality_df:
        Pre-computed posterior quality summary from
        :func:`~async_abc.analysis.sensitivity.compute_sensitivity_quality_summary`.
        When provided, the heatmap uses ``wasserstein_mean`` as the cell value
        and ``wasserstein_std`` / ``n_replicates`` are recorded in the CSV and
        metadata.  When ``None``, falls back to the legacy mean-final-tolerance
        metric computed directly from the raw particle CSVs.
    """
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

    if quality_df is not None:
        # ------------------------------------------------------------------ #
        # New path: use pre-computed Wasserstein quality scores               #
        # ------------------------------------------------------------------ #
        # Persist the quality summary to the data directory for reproducibility.
        quality_df.to_csv(data_dir / "sensitivity_quality_summary.csv", index=False)

        base_meta = _nonbenchmark_plot_metadata(
            output_dir,
            plot_name="sensitivity_heatmap",
            title="Sensitivity heatmap",
            summary_plot=True,
        )
        n_rep_min = int(quality_df["n_replicates"].min()) if "n_replicates" in quality_df.columns else 0
        base_meta.setdefault("extra", {})["n_replicates_min"] = n_rep_min

        if scheduler_key and scheduler_vals is not None:
            # One figure per scheduler value — keeps scheduler as a separate
            # comparison axis rather than a visual facet column.
            for sv in scheduler_vals:
                sched_stem = output_dir.plots / f"sensitivity_heatmap__scheduler_type={sv}"
                subset = quality_df[quality_df[scheduler_key].astype(str) == str(sv)]
                plot_name = f"sensitivity_heatmap__scheduler_type={sv}"
                meta = _nonbenchmark_plot_metadata(
                    output_dir,
                    plot_name=plot_name,
                    title=f"Sensitivity heatmap — {scheduler_key}={sv}",
                    summary_plot=True,
                    extra={"scheduler_type": str(sv), "n_replicates_min": n_rep_min},
                )
                _emit_sensitivity_heatmap(
                    subset,
                    row_key=row_key, col_key=col_key, facet_key=facet_key,
                    row_vals=row_vals, col_vals=col_vals, facet_vals=facet_vals,
                    path_stem=sched_stem,
                    meta_path=output_dir.plots / f"{plot_name}_meta.json",
                    extra_meta=meta,
                )
        else:
            _emit_sensitivity_heatmap(
                quality_df,
                row_key=row_key, col_key=col_key, facet_key=facet_key,
                row_vals=row_vals, col_vals=col_vals, facet_vals=facet_vals,
                path_stem=output_dir.plots / "sensitivity_heatmap",
                meta_path=output_dir.plots / "sensitivity_heatmap_meta.json",
                extra_meta=base_meta,
            )
        return

    # ---------------------------------------------------------------------- #
    # Legacy fallback: mean-final-tolerance computed from raw particle CSVs  #
    # ---------------------------------------------------------------------- #
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

        stem_str = csv_path.stem[len("sensitivity_"):]
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


def _emit_sensitivity_heatmap(
    quality_df: pd.DataFrame,
    *,
    row_key: str,
    col_key: str,
    facet_key: Optional[str],
    row_vals: List,
    col_vals: List,
    facet_vals: Optional[List],
    path_stem: Path,
    meta_path: Path,
    extra_meta: Dict,
) -> None:
    """Build and save a single sensitivity heatmap figure from a quality_df slice.

    Uses ``wasserstein_mean`` for cell values and includes ``wasserstein_std``
    and ``n_replicates`` in the exported data CSV.  Calls :func:`sensitivity_heatmap`
    for the visual figure, then rewrites the data CSV in tidy format.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_labels = [str(v) for v in row_vals]
    col_labels = [str(v) for v in col_vals]

    # ------------------------------------------------------------------ #
    # Build mean and std matrices                                          #
    # ------------------------------------------------------------------ #
    def _get(rv, cv, fv=None) -> tuple[float, float, int]:
        """Return (mean, std, n_rep) for a cell."""
        if quality_df.empty:
            return float("nan"), float("nan"), 0
        mask = (
            (quality_df[row_key].astype(str) == str(rv)) &
            (quality_df[col_key].astype(str) == str(cv))
        )
        if fv is not None and facet_key:
            mask &= quality_df[facet_key].astype(str) == str(fv)
        subset = quality_df[mask]
        if subset.empty:
            return float("nan"), float("nan"), 0
        mean = float(subset["wasserstein_mean"].iloc[0]) if "wasserstein_mean" in subset.columns else float("nan")
        std = float(subset["wasserstein_std"].iloc[0]) if "wasserstein_std" in subset.columns else float("nan")
        n_rep = int(subset["n_replicates"].iloc[0]) if "n_replicates" in subset.columns else 0
        return mean, std, n_rep

    if facet_key and facet_vals is not None:
        mean_mat = np.full((len(facet_vals), len(row_vals), len(col_vals)), np.nan)
        std_mat = np.full_like(mean_mat, np.nan)
        tidy_rows: list[dict] = []
        for f_idx, fv in enumerate(facet_vals):
            for i, rv in enumerate(row_vals):
                for j, cv in enumerate(col_vals):
                    m, s, n = _get(rv, cv, fv)
                    mean_mat[f_idx, i, j] = m
                    std_mat[f_idx, i, j] = s
                    tidy_rows.append({
                        "facet": str(fv),
                        row_key: str(rv),
                        col_key: str(cv),
                        "wasserstein_mean": m,
                        "wasserstein_std": s,
                        "n_replicates": n,
                    })
        if not np.isfinite(mean_mat).any():
            return
        paths = sensitivity_heatmap(
            mean_mat,
            row_labels=row_labels,
            col_labels=col_labels,
            path_stem=path_stem,
            facet_labels=[str(v) for v in facet_vals],
        )
    else:
        mean_mat = np.full((len(row_vals), len(col_vals)), np.nan)
        std_mat = np.full_like(mean_mat, np.nan)
        tidy_rows = []
        for i, rv in enumerate(row_vals):
            for j, cv in enumerate(col_vals):
                m, s, n = _get(rv, cv)
                mean_mat[i, j] = m
                std_mat[i, j] = s
                tidy_rows.append({
                    row_key: str(rv),
                    col_key: str(cv),
                    "wasserstein_mean": m,
                    "wasserstein_std": s,
                    "n_replicates": n,
                })
        if not np.isfinite(mean_mat).any():
            return
        paths = sensitivity_heatmap(
            mean_mat,
            row_labels=row_labels,
            col_labels=col_labels,
            path_stem=path_stem,
        )

    # Rewrite the data CSV in tidy format with wasserstein_std included.
    csv_path = paths.get("csv")
    if csv_path and tidy_rows:
        fieldnames = list(tidy_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tidy_rows)

    _merge_metadata(paths.get("meta", meta_path), extra_meta)


def plot_ablation_summary(
    data_dir: Path,
    variants: List[Dict[str, Any]],
    output_dir: OutputDir,
    benchmark_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Bar chart comparing final posterior quality across ablation variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_names = [v.get("name", f"v{i}") for i, v in enumerate(variants)]
    benchmark_cfg = benchmark_cfg or {}
    variant_cfg_by_name = {v.get("name", f"v{i}"): v for i, v in enumerate(variants)}
    mean_wass = []
    ci_lows = []
    ci_highs = []
    n_values = []

    for name in variant_names:
        csv_path = data_dir / f"ablation_{name}.csv"
        if not csv_path.exists():
            mean_wass.append(float("nan"))
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))
            n_values.append(0)
            continue
        records = load_records(csv_path)
        true_params = _true_params_from_cfg(records, benchmark_cfg)
        archive_size = variant_cfg_by_name.get(name, {}).get("k")
        if not records or not true_params:
            mean_wass.append(float("nan"))
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))
            n_values.append(0)
            continue

        wass_per_rep: list[float] = []
        for result in final_state_results(records, archive_size=archive_size):
            samples = []
            for record in result.records:
                if all(param in record.params for param in true_params):
                    samples.append([float(record.params[param]) for param in true_params])
            if not samples:
                continue
            sample_arr = np.asarray(samples, dtype=float)
            target = np.tile(
                np.asarray([float(true_params[param]) for param in true_params], dtype=float),
                (len(sample_arr), 1),
            )
            if sample_arr.shape[1] == 1:
                from scipy.stats import wasserstein_distance

                quality = float(wasserstein_distance(sample_arr[:, 0], target[:, 0]))
            else:
                try:
                    import ot

                    quality = float(ot.sliced_wasserstein_distance(sample_arr, target, n_projections=50))
                except ImportError:
                    from scipy.stats import wasserstein_distance

                    quality = float(np.mean([
                        wasserstein_distance(sample_arr[:, idx], target[:, idx])
                        for idx in range(sample_arr.shape[1])
                    ]))
            wass_per_rep.append(quality)

        if wass_per_rep:
            values = np.asarray(wass_per_rep, dtype=float)
            mean, ci_low, ci_high = _summarize_scalar(values, ci_level=0.95)
            mean_wass.append(mean)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
            n_values.append(int(np.isfinite(values).sum()))
        else:
            mean_wass.append(float("nan"))
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))
            n_values.append(0)

    fig, ax = plt.subplots(figsize=(max(5, len(variant_names) * 1.2), 4))
    x = np.arange(len(variant_names))
    ax.bar(x, mean_wass, color="steelblue", alpha=0.8)
    for idx, (mean, ci_low, ci_high, n_obs) in enumerate(zip(mean_wass, ci_lows, ci_highs, n_values)):
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
    ax.set_ylabel("mean final Wasserstein")
    ax.set_title("Ablation comparison")
    fig.tight_layout()

    data = {
        "variant": variant_names,
        "mean_final_wasserstein": mean_wass,
        "mean_final_wasserstein_ci_low": ci_lows,
        "mean_final_wasserstein_ci_high": ci_highs,
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
    if benchmark_cfg.get("name") == "gaussian_mean":
        write_gaussian_analytic_summary(
            records,
            cfg=cfg,
            output_dir=output_dir,
            archive_size=archive_size,
        )
    audit_df = _write_benchmark_audit(
        records,
        true_params=true_params,
        output_dir=output_dir,
        archive_size=archive_size,
        min_particles_for_threshold=min_particles_for_threshold,
    )

    if plots_cfg.get("posterior"):
        plot_posterior(records, output_dir, cfg=cfg, archive_size=archive_size)
    if plots_cfg.get("archive_evolution") and emit_paper:
        plot_archive_evolution(records, output_dir, cfg=cfg)
    if plots_cfg.get("archive_evolution") and emit_diagnostics:
        plot_archive_evolution_diagnostic(records, output_dir, cfg=cfg)
    if plots_cfg.get("corner"):
        plot_corner(
            records,
            param_names=_param_names(records),
            output_dir=output_dir,
            true_params=true_params,
            cfg=cfg,
            archive_size=archive_size,
        )
    if plots_cfg.get("tolerance_trajectory") and emit_paper:
        plot_tolerance_trajectory(
            records,
            output_dir,
            cfg=cfg,
            true_params=true_params,
            archive_size=archive_size,
            checkpoint_count=len(_default_checkpoint_steps(records)) if _default_checkpoint_steps(records) else 8,
            ci_level=ci_level,
        )
    if plots_cfg.get("tolerance_trajectory") and emit_diagnostics:
        plot_tolerance_trajectory_diagnostic(
            records,
            output_dir,
            cfg=cfg,
            true_params=true_params,
            archive_size=archive_size,
            checkpoint_count=len(_default_checkpoint_steps(records)) if _default_checkpoint_steps(records) else 8,
        )
    if plots_cfg.get("quality_vs_time"):
        target = float(analysis_cfg.get("target_wasserstein", 1.0))
        checkpoint_count = len(_default_checkpoint_steps(records)) if _default_checkpoint_steps(records) else 8
        if emit_paper:
            plot_progress_summary(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
                ci_level=ci_level,
                audit_df=audit_df,
            )
            plot_quality_vs_wall_time(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
                ci_level=ci_level,
                audit_df=audit_df,
            )
            plot_quality_vs_posterior_samples(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                ci_level=ci_level,
                audit_df=audit_df,
            )
            plot_quality_vs_attempt_budget(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                ci_level=ci_level,
                audit_df=audit_df,
            )
            plot_time_to_target_summary(
                records,
                true_params=true_params,
                target_wasserstein=target,
                output_dir=output_dir,
                cfg=cfg,
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
                cfg=cfg,
                archive_size=archive_size,
                ci_level=ci_level,
                min_particles_for_threshold=min_particles_for_threshold,
                audit_df=audit_df,
            )
        if emit_diagnostics:
            plot_progress_diagnostic(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
            )
            plot_quality_vs_wall_time_diagnostic(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
                audit_df=audit_df,
            )
            plot_quality_vs_posterior_samples_diagnostic(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
            )
            plot_quality_vs_attempt_budget_diagnostic(
                records,
                true_params=true_params,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                checkpoint_count=checkpoint_count,
            )
            plot_time_to_target_diagnostic(
                records,
                true_params=true_params,
                target_wasserstein=target,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                min_particles_for_threshold=min_particles_for_threshold,
            )
            plot_attempts_to_target_diagnostic(
                records,
                true_params=true_params,
                target_wasserstein=target,
                output_dir=output_dir,
                cfg=cfg,
                archive_size=archive_size,
                min_particles_for_threshold=min_particles_for_threshold,
            )
 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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
    sk_map = _build_state_kind_map(quality_df)
    for method, group in summary_df.groupby("method", sort=True):
        group = group.sort_values("axis_value")
        sk_suffix = f" [{sk_map[method]}]" if method in sk_map else ""
        ax.plot(group["axis_value"], group["wasserstein"], linewidth=1.8, label=f"{method}{sk_suffix}")
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
    """Extract true parameter values from benchmark config using param names.

    Emits a warning when the config contains ``true_*`` numeric keys that do not
    match any inferred parameter column.  This catches naming mismatches (e.g.
    ``true_division_rate_normalized`` vs the inferred column ``division_rate``)
    that would otherwise cause quality-vs-time plots to be silently skipped.
    """
    inferred_names = set(_param_names(records))
    true_params: Dict[str, float] = {}
    for param in inferred_names:
        key = f"true_{param}"
        if key in benchmark_cfg:
            true_params[param] = float(benchmark_cfg[key])

    # Warn about config true_* keys that have no matching inferred column.
    if inferred_names:
        unmapped = [
            key
            for key, val in benchmark_cfg.items()
            if key.startswith("true_")
            and isinstance(val, (int, float))
            and key[len("true_"):] not in inferred_names
        ]
        if unmapped:
            logger.warning(
                "_true_params_from_cfg: benchmark config has true_* keys %s that do not "
                "match any inferred parameter column %s; quality plots will be skipped. "
                "Rename the config keys to match the inferred column names.",
                unmapped,
                sorted(inferred_names),
            )

    return true_params


def _default_checkpoint_steps(records: List[ParticleRecord], count: int = 8) -> List[int]:
    """Choose a small set of evenly spaced simulation checkpoints."""
    max_step = max((int(r.step) for r in records), default=0)
    if max_step <= 0:
        return []
    n = min(count, max_step)
    return sorted({int(round(x)) for x in np.linspace(1, max_step, num=n)})
