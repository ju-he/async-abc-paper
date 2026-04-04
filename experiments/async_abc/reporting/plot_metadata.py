"""Plot metadata helpers for report generation."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..io.paths import OutputDir
from ..utils.metadata import (
    infer_experiment_role,
    infer_method_comparison_roles,
    infer_stop_policy,
    infer_stop_policy_by_method,
)

_BENCHMARK_SEMANTICS: dict[str, dict[str, object]] = {
    "posterior": {
        "summary_plot": True,
        "paper_primary": True,
        "validity_metric": "posterior_recovery",
    },
    "corner": {
        "summary_plot": True,
        "paper_primary": True,
        "validity_metric": "posterior_recovery",
    },
    "archive_evolution": {
        "summary_plot": True,
        "validity_metric": "tolerance_progress",
        "performance_metric": "attempt_budget_progress",
    },
    "archive_evolution_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "tolerance_progress",
        "performance_metric": "attempt_budget_progress",
    },
    "tolerance_trajectory": {
        "summary_plot": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "tolerance_trajectory_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "progress_summary": {
        "summary_plot": True,
        "paper_primary": True,
        "validity_metric": "final_posterior_quality",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "progress_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "final_posterior_quality",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "quality_vs_wall_time": {
        "summary_plot": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "quality_vs_wall_time_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "quality_vs_posterior_samples": {
        "summary_plot": True,
        "paper_primary": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "posterior_sample_efficiency",
    },
    "quality_vs_posterior_samples_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "posterior_sample_efficiency",
    },
    "quality_vs_attempt_budget": {
        "summary_plot": True,
        "paper_primary": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "attempt_budget_progress",
    },
    "quality_vs_attempt_budget_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "wasserstein_to_true_params",
        "performance_metric": "attempt_budget_progress",
    },
    "time_to_target_summary": {
        "summary_plot": True,
        "validity_metric": "time_to_target_wasserstein",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "time_to_target_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "time_to_target_wasserstein",
        "performance_metric": "wall_clock_progress_supporting",
    },
    "attempts_to_target_summary": {
        "summary_plot": True,
        "validity_metric": "attempts_to_target_wasserstein",
        "performance_metric": "attempt_budget_progress",
    },
    "attempts_to_target_diagnostic": {
        "diagnostic_plot": True,
        "validity_metric": "attempts_to_target_wasserstein",
        "performance_metric": "attempt_budget_progress",
    },
}


def load_run_metadata(output_dir: OutputDir) -> Dict[str, Any]:
    """Return persisted run metadata when available."""
    meta_path = output_dir.data / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def nonbenchmark_plot_metadata(
    output_dir: OutputDir,
    *,
    plot_name: str,
    title: str,
    methods: list[str] | None = None,
    summary_plot: bool = False,
    diagnostic_plot: bool = False,
    skip_reason: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build plot metadata for non-benchmark figures from persisted run metadata."""
    persisted_meta = load_run_metadata(output_dir)
    metadata: Dict[str, Any] = {
        "plot_name": plot_name,
        "title": title,
        "summary_plot": bool(summary_plot),
        "diagnostic_plot": bool(diagnostic_plot),
        "experiment_name": output_dir.root.name,
        "benchmark": False,
        "methods": sorted(methods or []),
        "experiment_role": persisted_meta.get(
            "experiment_role",
            infer_experiment_role({"experiment_name": str(output_dir.root.name)}),
        ),
        "stop_policy": persisted_meta.get("stop_policy"),
        "stop_policy_by_method": persisted_meta.get("stop_policy_by_method", {}),
        "method_comparison_roles": persisted_meta.get("method_comparison_roles", {}),
        "wall_time_limit_s": persisted_meta.get("wall_time_limit_s"),
        "wall_time_budgets_s": persisted_meta.get("wall_time_budgets_s"),
        "n_replicates_observed": persisted_meta.get("n_replicates_observed"),
    }
    if skip_reason:
        metadata["skip_reason"] = skip_reason
    if extra:
        metadata.update(extra)
    return metadata


def benchmark_plot_metadata(
    cfg: Dict[str, Any],
    *,
    plot_name: str,
    output_dir: OutputDir | None = None,
    title: str | None = None,
    summary_plot: bool | None = None,
    diagnostic_plot: bool | None = None,
    paper_primary: bool | None = None,
    validity_metric: str | None = None,
    performance_metric: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return complete benchmark plot metadata with per-plot semantics."""
    semantics = dict(_BENCHMARK_SEMANTICS.get(_canonical_plot_name(plot_name), {}))

    if summary_plot is not None:
        semantics["summary_plot"] = bool(summary_plot)
    if diagnostic_plot is not None:
        semantics["diagnostic_plot"] = bool(diagnostic_plot)
    if paper_primary is not None:
        semantics["paper_primary"] = bool(paper_primary)
    if validity_metric is not None:
        semantics["validity_metric"] = validity_metric
    if performance_metric is not None:
        semantics["performance_metric"] = performance_metric

    is_primary = bool(semantics.get("paper_primary"))
    metadata: Dict[str, Any] = {
        "plot_name": plot_name,
        "benchmark": True,
        "summary_plot": bool(semantics.get("summary_plot", False)),
        "diagnostic_plot": bool(semantics.get("diagnostic_plot", False)),
        "experiment_name": cfg.get("experiment_name"),
        "experiment_role": infer_experiment_role(cfg),
        "stop_policy": infer_stop_policy(cfg),
        "stop_policy_by_method": infer_stop_policy_by_method(cfg),
        "method_comparison_roles": infer_method_comparison_roles(cfg),
        "wall_time_limit_s": cfg.get("inference", {}).get("max_wall_time_s"),
        "wall_time_budgets_s": cfg.get("scaling", {}).get("wall_time_budgets_s"),
        "validity_metric": semantics.get("validity_metric"),
        "performance_metric": semantics.get("performance_metric"),
        "paper_primary": is_primary,
        "evidence_role": "validity_primary" if is_primary else "validity_supporting",
    }
    if title is not None:
        metadata["title"] = title
    metadata.update(_benchmark_specific_extras(cfg, plot_name=plot_name, output_dir=output_dir))
    if extra:
        metadata.update(extra)
    return metadata


def _canonical_plot_name(plot_name: str) -> str:
    if plot_name.startswith("posterior_"):
        return "posterior"
    return plot_name


def _benchmark_specific_extras(
    cfg: Dict[str, Any],
    *,
    plot_name: str,
    output_dir: OutputDir | None,
) -> Dict[str, Any]:
    benchmark_name = str(cfg.get("benchmark", {}).get("name", ""))
    if benchmark_name != "gaussian_mean":
        return {}
    if _canonical_plot_name(plot_name) not in {"posterior", "corner", "progress_summary"}:
        return {}
    analytic_available = False
    if output_dir is not None:
        analytic_available = (output_dir.data / "gaussian_analytic_summary.csv").exists()
    return {
        "analytic_reference_available": analytic_available,
        "validity_metric": "analytic_posterior_mean_error",
    }
