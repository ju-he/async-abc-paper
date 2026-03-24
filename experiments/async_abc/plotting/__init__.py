"""Plotting utilities for the async-ABC experiments."""

from importlib import import_module

_EXPORTS = {
    "save_figure": (".export", "save_figure"),
    "get_git_hash": (".export", "get_git_hash"),
    "corner_plot": (".common", "corner_plot"),
    "gantt_plot": (".common", "gantt_plot"),
    "posterior_plot": (".common", "posterior_plot"),
    "posterior_quality_plot": (".common", "posterior_quality_plot"),
    "quality_vs_time_plot": (".common", "quality_vs_time_plot"),
    "scaling_plot": (".common", "scaling_plot"),
    "threshold_summary_plot": (".common", "threshold_summary_plot"),
    "tolerance_trajectory_plot": (".common", "tolerance_trajectory_plot"),
    "archive_evolution_plot": (".common", "archive_evolution_plot"),
    "sensitivity_heatmap": (".common", "sensitivity_heatmap"),
    "compute_wasserstein": (".common", "compute_wasserstein"),
    "plot_benchmark_diagnostics": (".reporters", "plot_benchmark_diagnostics"),
    "plot_corner": (".reporters", "plot_corner"),
    "plot_posterior": (".reporters", "plot_posterior"),
    "plot_archive_evolution": (".reporters", "plot_archive_evolution"),
    "plot_attempts_to_target_summary": (".reporters", "plot_attempts_to_target_summary"),
    "plot_quality_vs_attempt_budget": (".reporters", "plot_quality_vs_attempt_budget"),
    "plot_quality_vs_posterior_samples": (".reporters", "plot_quality_vs_posterior_samples"),
    "plot_quality_vs_wall_time_diagnostic": (".reporters", "plot_quality_vs_wall_time_diagnostic"),
    "plot_quality_vs_time": (".reporters", "plot_quality_vs_time"),
    "plot_scaling_summary": (".reporters", "plot_scaling_summary"),
    "plot_sensitivity_summary": (".reporters", "plot_sensitivity_summary"),
    "plot_time_to_target_summary": (".reporters", "plot_time_to_target_summary"),
    "plot_tolerance_trajectory": (".reporters", "plot_tolerance_trajectory"),
    "plot_worker_gantt": (".reporters", "plot_worker_gantt"),
    "plot_ablation_summary": (".reporters", "plot_ablation_summary"),
}

__all__ = [
    "save_figure",
    "get_git_hash",
    "corner_plot",
    "gantt_plot",
    "posterior_plot",
    "posterior_quality_plot",
    "quality_vs_time_plot",
    "scaling_plot",
    "threshold_summary_plot",
    "tolerance_trajectory_plot",
    "archive_evolution_plot",
    "sensitivity_heatmap",
    "compute_wasserstein",
    "plot_benchmark_diagnostics",
    "plot_corner",
    "plot_posterior",
    "plot_archive_evolution",
    "plot_attempts_to_target_summary",
    "plot_quality_vs_attempt_budget",
    "plot_quality_vs_posterior_samples",
    "plot_quality_vs_wall_time_diagnostic",
    "plot_quality_vs_time",
    "plot_scaling_summary",
    "plot_sensitivity_summary",
    "plot_time_to_target_summary",
    "plot_tolerance_trajectory",
    "plot_worker_gantt",
    "plot_ablation_summary",
]


def __getattr__(name: str):
    """Load plotting helpers lazily so package imports stay lightweight."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
