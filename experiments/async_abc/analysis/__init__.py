"""Analysis helpers for post-processing ParticleRecord outputs."""

from importlib import import_module

_EXPORTS = {
    "base_method_name": (".final_state", "base_method_name"),
    "barrier_overhead_fraction": (".barrier", "barrier_overhead_fraction"),
    "final_state_records": (".final_state", "final_state_records"),
    "final_state_results": (".final_state", "final_state_results"),
    "generation_spans": (".barrier", "generation_spans"),
    "posterior_quality_curve": (".convergence", "posterior_quality_curve"),
    "time_to_threshold": (".convergence", "time_to_threshold"),
    "wasserstein_at_checkpoints": (".convergence", "wasserstein_at_checkpoints"),
    "compute_ess": (".ess", "compute_ess"),
    "ess_over_time": (".ess", "ess_over_time"),
    "compute_rank": (".sbc", "compute_rank"),
    "empirical_coverage": (".sbc", "empirical_coverage"),
    "sbc_ranks": (".sbc", "sbc_ranks"),
    "loss_over_steps": (".trajectory", "loss_over_steps"),
    "tolerance_over_wall_time": (".trajectory", "tolerance_over_wall_time"),
    "tolerance_over_attempts": (".trajectory", "tolerance_over_attempts"),
}

__all__ = [
    "base_method_name",
    "barrier_overhead_fraction",
    "compute_ess",
    "compute_rank",
    "empirical_coverage",
    "ess_over_time",
    "final_state_records",
    "final_state_results",
    "generation_spans",
    "loss_over_steps",
    "posterior_quality_curve",
    "sbc_ranks",
    "time_to_threshold",
    "tolerance_over_attempts",
    "tolerance_over_wall_time",
    "wasserstein_at_checkpoints",
]


def __getattr__(name: str):
    """Load analysis helpers lazily so optional deps stay module-local."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
