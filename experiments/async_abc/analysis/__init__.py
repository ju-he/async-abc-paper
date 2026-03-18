"""Analysis helpers for post-processing ParticleRecord outputs."""

from importlib import import_module

_EXPORTS = {
    "barrier_overhead_fraction": (".barrier", "barrier_overhead_fraction"),
    "generation_spans": (".barrier", "generation_spans"),
    "time_to_threshold": (".convergence", "time_to_threshold"),
    "wasserstein_at_checkpoints": (".convergence", "wasserstein_at_checkpoints"),
    "compute_ess": (".ess", "compute_ess"),
    "ess_over_time": (".ess", "ess_over_time"),
    "compute_rank": (".sbc", "compute_rank"),
    "empirical_coverage": (".sbc", "empirical_coverage"),
    "sbc_ranks": (".sbc", "sbc_ranks"),
    "loss_over_steps": (".trajectory", "loss_over_steps"),
    "tolerance_over_wall_time": (".trajectory", "tolerance_over_wall_time"),
}

__all__ = [
    "barrier_overhead_fraction",
    "compute_ess",
    "compute_rank",
    "empirical_coverage",
    "ess_over_time",
    "generation_spans",
    "loss_over_steps",
    "sbc_ranks",
    "time_to_threshold",
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
