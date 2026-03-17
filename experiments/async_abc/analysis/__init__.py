"""Analysis helpers for post-processing ParticleRecord outputs."""

from .barrier import barrier_overhead_fraction, generation_spans
from .convergence import time_to_threshold, wasserstein_at_checkpoints
from .ess import compute_ess, ess_over_time
from .trajectory import loss_over_steps, tolerance_over_wall_time

__all__ = [
    "barrier_overhead_fraction",
    "compute_ess",
    "ess_over_time",
    "generation_spans",
    "loss_over_steps",
    "time_to_threshold",
    "tolerance_over_wall_time",
    "wasserstein_at_checkpoints",
]
