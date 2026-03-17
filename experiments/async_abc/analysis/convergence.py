"""Posterior convergence summaries."""

import numpy as np
import ot
import pandas as pd
from scipy.stats import wasserstein_distance

from ._helpers import records_to_frame


def _wasserstein_to_true_params(
    frame: pd.DataFrame,
    true_params: dict[str, float],
    n_projections: int,
) -> float:
    param_names = list(true_params.keys())
    samples = frame[param_names].to_numpy(dtype=float)
    target = np.tile(
        np.asarray([true_params[name] for name in param_names], dtype=float),
        (len(frame), 1),
    )
    if len(param_names) == 1:
        return float(wasserstein_distance(samples[:, 0], target[:, 0]))
    return float(
        ot.sliced_wasserstein_distance(
            samples,
            target,
            n_projections=n_projections,
        )
    )


def wasserstein_at_checkpoints(
    records,
    true_params: dict[str, float],
    checkpoint_steps: list[int],
    n_projections: int = 50,
) -> pd.DataFrame:
    """Compute Wasserstein distance at requested evaluation checkpoints."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["method", "replicate", "step", "wall_time", "wasserstein"]
        )

    rows = []
    for (method, replicate), group in frame.groupby(["method", "replicate"], sort=True):
        group = group.sort_values("step")
        for checkpoint in sorted(set(checkpoint_steps)):
            subset = group[group["step"] <= checkpoint]
            if subset.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "step": checkpoint,
                    "wall_time": float(subset["wall_time"].max()),
                    "wasserstein": _wasserstein_to_true_params(
                        subset,
                        true_params,
                        n_projections=n_projections,
                    ),
                }
            )
    return pd.DataFrame(rows)


def time_to_threshold(
    records,
    true_params: dict[str, float],
    target_wasserstein: float,
) -> pd.DataFrame:
    """Return the earliest wall time at which each run reaches a target distance."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["method", "replicate", "wall_time_to_threshold"]
        )

    rows = []
    for (method, replicate), group in frame.groupby(["method", "replicate"], sort=True):
        group = group.sort_values("step")
        wall_time_to_threshold = np.nan
        for step in group["step"].tolist():
            subset = group[group["step"] <= step]
            distance = _wasserstein_to_true_params(
                subset,
                true_params,
                n_projections=50,
            )
            if distance <= target_wasserstein:
                wall_time_to_threshold = float(subset["wall_time"].max())
                break
        rows.append(
            {
                "method": method,
                "replicate": replicate,
                "wall_time_to_threshold": wall_time_to_threshold,
            }
        )
    return pd.DataFrame(rows)
