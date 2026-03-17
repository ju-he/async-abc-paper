"""Effective sample size utilities."""

import numpy as np
import pandas as pd

from ._helpers import records_to_frame


def compute_ess(weights: np.ndarray) -> float:
    """Compute ESS = (sum(w))^2 / sum(w^2) for unnormalized weights."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return 0.0
    denom = float(np.square(w).sum())
    if denom == 0.0:
        return 0.0
    total = float(w.sum())
    return (total * total) / denom


def ess_over_time(records, method: str) -> pd.DataFrame:
    """Return cumulative ESS by step for a single method."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["method", "replicate", "step", "ess"])

    frame = frame[frame["method"] == method].sort_values(["replicate", "step"])
    rows = []
    for replicate, group in frame.groupby("replicate", sort=True):
        weights = []
        for row in group.itertuples(index=False):
            weights.append(1.0 if pd.isna(row.weight) else float(row.weight))
            rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "step": int(row.step),
                    "ess": compute_ess(np.asarray(weights, dtype=float)),
                }
            )
    return pd.DataFrame(rows)
