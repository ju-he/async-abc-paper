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


def ess_vs_n_at_fixed_S(
    records,
    *,
    window: int = 100,
    method_label: str | None = None,
) -> pd.DataFrame:
    """Sliding-window relative ESS as a function of total evaluations (W3.2).

    Computes ESS within a sliding window of size ``window`` over the
    cumulative stream of evaluated particles. Each row reports the
    *relative* ESS (``ESS / window``) so curves from different methods or
    snapshot-buffer sizes are directly comparable.

    This is the empirical instrument for paper §4 Condition (C4): a
    streaming-AMIS run at fixed snapshot-buffer size ``S`` should produce
    a relative ESS that **stabilises** as ``n`` grows. Plotting curves
    across a sweep of ``S`` values (paper §17 — "snapshot buffer size is
    fixed") shows whether the empirical (C4) holds at the chosen ``S``
    or whether ``S`` needs to scale with ``n``.

    Parameters
    ----------
    records:
        Iterable of :class:`ParticleRecord` (or pandas-row-like dicts).
    window:
        Sliding-window size for the ESS estimator. Larger windows give
        smoother curves; smaller windows track local degeneracy.
    method_label:
        If given, filter records to this method first. Otherwise process
        all methods; the returned frame has a ``method`` column.

    Returns
    -------
    pd.DataFrame
        Columns: ``method``, ``replicate``, ``n``, ``ess``, ``relative_ess``.
        ``n`` is the index of the right edge of the sliding window.
    """
    frame = records_to_frame(records)
    columns = ["method", "replicate", "n", "ess", "relative_ess"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if method_label is not None:
        frame = frame[frame["method"] == method_label]
        if frame.empty:
            return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (method, replicate), group in frame.groupby(["method", "replicate"], sort=True):
        ordered = group.sort_values("step")
        weights = ordered["weight"].fillna(1.0).to_numpy(dtype=float)
        if weights.size == 0:
            continue
        for end in range(1, len(weights) + 1):
            start = max(0, end - window)
            w = weights[start:end]
            ess = compute_ess(w)
            rel = ess / float(w.size) if w.size > 0 else float("nan")
            rows.append(
                {
                    "method": method,
                    "replicate": int(replicate),
                    "n": int(end),
                    "ess": float(ess),
                    "relative_ess": float(rel),
                }
            )
    return pd.DataFrame(rows, columns=columns)
