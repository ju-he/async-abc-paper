"""Simple trajectory summaries over step and wall-clock time."""

import pandas as pd

from . import base_method_name
from ._helpers import records_to_frame


def tolerance_over_wall_time(records) -> pd.DataFrame:
    """Return tolerance trajectories over wall-clock time."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["method", "replicate", "wall_time", "tolerance"])
    cols = ["method", "replicate", "wall_time", "tolerance"]
    result = (
        frame.loc[frame["tolerance"].notna(), cols]
        .sort_values(["method", "replicate", "wall_time", "tolerance"])
        .reset_index(drop=True)
    )
    result = result.drop_duplicates(
        subset=["method", "replicate", "wall_time", "tolerance"],
        keep="last",
    ).reset_index(drop=True)
    if result.empty:
        return result
    for (method, replicate), group in result.groupby(["method", "replicate"], sort=False):
        if base_method_name(str(method)) != "async_propulate_abc":
            continue
        result.loc[group.index, "tolerance"] = group["tolerance"].cummin().to_numpy(dtype=float)
    return result.reset_index(drop=True)


def tolerance_over_attempts(records) -> pd.DataFrame:
    """Return tolerance trajectories over attempt count or step."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["method", "replicate", "attempt_count", "tolerance"])
    if "attempt_count" not in frame.columns:
        frame["attempt_count"] = frame["step"]
    result = frame.loc[frame["tolerance"].notna(), ["method", "replicate", "attempt_count", "tolerance"]].copy()
    result["attempt_count"] = pd.to_numeric(result["attempt_count"], errors="coerce").fillna(0).astype(int)
    result["tolerance"] = pd.to_numeric(result["tolerance"], errors="coerce")
    result = result.sort_values(["method", "replicate", "attempt_count", "tolerance"]).reset_index(drop=True)
    result = result.drop_duplicates(
        subset=["method", "replicate", "attempt_count", "tolerance"],
        keep="last",
    ).reset_index(drop=True)
    if result.empty:
        return result
    for (method, replicate), group in result.groupby(["method", "replicate"], sort=False):
        if base_method_name(str(method)) != "async_propulate_abc":
            continue
        result.loc[group.index, "tolerance"] = group["tolerance"].cummin().to_numpy(dtype=float)
    return result.reset_index(drop=True)


def loss_over_steps(records) -> pd.DataFrame:
    """Return loss trajectories over simulation steps."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["method", "replicate", "step", "loss"])
    cols = ["method", "replicate", "step", "loss"]
    return frame.loc[:, cols].sort_values(["method", "replicate", "step"]).reset_index(drop=True)
