"""Simple trajectory summaries over step and wall-clock time."""

import pandas as pd

from ._helpers import records_to_frame


def tolerance_over_wall_time(records) -> pd.DataFrame:
    """Return tolerance trajectories over wall-clock time."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["method", "replicate", "wall_time", "tolerance"])
    cols = ["method", "replicate", "wall_time", "tolerance"]
    return (
        frame.loc[frame["tolerance"].notna(), cols]
        .sort_values(["method", "replicate", "wall_time", "tolerance"])
        .reset_index(drop=True)
    )


def loss_over_steps(records) -> pd.DataFrame:
    """Return loss trajectories over simulation steps."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["method", "replicate", "step", "loss"])
    cols = ["method", "replicate", "step", "loss"]
    return frame.loc[:, cols].sort_values(["method", "replicate", "step"]).reset_index(drop=True)
