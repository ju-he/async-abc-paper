"""Generation barrier summaries for synchronous methods."""

import numpy as np
import pandas as pd

from ._helpers import records_to_frame


def generation_spans(records) -> pd.DataFrame:
    """Summarize start/end timing bounds for each generation."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "replicate",
                "generation",
                "gen_start",
                "gen_end",
                "gen_duration",
                "n_particles",
            ]
        )

    frame = frame[frame["generation"].notna()].copy()
    rows = []
    for (method, replicate, generation), group in frame.groupby(
        ["method", "replicate", "generation"],
        sort=True,
    ):
        gen_start = float(group["sim_start_time"].min())
        gen_end = float(group["sim_end_time"].max())
        rows.append(
            {
                "method": method,
                "replicate": replicate,
                "generation": int(generation),
                "gen_start": gen_start,
                "gen_end": gen_end,
                "gen_duration": gen_end - gen_start,
                "n_particles": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def barrier_overhead_fraction(records) -> pd.DataFrame:
    """Estimate the fraction of generation time spent waiting at barriers."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "replicate",
                "barrier_overhead_time",
                "total_generation_time",
                "barrier_overhead_fraction",
            ]
        )

    frame = frame[
        frame["generation"].notna()
        & frame["sim_start_time"].notna()
        & frame["sim_end_time"].notna()
    ].copy()

    rows = []
    for (method, replicate), run in frame.groupby(["method", "replicate"], sort=True):
        barrier_overhead_time = 0.0
        total_generation_time = 0.0
        for _, group in run.groupby("generation", sort=True):
            end_times = group["sim_end_time"].to_numpy(dtype=float)
            start_times = group["sim_start_time"].to_numpy(dtype=float)
            barrier_overhead_time += float(end_times.max() - end_times.min())
            total_generation_time += float(end_times.max() - start_times.min())

        fraction = np.nan
        if total_generation_time > 0.0:
            fraction = barrier_overhead_time / total_generation_time

        rows.append(
            {
                "method": method,
                "replicate": replicate,
                "barrier_overhead_time": barrier_overhead_time,
                "total_generation_time": total_generation_time,
                "barrier_overhead_fraction": fraction,
            }
        )

    return pd.DataFrame(rows)
