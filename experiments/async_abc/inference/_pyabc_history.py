"""Helpers for extracting observable pyABC generation metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def history_observable_frame(history, run_start: float | datetime) -> pd.DataFrame:
    """Return generation-indexed observable metadata from a pyABC history."""
    all_pops = history.get_all_populations().copy()
    if "t" in all_pops.columns:
        all_pops = all_pops[all_pops["t"] >= 0].copy()
    if all_pops.empty:
        return pd.DataFrame(columns=["generation", "epsilon", "generation_end", "attempt_count"])

    generations = pd.to_numeric(all_pops.get("t"), errors="coerce").fillna(-1).astype(int)
    frame = pd.DataFrame({"generation": generations})
    frame["epsilon"] = pd.to_numeric(all_pops.get("epsilon"), errors="coerce")
    frame["generation_end"] = [
        _seconds_since(run_start, value)
        for value in all_pops.get("population_end_time", pd.Series([None] * len(all_pops)))
    ]

    samples = _population_sample_counts(all_pops)
    frame["attempt_count"] = samples.cumsum().astype(int)
    return frame.set_index("generation", drop=False)


def _population_sample_counts(all_pops: pd.DataFrame) -> pd.Series:
    """Best-effort per-generation sample counts from pyABC metadata."""
    for column in (
        "samples",
        "nr_samples",
        "population_nr_samples",
        "n_samples",
    ):
        if column not in all_pops.columns:
            continue
        series = pd.to_numeric(all_pops[column], errors="coerce")
        if series.notna().any():
            return series.fillna(0).astype(int)

    for column in ("particles", "population_size", "n_particles"):
        if column not in all_pops.columns:
            continue
        series = pd.to_numeric(all_pops[column], errors="coerce")
        if series.notna().any():
            return series.fillna(0).astype(int)

    return pd.Series([0] * len(all_pops), index=all_pops.index, dtype=int)


def _seconds_since(run_start: float | datetime, value: Any) -> float | None:
    """Convert a pyABC timestamp into seconds elapsed since *run_start*."""
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(run_start, datetime):
        if isinstance(value, datetime):
            return float((value - run_start).total_seconds())
        if isinstance(value, str):
            try:
                parsed = pd.Timestamp(value).to_pydatetime()
            except Exception:
                return None
            return float((parsed - run_start).total_seconds())
        return None

    if isinstance(value, datetime):
        return float(value.timestamp()) - float(run_start)
    if isinstance(value, str):
        try:
            parsed = pd.Timestamp(value).to_pydatetime()
        except Exception:
            return None
        return float(parsed.timestamp()) - float(run_start)

    try:
        return float(value) - float(run_start)
    except Exception:
        return None
