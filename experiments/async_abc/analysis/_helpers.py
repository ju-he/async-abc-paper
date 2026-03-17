"""Shared helpers for analysis modules."""

from typing import Iterable

import pandas as pd

from ..io.records import ParticleRecord


def records_to_frame(records: Iterable[ParticleRecord] | pd.DataFrame) -> pd.DataFrame:
    """Convert ParticleRecord iterables into a flat DataFrame."""
    if isinstance(records, pd.DataFrame):
        return records.copy()

    rows = []
    for record in records:
        row = {
            "method": record.method,
            "replicate": record.replicate,
            "seed": record.seed,
            "step": record.step,
            "loss": record.loss,
            "weight": record.weight,
            "tolerance": record.tolerance,
            "wall_time": record.wall_time,
            "worker_id": record.worker_id,
            "sim_start_time": record.sim_start_time,
            "sim_end_time": record.sim_end_time,
            "generation": record.generation,
        }
        row.update(record.params)
        rows.append(row)
    return pd.DataFrame(rows)
