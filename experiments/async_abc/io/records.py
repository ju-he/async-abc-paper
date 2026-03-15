"""Result record schema and CSV writer.

ParticleRecord captures a single simulation evaluation.  RecordWriter appends
records to a CSV file, flattening the ``params`` dict into ``param_<key>``
columns.  The header is written once on first write; subsequent calls append.
"""
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ParticleRecord:
    """One simulation evaluation result.

    Parameters
    ----------
    method:
        Inference method name (e.g. ``"async_propulate_abc"``).
    replicate:
        Replicate index (0-based).
    seed:
        RNG seed used for this replicate.
    step:
        Simulation step / evaluation counter.
    params:
        Dict mapping parameter names to their sampled values.
    loss:
        Simulation distance / loss value.
    weight:
        Importance weight (None during prior phase).
    tolerance:
        Effective tolerance at the time of proposal (None during prior phase).
    wall_time:
        Wall-clock seconds elapsed since the start of the run.
    """

    method: str
    replicate: int
    seed: int
    step: int
    params: Dict[str, float]
    loss: float
    weight: Optional[float] = None
    tolerance: Optional[float] = None
    wall_time: float = 0.0


# Fixed columns that appear in the CSV before and after the param columns
_PREFIX_COLS = ["method", "replicate", "seed", "step"]
_SUFFIX_COLS = ["loss", "weight", "tolerance", "wall_time"]


def _record_to_row(record: ParticleRecord, param_keys: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": record.method,
        "replicate": record.replicate,
        "seed": record.seed,
        "step": record.step,
    }
    for k in param_keys:
        row[f"param_{k}"] = record.params.get(k, "")
    row["loss"] = record.loss
    row["weight"] = "" if record.weight is None else record.weight
    row["tolerance"] = "" if record.tolerance is None else record.tolerance
    row["wall_time"] = record.wall_time
    return row


class RecordWriter:
    """Appends :class:`ParticleRecord` objects to a CSV file.

    The header is inferred from the first batch of records.  Subsequent
    ``write`` calls append rows without re-writing the header, so the writer
    is safe to call multiple times across replicates.

    Parameters
    ----------
    path:
        Destination CSV file path.  Parent directories must already exist.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def write(self, records: List[ParticleRecord]) -> None:
        """Append *records* to the CSV.

        Parameters
        ----------
        records:
            List of :class:`ParticleRecord` to write.  Must be non-empty on the
            first call (so the header can be inferred from ``records[0].params``).
        """
        if not records:
            return

        # Determine parameter column order from the first record
        param_keys = list(records[0].params.keys())
        fieldnames = (
            _PREFIX_COLS
            + [f"param_{k}" for k in param_keys]
            + _SUFFIX_COLS
        )

        mode = "a" if self._header_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            for rec in records:
                writer.writerow(_record_to_row(rec, param_keys))
