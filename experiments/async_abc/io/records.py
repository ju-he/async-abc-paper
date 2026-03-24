"""Result record schema and CSV helpers.

ParticleRecord captures a single simulation evaluation.  RecordWriter appends
records to a CSV file, flattening the ``params`` dict into ``param_<key>``
columns.  The header is written once on first write; subsequent calls append.
"""
import csv
from dataclasses import dataclass
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
    record_kind:
        Semantic type of this row, e.g. ``"simulation_attempt"`` or
        ``"population_particle"``.
    time_semantics:
        Interpretation of ``wall_time``, e.g. ``"event_end"`` or
        ``"generation_end"``.
    attempt_count:
        Cumulative simulator-attempt budget observed at this record.
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
    worker_id: Optional[str] = None
    sim_start_time: Optional[float] = None
    sim_end_time: Optional[float] = None
    generation: Optional[int] = None
    record_kind: Optional[str] = None
    time_semantics: Optional[str] = None
    attempt_count: Optional[int] = None

    def to_csv_row(self, param_keys: List[str]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "method": self.method,
            "replicate": self.replicate,
            "seed": self.seed,
            "step": self.step,
        }
        for key in param_keys:
            row[f"param_{key}"] = self.params.get(key, "")
        row["loss"] = self.loss
        row["weight"] = "" if self.weight is None else self.weight
        row["tolerance"] = "" if self.tolerance is None else self.tolerance
        row["wall_time"] = self.wall_time
        row["worker_id"] = "" if self.worker_id is None else self.worker_id
        row["sim_start_time"] = "" if self.sim_start_time is None else self.sim_start_time
        row["sim_end_time"] = "" if self.sim_end_time is None else self.sim_end_time
        row["generation"] = "" if self.generation is None else self.generation
        row["record_kind"] = "" if self.record_kind is None else self.record_kind
        row["time_semantics"] = "" if self.time_semantics is None else self.time_semantics
        row["attempt_count"] = "" if self.attempt_count is None else self.attempt_count
        return row

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "ParticleRecord":
        params = {
            key.removeprefix("param_"): float(value)
            for key, value in row.items()
            if key.startswith("param_") and value != ""
        }
        return cls(
            method=row["method"],
            replicate=int(row["replicate"]),
            seed=int(row["seed"]),
            step=int(row["step"]),
            params=params,
            loss=float(row["loss"]),
            weight=_parse_optional_float(row.get("weight")),
            tolerance=_parse_optional_float(row.get("tolerance")),
            wall_time=float(row.get("wall_time", 0.0) or 0.0),
            worker_id=_parse_optional_str(row.get("worker_id")),
            sim_start_time=_parse_optional_float(row.get("sim_start_time")),
            sim_end_time=_parse_optional_float(row.get("sim_end_time")),
            generation=_parse_optional_int(row.get("generation")),
            record_kind=_parse_optional_str(row.get("record_kind")),
            time_semantics=_parse_optional_str(row.get("time_semantics")),
            attempt_count=_parse_optional_int(row.get("attempt_count")),
        )


# Fixed columns that appear in the CSV before and after the param columns
_PREFIX_COLS = ["method", "replicate", "seed", "step"]
_SUFFIX_COLS = [
    "loss",
    "weight",
    "tolerance",
    "wall_time",
    "worker_id",
    "sim_start_time",
    "sim_end_time",
    "generation",
    "record_kind",
    "time_semantics",
    "attempt_count",
]


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _parse_optional_str(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    return value


def _record_to_row(record: ParticleRecord, param_keys: List[str]) -> Dict[str, Any]:
    return record.to_csv_row(param_keys)


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


def load_records(path: Union[str, Path]) -> List[ParticleRecord]:
    """Load a CSV of :class:`ParticleRecord` rows."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return [ParticleRecord.from_csv_row(row) for row in csv.DictReader(f)]


def write_records(path: Union[str, Path], records: List[ParticleRecord]) -> None:
    """Write *records* to *path*, replacing any previous file."""
    path = Path(path)
    if path.exists():
        path.unlink()
    writer = RecordWriter(path)
    writer.write(records)
