"""Helpers for tracing simulator attempts across worker processes."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from ..io.records import ParticleRecord


def _current_worker_id() -> str:
    """Best-effort worker identifier across MPI and multiprocessing."""
    try:
        from mpi4py import MPI

        return str(int(MPI.COMM_WORLD.Get_rank()))
    except Exception:
        pass

    proc = multiprocessing.current_process()
    if proc._identity:
        return str(int(proc._identity[0] - 1))
    return str(os.getpid())


def _trace_file(trace_dir: Path) -> Path:
    worker_id = _current_worker_id()
    return trace_dir / f"worker_{worker_id}_pid_{os.getpid()}.jsonl"


def instrument_simulate(
    simulate_fn: Callable[[Dict[str, float], int], float],
    trace_dir: Path,
) -> Callable[[Dict[str, float], int], float]:
    """Wrap a simulator call and append attempt timing to a per-worker trace."""
    trace_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(params: Dict[str, float], seed: int) -> float:
        start_abs = time.time()
        loss = float(simulate_fn(params, seed=seed))
        end_abs = time.time()
        payload = {
            "params": {key: float(value) for key, value in params.items()},
            "seed": int(seed),
            "loss": loss,
            "start_abs": float(start_abs),
            "end_abs": float(end_abs),
            "worker_id": _current_worker_id(),
            "pid": int(os.getpid()),
        }
        with open(_trace_file(trace_dir), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")
        return loss

    return wrapped


def load_attempt_events(trace_dir: Path, *, run_start_abs: float) -> List[Dict[str, Any]]:
    """Load and normalize attempt traces to run-relative time."""
    events: List[Dict[str, Any]] = []
    if not trace_dir.exists():
        return events

    for path in sorted(trace_dir.glob("worker_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                start_abs = float(raw["start_abs"])
                end_abs = float(raw["end_abs"])
                events.append(
                    {
                        "params": {key: float(value) for key, value in raw.get("params", {}).items()},
                        "seed": int(raw["seed"]),
                        "loss": float(raw["loss"]),
                        "sim_start_time": start_abs - float(run_start_abs),
                        "sim_end_time": end_abs - float(run_start_abs),
                        "wall_time": end_abs - float(run_start_abs),
                        "worker_id": str(raw.get("worker_id", "")),
                        "pid": int(raw.get("pid", 0)),
                    }
                )
    return sorted(
        events,
        key=lambda event: (
            float(event["sim_end_time"]),
            float(event["sim_start_time"]),
            str(event["worker_id"]),
            int(event["pid"]),
        ),
    )


def attempt_records_from_events(
    events: Iterable[Dict[str, Any]],
    *,
    method_name: str,
    replicate: int,
    observable_attempt_counts: Iterable[int] | None = None,
) -> List[ParticleRecord]:
    """Convert attempt events into canonical attempt-level ParticleRecords."""
    cumulative_counts = [
        int(value)
        for value in observable_attempt_counts or []
        if value is not None and int(value) > 0
    ]
    records: List[ParticleRecord] = []
    generation_idx = 0

    for attempt_idx, event in enumerate(events, start=1):
        while generation_idx < len(cumulative_counts) and attempt_idx > cumulative_counts[generation_idx]:
            generation_idx += 1
        generation = generation_idx if generation_idx < len(cumulative_counts) else None
        records.append(
            ParticleRecord(
                method=method_name,
                replicate=int(replicate),
                seed=int(event["seed"]),
                step=int(attempt_idx),
                params={key: float(value) for key, value in event.get("params", {}).items()},
                loss=float(event["loss"]),
                weight=None,
                tolerance=None,
                wall_time=float(event["wall_time"]),
                worker_id=str(event["worker_id"]),
                sim_start_time=float(event["sim_start_time"]),
                sim_end_time=float(event["sim_end_time"]),
                generation=generation,
                record_kind="simulation_attempt",
                time_semantics="event_end",
                attempt_count=int(attempt_idx),
            )
        )
    return records
