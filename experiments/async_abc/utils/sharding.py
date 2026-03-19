"""Helpers for sharded experiment execution and finalization."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..io.paths import OutputDir


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def split_indices(total_units: int, num_shards: int) -> List[List[int]]:
    """Return balanced contiguous slices of ``range(total_units)``."""
    if total_units < 0:
        raise ValueError("total_units must be >= 0")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")

    base, remainder = divmod(total_units, num_shards)
    assignments: List[List[int]] = []
    start = 0
    for shard_idx in range(num_shards):
        size = base + (1 if shard_idx < remainder else 0)
        stop = start + size
        assignments.append(list(range(start, stop)))
        start = stop
    return assignments


def shard_indices(total_units: int, num_shards: int, shard_index: int) -> List[int]:
    """Return the unit indices assigned to one shard."""
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return split_indices(total_units, num_shards)[shard_index]


def is_shard_mode(args) -> bool:
    """Return whether CLI args request sharded execution."""
    return args.shard_index is not None or args.num_shards is not None or args.finalize_only


def validate_shard_args(args) -> None:
    """Validate shard-related CLI arguments."""
    if args.finalize_only and args.num_shards is None and args.shard_index is not None:
        raise ValueError("--shard-index requires --num-shards")
    if args.shard_index is None and args.num_shards is None:
        return
    if args.shard_index is None or args.num_shards is None:
        raise ValueError("--shard-index and --num-shards must be provided together")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")


@dataclass(frozen=True)
class ShardLayout:
    """Filesystem layout for one sharded experiment run."""

    output_root: Path
    experiment_name: str
    shard_index: Optional[int] = None

    @property
    def shard_root(self) -> Path:
        if self.shard_index is None:
            raise ValueError("shard_root is only defined for a specific shard")
        return self.output_root / "_shards" / self.experiment_name / f"shard-{self.shard_index:03d}"

    @property
    def shard_output_dir(self) -> OutputDir:
        if self.shard_index is None:
            raise ValueError("shard_output_dir is only defined for a specific shard")
        return OutputDir(self.shard_root, self.experiment_name)

    @property
    def shard_status_path(self) -> Path:
        if self.shard_index is None:
            raise ValueError("shard_status_path is only defined for a specific shard")
        return self.shard_root / "status.json"

    @property
    def plan_path(self) -> Path:
        return self.output_root / "_shards" / self.experiment_name / "plan.json"

    @property
    def merge_lock_path(self) -> Path:
        return self.output_root / "_shards" / self.experiment_name / "merge.lock"

    @property
    def merge_done_path(self) -> Path:
        return self.output_root / "_shards" / self.experiment_name / "merge.done.json"

    @property
    def final_output_dir(self) -> OutputDir:
        return OutputDir(self.output_root, self.experiment_name)


def _json_dump_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    """Return parsed JSON or ``{}`` if the file is missing."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path) as f:
        return json.load(f)


def build_plan_payload(
    *,
    experiment_name: str,
    config_path: str,
    unit_kind: str,
    full_total_units: int,
    actual_total_units: int,
    requested_num_shards: int,
    actual_num_shards: int,
    test_mode: bool,
    extend: bool,
    shard_assignments: List[List[int]],
    runner_script: Optional[str] = None,
    submitted_job_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable shard plan."""
    return {
        "experiment_name": experiment_name,
        "config_path": config_path,
        "runner_script": runner_script,
        "unit_kind": unit_kind,
        "full_total_units": int(full_total_units),
        "actual_total_units": int(actual_total_units),
        "requested_num_shards": int(requested_num_shards),
        "actual_num_shards": int(actual_num_shards),
        "test_mode": bool(test_mode),
        "extend": bool(extend),
        "shard_assignments": shard_assignments,
        "submitted_job_ids": list(submitted_job_ids or []),
        "created_at": _now_iso(),
    }


def ensure_plan(layout: ShardLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Write a shard plan if missing and return the stored plan."""
    existing = read_json(layout.plan_path)
    if existing:
        return existing
    _json_dump_atomic(layout.plan_path, payload)
    return payload


def update_plan(layout: ShardLayout, payload: Dict[str, Any]) -> None:
    """Overwrite the shard plan."""
    _json_dump_atomic(layout.plan_path, payload)


def prepare_shard_workspace(layout: ShardLayout) -> str:
    """Prepare one shard's local workspace.

    Returns ``"skip"`` if the shard is already complete, otherwise ``"run"``.
    """
    layout.shard_root.mkdir(parents=True, exist_ok=True)
    status = read_json(layout.shard_status_path)
    if status.get("state") == "completed":
        return "skip"
    shard_out = layout.shard_output_dir
    if shard_out.root.exists():
        shutil.rmtree(shard_out.root)
    shard_out.ensure()
    return "run"


def write_shard_status(
    layout: ShardLayout,
    *,
    state: str,
    unit_indices: List[int],
    elapsed_s: Optional[float] = None,
    estimated_full_s: Optional[float] = None,
    estimated_full_unsharded_s: Optional[float] = None,
    estimated_full_sharded_wall_s: Optional[float] = None,
    aggregate_compute_s: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist per-shard status metadata."""
    payload = {
        "state": state,
        "shard_index": layout.shard_index,
        "unit_indices": unit_indices,
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "timestamp": _now_iso(),
    }
    if elapsed_s is not None:
        payload["elapsed_s"] = float(elapsed_s)
    if estimated_full_s is not None:
        payload["estimated_full_s"] = float(estimated_full_s)
    if estimated_full_unsharded_s is not None:
        payload["estimated_full_unsharded_s"] = float(estimated_full_unsharded_s)
    if estimated_full_sharded_wall_s is not None:
        payload["estimated_full_sharded_wall_s"] = float(estimated_full_sharded_wall_s)
    if aggregate_compute_s is not None:
        payload["aggregate_compute_s"] = float(aggregate_compute_s)
    if extra:
        payload.update(extra)
    _json_dump_atomic(layout.shard_status_path, payload)


def load_shard_statuses(layout: ShardLayout, num_shards: int) -> List[Dict[str, Any]]:
    """Load status payloads for all shards."""
    statuses = []
    for idx in range(num_shards):
        shard_layout = ShardLayout(layout.output_root, layout.experiment_name, idx)
        statuses.append(read_json(shard_layout.shard_status_path))
    return statuses


def all_shards_completed(layout: ShardLayout, num_shards: int) -> bool:
    """Return whether every shard has completed."""
    return all(status.get("state") == "completed" for status in load_shard_statuses(layout, num_shards))


def _lock_owner_alive(lock_payload: Dict[str, Any]) -> bool:
    job_id = str(lock_payload.get("slurm_job_id") or "").strip()
    if not job_id:
        return False
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return True
    return bool(result.stdout.strip())


def acquire_merge_lock(layout: ShardLayout, *, owner_id: str) -> bool:
    """Acquire the merge lock, stealing stale locks when safe."""
    layout.merge_lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"owner_id": owner_id, "slurm_job_id": os.getenv("SLURM_JOB_ID", ""), "timestamp": _now_iso()}
    try:
        fd = os.open(layout.merge_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        lock_payload = read_json(layout.merge_lock_path)
        if layout.merge_done_path.exists():
            return False
        if lock_payload and _lock_owner_alive(lock_payload):
            return False
        try:
            layout.merge_lock_path.unlink()
        except FileNotFoundError:
            return False
        try:
            fd = os.open(layout.merge_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
    return True


def release_merge_lock(layout: ShardLayout) -> None:
    """Best-effort lock release."""
    try:
        layout.merge_lock_path.unlink()
    except FileNotFoundError:
        pass


def write_merge_done(layout: ShardLayout, payload: Dict[str, Any]) -> None:
    """Write the merge completion sentinel."""
    _json_dump_atomic(layout.merge_done_path, payload)


def estimate_sharded_wall_time(serial_estimate_s: float, total_units: int, requested_num_shards: int) -> float:
    """Estimate wall-clock for evenly sharded work."""
    if total_units <= 0:
        return float(serial_estimate_s)
    requested = max(1, min(int(requested_num_shards), int(total_units)))
    largest_shard = math.ceil(total_units / requested)
    return float(serial_estimate_s) * largest_shard / float(total_units)


def shard_timing_summary(statuses: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Summarize elapsed, wall, and aggregate compute from shard statuses."""
    starts = [float(status["started_at_s"]) for status in statuses if "started_at_s" in status]
    finishes = [float(status["finished_at_s"]) for status in statuses if "finished_at_s" in status]
    elapsed = None
    if starts and finishes:
        elapsed = max(finishes) - min(starts)
    aggregate = sum(float(status.get("elapsed_s", 0.0) or 0.0) for status in statuses)
    return {
        "elapsed_s": elapsed,
        "aggregate_compute_s": aggregate if aggregate > 0 else None,
    }


def publish_directory_atomically(source_dir: Path, target_dir: Path) -> None:
    """Publish a prepared directory by replacing the target path."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    os.replace(source_dir, target_dir)


def shard_output_dirs(layout: ShardLayout, num_shards: int) -> List[OutputDir]:
    """Return shard-local output directories for all shards."""
    return [
        ShardLayout(layout.output_root, layout.experiment_name, idx).shard_output_dir
        for idx in range(num_shards)
    ]


def merge_csv_group(
    sources: Iterable[Path],
    destination: Path,
    *,
    sort_key: Optional[Callable[[Dict[str, str]], Any]] = None,
) -> int:
    """Merge CSV rows from *sources* into *destination*."""
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    for path in sources:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
            rows.extend(list(reader))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        destination.write_text("")
        return 0
    if sort_key is not None:
        rows.sort(key=sort_key)
    with open(destination, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def maybe_finalize_sharded_run(
    *,
    layout: ShardLayout,
    actual_num_shards: int,
    owner_id: str,
    finalize_fn: Callable[[List[OutputDir], List[Dict[str, Any]]], Dict[str, Any]],
) -> bool:
    """Attempt to finalize a completed sharded run."""
    if layout.merge_done_path.exists():
        return False
    if not all_shards_completed(layout, actual_num_shards):
        return False
    if not acquire_merge_lock(layout, owner_id=owner_id):
        return False
    try:
        if layout.merge_done_path.exists():
            return False
        shard_dirs = shard_output_dirs(layout, actual_num_shards)
        statuses = load_shard_statuses(layout, actual_num_shards)
        payload = finalize_fn(shard_dirs, statuses) or {}
        payload.setdefault("finalized_at", _now_iso())
        payload.setdefault("owner_id", owner_id)
        write_merge_done(layout, payload)
        return True
    finally:
        release_merge_lock(layout)
