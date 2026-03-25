"""Helpers for best-effort MPI/SLURM rank coordination."""
from __future__ import annotations

import os
from typing import Any, List


def _try_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_comm():
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except ImportError:
        return None


def get_rank() -> int:
    """Return the current MPI/SLURM rank, defaulting to 0."""
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"):
        rank = _try_int(os.environ.get(key))
        if rank is not None:
            return rank

    comm = _get_comm()
    if comm is None:
        return 0
    try:
        return int(comm.Get_rank())
    except Exception:
        return 0


def get_world_size() -> int:
    """Return the current MPI/SLURM world size, defaulting to 1."""
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
        size = _try_int(os.environ.get(key))
        if size is not None:
            return size

    comm = _get_comm()
    if comm is None:
        return 1
    try:
        return int(comm.Get_size())
    except Exception:
        return 1


def is_root_rank() -> bool:
    """Return True on the root rank."""
    return get_rank() == 0


def allreduce_max(value: int) -> int:
    """Compute the maximum integer value across all ranks."""
    comm = _get_comm()
    if comm is None or get_world_size() == 1:
        return int(value)
    from mpi4py import MPI

    return int(comm.allreduce(int(value), op=MPI.MAX))


def any_true(value: bool) -> bool:
    """Return True if any rank reports True."""
    return bool(allreduce_max(1 if value else 0))


def allgather(value: Any) -> List[Any]:
    """Collect a value from each rank."""
    comm = _get_comm()
    if comm is None or get_world_size() == 1:
        return [value]
    return list(comm.allgather(value))
