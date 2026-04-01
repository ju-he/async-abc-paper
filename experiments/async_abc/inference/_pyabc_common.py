"""Shared utilities for pyABC-based inference methods.

Contains helpers used by both :mod:`pyabc_wrapper` (pyabc_smc) and
:mod:`abc_smc_baseline` to avoid code duplication.
"""
from pathlib import Path

from ..io.paths import OutputDir


def db_suffix(checkpoint_tag: str) -> str:
    """Return a filesystem-safe suffix derived from *checkpoint_tag*."""
    if not checkpoint_tag:
        return ""
    safe_tag = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(checkpoint_tag)
    )
    return f"__{safe_tag}" if safe_tag else ""


def prepare_db_path(
    output_dir: OutputDir,
    *,
    method_name: str,
    replicate: int,
    seed: int,
    checkpoint_tag: str,
) -> str:
    """Create (or clean) a SQLite database path for a pyABC run."""
    db_file = (
        output_dir.data
        / f"{method_name}_rep{replicate}_seed{seed}{db_suffix(checkpoint_tag)}.db"
    )
    for path in (db_file, Path(f"{db_file}-wal"), Path(f"{db_file}-shm")):
        if path.exists():
            path.unlink()
    return f"sqlite:///{db_file}"
