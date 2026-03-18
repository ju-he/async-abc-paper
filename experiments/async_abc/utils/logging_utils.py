"""Rank-aware logging setup for experiment scripts."""
from __future__ import annotations

import logging

from .mpi import is_root_rank


class _RootRankFilter(logging.Filter):
    """Suppress shared log output on non-root ranks."""

    def filter(self, record: logging.LogRecord) -> bool:
        return is_root_rank() or bool(getattr(record, "all_ranks", False))


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once per process."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
    handler.addFilter(_RootRankFilter())
    root.addHandler(handler)

    logging.captureWarnings(True)
