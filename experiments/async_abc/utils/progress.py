"""Shared progress logging helpers for inference methods."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .mpi import get_rank

logger = logging.getLogger(__name__)


def _format_metric_value(value: Any) -> str:
    """Return a compact, stable string representation for log metrics."""
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


@dataclass
class MethodProgressReporter:
    """Emit throttled, rank-aware progress messages for one method invocation."""

    method_name: str
    replicate: int
    interval_s: float = 10.0
    rank: int = field(default_factory=get_rank)
    logger_name: str = __name__
    _start_time: float | None = field(default=None, init=False, repr=False)
    _last_update_time: float | None = field(default=None, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(self.logger_name)

    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return max(0.0, time.monotonic() - self._start_time)

    def _emit(self, *, status: str, detail: str | None = None, metrics: dict[str, Any] | None = None) -> None:
        parts = [
            f"[progress][rank={self.rank}][{self.method_name} rep={self.replicate}]",
            f"elapsed={self._elapsed():.1f}s",
            f"status={status}",
        ]
        if detail:
            parts.append(f"detail={detail!r}")
        if metrics:
            for key, value in metrics.items():
                if value is None:
                    continue
                parts.append(f"{key}={_format_metric_value(value)}")
        self._logger.info(" ".join(parts), extra={"all_ranks": True})

    def start(self, total_hint: int | float | None = None, detail: str | None = None) -> None:
        """Log the start of a method invocation."""
        if self._start_time is None:
            self._start_time = time.monotonic()
        metrics = {}
        if total_hint is not None:
            metrics["total_hint"] = total_hint
        self._emit(status="start", detail=detail, metrics=metrics or None)

    def update(self, **metrics: Any) -> None:
        """Emit a throttled progress update."""
        if self._finished or self.interval_s <= 0:
            return
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        if self._last_update_time is not None and now - self._last_update_time < self.interval_s:
            return
        self._last_update_time = now
        self._emit(status="update", metrics=metrics or None)

    def finish(self, **metrics: Any) -> None:
        """Log the successful completion of a method invocation."""
        if self._finished:
            return
        if self._start_time is None:
            self._start_time = time.monotonic()
        self._finished = True
        self._emit(status="finish", metrics=metrics or None)

    def fail(self, exc: BaseException) -> None:
        """Log a failed method invocation."""
        if self._finished:
            return
        if self._start_time is None:
            self._start_time = time.monotonic()
        self._finished = True
        detail = str(exc) or exc.__class__.__name__
        self._emit(status="fail", detail=detail, metrics={"error": exc.__class__.__name__})
