"""Tests for async_abc.utils.progress."""
import logging

from async_abc.utils.logging_utils import _RootRankFilter
from async_abc.utils.progress import MethodProgressReporter


class _Clock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_progress_reporter_formats_messages(caplog, monkeypatch):
    import async_abc.utils.progress as progress_mod

    clock = _Clock()
    monkeypatch.setattr(progress_mod.time, "monotonic", clock)

    reporter = MethodProgressReporter("rejection_abc", replicate=4, interval_s=10.0, rank=2)

    with caplog.at_level(logging.INFO, logger="async_abc.utils.progress"):
        reporter.start(total_hint=12, detail="mode=rank_parallel")

    assert len(caplog.records) == 1
    assert "[progress][rank=2][rejection_abc rep=4]" in caplog.text
    assert "status=start" in caplog.text
    assert "total_hint=12" in caplog.text
    assert "detail='mode=rank_parallel'" in caplog.text


def test_progress_reporter_throttles_updates(caplog, monkeypatch):
    import async_abc.utils.progress as progress_mod

    clock = _Clock()
    monkeypatch.setattr(progress_mod.time, "monotonic", clock)

    reporter = MethodProgressReporter("pyabc_smc", replicate=0, interval_s=1.0, rank=0)

    with caplog.at_level(logging.INFO, logger="async_abc.utils.progress"):
        reporter.start()
        clock.now = 0.5
        reporter.update(simulations=1)
        clock.now = 0.8
        reporter.update(simulations=2)
        clock.now = 1.6
        reporter.update(simulations=3)

    update_messages = [record.getMessage() for record in caplog.records if "status=update" in record.getMessage()]
    assert len(update_messages) == 2
    assert "simulations=1" in update_messages[0]
    assert "simulations=3" in update_messages[1]


def test_progress_interval_nonpositive_disables_periodic_updates(caplog, monkeypatch):
    import async_abc.utils.progress as progress_mod

    clock = _Clock()
    monkeypatch.setattr(progress_mod.time, "monotonic", clock)

    reporter = MethodProgressReporter("abc_smc_baseline", replicate=1, interval_s=0.0, rank=0)

    with caplog.at_level(logging.INFO, logger="async_abc.utils.progress"):
        reporter.start()
        reporter.update(simulations=1)
        reporter.finish(records=5)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert "status=start" in messages[0]
    assert "status=finish" in messages[1]


def test_root_rank_filter_keeps_all_rank_progress_logs(monkeypatch):
    import async_abc.utils.logging_utils as logging_utils

    monkeypatch.setattr(logging_utils, "is_root_rank", lambda: False)
    rank_filter = _RootRankFilter()

    regular_record = logging.LogRecord("x", logging.INFO, __file__, 1, "regular", args=(), exc_info=None)
    progress_record = logging.LogRecord("x", logging.INFO, __file__, 1, "progress", args=(), exc_info=None)
    progress_record.all_ranks = True

    assert rank_filter.filter(regular_record) is False
    assert rank_filter.filter(progress_record) is True
