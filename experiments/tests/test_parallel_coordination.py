"""Tests for rank-aware experiment coordination helpers."""
from async_abc.io.records import ParticleRecord
from async_abc.utils import runner as runner_utils


def _sentinel_records() -> list[ParticleRecord]:
    return [
        ParticleRecord(
            method="sentinel",
            replicate=0,
            seed=1,
            step=1,
            params={"x": 0.0},
            loss=0.0,
        )
    ]


class TestRunMethodDistributed:
    def test_rank_zero_method_skips_non_root(self, monkeypatch, tmp_output_dir):
        calls = []

        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "allgather", lambda value: [value])
        monkeypatch.setattr(runner_utils, "method_execution_mode", lambda name: "rank_zero")
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: calls.append((args, kwargs)) or _sentinel_records(),
        )

        records = runner_utils.run_method_distributed(
            "pyabc_smc",
            lambda params, seed: 0.0,
            {"x": (-1.0, 1.0)},
            {"max_simulations": 1},
            tmp_output_dir,
            replicate=0,
            seed=1,
        )

        assert records == []
        assert calls == []

    def test_all_ranks_method_runs_everywhere_but_keeps_root_records(self, monkeypatch, tmp_output_dir):
        calls = []

        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "allgather", lambda value: [value])
        monkeypatch.setattr(runner_utils, "method_execution_mode", lambda name: "all_ranks")
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: calls.append((args, kwargs)) or _sentinel_records(),
        )

        records = runner_utils.run_method_distributed(
            "async_propulate_abc",
            lambda params, seed: 0.0,
            {"x": (-1.0, 1.0)},
            {"max_simulations": 1},
            tmp_output_dir,
            replicate=0,
            seed=1,
        )

        assert records == []
        assert len(calls) == 1

    def test_distributed_import_error_is_re_raised_as_import_error(self, monkeypatch, tmp_output_dir):
        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: True)
        monkeypatch.setattr(runner_utils, "allgather", lambda value: [value])
        monkeypatch.setattr(runner_utils, "method_execution_mode", lambda name: "rank_zero")
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("missing dep")),
        )

        try:
            runner_utils.run_method_distributed(
                "pyabc_smc",
                lambda params, seed: 0.0,
                {"x": (-1.0, 1.0)},
                {"max_simulations": 1},
                tmp_output_dir,
                replicate=0,
                seed=1,
            )
        except ImportError as exc:
            assert "missing dep" in str(exc)
        else:
            raise AssertionError("Expected ImportError")
