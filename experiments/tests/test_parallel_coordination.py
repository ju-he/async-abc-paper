"""Tests for rank-aware experiment coordination helpers."""
import pytest

from async_abc.io.records import ParticleRecord
from async_abc.io.paths import OutputDir
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
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()

        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "method_execution_mode_for_cfg", lambda name, cfg, simulate_fn=None: "rank_zero")
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: calls.append((args, kwargs)) or _sentinel_records(),
        )
        monkeypatch.setattr(
            runner_utils,
            "_wait_for_rank_zero_status",
            lambda path: {"kind": "ok", "message": ""},
        )

        records = runner_utils.run_method_distributed(
            "pyabc_smc",
            lambda params, seed: 0.0,
            {"x": (-1.0, 1.0)},
            {"max_simulations": 1},
            output_dir,
            replicate=0,
            seed=1,
        )

        assert records == []
        assert calls == []

    def test_rank_zero_method_on_root_avoids_allgather(self, monkeypatch, tmp_output_dir):
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()
        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: True)
        monkeypatch.setattr(
            runner_utils,
            "allgather",
            lambda value: (_ for _ in ()).throw(AssertionError("allgather should not be used")),
        )
        monkeypatch.setattr(runner_utils, "method_execution_mode_for_cfg", lambda name, cfg, simulate_fn=None: "rank_zero")
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: _sentinel_records(),
        )

        records = runner_utils.run_method_distributed(
            "pyabc_smc",
            lambda params, seed: 0.0,
            {"x": (-1.0, 1.0)},
            {"max_simulations": 1},
            output_dir,
            replicate=0,
            seed=1,
        )

        assert records == _sentinel_records()

    def test_all_ranks_method_runs_everywhere_but_keeps_root_records(self, monkeypatch, tmp_output_dir):
        calls = []
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()

        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "allgather", lambda value: [value])
        monkeypatch.setattr(runner_utils, "method_execution_mode_for_cfg", lambda name, cfg, simulate_fn=None: "all_ranks")
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
            output_dir,
            replicate=0,
            seed=1,
        )

        assert records == []
        assert len(calls) == 1

    @pytest.mark.parametrize("method_name", ["pyabc_smc", "abc_smc_baseline"])
    def test_pyabc_mpi_backend_overrides_rank_zero_mode(self, monkeypatch, tmp_output_dir, method_name):
        calls = []
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()

        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "allgather", lambda value: [value])
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: calls.append((args, kwargs)) or _sentinel_records(),
        )

        records = runner_utils.run_method_distributed(
            method_name,
            lambda params, seed: 0.0,
            {"x": (-1.0, 1.0)},
            {"max_simulations": 1, "n_workers": 4, "parallel_backend": "mpi"},
            output_dir,
            replicate=0,
            seed=1,
        )

        assert records == []
        assert len(calls) == 1

    @pytest.mark.parametrize("method_name", ["pyabc_smc", "abc_smc_baseline"])
    def test_pyabc_multicore_backend_stays_rank_zero(self, monkeypatch, tmp_output_dir, method_name):
        calls = []
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()

        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "method_execution_mode_for_cfg", lambda name, cfg, simulate_fn=None: "rank_zero")
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: calls.append((args, kwargs)) or _sentinel_records(),
        )
        monkeypatch.setattr(
            runner_utils,
            "_wait_for_rank_zero_status",
            lambda path: {"kind": "ok", "message": ""},
        )

        records = runner_utils.run_method_distributed(
            method_name,
            lambda params, seed: 0.0,
            {"x": (-1.0, 1.0)},
            {"max_simulations": 1, "n_workers": 1, "parallel_backend": "multicore"},
            output_dir,
            replicate=0,
            seed=1,
        )

        assert records == []
        assert calls == []

    @pytest.mark.parametrize("method_name", ["pyabc_smc", "abc_smc_baseline"])
    def test_pyabc_mpi_request_still_runs_on_all_ranks_for_benchmarks_marked_unsafe(
        self, monkeypatch, tmp_output_dir, method_name
    ):
        calls = []
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()

        class UnsafeBenchmark:
            MULTIPROCESSING_SAFE = False

            def simulate(self, params, seed):
                return 0.0

        benchmark = UnsafeBenchmark()

        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "allgather", lambda value: [value])
        monkeypatch.setattr(
            runner_utils,
            "run_method",
            lambda *args, **kwargs: calls.append((args, kwargs)) or _sentinel_records(),
        )

        records = runner_utils.run_method_distributed(
            method_name,
            benchmark.simulate,
            {"x": (-1.0, 1.0)},
            {"max_simulations": 1, "n_workers": 4, "parallel_backend": "mpi"},
            output_dir,
            replicate=0,
            seed=1,
        )

        assert records == []
        assert len(calls) == 1

    def test_distributed_import_error_is_re_raised_as_import_error(self, monkeypatch, tmp_output_dir):
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()
        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: True)
        monkeypatch.setattr(runner_utils, "method_execution_mode_for_cfg", lambda name, cfg, simulate_fn=None: "rank_zero")
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
                output_dir,
                replicate=0,
                seed=1,
            )
        except ImportError as exc:
            assert "missing dep" in str(exc)
        else:
            raise AssertionError("Expected ImportError")

    def test_non_root_re_raises_root_exception_from_status_file(self, monkeypatch, tmp_output_dir):
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()
        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: False)
        monkeypatch.setattr(runner_utils, "method_execution_mode_for_cfg", lambda name, cfg, simulate_fn=None: "rank_zero")
        monkeypatch.setattr(
            runner_utils,
            "_wait_for_rank_zero_status",
            lambda path: {"kind": "Exception", "message": "boom"},
        )

        try:
            runner_utils.run_method_distributed(
                "pyabc_smc",
                lambda params, seed: 0.0,
                {"x": (-1.0, 1.0)},
                {"max_simulations": 1},
                output_dir,
                replicate=0,
                seed=1,
            )
        except RuntimeError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError")


class TestRunExperimentCleanup:
    def test_closes_created_benchmark(self, monkeypatch, tmp_output_dir):
        closed = []
        output_dir = OutputDir(tmp_output_dir.parent, tmp_output_dir.name).ensure()

        class DummyBenchmark:
            limits = {"x": (-1.0, 1.0)}

            @staticmethod
            def simulate(params, seed):
                return 0.0

            def close(self):
                closed.append(True)

        monkeypatch.setattr(runner_utils, "make_benchmark", lambda cfg: DummyBenchmark())
        monkeypatch.setattr(runner_utils, "run_method_distributed", lambda *args, **kwargs: [])
        monkeypatch.setattr(runner_utils, "is_root_rank", lambda: True)

        cfg = {
            "benchmark": {"name": "dummy"},
            "methods": ["rejection_abc"],
            "inference": {"max_simulations": 1},
            "execution": {"n_replicates": 1, "base_seed": 1},
        }

        runner_utils.run_experiment(cfg, output_dir)

        assert closed == [True]
