"""Tests for async_abc.inference.*"""
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from async_abc.benchmarks.gaussian_mean import GaussianMean
from async_abc.inference.method_registry import METHOD_REGISTRY, method_execution_mode, run_method
from async_abc.inference.propulate_abc import run_propulate_abc
from async_abc.io.records import ParticleRecord
from async_abc.utils.progress import MethodProgressReporter


def _test_inference_cfg():
    return {
        "max_simulations": 12,
        "n_workers": 1,
        "k": 5,
        "tol_init": 5.0,
        "scheduler_type": "acceptance_rate",
        "perturbation_scale": 0.8,
    }


def _gaussian_bm():
    return GaussianMean(
        {
            "observed_data_seed": 0,
            "n_obs": 30,
            "true_mu": 0.0,
            "prior_low": -5.0,
            "prior_high": 5.0,
        }
    )


def _population_records(records):
    return [
        record
        for record in records
        if record.record_kind in (None, "", "population_particle")
    ]


def _make_fake_executor():
    """Return a minimal executor stub for fake MPI executor tests."""
    return types.SimpleNamespace(
        shutdown=lambda **kwargs: None,
        map=map,
    )


class _RecordingExecutor:
    def __init__(self):
        self.shutdown_calls = []
        self.map = map

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


def _install_fake_mpi_executor(monkeypatch, *, executor):
    """Install a fake mpi4py for unit testing.

    CommWorldMap.map falls back to local execution when Get_size() == 1,
    so the fake COMM_WORLD only needs Get_rank and Get_size.

    The concurrent_futures legacy path still uses MPICommExecutor, so
    the fake returns *executor* from __enter__ for that path.
    """
    fake_mpi = types.ModuleType("mpi4py")
    fake_mpi.MPI = types.SimpleNamespace(
        COMM_WORLD=types.SimpleNamespace(
            Get_size=lambda: 1,
            Get_rank=lambda: 0,
            Barrier=lambda: None,
        ),
    )

    fake_futures = types.ModuleType("mpi4py.futures")

    class _FakeMPICommExecutor:
        def __init__(self, comm, root=0):
            pass
        def __enter__(self):
            return executor
        def __exit__(self, *a):
            return False

    fake_futures.MPICommExecutor = _FakeMPICommExecutor
    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi)
    monkeypatch.setitem(sys.modules, "mpi4py.futures", fake_futures)


class _FakeIndividual:
    def __init__(
        self,
        params,
        *,
        generation,
        tolerance,
        evaltime,
        evalperiod,
        rank,
        loss,
    ):
        self._params = params
        self.generation = generation
        self.tolerance = tolerance
        self.evaltime = evaltime
        self.evalperiod = evalperiod
        self.rank = rank
        self.loss = loss
        self.weight = 1.0

    def __getitem__(self, key):
        return self._params[key]


class _FakeABCPMC:
    def __init__(self, *, limits, perturbation_scale, k, tol, scheduler_type, rng, **kwargs):
        self.limits = limits
        self.perturbation_scale = perturbation_scale
        self.k = k
        self.tol = tol
        self.scheduler_type = scheduler_type
        self.rng = rng
        self.extra = kwargs


class _FakePropulator:
    def __init__(self, *, loss_fn, propagator, rng, generations, checkpoint_path, **kwargs):
        self.loss_fn = loss_fn
        self.propagator = propagator
        self.rng = rng
        self.generations = generations
        self.checkpoint_path = checkpoint_path
        self.population = []
        self.extra = kwargs

    def propulate(self, **kwargs):
        base_time = time.time() + 1.0
        for generation in range(self.generations):
            params = {
                key: float(lo + (hi - lo) * self.rng.random())
                for key, (lo, hi) in self.propagator.limits.items()
            }
            ind = _FakeIndividual(
                params,
                generation=generation,
                tolerance=max(0.1, self.propagator.tol - 0.1 * generation),
                evaltime=base_time + 0.01 * (generation + 1),
                evalperiod=0.01,
                rank=generation % 2,
                loss=0.0,
            )
            ind.loss = self.loss_fn(ind)
            self.population.append(ind)


class _CapturingPropulator(_FakePropulator):
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        super().__init__(**kwargs)


class _CheckpointInspectingPropulator(_FakePropulator):
    saw_stale_marker = None

    def __init__(self, **kwargs):
        checkpoint_path = kwargs["checkpoint_path"]
        type(self).saw_stale_marker = (checkpoint_path / "stale.txt").exists()
        super().__init__(**kwargs)


class _FakeNoEvaltimePropulator(_FakePropulator):
    def propulate(self, **kwargs):
        self.population = []
        for generation in range(self.generations):
            params = {
                key: float(lo + (hi - lo) * self.rng.random())
                for key, (lo, hi) in self.propagator.limits.items()
            }
            ind = _FakeIndividual(
                params,
                generation=generation,
                tolerance=max(0.1, self.propagator.tol - 0.1 * generation),
                evaltime=None,
                evalperiod=None,
                rank=generation % 2,
                loss=0.0,
            )
            del ind.evaltime
            del ind.evalperiod
            ind.loss = self.loss_fn(ind)
            self.population.append(ind)


class _WallTimeLocalPropulator:
    clock = None

    def __init__(self, *, loss_fn, propagator, rng, generations, checkpoint_path, **kwargs):
        self.loss_fn = loss_fn
        self.propagator = propagator
        self.rng = rng
        self.generations = generations
        self.checkpoint_path = checkpoint_path
        self.population = []
        self.extra = kwargs
        self.generation = 0
        self.propulate_comm = None
        self.worker_sub_comm = None

    def _evaluate_individual(self):
        clock = type(self).clock
        assert clock is not None
        params = {
            key: float(lo + (hi - lo) * self.rng.random())
            for key, (lo, hi) in self.propagator.limits.items()
        }
        start = clock()
        clock.now += 0.02
        end = clock()
        ind = _FakeIndividual(
            params,
            generation=self.generation,
            tolerance=max(0.1, self.propagator.tol - 0.1 * self.generation),
            evaltime=end,
            evalperiod=end - start,
            rank=self.generation % 2,
            loss=0.0,
        )
        ind.loss = self.loss_fn(ind)
        self.population.append(ind)


class _FakeBarrierComm:
    def __init__(self):
        self.rank = 0
        self.size = 2
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1

    def Allreduce(self, sendbuf, recvbuf, op=None):
        recvbuf[:] = sendbuf


class _WallTimeCommPropulator:
    clock = None
    last_instance = None

    def __init__(self, *, loss_fn, propagator, rng, generations, checkpoint_path, **kwargs):
        type(self).last_instance = self
        self.loss_fn = loss_fn
        self.propagator = propagator
        self.rng = rng
        self.generations = generations
        self.checkpoint_path = checkpoint_path
        self.population = []
        self.extra = kwargs
        self.generation = 0
        self.propulate_comm = kwargs.get("propulate_comm")
        self.island_comm = kwargs.get("island_comm")
        self.worker_sub_comm = kwargs.get("worker_sub_comm")
        self.intra_requests = []
        self.cleanup_calls = 0
        self.dump_calls = 0
        self.final_dump_calls = 0

    def _evaluate_individual(self):
        clock = type(self).clock
        assert clock is not None
        params = {
            key: float(lo + (hi - lo) * self.rng.random())
            for key, (lo, hi) in self.propagator.limits.items()
        }
        start = clock()
        clock.now += 0.02
        end = clock()
        ind = _FakeIndividual(
            params,
            generation=self.generation,
            tolerance=max(0.1, self.propagator.tol - 0.1 * self.generation),
            evaltime=end,
            evalperiod=end - start,
            rank=self.generation % 2,
            loss=0.0,
        )
        ind.loss = self.loss_fn(ind)
        self.population.append(ind)
        self.intra_requests.append(object())

    def _receive_intra_island_individuals(self):
        return None

    def _intra_send_cleanup(self):
        self.cleanup_calls += 1
        self.intra_requests.clear()

    def _dump_checkpoint(self):
        self.dump_calls += 1

    def _determine_worker_dumping_next(self):
        return False

    def _dump_final_checkpoint(self):
        self.final_dump_calls += 1


class _Clock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def fake_propulate_env():
    import async_abc.inference.propulate_abc as mod

    original = (mod.Propulator, mod.ABCPMC)
    mod.Propulator = _FakePropulator
    mod.ABCPMC = _FakeABCPMC
    try:
        yield mod
    finally:
        mod.Propulator, mod.ABCPMC = original


def _run_fake_propulate(tmp_path_factory, cfg=None, *, replicate=0, seed=7):
    from async_abc.io.paths import OutputDir

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp(f"propulate_{seed}"), "run").ensure()
    return run_propulate_abc(
        bm.simulate,
        bm.limits,
        cfg or _test_inference_cfg(),
        od,
        replicate=replicate,
        seed=seed,
    )


@pytest.fixture(scope="module")
def propulate_records_default(fake_propulate_env, tmp_path_factory):
    return _run_fake_propulate(tmp_path_factory, replicate=0, seed=7)


@pytest.fixture(scope="module")
def propulate_records_alt_seed(fake_propulate_env, tmp_path_factory):
    return _run_fake_propulate(tmp_path_factory, replicate=1, seed=11)


@pytest.fixture(scope="module")
def propulate_records_tolerance(fake_propulate_env, tmp_path_factory):
    cfg = {**_test_inference_cfg(), "max_simulations": 20}
    return _run_fake_propulate(tmp_path_factory, cfg=cfg, seed=13)


@pytest.fixture(scope="module")
def propulate_records_no_evaltime(tmp_path_factory):
    import async_abc.inference.propulate_abc as mod

    original = (mod.Propulator, mod.ABCPMC)
    mod.Propulator = _FakeNoEvaltimePropulator
    mod.ABCPMC = _FakeABCPMC
    try:
        return _run_fake_propulate(tmp_path_factory, seed=17)
    finally:
        mod.Propulator, mod.ABCPMC = original


@pytest.fixture(scope="module")
def pyabc_records_default(tmp_path_factory):
    pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
    from async_abc.io.paths import OutputDir
    from async_abc.inference.pyabc_wrapper import run_pyabc_smc

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("pyabc_default"), "pyabc").ensure()
    return run_pyabc_smc(
        bm.simulate, bm.limits, {**_test_inference_cfg(), "max_simulations": 30}, od, replicate=0, seed=1
    )


@pytest.fixture(scope="module")
def abc_smc_baseline_records_default(tmp_path_factory):
    pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
    from async_abc.io.paths import OutputDir
    from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("abc_smc_default"), "abc_smc").ensure()
    cfg = {**_test_inference_cfg(), "n_generations": 2}
    return run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)


@pytest.fixture(scope="module")
def abc_smc_baseline_records_generation(tmp_path_factory):
    pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
    from async_abc.io.paths import OutputDir
    from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("abc_smc_generation"), "abc_smc").ensure()
    cfg = {**_test_inference_cfg(), "n_generations": 3}
    return run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)


@pytest.fixture(scope="module")
def abc_smc_baseline_records_loss(tmp_path_factory):
    pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
    from async_abc.io.paths import OutputDir
    from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("abc_smc_loss"), "abc_smc").ensure()
    cfg = {**_test_inference_cfg(), "n_generations": 2}
    return run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)


@pytest.fixture(scope="module")
def abc_smc_baseline_records_tolerance(tmp_path_factory):
    pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
    from async_abc.io.paths import OutputDir
    from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("abc_smc_tolerance"), "abc_smc").ensure()
    cfg = {**_test_inference_cfg(), "n_generations": 3, "max_simulations": 40}
    return run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)


@pytest.fixture(scope="module")
def rejection_records_default(tmp_path_factory):
    from async_abc.io.paths import OutputDir
    from async_abc.inference.rejection_abc import run_rejection_abc

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("rejection_default"), "rejection").ensure()
    return run_rejection_abc(bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1)


@pytest.fixture(scope="module")
def rejection_records_tol3(tmp_path_factory):
    from async_abc.io.paths import OutputDir
    from async_abc.inference.rejection_abc import run_rejection_abc

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("rejection_tol3"), "rejection").ensure()
    cfg = {**_test_inference_cfg(), "tol_init": 3.0}
    return run_rejection_abc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)


@pytest.fixture(scope="module")
def rejection_records_small_k(tmp_path_factory):
    from async_abc.io.paths import OutputDir
    from async_abc.inference.rejection_abc import run_rejection_abc

    bm = _gaussian_bm()
    od = OutputDir(tmp_path_factory.mktemp("rejection_small_k"), "rejection").ensure()
    cfg = {**_test_inference_cfg(), "k": 3}
    return run_rejection_abc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)


class TestMethodRegistry:
    def test_contains_async_propulate_abc(self):
        assert "async_propulate_abc" in METHOD_REGISTRY

    def test_contains_pyabc_smc(self):
        assert "pyabc_smc" in METHOD_REGISTRY

    def test_all_values_are_callable(self):
        assert all(callable(fn) for fn in METHOD_REGISTRY.values())


class TestRunMethod:
    def test_dispatches_async_abc_without_running_real_inference(self, tmp_output_dir, monkeypatch):
        from async_abc.io.paths import OutputDir

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "test").ensure()
        sentinel = [ParticleRecord(method="sentinel", replicate=0, seed=1, step=1, params={"mu": 0.0}, loss=0.0, weight=1.0, tolerance=1.0, wall_time=0.0)]
        monkeypatch.setitem(METHOD_REGISTRY, "async_propulate_abc", lambda *args: sentinel)
        assert run_method("async_propulate_abc", bm.simulate, bm.limits, _test_inference_cfg(), od, 0, 1) == sentinel

    def test_unknown_name_raises(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "test").ensure()
        with pytest.raises(KeyError, match="not_a_method"):
            run_method("not_a_method", bm.simulate, bm.limits, _test_inference_cfg(), od, 0, 1)


class TestMethodProgress:
    def test_rejection_abc_emits_periodic_progress(self, tmp_output_dir, caplog, monkeypatch):
        import async_abc.utils.progress as progress_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc

        clock = _Clock()
        monkeypatch.setattr(progress_mod.time, "monotonic", clock)

        def simulate(_params, seed):
            clock.now += 1.0
            return 0.0 if seed % 2 == 0 else 100.0

        reporter = MethodProgressReporter("rejection_abc", replicate=0, interval_s=0.5, rank=0)
        od = OutputDir(tmp_output_dir, "rejection_progress").ensure()

        with caplog.at_level(logging.INFO, logger="async_abc.utils.progress"):
            reporter.start(total_hint=5, detail="mode=test")
            run_rejection_abc(
                simulate,
                {"x": (-1.0, 1.0)},
                {**_test_inference_cfg(), "max_simulations": 5, "k": 2},
                od,
                replicate=0,
                seed=1,
                progress=reporter,
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any("status=update" in message and "accepted=" in message for message in messages)
        assert any("status=finish" in message and "budget=5" in message for message in messages)

    def test_propulate_abc_uses_wrapper_progress_and_suppresses_native_logs(
        self, tmp_output_dir, caplog, monkeypatch
    ):
        import async_abc.inference.propulate_abc as mod
        import async_abc.utils.progress as progress_mod
        from async_abc.io.paths import OutputDir

        clock = _Clock()
        monkeypatch.setattr(progress_mod.time, "monotonic", clock)

        class LoggingFakePropulator(_FakePropulator):
            def propulate(self, **kwargs):
                logging.getLogger("propulate.propulator").info("native propulate info")
                super().propulate(**kwargs)

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                original_loss_fn = self.loss_fn

                def wrapped_loss_fn(ind):
                    clock.now += 1.0
                    return original_loss_fn(ind)

                self.loss_fn = wrapped_loss_fn

        original = (mod.Propulator, mod.ABCPMC)
        mod.Propulator = LoggingFakePropulator
        mod.ABCPMC = _FakeABCPMC
        try:
            bm = _gaussian_bm()
            od = OutputDir(tmp_output_dir, "propulate_progress").ensure()
            reporter = MethodProgressReporter("async_propulate_abc", replicate=0, interval_s=0.5, rank=0)

            with caplog.at_level(logging.INFO):
                reporter.start(total_hint=4, detail="mode=test")
                records = mod.run_propulate_abc(
                    bm.simulate,
                    bm.limits,
                    {**_test_inference_cfg(), "max_simulations": 4},
                    od,
                    replicate=0,
                    seed=7,
                    progress=reporter,
                )
        finally:
            mod.Propulator, mod.ABCPMC = original

        assert records
        messages = [record.getMessage() for record in caplog.records]
        assert any("status=update" in message and "evaluations=" in message for message in messages)
        assert any("status=finish" in message and "records=" in message for message in messages)
        assert all("native propulate info" not in message for message in messages)

    def test_pyabc_wrapper_emits_progress(self, tmp_output_dir, caplog, monkeypatch):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
        import async_abc.utils.progress as progress_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        clock = _Clock()
        monkeypatch.setattr(progress_mod.time, "monotonic", clock)

        def simulate(params, seed):
            del params, seed
            clock.now += 1.0
            return 0.0

        od = OutputDir(tmp_output_dir, "pyabc_progress").ensure()
        reporter = MethodProgressReporter("pyabc_smc", replicate=0, interval_s=0.5, rank=0)

        with caplog.at_level(logging.INFO, logger="async_abc.utils.progress"):
            reporter.start(total_hint=8, detail="mode=test")
            records = run_pyabc_smc(
                simulate,
                {"mu": (-1.0, 1.0)},
                {**_test_inference_cfg(), "max_simulations": 8, "n_workers": 1},
                od,
                replicate=0,
                seed=1,
                progress=reporter,
            )

        assert records
        messages = [record.getMessage() for record in caplog.records]
        assert any("status=update" in message and "simulations=" in message for message in messages)
        assert any("status=finish" in message and "records=" in message for message in messages)

    def test_pyabc_wrapper_finish_simulations_matches_traced_attempts(self, tmp_output_dir, caplog, monkeypatch):
        """Regression (Phase 4): finish log simulations must equal traced attempt count, not eval_count.

        Under MPI, pyabc_model executes on worker ranks so the root-local eval_count stays
        at 0 while all real work is traced on disk. The finish log must use len(attempt_events)
        from the trace, not eval_count, so the count is correct in both single- and multi-rank runs.
        """
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
        import async_abc.utils.progress as progress_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        clock = _Clock()
        monkeypatch.setattr(progress_mod.time, "monotonic", clock)

        def simulate(params, seed):
            del params, seed
            clock.now += 1.0
            return 0.0

        od = OutputDir(tmp_output_dir, "pyabc_progress_count").ensure()
        reporter = MethodProgressReporter("pyabc_smc", replicate=0, interval_s=0.0, rank=0)

        with caplog.at_level(logging.INFO, logger="async_abc.utils.progress"):
            reporter.start()
            records = run_pyabc_smc(
                simulate,
                {"mu": (-1.0, 1.0)},
                {**_test_inference_cfg(), "max_simulations": 8, "n_workers": 1},
                od,
                replicate=1,
                seed=2,
                progress=reporter,
            )

        attempt_records = [r for r in records if r.record_kind == "simulation_attempt"]
        n_attempts = len(attempt_records)

        finish_messages = [
            record.getMessage()
            for record in caplog.records
            if "status=finish" in record.getMessage()
        ]
        assert finish_messages, "expected a finish log message"
        finish_msg = finish_messages[-1]
        assert f"simulations={n_attempts}" in finish_msg, (
            f"finish log reports wrong simulation count: expected simulations={n_attempts} "
            f"but got: {finish_msg}"
        )
        assert "generations=" in finish_msg, "finish log must include generations count"

    def test_abc_smc_baseline_emits_progress(self, tmp_output_dir, caplog, monkeypatch):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")
        import async_abc.utils.progress as progress_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        clock = _Clock()
        monkeypatch.setattr(progress_mod.time, "monotonic", clock)

        def simulate(params, seed):
            del params, seed
            clock.now += 1.0
            return 0.0

        od = OutputDir(tmp_output_dir, "baseline_progress").ensure()
        reporter = MethodProgressReporter("abc_smc_baseline", replicate=0, interval_s=0.5, rank=0)

        with caplog.at_level(logging.INFO, logger="async_abc.utils.progress"):
            reporter.start(total_hint=8, detail="mode=test")
            records = run_abc_smc_baseline(
                simulate,
                {"mu": (-1.0, 1.0)},
                {**_test_inference_cfg(), "max_simulations": 8, "n_generations": 2, "n_workers": 1},
                od,
                replicate=0,
                seed=1,
                progress=reporter,
            )

        assert records
        messages = [record.getMessage() for record in caplog.records]
        assert any("status=update" in message and "simulations=" in message for message in messages)
        assert any("status=finish" in message and "generations=" in message for message in messages)


class TestRunPropulateAbc:
    def test_completes_in_reasonable_time(self, fake_propulate_env, tmp_path_factory):
        t0 = time.time()
        records = _run_fake_propulate(tmp_path_factory, seed=19)
        assert time.time() - t0 < 5.0
        assert records

    def test_returns_list_of_particle_records(self, propulate_records_default):
        assert all(isinstance(record, ParticleRecord) for record in propulate_records_default)

    def test_records_have_required_fields(self, propulate_records_default):
        record = propulate_records_default[0]
        assert record.method == "async_propulate_abc"
        assert record.replicate == 0
        assert record.seed == 7
        assert isinstance(record.step, int) and record.step >= 1
        assert "mu" in record.params
        assert math.isfinite(record.loss) and record.loss >= 0.0
        assert record.wall_time >= 0.0
        assert record.record_kind == "simulation_attempt"
        assert record.time_semantics == "event_end"
        assert record.attempt_count is not None and record.attempt_count >= 1

    def test_records_have_sim_end_time(self, propulate_records_default):
        assert all(record.sim_end_time is not None for record in propulate_records_default)
        assert all(record.wall_time == pytest.approx(record.sim_end_time) for record in propulate_records_default)
        assert [record.attempt_count for record in propulate_records_default] == [record.step for record in propulate_records_default]

    def test_sim_end_time_is_none_when_evaltime_absent(self, propulate_records_no_evaltime):
        assert all(record.sim_end_time is None for record in propulate_records_no_evaltime)
        assert all(record.wall_time == 0.0 for record in propulate_records_no_evaltime)

    def test_records_count_matches_max_simulations(self, propulate_records_default):
        assert len(propulate_records_default) == _test_inference_cfg()["max_simulations"]

    def test_test_mode_scales_generation_budget_by_world_size(self, monkeypatch):
        import async_abc.inference.propulate_abc as mod

        monkeypatch.setattr(mod, "_propulate_world_size", lambda: 48)

        assert mod._effective_generation_budget(100, {"test_mode": True}) == 3
        assert mod._effective_generation_budget(100, {"test_mode": False}) == 100

    def test_total_simulation_budget_scales_generation_budget_by_world_size(self, monkeypatch):
        import async_abc.inference.propulate_abc as mod

        monkeypatch.setattr(mod, "_propulate_world_size", lambda: 48)

        assert mod._effective_generation_budget(
            100, {"propulate_budget_mode": "total_simulations", "test_mode": False}
        ) == 3

    def test_wall_time_cap_stops_propulate_early(self, tmp_path, monkeypatch):
        import async_abc.inference.propulate_abc as mod
        from async_abc.io.paths import OutputDir

        original = (mod.Propulator, mod.ABCPMC)
        clock = _Clock(start=100.0)
        _WallTimeLocalPropulator.clock = clock
        monkeypatch.setattr(mod.time, "time", clock)
        monkeypatch.setattr(mod, "_make_propulate_comm", lambda: None)
        mod.Propulator = _WallTimeLocalPropulator
        mod.ABCPMC = _FakeABCPMC
        try:
            bm = _gaussian_bm()
            od = OutputDir(tmp_path, "propulate_walltime").ensure()
            records = run_propulate_abc(
                bm.simulate,
                bm.limits,
                {**_test_inference_cfg(), "max_simulations": 50, "max_wall_time_s": 0.05},
                od,
                replicate=0,
                seed=31,
            )
        finally:
            mod.Propulator, mod.ABCPMC = original

        assert 0 < len(records) < 50
        assert records[-1].wall_time <= 0.06
        assert [record.step for record in records] == list(range(1, len(records) + 1))

    def test_steps_are_monotone(self, propulate_records_default):
        steps = [record.step for record in propulate_records_default]
        assert steps == sorted(steps)

    def test_tolerance_non_increasing_once_set(self, propulate_records_tolerance):
        tolerances = [record.tolerance for record in propulate_records_tolerance if record.tolerance is not None]
        assert tolerances == sorted(tolerances, reverse=True)

    def test_different_seeds_give_different_results(self, propulate_records_default, propulate_records_alt_seed):
        losses_a = [record.loss for record in propulate_records_default]
        losses_b = [record.loss for record in propulate_records_alt_seed]
        assert losses_a != losses_b

    def test_stable_seed_changes_with_rank(self):
        import async_abc.inference.propulate_abc as mod

        seed_rank_0 = mod._stable_seed(7, "propagator", 0)
        seed_rank_1 = mod._stable_seed(7, "propagator", 1)

        assert seed_rank_0 != seed_rank_1

    def test_eval_seed_changes_with_rank_and_params(self):
        import async_abc.inference.propulate_abc as mod

        params_a = {"mu": 0.1}
        params_b = {"mu": 0.2}

        assert mod._eval_seed(11, 0, 2, params_a) != mod._eval_seed(11, 1, 2, params_a)
        assert mod._eval_seed(11, 0, 2, params_a) != mod._eval_seed(11, 0, 2, params_b)
        assert mod._eval_seed(11, 0, 2, params_a) == mod._eval_seed(11, 0, 2, {"mu": 0.1})

    def test_replicate_index_stored(self, propulate_records_alt_seed):
        assert all(record.replicate == 1 for record in propulate_records_alt_seed)

    def test_uses_isolated_comm_for_parallel_propulate_runs(self, tmp_path_factory, monkeypatch):
        pytest.importorskip("mpi4py", reason="mpi4py not installed")
        import async_abc.inference.propulate_abc as mod

        original = (mod.Propulator, mod.ABCPMC)
        freed = []

        class _FakeComm:
            pass

        fake_comm = _FakeComm()
        monkeypatch.setattr(mod, "_make_propulate_comm", lambda: fake_comm)
        monkeypatch.setattr(mod, "_free_propulate_comm", lambda comm: freed.append(comm))
        mod.Propulator = _CapturingPropulator
        mod.ABCPMC = _FakeABCPMC
        try:
            records = _run_fake_propulate(tmp_path_factory, seed=23)
        finally:
            mod.Propulator, mod.ABCPMC = original

        assert records
        assert _CapturingPropulator.last_kwargs["island_comm"] is fake_comm
        assert _CapturingPropulator.last_kwargs["propulate_comm"] is fake_comm
        assert freed == [fake_comm]

    def test_test_mode_clears_stale_propulate_checkpoint(self, tmp_path):
        import async_abc.inference.propulate_abc as mod
        from async_abc.io.paths import OutputDir

        original = (mod.Propulator, mod.ABCPMC)
        _CheckpointInspectingPropulator.saw_stale_marker = None
        mod.Propulator = _CheckpointInspectingPropulator
        mod.ABCPMC = _FakeABCPMC
        try:
            bm = _gaussian_bm()
            od = OutputDir(tmp_path, "propulate_checkpoint_cleanup").ensure()
            checkpoint_dir = od.logs / "propulate_rep0_seed29"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "stale.txt").write_text("stale")

            records = run_propulate_abc(
                bm.simulate,
                bm.limits,
                {**_test_inference_cfg(), "test_mode": True},
                od,
                replicate=0,
                seed=29,
            )
        finally:
            mod.Propulator, mod.ABCPMC = original

        assert records
        assert _CheckpointInspectingPropulator.saw_stale_marker is False

    def test_wall_time_parallel_propulate_cleans_up_intra_island_sends(self, tmp_path, monkeypatch):
        import async_abc.inference.propulate_abc as mod
        from async_abc.io.paths import OutputDir

        original = (mod.Propulator, mod.ABCPMC)
        clock = _Clock(start=100.0)
        fake_comm = _FakeBarrierComm()
        freed = []

        _WallTimeCommPropulator.clock = clock
        _WallTimeCommPropulator.last_instance = None
        monkeypatch.setattr(mod.time, "time", clock)
        monkeypatch.setattr(mod, "_make_propulate_comm", lambda: fake_comm)
        monkeypatch.setattr(mod, "_free_propulate_comm", lambda comm: freed.append(comm))
        mod.Propulator = _WallTimeCommPropulator
        mod.ABCPMC = _FakeABCPMC
        try:
            bm = _gaussian_bm()
            od = OutputDir(tmp_path, "propulate_walltime_parallel").ensure()
            records = run_propulate_abc(
                bm.simulate,
                bm.limits,
                {**_test_inference_cfg(), "max_simulations": 50, "max_wall_time_s": 0.05},
                od,
                replicate=0,
                seed=37,
            )
        finally:
            mod.Propulator, mod.ABCPMC = original

        propulator = _WallTimeCommPropulator.last_instance
        assert records
        assert propulator is not None
        assert propulator.intra_requests == []
        assert propulator.final_dump_calls == 1
        assert freed == [fake_comm]

    def test_incompatible_individual_raises_clear_error(self, fake_propulate_env, tmp_path):
        from async_abc.io.paths import OutputDir

        class _BadIndividual(_FakeIndividual):
            def __getitem__(self, key):
                raise KeyError(key)

            def keys(self):
                return ["alpha", "beta"]

        class _BadPropulator(_FakePropulator):
            def propulate(self, **kwargs):
                self.population = [
                    _BadIndividual(
                        {"alpha": 1.0, "beta": 2.0},
                        generation=0,
                        tolerance=1.0,
                        evaltime=time.time(),
                        evalperiod=0.01,
                        rank=0,
                        loss=0.0,
                    )
                ]

        import async_abc.inference.propulate_abc as mod

        original = mod.Propulator
        mod.Propulator = _BadPropulator
        try:
            bm = _gaussian_bm()
            od = OutputDir(tmp_path, "bad_individual").ensure()
            with pytest.raises(RuntimeError, match="checkpoint reuse or MPI message cross-contamination"):
                run_propulate_abc(
                    bm.simulate,
                    bm.limits,
                    _test_inference_cfg(),
                    od,
                    replicate=0,
                    seed=29,
                )
        finally:
            mod.Propulator = original


class TestPyabcWrapper:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_pyabc_runs_on_gaussian(self, pyabc_records_default):
        assert isinstance(pyabc_records_default, list)
        assert all(isinstance(record, ParticleRecord) for record in pyabc_records_default)


class TestPyabcWrapperFixes:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_particle_loss_is_actual_distance(self, pyabc_records_default):
        population_records = _population_records(pyabc_records_default)
        first_gen_losses = [record.loss for record in population_records[:10] if record.loss is not None]
        assert len(set(round(loss, 8) for loss in first_gen_losses)) > 1

    def test_tolerance_field_is_generation_epsilon(self, pyabc_records_default):
        population_records = _population_records(pyabc_records_default)
        assert population_records
        assert all(record.tolerance is not None and record.tolerance >= 0.0 for record in population_records)

    def test_singlecore_sampler_when_n_workers_1(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        records = run_pyabc_smc(bm.simulate, bm.limits, {**_test_inference_cfg(), "n_workers": 1}, od, replicate=0, seed=1)
        assert records

    def test_mpi_sampler_when_n_workers_gt_1(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.pyabc_wrapper as wrapper_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            wrapper_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        run_pyabc_smc(bm.simulate, bm.limits, {**_test_inference_cfg(), "n_workers": 2}, od, replicate=0, seed=1)
        assert calls == [("mpi", "mapping", True, False, None)]

    def test_same_seed_reproducible(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        bm = _gaussian_bm()
        od1 = OutputDir(Path(tmp_output_dir) / "r1", "pyabc").ensure()
        od2 = OutputDir(Path(tmp_output_dir) / "r2", "pyabc").ensure()
        r1 = run_pyabc_smc(bm.simulate, bm.limits, _test_inference_cfg(), od1, replicate=0, seed=42)
        r2 = run_pyabc_smc(bm.simulate, bm.limits, _test_inference_cfg(), od2, replicate=0, seed=42)
        assert r1[0].params == r2[0].params

    def test_seed_derivation_is_stable_across_subprocesses(self):
        code = "\n".join(
            [
                "from async_abc.utils.seeding import canonical_param_key, stable_seed",
                "params = {'mu': 1.23456789}",
                "print(stable_seed(42, canonical_param_key(params)))",
            ]
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
        first = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
        second = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
        assert first == second

    def test_epsilon_extraction_robust(self, pyabc_records_default):
        population_records = _population_records(pyabc_records_default)
        assert population_records
        assert all(record.tolerance is not None for record in population_records)

    def test_wall_time_uses_observable_generation_snapshots(self, pyabc_records_default):
        population_records = _population_records(pyabc_records_default)
        assert population_records
        assert len({round(record.wall_time, 8) for record in population_records}) > 1

    def test_all_records_complete_schema(self, pyabc_records_default):
        population_records = _population_records(pyabc_records_default)
        assert population_records
        for record in population_records:
            assert record.method == "pyabc_smc"
            assert isinstance(record.step, int) and record.step >= 1
            assert math.isfinite(record.loss) and record.loss >= 0.0
            assert record.tolerance is not None and record.tolerance >= 0.0
            assert record.wall_time >= 0.0
            assert "mu" in record.params
            assert record.record_kind == "population_particle"
            assert record.time_semantics == "generation_end"
            assert record.generation is not None
            assert record.attempt_count is not None and record.attempt_count >= 0

    def test_multicore_parallel_population_losses_are_finite(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc_parallel").ensure()
        records = run_pyabc_smc(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "max_simulations": 20, "n_workers": 2, "parallel_backend": "multicore"},
            od,
            replicate=0,
            seed=42,
        )
        population_records = _population_records(records)
        assert population_records
        assert all(math.isfinite(record.loss) for record in population_records)


class TestRejectionAbc:
    def test_importable(self):
        from async_abc.inference.rejection_abc import run_rejection_abc

        assert callable(run_rejection_abc)

    def test_in_registry(self):
        assert "rejection_abc" in METHOD_REGISTRY

    def test_returns_particle_records(self, rejection_records_default):
        assert isinstance(rejection_records_default, list)
        assert all(isinstance(record, ParticleRecord) for record in rejection_records_default)

    def test_method_name(self, rejection_records_default):
        assert all(record.method == "rejection_abc" for record in rejection_records_default)

    def test_all_accepted_within_tolerance(self, rejection_records_default):
        assert all(record.loss <= _test_inference_cfg()["tol_init"] + 1e-9 for record in rejection_records_default)

    def test_tolerance_field_equals_tol_init(self, rejection_records_tol3):
        assert all(record.tolerance == pytest.approx(3.0) for record in rejection_records_tol3)
        assert all(record.record_kind == "accepted_particle" for record in rejection_records_tol3)
        assert all(record.time_semantics == "event_end" for record in rejection_records_tol3)

    def test_steps_monotone(self, rejection_records_default):
        steps = [record.step for record in rejection_records_default]
        assert steps == sorted(steps)
        assert steps[0] >= 1
        attempts = [record.attempt_count for record in rejection_records_default]
        assert attempts == sorted(attempts)
        assert all(attempt >= step for attempt, step in zip(attempts, steps))

    def test_returns_at_most_k_particles(self, rejection_records_small_k):
        assert len(rejection_records_small_k) <= 3

    def test_rejection_abc_respects_max_wall_time_s(self, tmp_path):
        """rejection_abc must stop early when max_wall_time_s is exceeded."""
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc

        bm = _gaussian_bm()
        od = OutputDir(tmp_path, "rejection_walltime").ensure()

        def slow_simulate(params, seed):
            time.sleep(0.05)
            return 0.0  # always accept

        cfg = {
            **_test_inference_cfg(),
            "max_simulations": 10_000,
            "k": 10_000,
            "tol_init": 100.0,
            "max_wall_time_s": 0.2,
        }
        records = run_rejection_abc(
            slow_simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        # With 0.05s sleep per sim and 0.2s budget, expect ~4 accepted
        # (certainly far fewer than 10_000)
        assert 0 < len(records) < 100
        assert records[-1].wall_time <= 0.35  # generous tolerance

    def test_rejection_abc_without_wall_time_runs_full_budget(self, tmp_path):
        """Without max_wall_time_s, rejection_abc runs until k or max_simulations."""
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc

        bm = _gaussian_bm()
        od = OutputDir(tmp_path, "rejection_no_walltime").ensure()

        cfg = {
            **_test_inference_cfg(),
            "max_simulations": 20,
            "k": 5,
            "tol_init": 100.0,
            # no max_wall_time_s
        }
        records = run_rejection_abc(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        # tol_init=100 means everything is accepted → should get exactly k=5
        assert len(records) == 5

    def test_rejection_abc_uses_strict_less_than(self, tmp_path):
        """A particle with loss == tol_init should NOT be accepted (strict <)."""
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc

        od = OutputDir(tmp_path, "rejection_strict_lt").ensure()

        def exact_tol_simulate(params, seed):
            return 5.0  # exactly equal to tol_init

        cfg = {
            **_test_inference_cfg(),
            "max_simulations": 10,
            "k": 10,
            "tol_init": 5.0,
        }
        records = run_rejection_abc(
            exact_tol_simulate, {"x": (-1.0, 1.0)}, cfg, od, replicate=0, seed=1,
        )
        # loss == tol_init → strict < means none accepted
        assert len(records) == 0

    def test_method_execution_mode_unknown_raises(self):
        """method_execution_mode with an unknown name should raise KeyError."""
        from async_abc.inference.method_registry import method_execution_mode
        with pytest.raises(KeyError, match="not_a_method"):
            method_execution_mode("not_a_method")


class TestAbcSmcBaseline:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_importable(self):
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        assert callable(run_abc_smc_baseline)

    def test_in_registry(self):
        assert "abc_smc_baseline" in METHOD_REGISTRY

    def test_static_execution_mode_defaults_to_rank_zero(self):
        assert method_execution_mode("abc_smc_baseline") == "rank_zero"

    def test_returns_particle_records(self, abc_smc_baseline_records_default):
        assert all(isinstance(record, ParticleRecord) for record in abc_smc_baseline_records_default)

    def test_method_name(self, abc_smc_baseline_records_default):
        assert all(record.method == "abc_smc_baseline" for record in abc_smc_baseline_records_default)

    def test_tolerance_non_increasing(self, abc_smc_baseline_records_tolerance):
        tolerances = [
            record.tolerance
            for record in _population_records(abc_smc_baseline_records_tolerance)
            if record.tolerance is not None
        ]
        assert tolerances == sorted(tolerances, reverse=True)

    def test_loss_is_actual_distance(self, abc_smc_baseline_records_loss):
        first_gen = _population_records(abc_smc_baseline_records_loss)[:10]
        if len(first_gen) > 1:
            assert len(set(round(record.loss, 8) for record in first_gen)) > 1

    def test_steps_monotone(self, abc_smc_baseline_records_default):
        steps = [record.step for record in _population_records(abc_smc_baseline_records_default)]
        assert steps == sorted(steps)

    def test_n_generations_respected(self, abc_smc_baseline_records_generation):
        assert abc_smc_baseline_records_generation

    def test_n_generations_defaults(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        records = run_abc_smc_baseline(bm.simulate, bm.limits, _test_inference_cfg(), od, 0, 1)
        assert isinstance(records, list)

    def test_records_have_generation(self, abc_smc_baseline_records_generation):
        generations = sorted({record.generation for record in _population_records(abc_smc_baseline_records_generation)})
        assert generations == list(range(len(generations)))

    def test_records_have_generation_timing(self, abc_smc_baseline_records_default):
        timed = [
            record
            for record in _population_records(abc_smc_baseline_records_default)
            if record.sim_start_time is not None and record.sim_end_time is not None
        ]
        assert timed
        assert all(record.sim_end_time >= record.sim_start_time for record in timed)

    def test_multicore_parallel_population_losses_are_finite(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc_parallel").ensure()
        records = run_abc_smc_baseline(
            bm.simulate,
            bm.limits,
            {
                **_test_inference_cfg(),
                "max_simulations": 20,
                "n_generations": 2,
                "n_workers": 2,
                "parallel_backend": "multicore",
            },
            od,
            replicate=0,
            seed=42,
        )
        population_records = _population_records(records)
        assert population_records
        assert all(math.isfinite(record.loss) for record in population_records)

    def test_abc_smc_baseline_defaults_high_n_generations_with_wall_time(self, tmp_output_dir):
        """When max_wall_time_s is set and n_generations is absent, use high default."""
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc_walltime").ensure()
        cfg = {
            **_test_inference_cfg(),
            "max_simulations": 50,
            "max_wall_time_s": 0.5,
            # n_generations intentionally absent — should default high
        }
        records = run_abc_smc_baseline(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        # Should produce records (stopped by wall time or max_sims, not by
        # a low n_generations=5 default)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_abc_smc_baseline_keeps_default_n_generations_with_wall_time(
        self, tmp_output_dir, monkeypatch,
    ):
        """max_wall_time_s no longer inflates n_generations."""
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(baseline_mod, "_run_abc_smc_baseline_with_sampler", spy)

        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc_ngen").ensure()
        cfg = {
            **_test_inference_cfg(),
            "max_simulations": 50,
            "max_wall_time_s": 1.0,
            # n_generations intentionally absent
        }
        run_abc_smc_baseline(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        assert captured["n_generations"] == 5, (
            f"Expected default n_generations=5 when max_wall_time_s is set, "
            f"got {captured['n_generations']}"
        )


class TestBuildPyabcSampler:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_multicore_n1_returns_singlecore_sampler(self):
        import pyabc
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        sampler = build_pyabc_sampler(n_procs=1, parallel_backend="multicore")
        assert isinstance(sampler, pyabc.SingleCoreSampler)

    def test_multicore_n4_returns_multicore_sampler(self, monkeypatch):
        import pyabc
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        calls = []
        original = pyabc.MulticoreEvalParallelSampler
        monkeypatch.setattr(pyabc, "MulticoreEvalParallelSampler", lambda n: (calls.append(n), original(n))[1])
        build_pyabc_sampler(n_procs=4, parallel_backend="multicore")
        assert calls == [4]

    def test_mpi_backend_defaults_to_mapping_sampler(self):
        import pyabc
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        sampler = build_pyabc_sampler(
            n_procs=8,
            parallel_backend="mpi",
            mpi_map=lambda f, xs: list(map(f, xs)),
        )
        assert isinstance(sampler, pyabc.MappingSampler)

    def test_mpi_concurrent_futures_backend_returns_concurrent_future_sampler(self):
        import pyabc
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        sampler = build_pyabc_sampler(
            n_procs=8,
            parallel_backend="mpi",
            mpi_sampler="concurrent_futures",
            cfuture_executor=object(),
        )
        assert isinstance(sampler, pyabc.ConcurrentFutureSampler)

    def test_mpi_concurrent_futures_backend_forwards_client_max_jobs(self, monkeypatch):
        import pyabc
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        calls = []
        original = pyabc.ConcurrentFutureSampler

        def _spy(*args, **kwargs):
            calls.append(kwargs["client_max_jobs"])
            return original(*args, **kwargs)

        monkeypatch.setattr(pyabc, "ConcurrentFutureSampler", _spy)
        sampler = build_pyabc_sampler(
            n_procs=8,
            parallel_backend="mpi",
            mpi_sampler="concurrent_futures",
            cfuture_executor=object(),
            client_max_jobs=8,
        )
        assert isinstance(sampler, original)
        assert calls == [8]

    def test_mpi_mapping_backend_requires_existing_map(self):
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        with pytest.raises(ValueError, match="communicator-backed map callable"):
            build_pyabc_sampler(n_procs=16, parallel_backend="mpi", mpi_sampler="mapping")

    def test_mpi_concurrent_futures_backend_requires_existing_executor(self):
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        with pytest.raises(ValueError, match="communicator-backed executor"):
            build_pyabc_sampler(
                n_procs=16,
                parallel_backend="mpi",
                mpi_sampler="concurrent_futures",
            )

    def test_mpi_n1_still_uses_mapping_sampler_by_default(self):
        import pyabc
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        sampler = build_pyabc_sampler(
            n_procs=1,
            parallel_backend="mpi",
            mpi_map=lambda f, xs: list(map(f, xs)),
        )
        assert isinstance(sampler, pyabc.MappingSampler)

    def test_invalid_backend_raises_value_error(self):
        from async_abc.inference.pyabc_sampler import build_pyabc_sampler

        with pytest.raises(ValueError, match="parallel_backend"):
            build_pyabc_sampler(n_procs=2, parallel_backend="redis")

    def test_resolve_pyabc_client_max_jobs_defaults_to_worker_count_for_mpi(self):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_client_max_jobs

        assert (
            resolve_pyabc_client_max_jobs(
                {},
                parallel_backend="mpi",
                n_procs=7,
                mpi_sampler="concurrent_futures",
            )
            == 7
        )

    def test_resolve_pyabc_client_max_jobs_honors_explicit_override(self):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_client_max_jobs

        assert (
            resolve_pyabc_client_max_jobs(
                {"pyabc_client_max_jobs": 3},
                parallel_backend="mpi",
                n_procs=7,
                mpi_sampler="concurrent_futures",
            )
            == 3
        )

    def test_resolve_pyabc_client_max_jobs_ignored_for_mapping(self, caplog):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_client_max_jobs

        with caplog.at_level(logging.WARNING):
            assert (
                resolve_pyabc_client_max_jobs(
                    {"pyabc_client_max_jobs": 3},
                    parallel_backend="mpi",
                    n_procs=7,
                    mpi_sampler="mapping",
                )
                is None
            )
        assert "Ignoring pyabc_client_max_jobs=3" in caplog.text

    def test_resolve_pyabc_client_max_jobs_ignored_for_non_mpi(self):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_client_max_jobs

        assert (
            resolve_pyabc_client_max_jobs(
                {"pyabc_client_max_jobs": 3},
                parallel_backend="multicore",
                n_procs=7,
            )
            is None
        )

    def test_resolve_pyabc_mpi_sampler_defaults_to_mapping(self):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_mpi_sampler

        assert (
            resolve_pyabc_mpi_sampler(
                {},
                parallel_backend="mpi",
                method_name="pyabc_smc",
            )
            == "mapping"
        )

    def test_resolve_pyabc_mpi_sampler_honors_explicit_mapping(self):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_mpi_sampler

        assert (
            resolve_pyabc_mpi_sampler(
                {"pyabc_mpi_sampler": "mapping"},
                parallel_backend="mpi",
                method_name="abc_smc_baseline",
            )
            == "mapping"
        )

    def test_resolve_pyabc_mpi_sampler_honors_legacy_alias_with_warning(self, caplog):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_mpi_sampler

        with caplog.at_level(logging.WARNING):
            assert (
                resolve_pyabc_mpi_sampler(
                    {"pyabc_mpi_sampler": "concurrent_futures_legacy"},
                    parallel_backend="mpi",
                    method_name="abc_smc_baseline",
                )
                == "concurrent_futures"
            )
        assert "deprecated; use pyabc_mpi_sampler=concurrent_futures instead" in caplog.text

    def test_resolve_pyabc_mpi_sampler_ignored_for_non_mpi(self):
        from async_abc.inference.pyabc_sampler import resolve_pyabc_mpi_sampler

        assert (
            resolve_pyabc_mpi_sampler(
                {},
                parallel_backend="multicore",
                method_name="pyabc_smc",
            )
            is None
        )

    def test_tracked_future_executor_waits_only_for_non_cancelled_pending_futures(self, monkeypatch):
        from async_abc.inference.pyabc_sampler import TrackedFutureExecutor

        waited = []

        class _InnerExecutor:
            def submit(self, fn, /, *args, **kwargs):
                return fn(*args, **kwargs)

        class _FakeFuture:
            def __init__(self, *, done=False, cancelled=False):
                self._done = done
                self._cancelled = cancelled

            def done(self):
                return self._done

            def cancelled(self):
                return self._cancelled

        tracker = TrackedFutureExecutor(_InnerExecutor())
        futures = [
            _FakeFuture(done=False, cancelled=False),
            _FakeFuture(done=False, cancelled=True),
            _FakeFuture(done=True, cancelled=False),
        ]
        for future in futures:
            tracker.submit(lambda future=future: future)

        import async_abc.inference.pyabc_sampler as sampler_mod

        monkeypatch.setattr(
            sampler_mod,
            "_wait",
            lambda pending: waited.append(list(pending)),
        )
        waited_count = tracker.wait_for_pending(exclude_cancelled=True)

        assert waited_count == 1
        assert waited == [[futures[0]]]


class TestPyabcCommon:
    """Tests for shared pyABC utilities in _pyabc_common."""

    def test_db_suffix_empty(self):
        from async_abc.inference._pyabc_common import db_suffix
        assert db_suffix("") == ""

    def test_db_suffix_with_tag(self):
        from async_abc.inference._pyabc_common import db_suffix
        assert db_suffix("my_tag") == "__my_tag"

    def test_db_suffix_sanitizes_special_chars(self):
        from async_abc.inference._pyabc_common import db_suffix
        result = db_suffix("a/b:c")
        assert "/" not in result
        assert ":" not in result

    def test_prepare_db_path_matches_old_behavior(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference._pyabc_common import prepare_db_path

        od = OutputDir(tmp_output_dir, "test_common").ensure()
        path = prepare_db_path(
            od, method_name="pyabc_smc", replicate=0, seed=42, checkpoint_tag="",
        )
        assert path.startswith("sqlite:///")
        assert "pyabc_smc_rep0_seed42" in path


class TestPyabcWrapperMpiBackend:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_mpi_backend_key_is_forwarded(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.pyabc_wrapper as wrapper_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            wrapper_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc_mpi").ensure()
        run_pyabc_smc(bm.simulate, bm.limits, {**_test_inference_cfg(), "parallel_backend": "mpi", "n_workers": 2}, od, replicate=0, seed=1)
        assert calls == [("mpi", "mapping", True, False, None)]

    def test_mpi_legacy_backend_is_forwarded(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.pyabc_wrapper as wrapper_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            wrapper_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None and cfuture_executor._inner is executor, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc_mpi_legacy").ensure()
        run_pyabc_smc(
            bm.simulate,
            bm.limits,
            {
                **_test_inference_cfg(),
                "parallel_backend": "mpi",
                "n_workers": 2,
                "pyabc_mpi_sampler": "concurrent_futures",
            },
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [("mpi", "concurrent_futures", False, True, 2)]

    def test_absent_parallel_backend_defaults_to_multicore(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.pyabc_wrapper as wrapper_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        calls = []
        monkeypatch.setattr(
            wrapper_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc_default").ensure()
        run_pyabc_smc(bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1)
        assert calls == [("multicore", None)]

    def test_parallel_run_defaults_to_mpi(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.pyabc_wrapper as wrapper_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            wrapper_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc_parallel_default").ensure()
        run_pyabc_smc(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "n_workers": 4},
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [("mpi", "mapping", True, False, None)]

    def test_parallel_multicore_request_is_overridden_to_mpi(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.pyabc_wrapper as wrapper_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            wrapper_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc_parallel_override").ensure()
        run_pyabc_smc(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "parallel_backend": "multicore", "n_workers": 4},
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [("mpi", "mapping", True, False, None)]

    def test_parallel_default_uses_mpi_even_for_benchmarks_marked_unsafe(
        self, tmp_output_dir, monkeypatch
    ):
        import pyabc
        import async_abc.inference.pyabc_wrapper as wrapper_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            wrapper_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((n_procs, parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        bm.MULTIPROCESSING_SAFE = False
        od = OutputDir(tmp_output_dir, "pyabc_cpm_safe").ensure()
        run_pyabc_smc(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "n_workers": 4},
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [(4, "mpi", "mapping", True, False, None)]

    def test_non_root_returns_no_records_under_mpi_backend(self):
        """Non-root behaviour is tested via MPI integration tests.

        With CommWorldMap (default mapping path), the worker loop only runs
        when Get_size() > 1, which requires real MPI.  The concurrent_futures
        legacy path still has non-root semantics via MPICommExecutor but is
        opt-in only.
        """
        pytest.skip("Non-root path requires real MPI (CommWorldMap worker_loop)")


class TestAbcSmcBaselineMpiBackend:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_mpi_backend_key_is_forwarded(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            baseline_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "smc_mpi").ensure()
        run_abc_smc_baseline(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "n_generations": 2, "parallel_backend": "mpi", "n_workers": 2},
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [("mpi", "mapping", True, False, None)]

    def test_mpi_legacy_backend_key_is_forwarded(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            baseline_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None and cfuture_executor._inner is executor, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "smc_mpi_legacy").ensure()
        run_abc_smc_baseline(
            bm.simulate,
            bm.limits,
            {
                **_test_inference_cfg(),
                "n_generations": 2,
                "parallel_backend": "mpi",
                "n_workers": 2,
                "pyabc_mpi_sampler": "concurrent_futures",
            },
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [("mpi", "concurrent_futures", False, True, 2)]

    def test_absent_parallel_backend_defaults_to_multicore(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        calls = []
        monkeypatch.setattr(
            baseline_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "smc_default").ensure()
        run_abc_smc_baseline(bm.simulate, bm.limits, {**_test_inference_cfg(), "n_generations": 2}, od, replicate=0, seed=1)
        assert calls == [("multicore", None)]

    def test_parallel_run_defaults_to_mpi(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            baseline_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "smc_parallel_default").ensure()
        run_abc_smc_baseline(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "n_generations": 2, "n_workers": 4},
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [("mpi", "mapping", True, False, None)]

    def test_parallel_multicore_request_is_overridden_to_mpi(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            baseline_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "smc_parallel_override").ensure()
        run_abc_smc_baseline(
            bm.simulate,
            bm.limits,
            {
                **_test_inference_cfg(),
                "n_generations": 2,
                "parallel_backend": "multicore",
                "n_workers": 4,
            },
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [("mpi", "mapping", True, False, None)]

    def test_parallel_default_uses_mpi_even_for_benchmarks_marked_unsafe(
        self, tmp_output_dir, monkeypatch
    ):
        import pyabc
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        calls = []
        executor = _make_fake_executor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            baseline_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: (
                calls.append((n_procs, parallel_backend, mpi_sampler, callable(mpi_map), cfuture_executor is not None, client_max_jobs)),
                pyabc.SingleCoreSampler(),
            )[1],
            raising=False,
        )
        bm = _gaussian_bm()
        bm.MULTIPROCESSING_SAFE = False
        od = OutputDir(tmp_output_dir, "smc_cpm_safe").ensure()
        run_abc_smc_baseline(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "n_generations": 2, "n_workers": 4},
            od,
            replicate=0,
            seed=1,
        )
        assert calls == [(4, "mpi", "mapping", True, False, None)]

    def test_non_root_returns_no_records_under_mpi_backend(self):
        """Non-root behaviour is tested via MPI integration tests.

        With CommWorldMap (default mapping path), the worker loop only runs
        when Get_size() > 1, which requires real MPI.
        """
        pytest.skip("Non-root path requires real MPI (CommWorldMap worker_loop)")

    def test_does_not_call_shutdown_inside_mpi_context(self, tmp_output_dir, monkeypatch):
        import pyabc
        import async_abc.inference.abc_smc_baseline as baseline_mod
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline

        executor = _RecordingExecutor()
        _install_fake_mpi_executor(monkeypatch, executor=executor)
        monkeypatch.setattr(
            baseline_mod,
            "build_pyabc_sampler",
            lambda n_procs, parallel_backend, mpi_sampler=None, mpi_map=None, cfuture_executor=None, client_max_jobs=None: pyabc.SingleCoreSampler(),
            raising=False,
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "smc_no_inner_shutdown").ensure()

        run_abc_smc_baseline(
            bm.simulate,
            bm.limits,
            {**_test_inference_cfg(), "n_generations": 2, "parallel_backend": "mpi", "n_workers": 2},
            od,
            replicate=0,
            seed=1,
        )

        assert executor.shutdown_calls == []


class TestMpiIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_mpirun_not_usable(self):
        if shutil.which("mpirun") is None:
            pytest.skip("mpirun not on PATH")
        probe = subprocess.run(
            ["mpirun", "-n", "1", "--stdin", "none", sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if probe.returncode != 0:
            pytest.skip("mpirun failed a trivial probe")

    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    @pytest.fixture(autouse=True)
    def skip_if_no_mpi4py(self):
        pytest.importorskip("mpi4py", reason="mpi4py not installed — skipping")

    def test_mpi_backend_via_mpirun(self, tmp_path):
        helper = Path(__file__).parent / "mpi_integration_helper.py"
        output_file = tmp_path / "mpi_result.json"
        result = subprocess.run(
            [
                "mpirun",
                "-n",
                "2",
                "--stdin",
                "none",
                sys.executable,
                str(helper),
                str(output_file),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(output_file.read_text())
        assert data["n_records"] > 0
        assert data["method"] == "pyabc_smc"
        assert data["pyabc_mpi_sampler"] == "mapping"
        assert data["barrier_reached"]

    def test_abc_smc_baseline_shutdown_does_not_hang(self, tmp_path):
        """Verify wall-time-limited abc_smc_baseline MPI runs tear down quickly.

        The default futures-based MPI sampler must finish well within the
        timeout and all ranks must reach the post-executor barrier.
        """
        helper = Path(__file__).parent / "mpi_abc_smc_baseline_helper.py"
        output_file = tmp_path / "abc_smc_baseline_result.json"
        result = subprocess.run(
            [
                "mpirun",
                "-n",
                "2",
                "--stdin",
                "none",
                sys.executable,
                str(helper),
                str(output_file),
                "concurrent_futures",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(output_file.read_text())
        assert data["n_records"] > 0
        assert data["method"] == "abc_smc_baseline"
        assert data["world_size"] == 2
        assert data["pyabc_mpi_sampler"] == "concurrent_futures"
        assert data["client_max_jobs"] is None
        assert data["max_elapsed_s"] < 60.0
        assert data["elapsed_spread_s"] < 20.0

    @pytest.mark.parametrize(
        ("n_ranks", "mpi_sampler", "client_max_jobs", "max_wall_time_s", "timeout_s"),
        [
            (2, "concurrent_futures", None, None, 60),
            (4, "concurrent_futures", None, None, 90),
        ],
    )
    def test_abc_smc_baseline_mpi_diagnostics_bounded_backlog(
        self,
        tmp_path,
        n_ranks,
        mpi_sampler,
        client_max_jobs,
        max_wall_time_s,
        timeout_s,
    ):
        helper = Path(__file__).parent / "mpi_abc_smc_baseline_helper.py"
        output_file = tmp_path / f"abc_smc_baseline_diag_n{n_ranks}.json"
        result = subprocess.run(
            [
                "mpirun",
                "-n",
                str(n_ranks),
                "--stdin",
                "none",
                sys.executable,
                str(helper),
                str(output_file),
                mpi_sampler,
                "none" if client_max_jobs is None else str(client_max_jobs),
                str(max_wall_time_s),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(output_file.read_text())
        assert data["n_records"] > 0
        assert data["method"] == "abc_smc_baseline"
        assert data["world_size"] == n_ranks
        assert data["pyabc_mpi_sampler"] == mpi_sampler
        assert data["client_max_jobs"] == client_max_jobs
        assert data["barrier_reached"], "COMM_WORLD barrier was not reached"
        assert data["max_elapsed_s"] < timeout_s
        # Keep this generous: we want a diagnostic guardrail, not a brittle
        # exact-performance assertion tied to one MPI implementation.
        assert data["elapsed_spread_s"] < 45.0
        assert data["barrier_reached"], "COMM_WORLD barrier was not reached — ranks may have diverged"

    def test_pyabc_smc_mpi_diagnostics_mapping(self, tmp_path):
        helper = Path(__file__).parent / "mpi_integration_helper.py"
        output_file = tmp_path / "pyabc_smc_diag.json"
        result = subprocess.run(
            [
                "mpirun",
                "-n",
                "2",
                "--stdin",
                "none",
                sys.executable,
                str(helper),
                str(output_file),
                "concurrent_futures",
                "none",
                "0.5",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(output_file.read_text())
        assert data["n_records"] > 0
        assert data["method"] == "pyabc_smc"
        assert data["world_size"] == 2
        assert data["pyabc_mpi_sampler"] == "concurrent_futures"
        assert data["barrier_reached"]
        assert data["max_elapsed_s"] < 60.0
        assert data["elapsed_spread_s"] < 20.0

    def test_pyabc_smc_mpi_legacy_remains_selectable(self, tmp_path):
        helper = Path(__file__).parent / "mpi_integration_helper.py"
        output_file = tmp_path / "pyabc_smc_legacy_diag.json"
        result = subprocess.run(
            [
                "mpirun",
                "-n",
                "2",
                "--stdin",
                "none",
                sys.executable,
                str(helper),
                str(output_file),
                "concurrent_futures_legacy",
                "2",
                "0.5",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(output_file.read_text())
        assert data["n_records"] > 0
        assert data["method"] == "pyabc_smc"
        assert data["pyabc_mpi_sampler"] == "concurrent_futures_legacy"
        assert data["client_max_jobs"] == 2
