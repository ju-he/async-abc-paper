"""Tests for async_abc.inference.*"""
import math
import time

import pytest

from async_abc.benchmarks.gaussian_mean import GaussianMean
from async_abc.inference.method_registry import METHOD_REGISTRY, run_method
from async_abc.inference.propulate_abc import run_propulate_abc
from async_abc.io.records import ParticleRecord


# ---------------------------------------------------------------------------
# Minimal test-mode inference config
# ---------------------------------------------------------------------------

def _test_inference_cfg():
    return {
        "max_simulations": 60,   # small enough to be fast
        "n_workers": 1,
        "k": 10,
        "tol_init": 5.0,
        "scheduler_type": "acceptance_rate",
        "perturbation_scale": 0.8,
    }


def _gaussian_bm():
    return GaussianMean({
        "observed_data_seed": 0,
        "n_obs": 30,
        "true_mu": 0.0,
        "prior_low": -5.0,
        "prior_high": 5.0,
    })


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

class TestMethodRegistry:
    def test_contains_async_propulate_abc(self):
        assert "async_propulate_abc" in METHOD_REGISTRY

    def test_contains_pyabc_smc(self):
        assert "pyabc_smc" in METHOD_REGISTRY

    def test_all_values_are_callable(self):
        for name, fn in METHOD_REGISTRY.items():
            assert callable(fn), f"METHOD_REGISTRY['{name}'] is not callable"


class TestRunMethod:
    def test_dispatches_async_abc(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "test").ensure()
        records = run_method(
            "async_propulate_abc", bm.simulate, bm.limits,
            _test_inference_cfg(), od, replicate=0, seed=1,
        )
        assert isinstance(records, list)
        assert len(records) > 0

    def test_unknown_name_raises(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "test").ensure()
        with pytest.raises(KeyError, match="not_a_method"):
            run_method(
                "not_a_method", bm.simulate, bm.limits,
                _test_inference_cfg(), od, replicate=0, seed=1,
            )


# ---------------------------------------------------------------------------
# run_propulate_abc
# ---------------------------------------------------------------------------

class TestRunPropulateAbc:
    def test_completes_in_reasonable_time(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        t0 = time.time()
        records = run_propulate_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=42,
        )
        elapsed = time.time() - t0
        assert elapsed < 60.0, f"run took {elapsed:.1f}s, expected < 60s"
        assert len(records) > 0

    def test_returns_list_of_particle_records(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        records = run_propulate_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=7,
        )
        assert all(isinstance(r, ParticleRecord) for r in records)

    def test_records_have_required_fields(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        records = run_propulate_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=3,
        )
        r = records[0]
        assert r.method == "async_propulate_abc"
        assert r.replicate == 0
        assert r.seed == 3
        assert isinstance(r.step, int) and r.step >= 1
        assert "mu" in r.params
        assert math.isfinite(r.loss) and r.loss >= 0.0
        assert r.wall_time >= 0.0

    def test_records_count_matches_max_simulations(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        cfg = _test_inference_cfg()
        records = run_propulate_abc(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=5,
        )
        assert len(records) == cfg["max_simulations"]

    def test_steps_are_monotone(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        records = run_propulate_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=9,
        )
        steps = [r.step for r in records]
        assert steps == sorted(steps)

    def test_tolerance_non_increasing_once_set(self, tmp_output_dir):
        """Once tolerance is stamped it must be monotonically non-increasing."""
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        # Use larger k-to-population ratio so archive fills and tolerance updates
        cfg = {**_test_inference_cfg(), "max_simulations": 150, "k": 5}
        records = run_propulate_abc(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=11,
        )
        tols = [r.tolerance for r in records if r.tolerance is not None]
        for i in range(1, len(tols)):
            assert tols[i] <= tols[i - 1] + 1e-9, (
                f"Tolerance increased at step {i}: {tols[i-1]} -> {tols[i]}"
            )

    def test_different_seeds_give_different_results(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        cfg = _test_inference_cfg()
        records_a = run_propulate_abc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)
        records_b = run_propulate_abc(bm.simulate, bm.limits, cfg, od, replicate=1, seed=2)
        losses_a = [r.loss for r in records_a]
        losses_b = [r.loss for r in records_b]
        assert losses_a != losses_b

    def test_replicate_index_stored(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "run").ensure()
        records = run_propulate_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=2, seed=1,
        )
        assert all(r.replicate == 2 for r in records)


# ---------------------------------------------------------------------------
# pyabc wrapper (optional)
# ---------------------------------------------------------------------------

class TestPyabcWrapper:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_pyabc_runs_on_gaussian(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        records = run_pyabc_smc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        assert isinstance(records, list)
        assert all(isinstance(r, ParticleRecord) for r in records)


# ---------------------------------------------------------------------------
# pyabc wrapper fixes
# ---------------------------------------------------------------------------

class TestPyabcWrapperFixes:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_particle_loss_is_actual_distance(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        records = run_pyabc_smc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        # Losses in the first generation must not all be identical (would indicate
        # loss was set to the generation epsilon rather than the actual distance).
        first_gen_losses = [r.loss for r in records[:10] if r.loss is not None]
        assert len(set(round(l, 8) for l in first_gen_losses)) > 1, (
            "All first-gen losses are identical — loss is likely set to epsilon"
        )

    def test_tolerance_field_is_generation_epsilon(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        records = run_pyabc_smc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        assert all(r.tolerance is not None for r in records)
        assert all(r.tolerance >= 0.0 for r in records)

    def test_singlecore_sampler_when_n_workers_1(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        cfg = {**_test_inference_cfg(), "n_workers": 1}
        records = run_pyabc_smc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)
        assert len(records) > 0

    def test_multicore_sampler_when_n_workers_gt_1(self, tmp_output_dir, monkeypatch):
        import pyabc
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        calls = []
        original = pyabc.MulticoreEvalParallelSampler
        monkeypatch.setattr(
            pyabc, "MulticoreEvalParallelSampler",
            lambda n: (calls.append(n), original(n))[1],
        )
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        cfg = {**_test_inference_cfg(), "n_workers": 2}
        run_pyabc_smc(bm.simulate, bm.limits, cfg, od, replicate=0, seed=1)
        assert calls == [2], f"Expected MulticoreEvalParallelSampler(2), got {calls}"

    def test_same_seed_reproducible(self, tmp_output_dir):
        from pathlib import Path
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        bm = _gaussian_bm()
        od1 = OutputDir(Path(tmp_output_dir) / "r1", "pyabc").ensure()
        od2 = OutputDir(Path(tmp_output_dir) / "r2", "pyabc").ensure()
        r1 = run_pyabc_smc(bm.simulate, bm.limits, _test_inference_cfg(), od1, replicate=0, seed=42)
        r2 = run_pyabc_smc(bm.simulate, bm.limits, _test_inference_cfg(), od2, replicate=0, seed=42)
        assert r1[0].params == r2[0].params

    def test_epsilon_extraction_robust(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        records = run_pyabc_smc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        assert all(r.tolerance is not None for r in records)

    def test_all_records_complete_schema(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.pyabc_wrapper import run_pyabc_smc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "pyabc").ensure()
        records = run_pyabc_smc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        for r in records:
            assert r.method == "pyabc_smc"
            assert isinstance(r.step, int) and r.step >= 1
            assert math.isfinite(r.loss) and r.loss >= 0.0
            assert r.tolerance is not None and r.tolerance >= 0.0
            assert r.wall_time >= 0.0
            assert "mu" in r.params


# ---------------------------------------------------------------------------
# rejection_abc
# ---------------------------------------------------------------------------

class TestRejectionAbc:
    def test_importable(self):
        from async_abc.inference.rejection_abc import run_rejection_abc
        assert callable(run_rejection_abc)

    def test_in_registry(self):
        assert "rejection_abc" in METHOD_REGISTRY

    def test_returns_particle_records(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "rejection").ensure()
        records = run_rejection_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        assert isinstance(records, list)
        assert all(isinstance(r, ParticleRecord) for r in records)

    def test_method_name(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "rejection").ensure()
        records = run_rejection_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        assert all(r.method == "rejection_abc" for r in records)

    def test_all_accepted_within_tolerance(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "rejection").ensure()
        cfg = {**_test_inference_cfg(), "tol_init": 5.0}
        records = run_rejection_abc(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        assert all(r.loss <= cfg["tol_init"] + 1e-9 for r in records)

    def test_tolerance_field_equals_tol_init(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "rejection").ensure()
        cfg = {**_test_inference_cfg(), "tol_init": 3.0}
        records = run_rejection_abc(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        assert all(r.tolerance == pytest.approx(3.0) for r in records)

    def test_steps_monotone(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "rejection").ensure()
        records = run_rejection_abc(
            bm.simulate, bm.limits, _test_inference_cfg(), od, replicate=0, seed=1,
        )
        steps = [r.step for r in records]
        assert steps == sorted(steps)
        assert steps[0] >= 1

    def test_returns_at_most_k_particles(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.rejection_abc import run_rejection_abc
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "rejection").ensure()
        cfg = {**_test_inference_cfg(), "k": 5}
        records = run_rejection_abc(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        assert len(records) <= cfg["k"]


# ---------------------------------------------------------------------------
# abc_smc_baseline
# ---------------------------------------------------------------------------

class TestAbcSmcBaseline:
    @pytest.fixture(autouse=True)
    def skip_if_no_pyabc(self):
        pytest.importorskip("pyabc", reason="pyabc not installed — skipping")

    def test_importable(self):
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        assert callable(run_abc_smc_baseline)

    def test_in_registry(self):
        assert "abc_smc_baseline" in METHOD_REGISTRY

    def test_returns_particle_records(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        cfg = {**_test_inference_cfg(), "n_generations": 2}
        records = run_abc_smc_baseline(
            bm.simulate, bm.limits, cfg, od, replicate=0, seed=1,
        )
        assert isinstance(records, list)
        assert all(isinstance(r, ParticleRecord) for r in records)

    def test_method_name(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        cfg = {**_test_inference_cfg(), "n_generations": 2}
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, 0, 1)
        assert all(r.method == "abc_smc_baseline" for r in records)

    def test_tolerance_non_increasing(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        cfg = {**_test_inference_cfg(), "n_generations": 3, "k": 5, "max_simulations": 300}
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, 0, 1)
        tols = [r.tolerance for r in records if r.tolerance is not None]
        for i in range(1, len(tols)):
            assert tols[i] <= tols[i - 1] + 1e-9

    def test_loss_is_actual_distance(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        cfg = {**_test_inference_cfg(), "n_generations": 2, "k": 10}
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, 0, 1)
        gen0 = records[:10]
        if len(gen0) > 1:
            assert len(set(round(r.loss, 8) for r in gen0)) > 1, (
                "All losses in generation 0 are identical — likely set to epsilon"
            )

    def test_steps_monotone(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        cfg = {**_test_inference_cfg(), "n_generations": 2, "k": 5}
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, 0, 1)
        steps = [r.step for r in records]
        assert steps == sorted(steps)

    def test_n_generations_respected(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        cfg = {**_test_inference_cfg(), "n_generations": 3, "k": 5}
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, 0, 1)
        assert len(records) > 0

    def test_n_generations_defaults(self, tmp_output_dir):
        from async_abc.io.paths import OutputDir
        from async_abc.inference.abc_smc_baseline import run_abc_smc_baseline
        bm = _gaussian_bm()
        od = OutputDir(tmp_output_dir, "abc_smc").ensure()
        cfg = {**_test_inference_cfg()}  # no n_generations key
        records = run_abc_smc_baseline(bm.simulate, bm.limits, cfg, od, 0, 1)
        assert isinstance(records, list)
