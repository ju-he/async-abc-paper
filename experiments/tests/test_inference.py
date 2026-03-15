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
