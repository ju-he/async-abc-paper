"""Review II.4.2 / II.3.b / II.9.7 — weighted and analytic quality metrics.

The paper attributes the param-bias result to the AMIS importance weights,
but the shipped metric was computed on the unweighted archive. These tests
pin the new weighted metric (which resamples each method's REPORTED
posterior by its own weights), the analytic-posterior Wasserstein for the
Gaussian-mean benchmark, and the determinism of the sliced-Wasserstein
distance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.analysis.convergence import _wasserstein_to_true_params
from async_abc.benchmarks.gaussian_mean import GaussianMean
from async_abc.io.records import ParticleRecord
from async_abc.reporting.runtime_summary import (
    _final_quality_wasserstein_analytic,
    _final_quality_wasserstein_weighted,
    _weighted_final_frame,
)


def _async_record(step, mu, posterior_weight, *, loss=0.1):
    return ParticleRecord(
        method="async_propulate_abc",
        replicate=0,
        seed=1,
        step=step,
        params={"mu": mu},
        loss=loss,
        weight=1.0,
        posterior_weight=posterior_weight,
        tolerance=1.0,
        wall_time=float(step),
        record_kind="simulation_attempt",
        time_semantics="event_end",
        attempt_count=step,
    )


def _sync_record(step, mu, weight, *, generation=0):
    return ParticleRecord(
        method="abc_smc_baseline",
        replicate=0,
        seed=1,
        step=step,
        params={"mu": mu},
        loss=0.1,
        weight=weight,
        tolerance=1.0,
        wall_time=float(step),
        generation=generation,
        record_kind="population_particle",
        time_semantics="generation_end",
        attempt_count=step,
    )


class TestWeightedFinalFrame:
    def test_async_uses_posterior_weight(self):
        """Nearly all posterior mass on mu=2.0 → the resample concentrates there."""
        records = [
            _async_record(1, 0.0, posterior_weight=1e-9),
            _async_record(2, 2.0, posterior_weight=1.0),
            _async_record(3, -3.0, posterior_weight=1e-9),
        ]
        frame = _weighted_final_frame(records, param_names=["mu"], archive_size=None)
        assert frame is not None
        assert (frame["mu"] == 2.0).mean() > 0.99

    def test_sync_uses_population_weight(self):
        records = [
            _sync_record(1, 0.0, weight=1e-9),
            _sync_record(2, 1.5, weight=1.0),
        ]
        frame = _weighted_final_frame(records, param_names=["mu"], archive_size=None)
        assert frame is not None
        assert (frame["mu"] == 1.5).mean() > 0.99

    def test_async_without_posterior_weight_falls_back_to_final_state(self):
        records = [
            _async_record(1, 0.5, posterior_weight=None),
            _async_record(2, 0.5, posterior_weight=None),
        ]
        frame = _weighted_final_frame(records, param_names=["mu"], archive_size=None)
        assert frame is not None
        assert set(frame["mu"]) == {0.5}

    def test_deterministic_resample(self):
        records = [
            _async_record(i, float(i) / 10.0, posterior_weight=1.0 / (i + 1))
            for i in range(1, 8)
        ]
        a = _weighted_final_frame(records, param_names=["mu"], archive_size=None)
        b = _weighted_final_frame(records, param_names=["mu"], archive_size=None)
        pd.testing.assert_frame_equal(a, b)

    def test_weighted_metric_reflects_weighting(self):
        """Weighted W-to-truth (truth mu=0) sees the reported posterior, not
        the raw particle set: mass on mu=2 → distance ≈ 2."""
        records = [
            _async_record(1, 0.0, posterior_weight=1e-9),
            _async_record(2, 2.0, posterior_weight=1.0),
        ]
        w = _final_quality_wasserstein_weighted(
            records, true_params={"mu": 0.0}, archive_size=None
        )
        assert w == pytest.approx(2.0, abs=0.05)


class TestAnalyticPosteriorMetric:
    def _cfg(self):
        return {
            "benchmark": {
                "name": "gaussian_mean",
                "observed_data_seed": 42,
                "n_obs": 100,
                "true_mu": 0.0,
                "sigma_obs": 1.0,
                "prior_low": -5.0,
                "prior_high": 5.0,
            },
        }

    def test_analytic_samples_shape_and_support(self):
        bm = GaussianMean(self._cfg()["benchmark"])
        samples = bm.analytic_posterior_samples(5000, seed=0)
        assert samples.shape == (5000,)
        assert samples.min() >= bm.prior_low
        assert samples.max() <= bm.prior_high
        assert float(np.mean(samples)) == pytest.approx(
            bm.analytic_posterior_mean(), abs=0.02
        )

    def test_metric_near_zero_for_exact_posterior(self):
        """Records drawn FROM the analytic posterior with uniform weights →
        the analytic-W metric is ~0 while the point-mass metric floors at the
        posterior spread (the review's II.3.a distinction, verified)."""
        cfg = self._cfg()
        bm = GaussianMean(cfg["benchmark"])
        draws = bm.analytic_posterior_samples(400, seed=7)
        records = [
            _async_record(i + 1, float(mu), posterior_weight=1.0)
            for i, mu in enumerate(draws)
        ]
        w_analytic = _final_quality_wasserstein_analytic(
            records, cfg=cfg, archive_size=None
        )
        assert w_analytic < 0.02
        w_point = _final_quality_wasserstein_weighted(
            records, true_params={"mu": 0.0}, archive_size=None
        )
        assert w_point > 2.0 * w_analytic  # point-mass metric floors at the spread

    def test_nan_for_other_benchmarks(self):
        records = [_async_record(1, 0.0, posterior_weight=1.0)]
        w = _final_quality_wasserstein_analytic(
            records, cfg={"benchmark": {"name": "lotka_volterra"}}, archive_size=None
        )
        assert np.isnan(w)


class TestSlicedWassersteinDeterminism:
    def test_multid_repeatable(self):
        pytest.importorskip("ot")
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            rng.normal(size=(200, 2)), columns=["theta1", "theta2"]
        )
        truth = {"theta1": 0.0, "theta2": 0.0}
        a = _wasserstein_to_true_params(frame, truth, n_projections=50)
        b = _wasserstein_to_true_params(frame, truth, n_projections=50)
        assert a == b
