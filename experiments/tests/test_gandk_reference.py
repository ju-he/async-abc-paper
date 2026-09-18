"""The g-and-k reference posterior rests on an approximation; check it.

``GandK.reference_posterior_samples`` targets p(theta | s_obs) -- the posterior
given the seven octile summaries the ABC discrepancy is built from, which is the
ABC algorithm's actual target. It is not analytic. It uses the asymptotic
multivariate-normal law of sample quantiles, so the figures that score a
recovered posterior against it are only as good as that law is at n_obs = 1000.
These tests are what make that claim checkable rather than asserted.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from async_abc.benchmarks.gandk import (  # noqa: E402
    GandK,
    _QUANTILE_LEVELS,
    _gandk_quantile,
    _summary_stats,
)

CFG = {"name": "gandk", "observed_data_seed": 42, "n_obs": 1000,
       "true_A": 3.0, "true_B": 1.0, "true_g": 2.0, "true_k": 0.5}
TRUTH = {"A": 3.0, "B": 1.0, "g": 2.0, "k": 0.5}


@pytest.fixture(scope="module")
def bm():
    return GandK(CFG)


class TestSummaryLaw:
    def test_density_matches_numerical_differentiation(self, bm):
        """f(Q(u)) = phi(z)/(dQ/dz) -- check dQ/dz against a finite difference."""
        u = np.array([0.125, 0.25, 0.5, 0.75, 0.875])
        h = 1e-6
        numeric = (_gandk_quantile(u + h, 3.0, 1.0, 2.0, 0.5)
                   - _gandk_quantile(u - h, 3.0, 1.0, 2.0, 0.5)) / (2 * h)
        z = stats.norm.ppf(u)
        analytic = bm._dQ_dz(z, 3.0, 1.0, 2.0, 0.5) / stats.norm.pdf(z)
        assert np.allclose(numeric, analytic, rtol=1e-6)

    def test_asymptotic_moments_match_simulated_octiles(self, bm):
        """The whole reference rests on this. Simulate datasets and compare."""
        rng = np.random.default_rng(0)
        reps = 4000
        sims = np.array([
            _summary_stats(_gandk_quantile(rng.uniform(0, 1, CFG["n_obs"]),
                                           3.0, 1.0, 2.0, 0.5))
            for _ in range(reps)
        ])
        mean, cov = bm.summary_mean_cov(TRUTH)
        emp_mean, emp_cov = sims.mean(0), np.cov(sims.T)

        # Means: within Monte Carlo error of the simulated mean.
        mc_se = np.sqrt(np.diag(emp_cov) / reps)
        assert np.all(np.abs(emp_mean - mean) < 4 * mc_se)
        # Spread: within 5% (measured ~0.3% at reps=40000).
        assert np.allclose(np.sqrt(np.diag(cov)), np.sqrt(np.diag(emp_cov)), rtol=0.05)
        # Correlation structure, which is what makes this multivariate.
        def corr(c):
            s = np.sqrt(np.diag(c))
            return c / np.outer(s, s)
        assert np.max(np.abs(corr(cov) - corr(emp_cov))) < 0.05

    def test_log_likelihood_is_maximised_near_the_truth(self, bm):
        base = bm.summary_log_likelihood(TRUTH)
        for name, offset in (("A", 0.5), ("B", 0.5), ("g", 1.0), ("k", 0.3)):
            for sign in (+1, -1):
                off = dict(TRUTH)
                off[name] = TRUTH[name] + sign * offset
                lo, hi = bm.limits[name]
                if not (lo <= off[name] <= hi):
                    continue
                assert bm.summary_log_likelihood(off) < base

    def test_log_likelihood_rejects_outside_the_prior_box(self, bm):
        off = dict(TRUTH)
        off["A"] = bm.limits["A"][1] + 1.0
        assert bm.summary_log_likelihood(off) == -np.inf


class TestReferencePosterior:
    @pytest.fixture(scope="class")
    def chains(self, bm):
        return (bm.reference_posterior_samples(4000, seed=0, burn_in=8000, thin=8),
                bm.reference_posterior_samples(4000, seed=1, burn_in=8000, thin=8))

    def test_independent_chains_agree(self, chains):
        """Convergence check: two chains from different seeds must land on the
        same distribution, or the reference is not a reference."""
        a, b = chains
        separation = np.abs(a.mean(0) - b.mean(0)) / a.std(0)
        assert separation.max() < 0.25, f"chains disagree: {separation}"

    def test_shape_and_support(self, bm, chains):
        a, _ = chains
        assert a.shape == (4000, 4)
        for j, name in enumerate(bm.limits):
            lo, hi = bm.limits[name]
            assert a[:, j].min() >= lo and a[:, j].max() <= hi

    def test_covers_the_true_parameters(self, bm, chains):
        a, _ = chains
        for j, name in enumerate(bm.limits):
            lo, hi = np.quantile(a[:, j], [0.01, 0.99])
            assert lo <= TRUTH[name] <= hi, f"{name} outside the 98% interval"

    def test_is_much_tighter_than_the_prior(self, bm, chains):
        """A reference that is barely narrower than the prior would make any
        posterior look good."""
        a, _ = chains
        for j, name in enumerate(bm.limits):
            lo, hi = bm.limits[name]
            prior_sd = (hi - lo) / np.sqrt(12)
            assert a[:, j].std() < 0.35 * prior_sd
