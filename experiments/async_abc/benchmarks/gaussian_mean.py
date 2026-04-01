"""Gaussian mean benchmark.

Model:
    y_i ~ N(theta, sigma_obs^2),  i = 1 ... n_obs
    theta ~ Uniform(prior_low, prior_high)

Observed data are generated once from ``true_mu`` using ``observed_data_seed``.
The ABC summary statistic is the sample mean.  Distance is |mean(sim) - mean(obs)|.

Analytic posterior under the Uniform prior:
    The posterior mean is the MLE (observed_mean) clipped to the prior bounds,
    since the uniform prior is flat over its support.
"""
from typing import Dict, Tuple

import numpy as np


class GaussianMean:
    """ABC benchmark: infer the mean of a Gaussian.

    Parameters
    ----------
    config:
        Benchmark sub-config dict.  Recognised keys:

        - ``observed_data_seed`` (int, default 42)
        - ``n_obs`` (int, default 50)
        - ``true_mu`` (float, default 0.0) — ground-truth parameter
        - ``sigma_obs`` (float, default 1.0) — known observation noise std
        - ``prior_low`` / ``prior_high`` (float, defaults -5 / 5)
    """

    def __init__(self, config: dict) -> None:
        self.n_obs = config.get("n_obs", 50)
        self.sigma_obs = config.get("sigma_obs", 1.0)
        self.true_mu = config.get("true_mu", 0.0)
        self.prior_low = config.get("prior_low", -5.0)
        self.prior_high = config.get("prior_high", 5.0)

        rng = np.random.default_rng(config.get("observed_data_seed", 42))
        self.observed_data = rng.normal(self.true_mu, self.sigma_obs, self.n_obs)
        self.observed_mean = float(np.mean(self.observed_data))

        self.limits: Dict[str, Tuple[float, float]] = {
            "mu": (float(self.prior_low), float(self.prior_high))
        }

    def simulate(self, params: dict, seed: int) -> float:
        """Simulate and return the ABC distance.

        Parameters
        ----------
        params:
            Dict with key ``"mu"``.
        seed:
            Integer RNG seed for reproducibility.

        Returns
        -------
        float
            |mean(simulated) - mean(observed)|
        """
        rng = np.random.default_rng(seed)
        mu = float(params["mu"])
        sim_data = rng.normal(mu, self.sigma_obs, self.n_obs)
        sim_mean = float(np.mean(sim_data))
        return abs(sim_mean - self.observed_mean)

    def analytic_posterior_mean(self) -> float:
        """Posterior mean of theta under the Uniform(prior_low, prior_high) prior.

        For a flat prior the posterior is proportional to the likelihood, which
        is Gaussian in the sufficient statistic (sample mean).  The posterior
        mean is therefore the MLE (observed_mean) clipped to the prior bounds.

        Returns
        -------
        float
        """
        return float(np.clip(self.observed_mean, self.prior_low, self.prior_high))
