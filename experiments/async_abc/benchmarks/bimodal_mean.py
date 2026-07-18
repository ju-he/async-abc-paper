"""Bimodal (sign-unidentifiable) mean benchmark.

Model:
    y_i ~ N(theta^2, sigma_obs^2),  i = 1 ... n_obs
    theta ~ Uniform(prior_low, prior_high)

The summary statistic is the sample mean and the distance is
|mean(sim) - mean(obs)|. Because the map theta -> theta^2 folds the sign, +theta
and -theta produce identical likelihoods, so the posterior is **symmetric
bimodal** with modes at +/- sqrt(mean_obs). This is the standard test for
mode loss: a top-k archive that collapses onto one sign misses half the
posterior, which SBC detects as rank non-uniformity (external review concern 6).

Used for simulation-based calibration on a multimodal target; there is no single
"analytic posterior mean" (the two modes cancel), so quality is assessed by SBC
rank calibration and a mode-coverage diagnostic rather than a point estimate.
"""
from typing import Dict, Tuple

import numpy as np


class BimodalMean:
    """ABC benchmark with a symmetric bimodal posterior (sign-unidentifiable mean).

    Parameters
    ----------
    config:
        Benchmark sub-config dict. Recognised keys:

        - ``observed_data_seed`` (int, default 42)
        - ``n_obs`` (int, default 100)
        - ``true_theta`` (float, default 1.5) — ground-truth parameter (its sign
          is unidentifiable; the two posterior modes sit at +/- |true_theta|)
        - ``sigma_obs`` (float, default 1.0) — known observation noise std
        - ``prior_low`` / ``prior_high`` (float, defaults -3 / 3)
    """

    def __init__(self, config: dict) -> None:
        self.n_obs = config.get("n_obs", 100)
        self.sigma_obs = config.get("sigma_obs", 1.0)
        self.true_theta = config.get("true_theta", 1.5)
        self.prior_low = config.get("prior_low", -3.0)
        self.prior_high = config.get("prior_high", 3.0)

        rng = np.random.default_rng(config.get("observed_data_seed", 42))
        mean_signal = float(self.true_theta) ** 2
        self.observed_data = rng.normal(mean_signal, self.sigma_obs, self.n_obs)
        self.observed_mean = float(np.mean(self.observed_data))

        self.limits: Dict[str, Tuple[float, float]] = {
            "theta": (float(self.prior_low), float(self.prior_high))
        }

    def simulate(self, params: dict, seed: int) -> float:
        """Return the ABC distance |mean(sim) - mean(obs)|.

        The simulated data depend on ``theta`` only through ``theta**2``, so
        ``theta`` and ``-theta`` are statistically indistinguishable.
        """
        rng = np.random.default_rng(seed)
        theta = float(params["theta"])
        sim_data = rng.normal(theta ** 2, self.sigma_obs, self.n_obs)
        sim_mean = float(np.mean(sim_data))
        return abs(sim_mean - self.observed_mean)

    def posterior_modes(self) -> Tuple[float, float]:
        """The two symmetric posterior modes ``+/- sqrt(max(mean_obs, 0))``."""
        root = float(np.sqrt(max(self.observed_mean, 0.0)))
        return (-root, root)
