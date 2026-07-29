"""Multivariate Gaussian-mean benchmark (dimension-scaling study).

Model:
    y_i ~ N(theta, sigma_obs^2 I_d),  i = 1 ... n_obs,  theta in R^d
    theta_j ~ Uniform(prior_low, prior_high)  independently for j = 1 ... d

This is the ``gaussian_mean`` benchmark with the dimension made a free
parameter, so that the archive size ``k`` and the AMIS buffer size ``S`` can be
swept against ``dim`` on a target whose posterior is known exactly at every
dimension. The summary statistic is the per-coordinate sample mean and the
discrepancy is the Euclidean distance between simulated and observed summaries
(this reduces to the 1-D benchmark's ``|mean(sim) - mean(obs)|`` at ``dim=1``).

Why this benchmark exists
-------------------------
The proposal covariance is a *full* ``d x d`` weighted covariance estimated from
the ``k`` archive members, with only a numerical jitter -- no shrinkage and no
diagonal fallback. Estimating ``d(d+1)/2`` free parameters from an archive of
size ``k`` therefore imposes a floor on ``k`` that grows quadratically in ``d``,
while the importance weights' dynamic range (and hence the value of a richer
``S``-snapshot denominator) grows with ``d`` as well. Both effects are
measurable here against an exact reference.

Parameters are named ``mu1 ... mud`` so that SBC reports per-coordinate ranks.
"""
from typing import Dict, Tuple

import numpy as np


class GaussianMeanND:
    """ABC benchmark: infer the mean vector of an isotropic Gaussian.

    Parameters
    ----------
    config:
        Benchmark sub-config dict. Recognised keys:

        - ``observed_data_seed`` (int, default 42)
        - ``dim`` (int, default 1) — dimension ``d`` of the mean vector
        - ``n_obs`` (int, default 100)
        - ``true_mu`` (float or list, default 0.0) — ground truth; a scalar is
          broadcast to all ``d`` coordinates
        - ``sigma_obs`` (float, default 1.0) — known observation noise std
        - ``prior_low`` / ``prior_high`` (float, defaults -5 / 5)
    """

    def __init__(self, config: dict) -> None:
        self.dim = int(config.get("dim", 1))
        if self.dim < 1:
            raise ValueError(f"gaussian_mean_nd requires dim >= 1, got {self.dim}")
        self.n_obs = int(config.get("n_obs", 100))
        self.sigma_obs = float(config.get("sigma_obs", 1.0))
        self.prior_low = float(config.get("prior_low", -5.0))
        self.prior_high = float(config.get("prior_high", 5.0))

        self.param_names = [f"mu{j + 1}" for j in range(self.dim)]

        # Ground truth may arrive either as a single ``true_mu`` (scalar or
        # length-d list, for a fixed-truth run) or as per-coordinate
        # ``true_mu1 ... true_mud`` keys. SBC uses the latter: it redraws the
        # truth every trial and injects one ``true_<param>`` key per parameter
        # name (sbc_runner._true_param_config). Reading only ``true_mu`` would
        # silently ignore that injection and generate every trial's observed
        # data from the same default truth, making the SBC ranks meaningless.
        default = np.broadcast_to(
            np.asarray(config.get("true_mu", 0.0), dtype=float), (self.dim,)
        ).astype(float)
        self.true_mu = np.asarray(
            [float(config.get(f"true_{name}", default[j]))
             for j, name in enumerate(self.param_names)],
            dtype=float,
        )

        rng = np.random.default_rng(config.get("observed_data_seed", 42))
        self.observed_data = rng.normal(
            self.true_mu, self.sigma_obs, size=(self.n_obs, self.dim)
        )
        self.observed_mean = np.mean(self.observed_data, axis=0)

        self.limits: Dict[str, Tuple[float, float]] = {
            name: (self.prior_low, self.prior_high) for name in self.param_names
        }

    def _theta(self, params: dict) -> np.ndarray:
        return np.asarray([float(params[name]) for name in self.param_names], dtype=float)

    def simulate(self, params: dict, seed: int) -> float:
        """Simulate and return the Euclidean summary distance.

        At ``dim=1`` this is exactly the 1-D benchmark's absolute difference.
        """
        rng = np.random.default_rng(seed)
        theta = self._theta(params)
        sim_data = rng.normal(theta, self.sigma_obs, size=(self.n_obs, self.dim))
        sim_mean = np.mean(sim_data, axis=0)
        return float(np.linalg.norm(sim_mean - self.observed_mean))

    def analytic_posterior_mean(self) -> Dict[str, float]:
        """Per-coordinate posterior mean under the flat Uniform prior.

        The prior is flat over its box, so the posterior is proportional to the
        likelihood and its mean is the MLE (the observed per-coordinate mean)
        clipped to the prior bounds.
        """
        clipped = np.clip(self.observed_mean, self.prior_low, self.prior_high)
        return {name: float(value) for name, value in zip(self.param_names, clipped)}

    def analytic_posterior_samples(self, n: int, seed: int) -> np.ndarray:
        """Draw *n* samples from the exact posterior, shape ``(n, dim)``.

        Under the flat box prior the posterior factorises over coordinates into
        ``N(observed_mean_j, sigma_obs^2 / n_obs)`` truncated to the prior box.
        """
        from scipy.stats import truncnorm

        rng = np.random.default_rng(seed)
        scale = self.sigma_obs / np.sqrt(self.n_obs)
        out = np.empty((int(n), self.dim), dtype=float)
        for j in range(self.dim):
            loc = float(self.observed_mean[j])
            a = (self.prior_low - loc) / scale
            b = (self.prior_high - loc) / scale
            out[:, j] = truncnorm.rvs(
                a, b, loc=loc, scale=scale, size=int(n), random_state=rng
            )
        return out
