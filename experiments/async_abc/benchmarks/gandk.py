"""G-and-k distribution benchmark.

The g-and-k distribution is defined by its quantile function:

    Q(p; A, B, g, k) = A + B * (1 + c * tanh(g*z/2)) * (1 + z^2)^k * z

where z = Phi^{-1}(p) is the standard normal quantile and c = 0.8.

Parameters:
    A: location
    B: scale  (B > 0)
    g: skewness
    k: kurtosis (k > -0.5)

Summary statistics: 7 octile-spaced quantiles of the data at
p = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875].

Reference:
    Allingham, D., King, R.A.R., & Mengersen, K.L. (2009).
    Bayesian estimation of quantile distributions.
    Statistics and Computing, 19(2), 189-201.
"""
from typing import Dict, Tuple

import numpy as np
from scipy import stats


_QUANTILE_LEVELS = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
_C = 0.8


def _gandk_quantile(u: np.ndarray, A: float, B: float, g: float, k: float) -> np.ndarray:
    """Evaluate the g-and-k quantile function at probability points *u*."""
    u = np.clip(u, 1e-8, 1.0 - 1e-8)
    z = stats.norm.ppf(u)
    if g == 0.0:
        tanh_part = 0.0
    else:
        tanh_part = np.tanh(g * z / 2.0)
    return A + B * (1.0 + _C * tanh_part) * (1.0 + z ** 2) ** k * z


def _summary_stats(data: np.ndarray) -> np.ndarray:
    return np.array([np.quantile(data, q) for q in _QUANTILE_LEVELS])


class GandK:
    """ABC benchmark: infer parameters of the g-and-k distribution.

    Parameters
    ----------
    config:
        Benchmark sub-config dict.  Recognised keys:

        - ``observed_data_seed`` (int, default 42)
        - ``n_obs`` (int, default 1000)
        - ``true_A``, ``true_B``, ``true_g``, ``true_k`` (float defaults: 3, 1, 2, 0.5)
    """

    def __init__(self, config: dict) -> None:
        self.n_obs = config.get("n_obs", 1000)
        true_A = config.get("true_A", 3.0)
        true_B = config.get("true_B", 1.0)
        true_g = config.get("true_g", 2.0)
        true_k = config.get("true_k", 0.5)

        rng = np.random.default_rng(config.get("observed_data_seed", 42))
        u_obs = rng.uniform(0.0, 1.0, self.n_obs)
        self.observed_data = _gandk_quantile(u_obs, true_A, true_B, true_g, true_k)
        self.observed_stats = _summary_stats(self.observed_data)

        self.limits: Dict[str, Tuple[float, float]] = {
            "A": (0.0, 6.0),
            "B": (0.1, 4.0),
            "g": (0.0, 5.0),
            "k": (0.0, 1.0),
        }

    def simulate(self, params: dict, seed: int) -> float:
        """Simulate and return the ABC distance.

        Parameters
        ----------
        params:
            Dict with keys ``A``, ``B``, ``g``, ``k``.
        seed:
            Integer RNG seed.

        Returns
        -------
        float
            Euclidean distance between simulated and observed summary statistics.
        """
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0, self.n_obs)
        sim_data = _gandk_quantile(
            u, float(params["A"]), float(params["B"]),
            float(params["g"]), float(params["k"])
        )
        sim_stats = _summary_stats(sim_data)
        return float(np.linalg.norm(sim_stats - self.observed_stats))

    # ------------------------------------------------------------------
    # Reference posterior
    # ------------------------------------------------------------------
    # The ABC target here is p(theta | s_obs) with s_obs the seven octiles --
    # NOT the full-data posterior, because the octiles are not sufficient.
    # Scoring a recovered posterior against the full-data posterior would
    # measure the summaries' information loss rather than the sampler.
    #
    # There is no closed form for p(theta | s_obs), but there is an accurate
    # one: sample quantiles are asymptotically multivariate normal,
    #
    #     qhat ~ MVN( Q(p; theta),  Sigma(theta) / n ),
    #     Sigma_ij = [min(p_i, p_j) - p_i p_j] / [f(Q(p_i)) f(Q(p_j))],
    #
    # and for a quantile-defined law the density comes straight from the
    # quantile function: f(Q(u)) = phi(z) / (dQ/dz) with z = Phi^{-1}(u)
    # (Rayner & MacGillivray 2002). That gives an explicit likelihood for the
    # summaries, which MCMC turns into reference draws.
    #
    # Both halves are checked rather than asserted -- see
    # ``tests/test_gandk_reference.py``: the density against numerical
    # differentiation of the quantile function, and the covariance against the
    # octiles of 40,000 simulated datasets (standard deviations within 0.3%,
    # correlations within 0.01 at the n_obs = 1000 used here).

    def _dQ_dz(self, z: np.ndarray, A: float, B: float, g: float, k: float) -> np.ndarray:
        """d/dz of the quantile function, z = Phi^{-1}(u)."""
        T = 1.0 + _C * np.tanh(g * z / 2.0)
        T_prime = _C * (g / 2.0) / np.cosh(g * z / 2.0) ** 2
        R = (1.0 + z ** 2) ** k
        R_prime = k * (1.0 + z ** 2) ** (k - 1.0) * 2.0 * z
        return B * (T_prime * R * z + T * R_prime * z + T * R)

    def summary_mean_cov(self, params: Dict[str, float]):
        """Asymptotic mean and covariance of the octile summary vector."""
        A, B, g, k = (float(params[n]) for n in ("A", "B", "g", "k"))
        z = stats.norm.ppf(_QUANTILE_LEVELS)
        mean = _gandk_quantile(_QUANTILE_LEVELS, A, B, g, k)
        dens = stats.norm.pdf(z) / self._dQ_dz(z, A, B, g, k)
        pp = (np.minimum.outer(_QUANTILE_LEVELS, _QUANTILE_LEVELS)
              - np.outer(_QUANTILE_LEVELS, _QUANTILE_LEVELS))
        return mean, pp / np.outer(dens, dens) / self.n_obs

    def summary_log_likelihood(self, params: Dict[str, float]) -> float:
        """log p(s_obs | theta) under the asymptotic summary law."""
        if not all(lo <= float(params[n]) <= hi for n, (lo, hi) in self.limits.items()):
            return -np.inf
        try:
            mean, cov = self.summary_mean_cov(params)
        except (FloatingPointError, ValueError):
            return -np.inf
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            return -np.inf
        try:
            return float(stats.multivariate_normal.logpdf(
                self.observed_stats, mean=mean, cov=cov
            ))
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf

    def reference_posterior_samples(
        self, n: int, seed: int = 0, *, burn_in: int = 20_000, thin: int = 20
    ) -> np.ndarray:
        """Draws from p(theta | s_obs) by random-walk Metropolis.

        Returns an ``(n, 4)`` array in ``self.limits`` order. This is the
        reference the reported posterior is scored against; it is a *reference*
        rather than an analytic posterior, and the approximation it rests on is
        the asymptotic summary law validated in the tests.
        """
        rng = np.random.default_rng(seed)
        names = list(self.limits)
        lo = np.array([self.limits[nm][0] for nm in names])
        hi = np.array([self.limits[nm][1] for nm in names])

        theta = np.array([3.0, 1.0, 2.0, 0.5])
        theta = np.clip(theta, lo, hi)
        logp = self.summary_log_likelihood(dict(zip(names, theta)))
        # Scales are per-parameter fractions of the prior width; adapted during
        # burn-in towards a 0.234 acceptance rate.
        step = 0.02 * (hi - lo)

        draws = np.empty((n, len(names)))
        kept = accepted = 0
        i = 0
        while kept < n:
            prop = theta + step * rng.normal(size=len(names))
            lp = self.summary_log_likelihood(dict(zip(names, prop)))
            if np.log(rng.uniform()) < lp - logp:
                theta, logp = prop, lp
                accepted += 1
            i += 1
            if i <= burn_in:
                if i % 500 == 0:
                    rate = accepted / i
                    step *= np.exp((rate - 0.234) * 2.0)
            elif (i - burn_in) % thin == 0:
                draws[kept] = theta
                kept += 1
        return draws
