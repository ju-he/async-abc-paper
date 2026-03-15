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
