"""Simulation-based calibration helpers."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_rank(posterior_samples: np.ndarray, true_value: float) -> int:
    """Rank of the true value among posterior samples."""
    samples = np.sort(np.asarray(posterior_samples, dtype=float).ravel())
    return int(np.searchsorted(samples, float(true_value), side="left"))


def compute_rank_weighted(
    posterior_samples: np.ndarray,
    weights,
    true_value: float,
    *,
    seed=None,
) -> int:
    """Rank of the true value in a weighted-resampled posterior.

    Draws ``len(posterior_samples)`` samples with replacement using ``weights``
    as probabilities, then returns the rank of ``true_value`` in the resampled
    array.  Falls back to :func:`compute_rank` when weights are None or all zero.
    """
    samples = np.asarray(posterior_samples, dtype=float).ravel()
    if weights is None:
        return compute_rank(samples, true_value)
    w = np.asarray(weights, dtype=float).ravel()
    w_sum = w.sum()
    if w_sum <= 0.0:
        return compute_rank(samples, true_value)
    w = w / w_sum
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=len(samples), replace=True, p=w)
    resampled = np.sort(samples[idx])
    return int(np.searchsorted(resampled, float(true_value), side="left"))


def _resample_with_weights(samples: np.ndarray, weights, *, seed=None) -> np.ndarray:
    """Return equal-weight resample; if weights None/zero, return sorted copy."""
    if weights is None:
        return np.sort(samples)
    w = np.asarray(weights, dtype=float).ravel()
    w_sum = w.sum()
    if w_sum <= 0.0:
        return np.sort(samples)
    w = w / w_sum
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=len(samples), replace=True, p=w)
    return samples[idx]


def sbc_ranks(trials: list[dict]) -> pd.DataFrame:
    """Compute SBC ranks for trial records."""
    rows = []
    for idx, trial in enumerate(trials):
        samples = np.asarray(trial["posterior_samples"], dtype=float).ravel()
        weights = trial.get("posterior_weights")
        seed = trial.get("trial", idx)
        rank = (
            compute_rank_weighted(samples, weights, float(trial["true_value"]), seed=seed)
            if weights is not None
            else compute_rank(samples, float(trial["true_value"]))
        )
        rows.append(
            {
                "trial": int(trial.get("trial", idx)),
                "method": trial.get("method"),
                "benchmark": trial.get("benchmark"),
                "param": trial.get("param", "param"),
                "true_value": float(trial["true_value"]),
                "rank": rank,
                "n_samples": int(len(samples)),
            }
        )
    return pd.DataFrame(rows)


def empirical_coverage(
    trials: list[dict],
    coverage_levels: list[float],
) -> pd.DataFrame:
    """Estimate empirical equal-tailed coverage across SBC trials."""
    rows = []
    for level in coverage_levels:
        alpha = float(level)
        lower_q = (1.0 - alpha) / 2.0
        upper_q = 1.0 - lower_q
        for idx, trial in enumerate(trials):
            samples = np.asarray(trial["posterior_samples"], dtype=float).ravel()
            weights = trial.get("posterior_weights")
            seed = trial.get("trial", idx)
            if weights is not None:
                resampled = _resample_with_weights(samples, weights, seed=seed)
                lower = float(np.quantile(resampled, lower_q))
                upper = float(np.quantile(resampled, upper_q))
            else:
                lower = float(np.quantile(samples, lower_q))
                upper = float(np.quantile(samples, upper_q))
            rows.append(
                {
                    "trial": int(trial.get("trial", idx)),
                    "method": trial.get("method"),
                    "benchmark": trial.get("benchmark"),
                    "param": trial.get("param", "param"),
                    "coverage_level": alpha,
                    "covered": float(lower <= float(trial["true_value"]) <= upper),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["benchmark", "method", "param", "coverage_level", "empirical_coverage"])

    summary = (
        frame.groupby(["benchmark", "method", "param", "coverage_level"], dropna=False, sort=True)["covered"]
        .agg(empirical_coverage="mean", n_trials="count")
        .reset_index()
    )
    return summary


def _weighted_mean_sd(samples: np.ndarray, weights) -> tuple[float, float]:
    """Weighted (or unweighted) mean and unbiased-ish standard deviation.

    For weighted samples we use the standard reliability-weighted estimator:
    sd^2 = sum(w_i (x_i - mean)^2) / sum(w_i). This is the maximum-likelihood
    estimator under the assumption that weights are reliability weights; it is
    consistent under the AMIS CLT (Cornuet et al. 2012). For unweighted samples
    it is the population sd (ddof=0), matching np.std defaults.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if weights is None:
        return float(np.mean(samples)), float(np.std(samples))
    w = np.asarray(weights, dtype=float).ravel()
    w_sum = w.sum()
    if w_sum <= 0.0:
        return float(np.mean(samples)), float(np.std(samples))
    mean = float(np.average(samples, weights=w))
    var = float(np.average((samples - mean) ** 2, weights=w))
    return mean, float(np.sqrt(var))


def gaussian_credible_coverage(
    trials: list[dict],
    coverage_levels: list[float],
) -> pd.DataFrame:
    """Empirical coverage of asymptotic Gaussian credible intervals.

    For each trial computes the posterior mean ± z(level) · sd interval
    using the (possibly importance-weighted) posterior samples, where
    ``z(level) = norm.ppf((1 + level) / 2)``. Averages "covered" over
    trials.

    This is the empirical demonstration of the AMIS-derived CLT
    (paper §4, Theorem 2): if the asymptotic Gaussian approximation
    is valid at the run's bandwidth, the empirical coverage should
    converge to the nominal level. Equal-tailed quantile coverage
    (``empirical_coverage``) tests posterior shape; this test
    specifically targets the CLT prediction.
    """
    rows = []
    for level in coverage_levels:
        alpha = float(level)
        z = float(norm.ppf((1.0 + alpha) / 2.0))
        for idx, trial in enumerate(trials):
            samples = np.asarray(trial["posterior_samples"], dtype=float).ravel()
            weights = trial.get("posterior_weights")
            mean, sd = _weighted_mean_sd(samples, weights)
            lower = mean - z * sd
            upper = mean + z * sd
            rows.append(
                {
                    "trial": int(trial.get("trial", idx)),
                    "method": trial.get("method"),
                    "benchmark": trial.get("benchmark"),
                    "param": trial.get("param", "param"),
                    "coverage_level": alpha,
                    "covered": float(lower <= float(trial["true_value"]) <= upper),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["benchmark", "method", "param", "coverage_level", "gaussian_coverage", "n_trials"]
        )

    summary = (
        frame.groupby(["benchmark", "method", "param", "coverage_level"], dropna=False, sort=True)["covered"]
        .agg(gaussian_coverage="mean", n_trials="count")
        .reset_index()
    )
    return summary
