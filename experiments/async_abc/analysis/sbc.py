"""Simulation-based calibration helpers."""

import numpy as np
import pandas as pd


def compute_rank(posterior_samples: np.ndarray, true_value: float) -> int:
    """Rank of the true value among posterior samples."""
    samples = np.sort(np.asarray(posterior_samples, dtype=float).ravel())
    return int(np.searchsorted(samples, float(true_value), side="left"))


def sbc_ranks(trials: list[dict]) -> pd.DataFrame:
    """Compute SBC ranks for trial records."""
    rows = []
    for idx, trial in enumerate(trials):
        samples = np.asarray(trial["posterior_samples"], dtype=float).ravel()
        rows.append(
            {
                "trial": int(trial.get("trial", idx)),
                "method": trial.get("method"),
                "param": trial.get("param", "param"),
                "true_value": float(trial["true_value"]),
                "rank": compute_rank(samples, float(trial["true_value"])),
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
            lower = float(np.quantile(samples, lower_q))
            upper = float(np.quantile(samples, upper_q))
            rows.append(
                {
                    "trial": int(trial.get("trial", idx)),
                    "method": trial.get("method"),
                    "param": trial.get("param", "param"),
                    "coverage_level": alpha,
                    "covered": float(lower <= float(trial["true_value"]) <= upper),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["method", "param", "coverage_level", "empirical_coverage"])

    summary = (
        frame.groupby(["method", "param", "coverage_level"], dropna=False, sort=True)["covered"]
        .mean()
        .reset_index(name="empirical_coverage")
    )
    return summary
