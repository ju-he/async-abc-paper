"""Posterior-quality summary for the sensitivity grid experiment.

``compute_sensitivity_quality_summary`` reads the per-variant CSVs produced by
``sensitivity_runner.py``, computes the Wasserstein distance between the final
posterior samples and the true parameters for each replicate, and returns a
tidy DataFrame with one row per grid variant.

Key design decisions
---------------------
- **Budget-keyed tail window**: the "final" samples are identified by
  ``step >= max_simulations * (1 - tail_fraction)``, not by the last N rows.
  This ensures variants with fewer accepted particles (e.g. tight ``tol_init``)
  are treated consistently with variants that accepted many particles.
- **Per-replicate aggregation**: Wasserstein is computed separately for each
  replicate, then mean ± std are reported.  This preserves cross-replicate
  variance and exposes it in the heatmap.
- **No absolute tolerance**: the metric is posterior quality, not raw ε.
  Variants differing only in ``tol_init_multiplier`` are fully comparable.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def true_params_from_benchmark_cfg(benchmark_cfg: dict) -> dict[str, float]:
    """Extract true parameter values from a benchmark sub-config dict.

    Looks for keys of the form ``"true_{name}"`` and returns
    ``{name: value}`` for each one found.

    Example
    -------
    >>> true_params_from_benchmark_cfg({"name": "gaussian_mean", "true_mu": 0.0})
    {'mu': 0.0}
    """
    return {
        key[len("true_"):]: float(val)
        for key, val in benchmark_cfg.items()
        if key.startswith("true_") and isinstance(val, (int, float, str))
        and _is_numeric(val)
    }


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def compute_sensitivity_quality_summary(
    data_dir: Path,
    grid: dict[str, list],
    true_params: dict[str, float],
    max_simulations: int,
    tail_fraction: float = 0.1,
) -> pd.DataFrame:
    """Return a DataFrame with posterior quality per grid variant.

    Parameters
    ----------
    data_dir:
        Directory containing ``sensitivity_*.csv`` files.
    grid:
        The sensitivity grid dict ``{param: [values]}``, same as in the config.
    true_params:
        Mapping from parameter name to its true value, e.g. ``{"mu": 0.0}``.
    max_simulations:
        Total simulation budget used for each variant run.  Determines the
        tail window: rows with ``step >= max_simulations * (1 - tail_fraction)``
        are treated as the final posterior sample.
    tail_fraction:
        Fraction of the simulation budget that counts as "final".  Default 0.1.

    Returns
    -------
    pd.DataFrame
        One row per grid variant with columns:

        - one column per grid key (parsed from the filename)
        - ``wasserstein_mean`` – mean Wasserstein distance across replicates
        - ``wasserstein_std``  – std of Wasserstein across replicates (NaN if
          only one replicate)
        - ``n_replicates``     – number of replicates found
    """
    import itertools

    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    rows_out: list[dict[str, Any]] = []

    for combo in itertools.product(*value_lists):
        variant: dict[str, Any] = dict(zip(keys, combo))
        variant_name = _variant_name(variant)
        csv_path = data_dir / f"sensitivity_{variant_name}.csv"

        row: dict[str, Any] = {k: v for k, v in variant.items()}
        if not csv_path.exists():
            row["wasserstein_mean"] = float("nan")
            row["wasserstein_std"] = float("nan")
            row["n_replicates"] = 0
            rows_out.append(row)
            continue

        wass_per_rep = _wasserstein_per_replicate(
            csv_path, true_params, max_simulations, tail_fraction
        )
        if not wass_per_rep:
            row["wasserstein_mean"] = float("nan")
            row["wasserstein_std"] = float("nan")
            row["n_replicates"] = 0
        else:
            arr = np.array(wass_per_rep, dtype=float)
            finite = arr[np.isfinite(arr)]
            row["wasserstein_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            row["wasserstein_std"] = float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
            row["n_replicates"] = len(finite)
        rows_out.append(row)

    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _variant_name(variant: dict[str, Any]) -> str:
    """Canonical variant filename stem (matches sensitivity_runner.py)."""
    parts = [f"{k}={v}" for k, v in sorted(variant.items())]
    return "__".join(parts)


def _wasserstein_per_replicate(
    csv_path: Path,
    true_params: dict[str, float],
    max_simulations: int,
    tail_fraction: float,
) -> list[float]:
    """Load CSV, split by replicate, compute Wasserstein for each."""
    frame = _read_variant_csv(csv_path)
    if frame.empty:
        return []

    param_names = list(true_params.keys())
    # Map param_X → X if needed
    for name in param_names:
        prefixed = f"param_{name}"
        if name not in frame.columns and prefixed in frame.columns:
            frame[name] = pd.to_numeric(frame[prefixed], errors="coerce")
        elif name in frame.columns:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    missing = [name for name in param_names if name not in frame.columns]
    if missing:
        return []

    frame["step"] = pd.to_numeric(frame["step"], errors="coerce").fillna(0).astype(int)
    threshold = int(max_simulations * (1.0 - tail_fraction))

    results: list[float] = []
    for _rep, group in frame.groupby("replicate", sort=True):
        # Clamp to [threshold, max_simulations] — exclude rows beyond the budget
        in_budget = group[group["step"] <= max_simulations]
        tail = in_budget[in_budget["step"] >= threshold]
        if tail.empty:
            # Fall back to all in-budget rows if tail window is empty
            tail = in_budget if not in_budget.empty else group
        samples = tail[param_names].dropna().to_numpy(dtype=float)
        if len(samples) == 0:
            continue
        results.append(_wasserstein(samples, true_params, param_names))

    return results


def _wasserstein(
    samples: np.ndarray,
    true_params: dict[str, float],
    param_names: list[str],
) -> float:
    """Wasserstein distance between samples and a point mass at true_params."""
    target = np.tile(
        np.array([true_params[n] for n in param_names], dtype=float),
        (len(samples), 1),
    )
    if samples.shape[1] == 1:
        return float(wasserstein_distance(samples[:, 0], target[:, 0]))
    try:
        import ot
        return float(ot.sliced_wasserstein_distance(samples, target, n_projections=50))
    except ImportError:
        return float(np.mean([
            wasserstein_distance(samples[:, i], target[:, i])
            for i in range(samples.shape[1])
        ]))


def _read_variant_csv(path: Path) -> pd.DataFrame:
    """Read a sensitivity variant CSV into a DataFrame."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
