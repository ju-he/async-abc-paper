"""Benchmark-specific summary artifacts."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..analysis import final_state_results
from ..io.paths import OutputDir
from ..io.records import ParticleRecord


def write_gaussian_analytic_summary(
    records: List[ParticleRecord],
    *,
    cfg: Dict[str, Any],
    output_dir: OutputDir,
    archive_size: int | None = None,
) -> None:
    """Write Gaussian posterior mean error summaries when an analytic target exists."""
    benchmark_cfg = cfg.get("benchmark", {})
    if benchmark_cfg.get("name") != "gaussian_mean":
        return

    try:
        from ..benchmarks.gaussian_mean import GaussianMean
    except Exception:
        return

    analytic_mean = float(GaussianMean(benchmark_cfg).analytic_posterior_mean())
    rows: list[dict[str, object]] = []
    for result in final_state_results(records, archive_size=archive_size):
        sample_values = [float(record.params["mu"]) for record in result.records if "mu" in record.params]
        if not sample_values:
            continue
        posterior_mean = float(np.mean(np.asarray(sample_values, dtype=float)))
        rows.append(
            {
                "method": str(result.method),
                "replicate": int(result.replicate),
                "posterior_mean": posterior_mean,
                "analytic_posterior_mean": analytic_mean,
                "analytic_posterior_mean_abs_error": abs(posterior_mean - analytic_mean),
                "n_particles_used": int(result.n_particles_used),
            }
        )
    if not rows:
        return

    pd.DataFrame(rows).sort_values(["method", "replicate"]).to_csv(
        output_dir.data / "gaussian_analytic_summary.csv",
        index=False,
    )
    summary = {
        "analytic_posterior_mean": analytic_mean,
        "mean_abs_error": float(np.mean([float(row["analytic_posterior_mean_abs_error"]) for row in rows])),
        "max_abs_error": float(np.max([float(row["analytic_posterior_mean_abs_error"]) for row in rows])),
        "n_rows": len(rows),
    }
    with open(output_dir.data / "gaussian_analytic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _posterior_weight_of(record) -> float:
    """Retroactive AMIS posterior weight if present, else streaming weight, else 1.0.

    Mirrors ``sbc_runner._posterior_weight_of`` so the weighted posterior here is
    the same estimator SBC and the corner plots report (review concern 5 / 6).
    """
    pw = getattr(record, "posterior_weight", None)
    if pw is not None:
        return float(pw)
    if record.weight is not None:
        return float(record.weight)
    return 1.0


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: List[float]
) -> List[float]:
    """Interpolated weighted quantiles (weights need not be normalised)."""
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    total = float(w.sum())
    if total <= 0.0:
        return [float(np.quantile(v, q)) for q in quantiles]
    # Cumulative weight at the midpoint of each atom (Hazen-style plotting position).
    cdf = (np.cumsum(w) - 0.5 * w) / total
    return [float(np.interp(q, cdf, v)) for q in quantiles]


def write_gaussian_weighted_posterior_summary(
    records: List[ParticleRecord],
    *,
    cfg: Dict[str, Any],
    output_dir: OutputDir,
    archive_size: int | None = None,
) -> None:
    """Weighted-posterior quality metric for the coupling study (review concern 5, T2.1b).

    ``write_gaussian_analytic_summary`` reports the *unweighted* top-k archive mean,
    which the Tier-1 ablation found to be AMIS-insensitive under runtime–parameter
    coupling. This metric instead reweights the archive by the retroactive AMIS
    ``posterior_weight`` (the estimator SBC and the corner plots report) and records,
    per (method, replicate): weighted posterior mean / variance / quantiles (5/50/95),
    the analytic mean/std, weighted-mean absolute error, a 90%-CI coverage proxy for
    the analytic mean, and weight-health diagnostics (ESS, ESS fraction, max normalised
    weight). Aggregating across replicates per (method, sigma) gives the AMIS vs no-AMIS
    comparison the reviewer asked for (bias in mean, variance, quantiles, coverage).
    """
    benchmark_cfg = cfg.get("benchmark", {})
    if benchmark_cfg.get("name") != "gaussian_mean":
        return

    try:
        from ..benchmarks.gaussian_mean import GaussianMean
    except Exception:
        return

    bm = GaussianMean(benchmark_cfg)
    analytic_mean = float(bm.analytic_posterior_mean())
    analytic_std = float(bm.sigma_obs) / float(np.sqrt(bm.n_obs))

    rows: list[dict[str, object]] = []
    for result in final_state_results(records, archive_size=archive_size):
        pairs = [
            (float(r.params["mu"]), _posterior_weight_of(r))
            for r in result.records
            if "mu" in r.params
        ]
        if not pairs:
            continue
        samples = np.asarray([p[0] for p in pairs], dtype=float)
        weights = np.asarray([p[1] for p in pairs], dtype=float)
        # AMIS weights are >= 0 by construction; drop non-finite/negative and fall
        # back to uniform only if the entire vector is degenerate.
        weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(samples)
        w = weights / float(weights.sum())
        wmean = float(np.sum(w * samples))
        wvar = float(np.sum(w * (samples - wmean) ** 2))
        q05, q50, q95 = _weighted_quantiles(samples, w, [0.05, 0.5, 0.95])
        ess = float(1.0 / np.sum(w ** 2))
        rows.append(
            {
                "method": str(result.method),
                "replicate": int(result.replicate),
                "weighted_posterior_mean": wmean,
                "weighted_posterior_var": wvar,
                "weighted_posterior_q05": q05,
                "weighted_posterior_q50": q50,
                "weighted_posterior_q95": q95,
                "analytic_posterior_mean": analytic_mean,
                "analytic_posterior_std": analytic_std,
                "weighted_mean_abs_error": abs(wmean - analytic_mean),
                "analytic_mean_in_90ci": int(q05 <= analytic_mean <= q95),
                "ess": ess,
                "ess_fraction": ess / float(samples.size),
                "max_norm_weight": float(w.max()),
                "n_particles_used": int(samples.size),
            }
        )
    if not rows:
        return

    pd.DataFrame(rows).sort_values(["method", "replicate"]).to_csv(
        output_dir.data / "gaussian_weighted_posterior_summary.csv",
        index=False,
    )
