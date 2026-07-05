"""Audit helpers for determining whether paper-facing plots are trustworthy."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ._helpers import records_to_frame
from .final_state import final_state_results
from .final_state import base_method_name


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation that returns NaN on degenerate inputs."""
    if x.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0 or not np.isfinite(sx) or not np.isfinite(sy):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def simulation_time_bias_report(records: Iterable) -> dict[str, object]:
    """Diagnose simulation-time bias in an asynchronous-ABC run (W3.3).

    Steady-state asynchronous ABC keeps every accepted particle, so faster-
    simulating parameter regions can be over-represented in the archive
    relative to slow-simulating regions of the posterior. Paper §17
    acknowledges this; this report quantifies it per run.

    Inputs are :class:`ParticleRecord`-shaped rows. Returns a dictionary
    summarising:

    - ``mean_sim_time_accepted`` / ``mean_sim_time_rejected``: average
      per-evaluation wall-time, segmented by acceptance (loss < tolerance).
      Equal means imply no bias; a large gap means simulator runtime
      correlates with acceptance.
    - ``pearson_corr_simtime_loss``: Pearson r between per-evaluation
      simulation time and loss across all records. A positive r indicates
      slow simulations tend to have high loss (poor regions); a negative
      r indicates the asymmetry that creates the bias.
    - ``chi2_p_value``: χ² test of independence between (sim_time-low,
      sim_time-high) and (accepted, rejected) using median splits. Small
      p-values reject independence and confirm bias.

    Records without ``sim_start_time`` / ``sim_end_time`` (e.g. sync methods
    that record only generation-level spans) are skipped from the per-
    evaluation analysis; the report still returns the run-level totals.
    """
    frame = records_to_frame(records)
    if frame.empty:
        return {
            "n_records": 0,
            "mean_sim_time_accepted": float("nan"),
            "mean_sim_time_rejected": float("nan"),
            "pearson_corr_simtime_loss": float("nan"),
            "chi2_p_value": float("nan"),
            "skip_reason": "empty_records",
        }

    # Derive per-evaluation sim_time as (sim_end - sim_start). Records that
    # lack both timestamps are skipped (sim_time = NaN); sync methods that
    # only emit generation-level spans will fall into this bucket.
    def _row_sim_time(row) -> float:
        s_start = row.get("sim_start_time", None)
        s_end = row.get("sim_end_time", None)
        if pd.notna(s_start) and pd.notna(s_end):
            try:
                dt = float(s_end) - float(s_start)
                if np.isfinite(dt) and dt > 0:
                    return dt
            except (TypeError, ValueError):
                pass
        return float("nan")

    sim_times = frame.apply(_row_sim_time, axis=1).to_numpy()
    losses = pd.to_numeric(frame["loss"], errors="coerce").to_numpy()
    tolerances = pd.to_numeric(frame.get("tolerance", pd.Series([])), errors="coerce")
    # An evaluation is "accepted" if its loss is below its stamped tolerance.
    # Records with no stamped tolerance (prior-phase or sync rejection-ABC)
    # use the run's initial tolerance via a fallback below.
    if tolerances.empty:
        tol_arr = np.full_like(losses, np.nan, dtype=float)
    else:
        tol_arr = tolerances.to_numpy()
    fallback_tol = float(np.nanmax(tol_arr)) if np.any(np.isfinite(tol_arr)) else float("inf")
    eff_tol = np.where(np.isfinite(tol_arr), tol_arr, fallback_tol)
    accepted_mask = (losses < eff_tol) & np.isfinite(losses)

    finite_mask = np.isfinite(sim_times) & np.isfinite(losses)
    n_finite = int(np.sum(finite_mask))
    if n_finite == 0:
        return {
            "n_records": int(len(frame)),
            "mean_sim_time_accepted": float("nan"),
            "mean_sim_time_rejected": float("nan"),
            "pearson_corr_simtime_loss": float("nan"),
            "chi2_p_value": float("nan"),
            "skip_reason": "no_finite_sim_time_or_loss",
        }

    sim_finite = sim_times[finite_mask]
    loss_finite = losses[finite_mask]
    accepted_finite = accepted_mask[finite_mask]

    mean_acc = float(np.mean(sim_finite[accepted_finite])) if accepted_finite.any() else float("nan")
    mean_rej = float(np.mean(sim_finite[~accepted_finite])) if (~accepted_finite).any() else float("nan")
    pearson = _safe_pearson(sim_finite, loss_finite)

    # χ² independence test on median-split sim_time vs accepted.
    chi2_p: float
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        chi2_p = float("nan")
    else:
        median = float(np.median(sim_finite))
        slow = sim_finite >= median
        # Build a 2x2 contingency table.
        table = np.array(
            [
                [int(np.sum(slow & accepted_finite)), int(np.sum(slow & ~accepted_finite))],
                [int(np.sum(~slow & accepted_finite)), int(np.sum(~slow & ~accepted_finite))],
            ],
            dtype=int,
        )
        if table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
            chi2_p = float("nan")
        else:
            _, chi2_p, _, _ = chi2_contingency(table)
            chi2_p = float(chi2_p)

    return {
        "n_records": int(len(frame)),
        "n_accepted": int(np.sum(accepted_finite)),
        "n_rejected": int(np.sum(~accepted_finite)),
        "mean_sim_time_accepted": mean_acc,
        "mean_sim_time_rejected": mean_rej,
        "pearson_corr_simtime_loss": pearson,
        "chi2_p_value": chi2_p,
        "skip_reason": None,
    }

FALLBACK_LOSS_THRESHOLD = 1e6
PATHOLOGICAL_FALLBACK_FRACTION = 0.95


def benchmark_plot_audit(
    records: Iterable,
    *,
    true_params: dict[str, float],
    archive_size: int | None = None,
    min_particles_for_threshold: int = 100,
) -> pd.DataFrame:
    """Return per-method/per-replicate plot validity diagnostics."""
    frame = records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "replicate",
                "final_tolerance",
                "tolerance_monotone",
                "wall_time_span",
                "attempt_count_span",
                "final_posterior_size",
                "fallback_or_extinction_fraction",
                "has_true_params",
                "paper_quality_plots_allowed",
                "paper_threshold_plots_allowed",
                "invalid_reason",
            ]
        )

    frame = frame.copy()
    frame["wall_time"] = pd.to_numeric(frame.get("wall_time"), errors="coerce")
    frame["sim_start_time"] = pd.to_numeric(frame.get("sim_start_time"), errors="coerce")
    frame["sim_end_time"] = pd.to_numeric(frame.get("sim_end_time"), errors="coerce")
    frame["attempt_count"] = pd.to_numeric(frame.get("attempt_count"), errors="coerce")
    frame["step"] = pd.to_numeric(frame.get("step"), errors="coerce")
    frame["tolerance"] = pd.to_numeric(frame.get("tolerance"), errors="coerce")
    frame["loss"] = pd.to_numeric(frame.get("loss"), errors="coerce")

    final_sizes = {
        (result.method, int(result.replicate)): int(result.n_particles_used)
        for result in final_state_results(records, archive_size=archive_size)
    }

    rows: list[dict[str, object]] = []
    has_true_params = bool(true_params)

    for (method, replicate), group in frame.groupby(["method", "replicate"], sort=True):
        group = group.sort_values(["wall_time", "sim_end_time", "step"], na_position="last").reset_index(drop=True)
        tolerances = group["tolerance"].dropna().to_numpy(dtype=float)
        if tolerances.size and base_method_name(str(method)) == "async_propulate_abc":
            tolerances = np.minimum.accumulate(tolerances)
        final_tolerance = float(np.nanmin(tolerances)) if tolerances.size else float("nan")
        tolerance_monotone = bool(np.all(np.diff(tolerances) <= 1e-12)) if tolerances.size > 1 else True

        wall_candidates = pd.concat(
            [
                group["sim_start_time"],
                group["sim_end_time"],
                group["wall_time"],
            ],
            axis=0,
        ).dropna()
        if wall_candidates.empty:
            wall_time_span = float("nan")
        else:
            wall_time_span = float(wall_candidates.max() - wall_candidates.min())

        attempts = group["attempt_count"].dropna()
        if attempts.empty:
            attempts = group["step"].dropna()
        if attempts.empty:
            attempt_count_span = float("nan")
        else:
            attempt_count_span = float(attempts.max() - attempts.min())

        finite_losses = group["loss"].replace([np.inf, -np.inf], np.nan).dropna()
        if finite_losses.empty:
            fallback_fraction = float("nan")
        else:
            fallback_fraction = float((finite_losses >= FALLBACK_LOSS_THRESHOLD).mean())

        final_posterior_size = int(final_sizes.get((method, int(replicate)), 0))

        invalid_reasons: list[str] = []
        if not has_true_params:
            invalid_reasons.append("missing_true_params")
        if not np.isfinite(wall_time_span) or wall_time_span <= 0.0:
            invalid_reasons.append("missing_wall_time_span")
        if not np.isfinite(attempt_count_span) or attempt_count_span <= 0.0:
            invalid_reasons.append("missing_attempt_count_span")
        if final_posterior_size <= 0:
            invalid_reasons.append("empty_final_posterior")
        if not tolerance_monotone:
            invalid_reasons.append("non_monotone_tolerance")
        if np.isfinite(fallback_fraction) and fallback_fraction >= PATHOLOGICAL_FALLBACK_FRACTION:
            invalid_reasons.append("pathological_fallback_or_extinction")

        paper_quality_allowed = not invalid_reasons
        paper_threshold_allowed = paper_quality_allowed and final_posterior_size >= int(min_particles_for_threshold)
        if paper_quality_allowed and not paper_threshold_allowed:
            invalid_reasons.append("insufficient_posterior_samples_for_threshold")

        rows.append(
            {
                "method": method,
                "replicate": int(replicate),
                "final_tolerance": final_tolerance,
                "tolerance_monotone": bool(tolerance_monotone),
                "wall_time_span": wall_time_span,
                "attempt_count_span": attempt_count_span,
                "final_posterior_size": final_posterior_size,
                "fallback_or_extinction_fraction": fallback_fraction,
                "has_true_params": has_true_params,
                "paper_quality_plots_allowed": bool(paper_quality_allowed),
                "paper_threshold_plots_allowed": bool(paper_threshold_allowed),
                "invalid_reason": ";".join(invalid_reasons),
            }
        )

    return pd.DataFrame(rows).sort_values(["method", "replicate"]).reset_index(drop=True)


def lotka_tol_init_diagnostic(
    records: Iterable,
    *,
    fallback_loss_threshold: float = FALLBACK_LOSS_THRESHOLD,
    recommended_quantile: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Summarize fallback prevalence and a recommended Lotka ``tol_init``.

    The recommendation is based on the specified quantile of all finite,
    non-fallback losses observed across the run.
    """
    frame = records_to_frame(records)
    columns = [
        "method",
        "replicate",
        "finite_loss_count",
        "non_fallback_loss_count",
        "fallback_or_extinction_fraction",
        "non_fallback_loss_p50",
        "non_fallback_loss_p95",
        "recommended_tol_init",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns), {
            "pathological_fallback": False,
            "fallback_loss_threshold": float(fallback_loss_threshold),
            "recommended_quantile": float(recommended_quantile),
            "recommended_tol_init": None,
            "non_fallback_loss_count": 0,
        }

    frame = frame.copy()
    frame["loss"] = pd.to_numeric(frame.get("loss"), errors="coerce")
    rows: list[dict[str, object]] = []
    all_non_fallback_losses: list[float] = []

    for (method, replicate), group in frame.groupby(["method", "replicate"], sort=True):
        losses = group["loss"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        non_fallback = losses[losses < float(fallback_loss_threshold)]
        if non_fallback.size:
            all_non_fallback_losses.extend(non_fallback.tolist())
        rows.append(
            {
                "method": method,
                "replicate": int(replicate),
                "finite_loss_count": int(losses.size),
                "non_fallback_loss_count": int(non_fallback.size),
                "fallback_or_extinction_fraction": float((losses >= float(fallback_loss_threshold)).mean()) if losses.size else float("nan"),
                "non_fallback_loss_p50": float(np.quantile(non_fallback, 0.5)) if non_fallback.size else float("nan"),
                "non_fallback_loss_p95": float(np.quantile(non_fallback, 0.95)) if non_fallback.size else float("nan"),
                "recommended_tol_init": float(np.quantile(non_fallback, float(recommended_quantile))) if non_fallback.size else float("nan"),
            }
        )

    diagnostic_df = pd.DataFrame(rows).sort_values(["method", "replicate"]).reset_index(drop=True)
    finite_fallback = diagnostic_df["fallback_or_extinction_fraction"].replace([np.inf, -np.inf], np.nan).dropna()
    overall_fallback = float(finite_fallback.mean()) if not finite_fallback.empty else float("nan")
    if all_non_fallback_losses:
        overall_losses = np.asarray(all_non_fallback_losses, dtype=float)
        recommended_tol_init = float(np.quantile(overall_losses, float(recommended_quantile)))
        p50 = float(np.quantile(overall_losses, 0.5))
        p95 = float(np.quantile(overall_losses, 0.95))
    else:
        recommended_tol_init = None
        p50 = None
        p95 = None

    summary = {
        "pathological_fallback": bool(np.isfinite(overall_fallback) and overall_fallback > 0.5),
        "overall_fallback_fraction": overall_fallback,
        "fallback_loss_threshold": float(fallback_loss_threshold),
        "recommended_quantile": float(recommended_quantile),
        "recommended_tol_init": recommended_tol_init,
        "non_fallback_loss_count": int(len(all_non_fallback_losses)),
        "non_fallback_loss_p50": p50,
        "non_fallback_loss_p95": p95,
    }
    return diagnostic_df, summary
