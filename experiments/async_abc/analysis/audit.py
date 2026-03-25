"""Audit helpers for determining whether paper-facing plots are trustworthy."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ._helpers import records_to_frame
from .final_state import final_state_results

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
