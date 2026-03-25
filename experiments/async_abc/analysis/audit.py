"""Audit helpers for determining whether paper-facing plots are trustworthy."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ._helpers import records_to_frame
from .final_state import final_state_results

FALLBACK_LOSS_THRESHOLD = 1e6


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
