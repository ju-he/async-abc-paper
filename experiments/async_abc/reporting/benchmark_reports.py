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
