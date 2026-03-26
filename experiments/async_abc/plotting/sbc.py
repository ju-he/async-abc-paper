"""Shared plotting helpers for SBC summaries."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..io.paths import OutputDir
from .export import save_figure


def plot_rank_histogram(ranks_df: Any, output_dir: OutputDir) -> None:
    """Render the standard SBC rank histogram figure."""
    if ranks_df.empty:
        return

    methods = list(ranks_df["method"].dropna().unique())
    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(6, max(3.5, 2.8 * len(methods))),
        squeeze=False,
    )
    for idx, method in enumerate(methods):
        ax = axes[idx, 0]
        group = ranks_df[ranks_df["method"] == method]
        bins = int(group["n_samples"].max()) + 1 if not group.empty else 10
        ax.hist(group["rank"], bins=min(max(bins, 5), 30), color="steelblue", alpha=0.8)
        ax.set_title(f"Rank histogram: {method}")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "rank_histogram",
        data={col: ranks_df[col].tolist() for col in ranks_df.columns},
        metadata={
            "plot_name": "rank_histogram",
            "title": "Rank histogram",
            "summary_plot": True,
            "diagnostic_plot": False,
            "experiment_name": output_dir.root.name,
            "benchmark": False,
            "methods": methods,
        },
    )


def plot_coverage_table(coverage_df: Any, output_dir: OutputDir) -> None:
    """Render the standard SBC empirical coverage figure."""
    if coverage_df.empty:
        return

    plot_df = coverage_df.copy()
    grouped = plot_df.groupby(["method", "param"], dropna=False, sort=False)
    if "n_trials" not in plot_df.columns:
        plot_df["n_trials"] = grouped["empirical_coverage"].transform("count")
    if "empirical_coverage_ci_low" not in plot_df.columns or "empirical_coverage_ci_high" not in plot_df.columns:
        z = 1.959963984540054
        ci_low: list[float] = []
        ci_high: list[float] = []
        for row in plot_df.itertuples(index=False):
            n = max(int(getattr(row, "n_trials", 0)), 1)
            p = float(row.empirical_coverage)
            denom = 1.0 + (z * z) / n
            center = (p + (z * z) / (2.0 * n)) / denom
            margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
            ci_low.append(max(0.0, center - margin))
            ci_high.append(min(1.0, center + margin))
        plot_df["empirical_coverage_ci_low"] = ci_low
        plot_df["empirical_coverage_ci_high"] = ci_high

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in plot_df.groupby("method", dropna=False, sort=True):
        group = group.sort_values("coverage_level")
        ax.plot(group["coverage_level"], group["empirical_coverage"], marker="o", label=method or "method")
        if {"empirical_coverage_ci_low", "empirical_coverage_ci_high"} <= set(group.columns):
            ax.fill_between(
                group["coverage_level"],
                group["empirical_coverage_ci_low"],
                group["empirical_coverage_ci_high"],
                alpha=0.15,
            )
    line = np.linspace(0.0, 1.0, 50)
    ax.plot(line, line, linestyle="--", color="grey", label="ideal")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("SBC empirical coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir.plots / "coverage_table",
        data={col: plot_df[col].tolist() for col in plot_df.columns},
        metadata={
            "plot_name": "coverage_table",
            "title": "SBC empirical coverage",
            "summary_plot": True,
            "diagnostic_plot": False,
            "experiment_name": output_dir.root.name,
            "benchmark": False,
            "methods": sorted(plot_df["method"].dropna().unique().tolist()),
        },
    )
