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

    has_benchmark_col = "benchmark" in ranks_df.columns and not ranks_df["benchmark"].dropna().empty
    if has_benchmark_col:
        panels = list(
            ranks_df[["benchmark", "method"]].dropna().drop_duplicates()
            .itertuples(index=False, name=None)
        )
    else:
        panels = [(None, m) for m in ranks_df["method"].dropna().unique()]

    n_panels = len(panels)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(6, max(3.5, 3.2 * n_panels)),
        squeeze=False,
    )
    for idx, (bench, method) in enumerate(panels):
        ax = axes[idx, 0]
        mask = ranks_df["method"] == method
        if has_benchmark_col and bench is not None:
            mask = mask & (ranks_df["benchmark"] == bench)
        group = ranks_df[mask]
        bins = int(group["n_samples"].max()) + 1 if not group.empty else 10
        ax.hist(group["rank"], bins=bins, color="steelblue", alpha=0.8)
        if group["rank"].shape[0] > 0:
            ax.axhline(len(group) / bins, color="grey", ls="--", lw=0.8, label="uniform")
        title = f"Rank histogram: {method}"
        if bench is not None:
            title = f"[{bench}] {title}"
        ax.set_title(title)
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
            "methods": [m for _, m in panels],
            "n_panels": n_panels,
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

    has_benchmark_col = "benchmark" in plot_df.columns and not plot_df["benchmark"].dropna().empty
    if has_benchmark_col:
        group_keys = ["benchmark", "method"]
    else:
        group_keys = ["method"]

    fig, ax = plt.subplots(figsize=(6, 4))
    combo_labels = []
    for keys, group in plot_df.groupby(group_keys, dropna=False, sort=True):
        if has_benchmark_col:
            bench, method = keys
            label = f"{bench} / {method}" if bench else str(method)
        else:
            method = keys
            label = str(method) if method else "method"
        combo_labels.append(label)
        group = group.sort_values("coverage_level")
        ax.plot(group["coverage_level"], group["empirical_coverage"], marker="o", label=label)
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
            "methods": combo_labels,
        },
    )
