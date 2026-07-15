#!/usr/bin/env python3
"""Hyperparameter sensitivity heatmap (fig_sensitivity_heatmap.pdf).

Posterior quality (Wasserstein distance to the true mu, lower is better) of the
asynchronous method across the Gaussian-mean hyperparameter grid. Each panel
sweeps perturbation scale (x) against the initial-tolerance multiplier (y);
panels are faceted by archive size k. Every cell is the mean Wasserstein over
five replicates and the two smooth kernels (gaussian, epanechnikov).

The tolerance scheduler is fixed to ``acceptance_rate``: under a smooth kernel
the async selector chooses epsilon by ESS-retention bisection and the scheduler
type is inert, so it is not a swept axis here (the three schedulers would give
identical columns). Quality is stable across the grid (Wasserstein ~0.08-0.15)
with no failure region; larger perturbation scales and initial tolerances
degrade quality only mildly.

Data: ``sensitivity/data/sensitivity_quality_summary.csv`` (one row per grid
combo, produced by compute_sensitivity_quality_summary over the 64 combo runs).
Default draws from the vendored CSV; ``--refresh`` re-derives it. Styling via
async_abc.plotting.paper_style.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _figdata as fd
from async_abc.plotting import paper_style as ps

FIG = "fig_sensitivity_heatmap"
K_VALUES = [50, 200]
PERT = [0.4, 0.8, 1.5, 2.0]
TOL = [0.5, 1.0, 2.0, 5.0]


def aggregate(root: Path):
    df = pd.read_csv(root / "sensitivity" / "data" / "sensitivity_quality_summary.csv")
    # Average over the two smoothing kernels (and any residual duplicates).
    cell = (df.groupby(["k", "perturbation_scale", "tol_init_multiplier"])["wasserstein_mean"]
            .mean().reset_index().rename(columns={"wasserstein_mean": "wasserstein"}))
    return {"sensitivity_heatmap": cell}


def draw(frames):
    cell = frames["sensitivity_heatmap"]
    vmin = float(cell["wasserstein"].min())
    vmax = float(cell["wasserstein"].max())

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=ps.fig_size(1.0, aspect=0.5))
    im = None
    for ax, k in zip(axes, K_VALUES):
        grid = np.full((len(TOL), len(PERT)), np.nan)
        sub = cell[cell["k"] == k]
        for _, r in sub.iterrows():
            i = TOL.index(float(r["tol_init_multiplier"]))
            j = PERT.index(float(r["perturbation_scale"]))
            grid[i, j] = r["wasserstein"]
        im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(PERT)), [f"{p:g}" for p in PERT])
        ax.set_yticks(range(len(TOL)), [f"{t:g}" for t in TOL])
        ax.set_xlabel("perturbation scale")
        # annotate each cell with its value
        for i in range(len(TOL)):
            for j in range(len(PERT)):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v < (vmin + vmax) / 2 else "black")
        ps.panel_tag(ax, f"($k={k}$)")
    axes[0].set_ylabel("initial-tolerance multiplier")
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label("Wasserstein to true $\\mu$")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fd.add_refresh_arg(parser)
    args = parser.parse_args()
    ps.apply()

    if args.refresh is not None:
        frames = aggregate(Path(args.refresh))
        vendor = frames
    else:
        frames = fd.load_vendored(FIG)
        vendor = None

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, FIG, data=vendor)
    print(f"wrote {saved['pdf']}")
    cell = frames["sensitivity_heatmap"]
    print(f"  Wasserstein range across grid: {cell['wasserstein'].min():.3f} - {cell['wasserstein'].max():.3f}")


if __name__ == "__main__":
    main()
