#!/usr/bin/env python3
"""Cellular Potts posterior-recovery DIFFERENCE panel (fig_cpm_recovery_diff.pdf).

Reviewer asked for a difference plot so that "comparable" is legible on the CPM
posterior-recovery panel. This plots W_async(t) - W_sync(t) (Wasserstein to the
reference) versus wall-clock, with a zero reference line and an uncertainty envelope
from the per-method inter-quartile ranges over the five replicates. Each method's
per-replicate trajectory is last-observation-carried-forward onto a shared time grid
(the same alignment used for the recovery curves), then summarized.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = ("/home/juhe/remotes/scratch/herold2/async-abc/run_full_20260626_1816/"
       "cellular_potts/plots/quality_vs_attempt_budget_diagnostic_data.csv")
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_cpm_recovery_diff.pdf"
ASYNC, SYNC = "async_propulate_abc", "abc_smc_baseline"


def _locf_curves(df: pd.DataFrame, method: str, grid: np.ndarray) -> np.ndarray:
    """reps x grid array of LOCF Wasserstein for one method."""
    sub = df[df["method"] == method]
    rows = []
    for _, g in sub.groupby("replicate"):
        g = g.sort_values("wall_time")
        t = g["wall_time"].to_numpy(dtype=float)
        w = g["wasserstein"].to_numpy(dtype=float)
        idx = np.searchsorted(t, grid, side="right") - 1
        wr = np.where(idx >= 0, w[np.clip(idx, 0, len(w) - 1)], np.nan)
        rows.append(wr)
    return np.array(rows)


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["method"].isin([ASYNC, SYNC])]
    tmax = 1800.0  # matched budget
    grid = np.linspace(60.0, tmax, 80)

    a = _locf_curves(df, ASYNC, grid)
    s = _locf_curves(df, SYNC, grid)
    a_med, a_lo, a_hi = np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0)
    s_med, s_lo, s_hi = np.nanmedian(s, 0), np.nanpercentile(s, 25, 0), np.nanpercentile(s, 75, 0)

    diff = a_med - s_med               # < 0  => async better
    band_lo = a_lo - s_hi              # conservative envelope of the difference
    band_hi = a_hi - s_lo

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.axhline(0.0, color="0.5", ls=":", lw=1.2)
    ax.fill_between(grid, band_lo, band_hi, color="#6a3d9a", alpha=0.15,
                    label="inter-quartile envelope")
    ax.plot(grid, diff, "-", color="#6a3d9a", lw=2.2, label="async $-$ sync (median)")
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel(r"$W_{\mathrm{async}}-W_{\mathrm{sync}}$")
    ax.set_title("Cellular Potts: posterior-quality difference")
    # annotate which direction is good
    ax.text(0.98, 0.04, "below 0: async better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="0.4")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, ls=":", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"median diff range: [{np.nanmin(diff):+.3f}, {np.nanmax(diff):+.3f}]")
    print(f"final diff (t={grid[-1]:.0f}s): {diff[-1]:+.3f}  (async {a_med[-1]:.3f} vs sync {s_med[-1]:.3f})")
    print(f"envelope straddles zero at final t: {band_lo[-1] < 0 < band_hi[-1]}")


if __name__ == "__main__":
    main()
