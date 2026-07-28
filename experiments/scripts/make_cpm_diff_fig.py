#!/usr/bin/env python3
"""Cellular Potts posterior-recovery DIFFERENCE panel (fig_cpm_recovery_diff.pdf).

Reviewer asked for a difference plot so that "comparable" is legible on the CPM
posterior-recovery panel. This plots W_async(t) - W_sync(t) (Wasserstein to the
reference posterior) versus wall-clock time, with a zero reference line and an
uncertainty envelope from the per-method inter-quartile ranges over the five
replicates. Each method's per-replicate trajectory is last-observation-carried-
forward onto a shared time grid (the same alignment used for the recovery
curves), then summarized; the difference is async-median minus sync-median.

The grid is truncated at the shared *measured support* — the replicate-median last
real checkpoint of the worse-covered method (asynchronous, ~1.0e3 s) — because
quality checkpoints are recorded on an evaluation counter rather than on wall-clock,
so the faster method exhausts its retained checkpoints first. Past that point the
asynchronous curve would be a carried-forward constant differenced against a
still-updating synchronous one.

Default draws from the vendored CSV; ``--refresh`` re-derives it from the
campaign output (``<root>/cellular_potts/plots/quality_vs_wall_time_
diagnostic_data.csv`` — the same series Fig.~\\ref{fig:posterior-recovery} plots,
so this really is those curves differenced). Styling via
async_abc.plotting.paper_style (Type-42, Okabe-Ito, print width).
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

ASYNC, SYNC = "async_propulate_abc", "abc_smc_baseline"

# Matched-budget wall-clock grid (identical to the recovery-curve alignment). The
# upper end is derived from the data (see ``_support_cap``), not hardcoded: the
# asynchronous checkpoints run out at ~1.0e3 s, so a fixed 1800 s grid would plot
# ~45% carried-forward constant.
GRID_LO, N_GRID = 60.0, 80


def _support_cap(df: pd.DataFrame) -> float:
    """Last wall-clock time at which both methods still have real observations."""
    caps = []
    for m in (ASYNC, SYNC):
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        caps.append(float(sub.groupby("replicate")["wall_time"].max().median()))
    return min(caps)


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


def aggregate(root: Path):
    """Derive the plotted async-minus-sync difference frame from the campaign CSV."""
    csv = root / "cellular_potts" / "plots" / "quality_vs_wall_time_diagnostic_data.csv"
    df = pd.read_csv(csv)
    df = df[df["method"].isin([ASYNC, SYNC])].copy()
    df["wall_time"] = pd.to_numeric(df["wall_time"], errors="coerce")
    df["wasserstein"] = pd.to_numeric(df["wasserstein"], errors="coerce")
    df = df.dropna(subset=["wall_time", "wasserstein"])
    grid = np.linspace(GRID_LO, _support_cap(df), N_GRID)

    a = _locf_curves(df, ASYNC, grid)
    s = _locf_curves(df, SYNC, grid)
    a_med, a_lo, a_hi = np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0)
    s_med, s_lo, s_hi = np.nanmedian(s, 0), np.nanpercentile(s, 25, 0), np.nanpercentile(s, 75, 0)

    diff = a_med - s_med               # < 0  => async better
    band_lo = a_lo - s_hi              # conservative envelope of the difference
    band_hi = a_hi - s_lo

    frame = pd.DataFrame(
        {"wall_time": grid, "diff_median": diff, "band_lo": band_lo, "band_hi": band_hi}
    )
    return {"cpm_recovery_diff": frame}


def draw(frames):
    d = frames["cpm_recovery_diff"]
    grid = d["wall_time"].to_numpy(dtype=float)
    diff = d["diff_median"].to_numpy(dtype=float)
    band_lo = d["band_lo"].to_numpy(dtype=float)
    band_hi = d["band_hi"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=ps.fig_size(0.55, aspect=0.68))
    ax.axhline(0.0, color=ps.COLORS["reference"], ls=":", lw=1.0)
    ax.fill_between(grid, band_lo, band_hi, color=ps.COLORS["async"], alpha=0.15,
                    label="inter-quartile envelope")
    ax.plot(grid, diff, color=ps.COLORS["async"], ls=ps.LINESTYLES["async"], lw=1.4,
            label=r"async $-$ sync (median)")
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel(r"$W_{\mathrm{async}} - W_{\mathrm{sync}}$")
    # Direction cue (no in-figure title): below the zero line means async recovers better.
    ax.text(0.98, 0.04, "below 0: async better", transform=ax.transAxes,
            ha="right", va="bottom", color=ps.COLORS["neutral"])
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout()
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
        frames = fd.load_vendored("fig_cpm_recovery_diff")
        vendor = None  # already committed; don't rewrite

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, "fig_cpm_recovery_diff", data=vendor)
    print(f"wrote {saved['pdf']}")

    d = frames["cpm_recovery_diff"]
    diff = d["diff_median"].to_numpy(dtype=float)
    band_lo = d["band_lo"].to_numpy(dtype=float)
    band_hi = d["band_hi"].to_numpy(dtype=float)
    grid = d["wall_time"].to_numpy(dtype=float)
    print(f"median diff range: [{np.nanmin(diff):+.3f}, {np.nanmax(diff):+.3f}]")
    print(f"final diff (t={grid[-1]:.0f}s): {diff[-1]:+.3f}")
    print(f"envelope straddles zero at final t: {band_lo[-1] < 0 < band_hi[-1]}")


if __name__ == "__main__":
    main()
