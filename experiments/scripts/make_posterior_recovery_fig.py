#!/usr/bin/env python3
"""Posterior-recovery figure (fig_posterior_recovery.pdf).

Three panels — g-and-k (left), Lotka--Volterra (middle), Cellular Potts (right)
— each showing posterior quality (Wasserstein distance to the truth/reference,
lower is better) versus wall-clock time for the asynchronous method vs. the
synchronous baseline. Per method, each replicate's quality-vs-time trace is
last-observation-carried-forward onto a shared per-benchmark time grid, then
summarised as median + inter-quartile range over the five replicates.

Each panel is truncated at the shared *measured support* (see ``_support_cap``):
quality checkpoints are recorded on an evaluation counter rather than on wall-clock,
so the faster method runs out of retained checkpoints first, and plotting past that
point would show a carried-forward constant rather than a measurement. On Cellular
Potts this caps the panel at ~1.0e3 s of the 3.6e3 s budget.

Data: ``<bench>/plots/quality_vs_wall_time_diagnostic_data.csv`` (per-replicate
checkpoints) from the campaign quality runs. Default draws from the vendored CSV;
``--refresh`` re-derives it. Styling via async_abc.plotting.paper_style.
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

FIG = "fig_posterior_recovery"
BENCHES = [("gandk", "g-and-k"), ("lotka_volterra", "Lotka–Volterra"),
           ("cellular_potts", "Cellular Potts")]
METHODS = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
N_GRID = 80


# Panels whose two methods have materially asymmetric checkpoint coverage, where
# carrying the last observation forward would compare a stale curve against a still-
# updating one. On Cellular Potts the asynchronous traces cover only ~28% of the
# budget against the synchronous ~71%; on g-and-k and Lotka--Volterra both methods
# cover ~86-100%, so those panels stay on the full budget.
TRUNCATE_TO_SUPPORT = {"cellular_potts"}


def _support_cap(df: pd.DataFrame) -> float:
    """Last wall-clock time at which *both* methods still have real observations.

    Quality checkpoints are recorded on an evaluation counter, not on wall-clock, so
    the faster method exhausts its retained checkpoints earlier: on Cellular Potts the
    asynchronous traces stop at ${\\approx}1.0\\times10^3$ s while the synchronous ones
    run past $3.3\\times10^3$ s. Carrying the last observation forward beyond that
    point would compare a *stale* asynchronous value against a still-updating
    synchronous one, so we cap the shared grid where either method's replicate-median
    support ends. Everything plotted is therefore backed by real checkpoints.
    """
    caps = []
    for m in METHODS:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        caps.append(float(sub.groupby("replicate")["wall_time"].max().median()))
    return min(caps)


def _locf_curve(sub: pd.DataFrame, grid: np.ndarray):
    reps = []
    for _rep, g in sub.groupby("replicate"):
        g = g.sort_values("wall_time")
        wt = g["wall_time"].to_numpy(float)
        w = g["wasserstein"].to_numpy(float)
        idx = np.searchsorted(wt, grid, side="right") - 1
        vals = np.where(idx >= 0, w[idx.clip(0, len(w) - 1)], np.nan)
        reps.append(vals)
    a = np.asarray(reps, dtype=float)
    return (np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0))


def aggregate(root: Path):
    rows = []
    for key, _label in BENCHES:
        df = pd.read_csv(root / key / "plots" / "quality_vs_wall_time_diagnostic_data.csv")
        df = df[df["method"].isin(METHODS)].copy()
        df["wall_time"] = pd.to_numeric(df["wall_time"], errors="coerce")
        df["wasserstein"] = pd.to_numeric(df["wasserstein"], errors="coerce")
        df["replicate"] = pd.to_numeric(df["replicate"], errors="coerce")
        df = df.dropna(subset=["wall_time", "wasserstein", "replicate"])
        pos = df["wall_time"] > 0
        hi = _support_cap(df) if key in TRUNCATE_TO_SUPPORT else float(df["wall_time"].max())
        lo = float(df.loc[pos, "wall_time"].min())
        grid = np.linspace(lo, hi, N_GRID)
        for m in METHODS:
            med, q1, q3 = _locf_curve(df[df["method"] == m], grid)
            for t, a, b, c in zip(grid, med, q1, q3):
                rows.append(dict(benchmark=key, method=m, wall_time=t,
                                 w_median=a, w_lo=b, w_hi=c))
    return {"posterior_recovery": pd.DataFrame(rows)}


def draw(frames):
    rec = frames["posterior_recovery"]
    fig, axes = plt.subplots(1, 3, figsize=ps.fig_size(1.0, aspect=0.34))
    tags = ["(a) g-and-k", "(b) Lotka–Volterra", "(c) Cellular Potts"]
    for ax, (key, _label), tag in zip(axes, BENCHES, tags):
        sub = rec[rec["benchmark"] == key]
        for m in METHODS:
            k = KEY[m]
            s = sub[sub["method"] == m]
            ax.fill_between(s["wall_time"], s["w_lo"], s["w_hi"],
                            color=ps.COLORS[k], alpha=0.18, lw=0)
            ax.plot(s["wall_time"], s["w_median"], color=ps.COLORS[k],
                    ls=ps.LINESTYLES[k], label=ps.LABELS[k])
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock time (s)")
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        ps.panel_tag(ax, tag)
    axes[0].set_ylabel("Wasserstein to truth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
    rec = frames["posterior_recovery"]
    for key, _ in BENCHES:
        sub = rec[rec["benchmark"] == key]
        fin = sub[sub["wall_time"] == sub["wall_time"].max()]
        vals = ", ".join(f"{KEY[m]}={fin[fin.method==m]['w_median'].iloc[0]:.3f}" for m in METHODS)
        print(f"  {key:>16} final W (median): {vals}")


if __name__ == "__main__":
    main()
