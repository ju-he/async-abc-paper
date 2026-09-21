#!/usr/bin/env python3
"""Tolerance against simulations spent, every benchmark (fig_eps_curves.pdf).

One panel per benchmark: the k-th (k=100) order statistic of each method's own
losses after n simulations (arrival order, simulation_attempt rows only) --
the bandwidth at which exactly k of its draws would be accepted -- median and
replicate range over five replicates, asynchronous against the matched
synchronous baseline. The vertical gap at equal n is the per-simulation
efficiency; the horizontal extent is what the wall clock bought; the slope over
the last decade of n (in the legend) is the exponent that decides whether more
simulations keep converting into a tighter tolerance. Curves come from
``tab_matched_eps/matched_eps_curves.csv`` (``make_matched_eps_table.py``).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _figdata as fd
from async_abc.plotting import paper_style as ps

NAME = "fig_eps_curves"
PANELS = [("gaussian_mean", "(a) Gaussian mean, 1-D, 0.1 ms"), ("gandk", "(b) g-and-k, 4-D, 2 ms"),
          ("lotka_volterra", "(c) Lotka–Volterra, 4-D, 4 ms"), ("cellular_potts", "(d) Cellular Potts, 2-D, 13 s")]


def aggregate(root):
    return {"eps_curves": pd.read_csv(ps.DATA_DIR / "tab_matched_eps" / "matched_eps_curves.csv")}


def _exponent(sub: pd.DataFrame) -> float:
    med = sub.groupby("n")["eps"].median()
    med = med[med.index >= med.index.max() / 10]
    if len(med) < 3:
        return float("nan")
    return float(np.polyfit(np.log(med.index.to_numpy(float)), np.log(med.to_numpy()), 1)[0])


def draw(frames):
    df = frames["eps_curves"]
    fig, axes = plt.subplots(2, 2, figsize=ps.fig_size(0.85, aspect=0.8))
    axes = axes.ravel()
    for ax, (bench, title) in zip(axes, PANELS):
        sub = df[df["benchmark"] == bench]
        for key in ("sync", "async"):
            s = sub[sub["method"] == key]
            if s.empty:
                continue
            med = s.groupby("n")["eps"].agg(["median", "min", "max"])
            ax.fill_between(med.index, med["min"], med["max"], color=ps.COLORS[key], alpha=0.15, lw=0)
            ax.plot(med.index, med["median"], color=ps.COLORS[key], ls=ps.LINESTYLES[key],
                    label=f"{ps.LABELS[key]} ($n^{{{_exponent(s):+.2f}}}$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if bench == "lotka_volterra":
            # 98% of Lotka-Volterra simulations go extinct and return the fallback
            # discrepancy (1e6), so eps_(100) sits on that plateau until ~5,000
            # draws have produced 100 survivors. The plateau is cut off here.
            ax.set_ylim(top=3e3)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_xlabel("simulations, $n$")
        ax.grid(True, ls=":", lw=0.4, alpha=0.6)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.8)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("$\\epsilon_{(100)}$")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fd.run(NAME, __doc__, aggregate, draw)
