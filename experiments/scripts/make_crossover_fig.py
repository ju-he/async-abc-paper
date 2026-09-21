#!/usr/bin/env python3
"""Where asynchrony pays: throughput ratio against cost per simulation (fig_crossover.pdf).

One point per benchmark from ``tab_matched_eps`` (``make_matched_eps_table.py``):
the asynchronous arm's simulation throughput relative to the matched synchronous
baseline at equal wall clock, against the mean cost of one simulation. The
asynchronous method pays a roughly fixed per-arrival coordination cost (rebuild
the proposal, weight the arrival, update the archive) that dominates a
microsecond simulator and vanishes against a seconds-long one, so the ratio is
monotone in the simulator's cost and crosses one at about four milliseconds
per simulation. The per-simulation efficiency of the sampler (the tolerance it
reaches at a matched simulation count, second marker) rises with the cost too,
and is at or above the baseline's on every benchmark but the one-dimensional
analytic target.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import _figdata as fd
from async_abc.plotting import paper_style as ps

NAME = "fig_crossover"
LABEL = {"gaussian_mean": "Gaussian mean (1-D)", "gandk": "g-and-k (4-D)",
         "lotka_volterra": "Lotka–Volterra (4-D)", "cellular_potts": "Cellular Potts (2-D)"}
# (dx, dy) in points and horizontal alignment, chosen so no label crosses a curve or the legend.
OFFSET = {"gaussian_mean": (6, -10, "left"), "gandk": (-6, 4, "right"),
          "lotka_volterra": (6, -10, "left"), "cellular_potts": (-6, 4, "right")}


def aggregate(root):
    return {"crossover": pd.read_csv(ps.DATA_DIR / "tab_matched_eps" / "matched_eps_summary.csv")}


def draw(frames):
    df = frames["crossover"].sort_values("sim_cost_s")
    fig, ax = plt.subplots(figsize=ps.fig_size(0.55, aspect=0.8))
    ax.axhline(1.0, color=ps.COLORS["reference"], ls=(0, (5, 3)), lw=0.8)
    ax.plot(df["sim_cost_s"], df["throughput_ratio"], marker=ps.MARKERS["async"], color=ps.COLORS["async"],
            ls="-", label="throughput ratio (equal wall clock)")
    ax.plot(df["sim_cost_s"], df["per_simulation_ratio"], marker=ps.MARKERS["sync"], color=ps.COLORS["sync"],
            ls="--", label="per-simulation efficiency, $\\epsilon_{\\mathrm{sync}}/\\epsilon_{\\mathrm{async}}$ (matched $n$)")
    for _, r in df.iterrows():
        dx, dy, ha = OFFSET.get(r.benchmark, (4, -10, "left"))
        ax.annotate(LABEL.get(r.benchmark, r.benchmark), (r.sim_cost_s, r.throughput_ratio),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.5, color="0.25")
    ax.text(0.985, 1.0, "parity", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=6.5, color=ps.COLORS["reference"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 8.0)
    ax.set_xlabel("cost of one simulation (s)")
    ax.set_ylabel("ratio, async over sync")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize="small")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fd.run(NAME, __doc__, aggregate, draw)
