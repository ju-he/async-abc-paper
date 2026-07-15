#!/usr/bin/env python3
"""Straggler robustness figure (fig_straggler_throughput.pdf).

One worker is given a permanent post-evaluation delay of 0.1 s scaled by
0x (control), 1x, 5x, 10x, 20x, identically for both methods (16 workers,
Gaussian-mean, 300 s). Throughput (simulations/s within the timed budget) is
plotted per method as median + inter-quartile range over 5 replicates.

The synchronous baseline barriers on the slowest worker each generation, so its
throughput collapses as the straggler slows (~8800 -> ~60 sims/s from the
control to 20x). The asynchronous method never waits, so it stays essentially
flat (~3900 sims/s), overtaking the baseline from 1x onward. At the 0x control
the barrier is cheap for a near-instant simulator, so the baseline is faster
there -- the honest boundary of the async advantage.

Default draws from the vendored CSV; ``--refresh`` re-derives it from the
campaign straggler run. Styling via async_abc.plotting.paper_style.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _figdata as fd
from async_abc.plotting import paper_style as ps

FIG = "fig_straggler_throughput"
ORDER = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
FACTORS = [0.0, 1.0, 5.0, 10.0, 20.0]


def aggregate(root: Path):
    df = pd.read_csv(root / "straggler" / "data" / "throughput_vs_slowdown_summary.csv")
    g = (
        df.groupby(["base_method", "slowdown_factor"])["throughput_sims_per_s"]
        .agg(throughput_median="median",
             throughput_q1=lambda s: s.quantile(0.25),
             throughput_q3=lambda s: s.quantile(0.75),
             n_replicates="count")
        .reset_index()
        .sort_values(["base_method", "slowdown_factor"])
    )
    return {"straggler_throughput": g}


def draw(frames):
    g = frames["straggler_throughput"]
    x = np.arange(len(FACTORS))
    fig, ax = plt.subplots(figsize=ps.fig_size(0.6, aspect=0.72))
    for m in ORDER:
        k = KEY[m]
        sub = g[g["base_method"] == m].set_index("slowdown_factor").reindex(FACTORS)
        ax.fill_between(x, sub["throughput_q1"], sub["throughput_q3"],
                        color=ps.COLORS[k], alpha=0.18, linewidth=0)
        ax.plot(x, sub["throughput_median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                ls=ps.LINESTYLES[k], label=ps.LABELS[k])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["0\n(control)", "1", "5", "10", "20"])
    ax.set_xlabel("straggler slowdown factor (x base 0.1 s delay)")
    ax.set_ylabel("throughput (simulations / s)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout()
    return fig


def main() -> None:
    import argparse
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
    g = frames["straggler_throughput"]
    for m in ORDER:
        sub = g[g["base_method"] == m]
        vals = ", ".join(f"{f:g}x={v:.0f}" for f, v in
                         zip(sub["slowdown_factor"], sub["throughput_median"]))
        print(f"  {KEY[m]:>5} median sims/s: {vals}")


if __name__ == "__main__":
    main()
