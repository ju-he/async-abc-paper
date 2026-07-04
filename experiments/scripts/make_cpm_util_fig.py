#!/usr/bin/env python3
"""Cellular Potts worker utilization async vs.\ sync (fig_cpm_util.pdf).

Instruments the \\S7.3 claim that the generation barrier --- not per-arrival overhead
--- is the binding cost on the cost-bearing CPM simulator. Uses the worker_utilization
column already recorded in the existing CPM strong-scaling run (run_cpm_20260626_1906,
k=100, 1800 s, five replicates per cell); no re-run. The asynchronous method stays
near-fully utilized at both worker counts, while the synchronous baseline idles at the
barrier --- and its idle fraction *grows* with scale (39% -> 59% from 48 to 96 workers)
as more workers wait on the slowest simulation per generation, which is exactly why its
throughput plateaus while the asynchronous method keeps scaling.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCR = "/home/juhe/remotes/scratch/herold2/async-abc"
D = f"{SCR}/run_cpm_20260626_1906/scaling_cpm/data"
EXT = f"{SCR}/cpm_scaling_ext_20260630/scaling_cpm/data"
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_cpm_util.pdf"
WORKERS = [48, 96, 192, 384]
# Asynchronous utilisation from the async scaling runs (48/96 original, 192/384 extension).
DIR_ASYNC = {48: (D, 100), 96: (D, 100), 192: (EXT, 100), 384: (EXT, 100)}
# Synchronous utilisation from the FAIR baseline (population = worker count) at the
# cap-affected multi-node counts; 48/96 already used population 100 >= workers, so
# they are fair as-is and come from the same original run.
DIR_SYNC = {48: (D, 100), 96: (D, 100),
            192: (f"{SCR}/cpm_fair_w192_20260701/scaling_cpm/data", 192),
            384: (f"{SCR}/cpm_fair_w384_20260701/scaling_cpm/data", 384)}
STYLE = {
    "async_propulate_abc": dict(label="Asynchronous (ours)", color="#1f77b4"),
    "abc_smc_baseline":    dict(label="Synchronous baseline (population = cores)", color="#d62728"),
}
ORDER = ["async_propulate_abc", "abc_smc_baseline"]


def _util_one(src, w, method):
    d, k = src
    df = pd.read_csv(f"{d}/throughput_summary_w{w}_k{k}.csv")
    sub = df[df["base_method"] == method]["worker_utilization"]
    return (100 * sub.mean(), 100 * sub.std())


def _util(w: int):
    return {
        "async_propulate_abc": _util_one(DIR_ASYNC[w], w, "async_propulate_abc"),
        "abc_smc_baseline": _util_one(DIR_SYNC[w], w, "abc_smc_baseline"),
    }


def main() -> None:
    util = {w: _util(w) for w in WORKERS}

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    x = np.arange(len(WORKERS))
    bw = 0.36
    for j, m in enumerate(ORDER):
        means = [util[w][m][0] for w in WORKERS]
        sds = [util[w][m][1] for w in WORKERS]
        bars = ax.bar(x + (j - 0.5) * bw, means, bw, yerr=sds, capsize=4,
                      color=STYLE[m]["color"], label=STYLE[m]["label"],
                      edgecolor="white", linewidth=0.6)
        for b, mn in zip(bars, means):
            ax.annotate(f"{mn:.0f}%", xy=(b.get_x() + b.get_width() / 2, mn),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w} workers" for w in WORKERS])
    ax.set_ylabel("worker utilization (%)")
    ax.set_ylim(0, 112)
    # Legend outside, above the axes, so it never overlaps the (near-100%) bars.
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, borderaxespad=0.0)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    for w in WORKERS:
        for m in ORDER:
            mn, sd = util[w][m]
            print(f"w{w:>3} {m:>22}: util {mn:5.1f}% (idle {100-mn:4.1f}%)  sd {sd:.1f}")


if __name__ == "__main__":
    main()
