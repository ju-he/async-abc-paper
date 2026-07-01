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

D = "/home/juhe/remotes/scratch/herold2/async-abc/run_cpm_20260626_1906/scaling_cpm/data"
EXT = "/home/juhe/remotes/scratch/herold2/async-abc/cpm_scaling_ext_20260630/scaling_cpm/data"
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_cpm_util.pdf"
# 48/96 from the original CPM scaling run; 192/384 from the node-scaling extension.
WORKERS = [48, 96, 192, 384]
DIR_FOR = {48: D, 96: D, 192: EXT, 384: EXT}
STYLE = {
    "async_propulate_abc": dict(label="asynchronous (ours)", color="#1f77b4"),
    "abc_smc_baseline":    dict(label="synchronous baseline", color="#d62728"),
}
ORDER = ["async_propulate_abc", "abc_smc_baseline"]


def _util(w: int):
    df = pd.read_csv(f"{DIR_FOR[w]}/throughput_summary_w{w}_k100.csv")
    out = {}
    for m, sub in df.groupby("base_method"):
        out[m] = (100 * sub["worker_utilization"].mean(), 100 * sub["worker_utilization"].std())
    return out


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
    ax.set_ylim(0, 109)
    ax.set_title("Cellular Potts: worker utilization")
    ax.legend(frameon=False, loc="lower left")
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
