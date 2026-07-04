#!/usr/bin/env python3
"""Generate the 5-point Cellular Potts strong-scaling figure (fig_cpm_scaling.pdf).

Throughput (sims/s) vs worker count, asynchronous vs matched synchronous baseline,
k=100, on the cost-bearing cellsInSilico simulator. Points 1/4/16 come from the
fill-in run, 48/96 from the original CPM scaling run. Unlike the near-instantaneous
Lotka-Volterra case, the async method scales ~linearly here (even across the
48->96 single-node->multi-node boundary) because real per-evaluation cost amortizes
the per-arrival coordination, while the synchronous baseline plateaus.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILL = "/home/juhe/remotes/scratch/herold2/async-abc/run_cpm_fillin_20260628/scaling_cpm/data"
EXIST = "/home/juhe/remotes/scratch/herold2/async-abc/run_cpm_20260626_1906/scaling_cpm/data"
EXT = "/home/juhe/remotes/scratch/herold2/async-abc/cpm_scaling_ext_20260630/scaling_cpm/data"
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_cpm_scaling.pdf"

# 1/4/16 from the fill-in run; 48/96 from the original CPM scaling run; 192/384 from
# the node-scaling extension (cpm_scaling_ext_20260630, k=100). Medians over the
# available replicates (the single slow 384-worker replicate is robustly handled by
# the median).
POINTS = [(1, FILL), (4, FILL), (16, FILL), (48, EXIST), (96, EXIST), (192, EXT), (384, EXT)]
ASYNC, SYNC = "async_propulate_abc", "abc_smc_baseline"


def _median(d: str, w: int, method: str) -> float:
    f = os.path.join(d, f"throughput_summary_w{w}_k100.csv")
    df = pd.read_csv(f)
    s = df[df["base_method"] == method]["throughput_sims_per_s"]
    return float(np.median(s)) if len(s) else float("nan")


def main() -> None:
    workers = [w for w, _ in POINTS]
    a = [_median(d, w, ASYNC) for w, d in POINTS]
    s = [_median(d, w, SYNC) for w, d in POINTS]

    # Ideal-linear reference anchored at the 1-worker async throughput.
    ideal = [a[0] * w for w in workers]

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(6.0, 4.3))

    ax.plot(workers, ideal, ":", color="0.55", lw=1.4, label="ideal linear", zorder=1)
    ax.plot(workers, a, "-o", color="#1f77b4", lw=2.0, ms=8, label="asynchronous (ours)", zorder=3)
    ax.plot(workers, s, "--s", color="#d62728", lw=2.0, ms=7, mfc="white", label="synchronous baseline", zorder=2)

    # Mark the single-node -> multi-node boundary (48 workers = one 48-core node);
    # the curve continues to scale across it all the way to 8 nodes (384 workers).
    ax.axvline(48, color="0.8", lw=1.0, ls="-", zorder=0)
    ax.annotate("1 node $\\to$ multi-node", xy=(48, ax.get_ylim()[0]), xytext=(50, 0.45),
                fontsize=9, color="0.4", rotation=90, va="bottom", ha="left")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers])
    ax.set_xlabel("workers")
    ax.set_ylabel("throughput (simulations / s)")
    ax.set_title("Cellular Potts strong scaling ($k{=}100$)")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(frameon=False, loc="upper left")

    # Annotate the largest-scale gap (8 nodes / 384 workers).
    ax.annotate(f"{a[-1]/s[-1]:.0f}$\\times$", xy=(workers[-1], a[-1]),
                xytext=(workers[-1] * 0.5, a[-1] * 1.35), fontsize=11, color="#1f77b4")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("workers:", workers)
    print("async  :", [round(x, 2) for x in a])
    print("sync   :", [round(x, 2) for x in s])
    # parallel efficiency relative to the single full node (48 workers)
    i48 = workers.index(48)
    print(f"async eff @384 vs 48-worker node-rate = {a[-1]/(a[i48]*(workers[-1]/48)):.2%}")
    print(f"async parallel efficiency @384 vs 1 worker = {a[-1]/(a[0]*workers[-1]):.2%}")
    print(f"async/sync @384 = {a[-1]/s[-1]:.2f}x")


if __name__ == "__main__":
    main()
