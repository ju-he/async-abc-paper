#!/usr/bin/env python3
"""Replot the Lotka-Volterra strong-scaling throughput figure (fig_scaling_throughput.pdf).

Clean k=100 async-vs-synchronous view with inter-quartile bands over the five
replicates and the single-node->multi-node boundary marked. Unlike the cost-bearing
Cellular Potts case, async throughput here peaks at the full-node boundary (48
workers) and declines once the job spans multiple nodes, because the near-instantaneous
simulator leaves per-arrival coordination dominant -- the honest node-boundary story.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SUPERSEDED by make_scaling_combined_fig.py, whose left panel is this figure and
# whose data is vendored under experiments/data/paper_figures/fig_scaling_combined/.
# The paper includes fig_scaling_combined.pdf, not fig_scaling_throughput.pdf. Kept
# for the single-benchmark view; not part of the reproducibility path.
#
# NOTE the synchronous line here is the k=100 baseline at every worker count, NOT the
# FAIR baseline (population = worker count) the paper reports at 144-288 workers. The
# crossover the paper describes is against the fair baseline; this figure's sync line
# plateaus near 3400 sims/s instead. Read the comparison from fig_scaling_combined.
#
# The campaign run. This used to point at run_full_20260626_1816 (pre-rerun) plus
# lv_scaling_packed_20260630, which return 595 sims/s for k=100 async at 48 workers
# against the 6240 the paper reports -- the propagator changed in the rerun, so those
# roots answer a different question and silently disagree with every quoted number.
DATA = "/home/juhe/remotes/scratch/herold2/async-abc/rerun_20260707/scaling/data"
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_scaling_throughput.pdf"
# Multi-node points are FULLY-PACKED nodes, exact multiples of 48 (144=3, 192=4,
# 240=5, 288=6 nodes), so every node runs 48 ranks with no idle cores -- unlike the
# earlier 128/256 points (3/6 nodes with 16/32 idle cores) that under-packed the
# trailing node. One root now covers every worker count.
WORKERS = [1, 4, 16, 48, 144, 192, 240, 288]
DIR_FOR = {w: DATA for w in WORKERS}
ASYNC, SYNC = "async_propulate_abc", "abc_smc_baseline"


def _stats(w: int, method: str):
    f = os.path.join(DIR_FOR[w], f"throughput_summary_w{w}_k100.csv")
    df = pd.read_csv(f)
    s = df[df["base_method"] == method]["throughput_sims_per_s"].to_numpy()
    return np.median(s), np.percentile(s, 25), np.percentile(s, 75)


def main() -> None:
    a = np.array([_stats(w, ASYNC) for w in WORKERS])   # cols: med, q1, q3
    s = np.array([_stats(w, SYNC) for w in WORKERS])

    # Evenly-spaced categorical x (worker counts are irregular: 1/4/16/48 then full-node
    # multiples 144..288), so the multi-node points stay legible.
    x = np.arange(len(WORKERS))
    nodes = [max(1, w // 48) if w >= 48 else 0 for w in WORKERS]
    labels = [f"{w}" if w < 48 else f"{w}\n({w // 48}n)" for w in WORKERS]

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    # single-node -> multi-node boundary sits between 48 (1 node) and 144 (3 nodes)
    i48 = WORKERS.index(48)
    ax.axvline(i48 + 0.5, color="0.7", lw=1.0, ls="--", zorder=0)
    ax.annotate("single node $\\to$ multi-node (fully packed)", xy=(i48 + 0.5, 0),
                xytext=(i48 + 0.42, 30), fontsize=8.5, color="0.4", rotation=90,
                va="bottom", ha="center")

    for arr, color, mk, lab, mfc in [
        (a, "#1f77b4", "o", "asynchronous (ours)", "#1f77b4"),
        (s, "#d62728", "s", "synchronous baseline", "white"),
    ]:
        ax.fill_between(x, arr[:, 1], arr[:, 2], color=color, alpha=0.15, zorder=1)
        ax.plot(x, arr[:, 0], "-" if mk == "o" else "--", marker=mk, color=color,
                lw=2.0, ms=8 if mk == "o" else 7, mfc=mfc, label=lab, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("workers (n = full 48-core nodes)")
    ax.set_ylabel("throughput (simulations / s)")
    ax.set_title("Lotka--Volterra strong scaling ($k{=}100$)")
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("workers     :", WORKERS)
    print("async median:", [round(x, 0) for x in a[:, 0]])
    print("sync median :", [round(x, 0) for x in s[:, 0]])


if __name__ == "__main__":
    main()
