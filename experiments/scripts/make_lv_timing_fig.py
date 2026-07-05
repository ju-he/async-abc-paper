#!/usr/bin/env python3
"""Decompose asynchronous worker time on Lotka--Volterra strong scaling (fig_lv_timing.pdf).

Backs the \\S7.2 claim that a near-instantaneous simulator leaves per-arrival
coordination dominant: for each worker count we split aggregate worker wall-time into
(i) simulation, (ii) per-arrival proposal/AMIS reconstruction, and (iii) the residual
coordination + idle (MPI exchange, serialization, waiting). The split is measured, not
inferred -- simulation time from the per-attempt event spans in raw_results, proposal
time from the env-gated phase-timing log (ASYNC_ABC_PHASE_TIMING=1), and coordination
as the residual wall - sim - proposal.

Data: dedicated instrumented async-only run scaling_timing_20260629 (k=100, 300 s,
3 replicates), worker counts 16/48/128/256 = 1/1/3/6 nodes (48 cores/node). The
productive (simulation) fraction collapses from ~9% to <1% as the job scales, and the
coordination residual crosses 95% once the job spans multiple nodes -- a property of
coordinating a near-instantaneous simulator, not of the algorithm.
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = "/home/juhe/remotes/scratch/herold2/async-abc/scaling_timing_20260629/scaling/data"
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_lv_timing.pdf"
WORKERS = [16, 48, 128, 256]
NODES = {16: 1, 48: 1, 128: 3, 256: 6}  # 48 cores / node


def _decompose(w: int):
    """Aggregate (sim, proposal, coordination+idle) fractions of worker wall-time."""
    pt = pd.concat(
        [pd.read_csv(f) for f in glob.glob(f"{DATA}/phase_timing_w{w}_k100_rep*_rank*.csv")],
        ignore_index=True,
    )
    rr = pd.read_csv(
        f"{DATA}/raw_results_w{w}_k100.csv",
        usecols=["sim_start_time", "sim_end_time"],
    )
    sim_s = float((rr["sim_end_time"] - rr["sim_start_time"]).sum())
    prop_s = float(pt["proposal_s"].sum())
    wall_s = float(pt["wall_s"].sum())
    coord_s = wall_s - sim_s - prop_s
    if coord_s < 0:
        raise ValueError(f"w{w}: negative coordination residual ({coord_s:.1f}s)")
    return np.array([sim_s, prop_s, coord_s]) / wall_s


def main() -> None:
    frac = np.array([_decompose(w) for w in WORKERS])  # rows: sim, prop, coord

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 12})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    x = np.arange(len(WORKERS))
    labels = ["simulation", "proposal / AMIS reconstruction", "coordination + idle (MPI, wait)"]
    colors = ["#2ca02c", "#1f77b4", "#bcbcbc"]
    bottom = np.zeros(len(WORKERS))
    for j, (lab, c) in enumerate(zip(labels, colors)):
        ax.bar(x, 100 * frac[:, j], bottom=100 * bottom, width=0.62, color=c,
               label=lab, edgecolor="white", linewidth=0.6)
        bottom += frac[:, j]

    # annotate the productive (simulation) fraction on each bar
    for i, f in enumerate(frac[:, 0]):
        ax.annotate(f"{100 * f:.1f}%", xy=(x[i], 100 * f), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color="#1a661a", fontweight="bold")

    # single-node -> multi-node boundary sits between 48 (1 node) and 128 (3 nodes)
    ax.axvline(1.5, color="0.45", lw=1.0, ls="--", zorder=0)
    ax.annotate("single node $\\to$ multi-node", xy=(1.5, 50), xytext=(1.62, 50),
                fontsize=10, color="0.35", rotation=90, va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n({NODES[w]} node{'s' if NODES[w] > 1 else ''})" for w in WORKERS])
    ax.set_xlabel("workers (48 cores / node)")
    ax.set_ylabel("share of worker wall-time (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Lotka--Volterra: where asynchronous worker time goes ($k{=}100$)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="0.8",
              loc="upper left", bbox_to_anchor=(0.015, 0.985))
    ax.margins(x=0.04)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"{'w':>4} {'nodes':>5} {'sim%':>6} {'prop%':>6} {'coord%':>7}")
    for i, w in enumerate(WORKERS):
        print(f"{w:>4} {NODES[w]:>5} {100*frac[i,0]:>5.1f} {100*frac[i,1]:>5.1f} {100*frac[i,2]:>6.1f}")


if __name__ == "__main__":
    main()
