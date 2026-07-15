#!/usr/bin/env python3
"""Where asynchronous worker time goes on Lotka--Volterra scaling (fig_lv_timing.pdf).

Backs the \\S7.2 claim that a near-instantaneous simulator leaves per-arrival
coordination dominant: for each worker count the aggregate worker wall-time is
split into (i) productive simulation and (ii) coordination + idle (MPI exchange,
per-arrival proposal/AMIS reconstruction, serialization, waiting). The productive
fraction is the measured ``worker_utilization`` (share of worker wall-clock spent
inside the simulator); the remainder is coordination + idle.

Data: the instrumented async-only Lotka--Volterra timing sweep
(``_scaling_timing``; k=100, worker counts 16/48/128/256 = 1/1/3/6 nodes at
48 cores/node). The productive fraction collapses from ~55% to ~2% as the job
scales and coordination overwhelmingly dominates once the job spans multiple
nodes -- a property of coordinating a near-instantaneous simulator, not of the
algorithm.

Note: this rerun's timing run did not emit the per-arrival phase-timing log, so
the previous three-way (simulation / proposal / coordination) split is reduced to
this two-way productive-vs-coordination split; the collapse of the productive
fraction is unchanged. Default draws from the vendored CSV; ``--refresh``
re-derives it. Styling via async_abc.plotting.paper_style.
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

FIG = "fig_lv_timing"
WORKERS = [16, 48, 128, 256]
NODES = {16: 1, 48: 1, 128: 3, 256: 6}  # 48 cores / node


def aggregate(root: Path):
    data = root / "_scaling_timing" / "scaling" / "data"
    rows = []
    for w in WORKERS:
        df = pd.read_csv(data / f"throughput_summary_w{w}_k100.csv")
        a = df[df["base_method"] == "async_propulate_abc"]
        productive = float(a["worker_utilization"].mean())
        rows.append(dict(n_workers=w, nodes=NODES[w],
                         productive_fraction=productive,
                         coordination_fraction=1.0 - productive))
    return {"lv_timing": pd.DataFrame(rows)}


def draw(frames):
    df = frames["lv_timing"].set_index("n_workers").reindex(WORKERS)
    x = np.arange(len(WORKERS))
    prod = df["productive_fraction"].to_numpy() * 100
    coord = df["coordination_fraction"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=ps.fig_size(0.66, aspect=0.72))
    ax.bar(x, prod, width=0.62, color=ps.COLORS["async"], edgecolor="white",
           linewidth=0.6, label="productive simulation")
    ax.bar(x, coord, width=0.62, bottom=prod, color=ps.COLORS["neutral"],
           edgecolor="white", linewidth=0.6, hatch="//",
           label="coordination + idle (MPI, proposal, wait)")
    for i, p in enumerate(prod):
        ax.annotate(f"{p:.1f}%", xy=(x[i], p), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, color=ps.COLORS["async"], fontweight="bold")

    # single-node -> multi-node boundary sits between 48 (1 node) and 128 (3 nodes)
    ax.axvline(1.5, color=ps.COLORS["reference"], lw=0.8, ls="--", zorder=5)
    ax.annotate("single node $\\to$ multi-node", xy=(1.5, 52), xytext=(1.6, 52),
                fontsize=6, color=ps.COLORS["reference"], rotation=90,
                va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n({NODES[w]} node{'s' if NODES[w] > 1 else ''})" for w in WORKERS])
    ax.set_xlabel("workers (48 cores / node)")
    ax.set_ylabel("share of worker wall-time (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.8",
              loc="upper right", fontsize=6)
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
        frames = fd.load_vendored(FIG)
        vendor = None

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, FIG, data=vendor)
    print(f"wrote {saved['pdf']}")
    for _, r in frames["lv_timing"].iterrows():
        print(f"  w={int(r['n_workers']):>3} ({int(r['nodes'])} node): "
              f"productive {100*r['productive_fraction']:.1f}%")


if __name__ == "__main__":
    main()
