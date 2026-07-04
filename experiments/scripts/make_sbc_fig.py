#!/usr/bin/env python3
"""Simulation-based calibration figures (fig_sbc_coverage.pdf, fig_sbc_rank.pdf).

Dedicated replot so the SBC panels use readable legend labels (Asynchronous / Synchronous
baseline) instead of the raw method keys, and the house colour convention (async = blue,
sync = red). Left: empirical vs nominal coverage with the perfect-calibration diagonal.
Right: rank histogram with the uniform expectation and a 99% band. Reads coverage.csv and
sbc_ranks.csv from the run_full SBC experiment (Gaussian-mean, mu, 100 trials); no
re-derivation.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

DATA = "/home/juhe/remotes/scratch/herold2/async-abc/run_full_20260626_1816/sbc/data"
FIGDIR = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures"

STYLE = {
    "async_propulate_abc": dict(label="Asynchronous (ours)", color="#1f77b4", marker="o", mfc="#1f77b4", ls="-"),
    "abc_smc_baseline":    dict(label="Synchronous baseline", color="#d62728", marker="s", mfc="white", ls="--"),
}
ORDER = ["async_propulate_abc", "abc_smc_baseline"]


def coverage_fig() -> None:
    df = pd.read_csv(f"{DATA}/coverage.csv")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color="0.4", ls=":", lw=1.2, label="perfect calibration")
    for m in ORDER:
        sub = df[df["method"] == m].sort_values("coverage_level")
        st = STYLE[m]
        ax.plot(sub["coverage_level"], sub["empirical_coverage"], marker=st["marker"],
                color=st["color"], mfc=st["mfc"], ls=st["ls"], lw=1.8, ms=8, label=st["label"])
    ax.set_xlabel("nominal coverage level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.35, 1.0)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_sbc_coverage.pdf", bbox_inches="tight")
    print(f"wrote {FIGDIR}/fig_sbc_coverage.pdf")


def rank_fig() -> None:
    df = pd.read_csv(f"{DATA}/sbc_ranks.csv")
    n_bins = 10
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    # uniform expectation + 99% band (binomial), computed from the async trial count
    n_trials = int(df[df["method"] == ORDER[0]].shape[0])
    exp = n_trials / n_bins
    lo, hi = binom.ppf(0.005, n_trials, 1 / n_bins), binom.ppf(0.995, n_trials, 1 / n_bins)
    ax.axhspan(lo, hi, color="0.85", zorder=0, label="99% uniform band")
    ax.axhline(exp, color="0.4", ls=":", lw=1.2)

    edges = np.linspace(0, 100, n_bins + 1)
    for m in ORDER:
        ranks = df[df["method"] == m]["rank"].to_numpy()
        counts, _ = np.histogram(ranks, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(np.r_[edges[0], centers, edges[-1]], np.r_[counts[0], counts, counts[-1]],
                where="mid", color=STYLE[m]["color"], lw=1.9, label=STYLE[m]["label"])
    ax.set_xlabel("rank statistic")
    ax.set_ylabel("count (100 trials)")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    # Legend above the axes so it never overlaps the tallest bins.
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_sbc_rank.pdf", bbox_inches="tight")
    print(f"wrote {FIGDIR}/fig_sbc_rank.pdf")


if __name__ == "__main__":
    coverage_fig()
    rank_fig()
