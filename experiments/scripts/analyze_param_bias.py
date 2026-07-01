#!/usr/bin/env python3
"""Quantify + plot the every-particle bias from the parameter-coupled-delay run (WS4).

Benchmark: Gaussian mean (true mu=0, prior [-5,5]). The simulator's post-evaluation
delay is coupled to the inferred parameter -- delay = base + coupling that grows with
|mu - center|, capped at delay_cap -- so one region of parameter space is evaluated far
more often than its mirror image. The coupling strength sweeps sigma in {0, 0.5, 1, 2}
(0 = uniform-runtime reference). Five replicates per cell, async vs.\ the discard-
latecomers synchronous baseline.

Reads runtime_performance_summary.csv (the non-shard runner does NOT emit
gaussian_analytic_summary.csv). Two-panel figure: (left) aggregate throughput vs.\
coupling -- confirms the gradient bites (~40% drop); (right) posterior error (Wasserstein
to the analytic Gaussian posterior) vs.\ coupling -- async stays flat and comparable to
sync, i.e.\ the AMIS importance weights compensate for the raw over-representation.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/juhe/remotes/scratch/herold2/async-abc/parambias2_20260629/"
    "runtime_heterogeneity/data/runtime_performance_summary.csv"
)
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_param_bias.pdf"

STYLE = {
    "async_propulate_abc": dict(label="asynchronous (ours)", color="#1f77b4", marker="o", mfc="#1f77b4", ls="-"),
    "abc_smc_baseline":    dict(label="synchronous baseline", color="#d62728", marker="s", mfc="white", ls="--"),
}
ORDER = ["async_propulate_abc", "abc_smc_baseline"]


def _agg(df: pd.DataFrame):
    return (df.groupby(["base_method", "sigma"])
            .agg(W=("final_quality_wasserstein", "mean"),
                 W_sd=("final_quality_wasserstein", "std"),
                 thr=("throughput_sims_per_s", "mean"),
                 thr_sd=("throughput_sims_per_s", "std"),
                 n=("replicate", "count"))
            .reset_index())


def main() -> None:
    df = pd.read_csv(CSV)
    g = _agg(df)

    print(f"{'method':>22} {'coupl':>6} {'W':>8} {'(sd)':>7} {'thr':>8} {'(sd)':>8} {'n':>3}")
    for _, r in g.sort_values(["base_method", "sigma"]).iterrows():
        print(f"{r['base_method']:>22} {r['sigma']:>6.1f} {r['W']:>8.4f} {r['W_sd']:>7.4f} "
              f"{r['thr']:>8.1f} {r['thr_sd']:>8.1f} {int(r['n']):>3}")

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, (axT, axW) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    for method in ORDER:
        sub = g[g["base_method"] == method].sort_values("sigma")
        st = STYLE[method]
        axT.errorbar(sub["sigma"], sub["thr"], yerr=sub["thr_sd"], capsize=3,
                     marker=st["marker"], color=st["color"], mfc=st["mfc"], ls=st["ls"],
                     lw=1.8, ms=7, label=st["label"])
        axW.errorbar(sub["sigma"], sub["W"], yerr=sub["W_sd"], capsize=3,
                     marker=st["marker"], color=st["color"], mfc=st["mfc"], ls=st["ls"],
                     lw=1.8, ms=7, label=st["label"])

    axT.set_xlabel("runtime$\\to$parameter coupling strength")
    axT.set_ylabel("throughput (simulations / s)")
    axT.set_title("(a) coupling bites: throughput falls")
    axT.set_ylim(bottom=0)
    axT.grid(True, ls=":", lw=0.5, alpha=0.6)
    axT.legend(frameon=False, loc="lower left")

    axW.set_xlabel("runtime$\\to$parameter coupling strength")
    axW.set_ylabel("posterior error (Wasserstein)")
    axW.set_title("(b) posterior error stays flat")
    axW.set_ylim(bottom=0)
    axW.grid(True, ls=":", lw=0.5, alpha=0.6)
    axW.legend(frameon=False, loc="lower left")

    for ax in (axT, axW):
        ax.set_xticks([0.0, 0.5, 1.0, 2.0])

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
