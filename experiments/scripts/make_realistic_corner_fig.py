#!/usr/bin/env python3
"""Cellular Potts posterior corner (fig_cpm_corner.pdf).

Dedicated replot so the CPM posterior corner follows the same house style as the other
figures: async = blue, sync = red, rejection = green (the reporter default drew async in
orange, which did not match the rest of the paper), with readable legend labels instead
of the raw method keys. Reference values are drawn as dashed lines.

Note on sample counts: the async posterior in this run is represented by only n=14 points
against n=500 for the sync/rejection baselines (shown in the legend). The async cloud is
therefore sparse but tightly concentrated on motility near its reference; the division
rate is weakly identified for every method. Reads corner_data.csv; no re-derivation.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

CSV = ("/home/juhe/remotes/scratch/herold2/async-abc/run_full_20260626_1816/"
       "cellular_potts/plots/corner_data.csv")
# Raw per-particle records, used to reconstruct the async posterior from its actual
# reported estimator (AMIS posterior_weight) instead of the hard tolerance cut that the
# generic final-state extractor applies (that cut leaves only n=14 async points because
# async drove the tolerance far lower than the sync baseline).
RAW = ("/home/juhe/remotes/scratch/herold2/async-abc/run_full_20260626_1816/"
       "cellular_potts/data/raw_results.csv")
N_ASYNC_RESAMPLE = 500          # match the sync/rejection final-population size
RESAMPLE_SEED = 20260701
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_cpm_corner.pdf"

REF = {"division_rate": 0.049905, "motility": 0.2}
STYLE = {
    "async_propulate_abc": dict(label="Asynchronous (ours)", color="#1f77b4"),
    "abc_smc_baseline":    dict(label="Synchronous baseline", color="#d62728"),
    "rejection_abc":       dict(label="Rejection ABC",        color="#2ca02c"),
}
ORDER = ["async_propulate_abc", "abc_smc_baseline", "rejection_abc"]
GRID = np.linspace(0.0, 1.0, 256)


def _kde(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return None
    try:
        return gaussian_kde(x)(GRID)
    except np.linalg.LinAlgError:
        return None


def _async_amis_posterior():
    """Resample the async posterior from its AMIS posterior_weight over the full history."""
    raw = pd.read_csv(RAW)
    a = raw[raw["method"] == "async_propulate_abc"].copy()
    w = a["posterior_weight"].to_numpy(float)
    ok = np.isfinite(w) & (w > 0)
    a, w = a[ok], w[ok]
    w = w / w.sum()
    idx = np.random.default_rng(RESAMPLE_SEED).choice(len(a), size=N_ASYNC_RESAMPLE, p=w)
    s = a.iloc[idx]
    return pd.DataFrame({"method": "async_propulate_abc",
                         "division_rate": s["param_division_rate"].to_numpy(),
                         "motility": s["param_motility"].to_numpy()})


def main() -> None:
    df = pd.read_csv(CSV)
    # Replace the hard-cut async points (n=14) with a draw from the AMIS posterior.
    df = pd.concat([df[df["method"] != "async_propulate_abc"], _async_amis_posterior()],
                   ignore_index=True)
    counts = {m: int((df["method"] == m).sum()) for m in ORDER}

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.4),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.08})
    ax_d, ax_leg = axes[0]
    ax_j, ax_m = axes[1]

    for m in ORDER:
        sub = df[df["method"] == m]
        c = STYLE[m]["color"]
        # division-rate marginal (top-left)
        k = _kde(sub["division_rate"])
        if k is not None:
            ax_d.plot(GRID, k, color=c, lw=1.6)
            ax_d.fill_between(GRID, k, color=c, alpha=0.12, linewidth=0)
        # motility marginal (bottom-right)
        k = _kde(sub["motility"])
        if k is not None:
            ax_m.plot(GRID, k, color=c, lw=1.6)
            ax_m.fill_between(GRID, k, color=c, alpha=0.12, linewidth=0)
        # joint scatter (bottom-left). Async is sparse, so draw it last and larger.
        big = m == "async_propulate_abc"
        ax_j.scatter(sub["division_rate"], sub["motility"], s=26 if big else 9,
                     color=c, alpha=0.85 if big else 0.35,
                     edgecolor="white" if big else "none", linewidth=0.4, zorder=3 if big else 2)

    # reference lines
    ax_d.axvline(REF["division_rate"], color="0.35", ls="--", lw=1.0)
    ax_m.axvline(REF["motility"], color="0.35", ls="--", lw=1.0)
    ax_j.axvline(REF["division_rate"], color="0.35", ls="--", lw=1.0)
    ax_j.axhline(REF["motility"], color="0.35", ls="--", lw=1.0)

    for ax in (ax_d, ax_j, ax_m):
        ax.set_xlim(0, 1)
    ax_j.set_ylim(0, 1)
    ax_d.set_xticklabels([])
    ax_d.set_yticks([])
    ax_m.set_yticks([])
    ax_j.set_xlabel("division rate")
    ax_j.set_ylabel("motility")
    ax_m.set_xlabel("motility")

    # legend (top-right cell), with honest per-method sample counts
    ax_leg.axis("off")
    handles = [plt.Line2D([0], [0], color=STYLE[m]["color"], lw=2.4,
                          label=f"{STYLE[m]['label']}  (n={counts[m]})") for m in ORDER]
    handles.append(plt.Line2D([0], [0], color="0.35", ls="--", lw=1.0, label="reference value"))
    ax_leg.legend(handles=handles, loc="center", frameon=False, borderaxespad=0.0)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("sample counts:", counts)


if __name__ == "__main__":
    main()
