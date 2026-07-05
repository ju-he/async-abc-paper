#!/usr/bin/env python3
"""Gaussian-mean posterior recovery vs. wall-clock time (fig_gaussian_recovery.pdf).

Dedicated replot so the figure follows the same house style as the parameter-bias
and strong-scaling figures (async = blue solid, sync = red dashed), fixes the color
mismatch of the reporter default (which drew async in orange, sync in blue), and moves
the full-range inset into the empty top band so it no longer lies on top of the
converging async/sync curves. Reads the quality-vs-wall-time summary CSV emitted by the
validated Gaussian rerun; no re-derivation.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = ("/home/juhe/remotes/scratch/herold2/async-abc/gaussian_rerun_20260628/"
       "gaussian_mean/plots/quality_vs_wall_time_data.csv")
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_gaussian_recovery.pdf"

STYLE = {
    "async_propulate_abc": dict(label="Asynchronous (ours)", color="#1f77b4", marker="o", mfc="#1f77b4", ls="-"),
    "abc_smc_baseline":    dict(label="Synchronous baseline", color="#d62728", marker="s", mfc="white", ls="--"),
}
ORDER = ["async_propulate_abc", "abc_smc_baseline"]
REJ_COLOR = "#7f7f7f"


def _curve(df: pd.DataFrame, method: str):
    s = df[df["method"] == method].sort_values("axis_value")
    return (s["axis_value"].to_numpy(), s["wasserstein"].to_numpy(),
            s["wasserstein_ci_low"].to_numpy(), s["wasserstein_ci_high"].to_numpy())


def main() -> None:
    df = pd.read_csv(CSV)
    rej = df[df["method"] == "rejection_abc"].sort_values("axis_value")["wasserstein"]
    # Report the rejection floor at the full budget (matches the caption's W ~ 2.4).
    rej_w = float(rej.iloc[-1]) if len(rej) else None

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for m in ORDER:
        t, w, lo, hi = _curve(df, m)
        st = STYLE[m]
        ax.plot(t, w, marker=st["marker"], color=st["color"], mfc=st["mfc"], ls=st["ls"],
                lw=1.8, ms=7, label=st["label"])
        ax.fill_between(t, lo, hi, color=st["color"], alpha=0.15, linewidth=0)

    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("Wasserstein distance to analytic posterior")
    ax.set_ylim(0, 0.16)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    # Legend above the axes so it never overlaps the curves or the inset.
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, borderaxespad=0.0)

    # Inset in the empty top band: the full range including the rejection-ABC floor,
    # placed clear of the async/sync curves and their bands (which top out near 0.09).
    if rej_w is not None:
        axin = ax.inset_axes([0.44, 0.60, 0.53, 0.36])
        for m in ORDER:
            t, w, _, _ = _curve(df, m)
            st = STYLE[m]
            axin.plot(t, w, color=st["color"], ls=st["ls"], lw=1.4)
        axin.axhline(rej_w, color=REJ_COLOR, ls="-", lw=1.6)
        xr = axin.get_xlim()
        axin.text(xr[0] + 0.5 * (xr[1] - xr[0]), rej_w, f"rejection ABC  $W\\approx{rej_w:.1f}$",
                  color=REJ_COLOR, fontsize=8, va="bottom", ha="center")
        axin.set_ylim(0, rej_w * 1.18)
        axin.set_title("full range", fontsize=9)
        axin.tick_params(labelsize=8)
        axin.set_xlabel("wall-clock time (s)", fontsize=8, labelpad=1)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    for m in ORDER + ["rejection_abc"]:
        s = df[df["method"] == m]
        if len(s):
            print(f"{m:>22}: final W = {s.sort_values('axis_value')['wasserstein'].iloc[-1]:.4f}  "
                  f"({int(s['n_replicates'].max())} reps)")


if __name__ == "__main__":
    main()
