#!/usr/bin/env python3
"""Replot the ablation figure with legible labels and a visible AMIS effect.

Addresses review round 2 (Fig. 10): the previous panels had tiny labels and the
AMIS-isolation curve was dominated by the initial prior-to-posterior transient, hiding
the converged gap. Here the left panel groups the two ingredient *removals* the claim
names (no AMIS, hard kernel) against the full method and the remaining hyperparameter
variants; the right panel zooms onto the converged regime so the persistent full-method
< no-AMIS gap is visible, with an inset retaining the full convergence transient.

Benchmark: Gaussian-mean (analytic posterior), 48 workers, 300 s budget, k=100, five
replicates, asynchronous method. Reads the summary CSVs emitted by the original run
(ablation_comparison_data.csv, ablation_amis_isolation_data.csv); no re-derivation.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = "/home/juhe/remotes/scratch/herold2/async-abc/run_full_20260626_1816/ablation/plots"
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_ablation.pdf"

# display name, group: 'full' | 'removal' (ingredient removed) | 'variant' (hyperparameter)
LABELS = {
    "full_model": ("full method", "full"),
    "no_amis": ("no AMIS", "removal"),
    "hard_kernel_baseline": ("hard kernel", "removal"),
    "epanechnikov_kernel": ("Epanechnikov", "variant"),
    "small_archive": ("small archive", "variant"),
    "slow_decay": ("slow decay", "variant"),
    "large_perturbation": ("large perturb.", "variant"),
    "fixed_perturbation": ("fixed perturb.", "variant"),
    "no_archive_truncation": ("no arch. trunc.", "variant"),
}
GROUP_COLOR = {"full": "#2ca02c", "removal": "#d62728", "variant": "#9e9e9e"}


def main() -> None:
    comp = pd.read_csv(f"{P}/ablation_comparison_data.csv")
    traj = pd.read_csv(f"{P}/ablation_amis_isolation_data.csv")

    # order: full first, then the two ingredient removals, then variants
    order_rank = {"full": 0, "removal": 1, "variant": 2}
    comp["grp"] = comp["variant"].map(lambda v: LABELS.get(v, (v, "variant"))[1])
    comp = comp.sort_values(by=["grp", "mean_final_wasserstein"],
                            key=lambda s: s.map(order_rank) if s.name == "grp" else s)

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
                         "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- left: bar chart, color-grouped ----
    x = np.arange(len(comp))
    means = comp["mean_final_wasserstein"].to_numpy()
    lo = comp["mean_final_wasserstein_ci_low"].to_numpy()
    hi = comp["mean_final_wasserstein_ci_high"].to_numpy()
    colors = [GROUP_COLOR[g] for g in comp["grp"]]
    axL.bar(x, means, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
    axL.errorbar(x, means, yerr=[means - lo, hi - means], fmt="none",
                 ecolor="0.25", elinewidth=1.2, capsize=3)
    full_y = float(comp.loc[comp["variant"] == "full_model", "mean_final_wasserstein"].iloc[0])
    axL.axhline(full_y, color="#2ca02c", ls=":", lw=1.2, zorder=0)
    axL.set_xticks(x)
    axL.set_xticklabels([LABELS.get(v, (v, ""))[0] for v in comp["variant"]],
                        rotation=35, ha="right")
    axL.set_ylabel("final Wasserstein to truth")
    axL.set_title("(a) full method vs. component changes")
    # Extra top headroom so the error bars sit clear of the legend.
    axL.set_ylim(0, max(hi) * 1.30)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ["full", "removal", "variant"]]
    axL.legend(handles, ["full method", "ingredient removed", "hyperparameter variant"],
               frameon=False, loc="upper left", fontsize=10)

    # ---- right: AMIS isolation, zoomed to converged regime + inset transient ----
    def _curve(v):
        s = traj[traj["variant"] == v].sort_values("wall_time")
        return s["wall_time"].to_numpy(), s["mean"].to_numpy(), s["std"].fillna(0).to_numpy()

    style = {"full_model": ("#2ca02c", "full method", "-"),
             "no_amis": ("#d62728", "no AMIS", "--")}
    for v, (c, lab, ls) in style.items():
        t, m, sd = _curve(v)
        axR.plot(t, m, ls, color=c, lw=2.0, label=lab)
        axR.fill_between(t, m - sd, m + sd, color=c, alpha=0.15)
    # Extra top headroom leaves an empty band above the two converged curves
    # for the inset, so the inset never lies on top of the data.
    axR.set_ylim(0.055, 0.104)
    axR.set_xlim(8, 305)
    axR.set_xlabel("wall-clock time (s)")
    axR.set_ylabel("Wasserstein to truth")
    axR.set_title("(b) AMIS isolation (converged regime)")
    axR.grid(True, ls=":", lw=0.5, alpha=0.6)
    axR.legend(frameon=False, loc="lower left")

    # inset: full transient from the prior, placed in the empty top band so it
    # sits clear of both converged curves and the legend.
    axin = axR.inset_axes([0.40, 0.53, 0.57, 0.35])
    for v, (c, lab, ls) in style.items():
        t, m, sd = _curve(v)
        axin.plot(t, m, ls, color=c, lw=1.3)
    axin.set_yscale("log")
    axin.set_title("full trajectory (log)", fontsize=9)
    axin.tick_params(labelsize=8)
    axin.set_xlabel("s", fontsize=8, labelpad=0)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(comp[["variant", "mean_final_wasserstein", "grp"]].to_string(index=False))
    for v in style:
        s = traj[(traj["variant"] == v) & (traj["wall_time"] > 40)]
        print(f"{v:>12} converged mean Wass = {s['mean'].mean():.4f}")


if __name__ == "__main__":
    main()
