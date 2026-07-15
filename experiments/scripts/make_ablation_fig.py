#!/usr/bin/env python3
"""Ablation figure (fig_ablation.pdf) — component study on Gaussian-mean.

Two panels in one print-width row:

* **(a)** final Wasserstein-to-truth per ablation variant (bars, 95% CI),
  grouped as the *full method*, the two ingredient *removals* the claim names
  (no AMIS, hard kernel), and the hyperparameter *variants*.
* **(b)** the AMIS-isolation quality trajectory vs wall-clock time for the
  full method against its no-AMIS ablation, zoomed onto the converged regime
  so the persistent gap is legible.

Benchmark: Gaussian-mean (analytic posterior), 48 workers, 300 s budget,
k=100, five replicates, asynchronous method.

Default draws from the vendored CSVs; ``--refresh`` re-derives them from the
campaign output (``ablation/plots/ablation_{comparison,amis_isolation}_data.csv``).
Styling via async_abc.plotting.paper_style (Type-42, Okabe-Ito, print width).

Review fix (CVD, §5.2 / II.7.b.4): the previous bar chart encoded the three
variant classes by red/green/grey colour alone, which is colourblind- and
grayscale-unsafe. Classes are now an Okabe-Ito colour (async blue = full,
vermillion = ingredient removed, neutral grey = hyperparameter variant) with a
redundant hatch (solid / ``//`` / ``xx``) so the bars separate in grayscale.
The former log-scale inset in panel (b) is dropped: at print width its labels
would fall below 6 pt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import _figdata as fd
from async_abc.plotting import paper_style as ps

# display name, class: 'full' | 'removal' (ingredient removed) | 'variant' (hyperparameter)
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

# CVD-safe: Okabe-Ito colour + redundant hatch per class (grayscale-safe).
CLASS_COLOR = {
    "full": ps.COLORS["async"],      # blue  #0072B2
    "removal": ps.COLORS["sync"],    # vermillion #D55E00
    "variant": ps.COLORS["neutral"], # grey  #999999
}
CLASS_HATCH = {"full": "", "removal": "//", "variant": "xx"}
CLASS_LEGEND = {
    "full": "full method",
    "removal": "ingredient removed",
    "variant": "hyperparameter variant",
}
CLASS_ORDER = {"full": 0, "removal": 1, "variant": 2}

# Trajectories plotted in panel (b): the AMIS-isolation pair.
TRAJ = {
    "full_model": dict(color=ps.COLORS["async"], ls="-", marker="o", label="full method"),
    "no_amis": dict(color=ps.COLORS["sync"], ls="--", marker="s", label="no AMIS"),
}


def _grp(variant: str) -> str:
    return LABELS.get(variant, (variant, "variant"))[1]


def panel_comparison(ax, comp: pd.DataFrame) -> None:
    """Panel (a): final-quality bars per variant with 95% CI, class colour+hatch."""
    comp = comp.copy()
    comp["grp"] = comp["variant"].map(_grp)
    # full first, then removals, then variants; sorted by mean within a class.
    comp = comp.sort_values(
        by=["grp", "mean_final_wasserstein"],
        key=lambda s: s.map(CLASS_ORDER) if s.name == "grp" else s,
    ).reset_index(drop=True)

    x = np.arange(len(comp))
    means = comp["mean_final_wasserstein"].to_numpy()
    lo = comp["mean_final_wasserstein_ci_low"].to_numpy()
    hi = comp["mean_final_wasserstein_ci_high"].to_numpy()
    grps = comp["grp"].tolist()

    bars = ax.bar(
        x, means,
        color=[CLASS_COLOR[g] for g in grps],
        edgecolor=ps.COLORS["reference"], linewidth=0.5,
    )
    for patch, g in zip(bars.patches, grps):
        if CLASS_HATCH[g]:
            patch.set_hatch(CLASS_HATCH[g])
    ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="none",
                ecolor=ps.COLORS["reference"], elinewidth=0.8, capsize=2)

    full_y = float(comp.loc[comp["variant"] == "full_model", "mean_final_wasserstein"].iloc[0])
    ax.axhline(full_y, color=ps.COLORS["async"], ls=":", lw=1.0, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(v, (v, ""))[0] for v in comp["variant"]],
                       rotation=35, ha="right")
    ax.set_ylabel("final Wasserstein to truth")
    ax.set_ylim(0, max(hi) * 1.30)

    handles = [
        Patch(facecolor=CLASS_COLOR[c], edgecolor=ps.COLORS["reference"],
              hatch=CLASS_HATCH[c] or None, label=CLASS_LEGEND[c])
        for c in ("full", "removal", "variant")
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 0.93), handlelength=1.4)
    ps.panel_tag(ax, "(a)")


def panel_amis(ax, traj: pd.DataFrame) -> None:
    """Panel (b): AMIS-isolation trajectory, zoomed to the converged regime."""
    for v, st in TRAJ.items():
        s = traj[traj["variant"] == v].sort_values("wall_time")
        t = s["wall_time"].to_numpy()
        m = s["mean"].to_numpy()
        sd = s["std"].fillna(0).to_numpy()
        ax.plot(t, m, color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=2.5, lw=1.4, label=st["label"])
        ax.fill_between(t, m - sd, m + sd, color=st["color"], alpha=0.15, lw=0)

    ax.set_xlim(8, 305)
    ax.set_ylim(0.055, 0.100)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("Wasserstein to truth")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8)
    ps.panel_tag(ax, "(b)")


def draw(frames: dict) -> "plt.Figure":
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=ps.fig_size(1.0, aspect=0.42),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    panel_comparison(axL, frames["ablation_comparison"])
    panel_amis(axR, frames["ablation_amis"])
    fig.tight_layout()
    return fig


def aggregate(root: Path) -> dict:
    """Return exactly the small frames plotted in each panel."""
    plots = root / "ablation" / "plots"
    comp = pd.read_csv(plots / "ablation_comparison_data.csv")[
        ["variant", "mean_final_wasserstein",
         "mean_final_wasserstein_ci_low", "mean_final_wasserstein_ci_high"]
    ]
    amis = pd.read_csv(plots / "ablation_amis_isolation_data.csv")
    # panel (b) plots only the AMIS-isolation pair; the wall_time~0 prior
    # transient (mean~4) lies outside the zoomed window, so drop it too —
    # the vendored frame is exactly what is drawn.
    amis = amis[amis["variant"].isin(TRAJ) & (amis["wall_time"] > 1.0)][
        ["wall_time", "mean", "std", "variant"]
    ].reset_index(drop=True)
    return {"ablation_comparison": comp, "ablation_amis": amis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fd.add_refresh_arg(parser)
    args = parser.parse_args()
    ps.apply()

    if args.refresh is not None:
        frames = aggregate(Path(args.refresh))
        vendor = frames
    else:
        frames = fd.load_vendored("fig_ablation")
        vendor = None

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, "fig_ablation", data=vendor)
    print(f"wrote {saved['pdf']}")
    if vendor is not None:
        for k, v in saved.items():
            if k.startswith("csv:"):
                print(f"  vendored {v}")


if __name__ == "__main__":
    main()
