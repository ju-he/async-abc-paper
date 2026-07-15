#!/usr/bin/env python3
"""Cellular Potts posterior corner (fig_cpm_corner.pdf).

Joint distribution of the two Cellular-Potts parameters (division rate, motility)
for each method, plus their two marginals. The joint shows a broad, diagonal
weak-identifiability ridge: motility is fairly well constrained near its truth
while the division rate is only weakly identified for every method.

Data (only touched with ``--refresh``):

* ``cellular_potts/plots/corner_data.csv`` supplies the synchronous and rejection
  posteriors (n=500 each).
* ``cellular_potts/data/raw_results.csv`` supplies the asynchronous posterior:
  ``corner_data.csv`` carries no async rows because the generic final-state
  extractor hard-cuts on the tolerance and async drove the tolerance far below
  the sync baseline (leaving only a handful of points). Instead we resample the
  async posterior from its actual AMIS ``posterior_weight`` estimator over the
  full history, drawing :data:`N_ASYNC_RESAMPLE` points to match the baselines.

The vendored ``corner_samples.csv`` is exactly the plotted point cloud (method,
division_rate, motility) for all three methods, so the default path redraws
without touching the 57 MB raw file.

Styling via async_abc.plotting.paper_style (Type-42, Okabe-Ito, print width).
Review §5.2 (CVD): the synchronous (vermillion) and rejection (green) methods are
a red/green pair, so they are additionally separated by marker shape (joint
scatter), contour linestyle (joint density) and line style (marginals) — legible
in grayscale and for colorblind readers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import _figdata as fd
from async_abc.plotting import paper_style as ps

ORDER = ["async_propulate_abc", "abc_smc_baseline", "rejection_abc"]
KEY = {
    "async_propulate_abc": "async",
    "abc_smc_baseline": "sync",
    "rejection_abc": "rejection",
}
# contour() wants named linestyles, not the "-"/"--"/":" plot() aliases.
_LS_NAME = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot"}

REF = {"division_rate": 0.049905, "motility": 0.2}
N_ASYNC_RESAMPLE = 500          # match the sync/rejection final-population size
RESAMPLE_SEED = 20260701
GRID = np.linspace(0.0, 1.0, 256)          # 1-D marginal grid
_G2 = np.linspace(0.0, 1.0, 120)           # 2-D joint-density grid


def _async_amis_posterior(raw: pd.DataFrame) -> pd.DataFrame:
    """Resample the async posterior from its AMIS posterior_weight over the full history."""
    a = raw[raw["method"] == "async_propulate_abc"]
    w = a["posterior_weight"].to_numpy(float)
    ok = np.isfinite(w) & (w > 0)
    a, w = a[ok], w[ok]
    w = w / w.sum()
    idx = np.random.default_rng(RESAMPLE_SEED).choice(len(a), size=N_ASYNC_RESAMPLE, p=w)
    s = a.iloc[idx]
    return pd.DataFrame({
        "method": "async_propulate_abc",
        "division_rate": s["param_division_rate"].to_numpy(),
        "motility": s["param_motility"].to_numpy(),
    })


def aggregate(root: Path):
    """Return the plotted point cloud: method, division_rate, motility (n=500 each)."""
    cp = root / "cellular_potts"
    df = pd.read_csv(cp / "plots" / "corner_data.csv")   # sync + rejection posteriors
    df = df[df["method"] != "async_propulate_abc"]       # drop any hard-cut async points
    raw = pd.read_csv(cp / "data" / "raw_results.csv")   # async lives only here
    df = pd.concat([df, _async_amis_posterior(raw)], ignore_index=True)
    df = df[["method", "division_rate", "motility"]]
    df["method"] = pd.Categorical(df["method"], categories=ORDER, ordered=True)
    return {"corner_samples": df.sort_values("method").reset_index(drop=True)}


def _kde1(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return None
    try:
        return gaussian_kde(x)(GRID)
    except np.linalg.LinAlgError:
        return None


def _kde2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.ptp(x) < 1e-9 or np.ptp(y) < 1e-9:
        return None
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return None
    xx, yy = np.meshgrid(_G2, _G2)
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, z


def draw(frames):
    df = frames["corner_samples"]
    counts = {m: int((df["method"] == m).sum()) for m in ORDER}

    fig, axes = plt.subplots(2, 2, figsize=ps.fig_size(0.62, aspect=0.9),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.08})
    ax_d, ax_leg = axes[0]
    ax_j, ax_m = axes[1]

    for m in ORDER:
        sub = df[df["method"] == m]
        k = KEY[m]
        color = ps.COLORS[k]
        ls = ps.LINESTYLES[k]
        marker = ps.MARKERS[k]

        # division-rate marginal (top-left) — distinct linestyle per method.
        yk = _kde1(sub["division_rate"])
        if yk is not None:
            ax_d.plot(GRID, yk, color=color, ls=ls, lw=1.3)
            ax_d.fill_between(GRID, yk, color=color, alpha=0.10, linewidth=0)
        # motility marginal (bottom-right).
        yk = _kde1(sub["motility"])
        if yk is not None:
            ax_m.plot(GRID, yk, color=color, ls=ls, lw=1.3)
            ax_m.fill_between(GRID, yk, color=color, alpha=0.10, linewidth=0)

        # joint (bottom-left): scatter with a distinct MARKER, then a KDE contour
        # with a distinct LINESTYLE — two redundant, color-independent encodings.
        ax_j.scatter(sub["division_rate"], sub["motility"], s=7, marker=marker,
                     color=color, alpha=0.30, edgecolor="none", zorder=2)
        kde2 = _kde2(sub["division_rate"], sub["motility"])
        if kde2 is not None:
            xx, yy, z = kde2
            ax_j.contour(xx, yy, z, levels=np.array([0.35, 0.7]) * z.max(),
                         colors=color, linestyles=_LS_NAME[ls], linewidths=1.1,
                         alpha=0.95, zorder=4)

    # reference values.
    ref_kw = dict(color=ps.COLORS["reference"], ls=(0, (5, 3)), lw=0.8, alpha=0.7, zorder=1)
    ax_d.axvline(REF["division_rate"], **ref_kw)
    ax_m.axvline(REF["motility"], **ref_kw)
    ax_j.axvline(REF["division_rate"], **ref_kw)
    ax_j.axhline(REF["motility"], **ref_kw)

    for ax in (ax_d, ax_j, ax_m):
        ax.set_xlim(0, 1)
    ax_j.set_ylim(0, 1)
    ax_d.set_xticklabels([])
    ax_d.set_yticks([])
    ax_m.set_yticks([])
    ax_j.set_xlabel("division rate")
    ax_j.set_ylabel("motility")
    ax_m.set_xlabel("motility")
    for ax in (ax_d, ax_j, ax_m):
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)

    # legend in the empty top-right cell; markers + linestyles carry the method
    # identity in grayscale, honest per-method sample counts alongside.
    ax_leg.axis("off")
    handles = [
        Line2D([0], [0], color=ps.COLORS[KEY[m]], marker=ps.MARKERS[KEY[m]],
               ls=ps.LINESTYLES[KEY[m]], lw=1.3, markersize=5,
               label=f"{ps.LABELS[KEY[m]]}\n(n={counts[m]})")
        for m in ORDER
    ]
    handles.append(Line2D([0], [0], **{**ref_kw, "alpha": 1.0}, label="reference value"))
    ax_leg.legend(handles=handles, loc="center", frameon=False,
                  borderaxespad=0.0, handlelength=1.8, labelspacing=1.0)

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
        frames = fd.load_vendored("fig_cpm_corner")
        vendor = None

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, "fig_cpm_corner", data=vendor)
    print(f"wrote {saved['pdf']}")
    if vendor is not None:
        for k, v in saved.items():
            if k.startswith("csv:"):
                print(f"  vendored {v}")


if __name__ == "__main__":
    main()
