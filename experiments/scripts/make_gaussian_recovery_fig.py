#!/usr/bin/env python3
"""Gaussian-mean posterior recovery vs. wall-clock time (fig_gaussian_recovery.pdf).

Wasserstein distance to the *true* mean versus wall-clock budget for the three
methods. The asynchronous method (archive) tracks the matched synchronous
baseline (generation) throughout the budget, with replicate confidence bands;
rejection ABC sits far off scale at W ~ 2.4 and is reported as a text reference
on the axes (the old full-range inset printed below ~6 pt and was dropped for
legibility, review II.7).

Data: the validated Gaussian-mean rerun quality summary,
``gaussian_mean/plots/quality_vs_wall_time_data.csv`` (method, axis_value =
wall-clock seconds, wasserstein, ci_low, ci_high, n_replicates).

Default draws from the vendored CSV; ``--refresh`` re-derives it from the
campaign output. Styling via async_abc.plotting.paper_style (Type-42, Okabe-Ito,
print width).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _figdata as fd
from async_abc.plotting import paper_style as ps

# Curves drawn as lines+bands; rejection is annotated as an off-scale reference.
ORDER = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {
    "async_propulate_abc": "async",
    "abc_smc_baseline": "sync",
    "rejection_abc": "rejection",
}


def _style(method: str) -> dict:
    k = KEY[method]
    return dict(color=ps.COLORS[k], marker=ps.MARKERS[k], ls=ps.LINESTYLES[k], label=ps.LABELS[k])


def _curve(df, method: str):
    s = df[df["method"] == method].sort_values("axis_value")
    return (s["axis_value"].to_numpy(), s["wasserstein"].to_numpy(),
            s["wasserstein_ci_low"].to_numpy(), s["wasserstein_ci_high"].to_numpy())


def recovery_fig(df):
    fig, ax = plt.subplots(figsize=ps.fig_size(0.6, aspect=0.68))

    for m in ORDER:
        t, w, lo, hi = _curve(df, m)
        st = _style(m)
        ax.plot(t, w, marker=st["marker"], color=st["color"], mfc=st["color"],
                ls=st["ls"], label=st["label"])
        ax.fill_between(t, lo, hi, color=st["color"], alpha=0.15, linewidth=0)

    # Rejection ABC lives far off this scale (W ~ 2.4); report it as text rather
    # than an unreadable inset (review II.7). Use the floor across the budget.
    rej = df[df["method"] == "rejection_abc"].sort_values("axis_value")["wasserstein"]
    if len(rej):
        rej_w = float(rej.iloc[-1])
        ax.text(0.97, 0.95, f"Rejection ABC reference: $W\\approx{rej_w:.1f}$ (off scale)",
                transform=ax.transAxes, ha="right", va="top",
                color=ps.COLORS["rejection"], fontsize=7)

    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel(r"Wasserstein distance to true $\mu$")
    ax.set_xlim(0, 305)
    ax.set_ylim(0, 0.17)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    fig.tight_layout()
    return fig


def aggregate(root: Path):
    """Read the Gaussian-mean quality-vs-wall-time summary from the campaign."""
    import pandas as pd
    csv = root / "gaussian_mean" / "plots" / "quality_vs_wall_time_data.csv"
    df = pd.read_csv(csv)
    keep = ORDER + ["rejection_abc"]
    return {"quality_vs_wall_time": df[df["method"].isin(keep)].reset_index(drop=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fd.add_refresh_arg(parser)
    args = parser.parse_args()
    ps.apply()

    if args.refresh is not None:
        frames = aggregate(Path(args.refresh))
        vendor = frames
    else:
        frames = fd.load_vendored("fig_gaussian_recovery")
        vendor = None

    df = frames["quality_vs_wall_time"]
    fig = recovery_fig(df)
    saved = ps.save_paper_figure(fig, "fig_gaussian_recovery", data=vendor)
    print(f"wrote {saved['pdf']}")

    for m in ORDER + ["rejection_abc"]:
        s = df[df["method"] == m].sort_values("axis_value")
        if len(s):
            print(f"{m:>22}: final W = {s['wasserstein'].iloc[-1]:.4f}  "
                  f"({int(s['n_replicates'].max())} reps)")


if __name__ == "__main__":
    main()
