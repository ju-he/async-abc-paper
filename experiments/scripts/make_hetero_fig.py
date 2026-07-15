#!/usr/bin/env python3
"""Runtime-heterogeneity figures (fig_hetero_idle.pdf, fig_hetero_quality.pdf).

Log-normal post-evaluation runtime noise with spread sigma in {0,0.5,1,1.5,2}
(Gaussian-mean, 48 workers, 60 s budget, 5 replicates), both methods.

* ``fig_hetero_idle`` — worker utilization loss (idle fraction) versus sigma.
  Idle fraction = 1 - sum(busy)/(n_workers * span) per (method, replicate),
  matching reporting/runtime_summary.py. The synchronous baseline idles at
  generation barriers (~0.51 -> ~0.79); the asynchronous method stays near-fully
  utilized (~0.01 -> ~0.27).
* ``fig_hetero_quality`` — two panels: (a) posterior-mean error to the analytic
  Gaussian posterior versus sigma; (b) simulations completed within the budget
  versus sigma. The synchronous baseline's error climbs ~an order of magnitude
  and its completion is starved by barrier idling; the asynchronous method holds
  error flat and completes several-fold more.

All series are median + inter-quartile range over replicates. Default draws from
vendored CSVs; ``--refresh`` re-derives them. Styling via
async_abc.plotting.paper_style.
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

ORDER = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def _med_iqr(df, value_col, by=("base_method", "sigma")):
    return (df.groupby(list(by))[value_col]
            .agg(median="median",
                 q1=lambda s: s.quantile(0.25),
                 q3=lambda s: s.quantile(0.75),
                 n="count")
            .reset_index())


def _idle_per_run(debug: pd.DataFrame) -> pd.DataFrame:
    d = debug.copy()
    d["sigma"] = d["method"].str.extract(r"sigma([0-9.]+)").astype(float)
    rows = []
    for (bm, sig, rep), g in d.groupby(["base_method", "sigma", "replicate"]):
        span = float(g["max_end"].max() - g["min_start"].min())
        nw = int(g["worker_id"].nunique())
        if span <= 0 or nw == 0:
            continue
        idle = 1.0 - float(g["total_busy_s"].sum()) / (nw * span)
        sims = int(g["n_attempts"].sum())
        rows.append(dict(base_method=bm, sigma=sig, replicate=rep,
                         idle_fraction=min(max(idle, 0.0), 1.0), sims_completed=sims))
    return pd.DataFrame(rows)


# ---- idle figure -----------------------------------------------------------
def aggregate_idle(root: Path):
    debug = pd.read_csv(root / "runtime_heterogeneity" / "data" / "runtime_debug_summary.csv")
    per_run = _idle_per_run(debug)
    return {"idle": _med_iqr(per_run, "idle_fraction"),
            "sims": _med_iqr(per_run, "sims_completed")}


def draw_idle(idle):
    fig, ax = plt.subplots(figsize=ps.fig_size(0.6, aspect=0.72))
    for m in ORDER:
        k = KEY[m]
        sub = idle[idle["base_method"] == m].set_index("sigma").reindex(SIGMAS)
        ax.fill_between(SIGMAS, sub["q1"], sub["q3"], color=ps.COLORS[k], alpha=0.18, lw=0)
        ax.plot(SIGMAS, sub["median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                ls=ps.LINESTYLES[k], label=ps.LABELS[k])
    ax.set_xlabel(r"runtime-noise spread $\sigma$")
    ax.set_ylabel("worker idle fraction")
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    return fig


# ---- quality figure (2 panels) --------------------------------------------
def aggregate_quality(root: Path):
    hdir = root / "runtime_heterogeneity" / "data"
    q = pd.read_csv(hdir / "gaussian_analytic_summary.csv")
    q["sigma"] = q["method"].str.extract(r"sigma([0-9.]+)").astype(float)
    q["base_method"] = q["method"].str.replace(r"__sigma[0-9.]+$", "", regex=True)
    err = _med_iqr(q, "analytic_posterior_mean_abs_error")
    per_run = _idle_per_run(pd.read_csv(hdir / "runtime_debug_summary.csv"))
    sims = _med_iqr(per_run, "sims_completed")
    return {"quality": err, "sims": sims}


def draw_quality(frames):
    err, sims = frames["quality"], frames["sims"]
    fig, (axE, axS) = plt.subplots(1, 2, figsize=ps.fig_size(1.0, aspect=0.42))
    for m in ORDER:
        k = KEY[m]
        e = err[err["base_method"] == m].set_index("sigma").reindex(SIGMAS)
        axE.fill_between(SIGMAS, e["q1"], e["q3"], color=ps.COLORS[k], alpha=0.18, lw=0)
        axE.plot(SIGMAS, e["median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                 ls=ps.LINESTYLES[k], label=ps.LABELS[k])
        s = sims[sims["base_method"] == m].set_index("sigma").reindex(SIGMAS)
        axS.fill_between(SIGMAS, s["q1"], s["q3"], color=ps.COLORS[k], alpha=0.18, lw=0)
        axS.plot(SIGMAS, s["median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                 ls=ps.LINESTYLES[k], label=ps.LABELS[k])
    axE.set_yscale("log")
    axE.set_xlabel(r"runtime-noise spread $\sigma$")
    axE.set_ylabel("posterior-mean error")
    axE.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ps.panel_tag(axE, "(a)")
    axS.set_xlabel(r"runtime-noise spread $\sigma$")
    axS.set_ylabel("simulations completed")
    axS.set_ylim(bottom=0)
    axS.grid(True, ls=":", lw=0.4, alpha=0.6)
    ps.panel_tag(axS, "(b)")
    axS.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fd.add_refresh_arg(parser)
    args = parser.parse_args()
    ps.apply()
    refreshing = args.refresh is not None

    if refreshing:
        root = Path(args.refresh)
        idle_frames = aggregate_idle(root)
        qual_frames = aggregate_quality(root)
    else:
        idle_frames = {"idle": fd.load_vendored("fig_hetero_idle")["idle"]}
        qf = fd.load_vendored("fig_hetero_quality")
        qual_frames = {"quality": qf["quality"], "sims": qf["sims"]}

    fig_i = draw_idle(idle_frames["idle"])
    si = ps.save_paper_figure(fig_i, "fig_hetero_idle",
                              data={"idle": idle_frames["idle"]} if refreshing else None)
    print(f"wrote {si['pdf']}")

    fig_q = draw_quality(qual_frames)
    sq = ps.save_paper_figure(
        fig_q, "fig_hetero_quality",
        data={"quality": qual_frames["quality"], "sims": qual_frames["sims"]} if refreshing else None,
    )
    print(f"wrote {sq['pdf']}")

    idle = idle_frames["idle"]
    for m in ORDER:
        sub = idle[idle["base_method"] == m]
        rng = ", ".join(f"σ{s:g}={v:.3f}" for s, v in zip(sub["sigma"], sub["median"]))
        print(f"  {KEY[m]:>5} idle: {rng}")


if __name__ == "__main__":
    main()
