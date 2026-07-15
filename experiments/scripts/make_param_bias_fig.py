#!/usr/bin/env python3
"""Runtime-to-parameter coupling figure (fig_param_bias.pdf).

Gaussian-mean benchmark (analytic posterior, 48 workers, 60 s budget, 5
replicates). The simulator's post-evaluation delay is coupled to the inferred
parameter, so one region of parameter space is evaluated far more often than its
mirror image; the coupling strength sweeps sigma in {0, 0.5, 1, 2} (0 = uniform
runtime reference). Both methods.

Two panels, each median + inter-quartile range over replicates:
* (a) throughput (simulations/s) versus coupling strength — the gradient bites
  for both methods, but the asynchronous method sustains markedly higher
  throughput and degrades less.
* (b) posterior-mean error to the analytic Gaussian posterior versus coupling —
  the error stays flat for both methods, i.e. the over-representation induced by
  the coupling does not distort the reported posterior at these coupling levels.

Data: ``parameter_bias/data/{runtime_debug_summary,gaussian_analytic_summary}.csv``.
Default draws from the vendored CSVs; ``--refresh`` re-derives them. Styling via
async_abc.plotting.paper_style.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import _figdata as fd
from async_abc.plotting import paper_style as ps

FIG = "fig_param_bias"
ORDER = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
SIGMAS = [0.0, 0.5, 1.0, 2.0]


def _med_iqr(df, value_col, by=("base_method", "sigma")):
    return (df.groupby(list(by))[value_col]
            .agg(median="median",
                 q1=lambda s: s.quantile(0.25),
                 q3=lambda s: s.quantile(0.75))
            .reset_index())


def aggregate(root: Path):
    ddir = root / "parameter_bias" / "data"
    debug = pd.read_csv(ddir / "runtime_debug_summary.csv")
    debug["sigma"] = debug["method"].str.extract(r"sigma([0-9.]+)").astype(float)
    trows = []
    for (bm, sig, rep), g in debug.groupby(["base_method", "sigma", "replicate"]):
        span = float(g["max_end"].max() - g["min_start"].min())
        if span <= 0:
            continue
        trows.append(dict(base_method=bm, sigma=sig, replicate=rep,
                          throughput=int(g["n_attempts"].sum()) / span))
    thr = _med_iqr(pd.DataFrame(trows), "throughput")

    q = pd.read_csv(ddir / "gaussian_analytic_summary.csv")
    q["sigma"] = q["method"].str.extract(r"sigma([0-9.]+)").astype(float)
    q["base_method"] = q["method"].str.replace(r"__sigma[0-9.]+$", "", regex=True)
    err = _med_iqr(q, "analytic_posterior_mean_abs_error")
    return {"throughput": thr, "error": err}


def draw(frames):
    thr, err = frames["throughput"], frames["error"]
    fig, (axT, axE) = plt.subplots(1, 2, figsize=ps.fig_size(0.92, aspect=0.46))
    for m in ORDER:
        k = KEY[m]
        t = thr[thr["base_method"] == m].set_index("sigma").reindex(SIGMAS)
        axT.fill_between(SIGMAS, t["q1"], t["q3"], color=ps.COLORS[k], alpha=0.18, lw=0)
        axT.plot(SIGMAS, t["median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                 ls=ps.LINESTYLES[k], label=ps.LABELS[k])
        e = err[err["base_method"] == m].set_index("sigma").reindex(SIGMAS)
        axE.fill_between(SIGMAS, e["q1"], e["q3"], color=ps.COLORS[k], alpha=0.18, lw=0)
        axE.plot(SIGMAS, e["median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                 ls=ps.LINESTYLES[k], label=ps.LABELS[k])
    axT.set_xlabel(r"runtime$\to$parameter coupling strength")
    axT.set_ylabel("throughput (simulations / s)")
    axT.set_ylim(bottom=0)
    axT.grid(True, ls=":", lw=0.4, alpha=0.6)
    ps.panel_tag(axT, "(a)")
    axT.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.8",
               loc="upper right", fontsize=6)
    axE.set_xlabel(r"runtime$\to$parameter coupling strength")
    axE.set_ylabel("posterior-mean error")
    axE.set_ylim(bottom=0)
    axE.grid(True, ls=":", lw=0.4, alpha=0.6)
    ps.panel_tag(axE, "(b)")
    for ax in (axT, axE):
        ax.set_xticks(SIGMAS)
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
    for m in ORDER:
        t = frames["throughput"]
        sub = t[t["base_method"] == m]
        vals = ", ".join(f"σ{s:g}={v:.0f}" for s, v in zip(sub["sigma"], sub["median"]))
        print(f"  {KEY[m]:>5} throughput: {vals}")


if __name__ == "__main__":
    main()
