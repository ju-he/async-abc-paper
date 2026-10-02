#!/usr/bin/env python3
"""The barrier figure of Sec. 6.1 (fig_barrier.pdf): three panels in one.

(a) Straggler throughput: one of 16 workers carries a permanent post-evaluation
    delay of 0.1 s scaled by 0-20x (Gaussian mean, 300 s). pyABC barriers on
    the slow worker every generation and collapses ~8800 -> ~60 sims/s; the
    asynchronous method never waits and holds ~2600-3900.
(b) Worker idle fraction under lognormal runtime heterogeneity of spread sigma
    (Gaussian mean, 48 workers, 60 s). pyABC idles 0.51 -> 0.79 of worker
    time; the asynchronous method 0.01 -> 0.27.
(c) The barrier's cost predicted from the asynchronous arm's timing alone
    against the cost measured with the barrierized twin, on every workload
    (the former stand-alone fig_predictor).

Each panel draws from the vendored CSVs of the figure it absorbed
(fig_straggler_throughput, fig_hetero_idle, fig_predictor), so this script has
no --refresh path of its own: refresh those three and re-run this one. Panel
(c) reuses the series styling and the outside-the-domain set of
make_predictor_fig.py so the two cannot drift apart.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

import _figdata as fd
from async_abc.plotting import paper_style as ps
from make_predictor_fig import OUTSIDE, STYLE

NAME = "fig_barrier"
SOURCES = ("fig_straggler_throughput", "fig_hetero_idle", "fig_predictor")
ORDER = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
FACTORS = [0.0, 1.0, 5.0, 10.0, 20.0]
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def _straggler(ax, g):
    x = np.arange(len(FACTORS))
    for m in ORDER:
        k = KEY[m]
        sub = g[g["base_method"] == m].set_index("slowdown_factor").reindex(FACTORS)
        ax.fill_between(x, sub["throughput_q1"], sub["throughput_q3"],
                        color=ps.COLORS[k], alpha=0.18, linewidth=0)
        ax.plot(x, sub["throughput_median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                ls=ps.LINESTYLES[k], label=ps.LABELS[k])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1", "5", "10", "20"])
    ax.set_xlabel("straggler slowdown factor ($W{=}16$)")
    ax.set_ylabel("throughput (sims/s)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ps.panel_tag(ax, "(a)")


def _idle(ax, idle):
    for m in ORDER:
        k = KEY[m]
        sub = idle[idle["base_method"] == m].set_index("sigma").reindex(SIGMAS)
        ax.fill_between(SIGMAS, sub["q1"], sub["q3"], color=ps.COLORS[k], alpha=0.18, lw=0)
        ax.plot(SIGMAS, sub["median"], marker=ps.MARKERS[k], color=ps.COLORS[k],
                ls=ps.LINESTYLES[k], label=ps.LABELS[k])
    ax.set_xlabel(r"runtime-noise spread $\sigma$ ($W{=}48$)")
    ax.set_ylabel("worker idle fraction")
    ax.set_ylim(0, 1)
    ax.set_xticks(SIGMAS)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ps.panel_tag(ax, "(b)")


def _predictor(ax, df):
    med = (df.groupby(["workload", "level"])
             .agg(pred=("ratio_pred", "median"), meas=("ratio_meas", "median"),
                  meas_lo=("ratio_meas", "min"), meas_hi=("ratio_meas", "max"))
             .reset_index())
    lo, hi = 0.7, 700
    ax.plot([lo, hi], [lo, hi], color="0.45", ls=(0, (5, 3)), lw=0.8, zorder=1)
    # Workloads, not methods: keep clear of the async/sync colours of (a) and (b).
    palette = ["#000000", "#CC79A7", "#009E73", "#56B4E9"]
    for (wl, st), color in zip(STYLE.items(), palette):
        sub = med[med["workload"] == wl]
        inside = sub[[(wl, l) not in OUTSIDE for l in sub["level"]]]
        outside = sub[[(wl, l) in OUTSIDE for l in sub["level"]]]
        ax.errorbar(inside["pred"], inside["meas"],
                    yerr=[inside["meas"] - inside["meas_lo"], inside["meas_hi"] - inside["meas"]],
                    fmt=st["marker"], color=color, ms=4.5, lw=0.8, capsize=2, label=st["label"], zorder=3)
        if not outside.empty:
            ax.scatter(outside["pred"], outside["meas"], marker=st["marker"], s=24, facecolors="none",
                       edgecolors=color, lw=1.0, zorder=3)
    ax.scatter([], [], marker="o", s=24, facecolors="none", edgecolors="0.3", label="outside the model's domain")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Square by construction (xlim == ylim and a square gridspec cell), not by
    # set_aspect: a fixed-aspect axes defeats constrained layout's label fitting.
    ratio_fmt = FuncFormatter(lambda v, _: f"{v:g}$\\times$")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ratio_fmt)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("predicted from asynchronous timing")
    ax.set_ylabel("measured against the twin")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ps.panel_tag(ax, "(c)")


def draw(frames):
    width, _ = ps.fig_size(1.0)
    fig = plt.figure(figsize=(width, 0.62 * width), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])
    _straggler(ax_a, frames["straggler_throughput"])
    _idle(ax_b, frames["idle"])
    _predictor(ax_c, frames["predictor_rows"])
    # Panel (c)'s workload legend as a strip under the whole figure; the
    # method legend of (a)/(b) lives inside (a). Constrained layout reserves
    # the strip's space for an "outside" figure legend.
    # Both curves of (a) cross every in-panel quadrant, so the method legend of
    # (a)/(b) is a strip above the figure and the workload legend of (c) a
    # strip below it; constrained layout reserves space for "outside" legends.
    h_ab, l_ab = ax_a.get_legend_handles_labels()
    fig.legend(h_ab, l_ab, loc="outside upper center", ncol=2, frameon=False,
               handlelength=1.8, columnspacing=2.0, fontsize=7)
    h_c, l_c = ax_c.get_legend_handles_labels()
    fig.legend(h_c, l_c, loc="outside lower center", ncol=3, frameon=False,
               handlelength=1.2, columnspacing=1.4, fontsize=7)
    return fig


def main() -> None:
    ps.apply()
    frames = {}
    for src in SOURCES:
        frames.update(fd.load_vendored(src))
    fig = draw(frames)
    saved = ps.save_paper_figure(fig, NAME, metadata={"sources": list(SOURCES)})
    print(f"wrote {saved['pdf']}")


if __name__ == "__main__":
    main()
