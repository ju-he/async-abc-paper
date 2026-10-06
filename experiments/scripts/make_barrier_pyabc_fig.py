#!/usr/bin/env python3
"""The barrier against pyABC (fig_barrier_pyabc.pdf, appendix): two panels.

(a) Straggler throughput: one of 16 workers carries a permanent post-evaluation
    delay of 0.1 s scaled by 0-20x (Gaussian mean, 300 s). pyABC barriers on
    the slow worker every generation and collapses ~8800 -> ~60 sims/s; the
    asynchronous sampler never waits and holds ~2600-3900.
(b) Worker idle fraction under lognormal runtime heterogeneity of spread sigma
    (Gaussian mean, 48 workers, 60 s). pyABC idles 0.51 -> 0.79 of worker
    time; the asynchronous sampler 0.01 -> 0.27.

Both panels show medians and inter-quartile ranges over five replicates and
share one method legend above the figure. They were panels (a) and (b) of
fig_barrier until the main-text figure was reduced to the predictor panel
(make_barrier_fig.py); the data and styling are unchanged.

Each panel draws from the vendored CSVs of the stand-alone figure it absorbed
(fig_straggler_throughput, fig_hetero_idle), so this script has no --refresh
path of its own: refresh those two and re-run this one.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _figdata as fd
from async_abc.plotting import paper_style as ps

NAME = "fig_barrier_pyabc"
SOURCES = ("fig_straggler_throughput", "fig_hetero_idle")
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


def draw(frames):
    width, _ = ps.fig_size(1.0)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(width, 0.40 * width), layout="constrained")
    _straggler(ax_a, frames["straggler_throughput"])
    _idle(ax_b, frames["idle"])
    # Both curves of (a) cross every in-panel quadrant, so the shared method
    # legend is a strip above the figure; constrained layout reserves its space.
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False,
               handlelength=1.8, columnspacing=2.0, fontsize=7)
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
