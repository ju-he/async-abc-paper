#!/usr/bin/env python3
"""The barrier's cost, predicted from the asynchronous arm's timing alone (fig_predictor.pdf).

A generation of W evaluations cannot finish before its slowest member, so a
barrierized sampler's throughput is bounded by ``T_sync = W / E[max of W
per-evaluation durations]``. The per-evaluation duration distribution is read
off the *asynchronous* arm's own records -- its recorded simulation times plus
its measured per-evaluation coordination overhead -- and the asynchronous
throughput is measured, so ``T_async / T_sync`` is a prediction that uses no
synchronous or twin data at all. It is plotted against the ratio actually
measured with the barrierized twin (same propagator, a collective barrier
before each breed, nothing else) on every workload we have:

* persistent straggler (Gaussian mean, W=16, one rank slowed 0-20x),
* runtime heterogeneity (Gaussian mean, W=48, lognormal multiplier sigma 0-2),
* Cellular Potts 50^3 at W=48-384 (measured as the utilisation ratio, since the
  twin's simulations also ran longer for reasons the barrier cannot explain),
* Cellular Potts 80^3 at W=48 against the matched pyABC baseline.

Two regimes are marked where the model is known not to apply: a cost-free
simulator with no straggler, where the collective's own latency dominates and
the model has no latency term; and 384 CPM ranks, where the asynchronous
sample carries a contention tail the twin's generations did not show.

The rows are vendored (``predictor_rows.csv``); they were derived by
``.plans/ressources/predictor_twin_from_async_timing.py`` from the campaign
records on scratch (per-worker debug summaries and raw records).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _figdata as fd
from async_abc.plotting import paper_style as ps

NAME = "fig_predictor"
STYLE = {
    "straggler": dict(marker="o", label="persistent straggler ($W{=}16$)"),
    "hetero": dict(marker="s", label="runtime heterogeneity ($W{=}48$)"),
    "cpm50": dict(marker="^", label="Cellular Potts $50^3$ ($W{=}48$–$384$)"),
    "cpm80": dict(marker="D", label="Cellular Potts $80^3$ ($W{=}48$)"),
}
# (workload, level) pairs outside the model's stated domain.
OUTSIDE = {("straggler", 0.0), ("straggler", 1.0), ("cpm50", 384.0)}


def aggregate(root):
    raise SystemExit("fig_predictor has no campaign-root refresh path; see the module docstring")


def draw(frames):
    df = frames["predictor_rows"]
    med = (df.groupby(["workload", "level"])
             .agg(pred=("ratio_pred", "median"), meas=("ratio_meas", "median"),
                  meas_lo=("ratio_meas", "min"), meas_hi=("ratio_meas", "max"))
             .reset_index())
    fig, ax = plt.subplots(figsize=ps.fig_size(0.55, aspect=0.8))
    lo, hi = 0.7, 700
    ax.plot([lo, hi], [lo, hi], color=ps.COLORS["reference"], ls=(0, (5, 3)), lw=0.8, zorder=1)
    palette = [ps.COLORS["async"], ps.COLORS["sync"], ps.COLORS["rejection"], ps.COLORS.get("prior", "0.35")]
    for (wl, st), color in zip(STYLE.items(), palette):
        sub = med[med["workload"] == wl]
        inside = sub[[(wl, l) not in OUTSIDE for l in sub["level"]]]
        outside = sub[[(wl, l) in OUTSIDE for l in sub["level"]]]
        ax.errorbar(inside["pred"], inside["meas"],
                    yerr=[inside["meas"] - inside["meas_lo"], inside["meas_hi"] - inside["meas"]],
                    fmt=st["marker"], color=color, ms=5, lw=0.8, capsize=2, label=st["label"], zorder=3)
        if not outside.empty:
            ax.scatter(outside["pred"], outside["meas"], marker=st["marker"], s=28, facecolors="none",
                       edgecolors=color, lw=1.0, zorder=3)
    ax.scatter([], [], marker="o", s=28, facecolors="none", edgecolors="0.3", label="outside the model's domain")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted from asynchronous timing, $T_{\\mathrm{async}}/T_{\\mathrm{sync}}$")
    ax.set_ylabel("measured against the barrierized arm")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize="small", handlelength=1.2)
    fig.tight_layout()
    return fig


def main() -> None:
    ps.apply()
    frames = fd.load_vendored(NAME)
    fig = draw(frames)
    saved = ps.save_paper_figure(fig, NAME)
    print(f"wrote {saved['pdf']}")
    df = frames["predictor_rows"]
    med = df.groupby(["workload", "level"]).agg(pred=("ratio_pred", "median"), meas=("ratio_meas", "median")).reset_index()
    med["pred/meas"] = med["pred"] / med["meas"]
    med["in_model"] = [(w, l) not in OUTSIDE for w, l in zip(med.workload, med.level)]
    print(med.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    ok = med[med.in_model]
    print(f"\ninside the model's domain: {len(ok)} points, pred/meas median {ok['pred/meas'].median():.3f}, "
          f"range {ok['pred/meas'].min():.2f}-{ok['pred/meas'].max():.2f}")


if __name__ == "__main__":
    main()
