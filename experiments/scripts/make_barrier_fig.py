#!/usr/bin/env python3
"""The barrier figure of Sec. 6.1 (fig_barrier.pdf): one square panel.

The barrier's cost predicted from the asynchronous run's timing alone
(``T_async / T_sync`` with ``T_sync = W / E[max of W per-evaluation
durations]``, Sec. 5) against the cost measured with the barrierized twin, on
every workload the twin ran:

* persistent straggler (Gaussian mean, W=16, one rank slowed 0-20x),
* runtime heterogeneity (Gaussian mean, W=48, lognormal multiplier sigma 0-2),
* Cellular Potts 50^3 at W=48-384 (utilisation ratio, earlier parametrisation),
* Cellular Potts 80^3 at W=48 against the matched pyABC baseline.

Medians over replicates with the replicate range as error bars; the dashed line
is equality. The three configurations outside the model's domain keep their
open markers and are labelled in the panel ("cost-free simulator (0x, 1x)" for
the straggler controls, "384 workers" for the Cellular Potts contention case)
instead of carrying a legend entry. The workload legend sits under the panel
because the longest label does not fit in the panel's empty lower-right corner
at 0.55 linewidth.

The panel reads the vendored CSV of fig_predictor (``predictor_rows.csv``), so
this script has no --refresh path of its own: refresh make_predictor_fig.py
and re-run this one. Marker shapes and the outside-the-domain set are imported
from make_predictor_fig.py so the two cannot drift apart. The former panels
(a) and (b) (throughput under a straggler, idle fraction under runtime
heterogeneity, both against pyABC) now live in make_barrier_pyabc_fig.py
(fig_barrier_pyabc.pdf, appendix).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

import _figdata as fd
from async_abc.plotting import paper_style as ps
from make_predictor_fig import OUTSIDE, STYLE

NAME = "fig_barrier"
SOURCE = "fig_predictor"
# Legend labels of the four workload series (markers come from STYLE).
LABELS = {
    "straggler": "persistent straggler ($W{=}16$)",
    "hetero": "runtime heterogeneity ($W{=}48$)",
    "cpm50": "Cellular Potts $50^3$ ($W{=}48$–$384$, earlier param.)",
    "cpm80": "Cellular Potts $80^3$ ($W{=}48$)",
}
# In-panel annotations of the three configurations outside the model's domain
# (open markers): text, the (workload, level) points it refers to.
ANNOTATIONS = (
    ("cost-free simulator (0×, 1×)", [("straggler", 0.0), ("straggler", 1.0)]),
    ("384 workers", [("cpm50", 384.0)]),
)
LO, HI = 0.7, 700


def _annotate_outside(ax, med):
    pts = {(r.workload, r.level): (r.pred, r.meas) for r in med.itertuples()}
    # Straggler controls: text above the pair, one near-vertical leader line
    # dropping from the text's baseline to each point.
    text, keys = ANNOTATIONS[0]
    tx, ty = 0.85, 260
    ax.text(tx, ty, text, ha="left", va="center", fontsize=6.5, color="0.25", zorder=4)
    for k, x_from in zip(keys, (1.0, 18.0)):
        px, py = pts[k]
        ax.annotate("", xy=(px, py), xytext=(x_from, ty / 1.45), textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.5, shrinkA=0, shrinkB=3), zorder=2)
    # 384 workers: text to the right of the point, where the panel is empty.
    text, keys = ANNOTATIONS[1]
    (px, py), = [pts[k] for k in keys]
    ax.annotate(text, xy=(px, py), xytext=(7, -1), textcoords="offset points", ha="left", va="center",
                fontsize=6.5, color="0.25", zorder=4)


def draw(frames):
    df = frames["predictor_rows"]
    med = (df.groupby(["workload", "level"])
             .agg(pred=("ratio_pred", "median"), meas=("ratio_meas", "median"),
                  meas_lo=("ratio_meas", "min"), meas_hi=("ratio_meas", "max"))
             .reset_index())
    # Square panel at 0.55 linewidth; the extra height holds the one-column
    # workload legend under the panel.
    fig, ax = plt.subplots(figsize=ps.fig_size(0.55, aspect=1.24))
    ax.plot([LO, HI], [LO, HI], color="0.45", ls=(0, (5, 3)), lw=0.8, zorder=1)
    # Workloads, not methods: never the async/sync colours.
    for (wl, st), color in zip(STYLE.items(), ps.WORKLOADS):
        sub = med[med["workload"] == wl]
        inside = sub[[(wl, l) not in OUTSIDE for l in sub["level"]]]
        outside = sub[[(wl, l) in OUTSIDE for l in sub["level"]]]
        ax.errorbar(inside["pred"], inside["meas"],
                    yerr=[inside["meas"] - inside["meas_lo"], inside["meas_hi"] - inside["meas"]],
                    fmt=st["marker"], color=color, ms=4.5, lw=0.8, capsize=2, label=LABELS[wl], zorder=3)
        if not outside.empty:
            ax.scatter(outside["pred"], outside["meas"], marker=st["marker"], s=24, facecolors="none",
                       edgecolors=color, lw=1.0, zorder=3)
    _annotate_outside(ax, med)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_aspect("equal")
    ratio_fmt = FuncFormatter(lambda v, _: f"{v:g}$\\times$")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ratio_fmt)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("predicted from asynchronous timing")
    ax.set_ylabel("measured against the twin")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=1, fontsize=7,
               handlelength=1.2, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    return fig


def main() -> None:
    ps.apply()
    frames = fd.load_vendored(SOURCE)
    fig = draw(frames)
    saved = ps.save_paper_figure(fig, NAME, metadata={"sources": [SOURCE]})
    print(f"wrote {saved['pdf']}")


if __name__ == "__main__":
    main()
