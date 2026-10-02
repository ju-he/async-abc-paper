#!/usr/bin/env python3
"""Stripped predictor scatter for graphical-abstract concept 2.

Same data as fig_predictor (vendored predictor_rows.csv), reduced to three
marker types, medians only, the equality line, and no points outside the
model's domain. Output: ../out/ga2_predictor_panel.pdf
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from async_abc.plotting import paper_style as ps  # noqa: E402

OUT = HERE.parent / "out"
OUTSIDE = {("straggler", 0.0), ("straggler", 1.0), ("cpm50", 384.0)}
SERIES = {
    "straggler": ("persistent straggler", "o", ps.COLORS["async"]),
    "hetero": ("runtime heterogeneity", "s", ps.COLORS["sync"]),
    "cpm": ("Cellular Potts tissue simulator", "^", ps.COLORS["rejection"]),
}


def main() -> None:
    ps.apply()
    df = pd.read_csv(ROOT / "experiments/data/paper_figures/fig_predictor/predictor_rows.csv")
    med = (df.groupby(["workload", "level"])
             .agg(pred=("ratio_pred", "median"), meas=("ratio_meas", "median"))
             .reset_index())
    med = med[[(w, l) not in OUTSIDE for w, l in zip(med.workload, med.level)]]
    med["series"] = med.workload.replace({"cpm50": "cpm", "cpm80": "cpm"})

    fig, ax = plt.subplots(figsize=(2.9, 2.9))
    lo, hi = 0.8, 600
    ax.plot([lo, hi], [lo, hi], color="black", ls=(0, (5, 3)), lw=0.8, zorder=1)
    for key, (label, marker, color) in SERIES.items():
        sub = med[med.series == key]
        ax.scatter(sub.pred, sub.meas, marker=marker, s=34, color=color, label=label, zorder=3,
                   edgecolors="white", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fmt = FuncFormatter(lambda v, _: f"{v:g}$\\times$")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("predicted from the asynchronous run")
    ax.set_ylabel("measured with the barrierized twin")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, handletextpad=0.3, borderaxespad=0.3)
    ax.text(0.97, 0.05, "dashed: prediction = measurement", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="0.3")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "ga2_predictor_panel.pdf")
    print(f"wrote {OUT / 'ga2_predictor_panel.pdf'}")


if __name__ == "__main__":
    main()
