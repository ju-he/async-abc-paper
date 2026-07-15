#!/usr/bin/env python3
"""Combined strong-scaling figure (fig_scaling_combined.pdf).

Two panels sharing one method palette:

* **(a) Lotka--Volterra** (near-instant simulator): there is no per-evaluation
  compute to distribute, so *neither* method strong-scales -- the panel isolates
  coordination overhead. Async throughput peaks at the full-node boundary (48
  workers) and declines once the job spans multiple nodes, because per-arrival
  coordination dominates on a free simulator; the fair synchronous baseline is
  competitive and catches up at scale.
* **(b) Cellular Potts** (costly simulator): real per-evaluation cost amortizes
  the coordination, so async scales ~linearly to 8 nodes (384 workers) while even
  the fair synchronous baseline plateaus at the generation barrier.

Series drawn per panel (median + inter-quartile band over replicates):

* ``async`` -- asynchronous method, ``k=100`` at every worker count.
* ``sync`` -- FAIR synchronous ABC-SMC baseline. pyABC caps concurrency at the
  population size, so a run with more cores than particles idles the surplus. Low
  worker counts already have ``k=100 >= n_workers`` (fair as-is); above one full
  node the baseline is re-run with the population set to the world size
  (``k = n_workers``) so every allocated core is usable each generation.

Default draws from the vendored CSV; ``--refresh`` re-derives it from the
campaign output (LV: ``scaling/data``; CPM: ``scaling_cpm/data``; one campaign
dir per benchmark). Styling via async_abc.plotting.paper_style (Type-42,
Okabe-Ito, print width).
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

# base_method value in the campaign CSVs -> paper_style method key.
BASE = {"async": "async_propulate_abc", "sync": "abc_smc_baseline"}

# One campaign dir per benchmark. ``fair_threshold`` is the largest worker count
# at which the synchronous baseline already runs fair (k=100 >= n_workers); above
# it the baseline is drawn from the fair k=n_workers re-runs.
PANELS = {
    "lv": dict(
        tag="(a) Lotka–Volterra",
        subdir="scaling",
        workers=[1, 4, 16, 48, 144, 192, 240, 288],
        fair_threshold=48,
    ),
    "cpm": dict(
        tag="(b) Cellular Potts",
        subdir="scaling_cpm",
        workers=[1, 4, 16, 48, 96, 192, 384],
        fair_threshold=96,
    ),
}


def _k_for(method: str, w: int, fair_threshold: int) -> int:
    """Particle count k for a (method, worker) point: async is always k=100;
    the synchronous baseline uses k=100 while fair, else the fair k=n_workers run."""
    if method == "async":
        return 100
    return 100 if w <= fair_threshold else w


def _median_iqr(data_dir: Path, w: int, k: int, base_method: str):
    f = data_dir / f"throughput_summary_w{w}_k{k}.csv"
    df = pd.read_csv(f)
    s = df[df["base_method"] == base_method]["throughput_sims_per_s"].to_numpy()
    if not len(s):
        raise ValueError(f"no {base_method} rows in {f}")
    return float(np.median(s)), float(np.percentile(s, 25)), float(np.percentile(s, 75))


def aggregate(root: Path):
    """Return the single tidy frame plotted by both panels.

    Columns: panel, method, n_workers, k, throughput_median, throughput_lo,
    throughput_hi (throughput in simulations/s; lo/hi are the 25th/75th
    percentiles over replicates)."""
    rows = []
    for panel, cfg in PANELS.items():
        data = root / cfg["subdir"] / "data"
        for w in cfg["workers"]:
            for method, base in BASE.items():
                k = _k_for(method, w, cfg["fair_threshold"])
                med, lo, hi = _median_iqr(data, w, k, base)
                rows.append(dict(
                    panel=panel, method=method, n_workers=w, k=k,
                    throughput_median=med, throughput_lo=lo, throughput_hi=hi,
                ))
    return {"scaling_combined": pd.DataFrame(rows)}


def _series(df: pd.DataFrame, panel: str, method: str):
    sub = df[(df["panel"] == panel) & (df["method"] == method)].sort_values("n_workers")
    return (sub["n_workers"].to_numpy(),
            sub["throughput_median"].to_numpy(),
            sub["throughput_lo"].to_numpy(),
            sub["throughput_hi"].to_numpy())


def _style(method: str) -> dict:
    return dict(color=ps.COLORS[method], marker=ps.MARKERS[method],
                ls=ps.LINESTYLES[method], label=ps.LABELS[method],
                mfc="white" if method == "sync" else ps.COLORS[method])


def _panel_lv(ax, df: pd.DataFrame) -> None:
    """Lotka-Volterra: irregular worker counts -> categorical x with node labels."""
    workers = PANELS["lv"]["workers"]
    x = np.arange(len(workers))
    i48 = workers.index(48)
    ax.axvline(i48 + 0.5, color=ps.COLORS["neutral"], lw=0.8, ls=":", zorder=0)
    ax.annotate("single node $\\to$ multi-node", xy=(i48 + 0.5, 0), xytext=(i48 + 0.4, 30),
                fontsize=6, color=ps.COLORS["neutral"], rotation=90, va="bottom", ha="center")
    for method in ("async", "sync"):
        _, med, lo, hi = _series(df, "lv", method)
        st = _style(method)
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, zorder=1)
        ax.plot(x, med, marker=st["marker"], color=st["color"], ls=st["ls"],
                mfc=st["mfc"], label=st["label"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}" if w < 48 else f"{w}\n({w // 48}n)" for w in workers])
    ax.set_xlabel("workers (n = 48-core nodes)")
    ax.set_ylabel("throughput (sims / s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)


def _panel_cpm(ax, df: pd.DataFrame) -> None:
    """Cellular Potts: log-log throughput with an ideal-linear reference."""
    workers, a_med, _, _ = _series(df, "cpm", "async")
    ideal = a_med[0] * (workers / workers[0])
    ax.plot(workers, ideal, ls=":", color=ps.COLORS["neutral"], lw=1.0,
            label="ideal linear", zorder=1)
    for method in ("async", "sync"):
        _, med, _, _ = _series(df, "cpm", method)
        st = _style(method)
        ax.plot(workers, med, marker=st["marker"], color=st["color"], ls=st["ls"],
                mfc=st["mfc"], label=st["label"], zorder=3)
    ax.axvline(48, color=ps.COLORS["neutral"], lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers])
    ax.set_xlabel("workers")
    ax.set_ylabel("throughput (sims / s)")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    _, s_med, _, _ = _series(df, "cpm", "sync")
    if s_med[-1] > 0:
        # Speedup label placed in the gap between the two curves at max scale.
        ax.annotate(f"{a_med[-1] / s_med[-1]:.0f}$\\times$",
                    xy=(workers[-1] * 0.62, float(np.sqrt(a_med[-1] * s_med[-1]))),
                    color=ps.COLORS["async"], fontsize=8, ha="center", va="center")


def draw(frames):
    df = frames["scaling_combined"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=ps.fig_size(1.0, aspect=0.42))
    _panel_lv(axL, df)
    _panel_cpm(axR, df)
    ps.panel_tag(axL, PANELS["lv"]["tag"])
    ps.panel_tag(axR, PANELS["cpm"]["tag"])
    # Single shared legend beneath both panels (async, sync, ideal-linear).
    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
        frames = fd.load_vendored("fig_scaling_combined")
        vendor = None

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, "fig_scaling_combined", data=vendor)
    print(f"wrote {saved['pdf']}")
    if vendor is not None:
        for k, v in saved.items():
            if k.startswith("csv:"):
                print(f"  vendored {v}")


if __name__ == "__main__":
    main()
