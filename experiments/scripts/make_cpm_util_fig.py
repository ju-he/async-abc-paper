#!/usr/bin/env python3
"""Cellular-Potts worker-utilization strong-scaling figure (fig_cpm_util.pdf).

Instruments the generation-barrier claim on the cost-bearing CPM simulator: the
binding cost of the synchronous baseline is *waiting at the per-generation
barrier*, not per-arrival overhead. We plot the ``worker_utilization`` recorded
by the CPM strong-scaling sweep (five replicates per cell) against worker count.
The asynchronous method stays near-fully utilized at every scale, while the
synchronous baseline's utilization collapses as more workers wait on the slowest
simulation per generation --- which is why its throughput plateaus while the
asynchronous method keeps scaling.

Method selection (preserved from the original scratch-reading script):

* **Asynchronous** --- always the ``k=100`` run.
* **Synchronous** --- the *fair* configuration (population >= worker count), so
  the only idling measured is barrier idling, not idling from empty population
  slots. For ``w <= 100`` the standard ``k=100`` population already covers every
  worker; for ``w in {192, 384}`` we use the matched ``k=W`` fair variants.

Default draws from the vendored CSV; ``--refresh`` re-derives it from the
campaign output (``scaling_cpm/data/throughput_summary_w{w}_k{k}.csv``). Styling
via async_abc.plotting.paper_style (Type-42, Okabe-Ito, print width).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullLocator, ScalarFormatter

import _figdata as fd
from async_abc.plotting import paper_style as ps

ORDER = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
WORKERS = [1, 4, 16, 48, 96, 192, 384]


def _sync_k(w: int) -> int:
    """Fair synchronous population for ``w`` workers: k=100 covers w<=100, else k=W."""
    return 100 if w <= 100 else w


def _style(method: str) -> dict:
    k = KEY[method]
    return dict(color=ps.COLORS[k], marker=ps.MARKERS[k], ls=ps.LINESTYLES[k], label=ps.LABELS[k])


def aggregate(root: Path):
    data = root / "scaling_cpm" / "data"
    rows = []
    for w in WORKERS:
        for m in ORDER:
            k = 100 if m == "async_propulate_abc" else _sync_k(w)
            df = pd.read_csv(data / f"throughput_summary_w{w}_k{k}.csv")
            u = df[df["base_method"] == m]["worker_utilization"]
            rows.append(
                {
                    "n_workers": w,
                    "method": m,
                    "k": k,
                    "util_mean_pct": 100.0 * u.mean(),
                    "util_std_pct": 100.0 * u.std(),
                }
            )
    return {"cpm_utilization": pd.DataFrame(rows)}


def draw(frames):
    df = frames["cpm_utilization"]
    fig, ax = plt.subplots(figsize=ps.fig_size(0.52, aspect=0.68))
    for m in ORDER:
        sub = df[df["method"] == m].sort_values("n_workers")
        st = _style(m)
        x = sub["n_workers"].to_numpy()
        y = sub["util_mean_pct"].to_numpy()
        e = sub["util_std_pct"].to_numpy()
        ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.15, lw=0)
        ax.plot(x, y, marker=st["marker"], color=st["color"], mfc=st["color"],
                ls=st["ls"], label=st["label"])
    ax.set_xscale("log")
    ax.set_xticks(WORKERS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("workers (cores)")
    ax.set_ylabel("worker utilization (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8)
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
        frames = fd.load_vendored("fig_cpm_util")
        vendor = None  # already committed; don't rewrite

    fig = draw(frames)
    saved = ps.save_paper_figure(fig, "fig_cpm_util", data=vendor)
    print(f"wrote {saved['pdf']}")
    if vendor is not None:
        for kk, vv in saved.items():
            if kk.startswith("csv:"):
                print(f"  vendored {vv}")

    df = frames["cpm_utilization"]
    for m in ORDER:
        for _, r in df[df["method"] == m].sort_values("n_workers").iterrows():
            mn = r["util_mean_pct"]
            print(
                f"w{int(r['n_workers']):>3} k{int(r['k']):>3} {m:>22}: "
                f"util {mn:5.1f}% (idle {100 - mn:4.1f}%)  sd {r['util_std_pct']:.1f}"
            )


if __name__ == "__main__":
    main()
