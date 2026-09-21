#!/usr/bin/env python3
"""Cellular Potts: tolerance reached against simulations spent (fig_cpm_eps.pdf).

For each method the k-th (k=100) order statistic of its own losses over the first
``n`` arrivals -- the bandwidth at which exactly k of its draws would be accepted
-- against ``n``, on the fixed two-parameter production run (five replicates of
the asynchronous method and of the matched synchronous baseline, 3600 s each)
and the fairly resourced rejection corpus (13,000 prior draws on 48 ranks). The
curves are what "per-simulation efficiency" means in the text: at equal ``n``
the vertical gap is the per-simulation advantage, the horizontal extent of each
curve is what the wall clock bought, and the slope is the exponent that decides
whether throughput converts into a tighter tolerance (rejection: n^-1 in this
noise-dominated regime, so eps ~ k/n).

Only ``simulation_attempt`` rows enter the count. ``--refresh`` recomputes the
vendored curves from ``experiments/data/cpm_two_param_validation``.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _figdata as fd
from async_abc.analysis.reported_posterior import order_statistic_eps
from async_abc.plotting import paper_style as ps

NAME = "fig_cpm_eps"
DATA = Path(__file__).resolve().parents[1] / "data" / "cpm_two_param_validation"
K = 100
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync", "rejection": "rejection"}


def _curve(losses: np.ndarray, grid: np.ndarray) -> list[tuple[int, float]]:
    out = []
    for n in grid:
        if n > len(losses):
            break
        e = order_statistic_eps(list(losses[:n]), K)
        if e is not None:
            out.append((int(n), float(e)))
    return out


def aggregate(root: Path):
    df = pd.read_csv(root / "cpm_two_param_fixed/cellular_potts_two_param/data/raw_results.csv.gz")
    att = df[df["record_kind"] == "simulation_attempt"]
    grid = np.unique(np.geomspace(2 * K, 14000, 40).astype(int))
    rows = []
    for method in ("async_propulate_abc", "abc_smc_baseline"):
        for rep, g in att[att["method"] == method].groupby("replicate"):
            losses = g.sort_values("sim_end_time")["loss"].to_numpy(float)
            rows += [dict(method=KEY[method], replicate=int(rep), n=n, eps=e) for n, e in _curve(losses, grid)]
    corpus = []
    with tarfile.open(root / "cpm_fair_rejection.tar.gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".jsonl"):
                for line in io.TextIOWrapper(tf.extractfile(m)):
                    r = json.loads(line)
                    corpus.append((int(r["index"]), float(r["loss"])))
    losses = np.array([l for _, l in sorted(corpus)])
    rows += [dict(method="rejection", replicate=0, n=n, eps=e) for n, e in _curve(losses, grid)]
    return {"eps_curves": pd.DataFrame(rows)}


def _exponent(sub: pd.DataFrame, n_min: int = 1000) -> float:
    s = sub[sub["n"] >= n_min]
    if s["n"].nunique() < 3:
        return float("nan")
    med = s.groupby("n")["eps"].median()
    return float(np.polyfit(np.log(med.index.to_numpy(float)), np.log(med.to_numpy()), 1)[0])


def draw(frames):
    df = frames["eps_curves"]
    fig, ax = plt.subplots(figsize=ps.fig_size(0.55, aspect=0.72))
    for key in ("rejection", "sync", "async"):
        sub = df[df["method"] == key]
        if sub.empty:
            continue
        med = sub.groupby("n")["eps"].agg(["median", "min", "max"])
        color = ps.COLORS[key]
        if sub["replicate"].nunique() > 1:
            ax.fill_between(med.index, med["min"], med["max"], color=color, alpha=0.15, lw=0)
        b = _exponent(sub)
        ax.plot(med.index, med["median"], color=color, ls=ps.LINESTYLES[key], marker=ps.MARKERS[key],
                ms=3, markevery=6, label=f"{ps.LABELS[key]} ($n^{{{b:+.2f}}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("simulations spent, $n$")
    ax.set_ylabel(f"$\\epsilon_{{({K})}}$: tolerance at $k={K}$ accepted")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper right", handlelength=2.2, fontsize="small")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", nargs="?", const=str(DATA), default=None, metavar="DATA_ROOT")
    args = parser.parse_args()
    ps.apply()
    if args.refresh is not None:
        frames = aggregate(Path(args.refresh))
        vendor = frames
    else:
        frames = fd.load_vendored(NAME)
        vendor = None
    fig = draw(frames)
    saved = ps.save_paper_figure(fig, NAME, data=vendor)
    print(f"wrote {saved['pdf']}")
    df = frames["eps_curves"]
    for key in ("async", "sync", "rejection"):
        sub = df[df["method"] == key]
        print(f"  {key:>9}: exponent (n>=1000) {_exponent(sub):+.2f}; eps at n=5000: "
              f"{sub[sub.n == sub[sub.n <= 5000].n.max()].eps.median():.3g}; final eps {sub[sub.n == sub.n.max()].eps.median():.3g} at n={sub.n.max()}")


if __name__ == "__main__":
    main()
