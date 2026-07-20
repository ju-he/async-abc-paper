#!/usr/bin/env python3
"""Simulation-based calibration figures (fig_sbc_coverage.pdf, fig_sbc_rank.pdf).

Left: empirical vs nominal coverage with the perfect-calibration diagonal.
Right: rank histogram with the uniform expectation and a 99% binomial band.

Data: the reported full-history AMIS SBC run (Gaussian-mean, mu, 500 trials) —
``sbc/data/{coverage,sbc_ranks}.csv``. Async is well calibrated; the synchronous
baseline is over-confident (under-covers). The two methods use different rank
supports (the async full-history report resamples ``n_samples=2000`` draws, the
baseline reports its final population), so the rank panel plots the *normalized*
rank u = rank/n_samples on [0,1] and bins uniformly, with a flat expectation
n_trials/10 and a flat 99% binomial band (per-bin proportion 1/10).

Default draws from the vendored CSVs; ``--refresh`` re-derives them from the
campaign output. Styling via async_abc.plotting.paper_style (Type-42, Okabe-Ito,
print width).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

import _figdata as fd
from async_abc.plotting import paper_style as ps

ORDER = ["async_propulate_abc", "abc_smc_baseline"]
KEY = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
N_BINS = 10


def _style(method: str) -> dict:
    k = KEY[method]
    return dict(color=ps.COLORS[k], marker=ps.MARKERS[k], ls=ps.LINESTYLES[k], label=ps.LABELS[k])


def coverage_fig(coverage):
    fig, ax = plt.subplots(figsize=ps.fig_size(0.48, aspect=0.92))
    ax.plot([0, 1], [0, 1], color=ps.COLORS["reference"], ls=":", lw=1.0, label="perfect calibration")
    for m in ORDER:
        sub = coverage[coverage["method"] == m].sort_values("coverage_level")
        st = _style(m)
        ax.plot(sub["coverage_level"], sub["empirical_coverage"], marker=st["marker"],
                color=st["color"], mfc=st["color"], ls=st["ls"], label=st["label"])
    ax.set_xlabel("nominal coverage level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.35, 1.0)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    fig.tight_layout()
    return fig


def rank_fig(ranks):
    # Ranks live on {0,...,n_samples}; the two methods use different n_samples (the
    # async full-history report resamples 2000 draws, the baseline reports its final
    # population), so we plot the *normalized* rank u = rank/n_samples on [0,1] and bin
    # uniformly. Under calibration u is ~uniform, giving a flat expectation
    # n_trials/N_BINS and a flat 99% binomial band (per-bin proportion 1/N_BINS).
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    n_trials = int((ranks["method"] == ORDER[0]).sum())
    p_bin = 1.0 / N_BINS
    exp = n_trials * p_bin
    lo = float(binom.ppf(0.005, n_trials, p_bin))
    hi = float(binom.ppf(0.995, n_trials, p_bin))

    fig, ax = plt.subplots(figsize=ps.fig_size(0.48, aspect=0.92))
    # flat 99% uniform band + expectation across the normalized-rank axis
    ax.fill_between(edges, np.full(edges.size, lo), np.full(edges.size, hi), step="post",
                    color="0.85", zorder=0, label="99% uniform band")
    ax.step(edges, np.full(edges.size, exp), where="post", color=ps.COLORS["reference"],
            ls=":", lw=1.0)

    for m in ORDER:
        sub = ranks[ranks["method"] == m]
        u = sub["rank"].to_numpy() / sub["n_samples"].to_numpy()
        counts, _ = np.histogram(u, bins=edges)
        st = _style(m)
        ax.step(edges, np.r_[counts, counts[-1]], where="post",
                color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
    ax.set_xlabel("normalized rank statistic")
    ax.set_ylabel(f"count ({n_trials} trials)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)
    # Legend inside the axes so the tight bbox matches fig_sbc_coverage (review II.7.b.3).
    ax.legend(frameon=True, loc="upper center", ncol=1, handlelength=1.6, fontsize=6)
    fig.tight_layout()
    return fig


def aggregate(root: Path):
    data = root / "sbc" / "data"
    import pandas as pd
    return {
        "coverage": pd.read_csv(data / "coverage.csv"),
        "sbc_ranks": pd.read_csv(data / "sbc_ranks.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fd.add_refresh_arg(parser)
    args = parser.parse_args()
    ps.apply()

    if args.refresh is not None:
        frames = aggregate(Path(args.refresh))
    else:
        cov = fd.load_vendored("fig_sbc_coverage")["coverage"]
        rk = fd.load_vendored("fig_sbc_rank")["sbc_ranks"]
        frames = {"coverage": cov, "sbc_ranks": rk}

    fig_c = coverage_fig(frames["coverage"])
    saved_c = ps.save_paper_figure(
        fig_c, "fig_sbc_coverage",
        data={"coverage": frames["coverage"]} if args.refresh is not None else None,
    )
    print(f"wrote {saved_c['pdf']}")

    fig_r = rank_fig(frames["sbc_ranks"])
    saved_r = ps.save_paper_figure(
        fig_r, "fig_sbc_rank",
        data={"sbc_ranks": frames["sbc_ranks"]} if args.refresh is not None else None,
    )
    print(f"wrote {saved_r['pdf']}")


if __name__ == "__main__":
    main()
