#!/usr/bin/env python3
"""Additional illustration 3: the straggler factor E[max_{i<=W} T_i] / mu.

Curves against worker count W for lognormal runtime laws (the injected
heterogeneity family) and for the measured per-simulation durations of the
asynchronous arm on Cellular Potts 50^3 (staged campaign records, earlier
configuration, CV about 0.09). The expectation of the maximum of W i.i.d.
draws is computed exactly from the quantile function,
E[max] = int_0^1 F^{-1}(u) W u^{W-1} du, on a fine grid; for the empirical
distribution the same integral is the order-statistic sum. The three worker
counts the paper runs at are marked. Output: ../out/a3_straggler_factor.pdf
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from async_abc.plotting import paper_style as ps  # noqa: E402

OUT = HERE.parent / "out"
STAGING = Path("/home/juhe/async-abc-rerun-staging")
W_GRID = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
SIGMAS = [0.1, 0.2, 0.5, 1.0, 2.0]


def emax_lognormal(sigma: float, W: np.ndarray) -> np.ndarray:
    """E[max of W] / mean for LN(0, sigma), via the quantile integral."""
    u = (np.arange(1, 400001) - 0.5) / 400000.0
    q = np.exp(sigma * norm.ppf(u))
    mean = np.exp(0.5 * sigma**2)
    out = []
    for w in W:
        out.append(np.mean(q * w * u ** (w - 1)) / mean)
    return np.array(out)


def emax_empirical(samples: np.ndarray, W: np.ndarray) -> np.ndarray:
    """E[max of W] / mean for the empirical distribution of ``samples``."""
    t = np.sort(samples)
    n = len(t)
    i = np.arange(1, n + 1) / n
    im1 = np.arange(0, n) / n
    out = []
    for w in W:
        out.append(np.sum(t * (i**w - im1**w)) / t.mean())
    return np.array(out)


def load_cpm_durations() -> np.ndarray:
    df = pd.read_csv(STAGING / "cellular_potts/data/raw_results.csv",
                     usecols=["method", "sim_start_time", "sim_end_time"])
    a = df[df.method == "async_propulate_abc"]
    return (a.sim_end_time - a.sim_start_time).dropna().to_numpy()


def main() -> None:
    ps.apply()
    fig, ax = plt.subplots(figsize=ps.fig_size(0.62, aspect=0.78))
    greys = ["0.75", "0.6", "0.45", "0.3", "0.1"]
    offsets = {0.1: 4, 0.2: 2, 0.5: 0, 1.0: 0, 2.0: 0}
    for sigma, grey in zip(SIGMAS, greys):
        y = emax_lognormal(sigma, W_GRID)
        ax.plot(W_GRID, y, color=grey, lw=1.1)
        ax.annotate(f"lognormal $\\sigma={sigma:g}$", (W_GRID[-1], y[-1]), xytext=(4, offsets[sigma]),
                    textcoords="offset points", va="center", fontsize=6.5, color=grey)
    cpm = load_cpm_durations()
    y = emax_empirical(cpm, W_GRID)
    ax.plot(W_GRID, y, color=ps.COLORS["rejection"], lw=1.6, marker="^", ms=3)
    ax.annotate(f"Cellular Potts $50^3$,\nmeasured (CV {cpm.std() / cpm.mean():.2f})",
                (W_GRID[-1], y[-1]), xytext=(4, -9), textcoords="offset points",
                va="center", fontsize=6.5, color=ps.COLORS["rejection"])
    for w in (16, 48, 384):
        ax.axvline(w, color=ps.COLORS["neutral"], lw=0.6, ls=":")
        ax.text(w, 1.03, f"$W={w}$", ha="center", va="bottom", fontsize=6.5, color="0.4",
                transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 512)
    ax.set_xticks([1, 4, 16, 48, 128, 384])
    ax.set_xticklabels(["1", "4", "16", "48", "128", "384"])
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1$\\times$", "2$\\times$", "5$\\times$", "10$\\times$", "20$\\times$", "50$\\times$"])
    ax.set_xlabel("workers $W$")
    ax.set_ylabel("straggler factor $\\mathbb{E}[\\max_{i\\leq W}T_i]\\,/\\,\\mu$")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    fig.subplots_adjust(right=0.72)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "a3_straggler_factor.pdf")
    print(f"wrote {OUT / 'a3_straggler_factor.pdf'}")
    for sigma in SIGMAS:
        print(f"sigma={sigma:g}: W=16 {emax_lognormal(sigma, np.array([16]))[0]:.2f}  "
              f"W=48 {emax_lognormal(sigma, np.array([48]))[0]:.2f}  "
              f"W=384 {emax_lognormal(sigma, np.array([384]))[0]:.2f}")
    print(f"cpm measured: W=48 {emax_empirical(cpm, np.array([48]))[0]:.3f}  "
          f"W=384 {emax_empirical(cpm, np.array([384]))[0]:.3f}  (n={len(cpm)})")


if __name__ == "__main__":
    main()
