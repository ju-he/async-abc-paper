#!/usr/bin/env python3
"""Additional illustration 2: cumulative completions against wall clock.

(a) Cellular Potts 50^3, W = 48, k = 100: the asynchronous arm against the
    barrierized twin (same propagator, a collective barrier before each
    proposal). The twin is a staircase whose treads are the generation times.
(b) Persistent straggler, W = 16, one worker slowed 20x: the same pair on a
    log axis, where the barrier costs about 400x.

Inputs are the per-particle subsets pulled by fetch_records.sh into ../data.
If the asynchronous Cellular Potts subset from the scaling campaign is
missing, the staged production-campaign async arm (same simulator and
worker count, wall-limited run) is used and the caption should say so.
Output: ../out/a2_cumulative_completions.pdf
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from async_abc.plotting import paper_style as ps  # noqa: E402

DATA = HERE.parent / "data"
OUT = HERE.parent / "out"
STAGING = Path("/home/juhe/async-abc-rerun-staging")


def completions(df: pd.DataFrame, replicate: int, tmax: float):
    t = np.sort(df.loc[df.replicate == replicate, "wall_time"].to_numpy())
    t = t[t <= tmax]
    return t, np.arange(1, len(t) + 1)


def load(name: str) -> pd.DataFrame:
    p = DATA / name
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"{p} missing or empty: run fetch_records.sh first")
    return pd.read_csv(p)


def load_cpm_async() -> tuple[pd.DataFrame, str]:
    p = DATA / "a2_cpm_async_w48.csv"
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)
        if len(df):
            return df, "scaling campaign, W=48, k=100"
    df = pd.read_csv(STAGING / "cellular_potts/data/raw_results.csv",
                     usecols=["method", "replicate", "wall_time"])
    return df[df.method == "async_propulate_abc"], "production campaign async arm (fallback)"


def draw_panel(ax, t_a, c_a, t_s, c_s, tmax):
    ax.step(t_s, c_s, where="post", color=ps.COLORS["sync"], lw=1.2, ls="-",
            label="barrierized twin")
    ax.plot(t_a, c_a, color=ps.COLORS["async"], lw=1.2, label=ps.LABELS["async"])
    ax.set_xlim(0, tmax)
    ax.set_xlabel("wall clock (s)")
    ax.set_ylabel("simulations completed")
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)


def longest_tread(t_s, c_s):
    """Start, end and level of the longest flat stretch of the staircase."""
    gaps = np.diff(t_s)
    i = int(np.argmax(gaps))
    return t_s[i], t_s[i + 1], c_s[i]


def main() -> None:
    ps.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ps.fig_size(1.0, aspect=0.42))

    # (a) Cellular Potts
    twin = load("a2_cpm_twin_w48.csv")
    asy, src = load_cpm_async()
    rep = int(sorted(twin.replicate.unique())[0])
    tmax = 200.0
    t_s, c_s = completions(twin, rep, tmax)
    t_a, c_a = completions(asy, int(sorted(asy.replicate.unique())[0]), tmax)
    draw_panel(ax1, t_a, c_a, t_s, c_s, tmax)
    ps.panel_tag(ax1, "(a) Cellular Potts $50^3$, $W=48$")
    ax1.legend(frameon=False, loc="lower right", fontsize=6.5)
    g0, g1, lvl = longest_tread(t_s, c_s)
    ax1.annotate("all 48 workers wait\nfor the slowest simulation", xy=(0.5 * (g0 + g1), lvl),
                 xytext=(0.5 * (g0 + g1) - 5, lvl + 520), fontsize=6.5, ha="right", color="0.3",
                 arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    rate_a = c_a[-1] / t_a[-1]
    rate_s = c_s[-1] / t_s[-1]
    print(f"(a) async source: {src}; async {c_a[-1]} in {t_a[-1]:.0f}s ({rate_a:.2f}/s); "
          f"twin {c_s[-1]} in {t_s[-1]:.0f}s ({rate_s:.2f}/s); ratio {rate_a / rate_s:.3f}; "
          f"longest tread {g0:.0f}-{g1:.0f}s")

    # (b) persistent straggler 20x
    twin = load("a2_straggler_twin_f20.csv")
    asy = load("a2_straggler_async_f20.csv")
    tmax = 40.0
    rep = int(sorted(twin.replicate.unique())[0])
    t_s, c_s = completions(twin, rep, tmax)
    t_a, c_a = completions(asy, int(sorted(asy.replicate.unique())[0]), tmax)
    draw_panel(ax2, t_a, c_a, t_s, c_s, tmax)
    ax2.set_yscale("log")
    ax2.set_ylim(1, None)
    ps.panel_tag(ax2, "(b) persistent straggler $20\\times$, $W=16$")
    ax2.annotate("16 per generation,\none generation per 2 s", xy=(t_s[len(t_s) // 2], c_s[len(c_s) // 2]),
                 xytext=(0.55, 0.32), textcoords="axes fraction", fontsize=6.5, color="0.3",
                 arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    rate_a = c_a[-1] / t_a[-1]
    rate_s = c_s[-1] / t_s[-1]
    print(f"(b) async {c_a[-1]} in {t_a[-1]:.1f}s ({rate_a:.0f}/s); twin {c_s[-1]} in "
          f"{t_s[-1]:.1f}s ({rate_s:.2f}/s); ratio {rate_a / rate_s:.0f}")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "a2_cumulative_completions.pdf")
    print(f"wrote {OUT / 'a2_cumulative_completions.pdf'}")


if __name__ == "__main__":
    main()
