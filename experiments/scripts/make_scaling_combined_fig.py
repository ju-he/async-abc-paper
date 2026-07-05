#!/usr/bin/env python3
"""Combined strong-scaling figure: Lotka-Volterra (cheap sim) vs realistic workload
(costly sim), async vs a FAIR synchronous baseline (fig_scaling_combined.pdf).

The synchronous pyABC ABC-SMC baseline caps its concurrency at the population
size, so a run given more cores than particles leaves the surplus idle. To make
the scaling comparison fair, the multi-node baseline points are re-run with the
population size set to the world size (n_workers), so every allocated core can be
used every generation. Points at <=~100 workers already used k=100 particles >=
their worker count, so they are fair as-is.

Left (Lotka-Volterra, near-instant simulator): there is no per-evaluation compute
to distribute, so *neither* method strong-scales -- the panel isolates coordination
overhead. Async's streaming coordination costs it per arrival (its own island
broadcast is O(W) per eval), so on a free simulator the extra workers are not worth
it; the fair synchronous baseline is competitive.

Right (realistic workload, costly simulator): real per-evaluation cost amortizes the
coordination, so async scales ~linearly to 8 nodes while even the fair synchronous
baseline plateaus at the generation barrier (it must wait for the slowest of a
whole population each generation).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRATCH = "/home/juhe/remotes/scratch/herold2/async-abc"
OUT = "/home/juhe/bwSyncShare/Code/async-abc-paper/latex/sn-article-template/figures/fig_scaling_combined.pdf"

ASYNC, SYNC = "async_propulate_abc", "abc_smc_baseline"

# ---- Lotka-Volterra data sources -------------------------------------------------
LV_ORIG = f"{SCRATCH}/run_full_20260626_1816/scaling/data"          # 1/4/16/48 (both arms, k100)
LV_PACK = f"{SCRATCH}/lv_scaling_packed_20260630/scaling/data"      # 144-288 async (k100)
LV_FAIR = f"{SCRATCH}/lv_fair_baseline_20260701/scaling/data"       # 144-288 sync (k=W)
LV_WORKERS = [1, 4, 16, 48, 144, 192, 240, 288]

def _lv_src(w: int, method: str) -> tuple[str, int]:
    """Return (data_dir, k) for a Lotka-Volterra (worker, method) point."""
    if w <= 48:
        return LV_ORIG, 100                       # <=48 workers: k=100 already >= workers (fair)
    if method == ASYNC:
        return LV_PACK, 100                        # async multi-node, k=100 archive throughout
    return LV_FAIR, w                              # FAIR sync: population = world size

# ---- Realistic-workload data sources -------------------------------------------------
RW_FILL = f"{SCRATCH}/run_cpm_fillin_20260628/scaling_cpm/data"    # 1/4/16
RW_ORIG = f"{SCRATCH}/run_cpm_20260626_1906/scaling_cpm/data"      # 48/96
RW_EXT = f"{SCRATCH}/cpm_scaling_ext_20260630/scaling_cpm/data"    # 192/384 async
RW_FAIR = {192: f"{SCRATCH}/cpm_fair_w192_20260701/scaling_cpm/data",
           384: f"{SCRATCH}/cpm_fair_w384_20260701/scaling_cpm/data"}
RW_WORKERS = [1, 4, 16, 48, 96, 192, 384]

def _rw_src(w: int, method: str) -> tuple[str, int]:
    if w <= 16:
        return RW_FILL, 100
    if w <= 96:
        return RW_ORIG, 100                        # <=96 workers: k=100 >= workers (fair)
    if method == ASYNC:
        return RW_EXT, 100
    return RW_FAIR[w], w                            # FAIR sync: population = world size

# ---- shared reader ---------------------------------------------------------------
def _median_iqr(data_dir: str, w: int, k: int, method: str):
    f = os.path.join(data_dir, f"throughput_summary_w{w}_k{k}.csv")
    df = pd.read_csv(f)
    s = df[df["base_method"] == method]["throughput_sims_per_s"].to_numpy()
    if not len(s):
        return np.nan, np.nan, np.nan
    return np.median(s), np.percentile(s, 25), np.percentile(s, 75)


def _curve(workers, src_fn, method):
    med, q1, q3 = [], [], []
    for w in workers:
        d, k = src_fn(w, method)
        m, a, b = _median_iqr(d, w, k, method)
        med.append(m); q1.append(a); q3.append(b)
    return np.array(med), np.array(q1), np.array(q3)


A_STYLE = dict(color="#1f77b4", marker="o", ls="-", lw=2.0, ms=7, label="asynchronous (ours)")
S_STYLE = dict(color="#d62728", marker="s", ls="--", lw=2.0, ms=6, mfc="white",
               label="synchronous baseline (population = cores)")


def _panel_categorical(ax, workers, a, s, title):
    """Lotka-Volterra: irregular worker counts -> categorical x with node labels."""
    x = np.arange(len(workers))
    labels = [f"{w}" if w < 48 else f"{w}\n({w // 48}n)" for w in workers]
    i48 = workers.index(48)
    ax.axvline(i48 + 0.5, color="0.75", lw=1.0, ls=":", zorder=0)
    ax.annotate("1 node $\\to$ multi-node", xy=(i48 + 0.5, 0), xytext=(i48 + 0.42, 30),
                fontsize=8, color="0.45", rotation=90, va="bottom", ha="center")
    for arr, st in [(a, A_STYLE), (s, S_STYLE)]:
        ax.fill_between(x, arr[1], arr[2], color=st["color"], alpha=0.15, zorder=1)
        ax.plot(x, arr[0], marker=st["marker"], color=st["color"], ls=st["ls"], lw=st["lw"],
                ms=st["ms"], mfc=st.get("mfc", st["color"]), label=st["label"], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("workers (n = full 48-core nodes)")
    ax.set_ylabel("throughput (simulations / s)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)


def _panel_loglog(ax, workers, a, s, title):
    """Realistic workload: log-log throughput with an ideal-linear reference."""
    ideal = [a[0][0] * (w / workers[0]) for w in workers]
    ax.plot(workers, ideal, ":", color="0.55", lw=1.4, label="ideal linear", zorder=1)
    for arr, st in [(a, A_STYLE), (s, S_STYLE)]:
        ax.plot(workers, arr[0], marker=st["marker"], color=st["color"], ls=st["ls"], lw=st["lw"],
                ms=st["ms"], mfc=st.get("mfc", st["color"]), label=st["label"], zorder=3)
    ax.axvline(48, color="0.85", lw=1.0, ls="-", zorder=0)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(workers); ax.set_xticklabels([str(w) for w in workers])
    ax.set_xlabel("workers")
    ax.set_ylabel("throughput (simulations / s)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    if np.isfinite(a[0][-1]) and np.isfinite(s[0][-1]) and s[0][-1] > 0:
        ax.annotate(f"{a[0][-1] / s[0][-1]:.0f}$\\times$", xy=(workers[-1], a[0][-1]),
                    xytext=(workers[-1] * 0.5, a[0][-1] * 1.3), fontsize=11, color="#1f77b4")


def main() -> None:
    lv_a = _curve(LV_WORKERS, _lv_src, ASYNC)
    lv_s = _curve(LV_WORKERS, _lv_src, SYNC)
    rw_a = _curve(RW_WORKERS, _rw_src, ASYNC)
    rw_s = _curve(RW_WORKERS, _rw_src, SYNC)

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9.5})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    _panel_categorical(axL, LV_WORKERS, lv_a, lv_s, "Lotka--Volterra (near-instant simulator)")
    _panel_loglog(axR, RW_WORKERS, rw_a, rw_s, "Realistic workload (costly simulator)")

    # One shared legend beneath both panels.
    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("LV  workers:", LV_WORKERS)
    print("LV  async  :", [round(x, 1) for x in lv_a[0]])
    print("LV  sync   :", [round(x, 1) for x in lv_s[0]])
    print("RW workers:", RW_WORKERS)
    print("RW async  :", [round(x, 2) for x in rw_a[0]])
    print("RW sync   :", [round(x, 2) for x in rw_s[0]])


if __name__ == "__main__":
    main()
