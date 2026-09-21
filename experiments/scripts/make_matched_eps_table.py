#!/usr/bin/env python3
"""Tolerance at a matched simulation budget, every benchmark (tab:matched-eps, fig_crossover).

The quantity that compares methods on a common footing is the k-th (k=100)
order statistic of each method's own losses after ``n`` simulations: the
bandwidth at which exactly k of its draws would be accepted. Unlike posterior
contraction it needs no reference, and unlike the reported bandwidth it does not
depend on where a schedule happened to stop. Per (benchmark, method, replicate)
this script keeps that statistic on a log grid of ``n`` (arrival order, only
``simulation_attempt`` rows), plus the simulation count in the wall-clock budget
and the mean simulation duration. From those the table derives, per benchmark:

* throughput ratio -- simulations per wall clock, asynchronous over synchronous;
* per-simulation ratio -- eps at the synchronous arm's own count, synchronous
  over asynchronous (>1: the asynchronous sampler is more efficient per draw);
* equal-wall-clock ratio -- eps at the end of both runs;
* cost per simulation -- the crossover variable of fig_crossover.

Streams the production histories of the 2026-07-07 campaign from the scratch
mirror (gaussian_mean 5.6 GB, gandk 8.3 GB, lotka_volterra 3.9 GB; ~25 min over
sshfs) and reads the fixed two-parameter Cellular Potts run from the repo.
``--refresh`` re-vendors ``tab_matched_eps/{matched_eps_curves,matched_eps_summary}.csv``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from async_abc.analysis.reported_posterior import order_statistic_eps
from async_abc.plotting import paper_style as ps

SCRATCH = Path(os.environ.get("ASYNC_ABC_SCRATCH", "/home/juhe/remotes/scratch/herold2/async-abc"))
REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "cpm_two_param_validation"
K = 100
BENCHMARKS = {
    # name: (path, dimension)
    "gaussian_mean": (SCRATCH / "rerun_20260707/gaussian_mean/data/raw_results.csv", 1),
    "gandk": (SCRATCH / "rerun_20260707/gandk/data/raw_results.csv", 4),
    "lotka_volterra": (SCRATCH / "rerun_20260707/lotka_volterra/data/raw_results.csv", 4),
    "cellular_potts": (REPO_DATA / "cpm_two_param_fixed/cellular_potts_two_param/data/raw_results.csv.gz", 2),
}
METHODS = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}
COLS = ["method", "replicate", "loss", "sim_start_time", "sim_end_time", "record_kind"]


def _stream(path: Path) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=COLS, chunksize=2_000_000,
                             dtype={"method": "category", "record_kind": "category"}):
        chunk = chunk[(chunk["record_kind"] == "simulation_attempt") & chunk["method"].isin(METHODS)]
        parts.append(chunk.drop(columns="record_kind"))
    return pd.concat(parts, ignore_index=True)


def curves(name: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for (method, rep), g in df.groupby(["method", "replicate"], observed=True):
        g = g.sort_values("sim_end_time")
        losses = g["loss"].to_numpy(float)
        dur = (g["sim_end_time"] - g["sim_start_time"]).to_numpy(float)
        wall = float(g["sim_end_time"].max())
        n = len(g)
        grid = np.unique(np.concatenate([np.geomspace(2 * K, n, 60).astype(int), [n]]))
        for m in grid:
            e = order_statistic_eps(list(losses[:m]), K)
            if e is not None:
                rows.append(dict(benchmark=name, method=METHODS[str(method)], replicate=int(rep), n=int(m), eps=e,
                                 n_sims=n, wall_s=wall, mean_sim_s=float(np.nanmean(dur))))
    return rows


def aggregate() -> pd.DataFrame:
    rows = []
    for name, (path, _) in BENCHMARKS.items():
        print(f"[{name}] streaming {path} ...", flush=True)
        df = _stream(path)
        rows += curves(name, df)
        print(f"[{name}] {df.groupby('method', observed=True).size().to_dict()} attempts", flush=True)
    return pd.DataFrame(rows)


def _eps_at(sub: pd.DataFrame, n: int) -> float:
    """Median over replicates of the k-th order statistic at budget ``n`` (nearest grid point <= n)."""
    vals = []
    for _, g in sub.groupby("replicate"):
        g = g[g["n"] <= n]
        if not g.empty:
            vals.append(float(g.loc[g["n"].idxmax(), "eps"]))
    return float(np.median(vals)) if vals else float("nan")


def _eps_final(sub: pd.DataFrame) -> float:
    """Median over replicates of the k-th order statistic at each replicate's own final count."""
    vals = [float(g.loc[g["n"].idxmax(), "eps"]) for _, g in sub.groupby("replicate")]
    return float(np.median(vals)) if vals else float("nan")


def summarise(c: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, (_, dim) in BENCHMARKS.items():
        b = c[c["benchmark"] == name]
        if b.empty:
            continue
        a, s = b[b.method == "async"], b[b.method == "sync"]
        per_rep = b.groupby(["method", "replicate"]).agg(n_sims=("n_sims", "first"), wall=("wall_s", "first"),
                                                          sim_s=("mean_sim_s", "first")).reset_index()
        n_a = per_rep[per_rep.method == "async"].n_sims.median()
        n_s = per_rep[per_rep.method == "sync"].n_sims.median()
        n_match = int(min(n_a, n_s))
        eps_a_full, eps_s_full = _eps_final(a), _eps_final(s)
        eps_a_m, eps_s_m = _eps_at(a, n_match), _eps_at(s, n_match)
        out.append(dict(benchmark=name, dim=dim,
                        sim_cost_s=float(per_rep.sim_s.median()),
                        wall_s=float(per_rep.wall.median()),
                        n_async=n_a, n_sync=n_s, n_match=n_match,
                        throughput_ratio=n_a / n_s,
                        eps_async_matched=eps_a_m, eps_sync_matched=eps_s_m,
                        per_simulation_ratio=eps_s_m / eps_a_m,
                        eps_async_full=eps_a_full, eps_sync_full=eps_s_full,
                        equal_wall_clock_ratio=eps_s_full / eps_a_full))
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    vdir = ps.DATA_DIR / "tab_matched_eps"
    if args.refresh:
        c = aggregate()
        vdir.mkdir(parents=True, exist_ok=True)
        c.to_csv(vdir / "matched_eps_curves.csv", index=False)
        summ = summarise(c)
        summ.to_csv(vdir / "matched_eps_summary.csv", index=False)
        print(f"vendored {vdir}/matched_eps_{{curves,summary}}.csv")
    else:
        # The summary is a pure function of the vendored curves; recompute it so a
        # change of definition here never leaves a stale summary behind.
        c = pd.read_csv(vdir / "matched_eps_curves.csv")
        summ = summarise(c)
        summ.to_csv(vdir / "matched_eps_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
