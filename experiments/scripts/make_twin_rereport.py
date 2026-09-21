#!/usr/bin/env python3
"""Reported posterior of the twin campaigns at the schedule's bandwidth and re-reported at eps_(k).

Vendors, per (arm, level, replicate) of the straggler and heterogeneity twin campaigns,
the bandwidth the schedule actually reached, how many bandwidth searches ran, the calls
each rank made, and the reported posterior's W1 to the analytic Gaussian-mean posterior
both at that schedule bandwidth (the default reporting rule) and re-reported at the
k-th order statistic of the run's own losses (the check of the paper's C4 section).

    tab:twin         -> tab_twin/twin_straggler_rereport.csv
    tab:twin-hetero  -> tab_twin_hetero/twin_hetero_rereport.csv

Why: the kernel-aware bandwidth search is throttled to once per ``bisect_interval``
(= k) calls on each rank's own scheduler and is otherwise re-run only when that rank's
cached history is rebuilt after an out-of-order arrival. Under a collective barrier
arrivals are in lockstep and rebuilds do not happen, so the twin's bandwidth stalls at
its throttle points; on a run in which no rank makes k calls it never leaves tol_init.
The estimator itself is fine: re-reporting the same history at eps_(k) restores it.

Campaign output is read from ASYNC_ABC_SCRATCH (see make_twin_tables.py); the default
path draws from the committed CSVs and prints the tabulated medians.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
SCRATCH = Path(os.environ.get("ASYNC_ABC_SCRATCH", "/home/juhe/remotes/scratch/herold2/async-abc"))
PAPER_DATA = HERE.parents[1] / "data" / "paper_figures"
FACTORS = [0, 1, 5, 10, 20]
SIGMAS = ["0p0", "0p5", "1p0", "1p5", "2p0"]

STRAGGLER_ARMS = [
    ("twin_fine", "twin2_20260729/straggler_twin_f{f}"),
    ("twin_coarse", "twin2_20260729/straggler_twinB_f{f}"),
    ("async_matched", "twin3_20260921/straggler_async_sim_f{f}"),
]
HETERO_ARMS = [
    ("twin_fine", "heterotwin_20260730/hetero_twin_s{s}"),
    ("twin_coarse", "heterotwin_20260730/hetero_twinB_s{s}"),
]
HETERO_PUBLISHED = "rerun_20260707/runtime_heterogeneity"


def _score_dir(directory: Path, arm: str, level_of) -> list[dict]:
    from async_abc.analysis.reported_posterior import order_statistic_eps, reported_posterior, weighted_w1
    from async_abc.benchmarks.gaussian_mean import GaussianMean
    from async_abc.io.records import ParticleRecord

    cfg = json.loads((directory / "data" / "metadata.json").read_text())["config"]
    bench, inf = cfg["benchmark"], cfg["inference"]
    limits = {"mu": (bench["prior_low"], bench["prior_high"])}
    bm = GaussianMean(bench)
    reference = bm.analytic_posterior_samples(20_000, seed=0).reshape(-1, 1)
    df = pd.read_csv(directory / "data" / "raw_results.csv", low_memory=False)
    out = []
    for (method, replicate), g in df.groupby(["method", "replicate"]):
        level = level_of(cfg, str(method))
        if level is None:
            continue
        g = g.sort_values("wall_time")
        if arm == "pyabc":
            pop = g[g["record_kind"] == "population_particle"]
            last = pop[pop["generation"] == pop["generation"].max()]
            vals = last[["param_mu"]].to_numpy(float)
            w = last["weight"].to_numpy(float)
            w = None if np.isnan(w).all() else np.nan_to_num(w)
            out.append(dict(arm=arm, level=level, replicate=int(replicate), n=int(len(last)),
                            w1_sched=weighted_w1(vals, w, reference)))
            continue
        g = g[g["record_kind"] == "simulation_attempt"]
        calls = g["worker_id"].value_counts()
        recs = [ParticleRecord(
            method=str(r.method), replicate=int(r.replicate), seed=0, step=i + 1,
            params={"mu": float(r.param_mu)}, loss=float(r.loss),
            weight=None if pd.isna(r.weight) else float(r.weight),
            tolerance=None if pd.isna(r.tolerance) else float(r.tolerance),
            wall_time=float(r.wall_time),
        ) for i, r in enumerate(g.itertuples())]
        kw = dict(k=inf["k"], kernel=inf["kernel"], scheduler_type=inf["scheduler_type"],
                  amis_snapshots=inf["amis_snapshots"], perturbation_scale=inf["perturbation_scale"],
                  tol=inf["tol_init"])
        losses = [r.loss for r in recs]
        tols = [r.tolerance for r in recs if r.tolerance is not None]
        pos, wt = reported_posterior(recs, limits, **kw)
        eps_k = order_statistic_eps(losses, int(inf["k"]))
        pos_k, wt_k = reported_posterior(recs, limits, eps_final=eps_k, **kw)
        order = np.argsort(losses)[: int(inf["k"])]
        arch = np.array([[recs[j].params["mu"]] for j in order], dtype=float)
        out.append(dict(
            arm=arm, level=level, replicate=int(replicate), n=len(recs),
            calls_per_rank_min=int(calls.min()), calls_per_rank_median=float(calls.median()),
            calls_per_rank_max=int(calls.max()),
            eps_sched=float(min(tols)), n_distinct_eps=int(len(set(tols))), eps_k=float(eps_k),
            w1_sched=weighted_w1(pos, wt, reference), ess_sched=float(1.0 / np.sum((wt / wt.sum()) ** 2)),
            w1_rereport=weighted_w1(pos_k, wt_k, reference), ess_rereport=float(1.0 / np.sum((wt_k / wt_k.sum()) ** 2)),
            archive_w1=weighted_w1(arch, None, reference),
        ))
        print(f"  {arm:14} level={level:<4} rep{replicate} n={len(recs):6d} eps_sched={min(tols):.3g} "
              f"W1={out[-1]['w1_sched']:.3f} -> re-reported at eps_(k)={eps_k:.3g}: W1={out[-1]['w1_rereport']:.3f}", flush=True)
    return out


def refresh() -> None:
    rows = []
    for arm, tmpl in STRAGGLER_ARMS:
        for f in FACTORS:
            d = SCRATCH / tmpl.format(f=f)
            rows += _score_dir(d, arm, lambda cfg, m, f=f: float(f))
    frame = pd.DataFrame(rows)
    (PAPER_DATA / "tab_twin").mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_DATA / "tab_twin" / "twin_straggler_rereport.csv", index=False)
    rows = []
    for arm, tmpl in HETERO_ARMS:
        for s in SIGMAS:
            d = SCRATCH / tmpl.format(s=s)
            rows += _score_dir(d, arm, lambda cfg, m, s=s: float(s.replace("p", ".")))
    pub = SCRATCH / HETERO_PUBLISHED

    def sigma_async(cfg, m):
        t = re.search(r"__sigma([0-9.]+)", m)
        return float(t.group(1)) if (t and m.startswith("async")) else None

    def sigma_pyabc(cfg, m):
        t = re.search(r"__sigma([0-9.]+)", m)
        return float(t.group(1)) if (t and m.startswith("abc_smc")) else None

    rows += _score_dir(pub, "async", sigma_async)
    rows += _score_dir(pub, "pyabc", sigma_pyabc)
    frame = pd.DataFrame(rows)
    (PAPER_DATA / "tab_twin_hetero").mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_DATA / "tab_twin_hetero" / "twin_hetero_rereport.csv", index=False)


def report() -> None:
    for name in ("tab_twin/twin_straggler_rereport.csv", "tab_twin_hetero/twin_hetero_rereport.csv"):
        frame = pd.read_csv(PAPER_DATA / name)
        print(f"\n== {name} (medians over replicates)")
        for col in ("eps_sched", "n_distinct_eps", "calls_per_rank_median", "eps_k", "w1_sched", "w1_rereport", "ess_sched", "ess_rereport", "archive_w1"):
            if col in frame:
                print(f"-- {col}")
                print(frame.pivot_table(index="arm", columns="level", values=col, aggfunc="median").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-derive the CSVs from the campaign output on scratch")
    a = ap.parse_args()
    if a.refresh:
        refresh()
    report()
