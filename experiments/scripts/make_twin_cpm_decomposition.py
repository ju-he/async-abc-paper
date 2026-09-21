#!/usr/bin/env python3
"""Decompose the Cellular Potts twin's slowdown into barrier idle and simulation duration.

``tab:twin-cpm`` reported the barrierized twin 1.9-2.8x slower than the
asynchronous arm on the 50^3 workload and attributed the whole factor, and the
twin's replicate spread, to the barrier. Throughput is utilisation divided by
mean simulation duration, so the ratio factorises exactly:

    T_async / T_twin = (util_async / util_twin) x (dur_twin / dur_async)

The first factor is what the barrier costs (idle at the collective); the
second is the twin's simulations taking longer, which a barrier cannot cause
and which this table does not explain. Each twin generation's straggler factor
``E[max]/E[mean]`` of its own W durations is reported beside it: it is the
model's prediction of the first factor from the twin's own timing.

Reads the twin campaign (``cpmtwin_20260729/scaling_cpm_twin``) and the
asynchronous arm of the CPM scaling campaign (``rerun_20260707/scaling_cpm``,
same seeds) from the scratch mirror; ``--refresh`` re-vendors
``tab_twin_cpm/twin_cpm_decomposition.csv``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from async_abc.plotting import paper_style as ps

SCRATCH = Path(os.environ.get("ASYNC_ABC_SCRATCH", "/home/juhe/remotes/scratch/herold2/async-abc"))
TWIN = SCRATCH / "cpmtwin_20260729" / "scaling_cpm_twin" / "data"
ASYNC = SCRATCH / "rerun_20260707" / "scaling_cpm" / "data"
WORKERS = (48, 96, 192, 384)
COLS = ["method", "replicate", "seed", "worker_id", "sim_start_time", "sim_end_time", "record_kind", "generation"]


def _per_replicate(df: pd.DataFrame, arm: str, W: int) -> list[dict]:
    df = df[df["record_kind"] == "simulation_attempt"].copy()
    df["dur"] = df["sim_end_time"] - df["sim_start_time"]
    rows = []
    for rep, g in df.groupby("replicate"):
        span = float(g["sim_end_time"].max() - g["sim_start_time"].min())
        row = dict(arm=arm, n_workers=W, replicate=int(rep), seed=int(g["seed"].iloc[0]), n_sims=len(g),
                   elapsed_s=span, throughput=len(g) / span,
                   utilisation=float(g["dur"].sum() / (g["worker_id"].nunique() * span)),
                   mean_duration_s=float(g["dur"].mean()), duration_cv=float(g["dur"].std() / g["dur"].mean()))
        if arm == "twin":
            gen = g.groupby("generation")["dur"].agg(["max", "mean"])
            row["generation_straggler_factor"] = float((gen["max"] / gen["mean"]).mean())
        rows.append(row)
    return rows


def aggregate() -> pd.DataFrame:
    rows = []
    for W in WORKERS:
        twin = pd.read_csv(TWIN / f"raw_results_w{W}_k100.csv", usecols=lambda c: c in COLS)
        rows += _per_replicate(twin, "twin", W)
        a = pd.read_csv(ASYNC / f"raw_results_w{W}_k100.csv", usecols=lambda c: c in COLS)
        rows += _per_replicate(a[a["method"].str.startswith("async")], "async", W)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for W, g in df.groupby("n_workers"):
        a, t = g[g.arm == "async"], g[g.arm == "twin"]
        out.append(dict(
            n_workers=W,
            throughput_ratio=a.throughput.median() / t.throughput.median(),
            utilisation_ratio=a.utilisation.median() / t.utilisation.median(),
            duration_ratio=t.mean_duration_s.median() / a.mean_duration_s.median(),
            twin_generation_straggler_factor=t.generation_straggler_factor.mean(),
            async_utilisation=a.utilisation.median(), twin_utilisation=t.utilisation.median(),
            async_duration_s=a.mean_duration_s.median(), twin_duration_s=t.mean_duration_s.median(),
            twin_duration_min_s=t.mean_duration_s.min(), twin_duration_max_s=t.mean_duration_s.max(),
            async_duration_cv=a.duration_cv.median(), twin_duration_cv=t.duration_cv.median(),
        ))
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-derive from the scratch mirror and re-vendor")
    args = parser.parse_args()
    vdir = ps.DATA_DIR / "tab_twin_cpm"
    if args.refresh:
        df = aggregate()
        vdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(vdir / "twin_cpm_decomposition.csv", index=False)
        print(f"vendored {vdir / 'twin_cpm_decomposition.csv'}")
    else:
        df = pd.read_csv(vdir / "twin_cpm_decomposition.csv")
    pd.set_option("display.width", 250)
    print(summarise(df).to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
