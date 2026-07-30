#!/usr/bin/env python3
"""Vendor the per-replicate data behind the three barrierized-twin tables.

The twin tables are hand-written LaTeX (they are small and their row structure
differs per benchmark), so unlike the figures they have no ``make_*_fig.py`` to
regenerate them. This script supplies the other half of the contract: it writes
the per-replicate rows each table is computed from under
``experiments/data/paper_figures/`` and prints the tabulated values, so the
numbers in the paper can be checked against committed data without the cluster.

    tab:twin         -> tab_twin/twin_straggler_raw.csv
    tab:twin-hetero  -> tab_twin_hetero/twin_hetero_raw.csv
    tab:twin-cpm     -> tab_twin_cpm/twin_cpm_raw.csv

All three compare arms as *rates* -- simulations per second of active
wall-clock -- because only the barrier arm can be simulation-limited: a
collective barrier needs identical per-rank call counts, so a first-rank-hit
wall-clock stop deadlocks it. The asynchronous arm therefore runs wall-limited
and the twin to a fixed simulation count, and the two are never at a matched
budget. See the ``\\paragraph{Isolating synchronization}`` discussion.

The straggler table previously had only the asynchronous arm and the fine-grained
twin vendored; the coarse (baseline-matched granularity) arm is added here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRATCH = Path("/home/juhe/remotes/scratch/herold2/async-abc")
STRAGGLER_ROOT = SCRATCH / "twin2_20260729"
HETERO_ROOT = SCRATCH / "heterotwin_20260730"
CPM_TWIN = SCRATCH / "cpmtwin_20260729" / "scaling_cpm_twin" / "data"
CPM_ASYNC = SCRATCH / "rerun_20260707" / "scaling_cpm" / "data"
HETERO_PUBLISHED = SCRATCH / "rerun_20260707" / "runtime_heterogeneity" / "data"

PAPER_DATA = Path(__file__).resolve().parents[1] / "data" / "paper_figures"
FACTORS = [0, 1, 5, 10, 20]
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
CPM_WORKERS = [48, 96, 192, 384]
ASYNC, SYNC = "async_propulate_abc", "abc_smc_baseline"


def _write(rel: str, frame: pd.DataFrame) -> None:
    out = PAPER_DATA / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(frame.to_csv(index=False))
    print(f"wrote {out}  ({len(frame)} rows)")


def _median_by(frame: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    return frame.pivot_table(index=index, columns=columns, values=values, aggfunc="median")


def straggler() -> pd.DataFrame:
    """Asynchronous arm + both twin granularities on the straggler benchmark."""
    arms = [
        ("async", "straggler_async_wall", None),
        ("twin_fine", "straggler_twin", 1),      # one barrier per W=16 evaluations
        ("twin_coarse", "straggler_twinB", 7),   # per 7W=112, ~ the baseline's population of 100
    ]
    rows = []
    for arm, stem, barrier_every in arms:
        for factor in FACTORS:
            # The async arm is one run over all factors; the twin arms are per-factor.
            sub = STRAGGLER_ROOT / (stem if barrier_every is None else f"{stem}_f{factor}")
            path = sub / "data" / "throughput_vs_slowdown_summary.csv"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            table = pd.read_csv(path)
            table = table[table["slowdown_factor"] == float(factor)]
            if table.empty:
                raise SystemExit(f"no rows for slowdown {factor} in {path}")
            table = table.copy()
            table["arm"] = arm
            table["barrier_every"] = barrier_every
            rows.append(table)
    frame = pd.concat(rows, ignore_index=True)
    _write("tab_twin/twin_straggler_raw.csv", frame)

    thr = _median_by(frame, "slowdown_factor", "arm", "throughput_sims_per_s")
    print("\ntab:twin -- median throughput (sims/s of active wall-clock):")
    print(thr.round(2).to_string())
    print("\nratios (async / twin):")
    for arm in ("twin_fine", "twin_coarse"):
        print(f"  {arm:12}", "  ".join(
            f"{f}x={thr.loc[f, 'async'] / thr.loc[f, arm]:.0f}" for f in thr.index))
    qual = _median_by(frame, "slowdown_factor", "arm", "final_quality_wasserstein")
    print("\nmedian final Wasserstein (all arms, all factors):")
    print(qual.round(4).to_string())
    print(f"  range {qual.min().min():.4f} - {qual.max().max():.4f}")
    return frame


def _hetero_rates(root: Path, pattern: str) -> list[dict]:
    """Per-replicate rate from the per-worker debug summary of each twin arm."""
    rows = []
    for directory in sorted(root.glob(pattern)):
        meta = json.loads((directory / "data" / "metadata.json").read_text())
        cfg = meta.get("config", {})
        barrier_every = cfg.get("inference", {}).get("barrier_every", 1)
        sigma = cfg["heterogeneity"]["sigma_levels"][0]
        debug = pd.read_csv(directory / "data" / "runtime_debug_summary.csv")
        for replicate, group in debug.groupby("replicate"):
            attempts = int(group["n_attempts"].sum())
            wall = float(group["elapsed_wall_s"].max())
            rows.append(dict(
                arm="twin_coarse" if barrier_every == 2 else "twin_fine",
                barrier_every=barrier_every, sigma=float(sigma), replicate=int(replicate),
                n_simulations=attempts, elapsed_wall_s=wall,
                throughput_sims_per_s=attempts / wall,
            ))
    return rows


def heterogeneity() -> pd.DataFrame:
    """Both twin granularities plus the published async and pyABC arms."""
    rows = _hetero_rates(HETERO_ROOT, "hetero_twin*_s*")
    if not rows:
        raise SystemExit(f"no twin arms found under {HETERO_ROOT}")

    debug = pd.read_csv(HETERO_PUBLISHED / "runtime_debug_summary.csv")
    debug["sigma"] = debug["method"].str.extract(r"__sigma([0-9.]+)").astype(float)
    debug["base"] = debug["method"].str.replace(r"__sigma.*", "", regex=True)
    for (base, sigma, replicate), group in debug.groupby(["base", "sigma", "replicate"]):
        attempts = int(group["n_attempts"].sum())
        # Each arm's OWN elapsed wall: the baseline's runs end early at high sigma
        # (~54 s at 1.5, ~52 s at 2 of a 60 s budget), so dividing by its actual
        # elapsed credits it with the higher rate -- conservative for the twin claim.
        wall = float(group["elapsed_wall_s"].max())
        rows.append(dict(
            arm="async" if base == ASYNC else "pyabc_baseline", barrier_every=None,
            sigma=float(sigma), replicate=int(replicate), n_simulations=attempts,
            elapsed_wall_s=wall, throughput_sims_per_s=attempts / wall,
        ))
    frame = pd.DataFrame(rows)
    _write("tab_twin_hetero/twin_hetero_raw.csv", frame)

    thr = _median_by(frame, "sigma", "arm", "throughput_sims_per_s")
    print("\ntab:twin-hetero -- median throughput (sims/s):")
    print(thr.round(2).to_string())
    print("\nratios (async / arm):")
    for arm in ("twin_fine", "twin_coarse", "pyabc_baseline"):
        print(f"  {arm:15}", "  ".join(
            f"s{s}={thr.loc[s, 'async'] / thr.loc[s, arm]:.1f}" for s in thr.index))
    print("\ncoarse/fine speedup:",
          "  ".join(f"s{s}={thr.loc[s, 'twin_coarse'] / thr.loc[s, 'twin_fine']:.2f}"
                    for s in thr.index))
    return frame


def cellular_potts() -> pd.DataFrame:
    """Twin vs asynchronous arm vs the fair (population = W) pyABC baseline."""
    rows = []
    for workers in CPM_WORKERS:
        twin = pd.read_csv(CPM_TWIN / f"throughput_summary_w{workers}_k100.csv")
        twin = twin.copy()
        twin["arm"] = "twin"
        rows.append(twin)

        published = pd.read_csv(CPM_ASYNC / f"throughput_summary_w{workers}_k100.csv")
        async_arm = published[published["base_method"] == ASYNC].copy()
        async_arm["arm"] = "async"
        rows.append(async_arm)

        # The fair baseline's population equals the worker count, so its shard is
        # k=W -- but only where W exceeds the default population: at 48 and 96 the
        # k=100 population already fills every core, so no k=W shard was run and
        # the k=100 one IS the fair comparison (this reproduces the paper's
        # 3.8/4.5/7.8/14.2 sims/s across the four worker counts).
        fair_k = workers if workers >= 192 else 100
        fair_path = CPM_ASYNC / f"throughput_summary_w{workers}_k{fair_k}.csv"
        fair = pd.read_csv(fair_path)
        fair = fair[fair["base_method"] == SYNC].copy()
        fair["arm"] = "pyabc_fair_baseline"
        rows.append(fair)

    frame = pd.concat(rows, ignore_index=True)
    keep = [c for c in ("arm", "base_method", "method_variant", "k", "n_workers",
                        "replicate", "seed", "elapsed_wall_time_s", "n_simulations",
                        "throughput_sims_per_s", "worker_utilization",
                        "final_quality_wasserstein") if c in frame.columns]
    frame = frame[keep]
    _write("tab_twin_cpm/twin_cpm_raw.csv", frame)

    print("\ntab:twin-cpm -- median throughput (sims/s) and replicate spread:")
    print(f"{'W':>5} {'arm':>20} {'n':>3} {'median':>9} {'CV%':>7}")
    thr = {}
    for workers in CPM_WORKERS:
        for arm in ("async", "twin", "pyabc_fair_baseline"):
            series = frame.loc[
                (frame["n_workers"] == workers) & (frame["arm"] == arm),
                "throughput_sims_per_s",
            ]
            thr[(workers, arm)] = series.median()
            cv = 100 * series.std() / series.mean() if len(series) > 1 else float("nan")
            print(f"{workers:5d} {arm:>20} {len(series):3d} {series.median():9.2f} {cv:7.1f}")
    print("\nratios:")
    for arm in ("twin", "pyabc_fair_baseline"):
        print(f"  async / {arm:20}", "  ".join(
            f"w{w}={thr[(w, 'async')] / thr[(w, arm)]:.1f}" for w in CPM_WORKERS))
    return frame


def main() -> None:
    straggler()
    heterogeneity()
    cellular_potts()


if __name__ == "__main__":
    sys.exit(main())
