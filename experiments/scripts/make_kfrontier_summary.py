#!/usr/bin/env python3
"""Join the archive size's throughput cost to its calibration effect (tab:k-frontier).

The paper measures the two consequences of the archive size $k$ in different
places: its throughput cost in the strong-scaling study (per-arrival proposal
reconstruction is O(k)) and its calibration effect in the k/S dimension sweep.
Each on its own invites the conclusion that the two trade off. Putting them on
one axis shows they very nearly do not -- the best-calibrated archive is also
within a few percent of the fastest one tested.

Two sources, matched on everything but k:

* throughput -- Lotka-Volterra strong scaling, 180 s wall cap, 5 replicates,
  worker counts {1,4,16,48,144,192,240,288}. k in {100,1000} come from the
  main campaign run and k in {50,200,400,800} from the k-frontier run, which
  uses the same config apart from its k_values.
* calibration -- the d-dimensional Gaussian-mean SBC sweep at S=20, 1000
  trials per cell, d in {1,2,4,8,16}; the vendored ksweep_summary.csv.

The join is on k alone and the two benchmarks differ deliberately: the
calibration side needs a target with a known posterior at every d, and the
throughput side needs the regime where the O(k) cost is most exposed. Because
Lotka-Volterra evaluations take ~2 ms, coordination is the dominant term there
and the relative k cost measured on it is an UPPER bound -- on a cost-bearing
simulator the per-evaluation work amortizes it (cf. the Cellular Potts scaling).

Writes experiments/data/paper_figures/kfrontier_summary.csv. Reads the campaign
output from scratch, so unlike the make_*_fig.py generators it has no vendored
default path; it is a one-shot table builder, re-run only when the runs change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRATCH = Path("/home/juhe/remotes/scratch/herold2/async-abc")
RERUN = SCRATCH / "rerun_20260707" / "scaling" / "data"
KFRONTIER = SCRATCH / "kfrontier_20260729" / "scaling_kfrontier" / "data"
PAPER_DATA = Path(__file__).resolve().parents[1] / "data" / "paper_figures"
KSWEEP = PAPER_DATA / "ksweep_summary.csv"
OUT = PAPER_DATA / "kfrontier_summary.csv"

WORKER_COUNTS = [1, 4, 16, 48, 144, 192, 240, 288]
# k -> which run measured it. Both configs share benchmark, worker counts, wall
# cap (180 s) and replicate count (5); they differ only in scaling.k_values.
SOURCE_FOR_K = {50: KFRONTIER, 100: RERUN, 200: KFRONTIER,
                400: KFRONTIER, 800: KFRONTIER, 1000: RERUN}
ASYNC, SYNC = "async_propulate_abc", "abc_smc_baseline"
# Fair synchronous baseline: population = worker count, so it can occupy every
# core (paper II.8.6). Its shards are named k=W, hence read separately.
FAIR_BASELINE_WORKERS = [144, 192, 240, 288]
REFERENCE_K = 100


def _median_throughput(data_dir: Path, workers: int, k: int, method: str) -> float | None:
    path = data_dir / f"throughput_summary_w{workers}_k{k}.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    series = rows.loc[rows["base_method"] == method, "throughput_sims_per_s"]
    return float(series.median()) if len(series) else None


def _assert_matched_settings() -> None:
    """Fail loudly if the two runs are not comparable on wall cap / replicates."""
    seen: dict[tuple[int, int], tuple[float, int]] = {}
    for k, data_dir in SOURCE_FOR_K.items():
        for workers in WORKER_COUNTS:
            path = data_dir / f"throughput_summary_w{workers}_k{k}.csv"
            if not path.exists():
                continue
            rows = pd.read_csv(path)
            rows = rows[rows["base_method"] == ASYNC]
            if rows.empty:
                continue
            seen[(k, workers)] = (float(rows["max_wall_time_s"].median()), len(rows))
    caps = {cap for cap, _ in seen.values()}
    reps = {n for _, n in seen.values()}
    if caps != {180.0}:
        raise SystemExit(f"wall caps are not matched across the join: {sorted(caps)}")
    if reps != {5}:
        raise SystemExit(f"replicate counts are not matched across the join: {sorted(reps)}")


def _fair_baseline() -> dict[int, float]:
    """Median throughput of the population = worker-count synchronous baseline."""
    baseline = {}
    value = _median_throughput(RERUN, 48, REFERENCE_K, SYNC)
    if value is not None:
        # At 48 workers the k=100 population already exceeds the core count, so
        # the k=100 shard is the fair comparison; there is no k=48 shard.
        baseline[48] = value
    for workers in FAIR_BASELINE_WORKERS:
        value = _median_throughput(RERUN, workers, workers, SYNC)
        if value is None:
            raise SystemExit(f"missing fair-baseline shard w{workers}_k{workers}")
        baseline[workers] = value
    return baseline


def _crossover_workers(k: int, baseline: dict[int, float]) -> int | None:
    """Smallest worker count at which the fair baseline overtakes the async arm."""
    for workers in sorted(baseline):
        async_thr = _median_throughput(SOURCE_FOR_K[k], workers, k, ASYNC)
        if async_thr is not None and async_thr < baseline[workers]:
            return workers
    return None


def main() -> None:
    if not KSWEEP.exists():
        raise SystemExit(f"missing {KSWEEP}")
    _assert_matched_settings()

    sweep = pd.read_csv(KSWEEP)
    sweep["dev"] = sweep["cov"].to_numpy() - sweep["level"].to_numpy()
    # Mean signed deviation per (d, k) at the S=20 we use throughout, then the
    # worst and mean absolute deviation over d -- the same aggregation as
    # tab:ks-sweep, reduced over the dimension axis.
    per_dk = sweep[sweep["S"] == 20].groupby(["d", "k"])["dev"].mean().unstack()

    baseline = _fair_baseline()
    reference = _median_throughput(SOURCE_FOR_K[REFERENCE_K], 48, REFERENCE_K, ASYNC)
    if reference is None:
        raise SystemExit("missing the k=100 reference shard at 48 workers")

    records = []
    for k, data_dir in sorted(SOURCE_FOR_K.items()):
        peak = _median_throughput(data_dir, 48, k, ASYNC)
        if peak is None:
            raise SystemExit(f"missing the 48-worker shard for k={k}")
        abs_dev = per_dk[k].abs() if k in per_dk.columns else None
        crossover = _crossover_workers(k, baseline)
        record = {
            "k": k,
            "throughput_w48": peak,
            "throughput_rel_k100_pct": 100.0 * (peak / reference - 1.0),
            "worst_abs_dev": None if abs_dev is None else float(abs_dev.max()),
            "worst_abs_dev_d": None if abs_dev is None else int(abs_dev.idxmax()),
            "mean_abs_dev": None if abs_dev is None else float(abs_dev.mean()),
            "baseline_overtakes_at_workers": crossover,
            "baseline_overtakes_at_nodes": None if crossover is None else crossover // 48,
        }
        for workers in WORKER_COUNTS:
            record[f"throughput_w{workers}"] = _median_throughput(data_dir, workers, k, ASYNC)
        records.append(record)

    table = pd.DataFrame(records)
    OUT.write_text(table.to_csv(index=False))
    print(f"wrote {OUT}")

    cols = ["k", "throughput_w48", "throughput_rel_k100_pct", "worst_abs_dev",
            "worst_abs_dev_d", "mean_abs_dev", "baseline_overtakes_at_nodes"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # The point of the join: which archive sizes are Pareto-dominated on both
    # axes at once (faster AND better calibrated than the alternative).
    print("\nPareto check (higher throughput is better, lower worst_abs_dev is better):")
    scored = table.dropna(subset=["worst_abs_dev"])
    for _, row in scored.iterrows():
        better = scored[
            (scored["throughput_w48"] >= row["throughput_w48"])
            & (scored["worst_abs_dev"] <= row["worst_abs_dev"])
            & (scored["k"] != row["k"])
        ]
        verdict = "on the frontier" if better.empty else \
            "dominated by k=" + ",".join(str(int(v)) for v in better["k"])
        print(f"  k={int(row['k']):4d}: {verdict}")


if __name__ == "__main__":
    sys.exit(main())
