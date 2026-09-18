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
import re
import sys

import numpy as np
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
    _print_quality(frame, "slowdown_factor")
    return frame


# The three quality columns are NOT interchangeable, and reporting only the first
# is how "unchanged posterior quality throughout" got into the paper:
#
#   final_quality_wasserstein           unweighted top-k ARCHIVE vs a point mass at
#                                       the truth. Floored at the posterior's own
#                                       spread -- for this benchmark the analytic
#                                       posterior has sd sigma_obs/sqrt(n_obs)=0.1,
#                                       so a PERFECT posterior still scores
#                                       0.1*sqrt(2/pi) = 0.080. Every arm sits at
#                                       0.07-0.08 because the metric cannot resolve
#                                       anything finer, not because they agree.
#   final_quality_wasserstein_weighted  the estimator the paper actually reports
#                                       (full-history AMIS weights), vs the truth.
#   final_quality_wasserstein_analytic  that same estimator vs the ANALYTIC
#                                       posterior. Goes to ~0 when the posterior is
#                                       right, so it is the one with resolution.
#
# All three are printed, always. Anything quoted in the paper must name which.
_QUALITY_COLUMNS = [
    ("final_quality_wasserstein", "unweighted top-k archive vs truth (point mass)"),
    ("final_quality_wasserstein_weighted", "REPORTED weighted posterior vs truth"),
    ("final_quality_wasserstein_analytic", "REPORTED weighted posterior vs analytic"),
]


def _print_quality(frame: pd.DataFrame, index: str) -> None:
    for column, label in _QUALITY_COLUMNS:
        if column not in frame.columns or frame[column].isna().all():
            print(f"\nmedian {column}: NOT PRESENT in this table's source data")
            continue
        table = _median_by(frame, index, "arm", column)
        print(f"\nmedian {column}\n  ({label}):")
        print(table.round(4).to_string())
    if "n_simulations" in frame.columns:
        sims = _median_by(frame, index, "arm", "n_simulations")
        print("\nmedian n_simulations (the arms are NOT at a matched budget):")
        print(sims.round(0).to_string())


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


def _hetero_quality(root: Path, pattern: str, arm_of) -> list[dict]:
    """Weighted-posterior quality per (arm, sigma, replicate), from raw records.

    Every arm is scored the same way -- the estimator it reports, against the
    ANALYTIC posterior of the Gaussian-mean target the heterogeneity study uses.
    The twin arms already carry a run-time
    ``gaussian_weighted_posterior_summary.csv``; this replays them anyway and
    checks the two agree, because the published asynchronous arm has no such
    file (it ran with ``compute_posterior_weights=False``, so its
    ``posterior_weight`` column is empty) and a comparison is only worth making
    if both sides are built by the same code.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from async_abc.analysis.reported_posterior import reported_posterior, weighted_w1
    from async_abc.benchmarks.gaussian_mean import GaussianMean
    from async_abc.io.records import ParticleRecord

    out = []
    for directory in sorted(root.glob(pattern)):
        cfg = json.loads((directory / "data" / "metadata.json").read_text())["config"]
        bench, inf = cfg["benchmark"], cfg["inference"]
        limits = {"mu": (bench["prior_low"], bench["prior_high"])}
        bm = GaussianMean(bench)
        reference = bm.analytic_posterior_samples(20_000, seed=0).reshape(-1, 1)
        analytic_mean = float(bm.analytic_posterior_mean())

        df = pd.read_csv(directory / "data" / "raw_results.csv", low_memory=False)
        for (method, replicate), g in df.groupby(["method", "replicate"]):
            arm = arm_of(cfg, method)
            if arm is None:
                continue
            g = g.sort_values("wall_time")
            if arm.startswith("pyabc") or arm == "sync":
                pop = g[g["record_kind"] == "population_particle"]
                if pop.empty:
                    continue
                last = pop[pop["generation"] == pop["generation"].max()]
                vals = last[["param_mu"]].to_numpy(float)
                w = last["weight"].to_numpy(float)
                w = None if np.isnan(w).all() else np.nan_to_num(w)
                mean = float(vals.mean()) if w is None else float((vals[:, 0] * w).sum() / w.sum())
                ess = float(len(last)) if w is None else float(1.0 / ((w / w.sum()) ** 2).sum())
                n_used = int(len(last))
                # For a generational method the reported population IS the archive.
                extra = dict(archive_mean_abs_error=abs(mean - analytic_mean),
                             archive_w1_to_analytic=weighted_w1(vals, w, reference))
            else:
                g = g[g["record_kind"] != "population_particle"]
                recs = [ParticleRecord(
                    method=str(r.method), replicate=int(r.replicate), seed=0, step=i + 1,
                    params={"mu": float(r.param_mu)}, loss=float(r.loss),
                    weight=None if pd.isna(r.weight) else float(r.weight),
                    tolerance=None if pd.isna(r.tolerance) else float(r.tolerance),
                    wall_time=float(r.wall_time),
                ) for i, r in enumerate(g.itertuples())]
                if len(recs) < 2 * int(inf["k"]):
                    continue
                pos, wt = reported_posterior(
                    recs, limits, k=inf["k"], kernel=inf["kernel"],
                    scheduler_type=inf["scheduler_type"],
                    amis_snapshots=inf["amis_snapshots"],
                    perturbation_scale=inf["perturbation_scale"], tol=inf["tol_init"],
                )
                vals, w = pos, wt
                wn = wt / wt.sum()
                mean = float((pos[:, 0] * wn).sum())
                ess = float(1.0 / np.sum(wn ** 2))
                n_used = int(len(recs))
                # The top-k ARCHIVE, for comparison. These are different
                # objects and on a short run they disagree completely: the
                # archive is the k best-fitting particles, the full-history
                # estimator reweights everything drawn, prior phase included.
                # The paper's "0.003-0.034" is the archive number.
                order = np.argsort([r.loss for r in recs])[: int(inf["k"])]
                arch = np.array([[recs[j].params["mu"]] for j in order], dtype=float)
                extra = dict(
                    archive_mean_abs_error=abs(float(arch.mean()) - analytic_mean),
                    archive_w1_to_analytic=weighted_w1(arch, None, reference),
                )
            # The twin runs are one sigma per directory; the published run holds
            # all five in one, tagged on the method name.
            tag = re.search(r"__sigma([0-9.]+)", str(method))
            sigma = (float(tag.group(1)) if tag
                     else float(cfg["heterogeneity"]["sigma_levels"][0]))
            out.append(dict(
                arm=arm, sigma=sigma,
                replicate=int(replicate),
                full_history_mean_abs_error=abs(mean - analytic_mean),
                full_history_w1_to_analytic=weighted_w1(vals, w, reference),
                ess=ess, ess_fraction=ess / max(n_used, 1), n_particles_used=n_used,
                **extra,
            ))
    return out


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

    # Posterior quality per arm (review: the vendored table carried throughput
    # only, so the paper's "twin recovers the analytic mean to within
    # 0.003-0.034" had no committed data behind it).
    def _twin_arm(cfg, method):
        return "twin_coarse" if cfg["inference"].get("barrier_every", 1) == 2 else "twin_fine"

    def _published_arm(cfg, method):
        base = re.sub(r"__sigma.*", "", str(method))
        return {ASYNC: "async", SYNC: "pyabc_baseline"}.get(base)

    quality = pd.DataFrame(
        _hetero_quality(HETERO_ROOT, "hetero_twin*_s*", _twin_arm)
        + _hetero_quality(HETERO_PUBLISHED.parent.parent, HETERO_PUBLISHED.parts[-2],
                          _published_arm)
    )
    if not quality.empty:
        frame = frame.merge(quality, on=["arm", "sigma", "replicate"], how="left")
    _write("tab_twin_hetero/twin_hetero_raw.csv", frame)

    if not quality.empty:
        print("\ntab:twin-hetero -- reported posterior vs the ANALYTIC posterior:")
        for col in ("archive_mean_abs_error", "full_history_mean_abs_error",
                    "archive_w1_to_analytic", "full_history_w1_to_analytic",
                    "ess_fraction"):
            print(f"\n  median {col}:")
            print(_median_by(quality, "sigma", "arm", col).round(4).to_string())

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
    # No quality column. Table~\ref{tab:twin-cpm} makes no posterior-quality
    # claim -- it reports throughput and replicate spread -- and Cellular Potts
    # has no reference posterior to score against anyway (its parameters
    # confound; see the benchmark scoping in the paper). The column was present
    # but empty in every one of the 56 rows, which reads as a missing number
    # rather than as an absent claim, so it is dropped.
    keep = [c for c in ("arm", "base_method", "method_variant", "k", "n_workers",
                        "replicate", "seed", "elapsed_wall_time_s", "n_simulations",
                        "throughput_sims_per_s", "worker_utilization")
            if c in frame.columns]
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
