#!/usr/bin/env python3
"""Reported-posterior recovery vs. wall-clock time (fig_reported_recovery.pdf).

The companion to ``fig_gaussian_recovery``, and the one that answers the
question the paper actually asks.

``fig_gaussian_recovery`` plots $W_1$ between the *unweighted top-k archive* and
a *point mass* at the true mean. Two things are wrong with that as a measure of
posterior recovery, and both are visible in its own numbers:

* A point-mass target is minimised by a posterior that has collapsed, so it
  ranks an over-concentrated archive *above* a correct one.
* It is floored at the posterior's own spread. Here the analytic posterior has
  standard deviation $\\sigma_{obs}/\\sqrt{n_{obs}} = 0.1$, so an exactly correct
  posterior scores $0.1\\sqrt{2/\\pi} = 0.080$ -- which is where every method in
  that figure sits, and why they are indistinguishable.

This figure instead scores each method's *own reported estimator* against the
*analytic posterior*, so the quantity goes to zero when the posterior is right:

* ``async_propulate_abc`` -- the retroactive AMIS posterior over the evaluated
  history, replayed with ``ABCPMC.extract_posterior`` over the prefix that had
  completed by each checkpoint, i.e. what the run would have reported had it
  stopped there. This is the estimator Theorem 1 is stated for.
* ``abc_smc_baseline`` -- the last SMC generation completed by the checkpoint,
  which is what that method reports.
* ``rejection_abc`` -- its accepted set, uniformly weighted.

``--refresh`` re-derives from the campaign output; the replay is a post-hoc pass
over ``raw_results.csv`` and runs no simulation. It is not cheap (about a minute
of replay per million evaluated particles), so the vendored CSV is the default
path as for every other figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _figdata as fd
from async_abc.analysis.reported_posterior import reported_posterior, weighted_w1
from async_abc.benchmarks.gaussian_mean import GaussianMean
from async_abc.io.records import ParticleRecord
from async_abc.plotting import paper_style as ps

ASYNC, SYNC, REJECT = "async_propulate_abc", "abc_smc_baseline", "rejection_abc"
ORDER = [ASYNC, SYNC]
KEY = {ASYNC: "async", SYNC: "sync", REJECT: "rejection"}
N_CHECKPOINTS = 8
N_REFERENCE = 50_000
N_REFERENCE_MCMC = 20_000


def _records(frame: pd.DataFrame, params: list[str]) -> list[ParticleRecord]:
    """Stored rows -> ParticleRecord, in arrival order."""
    return [
        ParticleRecord(
            method=r.method, replicate=int(r.replicate), seed=0, step=i + 1,
            params={p: float(getattr(r, f"param_{p}")) for p in params},
            loss=float(r.loss),
            weight=None if pd.isna(r.weight) else float(r.weight),
            posterior_weight=(
                None if pd.isna(r.posterior_weight) else float(r.posterior_weight)
            ),
            tolerance=None if pd.isna(r.tolerance) else float(r.tolerance),
            proposal_tolerance=(
                None if not hasattr(r, "proposal_tolerance")
                or pd.isna(r.proposal_tolerance) else float(r.proposal_tolerance)
            ),
            wall_time=float(r.wall_time),
            generation=None if pd.isna(r.generation) else int(r.generation),
        )
        for i, r in enumerate(frame.itertuples())
    ]


def _async_curve(frame, limits, inference, reference, times, params):
    """Replay the reported AMIS posterior over each wall-clock prefix."""
    frame = frame[frame["record_kind"] != "population_particle"]
    recs = _records(frame.sort_values("wall_time"), params)
    walls = np.array([r.wall_time for r in recs])
    out = []
    for t in times:
        n = int(np.searchsorted(walls, t, side="right"))
        if n < max(2 * inference["k"], 200):
            continue
        pos, w = reported_posterior(
            recs[:n], limits,
            k=inference["k"], kernel=inference["kernel"],
            scheduler_type=inference["scheduler_type"],
            amis_snapshots=inference["amis_snapshots"],
            perturbation_scale=inference["perturbation_scale"],
            tol=inference["tol_init"],
        )
        wn = w / w.sum() if w.sum() > 0 else w
        ess = float(1.0 / np.sum(wn ** 2)) if w.sum() > 0 else float("nan")
        out.append({
            "wall_time": float(t), "n_records": n,
            "w1": weighted_w1(pos, w, reference),
            "ess": ess, "ess_fraction": ess / len(wn),
        })
    return out


def _generation_curve(frame, reference, times, params):
    """The last SMC generation completed by each checkpoint, as reported.

    Only ``population_particle`` rows count. The baseline also writes one
    ``simulation_attempt`` row per evaluation -- 4.9M of them against 5,300
    population particles on this benchmark -- and pooling the two would score an
    accumulated cloud of attempts rather than the posterior the method reports.
    """
    frame = frame[frame["record_kind"] == "population_particle"]
    if frame.empty:
        return []
    frame = frame.sort_values(["generation", "wall_time"])
    gen_end = frame.groupby("generation")["wall_time"].max()
    out = []
    for t in times:
        done = gen_end.index[gen_end <= t]
        if len(done) == 0:
            continue
        g = frame[frame["generation"] == done.max()]
        vals = g[[f"param_{p}" for p in params]].to_numpy(float)
        w = g["weight"].to_numpy(float)
        w = None if np.isnan(w).all() else np.nan_to_num(w)
        if w is None:                      # uniform over the population
            ess = float(len(g))
        else:
            wn = w / w.sum()
            ess = float(1.0 / np.sum(wn ** 2))
        out.append({
            "wall_time": float(t), "n_records": int(len(g)),
            "w1": weighted_w1(vals, w, reference),
            "ess": ess, "ess_fraction": ess / len(g),
        })
    return out


def _prefix_curve(frame, reference, times, params):
    """Rejection ABC: its accepted set so far, uniformly weighted."""
    frame = frame.sort_values("wall_time")
    walls = frame["wall_time"].to_numpy(float)
    vals = frame[[f"param_{p}" for p in params]].to_numpy(float)
    out = []
    for t in times:
        n = int(np.searchsorted(walls, t, side="right"))
        if n < 10:
            continue
        out.append({"wall_time": float(t), "n_records": n,
                    "w1": weighted_w1(vals[:n], None, reference),
                    "ess": float(n), "ess_fraction": 1.0})
    return out


BENCHMARKS = {
    "gaussian_mean": {
        "label": "(a) Gaussian mean",
        "reference": "analytic posterior",
        "params": ["mu"],
    },
    "gandk": {
        "label": "(b) g-and-k",
        "reference": "reference posterior",
        "params": ["A", "B", "g", "k"],
    },
}


def _reference(name: str, bench_cfg: dict, n_params: int) -> np.ndarray:
    """Reference draws for a benchmark, on the target the ABC run is aiming at."""
    if name == "gaussian_mean":
        return GaussianMean(bench_cfg).analytic_posterior_samples(
            N_REFERENCE, seed=0
        ).reshape(-1, n_params)
    if name == "gandk":
        from async_abc.benchmarks.gandk import GandK
        # p(theta | s_obs) -- the octile summaries the discrepancy is built
        # from, NOT the full-data posterior; see GandK.reference_posterior_samples.
        return GandK(bench_cfg).reference_posterior_samples(N_REFERENCE_MCMC, seed=0)
    raise KeyError(f"no reference posterior defined for {name}")


def _aggregate_one(root: Path, name: str) -> pd.DataFrame:
    spec = BENCHMARKS[name]
    params = spec["params"]
    exp = root / name
    cfg = json.loads((exp / "data" / "metadata.json").read_text())["config"]
    bench, inference = cfg["benchmark"], cfg["inference"]
    limits_source = {
        "gaussian_mean": lambda b: {"mu": (b["prior_low"], b["prior_high"])},
        "gandk": lambda b: dict(GandKLimits(b)),
    }
    limits = limits_source[name](bench)

    print(f"[{name}] building reference ...", flush=True)
    reference = _reference(name, bench, len(params))

    print(f"[{name}] reading raw_results.csv ...", flush=True)
    wanted = {"method", "replicate", "loss", "weight", "posterior_weight",
              "tolerance", "proposal_tolerance", "wall_time", "generation",
              "record_kind", *(f"param_{p}" for p in params)}
    df = pd.read_csv(exp / "data" / "raw_results.csv",
                     usecols=lambda c: c in wanted,
                     dtype={"method": "category", "record_kind": "category"})

    budget = float(inference.get("max_wall_time_s") or df["wall_time"].max())
    times = np.linspace(budget / N_CHECKPOINTS, budget, N_CHECKPOINTS)

    rows = []
    for (method, replicate), g in df.groupby(["method", "replicate"], observed=True):
        if method == ASYNC:
            curve = _async_curve(g, limits, inference, reference, times, params)
        elif method == SYNC:
            curve = _generation_curve(g, reference, times, params)
        else:
            curve = _prefix_curve(g, reference, times, params)
        for row in curve:
            rows.append({"benchmark": name, "method": method,
                         "replicate": int(replicate), **row})
        print(f"  {method} rep {replicate}: {len(curve)} checkpoints", flush=True)
    return pd.DataFrame(rows)


def GandKLimits(bench_cfg):
    from async_abc.benchmarks.gandk import GandK
    return GandK(bench_cfg).limits


def aggregate(root: Path):
    """Replay every method's reported estimator over a common wall-clock grid."""
    per_rep = pd.concat(
        [_aggregate_one(root, name) for name in BENCHMARKS if (root / name).exists()],
        ignore_index=True,
    )
    summary = (
        per_rep.groupby(["benchmark", "method", "wall_time"])
        .agg(w1_median=("w1", "median"),
             w1_q1=("w1", lambda s: s.quantile(0.25)),
             w1_q3=("w1", lambda s: s.quantile(0.75)),
             ess_median=("ess", "median"),
             ess_fraction_median=("ess_fraction", "median"),
             n_records_median=("n_records", "median"),
             n_replicates=("w1", "size"))
        .reset_index()
    )
    return {"reported_recovery": summary,
            "reported_recovery_per_replicate": per_rep}


def draw(frames):
    df = frames["reported_recovery"]
    benches = [b for b in BENCHMARKS if b in set(df["benchmark"])]
    fig, axes = plt.subplots(2, len(benches), squeeze=False,
                             figsize=ps.fig_size(1.0, aspect=0.75))

    for col, bench in enumerate(benches):
        spec = BENCHMARKS[bench]
        sub = df[df["benchmark"] == bench]
        ax_w, ax_e = axes[0][col], axes[1][col]

        for method in ORDER:
            m = sub[sub["method"] == method].sort_values("wall_time")
            if m.empty:
                continue
            k = KEY[method]
            style = dict(marker=ps.MARKERS[k], color=ps.COLORS[k],
                         mfc=ps.COLORS[k], ls=ps.LINESTYLES[k], label=ps.LABELS[k])
            ax_w.plot(m["wall_time"], m["w1_median"], **style)
            ax_w.fill_between(m["wall_time"], m["w1_q1"], m["w1_q3"],
                              color=ps.COLORS[k], alpha=0.15, linewidth=0)
            ax_e.plot(m["wall_time"], m["ess_median"], **style)

        rej = sub[sub["method"] == REJECT].sort_values("wall_time")
        if len(rej):
            ax_w.text(0.97, 0.94,
                      f"Rejection ABC: $W_1\\approx{rej['w1_median'].iloc[-1]:.3g}$",
                      transform=ax_w.transAxes, ha="right", va="top",
                      color=ps.COLORS["rejection"], fontsize=6.5)

        a = sub[sub["method"] == ASYNC].sort_values("wall_time")
        if len(a):
            ax_e.plot(a["wall_time"], a["n_records_median"], color="0.45",
                      ls=":", lw=0.9)
            ax_e.text(a["wall_time"].iloc[-1], a["n_records_median"].iloc[-1] * 1.4,
                      "evaluated", color="0.45", fontsize=6, ha="right")

        ax_w.set_title(spec["label"], fontsize=8, loc="left")
        ax_w.set_ylabel(rf"$W_1$ to {spec['reference']}" if col == 0 else "")
        ax_e.set_ylabel("effective sample size" if col == 0 else "")
        ax_e.set_xlabel("wall-clock time (s)")
        for ax in (ax_w, ax_e):
            ax.set_yscale("log")
            ax.grid(True, ls=":", lw=0.4, alpha=0.6)
        if col == 0:
            ax_w.legend(frameon=False, loc="lower left", handlelength=1.6)

    fig.tight_layout()
    return fig


def refresh_rejection(rerun_root: Path) -> None:
    """Replace the rejection-ABC rows of the vendored frames with the best-k reruns.

    The rejection arm of the 2026-07-07 campaign thresholded at the shared
    ``tol_init`` and so accepted essentially every draw (a prior sampler, see
    ``rejection_abc.py``). It was re-run alone in ``best_k`` mode (jobs
    14262064/14262065): spend the same budget, keep the k=100 best. Those
    records hold only the accepted set, so the rejection curve is its final
    value at every checkpoint, exactly as the figure already drew it.
    """
    frames = fd.load_vendored("fig_reported_recovery")
    summary, per_rep = frames["reported_recovery"], frames["reported_recovery_per_replicate"]
    new_rows = []
    for name in BENCHMARKS:
        exp = rerun_root / f"rerun_{name}_rej" / f"{name}_rejection_rerun"
        if not exp.exists():
            print(f"[{name}] no rerun under {exp}; keeping the vendored rejection rows")
            continue
        cfg = json.loads((exp / "data" / "metadata.json").read_text())["config"]
        params = BENCHMARKS[name]["params"]
        reference = _reference(name, cfg["benchmark"], len(params))
        df = pd.read_csv(exp / "data" / "raw_results.csv")
        df = df[df["method"] == REJECT]
        times = sorted(summary[(summary.benchmark == name) & (summary.method == ASYNC)]["wall_time"].unique())
        for rep, g in df.groupby("replicate"):
            vals = g[[f"param_{p}" for p in params]].to_numpy(float)
            w1 = weighted_w1(vals, None, reference)
            for t in times:
                new_rows.append({"benchmark": name, "method": REJECT, "replicate": int(rep), "wall_time": float(t),
                                 "n_records": int(len(g)), "w1": float(w1), "ess": float(len(g)), "ess_fraction": 1.0})
        print(f"[{name}] rejection best-k: {df.groupby('replicate').size().to_dict()} accepted per replicate", flush=True)
    new = pd.DataFrame(new_rows)
    per_rep = pd.concat([per_rep[~((per_rep.method == REJECT) & per_rep.benchmark.isin(new.benchmark.unique()))], new],
                        ignore_index=True)
    agg = (new.groupby(["benchmark", "method", "wall_time"])
              .agg(w1_median=("w1", "median"), w1_q1=("w1", lambda s: s.quantile(0.25)),
                   w1_q3=("w1", lambda s: s.quantile(0.75)), ess_median=("ess", "median"),
                   ess_fraction_median=("ess_fraction", "median"), n_records_median=("n_records", "median"),
                   n_replicates=("w1", "size")).reset_index())
    summary = pd.concat([summary[~((summary.method == REJECT) & summary.benchmark.isin(new.benchmark.unique()))], agg],
                        ignore_index=True).sort_values(["benchmark", "method", "wall_time"])
    frames = {"reported_recovery": summary, "reported_recovery_per_replicate": per_rep}
    ps.apply()
    saved = ps.save_paper_figure(draw(frames), "fig_reported_recovery", data=frames)
    print(f"wrote {saved['pdf']}")
    for name in new.benchmark.unique():
        row = agg[(agg.benchmark == name)].iloc[-1]
        print(f"  {name}: rejection W1 to reference {row.w1_median:.3g} [{row.w1_q1:.3g}, {row.w1_q3:.3g}]")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--refresh-rejection":
        refresh_rejection(Path(sys.argv[2]))
        sys.exit(0)
    fd.run("fig_reported_recovery", __doc__, aggregate, draw,
           metadata={"estimator": "each method's own reported posterior",
                     "target": "analytic posterior (not a point mass at the truth)",
                     "async_replay": "ABCPMC.extract_posterior over the wall-clock prefix"})
