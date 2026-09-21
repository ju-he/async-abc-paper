#!/usr/bin/env python3
"""Cellular Potts production comparison (tab:cpm-production).

One table for the three methods on the two-parameter Cellular Potts setup, at
both benchmark sizes and, at 50^3, with the bandwidth defect (``tol_init`` 10.0)
and with it fixed (``tol_init`` 0.1). Every quantity is read off the stored
records, and every cross-method budget counts **only** ``simulation_attempt``
rows: the synchronous baseline also writes ``population_particle`` rows
(accepted, so pre-filtered to small losses) and mixing them into a budget
silently flatters it.

Per (run, method, replicate):

* ``n_sims`` / ``throughput`` -- simulations completed in the 3600 s wall clock.
* ``utilisation`` / ``runtime_cv`` -- worker busy fraction and the in-run
  coefficient of variation of the simulation runtime (the straggler-factor input).
* ``eps_full`` -- the k-th (k=100) order statistic of the run's own losses: the
  bandwidth at which exactly k of its draws would be accepted. This is the
  cross-method quantity, because it describes where the sampler put its draws
  and not the bandwidth it happened to report at.
* ``eps_matched`` -- the same statistic over the first ``n_match`` arrivals of
  *both* arms, ``n_match`` being the synchronous arm's median simulation count in
  that run, so that throughput and per-simulation efficiency can be separated.
* ``contraction_*`` / ``covered_*`` / ``ess`` -- the reported posterior:
  the retroactive AMIS estimator (``posterior_weight``) for the asynchronous
  method, the final population for the synchronous baseline, the k=100 best
  draws of a fairly resourced (48-rank) prior corpus for rejection ABC. Units
  are the prior range, as in ``diag_cpm_posterior_contraction.py``.

Data (vendored under ``experiments/data/cpm_two_param_validation``): the three
production runs as gzipped ``raw_results.csv`` (repaired by
``repair_two_param_cpm_records.py``), the synchronous populations with their
pyABC importance weights in ``pyabc_populations.csv.gz`` (the CSV records lost
them, see that script), and the two rejection corpora as tarballs. The
synchronous posterior is its last generation completed inside the wall clock,
weighted. ``--refresh`` recomputes the summary
CSVs under ``experiments/data/paper_figures/tab_cpm_production`` from them;
the default path prints the table from the committed CSVs.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from async_abc.analysis.reported_posterior import order_statistic_eps  # noqa: E402
from async_abc.plotting import paper_style as ps  # noqa: E402
from diag_cpm_posterior_contraction import summarise  # noqa: E402

NAME = "tab_cpm_production"
DATA = Path(__file__).resolve().parents[1] / "data" / "cpm_two_param_validation"
NAMES = ["division_rate", "cell_volume"]
TRUTH_PHYS = {"division_rate": 0.009, "cell_volume": 500.0}
PHYS_RANGE = {"division_rate": (0.001, 0.2), "cell_volume": (200.0, 1200.0)}
K = 100
WALL_S = 3600.0
WORKERS = 48

RUNS = {
    # label: (records, tol_init, box)
    "50_control": ("cpm_two_param_production/cellular_potts_two_param/data/raw_results.csv.gz", 10.0, 50),
    "50_fixed": ("cpm_two_param_fixed/cellular_potts_two_param/data/raw_results.csv.gz", 0.1, 50),
    "80": ("cpm_80_comparison/cellular_potts_two_param_80/data/raw_results.csv.gz", 0.1, 80),
}
CORPORA = {
    "50": ("cpm_fair_rejection.tar.gz", "cpm_fair_rejection/corpus/"),
    "80": ("cpm_80_prior.tar.gz", "cpm_80_prior/corpus/"),
}
METHODS = {"async_propulate_abc": "async", "abc_smc_baseline": "sync"}


def unit_truth() -> np.ndarray:
    """Truth in prior-normalised (log-uniform) coordinates, the units of the records."""
    out = []
    for name in NAMES:
        lo, hi = PHYS_RANGE[name]
        out.append(np.log(TRUTH_PHYS[name] / lo) / np.log(hi / lo))
    return np.array(out)


def _read(root: Path, spec: str) -> pd.DataFrame:
    if "::" in spec:
        tar, member = spec.split("::")
        with tarfile.open(root / tar) as tf:
            return pd.read_csv(io.TextIOWrapper(tf.extractfile(member)))
    return pd.read_csv(root / spec)


def _corpus(root: Path, tar: str, prefix: str) -> pd.DataFrame:
    rows = []
    with tarfile.open(root / tar) as tf:
        for m in tf.getmembers():
            if m.name.startswith(prefix) and m.name.endswith(".jsonl"):
                for line in io.TextIOWrapper(tf.extractfile(m)):
                    r = json.loads(line)
                    rows.append(dict(index=int(r["index"]), loss=float(r["loss"]), eval_s=float(r["eval_s"]),
                                     **{f"param_{k}": float(v) for k, v in r["params"].items()}))
    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def _posterior_row(theta: np.ndarray, w: np.ndarray, truth: np.ndarray) -> dict:
    s = summarise(theta, w, truth, NAMES)
    return {
        "ess": s["ess"],
        **{f"contraction_{n}": s[n]["contraction"] for n in NAMES},
        **{f"covered_{n}": bool(s[n]["covered"]) for n in NAMES},
        **{f"bias_{n}": s[n]["bias"] for n in NAMES},
    }


def replicate_rows(root: Path) -> pd.DataFrame:
    truth = unit_truth()
    pops = pd.read_csv(root / "pyabc_populations.csv.gz")
    rows = []
    for run, (spec, tol_init, box) in RUNS.items():
        df = _read(root, spec)
        att = df[df["record_kind"] == "simulation_attempt"].copy()
        att["dur"] = att["sim_end_time"] - att["sim_start_time"]
        n_match = int(att[att["method"] == "abc_smc_baseline"].groupby("replicate").size().median())
        for method, short in METHODS.items():
            for rep, g in att[att["method"] == method].groupby("replicate"):
                g = g.sort_values("sim_end_time")
                losses = g["loss"].to_numpy(float)
                span = float(g["sim_end_time"].max() - g["sim_start_time"].min())
                row = dict(run=run, box=box, tol_init=tol_init, method=short, replicate=int(rep),
                           n_sims=len(g), n_match=n_match, throughput=len(g) / WALL_S,
                           utilisation=float(g["dur"].sum() / (WORKERS * span)),
                           runtime_cv=float(g["dur"].std() / g["dur"].mean()),
                           mean_runtime_s=float(g["dur"].mean()),
                           eps_full=order_statistic_eps(list(losses), K),
                           eps_matched=order_statistic_eps(list(losses[:n_match]), K))
                if short == "async":
                    w = g["posterior_weight"].to_numpy(float)
                    ok = np.isfinite(w) & (w > 0)
                    theta = g[[f"param_{n}" for n in NAMES]].to_numpy(float)[ok]
                    row.update(_posterior_row(theta, w[ok], truth))
                else:
                    # Last generation completed inside the wall clock (the CSV is
                    # trimmed to the deadline; the history is not), weighted.
                    pop = df[(df["method"] == method) & (df["record_kind"] == "population_particle")
                             & (df["replicate"] == rep)]
                    t_last = int(pop["generation"].max())
                    ref = pops[(pops["run"] == run) & (pops["replicate"] == rep) & (pops["generation"] == t_last)]
                    theta = ref[NAMES].to_numpy(float)
                    row.update(_posterior_row(theta, ref["weight"].to_numpy(float), truth))
                    row["n_generations"] = t_last + 1
                rows.append(row)
    for box, (tar, prefix) in CORPORA.items():
        c = _corpus(root, tar, prefix)
        losses = c["loss"].to_numpy(float)
        best = c.nsmallest(K, "loss")
        theta = best[[f"param_{n}" for n in NAMES]].to_numpy(float)
        row = dict(run=f"{box}_rejection", box=int(box), tol_init=np.nan, method="rejection", replicate=0,
                   n_sims=len(c), n_match=len(c), throughput=np.nan,
                   utilisation=np.nan, runtime_cv=float(c["eval_s"].std() / c["eval_s"].mean()),
                   mean_runtime_s=float(c["eval_s"].mean()),
                   eps_full=order_statistic_eps(list(losses), K), eps_matched=np.nan)
        row.update(_posterior_row(theta, np.ones(len(theta)), truth))
        rows.append(row)
    return pd.DataFrame(rows)


def summary(rep: pd.DataFrame) -> pd.DataFrame:
    num = ["n_sims", "throughput", "utilisation", "runtime_cv", "mean_runtime_s", "eps_full", "eps_matched", "ess"] \
        + [f"contraction_{n}" for n in NAMES] + [f"bias_{n}" for n in NAMES]
    agg = rep.groupby(["run", "box", "method"], sort=False)
    out = (agg[num].mean().add_suffix("_mean")
           .join(agg[num].std(ddof=1).add_suffix("_sd"))
           .join(agg[num].median().add_suffix("_median")))
    out["n_replicates"] = agg.size()
    for n in NAMES:
        out[f"covered_{n}"] = agg[f"covered_{n}"].sum()
    return out.reset_index()


def print_table(rep: pd.DataFrame, summ: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    cols = ["run", "method", "n_replicates", "n_sims_mean", "utilisation_mean", "runtime_cv_mean",
            "eps_full_median", "eps_matched_median", "contraction_division_rate_mean", "contraction_division_rate_sd",
            "contraction_cell_volume_mean", "contraction_cell_volume_sd", "covered_division_rate",
            "covered_cell_volume", "ess_mean"]
    print(summ[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    # the derived ratios the text quotes
    for run in ("50_control", "50_fixed", "80"):
        a = summ[(summ.run == run) & (summ.method == "async")].iloc[0]
        s = summ[(summ.run == run) & (summ.method == "sync")].iloc[0]
        thr = a.n_sims_median / s.n_sims_median
        per = s.eps_matched_median / a.eps_matched_median
        wall = s.eps_full_median / a.eps_full_median
        print(f"\n[{run}] throughput async/sync {thr:.2f}x; per-simulation (eps at matched n) {per:.2f}x; "
              f"eps at equal wall clock {wall:.2f}x; utilisation {a.utilisation_mean:.1%} vs {s.utilisation_mean:.1%} "
              f"({a.utilisation_mean / s.utilisation_mean:.2f}x); in-run runtime CV async {a.runtime_cv_mean:.2f} sync {s.runtime_cv_mean:.2f}")
        box = run.split("_")[0]
        r = summ[(summ.run == f"{box}_rejection")].iloc[0]
        print(f"      vs fair rejection ({int(r.n_sims_mean)} draws): eps {r.eps_full_mean:.3g} -> async {r.eps_full_mean / a.eps_full_median:.1f}x tighter; "
              f"rejection contraction {r.contraction_division_rate_mean:.0%}/{r.contraction_cell_volume_mean:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", nargs="?", const=str(DATA), default=None, metavar="DATA_ROOT",
                        help=f"recompute from the vendored records (default root {DATA}); otherwise print the committed CSVs")
    args = parser.parse_args()
    vdir = ps.DATA_DIR / NAME
    if args.refresh is not None:
        rep = replicate_rows(Path(args.refresh))
        summ = summary(rep)
        vdir.mkdir(parents=True, exist_ok=True)
        rep.to_csv(vdir / "cpm_production_replicates.csv", index=False)
        summ.to_csv(vdir / "cpm_production_summary.csv", index=False)
        print(f"vendored {vdir}/cpm_production_{{replicates,summary}}.csv")
    else:
        rep = pd.read_csv(vdir / "cpm_production_replicates.csv")
        summ = pd.read_csv(vdir / "cpm_production_summary.csv")
    print_table(rep, summ)


if __name__ == "__main__":
    main()
