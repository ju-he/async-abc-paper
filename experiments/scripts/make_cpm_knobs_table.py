#!/usr/bin/env python3
"""The two knobs of the reported posterior on Cellular Potts (tab:cpm-knobs, SI).

Three small tables from the two-parameter setup, all asynchronous arms, 48
workers, 3600 s, five replicate seeds where available:

* ``tol_init`` -- the bandwidth the schedule starts from. Its distance to the
  k-th order statistic of the run's own losses is the bandwidth transient the
  reported posterior is stuck in; the sweep shows the transient closing and the
  weak parameter's contraction recovering (10.0: the shipped value; 0.1: the
  production value, about 1/5 of the prior-predictive median discrepancy).
* ``k`` -- the archive size, at ``tol_init`` 0.1. Raising it broadens the
  proposal and loosens the reported bandwidth (eps ~ k^1 in this noise-dominated
  regime), so contraction falls while ESS rises.
* re-report -- one stored history (fixed run, replicate 0) re-reported at
  ``eps_final`` equal to the k-th order statistic for several k, at zero
  simulation cost: the retroactive counterpart of the ``tol_init`` fix, and the
  reporting-k side of the archive-size question.

Inputs: the vendored control and fixed runs, plus the sweep tarballs on scratch
(``cpm_tolsweep_{1,0p1,0p01}.tar.gz``, ``cpm_ksweep_{30,300}.tar.gz``) extracted
under ``--refresh <dir>``. Vendors ``tab_cpm_knobs/{tol_init,k,rereport}.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from async_abc.analysis.reported_posterior import order_statistic_eps, reported_posterior  # noqa: E402
from async_abc.plotting import paper_style as ps  # noqa: E402
from diag_cpm_posterior_contraction import summarise  # noqa: E402
from make_cpm_production_table import NAMES, unit_truth  # noqa: E402
from make_reported_recovery_fig import _records  # noqa: E402

NAME = "tab_cpm_knobs"
DATA = Path(__file__).resolve().parents[1] / "data" / "cpm_two_param_validation"
FIXED = DATA / "cpm_two_param_fixed/cellular_potts_two_param/data/raw_results.csv.gz"
CONTROL = DATA / "cpm_two_param_production/cellular_potts_two_param/data/raw_results.csv.gz"
INFERENCE = dict(k=100, kernel="gaussian", scheduler_type="acceptance_rate", amis_snapshots=20, perturbation_scale=0.8)
LIMITS = {n: (0.0, 1.0) for n in NAMES}
REREPORT_K = [10, 30, 100, 300, 1000]


def _arm(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["method"] == "async_propulate_abc") & (df["record_kind"] == "simulation_attempt")].copy()
    df["source"] = tag
    return df


def _rows(df: pd.DataFrame, *, k_run: int, truth: np.ndarray, **tags) -> list[dict]:
    rows = []
    for (src, rep), g in df.groupby(["source", "replicate"]):
        g = g.sort_values("sim_end_time")
        losses = g["loss"].to_numpy(float)
        w = g["posterior_weight"].to_numpy(float)
        ok = np.isfinite(w) & (w > 0)
        theta = g[[f"param_{n}" for n in NAMES]].to_numpy(float)[ok]
        s = summarise(theta, w[ok], truth, NAMES)
        eps_rep = float(np.nanmin(g["tolerance"].to_numpy(float)))
        eps_k = order_statistic_eps(list(losses), k_run)
        rows.append(dict(**tags, source=src, replicate=int(rep), n_sims=len(g), ess=s["ess"],
                         eps_reported=eps_rep, eps_order_statistic_k=eps_k,
                         eps_order_statistic_100=order_statistic_eps(list(losses), 100),
                         transient_ratio=eps_rep / eps_k if eps_k else np.nan,
                         **{f"contraction_{n}": s[n]["contraction"] for n in NAMES},
                         **{f"covered_{n}": bool(s[n]["covered"]) for n in NAMES}))
    return rows


def aggregate(sweeps: Path):
    truth = unit_truth()
    tol = []
    tol += _rows(_arm(CONTROL, "production_control"), k_run=100, truth=truth, tol_init=10.0)
    tol += _rows(_arm(sweeps / "cpm_tolsweep_1/cellular_potts_tolsweep_1/data/raw_results.csv", "tolsweep"),
                 k_run=100, truth=truth, tol_init=1.0)
    tol += _rows(pd.concat([_arm(FIXED, "production_fixed"),
                            _arm(sweeps / "cpm_tolsweep_0p1/cellular_potts_tolsweep_0p1/data/raw_results.csv", "tolsweep")]),
                 k_run=100, truth=truth, tol_init=0.1)
    tol += _rows(_arm(sweeps / "cpm_tolsweep_0p01/cellular_potts_tolsweep_0p01/data/raw_results.csv", "tolsweep"),
                 k_run=100, truth=truth, tol_init=0.01)
    ks = []
    ks += _rows(_arm(sweeps / "cpm_ksweep_30/cellular_potts_ksweep_30/data/raw_results.csv", "ksweep"), k_run=30, truth=truth, k=30)
    ks += [r for r in tol if r["tol_init"] == 0.1]
    for r in ks:
        r.setdefault("k", 100)
    ks += _rows(_arm(sweeps / "cpm_ksweep_300/cellular_potts_ksweep_300/data/raw_results.csv", "ksweep"), k_run=300, truth=truth, k=300)
    ks = [{k: v for k, v in r.items() if k != "tol_init"} for r in ks]

    # Re-report one history at the order statistic for several k.
    df = pd.read_csv(FIXED)
    g = df[(df["method"] == "async_propulate_abc") & (df["replicate"] == 0)
           & (df["record_kind"] != "population_particle")].sort_values("wall_time")
    recs = _records(g, NAMES)
    losses = g["loss"].to_numpy(float)
    rer = []
    for kr in REREPORT_K:
        eps = order_statistic_eps(list(losses), kr)
        pos, w = reported_posterior(recs, LIMITS, eps_final=eps, **INFERENCE)
        s = summarise(np.asarray(pos, float), np.asarray(w, float), truth, NAMES)
        rer.append(dict(report_k=kr, eps_final=eps, ess=s["ess"], n_records=len(recs),
                        **{f"contraction_{n}": s[n]["contraction"] for n in NAMES},
                        **{f"covered_{n}": bool(s[n]["covered"]) for n in NAMES}))
    return {"tol_init": pd.DataFrame(tol), "k": pd.DataFrame(ks), "rereport": pd.DataFrame(rer)}


def _print(frames) -> None:
    pd.set_option("display.width", 250)
    ff = lambda x: f"{x:.3g}"
    for key, by in (("tol_init", "tol_init"), ("k", "k")):
        df = frames[key]
        agg = df.groupby(by).agg(reps=("replicate", "size"), n_sims=("n_sims", "median"), ess=("ess", "median"),
                                 eps_reported=("eps_reported", "median"), eps_os_k=("eps_order_statistic_k", "median"),
                                 eps_os_100=("eps_order_statistic_100", "median"), transient=("transient_ratio", "median"),
                                 div=("contraction_division_rate", "mean"), div_sd=("contraction_division_rate", "std"),
                                 vol=("contraction_cell_volume", "mean"), vol_sd=("contraction_cell_volume", "std"),
                                 cov_div=("covered_division_rate", "sum"), cov_vol=("covered_cell_volume", "sum"))
        print(f"\n== {key}\n{agg.to_string(float_format=ff)}")
    print(f"\n== re-report of the fixed run, replicate 0\n{frames['rereport'].to_string(index=False, float_format=ff)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", metavar="SWEEPS_DIR", default=None)
    args = parser.parse_args()
    vdir = ps.DATA_DIR / NAME
    if args.refresh:
        frames = aggregate(Path(args.refresh))
        vdir.mkdir(parents=True, exist_ok=True)
        for k, v in frames.items():
            v.to_csv(vdir / f"{k}.csv", index=False)
        print(f"vendored {vdir}/{{tol_init,k,rereport}}.csv")
    else:
        frames = {k: pd.read_csv(vdir / f"{k}.csv") for k in ("tol_init", "k", "rereport")}
    _print(frames)


if __name__ == "__main__":
    main()
