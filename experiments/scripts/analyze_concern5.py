#!/usr/bin/env python3
"""Analyse the concern-5 coupling matrix (T2.1).

Reads the six parameter_bias experiments' summaries from a campaign root and
produces the two comparisons the reviewer asked for:

  (A) AMIS vs no-AMIS under runtime-parameter coupling — the *weighted* posterior
      (mean / variance / 5-95% width / ESS) per sigma, in a mild and a steep
      coupling regime. Tests whether the AMIS reweighting changes the posterior,
      unlike the AMIS-insensitive unweighted archive mean.

  (B) Censored vs drained — the reported (deadline-censored) run vs the
      drain-after-deadline run, per sigma. Movement of the weighted posterior
      mean between them = the deadline-censoring effect the AMIS proposal
      denominator does not correct.

Usage:
    python analyze_concern5.py --root /path/to/concern5_20260718 [--out report.md]

Each experiment dir is expected at <root>/<exp>/data/{gaussian_weighted_posterior_summary,
gaussian_analytic_summary}.csv .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_EXPERIMENTS = [
    "parameter_bias",
    "parameter_bias_no_amis",
    "parameter_bias_drain",
    "parameter_bias_strong",
    "parameter_bias_strong_no_amis",
    "parameter_bias_strong_drain",
]

_ASYNC = "async_propulate_abc"


def _sigma_of(method: str) -> float | None:
    if "__sigma" not in method:
        return None
    try:
        return float(method.split("__sigma", 1)[1])
    except ValueError:
        return None


def _base_of(method: str) -> str:
    return method.split("__", 1)[0]


def _load(root: Path, exp: str, fname: str) -> pd.DataFrame | None:
    p = root / exp / "data" / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["sigma"] = df["method"].map(_sigma_of)
    df["base_method"] = df["method"].map(_base_of)
    return df


def _agg_weighted(root: Path, exp: str) -> pd.DataFrame:
    df = _load(root, exp, "gaussian_weighted_posterior_summary.csv")
    if df is None:
        return pd.DataFrame()
    df = df[df["base_method"] == _ASYNC].copy()
    if df.empty:
        return df
    df["ci90_width"] = df["weighted_posterior_q95"] - df["weighted_posterior_q05"]
    g = (
        df.groupby("sigma")
        .agg(
            w_mean=("weighted_posterior_mean", "mean"),
            w_mean_err=("weighted_mean_abs_error", "mean"),
            w_var=("weighted_posterior_var", "mean"),
            ci90_width=("ci90_width", "mean"),
            ess_frac=("ess_fraction", "mean"),
            max_w=("max_norm_weight", "mean"),
            cover90=("analytic_mean_in_90ci", "mean"),
            n=("replicate", "count"),
        )
        .reset_index()
    )
    g["experiment"] = exp
    return g


def _agg_unweighted(root: Path, exp: str) -> pd.DataFrame:
    df = _load(root, exp, "gaussian_analytic_summary.csv")
    if df is None:
        return pd.DataFrame()
    df = df[df["base_method"] == _ASYNC].copy()
    if df.empty:
        return df
    g = (
        df.groupby("sigma")
        .agg(unw_mean_err=("analytic_posterior_mean_abs_error", "mean"),
             unw_mean=("posterior_mean", "mean"))
        .reset_index()
    )
    g["experiment"] = exp
    return g


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "  (no data)\n"
    show = df[cols].copy()
    return show.to_string(index=False, float_format=lambda x: f"{x: .4f}") + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    weighted = {e: _agg_weighted(root, e) for e in _EXPERIMENTS}
    unweighted = {e: _agg_unweighted(root, e) for e in _EXPERIMENTS}

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)

    emit("# Concern-5 (T2.1) analysis\n")
    emit(f"root: {root}\n")

    # ---- (A) AMIS vs no-AMIS, weighted posterior, per regime ----
    emit("## (A) AMIS vs no-AMIS under coupling — weighted posterior (async arm)\n")
    for regime, amis_exp, noamis_exp in [
        ("MILD (base 0.02s, cap 0.5s)", "parameter_bias", "parameter_bias_no_amis"),
        ("STEEP (base 0.5s, cap 8s)", "parameter_bias_strong", "parameter_bias_strong_no_amis"),
    ]:
        emit(f"### {regime}\n")
        a, na = weighted[amis_exp], weighted[noamis_exp]
        ua, una = unweighted[amis_exp], unweighted[noamis_exp]
        emit("AMIS (m=20) weighted:")
        emit(_fmt(a, ["sigma", "w_mean", "w_mean_err", "w_var", "ci90_width", "ess_frac", "max_w", "cover90", "n"]))
        emit("no-AMIS (m=0) weighted:")
        emit(_fmt(na, ["sigma", "w_mean", "w_mean_err", "w_var", "ci90_width", "ess_frac", "max_w", "cover90", "n"]))
        # side-by-side unweighted-vs-weighted mean error to show AMIS sensitivity
        if not a.empty and not ua.empty:
            m = a.merge(ua[["sigma", "unw_mean_err"]], on="sigma", how="left")
            m["weighted_minus_unweighted_err"] = m["w_mean_err"] - m["unw_mean_err"]
            emit("AMIS: weighted vs unweighted mean-abs-error (does reweighting change the metric?):")
            emit(_fmt(m, ["sigma", "unw_mean_err", "w_mean_err", "weighted_minus_unweighted_err"]))

    # ---- (B) censored vs drained ----
    emit("## (B) Censored vs drained — deadline-censoring effect (async arm, AMIS)\n")
    for regime, cens_exp, drain_exp in [
        ("MILD", "parameter_bias", "parameter_bias_drain"),
        ("STEEP", "parameter_bias_strong", "parameter_bias_strong_drain"),
    ]:
        emit(f"### {regime}: {cens_exp} (censored) vs {drain_exp} (drained)\n")
        c, d = weighted[cens_exp], weighted[drain_exp]
        if c.empty or d.empty:
            emit("  (missing data)\n")
            continue
        m = c.merge(d, on="sigma", suffixes=("_cens", "_drain"))
        m["mean_shift"] = m["w_mean_drain"] - m["w_mean_cens"]
        m["var_ratio"] = m["w_var_drain"] / m["w_var_cens"]
        emit(_fmt(m, ["sigma", "w_mean_cens", "w_mean_drain", "mean_shift",
                      "w_var_cens", "w_var_drain", "var_ratio"]))

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
