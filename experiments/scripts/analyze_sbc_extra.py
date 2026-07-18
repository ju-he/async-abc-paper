#!/usr/bin/env python3
"""Extra SBC diagnostics for the multidim (g-and-k) + multimodal (bimodal) studies
(external review concern 6): coverage with binomial CIs, a formal rank-uniformity
test, importance-weight health (ESS / max normalised weight / weight tail), and
--- for the bimodal target --- a mode-coverage diagnostic (does the top-k archive
keep both modes, and in particular the mode where the truth lies?).

Usage:
    python analyze_sbc_extra.py --root /path/to/concern6_20260718 [--out report.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ASYNC = "async_propulate_abc"


def _binom_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    se = (p * (1 - p) / n) ** 0.5
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def _coverage_with_ci(cov_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cov_path)
    lo, hi = zip(*[_binom_ci(r.empirical_coverage, int(r.n_trials)) for r in df.itertuples()])
    df["ci_lo"], df["ci_hi"] = lo, hi
    df["nominal_in_ci"] = [
        (l <= lvl <= h) for l, h, lvl in zip(df.ci_lo, df.ci_hi, df.coverage_level)
    ]
    return df


def _rank_uniformity(ranks_path: Path, n_bins: int = 20) -> pd.DataFrame:
    df = pd.read_csv(ranks_path)
    rows = []
    for (method, param), g in df.groupby(["method", "param"]):
        n_samp = int(g["n_samples"].iloc[0])
        # rank in [0, n_samp]; normalise to [0,1) and chi-square against uniform bins
        u = np.clip(g["rank"].to_numpy(float) / (n_samp + 1), 0, 1 - 1e-9)
        counts, _ = np.histogram(u, bins=n_bins, range=(0, 1))
        expected = len(u) / n_bins
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        pval = float(stats.chi2.sf(chi2, df=n_bins - 1))
        rows.append({"method": method, "param": param, "n_trials": len(u),
                     "chi2": chi2, "uniformity_p": pval,
                     "uniform_ok": pval > 0.05})
    return pd.DataFrame(rows)


def _weight_health(jsonl_path: Path, method: str = _ASYNC) -> pd.DataFrame:
    """Per-trial ESS fraction, max normalised weight, and top-1% weight mass, aggregated."""
    per_param: dict[str, list] = {}
    for line in open(jsonl_path):
        d = json.loads(line)
        if d["method"] != method:
            continue
        w = np.asarray(d["posterior_weights"], dtype=float)
        w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        ess_frac = float(1.0 / np.sum(w ** 2) / len(w))
        max_w = float(w.max())
        k_top = max(1, int(round(0.01 * len(w))))
        tail_mass = float(np.sort(w)[-k_top:].sum())  # mass in top 1% of weights
        per_param.setdefault(d["param"], []).append((ess_frac, max_w, tail_mass))
    rows = []
    for param, vals in per_param.items():
        a = np.asarray(vals)
        rows.append({
            "param": param, "n_trials": len(a),
            "ess_frac_median": float(np.median(a[:, 0])),
            "ess_frac_p05": float(np.quantile(a[:, 0], 0.05)),
            "max_w_median": float(np.median(a[:, 1])),
            "max_w_p95": float(np.quantile(a[:, 1], 0.95)),
            "top1pct_mass_median": float(np.median(a[:, 2])),
        })
    return pd.DataFrame(rows)


def _bimodal_mode_coverage(jsonl_path: Path, method: str = _ASYNC,
                           split: float = 0.5, mass_thresh: float = 0.05) -> dict:
    """Fraction of trials the archive keeps both modes, and keeps the true mode."""
    both = 0
    true_mode = 0
    n = 0
    for line in open(jsonl_path):
        d = json.loads(line)
        if d["method"] != method or d["benchmark"] != "bimodal_mean":
            continue
        s = np.asarray(d["posterior_samples"], dtype=float)
        w = np.asarray(d["posterior_weights"], dtype=float)
        w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
        if w.sum() <= 0:
            w = np.ones_like(s)
        w = w / w.sum()
        neg_mass = float(w[s < -split].sum())
        pos_mass = float(w[s > split].sum())
        n += 1
        if neg_mass > mass_thresh and pos_mass > mass_thresh:
            both += 1
        tv = float(d["true_value"])
        # mode where the truth lies (sign of true_value); is that mode represented?
        if (tv >= 0 and pos_mass > mass_thresh) or (tv < 0 and neg_mass > mass_thresh):
            true_mode += 1
    return {"n_trials": n,
            "frac_both_modes": both / n if n else float("nan"),
            "frac_true_mode_kept": true_mode / n if n else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    lines: list[str] = []
    def emit(s=""):
        lines.append(s)

    emit("# SBC extra diagnostics (concern 6): multidim + multimodal\n")
    for exp in ["sbc_gandk", "sbc_bimodal"]:
        d = root / exp / "data"
        if not (d / "coverage.csv").exists():
            emit(f"## {exp}: MISSING\n")
            continue
        emit(f"## {exp}\n")
        cov = _coverage_with_ci(d / "coverage.csv")
        emit("### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)\n")
        emit(cov[["method", "param", "coverage_level", "empirical_coverage",
                  "ci_lo", "ci_hi", "nominal_in_ci"]].to_string(index=False,
                  float_format=lambda x: f"{x: .3f}"))
        emit("")
        emit("### Rank uniformity (chi-square, 20 bins)\n")
        emit(_rank_uniformity(d / "sbc_ranks.csv").to_string(index=False,
             float_format=lambda x: f"{x: .4f}"))
        emit("")
        emit("### Async importance-weight health (per-trial, aggregated)\n")
        emit(_weight_health(d / "sbc_trials.jsonl").to_string(index=False,
             float_format=lambda x: f"{x: .4f}"))
        emit("")
        if exp == "sbc_bimodal":
            mc = _bimodal_mode_coverage(d / "sbc_trials.jsonl")
            emit("### Bimodal mode coverage (async top-k archive)\n")
            emit(f"trials={mc['n_trials']}  both-modes-kept={mc['frac_both_modes']:.3f}  "
                 f"true-value-mode-kept={mc['frac_true_mode_kept']:.3f}")
            emit("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
