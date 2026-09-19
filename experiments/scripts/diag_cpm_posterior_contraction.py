#!/usr/bin/env python3
"""Did the run's posterior contract, and does it match the screening forecast?

The forecast in `.plans/cpm_setup_proposal_2026-09-19.md` was read off a screening
corpus by rejection ABC (`diag_cpm_posterior_forecast.py`).  This reads the SAME
quantities off a real async ABC run's `raw_results.csv`, so the two are directly
comparable: contraction against the prior, bias against the truth, coverage of the
90% interval, and the posterior correlation between the two parameters --
everything in units of the prior range, which is what the parameters are stored in.

The estimator is the reported one: the retroactive AMIS posterior, read from the
`posterior_weight` column that `propulate_abc` writes from
`ABCPMC.extract_posterior`.  The unweighted top-k archive is printed beside it
because it is the quantity the rejection-ABC forecast is literally about, and
because a large gap between them is itself a finding.

    python experiments/scripts/diag_cpm_posterior_contraction.py \
        --results <run>/data/raw_results.csv \
        --config experiments/configs/cellular_potts_two_param.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

# sd of a uniform draw on the unit interval -- the yardstick every contraction is
# against, and the same constant diag_cpm_posterior_forecast.py uses.
PRIOR_SD = 1.0 / np.sqrt(12.0)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       quantiles: Sequence[float]) -> np.ndarray:
    """Quantiles of a weighted sample, by the usual interpolated CDF."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= np.sum(weights)
    return np.interp(np.asarray(quantiles, dtype=float), cdf, values)


def summarise(theta: np.ndarray, weights: np.ndarray, truth: np.ndarray,
              names: Sequence[str]) -> Dict:
    """Posterior location, spread and coverage, all in units of the prior range."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    ess = float(1.0 / np.sum(weights ** 2))
    out: Dict = {"n": int(len(theta)), "ess": ess}
    for j, name in enumerate(names):
        column = theta[:, j]
        mean = float(np.sum(weights * column))
        # ddof-style correction for a weighted sample, via the ESS.
        var = float(np.sum(weights * (column - mean) ** 2))
        sd = float(np.sqrt(var * ess / max(ess - 1.0, 1.0)))
        median, lo, hi = _weighted_quantile(column, weights, [0.5, 0.05, 0.95])
        out[name] = dict(
            median=float(median),
            bias=float(median - truth[j]),
            sd=sd,
            contraction=1.0 - sd / PRIOR_SD,
            covered=bool(lo <= truth[j] <= hi),
            interval=[float(lo), float(hi)],
        )
    if theta.shape[1] == 2 and len(theta) > 2:
        means = np.array([np.sum(weights * theta[:, j]) for j in range(2)])
        centred = theta - means
        cov = float(np.sum(weights * centred[:, 0] * centred[:, 1]))
        sds = [np.sqrt(np.sum(weights * centred[:, j] ** 2)) for j in range(2)]
        out["correlation"] = float(cov / (sds[0] * sds[1])) if min(sds) > 0 else float("nan")
    return out


def truth_in_prior_units(benchmark_cfg: Dict, names: Sequence[str]) -> np.ndarray:
    """The configured truth, normalised the way the recorded parameters are.

    ``true_params_scale: physical`` means the config states simulator units, which
    have to go through the parameter space's own physical_range and scale -- the
    module defaults are for a different parameter space entirely.
    """
    from async_abc.benchmarks.cellular_potts import (
        _resolve_repo_path,
        normalize_cpm_param,
    )

    with open(_resolve_repo_path(benchmark_cfg["parameter_space"])) as f:
        space = json.load(f)["parameters"]
    limits = {n: tuple(space[n]["physical_range"]) for n in names}
    scales = {n: space[n].get("scale", "linear") for n in names}
    physical = str(benchmark_cfg.get("true_params_scale", "physical")) == "physical"
    truth = []
    for name in names:
        value = float(benchmark_cfg[f"true_{name}"])
        truth.append(normalize_cpm_param(name, value, limits, scales) if physical else value)
    return np.asarray(truth)


def load_rows(results_csv: Path) -> List[Dict[str, str]]:
    import csv

    with open(results_csv, newline="") as f:
        return list(csv.DictReader(f))


def run(results_csv: Path, config_path: Path, *, top_k: int,
        methods: Sequence[str] | None) -> Dict:
    with open(config_path) as f:
        cfg = json.load(f)
    benchmark_cfg = cfg["benchmark"]
    rows = load_rows(results_csv)
    if not rows:
        raise ValueError(f"{results_csv} has no rows")
    names = sorted(key.removeprefix("param_") for key in rows[0] if key.startswith("param_"))
    truth = truth_in_prior_units(benchmark_cfg, names)

    print(f"# Posterior contraction -- {results_csv}")
    print(f"prior sd {PRIOR_SD:.4f}; truth (prior-normalised): "
          + ", ".join(f"{n} {t:.3f}" for n, t in zip(names, truth)) + "\n")

    forecast = {"division_rate": 0.91, "cell_volume": 0.64}
    results = {}
    keys = sorted({(r["method"], int(r["replicate"])) for r in rows})
    for method, replicate in keys:
        if methods and method not in methods:
            continue
        group = [r for r in rows
                 if r["method"] == method and int(r["replicate"]) == replicate]
        theta = np.array([[float(r[f"param_{n}"]) for n in names] for r in group])
        losses = np.array([float(r["loss"]) for r in group])
        finite = np.isfinite(losses)
        raw_weights = np.array([float(r["posterior_weight"] or "nan") for r in group])
        usable = finite & np.isfinite(raw_weights) & (raw_weights > 0)

        print(f"## {method} replicate {replicate} -- {len(group)} evaluations, "
              f"{int(finite.sum())} finite, {int(usable.sum())} with a posterior weight")
        entry = {}
        if usable.sum() > 2:
            entry["reported"] = summarise(theta[usable], raw_weights[usable], truth, names)
            _print("reported (AMIS)", entry["reported"], names, forecast)
        else:
            print("   no usable posterior weights -- reported estimator unavailable")
        if finite.sum() > 2:
            best = np.argsort(losses[finite])[:top_k]
            sub = theta[finite][best]
            entry["top_k"] = summarise(sub, np.ones(len(sub)), truth, names)
            _print(f"top-{top_k} archive", entry["top_k"], names, forecast)
        print()
        results[f"{method}_rep{replicate}"] = entry
    return {"truth": {n: float(t) for n, t in zip(names, truth)},
            "parameters": list(names), "runs": results}


def _print(label: str, stats: Dict, names: Sequence[str], forecast: Dict) -> None:
    print(f"   {label}: n {stats['n']}, ESS {stats['ess']:.1f}")
    for name in names:
        s = stats[name]
        expected = forecast.get(name)
        against = f"   (forecast {expected:.0%})" if expected is not None else ""
        print(f"     {name:<14} contraction {s['contraction']:>6.0%}{against}"
              f"   bias {s['bias']:+.3f}   sd {s['sd']:.3f}"
              f"   90% [{s['interval'][0]:.3f}, {s['interval'][1]:.3f}]"
              f"   {'covered' if s['covered'] else 'NOT COVERED'}")
    if "correlation" in stats:
        print(f"     correlation    {stats['correlation']:+.2f}   (forecast +0.04)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True,
                        help="raw_results.csv of the run")
    parser.add_argument("--config", type=Path, required=True,
                        help="the experiment config the run was launched with")
    parser.add_argument("--top-k", type=int, default=100,
                        help="archive size for the unweighted cross-check")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the numbers here")
    args = parser.parse_args()
    out = run(args.results, args.config, top_k=args.top_k, methods=args.methods)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
