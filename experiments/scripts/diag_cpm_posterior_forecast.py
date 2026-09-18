#!/usr/bin/env python3
"""What posterior would a CPM setup actually produce? Read it off the screening corpus.

The screening design's LHS stratum is a draw from the prior, and every draw carries its
discrepancy to the reference.  Rejection ABC with the prior as proposal is then nothing but
"keep the draws with the smallest rho" -- so the accepted subset IS a sample from the ABC
posterior of that setup, with no modelling assumption and no further simulation.  That
makes the design questions -- which blocks to weight, how many replicate seeds, how long to
run, where to put the truth -- answerable by reading, before a campaign is committed.

Two honest caveats on the numbers this prints.

  * Rejection ABC against a fixed prior sample is a LOWER bound on what the paper's
    adaptive sampler achieves per simulation; the comparison BETWEEN setups is the point,
    not the absolute contraction.
  * At acceptance q the posterior sample has q x n_lhs members, so the tightest tolerances
    are the least certain.  The effective sample size is printed with every row.

    python experiments/scripts/diag_cpm_posterior_forecast.py \
        --out experiments/data/cpm_screening/cpm_two_t501 --weighting scalars
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_OUT = EXPERIMENTS_DIR / "data" / "cpm_screening" / "cpm_two_t501"
SIZE_BLOCKS = ("log_n", "log_r95")
# sd of a uniform draw on the unit interval: the yardstick every contraction is against.
PRIOR_SD = 1.0 / np.sqrt(12.0)


def build_weights(scheme: str, blocks: Sequence[str]) -> Dict[str, float]:
    if scheme == "equal":
        return {b: 1.0 / len(blocks) for b in blocks}
    if scheme == "scalars":
        size = [b for b in blocks if b in SIZE_BLOCKS]
        if not size:
            raise KeyError(f"corpus has no size blocks; found {sorted(blocks)}")
        return {b: 1.0 / len(size) for b in size}
    raise ValueError(f"unknown weighting '{scheme}'")


def summarise(theta: np.ndarray, truth: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    """Posterior location, spread and coverage, all in units of the prior range."""
    out: Dict[str, Any] = {"n": int(len(theta))}
    for j, name in enumerate(names):
        column = theta[:, j]
        lo, hi = np.quantile(column, [0.05, 0.95])
        out[name] = dict(
            median=float(np.median(column)),
            bias=float(np.median(column) - truth[j]),
            sd=float(np.std(column, ddof=1)) if len(column) > 1 else float("nan"),
            contraction=(1.0 - float(np.std(column, ddof=1)) / PRIOR_SD) if len(column) > 1 else float("nan"),
            covered=bool(lo <= truth[j] <= hi),
            interval=[float(lo), float(hi)],
        )
    if theta.shape[1] == 2 and len(theta) > 2:
        out["correlation"] = float(np.corrcoef(theta[:, 0], theta[:, 1])[0, 1])
    return out


def run(out_dir: Path, *, scheme: str, seeds: int, bins: int, tstep: int | None,
        acceptances: Sequence[float], fit_seed: int,
        reference_mode: str = "median", reference_draws: int = 1) -> Dict[str, Any]:
    from async_abc.benchmarks.cellular_potts import _ensure_nastjapy_on_path

    _ensure_nastjapy_on_path()
    import diag_cpm_screening as screening

    with open(out_dir / "protocol.json") as f:
        protocol = json.load(f)
    recorded = protocol.get("candidates")
    if not recorded:
        raise KeyError(f"{out_dir / 'protocol.json'} has no 'candidates' record")
    for name in list(screening.CANDIDATES):
        if name not in recorded:
            del screening.CANDIDATES[name]
    for name, spec in recorded.items():
        screening.CANDIDATES.setdefault(name, {}).update(spec)
    screening.CAMPAIGN_BLOCKS, screening.TRAJECTORY_BLOCKS, screening.EXTRA_BLOCKS = {}, {}, {}

    rows = screening.load_corpus(out_dir / "corpus")
    if tstep is None:
        tstep = max(int(r["tstep"]) for r in rows)
    points = screening._units(rows, bins=bins, tsteps=[tstep], group=seeds)
    result = screening.screen(points, bins=bins, seed=fit_seed)
    zs, blocks = result["zs"], result["blocks"]
    block_norms, z_ref = result["block_norms"], result["z_ref"]
    names = list(screening.CANDIDATES)
    weights = build_weights(scheme, blocks)

    reference = next(i for i, e in points.items() if e["meta"]["stratum"] == "reference")
    truth = np.asarray([screening._to_unit(n, float(points[reference]["meta"]["params"][n]))
                        for n in names])

    theta = []
    draws = []
    for index, entry in points.items():
        if entry["meta"]["stratum"] != "lhs" or index not in zs:
            continue
        coords = np.asarray([screening._to_unit(n, float(entry["meta"]["params"][n]))
                             for n in names])
        for z in zs[index]:
            theta.append(coords)
            draws.append(z)
    theta = np.asarray(theta)
    if theta.size == 0:
        raise ValueError("corpus has no LHS stratum to draw a posterior from")

    # The observed data is one realisation in a real experiment and an average over
    # reference replicates in a twin one; the two are not interchangeable, and the
    # gap between them is a design decision, so both are measurable here.
    if reference_mode == "median":
        references = [z_ref]
    elif reference_mode == "single":
        pool = zs[reference]
        references = [pool[i % len(pool)] for i in range(reference_draws)]
    else:
        raise ValueError(f"unknown reference mode '{reference_mode}'")
    rho_sets = [np.asarray([screening._rho(z, ref, block_norms, weights) for z in draws])
                for ref in references]
    rho = rho_sets[0]
    order = np.argsort(rho)

    print(f"# Posterior forecast -- {out_dir.name}")
    print(f"weighting '{scheme}' over {len(weights)} of {len(blocks)} blocks, "
          f"{seeds} seed(s) per evaluation, snapshot t={tstep}, "
          f"{len(theta)} prior draws, reference = {reference_mode}"
          + (f" (x{len(references)} realisations)" if len(references) > 1 else "") + "\n")
    print("truth (prior-normalised): " + ", ".join(
        f"{n} {truth[j]:.3f} = {points[reference]['meta']['params'][n]:g}"
        for j, n in enumerate(names)) + "\n")
    header = " | ".join(f"{n} bias | {n} contraction | {n} 90%" for n in names)
    print(f"| accept | ESS | tolerance | {header} | corr |")
    print("|---" * (2 + 3 * len(names) + 2) + "|")

    forecasts: Dict[str, Any] = {}
    for q in acceptances:
        size = max(2, int(round(q * len(theta))))
        per_reference = []
        for values in rho_sets:
            keep = np.argsort(values)[:size]
            per_reference.append(summarise(theta[keep], truth, names))
        stats = per_reference[0]
        cells = []
        for n in names:
            biases = [r[n]["bias"] for r in per_reference]
            contractions = [r[n]["contraction"] for r in per_reference]
            covered = float(np.mean([r[n]["covered"] for r in per_reference]))
            spread = (f" ±{np.std(biases):.3f}" if len(per_reference) > 1 else "")
            cells.append(f"{np.mean(biases):+.3f}{spread} | "
                         f"{np.mean(contractions):.0%} | "
                         + (f"{covered:.0%}" if len(per_reference) > 1
                            else ("yes" if per_reference[0][n]["covered"] else "NO")))
        corr = stats.get("correlation")
        tol = float(np.mean([np.sort(v)[:size].max() for v in rho_sets]))
        print(f"| {q:.1%} | {stats['n']} | {tol:.4f} | " + " | ".join(cells)
              + f" | {'' if corr is None else f'{corr:+.2f}'} |")
        forecasts[f"{q:.4f}"] = dict(
            per_reference=per_reference,
            mean_bias={n: float(np.mean([r[n]["bias"] for r in per_reference])) for n in names},
            mean_contraction={n: float(np.mean([r[n]["contraction"] for r in per_reference]))
                              for n in names},
            coverage={n: float(np.mean([r[n]["covered"] for r in per_reference])) for n in names})
    return dict(protocol=protocol, weighting=scheme, seeds_per_eval=seeds,
                snapshot=tstep, parameters=names, reference_mode=reference_mode,
                reference_draws=len(references),
                truth={n: float(truth[j]) for j, n in enumerate(names)},
                forecasts=forecasts)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--weighting", choices=["scalars", "equal"], default="scalars")
    parser.add_argument("--seeds-per-eval", type=int, default=4)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--tstep", type=int, default=None,
                        help="snapshot to score on (default: the last one in the corpus)")
    parser.add_argument("--acceptance", type=float, nargs="+",
                        default=[0.20, 0.10, 0.05, 0.02, 0.01])
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--reference", choices=["median", "single"], default="median",
                        help="'median' pools every reference replicate (an idealised "
                             "low-noise observation); 'single' uses one reference "
                             "realisation at a time, at the SAME replicate count as an "
                             "evaluation, which is what a production reference is. Only "
                             "'single' estimates coverage, since coverage needs more "
                             "than one observed dataset.")
    parser.add_argument("--reference-draws", type=int, default=8,
                        help="how many single realisations to average the verdict over")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    report = run(out_dir, scheme=args.weighting, seeds=args.seeds_per_eval, bins=args.bins,
                 tstep=args.tstep, acceptances=args.acceptance, fit_seed=args.fit_seed,
                 reference_mode=args.reference, reference_draws=args.reference_draws)
    name = (args.report
            or f"posterior_forecast_{args.weighting}_k{args.seeds_per_eval}"
               f"_{args.reference}.json")
    with open(out_dir / name, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_dir / name}")


if __name__ == "__main__":
    main()
