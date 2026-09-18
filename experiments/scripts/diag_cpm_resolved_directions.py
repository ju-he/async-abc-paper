#!/usr/bin/env python3
"""How many parameter directions does the CPM discrepancy actually resolve?

Confounding is a property of the discrepancy SURFACE, not of any single feature, and a
per-parameter identifiability score cannot answer it: two parameters can have nearly
parallel response directions and still be separable if the residual difference clears the
Monte Carlo noise.  Around the reference theta the expected discrepancy is quadratic,

    rho(theta) ~ rho0 + (1/2) (theta - theta0)' H (theta - theta0),

so the eigenvectors of H are the directions a sampler can and cannot see, and the question
is how many of them are real.

Reading that off H directly needs the noise at the reference -- where rho is a near-zero
squared distance whose robust scale is badly determined, and an earlier version of this
diagnostic produced numbers that contradicted the rest of the screen because of it.  This
avoids the noise estimate entirely: fit the surface using only the top-r curvature
directions and measure how well it predicts HELD-OUT thetas.  Directions that are real
improve held-out prediction; directions that are noise do not.  Cross-validation supplies
the noise scale, so nothing has to be assumed about it.

Folds are by theta, never by evaluation, so a theta's replicate seeds never straddle a
fold -- otherwise the replicates leak the answer and every r looks equally good.

Run it on a corpus produced by ``diag_cpm_screening.py --mode simulate``::

    python experiments/scripts/diag_cpm_resolved_directions.py \
        --out experiments/data/cpm_screening/campaign_features --seeds-per-eval 4

Measured on that corpus: the shipped equal weighting over nine blocks resolves ONE
direction at held-out R2 0.202, while ``log_n`` and ``log_r95`` alone resolve TWO at 0.825.
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

DEFAULT_OUT = EXPERIMENTS_DIR / "data" / "cpm_screening" / "campaign_features"
SIZE_BLOCKS = ("log_n", "log_r95")
# Blocks the sibling nastjapy campaign contributed; scored here as a weighting, not as a
# feature list, because whether they help is a question about the metric.
CAMPAIGN = ("invasion_ratio", "surface_roughness")
# A gain below this in held-out R2 is not a direction, it is a fitted noise mode.
GAIN_FLOOR = 0.005


def _quadratic_design(displacement: np.ndarray) -> np.ndarray:
    """[1, linear, cross-terms] for a full second-order model."""
    n = displacement.shape[1]
    columns = [np.ones(len(displacement))]
    columns += [displacement[:, a] for a in range(n)]
    for a in range(n):
        for b in range(a, n):
            columns.append(displacement[:, a] * displacement[:, b] * (1.0 if a == b else 2.0))
    return np.column_stack(columns)


def _hessian(beta: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    k = 1 + n
    for a in range(n):
        for b in range(a, n):
            matrix[a, b] = matrix[b, a] = 2.0 * beta[k]
            k += 1
    return matrix


def resolved_directions(displacement: np.ndarray, rho: np.ndarray, groups: np.ndarray,
                        *, folds: int, seed: int) -> Dict[str, Any]:
    """Held-out R^2 of the discrepancy surface restricted to its top-r curvature directions."""
    n = displacement.shape[1]
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    if unique.size < folds:
        raise ValueError(f"{unique.size} thetas is too few for {folds}-fold cross-validation")
    error = np.zeros(n + 1)
    for held in np.array_split(unique, folds):
        test = np.isin(groups, held)
        train = ~test
        beta, *_ = np.linalg.lstsq(_quadratic_design(displacement[train]), rho[train], rcond=None)
        values, vectors = np.linalg.eigh(_hessian(beta, n))
        vectors = vectors[:, np.argsort(np.abs(values))[::-1]]
        for r in range(n + 1):
            if r == 0:
                prediction = np.full(int(test.sum()), rho[train].mean())
            else:
                basis = vectors[:, :r]
                fit, *_ = np.linalg.lstsq(_quadratic_design(displacement[train] @ basis),
                                          rho[train], rcond=None)
                prediction = _quadratic_design(displacement[test] @ basis) @ fit
            error[r] += float(np.sum((rho[test] - prediction) ** 2))
    total = float(np.sum((rho - rho.mean()) ** 2))
    r2 = 1.0 - error / total
    gains = np.diff(r2)
    resolved = int(np.argmax(np.concatenate([gains < GAIN_FLOOR, [True]])))
    return dict(r2=r2, gains=gains, resolved=resolved)


def weightings(blocks: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """The candidate block weightings, as shares of the distance budget."""
    size = [b for b in blocks if b in SIZE_BLOCKS]
    campaign = [b for b in blocks if b in CAMPAIGN]
    rest = [b for b in blocks if b not in SIZE_BLOCKS]
    if not size:
        raise KeyError(f"corpus has no size blocks; found {sorted(blocks)}")
    out = {f"equal ({len(blocks)} blocks, shipped)": {b: 1.0 / len(blocks) for b in blocks},
           "size scalars only": {b: 1.0 / len(size) for b in size}}
    for share in (0.25, 0.50):
        if campaign:
            w = {b: (1.0 - share) / len(size) for b in size}
            w.update({b: share / len(campaign) for b in campaign})
            out[f"scalars + campaign features at {share:.0%}"] = w
    if rest:
        out["size blocks removed"] = {b: 1.0 / len(rest) for b in rest}
    return out


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="screening run directory (must contain corpus/ and protocol.json)")
    parser.add_argument("--seeds-per-eval", type=int, default=4,
                        help="replicate seeds averaged into one evaluation")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--tstep", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from async_abc.benchmarks.cellular_potts import _ensure_nastjapy_on_path

    _ensure_nastjapy_on_path()
    import diag_cpm_screening as screening

    out_dir = Path(args.out).resolve()
    with open(out_dir / "protocol.json") as f:
        protocol = json.load(f)
    recorded = protocol.get("candidates")
    if not recorded:
        raise KeyError(f"{out_dir / 'protocol.json'} has no 'candidates' record; the priors "
                       "this corpus was generated under are unknown")
    for name in list(screening.CANDIDATES):
        if name not in recorded:
            del screening.CANDIDATES[name]
    for name, spec in recorded.items():
        screening.CANDIDATES.setdefault(name, {}).update(spec)

    extras = [b for b in CAMPAIGN]
    screening.TRAJECTORY_BLOCKS, screening.EXTRA_BLOCKS = {}, {}
    rows = screening.load_corpus(out_dir / "corpus")
    available = set().union(*(set(r["features"]) for r in rows[:64]))
    extras = [b for b in extras if b in available]
    screening.CAMPAIGN_BLOCKS = {b: {} for b in extras}

    points = screening._units(rows, bins=args.bins, tsteps=[args.tstep],
                              group=args.seeds_per_eval)
    result = screening.screen(points, bins=args.bins, seed=args.seed,
                              extras=extras or False)
    zs, blocks = result["zs"], result["blocks"]
    block_norms, z_ref = result["block_norms"], result["z_ref"]
    names = list(screening.CANDIDATES)

    reference = next(i for i, e in points.items() if e["meta"]["stratum"] == "reference")

    def coords(index: int) -> np.ndarray:
        params = points[index]["meta"]["params"]
        return np.asarray([screening._to_unit(n, float(params[n])) for n in names])

    origin = coords(reference)
    lhs = [i for i, e in points.items() if e["meta"]["stratum"] == "lhs" and i in zs]
    if not lhs:
        raise ValueError("corpus has no LHS stratum to fit the discrepancy surface on")

    print(f"# Resolved directions -- {out_dir.name}, {len(lhs)} thetas, "
          f"{sum(len(zs[i]) for i in lhs)} evaluations at {args.seeds_per_eval} seed(s) each, "
          f"{len(blocks)} blocks, {len(names)} parameters\n")
    print("| block weighting | " + " | ".join(f"r={r}" for r in range(len(names) + 1))
          + " | resolved |")
    print("|---" * (len(names) + 3) + "|")

    summary: Dict[str, Any] = {}
    for label, weights in weightings(blocks).items():
        displacement, rho, groups = [], [], []
        for index in lhs:
            delta = coords(index) - origin
            for z in zs[index]:
                displacement.append(delta)
                rho.append(screening._rho(z, z_ref, block_norms, weights))
                groups.append(index)
        found = resolved_directions(np.asarray(displacement), np.asarray(rho),
                                    np.asarray(groups), folds=args.folds, seed=args.seed)
        cells = " | ".join(f"{v:.3f}" for v in found["r2"])
        print(f"| {label} | {cells} | **{found['resolved']}** |")
        summary[label] = dict(held_out_r2=[float(v) for v in found["r2"]],
                              resolved=found["resolved"],
                              weights={k: float(v) for k, v in weights.items()})

    best = max(summary, key=lambda k: (summary[k]["resolved"], max(summary[k]["held_out_r2"])))
    print(f"\nBest: **{best}** -- {summary[best]['resolved']} direction(s), "
          f"held-out R2 {max(summary[best]['held_out_r2']):.3f}")
    report = out_dir / f"resolved_directions_k{args.seeds_per_eval}.json"
    with open(report, "w") as f:
        json.dump(dict(protocol=protocol, seeds_per_eval=args.seeds_per_eval,
                       parameters=names, weightings=summary, best=best), f, indent=2)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
