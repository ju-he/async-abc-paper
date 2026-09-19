#!/usr/bin/env python3
"""Rejection ABC on the production code path: draw the prior, keep the smallest rho.

`diag_cpm_posterior_forecast.py` reads a posterior off a *screening* corpus, whose
discrepancy is computed by a feature-space model fitted inside that script. This
draws the same prior but scores every draw through the shipped benchmark -- the
same `CellularPotts` object, feature-space model, replicate averaging and
reference the experiment runs with -- so it answers the question the forecast
cannot: is the forecast right about THIS setup, independently of any sampler?

It is also how the tolerance/acceptance mapping is measured. Nothing else in the
repo knows what discrepancy value corresponds to 2% acceptance on a CPM prior,
and a fixed-tolerance method cannot be configured without it.

Two modes, because the simulating is an MPI job and the reading is not::

    srun python experiments/scripts/diag_cpm_prior_corpus.py \
        --config experiments/configs/cellular_potts_two_param.json \
        --out <scratch>/cpm_prior_corpus --n-draws 2000

    python experiments/scripts/diag_cpm_prior_corpus.py --mode analyse \
        --out <scratch>/cpm_prior_corpus \
        --config experiments/configs/cellular_potts_two_param.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_ACCEPTANCES = (0.20, 0.10, 0.05, 0.02, 0.01)


def prior_draws(n_draws: int, names: Sequence[str], seed: int) -> np.ndarray:
    """The prior sample, identical on every rank so each can take its own slice."""
    return np.random.default_rng(seed).uniform(size=(int(n_draws), len(names)))


def run_simulate(args: argparse.Namespace) -> None:
    from mpi4py import MPI

    from async_abc.benchmarks.cellular_potts import CellularPotts

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    out_dir = Path(args.out).resolve()
    corpus_dir = out_dir / "corpus"
    with open(args.config) as f:
        cfg = json.load(f)
    benchmark_cfg = dict(cfg["benchmark"])
    benchmark_cfg["output_dir"] = str(out_dir / "sims" / f"rank{rank:04d}")

    if rank == 0:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "protocol.json", "w") as f:
            json.dump(dict(config=str(Path(args.config).resolve()),
                           benchmark=benchmark_cfg, n_draws=args.n_draws,
                           draw_seed=args.seed, eval_seed_base=args.eval_seed_base,
                           n_ranks=size), f, indent=2)
    comm.Barrier()

    benchmark = CellularPotts(benchmark_cfg)
    names = sorted(benchmark.limits)
    draws = prior_draws(args.n_draws, names, args.seed)
    mine = list(range(args.n_draws))[rank::size]
    if rank == 0:
        print(f"[prior] {args.n_draws} draws over {size} ranks "
              f"({len(mine)} on rank 0), parameters {names}", flush=True)

    written = failures = 0
    with open(corpus_dir / f"rank_{rank:04d}.jsonl", "w", encoding="utf-8") as sink:
        for index in mine:
            params = {name: float(draws[index, j]) for j, name in enumerate(names)}
            started = time.time()
            # The evaluation seed is a function of the draw index alone, so the
            # corpus is reproducible from protocol.json and a rerun on a
            # different rank count gives the same simulations.
            loss = float(benchmark.simulate(params, seed=args.eval_seed_base + index))
            sink.write(json.dumps(dict(index=index, params=params, loss=loss,
                                       eval_s=time.time() - started)) + "\n")
            written += 1
            if not np.isfinite(loss):
                failures += 1
            if written % 50 == 0:
                sink.flush()
    benchmark.close()

    totals = comm.gather((len(mine), written, failures), root=0)
    if rank == 0:
        n_eval = sum(t[0] for t in totals)
        n_written = sum(t[1] for t in totals)
        n_fail = sum(t[2] for t in totals)
        print(f"[prior] done: {n_eval} draws, {n_written} rows, {n_fail} non-finite",
              flush=True)
        if n_fail > n_eval * 0.1:
            raise RuntimeError(
                f"{n_fail}/{n_eval} evaluations returned a non-finite discrepancy; "
                "refusing to report a posterior on a corpus that broken"
            )


def load_corpus(out_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted((out_dir / "corpus").glob("rank_*.jsonl")):
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise FileNotFoundError(f"no corpus rows under {out_dir / 'corpus'}")
    return rows


def run_analyse(args: argparse.Namespace) -> Dict:
    from diag_cpm_posterior_contraction import (
        PRIOR_SD,
        summarise,
        truth_in_prior_units,
    )

    out_dir = Path(args.out).resolve()
    with open(args.config) as f:
        benchmark_cfg = json.load(f)["benchmark"]
    rows = load_corpus(out_dir)
    names = sorted(rows[0]["params"])
    truth = truth_in_prior_units(benchmark_cfg, names)

    theta = np.array([[row["params"][n] for n in names] for row in rows])
    loss = np.array([row["loss"] for row in rows])
    finite = np.isfinite(loss)
    seconds = np.array([row.get("eval_s", np.nan) for row in rows])

    print(f"# Rejection ABC on the production path -- {out_dir.name}")
    print(f"{len(rows)} prior draws, {int(finite.sum())} finite "
          f"({100 * (1 - finite.mean()):.1f}% failed), "
          f"median {np.nanmedian(seconds):.1f}s per evaluation")
    print("truth (prior-normalised): "
          + ", ".join(f"{n} {t:.3f}" for n, t in zip(names, truth)))
    print(f"discrepancy: median {np.median(loss[finite]):.4f}, "
          f"min {loss[finite].min():.4f}, prior sd {PRIOR_SD:.4f}\n")

    theta, loss = theta[finite], loss[finite]
    order = np.argsort(loss)
    header = ("| acceptance |  tolerance |  ESS | "
              + " | ".join(f"{n} contraction | bias | cov" for n in names)
              + " | corr |")
    print(header)
    print("|" + "---|" * (2 + len(names) * 3 + 2))
    results = {}
    for acceptance in args.acceptances:
        keep = max(3, int(round(acceptance * len(theta))))
        chosen = order[:keep]
        stats = summarise(theta[chosen], np.ones(keep), truth, names)
        tolerance = float(loss[chosen].max())
        results[f"{acceptance:g}"] = dict(stats, tolerance=tolerance, n=int(keep))
        cells = " | ".join(
            f"{stats[n]['contraction']:>12.0%} | {stats[n]['bias']:+.3f} | "
            f"{'yes' if stats[n]['covered'] else 'NO '}" for n in names)
        corr = stats.get("correlation", float("nan"))
        print(f"| {acceptance:>9.0%} | {tolerance:>10.4f} | {keep:>4d} | "
              f"{cells} | {corr:+.2f} |")
    print("\ncontraction is 1 - sd_post/sd_prior; 'cov' is whether the 90% "
          "posterior interval contains the truth.")

    payload = {"parameters": list(names),
               "truth": {n: float(t) for n, t in zip(names, truth)},
               "n_draws": int(len(rows)), "n_finite": int(finite.sum()),
               "median_eval_s": float(np.nanmedian(seconds)),
               "acceptances": results}
    (out_dir / "prior_corpus_posterior.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out_dir / 'prior_corpus_posterior.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["simulate", "analyse"], default="simulate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919,
                        help="seed of the prior draw itself")
    parser.add_argument("--eval-seed-base", type=int, default=5_000_000,
                        help="evaluation seeds are this plus the draw index")
    parser.add_argument("--acceptances", type=float, nargs="+",
                        default=list(DEFAULT_ACCEPTANCES))
    args = parser.parse_args()
    if args.mode == "simulate":
        run_simulate(args)
    else:
        run_analyse(args)


if __name__ == "__main__":
    main()
