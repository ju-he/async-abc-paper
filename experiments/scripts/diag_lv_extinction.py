#!/usr/bin/env python3
"""How much of a Lotka-Volterra simulation budget carries any information?

About two percent of it, and the paper did not say so.

``LotkaVolterra.simulate`` returns ``EXTINCTION_LOSS = 1e6`` whenever either
population hits zero before ``T_max``. Under a smooth kernel those particles get
a weight of effectively zero, so an extinct simulation costs a full simulator
call and contributes nothing to the posterior. At the benchmark's configured
true parameters (0.5, 0.025, 0.025, 0.5) with x0=50, y0=100, T_max=30, that is
the usual outcome rather than an edge case.

Measured (seed 0; ``bm.simulate`` as the benchmark actually calls it):

    at the true parameters          survival  2.55%  of 2000 simulations
    over 400 uniform prior draws    survival  1.67%  of 10000 simulations
                                    92.5% of prior thetas never survived once
                                    survival-rate quantiles over theta:
                                      median 0.00, p75 0.00, p90 0.00,
                                      p95 0.04, p99 0.52, max 0.92

The observed dataset is itself survival-conditioned: ``__init__`` re-draws the
observed trajectory (up to ``max_extinction_retries``, default 100) until one
survives, and at these parameters it takes more than eleven attempts. So the ABC
target is p(theta | s_obs, survived) against a survival-conditioned s_obs, which
is at least internally consistent -- the extinct particles' zero weight is the
P(survive | theta) factor of the likelihood, and it belongs there.

What it is not is free. Three consequences worth stating in the paper:

* The throughput and scaling results on this benchmark are unaffected -- an
  extinct simulation costs what a surviving one costs, and that cost is the
  systems quantity under test.
* The posterior-recovery results on this benchmark are a different matter. The
  informative sample is roughly 2% of the nominal budget, so a "1.2M evaluation"
  Lotka-Volterra run carries the information of about 25k, and the comparison
  between methods is substantially a comparison of how each copes with a
  98%-rejection regime.
* A reference posterior for this benchmark has to model the survival factor
  explicitly. Synthetic-likelihood MCMC needs M surviving datasets per theta,
  which at a 1.7% survival rate costs about 60x what the same reference costs on
  a benchmark without the absorbing state.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from async_abc.benchmarks.lotka_volterra import (  # noqa: E402
    EXTINCTION_LOSS,
    LotkaVolterra,
)

CFG = {"name": "lotka_volterra", "observed_data_seed": 42, "T_max": 30.0,
       "x0": 50, "y0": 100, "true_theta1": 0.5, "true_theta2": 0.025,
       "true_theta3": 0.025, "true_theta4": 0.5}
TRUTH = {"theta1": 0.5, "theta2": 0.025, "theta3": 0.025, "theta4": 0.5}


def main(n_theta: int = 400, n_sim: int = 25, n_truth: int = 2000) -> None:
    bm = LotkaVolterra(CFG)
    names = list(bm.limits)
    lo = np.array([bm.limits[n][0] for n in names])
    hi = np.array([bm.limits[n][1] for n in names])

    losses = np.array([bm.simulate(TRUTH, seed=s) for s in range(n_truth)])
    print(f"true parameters: survival {100 * (losses < EXTINCTION_LOSS).mean():.2f}% "
          f"of {n_truth}")

    rng = np.random.default_rng(0)
    rates, pooled = [], []
    for _ in range(n_theta):
        theta = dict(zip(names, rng.uniform(lo, hi)))
        sims = np.array([bm.simulate(theta, seed=int(rng.integers(1 << 30)))
                         for _ in range(n_sim)])
        rates.append(float((sims < EXTINCTION_LOSS).mean()))
        pooled.append(sims)
    rates = np.array(rates)
    pooled = np.concatenate(pooled)

    print(f"\n{n_theta} uniform prior draws x {n_sim} simulations:")
    print(f"  overall survival            {100 * (pooled < EXTINCTION_LOSS).mean():.2f}%")
    print(f"  thetas that never survived  {int((rates == 0).sum())}/{n_theta} "
          f"({100 * (rates == 0).mean():.1f}%)")
    qs = np.quantile(rates, [0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    print("  survival-rate quantiles     "
          + "  ".join(f"p{int(p*100)}={q:.2f}"
                      for p, q in zip([0.5, 0.75, 0.9, 0.95, 0.99, 1.0], qs)))


if __name__ == "__main__":
    main()
