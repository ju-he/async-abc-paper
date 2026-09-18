#!/usr/bin/env python3
"""Does the reported posterior's effective sample size grow with the budget?

It does not, and that is worth knowing: on the production Gaussian run the
reported estimator has ESS 261-303 out of 1.1-1.3 MILLION evaluated particles
(2.2e-4 of the history), and the ESS is flat across a 13x range of n within a
single run. That is why the posterior-accuracy curve of
``make_reported_recovery_fig.py`` is flat too -- thirteen times the compute buys
no additional effective particles.

This is the controlled version of that observation. It drives the shipped
propagator serially -- so no MPI, no arrival-order question -- over a grid of
archive size k and budget n, and measures the ESS of ``extract_posterior``.

Result (seed 17, Gaussian mean, sigma_obs/sqrt(n_obs) = 0.1):

        k        n      ESS      ESS/n   ESS/k  eps_final   n<=eps
       50     5000    181.3   3.63e-02    3.63     0.0046       98
       50    20000    127.3   6.36e-03    2.55     0.0008       80
       50    80000    156.7   1.96e-03    3.13     0.0002       93
      100     5000    278.2   5.56e-02    2.78     0.0080      176
      100    20000    186.6   9.33e-03    1.87     0.0013      117
      100    80000    281.3   3.52e-03    2.81     0.0004      167
      200     5000    649.1   1.30e-01    3.25     0.0225      391
      200    20000    521.9   2.61e-02    2.61     0.0036      340
      200    80000    536.2   6.70e-03    2.68     0.0009      345

ESS/k sits at 1.9-3.6 everywhere; ESS/n falls like 1/n. The effective sample
size of the reported posterior is set by the ARCHIVE SIZE, not by the budget.

The mechanism is visible in the last two columns. The estimator is reported at
the tightest bandwidth the run reached, and the bandwidth is chosen by
ESS-retention bisection against the archive, so it tracks the archive downwards:
eps_final falls by more than an order of magnitude from n=5000 to n=80000, and
the number of particles inside it stays at a few multiples of k. Everything
outside is kernel-weighted to near zero. The full history enters the denominator
but not, in any effective sense, the numerator.

This bears on three things the paper says: the flat accuracy curve
(S6.2), the CLT rate condition that the reported configuration was already
measured to miss (S4.3), and the reporting-support sweep, where coverage
improves with a larger reported support M -- consistent with an effective
support of a few k rather than of n.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "propulate"))

from propulate.propagators.abcpmc import ABCPMC  # noqa: E402

LIMITS = {"mu": (-5.0, 5.0)}
SIGMA_OBS, N_OBS = 1.0, 100
POST_SD = SIGMA_OBS / np.sqrt(N_OBS)


def run(k: int, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ybar = float(rng.normal(0.0, POST_SD))
    prop = ABCPMC(LIMITS, k=k, kernel="gaussian", scheduler_type="acceptance_rate",
                  amis_snapshots=20, perturbation_scale=0.8, tol=5.0,
                  rng=random.Random(seed))
    hist = []
    for i in range(n):
        child = prop(hist)
        child.generation = i
        child.loss = abs(float(rng.normal(child.position[0], POST_SD)) - ybar)
        hist.append(child)
    _, weights = prop.extract_posterior(hist)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    eps_final = min(h.tolerance for h in hist if h.tolerance is not None)
    return {
        "ess": float(1.0 / np.sum(w ** 2)),
        "eps_final": float(eps_final),
        "n_within_eps": int(sum(1 for h in hist if h.loss <= eps_final)),
    }


def main() -> None:
    print(f"{'k':>5} {'n':>8} {'ESS':>8} {'ESS/n':>10} {'ESS/k':>7} "
          f"{'eps_final':>10} {'n<=eps':>8}")
    for k in (50, 100, 200):
        for n in (5_000, 20_000, 80_000):
            r = run(k, n, seed=17)
            print(f"{k:>5} {n:>8} {r['ess']:>8.1f} {r['ess']/n:>10.2e} "
                  f"{r['ess']/k:>7.2f} {r['eps_final']:>10.4f} "
                  f"{r['n_within_eps']:>8}")


if __name__ == "__main__":
    main()
