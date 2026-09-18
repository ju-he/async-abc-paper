#!/usr/bin/env python3
"""What does the reported posterior cost to produce, in time and memory?

Appendix B states the retrospective AMIS pass takes "about 20 minutes on the
3e7-particle Gaussian history" and is evaluated in fixed-memory chunks. Neither
figure had a committed benchmark behind it, and both matter: the throughput
results exclude this pass by design, so its cost is exactly the difference
between "simulations per second" and "time to a posterior".

This measures it. ``extract_posterior`` is O(n * m * k) with m <= S+1 = 21 fixed
and chunked at 65536 particles, so the prediction is time linear in n and peak
memory flat in n. Both are checked here rather than assumed.

    python diag_retrospective_cost.py [--max-n 1000000] [--k 100]

The history is generated serially from a seed, so this needs no campaign output
and no MPI: ``extract_posterior`` is a pure function of the history it is handed,
and its cost does not depend on how that history was produced.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "propulate"))

from propulate.propagators.abcpmc import ABCPMC  # noqa: E402

LIMITS = {"mu": (-5.0, 5.0)}
POST_SD = 0.1
OUT = Path(__file__).resolve().parents[1] / "data" / "diagnostics"


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def build_history(n: int, k: int, seed: int = 3):
    """Drive the propagator for `n` evaluations and return it with its history."""
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
    return prop, hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-n", type=int, default=1_000_000)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    sizes = [n for n in (25_000, 50_000, 100_000, 250_000, 500_000,
                         1_000_000, 2_000_000, 3_000_000) if n <= args.max_n]
    print(f"building a {max(sizes)}-particle history ...", flush=True)
    prop, hist = build_history(max(sizes), args.k)

    rows = []
    print(f"\n{'n':>10} {'seconds':>9} {'us/1k':>8} {'peak RSS GB':>12} {'ESS':>8}")
    for n in sizes:
        t0 = time.perf_counter()
        _, weights = prop.extract_posterior(hist[:n])
        elapsed = time.perf_counter() - t0
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        rows.append({"n": n, "seconds": elapsed,
                     "us_per_1k": 1e6 * elapsed / n * 1000,
                     "peak_rss_gb": _peak_rss_gb(),
                     "ess": float(1.0 / np.sum(w ** 2))})
        r = rows[-1]
        print(f"{n:>10} {r['seconds']:>9.2f} {r['us_per_1k']:>8.1f} "
              f"{r['peak_rss_gb']:>12.3f} {r['ess']:>8.0f}", flush=True)

    # Linear fit through the origin: the cost model the appendix asserts.
    n_arr = np.array([r["n"] for r in rows], dtype=float)
    t_arr = np.array([r["seconds"] for r in rows], dtype=float)
    slope = float((n_arr * t_arr).sum() / (n_arr ** 2).sum())
    resid = float(np.max(np.abs(t_arr - slope * n_arr) / t_arr))
    print(f"\nlinear fit: {slope * 1e6:.1f} s per 1e6 particles "
          f"(max relative residual {resid:.1%})")
    for target in (1e7, 3e7):
        print(f"  extrapolated to {target:.0e} particles: "
              f"{slope * target / 60:.1f} minutes")

    OUT.mkdir(parents=True, exist_ok=True)
    artifact = OUT / "retrospective_pass_cost.json"
    artifact.write_text(json.dumps({
        "k": args.k, "amis_snapshots": 20, "chunk": 65536,
        "measurements": rows,
        "seconds_per_1e6_particles": slope * 1e6,
        "max_relative_residual_of_linear_fit": resid,
        "extrapolated_minutes_at_1e7": slope * 1e7 / 60,
        "extrapolated_minutes_at_3e7": slope * 3e7 / 60,
        "peak_rss_gb": max(r["peak_rss_gb"] for r in rows),
    }, indent=2) + "\n")
    print(f"\nwrote {artifact}")


if __name__ == "__main__":
    main()
