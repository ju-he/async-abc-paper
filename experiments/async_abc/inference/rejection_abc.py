"""Rejection ABC: pure-numpy baseline, no external ABC framework needed.

Draws parameters from the uniform prior, simulates, and accepts if the
discrepancy is below ``tol_init``.  Stops after ``max_simulations`` total
evaluations or ``k`` accepted particles, whichever comes first.
"""
from typing import Callable, Dict, List

from ..io.paths import OutputDir
from ..io.records import ParticleRecord


def run_rejection_abc(
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    progress=None,
) -> List[ParticleRecord]:
    """Run rejection ABC.

    Parameters
    ----------
    simulate_fn:
        Callable ``(params: dict, seed: int) -> float``.
    limits:
        Search-space limits dict ``{name: (lo, hi)}``.
    inference_cfg:
        ``config["inference"]`` sub-dict.
    output_dir:
        Unused — kept for interface compatibility.
    replicate:
        Replicate index stored in each record.
    seed:
        Base RNG seed.

    Returns
    -------
    List[ParticleRecord]
        At most ``k`` accepted particles (may be fewer if budget exhausted).
    """
    import numpy as np
    import time

    max_sims = inference_cfg["max_simulations"]
    k        = inference_cfg.get("k", 100)
    tol_init = inference_cfg.get("tol_init", 10.0)

    rng = np.random.default_rng(seed)
    param_names = list(limits.keys())
    lows  = np.array([limits[p][0] for p in param_names])
    highs = np.array([limits[p][1] for p in param_names])

    accepted = []  # list of (params, loss, wall_time)
    sim_count = 0
    run_start = time.time()

    while sim_count < max_sims and len(accepted) < k:
        sim_count += 1
        vals     = rng.uniform(lows, highs)
        params   = {p: float(v) for p, v in zip(param_names, vals)}
        sim_seed = int(rng.integers(0, 2**31))
        loss     = float(simulate_fn(params, seed=sim_seed))
        if loss <= tol_init:
            accepted.append((params, loss, time.time() - run_start))
        if progress is not None:
            progress.update(
                simulations=sim_count,
                accepted=len(accepted),
                acceptance_rate=(len(accepted) / sim_count) if sim_count else 0.0,
            )

    n = len(accepted)
    w = 1.0 / n if n > 0 else None
    if progress is not None:
        progress.finish(
            simulations=sim_count,
            accepted=n,
            acceptance_rate=(n / sim_count) if sim_count else 0.0,
            budget=max_sims,
            records=n,
        )

    return [
        ParticleRecord(
            method="rejection_abc",
            replicate=replicate,
            seed=seed,
            step=i + 1,
            params=p,
            loss=d,
            weight=w,
            tolerance=float(tol_init),
            wall_time=t,
        )
        for i, (p, d, t) in enumerate(accepted)
    ]
