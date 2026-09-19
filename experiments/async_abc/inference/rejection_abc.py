"""Rejection ABC: pure-numpy baseline, no external ABC framework needed.

Draws parameters from the uniform prior and keeps the ``k`` closest.

Two modes, and the default changed on 2026-09-19.

``"best_k"`` (default) spends the whole budget and reports the ``k`` smallest
discrepancies -- rejection ABC with a data-driven tolerance, which is the
standard form and the strongest version of the baseline at a given budget. It
needs no tolerance constant, always returns ``k`` particles, and reports at the
k-th order statistic of its own evaluations, which is the same rule the
asynchronous arm uses (``inference.reported_eps_rule``). Every arm of the
comparison is then read at the same kind of bandwidth rather than at whatever
constant or schedule each happened to land on.

``"threshold"`` is the previous behaviour: accept while ``loss < tol_init`` and
stop at ``k`` accepted. It is kept because it is what every run before that date
did, but it is not a good default -- ``tol_init`` is shared with the
asynchronous scheduler, which wants it *loose* because it treats it as a
starting bandwidth and tightens from there. At the shipped values that threshold
admitted 99.1% of prior draws on gaussian_mean, 85.9% on gandk and ~100% on
cellular_potts, so the arm stopped after ~100 evaluations and reported a sample
from the prior -- in figures that present it as a method comparator.

Wall-time semantics
-------------------
When ``max_wall_time_s`` is configured the loop polls the centralised
:class:`._pyabc_common.Deadline` between candidate draws and exits at the
first rank that trips the deadline — *first-rank-hit*, not collective.
Records produced before the deadline are kept; partial post-deadline state
is discarded by the caller.
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

    from ._pyabc_common import Deadline

    max_sims = inference_cfg["max_simulations"]
    k        = inference_cfg.get("k", 100)
    tol_init = inference_cfg.get("tol_init", 10.0)
    max_wall_time_s = inference_cfg.get("max_wall_time_s")
    mode = str(inference_cfg.get("rejection_mode", "best_k"))
    if mode not in ("best_k", "threshold"):
        raise ValueError(
            f"inference.rejection_mode must be 'best_k' or 'threshold', got {mode!r}"
        )

    rng = np.random.default_rng(seed)
    param_names = list(limits.keys())
    lows  = np.array([limits[p][0] for p in param_names])
    highs = np.array([limits[p][1] for p in param_names])

    accepted = []  # list of (params, loss, wall_time, attempt_count)
    sim_count = 0
    deadline = Deadline(max_wall_time_s)
    # run_start retained for backward-compat wall-time records (uses wall clock,
    # not monotonic, so timestamps remain comparable across processes).
    run_start = time.time()

    evaluated = []  # every draw, for the best_k selection
    while sim_count < max_sims:
        if mode == "threshold" and len(accepted) >= k:
            break
        if deadline.expired:
            break
        sim_count += 1
        vals     = rng.uniform(lows, highs)
        params   = {p: float(v) for p, v in zip(param_names, vals)}
        sim_seed = int(rng.integers(0, 2**31))
        loss     = float(simulate_fn(params, seed=sim_seed))
        if mode == "best_k":
            # A failed simulation is an infinitely-bad discrepancy; it must not be
            # selectable as one of the k closest, however small the budget.
            if np.isfinite(loss):
                evaluated.append((params, loss, time.time() - run_start, sim_count))
            accepted_so_far = min(len(evaluated), k)
        elif loss < tol_init:
            accepted.append((params, loss, time.time() - run_start, sim_count))
            accepted_so_far = len(accepted)
        else:
            accepted_so_far = len(accepted)
        if progress is not None:
            progress.update(
                simulations=sim_count,
                accepted=accepted_so_far,
                acceptance_rate=(accepted_so_far / sim_count) if sim_count else 0.0,
            )

    if mode == "best_k":
        # The k closest of everything evaluated: rejection ABC at the tolerance
        # this budget affords, rather than at a constant chosen in advance.
        evaluated.sort(key=lambda item: item[1])
        # Select by loss, but emit in evaluation order: the records are this
        # run's history, and every downstream reader (step monotonicity, the
        # wall-clock quality curves) reads them as a sequence of arrivals.
        accepted = sorted(evaluated[:k], key=lambda item: item[3])

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
            record_kind="accepted_particle",
            time_semantics="event_end",
            attempt_count=attempt_count,
        )
        for i, (p, d, t, attempt_count) in enumerate(accepted)
    ]
