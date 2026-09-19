"""Evaluate the estimator the paper actually reports, over time.

Why this module exists
----------------------
:mod:`async_abc.analysis.convergence` measures $W_1$ between the *unweighted
top-k archive* and a point mass at the true parameter. That statistic answers a
different question from the one the paper asks, and on a well-specified
benchmark it has almost no resolution: a point-mass target is minimised by a
posterior that has collapsed, so an over-concentrated archive scores *better*
than the correct answer, and a correct answer scores the posterior's own spread
rather than zero. On the Gaussian-mean straggler configuration the analytic
posterior has standard deviation $0.1$, so an exactly right posterior scores
$0.090$ -- indistinguishable from every arm actually measured there.

This module scores the *reported* estimator instead: the retroactive AMIS
posterior of ``ABCPMC.extract_posterior``, which is what Theorem 1 is stated for
and what SBC and the corner plots use, against a *reference posterior* rather
than against a point mass.

Prefix semantics
----------------
``reported_posterior_curve`` answers "what would this run have reported had it
stopped at wall-clock $t$?" by replaying ``extract_posterior`` over the records
that had completed by $t$. That is faithful to how the estimator is formed at
the end of a real run (``propulate_abc`` sorts the population by completion time,
drops post-deadline completions, and reweights what remains), and it is not the
same as slicing the stored full-history ``posterior_weight`` column: the
cumulative-mixture denominator over a prefix has fewer components than over the
whole history, so the weights genuinely differ.

The replay is in arrival order, matching the implementation. The proofs are
stated in proposal order; that gap is the paper's, not this module's, and is
documented in the appendix on assumption status.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "infer_n_bootstrap",
    "individuals_from_records",
    "order_statistic_eps",
    "reported_posterior",
    "reported_posterior_curve",
    "weighted_w1",
]


def _import_abcpmc():
    """Import the propagator from the sibling checkout without an install."""
    try:
        from propulate.propagators.abcpmc import ABCPMC
        from propulate.population import Individual
    except ImportError:  # pragma: no cover - depends on checkout layout
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "propulate"))
        from propulate.propagators.abcpmc import ABCPMC
        from propulate.population import Individual
    return ABCPMC, Individual


def infer_n_bootstrap(records) -> int:
    """Number of leading bootstrap (uniform-prior) draws, read off ``weight``.

    Runs recorded before ``proposal_tolerance`` existed do not mark where the
    prior phase ended -- but they do not have to be guessed at either. The
    propagator assigns ``weight = 1.0`` to exactly the prior draws (the proposal
    equals the prior, so the importance weight is pi/pi), and they form a
    contiguous leading run. Its length is therefore the boundary, exactly.

    It is *not* ``k``: it is ``k`` plus however many ranks were still in flight
    when the archive filled, so it varies per replicate. Measured on a
    ``k=100``, ``W=16`` run: 112, 113, 113, 115, 115 across five replicates,
    each of which reproduces that replicate's own stored ``posterior_weight`` to
    machine precision.

    The one draw this rule cannot separate is a late prior *fallback* (weight
    also 1.0, but with a stamped tolerance), which ``extract_posterior`` treats
    as archive-phase anyway -- its own documented third approximation, and
    measured at zero occurrences on the reported runs. Only the leading run is
    used, so a late fallback cannot shift the boundary.
    """
    weights = np.array(
        [1.0 if getattr(r, "weight", None) is None else float(r.weight)
         for r in records],
        dtype=float,
    )
    if weights.size == 0:
        return 0
    non_unit = np.flatnonzero(weights != 1.0)
    return int(weights.size if non_unit.size == 0 else non_unit[0])


def order_statistic_eps(losses, k: int) -> float | None:
    """The k-th smallest finite loss: the bandwidth at which exactly ``k`` accept.

    ``extract_posterior`` reports by default at the tightest bandwidth the
    *schedule* reached, which on an expensive simulator is nowhere near where
    the schedule is trying to go. The scheduler's own rule is documented as
    equilibrating near this order statistic -- the acceptance gate holds ε until
    ``population_size`` individuals sit below it -- but it approaches that point
    at a bounded rate, and an expensive run ends inside the transient. Measured
    on the two-parameter Cellular Potts setup: the reported bandwidth had not
    moved from ``tol_init`` after 3,000 evaluations and was still 68x above this
    statistic at 13,000, which cost 24 points of posterior contraction on the
    weaker of the two parameters at no saving in simulation.

    It is the natural rule for a *comparison* as well as for one run: it is the
    tolerance a rejection sampler with the same budget and the same archive size
    would report at, so every arm is reported on the same footing rather than on
    whatever bandwidth its own schedule happened to reach.

    Returns ``None`` when fewer than ``k`` finite losses exist, which leaves the
    caller on the default (schedule) bandwidth rather than inventing one.
    """
    finite = np.asarray([x for x in np.asarray(losses, dtype=float) if np.isfinite(x)])
    if finite.size < int(k) or int(k) < 1:
        return None
    return float(np.partition(finite, int(k) - 1)[int(k) - 1])


def individuals_from_records(
    records,
    limits: Dict[str, Sequence[float]],
    *,
    n_bootstrap: int | None = None,
) -> List[Any]:
    """Rebuild the propagator's ``Individual`` view of a stored history.

    ``extract_posterior`` needs exactly four fields per particle -- position,
    ``loss``, ``weight`` and the *stamped* tolerance -- because it replays each
    past proposal with :meth:`ABCPMC._build_proposal`, whose mixture weights come
    from the stored per-individual ``weight``.

    The stamped tolerance is ``proposal_tolerance``, not ``tolerance``. The
    latter is the running-minimum trajectory written for the tolerance plots,
    which carries the initial tolerance through the prior phase instead of
    ``None`` and so erases the bootstrap/archive boundary. Feeding it in
    unchanged makes ``extract_posterior`` give the bootstrap draws archive-phase
    proposals in the mixture denominator; measured against a real 16-rank run's
    own output that moves the weights to a correlation of 0.967, with the largest
    errors (up to 1.9x) on exactly the early prior draws near the box edge.

    Runs recorded before ``proposal_tolerance`` existed have no copy of that
    field, but the boundary is still recoverable exactly:
    :func:`infer_n_bootstrap` reads it off the ``weight`` column, and the replay
    then reproduces those runs' own stored ``posterior_weight`` to machine
    precision. It is applied automatically; pass ``n_bootstrap`` to override.

    ``records`` is consumed in the order given; callers are responsible for
    supplying arrival order (see the module docstring).
    """
    _, Individual = _import_abcpmc()
    names = list(limits)
    has_stamped = any(
        getattr(r, "proposal_tolerance", None) is not None for r in records
    )
    if not has_stamped and n_bootstrap is None and records:
        # Legacy file: recover the boundary from `weight` rather than degrade.
        n_bootstrap = infer_n_bootstrap(records)

    out = []
    for i, r in enumerate(records):
        params = r.params if hasattr(r, "params") else r
        ind = Individual(
            position=np.array([float(params[n]) for n in names], dtype=float),
            limits=limits,
        )
        ind.loss = None if r.loss is None else float(r.loss)
        ind.weight = None if r.weight is None else float(r.weight)
        if n_bootstrap is not None and i < int(n_bootstrap):
            ind.tolerance = None
        elif has_stamped:
            stamped = getattr(r, "proposal_tolerance", None)
            ind.tolerance = None if stamped is None else float(stamped)
        else:
            ind.tolerance = None if r.tolerance is None else float(r.tolerance)
        ind.generation = i
        out.append(ind)
    return out


def reported_posterior(
    records,
    limits: Dict[str, Sequence[float]],
    *,
    k: int = 100,
    kernel: str = "gaussian",
    scheduler_type: str = "quantile",
    amis_snapshots: int = 20,
    perturbation_scale: float = 0.8,
    n_proposals: int | None = None,
    eps_final: float | None = None,
    n_bootstrap: int | None = None,
    **propagator_kwargs,
):
    """Return ``(positions, weights)`` of the reported posterior over ``records``.

    A fresh propagator is constructed purely as the replay engine --
    ``extract_posterior`` is a pure function of the history it is handed, so the
    propagator carries nothing into it.
    """
    ABCPMC, _ = _import_abcpmc()
    inds = individuals_from_records(records, limits, n_bootstrap=n_bootstrap)
    if not inds:
        return np.empty((0, len(limits))), np.empty(0)
    prop = ABCPMC(
        limits,
        k=k,
        kernel=kernel,
        scheduler_type=scheduler_type,
        amis_snapshots=amis_snapshots,
        perturbation_scale=perturbation_scale,
        **propagator_kwargs,
    )
    positions, weights = prop.extract_posterior(
        inds, n_proposals=n_proposals, eps_final=eps_final
    )
    return np.asarray(positions, dtype=float), np.asarray(weights, dtype=float)


def weighted_w1(
    samples: np.ndarray,
    weights: np.ndarray | None,
    reference: np.ndarray,
    *,
    n_projections: int = 100,
    seed: int = 0,
) -> float:
    """$W_1$ between a weighted sample and an (unweighted) reference sample.

    Exact in one dimension; sliced over random projections above it. Unlike the
    point-mass metric this goes to zero for a correctly recovered posterior, so
    it can distinguish "right" from "collapsed".
    """
    from scipy.stats import wasserstein_distance

    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    if samples.shape[0] == 0 or reference.shape[0] == 0:
        return float("nan")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        if weights.sum() <= 0.0:
            return float("nan")

    d = samples.shape[1]
    if d == 1:
        return float(wasserstein_distance(
            samples[:, 0], reference[:, 0], u_weights=weights
        ))

    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    proj_s, proj_r = samples @ dirs.T, reference @ dirs.T
    return float(np.mean([
        wasserstein_distance(proj_s[:, j], proj_r[:, j], u_weights=weights)
        for j in range(n_projections)
    ]))


def reported_posterior_curve(
    records,
    limits: Dict[str, Sequence[float]],
    reference: np.ndarray,
    *,
    checkpoints: int = 12,
    min_records: int = 50,
    **posterior_kwargs,
) -> pd.DataFrame:
    """$W_1$(reported posterior, reference) at wall-clock checkpoints.

    Each row replays ``extract_posterior`` over the prefix of ``records`` that
    had completed by that wall-clock time, so the curve is the trajectory of the
    quantity the paper reports rather than of the archive it happens to hold.
    Cost is ``O(checkpoints * n * m * k)``; ``checkpoints`` is deliberately small.
    """
    ordered = sorted(records, key=lambda r: (float(r.wall_time or 0.0)))
    if len(ordered) < min_records:
        return pd.DataFrame(columns=["wall_time", "n_records", "w1", "ess_fraction"])

    times = np.asarray([float(r.wall_time or 0.0) for r in ordered])
    # Checkpoints uniform in RECORD COUNT, not in time: a run whose throughput
    # varies (the whole point of the straggler study) would otherwise spend most
    # of its checkpoints in whichever regime happened to be slow.
    idx = np.unique(np.linspace(min_records, len(ordered), checkpoints).astype(int))

    rows = []
    for n in idx:
        prefix = ordered[:n]
        positions, weights = reported_posterior(prefix, limits, **posterior_kwargs)
        if positions.size == 0:
            continue
        w = weights / weights.sum() if weights.sum() > 0 else weights
        rows.append({
            "wall_time": float(times[n - 1]),
            "n_records": int(n),
            "w1": weighted_w1(positions, weights, reference),
            "ess_fraction": (
                float(1.0 / np.sum(w ** 2)) / len(w) if weights.sum() > 0 else float("nan")
            ),
        })
    return pd.DataFrame(rows)
