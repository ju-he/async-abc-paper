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
    "individuals_from_records",
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

    Runs recorded before ``proposal_tolerance`` existed have no faithful copy of
    the field. For those, pass ``n_bootstrap`` to mark the leading prior draws
    explicitly; passing ``None`` falls back to ``tolerance`` and warns, because
    the boundary is *not* recoverable from the file -- it sits at ``k + O(W)``
    draws rather than ``k``, the surplus being the ranks still in flight when the
    archive fills (measured: 112 for a ``k=100``, ``W=16`` run, where assuming
    ``k`` leaves a residual and assuming 112 reproduces the run to 0.99999).

    ``records`` is consumed in the order given; callers are responsible for
    supplying arrival order (see the module docstring).
    """
    import warnings

    _, Individual = _import_abcpmc()
    names = list(limits)
    has_stamped = any(
        getattr(r, "proposal_tolerance", None) is not None for r in records
    )
    if not has_stamped and n_bootstrap is None and records:
        warnings.warn(
            "individuals_from_records: no proposal_tolerance on these records and "
            "no n_bootstrap given, so the bootstrap phase cannot be identified. "
            "Falling back to the running-minimum `tolerance`; the replayed "
            "posterior will differ from the run's own output (correlation ~0.97 "
            "on a measured 16-rank run). Re-run to get proposal_tolerance, or "
            "pass n_bootstrap explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )

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
