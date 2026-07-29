"""Barrierized twin of the asynchronous propagator (external review, concern 4).

The synchronous pyABC baseline differs from our method in more than the barrier:
it adds a probabilistic-rejection step, its generation length depends on the
resulting acceptance probability, and it retains classical population weighting.
Matching the kernel removes the acceptance *rule* as a confound, but a fully
clean attribution of the systems gains to barrier removal needs a *twin* of our
own algorithm that differs in the barrier and nothing else.

:class:`ABCPMCBarrier` is that twin. It subclasses the frozen ``ABCPMC``
propagator and adds one thing: a collective barrier immediately before each
breed. Proposals, kernel weights, archive rule, and the reported estimator are
inherited unchanged, so the asynchronous arm and the twin differ *only* in when
a worker is allowed to proceed.

Why this is a real barrier
--------------------------
Propulate's worker loop is serial per rank -- ``_evaluate_individual()`` calls
``_breed()`` (which calls the propagator) and only then runs the simulation.
Blocking at the top of ``__call__`` therefore stalls the worker *before* it
starts its next simulation, so under a straggler every rank waits for the
straggler to reach the barrier. That idle time is real and is picked up by the
existing utilization instrumentation as non-simulation wall-clock. One
``__call__`` happens per evaluation per rank (no rejection loop under a smooth
kernel), so the collective stays in lockstep.

Deliberately NOT implemented: history truncation to a batch boundary. Staging
the *information* as well as the timing would change the statistics on top of
the schedule, confounding "the barrier costs throughput" with "the barrier
changes the proposal". Keeping the inherited statistical behaviour bit-identical
makes this a pure synchronization contrast.

Nothing here touches Propulate itself -- the twin lives in the paper repo and is
not part of the upstream propagator.

.. warning::
   The barrier is collective over the attached communicator, so **every rank
   must call it the same number of times**. Wall-clock-limited runs stop ranks
   independently (first-rank-hit semantics) at differing iteration counts and
   would hang. Twin runs must therefore be *simulation-limited*, which gives
   every rank the same fixed generation budget. :func:`assert_barrier_safe`
   enforces this at construction time rather than letting a job hang on the
   cluster.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def assert_barrier_safe(inference_cfg: dict) -> None:
    """Refuse a barrierized run whose ranks would stop at different call counts.

    A collective barrier deadlocks unless every participating rank reaches it the
    same number of times. Under a wall-clock budget Propulate stops each rank as
    it trips the deadline, so the counts diverge. Fail loudly here instead of
    hanging for the whole walltime on a compute node.
    """
    if inference_cfg.get("max_wall_time_s") is not None:
        raise ValueError(
            "barrier=True requires a simulation-limited budget: a collective "
            "barrier needs identical per-rank call counts, but max_wall_time_s "
            "stops ranks independently (first-rank-hit) and would deadlock. "
            "Use propulate_budget_mode='total_simulations' with max_simulations "
            "and drop max_wall_time_s for the twin arm."
        )


def make_barrier_propagator_class(base_cls: type) -> type:
    """Build the twin class from the frozen ``ABCPMC`` base.

    ``ABCPMC`` is imported lazily by the runner (MPI-dependent), so the subclass
    is created at call time rather than at import time.
    """

    class ABCPMCBarrier(base_cls):  # type: ignore[valid-type,misc]
        """``ABCPMC`` plus a collective barrier before each breed."""

        def __init__(self, *args: Any, barrier: bool = False, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._barrier_enabled = bool(barrier)
            self._barrier_comm: Optional[Any] = None
            self._barrier_calls = 0

        def attach_comm(self, comm: Any) -> None:
            """Attach the communicator the barrier synchronises over.

            Called after ``Propulator`` construction (the propagator is only
            *invoked* later), so no Propulate code has to change. The right
            scope is ``propulator.propulate_comm`` -- exactly the ranks that
            breed.
            """
            self._barrier_comm = comm

        @property
        def barrier_calls(self) -> int:
            """Number of barriers executed (for the twin's own diagnostics)."""
            return self._barrier_calls

        def __call__(self, inds):  # noqa: D102 - inherited contract
            if self._barrier_enabled:
                if self._barrier_comm is None:
                    raise RuntimeError(
                        "ABCPMCBarrier(barrier=True) was never given a "
                        "communicator; call attach_comm(propulator.propulate_comm) "
                        "after constructing the Propulator."
                    )
                self._barrier_comm.Barrier()
                self._barrier_calls += 1
            return super().__call__(inds)

    return ABCPMCBarrier
