"""Shared pyABC sampler factory.

Centralises sampler construction for both
:mod:`pyabc_wrapper` and :mod:`abc_smc_baseline`, avoiding duplication and
keeping the MPI import path isolated behind the ``"mpi"`` backend branch.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CommWorldMap:
    """COMM_WORLD-based blocking parallel map for pyABC's MappingSampler.

    Replaces the legacy MPI executor to avoid ``Create_intercomm``/``Disconnect``
    fragility on ParaStation MPI at high rank counts.  Uses only ``bcast``,
    ``send``, and ``recv`` on ``COMM_WORLD`` — no inter-communicators.

    Usage (all ranks must call the same code path)::

        cmap = CommWorldMap(MPI.COMM_WORLD)
        if cmap.is_root:
            sampler = build_pyabc_sampler(..., mpi_map=cmap.map)
            result = run_abc(sampler=sampler, ...)
            cmap.shutdown()          # tells workers to exit
        else:
            cmap.worker_loop()       # blocks until shutdown
        comm.Barrier()
    """

    _SENTINEL = None  # end-of-batch marker for work items

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_root = self.rank == 0
        self._shutdown = False

    def map(self, fn, iterable):
        """Distribute *fn* over *iterable* across MPI workers.  Root-only.

        Broadcasts *fn* to all workers, then distributes items dynamically
        (one at a time to idle workers) for load balance.  Returns results
        in submission order.
        """
        items = list(iterable)

        # Single-process fallback: no workers to distribute to.
        if self.size <= 1:
            return [fn(item) for item in items]

        if not items:
            self.comm.bcast(("map", fn), root=0)
            for dest in range(1, self.size):
                self.comm.send(self._SENTINEL, dest=dest, tag=0)
            return []

        from mpi4py import MPI

        self.comm.bcast(("map", fn), root=0)

        results = [None] * len(items)
        next_item = 0
        active = 0

        # Seed each worker with one item
        for dest in range(1, self.size):
            if next_item < len(items):
                self.comm.send((next_item, items[next_item]), dest=dest, tag=0)
                next_item += 1
                active += 1
            else:
                self.comm.send(self._SENTINEL, dest=dest, tag=0)

        # Collect results and send more work dynamically
        while active > 0:
            status = MPI.Status()
            response = self.comm.recv(source=MPI.ANY_SOURCE, tag=1, status=status)
            source = status.Get_source()
            idx, result_or_error = response

            if isinstance(result_or_error, _WorkerError):
                # Drain remaining active workers before re-raising
                active -= 1
                for dest in range(1, self.size):
                    if dest != source:
                        self.comm.send(self._SENTINEL, dest=dest, tag=0)
                while active > 0:
                    self.comm.recv(source=MPI.ANY_SOURCE, tag=1)
                    active -= 1
                raise result_or_error.exc

            results[idx] = result_or_error
            active -= 1

            if next_item < len(items):
                self.comm.send((next_item, items[next_item]), dest=source, tag=0)
                next_item += 1
                active += 1
            else:
                self.comm.send(self._SENTINEL, dest=source, tag=0)

        return results

    def shutdown(self):
        """Signal workers to exit their loop.  Root-only, idempotent."""
        if self._shutdown:
            return
        self._shutdown = True
        if self.size > 1:
            self.comm.bcast(("shutdown", None), root=0)

    def worker_loop(self):
        """Process map batches until shutdown.  Workers-only (rank != 0)."""
        while True:
            tag, payload = self.comm.bcast(None, root=0)
            if tag == "shutdown":
                break
            fn = payload
            # Process items until sentinel
            while True:
                item = self.comm.recv(source=0, tag=0)
                if item is self._SENTINEL:
                    break
                idx, work = item
                try:
                    result = fn(work)
                    self.comm.send((idx, result), dest=0, tag=1)
                except Exception as exc:
                    self.comm.send((idx, _WorkerError(exc)), dest=0, tag=1)


class _WorkerError:
    """Wrapper to distinguish worker exceptions from normal results."""
    __slots__ = ("exc",)

    def __init__(self, exc):
        self.exc = exc


def _pyabc_parallel_safe(simulate_fn) -> bool:
    """Return whether a benchmark can be shipped to parallel pyABC workers."""
    owner = getattr(simulate_fn, "__self__", None)
    if owner is None:
        return True

    if hasattr(owner, "PYABC_PARALLEL_SAFE"):
        return bool(getattr(owner, "PYABC_PARALLEL_SAFE"))
    return bool(getattr(owner, "MULTIPROCESSING_SAFE", True))


def resolve_pyabc_parallel_backend(
    inference_cfg,
    method_name: str,
    simulate_fn=None,
) -> str:
    """Return the pyABC backend that should be used for this run.

    Policy:
    - single-worker runs default to ``multicore`` for a local single-core sampler
    - parallel runs always use ``mpi`` for consistent comparisons
    - an explicit ``multicore`` request is overridden to ``mpi`` when
      ``n_workers > 1``
    """
    n_workers = max(1, int(inference_cfg.get("n_workers", 1)))
    configured = inference_cfg.get("parallel_backend")

    if n_workers <= 1:
        return configured or "multicore"

    if configured is None:
        return "mpi"

    if configured == "multicore":
        logger.warning(
            "Overriding %s parallel_backend=multicore to mpi because n_workers=%d. "
            "Parallel pyABC runs use MPI for consistent scaling/comparison.",
            method_name,
            n_workers,
        )
        return "mpi"

    return configured


def resolve_pyabc_worker_count(
    simulate_fn,
    n_procs: int,
    parallel_backend: str,
    method_name: str,
) -> int:
    """Return a safe worker count for the given simulator/backend pair.

    Some native simulator stacks are not safe to execute behind pyABC's local
    multiprocessing samplers. Benchmarks can opt out by setting
    ``MULTIPROCESSING_SAFE = False`` on the simulator instance.
    """
    if parallel_backend != "multicore" or n_procs <= 1:
        return n_procs

    if _pyabc_parallel_safe(simulate_fn):
        return n_procs

    owner = getattr(simulate_fn, "__self__", None)
    owner_name = owner.__class__.__name__ if owner is not None else type(simulate_fn).__name__

    logger.warning(
        "Forcing %s to single-process pyABC execution because %s is marked "
        "unsafe for parallel pyABC workers.",
        method_name,
        owner_name,
    )
    return 1


def resolve_pyabc_client_max_jobs(
    inference_cfg,
    *,
    parallel_backend: str,
    n_procs: int,
    mpi_sampler: str | None = None,
) -> int | None:
    """Return the client-side outstanding-job limit for pyABC MPI samplers."""
    if parallel_backend != "mpi":
        return None

    configured = inference_cfg.get("pyabc_client_max_jobs")
    if mpi_sampler == "mapping":
        if configured not in (None, ""):
            logger.warning(
                "Ignoring pyabc_client_max_jobs=%s because pyabc_mpi_sampler=mapping "
                "does not use speculative client-side futures.",
                configured,
            )
        return None

    if configured in (None, ""):
        return max(1, int(n_procs))
    return max(1, int(configured))


def resolve_pyabc_mpi_sampler(
    inference_cfg,
    *,
    parallel_backend: str,
    method_name: str,
) -> str | None:
    """Return the pyABC MPI sampler strategy for this run.

    After Phase 3 cleanup, only the CommWorldMap-backed ``"mapping"`` path
    is supported. Legacy sampler names now raise ValueError.
    """
    if parallel_backend != "mpi":
        return None

    configured = inference_cfg.get("pyabc_mpi_sampler")
    if configured in (None, "", "mapping"):
        return "mapping"

    raise ValueError(
        f"Unknown pyabc_mpi_sampler={configured!r} for {method_name}. "
        "Valid values: 'mapping'."
    )


def build_pyabc_sampler(
    n_procs: int,
    parallel_backend: str,
    *,
    mpi_sampler: str | None = None,
    mpi_map=None,
    client_max_jobs: int | None = None,
):
    """Construct a pyABC sampler.

    Parameters
    ----------
    n_procs:
        Number of parallel workers.
    parallel_backend:
        ``"multicore"`` — use :class:`pyabc.MulticoreEvalParallelSampler`
        (or :class:`pyabc.SingleCoreSampler` when *n_procs* == 1).
        ``"mpi"`` — wrap an MPI-backed ``map`` callable inside
        :class:`pyabc.MappingSampler`.
    mpi_sampler:
        Strategy to use when *parallel_backend* is ``"mpi"``. Only
        ``"mapping"`` (CommWorldMap) is supported after Phase 3 cleanup.
    mpi_map:
        Communicator-backed ``map``-like callable for the ``"mapping"``
        MPI strategy, typically ``CommWorldMap.map``.
    client_max_jobs:
        Retained for forward-compat with non-mapping samplers. Currently
        unused; the mapping path does not consume it. Kept so call sites
        that pass it (e.g. wrapper's ``_run_with_map_callable``) do not
        need immediate edits.

    Returns
    -------
    pyabc.Sampler

    Raises
    ------
    ValueError
        If *parallel_backend* is not recognised, or if *mpi_sampler* is
        set to anything other than ``"mapping"``.
    """
    import pyabc

    if parallel_backend == "multicore":
        if n_procs == 1:
            return pyabc.SingleCoreSampler()
        return pyabc.MulticoreEvalParallelSampler(n_procs)

    elif parallel_backend == "mpi":
        mpi_sampler = mpi_sampler or "mapping"
        if mpi_sampler == "mapping":
            if mpi_map is None:
                raise ValueError(
                    "The 'mpi' parallel_backend with pyabc_mpi_sampler='mapping' "
                    "requires an existing communicator-backed map callable."
                )
            return pyabc.MappingSampler(map_=mpi_map)
        raise ValueError(
            f"Unknown mpi_sampler={mpi_sampler!r}. "
            "Valid values: 'mapping'."
        )

    else:
        raise ValueError(
            f"Unknown parallel_backend={parallel_backend!r}. "
            "Valid values: 'multicore', 'mpi'."
        )
