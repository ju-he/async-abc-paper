"""Shared pyABC sampler factory.

Centralises sampler construction for both
:mod:`pyabc_wrapper` and :mod:`abc_smc_baseline`, avoiding duplication and
keeping the MPI import path isolated behind the ``"mpi"`` backend branch.
"""

from __future__ import annotations

import logging
from concurrent.futures import wait as _wait

logger = logging.getLogger(__name__)


class TrackedFutureExecutor:
    """Wrap an executor and retain references to every submitted future."""

    def __init__(self, inner):
        self._inner = inner
        self._submitted: list = []

    def submit(self, fn, /, *args, **kwargs):
        future = self._inner.submit(fn, *args, **kwargs)
        self._submitted.append(future)
        return future

    def pending_futures(self, *, exclude_cancelled: bool = False):
        pending = [future for future in self._submitted if not future.done()]
        if exclude_cancelled:
            pending = [future for future in pending if not future.cancelled()]
        return pending

    def wait_for_pending(self, *, exclude_cancelled: bool = False) -> int:
        pending = self.pending_futures(exclude_cancelled=exclude_cancelled)
        if pending:
            _wait(pending)
        return len(pending)

    @property
    def submitted_count(self) -> int:
        return len(self._submitted)

    def __getattr__(self, name):
        return getattr(self._inner, name)


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
    """Return the pyABC MPI sampler strategy for this run."""
    if parallel_backend != "mpi":
        return None

    configured = inference_cfg.get("pyabc_mpi_sampler")
    if configured in (None, ""):
        return "mapping"

    if configured == "mapping":
        return "mapping"

    if configured == "concurrent_futures_legacy":
        logger.warning(
            "Using %s with pyabc_mpi_sampler=concurrent_futures_legacy. "
            "This retains the old futures-based MPI path, which has shown teardown "
            "hangs at large rank counts on the cluster and is kept only as an explicit opt-in fallback.",
            method_name,
        )
        return configured

    raise ValueError(
        f"Unknown pyabc_mpi_sampler={configured!r}. "
        "Valid values: 'mapping', 'concurrent_futures_legacy'."
    )


def build_pyabc_sampler(
    n_procs: int,
    parallel_backend: str,
    *,
    mpi_sampler: str | None = None,
    mpi_map=None,
    cfuture_executor=None,
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
        ``"mpi"`` — wrap an existing communicator-backed executor inside
        :class:`pyabc.ConcurrentFutureSampler`.
    mpi_sampler:
        Strategy to use when *parallel_backend* is ``"mpi"``.
        ``"mapping"`` uses :class:`pyabc.MappingSampler`;
        ``"concurrent_futures_legacy"`` uses
        :class:`pyabc.ConcurrentFutureSampler`.
    mpi_map:
        Existing blocking ``map``-like callable for the ``"mapping"``
        MPI strategy, typically provided by an MPI executor.
    cfuture_executor:
        Existing ``concurrent.futures``-style executor for the ``"mpi"``
        backend. Callers are expected to provision it from the already launched
        MPI communicator via ``MPICommExecutor`` when using the legacy futures
        strategy.
    client_max_jobs:
        Maximum number of outstanding futures pyABC may keep submitted when
        using the MPI backend. Defaults to pyABC's internal default unless an
        explicit value is provided.

    Returns
    -------
    pyabc.Sampler

    Raises
    ------
    ValueError
        If *parallel_backend* is not a recognised value.
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
        if mpi_sampler == "concurrent_futures_legacy":
            if cfuture_executor is None:
                raise ValueError(
                    "The 'mpi' parallel_backend with pyabc_mpi_sampler='concurrent_futures_legacy' "
                    "requires an existing communicator-backed executor."
                )
            kwargs = {"cfuture_executor": cfuture_executor}
            if client_max_jobs is not None:
                kwargs["client_max_jobs"] = int(client_max_jobs)
            return pyabc.ConcurrentFutureSampler(**kwargs)
        raise ValueError(
            f"Unknown mpi_sampler={mpi_sampler!r}. "
            "Valid values: 'mapping', 'concurrent_futures_legacy'."
        )

    else:
        raise ValueError(
            f"Unknown parallel_backend={parallel_backend!r}. "
            "Valid values: 'multicore', 'mpi'."
        )
