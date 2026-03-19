"""Shared pyABC sampler factory.

Centralises sampler construction for both
:mod:`pyabc_wrapper` and :mod:`abc_smc_baseline`, avoiding duplication and
keeping the MPI import path isolated behind the ``"mpi"`` backend branch.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_pyabc_parallel_backend(
    inference_cfg,
    method_name: str,
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

    owner = getattr(simulate_fn, "__self__", None)
    if owner is None:
        return n_procs

    if getattr(owner, "MULTIPROCESSING_SAFE", True):
        return n_procs

    logger.warning(
        "Forcing %s to single-process pyABC execution because %s is marked "
        "MULTIPROCESSING_SAFE=False.",
        method_name,
        owner.__class__.__name__,
    )
    return 1


def build_pyabc_sampler(n_procs: int, parallel_backend: str):
    """Construct a pyABC sampler.

    Parameters
    ----------
    n_procs:
        Number of parallel workers.
    parallel_backend:
        ``"multicore"`` — use :class:`pyabc.MulticoreEvalParallelSampler`
        (or :class:`pyabc.SingleCoreSampler` when *n_procs* == 1).
        ``"mpi"`` — wrap :class:`mpi4py.futures.MPIPoolExecutor` inside
        :class:`pyabc.ConcurrentFutureSampler` for multi-node execution.

    Returns
    -------
    pyabc.Sampler

    Raises
    ------
    ValueError
        If *parallel_backend* is not a recognised value.
    ImportError
        If ``mpi4py`` is not installed and backend is ``"mpi"``.
    """
    import pyabc

    if parallel_backend == "multicore":
        if n_procs == 1:
            return pyabc.SingleCoreSampler()
        return pyabc.MulticoreEvalParallelSampler(n_procs)

    elif parallel_backend == "mpi":
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError as exc:
            raise ImportError(
                "The 'mpi' parallel_backend requires mpi4py. "
                "Install it with: pip install mpi4py"
            ) from exc
        # max_workers is intentionally omitted: in bootstrap mode
        # (mpirun -n N python -m mpi4py.futures), the worker count is
        # determined by the MPI world size, not by a Python parameter.
        return pyabc.ConcurrentFutureSampler(
            cfuture_executor=MPIPoolExecutor()
        )

    else:
        raise ValueError(
            f"Unknown parallel_backend={parallel_backend!r}. "
            "Valid values: 'multicore', 'mpi'."
        )
