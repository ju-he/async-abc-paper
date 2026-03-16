"""Shared pyABC sampler factory.

Centralises sampler construction for both
:mod:`pyabc_wrapper` and :mod:`abc_smc_baseline`, avoiding duplication and
keeping the MPI import path isolated behind the ``"mpi"`` backend branch.
"""


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
