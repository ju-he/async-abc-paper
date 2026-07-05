"""Inference method registry.

``METHOD_REGISTRY`` maps method name strings to their runner callables.
All runners share the same signature::

    runner(simulate_fn, limits, inference_cfg, output_dir, replicate, seed, progress=None)
        -> List[ParticleRecord]

Use :func:`run_method` as the single dispatch point in experiment scripts.
"""
import inspect
from typing import Any, Callable, Dict, List

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from ..utils.progress import MethodProgressReporter
from .propulate_abc import run_propulate_abc
from .pyabc_sampler import resolve_pyabc_parallel_backend
from .pyabc_wrapper import run_pyabc_smc
from .rejection_abc import run_rejection_abc
from .abc_smc_baseline import run_abc_smc_baseline

_PYABC_METHODS = {"pyabc_smc", "abc_smc_baseline"}

METHOD_REGISTRY: Dict[str, Callable] = {
    "async_propulate_abc": run_propulate_abc,
    "pyabc_smc":           run_pyabc_smc,
    "rejection_abc":       run_rejection_abc,
    "abc_smc_baseline":    run_abc_smc_baseline,
}

METHOD_EXECUTION_MODE: Dict[str, str] = {
    "async_propulate_abc": "all_ranks",
    "pyabc_smc": "rank_zero",
    "rejection_abc": "rank_parallel",
    "abc_smc_baseline": "rank_zero",
}


def run_method(
    name: str,
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    progress: MethodProgressReporter | None = None,
    **kwargs,
) -> List[ParticleRecord]:
    """Dispatch to the named inference method.

    Parameters
    ----------
    name:
        Key in :data:`METHOD_REGISTRY`.
    simulate_fn:
        Benchmark simulator ``(params, seed) -> float``.
    limits:
        Search-space limits dict.
    inference_cfg:
        ``config["inference"]`` sub-dict.
    output_dir:
        :class:`~async_abc.io.paths.OutputDir` for the current run.
    replicate:
        Replicate index.
    seed:
        RNG seed for this replicate.
    **kwargs:
        Extra keyword arguments forwarded to the runner if it accepts them.

    Returns
    -------
    List[ParticleRecord]

    Raises
    ------
    KeyError
        If *name* is not in :data:`METHOD_REGISTRY`, with a helpful message
        listing valid names.
    """
    if name not in METHOD_REGISTRY:
        valid = sorted(METHOD_REGISTRY.keys())
        raise KeyError(
            f"'{name}' is not a registered method. "
            f"Available methods: {valid}"
        )
    runner = METHOD_REGISTRY[name]
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None

    # Determine which optional kwargs the runner accepts.
    extra_kw: Dict[str, Any] = {}
    if signature is not None:
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        for key, value in kwargs.items():
            if has_var_keyword or key in signature.parameters:
                extra_kw[key] = value
    else:
        extra_kw = dict(kwargs)

    accepts_progress = True
    if signature is not None:
        accepts_progress = any(
            p.kind == inspect.Parameter.VAR_KEYWORD or p.name == "progress"
            for p in signature.parameters.values()
        )

    if accepts_progress:
        return runner(
            simulate_fn,
            limits,
            inference_cfg,
            output_dir,
            replicate,
            seed,
            progress=progress,
            **extra_kw,
        )
    return runner(simulate_fn, limits, inference_cfg, output_dir, replicate, seed, **extra_kw)


def method_execution_mode(name: str) -> str:
    """Return how a method should execute under MPI."""
    if name in METHOD_EXECUTION_MODE:
        return METHOD_EXECUTION_MODE[name]
    if name in METHOD_REGISTRY:
        return "rank_zero"
    valid = sorted(METHOD_EXECUTION_MODE.keys())
    raise KeyError(
        f"'{name}' is not a registered method. "
        f"Available methods: {valid}"
    )


def method_execution_mode_for_cfg(
    name: str,
    inference_cfg: Dict[str, Any],
    simulate_fn: Callable | None = None,
) -> str:
    """Return the effective execution mode for a method given a runtime config.

    Identical to :func:`method_execution_mode` for most methods. For pyABC
    methods whose backend resolves to ``"mpi"`` the static ``"rank_zero"``
    default is upgraded to ``"all_ranks"`` so that all pre-allocated MPI ranks
    participate in the run.
    """
    mode = method_execution_mode(name)
    if mode in ("rank_zero", "rank_parallel") and name in _PYABC_METHODS:
        if resolve_pyabc_parallel_backend(
            inference_cfg,
            method_name=name,
            simulate_fn=simulate_fn,
        ) == "mpi":
            return "all_ranks"
    return mode
