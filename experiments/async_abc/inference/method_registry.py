"""Inference method registry.

``METHOD_REGISTRY`` maps method name strings to their runner callables.
All runners share the same signature::

    runner(simulate_fn, limits, inference_cfg, output_dir, replicate, seed)
        -> List[ParticleRecord]

Use :func:`run_method` as the single dispatch point in experiment scripts.
"""
from typing import Callable, Dict, List

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from .propulate_abc import run_propulate_abc
from .pyabc_wrapper import run_pyabc_smc

METHOD_REGISTRY: Dict[str, Callable] = {
    "async_propulate_abc": run_propulate_abc,
    "pyabc_smc": run_pyabc_smc,
}


def run_method(
    name: str,
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
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
    return METHOD_REGISTRY[name](
        simulate_fn, limits, inference_cfg, output_dir, replicate, seed
    )
