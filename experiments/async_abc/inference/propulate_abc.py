"""Propulate-ABC inference wrapper.

Wraps the existing ``ABCPMC`` propagator from ``propulate`` and runs it via
``Propulator`` for a fixed number of simulations.  After the run, all
evaluated individuals are converted to :class:`~async_abc.io.records.ParticleRecord`
objects.

The loss function passed to Propulate receives a ``propulate.Individual`` which
behaves like a dict (``ind["mu"]`` etc.).  Internally, the benchmark's
``simulate(params, seed)`` is called with a per-evaluation seed derived from
the run seed and the individual's generation counter.
"""
import random
import sys
import time
from importlib import invalidate_caches
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from ..io.paths import OutputDir
from ..io.records import ParticleRecord

_PROPULATE_ROOT = Path(__file__).resolve().parents[3] / "propulate"
Propulator = None
ABCPMC = None


def _ensure_propulate_imports() -> None:
    """Resolve propulate from the active environment or the repo-local fallback."""
    global Propulator, ABCPMC
    if Propulator is not None and ABCPMC is not None:
        return

    def _import_symbols():
        from propulate import Propulator as _Propulator
        try:
            from propulate.propagators.abcpmc import ABCPMC as _ABCPMC
        except ImportError:
            from propulate.propagators import ABCPMC as _ABCPMC
        return _Propulator, _ABCPMC

    try:
        _Propulator, _ABCPMC = _import_symbols()
    except ImportError as env_exc:
        if not _PROPULATE_ROOT.is_dir():
            raise ImportError(
                "The async_propulate_abc method requires 'propulate'. "
                "Install it in the active environment or provide the repo-local "
                f"'propulate' checkout at {_PROPULATE_ROOT}."
            ) from env_exc
        propulate_root = str(_PROPULATE_ROOT)
        if propulate_root not in sys.path:
            sys.path.insert(0, propulate_root)
        invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == "propulate" or module_name.startswith("propulate."):
                del sys.modules[module_name]
        try:
            _Propulator, _ABCPMC = _import_symbols()
        except ImportError as path_exc:
            raise ImportError(
                "The async_propulate_abc method requires 'propulate'. "
                "Import from the active environment failed, and the repo-local "
                f"fallback at {_PROPULATE_ROOT} was not usable."
            ) from path_exc

    if Propulator is None:
        Propulator = _Propulator
    if ABCPMC is None:
        ABCPMC = _ABCPMC


def _eval_seed(run_seed: int, generation: int) -> int:
    """Deterministic per-evaluation seed derived from the run seed and generation."""
    return abs(hash((run_seed, generation))) % (2**31)


def run_propulate_abc(
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
) -> List[ParticleRecord]:
    """Run the asynchronous ABC-PMC propagator via Propulate.

    Parameters
    ----------
    simulate_fn:
        Callable ``(params: dict, seed: int) -> float`` — the benchmark simulator.
    limits:
        Propulate-compatible limits dict, e.g. ``{"mu": (-5.0, 5.0)}``.
    inference_cfg:
        ``config["inference"]`` sub-dict.  Used keys:
        ``max_simulations``, ``k``, ``tol_init``,
        ``scheduler_type``, ``perturbation_scale``.
    output_dir:
        :class:`~async_abc.io.paths.OutputDir` — used for Propulate checkpoint path.
    replicate:
        Replicate index (stored in each record).
    seed:
        Base RNG seed for this replicate.

    Returns
    -------
    List[ParticleRecord]
        One record per simulation evaluation, in generation order.
    """
    _ensure_propulate_imports()

    max_sims = inference_cfg["max_simulations"]
    k = inference_cfg.get("k", 100)
    tol_init = inference_cfg.get("tol_init", 10.0)
    scheduler_type = inference_cfg.get("scheduler_type", "acceptance_rate")
    perturbation_scale = inference_cfg.get("perturbation_scale", 0.8)

    # Pass extra scheduler kwargs if present
    scheduler_kwargs = {}
    for key in ("percentile", "decay_factor", "low_rate", "high_rate",
                "shrink_factor", "expand_factor"):
        if key in inference_cfg:
            scheduler_kwargs[key] = inference_cfg[key]

    propagator = ABCPMC(
        limits=limits,
        perturbation_scale=perturbation_scale,
        k=k,
        tol=tol_init,
        scheduler_type=scheduler_type,
        rng=random.Random(seed),
        **scheduler_kwargs,
    )

    run_start = time.time()

    # Propulate's loss_fn receives an Individual (dict-like).
    # We extract param values and forward to the benchmark simulator.
    def loss_fn(ind) -> float:
        params = {key: float(ind[key]) for key in limits}
        sim_seed = _eval_seed(seed, int(ind.generation))
        return float(simulate_fn(params, seed=sim_seed))

    # Each (replicate, seed) gets its own checkpoint dir to prevent cross-contamination
    checkpoint_dir = output_dir.logs / f"propulate_rep{replicate}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    propulator = Propulator(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=random.Random(seed + 1),  # +1 to keep RNG streams separate
        generations=max_sims,
        checkpoint_path=checkpoint_dir,
    )

    propulator.propulate(logging_interval=max_sims + 1, debug=0)

    # Convert population to ParticleRecord list, sorted by generation
    population = sorted(propulator.population, key=lambda ind: ind.generation)
    records: List[ParticleRecord] = []
    for step, ind in enumerate(population, start=1):
        params = {key: float(ind[key]) for key in limits}
        weight = float(ind.weight) if ind.weight is not None else None
        tolerance = float(ind.tolerance) if ind.tolerance is not None else None
        sim_end_time = (
            float(ind.evaltime) - run_start
            if hasattr(ind, "evaltime") and ind.evaltime is not None
            else None
        )
        sim_start_time = (
            sim_end_time - float(ind.evalperiod)
            if hasattr(ind, "evalperiod") and ind.evalperiod is not None
            else None
        )
        generation = int(ind.generation) if getattr(ind, "generation", None) is not None else None
        records.append(ParticleRecord(
            method="async_propulate_abc",
            replicate=replicate,
            seed=seed,
            step=step,
            params=params,
            loss=float(ind.loss),
            weight=weight,
            tolerance=tolerance,
            wall_time=sim_end_time if sim_end_time is not None else 0.0,
            worker_id=str(ind.rank) if getattr(ind, "rank", None) is not None else None,
            sim_start_time=sim_start_time,
            sim_end_time=sim_end_time,
            generation=generation,
        ))

    return records
