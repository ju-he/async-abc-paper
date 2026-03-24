"""pyABC inference wrapper (optional dependency).

If ``pyabc`` is not installed, :func:`run_pyabc_smc` raises ``ImportError``
with installation instructions.  The function is still importable so that the
method registry can reference it unconditionally.
"""
import logging
from typing import Callable, Dict, List

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from ._pyabc_history import history_observable_frame
from .pyabc_sampler import (
    build_pyabc_sampler,
    resolve_pyabc_parallel_backend,
    resolve_pyabc_worker_count,
)

logger = logging.getLogger(__name__)


def _run_pyabc_smc_with_sampler(
    *,
    sampler,
    simulate_fn: Callable,
    limits: Dict,
    max_sims: int,
    k: int,
    tol_init: float,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    progress=None,
) -> List[ParticleRecord]:
    import pyabc
    import numpy as np
    import time

    # Build pyABC prior from limits
    prior = pyabc.Distribution(
        **{
            name: pyabc.RV("uniform", lo, hi - lo)
            for name, (lo, hi) in limits.items()
        }
    )

    # Per-call distance cache: maps frozen param key -> actual loss.
    # Must be local to avoid cross-replicate contamination.
    # Note: not populated for MPI backend (workers run in separate processes).
    _distance_cache: Dict = {}
    eval_count = 0

    def pyabc_model(params):
        nonlocal eval_count
        # Derive a stable per-evaluation seed from (run_seed, frozen_params).
        # Using a param hash is deterministic regardless of evaluation order,
        # which matters for parallel samplers.
        param_key = tuple(sorted((k_, round(v, 10)) for k_, v in params.items()))
        sim_seed = abs(hash((seed, param_key))) % (2**31)
        loss = float(simulate_fn(dict(params), seed=sim_seed))
        _distance_cache[param_key] = loss
        eval_count += 1
        if progress is not None:
            progress.update(simulations=eval_count)
        return {"distance": loss}

    def pyabc_distance(x, x0):
        return x["distance"]

    db_path = f"sqlite:///{output_dir.data / f'pyabc_rep{replicate}.db'}"

    abc = pyabc.ABCSMC(
        models=pyabc_model,
        parameter_priors=prior,
        distance_function=pyabc_distance,
        population_size=k,
        transitions=pyabc.MultivariateNormalTransition(),
        eps=pyabc.QuantileEpsilon(alpha=0.5),
        sampler=sampler,
    )
    abc.new(db_path, {"distance": 0.0})

    # Seed numpy before running so pyABC's internal prior sampling is reproducible
    np.random.seed(seed % (2**31))
    run_start = time.time()
    history = abc.run(
        minimum_epsilon=tol_init * 0.01,
        max_total_nr_simulations=max_sims,
    )

    observable = history_observable_frame(history, run_start)

    # Collect records from the pyABC history
    records: List[ParticleRecord] = []
    step = 0
    fallback_attempt_count = 0
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        row = observable.loc[t] if t in observable.index else None
        eps_t = float(row["epsilon"]) if row is not None else float("inf")
        generation_end = None if row is None else row["generation_end"]
        previous_end = 0.0 if generation_end is not None and t == 0 else None
        if t > 0 and (t - 1) in observable.index:
            previous_end = observable.loc[t - 1, "generation_end"]
        raw_attempt_count = None if row is None else row["attempt_count"]
        try:
            candidate_attempt_count = int(raw_attempt_count)
        except (TypeError, ValueError):
            candidate_attempt_count = None
        attempt_count = (
            candidate_attempt_count
            if candidate_attempt_count is not None and candidate_attempt_count > 0
            else fallback_attempt_count + len(df)
        )
        fallback_attempt_count = max(fallback_attempt_count + len(df), attempt_count)
        # Use enumerate so we index w by position, not DataFrame label
        for pos, (idx, row) in enumerate(df.iterrows()):
            step += 1
            params = {col: float(row[col]) for col in limits}
            param_key = tuple(sorted((k_, round(v, 10)) for k_, v in params.items()))
            actual_loss = _distance_cache.get(param_key, eps_t)
            weight_val = float(w.iloc[pos]) if hasattr(w, "iloc") else None
            records.append(ParticleRecord(
                method="pyabc_smc",
                replicate=replicate,
                seed=seed,
                step=step,
                params=params,
                loss=actual_loss,
                weight=weight_val,
                tolerance=eps_t,
                wall_time=generation_end if generation_end is not None else time.time() - run_start,
                sim_start_time=previous_end,
                sim_end_time=generation_end,
                generation=t,
                record_kind="population_particle",
                time_semantics="generation_end",
                attempt_count=attempt_count,
            ))

    if progress is not None:
        progress.finish(simulations=eval_count, records=len(records))
    return records


def run_pyabc_smc(
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    progress=None,
) -> List[ParticleRecord]:
    """Run synchronous ABC-SMC via pyABC.

    Parameters
    ----------
    simulate_fn:
        Callable ``(params: dict, seed: int) -> float``.
    limits:
        Propulate-style limits dict ``{name: (lo, hi)}``.
    inference_cfg:
        ``config["inference"]`` sub-dict.
    output_dir:
        Output directory (used for pyABC database path).
    replicate:
        Replicate index stored in each record.
    seed:
        Base RNG seed.

    Returns
    -------
    List[ParticleRecord]

    Raises
    ------
    ImportError
        If pyabc is not installed.
    """
    try:
        import pyabc
    except ImportError as exc:
        raise ImportError(
            "The pyabc method requires the 'pyabc' package. "
            "Install it with: pip install pyabc"
        ) from exc

    max_sims = inference_cfg["max_simulations"]
    k = inference_cfg.get("k", 100)
    tol_init = inference_cfg.get("tol_init", 10.0)
    n_procs          = inference_cfg.get("n_workers", 1)
    parallel_backend = resolve_pyabc_parallel_backend(
        inference_cfg,
        method_name="pyabc_smc",
        simulate_fn=simulate_fn,
    )
    n_procs = resolve_pyabc_worker_count(
        simulate_fn,
        int(n_procs),
        parallel_backend,
        method_name="pyabc_smc",
    )

    if parallel_backend == "mpi":
        try:
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor
        except ImportError as exc:
            raise ImportError(
                "The 'mpi' parallel_backend requires mpi4py. "
                "Install it with: pip install mpi4py"
            ) from exc

        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is None:
                return []
            sampler = build_pyabc_sampler(
                n_procs,
                parallel_backend,
                cfuture_executor=executor,
            )
            return _run_pyabc_smc_with_sampler(
                sampler=sampler,
                simulate_fn=simulate_fn,
                limits=limits,
                max_sims=max_sims,
                k=k,
                tol_init=tol_init,
                output_dir=output_dir,
                replicate=replicate,
                seed=seed,
                progress=progress,
            )

    sampler = build_pyabc_sampler(n_procs, parallel_backend)
    return _run_pyabc_smc_with_sampler(
        sampler=sampler,
        simulate_fn=simulate_fn,
        limits=limits,
        max_sims=max_sims,
        k=k,
        tol_init=tol_init,
        output_dir=output_dir,
        replicate=replicate,
        seed=seed,
        progress=progress,
    )
