"""Classical synchronous ABC-SMC baseline via pyABC.

Differs from :func:`~.pyabc_wrapper.run_pyabc_smc` in that it runs a fixed
number of SMC generations (``n_generations`` config key, default 5) rather
than targeting a minimum epsilon.  This makes the simulation budget more
predictable and the comparison fairer.

If ``pyabc`` is not installed, raises ``ImportError`` with installation
instructions.
"""
import logging
from datetime import datetime
from typing import Callable, Dict, List

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from .pyabc_sampler import (
    build_pyabc_sampler,
    resolve_pyabc_parallel_backend,
    resolve_pyabc_worker_count,
)

logger = logging.getLogger(__name__)


def _run_abc_smc_baseline_with_sampler(
    *,
    sampler,
    simulate_fn: Callable,
    limits: Dict,
    max_sims: int,
    k: int,
    n_generations: int,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    progress=None,
) -> List[ParticleRecord]:
    import pyabc
    import numpy as np

    # Build pyABC prior from limits
    prior = pyabc.Distribution(
        **{
            name: pyabc.RV("uniform", lo, hi - lo)
            for name, (lo, hi) in limits.items()
        }
    )

    # Per-call distance cache; local to avoid cross-replicate contamination.
    # Note: not populated for MPI backend (workers run in separate processes).
    _distance_cache: Dict = {}
    eval_count = 0

    def pyabc_model(params):
        nonlocal eval_count
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

    db_path = f"sqlite:///{output_dir.data / f'abc_smc_baseline_rep{replicate}.db'}"

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
    run_start = datetime.now()
    history = abc.run(
        minimum_epsilon=0.0,
        max_total_nr_simulations=max_sims,
        max_nr_populations=n_generations,
    )

    # Robust epsilon extraction: key by generation index
    all_pops = history.get_all_populations()
    all_pops = all_pops[all_pops["t"] >= 0].copy()
    eps_series = all_pops.set_index("t")["epsilon"]
    end_time_series = all_pops.set_index("t")["population_end_time"]

    records: List[ParticleRecord] = []
    step = 0
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        eps_t = float(eps_series.loc[t]) if t in eps_series.index else float("inf")
        generation_end_dt = end_time_series.loc[t] if t in end_time_series.index else None
        previous_end_dt = run_start if t == 0 else end_time_series.loc[t - 1]
        generation_start = (
            (previous_end_dt - run_start).total_seconds()
            if previous_end_dt is not None
            else None
        )
        generation_end = (
            (generation_end_dt - run_start).total_seconds()
            if generation_end_dt is not None
            else None
        )
        for pos, (idx, row) in enumerate(df.iterrows()):
            step += 1
            params = {col: float(row[col]) for col in limits}
            param_key = tuple(sorted((k_, round(v, 10)) for k_, v in params.items()))
            actual_loss = _distance_cache.get(param_key, eps_t)
            weight_val = float(w.iloc[pos]) if hasattr(w, "iloc") else None
            records.append(ParticleRecord(
                method="abc_smc_baseline",
                replicate=replicate,
                seed=seed,
                step=step,
                params=params,
                loss=actual_loss,
                weight=weight_val,
                tolerance=eps_t,
                wall_time=generation_end if generation_end is not None else 0.0,
                sim_start_time=generation_start,
                sim_end_time=generation_end,
                generation=t,
            ))

    if progress is not None:
        progress.finish(
            simulations=eval_count,
            generations=history.max_t + 1,
            records=len(records),
        )
    return records


def run_abc_smc_baseline(
    simulate_fn: Callable,
    limits: Dict,
    inference_cfg: Dict,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    progress=None,
) -> List[ParticleRecord]:
    """Run synchronous ABC-SMC with a fixed number of generations via pyABC.

    Parameters
    ----------
    simulate_fn:
        Callable ``(params: dict, seed: int) -> float``.
    limits:
        Search-space limits dict ``{name: (lo, hi)}``.
    inference_cfg:
        ``config["inference"]`` sub-dict.  Reads ``max_simulations``, ``k``,
        ``tol_init``, ``n_generations`` (default 5), ``n_workers`` (default 1).
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
            "The abc_smc_baseline method requires the 'pyabc' package. "
            "Install it with: pip install pyabc"
        ) from exc

    max_sims     = inference_cfg["max_simulations"]
    k            = inference_cfg.get("k", 100)
    n_generations = inference_cfg.get("n_generations", 5)
    n_procs          = inference_cfg.get("n_workers", 1)
    parallel_backend = resolve_pyabc_parallel_backend(
        inference_cfg,
        method_name="abc_smc_baseline",
        simulate_fn=simulate_fn,
    )
    n_procs = resolve_pyabc_worker_count(
        simulate_fn,
        int(n_procs),
        parallel_backend,
        method_name="abc_smc_baseline",
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
            return _run_abc_smc_baseline_with_sampler(
                sampler=sampler,
                simulate_fn=simulate_fn,
                limits=limits,
                max_sims=max_sims,
                k=k,
                n_generations=n_generations,
                output_dir=output_dir,
                replicate=replicate,
                seed=seed,
                progress=progress,
            )

    sampler = build_pyabc_sampler(n_procs, parallel_backend)
    return _run_abc_smc_baseline_with_sampler(
        sampler=sampler,
        simulate_fn=simulate_fn,
        limits=limits,
        max_sims=max_sims,
        k=k,
        n_generations=n_generations,
        output_dir=output_dir,
        replicate=replicate,
        seed=seed,
        progress=progress,
    )
