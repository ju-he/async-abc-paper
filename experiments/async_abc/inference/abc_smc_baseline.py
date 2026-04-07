"""Classical synchronous ABC-SMC baseline via pyABC.

Differs from :func:`~.pyabc_wrapper.run_pyabc_smc` in that it runs a fixed
number of SMC generations (``n_generations`` config key, default 5) rather
than targeting a minimum epsilon.  This makes the simulation budget more
predictable and the comparison fairer.

If ``pyabc`` is not installed, raises ``ImportError`` with installation
instructions.
"""
import logging
import time
from datetime import timedelta
from typing import Callable, Dict, List

from ..io.paths import OutputDir
from ..io.records import ParticleRecord
from ..utils.seeding import canonical_param_key, canonical_param_key_json, stable_seed
from ._attempt_trace import attempt_records_from_events, instrument_simulate, load_attempt_events
from ._pyabc_history import history_observable_frame
from .pyabc_sampler import (
    TrackedFutureExecutor,
    build_pyabc_sampler,
    resolve_pyabc_client_max_jobs,
    resolve_pyabc_mpi_sampler,
    resolve_pyabc_parallel_backend,
    resolve_pyabc_worker_count,
)

from ._pyabc_common import db_suffix as _db_suffix, prepare_db_path as _prepare_db_path

logger = logging.getLogger(__name__)



def _run_abc_smc_baseline_with_sampler(
    *,
    sampler,
    simulate_fn: Callable,
    limits: Dict,
    max_sims: int,
    k: int,
    tol_init: float,
    n_generations: int,
    output_dir: OutputDir,
    replicate: int,
    seed: int,
    checkpoint_tag: str = "",
    max_wall_time_s: float | None = None,
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

    _distance_cache: Dict = {}
    eval_count = 0

    trace_dir = output_dir.logs / f"abc_smc_baseline_rep{replicate}_seed{seed}{_db_suffix(checkpoint_tag)}_attempts"
    timed_simulate = instrument_simulate(simulate_fn, trace_dir)

    def pyabc_model(params):
        nonlocal eval_count
        param_key = canonical_param_key(params)
        sim_seed = stable_seed(seed, param_key)
        loss = float(timed_simulate(dict(params), seed=sim_seed))
        _distance_cache[param_key] = loss
        eval_count += 1
        if progress is not None:
            progress.update(simulations=eval_count)
        return {"distance": loss}

    def pyabc_distance(x, x0):
        return x["distance"]

    db_path = _prepare_db_path(
        output_dir,
        method_name="abc_smc_baseline",
        replicate=replicate,
        seed=seed,
        checkpoint_tag=checkpoint_tag,
    )

    abc = pyabc.ABCSMC(
        models=pyabc_model,
        parameter_priors=prior,
        distance_function=pyabc_distance,
        population_size=k,
        transitions=pyabc.MultivariateNormalTransition(),
        eps=pyabc.QuantileEpsilon(initial_epsilon=tol_init, alpha=0.5),
        sampler=sampler,
    )
    abc.new(db_path, {"distance": 0.0})

    # Seed numpy before running so pyABC's internal prior sampling is reproducible
    np.random.seed(seed % (2**31))
    run_start = time.time()
    # When a wall-time cap is set, it should be the sole binding stopping
    # criterion so that abc_smc_baseline produces data for the full time
    # window.  Generation and simulation caps remain as fallbacks for
    # non-wall-time runs (e.g. unit tests).
    if max_wall_time_s is not None:
        run_max_sims = int(1e12)
        run_max_populations = int(1e6)
    else:
        run_max_sims = max_sims
        run_max_populations = n_generations
    history = abc.run(
        minimum_epsilon=0.0,
        max_total_nr_simulations=run_max_sims,
        max_nr_populations=run_max_populations,
        max_walltime=(
            timedelta(seconds=float(max_wall_time_s))
            if max_wall_time_s is not None
            else None
        ),
    )

    observable = history_observable_frame(history, run_start)

    attempt_events = load_attempt_events(trace_dir, run_start_abs=float(run_start))
    loss_by_param_key = {
        str(event["param_key"]): float(event["loss"])
        for event in attempt_events
    }

    records: List[ParticleRecord] = []
    records.extend(
        attempt_records_from_events(
            attempt_events,
            method_name="abc_smc_baseline",
            replicate=replicate,
            observable_attempt_counts=observable["attempt_count"].tolist(),
        )
    )
    step = 0
    fallback_attempt_count = 0
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        row = observable.loc[t] if t in observable.index else None
        eps_t = float(row["epsilon"]) if row is not None else float("inf")
        generation_end = None if row is None else row["generation_end"]
        generation_start = 0.0 if generation_end is not None and t == 0 else None
        if t > 0:
            generation_start = observable.loc[t - 1, "generation_end"]
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
        for pos, (idx, row) in enumerate(df.iterrows()):
            step += 1
            params = {col: float(row[col]) for col in limits}
            param_key = canonical_param_key(params)
            actual_loss = _distance_cache.get(param_key)
            if actual_loss is None:
                actual_loss = loss_by_param_key.get(canonical_param_key_json(params))
            if actual_loss is None:
                raise RuntimeError(
                    "Could not match pyABC population particle to any traced simulation attempt. "
                    "This indicates incomplete attempt tracing or parameter-key mismatch."
                )
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
                record_kind="population_particle",
                time_semantics="generation_end",
                attempt_count=attempt_count,
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
    mpi_executor=None,
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
    tol_init = inference_cfg.get("tol_init", 10.0)
    n_procs          = inference_cfg.get("n_workers", 1)
    max_wall_time_s = inference_cfg.get("max_wall_time_s")
    max_wall_time_s = None if max_wall_time_s in (None, "") else float(max_wall_time_s)
    n_generations = inference_cfg.get("n_generations", 5)
    parallel_backend = resolve_pyabc_parallel_backend(
        inference_cfg,
        method_name="abc_smc_baseline",
        simulate_fn=simulate_fn,
    )
    checkpoint_tag = str(inference_cfg.get("_checkpoint_tag", ""))
    n_procs = resolve_pyabc_worker_count(
        simulate_fn,
        int(n_procs),
        parallel_backend,
        method_name="abc_smc_baseline",
    )
    client_max_jobs = resolve_pyabc_client_max_jobs(
        inference_cfg,
        parallel_backend=parallel_backend,
        n_procs=int(n_procs),
        mpi_sampler=resolve_pyabc_mpi_sampler(
            inference_cfg,
            parallel_backend=parallel_backend,
            method_name="abc_smc_baseline",
        ),
    )
    mpi_sampler = resolve_pyabc_mpi_sampler(
        inference_cfg,
        parallel_backend=parallel_backend,
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

        def _run_with_executor(executor):
            logger.info(
                "[abc_smc_baseline] mpi sampler config: pyabc_mpi_sampler=%s n_workers=%d client_max_jobs=%s",
                mpi_sampler,
                int(n_procs),
                (
                    str(int(client_max_jobs))
                    if client_max_jobs is not None
                    else "ignored"
                ),
            )
            if mpi_sampler == "mapping":
                sampler = build_pyabc_sampler(
                    n_procs,
                    parallel_backend,
                    mpi_sampler=mpi_sampler,
                    mpi_map=executor.map,
                    client_max_jobs=client_max_jobs,
                )
            else:
                tracker = TrackedFutureExecutor(executor)
                sampler = build_pyabc_sampler(
                    n_procs,
                    parallel_backend,
                    mpi_sampler=mpi_sampler,
                    cfuture_executor=tracker,
                    client_max_jobs=client_max_jobs,
                )
            return _run_abc_smc_baseline_with_sampler(
                sampler=sampler,
                simulate_fn=simulate_fn,
                limits=limits,
                max_sims=max_sims,
                k=k,
                tol_init=tol_init,
                n_generations=n_generations,
                output_dir=output_dir,
                replicate=replicate,
                seed=seed,
                checkpoint_tag=checkpoint_tag,
                max_wall_time_s=max_wall_time_s,
                progress=progress,
            )

        # Shared executor path: caller manages MPICommExecutor lifecycle.
        # Only root calls this — workers are in the server recv loop.
        if mpi_executor is not None:
            return _run_with_executor(mpi_executor)

        # Self-managed executor path: create and destroy per call.
        result: List[ParticleRecord] = []
        context_exit_start = None
        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:
                result = _run_with_executor(executor)
                context_exit_start = time.monotonic()
            # Workers (executor is None) fall through here without returning,
            # so that all ranks exit the with-block together.
        if context_exit_start is not None:
            logger.debug(
                "[abc_smc_baseline] MPICommExecutor context exited in %.3fs",
                time.monotonic() - context_exit_start,
            )
        # Barrier after teardown: prevents workers from racing ahead to
        # COMM_WORLD collectives while root is still in Disconnect().
        if MPI.COMM_WORLD.Get_size() > 1:
            barrier_start = time.monotonic()
            MPI.COMM_WORLD.Barrier()
            logger.debug(
                "[abc_smc_baseline] post-exit COMM_WORLD barrier completed in %.3fs",
                time.monotonic() - barrier_start,
            )
        return result

    sampler = build_pyabc_sampler(
        n_procs,
        parallel_backend,
        client_max_jobs=client_max_jobs,
    )
    return _run_abc_smc_baseline_with_sampler(
        sampler=sampler,
        simulate_fn=simulate_fn,
        limits=limits,
        max_sims=max_sims,
        k=k,
        tol_init=tol_init,
        n_generations=n_generations,
        output_dir=output_dir,
        replicate=replicate,
        seed=seed,
        checkpoint_tag=checkpoint_tag,
        max_wall_time_s=max_wall_time_s,
        progress=progress,
    )
